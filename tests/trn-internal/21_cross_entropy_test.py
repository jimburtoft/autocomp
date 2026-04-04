"""Test harness for cross-entropy forward NKI kernel.

AutoComp test harness interface:
  - ref: @nki.jit NKI kernel (correctness baseline, in preamble before SUBSTITUTE HERE)
  - test: @nki.jit NKI kernel (optimization target, in postamble after SUBSTITUTE HERE)
  - test_nki(ref_func, test_func) -> bool: correctness check
  - benchmark_nki(nki_func): latency measurement via nki.benchmark

Source: nki-library/core/loss/cross_entropy.py (standalone version, no LNC sharding)
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_xla.core import xla_model as xm


# --- Inlined helpers ---


def div_ceil(n, d):
    return (n + d - 1) // d


# Test dimensions (smaller than production for fast evaluation)
# num_positions must be divisible by positions_per_batch
TEST_NUM_POSITIONS = 32
TEST_VOCAB_SIZE = 4096
TEST_POSITIONS_PER_BATCH = 16
TEST_CHUNK_SIZE = 2048


def _cross_entropy_body(
    logits_hbm,
    targets_hbm,
    positions_per_batch,
    chunk_size,
    dtype,
):
    """Core cross-entropy forward computation (single-core, no LNC sharding).

    Online log-sum-exp algorithm with batched processing.
    Shared between ref and test to avoid duplicating ~150 lines of kernel code.

    Args:
        logits_hbm: [num_positions, vocab_size], float32 input logits
        targets_hbm: [num_positions, 1], int32 target indices (2D for DMA compat)
    Returns:
        (loss_hbm, lse_state_hbm): each [num_positions, 1]
    """
    num_positions = logits_hbm.shape[0]
    vocab_size = logits_hbm.shape[1]
    num_chunks = div_ceil(vocab_size, chunk_size)
    num_batches = div_ceil(num_positions, positions_per_batch)

    # Output tensors in HBM (2D for DMA compatibility)
    loss_hbm = nl.ndarray((num_positions, 1), dtype=dtype, buffer=nl.hbm)
    lse_state_hbm = nl.ndarray((num_positions, 1), dtype=dtype, buffer=nl.hbm)

    # Pre-allocate SBUF buffers
    batch_targets = nl.ndarray((positions_per_batch, 1), dtype=nl.int32, buffer=nl.sbuf)
    batch_m = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_d = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_chunk = nl.ndarray(
        (positions_per_batch, chunk_size), dtype=dtype, buffer=nl.sbuf
    )
    batch_chunk_max = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_m_new = nl.ndarray((positions_per_batch, 1), dtype=nl.float32, buffer=nl.sbuf)
    batch_m_diff = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_correction = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_d_corrected = nl.ndarray(
        (positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf
    )
    batch_exp_chunk = nl.ndarray(
        (positions_per_batch, chunk_size), dtype=dtype, buffer=nl.sbuf
    )
    batch_sum_exp = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_d_new = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_log_d = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_lse = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)
    batch_target_logits = nl.ndarray(
        (positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf
    )
    batch_loss = nl.ndarray((positions_per_batch, 1), dtype=dtype, buffer=nl.sbuf)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * positions_per_batch
        actual_batch = min(positions_per_batch, num_positions - batch_start)

        if actual_batch <= 0:
            break

        # Initialize running max and sum
        nisa.memset(dst=batch_m, value=-float("inf"))
        nisa.memset(dst=batch_d, value=0.0)

        # Load target indices for this batch
        nisa.dma_copy(
            dst=batch_targets[0:actual_batch, 0:1],
            src=targets_hbm[batch_start : batch_start + actual_batch, 0:1],
        )

        # Process vocabulary in chunks (online log-sum-exp)
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, vocab_size)
            actual_chunk_len = chunk_end - chunk_start

            # Pad partial chunks with -inf
            if actual_chunk_len < chunk_size:
                nisa.memset(dst=batch_chunk, value=-float("inf"))

            # Load logits chunk
            nisa.dma_copy(
                dst=batch_chunk[0:actual_batch, 0:actual_chunk_len],
                src=logits_hbm[
                    batch_start : batch_start + actual_batch,
                    chunk_start:chunk_end,
                ],
            )

            # Max of current chunk
            nisa.tensor_reduce(
                op=nl.maximum,
                data=batch_chunk,
                dst=batch_chunk_max,
                axis=1,
            )

            # Update running max: m_new = max(m, chunk_max)
            nisa.tensor_tensor(
                dst=batch_m_new,
                data1=batch_m,
                data2=batch_chunk_max,
                op=nl.maximum,
            )

            # Correction factor: exp(m_old - m_new)
            nisa.tensor_tensor(
                dst=batch_m_diff,
                data1=batch_m,
                data2=batch_m_new,
                op=nl.subtract,
            )
            nisa.activation(op=nl.exp, data=batch_m_diff, dst=batch_correction)

            # Correct previous sum: d_corrected = d * correction
            nisa.tensor_tensor(
                dst=batch_d_corrected,
                data1=batch_d,
                data2=batch_correction,
                op=nl.multiply,
            )

            # exp(chunk - m_new)
            nisa.tensor_scalar(
                dst=batch_exp_chunk,
                data=batch_chunk,
                op0=nl.subtract,
                operand0=batch_m_new,
            )
            nisa.activation(op=nl.exp, data=batch_exp_chunk, dst=batch_exp_chunk)

            # Sum of exponentials in this chunk
            nisa.tensor_reduce(
                op=nl.add, data=batch_exp_chunk, dst=batch_sum_exp, axis=1
            )

            # d_new = d_corrected + sum_exp
            nisa.tensor_tensor(
                dst=batch_d_new,
                data1=batch_d_corrected,
                data2=batch_sum_exp,
                op=nl.add,
            )

            # Update state for next iteration
            nisa.tensor_copy(src=batch_m_new, dst=batch_m)
            nisa.tensor_copy(src=batch_d_new, dst=batch_d)

        # Compute LSE = m + log(d)
        nisa.activation(op=nl.log, data=batch_d, dst=batch_log_d)
        nisa.tensor_tensor(dst=batch_lse, data1=batch_m, data2=batch_log_d, op=nl.add)

        # Gather target logits using .ap() indirect access
        for position_idx in range(actual_batch):
            absolute_position = batch_start + position_idx
            nisa.dma_copy(
                dst=batch_target_logits[position_idx : position_idx + 1, :],
                src=logits_hbm.ap(
                    pattern=[[vocab_size, 1], [1, 1]],
                    offset=absolute_position * vocab_size,
                    scalar_offset=batch_targets.ap(
                        pattern=[[1, 1], [1, 1]],
                        offset=position_idx,
                    ),
                    indirect_dim=1,
                ),
            )

        # loss = lse - target_logit
        nisa.tensor_tensor(
            dst=batch_loss,
            data1=batch_lse,
            data2=batch_target_logits,
            op=nl.subtract,
        )

        # Store results to HBM
        nisa.dma_copy(
            dst=lse_state_hbm[batch_start : batch_start + actual_batch, 0:1],
            src=batch_lse[0:actual_batch, 0:1],
        )
        nisa.dma_copy(
            dst=loss_hbm[batch_start : batch_start + actual_batch, 0:1],
            src=batch_loss[0:actual_batch, 0:1],
        )

    return loss_hbm, lse_state_hbm


@nki.jit
def ref(logits_hbm, targets_hbm):
    """Reference cross-entropy forward.

    Args:
        logits_hbm: [num_positions, vocab_size], float32
        targets_hbm: [num_positions, 1], int32
    Returns:
        (loss_hbm, lse_state_hbm): each [num_positions, 1]
    """
    return _cross_entropy_body(
        logits_hbm,
        targets_hbm,
        positions_per_batch=TEST_POSITIONS_PER_BATCH,
        chunk_size=TEST_CHUNK_SIZE,
        dtype=nl.float32,
    )


# SUBSTITUTE HERE


@nki.jit
def test(logits_hbm, targets_hbm):
    """Test cross-entropy forward (optimization target).

    Args:
        logits_hbm: [num_positions, vocab_size], float32
        targets_hbm: [num_positions, 1], int32
    Returns:
        (loss_hbm, lse_state_hbm): each [num_positions, 1]
    """
    return _cross_entropy_body(
        logits_hbm,
        targets_hbm,
        positions_per_batch=TEST_POSITIONS_PER_BATCH,
        chunk_size=TEST_CHUNK_SIZE,
        dtype=nl.float32,
    )


def cpu_reference(logits_np, targets_np):
    """CPU reference: standard cross-entropy loss per position.

    loss[i] = log(sum(exp(logits[i,:]))) - logits[i, target[i]]
            = logsumexp(logits[i,:]) - logits[i, target[i]]
    """
    import scipy.special

    num_positions = logits_np.shape[0]
    loss = np.zeros(num_positions, dtype=np.float32)
    lse = np.zeros(num_positions, dtype=np.float32)
    for i in range(num_positions):
        lse_val = scipy.special.logsumexp(logits_np[i, :].astype(np.float64))
        lse[i] = lse_val
        loss[i] = lse_val - logits_np[i, targets_np[i]]
    return loss.astype(np.float32), lse.astype(np.float32)


def test_nki(ref_func, test_func):
    """Correctness check: compare NKI ref, NKI test, and CPU reference."""
    device = xm.xla_device()

    num_positions = TEST_NUM_POSITIONS
    vocab_size = TEST_VOCAB_SIZE

    for seed in range(2):
        np.random.seed(42 + seed)
        logits_np = (np.random.randn(num_positions, vocab_size) * 2.0).astype(
            np.float32
        )
        targets_np = np.random.randint(0, vocab_size, size=(num_positions,)).astype(
            np.int32
        )

        # CPU reference
        cpu_loss, cpu_lse = cpu_reference(logits_np, targets_np)

        # Create device tensors (targets as 2D for NKI DMA compatibility)
        logits_dev = torch.tensor(logits_np, dtype=torch.float32, device=device)
        targets_dev = torch.tensor(
            targets_np.reshape(-1, 1), dtype=torch.int32, device=device
        )

        # Run NKI ref
        ref_result = ref_func(logits_dev, targets_dev)
        ref_loss = ref_result[0].detach().cpu().float().numpy().flatten()

        # Run NKI test
        test_result = test_func(logits_dev, targets_dev)
        test_loss = test_result[0].detach().cpu().float().numpy().flatten()

        # NKI ref vs NKI test (should be identical)
        cos_ref_test = np.dot(ref_loss, test_loss) / (
            np.linalg.norm(ref_loss) * np.linalg.norm(test_loss) + 1e-12
        )

        # NKI test vs CPU
        cos_test_cpu = np.dot(test_loss, cpu_loss) / (
            np.linalg.norm(test_loss) * np.linalg.norm(cpu_loss) + 1e-12
        )

        print(
            f"  seed={42 + seed}: cos(ref,test)={cos_ref_test:.6f}, cos(test,cpu)={cos_test_cpu:.6f}"
        )

        if cos_ref_test < 0.999:
            print(f"  FAIL: NKI ref vs test cosine {cos_ref_test:.6f} < 0.999")
            return False

        if cos_test_cpu < 0.999:
            print(f"  FAIL: NKI test vs CPU cosine {cos_test_cpu:.6f} < 0.999")
            return False

    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()

    num_positions = TEST_NUM_POSITIONS
    vocab_size = TEST_VOCAB_SIZE

    np.random.seed(42)
    logits_np = (np.random.randn(num_positions, vocab_size) * 2.0).astype(np.float32)
    targets_np = np.random.randint(0, vocab_size, size=(num_positions,)).astype(
        np.int32
    )

    logits_dev = torch.tensor(logits_np, dtype=torch.float32, device=device)
    targets_dev = torch.tensor(
        targets_np.reshape(-1, 1), dtype=torch.int32, device=device
    )

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(logits_dev, targets_dev)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))


if __name__ == "__main__":
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)
