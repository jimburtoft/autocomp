"""Cross-entropy forward pass using online log-sum-exp algorithm with batched processing.

Standalone version of nki-library cross_entropy_forward kernel. Single-core (no LNC
sharding) for AutoComp optimization. All nki-library dependencies inlined.

Source: nki-library/core/loss/cross_entropy.py (316 lines)
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


# --- Inlined helpers ---


def div_ceil(n, d):
    return (n + d - 1) // d


# Test dimensions (smaller than production for fast AutoComp evaluation)
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
    Shared between ref and test to avoid code duplication.

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

    loss_hbm = nl.ndarray((num_positions, 1), dtype=dtype, buffer=nl.shared_hbm)
    lse_state_hbm = nl.ndarray((num_positions, 1), dtype=dtype, buffer=nl.shared_hbm)

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

        nisa.memset(dst=batch_m, value=-float("inf"))
        nisa.memset(dst=batch_d, value=0.0)

        nisa.dma_copy(
            dst=batch_targets[0:actual_batch, 0:1],
            src=targets_hbm[batch_start : batch_start + actual_batch, 0:1],
        )

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, vocab_size)
            actual_chunk_len = chunk_end - chunk_start

            if actual_chunk_len < chunk_size:
                nisa.memset(dst=batch_chunk, value=-float("inf"))

            nisa.dma_copy(
                dst=batch_chunk[0:actual_batch, 0:actual_chunk_len],
                src=logits_hbm[
                    batch_start : batch_start + actual_batch,
                    chunk_start:chunk_end,
                ],
            )

            nisa.tensor_reduce(
                op=nl.maximum,
                data=batch_chunk,
                dst=batch_chunk_max,
                axis=1,
            )

            nisa.tensor_tensor(
                dst=batch_m_new,
                data1=batch_m,
                data2=batch_chunk_max,
                op=nl.maximum,
            )

            nisa.tensor_tensor(
                dst=batch_m_diff,
                data1=batch_m,
                data2=batch_m_new,
                op=nl.subtract,
            )

            nisa.activation(op=nl.exp, data=batch_m_diff, dst=batch_correction)

            nisa.tensor_tensor(
                dst=batch_d_corrected,
                data1=batch_d,
                data2=batch_correction,
                op=nl.multiply,
            )

            nisa.tensor_scalar(
                dst=batch_exp_chunk,
                data=batch_chunk,
                op0=nl.subtract,
                operand0=batch_m_new,
            )

            nisa.activation(op=nl.exp, data=batch_exp_chunk, dst=batch_exp_chunk)

            nisa.tensor_reduce(
                op=nl.add, data=batch_exp_chunk, dst=batch_sum_exp, axis=1
            )

            nisa.tensor_tensor(
                dst=batch_d_new,
                data1=batch_d_corrected,
                data2=batch_sum_exp,
                op=nl.add,
            )

            nisa.tensor_copy(src=batch_m_new, dst=batch_m)
            nisa.tensor_copy(src=batch_d_new, dst=batch_d)

        nisa.activation(op=nl.log, data=batch_d, dst=batch_log_d)
        nisa.tensor_tensor(dst=batch_lse, data1=batch_m, data2=batch_log_d, op=nl.add)

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

        nisa.tensor_tensor(
            dst=batch_loss,
            data1=batch_lse,
            data2=batch_target_logits,
            op=nl.subtract,
        )

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
