"""
HunyuanVideo-1.5 DiT Flash Attention (Dense Path) — NKI 0.4.0b4 kernel.

The HunyuanVideo-1.5 DiT transformer uses 54 double-stream blocks, each
performing joint attention over concatenated image + text tokens.
At 480p 5-frame (the default configuration):
  - Image tokens: 2*30*53 = 3180  (T=2, H=30, W=53 latent grid)
  - Text tokens:  320              (byT5 + LLM)
  - Total seq:    3500 -> padded to 4096 for NKI tile alignment

Architecture per TP rank (TP=4, 16 heads total):
  - batch_heads = 4 (local heads per rank)
  - head_dim = 128
  - Non-causal, no mask (text padding zeroed upstream)

The attention kernel is called 54x per denoising step (~327ms/step total).
Optimizing this kernel directly reduces end-to-end generation latency.

Layout:
  Q: (batch_heads, seqlen_q, d_head)   -- bf16
  K: (batch_heads, d_head, seqlen_kv)  -- bf16 (transposed)
  V: (batch_heads, seqlen_kv, d_head)  -- bf16
  Output: (batch_heads, seqlen_q, d_head) -- bf16

NKI 0.4.0b4 ISA-level style:
  - Explicit buffer placement (sbuf, psum, shared_hbm)
  - nisa.tensor_scalar for per-lane broadcasting (PMAX,1) over (PMAX,F)
  - nisa.activation(dst, op, data) for exp, reciprocal
  - nisa.tensor_reduce(dst, op, data, axis) for reductions
  - nisa.nc_matmul for matrix multiply (bf16 inputs, fp32 psum output)
  - No operator overloading (* + -), no nl.arange indexing
"""

import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa
import torch


# Tile size constants
PMAX = 128  # Partition dim = head_dim = 128
K_TILE = 512  # K/V chunk size for flash attention tiling
LARGE_NEG = -9984.0  # bf16-safe large negative
SCALE = 1.0 / math.sqrt(128)  # Precomputed attention scale

# HunyuanVideo-specific dimensions
BATCH_HEADS = 4  # local heads per TP rank (16 heads / TP=4)
D_HEAD = 128  # head dimension
SEQ_LEN = 4096  # padded from 3500 (3180 img + 320 text) to 4096 for NKI alignment


# SUBSTITUTE HERE


@nki.jit
def ref(q_hbm, k_hbm, v_hbm):
    """Flash attention with fp32 online softmax for HunyuanVideo DiT.

    NKI 0.4.0b4 ISA API. Non-causal, bf16 I/O, fp32 softmax.

    Args:
        q_hbm: [batch, seqlen_q, d_head] bf16
        k_hbm: [batch, d_head, seqlen_kv] bf16 (transposed)
        v_hbm: [batch, seqlen_kv, d_head] bf16
    Returns:
        output: [batch, seqlen_q, d_head] bf16
    """
    batch, seqlen_q, d_head = q_hbm.shape
    _, _, seqlen_kv = k_hbm.shape

    n_q_tiles = seqlen_q // PMAX
    n_kv_chunks = seqlen_kv // K_TILE

    output = nl.ndarray(
        (batch, seqlen_q, d_head), dtype=q_hbm.dtype, buffer=nl.shared_hbm
    )

    for b in nl.affine_range(batch):
        for qt in nl.affine_range(n_q_tiles):
            q_offset = qt * PMAX

            # Load Q tile bf16, scale to fp32, cast back to bf16 for matmul
            q_tile = nl.ndarray((PMAX, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_tile, src=q_hbm[b, nl.ds(q_offset, PMAX), :])
            q_f32 = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(q_f32, q_tile, op0=nl.multiply, operand0=SCALE)
            q_scaled = nl.ndarray((PMAX, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q_scaled, src=q_f32)

            # Online softmax state
            running_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_max, LARGE_NEG)
            running_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_sum, 0.0)
            out_acc = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(out_acc, 0.0)

            for kv_idx in nl.sequential_range(n_kv_chunks):
                kv_offset = kv_idx * K_TILE

                # Load K^T chunk [d_head, K_TILE] bf16
                k_chunk = nl.ndarray(
                    (d_head, K_TILE), dtype=nl.bfloat16, buffer=nl.sbuf
                )
                nisa.dma_copy(dst=k_chunk, src=k_hbm[b, :, nl.ds(kv_offset, K_TILE)])

                # QK matmul: bf16 @ bf16 -> fp32 in PSUM
                scores = nl.ndarray((PMAX, K_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(scores, q_scaled, k_chunk)

                # Copy scores to SBUF
                scores_sbuf = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=scores_sbuf, src=scores)

                # Online softmax: chunk_max
                chunk_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(chunk_max, nl.max, scores_sbuf, axis=(1,))

                # new_max
                new_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(new_max, running_max, chunk_max, op=nl.maximum)

                # correction = exp(running_max - new_max)
                max_diff = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(max_diff, running_max, new_max, op=nl.subtract)
                correction = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(correction, nl.exp, max_diff)

                # Rescale: tensor_scalar broadcasts (PMAX,1) over (PMAX,F)
                nisa.tensor_scalar(
                    out_acc, out_acc, op0=nl.multiply, operand0=correction
                )
                nisa.tensor_scalar(
                    running_sum, running_sum, op0=nl.multiply, operand0=correction
                )

                # exp(scores - new_max)
                scores_shifted = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    scores_shifted, scores_sbuf, op0=nl.subtract, operand0=new_max
                )
                exp_scores = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.activation(exp_scores, nl.exp, scores_shifted)

                # chunk_sum
                chunk_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(chunk_sum, nl.add, exp_scores, axis=(1,))
                nisa.tensor_tensor(running_sum, running_sum, chunk_sum, op=nl.add)

                # Update running_max
                nisa.tensor_copy(dst=running_max, src=new_max)

                # P @ V in sub-tiles (K_TILE / PMAX = 4)
                for vt in range(K_TILE // PMAX):
                    v_offset = kv_offset + vt * PMAX
                    v_tile = nl.ndarray(
                        (PMAX, d_head), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.dma_copy(dst=v_tile, src=v_hbm[b, nl.ds(v_offset, PMAX), :])

                    # Cast P-slice to bf16 for matmul
                    p_slice = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=p_slice, src=exp_scores[:, nl.ds(vt * PMAX, PMAX)]
                    )
                    p_bf16 = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=p_bf16, src=p_slice)

                    # P@V matmul
                    pv = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(pv, p_bf16, v_tile)
                    pv_sbuf = nl.ndarray(
                        (PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=pv_sbuf, src=pv)
                    nisa.tensor_tensor(out_acc, out_acc, pv_sbuf, op=nl.add)

            # Normalize: out_acc * reciprocal(running_sum)
            inv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(inv_sum, nl.reciprocal, running_sum)
            out_norm = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(out_norm, out_acc, op0=nl.multiply, operand0=inv_sum)

            # Cast to bf16 and store
            out_bf16 = nl.ndarray((PMAX, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_bf16, src=out_norm)
            nisa.dma_copy(dst=output[b, nl.ds(q_offset, PMAX), :], src=out_bf16)

    return output


def reference_attention_cpu(q, k, v, scale):
    """CPU reference attention for correctness verification."""
    scores = torch.bmm(q, k) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.bmm(attn_weights, v)


def test_nki(ref_func, test_func):
    """Correctness check: compare ref and test vs CPU attention."""
    batch = BATCH_HEADS
    seqlen = SEQ_LEN
    d_head = D_HEAD
    scale = SCALE

    device = torch.device("neuron")

    for seed in range(2):
        np.random.seed(42 + seed)
        q_np = (np.random.randn(batch, seqlen, d_head) * 0.1).astype(np.float32)
        k_np = (np.random.randn(batch, d_head, seqlen) * 0.1).astype(np.float32)
        v_np = (np.random.randn(batch, seqlen, d_head) * 0.1).astype(np.float32)

        # CPU reference
        cpu_out = reference_attention_cpu(
            torch.tensor(q_np), torch.tensor(k_np), torch.tensor(v_np), scale
        ).numpy()

        # NKI on device (bf16 inputs)
        q_dev = torch.tensor(q_np, dtype=torch.bfloat16, device=device)
        k_dev = torch.tensor(k_np, dtype=torch.bfloat16, device=device)
        v_dev = torch.tensor(v_np, dtype=torch.bfloat16, device=device)

        result_ref = ref_func(q_dev, k_dev, v_dev)
        result_test = test_func(q_dev, k_dev, v_dev)

        ref_out = result_ref.detach().cpu().float().numpy()
        test_out = result_test.detach().cpu().float().numpy()

        cos_ref_test = np.dot(ref_out.flatten(), test_out.flatten()) / (
            np.linalg.norm(ref_out.flatten()) * np.linalg.norm(test_out.flatten())
            + 1e-12
        )

        cos_test_cpu = np.dot(test_out.flatten(), cpu_out.flatten()) / (
            np.linalg.norm(test_out.flatten()) * np.linalg.norm(cpu_out.flatten())
            + 1e-12
        )

        print(
            f"  seed={42 + seed}: cos(ref,test)={cos_ref_test:.6f}, cos(test,cpu)={cos_test_cpu:.6f}"
        )

        if cos_ref_test < 0.99:
            print(f"  FAIL: NKI ref vs test cosine {cos_ref_test:.6f} < 0.99")
            return False

        if cos_test_cpu < 0.99:
            print(f"  FAIL: NKI test vs CPU cosine {cos_test_cpu:.6f} < 0.99")
            return False

    return True


def benchmark_nki(nki_func):
    """Latency benchmark using wall-clock timing (PyTorch Native eager mode)."""
    import time

    batch = BATCH_HEADS
    seqlen = SEQ_LEN
    d_head = D_HEAD

    device = torch.device("neuron")
    np.random.seed(42)
    q_np = (np.random.randn(batch, seqlen, d_head) * 0.1).astype(np.float32)
    k_np = (np.random.randn(batch, d_head, seqlen) * 0.1).astype(np.float32)
    v_np = (np.random.randn(batch, seqlen, d_head) * 0.1).astype(np.float32)

    q_dev = torch.tensor(q_np, dtype=torch.bfloat16, device=device)
    k_dev = torch.tensor(k_np, dtype=torch.bfloat16, device=device)
    v_dev = torch.tensor(v_np, dtype=torch.bfloat16, device=device)

    # Warmup (includes first compilation if not cached)
    for _ in range(3):
        nki_func(q_dev, k_dev, v_dev)

    # Timed iterations
    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        nki_func(q_dev, k_dev, v_dev)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iters) * 1000.0
    print(f"Latency: {avg_ms:.3f} ms (avg over {iters} iters, wall-clock)")


if __name__ == "__main__":
    # When run standalone (no SUBSTITUTE HERE), use ref as test
    if "test" not in dir():
        test = ref
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)
