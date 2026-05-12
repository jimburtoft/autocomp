"""
HunyuanVideo-1.5 DiT Flash Attention — Reference NKI kernel (NKI 0.4.0b4 ISA API).

Flash attention with FP32 online softmax for the HunyuanVideo-1.5 DiT.
Non-causal, bf16 I/O, fp32 softmax accumulation.

NKI 0.4.0b4 ISA-level style:
  - Explicit buffer placement (sbuf, psum, shared_hbm)
  - nisa.tensor_scalar for per-lane broadcasting (PMAX,1) over (PMAX,F)
  - nisa.activation for exp, reciprocal
  - nisa.nc_matmul for matrix multiply (bf16 inputs, fp32 psum output)
  - No operator overloading (* + -), no nl.arange indexing
"""

import nki
import nki.language as nl
import nki.isa as nisa
import math


# Tile size constants
PMAX = 128
K_TILE = 512
LARGE_NEG = -9984.0
SCALE = 1.0 / math.sqrt(128)


@nki.jit
def test(q_hbm, k_hbm, v_hbm):
    """Flash attention with fp32 online softmax — baseline reference.

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

                # Copy scores to SBUF for scalar ops
                scores_sbuf = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=scores_sbuf, src=scores)

                # Online softmax: chunk_max
                chunk_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(chunk_max, nl.max, scores_sbuf, axis=(1,))

                # new_max = max(running_max, chunk_max)
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

                # chunk_sum = sum(exp_scores)
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

                    # P@V: stationary[PMAX, PMAX] @ moving[PMAX, d_head] -> [PMAX, d_head]
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
