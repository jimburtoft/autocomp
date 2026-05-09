"""
HunyuanVideo-1.5 DiT Flash Attention — Reference NKI kernel (NKI 0.3.0 nl API).

Flash attention with FP32 online softmax for the HunyuanVideo-1.5 DiT.
Non-causal, bf16 I/O, fp32 softmax accumulation.

Baseline device-only latency: 18.39 ms (neuron-profile, trn2.3xlarge LNC=2)
  batch_heads=4, seq_len=4096, d_head=128

Profiling breakdown:
  - Vector engine: 87% active (softmax path dominates)
  - Tensor engine: 27.6% active (matmul underutilized)
  - MFU: 1.2%

Optimization targets:
  1. Reduce vector engine pressure (fuse exp+sum, fuse correction passes)
  2. Pipeline DMA loads with compute
  3. Wider K_TILE or batched V tiles if SBUF allows
  4. Use nl.loop_reduce for running statistics
"""

import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import torch


# Tile size constants
PMAX = 128
K_TILE = 512
LARGE_NEG = -9984.0


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
    scale = 1.0 / math.sqrt(d_head)

    n_q_tiles = seqlen_q // PMAX
    n_kv_chunks = seqlen_kv // K_TILE

    output = nl.ndarray(
        (batch, seqlen_q, d_head), dtype=q_hbm.dtype, buffer=nl.shared_hbm
    )

    for b in nl.affine_range(batch):
        for qt in nl.affine_range(n_q_tiles):
            q_offset = qt * PMAX

            # Load Q tile and scale
            q_idx = nl.arange(PMAX)[:, None]
            d_idx = nl.arange(d_head)[None, :]
            q_tile = nl.load(q_hbm[b, q_offset + q_idx, d_idx])
            q_tile = q_tile * scale

            # Index tensors for in-place updates
            i_p = nl.arange(PMAX)[:, None]
            i_1 = nl.arange(1)[None, :]
            i_d = nl.arange(d_head)[None, :]

            # Online softmax state
            running_max = nl.full((PMAX, 1), fill_value=LARGE_NEG, dtype=nl.float32)
            running_sum = nl.zeros((PMAX, 1), dtype=nl.float32)
            out_acc = nl.zeros((PMAX, d_head), dtype=nl.float32)

            for kv_idx in nl.sequential_range(n_kv_chunks):
                kv_offset = kv_idx * K_TILE

                # Load K^T chunk [d_head, K_TILE]
                k_p_idx = nl.arange(d_head)[:, None]
                k_f_idx = nl.arange(K_TILE)[None, :]
                k_chunk = nl.load(k_hbm[b, k_p_idx, kv_offset + k_f_idx])

                # Q @ K -> scores [PMAX, K_TILE]
                scores = nl.matmul(q_tile, k_chunk)

                # Online softmax
                chunk_max = nl.max(scores, axis=[1])
                new_max = nl.maximum(running_max, chunk_max)
                correction = nl.exp(running_max - new_max)

                # Rescale accumulators
                running_sum[i_p, i_1] = running_sum * correction
                out_acc[i_p, i_d] = out_acc * correction

                # exp(scores - new_max)
                exp_scores = nl.exp(scores - new_max)
                chunk_sum = nl.sum(exp_scores, axis=[1])
                running_sum[i_p, i_1] = running_sum + chunk_sum
                running_max[i_p, i_1] = new_max

                # P @ V in sub-tiles (partition limit = 128)
                n_v_tiles = K_TILE // PMAX  # 4
                for vt in range(n_v_tiles):
                    v_tile_offset = kv_offset + vt * PMAX
                    vt_p_idx = nl.arange(PMAX)[:, None]
                    vt_f_idx = nl.arange(d_head)[None, :]
                    v_tile = nl.load(v_hbm[b, v_tile_offset + vt_p_idx, vt_f_idx])

                    p_slice = exp_scores[:, nl.ds(vt * PMAX, PMAX)]
                    pv_tile = nl.matmul(p_slice, v_tile)
                    out_acc[i_p, i_d] = out_acc + pv_tile

            # Normalize and store
            out_normalized = out_acc / running_sum
            nl.store(output[b, q_offset + q_idx, d_idx], value=out_normalized)

    return output
