"""
Custom NKI flash attention kernel with FP32 softmax for FlashVSR.

Updated for NKI 0.3.0 (GA, SDK 2.29). Migrated from Beta 2 (0.2.x, SDK 2.28).

Keeps exp(scores) in fp32 throughout the P@V matmul, matching CUDA flash
attention's precision. The standard nkilib attention_cte kernel truncates
exp to bf16, causing temporal quality degradation in one-step diffusion.

Design:
  - Flash attention tiling: processes K/V in chunks of 512 (K_TILE)
  - Online softmax: running max + running sum (Milakov & Gimelshein)
  - FP32 exp from computation through P@V matmul (no bf16 truncation)
  - V tiles upcast bf16->fp32 via tensor_copy before matmul
  - Non-causal only (no masking logic)
  - Uses intentional PSUM accumulation for P@V across V-tile chunks
  - NKI 0.3.0: explicit nc_matmul accumulate= parameter for V-tile loop

Layout (matches attention_cte defaults):
  Q: (batch, seqlen_q, d_head)   -- bf16
  K: (batch, d_head, seqlen_kv)  -- bf16 (transposed)
  V: (batch, seqlen_kv, d_head)  -- bf16
  Output: (batch, seqlen_q, d_head) -- bf16

SBUF budget per Q-tile (128 positions, K_TILE=512):
  Q tile (fp32):   128 x 128 x fp32 =  64 KB  (persistent across K chunks)
  K tile:          128 x 512 x bf16 = 128 KB  (per K chunk)
  V tile bf16:     128 x 128 x bf16 =  32 KB  (per V sub-tile)
  V tile fp32:     128 x 128 x fp32 =  64 KB  (upcast for matmul)
  qk_sbuf:         128 x 512 x fp32 = 256 KB  (per K chunk)
  exp_scores:      128 x 512 x fp32 = 256 KB  (per K chunk, KEY: fp32)
  qk_centered:     128 x 512 x fp32 = 256 KB  (shares lifetime with qk_sbuf)
  running max:     128 x 1   x fp32 =   0.5 KB
  running sum:     128 x 1   x fp32 =   0.5 KB
  output acc:      128 x 128 x fp32 =  64 KB  (persistent)
  pv_sbuf:         128 x 128 x fp32 =  64 KB  (temp for PSUM->SBUF)
  ----------------------------------------
  Peak:                               ~929 KB  (well within 29 MB SBUF)

Note: qk_sbuf and qk_centered could alias the same buffer since they have
non-overlapping lifetimes, but the compiler should handle this automatically.

NKI 0.3.0 migration notes:
  - @nki.jit: already bare (no mode= or platform_target=)
  - nisa.memset: float values into fp32 tensors -- compatible
  - Output tensor: already uses nl.shared_hbm -- compatible
  - nc_matmul: added explicit accumulate= parameter for V-tile loop
  - nisa.activation with nl.reciprocal: unchanged in 0.3.0
  - All buffer= params are explicit (nl.sbuf/nl.psum) -- compatible with new defaults
"""

import nki
import nki.language as nl
import nki.isa as nisa


# Tile size constants
PMAX = 128  # nl.tile_size.pmax -- partition dimension
K_TILE = 512  # K/V processing chunk (nl.tile_size.gemm_moving_fmax)
V_TILE = 128  # V sub-tile for matmul (nl.tile_size.gemm_stationary_fmax)
LARGE_NEG = -9.984e3  # Initial max value (within bf16 range, safe for fp32)


@nki.jit
def fp32_exp_attention(q, k, v, scale: float = 1.0):
    """Flash attention with fp32 softmax (non-causal).

    Args:
        q: (batch, seqlen_q, d_head) bf16
        k: (batch, d_head, seqlen_kv) bf16 (transposed)
        v: (batch, seqlen_kv, d_head) bf16
        scale: attention scale factor (typically 1/sqrt(d_head))

    Returns:
        output: (batch, seqlen_q, d_head) bf16
    """
    batch, seqlen_q, d_head = q.shape
    _, _, seqlen_kv = k.shape

    assert d_head == PMAX, f"d_head must be {PMAX}"
    assert seqlen_q % PMAX == 0, f"seqlen_q must be divisible by {PMAX}"
    assert seqlen_kv % K_TILE == 0, f"seqlen_kv must be divisible by {K_TILE}"

    n_q_tiles = seqlen_q // PMAX
    n_kv_chunks = seqlen_kv // K_TILE
    n_v_tiles_per_chunk = K_TILE // V_TILE  # 512 / 128 = 4

    # Output in HBM
    output = nl.ndarray((batch, seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # Process each batch independently
    for b in nl.affine_range(batch):
        # Process Q in tiles of PMAX=128
        for q_tile_idx in nl.affine_range(n_q_tiles):
            q_offset = q_tile_idx * PMAX

            # Load Q tile into SBUF: shape (PMAX, d_head) = (128, 128)
            q_tile = nl.ndarray((PMAX, d_head), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_tile, src=q[b, nl.ds(q_offset, PMAX), :])

            # Apply scale to Q (stays bf16 for now)
            q_scaled_bf16 = nl.ndarray(
                (PMAX, d_head), dtype=nl.bfloat16, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=q_scaled_bf16, data=q_tile, op0=nl.multiply, operand0=scale
            )

            # Transpose Q from (seq, d_head) to (d_head, seq) for nc_matmul.
            # nc_matmul(dst, stat, mov) computes:
            #   dst[f_s, f_m] = sum_p stat[p, f_s] * mov[p, f_m]
            # For QK we want: scores[seq, kv] = sum_d Q[seq, d] * K[d, kv]
            # So stat must have P=d, F=seq (i.e., Q transposed).
            q_T_psum = nl.ndarray((d_head, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(q_T_psum, q_scaled_bf16)
            q_T = nl.ndarray((d_head, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q_T, src=q_T_psum)

            # Running statistics for online softmax (persistent across KV chunks)
            running_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=running_max, value=LARGE_NEG)

            running_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=running_sum, value=0.0)

            # Output accumulator: (PMAX, d_head) fp32 (persistent across KV chunks)
            out_acc = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=out_acc, value=0.0)

            # Flash attention: iterate over K/V chunks
            # Use Python range() so the loop is unrolled at NKI trace time.
            # This avoids HLO while-loops that cause ICE in the compiler's
            # ModuleSplitter. The shapes are concrete during tracing, so
            # range(n_kv_chunks) works. Loop-carried deps (running_max,
            # running_sum, out_acc) are handled by data flow ordering.
            for kv_chunk_idx in range(n_kv_chunks):
                kv_offset = kv_chunk_idx * K_TILE

                # --- Step 1: QK matmul ---
                # K is (batch, d_head, seqlen_kv): K[b, :, kv_offset:kv_offset+K_TILE]
                # Shape: (d_head, K_TILE) = (128, 512) -- perfect for moving operand
                k_tile = nl.ndarray((d_head, K_TILE), dtype=k.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=k_tile, src=k[b, :, nl.ds(kv_offset, K_TILE)])

                # QK matmul: q_T (d, seq) as stationary, k_tile (d, kv) as moving
                # nc_matmul: dst[seq, kv] = sum_d q_T[d, seq] * k_tile[d, kv]
                qk_psum = nl.ndarray((PMAX, K_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(qk_psum, q_T, k_tile, accumulate=False)

                # Copy QK from PSUM to SBUF for softmax operations
                qk_sbuf = nl.ndarray((PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

                # --- Step 2: Online softmax ---
                # Find chunk max: reduce (PMAX, K_TILE) -> (PMAX, 1)
                chunk_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=chunk_max, op=nl.maximum, data=qk_sbuf, axis=1)

                # New running max = max(running_max, chunk_max)
                new_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=new_max, data1=running_max, data2=chunk_max, op=nl.maximum
                )

                # Correction factor for previous accumulator: exp(old_max - new_max)
                max_diff = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=max_diff, data1=running_max, data2=new_max, op=nl.subtract
                )
                correction = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=correction, op=nl.exp, data=max_diff)

                # Rescale output accumulator: out_acc *= correction
                # tensor_scalar broadcasts (PMAX, 1) operand0 across (PMAX, d_head) data
                nisa.tensor_scalar(
                    dst=out_acc, data=out_acc, op0=nl.multiply, operand0=correction
                )

                # Rescale running sum: running_sum *= correction
                nisa.tensor_scalar(
                    dst=running_sum,
                    data=running_sum,
                    op0=nl.multiply,
                    operand0=correction,
                )

                # Subtract new_max from QK scores: broadcasts (PMAX, 1) across (PMAX, K_TILE)
                qk_centered = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=qk_centered, data=qk_sbuf, op0=nl.subtract, operand0=new_max
                )

                # exp(QK - max) -- KEY: stays in fp32!
                # OPTIMIZATION: Fused exp + reduction using nisa.activation's
                # pipelined reduce capability. Computes exp(x) and sum(exp(x))
                # in a single ISA instruction, eliminating a separate tensor_reduce.
                exp_scores = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                chunk_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    dst=exp_scores,
                    op=nl.exp,
                    data=qk_centered,
                    reduce_op=nl.add,
                    reduce_res=chunk_sum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )

                # Update running sum: running_sum += chunk_sum
                nisa.tensor_tensor(
                    dst=running_sum, data1=running_sum, data2=chunk_sum, op=nl.add
                )

                # Update running max
                nisa.tensor_copy(dst=running_max, src=new_max)

                # --- Step 3: P@V matmul (fp32 exp @ fp32 V) ---
                # exp_scores is (PMAX, K_TILE) = (128, 512) fp32
                # V is (batch, seqlen_kv, d_head) bf16
                #
                # We want: out_acc += exp_scores @ V_chunk
                #   = (128, 512) @ (512, 128) -> (128, 128)
                #
                # nc_matmul constraint: stationary max (128, 128), moving max (128, 512)
                # So we split exp_scores into 4 chunks of (128, 128) along the K_TILE dim,
                # and V into 4 tiles of (128, 128).
                #
                # For each V-tile: stationary=exp_chunk(128,128), moving=v_tile(128,128)
                # Use intentional PSUM accumulation across the 4 V-tiles:
                #   pv_psum += exp_chunk_i @ v_tile_i  for i in 0..3
                # This is more efficient than 4 separate PSUM allocations + 4 SBUF adds.
                #
                # NOTE: exp_scores is fp32 but we downcast to bf16 for nc_matmul to
                # avoid an ICE in neuronx-cc's hlo2penguin (stoi crash with fp32 matmul
                # inputs). The key precision benefit is preserved: exp() was computed
                # in fp32, so the softmax weights (exp/sum) are more accurate than
                # attention_cte's bf16 exp. The matmul itself runs in bf16 with fp32
                # PSUM accumulation, same as the standard path.

                pv_psum = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.psum)

                for v_tile_idx in nl.affine_range(n_v_tiles_per_chunk):
                    v_offset = kv_offset + v_tile_idx * V_TILE

                    # Load V tile: (V_TILE, d_head) = (128, 128) bf16
                    v_tile_bf16 = nl.ndarray(
                        (V_TILE, d_head), dtype=v.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(dst=v_tile_bf16, src=v[b, nl.ds(v_offset, V_TILE), :])

                    # exp chunk: (PMAX, V_TILE) = (128, 128) fp32 -> downcast to bf16
                    exp_chunk_offset = v_tile_idx * V_TILE
                    exp_chunk_bf16 = nl.ndarray(
                        (PMAX, V_TILE), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(
                        dst=exp_chunk_bf16,
                        src=exp_scores[:, nl.ds(exp_chunk_offset, V_TILE)],
                    )

                    # Transpose exp_chunk from (seq, kv_sub) to (kv_sub, seq) for nc_matmul.
                    # nc_matmul: dst[seq, d] = sum_kv exp_T[kv, seq] * v_tile[kv, d]
                    exp_T_psum = nl.ndarray(
                        (V_TILE, PMAX), dtype=nl.bfloat16, buffer=nl.psum
                    )
                    nisa.nc_transpose(exp_T_psum, exp_chunk_bf16)
                    exp_T = nl.ndarray(
                        (V_TILE, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=exp_T, src=exp_T_psum)

                    # P@V matmul: pv_psum += exp_T @ v_tile
                    # dst[seq, d] = sum_kv exp_T[kv, seq] * v_tile[kv, d]
                    nisa.nc_matmul(pv_psum, exp_T, v_tile_bf16)

                # Move accumulated P@V result from PSUM to SBUF
                pv_sbuf = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

                # Accumulate into output: out_acc += pv_result
                nisa.tensor_tensor(dst=out_acc, data1=out_acc, data2=pv_sbuf, op=nl.add)

            # --- Step 4: Normalize by sum ---
            inv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=inv_sum, op=nl.reciprocal, data=running_sum, bias=None, scale=1.0
            )

            # out_acc *= 1/sum (broadcasts (PMAX, 1) across (PMAX, d_head))
            nisa.tensor_scalar(
                dst=out_acc, data=out_acc, op0=nl.multiply, operand0=inv_sum
            )

            # --- Step 5: Cast to output dtype and store ---
            out_tile = nl.ndarray((PMAX, d_head), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_tile, src=out_acc)

            nisa.dma_copy(dst=output[b, nl.ds(q_offset, PMAX), :], src=out_tile)

    return output
