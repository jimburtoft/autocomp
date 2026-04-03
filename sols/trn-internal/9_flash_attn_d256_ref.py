"""
Flash attention for d=256 with causal masking (Qwen3.5).

Supports head_dim=256 by tiling the QK matmul contraction dimension in 2
chunks of 128. Uses the NKI affine_select for causal masking and online
softmax (FlashAttention algorithm).

Layout:
  Q: (bs, n_heads, 256, seq_q)  -- bfloat16
  K: (bs, nk_heads, 256, seq_k) -- bfloat16
  V: (bs, nv_heads, seq_v, 256) -- bfloat16
  Output: (bs, n_heads, seq_q, 256) -- bfloat16

ISA-level implementation: uses nisa.dma_copy, nisa.nc_matmul, nisa.tensor_copy,
nisa.tensor_scalar, nisa.tensor_tensor, nisa.tensor_reduce, nisa.activation,
nisa.activation_reduce, nisa.affine_select, nisa.nc_transpose, nisa.memset.
No mutation-style assignments (no `a[:,:] = expr`).
"""

import nki
import nki.language as nl
import nki.isa as nisa

B_P = 128  # partition dim max
B_F = 512  # free dim max for matmul moving operand
D_TILE = 128  # head_dim tile size (256 / 2)


@nki.jit
def flash_attn_d256(q, k, v, use_causal_mask=True):
    """
    Flash attention for head_dim=256.

    Args:
        q: (bs, n_heads, 256, seq_q) -- bfloat16
        k: (bs, nk_heads, 256, seq_k) -- bfloat16
        v: (bs, nv_heads, seq_v, 256) -- bfloat16
        use_causal_mask: bool

    Returns:
        o: (bs, n_heads, seq_q, 256) -- bfloat16

    The QK matmul is tiled: QK = Q0^T @ K0 + Q1^T @ K1
    where Q0/Q1 are the first/second 128 dims of Q along head_dim,
    and similarly for K0/K1.
    """
    b, h, d, seqlen_q = q.shape
    _, k_h, _, seqlen_k = k.shape
    assert d == 256
    assert seqlen_k % B_F == 0

    q_h_per_k_h = h // k_h

    o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    scale = 1.0 / (d**0.5)
    n_q_tiles = seqlen_q // B_P
    n_kv_tiles = seqlen_k // B_F
    NEG_INF = -9984.0

    for batch_id in nl.affine_range(b):
        for head_id in nl.affine_range(k_h):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                for qi in nl.sequential_range(n_q_tiles):
                    # Accumulators
                    o_acc = nl.ndarray(
                        (nl.par_dim(B_P), d), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.memset(o_acc, 0.0)
                    m_acc = nl.ndarray(
                        (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.memset(m_acc, NEG_INF)
                    l_acc = nl.ndarray(
                        (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.memset(l_acc, NEG_INF)

                    # Load Q tile: 2 chunks of (D_TILE, B_P)
                    # Q layout: (bs, n_heads, 256, seq_q) -- index directly from q
                    q_head = head_id * q_h_per_k_h + i_q_h

                    # Load q0 from HBM, then scale
                    q0_raw = nl.ndarray(
                        (D_TILE, B_P), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.dma_copy(
                        dst=q0_raw,
                        src=q[batch_id, q_head, nl.ds(0, D_TILE), nl.ds(qi * B_P, B_P)],
                    )
                    # Scale q0: upcast to fp32, multiply, downcast to bf16
                    q0_f32 = nl.ndarray((D_TILE, B_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(q0_f32, q0_raw, op0=nl.multiply, operand0=scale)
                    q0 = nl.ndarray((D_TILE, B_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=q0, src=q0_f32)

                    # Load q1 from HBM, then scale
                    q1_raw = nl.ndarray(
                        (D_TILE, B_P), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.dma_copy(
                        dst=q1_raw,
                        src=q[
                            batch_id,
                            q_head,
                            nl.ds(D_TILE, D_TILE),
                            nl.ds(qi * B_P, B_P),
                        ],
                    )
                    q1_f32 = nl.ndarray((D_TILE, B_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(q1_f32, q1_raw, op0=nl.multiply, operand0=scale)
                    q1 = nl.ndarray((D_TILE, B_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=q1, src=q1_f32)

                    for kvi in nl.sequential_range(n_kv_tiles):
                        # Causal: skip if Q tile is entirely before K tile
                        if use_causal_mask:
                            skip_condition = qi * B_P < kvi * B_F
                        else:
                            skip_condition = False

                        if not skip_condition:
                            # Load K: 2 chunks of (par_dim(D_TILE), B_F)
                            k0 = nl.ndarray(
                                (nl.par_dim(D_TILE), B_F),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            nisa.dma_copy(
                                dst=k0,
                                src=k[
                                    batch_id,
                                    head_id,
                                    nl.ds(0, D_TILE),
                                    nl.ds(kvi * B_F, B_F),
                                ],
                            )
                            k1 = nl.ndarray(
                                (nl.par_dim(D_TILE), B_F),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            nisa.dma_copy(
                                dst=k1,
                                src=k[
                                    batch_id,
                                    head_id,
                                    nl.ds(D_TILE, D_TILE),
                                    nl.ds(kvi * B_F, B_F),
                                ],
                            )

                            # Tiled QK matmul: q0^T @ k0 + q1^T @ k1
                            # nc_matmul accumulates into psum across calls
                            qk = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.psum
                            )
                            nisa.nc_matmul(qk, q0, k0)
                            # Second matmul accumulates into same psum tile
                            nisa.nc_matmul(qk, q1, k1)

                            # Move QK from PSUM to SBUF
                            qk_sbuf = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.sbuf
                            )

                            # Apply causal mask
                            if use_causal_mask:
                                i_q, i_k = nl.mgrid[0:B_P, 0:B_F]
                                q_pos = qi * B_P + i_q
                                k_pos = kvi * B_F + i_k
                                pred_causal = q_pos >= k_pos

                                nisa.affine_select(
                                    dst=qk_sbuf,
                                    pred=pred_causal,
                                    on_true_tile=qk,
                                    on_false_value=NEG_INF,
                                    dtype=nl.float32,
                                )
                            else:
                                nisa.tensor_copy(dst=qk_sbuf, src=qk)

                            # Row max
                            new_max = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_reduce(new_max, nl.maximum, qk_sbuf, axis=1)

                            # m_prev = copy of m_acc
                            m_prev = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_copy(dst=m_prev, src=m_acc)

                            # m_acc = max(m_prev, new_max)
                            nisa.tensor_tensor(m_acc, m_prev, new_max, op=nl.maximum)

                            # alpha = exp(m_prev - m_acc) for rescaling
                            # nisa.activation: dst = act_fn(src * scale + bias)
                            # We want exp(m_prev - m_cur) = exp(m_cur * (-1) + m_prev)
                            alpha = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.activation(
                                alpha, nl.exp, m_acc, bias=m_prev, scale=-1.0
                            )

                            # Rescale o_acc *= alpha
                            nisa.tensor_scalar(
                                o_acc, o_acc, op0=nl.multiply, operand0=alpha
                            )

                            # exp(qk - max) and row sum via activation_reduce
                            p = nl.ndarray(
                                (nl.par_dim(B_P), B_F),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            p_sum = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.activation_reduce(
                                dst=p,
                                act_fn=nl.exp,
                                src=qk_sbuf,
                                bias=-1.0 * m_acc,
                                scale=1.0,
                                reduce_op=nl.add,
                                reduce_res=p_sum,
                                dtype=nl.bfloat16,
                            )

                            # Load V: (n_v_sub, par_dim(B_P), d)
                            n_v_sub = B_F // B_P
                            v_tile = nl.ndarray(
                                (n_v_sub, nl.par_dim(B_P), d),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            for vi in nl.affine_range(n_v_sub):
                                nisa.dma_copy(
                                    dst=v_tile[vi],
                                    src=v[
                                        batch_id,
                                        head_id,
                                        nl.ds(kvi * B_F + vi * B_P, B_P),
                                        :,
                                    ],
                                )

                            # Transpose p for PV matmul: need (par_dim(B_P), B_F)
                            # p is already (par_dim(B_P), B_F) bf16 in sbuf
                            # nc_transpose transposes a (P, F) tile in SBUF through PSUM
                            p_t = nl.ndarray(
                                (nl.par_dim(B_P), B_F),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            for ti in nl.affine_range(B_F // B_P):
                                p_t_psum = nl.ndarray(
                                    (nl.par_dim(B_P), B_P),
                                    dtype=nl.float32,
                                    buffer=nl.psum,
                                )
                                nisa.nc_transpose(p_t_psum, p[:, nl.ds(ti * B_P, B_P)])
                                p_t_chunk = nl.ndarray(
                                    (nl.par_dim(B_P), B_P),
                                    dtype=nl.bfloat16,
                                    buffer=nl.sbuf,
                                )
                                nisa.tensor_copy(dst=p_t_chunk, src=p_t_psum)
                                nisa.tensor_copy(
                                    dst=p_t[:, nl.ds(ti * B_P, B_P)],
                                    src=p_t_chunk,
                                )

                            # PV matmul: (B_P, B_F) @ (B_F, 256) -> (B_P, 256)
                            # Tiled across n_v_sub sub-tiles
                            pv = nl.ndarray(
                                (nl.par_dim(B_P), d), dtype=nl.float32, buffer=nl.psum
                            )
                            nisa.memset(pv, 0.0)
                            for vi in nl.affine_range(n_v_sub):
                                nisa.nc_matmul(
                                    pv,
                                    p_t[:, nl.ds(vi * B_P, B_P)],
                                    v_tile[vi],
                                )

                            # o_acc += pv (move pv from PSUM to SBUF first)
                            pv_sbuf = nl.ndarray(
                                (nl.par_dim(B_P), d), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_copy(dst=pv_sbuf, src=pv)
                            nisa.tensor_tensor(o_acc, o_acc, pv_sbuf, op=nl.add)

                            # Update log-sum-exp: l_acc = m_cur + log(exp(l_prev - m_cur) + p_sum)
                            # exp_l = exp(m_acc * (-1) + l_acc) = exp(l_acc - m_acc)
                            exp_l = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.activation(
                                exp_l, nl.exp, m_acc, bias=l_acc, scale=-1.0
                            )
                            # log_arg = exp_l + p_sum
                            log_arg = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_tensor(log_arg, exp_l, p_sum, op=nl.add)
                            # log_val = log(log_arg)
                            log_val = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.activation(log_val, nl.log, log_arg)
                            # l_acc = m_acc + log_val
                            nisa.tensor_tensor(l_acc, m_acc, log_val, op=nl.add)

                    # Final rescale: out = o_acc * exp(m_acc - l_acc)
                    # final_exp = exp(l_acc * (-1) + m_acc) = exp(m_acc - l_acc)
                    final_exp = nl.ndarray(
                        (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.activation(final_exp, nl.exp, l_acc, bias=m_acc, scale=-1.0)

                    # out = o_acc * final_exp, cast to bf16
                    nisa.tensor_scalar(
                        o_acc, o_acc, op0=nl.multiply, operand0=final_exp
                    )
                    out = nl.ndarray(
                        (nl.par_dim(B_P), d), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=out, src=o_acc)

                    # Store to HBM
                    nisa.dma_copy(
                        dst=o[
                            batch_id,
                            head_id * q_h_per_k_h + i_q_h,
                            nl.ds(qi * B_P, B_P),
                            :,
                        ],
                        src=out,
                    )

    return o
