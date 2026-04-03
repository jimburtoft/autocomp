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
"""

import numpy as np
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
                    o_acc = nl.zeros(
                        (nl.par_dim(B_P), d), dtype=np.float32, buffer=nl.sbuf
                    )
                    m_acc = nl.full(
                        (nl.par_dim(B_P), 1), fill_value=NEG_INF, dtype=np.float32
                    )
                    l_acc = nl.full(
                        (nl.par_dim(B_P), 1), fill_value=NEG_INF, dtype=np.float32
                    )

                    # Load Q tile: 2 chunks of (128, 128)
                    q_hbm = q[batch_id, head_id * q_h_per_k_h + i_q_h]
                    q0 = nl.ndarray((D_TILE, B_P), dtype=nl.bfloat16)
                    q0[:, :] = (
                        nl.load(q_hbm[nl.ds(0, D_TILE), nl.ds(qi * B_P, B_P)]) * scale
                    )
                    q1 = nl.ndarray((D_TILE, B_P), dtype=nl.bfloat16)
                    q1[:, :] = (
                        nl.load(q_hbm[nl.ds(D_TILE, D_TILE), nl.ds(qi * B_P, B_P)])
                        * scale
                    )

                    for kvi in nl.sequential_range(n_kv_tiles):
                        # Causal: skip if Q tile is entirely before K tile
                        if use_causal_mask:
                            skip_condition = qi * B_P < kvi * B_F
                        else:
                            skip_condition = False

                        if not skip_condition:
                            # Load K: 2 chunks of (par_dim(128), 512)
                            k0 = nl.ndarray(
                                (nl.par_dim(D_TILE), B_F), dtype=nl.bfloat16
                            )
                            k0[:, :] = nl.load(
                                k[
                                    batch_id,
                                    head_id,
                                    nl.ds(0, D_TILE),
                                    nl.ds(kvi * B_F, B_F),
                                ]
                            )
                            k1 = nl.ndarray(
                                (nl.par_dim(D_TILE), B_F), dtype=nl.bfloat16
                            )
                            k1[:, :] = nl.load(
                                k[
                                    batch_id,
                                    head_id,
                                    nl.ds(D_TILE, D_TILE),
                                    nl.ds(kvi * B_F, B_F),
                                ]
                            )

                            # Tiled QK matmul: (128,128)^T @ (p128,512) -> (p128,512) accumulated
                            qk = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=np.float32, buffer=nl.psum
                            )
                            qk[:, :] = nl.matmul(q0, k0, transpose_x=True)
                            qk[:, :] += nl.matmul(q1, k1, transpose_x=True)

                            # Move to SBUF for masking
                            qk_sbuf = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=np.float32, buffer=nl.sbuf
                            )

                            # Apply causal mask
                            if use_causal_mask:
                                i_q, i_k = nl.mgrid[0:B_P, 0:B_F]
                                q_pos = qi * B_P + i_q
                                k_pos = kvi * B_F + i_k
                                pred_causal = q_pos >= k_pos

                                qk_sbuf[:, :] = nisa.affine_select(
                                    pred=pred_causal,
                                    on_true_tile=qk,
                                    on_false_value=NEG_INF,
                                    dtype=np.float32,
                                )
                            else:
                                qk_sbuf[:, :] = nl.copy(qk, dtype=np.float32)

                            # Row max
                            new_max = nisa.tensor_reduce(
                                np.max,
                                qk_sbuf,
                                axis=(1,),
                                dtype=np.float32,
                                negate=False,
                            )

                            m_prev = nl.copy(m_acc[:, 0])
                            m_acc[:, 0] = nl.maximum(m_prev, new_max)
                            m_cur = m_acc[:, 0]

                            # Rescale previous output
                            alpha = nisa.activation(
                                np.exp, m_cur, bias=m_prev, scale=-1.0
                            )
                            o_acc[...] = nl.multiply(o_acc, alpha)

                            # exp(qk - max) and row sum
                            p = nl.ndarray((nl.par_dim(B_P), B_F), dtype=nl.bfloat16)
                            p_sum = nl.ndarray((nl.par_dim(B_P), 1), dtype=np.float32)
                            p[:, :] = nisa.activation_reduce(
                                np.exp,
                                qk_sbuf,
                                bias=-1 * m_cur,
                                scale=1.0,
                                reduce_op=nl.add,
                                reduce_res=p_sum[:, 0],
                                dtype=nl.bfloat16,
                            )

                            # Load V: (B_F//B_P, par_dim(B_P), 256)
                            n_v_sub = B_F // B_P
                            v_tile = nl.ndarray(
                                (n_v_sub, nl.par_dim(B_P), d), dtype=nl.bfloat16
                            )
                            for vi in nl.affine_range(n_v_sub):
                                v_tile[vi, :, :] = nl.load(
                                    v[
                                        batch_id,
                                        head_id,
                                        nl.ds(kvi * B_F + vi * B_P, B_P),
                                        :,
                                    ],
                                    dtype=nl.bfloat16,
                                )

                            # Transpose p for PV matmul
                            p_t = nl.ndarray((nl.par_dim(B_P), B_F), dtype=nl.bfloat16)
                            for ti in nl.affine_range(B_F // B_P):
                                p_t_tmp = nl.ndarray(
                                    (nl.par_dim(B_P), B_P),
                                    dtype=np.float32,
                                    buffer=nl.psum,
                                )
                                p_t_tmp[:, :] = nisa.nc_transpose(
                                    p[:, nl.ds(ti * B_P, B_P)]
                                )
                                p_t[:, nl.ds(ti * B_P, B_P)] = nl.copy(
                                    p_t_tmp, dtype=nl.bfloat16
                                )

                            # PV matmul: (B_P, B_F) @ (B_F, 256) -> (B_P, 256) in PSUM
                            pv = nl.zeros(
                                (nl.par_dim(B_P), d),
                                dtype=np.float32,
                                buffer=nl.psum,
                                lazy_initialization=True,
                            )
                            for vi in nl.affine_range(n_v_sub):
                                pv[:, :] += nl.matmul(
                                    p_t[:, nl.ds(vi * B_P, B_P)],
                                    v_tile[vi, :, :],
                                    transpose_x=True,
                                )

                            o_acc[:, :] = nl.add(o_acc, pv)

                            # Update log-sum-exp
                            exp_l = nisa.activation(
                                nl.exp, m_cur, bias=l_acc[:, 0], scale=-1.0
                            )
                            l_acc[:, 0] = nl.add(
                                m_cur, nisa.activation(nl.log, exp_l, bias=p_sum[:, 0])
                            )

                    # Final rescale and store
                    final_exp = nisa.activation(
                        np.exp, l_acc[:, 0], bias=m_acc[:, 0], scale=-1.0
                    )
                    out = nl.multiply(o_acc, final_exp, dtype=nl.bfloat16)
                    nl.store(
                        o[
                            batch_id,
                            head_id * q_h_per_k_h + i_q_h,
                            nl.ds(qi * B_P, B_P),
                            :,
                        ],
                        out,
                    )

    return o
