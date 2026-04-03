import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_xla.core import xla_model as xm


B_P = 128  # partition dim max
B_F = 512  # free dim max for matmul moving operand
D_TILE = 128  # head_dim tile size (256 / 2)


@nki.jit
def ref(q, k, v, use_causal_mask=True):
    """Reference flash attention for head_dim=256 (causal), ISA-level style."""
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

                    q_head = head_id * q_h_per_k_h + i_q_h

                    # Load and scale q0
                    q0_raw = nl.ndarray(
                        (D_TILE, B_P), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.dma_copy(
                        dst=q0_raw,
                        src=q[batch_id, q_head, nl.ds(0, D_TILE), nl.ds(qi * B_P, B_P)],
                    )
                    q0_f32 = nl.ndarray((D_TILE, B_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(q0_f32, q0_raw, op0=nl.multiply, operand0=scale)
                    q0 = nl.ndarray((D_TILE, B_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=q0, src=q0_f32)

                    # Load and scale q1
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
                        if use_causal_mask:
                            skip_condition = qi * B_P < kvi * B_F
                        else:
                            skip_condition = False

                        if not skip_condition:
                            # Load K chunks
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

                            # Tiled QK matmul (accumulates in psum)
                            qk = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.psum
                            )
                            nisa.nc_matmul(qk, q0, k0)
                            nisa.nc_matmul(qk, q1, k1)

                            # Move to SBUF with causal mask
                            qk_sbuf = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.sbuf
                            )

                            if use_causal_mask:
                                qk_sbuf_raw = nl.ndarray(
                                    (nl.par_dim(B_P), B_F),
                                    dtype=nl.float32,
                                    buffer=nl.sbuf,
                                )
                                nisa.tensor_copy(dst=qk_sbuf_raw, src=qk)
                                nisa.affine_select(
                                    dst=qk_sbuf,
                                    pattern=[[0, 1], [0, 1], [0, 1], [-1, B_F]],
                                    offset=qi * B_P - kvi * B_F,
                                    channel_multiplier=1,
                                    on_true_tile=qk_sbuf_raw,
                                    on_false_value=NEG_INF,
                                    cmp_op=nl.greater_equal,
                                )
                            else:
                                nisa.tensor_copy(dst=qk_sbuf, src=qk)

                            # Row max
                            new_max = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_reduce(new_max, nl.maximum, qk_sbuf, axis=1)

                            # m_prev = copy(m_acc), m_acc = max(m_prev, new_max)
                            m_prev = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_copy(dst=m_prev, src=m_acc)
                            nisa.tensor_tensor(m_acc, m_prev, new_max, op=nl.maximum)

                            # alpha = exp(m_prev - m_acc)
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

                            # exp(qk - max) and row sum
                            p = nl.ndarray(
                                (nl.par_dim(B_P), B_F),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            p_sum = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            neg_m_acc = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_scalar(
                                neg_m_acc, m_acc, op0=nl.multiply, operand0=-1.0
                            )
                            nisa.activation_reduce(
                                p,
                                nl.exp,
                                qk_sbuf,
                                nl.add,
                                p_sum,
                                bias=neg_m_acc,
                                scale=1.0,
                            )

                            # Load V tiles
                            n_v_sub = B_F // B_P
                            v_tiles = []
                            for vi in range(n_v_sub):
                                vt = nl.ndarray(
                                    (nl.par_dim(B_P), d),
                                    dtype=nl.bfloat16,
                                    buffer=nl.sbuf,
                                )
                                nisa.dma_copy(
                                    dst=vt,
                                    src=v[
                                        batch_id,
                                        head_id,
                                        nl.ds(kvi * B_F + vi * B_P, B_P),
                                        :,
                                    ],
                                )
                                v_tiles.append(vt)

                            # Transpose p for PV matmul
                            p_t = nl.ndarray(
                                (nl.par_dim(B_P), B_F),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            for ti in nl.affine_range(B_F // B_P):
                                p_t_psum = nl.ndarray(
                                    (nl.par_dim(B_P), B_P),
                                    dtype=nl.bfloat16,
                                    buffer=nl.psum,
                                )
                                nisa.nc_transpose(p_t_psum, p[:, nl.ds(ti * B_P, B_P)])
                                nisa.tensor_copy(
                                    dst=p_t[:, nl.ds(ti * B_P, B_P)], src=p_t_psum
                                )

                            # PV matmul
                            pv = nl.ndarray(
                                (nl.par_dim(B_P), d), dtype=nl.float32, buffer=nl.psum
                            )
                            nisa.memset(pv, 0.0)
                            for vi in range(n_v_sub):
                                nisa.nc_matmul(
                                    pv, p_t[:, nl.ds(vi * B_P, B_P)], v_tiles[vi]
                                )

                            # o_acc += pv
                            pv_sbuf = nl.ndarray(
                                (nl.par_dim(B_P), d), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_copy(dst=pv_sbuf, src=pv)
                            nisa.tensor_tensor(o_acc, o_acc, pv_sbuf, op=nl.add)

                            # Update log-sum-exp
                            exp_l = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.activation(
                                exp_l, nl.exp, m_acc, bias=l_acc, scale=-1.0
                            )
                            log_arg = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_tensor(log_arg, exp_l, p_sum, op=nl.add)
                            log_val = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.activation(log_val, nl.log, log_arg)
                            nisa.tensor_tensor(l_acc, m_acc, log_val, op=nl.add)

                    # Final rescale and store
                    final_exp = nl.ndarray(
                        (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.activation(final_exp, nl.exp, l_acc, bias=m_acc, scale=-1.0)
                    nisa.tensor_scalar(
                        o_acc, o_acc, op0=nl.multiply, operand0=final_exp
                    )
                    out = nl.ndarray(
                        (nl.par_dim(B_P), d), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=out, src=o_acc)
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


# SUBSTITUTE HERE


@nki.jit
def test(q, k, v, use_causal_mask=True):
    """Test flash attention for head_dim=256 (causal), ISA-level style."""
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

                    q_head = head_id * q_h_per_k_h + i_q_h

                    # Load and scale q0
                    q0_raw = nl.ndarray(
                        (D_TILE, B_P), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.dma_copy(
                        dst=q0_raw,
                        src=q[batch_id, q_head, nl.ds(0, D_TILE), nl.ds(qi * B_P, B_P)],
                    )
                    q0_f32 = nl.ndarray((D_TILE, B_P), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(q0_f32, q0_raw, op0=nl.multiply, operand0=scale)
                    q0 = nl.ndarray((D_TILE, B_P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=q0, src=q0_f32)

                    # Load and scale q1
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
                        if use_causal_mask:
                            skip_condition = qi * B_P < kvi * B_F
                        else:
                            skip_condition = False

                        if not skip_condition:
                            # Load K chunks
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

                            # Tiled QK matmul
                            qk = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.psum
                            )
                            nisa.nc_matmul(qk, q0, k0)
                            nisa.nc_matmul(qk, q1, k1)

                            # Move to SBUF with causal mask
                            qk_sbuf = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.sbuf
                            )

                            if use_causal_mask:
                                qk_sbuf_raw = nl.ndarray(
                                    (nl.par_dim(B_P), B_F),
                                    dtype=nl.float32,
                                    buffer=nl.sbuf,
                                )
                                nisa.tensor_copy(dst=qk_sbuf_raw, src=qk)
                                nisa.affine_select(
                                    dst=qk_sbuf,
                                    pattern=[[0, 1], [0, 1], [0, 1], [-1, B_F]],
                                    offset=qi * B_P - kvi * B_F,
                                    channel_multiplier=1,
                                    on_true_tile=qk_sbuf_raw,
                                    on_false_value=NEG_INF,
                                    cmp_op=nl.greater_equal,
                                )
                            else:
                                nisa.tensor_copy(dst=qk_sbuf, src=qk)

                            # Row max
                            new_max = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_reduce(new_max, nl.maximum, qk_sbuf, axis=1)

                            m_prev = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_copy(dst=m_prev, src=m_acc)
                            nisa.tensor_tensor(m_acc, m_prev, new_max, op=nl.maximum)

                            alpha = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.activation(
                                alpha, nl.exp, m_acc, bias=m_prev, scale=-1.0
                            )

                            nisa.tensor_scalar(
                                o_acc, o_acc, op0=nl.multiply, operand0=alpha
                            )

                            p = nl.ndarray(
                                (nl.par_dim(B_P), B_F),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            p_sum = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            neg_m_acc = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_scalar(
                                neg_m_acc, m_acc, op0=nl.multiply, operand0=-1.0
                            )
                            nisa.activation_reduce(
                                p,
                                nl.exp,
                                qk_sbuf,
                                nl.add,
                                p_sum,
                                bias=neg_m_acc,
                                scale=1.0,
                            )

                            # Load V tiles
                            n_v_sub = B_F // B_P
                            v_tiles = []
                            for vi in range(n_v_sub):
                                vt = nl.ndarray(
                                    (nl.par_dim(B_P), d),
                                    dtype=nl.bfloat16,
                                    buffer=nl.sbuf,
                                )
                                nisa.dma_copy(
                                    dst=vt,
                                    src=v[
                                        batch_id,
                                        head_id,
                                        nl.ds(kvi * B_F + vi * B_P, B_P),
                                        :,
                                    ],
                                )
                                v_tiles.append(vt)

                            # Transpose p
                            p_t = nl.ndarray(
                                (nl.par_dim(B_P), B_F),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            for ti in nl.affine_range(B_F // B_P):
                                p_t_psum = nl.ndarray(
                                    (nl.par_dim(B_P), B_P),
                                    dtype=nl.bfloat16,
                                    buffer=nl.psum,
                                )
                                nisa.nc_transpose(p_t_psum, p[:, nl.ds(ti * B_P, B_P)])
                                nisa.tensor_copy(
                                    dst=p_t[:, nl.ds(ti * B_P, B_P)], src=p_t_psum
                                )

                            # PV matmul
                            pv = nl.ndarray(
                                (nl.par_dim(B_P), d), dtype=nl.float32, buffer=nl.psum
                            )
                            nisa.memset(pv, 0.0)
                            for vi in range(n_v_sub):
                                nisa.nc_matmul(
                                    pv, p_t[:, nl.ds(vi * B_P, B_P)], v_tiles[vi]
                                )

                            pv_sbuf = nl.ndarray(
                                (nl.par_dim(B_P), d), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_copy(dst=pv_sbuf, src=pv)
                            nisa.tensor_tensor(o_acc, o_acc, pv_sbuf, op=nl.add)

                            # Update log-sum-exp
                            exp_l = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.activation(
                                exp_l, nl.exp, m_acc, bias=l_acc, scale=-1.0
                            )
                            log_arg = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.tensor_tensor(log_arg, exp_l, p_sum, op=nl.add)
                            log_val = nl.ndarray(
                                (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                            )
                            nisa.activation(log_val, nl.log, log_arg)
                            nisa.tensor_tensor(l_acc, m_acc, log_val, op=nl.add)

                    # Final rescale and store
                    final_exp = nl.ndarray(
                        (nl.par_dim(B_P), 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.activation(final_exp, nl.exp, l_acc, bias=m_acc, scale=-1.0)
                    nisa.tensor_scalar(
                        o_acc, o_acc, op0=nl.multiply, operand0=final_exp
                    )
                    out = nl.ndarray(
                        (nl.par_dim(B_P), d), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=out, src=o_acc)
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


# ==============================================================================
# CPU reference
# ==============================================================================


def reference_causal_attention_cpu(q, k, v):
    """CPU reference: q(b,h,d,sq), k(b,h,d,sk), v(b,h,sk,d) -> (b,h,sq,d)

    All inputs/outputs are torch float32 tensors.
    """
    import torch.nn.functional as F

    d = q.shape[2]
    q_t = q.permute(0, 1, 3, 2).float()  # (b,h,sq,d)
    k_t = k.permute(0, 1, 3, 2).float()  # (b,h,sk,d)
    v_t = v.float()  # (b,h,sk,d)
    scale = 1.0 / (d**0.5)
    attn = q_t @ k_t.transpose(-2, -1) * scale
    mask = torch.triu(
        torch.ones(q_t.shape[2], k_t.shape[2], dtype=torch.bool), diagonal=1
    )
    attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return attn @ v_t


# ==============================================================================
# Correctness check
# ==============================================================================


def test_nki(ref_func, test_func):
    """Correctness check: compare ref and test vs CPU causal attention."""
    device = xm.xla_device()

    bs = 1
    heads = 1
    d = 256
    seqlen = 512  # 4 Q tiles, 1 KV tile

    for seed in range(2):
        np.random.seed(42 + seed)
        # Q: (bs, heads, d, seqlen) -- bf16 layout
        q_np = (np.random.randn(bs, heads, d, seqlen) * 0.1).astype(np.float32)
        # K: (bs, heads, d, seqlen) -- bf16 layout
        k_np = (np.random.randn(bs, heads, d, seqlen) * 0.1).astype(np.float32)
        # V: (bs, heads, seqlen, d) -- bf16 layout
        v_np = (np.random.randn(bs, heads, seqlen, d) * 0.1).astype(np.float32)

        # CPU reference
        cpu_out = reference_causal_attention_cpu(
            torch.tensor(q_np), torch.tensor(k_np), torch.tensor(v_np)
        ).numpy()

        # NKI on device (bf16 inputs)
        q_dev = torch.tensor(q_np, dtype=torch.bfloat16, device=device)
        k_dev = torch.tensor(k_np, dtype=torch.bfloat16, device=device)
        v_dev = torch.tensor(v_np, dtype=torch.bfloat16, device=device)

        result_ref = ref_func(q_dev, k_dev, v_dev, use_causal_mask=True)
        result_test = test_func(q_dev, k_dev, v_dev, use_causal_mask=True)

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


# ==============================================================================
# Benchmark
# ==============================================================================


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    bs = 1
    heads = 1
    d = 256
    seqlen = 512

    np.random.seed(42)
    q_np = (np.random.randn(bs, heads, d, seqlen) * 0.1).astype(np.float32)
    k_np = (np.random.randn(bs, heads, d, seqlen) * 0.1).astype(np.float32)
    v_np = (np.random.randn(bs, heads, seqlen, d) * 0.1).astype(np.float32)

    q_dev = torch.tensor(q_np, dtype=torch.bfloat16, device=device)
    k_dev = torch.tensor(k_np, dtype=torch.bfloat16, device=device)
    v_dev = torch.tensor(v_np, dtype=torch.bfloat16, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(q_dev, k_dev, v_dev, use_causal_mask=True)
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
