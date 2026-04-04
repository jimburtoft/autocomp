import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_xla.core import xla_model as xm


# Constants matching the ref kernel
P_MAX = 128
D_MODEL = 1280
N_HEADS = 20
HEAD_DIM = 64
MLP_DIM = 5120
N_HIDDEN = D_MODEL // P_MAX  # 10
N_MLP = MLP_DIM // P_MAX  # 40

# Test uses fewer seq tiles for speed (3 tiles = 384 positions)
TEST_N_SEQ_TILES = 3
TEST_SEQ_PAD = TEST_N_SEQ_TILES * P_MAX  # 384


def _transpose_to_sbuf(x):
    """nc_transpose x from SBUF -> PSUM -> SBUF."""
    x_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=x_t_psum, data=x)
    x_t = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x_t, src=x_t_psum)
    return x_t


def _prepare_weight(w_hbm):
    """Load and transpose a [P_MAX, P_MAX] weight tile from HBM."""
    w = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=w_hbm)
    w_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=w_t_psum, data=w)
    w_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w_t, src=w_t_psum)
    return w_t


def _layer_norm_tile(x_tile, weight_tiled, bias_tiled, F, eps=1e-5):
    """LayerNorm on a [P_MAX, F] tile entirely in SBUF."""
    inv_F = 1.0 / float(F)

    x_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=x_f32, src=x_tile)

    sum_x = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sum_x, op=nl.add, data=x_f32, axis=1)
    mean = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mean, data=sum_x, op0=nl.multiply, operand0=inv_F, engine=nisa.vector_engine
    )

    centered = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=centered,
        data=x_f32,
        op0=nl.subtract,
        operand0=mean,
        engine=nisa.vector_engine,
    )

    sq = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=centered, data2=centered, op=nl.multiply)
    sum_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sum_sq, op=nl.add, data=sq, axis=1)
    var = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=var, data=sum_sq, op0=nl.multiply, operand0=inv_F, engine=nisa.vector_engine
    )

    var_eps = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=var_eps, data=var, op0=nl.add, operand0=eps, engine=nisa.vector_engine
    )
    rsqrt_std = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rsqrt_std, op=nl.rsqrt, data=var_eps, bias=None, scale=1.0)

    norm_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=norm_f32,
        data=centered,
        op0=nl.multiply,
        operand0=rsqrt_std,
        engine=nisa.vector_engine,
    )

    w_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=w_f32, src=weight_tiled)
    scaled = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scaled, data1=norm_f32, data2=w_f32, op=nl.multiply)

    b_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b_f32, src=bias_tiled)
    result_f32 = nl.ndarray((P_MAX, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=result_f32, data1=scaled, data2=b_f32, op=nl.add)

    result = nl.ndarray((P_MAX, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result, src=result_f32)
    return result


def _attention_for_seq_tile(
    q_hbm,
    k_hbm,
    v_hbm,
    out_hbm,
    j_tile,
    n_seq_tiles,
    n_heads,
    head_dim,
    scale,
):
    """Bidirectional flash attention for query tile j_tile, all heads."""
    Hd = n_heads * head_dim
    j_start = j_tile * P_MAX

    for h in nl.affine_range(n_heads):
        hd_start = h * head_dim

        q_tile = nl.ndarray((P_MAX, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_tile,
            src=q_hbm[j_start : j_start + P_MAX, hd_start : hd_start + head_dim],
        )

        q_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=q_padded, value=0.0)
        nisa.tensor_copy(dst=q_padded[0:P_MAX, 0:head_dim], src=q_tile)
        q_t = _transpose_to_sbuf(q_padded)

        m_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=m_prev, value=-1e30)
        l_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=l_prev, value=0.0)
        o_acc = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=o_acc, value=0.0)

        for k_tile_idx in nl.sequential_range(n_seq_tiles):
            k_start = k_tile_idx * P_MAX

            k_tile_sb = nl.ndarray((P_MAX, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_tile_sb,
                src=k_hbm[k_start : k_start + P_MAX, hd_start : hd_start + head_dim],
            )

            v_tile_sb = nl.ndarray((P_MAX, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=v_tile_sb,
                src=v_hbm[k_start : k_start + P_MAX, hd_start : hd_start + head_dim],
            )

            k_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=k_padded, value=0.0)
            nisa.tensor_copy(dst=k_padded[0:P_MAX, 0:head_dim], src=k_tile_sb)
            k_t = _transpose_to_sbuf(k_padded)

            logits_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=logits_psum, stationary=q_t, moving=k_t)
            logits = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=logits, src=logits_psum)

            logits_scaled = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=logits_scaled,
                data=logits,
                op0=nl.multiply,
                operand0=scale,
                engine=nisa.vector_engine,
            )

            tile_max = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=logits_scaled, axis=1)

            m_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m_new, data1=m_prev, data2=tile_max, op=nl.maximum)

            m_diff = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m_diff, data1=m_prev, data2=m_new, op=nl.subtract)
            correction = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=correction, op=nl.exp, data=m_diff, bias=None, scale=1.0
            )

            logits_shifted = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=logits_shifted,
                data=logits_scaled,
                op0=nl.subtract,
                operand0=m_new,
                engine=nisa.vector_engine,
            )
            exp_logits = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=exp_logits, op=nl.exp, data=logits_shifted, bias=None, scale=1.0
            )

            l_corrected = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=l_corrected, data1=l_prev, data2=correction, op=nl.multiply
            )
            tile_sum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_logits, axis=1)
            l_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=l_new, data1=l_corrected, data2=tile_sum, op=nl.add)

            o_scaled = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=o_scaled,
                data=o_acc,
                op0=nl.multiply,
                operand0=correction,
                engine=nisa.vector_engine,
            )

            exp_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=exp_bf16, src=exp_logits)
            exp_t = _transpose_to_sbuf(exp_bf16)

            pv_psum = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=pv_psum, stationary=exp_t, moving=v_tile_sb)
            pv_sbuf = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

            nisa.tensor_tensor(dst=o_acc, data1=o_scaled, data2=pv_sbuf, op=nl.add)

            nisa.tensor_copy(dst=m_prev, src=m_new)
            nisa.tensor_copy(dst=l_prev, src=l_new)

        inv_l = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_l, data=l_prev)
        o_final = nl.ndarray((P_MAX, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=o_final,
            data=o_acc,
            op0=nl.multiply,
            operand0=inv_l,
            engine=nisa.vector_engine,
        )

        o_out = nl.ndarray((P_MAX, head_dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_out, src=o_final)
        nisa.dma_copy(
            dst=out_hbm[j_start : j_start + P_MAX, hd_start : hd_start + head_dim],
            src=o_out,
        )


def _encoder_layer_body(
    x_in,
    attn_ln_w,
    attn_ln_b,
    qkv_w,
    qkv_b,
    out_w,
    out_b,
    mlp_ln_w,
    mlp_ln_b,
    fc1_w,
    fc1_b,
    fc2_w,
    fc2_b,
    n_seq_tiles,
    n_heads,
    head_dim,
    eps,
):
    """Shared encoder layer body used by both ref and test."""
    S = n_seq_tiles * P_MAX
    Hd = n_heads * head_dim
    scale = 1.0 / (head_dim**0.5)

    x_out = nl.ndarray((S, D_MODEL), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    q_buf = nl.ndarray((S, Hd), dtype=nl.bfloat16, buffer=nl.private_hbm)
    k_buf = nl.ndarray((S, Hd), dtype=nl.bfloat16, buffer=nl.private_hbm)
    v_buf = nl.ndarray((S, Hd), dtype=nl.bfloat16, buffer=nl.private_hbm)
    attn_out_buf = nl.ndarray((S, Hd), dtype=nl.bfloat16, buffer=nl.private_hbm)
    post_attn_buf = nl.ndarray((S, D_MODEL), dtype=nl.bfloat16, buffer=nl.private_hbm)

    ln1_w = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln1_w, src=attn_ln_w)
    ln1_b = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln1_b, src=attn_ln_b)

    # Phase 1-2: LayerNorm + QKV projection
    for s_tile in nl.sequential_range(n_seq_tiles):
        s_start = s_tile * P_MAX

        x_tile = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x_in[s_start : s_start + P_MAX, 0:D_MODEL])

        x_normed = _layer_norm_tile(x_tile, ln1_w, ln1_b, D_MODEL, eps)

        xc_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_0, src=x_normed[0:P_MAX, 0:P_MAX])
        xc_t_0 = _transpose_to_sbuf(xc_0)

        xc_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_1, src=x_normed[0:P_MAX, P_MAX : 2 * P_MAX])
        xc_t_1 = _transpose_to_sbuf(xc_1)

        xc_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_2, src=x_normed[0:P_MAX, 2 * P_MAX : 3 * P_MAX])
        xc_t_2 = _transpose_to_sbuf(xc_2)

        xc_3 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_3, src=x_normed[0:P_MAX, 3 * P_MAX : 4 * P_MAX])
        xc_t_3 = _transpose_to_sbuf(xc_3)

        xc_4 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_4, src=x_normed[0:P_MAX, 4 * P_MAX : 5 * P_MAX])
        xc_t_4 = _transpose_to_sbuf(xc_4)

        xc_5 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_5, src=x_normed[0:P_MAX, 5 * P_MAX : 6 * P_MAX])
        xc_t_5 = _transpose_to_sbuf(xc_5)

        xc_6 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_6, src=x_normed[0:P_MAX, 6 * P_MAX : 7 * P_MAX])
        xc_t_6 = _transpose_to_sbuf(xc_6)

        xc_7 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_7, src=x_normed[0:P_MAX, 7 * P_MAX : 8 * P_MAX])
        xc_t_7 = _transpose_to_sbuf(xc_7)

        xc_8 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_8, src=x_normed[0:P_MAX, 8 * P_MAX : 9 * P_MAX])
        xc_t_8 = _transpose_to_sbuf(xc_8)

        xc_9 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc_9, src=x_normed[0:P_MAX, 9 * P_MAX : 10 * P_MAX])
        xc_t_9 = _transpose_to_sbuf(xc_9)

        xc_t = (
            xc_t_0,
            xc_t_1,
            xc_t_2,
            xc_t_3,
            xc_t_4,
            xc_t_5,
            xc_t_6,
            xc_t_7,
            xc_t_8,
            xc_t_9,
        )

        # Q projection
        for o in nl.static_range(N_HIDDEN):
            q_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=q_acc, value=0.0)
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    qkv_w[o * P_MAX : (o + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=xc_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=q_acc, data1=q_acc, data2=ps, op=nl.add)
            q_out = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q_out, src=q_acc)
            q_bias = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_bias, src=qkv_b[0:P_MAX, o * P_MAX : (o + 1) * P_MAX])
            q_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=q_biased, data1=q_out, data2=q_bias, op=nl.add)
            nisa.dma_copy(
                dst=q_buf[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=q_biased,
            )

        # K projection
        for o in nl.static_range(N_HIDDEN):
            k_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=k_acc, value=0.0)
            w_row_off = D_MODEL + o * P_MAX
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    qkv_w[w_row_off : w_row_off + P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=xc_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=k_acc, data1=k_acc, data2=ps, op=nl.add)
            k_out = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_out, src=k_acc)
            k_bias = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_bias,
                src=qkv_b[0:P_MAX, D_MODEL + o * P_MAX : D_MODEL + (o + 1) * P_MAX],
            )
            k_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=k_biased, data1=k_out, data2=k_bias, op=nl.add)
            nisa.dma_copy(
                dst=k_buf[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=k_biased,
            )

        # V projection
        for o in nl.static_range(N_HIDDEN):
            v_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=v_acc, value=0.0)
            w_row_off = 2 * D_MODEL + o * P_MAX
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    qkv_w[w_row_off : w_row_off + P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=xc_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=v_acc, data1=v_acc, data2=ps, op=nl.add)
            v_out = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_out, src=v_acc)
            v_bias = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=v_bias,
                src=qkv_b[
                    0:P_MAX, 2 * D_MODEL + o * P_MAX : 2 * D_MODEL + (o + 1) * P_MAX
                ],
            )
            v_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=v_biased, data1=v_out, data2=v_bias, op=nl.add)
            nisa.dma_copy(
                dst=v_buf[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=v_biased,
            )

    # Phase 3: Bidirectional flash attention
    for j_tile in nl.static_range(TEST_N_SEQ_TILES):
        _attention_for_seq_tile(
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            j_tile,
            n_seq_tiles,
            n_heads,
            head_dim,
            scale,
        )

    # Phase 4: Output projection + residual add
    for s_tile in nl.sequential_range(n_seq_tiles):
        s_start = s_tile * P_MAX

        ao_c_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=ao_c_0, src=attn_out_buf[s_start : s_start + P_MAX, 0:P_MAX])
        ao_t_0 = _transpose_to_sbuf(ao_c_0)

        ao_c_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_1, src=attn_out_buf[s_start : s_start + P_MAX, P_MAX : 2 * P_MAX]
        )
        ao_t_1 = _transpose_to_sbuf(ao_c_1)

        ao_c_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_2,
            src=attn_out_buf[s_start : s_start + P_MAX, 2 * P_MAX : 3 * P_MAX],
        )
        ao_t_2 = _transpose_to_sbuf(ao_c_2)

        ao_c_3 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_3,
            src=attn_out_buf[s_start : s_start + P_MAX, 3 * P_MAX : 4 * P_MAX],
        )
        ao_t_3 = _transpose_to_sbuf(ao_c_3)

        ao_c_4 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_4,
            src=attn_out_buf[s_start : s_start + P_MAX, 4 * P_MAX : 5 * P_MAX],
        )
        ao_t_4 = _transpose_to_sbuf(ao_c_4)

        ao_c_5 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_5,
            src=attn_out_buf[s_start : s_start + P_MAX, 5 * P_MAX : 6 * P_MAX],
        )
        ao_t_5 = _transpose_to_sbuf(ao_c_5)

        ao_c_6 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_6,
            src=attn_out_buf[s_start : s_start + P_MAX, 6 * P_MAX : 7 * P_MAX],
        )
        ao_t_6 = _transpose_to_sbuf(ao_c_6)

        ao_c_7 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_7,
            src=attn_out_buf[s_start : s_start + P_MAX, 7 * P_MAX : 8 * P_MAX],
        )
        ao_t_7 = _transpose_to_sbuf(ao_c_7)

        ao_c_8 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_8,
            src=attn_out_buf[s_start : s_start + P_MAX, 8 * P_MAX : 9 * P_MAX],
        )
        ao_t_8 = _transpose_to_sbuf(ao_c_8)

        ao_c_9 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ao_c_9,
            src=attn_out_buf[s_start : s_start + P_MAX, 9 * P_MAX : 10 * P_MAX],
        )
        ao_t_9 = _transpose_to_sbuf(ao_c_9)

        ao_t = (
            ao_t_0,
            ao_t_1,
            ao_t_2,
            ao_t_3,
            ao_t_4,
            ao_t_5,
            ao_t_6,
            ao_t_7,
            ao_t_8,
            ao_t_9,
        )

        for o in nl.static_range(N_HIDDEN):
            out_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=out_acc, value=0.0)
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    out_w[o * P_MAX : (o + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=ao_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=out_acc, data1=out_acc, data2=ps, op=nl.add)

            proj_chunk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=proj_chunk, src=out_acc)
            ob = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=ob, src=out_b[0:P_MAX, o * P_MAX : (o + 1) * P_MAX])
            proj_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=proj_biased, data1=proj_chunk, data2=ob, op=nl.add)
            x_orig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=x_orig,
                src=x_in[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
            )
            x_res = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=x_res, data1=x_orig, data2=proj_biased, op=nl.add)
            nisa.dma_copy(
                dst=post_attn_buf[
                    s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX
                ],
                src=x_res,
            )

    # Phase 5-8: LayerNorm + MLP + residual
    ln2_w = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln2_w, src=mlp_ln_w)
    ln2_b = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln2_b, src=mlp_ln_b)

    for s_tile in nl.sequential_range(n_seq_tiles):
        s_start = s_tile * P_MAX

        pa_tile = nl.ndarray((P_MAX, D_MODEL), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=pa_tile, src=post_attn_buf[s_start : s_start + P_MAX, 0:D_MODEL]
        )

        x_normed = _layer_norm_tile(pa_tile, ln2_w, ln2_b, D_MODEL, eps)

        mc_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_0, src=x_normed[0:P_MAX, 0:P_MAX])
        mc_t_0 = _transpose_to_sbuf(mc_0)

        mc_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_1, src=x_normed[0:P_MAX, P_MAX : 2 * P_MAX])
        mc_t_1 = _transpose_to_sbuf(mc_1)

        mc_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_2, src=x_normed[0:P_MAX, 2 * P_MAX : 3 * P_MAX])
        mc_t_2 = _transpose_to_sbuf(mc_2)

        mc_3 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_3, src=x_normed[0:P_MAX, 3 * P_MAX : 4 * P_MAX])
        mc_t_3 = _transpose_to_sbuf(mc_3)

        mc_4 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_4, src=x_normed[0:P_MAX, 4 * P_MAX : 5 * P_MAX])
        mc_t_4 = _transpose_to_sbuf(mc_4)

        mc_5 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_5, src=x_normed[0:P_MAX, 5 * P_MAX : 6 * P_MAX])
        mc_t_5 = _transpose_to_sbuf(mc_5)

        mc_6 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_6, src=x_normed[0:P_MAX, 6 * P_MAX : 7 * P_MAX])
        mc_t_6 = _transpose_to_sbuf(mc_6)

        mc_7 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_7, src=x_normed[0:P_MAX, 7 * P_MAX : 8 * P_MAX])
        mc_t_7 = _transpose_to_sbuf(mc_7)

        mc_8 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_8, src=x_normed[0:P_MAX, 8 * P_MAX : 9 * P_MAX])
        mc_t_8 = _transpose_to_sbuf(mc_8)

        mc_9 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mc_9, src=x_normed[0:P_MAX, 9 * P_MAX : 10 * P_MAX])
        mc_t_9 = _transpose_to_sbuf(mc_9)

        mc_t = (
            mc_t_0,
            mc_t_1,
            mc_t_2,
            mc_t_3,
            mc_t_4,
            mc_t_5,
            mc_t_6,
            mc_t_7,
            mc_t_8,
            mc_t_9,
        )

        # FC2 accumulators
        fc2_acc_0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_0, value=0.0)
        fc2_acc_1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_1, value=0.0)
        fc2_acc_2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_2, value=0.0)
        fc2_acc_3 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_3, value=0.0)
        fc2_acc_4 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_4, value=0.0)
        fc2_acc_5 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_5, value=0.0)
        fc2_acc_6 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_6, value=0.0)
        fc2_acc_7 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_7, value=0.0)
        fc2_acc_8 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_8, value=0.0)
        fc2_acc_9 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=fc2_acc_9, value=0.0)

        fc2_acc = (
            fc2_acc_0,
            fc2_acc_1,
            fc2_acc_2,
            fc2_acc_3,
            fc2_acc_4,
            fc2_acc_5,
            fc2_acc_6,
            fc2_acc_7,
            fc2_acc_8,
            fc2_acc_9,
        )

        for h in nl.static_range(N_MLP):
            fc1_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=fc1_acc, value=0.0)
            for i in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    fc1_w[h * P_MAX : (h + 1) * P_MAX, i * P_MAX : (i + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=mc_t[i], moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(dst=fc1_acc, data1=fc1_acc, data2=ps, op=nl.add)

            fc1_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fc1_bf16, src=fc1_acc)
            fb1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=fb1, src=fc1_b[0:P_MAX, h * P_MAX : (h + 1) * P_MAX])
            fc1_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=fc1_biased, data1=fc1_bf16, data2=fb1, op=nl.add)

            # GELU: x * sigmoid(1.702 * x)
            scaled_x = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=scaled_x,
                data=fc1_biased,
                op0=nl.multiply,
                operand0=1.702,
                engine=nisa.vector_engine,
            )
            sig_x = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(
                dst=sig_x, op=nl.sigmoid, data=scaled_x, bias=None, scale=1.0
            )
            gelu_out = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=gelu_out, data1=fc1_biased, data2=sig_x, op=nl.multiply
            )

            gelu_t = _transpose_to_sbuf(gelu_out)
            for o in nl.static_range(N_HIDDEN):
                w_t = _prepare_weight(
                    fc2_w[o * P_MAX : (o + 1) * P_MAX, h * P_MAX : (h + 1) * P_MAX]
                )
                p = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p, stationary=gelu_t, moving=w_t)
                ps = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ps, src=p)
                nisa.tensor_tensor(
                    dst=fc2_acc[o], data1=fc2_acc[o], data2=ps, op=nl.add
                )

        # Add FC2 bias, residual, store
        for o in nl.static_range(N_HIDDEN):
            fc2_chunk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fc2_chunk, src=fc2_acc[o])
            fb2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=fb2, src=fc2_b[0:P_MAX, o * P_MAX : (o + 1) * P_MAX])
            fc2_biased = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=fc2_biased, data1=fc2_chunk, data2=fb2, op=nl.add)

            pa_orig = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=pa_orig,
                src=post_attn_buf[
                    s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX
                ],
            )
            x_final = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=x_final, data1=pa_orig, data2=fc2_biased, op=nl.add)
            nisa.dma_copy(
                dst=x_out[s_start : s_start + P_MAX, o * P_MAX : (o + 1) * P_MAX],
                src=x_final,
            )

    return x_out


@nki.jit
def ref(
    x_in,
    attn_ln_w,
    attn_ln_b,
    qkv_w,
    qkv_b,
    out_w,
    out_b,
    mlp_ln_w,
    mlp_ln_b,
    fc1_w,
    fc1_b,
    fc2_w,
    fc2_b,
    n_seq_tiles: int = TEST_N_SEQ_TILES,
    n_heads: int = N_HEADS,
    head_dim: int = HEAD_DIM,
    eps: float = 1e-5,
) -> nl.ndarray:
    """Reference: Whisper encoder layer fused kernel."""
    return _encoder_layer_body(
        x_in,
        attn_ln_w,
        attn_ln_b,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        mlp_ln_w,
        mlp_ln_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        n_seq_tiles,
        n_heads,
        head_dim,
        eps,
    )


# SUBSTITUTE HERE


@nki.jit
def test(
    x_in,
    attn_ln_w,
    attn_ln_b,
    qkv_w,
    qkv_b,
    out_w,
    out_b,
    mlp_ln_w,
    mlp_ln_b,
    fc1_w,
    fc1_b,
    fc2_w,
    fc2_b,
    n_seq_tiles: int = TEST_N_SEQ_TILES,
    n_heads: int = N_HEADS,
    head_dim: int = HEAD_DIM,
    eps: float = 1e-5,
) -> nl.ndarray:
    """Test: Whisper encoder layer fused kernel (copy of ref, to be optimized)."""
    return _encoder_layer_body(
        x_in,
        attn_ln_w,
        attn_ln_b,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        mlp_ln_w,
        mlp_ln_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        n_seq_tiles,
        n_heads,
        head_dim,
        eps,
    )


def _layer_norm_np(x, weight, bias, eps=1e-5):
    """NumPy LayerNorm over the last dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * weight + bias


def _gelu_quick_np(x):
    """GELU quick approximation: x * sigmoid(1.702 * x)."""
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))


def _softmax_np(x, axis=-1):
    """Numerically stable softmax."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def cpu_reference(
    x_in,
    attn_ln_w,
    attn_ln_b,
    qkv_w,
    qkv_b,
    out_w,
    out_b,
    mlp_ln_w,
    mlp_ln_b,
    fc1_w,
    fc1_b,
    fc2_w,
    fc2_b,
):
    """CPU reference: full Whisper encoder layer.

    Tensor layouts match the NKI kernel's pre-tiled format:
      x_in:     [S, D_MODEL]         bf16 input activations
      attn_ln_w: [P_MAX, D_MODEL]    pre-tiled LN weight (each row identical)
      attn_ln_b: [P_MAX, D_MODEL]    pre-tiled LN bias
      qkv_w:    [3*D_MODEL, D_MODEL] stacked Q,K,V weight matrices
      qkv_b:    [P_MAX, 3*D_MODEL]   pre-tiled QKV bias (each row identical)
      out_w:    [D_MODEL, D_MODEL]   output projection weight
      out_b:    [P_MAX, D_MODEL]     pre-tiled output bias
      mlp_ln_w: [P_MAX, D_MODEL]     pre-tiled MLP LN weight
      mlp_ln_b: [P_MAX, D_MODEL]     pre-tiled MLP LN bias
      fc1_w:    [MLP_DIM, D_MODEL]   FC1 weight
      fc1_b:    [P_MAX, MLP_DIM]     pre-tiled FC1 bias
      fc2_w:    [D_MODEL, MLP_DIM]   FC2 weight
      fc2_b:    [P_MAX, D_MODEL]     pre-tiled FC2 bias
    """
    S = x_in.shape[0]
    x = x_in.astype(np.float32)

    # Extract un-tiled weights (row 0 of pre-tiled format)
    ln1_w = attn_ln_w[0].astype(np.float32)
    ln1_b = attn_ln_b[0].astype(np.float32)
    qkv_W = qkv_w.astype(np.float32)
    qkv_B = qkv_b[0].astype(np.float32)
    out_W = out_w.astype(np.float32)
    out_B = out_b[0].astype(np.float32)
    ln2_w = mlp_ln_w[0].astype(np.float32)
    ln2_b = mlp_ln_b[0].astype(np.float32)
    fc1_W = fc1_w.astype(np.float32)
    fc1_B = fc1_b[0].astype(np.float32)
    fc2_W = fc2_w.astype(np.float32)
    fc2_B = fc2_b[0].astype(np.float32)

    # Phase 1: LayerNorm (pre-attention)
    x_normed = _layer_norm_np(x, ln1_w, ln1_b)

    # Phase 2: QKV projection
    qkv = x_normed @ qkv_W.T + qkv_B
    Q = qkv[:, :D_MODEL]
    K = qkv[:, D_MODEL : 2 * D_MODEL]
    V = qkv[:, 2 * D_MODEL :]

    # Phase 3: Multi-head attention
    Q = Q.reshape(S, N_HEADS, HEAD_DIM)
    K = K.reshape(S, N_HEADS, HEAD_DIM)
    V = V.reshape(S, N_HEADS, HEAD_DIM)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_out = np.zeros((S, N_HEADS, HEAD_DIM), dtype=np.float32)
    for h in range(N_HEADS):
        scores = Q[:, h, :] @ K[:, h, :].T * scale
        weights = _softmax_np(scores, axis=-1)
        attn_out[:, h, :] = weights @ V[:, h, :]

    attn_out = attn_out.reshape(S, D_MODEL)

    # Phase 4: Output projection + residual
    proj = attn_out @ out_W.T + out_B
    x = x + proj

    # Phase 5: LayerNorm (pre-MLP)
    x_normed2 = _layer_norm_np(x, ln2_w, ln2_b)

    # Phase 6-7: FC1 + GELU
    fc1_out = x_normed2 @ fc1_W.T + fc1_B
    gelu_out = _gelu_quick_np(fc1_out)

    # Phase 8: FC2 + residual
    fc2_out = gelu_out @ fc2_W.T + fc2_B
    x = x + fc2_out

    return x.astype(np.float32)


def _make_pretiled_1d(vec, tile_rows=P_MAX):
    """Broadcast a 1D vector [F] into pre-tiled [tile_rows, F] (each row identical)."""
    return np.tile(vec[np.newaxis, :], (tile_rows, 1))


def _make_inputs(seed=42):
    """Generate random test inputs in the pre-tiled format expected by the kernel."""
    rng = np.random.RandomState(seed)
    S = TEST_SEQ_PAD

    x_in = rng.randn(S, D_MODEL).astype(np.float32) * 0.02

    attn_ln_w_1d = rng.randn(D_MODEL).astype(np.float32) * 0.1 + 1.0
    attn_ln_b_1d = rng.randn(D_MODEL).astype(np.float32) * 0.01
    attn_ln_w = _make_pretiled_1d(attn_ln_w_1d)
    attn_ln_b = _make_pretiled_1d(attn_ln_b_1d)

    qkv_w = rng.randn(3 * D_MODEL, D_MODEL).astype(np.float32) * 0.02
    qkv_b_1d = rng.randn(3 * D_MODEL).astype(np.float32) * 0.01
    qkv_b = _make_pretiled_1d(qkv_b_1d)

    out_w = rng.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    out_b_1d = rng.randn(D_MODEL).astype(np.float32) * 0.01
    out_b = _make_pretiled_1d(out_b_1d)

    mlp_ln_w_1d = rng.randn(D_MODEL).astype(np.float32) * 0.1 + 1.0
    mlp_ln_b_1d = rng.randn(D_MODEL).astype(np.float32) * 0.01
    mlp_ln_w = _make_pretiled_1d(mlp_ln_w_1d)
    mlp_ln_b = _make_pretiled_1d(mlp_ln_b_1d)

    fc1_w = rng.randn(MLP_DIM, D_MODEL).astype(np.float32) * 0.02
    fc1_b_1d = rng.randn(MLP_DIM).astype(np.float32) * 0.01
    fc1_b = _make_pretiled_1d(fc1_b_1d)

    fc2_w = rng.randn(D_MODEL, MLP_DIM).astype(np.float32) * 0.02
    fc2_b_1d = rng.randn(D_MODEL).astype(np.float32) * 0.01
    fc2_b = _make_pretiled_1d(fc2_b_1d)

    return (
        x_in,
        attn_ln_w,
        attn_ln_b,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        mlp_ln_w,
        mlp_ln_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
    )


def _to_device(inputs, device):
    """Send all inputs to device as bf16 tensors."""
    return tuple(torch.tensor(arr, dtype=torch.bfloat16).to(device) for arr in inputs)


def test_nki(ref_func, test_func):
    """Correctness: compare ref NKI kernel vs test NKI kernel (and vs CPU)."""
    device = xm.xla_device()

    for seed in [42, 123]:
        inputs = _make_inputs(seed)
        cpu_out = cpu_reference(*inputs)

        dev_inputs = _to_device(inputs, device)

        ref_out_t = ref_func(*dev_inputs, n_seq_tiles=TEST_N_SEQ_TILES)
        test_out_t = test_func(*dev_inputs, n_seq_tiles=TEST_N_SEQ_TILES)
        xm.mark_step()

        ref_out = ref_out_t.cpu().float().numpy()
        test_out = test_out_t.cpu().float().numpy()

        def cosine_sim(a, b):
            a_flat = a.flatten()
            b_flat = b.flatten()
            return np.dot(a_flat, b_flat) / (
                np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12
            )

        cos_ref_test = cosine_sim(ref_out, test_out)
        cos_test_cpu = cosine_sim(test_out, cpu_out)

        print(
            f"  seed={seed}: cos(ref,test)={cos_ref_test:.6f}, cos(test,cpu)={cos_test_cpu:.6f}"
        )

        if cos_ref_test < 0.99:
            print(f"  FAIL: NKI ref vs test cosine {cos_ref_test:.6f} < 0.99")
            return False

        if cos_test_cpu < 0.98:
            print(f"  FAIL: NKI test vs CPU cosine {cos_test_cpu:.6f} < 0.98")
            return False

    return True


def benchmark_nki(nki_func):
    """Benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    inputs = _make_inputs(42)
    dev_inputs = _to_device(inputs, device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(*dev_inputs, n_seq_tiles=TEST_N_SEQ_TILES)
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
