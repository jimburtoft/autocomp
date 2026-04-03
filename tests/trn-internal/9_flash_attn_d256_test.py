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
    """Reference flash attention for head_dim=256 (causal)."""
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
                        if use_causal_mask:
                            skip_condition = qi * B_P < kvi * B_F
                        else:
                            skip_condition = False

                        if not skip_condition:
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

                            qk = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.psum
                            )
                            qk[:, :] = nl.matmul(q0, k0, transpose_x=True)
                            qk[:, :] += nl.matmul(q1, k1, transpose_x=True)

                            qk_sbuf = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.sbuf
                            )

                            if use_causal_mask:
                                i_q, i_k = nl.mgrid[0:B_P, 0:B_F]
                                q_pos = qi * B_P + i_q
                                k_pos = kvi * B_F + i_k
                                pred_causal = q_pos >= k_pos

                                qk_sbuf[:, :] = nisa.affine_select(
                                    pred=pred_causal,
                                    on_true_tile=qk,
                                    on_false_value=NEG_INF,
                                    dtype=nl.float32,
                                )
                            else:
                                qk_sbuf[:, :] = nl.copy(qk, dtype=nl.float32)

                            new_max = nisa.tensor_reduce(
                                nl.max,
                                qk_sbuf,
                                axis=(1,),
                                dtype=nl.float32,
                                negate=False,
                            )

                            m_prev = nl.copy(m_acc[:, 0])
                            m_acc[:, 0] = nl.maximum(m_prev, new_max)
                            m_cur = m_acc[:, 0]

                            alpha = nisa.activation(
                                nl.exp, m_cur, bias=m_prev, scale=-1.0
                            )
                            o_acc[...] = nl.multiply(o_acc, alpha)

                            p = nl.ndarray((nl.par_dim(B_P), B_F), dtype=nl.bfloat16)
                            p_sum = nl.ndarray((nl.par_dim(B_P), 1), dtype=nl.float32)
                            p[:, :] = nisa.activation_reduce(
                                nl.exp,
                                qk_sbuf,
                                bias=-1 * m_cur,
                                scale=1.0,
                                reduce_op=nl.add,
                                reduce_res=p_sum[:, 0],
                                dtype=nl.bfloat16,
                            )

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

                            p_t = nl.ndarray((nl.par_dim(B_P), B_F), dtype=nl.bfloat16)
                            for ti in nl.affine_range(B_F // B_P):
                                p_t_tmp = nl.ndarray(
                                    (nl.par_dim(B_P), B_P),
                                    dtype=nl.float32,
                                    buffer=nl.psum,
                                )
                                p_t_tmp[:, :] = nisa.nc_transpose(
                                    p[:, nl.ds(ti * B_P, B_P)]
                                )
                                p_t[:, nl.ds(ti * B_P, B_P)] = nl.copy(
                                    p_t_tmp, dtype=nl.bfloat16
                                )

                            pv = nl.ndarray(
                                (nl.par_dim(B_P), d),
                                dtype=nl.float32,
                                buffer=nl.psum,
                            )
                            nisa.memset(pv, 0.0)
                            for vi in nl.affine_range(n_v_sub):
                                pv[:, :] += nl.matmul(
                                    p_t[:, nl.ds(vi * B_P, B_P)],
                                    v_tile[vi, :, :],
                                    transpose_x=True,
                                )

                            o_acc[:, :] = nl.add(o_acc, pv)

                            exp_l = nisa.activation(
                                nl.exp, m_cur, bias=l_acc[:, 0], scale=-1.0
                            )
                            l_acc[:, 0] = nl.add(
                                m_cur, nisa.activation(nl.log, exp_l, bias=p_sum[:, 0])
                            )

                    final_exp = nisa.activation(
                        nl.exp, l_acc[:, 0], bias=m_acc[:, 0], scale=-1.0
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


# SUBSTITUTE HERE


@nki.jit
def test(q, k, v, use_causal_mask=True):
    """Test flash attention for head_dim=256 (causal)."""
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
                        if use_causal_mask:
                            skip_condition = qi * B_P < kvi * B_F
                        else:
                            skip_condition = False

                        if not skip_condition:
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

                            qk = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.psum
                            )
                            qk[:, :] = nl.matmul(q0, k0, transpose_x=True)
                            qk[:, :] += nl.matmul(q1, k1, transpose_x=True)

                            qk_sbuf = nl.ndarray(
                                (nl.par_dim(B_P), B_F), dtype=nl.float32, buffer=nl.sbuf
                            )

                            if use_causal_mask:
                                i_q, i_k = nl.mgrid[0:B_P, 0:B_F]
                                q_pos = qi * B_P + i_q
                                k_pos = kvi * B_F + i_k
                                pred_causal = q_pos >= k_pos

                                qk_sbuf[:, :] = nisa.affine_select(
                                    pred=pred_causal,
                                    on_true_tile=qk,
                                    on_false_value=NEG_INF,
                                    dtype=nl.float32,
                                )
                            else:
                                qk_sbuf[:, :] = nl.copy(qk, dtype=nl.float32)

                            new_max = nisa.tensor_reduce(
                                nl.max,
                                qk_sbuf,
                                axis=(1,),
                                dtype=nl.float32,
                                negate=False,
                            )

                            m_prev = nl.copy(m_acc[:, 0])
                            m_acc[:, 0] = nl.maximum(m_prev, new_max)
                            m_cur = m_acc[:, 0]

                            alpha = nisa.activation(
                                nl.exp, m_cur, bias=m_prev, scale=-1.0
                            )
                            o_acc[...] = nl.multiply(o_acc, alpha)

                            p = nl.ndarray((nl.par_dim(B_P), B_F), dtype=nl.bfloat16)
                            p_sum = nl.ndarray((nl.par_dim(B_P), 1), dtype=nl.float32)
                            p[:, :] = nisa.activation_reduce(
                                nl.exp,
                                qk_sbuf,
                                bias=-1 * m_cur,
                                scale=1.0,
                                reduce_op=nl.add,
                                reduce_res=p_sum[:, 0],
                                dtype=nl.bfloat16,
                            )

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

                            p_t = nl.ndarray((nl.par_dim(B_P), B_F), dtype=nl.bfloat16)
                            for ti in nl.affine_range(B_F // B_P):
                                p_t_tmp = nl.ndarray(
                                    (nl.par_dim(B_P), B_P),
                                    dtype=nl.float32,
                                    buffer=nl.psum,
                                )
                                p_t_tmp[:, :] = nisa.nc_transpose(
                                    p[:, nl.ds(ti * B_P, B_P)]
                                )
                                p_t[:, nl.ds(ti * B_P, B_P)] = nl.copy(
                                    p_t_tmp, dtype=nl.bfloat16
                                )

                            pv = nl.ndarray(
                                (nl.par_dim(B_P), d),
                                dtype=nl.float32,
                                buffer=nl.psum,
                            )
                            nisa.memset(pv, 0.0)
                            for vi in nl.affine_range(n_v_sub):
                                pv[:, :] += nl.matmul(
                                    p_t[:, nl.ds(vi * B_P, B_P)],
                                    v_tile[vi, :, :],
                                    transpose_x=True,
                                )

                            o_acc[:, :] = nl.add(o_acc, pv)

                            exp_l = nisa.activation(
                                nl.exp, m_cur, bias=l_acc[:, 0], scale=-1.0
                            )
                            l_acc[:, 0] = nl.add(
                                m_cur, nisa.activation(nl.log, exp_l, bias=p_sum[:, 0])
                            )

                    final_exp = nisa.activation(
                        nl.exp, l_acc[:, 0], bias=m_acc[:, 0], scale=-1.0
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
