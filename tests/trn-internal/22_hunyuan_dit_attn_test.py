"""
HunyuanVideo-1.5 DiT Flash Attention (Dense Path) — NKI 0.3.0 kernel.

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

Baseline device-only latency: 18.39 ms (neuron-profile, trn2.3xlarge LNC=2)
  - Vector engine dominant (87%) -- softmax ops
  - Tensor engine 27.6% -- matmul underutilized
  - MFU: 1.2%

Optimization opportunities:
  - Reduce V sub-tile iterations (wider P@V accumulation)
  - Fuse correction * out_acc + pv into single pass
  - Pipeline K load with QK matmul
  - Use nl.loop_reduce for sum/max fusion
"""

import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import torch


# Tile size constants
PMAX = 128  # Partition dim = head_dim = 128
K_TILE = 512  # K/V chunk size for flash attention tiling
LARGE_NEG = -9984.0  # bf16-safe large negative

# HunyuanVideo-specific dimensions
BATCH_HEADS = 4  # local heads per TP rank (16 heads / TP=4)
D_HEAD = 128  # head dimension
SEQ_LEN = 4096  # padded from 3500 (3180 img + 320 text) to 4096 for NKI alignment


# SUBSTITUTE HERE


@nki.jit
def ref(q_hbm, k_hbm, v_hbm):
    """Flash attention with fp32 online softmax for HunyuanVideo DiT.

    NKI 0.3.0 nl API. Non-causal, bf16 I/O, fp32 softmax.

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

            # Load Q tile [PMAX, d_head] and scale
            q_idx = nl.arange(PMAX)[:, None]
            d_idx = nl.arange(d_head)[None, :]
            q_tile = nl.load(q_hbm[b, q_offset + q_idx, d_idx])
            q_tile = q_tile * scale

            # Index tensors for in-place updates (NKI 0.3.0 scoping)
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

                # Rescale accumulators (in-place for NKI 0.3.0 scoping)
                running_sum[i_p, i_1] = running_sum * correction
                out_acc[i_p, i_d] = out_acc * correction

                # exp(scores - new_max)
                exp_scores = nl.exp(scores - new_max)
                chunk_sum = nl.sum(exp_scores, axis=[1])
                running_sum[i_p, i_1] = running_sum + chunk_sum
                running_max[i_p, i_1] = new_max

                # P @ V in sub-tiles of PMAX (partition limit = 128)
                n_v_tiles = K_TILE // PMAX  # 4
                for vt in range(n_v_tiles):
                    v_tile_offset = kv_offset + vt * PMAX
                    vt_p_idx = nl.arange(PMAX)[:, None]
                    vt_f_idx = nl.arange(d_head)[None, :]
                    v_tile = nl.load(v_hbm[b, v_tile_offset + vt_p_idx, vt_f_idx])

                    p_slice = exp_scores[:, nl.ds(vt * PMAX, PMAX)]
                    pv_tile = nl.matmul(p_slice, v_tile)
                    out_acc[i_p, i_d] = out_acc + pv_tile

            # Final normalization and store
            out_normalized = out_acc / running_sum
            nl.store(output[b, q_offset + q_idx, d_idx], value=out_normalized)

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
    scale = 1.0 / math.sqrt(d_head)

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
        q_dev = torch.tensor(q_np, dtype=torch.bfloat16)
        k_dev = torch.tensor(k_np, dtype=torch.bfloat16)
        v_dev = torch.tensor(v_np, dtype=torch.bfloat16)

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
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    batch = BATCH_HEADS
    seqlen = SEQ_LEN
    d_head = D_HEAD

    np.random.seed(42)
    q_np = (np.random.randn(batch, seqlen, d_head) * 0.1).astype(np.float32)
    k_np = (np.random.randn(batch, d_head, seqlen) * 0.1).astype(np.float32)
    v_np = (np.random.randn(batch, seqlen, d_head) * 0.1).astype(np.float32)

    q_dev = torch.tensor(q_np, dtype=torch.bfloat16)
    k_dev = torch.tensor(k_np, dtype=torch.bfloat16)
    v_dev = torch.tensor(v_np, dtype=torch.bfloat16)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(q_dev, k_dev, v_dev)
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
