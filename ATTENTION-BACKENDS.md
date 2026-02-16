---
layout: default
title: Attention Backends
---

# 05 — Attention Backends

Attention is the most compute-intensive operation in every transformer-based LLM.
vLLM abstracts it behind a **backend registry** so that the engine can dispatch to
the fastest available kernel for the current hardware, model architecture, and
quantization configuration—without changing a single line in the model definition.

> **Official reference:** <https://docs.vllm.ai/en/latest/design/attention_backends.html>

![Attention Backends](diagrams/attention-backends.png)

---

## 1 — Why Multiple Backends?

No single attention kernel is optimal everywhere:

| Concern | Impact on backend choice |
|---------|--------------------------|
| **GPU vendor** | NVIDIA ships FlashAttention; AMD uses its own ROCm kernels. |
| **GPU generation** | Hopper GPUs unlock FP8 paths in FlashAttention v3; Ampere cannot. |
| **Head dimension** | Some kernels only support `head_dim` 64/128; others handle 256+. |
| **KV cache dtype** | FP8 KV caches need a backend that can dequantize on the fly. |
| **Prefix caching** | FlashInfer has first-class prefix-aware paged attention. |
| **Architecture** | DeepSeek-V2/V3 MLA models compress KV differently than standard MHA. |
| **State-space models** | Mamba layers bypass attention entirely—they need their own "backend." |

Because of this diversity, vLLM keeps an extensible **registry** that maps a
symbolic name (`FLASH_ATTN`, `FLASHINFER`, `TRITON_MLA`, …) to a concrete
implementation class.

---

## 2 — Backend Selection Logic

The selector lives in `vllm/v1/attention/backends/selector.py`.  The decision
tree is evaluated **once** at engine start-up and is deterministic:

```
┌──────────────────────────────────────────────┐
│  1. Did the user set --attention-backend?     │
│     YES ──► use that backend directly         │
│     NO  ──► continue                          │
├──────────────────────────────────────────────┤
│  2. Does the model use MLA (Multi-head        │
│     Latent Attention)?                        │
│     YES ──► pick best MLA backend for         │
│              this platform                    │
│     NO  ──► continue                          │
├──────────────────────────────────────────────┤
│  3. Is the model a non-attention arch?        │
│     (Mamba, RWKV, linear attention)           │
│     YES ──► pick MAMBA1/MAMBA2/LINEAR/…       │
│     NO  ──► continue                          │
├──────────────────────────────────────────────┤
│  4. What platform are we on?                  │
│     CUDA (NVIDIA) ──► FlashAttention or       │
│                        FlashInfer             │
│     ROCm (AMD)    ──► ROCM_ATTN              │
│     CPU           ──► TORCH_SDPA             │
│     TPU           ──► PALLAS_ATTN            │
│     HPU (Gaudi)   ──► HPU_ATTN               │
├──────────────────────────────────────────────┤
│  5. Within CUDA, refine:                      │
│     • head_size supported?                    │
│     • kv_cache_dtype compatible?              │
│     • block_size compatible?                  │
│     • FlashAttention v3 available (Hopper)?   │
│     ──► pick best match                       │
└──────────────────────────────────────────────┘
```

### User Override

The simplest way to force a backend is the `--attention-backend` CLI flag:

```bash
vllm serve meta-llama/Llama-3.1-8B --attention-backend FLASHINFER
```

This skips all auto-detection and directly loads the requested backend.  It is
useful for benchmarking or working around a newly discovered bug.

### Platform Check

After the user override, the selector queries the **current platform**
(`vllm.platforms`).  Each platform advertises which backends it supports.  For
example, `CudaPlatform` registers `FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`,
`FLEX_ATTENTION`, etc., while `RocmPlatform` registers `ROCM_ATTN`.

### Capabilities Check

Even within a platform, not every backend handles every configuration:

* `FLASH_ATTN` only supports head sizes 64, 128, and 256.
* `FLASHINFER` supports a wider range of head sizes and FP8 KV caches.
* `FLASH_ATTN_V3` requires SM90+ (Hopper / Blackwell).
* Some backends do not support `block_size=1` (used by some schedulers).

The selector calls each candidate's `check_supported(head_size, dtype,
kv_cache_dtype, block_size)` class method and picks the first that passes.

---

## 3 — Standard Attention Backends

### 3.1 — FLASH_ATTN (FlashAttention v2)

| Property | Detail |
|----------|--------|
| **Kernel source** | [flash-attn](https://github.com/Dao-AILab/flash-attention) CUDA library |
| **Supported GPUs** | Ampere (SM80), Ada (SM89), Hopper (SM90) |
| **Supported dtypes** | fp16, bf16 |
| **KV cache dtypes** | fp16, bf16 |
| **Head sizes** | 64, 128, 256 |
| **Best for** | General-purpose NVIDIA inference; most production deployments |

FlashAttention v2 rewrites the attention algorithm to avoid materialising the
full `N × N` attention matrix.  Instead it tiles the computation and keeps the
running softmax statistics in registers, achieving **O(N)** HBM reads instead of
**O(N²)**.  This is the most battle-tested backend in vLLM.

**Prefill** uses `flash_attn_varlen_func` which accepts variable-length
sequences packed into a single tensor.  **Decode** uses paged attention kernels
that follow the block table to scatter-gather KV from the block pool.

### 3.2 — FLASH_ATTN_V3 (FlashAttention v3)

| Property | Detail |
|----------|--------|
| **Kernel source** | FlashAttention v3 branch |
| **Minimum GPU** | Hopper SM90+ |
| **Key feature** | Native FP8 (E4M3) attention, asynchronous warp specialisation |
| **Best for** | FP8 models on H100/H200 |

FlashAttention v3 exploits Hopper-specific hardware:

* **Warp-specialised pipelining** — producer warps issue TMA loads while
  consumer warps execute tensor-core MMAs, hiding memory latency.
* **FP8 tensor cores** — the kernel can consume FP8 Q/K/V directly, doubling
  the arithmetic throughput and halving memory traffic.
* **Persistent kernel design** — a single kernel stays resident across the
  entire sequence, reducing launch overhead.

Use v3 when running FP8-quantised models on Hopper or newer:

```bash
vllm serve meta-llama/Llama-3.1-70B --quantization fp8 --attention-backend FLASH_ATTN_V3
```

### 3.3 — FLASHINFER

| Property | Detail |
|----------|--------|
| **Kernel source** | [flashinfer](https://github.com/flashinfer-ai/flashinfer) library |
| **Supported GPUs** | Ampere+ |
| **Key feature** | First-class paged/prefix-caching KV, broad head-size support |
| **Best for** | Workloads with heavy prefix caching |

FlashInfer was designed from the ground up for **serving** (as opposed to
training).  Its paged-attention kernels can natively follow block tables without
a separate "paged attention" wrapper, and it has deep support for
**prefix-aware** KV cache layouts that skip redundant computation when many
requests share a common system prompt.

FlashInfer also supports a wider set of head dimensions (including 96, 192, and
others) and KV cache dtypes (including FP8 E4M3).

### 3.4 — TRITON_ATTN

| Property | Detail |
|----------|--------|
| **Kernel source** | Pure Triton (Python) kernels shipped with vLLM |
| **Supported GPUs** | Anything Triton supports (NVIDIA, some AMD) |
| **Key feature** | Portable, easy to modify |
| **Best for** | Development, debugging, custom attention variants |

Triton attention is written entirely in Triton DSL.  It is slower than
FlashAttention or FlashInfer for large batch sizes but has the advantage of
being **easy to read and extend**.  If you are prototyping a new attention
variant (e.g., sliding window with a custom mask), starting with Triton is
usually the fastest path.

### 3.5 — FLEX_ATTENTION

| Property | Detail |
|----------|--------|
| **Kernel source** | `torch.nn.attention.flex_attention` (PyTorch native) |
| **Supported GPUs** | Whatever PyTorch supports |
| **Key feature** | Leverages upstream PyTorch's FlexAttention API |
| **Best for** | Future-proofing; models that define custom score_mod functions |

FlexAttention is PyTorch's official composable attention API.  It lets users
define arbitrary per-element masking and scoring functions that get compiled into
a fused kernel via `torch.compile`.  vLLM wraps this API so that models can
use `score_mod` / `block_mask` without leaving the vLLM execution loop.

### 3.6 — TORCH_SDPA

| Property | Detail |
|----------|--------|
| **Kernel source** | `torch.nn.functional.scaled_dot_product_attention` |
| **Supported GPUs** | All (falls back to math kernel on CPU) |
| **Key feature** | No external dependencies |
| **Best for** | ViT encoders, encoder-decoder cross-attention |

TORCH_SDPA is the simplest backend.  It dispatches internally to whatever
PyTorch decides is optimal (FlashAttention, efficient attention, or the maths
fallback).  vLLM uses it mainly for **vision encoders** (e.g., the ViT in
LLaVA) where sequences are short and paged attention is unnecessary.

### 3.7 — ROCM_ATTN

| Property | Detail |
|----------|--------|
| **Kernel source** | ROCm-optimised kernels (CK-based) |
| **Supported GPUs** | AMD MI250, MI300X |
| **Best for** | AMD data-centre GPUs |

The ROCm backend wraps AMD's Composable Kernel (CK) library, which implements
fused attention with similar tiling strategies to FlashAttention but targeting
the CDNA architecture's matrix cores.

---

## 4 — MLA Backends (Multi-head Latent Attention)

### What is MLA?

**Multi-head Latent Attention** (introduced in DeepSeek-V2) compresses the KV
cache by projecting K and V into a low-rank **latent space** before caching:

```
Standard MHA:
  cache_k = W_k @ x          # shape: [num_heads, head_dim]
  cache_v = W_v @ x          # shape: [num_heads, head_dim]
  ──► per-token cache ≈ 2 × num_heads × head_dim

MLA:
  latent = W_down @ x        # shape: [latent_dim]   (latent_dim << num_heads × head_dim)
  cache   = latent           # shape: [latent_dim]
  k = W_up_k @ latent        # reconstructed at attention time
  v = W_up_v @ latent        # reconstructed at attention time
  ──► per-token cache ≈ latent_dim   (4-16× smaller)
```

This dramatically reduces KV cache memory for long-context inference but
requires special kernels that perform the up-projection fused with attention.

### MLA Backend Table

| Backend | Implementation | Notes |
|---------|---------------|-------|
| `FLASHINFER_MLA` | FlashInfer with fused up-proj + attention | Default MLA backend on CUDA |
| `TRITON_MLA` | Pure Triton kernels | Portable MLA fallback |
| `CUTLASS_MLA` | NVIDIA CUTLASS templates | High perf on Hopper |
| `FLASHMLA` | Dedicated FlashMLA library | From DeepSeek team |
| `FLASH_ATTN_MLA` | FlashAttention patched for MLA | Experimental |

Each of these also has a `*_SPARSE` variant (`FLASHINFER_MLA_SPARSE`,
`TRITON_MLA_SPARSE`, etc.) that adds **sparse attention** on top of MLA—useful
for very long contexts where even the latent KV cache becomes large.

### Selecting an MLA Backend

The selector checks for MLA automatically when the model config specifies a
`DeepseekV2` (or V3) architecture.  You can override:

```bash
vllm serve deepseek-ai/DeepSeek-V3 --attention-backend FLASHINFER_MLA
```

---

## 5 — Non-Attention Backends

Not every sequence model uses self-attention.  vLLM supports **state-space
models** (SSMs) and other architectures via specialised backends that slot into
the same registry.

### 5.1 — MAMBA1 / MAMBA2

Mamba layers replace the attention matrix with a **selective state-space model**
(S6).  The hidden state is updated recurrently:

```
h[t] = A_bar · h[t-1] + B_bar · x[t]
y[t] = C · h[t]
```

During prefill the recurrence is unrolled with an efficient parallel scan.
During decode, only a single step is executed per token.  The "KV cache"
equivalent is the **SSM state tensor** `h[t]`.

* `MAMBA1` implements the original Mamba architecture (Gu & Dao, 2023).
* `MAMBA2` implements Mamba-2 with structured state-space duality (SSD).

### 5.2 — LINEAR

Linear attention replaces softmax attention with a kernel feature map:

```
Attention(Q, K, V) = φ(Q) · (φ(K)^T · V)
```

By changing the order of multiplication the cost drops from O(N²) to O(N).
The LINEAR backend stores running sums instead of full KV caches.

### 5.3 — SHORT_CONV

Some hybrid architectures (e.g., Jamba, Zamba) interleave Mamba layers with
short 1-D convolution layers.  The `SHORT_CONV` backend handles these by
maintaining a small sliding-window state buffer.

---

## 6 — Prefill vs Decode: Different Backends

vLLM can use **different backends for prefill and decode** because the two
phases have very different computational profiles:

| Phase | Characteristic | Preferred kernel style |
|-------|---------------|----------------------|
| **Prefill** | Long sequences, compute-bound, variable length | `flash_attn_varlen` / FlashInfer batch-prefill |
| **Decode** | Single new token per request, memory-bound, paged KV | Paged attention / FlashInfer decode |

During prefill, all tokens are processed at once.  The kernel receives packed
variable-length sequences and produces both the output and the KV cache entries.

During decode, only the **new** token's Q vector is computed.  The kernel must
read the **entire** cached K/V for each request (following the block table),
compute attention, and return a single output vector per request.  This is
heavily memory-bandwidth-bound, which is why CUDA graph replay (see
[Compilation](COMPILATION)) helps so much—it eliminates CPU-side overhead
that would otherwise dominate.

### Block Table Lookup

```
Request 42, generated 317 tokens:
  block_table[42] = [blk_5, blk_12, blk_3, blk_97, blk_61]

Decode attention kernel:
  for each block_id in block_table[42]:
      load K[block_id], V[block_id] from GPU memory
      compute partial attention
  combine partials → output
```

The block table is allocated and managed by the **block manager** (see
[Configuration](CONFIGURATION)).

---

## 7 — FlashAttention v2 vs v3 — Detailed Comparison

| Feature | FlashAttention v2 | FlashAttention v3 |
|---------|-------------------|-------------------|
| Minimum GPU | Ampere (SM80) | Hopper (SM90) |
| Data types | fp16, bf16 | fp16, bf16, **fp8 (E4M3)** |
| Warp scheduling | Standard | **Warp-specialised** (producer/consumer) |
| Memory loads | Synchronous | **Asynchronous TMA** |
| Kernel design | Tiled, relaunch per tile | **Persistent**, stays resident |
| Typical speedup | 1× (baseline) | ~1.5–2× on H100 |
| KV cache FP8 | Not native | Native, zero-overhead dequant |

**When to use v3:**
- You have H100 / H200 / B100 / B200 GPUs.
- Your model uses FP8 quantisation.
- You are latency-sensitive and want every ounce of performance.

**When to stick with v2:**
- You are on A100 or older.
- Your workload is bf16 / fp16 only.
- You need maximum kernel stability (v2 is more mature).

---

## 8 — Configuration Reference

### CLI Flags

| Flag | Values | Default |
|------|--------|---------|
| `--attention-backend` | `FLASH_ATTN`, `FLASH_ATTN_V3`, `FLASHINFER`, `TRITON_ATTN`, `FLEX_ATTENTION`, `TORCH_SDPA`, `ROCM_ATTN`, `FLASHINFER_MLA`, `TRITON_MLA`, `CUTLASS_MLA`, etc. | Auto-detect |
| `--kv-cache-dtype` | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | `auto` |
| `--block-size` | 16 (typical) | Auto |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `VLLM_ATTENTION_BACKEND` | Same as `--attention-backend` but via env var |
| `VLLM_USE_FLASHINFER` | Legacy flag (deprecated in favour of the above) |

---

## 9 — Adding a New Backend

The backend registry is designed to be extensible.  To add a new backend:

1. **Create the backend class** in `vllm/v1/attention/backends/your_backend.py`.
   Implement the `AttentionBackend` interface:
   - `forward_prefill(query, key, value, …) → output`
   - `forward_decode(query, key_cache, value_cache, block_table, …) → output`
   - Class method `check_supported(head_size, dtype, kv_cache_dtype, block_size) → bool`

2. **Register** the backend enum value in `vllm/v1/attention/backends/registry.py`.

3. **Update the selector** in `selector.py` so auto-detection can find it.

4. **Write tests** in `tests/v1/attention/`.

---

## 10 — Key Files

| File | Purpose |
|------|---------|
| `vllm/v1/attention/backends/registry.py` | `AttentionBackendEnum` — canonical list of all backend names |
| `vllm/v1/attention/backends/selector.py` | `get_attn_backend()` — auto-detection logic |
| `vllm/v1/attention/backends/flash_attn.py` | FlashAttention v2 wrapper |
| `vllm/v1/attention/backends/flash_attn_v3.py` | FlashAttention v3 wrapper |
| `vllm/v1/attention/backends/flashinfer.py` | FlashInfer wrapper (standard + MLA) |
| `vllm/v1/attention/backends/triton_attn.py` | Triton-based attention |
| `vllm/v1/attention/backends/mla/` | MLA-specific implementations (CUTLASS, Triton, FlashMLA) |
| `vllm/v1/attention/backends/mamba.py` | Mamba SSM backends |
| `vllm/model_executor/layers/attention/` | `Attention` module that delegates to the selected backend |

---

## 11 — Summary

```
                     ┌─────────────────┐
                     │  User override?  │
                     └───────┬─────────┘
                             │ no
                     ┌───────▼─────────┐
                     │  MLA required?   │──yes──► FLASHINFER_MLA / TRITON_MLA / …
                     └───────┬─────────┘
                             │ no
                     ┌───────▼─────────┐
                     │  SSM / Linear?   │──yes──► MAMBA1 / MAMBA2 / LINEAR / …
                     └───────┬─────────┘
                             │ no
                     ┌───────▼─────────┐
                     │   Platform?      │
                     └───────┬─────────┘
                      ┌──────┼──────┐
                   CUDA    ROCm   CPU/TPU
                      │      │      │
               FlashAttn  ROCM   TORCH_SDPA
               FlashInfer  ATTN   PALLAS
               Triton
```

The attention backend is one of the most performance-critical choices in vLLM.
For most NVIDIA users the default auto-detection picks the right one, but
understanding *why* each exists helps when debugging latency, enabling FP8, or
supporting a new model architecture.

