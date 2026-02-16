---
layout: default
title: Quantization
---

# 06 - Quantization

Quantization is a technique to make large AI models smaller and faster by using less precise numbers. This page explains why it matters, how it works, and which method to choose.

![Quantization Methods](diagrams/quantization-methods.png)

![Memory Savings](diagrams/memory-savings.png)

---

## Why Quantize?

Large language models store billions of numbers (called *weights*). The precision of those numbers directly determines how much memory the model needs.

Here's the simple math for a 70-billion-parameter model:

| Precision | Bytes per Weight | Total Memory | Fits on 1 GPU (80 GB)? |
|-----------|-----------------|-------------|------------------------|
| FP16 (16-bit) | 2 bytes | **140 GB** | No |
| INT8 (8-bit) | 1 byte | **70 GB** | Barely |
| INT4 (4-bit) | 0.5 bytes | **35 GB** | Yes, with room to spare! |

Think of it like image compression: a raw photo might be 25 MB, but a compressed JPEG looks almost the same at 2 MB. Quantization does the same thing for model weights -- it reduces precision just enough to save memory without noticeably hurting quality.

**In short: quantization lets you run bigger models on fewer GPUs.**

### Real Memory Math: Llama-3.1-70B

Let's walk through the full memory budget for serving Llama-3.1-70B on an H100 (80 GB):

| Component | FP16 | FP8 | INT4 (GPTQ) |
|-----------|------|-----|-------------|
| Model weights | 140 GB | 70 GB | 35 GB |
| KV cache (2K context, 32 users) | ~8 GB | ~4 GB (FP8 KV) | ~8 GB |
| Activations + overhead | ~2 GB | ~2 GB | ~2 GB |
| **Total** | **~150 GB (2× H100)** | **~76 GB (1× H100, tight)** | **~45 GB (1× H100, comfortable)** |

The KV cache formula: `2 × num_layers × num_kv_heads × head_dim × bytes_per_value × seq_len × batch_size`. For Llama-3.1-70B with 80 layers and 8 KV heads: `2 × 80 × 8 × 128 × 2 × 2048 × 32 ≈ 8.4 GB` in FP16.

---

## Two Flavors of Quantization

### Weight-Only Quantization

Only the model's stored weights are compressed. During computation, the weights are converted back to full precision on-the-fly.

- **Pros**: Easy to apply, works on most GPUs, good quality
- **Cons**: The math still happens in higher precision, so you don't get the full speed benefit
- **Methods**: GPTQ, AWQ, BitsAndBytes, GGUF, NVFP4

*Analogy*: Like storing a recipe in shorthand but reading it back in full sentences when you cook.

### Weight + Activation Quantization

Both the weights *and* the intermediate calculations (activations) use lower precision. This means the actual math operations (matrix multiplications) run in low precision too, which is significantly faster on supported hardware.

- **Pros**: Faster inference, lower memory for both weights and computations
- **Cons**: Requires newer GPUs with hardware support; slightly more quality risk
- **Methods**: FP8 (E4M3), INT8 (W8A8)

*Analogy*: Like doing the entire recipe in shorthand -- faster but you need a kitchen that understands shorthand.

---

## Under the Hood: FP8 at the Bit Level

FP8 E4M3 uses 8 bits to represent a floating-point number: **1 sign bit + 4 exponent bits + 3 mantissa bits**.

```
FP16 (16 bits):  [1 sign][5 exponent][10 mantissa]  → range ±65504, precision ~0.001
FP8 E4M3 (8 bits): [1 sign][4 exponent][3 mantissa]  → range ±448,   precision ~0.125
FP8 E5M2 (8 bits): [1 sign][5 exponent][2 mantissa]  → range ±57344, precision ~0.25
```

**E4M3** is used for weights and activations -- it has more mantissa bits so values are more precise. **E5M2** is used for gradients during training -- it has more exponent bits so it can represent a wider range. For inference with vLLM, you almost always want E4M3.

The key insight: H100 GPUs have dedicated FP8 Tensor Cores that compute matrix multiplications in FP8 at **2x the throughput** of FP16. This means FP8 isn't just a memory optimization -- it's a speed optimization too.

---

## Under the Hood: Why Marlin Kernels Are Fast

When you see `GPTQ_Marlin` or `AWQ_Marlin` in vLLM, the "Marlin" part refers to specialized CUDA kernels that make weight-only quantization much faster than the naive approach.

**Raw GPTQ** dequantizes weights to FP16 and then does a standard FP16 matrix multiply. This is memory-bound -- the GPU spends most of its time loading weights from memory.

**Marlin kernels** use a different strategy:
1. **Fused dequantization**: Weights are dequantized inside the matrix multiply kernel itself, not as a separate step
2. **Asynchronous memory access**: While one group of weights is being computed, the next group is being loaded from memory
3. **Optimized memory layout**: Weights are rearranged (permuted) at load time into a layout that maximizes GPU memory bandwidth

The result: Marlin kernels achieve near-FP16 throughput while using 4-bit weights. In vLLM, the `gptq_marlin.py` file handles the weight repacking, and the actual kernel code lives in `csrc/quantization/`.

```python
# From vllm/model_executor/layers/quantization/gptq_marlin.py
# Marlin uses specific scalar types for different bit widths:
TYPE_MAP = {
    (4, True): scalar_types.uint4b8,   # 4-bit symmetric
    (8, True): scalar_types.uint8b128, # 8-bit symmetric
}
```

---

## Supported Methods

| Method | Bits | Type | Speed | Quality | Hardware Needed |
|--------|------|------|-------|---------|-----------------|
| FP8 (E4M3) | 8 | Weight+Activation | Very Fast | Near-lossless | Hopper+ (H100, etc.) |
| GPTQ (Marlin) | 4/8 | Weight-only | Fast | Good | Ampere+ (A100, RTX 3090+) |
| AWQ (Marlin) | 4 | Weight-only | Fast | Good | Ampere+ |
| INT8 (W8A8) | 8 | Weight+Activation | Fast | Good | Turing+ (T4, RTX 2080+) |
| BitsAndBytes | 4/8 | Weight-only | Moderate | Good | Any CUDA GPU |
| NVFP4 | 4 | Weight-only | Fast | Good | Hopper+ |
| MXFP4/6/8 | 4/6/8 | Weight-only | Fast | Varies | Hopper+ |
| GGUF | 2-8 | Weight-only | Varies | Varies | Any GPU |
| TorchAO | Various | Multiple | Fast | Good | Varies |
| CompressedTensors | Various | Multiple | Fast | Good | Varies |

---

## Performance vs Quality Tradeoffs

Benchmark numbers vary by model and task, but here are typical ranges for Llama-class models:

| Method | Tokens/sec (relative) | Perplexity Increase | Memory Savings |
|--------|----------------------|---------------------|----------------|
| FP16 (baseline) | 1.0x | 0 | 0% |
| FP8 | 1.5-2.0x | < 0.1 | 50% |
| GPTQ-Marlin 4-bit | 1.3-1.7x | 0.1-0.3 | 75% |
| AWQ-Marlin 4-bit | 1.3-1.7x | 0.1-0.3 | 75% |
| INT8 W8A8 | 1.2-1.5x | < 0.1 | 50% |
| BitsAndBytes 4-bit | 0.8-1.0x | 0.2-0.5 | 75% |
| GGUF 2-bit | 0.7-0.9x | 1.0+ | 87% |

Note: BitsAndBytes is often *slower* than FP16 despite using less memory because its dequantization overhead is higher than Marlin-based methods.

---

## KV Cache Quantization

The KV cache stores information the model needs to "remember" earlier parts of the conversation. By default it uses 2 bytes per value (bf16/fp16). You can cut that in half with FP8, giving you **50% memory savings** on the cache -- which means room for longer conversations or more concurrent users.

| KV Cache Dtype | Memory per Token | When to Use |
|---------------|-----------------|-------------|
| `auto` (bf16) | 2 bytes | Default -- best quality |
| `fp8` / `fp8_e4m3` | 1 byte | When you need more memory for longer sequences or more users |
| `fp8_e5m2` | 1 byte | Alternative FP8 format |

---

## Decision Flowchart: Which Method Should I Use?

```
START: What GPU do you have?
│
├── H100 / H200 / Hopper+ ?
│   └── Use FP8 (E4M3)
│       ├── Need even more compression? → Add --kv-cache-dtype fp8_e4m3
│       └── Need 4-bit? → Use NVFP4
│
├── A100 / A10G / Ampere ?
│   ├── Quality critical? → GPTQ-Marlin 8-bit or INT8 W8A8
│   └── Memory critical? → AWQ-Marlin 4-bit or GPTQ-Marlin 4-bit
│
├── T4 / V100 / RTX 2080 (Turing) ?
│   ├── INT8 W8A8 (if supported)
│   └── BitsAndBytes 4-bit (universal fallback)
│
└── Older / consumer GPU?
    ├── BitsAndBytes 4-bit or 8-bit
    └── GGUF 2-4 bit (absolute minimum memory)
```

**Quick rule of thumb:** Always try FP8 first if your GPU supports it. It's the best quality-to-speed ratio. Only go to 4-bit if you absolutely need the memory savings.

---

## How It Works Under the Hood

**Loading phase (happens once at startup):**

| Step | What Happens |
|------|-------------|
| 1 | vLLM detects quantization from the model checkpoint (or from your `--quantization` flag) |
| 2 | A `QuantizationConfig` is created that selects the right method and optimized GPU kernel |
| 3 | The weight loader unpacks the quantized weights from disk |
| 4 | Scaling factors and zero-points are loaded alongside the weights (these help "decode" the compressed values) |

**Inference phase (happens every step):**
- **Weight-only**: Compressed weights are decompressed on-the-fly during each matrix multiplication
- **Weight+Activation**: Both weights and activations stay in low precision, and the matrix multiplication itself runs in low precision using specialized hardware units

---

## Gotchas and Common Mistakes

1. **"My GPTQ model is slow"** -- You're probably using raw GPTQ instead of GPTQ-Marlin. Check logs for `Using Marlin kernel`. If not present, your GPU may not support Marlin (needs Ampere+) or the model's group size is unsupported.

2. **"FP8 gives me errors on A100"** -- FP8 weight+activation quantization requires Hopper GPUs. On Ampere, use `--quantization fp8` only if you have pre-quantized FP8 weights (weight-only mode).

3. **"BitsAndBytes is slower than FP16"** -- This is expected for serving. BitsAndBytes is optimized for fine-tuning, not inference throughput. Switch to GPTQ-Marlin or AWQ-Marlin for better serving performance.

4. **"Model quality dropped significantly"** -- 4-bit quantization hits harder on smaller models. A 7B model at 4-bit loses more quality than a 70B model at 4-bit. Consider using 8-bit for models under 13B parameters.

5. **"KV cache FP8 increases latency"** -- The quantize/dequantize overhead is usually negligible, but if you're latency-sensitive with short sequences, stick with `auto`.

---

## Using Quantization

```bash
# Auto-detect from checkpoint (many models on HuggingFace come pre-quantized)
vllm serve TheBloke/Llama-2-7B-GPTQ

# Explicit FP8 quantization (vLLM quantizes at load time)
vllm serve meta-llama/Llama-3.1-8B --quantization fp8

# KV cache quantization (saves memory for the cache, not the model weights)
vllm serve meta-llama/Llama-3.1-8B --kv-cache-dtype fp8_e4m3
```

---

## Key Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/layers/quantization/` | All quantization method implementations (29 files) |
| `vllm/model_executor/layers/quantization/fp8.py` | FP8 implementation |
| `vllm/model_executor/layers/quantization/gptq_marlin.py` | GPTQ with Marlin kernels |
| `vllm/model_executor/layers/quantization/awq_marlin.py` | AWQ with Marlin kernels |
| `vllm/model_executor/layers/quantization/modelopt.py` | NVIDIA ModelOpt integration |
| `csrc/quantization/` | CUDA kernel source for quantization ops |

---

## Related Concepts

- **[Configuration](CONFIGURATION)** -- How to set quantization flags like `--quantization` and `--kv-cache-dtype` when starting vLLM
- **[Model System](MODEL-SYSTEM)** -- How vLLM loads models and applies quantization during the weight loading step
- **[Distributed Inference](DISTRIBUTED)** -- If your model is too big even after quantization, you can split it across multiple GPUs instead (or combine both approaches!)
- **[Scheduling & KV Cache](SCHEDULING-KV-CACHE)** -- KV cache quantization directly affects how many requests the scheduler can handle at once
- **[Official Docs: Quantization](https://docs.vllm.ai/en/latest/quantization/supported_hardware.html)** -- The official vLLM quantization guide with hardware compatibility details
- **[Official Docs: FP8](https://docs.vllm.ai/en/latest/quantization/fp8.html)** -- Detailed guide for FP8 quantization setup
