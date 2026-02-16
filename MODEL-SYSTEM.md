---
layout: default
title: Model System
---

# 04 - Model System

This page explains how vLLM finds, loads, and runs AI models. When you give vLLM a model name like `meta-llama/Llama-3.1-8B`, a lot happens behind the scenes before the first token is generated. Let's walk through it.

## How Model Discovery Works

![Model Registry](diagrams/model-registry.png)

When you tell vLLM to serve a model, here's what happens step by step:

1. **You provide a model name** -- This can be a HuggingFace model ID (like `meta-llama/Llama-3.1-8B`) or a path to a local folder
2. **vLLM downloads the config** -- It fetches `config.json` from HuggingFace (or reads it locally). This file describes the model's architecture: how many layers, how many attention heads, hidden size, etc.
3. **The registry looks up the architecture** -- vLLM has a built-in registry that maps architecture names (like `"LlamaForCausalLM"`) to the right Python class that knows how to run that model
4. **If the architecture isn't found** -- vLLM falls back to the Transformers backend (more on this below)

Think of the registry like a phone book: you look up the model's architecture name and get back the class that can run it.

---

## Real Example: Loading Llama-3.1-8B Step by Step

Here's exactly what happens when you run `vllm serve meta-llama/Llama-3.1-8B`:

**Step 1: Config download.** vLLM fetches `config.json` from HuggingFace. Inside, it finds `"architectures": ["LlamaForCausalLM"]`.

**Step 2: Registry lookup.** The `_TEXT_GENERATION_MODELS` dict in `registry.py` contains this entry:

```python
# From vllm/model_executor/models/registry.py
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    # ...200+ more entries...
}
```

The tuple `("llama", "LlamaForCausalLM")` means: import `LlamaForCausalLM` from the module `vllm.model_executor.models.llama`.

**Step 3: Lazy inspection.** Before actually importing the model class (which could trigger CUDA initialization), vLLM inspects the class in a subprocess to learn its capabilities. More on this in the LazyRegisteredModel section below.

**Step 4: Model creation.** The `LlamaForCausalLM` class (in `llama.py`) is instantiated with empty weights -- just the structure.

**Step 5: Weight loading.** Safetensors files are downloaded and the weights are loaded. For Llama-3.1-8B, that's about 16 GB of FP16 weights. If you specified `--tensor-parallel-size 2`, each GPU receives half the weights.

**Step 6: Post-init warmup.** CUDA graphs are recorded at various batch sizes, torch.compile runs, and the model is ready to serve.

---

## What Models Does vLLM Support?

vLLM supports a wide range of model types:

| Category | Count | Examples |
|----------|-------|---------|
| Text Generation | 100+ | Llama, Qwen, Mistral, GPT-NeoX, Falcon |
| Mixture of Experts (MoE) | 15+ | Mixtral, DeepSeek-V2, Qwen2-MoE |
| Multimodal (text + images/audio) | 40+ | LLaVA, Qwen-VL, InternVL, Phi3-V |
| Embedding / Pooling | 15+ | BERT, RoBERTa, ColBERT |
| Encoder-Decoder | 5+ | Whisper, Nemotron-Parse |
| Speculative Decoding | 5+ | Eagle, Medusa, MTP |

---

## Model Interfaces: How Models Tell vLLM What They Can Do

Every model in vLLM implements one or more *interfaces* -- think of these as labels that say "I can do X." This lets vLLM treat different models uniformly while still supporting their unique features.

**Base interfaces (every model uses one of these):**

| Interface | What It Means | Used For |
|-----------|--------------|----------|
| `VllmModel` | "I can process inputs and run a forward pass" | All models |
| `VllmModelForTextGeneration` | "I can generate text token by token" | Chat and completion models (Llama, GPT, etc.) |
| `VllmModelForPooling` | "I can produce embeddings or classifications" | Embedding models (BERT, etc.) |

**Feature mix-ins (optional capabilities):**

| Mix-in | What It Means |
|--------|--------------|
| `SupportsMultiModal` | "I can process images, video, or audio alongside text" |
| `SupportsLoRA` | "I can use LoRA adapters for fine-tuned behavior" |
| `SupportsPP` | "I can be split across GPUs with pipeline parallelism" |
| `MixtureOfExperts` | "I use expert routing (Mixtral, DeepSeek-V2 style)" |
| `HasInnerState` | "I maintain internal state between steps (Mamba/Jamba)" |
| `IsAttentionFree` | "I don't use traditional attention (Mamba-style)" |
| `IsHybrid` | "I mix attention layers with Mamba-style layers" |

---

## Under the Hood: LazyRegisteredModel and Subprocess Safety

One of the most clever parts of the registry is `_LazyRegisteredModel`. Here's the problem it solves:

**The CUDA fork problem:** When vLLM starts, it may fork child processes (for multi-GPU serving). If CUDA has been initialized in the parent process before forking, the child processes will crash with `RuntimeError: Cannot re-initialize CUDA in forked subprocess`. Simply importing a model class can trigger CUDA initialization (e.g., through `torch.cuda` calls in module-level code).

**The solution:** `_LazyRegisteredModel` never imports the model class in the main process. Instead, it:

1. **Checks a JSON cache** at `~/.cache/vllm/modelinfos/` for previously computed model info
2. **If cache misses**, spawns a **separate subprocess** to import the model class and inspect its interfaces
3. **Saves the result** back to the cache so future startups are instant

```python
# From vllm/model_executor/models/registry.py (line ~789)
# Performed in another process to avoid initializing CUDA
mi = _run_in_subprocess(
    lambda: _ModelInfo.from_model_cls(self.load_model_cls())
)
```

The `_run_in_subprocess` function uses `cloudpickle` to serialize a lambda, runs it in a fresh Python process via `subprocess.run`, and reads back the result from a temp file. This way, the main process never touches CUDA until it's actually ready to.

**What gets inspected:** The subprocess checks which interfaces the model implements -- does it support LoRA? Multimodal? Pipeline parallelism? This information is cached as a `_ModelInfo` dataclass so the main process can make decisions without ever loading the model class.

---

## The Model Loading Pipeline

Loading a model involves four main steps:

| Step | What Happens | In Simple Terms |
|------|-------------|-----------------|
| 1. Create model class | vLLM looks up the right class from the registry | "Find the right blueprint" |
| 2. Initialize skeleton | Create the model structure with placeholder weights | "Build the frame of the house" |
| 3. Load weights | Download safetensors files, split weights for multi-GPU setups, apply quantization, map weight names | "Move in the furniture" |
| 4. Post-init | Warm up CUDA graphs, run compilation | "Do a test run to make sure everything works" |

### What Happens During Weight Loading

| Feature | What It Does |
|---------|-------------|
| Lazy loading | Models aren't fully loaded until actually needed -- this prevents wasting GPU memory in helper processes |
| Tensor Parallelism sharding | If you're using multiple GPUs, weights are automatically split across them |
| Pipeline Parallelism | Different layers go to different GPUs in the pipeline |
| Quantized weights | If the model is quantized (e.g., GPTQ, AWQ), special loaders handle the compressed format |
| Weight mapping | A `WeightsMapper` translates between HuggingFace weight names and vLLM's internal names (they're sometimes different) |
| Safetensors format | The default file format -- it's safe, fast, and memory-efficient because files can be memory-mapped |

---

## Under the Hood: The Registry Lookup Code Pattern

When `resolve_model_cls` is called, the registry follows a specific priority order:

```python
# Simplified from vllm/model_executor/models/registry.py
def resolve_model_cls(self, architectures, model_config):
    # 1. If user explicitly requested transformers backend
    if model_config.model_impl == "transformers":
        return self._try_resolve_transformers(architectures[0])

    # 2. If architecture not in registry AND model_impl == "auto",
    #    try transformers fallback BEFORE raising error
    if all(arch not in self.models for arch in architectures):
        return self._try_resolve_transformers(architectures[0])

    # 3. Normal path: look up in registry
    for arch in architectures:
        normalized = self._normalize_arch(arch, model_config)
        model_cls = self._try_load_model_cls(normalized)
        if model_cls is not None:
            return (model_cls, arch)

    # 4. Last-resort transformers fallback
    return self._try_resolve_transformers(architectures[0])
```

The triple fallback to transformers ensures that nearly any HuggingFace model can work with vLLM, even without a native implementation.

---

## The Transformers Fallback: No Code Needed

What if you want to run a model that isn't in vLLM's registry? vLLM has a fallback: it can use the model's implementation directly from the HuggingFace Transformers library.

**How it works:**
- If the model architecture isn't found in the registry, vLLM automatically tries to load it via Transformers
- This means any model that works with `transformers.AutoModelForCausalLM` can work with vLLM
- You may need to pass `--trust-remote-code` if the model uses custom Python code
- You can force it explicitly with `--model-impl transformers`

**Trade-off:** The Transformers fallback works, but native vLLM implementations are faster because they use optimized attention kernels and support features like tensor parallelism out of the box.

---

## How to Run Your First Model

Here's the quickest way to get started:

```bash
# Install vLLM
pip install vllm

# Serve a model (this downloads it automatically from HuggingFace)
vllm serve meta-llama/Llama-3.1-8B

# That's it! The API is now running at http://localhost:8000
# Test it with curl:
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B", "prompt": "Hello, world!", "max_tokens": 50}'
```

**Common options:**

```bash
# Use a specific GPU (e.g., GPU 1)
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B

# Run a quantized model (uses less memory)
vllm serve meta-llama/Llama-3.1-8B --quantization fp8

# Run a gated model (requires HuggingFace token)
vllm serve meta-llama/Llama-3.1-8B --token YOUR_HF_TOKEN

# Run a model with custom code
vllm serve some-org/custom-model --trust-remote-code
```

---

## Troubleshooting Model Loading

| Problem | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Cannot re-initialize CUDA in forked subprocess` | CUDA initialized before forking | Usually a vLLM bug -- report it. As a workaround, set `VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| `KeyError: 'SomeModelForCausalLM'` | Architecture not in registry | Try `--model-impl transformers` to use the HuggingFace fallback |
| `OutOfMemoryError` during loading | Model too big for GPU | Use `--quantization fp8` or `--tensor-parallel-size 2` to split across GPUs |
| `ValueError: No model architectures are specified` | `config.json` is missing `architectures` field | This is a broken checkpoint -- contact the model author |
| Model loads but outputs are garbage | Wrong dtype or broken quantization config | Try `--dtype auto` and check that the quantization method matches the checkpoint |
| Loading is very slow on first run | Registry cache miss -- inspecting model in subprocess | Normal on first run. Subsequent starts use the cache at `~/.cache/vllm/modelinfos/` |

---

## Adding a New Model to vLLM

If you want to add native support for a new model architecture:

1. Create a model file at `vllm/model_executor/models/your_model.py`
2. Implement the `VllmModelForTextGeneration` interface
3. Add a `load_weights()` method to load HuggingFace weights
4. Register it in `_TEXT_GENERATION_MODELS` in `registry.py`
5. **Or skip all that:** Just use the Transformers fallback -- no code changes needed!

---

## Key Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/models/registry.py` | Model registry -- maps architecture names to classes |
| `vllm/model_executor/models/interfaces.py` | Feature interfaces (LoRA, multimodal, etc.) |
| `vllm/model_executor/models/interfaces_base.py` | Base interfaces (VllmModel, etc.) |
| `vllm/model_executor/models/utils.py` | AutoWeightsLoader for downloading and sharding weights |
| `vllm/model_executor/model_loader/` | The full model loading pipeline |
| `vllm/model_executor/layers/` | Shared building blocks (attention, MoE routing, etc.) |

---

## Related Concepts

- **[Configuration](CONFIGURATION)** -- How to set model-related flags like `--model`, `--dtype`, `--trust-remote-code`, and `--tokenizer`
- **[Quantization](QUANTIZATION)** -- How to load quantized models that use less memory (applied during step 3 of loading)
- **[Attention Backends](ATTENTION-BACKENDS)** -- The optimized attention kernels that make native vLLM models faster than the Transformers fallback
- **[Distributed Inference](DISTRIBUTED)** -- How weights are split across multiple GPUs using tensor and pipeline parallelism
- **[Official Docs: Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)** -- Full list of natively supported model architectures
- **[Official Docs: Using Models](https://docs.vllm.ai/en/latest/models/generative_models.html)** -- The official guide for loading and running models
