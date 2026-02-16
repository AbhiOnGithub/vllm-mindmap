---
layout: default
title: Codebase Map
---

# 12 - Codebase Map

This page is your roadmap to the vLLM codebase. With 245+ model files, 29 quantization implementations, and thousands of lines in a single runner file, knowing where to look is half the battle. Use this page as your compass.

## Architecture Layers

The codebase is organized in layers, from user-facing APIs down to GPU kernels. Understanding this layering helps you find the right file fast.

| Layer | Directories | What Lives Here |
|-------|------------|-----------------|
| **API / Entrypoints** | `entrypoints/`, `cli/` | FastAPI server, OpenAI-compatible endpoints, `vllm serve` CLI |
| **Engine** | `v1/engine/` | EngineCore loop, ZMQ clients, async/sync frontends |
| **Scheduling** | `v1/core/sched/`, `v1/core/` | Request scheduling, KV cache block management, block pool |
| **Execution** | `v1/worker/`, `v1/executor/` | GPU workers, model runner (forward pass), input batching |
| **Model** | `model_executor/models/` | 245 model files, registry, interfaces |
| **Layers** | `model_executor/layers/` | Attention, quantization, MoE, rotary embeddings |
| **Compilation** | `compilation/` | torch.compile wrapper, CUDA graphs, fusion passes |
| **Distributed** | `distributed/` | Tensor/pipeline parallelism, NCCL communicators |
| **Config** | `config/` | All configuration dataclasses |
| **C++/CUDA** | `csrc/` | Hand-written CUDA kernels for attention, quantization, MoE |

## Directory → Purpose Quick Reference

```
vllm/
├── v1/                          ★ V1 ENGINE (current default)
│   ├── engine/
│   │   ├── core.py              EngineCore - main execution loop
│   │   ├── core_client.py       ZMQ clients (Inproc/Sync/Async/DP)
│   │   ├── async_llm.py         AsyncLLM - async front-end
│   │   ├── llm_engine.py        LLMEngine - sync V0-compat
│   │   ├── input_processor.py   Tokenization, MM preprocessing
│   │   └── output_processor.py  Detokenization, streaming
│   │
│   ├── core/
│   │   ├── sched/
│   │   │   ├── scheduler.py     ★ Main scheduler logic
│   │   │   ├── output.py        SchedulerOutput structures
│   │   │   └── request.py       Request state tracking
│   │   ├── kv_cache_manager.py  KV cache block management
│   │   ├── block_pool.py        Physical block pool + LRU
│   │   ├── kv_cache_coordinator.py  Multi-group coordination
│   │   └── kv_cache_utils.py    Config generation, memory checks
│   │
│   ├── worker/
│   │   ├── gpu_worker.py        Per-GPU worker process
│   │   ├── gpu_model_runner.py  ★ Forward pass execution
│   │   └── gpu_input_batch.py   Input batch construction
│   │
│   ├── attention/
│   │   └── backends/            21 attention backend implementations
│   │       ├── flash_attn.py
│   │       ├── flashinfer.py
│   │       └── ...
│   │
│   ├── sample/
│   │   ├── sampler.py           Token sampling logic
│   │   └── rejection_sampler.py Speculative decoding verification
│   │
│   ├── executor/
│   │   ├── abstract.py          Executor base class
│   │   └── multiproc_executor.py Multi-process executor
│   │
│   ├── spec_decode/             Speculative decoding proposers
│   └── kv_cache_interface.py    KVCacheSpec, KVCacheTensor, KVCacheConfig
│
├── model_executor/              ★ MODEL SYSTEM
│   ├── models/
│   │   ├── registry.py          ★ Model registry (200+ architectures)
│   │   ├── llama.py             Llama implementation
│   │   ├── qwen2.py             Qwen2 implementation
│   │   ├── deepseek_v2.py       DeepSeek V2/V3
│   │   ├── interfaces.py        Feature interfaces
│   │   └── ...                  245 model files total
│   │
│   ├── layers/
│   │   ├── attention/           Attention layer implementations
│   │   ├── quantization/        ★ 29 quantization files
│   │   ├── fused_moe/           MoE kernel implementations
│   │   └── rotary_embedding.py  RoPE implementations
│   │
│   └── model_loader/            Weight loading pipeline
│
├── entrypoints/                 ★ USER-FACING INTERFACES
│   ├── openai/
│   │   ├── api_server.py        FastAPI server
│   │   ├── serving_chat.py      Chat completions
│   │   └── serving_completion.py Legacy completions
│   │
│   └── cli/
│       ├── main.py              CLI entry point
│       └── serve.py             `vllm serve` command
│
├── distributed/                 ★ DISTRIBUTED SYSTEMS
│   ├── parallel_state.py        Process groups, communication
│   ├── device_communicators/    NCCL, Gloo communicators
│   ├── eplb/                    Expert load balancing
│   └── kv_transfer/             Disaggregated prefill
│
├── compilation/                 ★ PERFORMANCE
│   ├── wrapper.py               torch.compile wrapper
│   ├── cuda_graph.py            CUDA graph capture/replay
│   └── passes/                  Custom optimization passes
│       └── fusion/              10 fusion pass implementations
│
├── config/                      All configuration classes
│   ├── vllm.py                  VllmConfig (master)
│   ├── model.py                 ModelConfig
│   ├── cache.py                 CacheConfig
│   └── compilation.py           CompilationConfig
│
├── lora/                        LoRA adapter system
├── multimodal/                  Multimodal input processing
├── reasoning/                   Reasoning model support
├── tool_parsers/                30+ tool call parsers
├── tokenizers/                  Tokenizer implementations
├── logger.py                    Logging configuration
├── envs.py                      Environment variables
└── utils/                       Shared utilities

tests/                           Test suite (56 top-level entries)
├── v1/                          V1-specific tests
├── models/                      Model-specific tests
├── kernels/                     Kernel tests
├── quantization/                Quantization tests
├── distributed/                 Multi-GPU tests
├── compile/                     Compilation tests
├── lora/                        LoRA tests
├── entrypoints/                 API/endpoint tests
└── ...

docs/                            Documentation
├── design/                      Architecture docs
├── features/                    Feature docs
├── configuration/               Config guides
└── serving/                     Serving guides

csrc/                            C++/CUDA kernels
├── attention/                   Attention kernels
├── quantization/                Quantization kernels (CUDA)
└── moe/                         MoE kernels
```

## Where to Start Reading (and WHY)

| Priority | File | Why Read This |
|----------|------|---------------|
| 1 | `vllm/v1/engine/core.py` | **The heartbeat.** Every request flows through EngineCore's `step()` loop. Understanding this one file tells you how scheduling, execution, and output processing connect. |
| 2 | `vllm/v1/core/sched/scheduler.py` | **The brain.** Decides which requests run each step, manages preemption, and allocates KV cache blocks. If you're debugging throughput issues, start here. |
| 3 | `vllm/v1/worker/gpu_model_runner.py` | **The muscle.** The largest file in the codebase -- it handles input preparation, forward pass execution, CUDA graph warmup, and output extraction. Read the `execute_model()` method first. |
| 4 | `vllm/model_executor/models/registry.py` | **The phone book.** Maps architecture strings like `"LlamaForCausalLM"` to Python classes. Understanding lazy loading and subprocess inspection here is key to understanding model init. |
| 5 | `vllm/config/vllm.py` | **The master config.** `VllmConfig` aggregates all sub-configs (model, cache, parallel, compilation). Every component receives this. |
| 6 | `vllm/entrypoints/openai/api_server.py` | **The front door.** FastAPI app that exposes the OpenAI-compatible API. Short and readable -- good for understanding how HTTP requests enter the system. |
| 7 | `vllm/v1/core/kv_cache_manager.py` | **The memory manager.** Allocates and frees KV cache blocks. Surprisingly small for such a critical component -- well-factored. |
| 8 | `vllm/v1/sample/sampler.py` | **The output picker.** Converts logits to token IDs using temperature, top-p, top-k. Small and self-contained. |

### Under the Hood: Why gpu_model_runner.py is So Large

This file is the single largest in the codebase (thousands of lines) because it handles everything that happens on a GPU worker:
- Input tensor preparation (padding, position IDs, attention metadata)
- CUDA graph warmup at multiple batch sizes
- The actual `model.forward()` call
- Speculative decoding draft/verify logic
- Multi-modal input handling
- Profile runs for memory estimation

If you're looking for a specific behavior, search for these key methods: `execute_model`, `_prepare_inputs`, `_execute_graph_replay`, `profile_run`.

## Test Organization

Tests mirror the source structure. Key test directories:

| Directory | What It Tests | Run With |
|-----------|--------------|----------|
| `tests/v1/` | V1 engine, scheduler, workers | `pytest tests/v1/` |
| `tests/models/` | Per-model correctness (output matches HuggingFace) | `pytest tests/models/` |
| `tests/kernels/` | CUDA kernel correctness | `pytest tests/kernels/` |
| `tests/quantization/` | Quantization method accuracy | `pytest tests/quantization/` |
| `tests/distributed/` | Multi-GPU TP/PP tests | Requires multiple GPUs |
| `tests/entrypoints/` | API server, OpenAI compatibility | `pytest tests/entrypoints/` |
| `tests/compile/` | Compilation and CUDA graph tests | `pytest tests/compile/` |
| `tests/lora/` | LoRA adapter loading and inference | `pytest tests/lora/` |

Model-specific tests in `tests/models/registry.py` contain example HuggingFace model IDs for every registered architecture, making it a useful reference for finding test models.

## Files You Should Read First

| Priority | File | Why |
|----------|------|-----|
| 1 | `vllm/v1/engine/core.py` | Understand the main engine loop |
| 2 | `vllm/v1/core/sched/scheduler.py` | Understand scheduling |
| 3 | `vllm/v1/worker/gpu_model_runner.py` | Understand GPU execution |
| 4 | `vllm/model_executor/models/registry.py` | Understand model system |
| 5 | `vllm/config/vllm.py` | Understand configuration |
| 6 | `vllm/entrypoints/openai/api_server.py` | Understand API serving |
| 7 | `vllm/v1/core/kv_cache_manager.py` | Understand KV cache |
| 8 | `vllm/v1/sample/sampler.py` | Understand token sampling |
