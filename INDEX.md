# vLLM Complete Mind-Map

> A deep structural map of vLLM internals for a newcomer to build mental models fast.

## Navigation

| # | File | Covers |
|---|------|--------|
| 01 | [Architecture Overview](01-ARCHITECTURE.md) | V1 engine design, process model, how pieces fit together |
| 02 | [Request Lifecycle](02-REQUEST-LIFECYCLE.md) | Step-by-step journey of a single inference request |
| 03 | [Scheduling & KV Cache](03-SCHEDULING-KV-CACHE.md) | Scheduler, PagedAttention, block management, prefix caching |
| 04 | [Model System](04-MODEL-SYSTEM.md) | Model registry, loading, 200+ architectures, weight sharding |
| 05 | [Attention Backends](05-ATTENTION-BACKENDS.md) | 20+ attention kernels, MLA, backend selection |
| 06 | [Quantization](06-QUANTIZATION.md) | GPTQ, AWQ, FP8, INT8, MXFP4 and 15+ methods |
| 07 | [Distributed Inference](07-DISTRIBUTED.md) | TP, PP, DP, EP, multi-node, NCCL |
| 08 | [Compilation & Performance](08-COMPILATION.md) | torch.compile, CUDA graphs, optimization levels |
| 09 | [API & Serving](09-API-SERVING.md) | OpenAI server, CLI, endpoints, streaming |
| 10 | [Advanced Features](10-ADVANCED-FEATURES.md) | LoRA, multimodal, tool calling, speculative decoding, reasoning |
| 11 | [Configuration Reference](11-CONFIGURATION.md) | Every config class, env vars, tuning knobs |
| 12 | [Codebase Map](12-CODEBASE-MAP.md) | Directory → purpose quick reference |

## Quick Mental Model

![vLLM V1 Architecture](diagrams/architecture-overview.png)
