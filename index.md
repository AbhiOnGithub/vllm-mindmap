# vLLM Complete Mind-Map

> A deep structural map of vLLM internals for a newcomer to build mental models fast.

## Navigation

| Topic | Covers |
|-------|--------|
| [Architecture Overview](ARCHITECTURE) | V1 engine design, process model, how pieces fit together |
| [Request Lifecycle](REQUEST-LIFECYCLE) | Step-by-step journey of a single inference request |
| [Scheduling & KV Cache](SCHEDULING-KV-CACHE) | Scheduler, PagedAttention, block management, prefix caching |
| [Model System](MODEL-SYSTEM) | Model registry, loading, 200+ architectures, weight sharding |
| [Attention Backends](ATTENTION-BACKENDS) | 20+ attention kernels, MLA, backend selection |
| [Quantization](QUANTIZATION) | GPTQ, AWQ, FP8, INT8, MXFP4 and 15+ methods |
| [Distributed Inference](DISTRIBUTED) | TP, PP, DP, EP, multi-node, NCCL |
| [Compilation & Performance](COMPILATION) | torch.compile, CUDA graphs, optimization levels |
| [API & Serving](API-SERVING) | OpenAI server, CLI, endpoints, streaming |
| [Advanced Features](ADVANCED-FEATURES) | LoRA, multimodal, tool calling, speculative decoding, reasoning |
| [Configuration Reference](CONFIGURATION) | Every config class, env vars, tuning knobs |
| [Codebase Map](CODEBASE-MAP) | Directory to purpose quick reference |

## Quick Mental Model

![vLLM V1 Architecture](diagrams/architecture-overview.png)
