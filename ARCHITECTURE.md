---
layout: default
title: Architecture Overview
---

# 01 - Architecture Overview

![Architecture Overview](diagrams/architecture-overview.png)

> **Reference**: [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html)

---

## Table of Contents

1. [Why V1 Was Built](#why-v1-was-built)
2. [V1 vs V0 Comparison](#v1-vs-v0-comparison)
3. [V1 Multi-Process Design](#v1-multi-process-design)
4. [ZMQ / IPC Communication](#zmq--ipc-communication)
5. [Client Variants](#client-variants)
6. [Data Parallel Architecture](#data-parallel-architecture)
7. [Key Files](#key-files)

---

## Why V1 Was Built

### V0 Limitations

The original vLLM architecture (V0, sometimes called "legacy") was a single-process,
synchronous design. While it proved the value of PagedAttention and continuous batching,
it had several fundamental bottlenecks that motivated the rewrite.

**1. Single Process Bottleneck**

In V0, the entire pipeline -- API handling, scheduling, tokenization, detokenization,
and engine coordination -- ran in a single Python process. This meant:

- The GIL (Global Interpreter Lock) serialized CPU-bound work.
- Tokenization and detokenization blocked the main engine loop.
- FastAPI's async event loop competed with engine scheduling for CPU time.
- A slow tokenizer call could delay GPU execution, leaving expensive hardware idle.

**2. Synchronous `step()` Loop**

V0's core loop was `LLMEngine.step()`, a synchronous function that executed one full
cycle of schedule-execute-process per call:

```python
# V0 pseudocode -- everything is blocking
while True:
    scheduler_output = scheduler.schedule()        # CPU
    model_output     = workers.execute(scheduler_output)  # GPU (blocking wait)
    engine.process_output(model_output)            # CPU (detokenize, etc.)
    #  ^^^ GPU is IDLE while CPU processes output
```

The GPU sat idle during CPU phases (scheduling, detokenization). On production
workloads with thousands of concurrent requests, this serialization wasted 10-30%
of potential throughput.

**3. No Native Data Parallelism**

V0 had no built-in mechanism to run multiple engine replicas behind a load balancer.
Operators had to manually deploy N separate vLLM processes and add an external
load balancer (e.g., nginx). This made scaling awkward and prevented intelligent
request routing based on prefix cache state.

**4. Tight Coupling**

V0 mixed API-level concerns (HTTP handling, chat template rendering) with
engine-level concerns (scheduling, KV cache management) in the same process
address space. This made it difficult to:

- Test components in isolation.
- Replace the executor backend without affecting the API layer.
- Run the engine in a separate container from the API server.

### The V1 Vision

V1 was designed from scratch to address all of the above. The key design principles:

- **Process isolation**: Separate the API server (P0) from the engine core (P1).
- **Overlap CPU and GPU**: While the GPU executes step N, the CPU processes
  outputs from step N-1 and schedules step N+1.
- **Native data parallelism**: First-class support for multiple EngineCore
  replicas with prefix-aware load balancing.
- **Clean interfaces**: Well-defined message boundaries between processes
  using ZeroMQ (ZMQ) for inter-process communication.

V1 became the default engine starting with **vLLM v0.8**.

---

## V1 vs V0 Comparison

| Aspect | V0 (Legacy) | V1 (Current Default) |
|--------|-------------|---------------------|
| Engine | Single-process `LLMEngine.step()` | Multi-process: EngineCore in background |
| Communication | In-process Python calls | ZMQ/IPC between processes |
| Scheduling | Synchronous, blocks GPU | Async, overlaps with GPU compute |
| Data Parallel | Not built-in, manual process management | Native DP with prefix-aware load balancing |
| Input Processing | Inline in engine loop | Dedicated InputProcessor in API process |
| Output Processing | Inline in engine loop | Dedicated OutputProcessor in API process |
| Tokenization | Blocks engine step | Runs in API process, never blocks EngineCore |
| Default Since | v0.1 - v0.7 | v0.8+ |
| Status | Deprecated (removed in v0.9) | Active development |

---

## V1 Multi-Process Design

V1 splits vLLM into two main processes, with GPU workers spawned as child
processes (or threads) of the EngineCore.

### Process 0: API Server

The API server process (P0) handles everything **before** the engine and
everything **after** the engine. It never touches the GPU directly.

Responsibilities:

- **HTTP endpoint handling**: FastAPI serves the OpenAI-compatible API
  (`/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, etc.).
- **Chat template rendering**: Converts `messages` format to a prompt string
  using the model's Jinja2 chat template.
- **Tokenization**: Converts the prompt string to `token_ids` using the
  HuggingFace tokenizer. This is CPU-intensive and was a bottleneck when
  it lived in the engine process.
- **Multimodal preprocessing**: Extracts and preprocesses images, audio,
  or video from the request using the model's input processor.
- **Detokenization**: Converts output `token_ids` back to text, one token
  at a time for streaming, using an incremental detokenizer.
- **Streaming**: Sends Server-Sent Events (SSE) to the HTTP client as
  tokens are generated.

Key components in P0 (see the blue section in the architecture diagram above):

- **FastAPI App** -- OpenAI Serving Chat / Completions / Embeddings
- **AsyncLLM** -- front-end engine interface
- **InputProcessor** -- Tokenizer + Multimodal input mapper
- **OutputProcessor** -- IncrementalDetokenizer, LogprobsProcessor, Stop-string checker
- **EngineCoreClient** -- ZMQ sender/receiver, serializes requests via msgspec

### Process 1: EngineCore

The EngineCore process (P1) is the brain of vLLM. It runs the main engine
loop and coordinates GPU execution. It has **no knowledge** of HTTP, chat
templates, or tokenization -- it only works with token IDs and block tables.

Responsibilities:

- **Scheduling**: Decides which requests to run each step, manages
  waiting/running queues, handles preemption.
- **KV Cache Management**: Allocates and frees KV cache blocks, manages
  the prefix cache (hash-based block deduplication).
- **Execution Coordination**: Sends `SchedulerOutput` to GPU workers,
  collects `ModelRunnerOutput`.
- **Request State Tracking**: Maintains per-request state (generated tokens,
  stop conditions, block assignments).

Key components in P1 (see the green section in the architecture diagram above):

- **Main Loop** (runs continuously): receive requests via ZMQ, `scheduler.schedule()`, `executor.execute_model()`, `scheduler.update_from_output()`, send outputs via ZMQ
- **Scheduler** -- waiting queue (FIFO), running list, preemption policy (FCFS or Priority)
- **KVCacheManager** -- BlockPool (physical block allocation), prefix cache (hash to block), per-request block tables
- **Executor** -- manages GPU worker processes, handles TP and PP communication

### GPU Workers

Each GPU in the system runs a `GPUWorker` process (or thread). Workers are
managed by the Executor and do not communicate with P0 directly.

Responsibilities:

- **Model Runner**: Holds the model weights, manages the `InputBatch`,
  and runs the forward pass.
- **Attention Backend**: Executes PagedAttention kernels (FlashAttention,
  FlashInfer, etc.) using block tables from the KV cache.
- **Sampling**: Applies temperature, top-k, top-p, and penalties, then
  samples the next token.

Each GPU worker (see the orange section in the architecture diagram above) contains:

- **GPUModelRunner** -- InputBatch, Model (nn.Module), CUDA Graphs
- **Attention Backend** -- FlashAttention, FlashInfer, or other PagedAttention kernel
- **Sampler** -- Temperature, Top-k/Top-p, Penalties, Logprobs

### Full System Diagram

> The architecture diagram at the top of this page shows the complete system: P0 (blue), P1 (green), GPU workers (orange), and ZMQ/IPC (red arrow) connecting them.

---

## ZMQ / IPC Communication

V1 uses ZeroMQ (ZMQ) with IPC (inter-process communication) transport for
all communication between P0 and P1. ZMQ was chosen because:

- **Zero-copy messaging**: Large tensors can be sent without copying.
- **IPC transport**: Uses Unix domain sockets, avoiding TCP overhead.
- **Async-compatible**: Works with Python's asyncio event loop.
- **Language-agnostic framing**: Binary message framing without protocol overhead.

### Message Types

**P0 -> P1 (Requests)**:

| Message | Contents | When |
|---------|----------|------|
| `EngineCoreRequest` | `request_id`, `token_ids`, `sampling_params`, `mm_inputs` | New request arrives |
| `EngineCoreAbort` | `request_id` | Client disconnects |
| `EngineCoreProfile` | profiling flags | Profiling control |

**P1 -> P0 (Outputs)**:

| Message | Contents | When |
|---------|----------|------|
| `EngineCoreOutput` | `request_id`, `new_token_ids`, `finish_reason`, `logprobs` | Every engine step |

### Serialization

Messages are serialized using **msgspec** (a fast, typed serialization library).
msgspec was chosen over pickle for:

- Type safety (schemas are defined as `msgspec.Struct` classes).
- Speed (10-50x faster than pickle for small messages).
- Security (no arbitrary code execution on deserialization).

### Communication Pattern

The communication is **asynchronous** -- P0 can send `EngineCoreRequest` via ZMQ at any time, and P1 sends `EngineCoreOutput` back after every engine step. This allows true overlap:

- While GPU executes step N, P0 processes outputs from step N-1.
- While P0 processes outputs, P1 schedules step N+1.

---

## Client Variants

The `EngineCoreClient` has four implementations, each suited for a different
deployment scenario. The client is selected automatically based on configuration
but can be overridden.

### InprocClient

| Property | Value |
|----------|-------|
| Module | `vllm/v1/engine/core_client.py` |
| Process Model | Single process (no ZMQ) |
| API | Synchronous |
| Use Case | Offline batch inference, V0 compatibility, testing |

`InprocClient` runs the EngineCore **in the same process** as the API layer.
There is no IPC -- it calls `EngineCore.step()` directly. This is used for:

- The `LLM` class (offline batch inference).
- Unit tests that need deterministic behavior.
- Debugging where multi-process makes tracing difficult.

In this mode, everything runs in a single process: `LLM` -> `InprocClient` -> `EngineCore` (direct call) -> `Executor` -> Workers.

### SyncMPClient

| Property | Value |
|----------|-------|
| Module | `vllm/v1/engine/core_client.py` |
| Process Model | Multi-process with ZMQ |
| API | Synchronous (blocking) |
| Use Case | Synchronous `LLMEngine` users, simple scripts |

`SyncMPClient` spawns the EngineCore in a separate process and communicates
via ZMQ, but exposes a **synchronous** (blocking) API. Each call to
`get_output()` blocks until the engine produces a result.

### AsyncMPClient

| Property | Value |
|----------|-------|
| Module | `vllm/v1/engine/core_client.py` |
| Process Model | Multi-process with ZMQ |
| API | Asynchronous (asyncio) |
| Use Case | API server (default), high-throughput serving |

`AsyncMPClient` is the **default client for production serving**. It spawns
the EngineCore in a separate process and uses ZMQ with asyncio integration.
Features:

- Non-blocking `add_request()` and output polling.
- Integrates with FastAPI's async event loop.
- Supports concurrent request handling without thread pools.

This is the standard two-process model shown in the architecture diagram above: API Server (P0) communicates with EngineCore (P1) via ZMQ.

### DPLBAsyncMPClient

| Property | Value |
|----------|-------|
| Module | `vllm/v1/engine/core_client.py` |
| Process Model | Multi-process with multiple EngineCores |
| API | Asynchronous (asyncio) |
| Use Case | Data parallel deployment on multi-GPU nodes |

`DPLBAsyncMPClient` (Data Parallel Load Balanced Async Multi-Process Client)
manages **multiple EngineCore replicas** and routes requests intelligently.
See [Data Parallel Architecture](#data-parallel-architecture) below.

### Client Selection Logic

```
if data_parallel_size > 1:
    client = DPLBAsyncMPClient
elif asyncio mode:
    if multiprocess enabled:
        client = AsyncMPClient        # default for API server
    else:
        client = InprocClient          # --disable-frontend-multiprocessing
else:
    if multiprocess enabled:
        client = SyncMPClient
    else:
        client = InprocClient          # LLM class, offline
```

---

## Data Parallel Architecture

Data Parallelism (DP) allows running multiple independent EngineCore replicas,
each with its own set of GPUs, behind a single API server. This provides
linear throughput scaling for request-bound workloads.

### How It Works

With DP, a single API Server (P0) connects to multiple EngineCore replicas via `DPLBAsyncMPClient`. Each replica has its own Scheduler, KVCacheManager, Executor, and GPU set. For example, DP=3 with TP=2 uses 6 GPUs total: replica 0 on GPUs 0-1, replica 1 on GPUs 2-3, replica 2 on GPUs 4-5. See the [Distributed Inference](DISTRIBUTED) page for the full parallelism diagram.

### Load Balancing Strategy

The `DPLBAsyncMPClient` uses a **prefix-aware** load balancing strategy:

1. **Round-robin** as a baseline to distribute requests evenly.
2. **Prefix affinity**: Requests with similar prompt prefixes are routed to
   the same replica, maximizing prefix cache hit rates.
3. **Queue-length awareness**: Avoids routing to overloaded replicas.

This is more sophisticated than external load balancers, which have no
visibility into prefix cache state or engine queue depths.

### Configuration

```bash
# Start with 4-way data parallelism, each replica using 2 GPUs (TP=2)
vllm serve meta-llama/Llama-3.1-70B \
    --data-parallel-size 4 \
    --tensor-parallel-size 2

# Total GPUs needed: data_parallel_size * tensor_parallel_size = 8
```

---

## Key Files

| File | Purpose | Details |
|------|---------|---------|
| `vllm/v1/engine/core.py` | EngineCore | Main loop: receive requests, schedule, execute, return outputs. Runs in P1. |
| `vllm/v1/engine/core_client.py` | EngineCoreClient variants | All four client types (InprocClient, SyncMPClient, AsyncMPClient, DPLBAsyncMPClient). ZMQ setup and message serialization. |
| `vllm/v1/engine/async_llm.py` | AsyncLLM | Async front-end used by API server. Manages request lifecycle, connects InputProcessor, OutputProcessor, and EngineCoreClient. |
| `vllm/v1/engine/llm_engine.py` | LLMEngine | Synchronous V0-compatible interface. Wraps SyncMPClient or InprocClient. Used by the `LLM` class for offline batch inference. |
| `vllm/v1/engine/output_processor.py` | OutputProcessor | Detokenizes output tokens incrementally, processes logprobs, checks stop conditions, assembles `RequestOutput` for streaming. |
| `vllm/v1/engine/input_processor.py` | InputProcessor | Tokenizes prompts, preprocesses multimodal inputs, creates `EngineCoreRequest` objects ready for transmission to P1. |
| `vllm/v1/core/sched/scheduler.py` | Scheduler | Request scheduling: FCFS queuing, preemption, chunked prefill budget control. Produces `SchedulerOutput` each step. |
| `vllm/v1/core/kv_cache_manager.py` | KVCacheManager | High-level KV cache interface: block allocation, prefix cache lookup, per-request block table management. |
| `vllm/v1/worker/gpu_worker.py` | GPUWorker | Per-GPU worker process. Initializes GPU, creates model runner, handles weight loading and CUDA graph capture. |
| `vllm/v1/worker/gpu_model_runner.py` | GPUModelRunner | Manages the model forward pass: builds InputBatch, runs the model, returns logits. Handles CUDA graph execution. |
| `vllm/v1/executor/abstract.py` | Executor base | Abstract base for executors. Subclasses manage different execution backends (multiprocessing, Ray, etc.). |
| `vllm/entrypoints/openai/api_server.py` | API Server | FastAPI application. Registers OpenAI-compatible routes, creates `AsyncLLM`, handles HTTP lifecycle. |
| `vllm/entrypoints/openai/serving_chat.py` | Chat endpoint | Implements `/v1/chat/completions`: chat template rendering, streaming SSE, tool call parsing. |

---

## Further Reading

- [vLLM Architecture Overview (official)](https://docs.vllm.ai/en/latest/design/arch_overview.html)
- [Request Lifecycle](REQUEST-LIFECYCLE) -- how a request flows through V1
- [Scheduling & KV Cache](SCHEDULING-KV-CACHE) -- deep dive into the scheduler and PagedAttention
- [Model System](MODEL-SYSTEM) -- how models are registered, loaded, and executed

