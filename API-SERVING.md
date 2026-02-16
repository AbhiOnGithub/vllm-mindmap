---
layout: default
title: API & Serving
---

# 09 - API & Serving

vLLM provides a production-ready serving stack that exposes OpenAI-compatible REST endpoints,
CLI tools for development and benchmarking, and a Python API for offline batch inference.
This page covers the full serving layer in detail.

> **Reference**: [OpenAI Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

![API Server](diagrams/api-server.png)

---

## OpenAI-Compatible Server

The server is launched with a single command and provides drop-in compatibility with the
OpenAI API specification. Any client library or tool that works with the OpenAI API can be
pointed at a vLLM server by changing the `base_url`.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Complete Endpoint Reference

The API server implements the following endpoints. Endpoints prefixed with `/v1/` follow the
[OpenAI API specification](https://platform.openai.com/docs/api-reference).

| Endpoint | Method | Purpose | Key Parameters |
|----------|--------|---------|----------------|
| `/v1/chat/completions` | POST | Chat-style completions (primary endpoint) | `messages`, `model`, `temperature`, `max_tokens`, `stream`, `tools`, `response_format` |
| `/v1/completions` | POST | Legacy text completions (prompt-in/text-out) | `prompt`, `model`, `max_tokens`, `temperature`, `logprobs` |
| `/v1/embeddings` | POST | Generate text embeddings | `input`, `model`, `encoding_format` |
| `/v1/responses` | POST | OpenAI Responses API (structured responses) | `input`, `model`, `instructions` |
| `/v1/audio/transcriptions` | POST | Speech-to-text transcription (Whisper models) | `file`, `model`, `language`, `response_format` |
| `/v1/audio/translations` | POST | Audio translation to English | `file`, `model` |
| `/v1/score` | POST | Cross-encoder scoring / reranking | `text_1`, `text_2`, `model` |
| `/v1/rerank` | POST | Document reranking | `query`, `documents`, `model` |
| `/v1/models` | GET | List available models (including LoRA adapters) | — |
| `/v1/models/{model_id}` | GET | Get specific model info | — |
| `/v1/realtime` | WS | Real-time bidirectional WebSocket API | WebSocket connection |
| `/health` | GET | Server health check (returns 200 when ready) | — |
| `/ping` | GET | Alias for health check | — |
| `/metrics` | GET | Prometheus metrics (latency, throughput, cache stats) | — |
| `/version` | GET | vLLM version info | — |
| `/tokenize` | POST | Tokenize text without inference | `prompt`, `model` |
| `/detokenize` | POST | Convert token IDs back to text | `tokens`, `model` |

### Chat Completions Endpoint Deep Dive

The `/v1/chat/completions` endpoint is the most commonly used. It supports the full range
of OpenAI chat parameters plus vLLM-specific extensions via `extra_body`:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    n=1,                    # Number of completions
    stream=True,            # Enable SSE streaming
    logprobs=True,          # Return log probabilities
    top_logprobs=5,         # Number of top logprobs per token
    # vLLM-specific extensions
    extra_body={
        "min_tokens": 10,                              # Minimum tokens to generate
        "repetition_penalty": 1.1,                     # Repetition penalty
        "guided_json": {"type": "object", ...},        # JSON schema constraint
        "guided_regex": r"\d{3}-\d{4}",               # Regex constraint
        "chat_template_kwargs": {"enable_thinking": True},  # Template kwargs
    },
)
```

### Embeddings Endpoint

For embedding models (e.g., `intfloat/e5-mistral-7b-instruct`):

```python
response = client.embeddings.create(
    model="intfloat/e5-mistral-7b-instruct",
    input=["Hello world", "How are you?"],
    encoding_format="float",   # "float" or "base64"
)
# response.data[0].embedding → [0.123, -0.456, ...]
```

---

## Request Processing Pipeline

Every incoming HTTP request flows through a carefully designed pipeline from raw HTTP
to GPU inference and back. Understanding this pipeline is critical for debugging and tuning.

![Request Lifecycle](diagrams/request-lifecycle.png)

```
HTTP Request (JSON body)
  │
  ▼
┌─────────────────────────────────────────────┐
│  FastAPI Application (api_server.py)         │
│                                              │
│  ┌─── CORS Middleware ──────────────────┐    │
│  │  Validates Origin headers            │    │
│  └──────────────────────────────────────┘    │
│  ┌─── Authentication Middleware ────────┐    │
│  │  Checks --api-key / Bearer token     │    │
│  └──────────────────────────────────────┘    │
│  ┌─── Request ID Middleware ───────────┐     │
│  │  Assigns unique X-Request-Id header  │    │
│  └──────────────────────────────────────┘    │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Serving Layer                               │
│  (OpenAIServingChat / OpenAIServingCompletion)│
│                                              │
│  1. Validate request against model caps      │
│  2. Resolve LoRA adapter (if specified)      │
│  3. Apply chat template (Jinja2 rendering)   │
│  4. Parse tool/function call format          │
│  5. Construct SamplingParams                 │
│  6. Handle multimodal inputs (images, etc.)  │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  InputProcessor (input_processor.py)         │
│                                              │
│  1. Tokenize prompt text                     │
│  2. Process multimodal data (resize, encode) │
│  3. Insert placeholder tokens for media      │
│  4. Build ProcessorInputs                    │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  AsyncLLM.add_request()                      │
│                                              │
│  Sends to EngineCore via ZMQ IPC             │
│  (EngineCoreClient manages the connection)   │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  EngineCore (Scheduler → Worker → Sampler)   │
│  ... GPU processing ...                      │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  OutputProcessor (output_processor.py)        │
│                                              │
│  1. Incremental detokenization               │
│  2. Log probability formatting               │
│  3. Stop condition checking (stop strings,   │
│     max_tokens, EOS token)                   │
│  4. Build RequestOutput with finish reason   │
└─────────────┬───────────────────────────────┘
              │
              ▼
HTTP Response (JSON) or SSE Stream
```

### Key Design Decisions

1. **ZMQ for IPC**: The API server and EngineCore communicate over ZeroMQ intra-process
   sockets, decoupling input processing from GPU execution.
2. **Async throughout**: The serving layer uses Python `asyncio` from HTTP handler to engine
   request, enabling high concurrency even with a single API server process.
3. **Incremental detokenization**: Tokens are detokenized as they arrive, enabling low-latency
   streaming without waiting for the full response.

---

## Streaming (SSE) vs Non-Streaming

### Non-Streaming Mode

In non-streaming mode, the server collects all generated tokens, assembles the complete
response, and returns a single JSON object:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": false
  }'
```

Response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
}
```

### Streaming Mode (Server-Sent Events)

With `"stream": true`, the server sends tokens as they are generated using the SSE protocol.
Each event is a `data:` line followed by a JSON chunk:

```
data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"content":" How"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}

data: [DONE]
```

### Stream Options

The `stream_options` parameter controls streaming behavior:

```json
{
  "stream": true,
  "stream_options": {
    "include_usage": true,
    "continuous_usage_stats": true
  }
}
```

- `include_usage`: Include token usage in the final chunk
- `continuous_usage_stats`: Report usage after every chunk (useful for monitoring)

### When to Use Streaming

| Scenario | Recommendation |
|----------|---------------|
| Interactive chat UI | Stream — users see tokens as they arrive |
| Batch processing | Non-stream — simpler to parse |
| Long responses | Stream — reduces perceived latency |
| Tool calling | Non-stream — need complete tool call JSON |
| Latency-sensitive APIs | Stream — first token arrives faster (lower TTFT) |

---

## Chat Template Rendering (Jinja2)

vLLM uses Jinja2 templates to convert the structured `messages` array into the raw prompt
format expected by each model. This is critical because different models expect different
prompt formats.

### How It Works

1. The chat template is loaded from the model's `tokenizer_config.json` (the `chat_template` field)
2. The `messages` array is passed to the Jinja2 template engine
3. The template renders the final prompt string (e.g., with `<|im_start|>` tokens for ChatML)
4. Special tokens like BOS/EOS are handled per the template logic

### Example: ChatML Format (Qwen, many others)

Template renders:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

### Example: Llama 3 Format

Template renders:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

### Custom Templates

You can override the chat template:

```bash
# Use a template file
vllm serve model --chat-template ./my_template.jinja

# Templates can include tool definitions, system prompts, etc.
```

### Template Variables Available

| Variable | Type | Description |
|----------|------|-------------|
| `messages` | list | The conversation messages |
| `tools` | list | Available tool definitions (if provided) |
| `add_generation_prompt` | bool | Whether to add the assistant prompt prefix |
| `bos_token` | str | Beginning of sequence token |
| `eos_token` | str | End of sequence token |
| `enable_thinking` | bool | Enable reasoning (for models like Qwen3) |

---

## Multi-Process API Server

When serving high-throughput workloads, input tokenization and output detokenization can
become CPU bottlenecks. The `--api-server-count` flag launches multiple API server processes
that share a single EngineCore.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --api-server-count 4
```

### Architecture

```
                Load Balancer (optional)
                        │
       ┌────────────────┼────────────────┐
       │                │                │
 ┌─────▼─────┐  ┌──────▼────┐  ┌───────▼───┐  ┌──────────┐
 │ API Srv 0  │  │ API Srv 1 │  │ API Srv 2 │  │ API Srv 3│
 │ (FastAPI)  │  │ (FastAPI) │  │ (FastAPI)  │  │ (FastAPI)│
 │ Port 8000  │  │ Port 8001 │  │ Port 8002  │  │ Port 8003│
 │            │  │           │  │            │  │          │
 │ Tokenize   │  │ Tokenize  │  │ Tokenize   │  │ Tokenize │
 │ Detokenize │  │ Detokenize│  │ Detokenize │  │ Detokenize│
 └─────┬──────┘  └─────┬─────┘  └─────┬──────┘  └─────┬────┘
       │               │              │               │
       └───────────────┴──────┬───────┴───────────────┘
                              │ ZMQ IPC
                     ┌────────▼────────┐
                     │   EngineCore    │
                     │  (single proc)  │
                     │                 │
                     │ Scheduler       │
                     │ Workers (GPUs)  │
                     │ KV Cache Mgmt   │
                     └─────────────────┘
```

### When to Use Multi-Process API Server

- **Long prompts**: Tokenization of long inputs can take significant CPU time
- **High QPS**: Many concurrent requests strain a single Python process
- **Multimodal workloads**: Image/video preprocessing is CPU-intensive
- **Always profile first**: Use `--collect-detailed-traces` to identify if input/output
  processing is actually your bottleneck before scaling API servers

---

## CLI Commands

vLLM provides several CLI commands for different use cases.

> **Reference**: [CLI Documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

### `vllm serve`

The primary command to launch an API server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --quantization fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --api-server-count 2 \
  --data-parallel-size 2 \
  --uvicorn-log-level warning
```

### `vllm chat`

Interactive chat session in your terminal:

```bash
vllm chat meta-llama/Llama-3.1-8B-Instruct
# Starts an interactive REPL:
# User: What is the capital of France?
# Assistant: The capital of France is Paris.
```

### `vllm complete`

Offline text completion:

```bash
vllm complete meta-llama/Llama-3.1-8B --prompt "The future of AI is"
```

### `vllm bench`

Benchmarking tool for measuring throughput and latency:

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 1000 \
  --input-len 512 \
  --output-len 128
```

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --batch-size 32
```

### `vllm collect-env`

Print environment information for debugging:

```bash
vllm collect-env
# Outputs: Python version, PyTorch version, CUDA version, GPU info, etc.
```

---

## Python Offline API

For batch processing without running a server, use the `LLM` class directly.

> **Reference**: [Offline Inference](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

### Basic Usage

```python
from vllm import LLM, SamplingParams

# Initialize the model (loads weights, allocates KV cache)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=256,
    min_tokens=10,
    repetition_penalty=1.1,
    stop=["</s>", "\n\n"],
    logprobs=5,
)

# Generate completions
prompts = [
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "What is the meaning of life?",
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated!r}\n")
```

### Chat API (Offline)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

messages_batch = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ],
    [
        {"role": "user", "content": "Write a limerick about GPUs."},
    ],
]

outputs = llm.chat(messages_batch, SamplingParams(temperature=0.8, max_tokens=256))
for output in outputs:
    print(output.outputs[0].text)
```

### SamplingParams Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Randomness (0 = greedy, higher = more random) |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `top_k` | -1 | Top-K sampling (-1 = disabled) |
| `max_tokens` | 16 | Maximum tokens to generate |
| `min_tokens` | 0 | Minimum tokens before stop conditions apply |
| `repetition_penalty` | 1.0 | Penalty for repeated tokens |
| `frequency_penalty` | 0.0 | Penalty based on token frequency |
| `presence_penalty` | 0.0 | Penalty based on token presence |
| `stop` | None | Stop strings (generation stops when encountered) |
| `stop_token_ids` | None | Stop token IDs |
| `logprobs` | None | Number of log probabilities to return |
| `best_of` | 1 | Generate N sequences, return best (beam search) |
| `seed` | None | Random seed for reproducibility |
| `n` | 1 | Number of output sequences per prompt |

---

## Authentication and CORS Configuration

### API Key Authentication

Protect your server with an API key:

```bash
vllm serve model --api-key "my-secret-key-12345"
```

Clients must include the key in the `Authorization` header:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"model": "model", "messages": [{"role": "user", "content": "Hi"}]}'
```

Or set it as the `api_key` in the OpenAI client:

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="my-secret-key-12345")
```

### CORS Configuration

For browser-based clients, configure Cross-Origin Resource Sharing:

```bash
vllm serve model \
  --allowed-origins '["http://localhost:3000", "https://myapp.com"]' \
  --allowed-methods '["GET", "POST"]' \
  --allowed-headers '["*"]'
```

| Flag | Description |
|------|-------------|
| `--allowed-origins` | JSON list of allowed origins (or `["*"]` for all) |
| `--allowed-methods` | JSON list of allowed HTTP methods |
| `--allowed-headers` | JSON list of allowed headers |

### SSL/TLS Configuration

For HTTPS support:

```bash
vllm serve model \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem \
  --ssl-ca-certs /path/to/ca.pem
```

---

## Prometheus Metrics

The `/metrics` endpoint exposes Prometheus-formatted metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:num_requests_running` | Gauge | Currently running requests |
| `vllm:num_requests_waiting` | Gauge | Requests waiting in queue |
| `vllm:num_preemptions_total` | Counter | Total preemptions |
| `vllm:gpu_cache_usage_perc` | Gauge | GPU KV cache utilization |
| `vllm:cpu_cache_usage_perc` | Gauge | CPU KV cache utilization |
| `vllm:request_success_total` | Counter | Successful requests |
| `vllm:avg_prompt_throughput_toks_per_s` | Gauge | Prompt processing throughput |
| `vllm:avg_generation_throughput_toks_per_s` | Gauge | Generation throughput |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end request latency |
| `vllm:time_to_first_token_seconds` | Histogram | Time to first token (TTFT) |
| `vllm:time_per_output_token_seconds` | Histogram | Inter-token latency (ITL) |

Access them at:
```bash
curl http://localhost:8000/metrics
```

---

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/entrypoints/openai/api_server.py` | ~800 | Main FastAPI application: endpoint registration, middleware, server startup |
| `vllm/entrypoints/openai/serving_chat.py` | ~1200 | Chat completions handler: template rendering, tool parsing, streaming |
| `vllm/entrypoints/openai/serving_completion.py` | ~400 | Legacy completions handler |
| `vllm/entrypoints/openai/serving_embedding.py` | ~200 | Embedding endpoint handler |
| `vllm/entrypoints/openai/serving_engine.py` | ~500 | Base class for all serving handlers |
| `vllm/entrypoints/openai/serving_responses.py` | ~300 | Responses API handler |
| `vllm/entrypoints/openai/protocol.py` | ~800 | Pydantic models for request/response schemas |
| `vllm/entrypoints/openai/tool_parsers/` | dir | Tool/function call output parsers |
| `vllm/entrypoints/cli/main.py` | ~100 | CLI entry point (argparse) |
| `vllm/entrypoints/cli/serve.py` | ~200 | `vllm serve` command implementation |
| `vllm/entrypoints/cli/bench.py` | ~300 | `vllm bench` benchmarking tools |
| `vllm/v1/engine/input_processor.py` | ~300 | Tokenization and multimodal preprocessing |
| `vllm/v1/engine/output_processor.py` | ~400 | Detokenization and streaming output assembly |

---

## Quick Start Recipes

### Minimal Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

### Production Server

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --enable-prefix-caching \
  --api-key "$VLLM_API_KEY" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-server-count 2 \
  --uvicorn-log-level warning
```

### Multi-LoRA Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules sql-lora=./adapters/sql \
                 code-lora=./adapters/code \
  --max-lora-rank 32
```

### Multimodal Server

```bash
vllm serve llava-hf/llava-v1.6-mistral-7b-hf \
  --max-model-len 4096 \
  --chat-template ./llava_template.jinja
```

