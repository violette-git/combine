# PowerQuant

Efficient LLM inference by combining two complementary optimizations:

| | [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) | [TurboQuant](https://github.com/tonbistudio/turboquant-pytorch) |
|---|---|---|
| **What** | Sparse-activation inference engine | KV cache vector quantization |
| **How** | Hot neurons on GPU, cold on CPU | Rotation + Lloyd-Max scalar quant |
| **Gain** | Up to 11× faster than llama.cpp | ~5× smaller KV cache |
| **Format** | PowerInfer GGUF (`.gguf`) | Any HuggingFace causal LM |
| **Precision** | INT4 weights | ~3-bit avg KV (K4/V2 default) |

Together they let you run **larger models** and **longer contexts** on consumer hardware.

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [CLI reference](#cli-reference)
6. [Python API](#python-api)
7. [Metrics](#metrics)
8. [Configuration](#configuration)
9. [Benchmarks](#benchmarks)
10. [Architecture](#architecture)

---

## How it works

### PowerInfer — sparse weight inference

Most LLM neurons follow a **power-law activation pattern**: a small set of
"hot" neurons fire on nearly every input, while the majority ("cold" neurons)
activate rarely and differently per input.

PowerInfer exploits this:
- **Hot neurons** are preloaded on GPU — fast, always ready.
- **Cold neurons** are computed on CPU — only when needed, using sparse operators.

Result: up to 11× speedup vs llama.cpp on an RTX 4090 while maintaining
the same output quality as a full-precision run.

Requires models in **PowerInfer GGUF format** — standard GGUF extended with
neuron activation statistics and predictor weights. Convert any HuggingFace
model with `powerquant convert`.

### TurboQuant — KV cache compression

The key-value (KV) cache grows linearly with context length and is a major
memory bottleneck for long-context generation.

TurboQuant V3 compresses the KV cache using:
1. **Random rotation** — applies a random orthogonal matrix to normalize the
   coordinate distribution of each vector.
2. **Lloyd-Max quantization** — solves for optimal scalar quantization centroids
   for the resulting distribution (Beta/Gaussian).
3. **Bit-packing** — packs multiple indices per byte for maximum storage efficiency.
4. **Asymmetric bits** — keys use more bits than values (keys are more
   precision-sensitive for attention score computation).
5. **Residual window** — the most recent `residual_window` tokens stay in fp16
   to preserve generation quality at the end of the sequence.
6. **Layer protection** — the first and last N layers use full precision, as
   these are most sensitive to quantization error.

At K4/V2 (the default), TurboQuant achieves **~5× KV memory reduction** with
minimal impact on generation quality.

---

## Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| transformers | 4.40+ |
| CUDA (optional) | 11.8+ for GPU acceleration |
| cmake | 3.17+ (for PowerInfer build) |
| C++ compiler | GCC 9+ / Clang 12+ / MSVC 2019+ |

PowerInfer requires a compiled C++ binary. The `install.sh` script handles
cloning, configuring, and building it automatically.

TurboQuant (HF backend) works CPU-only, but GPU is strongly recommended for
practical inference speeds.

---

## Installation

### Option A — Full install (recommended)

Downloads dependencies, clones PowerInfer, builds the C++ binary, and installs
the `powerquant` Python package in one command:

```bash
git clone https://github.com/violette-git/combine
cd combine
bash install.sh
```

**GPU detection is automatic:**
- NVIDIA GPU detected → builds with CUDA (`-DLLAMA_CUBLAS=ON`)
- AMD GPU detected → builds with HIP (`-DLLAMA_HIPBLAS=ON`)
- No GPU → CPU-only build

**Options:**

```bash
bash install.sh --cpu-only        # force CPU-only build of PowerInfer
bash install.sh --skip-powerinfer # Python/TurboQuant only (no C++ build)
```

### Option B — Python-only (TurboQuant / HF backend only)

If you only want the HuggingFace + TurboQuant backend and don't need PowerInfer:

```bash
pip install -r requirements.txt
pip install -e .
```

For CUDA-enabled PyTorch (replace `cu128` with your CUDA version):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

### Verify

```bash
powerquant info
```

Expected output:

```
──────────────────────────────────────────────────────
  PowerQuant v0.1.0
──────────────────────────────────────────────────────
  Python    : 3.11.5
  Platform  : Linux x86_64
  PyTorch   : 2.3.0+cu121
    GPU 0  : NVIDIA RTX 4090  (23.7 GB total, 22841 MB free)
  Transformers : 4.43.0
  bitsandbytes : 0.43.1
  PowerInfer   : /path/to/combine/vendor/PowerInfer/build/bin/main  [ready]
──────────────────────────────────────────────────────
```

---

## Quick start

### Interactive chat

```bash
# HuggingFace model with 4-bit weights + TurboQuant KV compression
powerquant chat --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit

# Same but show metric summary after each turn
powerquant chat --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit --metrics

# Full precision (no quantization, no TurboQuant)
powerquant chat --model gpt2 --no-turboquant

# PowerInfer GGUF model
powerquant chat --model ./llama2-7b.gguf
```

### Single generation

```bash
# Generate and print
powerquant generate --model gpt2 --prompt "The future of AI is"

# With detailed metrics report
powerquant generate --model gpt2 --prompt "Hello world" --metrics

# Stream output token by token
powerquant generate --model gpt2 --prompt "Tell me a story" --stream

# Machine-readable JSON metrics
powerquant generate --model gpt2 --prompt "Hello" --json
```

### Run benchmark

```bash
# Tests needle-in-haystack retrieval at 2K/4K/8K context
powerquant benchmark --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit

# Specific context lengths
powerquant benchmark --model Qwen/Qwen2.5-3B-Instruct --context-lengths 1024,2048

# Export results as JSON
powerquant benchmark --model Qwen/Qwen2.5-3B-Instruct --json > results.json
```

---

## CLI reference

### `powerquant chat`

Interactive streaming chat session.

```
--model, -m MODEL         HuggingFace model ID, local path, or .gguf file  [required]
--system TEXT             System prompt prepended to every turn
--metrics                 Print one-line metric summary after each response
--load-in-4bit            4-bit NF4 weight quantization (bitsandbytes)
--load-in-8bit            8-bit weight quantization (bitsandbytes)
--no-turboquant           Disable TurboQuant KV compression
--key-bits N              Key vector bit-width (default: 4)
--value-bits N            Value vector bit-width (default: 2)
--residual-window N       Uncompressed recent tokens (default: 128)
--protected-layers N      First/last N layers at full precision (default: 4)
--max-new-tokens N        Max tokens per response (default: 512)
--temperature F           Sampling temperature (default: 0.7)
```

**Session commands:**
- `exit` / `quit` — end the session
- `clear` — reset conversation history
- `metrics` — toggle per-turn metric summary on/off

### `powerquant generate`

Single prompt generation.

```
--model, -m MODEL         HuggingFace model ID, local path, or .gguf file  [required]
--prompt, -p TEXT         Input prompt  [required]
--metrics                 Print full metrics report after generation
--json                    Output metrics as JSON to stdout
--quiet                   Suppress the one-line metric summary on stderr
--stream                  Stream tokens as produced (no metrics)
--greedy                  Greedy decoding (no sampling)
[+ all --load-in-*, --*-turboquant, --key-bits, ... args from chat]
```

### `powerquant benchmark`

Needle-in-haystack accuracy + throughput test.

```
--model, -m MODEL         HuggingFace model ID  [required]
--load-in-4bit            4-bit weight quantization
--context-lengths LIST    Comma-separated sizes (default: 2048,4096,8192)
--json                    Also print full results as JSON
```

Tests five configurations:
1. FP16 baseline (no compression)
2. K4/V2 ~5.1× (default)
3. K4/V4 ~3.2× (high quality)
4. K3/V2 ~6.1× (aggressive)
5. K4/V2 no residual window

### `powerquant convert`

Convert a HuggingFace model to PowerInfer GGUF format.

```
--model, -m MODEL_ID      HuggingFace model ID  [required]
--output, -o PATH         Output .gguf file path  [required]
--no-quantize             Keep fp16 (skip INT4 quantization)
```

Example:
```bash
powerquant convert --model meta-llama/Llama-2-7b-hf --output llama2-7b.gguf
```

### `powerquant build`

Build the PowerInfer C++ binary from source.

```
--cpu-only    Build without GPU support
--hip         Build with AMD HIP/ROCm
```

### `powerquant info`

Show Python, PyTorch, GPU, and PowerInfer availability.

---

## Python API

### Basic usage

```python
from powerquant import Engine

# Load with 4-bit weights + TurboQuant KV compression (recommended)
engine = Engine.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit=True,
)

# Generate — returns a string
text = engine.generate("Explain transformer attention in two sentences.")
print(text)

# Generate with metrics — returns GenerationResult
result = engine.generate_with_metrics("Explain transformer attention.")
print(result.text)
print(result.summary())       # one-line
result.print_report()         # full table
metrics = result.to_dict()    # for logging/JSON
```

### Streaming

```python
# Stream tokens (for interactive UIs)
for token in engine.generate_stream("Tell me a story."):
    print(token, end="", flush=True)
print()
```

### Interactive chat

```python
engine.chat(
    system_prompt="You are a helpful assistant.",
    show_metrics=True,   # print stats after each turn
)
```

### Fine-grained configuration

```python
from powerquant import Engine, EngineConfig, TurboQuantConfig

config = EngineConfig(
    load_in_4bit=True,
    use_turboquant=True,
    turboquant=TurboQuantConfig(
        key_bits=4,
        value_bits=4,          # higher bits = better quality
        residual_window=256,   # larger window = better quality
        protected_layers=6,    # more protected layers = safer
    ),
    max_new_tokens=1024,
    temperature=0.8,
)

engine = Engine("Qwen/Qwen2.5-7B-Instruct", config)
result = engine.generate_with_metrics("Hello!")
result.print_report()
```

### PowerInfer backend

```python
# Auto-detected from .gguf extension
engine = Engine.from_pretrained("/path/to/model.gguf")

# Or explicit
engine = Engine.powerinfer(
    "/path/to/model.gguf",
    vram_budget_gb=8,
    n_threads=12,
)

result = engine.generate_with_metrics("What is a transformer?")
result.print_report()
```

### Direct TurboQuantCache (lower-level)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from powerquant import TurboQuantCache
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

cache = TurboQuantCache(
    key_bits=4,
    value_bits=2,
    residual_window=64,
    n_layers=model.config.n_layer,
)

inputs = tokenizer("The quick brown fox", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        past_key_values=cache,
        use_cache=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(cache.memory_report())
```

---

## Metrics

Every call to `generate_with_metrics()` returns a `GenerationResult` object.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Generated response text |
| `input_tokens` | `int` | Tokens in the prompt |
| `output_tokens` | `int` | Tokens generated |
| `latency_s` | `float` | Total wall-clock time (seconds) |
| `prefill_s` | `float` | Time to first token / prefill time (seconds) |
| `tokens_per_second` | `float` | Output throughput |
| `kv.compressed_mb` | `float` | TurboQuant compressed KV size (MB) |
| `kv.fp16_equiv_mb` | `float` | What the cache would use in fp16 (MB) |
| `kv.compression_ratio` | `float` | Compression factor (e.g. 4.8 = 4.8× smaller) |
| `kv.savings_mb` | `float` | Memory saved vs fp16 (MB) |
| `kv.compressed_tokens` | `int` | Tokens stored compressed |
| `kv.fp16_tokens` | `int` | Tokens in the residual window (uncompressed) |
| `kv.key_bits` | `int` | Effective key bit-width used |
| `kv.value_bits` | `int` | Effective value bit-width used |
| `gpu.before_mb` | `int` | GPU memory before generation (MB) |
| `gpu.peak_mb` | `int` | Peak GPU memory during generation (MB) |
| `gpu.after_mb` | `int` | GPU memory after generation (MB) |
| `gpu.delta_mb` | `int` | Additional GPU memory used (after − before) |
| `backend` | `str` | `"hf"` or `"powerinfer"` |
| `model` | `str` | Model name/path |
| `weight_quantization` | `str` | `"fp16"`, `"int8"`, or `"int4"` |

### Methods

```python
result.summary()       # → "[43.2 tok/s | 128 in / 87 out | KV 4.8x | GPU +8 MB]"
result.print_report()  # → formatted multi-section table to stdout
result.to_dict()       # → plain dict for JSON export / logging
```

### Example report

```
────────────────────────────────────────────────────────
  PowerQuant Generation Report
────────────────────────────────────────────────────────
  Model   : Qwen/Qwen2.5-3B-Instruct
  Backend : hf
  Weights : int4

  ── Tokens ──────────────────────────────────────
  Input tokens      :       128
  Output tokens     :        87
  Total tokens      :       215

  ── Timing ──────────────────────────────────────
  Total latency     :    2.013 s
  Prefill (TTFT)    :    0.312 s
  Decode time       :    1.701 s
  Throughput        :     43.2 tok/s

  ── KV Cache (TurboQuant) ───────────────────────
  Key bits          :        4
  Value bits        :        2
  Residual window   :      128 tokens
  Compressed tokens :        0
  FP16 window       :      128 tokens
  Compressed size   :     0.00 MB
  FP16 equivalent   :     0.00 MB
  Compression ratio :     4.80x
  Memory saved      :    12.40 MB

  ── GPU Memory ──────────────────────────────────
  Before generate   :    3,721 MB
  Peak during run   :    3,849 MB
  After generate    :    3,734 MB
  Delta             :      +13 MB
────────────────────────────────────────────────────────
```

### JSON export

```python
import json

result = engine.generate_with_metrics("Hello!")
print(json.dumps(result.to_dict(), indent=2))
```

```json
{
  "input_tokens": 5,
  "output_tokens": 34,
  "total_tokens": 39,
  "latency_s": 0.812,
  "prefill_s": 0.091,
  "tokens_per_second": 41.87,
  "kv": {
    "compressed_mb": 0.0,
    "fp16_equiv_mb": 0.0,
    "compression_ratio": 1.0,
    "compressed_tokens": 0,
    "fp16_tokens": 5,
    "layers_compressed": 0,
    "key_bits": 4,
    "value_bits": 2,
    "residual_window": 128,
    "savings_mb": 0.0
  },
  "gpu": {
    "before_mb": 3721,
    "peak_mb": 3812,
    "after_mb": 3728,
    "delta_mb": 7
  },
  "backend": "hf",
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "weight_quantization": "int4"
}
```

---

## Configuration

### EngineConfig

Top-level configuration object. Pass to `Engine(model, config)`.

```python
from powerquant import EngineConfig, TurboQuantConfig, PowerInferConfig

config = EngineConfig(
    # Backend selection
    backend="auto",          # "auto" | "hf" | "powerinfer"

    # Generation defaults (override per-call)
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,

    # HF backend: weight quantization
    load_in_4bit=False,
    load_in_8bit=False,
    torch_dtype="auto",      # "auto" | "float16" | "bfloat16"
    device_map="auto",       # "auto" | "cpu" | "cuda:0"

    # TurboQuant KV compression (HF backend)
    use_turboquant=True,
    turboquant=TurboQuantConfig(
        key_bits=4,
        value_bits=2,
        residual_window=128,
        protected_layers=4,
        protected_bits=8,
    ),

    # PowerInfer C++ backend
    powerinfer=PowerInferConfig(
        binary_path=None,      # auto-detected
        vram_budget_gb=None,   # no limit
        n_threads=8,
        n_gpu_layers=-1,       # auto
        context_size=4096,
    ),
)
```

### TurboQuant presets

| Preset | key_bits | value_bits | Approx ratio | Use case |
|--------|----------|------------|--------------|----------|
| High quality | 4 | 4 | ~3.2× | When quality matters most |
| **Default** | **4** | **2** | **~5.1×** | **Best balance** |
| Aggressive | 3 | 2 | ~6.1× | Maximum memory savings |
| No compression | 16 | 16 | 1× | Baseline / debugging |

`residual_window=128` is safe for most models. Increase to 256–512 for
better quality on very long contexts.

`protected_layers=4` protects the first and last 4 layers. Increase if you
see quality degradation on short sequences.

---

## Benchmarks

### Memory savings (7B model, 4K context, 32 heads, head_dim=128)

| Config | KV size | vs fp16 |
|--------|---------|---------|
| fp16 baseline | 512 MB | 1.0× |
| TurboQuant K4/V4 | 160 MB | 3.2× |
| TurboQuant K4/V2 | 100 MB | 5.1× |
| TurboQuant K3/V2 | 84 MB | 6.1× |

### Needle-in-haystack accuracy (Qwen2.5-3B, 4-bit)

Retrieves a secret code name from a long document. "FOUND" = correct retrieval.

| Config | 2K ctx | 4K ctx | 8K ctx |
|--------|--------|--------|--------|
| fp16 baseline | FOUND | FOUND | FOUND |
| K4/V2 (~5×) | FOUND | FOUND | FOUND |
| K4/V4 (~3×) | FOUND | FOUND | FOUND |
| K3/V2 (~6×) | FOUND | FOUND | MISS |

### PowerInfer throughput (RTX 4090, Falcon-40B)

| System | tok/s |
|--------|-------|
| llama.cpp (fp16) | ~1.1 |
| PowerInfer INT4 | ~13.2 (peak 29.1) |

---

## Architecture

```
powerquant/
│
├── engine.py              Unified Engine class
│                          └── auto-selects backend from model format
│
├── config.py              EngineConfig, TurboQuantConfig, PowerInferConfig
│
├── metrics.py             GenerationResult, KVMetrics, GPUMetrics, MetricsTimer
│
├── cli.py                 `powerquant` CLI entry point
│
├── backends/
│   ├── hf.py              HuggingFace backend
│   │                      └── loads model via transformers
│   │                      └── wraps generate() with TurboQuantCache
│   │                      └── collects timing + GPU + KV metrics
│   │
│   └── powerinfer.py      PowerInfer C++ backend
│                          └── subprocess wrapper for the C++ binary
│                          └── model conversion HF → PowerInfer GGUF
│                          └── collects timing metrics
│
└── turboquant/            KV cache compression library (from turboquant-pytorch)
    ├── lloyd_max.py        Lloyd-Max optimal scalar quantizer
    │                       └── solve_lloyd_max(): iterative centroid solver
    │                       └── LloydMaxCodebook: precomputed codebook
    │
    ├── turboquant.py       Core quantizers
    │                       └── TurboQuantMSE: rotation + Lloyd-Max (stage 1)
    │                       └── TurboQuantProd: MSE + QJL inner product (stage 1+2)
    │                       └── TurboQuantKVCache: original cache wrapper (V1/V2)
    │
    ├── compressors_v3.py   TurboQuant V3 (MSE-only, asymmetric)
    │                       └── MSECompressor: bit-packed compress/decompress
    │                       └── TurboQuantV3: asymmetric K/V + residual window
    │
    └── cache.py            DynamicCache integration
                            └── TurboQuantCache: drop-in HF cache replacement
                            └── Manages per-layer compressed chunk storage
                            └── memory_report(): compression stats
```

### Data flow: HF + TurboQuant

```
User prompt
    │
    ▼
tokenizer.encode()
    │
    ▼
model.generate(past_key_values=TurboQuantCache)
    │
    ├── [each transformer layer]
    │       │
    │       ▼
    │   TurboQuantCache.update(key_states, value_states, layer_idx)
    │       │
    │       ├── Append new tokens to fp16 buffer
    │       ├── If buffer > residual_window:
    │       │       compress overflow with TurboQuantV3
    │       │       MSECompressor.compress():
    │       │           normalize → rotate (Pi) → quantize → bit-pack
    │       ├── Decompress all compressed chunks
    │       │       MSECompressor.decompress():
    │       │           unpack bits → dequantize → unrotate (Pi^T) → rescale
    │       └── Concatenate decompressed + fp16 window → return to attention
    │
    ▼
tokenizer.decode()
    │
    ▼
GenerationResult(text, timing, kv_metrics, gpu_metrics)
```

### Data flow: PowerInfer

```
User prompt
    │
    ▼
PowerInferBackend.generate_with_metrics()
    │
    ▼
subprocess: ./vendor/PowerInfer/build/bin/main -m model.gguf -p prompt ...
    │
    ├── [C++ engine]
    │       ├── Load hot neurons → GPU
    │       ├── For each token:
    │       │       predictor: which cold neurons activate?
    │       │       hot neurons: GPU matmul
    │       │       cold neurons: CPU sparse matmul
    │       └── Stream token to stdout
    │
    ▼
GenerationResult(text, timing, gpu_metrics)
```

---

## Troubleshooting

**`PowerInfer binary not found`**
```bash
powerquant build          # rebuild from source
# or:
bash install.sh           # full reinstall
```

**`CUDA out of memory` during inference**
- Use `--load-in-4bit` to reduce weight memory
- Use `--vram-budget X` (PowerInfer) to limit hot neuron GPU allocation
- Reduce `--residual-window` to decrease fp16 portion of KV cache

**`bitsandbytes` not found (4-bit/8-bit unavailable)**
```bash
pip install bitsandbytes
```
Note: bitsandbytes requires a CUDA-capable GPU. CPU-only systems cannot use
4-bit or 8-bit weight quantization.

**TurboQuant causes slow generation**
TurboQuant adds compression/decompression overhead per forward pass. For
short sequences (< `residual_window` tokens), everything stays in fp16 and
there is no overhead. For very long sequences, the decompression cost is a
single matrix multiply per layer — typically <5% of total generation time on GPU.

**`DynamicCache` compatibility errors**
TurboQuantCache requires `transformers >= 4.40`. Update with:
```bash
pip install --upgrade transformers
```

---

## License

MIT. See [LICENSE](LICENSE).

**Third-party sources:**
- PowerInfer: MIT — https://github.com/Tiiny-AI/PowerInfer
- TurboQuant: MIT — https://github.com/tonbistudio/turboquant-pytorch
