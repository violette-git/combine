"""
Example: Python API usage for PowerQuant.

Shows the main ways to use PowerQuant programmatically.
"""

# ──────────────────────────────────────────────────────────────
# 1. Simplest usage: Engine.from_pretrained
# ──────────────────────────────────────────────────────────────
from powerquant import Engine

# HuggingFace model with TurboQuant KV compression (4-bit weights + compressed KV cache)
engine = Engine.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit=True,   # 4-bit weight quantization via bitsandbytes
    key_bits=4,          # TurboQuant: 4 bits for attention keys
    value_bits=2,        # TurboQuant: 2 bits for attention values (~5x KV compression)
    residual_window=128, # keep last 128 tokens in fp16 for generation quality
)

response = engine.generate(
    "Explain transformer attention in two sentences.",
    max_new_tokens=200,
)
print(response)


# ──────────────────────────────────────────────────────────────
# 2. Streaming generation
# ──────────────────────────────────────────────────────────────
print("\nStreaming:")
for token in engine.generate_stream("What is sparse neural activation?"):
    print(token, end="", flush=True)
print()


# ──────────────────────────────────────────────────────────────
# 3. Fine-grained configuration
# ──────────────────────────────────────────────────────────────
from powerquant import EngineConfig, TurboQuantConfig

config = EngineConfig(
    # Weight quantization
    load_in_4bit=True,
    # TurboQuant KV compression
    use_turboquant=True,
    turboquant=TurboQuantConfig(
        key_bits=4,
        value_bits=4,         # higher value bits = better quality
        residual_window=256,  # larger window = better quality, less compression
        protected_layers=6,   # first/last 6 layers at full precision
    ),
    # Generation defaults
    max_new_tokens=1024,
    temperature=0.8,
)

engine2 = Engine("Qwen/Qwen2.5-3B-Instruct", config)
print(engine2.generate("Hello!"))


# ──────────────────────────────────────────────────────────────
# 4. Direct TurboQuantCache (lower-level access)
# ──────────────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from powerquant import TurboQuantCache

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
print("\nKV cache memory report:", cache.memory_report())


# ──────────────────────────────────────────────────────────────
# 5. PowerInfer GGUF backend
# ──────────────────────────────────────────────────────────────
# (requires PowerInfer built via install.sh and a GGUF model file)

# engine_pi = Engine.powerinfer("/path/to/llama-2-7b.gguf", vram_budget_gb=6)
# print(engine_pi.generate("Hello, what can you do?"))

# Convert a HuggingFace model to PowerInfer GGUF:
# from powerquant.backends.powerinfer import PowerInferBackend
# backend = PowerInferBackend()
# backend.convert_model("meta-llama/Llama-2-7b-hf", "llama2-7b.gguf", quantize=True)
