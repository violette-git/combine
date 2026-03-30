"""
PowerQuant: Efficient LLM inference via PowerInfer + TurboQuant.

Combines two orthogonal optimizations:

  PowerInfer (https://github.com/Tiiny-AI/PowerInfer)
    Sparse-activation inference engine. Exploits the observation that most
    LLM neurons activate in power-law patterns: "hot" neurons are preloaded
    on GPU, "cold" neurons computed on CPU. Achieves up to 11x speedup vs
    llama.cpp on consumer GPUs. Works with .gguf model files.

  TurboQuant (https://github.com/tonbistudio/turboquant-pytorch)
    KV cache compression via vector quantization (ICLR 2026). Rotates KV
    vectors to normalize coordinate distributions, then applies Lloyd-Max
    optimal scalar quantization. V3 uses MSE-only compression with asymmetric
    key/value bit widths. Achieves ~5x compression at 3-bit average precision.
    Works with any HuggingFace model.

Quick start:
    from powerquant import Engine

    # HuggingFace model with TurboQuant KV compression + 4-bit weights
    engine = Engine.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit=True,
        key_bits=4,
        value_bits=2,
    )
    print(engine.generate("What is sparse neural activation?"))

    # Interactive chat
    engine.chat(system_prompt="You are a helpful assistant.")

    # PowerInfer GGUF model
    engine = Engine.powerinfer("/path/to/model.gguf", vram_budget_gb=8)
    print(engine.generate("Hello!"))
"""

from .engine import Engine
from .config import EngineConfig, TurboQuantConfig, PowerInferConfig
from .turboquant import TurboQuantCache, TurboQuantV3

__version__ = "0.1.0"
__all__ = [
    "Engine",
    "EngineConfig",
    "TurboQuantConfig",
    "PowerInferConfig",
    "TurboQuantCache",
    "TurboQuantV3",
]
