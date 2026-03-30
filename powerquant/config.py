"""
Configuration dataclasses for PowerQuant.

Every setting has a sensible default. The most important knobs:
  - backend: "auto" picks based on model format (.gguf -> PowerInfer, else HF)
  - load_in_4bit: combine with TurboQuant for maximum memory savings
  - key_bits / value_bits: TurboQuant precision (4/2 is a good default)
  - residual_window: recent tokens kept in fp16 (128 is a good default)
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TurboQuantConfig:
    """
    TurboQuant V3 KV cache compression settings.

    Recommended presets:
        - High quality: key_bits=4, value_bits=4
        - Balanced:     key_bits=4, value_bits=2  (default, ~5x compression)
        - Aggressive:   key_bits=3, value_bits=2  (~6x compression)

    residual_window: number of recent tokens stored in fp16 (uncompressed).
        Higher = better generation quality, slightly less compression.
        128 is safe for most models. Set to 0 to compress everything.

    protected_layers: first and last N layers of the model use protected_bits
        instead of key_bits/value_bits. This preserves accuracy in the most
        sensitive transformer layers. Set to 0 to disable protection.
    """
    key_bits: int = 4
    value_bits: int = 2
    residual_window: int = 128
    protected_layers: int = 4
    protected_bits: int = 8
    seed: int = 42


@dataclass
class PowerInferConfig:
    """
    PowerInfer C++ backend settings.

    binary_path: explicit path to the PowerInfer 'main' binary.
        Leave None for auto-detection from vendor/PowerInfer/build.
    vram_budget_gb: limit GPU memory usage. PowerInfer will fit as many
        hot neurons on GPU as allowed, spilling the rest to CPU.
        None = use all available VRAM.
    n_gpu_layers: number of transformer layers to fully offload to GPU.
        -1 = auto (as many as VRAM allows). 0 = CPU-only.
    """
    binary_path: Optional[str] = None
    vram_budget_gb: Optional[float] = None
    n_threads: int = 8
    n_gpu_layers: int = -1
    context_size: int = 4096


@dataclass
class EngineConfig:
    """
    Top-level configuration for PowerQuant Engine.

    backend: which inference backend to use.
        "auto"       — .gguf model -> PowerInfer, else HuggingFace
        "hf"         — always use HuggingFace + TurboQuant
        "powerinfer" — always use PowerInfer C++ engine

    load_in_4bit / load_in_8bit: bitsandbytes weight quantization (HF backend only).
        Combine with TurboQuant for maximum memory savings:
          - 7B model FP16:        ~14 GB
          - 7B model 4-bit:       ~3.5 GB
          - 7B model 4-bit + TQ:  ~3.5 GB weights + 5x smaller KV cache

    use_turboquant: enable TurboQuant KV compression (HF backend only).
        Has no effect on the PowerInfer backend (which manages its own KV cache).
    """
    backend: Literal["auto", "hf", "powerinfer"] = "auto"

    # Generation defaults (can be overridden per-call)
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1

    # HF backend: weight quantization
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "auto"
    device_map: str = "auto"

    # TurboQuant KV compression
    use_turboquant: bool = True
    turboquant: TurboQuantConfig = field(default_factory=TurboQuantConfig)

    # PowerInfer backend
    powerinfer: PowerInferConfig = field(default_factory=PowerInferConfig)
