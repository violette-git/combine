"""
PowerQuant Engine: unified interface for efficient LLM inference.

Combines two orthogonal optimizations:

  1. PowerInfer sparse-activation inference (for GGUF models)
       - Hot neurons cached on GPU, cold neurons computed on CPU
       - Up to 11x faster than llama.cpp on consumer GPUs
       - Requires PowerInfer GGUF format (.gguf with activation statistics)

  2. TurboQuant KV cache compression (for HuggingFace models)
       - Compresses key-value cache by ~5x at 3-bit average precision
       - Enables much longer contexts on memory-constrained hardware
       - Works with any HuggingFace causal LM

Backend selection:
  - Model path ends in .gguf  →  PowerInfer backend
  - Model is a HF model ID    →  HF + TurboQuant backend
  - Override with config.backend = "hf" or "powerinfer"

Quick start:
    from powerquant import Engine

    # HuggingFace model with TurboQuant KV compression
    engine = Engine.from_pretrained("Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True)
    print(engine.generate("Explain transformers in one paragraph."))

    # PowerInfer GGUF model
    engine = Engine.from_pretrained("/path/to/model.gguf")
    print(engine.generate("Explain transformers in one paragraph."))
"""

import os
from pathlib import Path
from typing import Optional, Iterator

from .config import EngineConfig, TurboQuantConfig, PowerInferConfig
from .backends.hf import HFBackend, HFGenConfig
from .backends.powerinfer import PowerInferBackend, PowerInferGenConfig


class Engine:
    """
    Unified inference engine supporting both PowerInfer and HuggingFace backends.

    The engine auto-selects the backend based on the model format and
    available hardware, but can be forced to a specific backend via config.
    """

    def __init__(self, model: str, config: Optional[EngineConfig] = None):
        """
        Args:
            model: HuggingFace model ID, local model directory, or path to .gguf file.
            config: engine configuration. Defaults to EngineConfig() if None.
        """
        self.model = model
        self.config = config or EngineConfig()
        self._backend = None
        self._selected_backend: str = self._select_backend()

    def _select_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend
        # GGUF file -> PowerInfer; everything else -> HF
        if self.model.endswith(".gguf") or Path(self.model).suffix == ".gguf":
            return "powerinfer"
        return "hf"

    def _ensure_loaded(self):
        if self._backend is not None:
            return
        if self._selected_backend == "powerinfer":
            self._backend = PowerInferBackend(
                binary_path=self.config.powerinfer.binary_path
            )
        else:
            tq = self.config.turboquant
            self._backend = HFBackend(
                model_name=self.model,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
                use_turboquant=self.config.use_turboquant,
                key_bits=tq.key_bits,
                value_bits=tq.value_bits,
                residual_window=tq.residual_window,
                protected_layers=tq.protected_layers,
            )
            self._backend.load()

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text for the given prompt.

        Args:
            prompt: user input text.
            max_new_tokens: override config default.
            temperature: override config default.
            top_p: override config default.
            do_sample: override config default.
            system_prompt: optional system message (HF backend, if chat template available).

        Returns:
            Generated response string.
        """
        self._ensure_loaded()
        cfg = self.config

        if self._selected_backend == "powerinfer":
            pi_cfg = PowerInferGenConfig(
                max_new_tokens=max_new_tokens or cfg.max_new_tokens,
                temperature=temperature or cfg.temperature,
                top_p=top_p or cfg.top_p,
                n_threads=cfg.powerinfer.n_threads,
                vram_budget_gb=cfg.powerinfer.vram_budget_gb,
                n_gpu_layers=cfg.powerinfer.n_gpu_layers,
                context_size=cfg.powerinfer.context_size,
            )
            return self._backend.generate(self.model, prompt, pi_cfg)
        else:
            hf_cfg = HFGenConfig(
                max_new_tokens=max_new_tokens or cfg.max_new_tokens,
                temperature=temperature or cfg.temperature,
                top_p=top_p or cfg.top_p,
                top_k=cfg.top_k,
                do_sample=do_sample if do_sample is not None else cfg.do_sample,
                repetition_penalty=cfg.repetition_penalty,
            )
            return self._backend.generate(prompt, hf_cfg, system_prompt=system_prompt)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream generated tokens as they are produced.

        Yields decoded token strings one at a time. Use this for interactive
        chat interfaces where you want to display output progressively.
        """
        self._ensure_loaded()
        cfg = self.config

        if self._selected_backend == "powerinfer":
            pi_cfg = PowerInferGenConfig(
                max_new_tokens=max_new_tokens or cfg.max_new_tokens,
                temperature=temperature or cfg.temperature,
                n_threads=cfg.powerinfer.n_threads,
                vram_budget_gb=cfg.powerinfer.vram_budget_gb,
            )
            yield from self._backend.generate_stream(self.model, prompt, pi_cfg)
        else:
            hf_cfg = HFGenConfig(
                max_new_tokens=max_new_tokens or cfg.max_new_tokens,
                temperature=temperature or cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
            )
            yield from self._backend.generate_stream(prompt, hf_cfg, system_prompt=system_prompt)

    def chat(self, system_prompt: Optional[str] = None):
        """
        Start an interactive chat session in the terminal.

        Streams responses token-by-token. Type 'exit' or 'quit' to stop,
        'clear' to reset conversation history.
        """
        self._ensure_loaded()
        print(f"\nPowerQuant Chat [{self._selected_backend} backend]")
        print(f"Model: {self.model}")
        if self.config.use_turboquant and self._selected_backend == "hf":
            tq = self.config.turboquant
            print(f"TurboQuant: K{tq.key_bits}/V{tq.value_bits}, window={tq.residual_window}")
        print("Type 'exit' to quit, 'clear' to reset history.\n")

        history: list[dict] = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                history = [h for h in history if h["role"] == "system"]
                print("History cleared.\n")
                continue

            print("Assistant: ", end="", flush=True)
            response_parts = []
            for token in self.generate_stream(user_input, system_prompt=system_prompt):
                print(token, end="", flush=True)
                response_parts.append(token)
            print()

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": "".join(response_parts)})

    @property
    def backend_name(self) -> str:
        return self._selected_backend

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def unload(self):
        """Release model memory."""
        if self._backend is not None and hasattr(self._backend, "unload"):
            self._backend.unload()
        self._backend = None

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_turboquant: bool = True,
        key_bits: int = 4,
        value_bits: int = 2,
        residual_window: int = 128,
        **kwargs,
    ) -> "Engine":
        """
        Convenience constructor with the most common options as keyword args.

        Examples:
            # 4-bit quantization + TurboQuant (recommended for consumer GPUs)
            engine = Engine.from_pretrained("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)

            # Full precision HF model
            engine = Engine.from_pretrained("gpt2")

            # PowerInfer GGUF model
            engine = Engine.from_pretrained("/models/llama-2-7b.gguf")

            # High-quality TurboQuant (less compression, better accuracy)
            engine = Engine.from_pretrained("Qwen/Qwen2.5-3B", key_bits=4, value_bits=4)
        """
        config = EngineConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            use_turboquant=use_turboquant,
            turboquant=TurboQuantConfig(
                key_bits=key_bits,
                value_bits=value_bits,
                residual_window=residual_window,
            ),
            **kwargs,
        )
        return cls(model, config)

    @classmethod
    def powerinfer(cls, gguf_path: str, **kwargs) -> "Engine":
        """
        Create an engine using the PowerInfer backend.

        Args:
            gguf_path: path to a PowerInfer GGUF model file.

        Example:
            engine = Engine.powerinfer("/models/falcon-40b.gguf", vram_budget_gb=8)
        """
        pi_config = PowerInferConfig(**{
            k: v for k, v in kwargs.items()
            if k in PowerInferConfig.__dataclass_fields__
        })
        config = EngineConfig(backend="powerinfer", powerinfer=pi_config)
        return cls(gguf_path, config)
