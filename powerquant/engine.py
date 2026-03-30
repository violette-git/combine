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

Quick start::

    from powerquant import Engine

    # HuggingFace model with TurboQuant KV compression + 4-bit weights
    engine = Engine.from_pretrained("Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True)
    print(engine.generate("Explain transformers in one paragraph."))

    # Same call but with full metrics
    result = engine.generate_with_metrics("Explain transformers in one paragraph.")
    print(result.text)
    result.print_report()

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
from .metrics import GenerationResult


class Engine:
    """
    Unified inference engine supporting both PowerInfer and HuggingFace backends.

    The engine auto-selects the backend based on the model format and available
    hardware, but can be forced via ``config.backend``.

    **Backend auto-selection rules:**

    * Path ending in ``.gguf`` → PowerInfer C++ engine
    * HuggingFace model ID or local directory → HF + TurboQuant backend

    **Methods:**

    * :meth:`generate` — return generated text as a string
    * :meth:`generate_with_metrics` — return :class:`~powerquant.metrics.GenerationResult`
      with text + timing + KV memory + GPU memory metrics
    * :meth:`generate_stream` — yield tokens as they are produced
    * :meth:`chat` — interactive terminal session
    """

    def __init__(self, model: str, config: Optional[EngineConfig] = None):
        """
        Args:
            model: HuggingFace model ID, local model directory, or path to a
                   ``.gguf`` file. The file extension determines the backend.
            config: engine configuration. ``EngineConfig()`` defaults are used
                    if not provided.
        """
        self.model = model
        self.config = config or EngineConfig()
        self._backend = None
        self._selected_backend: str = self._select_backend()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend
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

    def _hf_gen_config(
        self,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        do_sample: Optional[bool],
    ) -> HFGenConfig:
        cfg = self.config
        return HFGenConfig(
            max_new_tokens=max_new_tokens or cfg.max_new_tokens,
            temperature=temperature or cfg.temperature,
            top_p=top_p or cfg.top_p,
            top_k=cfg.top_k,
            do_sample=do_sample if do_sample is not None else cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
        )

    def _pi_gen_config(
        self,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
    ) -> PowerInferGenConfig:
        cfg = self.config
        return PowerInferGenConfig(
            max_new_tokens=max_new_tokens or cfg.max_new_tokens,
            temperature=temperature or cfg.temperature,
            top_p=cfg.top_p,
            n_threads=cfg.powerinfer.n_threads,
            vram_budget_gb=cfg.powerinfer.vram_budget_gb,
            n_gpu_layers=cfg.powerinfer.n_gpu_layers,
            context_size=cfg.powerinfer.context_size,
        )

    # ── Public API ────────────────────────────────────────────────────────────

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
        Generate a text response for the given prompt.

        Args:
            prompt: user input text.
            max_new_tokens: maximum tokens to generate. Overrides config default.
            temperature: sampling temperature. Overrides config default.
            top_p: nucleus sampling probability. Overrides config default.
            do_sample: enable stochastic sampling. Overrides config default.
            system_prompt: prepend a system message (HF backend with chat template).

        Returns:
            The generated response as a plain string (prompt not included).

        .. tip::
            Use :meth:`generate_with_metrics` to get the same result plus
            detailed performance data.
        """
        return self.generate_with_metrics(
            prompt, max_new_tokens, temperature, top_p, do_sample, system_prompt
        ).text

    def generate_with_metrics(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate a response and return a rich metrics object.

        Returns a :class:`~powerquant.metrics.GenerationResult` that contains:

        * ``result.text`` — the generated string
        * ``result.tokens_per_second`` — output throughput
        * ``result.latency_s`` — wall-clock seconds for the full call
        * ``result.input_tokens`` / ``result.output_tokens`` — token counts
        * ``result.kv.compression_ratio`` — TurboQuant KV compression (HF only)
        * ``result.kv.compressed_mb`` / ``result.kv.fp16_equiv_mb`` — memory sizes
        * ``result.kv.savings_mb`` — memory saved vs a standard fp16 cache
        * ``result.gpu.before_mb`` / ``result.gpu.peak_mb`` — GPU memory
        * ``result.summary()`` — one-line summary string
        * ``result.print_report()`` — formatted multi-section report to stdout
        * ``result.to_dict()`` — all metrics as a plain dict

        Example::

            result = engine.generate_with_metrics(
                "Summarize the history of AI in three sentences."
            )
            print(result.text)
            print(result.summary())
            # → [43.2 tok/s | 12 in / 87 out | KV 4.8x compressed | GPU +8 MB]

            result.print_report()
            # → formatted table with all metrics sections

            metrics = result.to_dict()  # for logging/JSON export
        """
        self._ensure_loaded()

        if self._selected_backend == "powerinfer":
            return self._backend.generate_with_metrics(
                self.model, prompt, self._pi_gen_config(max_new_tokens, temperature)
            )
        else:
            return self._backend.generate_with_metrics(
                prompt,
                self._hf_gen_config(max_new_tokens, temperature, top_p, do_sample),
                system_prompt=system_prompt,
            )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Yield generated tokens one at a time as they are produced.

        Use this for interactive interfaces where you want to display output
        progressively (e.g., a chat UI or terminal that streams responses).

        .. note::
            To also capture metrics from a streaming call, use the lower-level
            ``backend.generate_stream_with_metrics()`` directly (HF backend only).
        """
        self._ensure_loaded()

        if self._selected_backend == "powerinfer":
            yield from self._backend.generate_stream(
                self.model, prompt, self._pi_gen_config(max_new_tokens, temperature)
            )
        else:
            yield from self._backend.generate_stream(
                prompt,
                self._hf_gen_config(max_new_tokens, temperature, None, None),
                system_prompt=system_prompt,
            )

    def chat(
        self,
        system_prompt: Optional[str] = None,
        show_metrics: bool = False,
    ):
        """
        Start an interactive chat session in the terminal.

        Responses are streamed token-by-token for a responsive feel.

        Commands during chat:
          * ``exit`` / ``quit`` — end the session
          * ``clear`` — reset conversation history
          * ``metrics`` — toggle per-turn metric summary lines (or pass
            ``show_metrics=True`` when calling this method)

        Args:
            system_prompt: optional system message prepended to every turn.
            show_metrics: print a one-line metric summary after each response.
        """
        self._ensure_loaded()
        print(f"\nPowerQuant Chat  [{self._selected_backend} backend]")
        print(f"Model : {self.model}")
        if self.config.use_turboquant and self._selected_backend == "hf":
            tq = self.config.turboquant
            print(
                f"KV    : TurboQuant K{tq.key_bits}/V{tq.value_bits}"
                f", residual window={tq.residual_window} tokens"
            )
        print("Type 'exit' to quit, 'clear' to reset, 'metrics' to toggle stats.\n")

        _show = show_metrics
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
            if user_input.lower() == "metrics":
                _show = not _show
                print(f"Metrics {'on' if _show else 'off'}.\n")
                continue

            print("Assistant: ", end="", flush=True)
            response_parts: list[str] = []

            if self._selected_backend == "hf" and hasattr(self._backend, "generate_stream_with_metrics"):
                result = None
                for token, maybe_result in self._backend.generate_stream_with_metrics(
                    user_input, system_prompt=system_prompt
                ):
                    if maybe_result is not None:
                        result = maybe_result
                    else:
                        print(token, end="", flush=True)
                        response_parts.append(token)
                print()
                if _show and result:
                    print(f"  {result.summary()}")
            else:
                for token in self.generate_stream(user_input, system_prompt=system_prompt):
                    print(token, end="", flush=True)
                    response_parts.append(token)
                print()

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": "".join(response_parts)})

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        """Name of the selected backend: ``"hf"`` or ``"powerinfer"``."""
        return self._selected_backend

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded into memory."""
        return self._backend is not None

    def unload(self):
        """Release model weights from memory (GPU and CPU)."""
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

        Args:
            model: HuggingFace model ID, local path, or ``.gguf`` file.
            load_in_4bit: 4-bit NF4 weight quantization (bitsandbytes, HF only).
            load_in_8bit: 8-bit weight quantization (bitsandbytes, HF only).
            use_turboquant: enable TurboQuant KV cache compression (HF only).
            key_bits: TurboQuant key bit-width (default 4).
            value_bits: TurboQuant value bit-width (default 2).
            residual_window: tokens kept uncompressed at end of sequence (default 128).
            **kwargs: additional :class:`~powerquant.config.EngineConfig` fields.

        Examples::

            # 4-bit weights + TurboQuant (recommended for consumer GPUs)
            engine = Engine.from_pretrained("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)

            # Full-precision HuggingFace model
            engine = Engine.from_pretrained("gpt2")

            # PowerInfer GGUF model (auto-detected from extension)
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
        Create an engine explicitly using the PowerInfer C++ backend.

        Args:
            gguf_path: path to a PowerInfer GGUF model file.
            **kwargs: :class:`~powerquant.config.PowerInferConfig` fields such as
                      ``vram_budget_gb``, ``n_threads``, ``n_gpu_layers``.

        Example::

            engine = Engine.powerinfer("/models/falcon-40b.gguf", vram_budget_gb=8)
            result = engine.generate_with_metrics("What is a transformer?")
            result.print_report()
        """
        pi_config = PowerInferConfig(**{
            k: v for k, v in kwargs.items()
            if k in PowerInferConfig.__dataclass_fields__
        })
        config = EngineConfig(backend="powerinfer", powerinfer=pi_config)
        return cls(gguf_path, config)
