"""
HuggingFace backend: standard transformers inference with optional TurboQuant
KV cache compression.

This backend loads any HuggingFace causal LM and optionally replaces the KV
cache with TurboQuantCache to compress long contexts, reducing GPU memory usage
by ~5x at minimal quality cost.

Combining weight quantization (bitsandbytes 4-bit) with KV cache compression
(TurboQuant) allows running significantly larger models on consumer hardware:
    - 7B model in 4-bit: ~3.5 GB weights + ~0.5 GB/1K tokens KV (fp16)
    - 7B model in 4-bit + TurboQuant K4/V2: ~3.5 GB weights + ~0.1 GB/1K tokens KV
"""

import gc
import time
import torch
from typing import Optional, Iterator
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

from ..turboquant.cache import TurboQuantCache
from ..metrics import GenerationResult, KVMetrics, GPUMetrics, MetricsTimer


@dataclass
class HFGenConfig:
    """Generation parameters for the HF backend."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


class HFBackend:
    """
    HuggingFace causal LM inference with optional TurboQuant KV cache compression.

    Supports any model on HuggingFace Hub or local directory.
    Optionally applies:
      - 4-bit / 8-bit weight quantization via bitsandbytes
      - TurboQuant V3 KV cache compression for long-context efficiency

    TurboQuant integration:
        When ``use_turboquant=True``, a :class:`TurboQuantCache` is passed as
        ``past_key_values`` to ``model.generate()``. The cache transparently
        compresses tokens beyond ``residual_window`` while keeping recent tokens
        in fp16 for quality.

    Metrics:
        Call ``generate_with_metrics()`` to get a
        :class:`~powerquant.metrics.GenerationResult` containing timing,
        throughput, KV memory, and GPU memory data alongside the generated text.
    """

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        use_turboquant: bool = True,
        key_bits: int = 4,
        value_bits: int = 2,
        residual_window: int = 128,
        protected_layers: int = 4,
    ):
        """
        Args:
            model_name: HuggingFace model ID or local path.
            load_in_4bit: enable bitsandbytes NF4 weight quantization (~4x smaller).
            load_in_8bit: enable bitsandbytes 8-bit weight quantization (~2x smaller).
            torch_dtype: weight dtype (``"auto"``, ``"float16"``, ``"bfloat16"``).
            device_map: device placement (``"auto"``, ``"cuda:0"``, ``"cpu"``).
            use_turboquant: compress KV cache with TurboQuant V3.
            key_bits: bit-width for key vectors (default 4).
            value_bits: bit-width for value vectors (default 2).
            residual_window: recent tokens kept uncompressed in fp16 (default 128).
            protected_layers: first/last N layers kept at full precision (default 4).
        """
        self.model_name = model_name
        self.use_turboquant = use_turboquant
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window
        self.protected_layers = protected_layers

        self.model = None
        self.tokenizer = None
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self._torch_dtype = torch_dtype
        self._device_map = device_map

    def load(self) -> "HFBackend":
        """Load model and tokenizer into memory. Returns self for chaining."""
        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {"device_map": self._device_map}

        if self._load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["dtype"] = torch.float16
        elif self._load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self._torch_dtype != "auto":
            model_kwargs["dtype"] = getattr(torch, self._torch_dtype)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        except ValueError as e:
            if "dispatched on the CPU" in str(e) and (self._load_in_4bit or self._load_in_8bit):
                # Model too large for GPU alone — enable CPU offloading and retry.
                print(
                    "  Model doesn't fit in GPU VRAM alone. "
                    "Enabling CPU offload (slower but will work)..."
                )
                model_kwargs["device_map"] = "auto"
                if self._load_in_4bit:
                    # For 4-bit, just use device_map=auto — do NOT set
                    # llm_int8_enable_fp32_cpu_offload (that flag is int8-only).
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                elif self._load_in_8bit:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            else:
                raise
        self.model.eval()

        n_layers = getattr(self.model.config, "num_hidden_layers", 32)
        quant = "int4" if self._load_in_4bit else "int8" if self._load_in_8bit else "fp16"
        print(
            f"Loaded {self.model_name} | layers={n_layers} | "
            f"weights={quant} | TurboQuant={self.use_turboquant}"
        )
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() // 1024 // 1024
            print(f"GPU memory after load: {mem_mb} MB")

        return self

    def _make_cache(self) -> Optional[TurboQuantCache]:
        """Build a fresh TurboQuantCache for a generation run."""
        if not self.use_turboquant:
            return None
        n_layers = getattr(self.model.config, "num_hidden_layers", 32)
        return TurboQuantCache(
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            residual_window=self.residual_window,
            protected_layers=self.protected_layers,
            n_layers=n_layers,
        )

    def _encode(self, prompt: str, max_length: int = 32768) -> dict:
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.model.device)

    def _weight_quant_label(self) -> str:
        if self._load_in_4bit:
            return "int4"
        if self._load_in_8bit:
            return "int8"
        return "fp16"

    def generate(
        self,
        prompt: str,
        config: Optional[HFGenConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text for the given prompt.

        Applies the model's chat template if the tokenizer supports it.
        Returns only the newly generated text (not the prompt).

        For full metrics, use :meth:`generate_with_metrics` instead.
        """
        return self.generate_with_metrics(prompt, config, system_prompt).text

    def generate_with_metrics(
        self,
        prompt: str,
        config: Optional[HFGenConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate text and return a :class:`~powerquant.metrics.GenerationResult`
        containing the text plus detailed performance metrics.

        Metrics collected:
          - ``input_tokens`` / ``output_tokens``: token counts
          - ``latency_s``: total wall-clock time in seconds
          - ``tokens_per_second``: output throughput
          - ``kv.*``: TurboQuant compression ratio, sizes, token counts
          - ``gpu.*``: GPU memory before/peak/after in MB

        Example::

            result = backend.generate_with_metrics("Hello!")
            print(result.text)
            print(f"{result.tokens_per_second:.1f} tok/s")
            result.print_report()
        """
        if self.model is None:
            self.load()

        cfg = config or HFGenConfig()
        formatted = self._format_prompt(prompt, system_prompt)
        inputs = self._encode(formatted)
        input_token_count = inputs["input_ids"].shape[1]
        cache = self._make_cache()

        with MetricsTimer() as timer:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature if cfg.do_sample else 1.0,
                    top_p=cfg.top_p if cfg.do_sample else 1.0,
                    top_k=cfg.top_k if cfg.do_sample else 0,
                    do_sample=cfg.do_sample,
                    repetition_penalty=cfg.repetition_penalty,
                    past_key_values=cache,
                    use_cache=True,
                )

        new_tokens = outputs[0][input_token_count:]
        output_token_count = new_tokens.shape[0]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        tps = output_token_count / timer.elapsed_s if timer.elapsed_s > 0 else 0.0

        kv = self._build_kv_metrics(cache)
        gpu = GPUMetrics(
            before_mb=timer.gpu_before_mb,
            peak_mb=timer.gpu_peak_mb,
            after_mb=timer.gpu_after_mb,
        )

        return GenerationResult(
            text=text,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            latency_s=timer.elapsed_s,
            tokens_per_second=tps,
            kv=kv,
            gpu=gpu,
            backend="hf",
            model=self.model_name,
            weight_quantization=self._weight_quant_label(),
        )

    def generate_stream(
        self,
        prompt: str,
        config: Optional[HFGenConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream generated tokens as they are produced.

        Uses :class:`~transformers.TextIteratorStreamer` for real-time output.
        Yields decoded token strings one at a time.

        .. note::
            For streamed output with metrics, use
            :meth:`generate_stream_with_metrics`.
        """
        for text, _ in self.generate_stream_with_metrics(prompt, config, system_prompt):
            yield text

    def generate_stream_with_metrics(
        self,
        prompt: str,
        config: Optional[HFGenConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[tuple[str, Optional[GenerationResult]]]:
        """
        Stream generated tokens and yield a final :class:`~powerquant.metrics.GenerationResult`.

        Yields ``(token_str, None)`` for each token during generation, then
        yields ``("", result)`` as the final item once generation completes.
        The caller can detect the end by checking ``result is not None``.

        Example::

            for token, result in backend.generate_stream_with_metrics("Hello!"):
                if result is not None:
                    result.print_report()
                else:
                    print(token, end="", flush=True)
        """
        if self.model is None:
            self.load()

        cfg = config or HFGenConfig()
        formatted = self._format_prompt(prompt, system_prompt)
        inputs = self._encode(formatted)
        input_token_count = inputs["input_ids"].shape[1]
        cache = self._make_cache()

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = {
            **inputs,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature if cfg.do_sample else 1.0,
            "top_p": cfg.top_p if cfg.do_sample else 1.0,
            "top_k": cfg.top_k if cfg.do_sample else 0,
            "do_sample": cfg.do_sample,
            "repetition_penalty": cfg.repetition_penalty,
            "past_key_values": cache,
            "use_cache": True,
            "streamer": streamer,
        }

        t_start = time.perf_counter()
        gpu_before = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_before = torch.cuda.memory_allocated() // (1024 * 1024)

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        parts: list[str] = []
        t_first = 0.0
        for text in streamer:
            if not t_first and text:
                t_first = time.perf_counter()
            yield text, None
            parts.append(text)

        thread.join()
        t_end = time.perf_counter()

        full_text = "".join(parts).strip()
        output_token_count = len(self.tokenizer.encode(full_text))
        latency = t_end - t_start
        tps = output_token_count / latency if latency > 0 else 0.0
        prefill = (t_first - t_start) if t_first else 0.0

        gpu_peak = gpu_after = 0
        if torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
            gpu_after = torch.cuda.memory_allocated() // (1024 * 1024)

        result = GenerationResult(
            text=full_text,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            latency_s=latency,
            prefill_s=prefill,
            tokens_per_second=tps,
            kv=self._build_kv_metrics(cache),
            gpu=GPUMetrics(before_mb=gpu_before, peak_mb=gpu_peak, after_mb=gpu_after),
            backend="hf",
            model=self.model_name,
            weight_quantization=self._weight_quant_label(),
        )
        yield "", result

    def _build_kv_metrics(self, cache: Optional[TurboQuantCache]) -> KVMetrics:
        """Extract KV memory metrics from a completed TurboQuantCache."""
        if cache is None:
            return KVMetrics(key_bits=16, value_bits=16)

        report = cache.memory_report()
        if "status" in report:
            # No compressed chunks yet (very short prompt, all in residual window)
            return KVMetrics(
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                residual_window=self.residual_window,
            )

        return KVMetrics(
            compressed_mb=report.get("compressed_mb", 0.0),
            fp16_equiv_mb=report.get("fp16_equiv_mb", 0.0),
            compression_ratio=report.get("compression_ratio", 1.0),
            layers_compressed=report.get("layers_compressed", 0),
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            residual_window=self.residual_window,
        )

    def _format_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Apply chat template if available, else return plain prompt."""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        if self.tokenizer.chat_template is None:
            return prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt

    def unload(self):
        """Release model and tokenizer from memory."""
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
