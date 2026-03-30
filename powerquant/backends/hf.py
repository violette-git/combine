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
        When `use_turboquant=True`, a TurboQuantCache is passed as `past_key_values`
        to `model.generate()`. The cache transparently compresses tokens beyond
        `residual_window` while keeping recent tokens in fp16 for quality.
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
            torch_dtype: weight dtype ("auto", "float16", "bfloat16").
            device_map: device placement ("auto", "cuda:0", "cpu").
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
        """Load model and tokenizer. Returns self for chaining."""
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
            model_kwargs["torch_dtype"] = torch.float16
        elif self._load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self._torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, self._torch_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

        n_layers = getattr(self.model.config, "num_hidden_layers", 32)
        print(
            f"Loaded {self.model_name} | layers={n_layers} | "
            f"4bit={self._load_in_4bit} | TurboQuant={self.use_turboquant}"
        )
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() // 1024 // 1024
            print(f"GPU memory: {mem_mb} MB")

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

    def generate(
        self,
        prompt: str,
        config: Optional[HFGenConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text for the given prompt.

        Applies chat template if the tokenizer supports it.
        Returns only the newly generated text (not the prompt).
        """
        if self.model is None:
            self.load()

        cfg = config or HFGenConfig()
        formatted = self._format_prompt(prompt, system_prompt)
        inputs = self._encode(formatted)
        cache = self._make_cache()

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

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_stream(
        self,
        prompt: str,
        config: Optional[HFGenConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream generated tokens as they are produced.

        Uses transformers TextIteratorStreamer for real-time output.
        Yields decoded token strings.
        """
        if self.model is None:
            self.load()

        cfg = config or HFGenConfig()
        formatted = self._format_prompt(prompt, system_prompt)
        inputs = self._encode(formatted)
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

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for text in streamer:
            yield text
        thread.join()

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

    def kv_memory_report(self, last_cache: Optional[TurboQuantCache] = None) -> dict:
        """
        Return KV cache memory statistics from the last generation run.
        Pass the cache object if you captured it; otherwise returns empty report.
        """
        if last_cache is None:
            return {"note": "Pass cache object to get memory report"}
        return last_cache.memory_report()

    def unload(self):
        """Release model from memory."""
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
