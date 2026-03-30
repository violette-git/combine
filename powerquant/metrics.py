"""
Metrics types returned by PowerQuant inference calls.

Every call to generate_with_metrics() returns a GenerationResult containing:
  - The generated text
  - Timing: latency, tokens/sec, time-to-first-token
  - Token counts: input, output
  - KV cache memory (TurboQuant): compressed size, fp16 equivalent, ratio
  - GPU memory: before/peak/after allocation
  - Configuration snapshot: backend, quantization, TurboQuant settings

Usage:
    result = engine.generate_with_metrics("Tell me about AI.")

    print(result.text)                   # the response
    print(result.tokens_per_second)      # throughput
    print(result.kv_compression_ratio)   # e.g. 4.8 (4.8x less KV memory)
    print(result.summary())             # formatted table
    result.print_report()               # full metrics to stdout
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVMetrics:
    """
    KV cache memory metrics from a TurboQuant-compressed generation run.

    All sizes are in megabytes.
    compression_ratio is how many times smaller the compressed cache is vs fp16.
    For example, ratio=4.8 means the cache uses 4.8x less memory than standard fp16.
    """
    compressed_mb: float = 0.0
    fp16_equiv_mb: float = 0.0
    compression_ratio: float = 1.0
    compressed_tokens: int = 0
    fp16_tokens: int = 0        # tokens kept uncompressed (residual window)
    layers_compressed: int = 0
    key_bits: int = 16
    value_bits: int = 16
    residual_window: int = 0

    @property
    def savings_mb(self) -> float:
        """Memory saved vs a standard fp16 cache."""
        return max(0.0, self.fp16_equiv_mb - self.compressed_mb)

    @property
    def is_active(self) -> bool:
        return self.compression_ratio > 1.01


@dataclass
class GPUMetrics:
    """GPU memory usage sampled around a generation call (in MB)."""
    before_mb: int = 0
    peak_mb: int = 0
    after_mb: int = 0

    @property
    def delta_mb(self) -> int:
        """Additional memory consumed by the generation (after - before)."""
        return max(0, self.after_mb - self.before_mb)


@dataclass
class GenerationResult:
    """
    Full result of a PowerQuant generation call including text and metrics.

    The .text attribute holds the generated string.
    All other fields are performance and memory metrics.

    Typical usage:
        result = engine.generate_with_metrics("Hello!")

        # Access the generated text
        print(result.text)

        # Throughput
        print(f"{result.tokens_per_second:.1f} tok/s")

        # KV compression
        print(f"KV ratio: {result.kv.compression_ratio:.2f}x")
        print(f"KV saved: {result.kv.savings_mb:.1f} MB")

        # Print a formatted report
        result.print_report()
    """

    # ── Output ────────────────────────────────────────────────────────────────
    text: str = ""

    # ── Token counts ─────────────────────────────────────────────────────────
    input_tokens: int = 0
    output_tokens: int = 0

    # ── Timing ───────────────────────────────────────────────────────────────
    latency_s: float = 0.0
    """Total wall-clock time from start of generate() to last token (seconds)."""

    prefill_s: float = 0.0
    """Time to process the input prompt and produce the first token (seconds).
    Only populated when time_first_token is measurable (streaming path)."""

    # ── Throughput ────────────────────────────────────────────────────────────
    tokens_per_second: float = 0.0
    """Output tokens divided by total latency."""

    # ── KV cache ─────────────────────────────────────────────────────────────
    kv: KVMetrics = field(default_factory=KVMetrics)

    # ── GPU memory ────────────────────────────────────────────────────────────
    gpu: GPUMetrics = field(default_factory=GPUMetrics)

    # ── Configuration snapshot ────────────────────────────────────────────────
    backend: str = "unknown"
    model: str = ""
    weight_quantization: str = "fp16"
    """One of: fp16, bf16, int8, int4."""

    # ── Internal ─────────────────────────────────────────────────────────────
    _t_start: float = field(default=0.0, repr=False, compare=False)
    _t_first_token: float = field(default=0.0, repr=False, compare=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        """
        One-line summary string suitable for printing after generation.

        Example:
            [43 tok/s | 128 in, 87 out | KV 4.8x | GPU +12 MB]
        """
        parts = []
        if self.tokens_per_second > 0:
            parts.append(f"{self.tokens_per_second:.1f} tok/s")
        if self.input_tokens or self.output_tokens:
            parts.append(f"{self.input_tokens} in / {self.output_tokens} out")
        if self.latency_s > 0:
            parts.append(f"{self.latency_s:.2f}s")
        if self.kv.is_active:
            parts.append(f"KV {self.kv.compression_ratio:.1f}x compressed")
        if self.gpu.delta_mb > 0:
            parts.append(f"GPU +{self.gpu.delta_mb} MB")
        return "[" + " | ".join(parts) + "]" if parts else ""

    def print_report(self):
        """Print a formatted multi-section metrics report to stdout."""
        w = 52
        print(f"\n{'─'*w}")
        print(f"  PowerQuant Generation Report")
        print(f"{'─'*w}")
        print(f"  Model   : {self.model}")
        print(f"  Backend : {self.backend}")
        print(f"  Weights : {self.weight_quantization}")

        print(f"\n  ── Tokens ──────────────────────────────────────")
        print(f"  Input tokens      : {self.input_tokens:>8,}")
        print(f"  Output tokens     : {self.output_tokens:>8,}")
        print(f"  Total tokens      : {self.total_tokens:>8,}")

        print(f"\n  ── Timing ──────────────────────────────────────")
        print(f"  Total latency     : {self.latency_s:>8.3f} s")
        if self.prefill_s > 0:
            print(f"  Prefill (TTFT)    : {self.prefill_s:>8.3f} s")
            decode_s = max(0.0, self.latency_s - self.prefill_s)
            if self.output_tokens > 1 and decode_s > 0:
                print(f"  Decode time       : {decode_s:>8.3f} s")
        print(f"  Throughput        : {self.tokens_per_second:>8.1f} tok/s")

        if self.kv.is_active:
            print(f"\n  ── KV Cache (TurboQuant) ───────────────────────")
            print(f"  Key bits          : {self.kv.key_bits:>8}")
            print(f"  Value bits        : {self.kv.value_bits:>8}")
            print(f"  Residual window   : {self.kv.residual_window:>8} tokens")
            print(f"  Compressed tokens : {self.kv.compressed_tokens:>8,}")
            print(f"  FP16 window       : {self.kv.fp16_tokens:>8,} tokens")
            print(f"  Compressed size   : {self.kv.compressed_mb:>8.2f} MB")
            print(f"  FP16 equivalent   : {self.kv.fp16_equiv_mb:>8.2f} MB")
            print(f"  Compression ratio : {self.kv.compression_ratio:>8.2f}x")
            print(f"  Memory saved      : {self.kv.savings_mb:>8.2f} MB")
        else:
            print(f"\n  ── KV Cache ────────────────────────────────────")
            print(f"  TurboQuant        : {'disabled':>8}")

        if self.gpu.before_mb > 0 or self.gpu.peak_mb > 0:
            print(f"\n  ── GPU Memory ──────────────────────────────────")
            print(f"  Before generate   : {self.gpu.before_mb:>8,} MB")
            print(f"  Peak during run   : {self.gpu.peak_mb:>8,} MB")
            print(f"  After generate    : {self.gpu.after_mb:>8,} MB")
            print(f"  Delta             : {self.gpu.delta_mb:>+8,} MB")

        print(f"{'─'*w}\n")

    def to_dict(self) -> dict:
        """
        Serialize metrics to a plain dict.

        Suitable for logging, JSON export, or programmatic comparison.
        Does not include the generated text by default — add result.text separately
        if you need it.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 4),
            "prefill_s": round(self.prefill_s, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "kv": {
                "compressed_mb": round(self.kv.compressed_mb, 3),
                "fp16_equiv_mb": round(self.kv.fp16_equiv_mb, 3),
                "compression_ratio": round(self.kv.compression_ratio, 3),
                "compressed_tokens": self.kv.compressed_tokens,
                "fp16_tokens": self.kv.fp16_tokens,
                "layers_compressed": self.kv.layers_compressed,
                "key_bits": self.kv.key_bits,
                "value_bits": self.kv.value_bits,
                "residual_window": self.kv.residual_window,
                "savings_mb": round(self.kv.savings_mb, 3),
            },
            "gpu": {
                "before_mb": self.gpu.before_mb,
                "peak_mb": self.gpu.peak_mb,
                "after_mb": self.gpu.after_mb,
                "delta_mb": self.gpu.delta_mb,
            },
            "backend": self.backend,
            "model": self.model,
            "weight_quantization": self.weight_quantization,
        }


class MetricsTimer:
    """
    Context manager that measures wall-clock time and GPU memory for a block.

    Usage:
        with MetricsTimer() as t:
            result = model.generate(...)
        print(t.elapsed_s, t.gpu_peak_mb)
    """

    def __init__(self, track_gpu: bool = True):
        self.track_gpu = track_gpu
        self.elapsed_s: float = 0.0
        self.gpu_before_mb: int = 0
        self.gpu_peak_mb: int = 0
        self.gpu_after_mb: int = 0
        self._t0: float = 0.0

    def __enter__(self) -> "MetricsTimer":
        try:
            import torch
            if self.track_gpu and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.gpu_before_mb = torch.cuda.memory_allocated() // (1024 * 1024)
        except ImportError:
            pass
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_s = time.perf_counter() - self._t0
        try:
            import torch
            if self.track_gpu and torch.cuda.is_available():
                self.gpu_peak_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
                self.gpu_after_mb = torch.cuda.memory_allocated() // (1024 * 1024)
        except ImportError:
            pass
