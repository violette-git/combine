"""
TurboQuant: KV cache compression via vector quantization.

Core algorithm: random rotation to normalize coordinate distributions,
followed by Lloyd-Max optimal scalar quantization. V3 improves on earlier
versions by using MSE-only compression (removing QJL) and asymmetric
bit allocation for keys vs values.

Quick start:
    from powerquant.turboquant import TurboQuantCache

    cache = TurboQuantCache(key_bits=4, value_bits=2, residual_window=128)
    outputs = model.generate(..., past_key_values=cache, use_cache=True)
"""

from .compressors_v3 import TurboQuantV3, MSECompressor
from .turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max
from .cache import TurboQuantCache

__all__ = [
    "TurboQuantCache",
    "TurboQuantV3",
    "MSECompressor",
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantKVCache",
    "LloydMaxCodebook",
    "solve_lloyd_max",
]
