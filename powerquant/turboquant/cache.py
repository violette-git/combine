"""
TurboQuantCache: a drop-in DynamicCache replacement that compresses KV tensors
using TurboQuant V3 during generation.

Drop-in usage:
    from powerquant.turboquant.cache import TurboQuantCache

    cache = TurboQuantCache(key_bits=4, value_bits=2, residual_window=128)
    outputs = model.generate(..., past_key_values=cache, use_cache=True)
"""

import torch
from transformers import DynamicCache
from .compressors_v3 import TurboQuantV3


class TurboQuantCache(DynamicCache):
    """
    DynamicCache subclass that transparently compresses KV tensors with TurboQuant V3.

    Strategy:
      - New tokens are added to a per-layer fp16 buffer.
      - When the buffer exceeds `residual_window` tokens, the overflow is compressed
        and stored as quantized chunks.
      - On each update(), all chunks are decompressed and concatenated with the fp16
        buffer before returning to the attention mechanism.

    This means the attention layer always sees full-precision (decompressed) tensors,
    while the actual storage is compressed. The overhead is the decompression cost
    per forward pass, which is fast (matmul + lookup).
    """

    def __init__(
        self,
        key_bits: int = 4,
        value_bits: int = 2,
        residual_window: int = 128,
        protected_layers: int = 4,
        n_layers: int = 36,
        seed: int = 42,
    ):
        super().__init__()
        # Newer transformers versions may not initialize these in __init__
        if not hasattr(self, "key_cache"):
            self.key_cache = []
        if not hasattr(self, "value_cache"):
            self.value_cache = []
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window
        self.protected_layers = protected_layers
        self.n_layers = n_layers
        self.seed = seed

        self._compressors: dict[int, TurboQuantV3] = {}
        self._chunks_k: dict[int, list] = {}
        self._chunks_v: dict[int, list] = {}
        self._fp16_recent_k: dict[int, list] = {}
        self._fp16_recent_v: dict[int, list] = {}
        self._total_seq: dict[int, int] = {}

    def _get_compressor(self, layer_idx: int, head_dim: int, device) -> TurboQuantV3:
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                residual_window=0,  # windowing is managed here, not inside V3
                layer_idx=layer_idx,
                n_layers=self.n_layers,
                protected_layers=self.protected_layers,
                seed=self.seed,
                device=str(device),
            )
        return self._compressors[layer_idx]

    def _init_layer(self, layer_idx: int):
        self._chunks_k[layer_idx] = []
        self._chunks_v[layer_idx] = []
        self._fp16_recent_k[layer_idx] = []
        self._fp16_recent_v[layer_idx] = []
        self._total_seq[layer_idx] = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Called by each transformer layer during the forward pass.

        Appends new key/value states, compresses overflow beyond residual_window,
        and returns the full decompressed K/V sequence for attention.
        """
        B, H, S_new, D = key_states.shape
        device = key_states.device

        if layer_idx not in self._chunks_k:
            self._init_layer(layer_idx)

        comp = self._get_compressor(layer_idx, D, device)
        self._total_seq[layer_idx] += S_new

        # Accumulate new tokens in fp16 buffer
        self._fp16_recent_k[layer_idx].append(key_states)
        self._fp16_recent_v[layer_idx].append(value_states)

        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        rw = self.residual_window

        # Compress overflow beyond residual window
        if rw > 0 and recent_k.shape[2] > rw:
            overflow = recent_k.shape[2] - rw
            to_compress_k = recent_k[:, :, :overflow, :]
            to_compress_v = recent_v[:, :, :overflow, :]

            ck, cv = comp.compress_kv(to_compress_k, to_compress_v)
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)

            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._fp16_recent_k[layer_idx] = [recent_k]
            self._fp16_recent_v[layer_idx] = [recent_v]

        # Decompress all historical chunks
        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []
        for ck, cv in zip(self._chunks_k[layer_idx], self._chunks_v[layer_idx]):
            dk, dv = comp.decompress_kv(ck, cv)
            parts_k.append(dk.to(key_states.dtype))
            parts_v.append(dv.to(value_states.dtype))

        # Concatenate decompressed history + fp16 recent window
        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        parts_k.append(recent_k)
        parts_v.append(recent_v)

        full_k = torch.cat(parts_k, dim=2)
        full_v = torch.cat(parts_v, dim=2)

        # Keep DynamicCache internal lists in sync so transformers internals work
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.zeros(0, device=device))
            self.value_cache.append(torch.zeros(0, device=device))
        self.key_cache[layer_idx] = full_k
        self.value_cache[layer_idx] = full_v

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seq.get(layer_idx, 0)

    def memory_report(self) -> dict:
        """
        Return a summary of compressed vs fp16 memory across all layers.
        """
        if not self._chunks_k:
            return {"status": "no compressed chunks yet"}

        total_compressed = 0
        total_fp16_equiv = 0
        for layer_idx, chunks_k in self._chunks_k.items():
            chunks_v = self._chunks_v[layer_idx]
            for ck, cv in zip(chunks_k, chunks_v):
                B, H, S, D = ck["shape"]
                comp = self._compressors.get(layer_idx)
                if comp:
                    info = comp.memory_bytes(B, H, S)
                    total_compressed += info["compressed_bytes"]
                    total_fp16_equiv += info["fp16_bytes"]

        ratio = total_fp16_equiv / total_compressed if total_compressed > 0 else 0
        return {
            "compressed_mb": total_compressed / 1e6,
            "fp16_equiv_mb": total_fp16_equiv / 1e6,
            "compression_ratio": round(ratio, 2),
            "layers_compressed": len([k for k in self._chunks_k if self._chunks_k[k]]),
        }
