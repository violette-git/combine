"""
Core TurboQuant quantizers: rotation-based MSE quantization and QJL inner product estimation.

Two-stage vector quantization with near-optimal distortion:
  Stage 1 (MSE): random rotation + per-coordinate Lloyd-Max quantization
  Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on residuals for unbiased inner products

Source: https://github.com/tonbistudio/turboquant-pytorch
"""

import torch
import math
from .lloyd_max import LloydMaxCodebook


def generate_rotation_matrix(d: int, seed: int = 42, device: str = "cpu") -> torch.Tensor:
    """
    Generate a Haar-distributed random orthogonal matrix via QR decomposition.

    Args:
        d: dimension
        seed: RNG seed for reproducibility
        device: target device

    Returns:
        Pi: (d, d) orthogonal matrix
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    A = torch.randn(d, d, generator=gen, device=device)
    Q, R = torch.linalg.qr(A)
    # Ensure uniform (Haar) distribution by sign-correcting
    signs = torch.diag(R).sign()
    return Q * signs.unsqueeze(0)


def generate_qjl_matrix(d: int, m: int = None, seed: int = 42, device: str = "cpu") -> torch.Tensor:
    """
    Generate a random projection matrix with i.i.d. N(0,1) entries for QJL.

    Args:
        d: input dimension
        m: output dimension (defaults to d)
        seed: RNG seed
        device: target device

    Returns:
        S: (m, d) projection matrix
    """
    if m is None:
        m = d
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(m, d, generator=gen, device=device)


class TurboQuantMSE:
    """
    Stage 1 quantizer: MSE-optimal compression via random rotation + Lloyd-Max.

    Rotates input vectors to approximately normalize coordinate distributions,
    then applies per-coordinate optimal scalar quantization.
    """

    def __init__(self, head_dim: int, bits: int, seed: int = 42, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.codebook = LloydMaxCodebook(head_dim, bits)
        self.centroids = self.codebook.centroids.to(device)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation: y = Pi @ x  (for unit-norm x)."""
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse rotation: x = Pi^T @ y."""
        return y @ self.Pi

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map rotated vectors to codebook indices. Input: (..., D)"""
        rotated = self.rotate(x)
        diffs = rotated.unsqueeze(-1) - self.centroids
        return diffs.abs().argmin(dim=-1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from indices. Output: (..., D)"""
        reconstructed = self.centroids[indices]
        return self.unrotate(reconstructed)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full compress+decompress cycle. Returns (reconstructed, indices)."""
        indices = self.quantize(x)
        reconstructed = self.dequantize(indices)
        return reconstructed, indices


class TurboQuantProd:
    """
    Stage 1+2 quantizer for unbiased inner product estimation.

    Combines (b-1)-bit MSE quantization with 1-bit QJL on residuals.
    Achieves near-unbiased inner product estimates with variance O(1/d).

    Storage per vector: (b-1)*d bits (MSE) + d bits (QJL signs) + 16 bits (residual norm)
    """

    def __init__(self, head_dim: int, bits: int, seed: int = 42, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        # Stage 1: (bits-1)-bit MSE
        self.mse = TurboQuantMSE(head_dim, bits - 1, seed=seed, device=device)
        # Stage 2: QJL projection for residuals
        self.S = generate_qjl_matrix(head_dim, head_dim, seed=seed + 1, device=device)

    def quantize(self, x: torch.Tensor) -> dict:
        """
        Compress x: (..., D) -> dict with mse_indices, qjl_signs, residual_norm.
        Assumes x is unit-norm.
        """
        x_mse, mse_indices = self.mse.forward(x)
        residual = x - x_mse
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)
        residual_unit = residual / (residual_norm + 1e-8)
        qjl_proj = residual_unit @ self.S.T
        qjl_signs = (qjl_proj >= 0).to(torch.bool)
        return {
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.to(torch.float16),
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        """Reconstruct the MSE component only (without QJL correction)."""
        return self.mse.dequantize(compressed["mse_indices"])

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute unbiased inner product estimate <y, x>.

        Formula: <y, x_mse> + ||r|| * sqrt(pi/2) / m * <S@y, signs>
        where r is the residual and signs = sign(S @ r / ||r||).
        """
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        mse_term = (y * x_mse).sum(dim=-1)

        # QJL correction
        s_y = y @ self.S.T  # (..., m)
        signs_float = compressed["qjl_signs"].float() * 2 - 1  # {-1, +1}
        residual_norm = compressed["residual_norm"].float()
        m = self.head_dim
        qjl_term = residual_norm.squeeze(-1) * math.sqrt(math.pi / 2) / m * (s_y * signs_float).sum(dim=-1)
        return mse_term + qjl_term

    def forward(self, x: torch.Tensor) -> dict:
        """Alias for quantize()."""
        return self.quantize(x)


class TurboQuantKVCache:
    """
    Drop-in KV cache replacement using TurboQuant compression.

    Compresses keys with TurboQuantProd (enabling direct inner product computation)
    and values with TurboQuantMSE (simpler MSE reconstruction suffices since
    value weighted sums average out per-vector errors).
    """

    def __init__(self, head_dim: int, bits: int = 4, seed: int = 42, device: str = "cpu"):
        self.key_compressor = TurboQuantProd(head_dim, bits, seed=seed, device=device)
        self.val_compressor = TurboQuantMSE(head_dim, bits - 1, seed=seed + 100, device=device)
        self._compressed_keys = []
        self._compressed_values = []
        self._key_norms = []

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Append new key-value pairs to the cache.
        keys, values: (B, H, S, D) — assumed unit-norm for keys.
        """
        B, H, S, D = keys.shape
        flat_k = keys.reshape(B * H * S, D)
        flat_v = values.reshape(B * H * S, D)

        key_norms = torch.norm(flat_k, dim=-1, keepdim=True)
        flat_k_unit = flat_k / (key_norms + 1e-8)

        self._compressed_keys.append(self.key_compressor.quantize(flat_k_unit))
        self._compressed_values.append(self.val_compressor.quantize(flat_k_unit))  # note: intentional
        self._key_norms.append(key_norms.to(torch.float16))
        # Store value indices
        self._compressed_values[-1] = self.val_compressor.quantize(flat_v / (torch.norm(flat_v, dim=-1, keepdim=True) + 1e-8))

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores via inner product estimation.
        queries: (B, H, 1, D)
        Returns: (B, H, 1, S_total)
        """
        B, H, _, D = queries.shape
        flat_q = queries.reshape(B * H, D)
        scores = []
        for ck in self._compressed_keys:
            s = self.key_compressor.inner_product(flat_q.unsqueeze(1), ck)
            scores.append(s)
        return torch.cat(scores, dim=-1).reshape(B, H, 1, -1)

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values. Returns (B*H, S_total, D)."""
        parts = []
        for cv in self._compressed_values:
            parts.append(self.val_compressor.dequantize(cv["mse_indices"] if isinstance(cv, dict) else cv))
        return torch.cat(parts, dim=0)

    def memory_usage_bits(self) -> dict:
        """Report storage in bits with compression ratio vs FP16."""
        if not self._compressed_keys:
            return {"compressed_bits": 0, "fp16_bits": 0, "ratio": 0.0}
        total_vectors = sum(ck["mse_indices"].numel() // self.key_compressor.mse.head_dim
                            for ck in self._compressed_keys)
        D = self.key_compressor.mse.head_dim
        key_bits_per = (self.key_compressor.bits - 1) * D + D + 16
        val_bits_per = (self.key_compressor.bits - 1) * D + 16
        compressed_bits = total_vectors * (key_bits_per + val_bits_per)
        fp16_bits = total_vectors * D * 16 * 2
        return {
            "compressed_bits": compressed_bits,
            "fp16_bits": fp16_bits,
            "ratio": fp16_bits / compressed_bits if compressed_bits > 0 else 0.0,
        }

    def __len__(self) -> int:
        return sum(ck["mse_indices"].shape[0] for ck in self._compressed_keys) if self._compressed_keys else 0
