import os
import sys

libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".python_libs"))
if os.path.exists(libs_dir) and libs_dir not in sys.path:
    sys.path.insert(0, libs_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for ternary quantization {-1, 0, 1}.
    Clamps scaled values to {-1, 0, 1} on forward pass, passes gradients directly on backward.
    """
    @staticmethod
    def forward(ctx, x):
        scale = x.abs().mean().clamp(min=1e-5)
        x_scaled = x / scale
        x_quantized = torch.round(x_scaled).clamp(-1, 1)
        x_dequantized = x_quantized * scale
        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ternary_quantize(x: torch.Tensor) -> torch.Tensor:
    """Applies ternary Straight-Through Estimator quantization to tensor x."""
    return TernaryQuantizeSTE.apply(x)


class TernaryPack(nn.Module):
    """
    QAT activation/weight module representing ternary quantization and packing.
    Recognized by Tenzo FX-to-MLIR exporter to emit `tenzo.ternary_pack`.
    """
    def __init__(self, values_per_byte: int = 4):
        super().__init__()
        self.values_per_byte = values_per_byte
        self.is_ternary_pack = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TernaryQuantizeSTE.apply(x)


class BitLinear(nn.Linear):
    """
    QAT layer for 1.58-bit (ternary) weights using Straight-Through Estimator.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.is_ternary = True

    def forward(self, input):
        quantized_weight = TernaryQuantizeSTE.apply(self.weight)
        return F.linear(input, quantized_weight, self.bias)


class QATBitSelfAttention(nn.Module):
    """
    PyTorch QAT Self-Attention module with ternary weights (BitLinear)
    and ternary packed Key/Value activations for tenzo.packed_attention.
    """
    def __init__(self, embed_dim=128, num_heads=None, head_dim=None, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.is_qat_attention = True

        self.q_proj = BitLinear(embed_dim, embed_dim, bias=bias)
        self.k_proj = BitLinear(embed_dim, embed_dim, bias=bias)
        self.v_proj = BitLinear(embed_dim, embed_dim, bias=bias)
        self.out_proj = BitLinear(embed_dim, embed_dim, bias=bias)
        self.k_pack = TernaryPack()
        self.v_pack = TernaryPack()

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        k_quant = self.k_pack(k)
        v_quant = self.v_pack(v)
        attn_out = F.scaled_dot_product_attention(q, k_quant, v_quant)
        return self.out_proj(attn_out)


class QATTransformerBlock(nn.Module):
    """
    Full Transformer block combining QATBitSelfAttention and BitLinear FFN.
    """
    def __init__(self, embed_dim=128, ff_dim=256, num_heads=None, head_dim=None, bias=False):
        super().__init__()
        self.attn = QATBitSelfAttention(embed_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, bias=bias)
        self.fc1 = BitLinear(embed_dim, ff_dim, bias=bias)
        self.relu = nn.ReLU()
        self.fc2 = BitLinear(ff_dim, embed_dim, bias=bias)

    def forward(self, x):
        h = x + self.attn(x)
        out = h + self.fc2(self.relu(self.fc1(h)))
        return out
