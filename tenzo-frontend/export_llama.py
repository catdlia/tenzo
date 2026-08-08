import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fx_to_mlir import export_torch_model_to_tenzo, HAS_TORCH
from qat import BitLinear

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, x, seq_pos=0):
        D = x.shape[-1]
        halfD = D // 2
        freq_idx = torch.arange(halfD, device=x.device, dtype=x.dtype)
        inv_freq = 1.0 / (10000.0 ** (2.0 * freq_idx / D))
        
        pos = float(seq_pos)
        angle = pos * inv_freq
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        x1 = x[..., :halfD]
        x2 = x[..., halfD:]
        
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        return torch.cat([rx1, rx2], dim=-1)

class BitSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = BitLinear(embed_dim, embed_dim)
        self.k_proj = BitLinear(embed_dim, embed_dim)
        self.v_proj = BitLinear(embed_dim, embed_dim)
        self.out_proj = BitLinear(embed_dim, embed_dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, S, E = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q, seq_pos=0)
        k = self.rope(k, seq_pos=0)
        
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, E)
        out = self.out_proj(attn_out)
        return out

class BitTransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, ff_dim=256, num_heads=4):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(embed_dim)
        self.attn = BitSelfAttention(embed_dim, num_heads=num_heads)
        self.post_attention_layernorm = LlamaRMSNorm(embed_dim)
        self.fc1 = BitLinear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = BitLinear(ff_dim, embed_dim)

    def forward(self, x):
        h = x + self.attn(self.input_layernorm(x))
        out = h + self.fc2(self.relu(self.fc1(self.post_attention_layernorm(h))))
        return out

class FullBitLlamaModel(nn.Module):
    def __init__(self, vocab_size=320, embed_dim=128, ff_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            BitTransformerBlock(embed_dim, ff_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = LlamaRMSNorm(embed_dim)
        self.lm_head = BitLinear(embed_dim, vocab_size)

    def forward(self, input_ids):
        # input_ids: tensor<1x1xi32>
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

def export_llama_model():
    out_dir = os.path.join(os.path.dirname(__file__), "export_output")
    os.makedirs(out_dir, exist_ok=True)

    print("[Export Llama] Constructing Full BitLlama Architecture (2 Layers)...")
    torch.manual_seed(42)
    model = FullBitLlamaModel(vocab_size=320, embed_dim=128, ff_dim=256, num_heads=4, num_layers=2)
    model.eval()

    sample_input = torch.tensor([[1]], dtype=torch.int32)

    print("[Export Llama] Exporting FX graph & packing 1.58-bit weights...")
    export_torch_model_to_tenzo(model, sample_input, output_dir=out_dir)

    # Export Vocabulary
    vocab_path = os.path.join(out_dir, "tokenizer.vocab")
    words = [
        "<unk>", "Tenzo", " ", "Edge", " ", "AI", "Hello", "world", ",", "I",
        " am", " a", " high", "-", "performance", " MLIR", " compiler", " running",
        " on", " AVX2", " micro", " kernels", ".", " What", " is", " the", " future",
        " of", " local", " LLM", " inference", "?"
    ]
    with open(vocab_path, "w") as f:
        for i in range(320):
            token_str = words[i % len(words)] if i < len(words) else f"token_{i}"
            f.write(f"{i} {token_str}\n")

    print(f"[Export Llama] Successfully exported full BitLlama model to {out_dir}")

if __name__ == "__main__":
    export_llama_model()
