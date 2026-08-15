import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fx_to_mlir import export_torch_model_to_tenzo, HAS_TORCH
if HAS_TORCH:
    try:
        from qat import BitLinear
    except ImportError:
        BitLinear = nn.Linear # fallback

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

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, ff_dim=256, num_heads=4):
        super().__init__()
        self.attn = BitSelfAttention(embed_dim, num_heads=num_heads)
        self.fc1 = BitLinear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = BitLinear(ff_dim, embed_dim)

    def forward(self, x):
        h = x + self.attn(x)
        out = h + self.fc2(self.relu(self.fc1(h)))
        return out

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "export_output")
    os.makedirs(out_dir, exist_ok=True)
    
    print("[Generation Test] Initializing model...")
    model = TransformerBlock(embed_dim=128, ff_dim=256)
    model.eval()

    # 1. Export MLIR for Decode Phase (SeqLen = 1)
    print("[Generation Test] Exporting MLIR...")
    sample_input = torch.randn(1, 1, 128)
    export_torch_model_to_tenzo(model, sample_input, output_dir=out_dir)
    
    # 2. Autoregressive Loop in PyTorch (5 tokens)
    print("[Generation Test] Simulating 5 tokens in PyTorch...")
    
    seq_inputs = []
    expected_outputs = []
    
    torch.manual_seed(42)
    
    # PyTorch KV Cache buffers
    py_cache_k = torch.zeros(1, 4, 1024, 32)
    py_cache_v = torch.zeros(1, 4, 1024, 32)
    
    with torch.no_grad():
        for i in range(5):
            new_token = torch.randn(1, 1, 128)
            seq_inputs.append(new_token)
            
            # Step 1: Attention projections + RoPE
            q = model.attn.q_proj(new_token).view(1, 1, 4, 32).transpose(1, 2)
            k = model.attn.k_proj(new_token).view(1, 1, 4, 32).transpose(1, 2)
            v = model.attn.v_proj(new_token).view(1, 1, 4, 32).transpose(1, 2)
            
            q = model.attn.rope(q, seq_pos=i)
            k = model.attn.rope(k, seq_pos=i)
            
            py_cache_k[:, :, i:i+1, :] = k
            py_cache_v[:, :, i:i+1, :] = v
            
            valid_k = py_cache_k[:, :, :i+1, :]
            valid_v = py_cache_v[:, :, :i+1, :]
            
            attn_out = F.scaled_dot_product_attention(q, valid_k, valid_v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, 128)
            attn_out = model.attn.out_proj(attn_out)
            
            h = new_token + attn_out
            out = h + model.fc2(model.relu(model.fc1(h)))
            
            expected_outputs.append(out)
            print(f"  -> Generated PyTorch token {i+1} (SeqPos={i})")

    # Save inputs (all 5 tokens in a flat list) and expected outputs
    input_bin_path = os.path.join(out_dir, "input.bin")
    expected_bin_path = os.path.join(out_dir, "expected.bin")
    
    # Pack them sequentially
    # We write 5 tokens of shape 1x1x128
    flat_inputs = torch.cat(seq_inputs, dim=1).numpy().astype(np.float32).tobytes()
    flat_outputs = torch.cat(expected_outputs, dim=1).numpy().astype(np.float32).tobytes()
    
    with open(input_bin_path, "wb") as f:
        f.write(flat_inputs)
        
    with open(expected_bin_path, "wb") as f:
        f.write(flat_outputs)
        
    vocab_path = os.path.join(out_dir, "tokenizer.vocab")
    with open(vocab_path, "w") as f:
        words = ["<unk>", "Tenzo", " ", "Edge", " ", "AI", "Hello", "World", "Transformer", "LLM"]
        for i in range(320):
            token_str = words[i % len(words)] if i < len(words) else f"token_{i}"
            f.write(f"{i} {token_str}\n")
            
    print(f"[Generation Test] Saved seq inputs ({len(flat_inputs)} bytes), expected outputs ({len(flat_outputs)} bytes), and tokenizer.vocab")
