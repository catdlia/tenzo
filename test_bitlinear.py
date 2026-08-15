import torch

# Simulate random ternary weight matrix
torch.manual_seed(42)
w = torch.randint(-1, 2, (2560, 2560)).float()
scale = w.abs().mean().clamp(min=1e-5)

# Simulated random activation from residual stream
x = torch.randn(1, 1, 2560) * 5.0  # std 5.0

# 1. Tenzo current implementation (Float32 with scale)
w_quant = torch.round(w.clamp(-1, 1)) * scale
out_tenzo = torch.matmul(x, w_quant.t())

# 2. Official BitLinear implementation (with activation quantization)
x_scale = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
x_quant = torch.round(x * x_scale).clamp(-128, 127)
w_quant_pure = torch.round(w.clamp(-1, 1))

out_official = torch.matmul(x_quant, w_quant_pure.t()) * (scale / x_scale)

print("--- Output Stats ---")
print(f"Tenzo Float32 variance: {out_tenzo.var().item():.4f}, max: {out_tenzo.max().item():.4f}")
print(f"BitNet TL1 variance: {out_official.var().item():.4f}, max: {out_official.max().item():.4f}")

# Check difference
diff = (out_tenzo - out_official).abs()
print(f"Mean Abs Diff: {diff.mean().item():.4f}")
