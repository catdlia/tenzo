from safetensors import safe_open
import numpy as np

# Open safetensors
with safe_open("export_output_bitnet/model.safetensors", framework="numpy", device="cpu") as f:
    weights = f.get_tensor("model.layers.0.mlp.down_proj.weight")

print(f"Weights shape: {weights.shape}, dtype: {weights.dtype}")

# It should be [20, 1280, 64] for TL1.
# Decode back to [N, K]
n_blocks, K_half, _ = weights.shape
N = n_blocks * 128
K = K_half * 2

decoded = np.zeros((N, K), dtype=np.int8)

# weights is int8, cast to uint8 to safely do bitwise ops
packed = weights.astype(np.uint8)

for b in range(n_blocks):
    for k in range(K_half):
        for c_half in range(64):
            val = packed[b, k, c_half]
            
            # Lower nibble
            low = val & 0x0F
            w_even_l = (low & 3) - 1
            w_odd_l = ((low >> 2) & 3) - 1
            
            # Upper nibble
            high = (val >> 4) & 0x0F
            w_even_u = (high & 3) - 1
            w_odd_u = ((high >> 2) & 3) - 1
            
            # Assign
            c_low = c_half
            decoded[b * 128 + c_low, k * 2] = w_even_l
            decoded[b * 128 + c_low, k * 2 + 1] = w_odd_l
            
            c_high = c_half + 64
            decoded[b * 128 + c_high, k * 2] = w_even_u
            decoded[b * 128 + c_high, k * 2 + 1] = w_odd_u

print("Decoded values:", np.unique(decoded))
print("First 10 values of first channel:", decoded[0, :10])
