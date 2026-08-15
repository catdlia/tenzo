import numpy as np

offset = 1313351680
n_blocks = 20
k_half = 1280
c_half = 64

# Open weights.bin
with open("export_output_bitnet/weights.bin", "rb") as f:
    f.seek(offset)
    raw = f.read(n_blocks * k_half * c_half)

packed = np.frombuffer(raw, dtype=np.int8).reshape(n_blocks, k_half, c_half)

# Let's decode it
decoded = np.zeros((n_blocks * 128, k_half * 2), dtype=np.int8)

for b in range(1): # test first block
    for k in range(2): # test first 2 K_half
        for c in range(64):
            val = packed[b, k, c]
            
            # C++ simulates i8
            # In python, val is int8 (-128 to 127). We can use it directly if we are careful with bitwise ops
            val_u = np.uint8(val)
            
            low = val_u & 0x0F
            w_even_l = (low & 3) - 1
            w_odd_l = ((low >> 2) & 3) - 1
            
            high = (val_u >> 4) & 0x0F
            w_even_u = (high & 3) - 1
            w_odd_u = ((high >> 2) & 3) - 1
            
            decoded[b * 128 + c, k * 2] = w_even_l
            decoded[b * 128 + c, k * 2 + 1] = w_odd_l
            
            decoded[b * 128 + c + 64, k * 2] = w_even_u
            decoded[b * 128 + c + 64, k * 2 + 1] = w_odd_u

print(decoded[:10, :4])
