import numpy as np

# Simulate python packing
np.random.seed(42)
N = 128
K = 32
K_half = K // 2
tern = np.random.randint(-1, 2, size=(N, K), dtype=np.int8)

w_even = tern[:, 0::2]
w_odd = tern[:, 1::2]
idx_matrix = (w_even + 1) | ((w_odd + 1) << 2)

idx_blocked = idx_matrix.reshape(1, 128, K_half)
idx_T = idx_blocked.transpose(0, 2, 1)
n_low_pack = idx_T[:, :, 0:64]
n_high_pack = idx_T[:, :, 64:128]
packed = (n_low_pack.astype(np.uint8)) | (n_high_pack.astype(np.uint8) << 4)

# Simulate C++ unpacking
n_block = 0
k_half_idx = 0

for chunk in range(4):
    for sub in range(2):
        for i in range(8):
            # Element index in the 16-byte chunk
            j = sub * 8 + i
            
            # Read from packed
            byte_val = packed[n_block, k_half_idx, chunk * 16 + j]
            
            # Lower nibble
            nibble_lower = byte_val & 0x0F
            w_even_plus_1_l = nibble_lower & 3
            w_even_l = w_even_plus_1_l - 1
            w_odd_shifted_l = nibble_lower >> 2
            w_odd_plus_1_l = w_odd_shifted_l & 3
            w_odd_l = w_odd_plus_1_l - 1
            
            # Expected lower
            c_low = chunk * 16 + sub * 8 + i
            expected_even_l = w_even[c_low, k_half_idx]
            expected_odd_l = w_odd[c_low, k_half_idx]
            
            if w_even_l != expected_even_l or w_odd_l != expected_odd_l:
                print(f"MISMATCH LOWER! c_low={c_low}")
            
            # Upper nibble
            nibble_upper = (byte_val >> 4) & 0x0F
            w_even_plus_1_u = nibble_upper & 3
            w_even_u = w_even_plus_1_u - 1
            w_odd_shifted_u = nibble_upper >> 2
            w_odd_plus_1_u = w_odd_shifted_u & 3
            w_odd_u = w_odd_plus_1_u - 1
            
            # Expected upper
            c_high = 64 + chunk * 16 + sub * 8 + i
            expected_even_u = w_even[c_high, k_half_idx]
            expected_odd_u = w_odd[c_high, k_half_idx]
            
            if w_even_u != expected_even_u or w_odd_u != expected_odd_u:
                print(f"MISMATCH UPPER! c_high={c_high}")
                
print("Simulation complete.")
