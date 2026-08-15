import numpy as np
import torch

def test_tl1_pack():
    np.random.seed(42)
    # Simulate a small weight matrix
    M = 32
    K = 128
    
    # Random ternary weights
    tern = np.random.randint(-1, 2, size=(M, K), dtype=np.int8)
    
    # 1. Microsoft TL1 dense mapping: (w0+1)*3 + (w1+1)
    w_even = tern[:, 0::2]  # [M, K/2]
    w_odd = tern[:, 1::2]   # [M, K/2]
    
    idx_matrix = (w_even + 1) * 3 + (w_odd + 1)  # [M, K/2] uint8, values 0..8
    
    N, K_half = idx_matrix.shape
    bm = 32
    n_blocks = N // bm
    
    # Reshape N into blocks of 32, then transpose K to be the middle dimension
    idx_blocked = idx_matrix.reshape(n_blocks, bm, K_half)
    idx_T = idx_blocked.transpose(0, 2, 1)  # [1, 64, 32]
    
    # Pack 32 channels into 16 bytes.
    # Channel 0..15 in lower nibble, channel 16..31 in upper nibble
    n_low_pack = idx_T[:, :, 0:16]    # [1, 64, 16]
    n_high_pack = idx_T[:, :, 16:32]  # [1, 64, 16]
    packed = (n_low_pack.astype(np.uint8)) | (n_high_pack.astype(np.uint8) << 4)
    
    # 2. Simulate Activations
    A = np.random.randint(-128, 127, size=(K,), dtype=np.int8)
    
    # Expected output:
    # dot product of A and tern
    # A is (K,), tern is (M, K)
    # result is (M,)
    expected = np.dot(tern.astype(np.int32), A.astype(np.int32))
    
    # 3. Simulate LUT execution
    result = np.zeros(M, dtype=np.int32)
    
    for k_half in range(K_half):
        # Compute LUT for this k_half
        # A[2k], A[2k+1]
        a0 = int(A[2*k_half])
        a1 = int(A[2*k_half + 1])
        
        lut = np.zeros(9, dtype=np.int16)
        # Reconstruct the 9 values
        for w0 in [-1, 0, 1]:
            for w1 in [-1, 0, 1]:
                idx = (w0 + 1) * 3 + (w1 + 1)
                lut[idx] = w0 * a0 + w1 * a1
                
        # Now lookup using packed
        packed_bytes = packed[0, k_half, :] # [16] bytes
        
        # Unpack
        idx_A = packed_bytes & 0x0F
        idx_B = (packed_bytes >> 4) & 0x0F
        
        # Lookup
        for i in range(16):
            val_A = lut[idx_A[i]]
            val_B = lut[idx_B[i]]
            
            result[i] += val_A
            result[i + 16] += val_B
            
    print("Expected:", expected[:10])
    print("LUT out :", result[:10])
    assert np.array_equal(expected, result), "LUT result does not match expected!"
    print("✅ test_tl1_pack passed successfully!")

if __name__ == "__main__":
    test_tl1_pack()
