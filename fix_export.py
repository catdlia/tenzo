with open("tenzo-frontend/export_bitnet.py", "r") as f:
    content = f.read()

content = content.replace(
"""                # Pack consecutive N pairs into 1 byte along the last dimension
                n_even_pack = idx_T[:, :, 0::2]  # [N/128, K/2, 64]
                n_odd_pack = idx_T[:, :, 1::2]   # [N/128, K/2, 64]
                packed = (n_even_pack.astype(np.uint8)) | (n_odd_pack.astype(np.uint8) << 4)""",
"""                # Pack N pairs into 1 byte: channels 0..63 in lower nibble, 64..127 in upper nibble
                n_low_pack = idx_T[:, :, 0:64]    # [N/128, K/2, 64]
                n_high_pack = idx_T[:, :, 64:128] # [N/128, K/2, 64]
                packed = (n_low_pack.astype(np.uint8)) | (n_high_pack.astype(np.uint8) << 4)"""
)

with open("tenzo-frontend/export_bitnet.py", "w") as f:
    f.write(content)
