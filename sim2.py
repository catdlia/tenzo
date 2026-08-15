import numpy as np

# Test negative bitwise extraction in python vs c++ behavior
n_low = np.uint8(11)  # binary 1011
n_high = np.uint8(9)  # binary 1001

packed_uint8 = n_low | (n_high << 4)  # 1001_1011 = 155
packed_int8 = np.int8(packed_uint8)   # 155 -> -101

# Simulate C++ i8
# C++ i8 has values -128..127. 
cpp_val = int(packed_int8)

# lower nibble extraction: (val & 0x0F)
lower_nibble = cpp_val & 0x0F
print(f"Lower nibble: {lower_nibble} (Expected {n_low})")

# upper nibble extraction: (val >> 4) & 0x0F
# in C++, if unsigned shift:
upper_nibble = (packed_uint8 >> 4) & 0x0F
print(f"Upper nibble (unsigned shift): {upper_nibble} (Expected {n_high})")

# Wait, in MLIR we use arith::ShRUIOp which is UNSIGNED shift.
# BUT we apply it to `wLoad_chunk` which is `vector<16xi8>`.
# In MLIR, ShRUIOp treats the operand as an unsigned integer!
# So -101 (1001_1011) is treated as 155 (1001_1011).
# 155 >> 4 = 9 (0000_1001).
# 9 & 0x0F = 9.
