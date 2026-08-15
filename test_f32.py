with open('src/passes/LinalgLowering.cpp', 'r') as f:
    content = f.read()

# I want to disable act1D_q logic and just use act1D directly.
# Look for a_even extraction:
# Value a_even = b3.create<tensor::ExtractOp>(loc3, act1D_q, ValueRange{k_idx});
# Value a_odd = b3.create<tensor::ExtractOp>(loc3, act1D_q, ValueRange{k_idx_plus_1});

import re

new_content = re.sub(
    r'Value a_even = b3\.create<tensor::ExtractOp>\(loc3, act1D_q, ValueRange\{k_idx\}\);',
    'Value a_even = b3.create<tensor::ExtractOp>(loc3, act1D, ValueRange{k_idx});',
    content
)

new_content = re.sub(
    r'Value a_odd = b3\.create<tensor::ExtractOp>\(loc3, act1D_q, ValueRange\{k_idx_plus_1\}\);',
    'Value a_odd = b3.create<tensor::ExtractOp>(loc3, act1D, ValueRange{k_idx_plus_1});',
    new_content
)

# And now, a_even and a_odd are f32!
# We don't need ExtSIOp for them!
new_content = re.sub(
    r'Value a_even_i32 = b3\.create<arith::ExtSIOp>\(loc3, b3\.getI32Type\(\), a_even\);',
    'Value a_even_i32 = a_even;',
    new_content
)
new_content = re.sub(
    r'Value a_odd_i32 = b3\.create<arith::ExtSIOp>\(loc3, b3\.getI32Type\(\), a_odd\);',
    'Value a_odd_i32 = a_odd;',
    new_content
)

# Broadcast them to vec8f32Type!
new_content = re.sub(
    r'Value a_even_vec = b3\.create<vector::BroadcastOp>\(loc3, vec8i32Type, a_even_i32\);',
    'Value a_even_vec = b3.create<vector::BroadcastOp>(loc3, vec8f32Type, a_even_i32);',
    new_content
)
new_content = re.sub(
    r'Value a_odd_vec = b3\.create<vector::BroadcastOp>\(loc3, vec8i32Type, a_odd_i32\);',
    'Value a_odd_vec = b3.create<vector::BroadcastOp>(loc3, vec8f32Type, a_odd_i32);',
    new_content
)

# Also, w_even_32_l is vec8i32Type. We need to convert it to vec8f32Type!
# m_even_l = a_even_vec * w_even_32_l
# We need to change MulIOp to MulFOp, and convert w_even_32_l to f32.
def repl_mul(match):
    prefix = match.group(1) # w_even_32_l or w_even_32_u etc
    return f"Value {prefix}_f32 = b3.create<arith::SIToFPOp>(loc3, vec8f32Type, {prefix});\n" \
           f"Value m_even_l = b3.create<arith::MulFOp>(loc3, a_even_vec, {prefix}_f32);"

new_content = re.sub(
    r'Value m_even_l = b3\.create<arith::MulIOp>\(loc3, a_even_vec, (w_even_32_l)\);',
    repl_mul,
    new_content
)

def repl_mul2(match):
    prefix = match.group(1)
    return f"Value {prefix}_f32 = b3.create<arith::SIToFPOp>(loc3, vec8f32Type, {prefix});\n" \
           f"Value m_odd_l = b3.create<arith::MulFOp>(loc3, a_odd_vec, {prefix}_f32);"

new_content = re.sub(
    r'Value m_odd_l = b3\.create<arith::MulIOp>\(loc3, a_odd_vec, (w_odd_32_l)\);',
    repl_mul2,
    new_content
)

def repl_mul3(match):
    prefix = match.group(1)
    return f"Value {prefix}_f32 = b3.create<arith::SIToFPOp>(loc3, vec8f32Type, {prefix});\n" \
           f"Value m_even_u = b3.create<arith::MulFOp>(loc3, a_even_vec, {prefix}_f32);"
new_content = re.sub(
    r'Value m_even_u = b3\.create<arith::MulIOp>\(loc3, a_even_vec, (w_even_32_u)\);',
    repl_mul3,
    new_content
)

def repl_mul4(match):
    prefix = match.group(1)
    return f"Value {prefix}_f32 = b3.create<arith::SIToFPOp>(loc3, vec8f32Type, {prefix});\n" \
           f"Value m_odd_u = b3.create<arith::MulFOp>(loc3, a_odd_vec, {prefix}_f32);"
new_content = re.sub(
    r'Value m_odd_u = b3\.create<arith::MulIOp>\(loc3, a_odd_vec, (w_odd_32_u)\);',
    repl_mul4,
    new_content
)

# And AddIOp to AddFOp
new_content = re.sub(r'b3\.create<arith::AddIOp>\(loc3, m_even_l, m_odd_l\)', 'b3.create<arith::AddFOp>(loc3, m_even_l, m_odd_l)', new_content)
new_content = re.sub(r'b3\.create<arith::AddIOp>\(loc3, m_even_u, m_odd_u\)', 'b3.create<arith::AddFOp>(loc3, m_even_u, m_odd_u)', new_content)

new_content = re.sub(r'b3\.create<arith::AddIOp>\(loc3, curAccs\[acc_idx_l\], m_sum_l\)', 'b3.create<arith::AddFOp>(loc3, curAccs[acc_idx_l], m_sum_l)', new_content)
new_content = re.sub(r'b3\.create<arith::AddIOp>\(loc3, curAccs\[acc_idx_u\], m_sum_u\)', 'b3.create<arith::AddFOp>(loc3, curAccs[acc_idx_u], m_sum_u)', new_content)

# And zeroAccs must be f32!
new_content = re.sub(
    r'Value zero8i32 = rewriter\.create<arith::ConstantOp>\(loc, DenseElementsAttr::get\(vec8i32Type, rewriter\.getI32IntegerAttr\(0\)\)\);',
    'Value zero8i32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec8f32Type, rewriter.getF32FloatAttr(0.0f)));',
    new_content
)

# And in the final loop:
# Value accF32_l = b.create<arith::SIToFPOp>(loc2, vec8f32Type, outAccs[acc_idx_l]);
# becomes just outAccs[acc_idx_l]
new_content = re.sub(
    r'Value accF32_l = b\.create<arith::SIToFPOp>\(loc2, vec8f32Type, outAccs\[acc_idx_l\]\);',
    'Value accF32_l = outAccs[acc_idx_l];',
    new_content
)
new_content = re.sub(
    r'Value accF32_l = outAccs\[acc_idx_l\];\n\s*Value accScaled_l = b\.create<arith::MulFOp>\(loc2, accF32_l, totalScaleVec\);',
    'Value accF32_l = outAccs[acc_idx_l];\n                        Value accScaled_l = b.create<arith::MulFOp>(loc2, accF32_l, totalScaleVec);',
    new_content
) # just fixing spacing
new_content = re.sub(
    r'Value accF32_u = b\.create<arith::SIToFPOp>\(loc2, vec8f32Type, outAccs\[acc_idx_u\]\);',
    'Value accF32_u = outAccs[acc_idx_u];',
    new_content
)

# totalScale becomes just scaleF
new_content = re.sub(
    r'Value totalScaleVec = b\.create<vector::BroadcastOp>\(loc2, vec8f32Type, totalScale\);',
    'Value totalScaleVec = b.create<vector::BroadcastOp>(loc2, vec8f32Type, scaleF);',
    new_content
)


with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.write(new_content)
