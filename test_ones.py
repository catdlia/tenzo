with open('src/passes/LinalgLowering.cpp', 'r') as f:
    content = f.read()

import re

# Instead of val_l and val_u, insert 1.0f!
new_content = re.sub(
    r'Value val_l = b\.create<vector::ExtractElementOp>\(loc2, accScaled_l, iVal\);',
    'Value val_l = b.create<arith::ConstantOp>(loc2, b.getF32FloatAttr(1.0f));',
    content
)

new_content = re.sub(
    r'Value val_u = b\.create<vector::ExtractElementOp>\(loc2, accScaled_u, iVal\);',
    'Value val_u = b.create<arith::ConstantOp>(loc2, b.getF32FloatAttr(1.0f));',
    new_content
)

with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.write(new_content)
