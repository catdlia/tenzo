with open('src/passes/LinalgLowering.cpp', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'Value zero8i32 = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vec8f32Type, rewriter.getF32FloatAttr(0.0f)));' in line:
        new_lines.append('        auto vec8f32Type = VectorType::get({8}, rewriter.getF32Type());\n')
        new_lines.append(line)
    elif 'auto vec8f32Type = VectorType::get({8}, rewriter.getF32Type());' in line:
        pass # already added
    else:
        new_lines.append(line)

with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.writelines(new_lines)
