with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

if "patterns.add<BitLinearTL1LoweringToLinalg>" not in content:
    content = content.replace(
        "patterns.add<MatMulLoweringToLinalg>(patterns.getContext());",
        "patterns.add<MatMulLoweringToLinalg>(patterns.getContext());\n    patterns.add<BitLinearTL1LoweringToLinalg>(patterns.getContext());"
    )

with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)
