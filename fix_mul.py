import re

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

with open("mul_pass.cpp", "r") as f:
    mul_content = f.read()

mul_code = mul_content[mul_content.find("// tenzo.mul"):mul_content.find("// tenzo.rope")]

if "struct MulLoweringToLinalg" not in content:
    content = content.replace(
        "void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {",
        mul_code + "\nvoid tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {\n"
    )
    content = content.replace(
        "patterns.add<SiLuLoweringToLinalg>(patterns.getContext());",
        "patterns.add<SiLuLoweringToLinalg>(patterns.getContext());\n    patterns.add<MulLoweringToLinalg>(patterns.getContext());"
    )

with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)

