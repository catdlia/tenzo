with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

with open("relu2_pass.cpp", "r") as f:
    relu2_content = f.read()

if "Relu2LoweringToLinalg" not in content:
    content = content.replace(
        "void tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {",
        relu2_content + "\nvoid tenzo::populateTenzoToLinalgConversionPatterns(RewritePatternSet &patterns) {\n"
    )
    content = content.replace(
        "patterns.add<ReluLoweringToLinalg>(patterns.getContext());",
        "patterns.add<ReluLoweringToLinalg>(patterns.getContext());\n    patterns.add<Relu2LoweringToLinalg>(patterns.getContext());\n    patterns.add<SiLuLoweringToLinalg>(patterns.getContext());"
    )

with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)

