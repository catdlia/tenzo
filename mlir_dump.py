with open("src/tests/GenerationTest.cpp", "r") as f:
    content = f.read()
if "module->dump();" not in content:
    content = content.replace("llvm::outs() << \"✅ Model compiled successfully!\\n\";", "llvm::outs() << \"✅ Model compiled successfully!\\n\";\n    module->dump();")
with open("src/tests/GenerationTest.cpp", "w") as f:
    f.write(content)
