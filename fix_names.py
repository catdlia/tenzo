with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

content = content.replace("adaptor.getLhs()", "adaptor.getInput()")
content = content.replace("adaptor.getRhs()", "adaptor.getWeights()")

with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)
