import re

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

start_idx = content.find("struct BitLinearTL1LoweringToLinalg")
# find the next struct
end_idx = content.find("struct ", start_idx + 10)
print(f"Start: {start_idx}, End: {end_idx}")

# extract the code
code = content[start_idx:end_idx]
with open("tl1_code.cpp", "w") as f:
    f.write(code)
