import re

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

start_idx = content.find("AffineMap scoresMap =")
if start_idx != -1:
    end_idx = content.find("--- Step 3: Softmax ---", start_idx)
    print(content[start_idx:end_idx])
else:
    print("Not found")
