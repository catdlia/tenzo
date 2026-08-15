import re

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

start_idx = content.find("struct KVCacheUpdateLoweringToLinalg")
if start_idx != -1:
    end_idx = content.find("struct ", start_idx + 10)
    print(content[start_idx:end_idx])
else:
    print("Not found")
