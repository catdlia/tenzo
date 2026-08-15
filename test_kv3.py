import re

with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

start_idx = content.find("tenzo.kv_cache_update -> tensor.insert_slice")
if start_idx != -1:
    end_idx = content.find("//===", start_idx + 100)
    if end_idx == -1: end_idx = len(content)
    print(content[start_idx:end_idx])
else:
    print("Not found")
