import re

with open("src/tests/GenerationTest.cpp", "r") as f:
    content = f.read()

# Replace outputs.push_back(&out_k_caches[l]); with outputs.push_back(kv_cache.get_k_cache(l));
content = content.replace("outputs.push_back(&out_k_caches[l]);", "outputs.push_back(kv_cache.get_k_cache(l));")
content = content.replace("outputs.push_back(&out_v_caches[l]);", "outputs.push_back(kv_cache.get_v_cache(l));")

with open("src/tests/GenerationTest.cpp", "w") as f:
    f.write(content)
