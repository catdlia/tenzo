import re

with open("src/tests/GenerationTest.cpp", "r") as f:
    content = f.read()

# Remove the memcpy loop
start = content.find("        for (int l = 0; l < num_layers; ++l) {\\n            size_t kv_num_floats")
if start != -1:
    end = content.find("        // When prefill completes", start)
    content = content[:start] + content[end:]

with open("src/tests/GenerationTest.cpp", "w") as f:
    f.write(content)
