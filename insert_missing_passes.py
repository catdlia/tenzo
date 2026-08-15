with open('src/passes/LinalgLowering.cpp', 'r') as f:
    content = f.read()

with open('relu2_pass.cpp', 'r') as f:
    relu2_content = f.read()

with open('mul_pass.cpp', 'r') as f:
    mul_content = f.read()

# insert before BitLinearElutLoweringToLinalg
insert_idx = content.find('struct BitLinearElutLoweringToLinalg')
content = content[:insert_idx] + relu2_content + '\n' + mul_content + '\n' + content[insert_idx:]

with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.write(content)
