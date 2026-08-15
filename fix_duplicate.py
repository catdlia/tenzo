with open('src/passes/LinalgLowering.cpp', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if i == 1130 and "tenzo.silu" in line:
        skip = True
    
    if skip:
        if line.strip() == "};" and i > 1130:
            skip = False
            continue
    else:
        new_lines.append(line)

with open('src/passes/LinalgLowering.cpp', 'w') as f:
    f.writelines(new_lines)
