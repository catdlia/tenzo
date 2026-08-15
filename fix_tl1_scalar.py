import re

with open("tl1_code.cpp", "r") as f:
    code = f.read()

# We want to replace the body of the `GenericOp` for BitLinearTL1.
# Wait, my current BitLinearTL1LoweringToLinalg uses scf::ForOp inside a GenericOp.
