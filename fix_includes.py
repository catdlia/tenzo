with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

content = content.replace('#include "mlir/Dialect/Func/IR/FuncOps.h"',
"""#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
""")

content = content.replace(
'target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, math::MathDialect, tensor::TensorDialect>();',
'target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, math::MathDialect, tensor::TensorDialect, scf::SCFDialect, vector::VectorDialect>();'
)

with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)
