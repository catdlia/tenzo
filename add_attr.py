with open("src/passes/LinalgLowering.cpp", "r") as f:
    content = f.read()

target = r"""        auto nLoop = rewriter.create<scf::ForOp>(loc, c0, cn_blocks, c1, ValueRange{emptyOut},
            [&](OpBuilder &b, Location loc2, Value n_idx, ValueRange args) {"""

replacement = r"""        auto nLoop = rewriter.create<scf::ForOp>(loc, c0, cn_blocks, c1, ValueRange{emptyOut},
            [&](OpBuilder &b, Location loc2, Value n_idx, ValueRange args) {
                Value outTensor = args[0];"""

if "nLoop->setAttr(" not in content:
    content = content.replace("b.create<scf::YieldOp>(loc2, newOutTensor);\n            });", "b.create<scf::YieldOp>(loc2, newOutTensor);\n            });\n        nLoop->setAttr(\"tenzo.parallelize\", rewriter.getUnitAttr());")

with open("src/passes/LinalgLowering.cpp", "w") as f:
    f.write(content)
