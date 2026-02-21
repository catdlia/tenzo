#!/bin/bash
for file in src/tests/*.cpp; do
    sed -i -e 's/mlir::registerLLVMDialectTranslation(context);/tenzo::registerAllTenzoDialectTranslations(context);\n    mlir::registerLLVMDialectTranslation(context);/g' $file
done
