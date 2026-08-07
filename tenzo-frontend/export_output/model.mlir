// Standalone Mock MLIR model for Zero-Copy Bridge Test
module {
  memref.global "public" constant @fc1_weight : memref<20x10xf32> = dense<0.0> {bit_width = 2 : i32, offset = 0 : i64, quant_scheme = "ternary", size_bytes = 64 : i64}

  func.func @main(%arg0: tensor<1x10xf32>) -> tensor<1x20xf32> {
    %0 = memref.get_global @fc1_weight : memref<20x10xf32>
    %1 = bufferization.to_tensor %0 : memref<20x10xf32> to tensor<20x10xf32>
    %2 = "tenzo.quantize"(%1) {bit_width = 2 : i32, quant_scheme = "ternary"} : (tensor<20x10xf32>) -> tensor<20x10xf32>
    %3 = "tenzo.matmul"(%arg0, %2) : (tensor<1x10xf32>, tensor<20x10xf32>) -> tensor<1x20xf32>
    return %3 : tensor<1x20xf32>
  }
}
