
module {
  func.func @main(%arg0: tensor<1x1x128xf32>, %arg1: memref<1x64x64xi8>) -> tensor<1x1x128xf32> attributes {llvm.emit_c_interface} {
    %scale = arith.constant 1.5 : f32
    %w_tens = bufferization.to_tensor %arg1 restrict : memref<1x64x64xi8> to tensor<1x64x64xi8>
    %out = "tenzo.bitlinear_tl1"(%arg0, %w_tens, %scale) : (tensor<1x1x128xf32>, tensor<1x64x64xi8>, f32) -> tensor<1x1x128xf32>
    return %out : tensor<1x1x128xf32>
  }
}
