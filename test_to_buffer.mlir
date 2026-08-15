"builtin.module"() ({ %0 = "tensor.empty"() : () -> tensor<1x1x128xf32> %1 = "bufferization.to_buffer"(%0) : (tensor<1x1x128xf32>) -> memref<1x1x128xf32> }) : () -> ()
