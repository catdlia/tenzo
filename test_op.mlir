"builtin.module"() ({ %0 = "arith.constant"() {value = 0.0 : f32} : () -> tensor<f32> %1 = "bufferization.to_memref"(%0) : (tensor<f32>) -> memref<f32> }) : () -> ()
