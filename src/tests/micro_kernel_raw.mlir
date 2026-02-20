// micro_kernel_raw.mlir - Micro-kernel with raw pointer interface
// Uses llvm.ptr instead of memref for direct C ABI compatibility

module {
  // Raw pointer version - compatible with C extern "C" void f(float*, float*, float*)
  llvm.func @micro_kernel_6x16_raw(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %c256 = llvm.mlir.constant(256 : i64) : i64

    // Row strides
    %stride_a = llvm.mlir.constant(256 : i64) : i64  // A is 6x256, stride = 256
    %stride_b = llvm.mlir.constant(16 : i64) : i64   // B is 256x16, stride = 16
    %stride_c = llvm.mlir.constant(16 : i64) : i64   // C is 6x16, stride = 16

    // Load C into 12 YMM registers
    // Row 0
    %c_ptr_0_0 = llvm.getelementptr %C[%c0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c0_0 = llvm.load %c_ptr_0_0 {alignment = 32} : !llvm.ptr -> vector<8xf32>
    %c_ptr_0_8 = llvm.getelementptr %C[%c8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c0_1 = llvm.load %c_ptr_0_8 {alignment = 32} : !llvm.ptr -> vector<8xf32>

    // Row 1
    %c_off_1 = llvm.mlir.constant(16 : i64) : i64
    %c_ptr_1_0 = llvm.getelementptr %C[%c_off_1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c1_0 = llvm.load %c_ptr_1_0 {alignment = 32} : !llvm.ptr -> vector<8xf32>
    %c_off_1_8 = llvm.mlir.constant(24 : i64) : i64
    %c_ptr_1_8 = llvm.getelementptr %C[%c_off_1_8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c1_1 = llvm.load %c_ptr_1_8 {alignment = 32} : !llvm.ptr -> vector<8xf32>

    // Row 2
    %c_off_2 = llvm.mlir.constant(32 : i64) : i64
    %c_ptr_2_0 = llvm.getelementptr %C[%c_off_2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c2_0 = llvm.load %c_ptr_2_0 {alignment = 32} : !llvm.ptr -> vector<8xf32>
    %c_off_2_8 = llvm.mlir.constant(40 : i64) : i64
    %c_ptr_2_8 = llvm.getelementptr %C[%c_off_2_8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c2_1 = llvm.load %c_ptr_2_8 {alignment = 32} : !llvm.ptr -> vector<8xf32>

    // Row 3
    %c_off_3 = llvm.mlir.constant(48 : i64) : i64
    %c_ptr_3_0 = llvm.getelementptr %C[%c_off_3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c3_0 = llvm.load %c_ptr_3_0 {alignment = 32} : !llvm.ptr -> vector<8xf32>
    %c_off_3_8 = llvm.mlir.constant(56 : i64) : i64
    %c_ptr_3_8 = llvm.getelementptr %C[%c_off_3_8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c3_1 = llvm.load %c_ptr_3_8 {alignment = 32} : !llvm.ptr -> vector<8xf32>

    // Row 4
    %c_off_4 = llvm.mlir.constant(64 : i64) : i64
    %c_ptr_4_0 = llvm.getelementptr %C[%c_off_4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c4_0 = llvm.load %c_ptr_4_0 {alignment = 32} : !llvm.ptr -> vector<8xf32>
    %c_off_4_8 = llvm.mlir.constant(72 : i64) : i64
    %c_ptr_4_8 = llvm.getelementptr %C[%c_off_4_8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c4_1 = llvm.load %c_ptr_4_8 {alignment = 32} : !llvm.ptr -> vector<8xf32>

    // Row 5
    %c_off_5 = llvm.mlir.constant(80 : i64) : i64
    %c_ptr_5_0 = llvm.getelementptr %C[%c_off_5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c5_0 = llvm.load %c_ptr_5_0 {alignment = 32} : !llvm.ptr -> vector<8xf32>
    %c_off_5_8 = llvm.mlir.constant(88 : i64) : i64
    %c_ptr_5_8 = llvm.getelementptr %C[%c_off_5_8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %c5_1 = llvm.load %c_ptr_5_8 {alignment = 32} : !llvm.ptr -> vector<8xf32>

    // K-loop (unrolled manually, using llvm.br for loop)
    llvm.br ^loop_header(%c0, %c0_0, %c0_1, %c1_0, %c1_1, %c2_0, %c2_1,
                         %c3_0, %c3_1, %c4_0, %c4_1, %c5_0, %c5_1
                         : i64, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
                           vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
                           vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>)

  ^loop_header(%k: i64, %acc0_0: vector<8xf32>, %acc0_1: vector<8xf32>,
               %acc1_0: vector<8xf32>, %acc1_1: vector<8xf32>,
               %acc2_0: vector<8xf32>, %acc2_1: vector<8xf32>,
               %acc3_0: vector<8xf32>, %acc3_1: vector<8xf32>,
               %acc4_0: vector<8xf32>, %acc4_1: vector<8xf32>,
               %acc5_0: vector<8xf32>, %acc5_1: vector<8xf32>):
    %cond = llvm.icmp "slt" %k, %c256 : i64
    llvm.cond_br %cond, ^loop_body, ^loop_exit(%acc0_0, %acc0_1, %acc1_0, %acc1_1,
                                               %acc2_0, %acc2_1, %acc3_0, %acc3_1,
                                               %acc4_0, %acc4_1, %acc5_0, %acc5_1
                                               : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
                                                 vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
                                                 vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>)

  ^loop_body:
    // Load B[k, 0:8] and B[k, 8:16]
    %b_row_offset = llvm.mul %k, %c16 : i64
    %b_ptr_0 = llvm.getelementptr %B[%b_row_offset] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %b0 = llvm.load %b_ptr_0 {alignment = 32} : !llvm.ptr -> vector<8xf32>
    %b_col_8 = llvm.add %b_row_offset, %c8 : i64
    %b_ptr_8 = llvm.getelementptr %B[%b_col_8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %b1 = llvm.load %b_ptr_8 {alignment = 32} : !llvm.ptr -> vector<8xf32>

    // Row 0: A[0, k]
    %a_ptr_0 = llvm.getelementptr %A[%k] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a0_scalar = llvm.load %a_ptr_0 : !llvm.ptr -> f32
    %a0 = llvm.mlir.undef : vector<8xf32>
    %a0_0 = llvm.insertelement %a0_scalar, %a0[%c0 : i64] : vector<8xf32>
    %a0_bcast = llvm.shufflevector %a0_0, %a0 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>
    %new0_0 = llvm.intr.fma(%a0_bcast, %b0, %acc0_0) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %new0_1 = llvm.intr.fma(%a0_bcast, %b1, %acc0_1) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>

    // Row 1: A[1, k]
    %a_off_1 = llvm.add %k, %stride_a : i64
    %a_ptr_1 = llvm.getelementptr %A[%a_off_1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a1_scalar = llvm.load %a_ptr_1 : !llvm.ptr -> f32
    %a1 = llvm.mlir.undef : vector<8xf32>
    %a1_0 = llvm.insertelement %a1_scalar, %a1[%c0 : i64] : vector<8xf32>
    %a1_bcast = llvm.shufflevector %a1_0, %a1 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>
    %new1_0 = llvm.intr.fma(%a1_bcast, %b0, %acc1_0) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %new1_1 = llvm.intr.fma(%a1_bcast, %b1, %acc1_1) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>

    // Row 2: A[2, k]
    %a_stride_2 = llvm.mlir.constant(512 : i64) : i64
    %a_off_2 = llvm.add %k, %a_stride_2 : i64
    %a_ptr_2 = llvm.getelementptr %A[%a_off_2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a2_scalar = llvm.load %a_ptr_2 : !llvm.ptr -> f32
    %a2 = llvm.mlir.undef : vector<8xf32>
    %a2_0 = llvm.insertelement %a2_scalar, %a2[%c0 : i64] : vector<8xf32>
    %a2_bcast = llvm.shufflevector %a2_0, %a2 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>
    %new2_0 = llvm.intr.fma(%a2_bcast, %b0, %acc2_0) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %new2_1 = llvm.intr.fma(%a2_bcast, %b1, %acc2_1) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>

    // Row 3: A[3, k]
    %a_stride_3 = llvm.mlir.constant(768 : i64) : i64
    %a_off_3 = llvm.add %k, %a_stride_3 : i64
    %a_ptr_3 = llvm.getelementptr %A[%a_off_3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a3_scalar = llvm.load %a_ptr_3 : !llvm.ptr -> f32
    %a3 = llvm.mlir.undef : vector<8xf32>
    %a3_0 = llvm.insertelement %a3_scalar, %a3[%c0 : i64] : vector<8xf32>
    %a3_bcast = llvm.shufflevector %a3_0, %a3 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>
    %new3_0 = llvm.intr.fma(%a3_bcast, %b0, %acc3_0) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %new3_1 = llvm.intr.fma(%a3_bcast, %b1, %acc3_1) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>

    // Row 4: A[4, k]
    %a_stride_4 = llvm.mlir.constant(1024 : i64) : i64
    %a_off_4 = llvm.add %k, %a_stride_4 : i64
    %a_ptr_4 = llvm.getelementptr %A[%a_off_4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a4_scalar = llvm.load %a_ptr_4 : !llvm.ptr -> f32
    %a4 = llvm.mlir.undef : vector<8xf32>
    %a4_0 = llvm.insertelement %a4_scalar, %a4[%c0 : i64] : vector<8xf32>
    %a4_bcast = llvm.shufflevector %a4_0, %a4 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>
    %new4_0 = llvm.intr.fma(%a4_bcast, %b0, %acc4_0) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %new4_1 = llvm.intr.fma(%a4_bcast, %b1, %acc4_1) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>

    // Row 5: A[5, k]
    %a_stride_5 = llvm.mlir.constant(1280 : i64) : i64
    %a_off_5 = llvm.add %k, %a_stride_5 : i64
    %a_ptr_5 = llvm.getelementptr %A[%a_off_5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a5_scalar = llvm.load %a_ptr_5 : !llvm.ptr -> f32
    %a5 = llvm.mlir.undef : vector<8xf32>
    %a5_0 = llvm.insertelement %a5_scalar, %a5[%c0 : i64] : vector<8xf32>
    %a5_bcast = llvm.shufflevector %a5_0, %a5 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>
    %new5_0 = llvm.intr.fma(%a5_bcast, %b0, %acc5_0) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %new5_1 = llvm.intr.fma(%a5_bcast, %b1, %acc5_1) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>

    // Increment k
    %k_next = llvm.add %k, %c1 : i64
    llvm.br ^loop_header(%k_next, %new0_0, %new0_1, %new1_0, %new1_1,
                         %new2_0, %new2_1, %new3_0, %new3_1,
                         %new4_0, %new4_1, %new5_0, %new5_1
                         : i64, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
                           vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
                           vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>)

  ^loop_exit(%final0_0: vector<8xf32>, %final0_1: vector<8xf32>,
             %final1_0: vector<8xf32>, %final1_1: vector<8xf32>,
             %final2_0: vector<8xf32>, %final2_1: vector<8xf32>,
             %final3_0: vector<8xf32>, %final3_1: vector<8xf32>,
             %final4_0: vector<8xf32>, %final4_1: vector<8xf32>,
             %final5_0: vector<8xf32>, %final5_1: vector<8xf32>):
    // Store C back
    llvm.store %final0_0, %c_ptr_0_0 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final0_1, %c_ptr_0_8 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final1_0, %c_ptr_1_0 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final1_1, %c_ptr_1_8 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final2_0, %c_ptr_2_0 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final2_1, %c_ptr_2_8 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final3_0, %c_ptr_3_0 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final3_1, %c_ptr_3_8 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final4_0, %c_ptr_4_0 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final4_1, %c_ptr_4_8 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final5_0, %c_ptr_5_0 {alignment = 32} : vector<8xf32>, !llvm.ptr
    llvm.store %final5_1, %c_ptr_5_8 {alignment = 32} : vector<8xf32>, !llvm.ptr

    llvm.return
  }
}

