// micro_kernel.mlir - MLIR source for the 6x16 micro-kernel
// This file is compiled to micro_kernel.o and linked with micro_bench
//
// Register allocation:
//   YMM0-YMM11:  12 accumulators (C: 6 rows × 2 vectors)
//   YMM12-YMM13: 2 B vectors
//   YMM14:       A broadcast
//   YMM15:       Spare

module {
  // Main micro-kernel function
  // A: [6 x 256] row-major
  // B: [256 x 16] packed column-major (column elements consecutive)
  // C: [6 x 16] row-major
  func.func @micro_kernel_6x16(%A: memref<6x256xf32>, %B: memref<256x16xf32>, %C: memref<6x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0.0 : f32

    // STEP 1: Load C[6][16] into 12 vector registers
    // Row 0
    %c0_0 = vector.transfer_read %C[%c0, %c0], %zero : memref<6x16xf32>, vector<8xf32>
    %c0_1 = vector.transfer_read %C[%c0, %c8], %zero : memref<6x16xf32>, vector<8xf32>
    // Row 1
    %i1 = arith.constant 1 : index
    %c1_0 = vector.transfer_read %C[%i1, %c0], %zero : memref<6x16xf32>, vector<8xf32>
    %c1_1 = vector.transfer_read %C[%i1, %c8], %zero : memref<6x16xf32>, vector<8xf32>
    // Row 2
    %i2 = arith.constant 2 : index
    %c2_0 = vector.transfer_read %C[%i2, %c0], %zero : memref<6x16xf32>, vector<8xf32>
    %c2_1 = vector.transfer_read %C[%i2, %c8], %zero : memref<6x16xf32>, vector<8xf32>
    // Row 3
    %i3 = arith.constant 3 : index
    %c3_0 = vector.transfer_read %C[%i3, %c0], %zero : memref<6x16xf32>, vector<8xf32>
    %c3_1 = vector.transfer_read %C[%i3, %c8], %zero : memref<6x16xf32>, vector<8xf32>
    // Row 4
    %i4 = arith.constant 4 : index
    %c4_0 = vector.transfer_read %C[%i4, %c0], %zero : memref<6x16xf32>, vector<8xf32>
    %c4_1 = vector.transfer_read %C[%i4, %c8], %zero : memref<6x16xf32>, vector<8xf32>
    // Row 5
    %i5 = arith.constant 5 : index
    %c5_0 = vector.transfer_read %C[%i5, %c0], %zero : memref<6x16xf32>, vector<8xf32>
    %c5_1 = vector.transfer_read %C[%i5, %c8], %zero : memref<6x16xf32>, vector<8xf32>

    // STEP 2: K-loop with explicit FMA unrolling
    %results:12 = scf.for %k = %c0 to %c256 step %c1
        iter_args(%acc0_0 = %c0_0, %acc0_1 = %c0_1,
                  %acc1_0 = %c1_0, %acc1_1 = %c1_1,
                  %acc2_0 = %c2_0, %acc2_1 = %c2_1,
                  %acc3_0 = %c3_0, %acc3_1 = %c3_1,
                  %acc4_0 = %c4_0, %acc4_1 = %c4_1,
                  %acc5_0 = %c5_0, %acc5_1 = %c5_1)
        -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
            vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
            vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {

      // Load B[k, 0:8] and B[k, 8:16]
      %b0 = vector.transfer_read %B[%k, %c0], %zero : memref<256x16xf32>, vector<8xf32>
      %b1 = vector.transfer_read %B[%k, %c8], %zero : memref<256x16xf32>, vector<8xf32>

      // Row 0: load A[0,k], broadcast, FMA
      %a0_scalar = memref.load %A[%c0, %k] : memref<6x256xf32>
      %a0 = vector.broadcast %a0_scalar : f32 to vector<8xf32>
      %new0_0 = vector.fma %a0, %b0, %acc0_0 : vector<8xf32>
      %new0_1 = vector.fma %a0, %b1, %acc0_1 : vector<8xf32>

      // Row 1
      %a1_scalar = memref.load %A[%i1, %k] : memref<6x256xf32>
      %a1 = vector.broadcast %a1_scalar : f32 to vector<8xf32>
      %new1_0 = vector.fma %a1, %b0, %acc1_0 : vector<8xf32>
      %new1_1 = vector.fma %a1, %b1, %acc1_1 : vector<8xf32>

      // Row 2
      %a2_scalar = memref.load %A[%i2, %k] : memref<6x256xf32>
      %a2 = vector.broadcast %a2_scalar : f32 to vector<8xf32>
      %new2_0 = vector.fma %a2, %b0, %acc2_0 : vector<8xf32>
      %new2_1 = vector.fma %a2, %b1, %acc2_1 : vector<8xf32>

      // Row 3
      %a3_scalar = memref.load %A[%i3, %k] : memref<6x256xf32>
      %a3 = vector.broadcast %a3_scalar : f32 to vector<8xf32>
      %new3_0 = vector.fma %a3, %b0, %acc3_0 : vector<8xf32>
      %new3_1 = vector.fma %a3, %b1, %acc3_1 : vector<8xf32>

      // Row 4
      %a4_scalar = memref.load %A[%i4, %k] : memref<6x256xf32>
      %a4 = vector.broadcast %a4_scalar : f32 to vector<8xf32>
      %new4_0 = vector.fma %a4, %b0, %acc4_0 : vector<8xf32>
      %new4_1 = vector.fma %a4, %b1, %acc4_1 : vector<8xf32>

      // Row 5
      %a5_scalar = memref.load %A[%i5, %k] : memref<6x256xf32>
      %a5 = vector.broadcast %a5_scalar : f32 to vector<8xf32>
      %new5_0 = vector.fma %a5, %b0, %acc5_0 : vector<8xf32>
      %new5_1 = vector.fma %a5, %b1, %acc5_1 : vector<8xf32>

      scf.yield %new0_0, %new0_1, %new1_0, %new1_1,
                %new2_0, %new2_1, %new3_0, %new3_1,
                %new4_0, %new4_1, %new5_0, %new5_1
          : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
            vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>,
            vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
    }

    // STEP 3: Store C back
    vector.transfer_write %results#0, %C[%c0, %c0] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#1, %C[%c0, %c8] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#2, %C[%i1, %c0] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#3, %C[%i1, %c8] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#4, %C[%i2, %c0] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#5, %C[%i2, %c8] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#6, %C[%i3, %c0] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#7, %C[%i3, %c8] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#8, %C[%i4, %c0] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#9, %C[%i4, %c8] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#10, %C[%i5, %c0] : vector<8xf32>, memref<6x16xf32>
    vector.transfer_write %results#11, %C[%i5, %c8] : vector<8xf32>, memref<6x16xf32>

    return
  }
}

