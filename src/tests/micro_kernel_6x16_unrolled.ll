; micro_kernel_6x16_unrolled.ll - Hand-written LLVM IR for 6x16 micro-kernel
; This generates optimal AVX2 code with perfect register allocation
; UNROLLED 4x (K-loop) to minimize branch overhead
;
; Function signature: void micro_kernel_6x16(float* A, float* B, float* C)
; A: [6 x 256] row-major
; B: [256 x 16] row-major (packed for micro-kernel)
; C: [6 x 16] row-major

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @micro_kernel_6x16(ptr nocapture readonly %A, ptr nocapture readonly %B, ptr nocapture %C) local_unnamed_addr #0 {
entry:
  ; Load C into 12 YMM registers
  %c0_ptr0 = getelementptr inbounds float, ptr %C, i64 0
  %c0_0 = load <8 x float>, ptr %c0_ptr0, align 32
  %c0_ptr8 = getelementptr inbounds float, ptr %C, i64 8
  %c0_1 = load <8 x float>, ptr %c0_ptr8, align 32

  %c1_ptr0 = getelementptr inbounds float, ptr %C, i64 16
  %c1_0 = load <8 x float>, ptr %c1_ptr0, align 32
  %c1_ptr8 = getelementptr inbounds float, ptr %C, i64 24
  %c1_1 = load <8 x float>, ptr %c1_ptr8, align 32

  %c2_ptr0 = getelementptr inbounds float, ptr %C, i64 32
  %c2_0 = load <8 x float>, ptr %c2_ptr0, align 32
  %c2_ptr8 = getelementptr inbounds float, ptr %C, i64 40
  %c2_1 = load <8 x float>, ptr %c2_ptr8, align 32

  %c3_ptr0 = getelementptr inbounds float, ptr %C, i64 48
  %c3_0 = load <8 x float>, ptr %c3_ptr0, align 32
  %c3_ptr8 = getelementptr inbounds float, ptr %C, i64 56
  %c3_1 = load <8 x float>, ptr %c3_ptr8, align 32

  %c4_ptr0 = getelementptr inbounds float, ptr %C, i64 64
  %c4_0 = load <8 x float>, ptr %c4_ptr0, align 32
  %c4_ptr8 = getelementptr inbounds float, ptr %C, i64 72
  %c4_1 = load <8 x float>, ptr %c4_ptr8, align 32

  %c5_ptr0 = getelementptr inbounds float, ptr %C, i64 80
  %c5_0 = load <8 x float>, ptr %c5_ptr0, align 32
  %c5_ptr8 = getelementptr inbounds float, ptr %C, i64 88
  %c5_1 = load <8 x float>, ptr %c5_ptr8, align 32

  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %k_next, %loop ]
  ; Accumulators
  %acc0_0 = phi <8 x float> [ %c0_0, %entry ], [ %it3_new0_0, %loop ]
  %acc0_1 = phi <8 x float> [ %c0_1, %entry ], [ %it3_new0_1, %loop ]
  %acc1_0 = phi <8 x float> [ %c1_0, %entry ], [ %it3_new1_0, %loop ]
  %acc1_1 = phi <8 x float> [ %c1_1, %entry ], [ %it3_new1_1, %loop ]
  %acc2_0 = phi <8 x float> [ %c2_0, %entry ], [ %it3_new2_0, %loop ]
  %acc2_1 = phi <8 x float> [ %c2_1, %entry ], [ %it3_new2_1, %loop ]
  %acc3_0 = phi <8 x float> [ %c3_0, %entry ], [ %it3_new3_0, %loop ]
  %acc3_1 = phi <8 x float> [ %c3_1, %entry ], [ %it3_new3_1, %loop ]
  %acc4_0 = phi <8 x float> [ %c4_0, %entry ], [ %it3_new4_0, %loop ]
  %acc4_1 = phi <8 x float> [ %c4_1, %entry ], [ %it3_new4_1, %loop ]
  %acc5_0 = phi <8 x float> [ %c5_0, %entry ], [ %it3_new5_0, %loop ]
  %acc5_1 = phi <8 x float> [ %c5_1, %entry ], [ %it3_new5_1, %loop ]

  ; ==========================================================
  ; ITERATION 0
  ; ==========================================================
  ; Load B[k, 0:16]
  %it0_b_row_off = mul nuw nsw i64 %k, 16
  %it0_b_ptr0 = getelementptr inbounds float, ptr %B, i64 %it0_b_row_off
  %it0_b0 = load <8 x float>, ptr %it0_b_ptr0, align 32
  %it0_b_off8 = add nuw nsw i64 %it0_b_row_off, 8
  %it0_b_ptr8 = getelementptr inbounds float, ptr %B, i64 %it0_b_off8
  %it0_b1 = load <8 x float>, ptr %it0_b_ptr8, align 32

  ; Row 0: A[0, k]
  %it0_a0_ptr = getelementptr inbounds float, ptr %A, i64 %k
  %it0_a0_scalar = load float, ptr %it0_a0_ptr, align 4
  %it0_a0_vec_tmp = insertelement <8 x float> poison, float %it0_a0_scalar, i64 0
  %it0_a0_vec = shufflevector <8 x float> %it0_a0_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it0_new0_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a0_vec, <8 x float> %it0_b0, <8 x float> %acc0_0)
  %it0_new0_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a0_vec, <8 x float> %it0_b1, <8 x float> %acc0_1)

  ; Row 1: A[1, k] (offset = 256 + k)
  %it0_a1_off = add nuw nsw i64 %k, 256
  %it0_a1_ptr = getelementptr inbounds float, ptr %A, i64 %it0_a1_off
  %it0_a1_scalar = load float, ptr %it0_a1_ptr, align 4
  %it0_a1_vec_tmp = insertelement <8 x float> poison, float %it0_a1_scalar, i64 0
  %it0_a1_vec = shufflevector <8 x float> %it0_a1_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it0_new1_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a1_vec, <8 x float> %it0_b0, <8 x float> %acc1_0)
  %it0_new1_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a1_vec, <8 x float> %it0_b1, <8 x float> %acc1_1)

  ; Row 2: A[2, k] (offset = 512 + k)
  %it0_a2_off = add nuw nsw i64 %k, 512
  %it0_a2_ptr = getelementptr inbounds float, ptr %A, i64 %it0_a2_off
  %it0_a2_scalar = load float, ptr %it0_a2_ptr, align 4
  %it0_a2_vec_tmp = insertelement <8 x float> poison, float %it0_a2_scalar, i64 0
  %it0_a2_vec = shufflevector <8 x float> %it0_a2_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it0_new2_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a2_vec, <8 x float> %it0_b0, <8 x float> %acc2_0)
  %it0_new2_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a2_vec, <8 x float> %it0_b1, <8 x float> %acc2_1)

  ; Row 3: A[3, k] (offset = 768 + k)
  %it0_a3_off = add nuw nsw i64 %k, 768
  %it0_a3_ptr = getelementptr inbounds float, ptr %A, i64 %it0_a3_off
  %it0_a3_scalar = load float, ptr %it0_a3_ptr, align 4
  %it0_a3_vec_tmp = insertelement <8 x float> poison, float %it0_a3_scalar, i64 0
  %it0_a3_vec = shufflevector <8 x float> %it0_a3_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it0_new3_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a3_vec, <8 x float> %it0_b0, <8 x float> %acc3_0)
  %it0_new3_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a3_vec, <8 x float> %it0_b1, <8 x float> %acc3_1)

  ; Row 4: A[4, k] (offset = 1024 + k)
  %it0_a4_off = add nuw nsw i64 %k, 1024
  %it0_a4_ptr = getelementptr inbounds float, ptr %A, i64 %it0_a4_off
  %it0_a4_scalar = load float, ptr %it0_a4_ptr, align 4
  %it0_a4_vec_tmp = insertelement <8 x float> poison, float %it0_a4_scalar, i64 0
  %it0_a4_vec = shufflevector <8 x float> %it0_a4_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it0_new4_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a4_vec, <8 x float> %it0_b0, <8 x float> %acc4_0)
  %it0_new4_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a4_vec, <8 x float> %it0_b1, <8 x float> %acc4_1)

  ; Row 5: A[5, k] (offset = 1280 + k)
  %it0_a5_off = add nuw nsw i64 %k, 1280
  %it0_a5_ptr = getelementptr inbounds float, ptr %A, i64 %it0_a5_off
  %it0_a5_scalar = load float, ptr %it0_a5_ptr, align 4
  %it0_a5_vec_tmp = insertelement <8 x float> poison, float %it0_a5_scalar, i64 0
  %it0_a5_vec = shufflevector <8 x float> %it0_a5_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it0_new5_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a5_vec, <8 x float> %it0_b0, <8 x float> %acc5_0)
  %it0_new5_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it0_a5_vec, <8 x float> %it0_b1, <8 x float> %acc5_1)

  ; ==========================================================
  ; ITERATION 1 (offset +1)
  ; ==========================================================
  %k1 = add nuw nsw i64 %k, 1

  ; Load B[k+1, 0:16]
  %it1_b_row_off = mul nuw nsw i64 %k1, 16
  %it1_b_ptr0 = getelementptr inbounds float, ptr %B, i64 %it1_b_row_off
  %it1_b0 = load <8 x float>, ptr %it1_b_ptr0, align 32
  %it1_b_off8 = add nuw nsw i64 %it1_b_row_off, 8
  %it1_b_ptr8 = getelementptr inbounds float, ptr %B, i64 %it1_b_off8
  %it1_b1 = load <8 x float>, ptr %it1_b_ptr8, align 32

  ; Row 0
  %it1_a0_ptr = getelementptr inbounds float, ptr %A, i64 %k1
  %it1_a0_scalar = load float, ptr %it1_a0_ptr, align 4
  %it1_a0_vec_tmp = insertelement <8 x float> poison, float %it1_a0_scalar, i64 0
  %it1_a0_vec = shufflevector <8 x float> %it1_a0_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it1_new0_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a0_vec, <8 x float> %it1_b0, <8 x float> %it0_new0_0)
  %it1_new0_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a0_vec, <8 x float> %it1_b1, <8 x float> %it0_new0_1)

  ; Row 1
  %it1_a1_off = add nuw nsw i64 %k1, 256
  %it1_a1_ptr = getelementptr inbounds float, ptr %A, i64 %it1_a1_off
  %it1_a1_scalar = load float, ptr %it1_a1_ptr, align 4
  %it1_a1_vec_tmp = insertelement <8 x float> poison, float %it1_a1_scalar, i64 0
  %it1_a1_vec = shufflevector <8 x float> %it1_a1_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it1_new1_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a1_vec, <8 x float> %it1_b0, <8 x float> %it0_new1_0)
  %it1_new1_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a1_vec, <8 x float> %it1_b1, <8 x float> %it0_new1_1)

  ; Row 2
  %it1_a2_off = add nuw nsw i64 %k1, 512
  %it1_a2_ptr = getelementptr inbounds float, ptr %A, i64 %it1_a2_off
  %it1_a2_scalar = load float, ptr %it1_a2_ptr, align 4
  %it1_a2_vec_tmp = insertelement <8 x float> poison, float %it1_a2_scalar, i64 0
  %it1_a2_vec = shufflevector <8 x float> %it1_a2_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it1_new2_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a2_vec, <8 x float> %it1_b0, <8 x float> %it0_new2_0)
  %it1_new2_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a2_vec, <8 x float> %it1_b1, <8 x float> %it0_new2_1)

  ; Row 3
  %it1_a3_off = add nuw nsw i64 %k1, 768
  %it1_a3_ptr = getelementptr inbounds float, ptr %A, i64 %it1_a3_off
  %it1_a3_scalar = load float, ptr %it1_a3_ptr, align 4
  %it1_a3_vec_tmp = insertelement <8 x float> poison, float %it1_a3_scalar, i64 0
  %it1_a3_vec = shufflevector <8 x float> %it1_a3_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it1_new3_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a3_vec, <8 x float> %it1_b0, <8 x float> %it0_new3_0)
  %it1_new3_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a3_vec, <8 x float> %it1_b1, <8 x float> %it0_new3_1)

  ; Row 4
  %it1_a4_off = add nuw nsw i64 %k1, 1024
  %it1_a4_ptr = getelementptr inbounds float, ptr %A, i64 %it1_a4_off
  %it1_a4_scalar = load float, ptr %it1_a4_ptr, align 4
  %it1_a4_vec_tmp = insertelement <8 x float> poison, float %it1_a4_scalar, i64 0
  %it1_a4_vec = shufflevector <8 x float> %it1_a4_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it1_new4_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a4_vec, <8 x float> %it1_b0, <8 x float> %it0_new4_0)
  %it1_new4_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a4_vec, <8 x float> %it1_b1, <8 x float> %it0_new4_1)

  ; Row 5
  %it1_a5_off = add nuw nsw i64 %k1, 1280
  %it1_a5_ptr = getelementptr inbounds float, ptr %A, i64 %it1_a5_off
  %it1_a5_scalar = load float, ptr %it1_a5_ptr, align 4
  %it1_a5_vec_tmp = insertelement <8 x float> poison, float %it1_a5_scalar, i64 0
  %it1_a5_vec = shufflevector <8 x float> %it1_a5_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it1_new5_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a5_vec, <8 x float> %it1_b0, <8 x float> %it0_new5_0)
  %it1_new5_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it1_a5_vec, <8 x float> %it1_b1, <8 x float> %it0_new5_1)

  ; ==========================================================
  ; ITERATION 2 (offset +2)
  ; ==========================================================
  %k2 = add nuw nsw i64 %k, 2

  ; Load B[k+2, 0:16]
  %it2_b_row_off = mul nuw nsw i64 %k2, 16
  %it2_b_ptr0 = getelementptr inbounds float, ptr %B, i64 %it2_b_row_off
  %it2_b0 = load <8 x float>, ptr %it2_b_ptr0, align 32
  %it2_b_off8 = add nuw nsw i64 %it2_b_row_off, 8
  %it2_b_ptr8 = getelementptr inbounds float, ptr %B, i64 %it2_b_off8
  %it2_b1 = load <8 x float>, ptr %it2_b_ptr8, align 32

  ; Row 0
  %it2_a0_ptr = getelementptr inbounds float, ptr %A, i64 %k2
  %it2_a0_scalar = load float, ptr %it2_a0_ptr, align 4
  %it2_a0_vec_tmp = insertelement <8 x float> poison, float %it2_a0_scalar, i64 0
  %it2_a0_vec = shufflevector <8 x float> %it2_a0_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it2_new0_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a0_vec, <8 x float> %it2_b0, <8 x float> %it1_new0_0)
  %it2_new0_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a0_vec, <8 x float> %it2_b1, <8 x float> %it1_new0_1)

  ; Row 1
  %it2_a1_off = add nuw nsw i64 %k2, 256
  %it2_a1_ptr = getelementptr inbounds float, ptr %A, i64 %it2_a1_off
  %it2_a1_scalar = load float, ptr %it2_a1_ptr, align 4
  %it2_a1_vec_tmp = insertelement <8 x float> poison, float %it2_a1_scalar, i64 0
  %it2_a1_vec = shufflevector <8 x float> %it2_a1_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it2_new1_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a1_vec, <8 x float> %it2_b0, <8 x float> %it1_new1_0)
  %it2_new1_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a1_vec, <8 x float> %it2_b1, <8 x float> %it1_new1_1)

  ; Row 2
  %it2_a2_off = add nuw nsw i64 %k2, 512
  %it2_a2_ptr = getelementptr inbounds float, ptr %A, i64 %it2_a2_off
  %it2_a2_scalar = load float, ptr %it2_a2_ptr, align 4
  %it2_a2_vec_tmp = insertelement <8 x float> poison, float %it2_a2_scalar, i64 0
  %it2_a2_vec = shufflevector <8 x float> %it2_a2_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it2_new2_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a2_vec, <8 x float> %it2_b0, <8 x float> %it1_new2_0)
  %it2_new2_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a2_vec, <8 x float> %it2_b1, <8 x float> %it1_new2_1)

  ; Row 3
  %it2_a3_off = add nuw nsw i64 %k2, 768
  %it2_a3_ptr = getelementptr inbounds float, ptr %A, i64 %it2_a3_off
  %it2_a3_scalar = load float, ptr %it2_a3_ptr, align 4
  %it2_a3_vec_tmp = insertelement <8 x float> poison, float %it2_a3_scalar, i64 0
  %it2_a3_vec = shufflevector <8 x float> %it2_a3_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it2_new3_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a3_vec, <8 x float> %it2_b0, <8 x float> %it1_new3_0)
  %it2_new3_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a3_vec, <8 x float> %it2_b1, <8 x float> %it1_new3_1)

  ; Row 4
  %it2_a4_off = add nuw nsw i64 %k2, 1024
  %it2_a4_ptr = getelementptr inbounds float, ptr %A, i64 %it2_a4_off
  %it2_a4_scalar = load float, ptr %it2_a4_ptr, align 4
  %it2_a4_vec_tmp = insertelement <8 x float> poison, float %it2_a4_scalar, i64 0
  %it2_a4_vec = shufflevector <8 x float> %it2_a4_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it2_new4_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a4_vec, <8 x float> %it2_b0, <8 x float> %it1_new4_0)
  %it2_new4_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a4_vec, <8 x float> %it2_b1, <8 x float> %it1_new4_1)

  ; Row 5
  %it2_a5_off = add nuw nsw i64 %k2, 1280
  %it2_a5_ptr = getelementptr inbounds float, ptr %A, i64 %it2_a5_off
  %it2_a5_scalar = load float, ptr %it2_a5_ptr, align 4
  %it2_a5_vec_tmp = insertelement <8 x float> poison, float %it2_a5_scalar, i64 0
  %it2_a5_vec = shufflevector <8 x float> %it2_a5_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it2_new5_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a5_vec, <8 x float> %it2_b0, <8 x float> %it1_new5_0)
  %it2_new5_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it2_a5_vec, <8 x float> %it2_b1, <8 x float> %it1_new5_1)

  ; ==========================================================
  ; ITERATION 3 (offset +3)
  ; ==========================================================
  %k3 = add nuw nsw i64 %k, 3

  ; Load B[k+3, 0:16]
  %it3_b_row_off = mul nuw nsw i64 %k3, 16
  %it3_b_ptr0 = getelementptr inbounds float, ptr %B, i64 %it3_b_row_off
  %it3_b0 = load <8 x float>, ptr %it3_b_ptr0, align 32
  %it3_b_off8 = add nuw nsw i64 %it3_b_row_off, 8
  %it3_b_ptr8 = getelementptr inbounds float, ptr %B, i64 %it3_b_off8
  %it3_b1 = load <8 x float>, ptr %it3_b_ptr8, align 32

  ; Row 0
  %it3_a0_ptr = getelementptr inbounds float, ptr %A, i64 %k3
  %it3_a0_scalar = load float, ptr %it3_a0_ptr, align 4
  %it3_a0_vec_tmp = insertelement <8 x float> poison, float %it3_a0_scalar, i64 0
  %it3_a0_vec = shufflevector <8 x float> %it3_a0_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it3_new0_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a0_vec, <8 x float> %it3_b0, <8 x float> %it2_new0_0)
  %it3_new0_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a0_vec, <8 x float> %it3_b1, <8 x float> %it2_new0_1)

  ; Row 1
  %it3_a1_off = add nuw nsw i64 %k3, 256
  %it3_a1_ptr = getelementptr inbounds float, ptr %A, i64 %it3_a1_off
  %it3_a1_scalar = load float, ptr %it3_a1_ptr, align 4
  %it3_a1_vec_tmp = insertelement <8 x float> poison, float %it3_a1_scalar, i64 0
  %it3_a1_vec = shufflevector <8 x float> %it3_a1_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it3_new1_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a1_vec, <8 x float> %it3_b0, <8 x float> %it2_new1_0)
  %it3_new1_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a1_vec, <8 x float> %it3_b1, <8 x float> %it2_new1_1)

  ; Row 2
  %it3_a2_off = add nuw nsw i64 %k3, 512
  %it3_a2_ptr = getelementptr inbounds float, ptr %A, i64 %it3_a2_off
  %it3_a2_scalar = load float, ptr %it3_a2_ptr, align 4
  %it3_a2_vec_tmp = insertelement <8 x float> poison, float %it3_a2_scalar, i64 0
  %it3_a2_vec = shufflevector <8 x float> %it3_a2_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it3_new2_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a2_vec, <8 x float> %it3_b0, <8 x float> %it2_new2_0)
  %it3_new2_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a2_vec, <8 x float> %it3_b1, <8 x float> %it2_new2_1)

  ; Row 3
  %it3_a3_off = add nuw nsw i64 %k3, 768
  %it3_a3_ptr = getelementptr inbounds float, ptr %A, i64 %it3_a3_off
  %it3_a3_scalar = load float, ptr %it3_a3_ptr, align 4
  %it3_a3_vec_tmp = insertelement <8 x float> poison, float %it3_a3_scalar, i64 0
  %it3_a3_vec = shufflevector <8 x float> %it3_a3_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it3_new3_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a3_vec, <8 x float> %it3_b0, <8 x float> %it2_new3_0)
  %it3_new3_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a3_vec, <8 x float> %it3_b1, <8 x float> %it2_new3_1)

  ; Row 4
  %it3_a4_off = add nuw nsw i64 %k3, 1024
  %it3_a4_ptr = getelementptr inbounds float, ptr %A, i64 %it3_a4_off
  %it3_a4_scalar = load float, ptr %it3_a4_ptr, align 4
  %it3_a4_vec_tmp = insertelement <8 x float> poison, float %it3_a4_scalar, i64 0
  %it3_a4_vec = shufflevector <8 x float> %it3_a4_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it3_new4_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a4_vec, <8 x float> %it3_b0, <8 x float> %it2_new4_0)
  %it3_new4_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a4_vec, <8 x float> %it3_b1, <8 x float> %it2_new4_1)

  ; Row 5
  %it3_a5_off = add nuw nsw i64 %k3, 1280
  %it3_a5_ptr = getelementptr inbounds float, ptr %A, i64 %it3_a5_off
  %it3_a5_scalar = load float, ptr %it3_a5_ptr, align 4
  %it3_a5_vec_tmp = insertelement <8 x float> poison, float %it3_a5_scalar, i64 0
  %it3_a5_vec = shufflevector <8 x float> %it3_a5_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %it3_new5_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a5_vec, <8 x float> %it3_b0, <8 x float> %it2_new5_0)
  %it3_new5_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %it3_a5_vec, <8 x float> %it3_b1, <8 x float> %it2_new5_1)

  ; Increment k by 4
  %k_next = add nuw nsw i64 %k, 4
  %cmp = icmp eq i64 %k_next, 256
  br i1 %cmp, label %exit, label %loop

exit:
  ; Store C back (use iteration 3 results)
  store <8 x float> %it3_new0_0, ptr %c0_ptr0, align 32
  store <8 x float> %it3_new0_1, ptr %c0_ptr8, align 32
  store <8 x float> %it3_new1_0, ptr %c1_ptr0, align 32
  store <8 x float> %it3_new1_1, ptr %c1_ptr8, align 32
  store <8 x float> %it3_new2_0, ptr %c2_ptr0, align 32
  store <8 x float> %it3_new2_1, ptr %c2_ptr8, align 32
  store <8 x float> %it3_new3_0, ptr %c3_ptr0, align 32
  store <8 x float> %it3_new3_1, ptr %c3_ptr8, align 32
  store <8 x float> %it3_new4_0, ptr %c4_ptr0, align 32
  store <8 x float> %it3_new4_1, ptr %c4_ptr8, align 32
  store <8 x float> %it3_new5_0, ptr %c5_ptr0, align 32
  store <8 x float> %it3_new5_1, ptr %c5_ptr8, align 32

  ret void
}

declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #1

attributes #0 = { nounwind "target-cpu"="native" "target-features"="+avx2,+fma" }
attributes #1 = { nounwind readnone speculatable willreturn }

