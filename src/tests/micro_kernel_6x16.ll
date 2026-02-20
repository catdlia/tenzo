; micro_kernel_6x16.ll - Hand-written LLVM IR for 6x16 micro-kernel
; This generates optimal AVX2 code with perfect register allocation
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
  %acc0_0 = phi <8 x float> [ %c0_0, %entry ], [ %new0_0, %loop ]
  %acc0_1 = phi <8 x float> [ %c0_1, %entry ], [ %new0_1, %loop ]
  %acc1_0 = phi <8 x float> [ %c1_0, %entry ], [ %new1_0, %loop ]
  %acc1_1 = phi <8 x float> [ %c1_1, %entry ], [ %new1_1, %loop ]
  %acc2_0 = phi <8 x float> [ %c2_0, %entry ], [ %new2_0, %loop ]
  %acc2_1 = phi <8 x float> [ %c2_1, %entry ], [ %new2_1, %loop ]
  %acc3_0 = phi <8 x float> [ %c3_0, %entry ], [ %new3_0, %loop ]
  %acc3_1 = phi <8 x float> [ %c3_1, %entry ], [ %new3_1, %loop ]
  %acc4_0 = phi <8 x float> [ %c4_0, %entry ], [ %new4_0, %loop ]
  %acc4_1 = phi <8 x float> [ %c4_1, %entry ], [ %new4_1, %loop ]
  %acc5_0 = phi <8 x float> [ %c5_0, %entry ], [ %new5_0, %loop ]
  %acc5_1 = phi <8 x float> [ %c5_1, %entry ], [ %new5_1, %loop ]

  ; Load B[k, 0:16]
  %b_row_off = mul nuw nsw i64 %k, 16
  %b_ptr0 = getelementptr inbounds float, ptr %B, i64 %b_row_off
  %b0 = load <8 x float>, ptr %b_ptr0, align 32
  %b_off8 = add nuw nsw i64 %b_row_off, 8
  %b_ptr8 = getelementptr inbounds float, ptr %B, i64 %b_off8
  %b1 = load <8 x float>, ptr %b_ptr8, align 32

  ; Row 0: A[0, k]
  %a0_ptr = getelementptr inbounds float, ptr %A, i64 %k
  %a0_scalar = load float, ptr %a0_ptr, align 4
  %a0_vec_tmp = insertelement <8 x float> poison, float %a0_scalar, i64 0
  %a0_vec = shufflevector <8 x float> %a0_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %new0_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a0_vec, <8 x float> %b0, <8 x float> %acc0_0)
  %new0_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a0_vec, <8 x float> %b1, <8 x float> %acc0_1)

  ; Row 1: A[1, k] (offset = 256 + k)
  %a1_off = add nuw nsw i64 %k, 256
  %a1_ptr = getelementptr inbounds float, ptr %A, i64 %a1_off
  %a1_scalar = load float, ptr %a1_ptr, align 4
  %a1_vec_tmp = insertelement <8 x float> poison, float %a1_scalar, i64 0
  %a1_vec = shufflevector <8 x float> %a1_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %new1_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a1_vec, <8 x float> %b0, <8 x float> %acc1_0)
  %new1_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a1_vec, <8 x float> %b1, <8 x float> %acc1_1)

  ; Row 2: A[2, k] (offset = 512 + k)
  %a2_off = add nuw nsw i64 %k, 512
  %a2_ptr = getelementptr inbounds float, ptr %A, i64 %a2_off
  %a2_scalar = load float, ptr %a2_ptr, align 4
  %a2_vec_tmp = insertelement <8 x float> poison, float %a2_scalar, i64 0
  %a2_vec = shufflevector <8 x float> %a2_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %new2_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a2_vec, <8 x float> %b0, <8 x float> %acc2_0)
  %new2_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a2_vec, <8 x float> %b1, <8 x float> %acc2_1)

  ; Row 3: A[3, k] (offset = 768 + k)
  %a3_off = add nuw nsw i64 %k, 768
  %a3_ptr = getelementptr inbounds float, ptr %A, i64 %a3_off
  %a3_scalar = load float, ptr %a3_ptr, align 4
  %a3_vec_tmp = insertelement <8 x float> poison, float %a3_scalar, i64 0
  %a3_vec = shufflevector <8 x float> %a3_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %new3_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a3_vec, <8 x float> %b0, <8 x float> %acc3_0)
  %new3_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a3_vec, <8 x float> %b1, <8 x float> %acc3_1)

  ; Row 4: A[4, k] (offset = 1024 + k)
  %a4_off = add nuw nsw i64 %k, 1024
  %a4_ptr = getelementptr inbounds float, ptr %A, i64 %a4_off
  %a4_scalar = load float, ptr %a4_ptr, align 4
  %a4_vec_tmp = insertelement <8 x float> poison, float %a4_scalar, i64 0
  %a4_vec = shufflevector <8 x float> %a4_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %new4_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a4_vec, <8 x float> %b0, <8 x float> %acc4_0)
  %new4_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a4_vec, <8 x float> %b1, <8 x float> %acc4_1)

  ; Row 5: A[5, k] (offset = 1280 + k)
  %a5_off = add nuw nsw i64 %k, 1280
  %a5_ptr = getelementptr inbounds float, ptr %A, i64 %a5_off
  %a5_scalar = load float, ptr %a5_ptr, align 4
  %a5_vec_tmp = insertelement <8 x float> poison, float %a5_scalar, i64 0
  %a5_vec = shufflevector <8 x float> %a5_vec_tmp, <8 x float> poison, <8 x i32> zeroinitializer
  %new5_0 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a5_vec, <8 x float> %b0, <8 x float> %acc5_0)
  %new5_1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a5_vec, <8 x float> %b1, <8 x float> %acc5_1)

  ; Increment k
  %k_next = add nuw nsw i64 %k, 1
  %cmp = icmp eq i64 %k_next, 256
  br i1 %cmp, label %exit, label %loop

exit:
  ; Store C back
  store <8 x float> %new0_0, ptr %c0_ptr0, align 32
  store <8 x float> %new0_1, ptr %c0_ptr8, align 32
  store <8 x float> %new1_0, ptr %c1_ptr0, align 32
  store <8 x float> %new1_1, ptr %c1_ptr8, align 32
  store <8 x float> %new2_0, ptr %c2_ptr0, align 32
  store <8 x float> %new2_1, ptr %c2_ptr8, align 32
  store <8 x float> %new3_0, ptr %c3_ptr0, align 32
  store <8 x float> %new3_1, ptr %c3_ptr8, align 32
  store <8 x float> %new4_0, ptr %c4_ptr0, align 32
  store <8 x float> %new4_1, ptr %c4_ptr8, align 32
  store <8 x float> %new5_0, ptr %c5_ptr0, align 32
  store <8 x float> %new5_1, ptr %c5_ptr8, align 32

  ret void
}

declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #1

attributes #0 = { nounwind "target-cpu"="native" "target-features"="+avx2,+fma" }
attributes #1 = { nounwind readnone speculatable willreturn }

