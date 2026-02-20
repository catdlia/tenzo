#!/usr/bin/env python3
"""
generate_unrolled_microkernel.py

Generates a FULLY UNROLLED micro-kernel LLVM IR with NO loops.
This should produce 3072 FMA instructions for maximum performance (120+ GFLOPS).

Target: 6x16x256 micro-kernel
- 6 rows
- 16 columns (2 AVX2 vectors of 8 floats)
- 256 K iterations (FULLY UNROLLED)

Total FMAs: 6 * 2 * 256 = 3072
"""

def generate_llvm_ir():
    MR = 6  # rows
    NR_VECS = 2  # number of vectors (16 elements = 2×8)
    K = 256  # depth

    ir = []

    # Header
    ir.append('; Auto-generated FULLY UNROLLED micro-kernel')
    ir.append('; 6x16x256 with ZERO loops - 3072 FMA instructions')
    ir.append('')
    ir.append('target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"')
    ir.append('target triple = "x86_64-unknown-linux-gnu"')
    ir.append('')
    ir.append('define dso_local void @micro_kernel_6x16(ptr nocapture readonly %A, ptr nocapture readonly %B, ptr nocapture %C) local_unnamed_addr #0 {')
    ir.append('entry:')

    # Load C into 12 accumulators
    ir.append('  ; Load C matrix into 12 YMM registers (accumulators)')
    for i in range(MR):
        for j in range(NR_VECS):
            offset = i * NR_VECS * 8 + j * 8
            acc_name = f'%c{i}_{j}_init'
            ptr_name = f'%c{i}_{j}_ptr'
            ir.append(f'  {ptr_name} = getelementptr inbounds float, ptr %C, i64 {offset}')
            ir.append(f'  {acc_name} = load <8 x float>, ptr {ptr_name}, align 32')

    ir.append('')
    ir.append('  ; FULLY UNROLLED K-loop: 256 iterations × 12 FMAs = 3072 FMAs')
    ir.append('')

    # Generate unrolled K-loop
    for k in range(K):
        ir.append(f'  ; ===== K = {k} =====')

        # Load B vectors
        b_offset_0 = k * 16
        b_offset_1 = k * 16 + 8
        ir.append(f'  %b{k}_ptr0 = getelementptr inbounds float, ptr %B, i64 {b_offset_0}')
        ir.append(f'  %b{k}_0 = load <8 x float>, ptr %b{k}_ptr0, align 32')
        ir.append(f'  %b{k}_ptr1 = getelementptr inbounds float, ptr %B, i64 {b_offset_1}')
        ir.append(f'  %b{k}_1 = load <8 x float>, ptr %b{k}_ptr1, align 32')

        # Process each row
        for i in range(MR):
            a_offset = i * K + k

            # Load and broadcast A[i, k]
            ir.append(f'  %a{i}_{k}_ptr = getelementptr inbounds float, ptr %A, i64 {a_offset}')
            ir.append(f'  %a{i}_{k}_scalar = load float, ptr %a{i}_{k}_ptr, align 4')
            ir.append(f'  %a{i}_{k}_tmp = insertelement <8 x float> poison, float %a{i}_{k}_scalar, i64 0')
            ir.append(f'  %a{i}_{k}_vec = shufflevector <8 x float> %a{i}_{k}_tmp, <8 x float> poison, <8 x i32> zeroinitializer')

            # FMA for both vectors
            for j in range(NR_VECS):
                if k == 0:
                    acc_prev = f'%c{i}_{j}_init'
                else:
                    acc_prev = f'%c{i}_{j}_k{k-1}'
                acc_new = f'%c{i}_{j}_k{k}'
                b_vec = f'%b{k}_{j}'
                a_vec = f'%a{i}_{k}_vec'

                ir.append(f'  {acc_new} = tail call <8 x float> @llvm.fma.v8f32(<8 x float> {a_vec}, <8 x float> {b_vec}, <8 x float> {acc_prev})')

        ir.append('')

    # Store results back
    ir.append('  ; Store accumulated results back to C')
    for i in range(MR):
        for j in range(NR_VECS):
            offset = i * NR_VECS * 8 + j * 8
            acc_final = f'%c{i}_{j}_k{K-1}'
            ptr_name = f'%c{i}_{j}_ptr'
            ir.append(f'  store <8 x float> {acc_final}, ptr {ptr_name}, align 32')

    ir.append('')
    ir.append('  ret void')
    ir.append('}')
    ir.append('')
    ir.append('declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #1')
    ir.append('')
    ir.append('attributes #0 = { nounwind "target-cpu"="haswell" "target-features"="+avx2,+fma" }')
    ir.append('attributes #1 = { nounwind readnone speculatable willreturn }')
    ir.append('')

    return '\n'.join(ir)

if __name__ == '__main__':
    import sys

    print("Generating FULLY UNROLLED micro-kernel LLVM IR...")
    print("This will create a VERY LARGE file with 3072 FMA instructions")
    print()

    ir_code = generate_llvm_ir()

    output_file = 'micro_kernel_fully_unrolled.ll'
    if len(sys.argv) > 1:
        output_file = sys.argv[1]

    with open(output_file, 'w') as f:
        f.write(ir_code)

    print(f"✅ Generated: {output_file}")

    # Count lines
    lines = ir_code.count('\n')
    fma_count = ir_code.count('@llvm.fma.v8f32')

    print(f"📊 Statistics:")
    print(f"   Lines: {lines}")
    print(f"   FMA calls: {fma_count}")
    print(f"   Expected FMAs: 3072")
    print()
    print("Next steps:")
    print(f"  1. Compile: llc {output_file} -O3 -mattr=+avx2,+fma -filetype=obj -o micro_kernel_unrolled.o")
    print(f"  2. Check:   objdump -d micro_kernel_unrolled.o | grep -c vfmadd")
    print(f"  3. Use:     Copy to micro_kernel_mlir.o and rebuild micro_bench")

