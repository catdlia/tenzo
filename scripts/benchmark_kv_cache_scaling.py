#!/usr/bin/env python3
"""
benchmark_kv_cache_scaling.py
Comprehensive Long-Context Benchmark for Tenzo KV-Cache Architectures:
- FP32 Standard KV-Cache
- INT8 Fused KV-Cache (4x compression)
- TL1 Fused (1.58-bit Ternary) KV-Cache (14.2x compression)

Measures:
- Memory footprint (MB) across sequence lengths (64, 256, 512, 1024, 2048, 4096, 8192)
- TTFT (Time To First Token) scaling
- Decode Throughput (tok/sec)
- Semantic Output Comparison
"""

import os
import sys
import time
import subprocess
import json

CONTEXT_LENGTHS = [32, 128, 256, 512, 1024]
KV_MODES = ["fp32", "int8_fused", "tl1_fused"]
PROMPTS = {
    "short": "In computer science, a compiler translates source code written in a high-level programming language into",
    "medium": (
        "Artificial intelligence and compilers are two foundational pillars of modern computer systems. "
        "A compiler optimizes human-readable programs into machine instructions that execute on silicon hardware. "
        "Large language models, such as 1.58-bit ternary architectures like BitNet, drastically reduce the computational "
        "and memory requirements of deep neural networks. In this technical report, we analyze the architectural impact of "
        "in-register KV-cache quantization and SIMD vectorized element-wise lookup tables. The key findings indicate that"
    ),
    "long": (
        "Artificial intelligence and compilers are two foundational pillars of modern computer systems. "
        "A compiler optimizes human-readable programs into machine instructions that execute on silicon hardware. "
        "Large language models, such as 1.58-bit ternary architectures like BitNet, drastically reduce the computational "
        "and memory requirements of deep neural networks. In this technical report, we analyze the architectural impact of "
        "in-register KV-cache quantization and SIMD vectorized element-wise lookup tables. "
        "Traditionally, the autoregressive generation loop is strictly memory-bandwidth bound. Every decoded token requires "
        "streaming all model weights from DRAM into CPU cache, alongside retrieving the past key-value representations for every "
        "transformer attention head. As the context window expands from hundreds to thousands of tokens, the KV-cache memory "
        "footprint grows linearly, saturating memory channels and causing severe cache thrashing. "
        "By applying extreme quantization to both the weight tensors and the dynamic KV-cache, Tenzo achieves remarkable memory efficiency. "
        "Specifically, the ternary 1.58-bit KV-cache quantization packs four channels into a single byte using custom AVX2 vector permutations. "
        "In our empirical evaluation, we demonstrate that"
    )
}

def calculate_kv_memory_mb(num_layers=30, num_kv_heads=5, head_dim=128, seq_len=8192, kv_mode="int8_fused"):
    total_elements = num_layers * seq_len * num_kv_heads * head_dim
    total_scales = num_layers * seq_len * num_kv_heads
    if kv_mode == "fp32":
        # 2 tensors (K and V) * 4 bytes
        bytes_total = 2 * total_elements * 4
    elif kv_mode == "int8_fused":
        # 2 tensors * 1 byte + 2 scales * 4 bytes
        bytes_total = 2 * total_elements * 1 + 2 * total_scales * 4
    elif kv_mode == "tl1_fused":
        # 2 tensors * 0.25 bytes + 2 scales * 4 bytes
        bytes_total = 2 * (total_elements // 4) + 2 * total_scales * 4
    else:
        bytes_total = 0
    return bytes_total / (1024 * 1024)

def run_benchmark_cpp(prompt, kv_mode, max_tokens=30):
    cmd = [
        "docker", "compose", "run", "--rm",
        "-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread",
        "dev",
        "/app/cmake-build-debug/tenzo_basic_inference",
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", "0.7",
        "-m", "/app/tenzo-frontend/export_output",
        "--kv-quant", kv_mode
    ]
    start_t = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_t

    stdout = res.stdout
    ttft = 0.0
    decode_speed = 0.0
    output_text = ""

    for line in stdout.splitlines():
        if "Time To First Token:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    ttft = float(parts[1].replace("ms", "").strip().split()[0])
                except:
                    pass
        elif "Decode Speed:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    decode_speed = float(parts[1].replace("tok/sec", "").strip().split()[0])
                except:
                    pass
        elif "[Tenzo C++ Engine] Output Stream:" in line:
            output_text = line.replace("[Tenzo C++ Engine] Output Stream:", "").strip()

    return {
        "kv_mode": kv_mode,
        "ttft_ms": ttft,
        "decode_speed": decode_speed,
        "output_text": output_text,
        "raw_output": stdout
    }

def main():
    print("=" * 80)
    print("📊 TENZO KV-CACHE SCALING & BENCHMARK REPORT (PHASE E)")
    print("=" * 80)

    # 1. Theoretical Memory Footprint Table
    print("\n📦 1. KV-Cache Memory Footprint Scaling (30 Layers, 5 KV-Heads, Head Dim 128):")
    print("-" * 75)
    print(f"{'Context Length':<16} | {'FP32 Standard':<15} | {'INT8 Fused':<15} | {'TL1 Fused (1.58b)':<18} | {'Compression':<10}")
    print("-" * 75)
    for seq in [128, 512, 1024, 2048, 4096, 8192, 16384]:
        m_fp32 = calculate_kv_memory_mb(seq_len=seq, kv_mode="fp32")
        m_i8 = calculate_kv_memory_mb(seq_len=seq, kv_mode="int8_fused")
        m_tl1 = calculate_kv_memory_mb(seq_len=seq, kv_mode="tl1_fused")
        comp = f"{m_fp32 / m_tl1:.1f}x"
        print(f"{seq:<16} | {m_fp32:>11.2f} MB | {m_i8:>11.2f} MB | {m_tl1:>14.2f} MB | {comp:>10}")
    print("-" * 75)

    # 2. Benchmark on Different Prompt Lengths
    results = {}
    print("\n🚀 2. Empirical Benchmark across KV-Cache Quantization Modes:")
    
    test_cases = [
        ("Short Prompt (18 tokens)", PROMPTS["short"]),
        ("Medium Context (80 tokens)", PROMPTS["medium"]),
        ("Long Context (200 tokens)", PROMPTS["long"]),
    ]

    for title, prompt in test_cases:
        print(f"\n--- {title} ---")
        for mode in ["fp32", "int8_fused", "tl1_fused"]:
            print(f"  • Running KV Mode: {mode:<12} ... ", end="", flush=True)
            res = run_benchmark_cpp(prompt, mode, max_tokens=30)
            results[(title, mode)] = res
            print(f"TTFT: {res['ttft_ms']:>6.1f} ms | Decode: {res['decode_speed']:>5.2f} tok/s")
            print(f"    Sample: \"{res['output_text'][:110]}...\"")

    # 3. Summary Matrix
    print("\n" + "=" * 80)
    print("📈 SUMMARY TABLE: TTFT & Decode Speed by Context Length")
    print("=" * 80)
    print(f"{'Context Case':<26} | {'KV Mode':<12} | {'TTFT (ms)':<10} | {'Decode (tok/s)':<15} | {'KV RAM (8K)':<12}")
    print("-" * 80)
    for (title, mode), res in results.items():
        kv_ram = f"{calculate_kv_memory_mb(seq_len=8192, kv_mode=mode):.1f} MB"
        print(f"{title:<26} | {mode:<12} | {res['ttft_ms']:>9.1f} | {res['decode_speed']:>13.2f} | {kv_ram:>12}")
    print("=" * 80)

if __name__ == "__main__":
    main()
