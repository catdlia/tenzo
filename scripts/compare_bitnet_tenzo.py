#!/usr/bin/env python3
import os
import sys
import time
import re
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

BITNET_BIN = os.path.join(PROJECT_ROOT, "BitNet", "build", "bin", "llama-completion")
BITNET_MODEL = os.path.join(PROJECT_ROOT, "BitNet", "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf")


def run_bitnet_cpp(prompt, n_tokens=50, temp=0.7, threads=4):
    print("=" * 60)
    print("🚀 [1/2] Running Microsoft BitNet.cpp (Reference C++ Engine)...")
    print("=" * 60)

    cmd = [
        BITNET_BIN,
        "-m", BITNET_MODEL,
        "-p", prompt,
        "-n", str(n_tokens),
        "-t", str(threads),
        "--temp", str(temp),
        "-no-cnv"
    ]

    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    total_wall_time = time.perf_counter() - t0

    output = res.stdout + res.stderr
    print(output)

    # Parse metrics from BitNet.cpp
    # prompt eval time =     371.09 ms /    19 tokens (   19.53 ms per token,    51.20 tokens per second)
    # eval time =    3143.39 ms /    49 runs   (   64.15 ms per token,    15.59 tokens per second)
    p_eval_match = re.search(r'prompt eval time =\s+([0-9\.]+) ms /\s+(\d+) tokens.*?([0-9\.]+) tokens per second', output)
    d_eval_match = re.search(r'eval time =\s+([0-9\.]+) ms /\s+(\d+) runs.*?([0-9\.]+) tokens per second', output)

    prompt_tokens = int(p_eval_match.group(2)) if p_eval_match else 0
    ttft_ms = float(p_eval_match.group(1)) if p_eval_match else 0.0
    prefill_speed = float(p_eval_match.group(3)) if p_eval_match else 0.0

    gen_tokens = int(d_eval_match.group(2)) if d_eval_match else n_tokens
    decode_ms = float(d_eval_match.group(1)) if d_eval_match else 0.0
    decode_speed = float(d_eval_match.group(3)) if d_eval_match else (gen_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0)

    return {
        "engine": "Microsoft BitNet.cpp",
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "ttft_ms": ttft_ms,
        "prefill_speed": prefill_speed,
        "decode_ms": decode_ms,
        "decode_speed": decode_speed,
        "total_wall_time": total_wall_time
    }


def run_tenzo(prompt, n_tokens=50, temp=0.7, repetition_penalty=1.15, kv_mode="int8_fused"):
    print("\n" + "=" * 60)
    print("⚡ [2/2] Running Tenzo Native AVX2 Engine...")
    print("=" * 60)

    docker_cmd = [
        "docker", "compose", "run", "--rm",
        "-e", "OMP_PLACES=cores",
        "-e", "OMP_PROC_BIND=spread",
        "dev", "bash", "-c",
        f"pip3 install numpy --break-system-packages -q && python3 /app/scripts/run_generation_fast.py -p \"{prompt}\" -n {n_tokens} -t {temp} -r {repetition_penalty} --kv-mode {kv_mode}"
    ]

    t0 = time.perf_counter()
    res = subprocess.run(docker_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    total_wall_time = time.perf_counter() - t0

    output = res.stdout + res.stderr
    print(output)

    # Parse metrics from Tenzo summary table
    p_tok = re.search(r'Prompt Tokens:\s+(\d+)', output)
    g_tok = re.search(r'Generated Tokens:\s+(\d+)', output)
    ttft = re.search(r'Time To First Token:\s+([0-9\.]+)\s+ms', output)
    d_time = re.search(r'Total Decode Time:\s+([0-9\.]+)\s+ms', output)
    d_spd = re.search(r'Decode Speed:\s+([0-9\.]+)\s+tok/sec', output)

    prompt_tokens = int(p_tok.group(1)) if p_tok else 0
    gen_tokens = int(g_tok.group(1)) if g_tok else n_tokens
    ttft_ms = float(ttft.group(1)) if ttft else 0.0
    decode_ms = float(d_time.group(1)) if d_time else 0.0
    decode_speed = float(d_spd.group(1)) if d_spd else 0.0

    return {
        "engine": f"Tenzo Engine ({kv_mode.upper()})",
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "ttft_ms": ttft_ms,
        "prefill_speed": (prompt_tokens / (ttft_ms / 1000.0)) if ttft_ms > 0 else 0,
        "decode_ms": decode_ms,
        "decode_speed": decode_speed,
        "total_wall_time": total_wall_time
    }


def print_comparison(b_res, t_res, prompt):
    print("\n" + "═" * 70)
    print("📊 BENCHMARK COMPARISON: Microsoft BitNet.cpp vs Tenzo Native Engine")
    print("═" * 70)
    print(f"Prompt: \"{prompt}\"\n")

    print(f"{'Metric':<30} | {'Microsoft BitNet.cpp':<20} | {'Tenzo Native Engine':<20}")
    print("-" * 75)
    print(f"{'Model Weights Format':<30} | {'TL1 / i2_s (INT8 LM)':<20} | {'TL1 / i2_s (FP32 LM)':<20}")
    print(f"{'KV-Cache Architecture':<30} | {'Standard FP32/FP16':<20} | {'Fused INT8 (4x comp)':<20}")
    print(f"{'Attention Head Routing':<30} | {'Standard Strided':<20} | {'Zero-Copy In-Place':<20}")
    print(f"{'Prompt / Context Tokens':<30} | {b_res['prompt_tokens']:<20} | {t_res['prompt_tokens']:<20}")
    b_ttft = f"{b_res['ttft_ms']:.2f} ms"
    t_ttft = f"{t_res['ttft_ms']:.2f} ms"
    b_spd = f"{b_res['decode_speed']:.2f} tok/sec"
    t_spd = f"{t_res['decode_speed']:.2f} tok/sec"
    b_lat = f"{1000.0/b_res['decode_speed']:.2f} ms" if b_res['decode_speed'] > 0 else "N/A"
    t_lat = f"{1000.0/t_res['decode_speed']:.2f} ms" if t_res['decode_speed'] > 0 else "N/A"
    b_bw = f"~{b_res['decode_speed'] * 0.85:.1f} GB/s"
    t_bw = f"~{t_res['decode_speed'] * 1.84:.1f} GB/s"

    print(f"{'Time To First Token (TTFT)':<30} | {b_ttft:<20} | {t_ttft:<20}")
    print(f"{'Decode Speed (tok/sec)':<30} | {b_spd:<20} | {t_spd:<20}")
    print(f"{'Per-Token Decode Latency':<30} | {b_lat:<20} | {t_lat:<20}")
    print(f"{'Memory Transferred / tok':<30} | {'~0.85 GB / tok':<20} | {'~1.84 GB / tok':<20}")
    print(f"{'Effective Memory Bandwidth':<30} | {b_bw:<20} | {t_bw:<20}")
    print("═" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Side-by-side benchmark: BitNet.cpp vs Tenzo Engine")
    parser.add_argument("-p", "--prompt", type=str, default="In computer science, a compiler translates source code written in a high-level programming language into", help="Input prompt")
    parser.add_argument("-n", "--tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("-t", "--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--kv-mode", type=str, default="int8_fused", choices=["fp32", "int8_fused"])
    args = parser.parse_args()

    b_res = run_bitnet_cpp(args.prompt, args.tokens, args.temp)
    t_res = run_tenzo(args.prompt, args.tokens, args.temp, kv_mode=args.kv_mode)
    print_comparison(b_res, t_res, args.prompt)
