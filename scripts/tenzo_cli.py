#!/usr/bin/env python3
"""
tenzo_cli.py — Tenzo Production AI Compiler & LLM Inference CLI (v0.3.0)

A unified, modern command-line interface inspired by llama.cpp, vLLM, and Ollama:
- Model auto-detection and local registry management (`tenzo list`)
- Hugging Face model pull and automated MLIR compilation (`tenzo pull <repo_id>`)
- Interactive multi-turn REPL chat (`tenzo chat`)
- Streaming single-shot generation (`tenzo run -p "..."`)
- Fine-grained quantization settings for weights and KV-cache
"""

import os
import sys
import argparse
import subprocess
import json
import shutil
import time

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[37m"

DEFAULT_MODELS_DIR = "/app/models"
LOCAL_MODELS_DIR = os.path.join(os.getcwd(), "models")
EXPORT_OUTPUT_DIR = os.path.join(os.getcwd(), "tenzo-frontend", "export_output")

def print_banner():
    banner = f"""{ANSI_CYAN}{ANSI_BOLD}  _____ _____ _   _ _____ ____  
 |_   _| ____| \\ | |__  / __ \\ 
   | | |  _| |  \\| | / / |  | |
   | | | |___| |\\  |/ /| |__| |
   |_| |_____|_| \\_/____\\____/ {ANSI_RESET}
 {ANSI_BOLD}⚡ Tenzo Compiler & LLM Inference CLI {ANSI_GREEN}v0.3.0{ANSI_RESET}
 {ANSI_DIM}High-Performance 1.58-bit BitNet Heterogeneous MLIR/AVX2 Engine{ANSI_RESET}
────────────────────────────────────────────────────────────────────────────"""
    print(banner)

def find_available_models():
    models = []
    # Check default export_output
    if os.path.exists(EXPORT_OUTPUT_DIR):
        w_bin = os.path.join(EXPORT_OUTPUT_DIR, "weights.bin")
        m_mlir = os.path.join(EXPORT_OUTPUT_DIR, "model.mlir")
        t_voc = os.path.join(EXPORT_OUTPUT_DIR, "tokenizer.vocab")
        if os.path.exists(w_bin) and os.path.exists(t_voc):
            size_mb = os.path.getsize(w_bin) / (1024 * 1024)
            models.append({
                "name": "bitnet-2b-default",
                "alias": "default",
                "path": "/app/tenzo-frontend/export_output",
                "local_path": EXPORT_OUTPUT_DIR,
                "size_mb": size_mb,
                "format": "TL1 (1.58b) + INT8 LM",
                "status": "Ready"
            })

    # Check models directory
    if os.path.exists(LOCAL_MODELS_DIR):
        for entry in os.listdir(LOCAL_MODELS_DIR):
            m_path = os.path.join(LOCAL_MODELS_DIR, entry)
            if os.path.isdir(m_path):
                w_bin = os.path.join(m_path, "weights.bin")
                if os.path.exists(w_bin):
                    size_mb = os.path.getsize(w_bin) / (1024 * 1024)
                    models.append({
                        "name": entry,
                        "alias": entry,
                        "path": f"/app/models/{entry}",
                        "local_path": m_path,
                        "size_mb": size_mb,
                        "format": "TL1 (1.58b)",
                        "status": "Ready"
                    })
    return models

def cmd_list(args):
    print_banner()
    models = find_available_models()
    print(f"\n{ANSI_BOLD}📦 Available Local Models & Registries:{ANSI_RESET}\n")
    print(f"{'NAME':<24} | {'ALIAS':<10} | {'FORMAT':<22} | {'SIZE (MB)':<10} | {'STATUS':<8}")
    print("─" * 80)
    if not models:
        print(f"  {ANSI_YELLOW}No models found in ./models or ./tenzo-frontend/export_output.{ANSI_RESET}")
        print(f"  Run {ANSI_CYAN}tenzo pull microsoft/bitnet-b1.58-2B-4T{ANSI_RESET} to download a model.")
    else:
        for m in models:
            print(f"{ANSI_BOLD}{m['name']:<24}{ANSI_RESET} | {m['alias']:<10} | {m['format']:<22} | {m['size_mb']:>8.1f} MB | {ANSI_GREEN}{m['status']:<8}{ANSI_RESET}")
    print("─" * 80)
    print(f"\n💡 Use a model with: {ANSI_CYAN}tenzo chat --model <name>{ANSI_RESET} or {ANSI_CYAN}tenzo run --model <name> -p \"...\"{ANSI_RESET}\n")

def cmd_pull(args):
    print_banner()
    repo_id = args.model_id
    print(f"\n📥 Pulling and converting model from Hugging Face: {ANSI_BOLD}{repo_id}{ANSI_RESET}...")
    
    clean_name = repo_id.split("/")[-1].lower().replace("_", "-")
    target_dir = os.path.join(LOCAL_MODELS_DIR, clean_name)
    os.makedirs(target_dir, exist_ok=True)

    export_script = os.path.join(os.getcwd(), "tenzo-frontend", "export_bitnet.py")
    if not os.path.exists(export_script):
        print(f"{ANSI_RED}❌ Export script not found at {export_script}{ANSI_RESET}")
        sys.exit(1)

    print(f"🚀 Executing Tenzo automated MLIR export & weight packing pipeline...")
    cmd = [
        "python3", export_script,
        "--model", repo_id,
        "--out", target_dir,
        "--layers", str(args.layers) if args.layers else "30",
        "--quant", args.quant
    ]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print(f"\n{ANSI_GREEN}✅ Successfully installed and compiled '{clean_name}'!{ANSI_RESET}")
        print(f"   Model directory: {target_dir}")
        print(f"   Run with: {ANSI_CYAN}python3 scripts/tenzo_cli.py chat --model {clean_name}{ANSI_RESET}\n")
    else:
        print(f"\n{ANSI_RED}❌ Failed to export model {repo_id}. Check log output above.{ANSI_RESET}")
        sys.exit(res.returncode)

def resolve_model_path(model_arg):
    models = find_available_models()
    for m in models:
        if model_arg in (m['name'], m['alias'], m['path'], m['local_path']):
            return m['path']
    if os.path.exists(model_arg):
        return model_arg
    return "/app/tenzo-frontend/export_output"

def cmd_run(args):
    model_path = resolve_model_path(args.model)
    cmd = [
        "docker", "compose", "run", "--rm",
        "-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread",
        "dev",
        "/app/cmake-build-debug/tenzo-inference",
        "-p", args.prompt if args.prompt else "In computer science, a compiler translates source code written in a high-level programming language into",
        "-n", str(args.max_tokens),
        "-t", str(args.temp),
        "--top-p", str(args.top_p),
        "--top-k", str(args.top_k),
        "--rep-penalty", str(args.rep_penalty),
        "-m", model_path,
        "-c", str(args.ctx_size),
        "--kv-quant", args.kv_quant
    ]
    subprocess.run(cmd)

def cmd_chat(args):
    model_path = resolve_model_path(args.model)
    cmd = [
        "docker", "compose", "run", "--rm", "-it",
        "-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread",
        "dev",
        "/app/cmake-build-debug/tenzo-inference",
        "--chat",
        "-t", str(args.temp),
        "--top-p", str(args.top_p),
        "--top-k", str(args.top_k),
        "--rep-penalty", str(args.rep_penalty),
        "-m", model_path,
        "-c", str(args.ctx_size),
        "--kv-quant", args.kv_quant,
        "--system", args.system if args.system else "You are Tenzo, a fast and helpful AI assistant."
    ]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Tenzo Compiler & LLM Inference CLI (v0.3.0)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 1. list
    sub_list = subparsers.add_parser("list", help="List all available local models and formats")
    sub_list.set_defaults(func=cmd_list)

    # 2. pull
    sub_pull = subparsers.add_parser("pull", help="Download and compile a model from Hugging Face")
    sub_pull.add_argument("model_id", type=str, help="Hugging Face repo ID (e.g. microsoft/bitnet-b1.58-2B-4T)")
    sub_pull.add_argument("--layers", type=int, default=30, help="Number of layers to export (default: 30)")
    sub_pull.add_argument("--quant", type=str, default="i2_s", choices=["i2_s", "i8_s", "f32"], help="Quantization mode (default: i2_s)")
    sub_pull.set_defaults(func=cmd_pull)

    # 3. run
    sub_run = subparsers.add_parser("run", help="Run single-shot prompt generation")
    sub_run.add_argument("-p", "--prompt", type=str, default="In computer science, a compiler translates source code written in a high-level programming language into", help="Input prompt")
    sub_run.add_argument("-m", "--model", type=str, default="default", help="Model name or path")
    sub_run.add_argument("-n", "--max-tokens", type=int, default=64, help="Maximum generated tokens")
    sub_run.add_argument("-t", "--temp", type=float, default=0.7, help="Sampling temperature")
    sub_run.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus cutoff")
    sub_run.add_argument("--top-k", type=int, default=40, help="Top-K filter")
    sub_run.add_argument("--rep-penalty", type=float, default=1.15, help="Repetition penalty multiplier")
    sub_run.add_argument("-c", "--ctx-size", type=int, default=8192, help="Context window capacity (default: 8192)")
    sub_run.add_argument("--kv-quant", type=str, default="tl1_fused", choices=["tl1_fused", "int8_fused", "fp32"], help="KV-Cache quantization mode")
    sub_run.set_defaults(func=cmd_run)

    # 4. chat
    sub_chat = subparsers.add_parser("chat", help="Launch interactive multi-turn REPL chat")
    sub_chat.add_argument("-m", "--model", type=str, default="default", help="Model name or path")
    sub_chat.add_argument("-t", "--temp", type=float, default=0.7, help="Sampling temperature")
    sub_chat.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus cutoff")
    sub_chat.add_argument("--top-k", type=int, default=40, help="Top-K filter")
    sub_chat.add_argument("--rep-penalty", type=float, default=1.15, help="Repetition penalty multiplier")
    sub_chat.add_argument("-c", "--ctx-size", type=int, default=8192, help="Context window capacity (default: 8192)")
    sub_chat.add_argument("--kv-quant", type=str, default="tl1_fused", choices=["tl1_fused", "int8_fused", "fp32"], help="KV-Cache quantization mode")
    sub_chat.add_argument("--system", type=str, default="You are Tenzo, a fast and helpful AI assistant.", help="System prompt")
    sub_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    if not args.command:
        print_banner()
        parser.print_help()
    else:
        args.func(args)

if __name__ == "__main__":
    main()
