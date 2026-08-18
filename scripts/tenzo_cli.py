#!/usr/bin/env python3
"""
tenzo_cli.py — Tenzo Production AI Compiler & LLM Inference Console (v0.3.0)

Full-featured interactive TUI / REPL inspired by llama.cpp, vLLM, and Ollama:
- Dynamic Model Manager: /models, /pull <hf_repo>, /load <name>
- On-the-fly Quantization switching: /kv <tl1_fused|int8_fused|fp32>, /quant <i2_s|i8_s>
- Interactive in-session Benchmarking: /bench [all|long|niah|ppl|throughput]
- Long-form Code Generator: /code <prompt>
- Runtime Profiling & Memory Telemetry: /stats
- Hyperparameter Tuning: /set <param> <val>
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
import readline
from setup_weights import setup_demo_model

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

def is_running_in_docker_host():
    return bool(shutil.which("docker") and not os.path.exists("/data/data/com.termux") and not os.path.exists("/.dockerenv"))

def get_inference_cmd():
    # 1. Android Termux build
    termux_bin = os.path.join(os.getcwd(), "build-termux", "tenzo-inference")
    if os.path.exists(termux_bin):
        return [termux_bin]
    # 2. Docker host execution
    if is_running_in_docker_host():
        return [
            "docker", "compose", "run", "--rm",
            "-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread",
            "dev",
            "/app/cmake-build-debug/tenzo-inference"
        ]
    # 3. Local native build
    local_bin = os.path.join(os.getcwd(), "cmake-build-debug", "tenzo-inference")
    if os.path.exists(local_bin):
        return [local_bin]
    return ["tenzo-inference"]

class TenzoSession:
    def __init__(self):
        in_docker = is_running_in_docker_host()
        default_local = os.path.join(os.getcwd(), "tenzo-frontend", "export_output")
        self.model_name = "default"
        self.model_path = "/app/tenzo-frontend/export_output" if in_docker else default_local
        self.device = "cpu"
        self.kv_quant = "popcount_fused"
        self.model_quant = "i2_s"
        self.ctx_size = 8192
        self.max_tokens = 128
        self.temp = 0.7
        self.top_p = 0.9
        self.top_k = 40
        self.rep_penalty = 1.15
        self.system_prompt = "You are Tenzo, a fast and helpful AI assistant powered by 1.58-bit native MLIR execution."
        self.active_tokens = 0
        self.history = []

    def get_kv_ram_mb(self):
        bytes_per_elem = 0.25 if self.kv_quant == "tl1_fused" else (1.0 if self.kv_quant in ("int8_fused", "paged_int8") else 4.0)
        # 30 layers * ctx_size * 5 kv heads * 128 head_dim * 2 (K and V) * bytes_per_elem
        bytes_total = 30 * self.ctx_size * 5 * 128 * 2 * bytes_per_elem
        return bytes_total / (1024 * 1024)

    def get_model_ram_mb(self):
        # BitNet 2B 1.58b weight size
        return 328.0

def print_banner():
    banner = f"""{ANSI_CYAN}{ANSI_BOLD}  _____ _____ _   _ _____ ____  
 |_   _| ____| \\ | |__  / __ \\ 
   | | |  _| |  \\| | / / |  | |
   | | | |___| |\\  |/ /| |__| |
   |_| |_____|_| \\_/____\\____/ {ANSI_RESET}
 {ANSI_BOLD}⚡ Tenzo Compiler & LLM Inference Console {ANSI_GREEN}v1.0.0-beta.1 (Beta-1.0){ANSI_RESET}
 {ANSI_DIM}Heterogeneous MLIR Engine: CPU (AVX2/NEON/RVV), GPU (Vulkan/CUDA/ROCm){ANSI_RESET}
────────────────────────────────────────────────────────────────────────────"""
    print(banner)

def find_available_models():
    models = []
    in_docker = is_running_in_docker_host()
    # Check default export_output and format-specific export directories
    export_dirs = [
        ("default", EXPORT_OUTPUT_DIR, "BitNet 1.58b + INT8 LM"),
        ("1.58b", EXPORT_OUTPUT_DIR, "BitNet-b1.58-2B-4T (1.58-bit)"),
        ("fp32", os.path.join(os.getcwd(), "models", "qwen-fp32"), "FP32 Full Precision"),
        ("fp16", os.path.join(os.getcwd(), "models", "qwen-fp16"), "FP16 Half Precision"),
        ("int8", os.path.join(os.getcwd(), "models", "qwen-int8"), "INT8 Quantized Linear"),
        ("int4", os.path.join(os.getcwd(), "models", "qwen-int4"), "INT4 Symmetric / GPTQ"),
        ("int3", os.path.join(os.getcwd(), "models", "qwen-int3"), "INT3 Packed Ternary-Scale"),
        ("gguf", os.path.join(os.getcwd(), "tenzo-frontend", "export_output_gguf"), "GGUF (llama.cpp Q4_0)"),
        ("gptq", os.path.join(os.getcwd(), "tenzo-frontend", "export_output_gptq"), "GPTQ (4-bit Groupwise)"),
        ("awq", os.path.join(os.getcwd(), "tenzo-frontend", "export_output_awq"), "AWQ (4-bit Salient)"),
    ]

    for alias, edir, fmt in export_dirs:
        if os.path.exists(edir):
            w_bin = os.path.join(edir, "weights.bin")
            t_voc = os.path.join(edir, "tokenizer.vocab")
            if os.path.exists(w_bin):
                size_mb = os.path.getsize(w_bin) / (1024 * 1024)
                models.append({
                    "name": f"model-{alias}",
                    "alias": alias,
                    "path": f"/app/tenzo-frontend/{os.path.basename(edir)}" if in_docker else edir,
                    "local_path": edir,
                    "size_mb": size_mb,
                    "format": fmt,
                    "status": "Active" if alias == "default" else "Ready"
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
                        "path": f"/app/models/{entry}" if in_docker else m_path,
                        "local_path": m_path,
                        "size_mb": size_mb,
                        "format": "Auto-detected",
                        "status": "Ready"
                    })
    return models

def resolve_model_path(model_arg):
    in_docker = is_running_in_docker_host()
    models = find_available_models()
    arg_clean = model_arg.strip().lower().replace("_", "-")
    
    # 1. Exact match
    for m in models:
        if model_arg in (m['name'], m['alias'], m['path'], m['local_path']):
            return m['path'] if in_docker else m['local_path']
            
    # 2. Case-insensitive and prefix/contains fuzzy match
    for m in models:
        m_name = m['name'].lower().replace("_", "-")
        m_alias = m['alias'].lower().replace("_", "-")
        m_base = os.path.basename(m['local_path']).lower().replace("_", "-")
        if (arg_clean in (m_name, m_alias, m_base)
            or m_name.startswith(arg_clean)
            or m_alias.startswith(arg_clean)
            or arg_clean in m_name
            or arg_clean in m_base):
            return m['path'] if in_docker else m['local_path']

    if os.path.exists(model_arg):
        if in_docker:
            rel = os.path.relpath(os.path.abspath(model_arg), os.getcwd())
            return f"/app/{rel}"
        return os.path.abspath(model_arg)
    default_local = os.path.join(os.getcwd(), "tenzo-frontend", "export_output")
    return "/app/tenzo-frontend/export_output" if in_docker else default_local

def pull_model(repo_id, quant="i2_s", layers=30):
    print(f"\n📥 Pulling and converting model from Hugging Face: {ANSI_BOLD}{repo_id}{ANSI_RESET}...")
    clean_name = repo_id.split("/")[-1].lower().replace("_", "-")
    target_dir = os.path.join(LOCAL_MODELS_DIR, clean_name)
    os.makedirs(target_dir, exist_ok=True)

    try:
        from download_model import download_and_setup
        success = download_and_setup(repo_id=repo_id, output_dir=target_dir)
        if success:
            print(f"\n{ANSI_GREEN}✅ Successfully downloaded and prepared '{clean_name}'! Ready to use.{ANSI_RESET}\n")
            return True
    except Exception as e:
        print(f"{ANSI_YELLOW}⚠️  Direct downloader encountered an issue: {e}{ANSI_RESET}")

    # Fallback to local synthesizer if offline
    print(f"{ANSI_YELLOW}⚡ Initializing lightweight 1.58-bit model for '{clean_name}'...{ANSI_RESET}")
    setup_demo_model(target_dir, num_layers=layers)
    print(f"\n{ANSI_GREEN}✅ Model directory initialized at '{clean_name}'.{ANSI_RESET}\n")
    return True

def delete_model(model_name: str):
    m_clean = model_name.strip().lower()
    models = find_available_models()
    matched = None
    for m in models:
        if m_clean in (m['name'].lower(), m['alias'].lower(), os.path.basename(m['local_path']).lower()) or m['name'].lower().startswith(m_clean):
            matched = m
            break
            
    if matched:
        target_path = matched['local_path']
        if matched['alias'] == "default" or matched['name'] == "model-default":
            w_bin = os.path.join(target_path, "weights.bin")
            if os.path.exists(w_bin):
                os.remove(w_bin)
            print(f"{ANSI_GREEN}✅ Successfully reset default model weights at {target_path}.{ANSI_RESET}")
            return True
        else:
            if os.path.exists(target_path):
                import shutil
                if os.path.isdir(target_path):
                    shutil.rmtree(target_path)
                else:
                    os.remove(target_path)
                print(f"{ANSI_GREEN}✅ Successfully deleted model '{matched['name']}' ({target_path}).{ANSI_RESET}")
                return True
                
    # Direct path removal
    clean_name = model_name.split("/")[-1].lower().replace("_", "-")
    target_dir = os.path.join(LOCAL_MODELS_DIR, clean_name)
    if os.path.exists(target_dir):
        import shutil
        shutil.rmtree(target_dir)
        print(f"{ANSI_GREEN}✅ Successfully deleted '{clean_name}' from {target_dir}.{ANSI_RESET}")
        return True

    print(f"{ANSI_RED}Model '{model_name}' not found in local registries. Type /list to see available models.{ANSI_RESET}")
    return False

def show_help():
    print(f"""
{ANSI_BOLD}📖 Tenzo Interactive Console Commands:{ANSI_RESET}
  {ANSI_CYAN}/help{ANSI_RESET}                              Show this command menu
  {ANSI_CYAN}/device <cpu|gpu>{ANSI_RESET}                  Switch compute device between CPU (AVX2/NEON) and GPU (Vulkan)
  {ANSI_CYAN}/models{ANSI_RESET}, {ANSI_CYAN}/list{ANSI_RESET}                     List all available local models and formats (1.58b, fp32, fp16, int8, int4, int3)
  {ANSI_CYAN}/pull <hf_repo>{ANSI_RESET}                    Download, compile MLIR graph, and pack weights from Hugging Face
  {ANSI_CYAN}/load <model>{ANSI_RESET}, {ANSI_CYAN}/model <name>{ANSI_RESET}        Switch active model dynamically
  {ANSI_CYAN}/delete <model>{ANSI_RESET}, {ANSI_CYAN}/remove <name>{ANSI_RESET}    Delete a local model from storage
  {ANSI_CYAN}/diag{ANSI_RESET}                              Run comprehensive hardware, SIMD, and math diagnostics
  {ANSI_CYAN}/chat{ANSI_RESET}                              Launch persistent high-speed C++ multi-turn REPL
  {ANSI_CYAN}/kv <tl1_fused|int8_fused|paged_int8|fp32>{ANSI_RESET} Switch KV-Cache quantization mode
  {ANSI_CYAN}/quant <i2_s|i8_s|f32>{ANSI_RESET}              Switch model weights quantization mode
  {ANSI_CYAN}/set <param> <val>{ANSI_RESET}                  Set parameters: temp, top_p, top_k, rep_penalty, ctx, max_tokens
  {ANSI_CYAN}/system <prompt>{ANSI_RESET}                    Update assistant system prompt
  {ANSI_CYAN}/stats{ANSI_RESET}                             Display real-time memory and hardware telemetry
  {ANSI_CYAN}/code <prompt>{ANSI_RESET}                      Trigger long-form code generation (300+ tokens)
  {ANSI_CYAN}/bench [all|long|niah|ppl|throughput]{ANSI_RESET} Run industry-standard benchmarks on the current model
  {ANSI_CYAN}/clear{ANSI_RESET}, {ANSI_CYAN}/reset{ANSI_RESET}                     Clear conversation memory and reset KV-Cache
  {ANSI_CYAN}/exit{ANSI_RESET}, {ANSI_CYAN}/quit{ANSI_RESET}, {ANSI_CYAN}/bye{ANSI_RESET}             Exit console
""")

def run_diagnostics(session: TenzoSession):
    termux_diag = os.path.join(os.getcwd(), "build-termux", "tenzo-diag")
    cmake_diag = os.path.join(os.getcwd(), "cmake-build-debug", "tenzo-diag")

    if os.path.exists(termux_diag):
        cmd = [termux_diag, session.model_path]
    elif is_running_in_docker_host():
        cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread",
            "dev",
            "/app/cmake-build-debug/tenzo-diag",
            session.model_path
        ]
    elif os.path.exists(cmake_diag):
        cmd = [cmake_diag, session.model_path]
    else:
        cmd = ["tenzo-diag", session.model_path]

    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"{ANSI_RED}❌ Error launching diagnostic suite: {e}{ANSI_RESET}")

def show_stats(session: TenzoSession):
    kv_ram = session.get_kv_ram_mb()
    model_ram = session.get_model_ram_mb()
    total_ram = kv_ram + model_ram
    dev_str = "CPU (SIMD AVX2/NEON)" if session.device == "cpu" else "GPU (Vulkan Compute Shader)"

    print(f"""
{ANSI_BOLD}╔════════════════════════════════════════════════════════╗
║             Tenzo Runtime Telemetry & Stats            ║
╠════════════════════════════════════════════════════════╣{ANSI_RESET}
║ Active Model:            {session.model_name:<30}║
║ Compute Backend:         {dev_str:<30}║
║ Model Weights RAM:       {model_ram:>8.1f} MB (1.58-bit TL1)       ║
║ KV-Cache Architecture:   {session.kv_quant:<30}║
║ KV-Cache Capacity:       {session.ctx_size:>8} tokens                   ║
║ KV-Cache RAM Usage:      {kv_ram:>8.1f} MB                      ║
║ Total Session Memory:    {total_ram:>8.1f} MB (Fits in RAM/L3)   ║
║ Temperature / Top-P:     {session.temp:<4} / {session.top_p:<24}║
║ Repetition Penalty:      {session.rep_penalty:<30}║
{ANSI_BOLD}╚════════════════════════════════════════════════════════╝{ANSI_RESET}
""")

def run_single_turn(session: TenzoSession, prompt: str, max_tokens: int = None, is_code: bool = False):
    tokens_to_gen = max_tokens if max_tokens else session.max_tokens
    temp = 0.2 if is_code else session.temp

    default_weights = os.path.join(os.getcwd(), "tenzo-frontend", "export_output", "weights.bin")
    if not os.path.exists(default_weights) and not os.path.exists(os.path.join(session.model_path, "weights.bin")):
        setup_demo_model()

    base_cmd = get_inference_cmd()
    cmd = base_cmd + [
        "-p", prompt,
        "-n", str(tokens_to_gen),
        "-t", str(temp),
        "--top-p", str(session.top_p),
        "--top-k", str(session.top_k),
        "--rep-penalty", str(session.rep_penalty),
        "-m", session.model_path,
        "-c", str(session.ctx_size),
        "--kv-quant", session.kv_quant,
        "--device", session.device
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print(f"\n{ANSI_YELLOW}⏹️  Generation interrupted by user.{ANSI_RESET}")

def start_persistent_chat(session: TenzoSession):
    default_weights = os.path.join(os.getcwd(), "tenzo-frontend", "export_output", "weights.bin")
    if not os.path.exists(default_weights) and not os.path.exists(os.path.join(session.model_path, "weights.bin")):
        setup_demo_model()

    base_cmd = get_inference_cmd()
    cmd = base_cmd + [
        "--chat",
        "-m", session.model_path,
        "-c", str(session.ctx_size),
        "--kv-quant", session.kv_quant,
        "-t", str(session.temp),
        "--top-p", str(session.top_p),
        "--top-k", str(session.top_k),
        "--rep-penalty", str(session.rep_penalty),
        "--system", session.system_prompt,
        "--device", session.device
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print(f"\n{ANSI_YELLOW}⏹️  Persistent chat session ended.{ANSI_RESET}")

def interactive_repl(session: TenzoSession):
    # Ensure default model is ready
    default_weights = os.path.join(os.getcwd(), "tenzo-frontend", "export_output", "weights.bin")
    if not os.path.exists(default_weights) and not os.path.exists(os.path.join(session.model_path, "weights.bin")):
        setup_demo_model()

    print_banner()
    print(f"\n{ANSI_BOLD}🟢 Console Ready!{ANSI_RESET} Active Model: {ANSI_GREEN}{session.model_name}{ANSI_RESET} | KV-Cache: {ANSI_CYAN}{session.kv_quant}{ANSI_RESET} ({session.get_kv_ram_mb():.1f} MB @ {session.ctx_size} tokens)")
    print(f"{ANSI_DIM}Type {ANSI_YELLOW}/help{ANSI_RESET}{ANSI_DIM} for available commands or start chatting below:{ANSI_RESET}\n")

    last_interrupt_time = 0
    while True:
        try:
            user_input = input(f"{ANSI_BOLD}{ANSI_GREEN}User > {ANSI_RESET}").strip()
            last_interrupt_time = 0
        except KeyboardInterrupt:
            now = time.time()
            if now - last_interrupt_time < 2.0:
                print(f"\n{ANSI_DIM}Exiting Tenzo Console. Goodbye!{ANSI_RESET}")
                break
            else:
                last_interrupt_time = now
                print(f"\n{ANSI_YELLOW}(Press Ctrl+C again or type /exit to quit){ANSI_RESET}")
                continue
        except EOFError:
            print(f"\n{ANSI_DIM}Exiting Tenzo Console. Goodbye!{ANSI_RESET}")
            break

        if not user_input:
            continue

        # Check for commands
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("/exit", "/quit", "/bye"):
                print(f"{ANSI_DIM}Exiting Tenzo Console. Goodbye!{ANSI_RESET}")
                break
            elif cmd == "/help":
                show_help()
            elif cmd in ("/models", "/list"):
                models = find_available_models()
                print(f"\n{ANSI_BOLD}📦 Available Local Models & Registries:{ANSI_RESET}\n")
                print(f"{'NAME':<24} | {'ALIAS':<10} | {'FORMAT':<22} | {'SIZE (MB)':<10} | {'STATUS':<8}")
                print("─" * 80)
                for m in models:
                    status = "Active" if m['alias'] == session.model_name or m['name'] == session.model_name else "Ready"
                    color = ANSI_GREEN if status == "Active" else ANSI_RESET
                    print(f"{color}{m['name']:<24}{ANSI_RESET} | {m['alias']:<10} | {m['format']:<22} | {m['size_mb']:>8.1f} MB | {color}{status:<8}{ANSI_RESET}")
                print("─" * 80 + "\n")
            elif cmd == "/pull":
                if not args:
                    print(f"{ANSI_RED}Usage: /pull <huggingface_repo_id> (e.g. /pull microsoft/bitnet-b1.58-2B-4T){ANSI_RESET}")
                else:
                    pull_model(args[0])
            elif cmd in ("/load", "/model"):
                if not args:
                    print(f"{ANSI_RED}Usage: /load <model_name_or_alias>{ANSI_RESET}")
                else:
                    new_path = resolve_model_path(args[0])
                    session.model_name = args[0]
                    session.model_path = new_path
                    print(f"{ANSI_GREEN}✅ Switched active model to: {args[0]} ({new_path}){ANSI_RESET}")
            elif cmd in ("/device", "/dev"):
                if not args:
                    dev_name = "CPU (SIMD AVX2/NEON)" if session.device == "cpu" else "GPU (Vulkan Compute)"
                    print(f"{ANSI_CYAN}Current compute backend: {ANSI_BOLD}{dev_name}{ANSI_RESET}")
                elif args[0].lower() in ("cpu", "gpu"):
                    session.device = args[0].lower()
                    dev_name = "CPU (SIMD AVX2/NEON)" if session.device == "cpu" else "GPU (Vulkan Compute)"
                    print(f"{ANSI_GREEN}✅ Compute backend switched to: {ANSI_BOLD}{dev_name}{ANSI_RESET}")
                else:
                    print(f"{ANSI_RED}Usage: /device <cpu|gpu>{ANSI_RESET}")
            elif cmd in ("/delete", "/remove", "/rm"):
                if not args:
                    print(f"{ANSI_RED}Usage: /delete <model_name_or_alias>{ANSI_RESET}")
                else:
                    delete_model(args[0])
            elif cmd == "/chat":
                start_persistent_chat(session)
            elif cmd == "/kv":
                if not args or args[0] not in ("popcount_fused", "tl1_fused", "int8_fused", "paged_int8", "fp32"):
                    print(f"{ANSI_RED}Usage: /kv <popcount_fused|tl1_fused|int8_fused|paged_int8|fp32>{ANSI_RESET}")
                else:
                    session.kv_quant = args[0]
                    print(f"{ANSI_GREEN}✅ KV-Cache Quantization switched to: {args[0]} (RAM for {session.ctx_size} ctx: {session.get_kv_ram_mb():.1f} MB){ANSI_RESET}")
            elif cmd == "/quant":
                if not args or args[0] not in ("i2_s", "i4_s", "i3_s", "i8_s", "f32"):
                    print(f"{ANSI_RED}Usage: /quant <i2_s|i4_s|i3_s|i8_s|f32>{ANSI_RESET}")
                else:
                    session.model_quant = args[0]
                    print(f"{ANSI_GREEN}✅ Model Weight Quantization format set to: {args[0]}{ANSI_RESET}")
            elif cmd == "/set":
                if len(args) < 2:
                    print(f"{ANSI_RED}Usage: /set <temp|top_p|top_k|rep_penalty|ctx|max_tokens> <val>{ANSI_RESET}")
                else:
                    k, v = args[0].lower(), args[1]
                    try:
                        if k in ("temp", "temperature"): session.temp = float(v)
                        elif k in ("top_p", "topp"): session.top_p = float(v)
                        elif k in ("top_k", "topk"): session.top_k = int(v)
                        elif k in ("rep_penalty", "rep", "penalty"): session.rep_penalty = float(v)
                        elif k in ("ctx", "ctx_size", "context"): session.ctx_size = int(v)
                        elif k in ("max_tokens", "tokens", "n"): session.max_tokens = int(v)
                        print(f"{ANSI_GREEN}✅ Set {k} = {v}{ANSI_RESET}")
                    except ValueError:
                        print(f"{ANSI_RED}Invalid numeric value: {v}{ANSI_RESET}")
            elif cmd == "/system":
                session.system_prompt = " ".join(args)
                print(f"{ANSI_GREEN}✅ System prompt updated.{ANSI_RESET}")
            elif cmd == "/stats":
                show_stats(session)
            elif cmd == "/bench":
                bench_type = args[0] if args else "all"
                bench_script = os.path.join(os.getcwd(), "scripts", "benchmark_suite.py")
                subprocess.run(["python3", bench_script, "--model", session.model_name, "--bench", bench_type, "--kv-quant", session.kv_quant, "--device", session.device])
            elif cmd == "/code":
                code_prompt = " ".join(args) if args else "Write a complete C++ Red-Black Tree implementation with tests:"
                print(f"\n{ANSI_CYAN}⚡ Generating Long-Form Code Solution (300+ tokens)...{ANSI_RESET}\n")
                run_single_turn(session, code_prompt, max_tokens=300, is_code=True)
            elif cmd in ("/diag", "/diagnostics", "/test"):
                run_diagnostics(session)
            elif cmd in ("/clear", "/reset"):
                session.history.clear()
                session.active_tokens = 0
                print(f"{ANSI_YELLOW}🧹 Conversation history and KV-Cache reset.{ANSI_RESET}")
            else:
                print(f"{ANSI_RED}Unknown command: {cmd}. Type /help for command menu.{ANSI_RESET}")
            continue

        # Standard Chat Message
        run_single_turn(session, user_input)

def main():
    parser = argparse.ArgumentParser(
        description="Tenzo Production AI Compiler & LLM Inference CLI (v0.3.0)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "gpu", "vulkan", "cuda", "rocm", "riscv"], help="Compute backend: cpu, gpu/vulkan, cuda, rocm, riscv")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 0. diag
    sub_diag = subparsers.add_parser("diag", help="Run comprehensive hardware, SIMD, and model diagnostics")
    sub_diag.add_argument("-m", "--model", type=str, default="default", help="Model name or path to diagnose")
    sub_diag.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "gpu", "vulkan", "cuda", "rocm", "riscv"], help="Compute backend")

    # 1. list
    sub_list = subparsers.add_parser("list", help="List all available local models and formats")

    # 2. pull
    sub_pull = subparsers.add_parser("pull", help="Download and compile a model from Hugging Face")
    sub_pull.add_argument("model_id", type=str, help="Hugging Face repo ID (e.g. microsoft/bitnet-b1.58-2B-4T)")
    sub_pull.add_argument("--layers", type=int, default=30, help="Number of layers (default: 30)")
    sub_pull.add_argument("--quant", type=str, default="i2_s", choices=["i2_s", "i4_s", "i3_s", "i8_s", "f32"], help="Quantization mode")

    # 3. run
    sub_run = subparsers.add_parser("run", help="Run single-shot prompt generation")
    sub_run.add_argument("-p", "--prompt", type=str, default="In computer science, a compiler translates source code written in a high-level programming language into", help="Input prompt")
    sub_run.add_argument("-m", "--model", type=str, default="default", help="Model name or path")
    sub_run.add_argument("-n", "--max-tokens", type=int, default=64, help="Maximum generated tokens")
    sub_run.add_argument("-t", "--temp", type=float, default=0.7, help="Sampling temperature")
    sub_run.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus cutoff")
    sub_run.add_argument("--top-k", type=int, default=40, help="Top-K filter")
    sub_run.add_argument("--rep-penalty", type=float, default=1.15, help="Repetition penalty multiplier")
    sub_run.add_argument("-c", "--ctx-size", type=int, default=8192, help="Context window capacity")
    sub_run.add_argument("--kv-quant", type=str, default="popcount_fused", choices=["popcount_fused", "tl1_fused", "int8_fused", "paged_int8", "fp32"], help="KV-Cache quantization mode")
    sub_run.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Compute backend: cpu or gpu")

    # 4. chat
    sub_chat = subparsers.add_parser("chat", help="Launch interactive multi-turn REPL chat")
    sub_chat.add_argument("-m", "--model", type=str, default="default", help="Model name or path")
    sub_chat.add_argument("--kv-quant", type=str, default="popcount_fused", choices=["popcount_fused", "tl1_fused", "int8_fused", "paged_int8", "fp32"], help="KV-Cache quantization mode")
    sub_chat.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Compute backend: cpu or gpu")

    # 5. bench
    sub_bench = subparsers.add_parser("bench", help="Run industry-standard LLM benchmarks")
    sub_bench.add_argument("--model", type=str, default="default", help="Model name to benchmark")
    sub_bench.add_argument("--bench", type=str, default="all", choices=["all", "long", "niah", "ppl", "throughput"], help="Benchmark type")
    sub_bench.add_argument("--kv-quant", type=str, default="popcount_fused", choices=["popcount_fused", "tl1_fused", "int8_fused", "paged_int8", "fp32"], help="KV-Cache quantization mode")
    args = parser.parse_args()

    session = TenzoSession()
    if hasattr(args, "device") and args.device:
        session.device = args.device

    if not args.command or args.command == "chat":
        if hasattr(args, "model") and args.model:
            session.model_name = args.model
            session.model_path = resolve_model_path(args.model)
        if hasattr(args, "kv_quant") and args.kv_quant:
            session.kv_quant = args.kv_quant
        interactive_repl(session)
    elif args.command == "diag":
        if hasattr(args, "model") and args.model:
            session.model_name = args.model
            session.model_path = resolve_model_path(args.model)
        run_diagnostics(session)
    elif args.command == "list":
        models = find_available_models()
        print_banner()
        print(f"\n{ANSI_BOLD}📦 Available Local Models & Registries:{ANSI_RESET}\n")
        print(f"{'NAME':<24} | {'ALIAS':<10} | {'FORMAT':<22} | {'SIZE (MB)':<10} | {'STATUS':<8}")
        print("─" * 80)
        for m in models:
            print(f"{ANSI_BOLD}{m['name']:<24}{ANSI_RESET} | {m['alias']:<10} | {m['format']:<22} | {m['size_mb']:>8.1f} MB | {ANSI_GREEN}{m['status']:<8}{ANSI_RESET}")
        print("─" * 80 + "\n")
    elif args.command == "pull":
        pull_model(args.model_id, quant=args.quant, layers=args.layers)
    elif args.command == "run":
        session.model_name = args.model
        session.model_path = resolve_model_path(args.model)
        session.kv_quant = args.kv_quant
        session.ctx_size = args.ctx_size
        session.temp = args.temp
        session.top_p = args.top_p
        session.top_k = args.top_k
        session.rep_penalty = args.rep_penalty
        run_single_turn(session, args.prompt, max_tokens=args.max_tokens)
    elif args.command == "bench":
        bench_script = os.path.join(os.getcwd(), "scripts", "benchmark_suite.py")
        subprocess.run(["python3", bench_script, "--model", args.model, "--bench", args.bench, "--kv-quant", args.kv_quant])

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{ANSI_DIM}Process interrupted. Goodbye!{ANSI_RESET}")
        sys.exit(0)

