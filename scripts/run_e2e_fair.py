#!/usr/bin/env python3
"""Block E: same-condition end-to-end comparison, Tenzo vs bitnet.cpp (llama-bench), on this laptop.

Conditions for both engines: 4 threads pinned to cpus 0-3 (Tenzo: dev container, cpuset 0-3, OMP_PLACES=cores spread;
bitnet.cpp: host taskset -c 0-3 -t 4), prompt lengths {16,128,512}, 128 generated tokens, greedy, 5 repetitions, MEDIAN reported.
Nothing else heavy may run meanwhile. All numbers come from the rows this script writes to --out (CSV).

Metrics per (engine, kv, prompt_len):
  pp_tps        prompt tokens / prefill seconds        (Tenzo: token-by-token prefill; llama.cpp: one batch)
  tg_tps        generated tokens / decode seconds      (Tenzo: decode AFTER the prompt; llama-bench -n: decode from an EMPTY context)
  pg_tps        (prompt+gen tokens) / total seconds    (the only like-for-like number: Tenzo measured, llama-bench -pg measured)
  rss_mb        peak resident set of the process (Tenzo: VmHWM at exit; llama-bench: wait4 ru_maxrss)
"""
import argparse, csv, json, os, re, statistics, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "BitNet/build/bin/llama-bench"
GGUF = ROOT / "BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
FIELDS = ["engine", "kv", "prompt_len", "gen_len", "rep", "pp_tps", "tg_tps", "pg_tps", "rss_mb", "threads", "cpus", "governor", "note"]


def governor():
    try: return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except Exception: return "unknown"


def run_tenzo(kv, pp, tg, ids_file, ctx):
    cmd = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "run", "--rm", "-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread", "dev",
           "/app/cmake-build-debug/tenzo-inference", "-m", "/app/models/bitnet-1.58b", "--kv-quant", kv, "-c", str(ctx),
           "--ids-file", "/app/" + str(ids_file), "--bench-pp", str(pp), "--bench-tg", str(tg)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
    m = re.search(r"BENCH_FAIR,(.*)", out)
    if not m: raise RuntimeError("no BENCH_FAIR line:\n" + out[-400:])
    d = dict(kv.split("=") for kv in m.group(1).split(","))
    pp_ms, tg_ms = float(d["pp_ms"]), float(d["tg_ms"])
    return {"pp_tps": pp / (pp_ms / 1e3), "tg_tps": tg / (tg_ms / 1e3), "pg_tps": (pp + tg) / ((pp_ms + tg_ms) / 1e3), "rss_mb": float(d["hwm_kb"]) / 1024}


def run_llama(kind, pp, tg, ctk, extra):
    env = dict(os.environ, LD_LIBRARY_PATH=str(BENCH.parent))
    args = ["taskset", "-c", "0-3", str(BENCH), "-m", str(GGUF), "-t", "4", "-r", "1", "-o", "json", "-ctk", ctk, "-ctv", ctk] + extra
    if kind == "pp": args += ["-p", str(pp), "-n", "0"]
    elif kind == "tg": args += ["-p", "0", "-n", str(tg)]
    else: args += ["-p", "0", "-n", "0", "-pg", f"{pp},{tg}"]
    import tempfile
    with tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        p = subprocess.Popen(args, stdout=fo, stderr=fe, env=env, text=True)
        hwm = 0.0
        while p.poll() is None:                       # VmHWM is a monotonic peak: the last successful read before exit is the peak
            try:
                for line in open(f"/proc/{p.pid}/status"):
                    if line.startswith("VmHWM:"): hwm = max(hwm, float(line.split()[1]) / 1024)
            except OSError: pass
            time.sleep(0.2)
        fo.seek(0); fe.seek(0); out, err = fo.read(), fe.read()
    rss = hwm if hwm > 0 else None
    if p.returncode != 0: raise RuntimeError(f"llama-bench failed ({p.returncode}): {err[-300:]}")
    rows = json.loads(out)
    return rows[0]["avg_ts"], rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--prompts", default="16,128,512"); ap.add_argument("--ids-file", default="results/2026-11-18_overnight/e2e_ids.txt")
    ap.add_argument("--tenzo-kv", default="fp32,int8_fused,tl1_fused"); ap.add_argument("--skip-llama", action="store_true")
    a = ap.parse_args()
    gov = governor(); rows = []
    prompts = [int(x) for x in a.prompts.split(",")]
    w = csv.DictWriter(open(a.out, "w", newline=""), fieldnames=FIELDS); w.writeheader()
    def emit(r):
        r.update(threads=4, cpus="0-3", governor=gov); w.writerow(r); sys.stdout.flush()
    for pp in prompts:
        for kv in a.tenzo_kv.split(","):
            for rep in range(a.reps):
                m = run_tenzo(kv, pp, a.gen, a.ids_file, 1024)
                emit({"engine": "tenzo", "kv": kv, "prompt_len": pp, "gen_len": a.gen, "rep": rep, **m, "note": "token-by-token prefill; tg after the prompt"})
        if a.skip_llama: continue
        for ctk in ("f16", "q8_0"):
            extra = [] if ctk == "f16" else ["-fa", "1"]
            for rep in range(a.reps):
                try:
                    pps, rss = run_llama("pp", pp, a.gen, ctk, extra); tgs, _ = run_llama("tg", pp, a.gen, ctk, extra); pgs, _ = run_llama("pg", pp, a.gen, ctk, extra)
                except RuntimeError as e:
                    emit({"engine": "bitnet.cpp", "kv": ctk, "prompt_len": pp, "gen_len": a.gen, "rep": rep, "note": "UNSUPPORTED: " + str(e)[:120]}); break
                emit({"engine": "bitnet.cpp", "kv": ctk, "prompt_len": pp, "gen_len": a.gen, "rep": rep, "pp_tps": pps, "tg_tps": tgs, "pg_tps": pgs, "rss_mb": rss,
                      "note": "tg_tps measured from an EMPTY context (llama-bench -n); pg_tps = llama-bench -pg"})


if __name__ == "__main__":
    main()
