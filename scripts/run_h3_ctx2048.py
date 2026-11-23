#!/usr/bin/env python3
"""Run H3 (ctx=2048) evaluation for golden_bitnet (fp32, tenzo_int8, S3+s4r32, S13+s4r32)
and Tenzo engine (int8_fused, ternary_rot, ternary_rot+alpha_mix).

Golden runs use GB.Golden directly with row-chunked (256-token) lm_logits_rows + double
log_softmax so peak RSS stays <= ~2.6 GB under MemoryMax=4G (full 2047x128256 float64
would allocate >5.2 GB). Supports batch=4 so all 4 windows of a file take 1 weight pass.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import subprocess
import sys
import time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
H3_DIR = ROOT / "results" / "2026-11-23_agent2" / "H3"
SCORES_DIR = H3_DIR / "scores"


def run_golden_ctx2048(lang: str, scheme: str | None, batch: int = 4, threads: int = 4) -> int:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    sname = "fp32" if scheme is None else scheme.replace(":", "_").replace("+", "_")
    out_p = SCORES_DIR / f"golden_{sname}_{lang}.txt"
    if out_p.exists() and sum(1 for ln in out_p.read_text(encoding="utf-8").splitlines() if ln.startswith("w ")) >= 4:
        print(f"[skip] {out_p} already complete", flush=True)
        return 0

    sys.path.insert(0, str(ROOT / "tools" / "reference"))
    import golden_bitnet as GB
    import kv_ptq

    if scheme is not None and scheme != "fp32":
        kv_ptq.install(scheme, GB)
    else:
        GB.kv_quant = lambda k, v, l: (k, v)
        GB.q_quant = None

    g = GB.Golden(GB.SNAP, torch.float32, threads)
    quiet = GB.Dumper(None, 0, [], [])
    win_path = ROOT / "results" / "reference" / f"windows_{lang}_2048.ids"
    wins = [list(map(int, ln.split())) for ln in win_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(wins) == 4 and all(len(w) == 2048 for w in wins)

    t0 = time.time()
    with out_p.open("w", encoding="utf-8") as f:
        f.write(f"# golden dtype=float32 kv={scheme or 'fp32'} ctx=2048 windows={len(wins)}\n")
        for b0 in range(0, len(wins), batch):
            idx = list(range(b0, min(b0 + batch, len(wins))))
            ids = torch.tensor([wins[w] for w in idx], dtype=torch.long)
            h = g.hidden(ids, quiet)
            for j, w in enumerate(idx):
                hj = h[j, :-1]  # [2047, C]
                targets = ids[j, 1:, None]
                lp: list[float] = []
                for r0 in range(0, hj.shape[0], 256):
                    lg = g.lm_logits_rows(hj[r0 : r0 + 256])
                    chunk_lp = torch.log_softmax(lg.double(), dim=-1).gather(1, targets[r0 : r0 + 256])[:, 0].tolist()
                    lp.extend(chunk_lp)
                    del lg
                nll = -sum(lp)
                f.write(f"w {w} {len(lp)} {nll:.9f}\n")
                f.write("lp " + " ".join(f"{x:.9f}" for x in lp) + "\n")
                f.flush()
                print(
                    f"[golden {sname} {lang}] w={w} n={len(lp)} ppl={math.exp(nll/len(lp)):.3f} "
                    f"t={time.time()-t0:.0f}s rss={GB.rss_gb():.2f}GB",
                    flush=True,
                )
    print(f"[done] golden {sname} {lang} in {time.time()-t0:.1f}s peak_rss={GB.peak_gb():.2f}GB", flush=True)
    return 0


def run_engine_all() -> int:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    runs = [
        ("int8_fused", "int8_fused", None),
        ("ternary_rot", "ternary_rot", None),
        ("ternary_rot_alpha", "ternary_rot", "/app/results/2026-11-23_agent2/H1/alpha_mix.txt"),
    ]
    for lang in ("en", "uk"):
        win_rel = f"results/reference/windows_{lang}_2048.ids"
        for label, kv_mode, alpha_env in runs:
            out_rel = f"results/2026-11-23_agent2/H3/scores/engine_{label}_{lang}.txt"
            log_p = ROOT / f"results/2026-11-23_agent2/H3/scores/engine_{label}_{lang}.log"
            out_p = ROOT / out_rel
            if out_p.exists() and sum(1 for ln in out_p.read_text(encoding="utf-8").splitlines() if ln.startswith("w ")) >= 4:
                print(f"[skip] {out_rel} already complete", flush=True)
                continue
            cmd = [
                "docker", "compose", "run", "--rm",
                "-e", "OMP_PLACES=cores",
                "-e", "OMP_PROC_BIND=spread",
            ]
            if alpha_env is not None:
                cmd.extend(["-e", f"TENZO_KV_ALPHA={alpha_env}"])
            cmd.extend([
                "dev", "nice", "-n", "19", "ionice", "-c3",
                "/app/cmake-build-debug/tenzo-inference",
                "-m", "/app/models/bitnet-1.58b-v2",
                "--kv-quant", kv_mode,
                "-c", "2048",
                "--score", f"/app/{win_rel}",
                "--score-out", f"/app/{out_rel}",
            ])
            print(f"[run] engine {label} {lang}", flush=True)
            with log_p.open("w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f"[FAIL] engine {label} {lang} exit={rc}", file=sys.stderr)
                return rc
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["golden-one", "golden-all", "engine-all"], required=True)
    ap.add_argument("--lang", choices=["en", "uk"], default="en")
    ap.add_argument("--scheme", default="fp32")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "golden-one":
        return run_golden_ctx2048(args.lang, None if args.scheme == "fp32" else args.scheme, args.batch, args.threads)

    if args.mode == "golden-all":
        for scheme in ("fp32", "tenzo_int8", "S3+s4r32", "S13+s4r32"):
            for lang in ("en", "uk"):
                sname = scheme.replace(":", "_").replace("+", "_")
                out_p = SCORES_DIR / f"golden_{sname}_{lang}.txt"
                log_p = SCORES_DIR / f"golden_{sname}_{lang}.log"
                if out_p.exists() and sum(1 for ln in out_p.read_text(encoding="utf-8").splitlines() if ln.startswith("w ")) >= 4:
                    print(f"[skip] {out_p} already complete", flush=True)
                    continue
                cmd = [
                    "systemd-run", "--user", "--scope",
                    "-p", "MemoryMax=4G", "-p", "MemorySwapMax=0", "--",
                    "nice", "-n", "19", "ionice", "-c3",
                    "python3", "scripts/run_h3_ctx2048.py",
                    "--mode", "golden-one", "--lang", lang, "--scheme", scheme,
                    "--batch", str(args.batch), "--threads", str(args.threads),
                ]
                print(f"[run] {sname} {lang}", flush=True)
                with log_p.open("w", encoding="utf-8") as lf:
                    rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print(f"[FAIL] {sname} {lang} exit={rc}", file=sys.stderr)
                    return rc
        return 0

    return run_engine_all()


if __name__ == "__main__":
    raise SystemExit(main())
