#!/usr/bin/env python3
"""T18 acceptance test: kill a training run mid-way (SIGKILL), resume it, and require a bit-identical loss curve.

Runs tenzo-frontend/train_lm.py three times on CPU with one thread (deterministic):
  1. reference: N steps without interruption;
  2. victim: same command, SIGKILLed after its first checkpoint, while it is somewhere past that checkpoint;
  3. resume: same command with --resume.
Pass iff, for every step, the last logged loss of runs 2+3 equals the reference loss exactly (float.hex()).
Usage: python3 tenzo-frontend/test_train_resume.py [--kv s3] [--steps 30]
"""
from __future__ import annotations
import argparse, json, pathlib, signal, subprocess, sys, tempfile, time

HERE = pathlib.Path(__file__).resolve().parent


def losses(log):
    out = {}
    for line in open(log):
        r = json.loads(line)
        out[r["step"]] = r["loss_hex"]  # later lines (after resume) overwrite earlier ones
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default="s3")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--save-every", type=int, default=10)
    a = ap.parse_args()
    data = HERE / "train_lm.py"   # any file with a few KB of bytes
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        base = [sys.executable, str(HERE / "train_lm.py"), "--data", str(data), "--bytes", "--d", "64", "--layers", "2",
                "--heads", "2", "--ff", "128", "--seq", "64", "--batch", "4", "--kv", a.kv, "--kv-recent", "16",
                "--steps", str(a.steps), "--save-every", str(a.save_every), "--warmup", "5", "--threads", "1"]
        subprocess.run(base + ["--out", str(td / "ref")], check=True, capture_output=True)
        victim = subprocess.Popen(base + ["--out", str(td / "k")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log = td / "k" / "log.jsonl"
        deadline = time.time() + 300
        while time.time() < deadline:
            n = sum(1 for _ in open(log)) if log.exists() else 0
            if (td / "k" / "latest").exists() and n >= a.save_every + 2:
                break
            time.sleep(0.05)
        victim.send_signal(signal.SIGKILL); victim.wait()
        killed_at = sum(1 for _ in open(log))
        r = subprocess.run(base + ["--out", str(td / "k"), "--resume"], check=True, capture_output=True, text=True)
        ref, got = losses(td / "ref" / "log.jsonl"), losses(log)
        bad = [s for s in ref if got.get(s) != ref[s]]
        print(f"kv={a.kv} steps={a.steps} killed after {killed_at} logged steps; {r.stdout.strip().splitlines()[0]}")
        print("PASS: bit-identical loss curve after SIGKILL + resume" if not bad and len(got) == len(ref)
              else f"FAIL: {len(bad)} mismatching steps, e.g. {bad[:5]}")
        sys.exit(0 if not bad and len(got) == len(ref) else 1)


if __name__ == "__main__":
    main()
