#!/usr/bin/env python3
"""Drive tools/reference/bitnet_ref_dump (libllama / bitnet.cpp i2_s) over prompts.json.
Input is ALIGNED: the llama run consumes the HF token ids (results/reference/hf/<id>.ids.txt); llama's own tokenisation of the
text is stored separately as <id>.ids_native.txt so tokenisation differences are a finding, not a hidden confound."""
import json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BIN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ov/bitnet_ref_dump"
GGUF = ROOT / "BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
out = ROOT / "results/reference/llama"; out.mkdir(parents=True, exist_ok=True)
for p in json.loads((ROOT / "tools/reference/prompts.json").read_text()):
    pid = p["id"]
    nat = subprocess.run([BIN, "tokenize", str(GGUF), p["text"]], capture_output=True, text=True, check=True).stdout
    line = [l for l in nat.splitlines() if l.startswith("TOKENS:")][0]
    (out / f"{pid}.ids_native.txt").write_text(line.split(":", 1)[1].strip() + "\n")
    hf_ids = ROOT / f"results/reference/hf/{pid}.ids.txt"
    shutil.copy(hf_ids, out / f"{pid}.ids.txt")
    r = subprocess.run([BIN, "prompt", str(GGUF), str(hf_ids), str(out / pid), "32"], capture_output=True, text=True)
    print(pid, r.returncode, r.stdout.strip()[:80], r.stderr.strip()[-120:] if r.returncode else "", flush=True)
