#!/usr/bin/env python3
"""Build the fixed WikiText-2 evaluation windows (protocol shared by every engine).

Protocol: tokenise the whole raw test split (concatenated rows, as in llama.cpp / HF PPL scripts) with the
canonical HF tokenizer WITHOUT special tokens; draw 16 non-overlapping start offsets with numpy RandomState(seed 0);
window = [BOS] + 511 consecutive text tokens (512 total). Every window is scored on its 511 predictions
(all tokens except the first = BOS).  Outputs (text, committed): windows.ids (one window per line), windows.json.
"""
import argparse, json, os, hashlib
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="/tmp/ov/data/wt2_test.raw")
ap.add_argument("--out", default="results/reference")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--len", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
text = open(a.raw, encoding="utf-8").read()
ids = tok(text, add_special_tokens=False).input_ids
bos = tok.bos_token_id
body = a.len - 1
rs = np.random.RandomState(a.seed)
starts = []
while len(starts) < a.n:
    s = int(rs.randint(0, len(ids) - body))
    if all(abs(s - t) >= body for t in starts):
        starts.append(s)
starts.sort()
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "windows.ids"), "w") as f:
    for s in starts:
        f.write(" ".join(str(x) for x in [bos] + ids[s:s + body]) + "\n")
meta = {"raw_sha256": hashlib.sha256(open(a.raw, "rb").read()).hexdigest(), "n_text_tokens": len(ids), "seed": a.seed,
        "window_len": a.len, "starts": starts, "bos": bos, "tokenizer": "microsoft/bitnet-b1.58-2B-4T (HF fast, add_special_tokens=False)"}
json.dump(meta, open(os.path.join(a.out, "windows.json"), "w"), indent=1)
print("text tokens", len(ids), "starts", starts)
