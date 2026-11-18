#!/usr/bin/env python3
"""Ukrainian evaluation windows from the first rows of Goader/kobza (fetched via the datasets-server first-rows API, 39 documents,
~180 KB JSON, saved as results/reference/kobza_first_rows.json). No randomness: documents are concatenated with a blank line in
dataset order and cut into consecutive non-overlapping windows [BOS] + 511 tokens (same protocol as make_windows.py)."""
import argparse, json, os, hashlib
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--len", type=int, default=512)
ap.add_argument("--src", default="results/reference/kobza_first_rows.json"); ap.add_argument("--out", default="results/reference")
a = ap.parse_args()
rows = json.load(open(a.src))["rows"]
text = "\n\n".join(r["row"]["text"] for r in rows)
tok = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
ids = tok(text, add_special_tokens=False).input_ids
body = a.len - 1
assert len(ids) >= a.n * body, f"only {len(ids)} tokens"
with open(os.path.join(a.out, "windows_uk.ids"), "w") as f:
    for w in range(a.n):
        f.write(" ".join(map(str, [tok.bos_token_id] + ids[w * body:(w + 1) * body])) + "\n")
json.dump({"source": "Goader/kobza first rows via datasets-server", "src_sha256": hashlib.sha256(open(a.src, "rb").read()).hexdigest(),
           "n_docs": len(rows), "n_text_tokens": len(ids), "windows": a.n, "window_len": a.len}, open(os.path.join(a.out, "windows_uk.json"), "w"), indent=1)
print("text tokens", len(ids), "chars", len(text), "windows", a.n)
