#!/usr/bin/env python3
"""Diff two dump dirs (golden vs HF hooks, or golden vs engine): per tensor rel-L2, max_abs, cosine; plus logits files."""
import json, os, sys, glob
import numpy as np
a, b = sys.argv[1], sys.argv[2]
ma = {(t["layer"], t["name"], t["pos"]): t["file"] for t in json.load(open(os.path.join(a, "manifest.json")))["tensors"]}
mb = {(t["layer"], t["name"], t["pos"]): t["file"] for t in json.load(open(os.path.join(b, "manifest.json")))["tensors"]}
print("| layer | tensor | pos | rel_l2 | max_abs | cos |\n|---|---|---|---|---|---|")
for k in [k for k in ma if k in mb]:
    x = np.fromfile(os.path.join(a, ma[k]), np.float32).astype(np.float64)
    y = np.fromfile(os.path.join(b, mb[k]), np.float32).astype(np.float64)
    if x.size != y.size:
        print("| %s | %s | %d | size %d vs %d | | |" % (k[0], k[1], k[2], x.size, y.size)); continue
    print("| %s | %s | %d | %.3e | %.3e | %.6f |" % (k[0], k[1], k[2], np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30),
          abs(x - y).max(), x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30)))
for la in sorted(glob.glob(os.path.join(sys.argv[3] if len(sys.argv) > 3 else a, "*.logits.f32"))):
    lb = os.path.join(sys.argv[4] if len(sys.argv) > 4 else b, os.path.basename(la))
    if os.path.exists(lb):
        x = np.fromfile(la, np.float32).astype(np.float64); y = np.fromfile(lb, np.float32).astype(np.float64)
        print("logits %s cos %.6f max_abs %.3f top1 %d/%d" % (os.path.basename(la), x @ y / np.linalg.norm(x) / np.linalg.norm(y),
              abs(x - y).max(), x.argmax(), y.argmax()))
