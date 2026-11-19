#!/usr/bin/env python3
"""RSS / peak-RSS table from the BENCH_FAIR lines written by tenzo-inference (rss_kb = VmRSS at exit, hwm_kb = VmHWM).
Usage: summarize_rss.py OUT.md LABEL=lines.txt [LABEL=lines.txt ...]   (lines: '<kv> ctx=<n> BENCH_FAIR,...')"""
import re, sys
out, specs = sys.argv[1], sys.argv[2:]
data = {}
for sp in specs:
    label, path = sp.split("=", 1)
    for line in open(path):
        m = re.match(r"(\S+) ctx=(\d+) BENCH_FAIR,.*rss_kb=(\d+),hwm_kb=(\d+)", line)
        if m: data[(label, m.group(1), int(m.group(2)))] = (int(m.group(3)) / 1024, int(m.group(4)) / 1024)
labels = [s.split("=")[0] for s in specs]
keys = sorted({(k, c) for (_, k, c) in data}, key=lambda t: (t[1], t[0]))
L = ["# Engine RSS: steady state (VmRSS) and peak (VmHWM)", "",
     "Measured with `tenzo-inference --bench-pp 16 --bench-tg 128 -c <ctx>` (model models/bitnet-1.58b-v2, int8 LM head, I2_S GEMV); RSS at exit after 16 prompt + 128 generated tokens.", "",
     "| kv mode | ctx | " + " | ".join("%s RSS MB | %s HWM MB" % (l, l) for l in labels) + " |", "|---|---|" + "---|---|" * len(labels)]
for k, c in keys:
    cells = []
    for l in labels:
        v = data.get((l, k, c)); cells += ["%.0f" % v[0], "%.0f" % v[1]] if v else ["-", "-"]
    L.append("| %s | %d | %s |" % (k, c, " | ".join(cells)))
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
