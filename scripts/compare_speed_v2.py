#!/usr/bin/env python3
"""Speed before/after fix 2 (8-bit activation LUT kernel). Medians over repetitions from two run_e2e_fair.py CSVs.
'before' = results/2026-11-18_overnight/e2e_fair.csv (old 6-bit kernel; the theta fix does not change speed),
'after'  = tonight's CSV. Machine drift is estimated from the bitnet.cpp f16 control point (prompt 128) present in both:
drift = after/before of that engine; 'drift-corrected ratio' = (tenzo after/before) / drift."""
import csv, statistics, sys
before, after, out = sys.argv[1], sys.argv[2], sys.argv[3]


def load(p):
    D = {}
    for r in csv.DictReader(open(p)):
        if not r.get("tg_tps"):
            continue
        k = (r["engine"], r["kv"], int(r["prompt_len"]))
        D.setdefault(k, {"pp": [], "tg": [], "pg": [], "rss": []})
        for m, c in (("pp", "pp_tps"), ("tg", "tg_tps"), ("pg", "pg_tps"), ("rss", "rss_mb")):
            if r.get(c):
                D[k][m].append(float(r[c]))
    return {k: {m: (statistics.median(v) if v else None) for m, v in d.items()} for k, d in D.items()}


B, A = load(before), load(after)
ctl = ("bitnet.cpp", "f16", 128)
drift = {m: (A[ctl][m] / B[ctl][m]) if ctl in A and ctl in B and A[ctl][m] and B[ctl][m] else None for m in ("pp", "tg", "pg")}
L = ["# Speed after fix 2 (8-bit activations, two-table LUT kernel)", "",
     "Medians of 5 repetitions, 4 threads on cpus 0-3, gen 128 (scripts/run_e2e_fair.py). before = 2026-11-18 CSV (6-bit kernel), "
     "after = tonight. Control point bitnet.cpp f16 prompt 128 measures machine drift between the days.", "",
     "| control point bitnet.cpp f16, prompt 128 | before | after | after/before (drift) |", "|---|---|---|---|"]
for m in ("pp", "tg", "pg"):
    if ctl in A and ctl in B:
        L.append("| %s tok/s | %.2f | %.2f | %.3f |" % (m, B[ctl][m], A[ctl][m], drift[m]))
L += ["", "| engine | kv | prompt | metric | before tok/s | after tok/s | after/before | drift-corrected |", "|---|---|---|---|---|---|---|---|"]
for k in sorted(A):
    if k[0] != "tenzo" or k not in B:
        continue
    for m in ("pp", "tg", "pg"):
        b, a = B[k][m], A[k][m]
        r = a / b
        dc = r / drift[m] if drift.get(m) else float("nan")
        L.append("| %s | %s | %d | %s | %.2f | %.2f | %.3f | %.3f |" % (k[0], k[1], k[2], m, b, a, r, dc))
L += ["", "| engine | kv | prompt | RSS MB before | RSS MB after |", "|---|---|---|---|---|"]
for k in sorted(A):
    if k in B and B[k]["rss"] and A[k]["rss"]:
        L.append("| %s | %s | %d | %.0f | %.0f |" % (k[0], k[1], k[2], B[k]["rss"], A[k]["rss"]))
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
