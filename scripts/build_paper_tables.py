#!/usr/bin/env python3
"""Paper tables from the result files (no number is typed by hand): results/paper/tables.md and tables.tex.

Every table is followed by: source files, git commit, generation date, machine (CPU, governor), load average at generation time and,
where the measurement files carry it, the load average at measurement time. A table whose inputs do not exist yet is replaced by a
"not available" note instead of being filled with anything. Speed rows are only taken from files where every row has load1 < 1.0
(the clean-machine rule).  Run: python3 scripts/build_paper_tables.py   (make paper-tables)
"""
import csv, glob, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "paper"
R19, R20 = ROOT / "results" / "2026-11-19_overnight", ROOT / "results" / "2026-11-20_overnight"
P2 = R20 / "part2"
sys.path.insert(0, str(ROOT / "tools" / "reference"))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return "unknown"


COMMIT = sh("git rev-parse --short HEAD")
DIRTY = " (+uncommitted changes)" if sh("git status --porcelain -- results scripts tools src | head -1") else ""
CPU = sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip()
GOV = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
LOAD = sh("cut -d' ' -f1-3 /proc/loadavg")
NOW = time.strftime("%Y-%m-%d %H:%M")


def footer(sources, extra=""):
    srcs = ", ".join("`%s`" % os.path.relpath(s, ROOT) for s in sources)
    return ("Source: %s. Commit %s%s, generated %s on %s, governor `%s`, load average at generation %s.%s"
            % (srcs, COMMIT, DIRTY, NOW, CPU, GOV, LOAD, (" " + extra) if extra else ""))


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def rd_scores(p):
    W = {}
    for line in open(p):
        if line.startswith("w "):
            _, i, n, x = line.split(); W[int(i)] = (int(n), float(x))
    return W


def ppl_over(W, ids):
    return math.exp(sum(W[i][1] for i in ids) / sum(W[i][0] for i in ids))


class Table:
    def __init__(self, title, header, rows, sources, note="", extra_footer=""):
        self.title, self.header, self.rows, self.sources, self.note, self.extra = title, header, rows, sources, note, extra_footer

    def md(self):
        out = ["### " + self.title, ""]
        if self.rows:
            out += ["| " + " | ".join(self.header) + " |", "|" + "---|" * len(self.header)]
            out += ["| " + " | ".join(str(c) for c in r) + " |" for r in self.rows]
        else:
            out.append("*Not available: %s*" % (self.note or "input files do not exist yet"))
        if self.rows and self.note:
            out += ["", self.note]
        out += ["", "<sub>" + footer(self.sources, self.extra) + "</sub>", ""]
        return "\n".join(out)

    def tex(self):
        esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("+", "+").replace("#", r"\#")
        if not self.rows:
            return "%% %s: not available\n" % self.title
        cols = "l" + "r" * (len(self.header) - 1)
        lines = [r"\begin{table}[t]\centering\small", r"\caption{%s}" % esc(self.title), r"\begin{tabular}{%s}\toprule" % cols,
                 " & ".join(esc(h) for h in self.header) + r" \\ \midrule"]
        lines += [" & ".join(esc(c) for c in r) + r" \\" for r in self.rows]
        lines += [r"\bottomrule\end{tabular}", r"\\[2pt]{\scriptsize %s}" % esc(footer(self.sources, self.extra)), r"\end{table}", ""]
        return "\n".join(lines)


def f3(x, d=3): return ("%." + str(d) + "f") % float(x)


# ----------------------------------------------------------------------------------------------------------------- tables
def t_ppl_v2():
    en, uk = R19 / "ppl_kv_v2.csv", R19 / "ppl_kv_v2_uk.csv"
    if not (en.exists() and uk.exists()):
        return Table("KV-mode perplexity v2", [], [], [en, uk])
    E, U = {r["kv_mode"]: r for r in read_csv(en)}, {r["kv_mode"]: r for r in read_csv(uk)}
    rows = []
    for m in E:
        e, u = E[m], U.get(m)
        rows.append([m, f3(e["ppl"], 2), "%s [%s, %s]" % (f3(e["ratio_vs_golden"], 3), f3(e["ratio_vs_golden_ci_lo"], 3), f3(e["ratio_vs_golden_ci_hi"], 3)),
                     f3(u["ppl"], 1) if u else "-",
                     "%s [%s, %s]" % (f3(u["ratio_vs_golden"], 3), f3(u["ratio_vs_golden_ci_lo"], 3), f3(u["ratio_vs_golden_ci_hi"], 3)) if u else "-"])
    meta = (R19 / "meta.txt")
    return Table("Perplexity of the KV modes, ratio to the fp32 golden reference (BitNet-b1.58-2B-4T)",
                 ["KV mode", "WikiText-2 PPL", "ratio to golden [95% CI]", "Kobza-UK PPL", "ratio to golden [95% CI]"], rows, [en, uk],
                 "WikiText-2: 16 x 511 scored tokens; Kobza Ukrainian: 8 x 511. Golden fp32 PPL: %s (EN), %s (UK). Deterministic (load-independent); computed in session with %s."
                 % (f3(E[next(iter(E))]["golden_fp32_ppl"], 2), f3(U[next(iter(U))]["golden_fp32_ppl"], 1), os.path.relpath(meta, ROOT)))


def screening_points():
    """(round, scheme, lang, dppl%) for the 4-window screenings, computed from score files; golden on the same windows."""
    import kv_ptq
    GE, GU = rd_scores(ROOT / "results/reference/golden/nll_golden_float32.txt"), rd_scores(ROOT / "results/reference/golden/nll_golden_float32_uk.txt")
    pts = {}
    srcs = [("r1", "en", R19 / "kv_ptq/screen4", GE), ("r2", "en", R20 / "kv_ptq2/screen_en4", GE), ("r2", "uk", R20 / "kv_ptq2/screen_uk4", GU),
            ("r1", "uk", R19 / "kv_ptq/final_uk8", GU)]
    for rnd, lang, d, G in srcs:
        for f in sorted(glob.glob(str(d / "*.txt"))):
            if "progress" in f: continue
            name = os.path.basename(f)[:-4]
            name = ("K8V:" + name[4:]) if name.startswith("K8V_") else ("KV8:" + name[4:]) if name.startswith("KV8_") else (name.replace("_s", "+s", 1) if "_s" in name and name.split("_s")[1][:1].isdigit() else name)
            T = rd_scores(f); ids = [i for i in sorted(T) if i < 4 and i in G]
            if len(ids) < 4: continue
            try:
                bits = kv_ptq.bits(name, 512)
            except Exception:
                continue
            pts[(name, lang)] = (rnd, bits, 100 * (ppl_over(T, ids) / ppl_over(G, ids) - 1))
    return pts


def t_pareto():
    try:
        pts = screening_points()
    except Exception as e:
        return Table("Ternary-KV PTQ screening", [], [], [], "cannot compute: %s" % e)
    names = sorted({n for n, _ in pts}, key=lambda n: (pts.get((n, "en"), pts.get((n, "uk")))[1], n))
    rows, front = [], []
    for n in names:
        en, uk = pts.get((n, "en")), pts.get((n, "uk"))
        bits = (en or uk)[1]
        rounds = "/".join(sorted({x[0] for x in (en, uk) if x}))
        rows.append([n, f3(bits, 2), ("%+.1f%%" % en[2]) if en else "-", ("%+.1f%%" % uk[2]) if uk else "-", rounds])
    # Pareto front on (bits, EN dPPL)
    best = 1e9
    for r in sorted(rows, key=lambda r: float(r[1])):
        if r[2] != "-" and float(r[2].rstrip("%")) < best:
            best = float(r[2].rstrip("%")); front.append(r[0])
    rows = [r + ["yes" if r[0] in front else ""] for r in rows]
    return Table("Ternary-KV post-training quantization, screening on 4 windows per language (rounds r1, r2)",
                 ["scheme", "bits/elem @ctx512", "EN dPPL", "UK dPPL", "round", "Pareto (EN)"], rows,
                 [R19 / "kv_ptq/screen4", R19 / "kv_ptq/final_uk8", R20 / "kv_ptq2/screen_en4", R20 / "kv_ptq2/screen_uk4",
                  ROOT / "results/reference/golden/nll_golden_float32.txt"],
                 "dPPL is relative to golden fp32-KV on the same 4 windows (no CI at 4 windows; full-set results in the next table). Bits include fp16 scales and the INT8 windows (tools/reference/kv_ptq.py).")


def t_full():
    cands = [("S3+s4r32 (uncalibrated)", R19 / "kv_ptq/final_en16/S3_s4r32.txt", P2 / "kv_final/uk32/S3_s4r32.txt"),
             ("S13+s4r32 (K INT4, V ternary)", P2 / "kv_final/en16/S13_s4r32.txt", P2 / "kv_final/uk32/S13_s4r32.txt")]
    csvp = P2 / "kv_calib.csv"
    rows = []
    if csvp.exists():
        by = {}
        for r in read_csv(csvp): by.setdefault(r["scheme"], {})[r["lang"]] = r
        for s, d in by.items():
            if "en" in d and "uk" in d:
                rows.append([s, f3(d["en"]["ratio"], 3) + " [%s, %s]" % (f3(d["en"]["ratio_ci_lo"], 3), f3(d["en"]["ratio_ci_hi"], 3)),
                             f3(d["uk"]["ratio"], 3) + " [%s, %s]" % (f3(d["uk"]["ratio_ci_lo"], 3), f3(d["uk"]["ratio_ci_hi"], 3)),
                             "%+.4f" % float(d["en"]["dnll"]), "%+.4f" % float(d["uk"]["dnll"])])
    return Table("Full-set results of the best ternary-KV schemes (16 EN + 32 UK windows), incl. calibration",
                 ["scheme", "EN PPL ratio to golden [CI]", "UK PPL ratio to golden [CI]", "EN dNLL (nats/token)", "UK dNLL"], rows, [csvp],
                 "Source rows are produced by scripts/analyze_kv_calib.py (paired bootstrap over windows).")


def t_disparity():
    rows = []
    srcs = []
    for label, p in (("BitNet-b1.58-2B-4T", P2 / "kv_calib.csv"), ("Qwen2.5-1.5B", P2 / "second_model.csv")):
        if not p.exists(): continue
        srcs.append(p)
        seen = {}
        for r in read_csv(p): seen.setdefault(r["scheme"], {})[r["lang"]] = r
        for s, d in seen.items():
            if "en" in d and "uk" in d:
                en, uk = d["en"], d["uk"]
                rows.append([label, s, f3(en["dnll"], 4), f3(uk["dnll"], 4), f3(uk["base_nll"], 2) + " vs " + f3(en["base_nll"], 2),
                             "%+.4f [%+.4f, %+.4f]" % (float(en["pooled_c_uk"]), float(en["pooled_c_ci_lo"]), float(en["pooled_c_ci_hi"]))])
    return Table("Language disparity of KV quantization: dNLL per token and the difficulty-adjusted Ukrainian effect",
                 ["model", "scheme", "EN dNLL", "UK dNLL", "base NLL UK vs EN", "c_UK at equal difficulty [95% CI]"], rows, srcs,
                 "c_UK: coefficient of the UK indicator in dNLL = a + b*NLL0 + c*UK over windows of both languages (scripts/analyze_kv_calib.py). The base NLL ranges of the two languages hardly overlap, so c_UK is an adjusted group difference, not a per-window causal effect.")


def t_parity():
    files = sorted(glob.glob(str(R19 / "parity_v2/parity_*.json")))
    if not files:
        return Table("Parity", [], [], [])
    rows = []
    for f in files:
        d = json.load(open(f)); kv = d["kv_quant"]
        for src in ("golden", "hf"):
            ms = [pr[src] for pr in d["prompts"].values() if isinstance(pr.get(src), dict) and "cosine_similarity" in pr[src]]
            if not ms: continue
            rows.append([kv, src, len(ms), "%d/%d" % (sum(m["gate_cosine"] for m in ms), len(ms)), "%d/%d" % (sum(m["gate_max_abs"] for m in ms), len(ms)),
                         "%d/%d" % (sum(m["top1_match"] for m in ms), len(ms)), "%.2f-%.2f" % (min(m["max_abs_diff"] for m in ms), max(m["max_abs_diff"] for m in ms))])
    # the two references against each other
    gl, hl = [], []
    for pid in ("p1_fact_en", "p2_tech_en", "p3_code", "p4_ukrainian", "p5_long_ctx", "p6_weird", "p7_math_en"):
        a = np.fromfile(ROOT / f"results/reference/golden/{pid}.logits.f32", np.float32).astype(float)
        b = np.fromfile(ROOT / f"results/reference/hf/{pid}.logits.f32", np.float32).astype(float)
        gl.append((a @ b / np.linalg.norm(a) / np.linalg.norm(b), abs(a - b).max(), a.argmax() == b.argmax()))
    rows.append(["(golden fp32 vs HF bf16)", "-", len(gl), "%d/%d" % (sum(c > 0.999 for c, _, _ in gl), len(gl)), "%d/%d" % (sum(m <= 0.25 for _, m, _ in gl), len(gl)),
                 "%d/%d" % (sum(t for _, _, t in gl), len(gl)), "%.2f-%.2f" % (min(m for _, m, _ in gl), max(m for _, m, _ in gl))])
    return Table("First-step logit parity of the engine (7 prompts); gates: cosine > 0.999 and max_abs <= 0.25",
                 ["KV mode", "reference", "prompts", "cosine gate", "max_abs gate", "top-1 equal", "max_abs range"], rows,
                 files + [ROOT / "results/reference/golden", ROOT / "results/reference/hf"], "The last row compares the two references with each other: the max_abs gate is not met even there.")


def t_speed():
    files = sorted(glob.glob(str(P2 / "bench/*.csv")))
    clean, srcs = [], []
    for f in files:
        rows = read_csv(f)
        if rows and "load1" in rows[0] and all(float(r["load1"]) < 1.0 for r in rows if r.get("load1")):
            clean.append((f, rows)); srcs.append(Path(f))
    if not clean:
        return Table("Speed", [], [], [Path(f) for f in files], "no clean-machine measurements yet (queue: scripts/pending_benchmarks.sh)")
    out = []
    for f, rows in clean:
        key = [k for k in ("variant", "engine") if k in rows[0]][0]
        for v in sorted({r[key] for r in rows}):
            sel = [r for r in rows if r[key] == v]
            pp = [float(r["pp_tps"]) for r in sel if r.get("pp_tps")]; tg = [float(r["tg_tps"]) for r in sel if r.get("tg_tps")]
            out.append([os.path.basename(f), v, len(sel), f3(np.median(pp), 2) if pp else "-", f3(np.median(tg), 2) if tg else "-",
                        "%.2f-%.2f" % (min(float(r["load1"]) for r in sel), max(float(r["load1"]) for r in sel))])
    return Table("Speed on a clean machine (medians)", ["file", "variant", "runs", "prefill tok/s", "decode tok/s", "load1 range"], out, srcs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = [t_ppl_v2(), t_pareto(), t_full(), t_disparity(), t_parity(), t_speed()]
    (OUT / "tables.md").write_text("# Paper tables (generated by scripts/build_paper_tables.py)\n\n" + "\n".join(t.md() for t in tables))
    (OUT / "tables.tex").write_text("% generated by scripts/build_paper_tables.py; requires booktabs\n" + "\n".join(t.tex() for t in tables))
    print("wrote", OUT / "tables.md", OUT / "tables.tex")
    for t in tables:
        print("-", t.title, "->", "%d rows" % len(t.rows))


if __name__ == "__main__":
    main()
