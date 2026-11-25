#!/usr/bin/env python3
"""Figure for scripts/analyze_disparity_decomposition.py (reads its CSV only)."""
import csv, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/2026-11-25_coordinator/decomposition")
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
COL = {"fertility": "#257de0", "composition": "#00b894", "fragility": "#f47230"}
LBL = {"fertility": "fertility (more tokens at EN's damage rate)",
       "composition": "composition (other difficulty mix)",
       "fragility": "fragility (more damage at equal difficulty)"}
NAME = {"bitnet": "BitNet-b1.58-2B-4T (ternary)", "eurollm": "EuroLLM-1.7B", "qwen": "Qwen2.5-1.5B"}
CORP = {"flores32": "FLORES-32", "flores5": "FLORES-48", "ntrex": "NTREX-24"}
plt.rcParams.update({"font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": "#bdbdbd", "axes.linewidth": 0.8})

rows = [r for r in csv.DictReader(open(D / "decomposition.csv")) if r["scheme"] == "S3_s4r32" and r["variant"] == "all"]
order = [("flores32", "uk"), ("flores5", "de"), ("flores5", "pl"), ("flores5", "uk"), ("flores5", "bg"), ("ntrex", "uk")]
models = ["bitnet", "eurollm", "qwen"]
per = {m: [r for key in order for r in rows if r["model"] == m and (r["corpus"], r["lang"]) == key] for m in models}
fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.2), sharex=True,
                         gridspec_kw=dict(height_ratios=[len(per[m]) for m in models], hspace=0.35))
for ax, m in zip(axes, models):
    rr = per[m]
    ys = list(range(len(rr)))[::-1]
    for y, r in zip(ys, rr):
        sh = {c: 100 * float(r[f"{c}_share"]) for c in COL}
        pos = 0.0
        for c in ("fertility", "fragility"):
            v = sh[c]
            if v >= 0:
                ax.barh(y, v, left=pos, color=COL[c], height=0.62, edgecolor="white", linewidth=1.5)
                if v >= 12:
                    ax.text(pos + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white")
                pos += v
            else:
                ax.barh(y, v, left=0, color=COL[c], height=0.62, edgecolor="white", linewidth=1.5)
        v = sh["composition"]
        neg_left = min(0.0, sh["fertility"])
        ax.barh(y, v, left=neg_left if v < 0 else pos, color=COL["composition"], height=0.62, edgecolor="white", linewidth=1.5)
        if abs(v) >= 12:
            ax.text((neg_left if v < 0 else pos) + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white")
        ax.text(162, y, f"D = {float(r['D_per_window']):+.1f}", va="center", ha="left", fontsize=7, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{CORP[r['corpus']]} · {r['lang']}" for r in rr])
    ax.axvline(0, color="#9c9c9c", lw=0.8)
    ax.axvline(100, color="#bdbdbd", lw=0.8, ls=":")
    ax.set_title(NAME[m], fontsize=8.5, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlim(-70, 160)
axes[-1].set_xlabel("share of the extra KV-quantization loss vs English for the same content (S3+s4r32), % of D")
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in COL]
fig.legend(handles, [LBL[c] for c in COL], loc="lower center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.005))
fig.text(0.985, 0.955, "D: extra nats per window", ha="right", fontsize=7, color=INK2)
fig.subplots_adjust(left=0.17, right=0.86, top=0.94, bottom=0.12)
fig.savefig(D / "decomposition.png", dpi=160)
print("wrote", D / "decomposition.png")
