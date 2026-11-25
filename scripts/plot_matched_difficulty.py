#!/usr/bin/env python3
"""Figures for scripts/analyze_matched_difficulty.py (reads its CSVs only)."""
import csv, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/2026-11-25_coordinator/matched_difficulty")
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
LANG = {"en": "#257de0", "uk": "#f47230", "de": "#00b894", "pl": "#f2a600", "bg": "#ee7fac"}
MODEL = {"bitnet": "#257de0", "eurollm": "#f47230"}
NAME = {"bitnet": "BitNet-b1.58-2B-4T", "eurollm": "EuroLLM-1.7B", "qwen": "Qwen2.5-1.5B"}
LABELS = ["<0.05", "0.05-0.3", "0.3-1", "1-2", "2-4", "4-7", ">=7"]
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": "#bdbdbd", "axes.linewidth": 0.8})

def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

bins = list(csv.DictReader(open(D / "matched_bins.csv")))
summ = list(csv.DictReader(open(D / "matched_difficulty.csv")))

# Figure 1: per-bin extra NLL, EN vs UK, S3+s4r32
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=False)
for ax, (corpus, model) in zip(axes, [("flores32", "bitnet"), ("flores32", "eurollm"), ("ntrex", "bitnet"), ("ntrex", "eurollm")]):
    ends = {}
    for L in ("en", "uk"):
        rows = [r for r in bins if r["corpus"] == corpus and r["model"] == model and r["scheme"] == "S3_s4r32" and r["lang"] == L]
        y = [float(r["mean_d"]) for r in rows]
        ax.plot(range(len(y)), y, color=LANG[L], lw=2, marker="o", ms=5, mec="white", mew=1.2, label=L.upper())
        ends[L] = y[-1]
    lo_, hi_ = ax.get_ylim(); close = abs(ends["en"] - ends["uk"]) < 0.06 * (hi_ - lo_)
    for L in ("en", "uk"):
        dy = (5 if ends[L] >= ends["en" if L == "uk" else "uk"] else -5) if close else 0
        ax.annotate(L.upper(), (len(LABELS) - 1, ends[L]), xytext=(6, dy), textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.axhline(0, color="#9c9c9c", lw=0.8)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=7.5)
    ax.set_title(f"{'FLORES-200' if corpus == 'flores32' else 'NTREX-128'} · {NAME[model]}", fontsize=9, color=INK)
    ax.set_xlabel("base NLL of the token (nats)"); style(ax)
axes[0].set_ylabel("extra NLL from S3+s4r32 KV (nats/token)")
axes[0].legend(frameon=False, loc="upper right", fontsize=8)
fig.suptitle("At equal base difficulty, Ukrainian tokens lose more than English ones — far more in BitNet than in EuroLLM",
             fontsize=10, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(D / "matched_bins.png", dpi=160); plt.close(fig)

# Figure 2: matched-difficulty UK/EN-style ratio (b<4) per language vs cost ratio kappa, FLORES-5
OFF = {("bitnet", "uk"): (-8, 7, "right"), ("bitnet", "bg"): (8, -4, "left"), ("eurollm", "de"): (-8, -7, "right"),
       ("eurollm", "pl"): (8, -6, "left"), ("eurollm", "bg"): (-8, 5, "right"), ("eurollm", "uk"): (8, 6, "left")}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for model in ("bitnet", "eurollm"):
    rows = [r for r in summ if r["corpus"] == "flores5" and r["model"] == model and r["scheme"] == "S3_s4r32"
            and r["variant"] == "lt4" and r["lang"] != "en"]
    for r in rows:
        k, y, lo, hi = float(r["kappa"]), float(r["ratio_vs_en"]), float(r["ratio_lo"]), float(r["ratio_hi"])
        ax.errorbar(k, y, yerr=[[y - lo], [hi - y]], fmt="o", ms=6, color=MODEL[model], mec="white", mew=1.2, elinewidth=1.2, capsize=0)
        dx, dy, ha = OFF.get((model, r["lang"]), (7, -3, "left"))
        ax.annotate(r["lang"], (k, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK, ha=ha, va="center")
    ax.plot([], [], "o", color=MODEL[model], label=NAME[model])
ax.axhline(1, color="#9c9c9c", lw=0.8)
ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"])
ax.set_xlabel("κ = base nats on the same content, relative to English")
ax.set_ylabel("extra NLL vs English at equal difficulty (×, b<4)")
ax.set_title("FLORES-200, 48 parallel windows, S3+s4r32 — 95% bootstrap CI", fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left"); style(ax)
fig.tight_layout(); fig.savefig(D / "matched_ratio_vs_kappa.png", dpi=160); plt.close(fig)
print("ok")
