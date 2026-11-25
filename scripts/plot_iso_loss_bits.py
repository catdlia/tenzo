#!/usr/bin/env python3
"""Figure for scripts/analyze_iso_loss_bits.py (reads its CSVs only)."""
import csv, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/2026-11-25_coordinator/iso_loss_bits")
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
LANG = {"en": "#257de0", "uk": "#f47230"}
FAM = {"token_r": dict(ls="-", marker="o", name="token window r"),
       "char_C": dict(ls="--", marker="s", name="character window C")}
TARGET = 0.005
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": "#bdbdbd", "axes.linewidth": 0.8})

curves = list(csv.DictReader(open(D / "iso_loss_curves.csv")))
iso = list(csv.DictReader(open(D / "iso_loss_bits.csv")))

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)

for fam, st in FAM.items():
    for L in ("en", "uk"):
        rows = sorted([r for r in curves if r["family"] == fam and r["lang"] == L], key=lambda r: float(r["bits_at_512"]))
        x = [float(r["bits_at_512"]) for r in rows]
        y = [100 * float(r["rel"]) for r in rows]
        lo = [100 * (float(r["rel"]) - float(r["rel_lo"])) for r in rows]
        hi = [100 * (float(r["rel_hi"]) - float(r["rel"])) for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], color=LANG[L], ls=st["ls"], lw=2 if fam == "token_r" else 1.4,
                    marker=st["marker"], ms=5, mec="white", mew=1.0, elinewidth=0.8, capsize=0, alpha=1 if fam == "token_r" else 0.75,
                    label=f"{L.upper()} · {st['name']}")
        if fam == "token_r":
            for r, xi, yi in zip(rows, x, y):
                if r["policy"].endswith(("r8", "r32", "r96")):
                    ax.annotate(r["policy"].split("s4")[1], (xi, yi), xytext=(4, 6), textcoords="offset points", fontsize=7, color=INK2)

ax.axhline(100 * TARGET, color="#9c9c9c", lw=0.8, ls=":")
ax.axhline(0, color="#bdbdbd", lw=0.8)
row = next(r for r in iso if r["family"] == "token_r" and abs(float(r["target_rel"]) - TARGET) < 1e-9)
xe, xu = float(row["bits_en"]), float(row["bits_uk"])
ax.annotate("", xy=(xu, 100 * TARGET), xytext=(xe, 100 * TARGET),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.0, shrinkA=0, shrinkB=0))
ax.text((xe + xu) / 2, 100 * TARGET + 0.06,
        f"+{float(row['tax_bits']):.2f} bits/elem [{float(row['tax_bits_lo']):.2f}, {float(row['tax_bits_hi']):.2f}]\n"
        f"recent INT8 slots {float(row['r_eff_en']):.0f} → {float(row['r_eff_uk']):.0f} (×{float(row['slots_ratio_uk_en']):.1f}), "
        f"chars ×{float(row['chars_ratio_uk_en']):.1f}",
        ha="center", va="bottom", fontsize=7.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), zorder=5)
ax.text(ax.get_xlim()[1], 100 * TARGET, " 0.5% target", va="center", ha="left", fontsize=7, color=INK2, clip_on=False)
ax.set_xlabel("effective KV bits per element at ctx 512 (S3 ternary + INT8 sinks/recent)")
ax.set_ylabel("relative loss, % of base NLL")
ax.set_title("FLORES-200 · BitNet-b1.58-2B-4T · S3: budget for equal loss, EN vs UK", fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig(D / "iso_loss_bits.png", dpi=160)
print("wrote", D / "iso_loss_bits.png")
