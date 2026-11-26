#!/usr/bin/env python3
"""Plot 10-language (9 non-EN) H2 fertility-vs-competence regression & responses."""

from __future__ import annotations

import csv
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG_DIR = ROOT / "results" / "2026-11-26_agent2" / "H2" / "regression"
OUT_PNG = ROOT / "results" / "2026-11-26_agent2" / "H2" / "fertility_vs_competence.png"

INK, INK2, GRID = "#191919", "#555555", "#e6e6e6"
MODEL_COLOR = {"bitnet": "#257de0", "eurollm": "#f47230"}
MODEL_LABEL = {"bitnet": "BitNet-b1.58-2B-4T", "eurollm": "EuroLLM-1.7B"}
plt.rcParams.update({
    "font.size": 8.5,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.edgecolor": "#bdbdbd",
    "axes.linewidth": 0.8,
})

LANG_OFF = {
    ("bitnet", "fr", "phi"): (6, -5, "left"),
    ("bitnet", "es", "phi"): (-6, 6, "right"),
    ("bitnet", "de", "phi"): (-6, -7, "right"),
    ("bitnet", "cs", "phi"): (-6, 5, "right"),
    ("bitnet", "uk", "phi"): (6, 4, "left"),
    ("bitnet", "pl", "phi"): (-6, -5, "right"),
    ("bitnet", "bg", "phi"): (-6, 6, "right"),
    ("bitnet", "hu", "phi"): (6, -4, "left"),
    ("bitnet", "el", "phi"): (-6, -6, "right"),
    ("bitnet", "fr", "kappa"): (-6, -4, "right"),
    ("bitnet", "es", "kappa"): (6, 5, "left"),
    ("bitnet", "de", "kappa"): (6, -5, "left"),
    ("bitnet", "cs", "kappa"): (-6, -6, "right"),
    ("bitnet", "uk", "kappa"): (6, -6, "left"),
    ("bitnet", "pl", "kappa"): (-6, 6, "right"),
    ("bitnet", "bg", "kappa"): (-6, 5, "right"),
    ("bitnet", "hu", "kappa"): (-6, -5, "right"),
    ("bitnet", "el", "kappa"): (-6, 7, "right"),
}


def load_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    responses = [
        ("R", "Excess rel. loss R (pp)"),
        ("ratio_lt4", "Matched-difficulty ratio (b<4)"),
        ("fragility", "Fragility term (nats / EN tok)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.2), sharex=False)
    for row_idx, (resp, ylabel) in enumerate(responses):
        langs = load_csv(REG_DIR / f"languages_flores11_S3_s4r32_{resp}.csv")
        reg = {r["model"]: r for r in load_csv(REG_DIR / f"regression_flores11_S3_s4r32_{resp}.csv")}
        for col_idx, xkey in enumerate(("phi", "kappa")):
            ax = axes[row_idx, col_idx]
            ax.grid(True, color=GRID, lw=0.7)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for m in ("bitnet", "eurollm"):
                pts = [r for r in langs if r["model"] == m]
                xs = np.array([float(r[xkey]) for r in pts])
                ys = np.array([float(r["y"]) for r in pts])
                lo = ys - np.array([float(r["y_lo"]) for r in pts])
                hi = np.array([float(r["y_hi"]) for r in pts]) - ys
                ax.errorbar(
                    xs, ys, yerr=[lo, hi], fmt="o", ms=5.5, color=MODEL_COLOR[m],
                    mec="white", mew=1.0, elinewidth=1.0, alpha=0.92, capsize=0,
                    label=MODEL_LABEL[m] if row_idx == 0 and col_idx == 0 else None,
                )
                for r, xv, yv in zip(pts, xs, ys):
                    dx, dy, ha = LANG_OFF.get((m, r["lang"], xkey), (5, 4, "left"))
                    ax.annotate(
                        r["lang"], (xv, yv), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.5, color=INK, ha=ha, va="center",
                    )
            rb, re = reg["bitnet"], reg["eurollm"]
            b_txt = (
                f"BitNet: β_φ={float(rb['beta_logphi']):+.2f} [{float(rb['beta_logphi_lo']):+.2f},{float(rb['beta_logphi_hi']):+.2f}], "
                f"β_κ={float(rb['beta_logkappa']):+.2f} [{float(rb['beta_logkappa_lo']):+.2f},{float(rb['beta_logkappa_hi']):+.2f}] → {rb['verdict']}\n"
                f"EuroLLM: β_φ={float(re['beta_logphi']):+.2f}, β_κ={float(re['beta_logkappa']):+.2f} → {re['verdict']}"
            )
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, loc="upper left", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(
                f"{'Fertility φ = tok_L / tok_EN' if xkey == 'phi' else 'Content cost κ = B_L / B_EN'}"
                + (f"  |  BitNet ρ_φ={float(rb['spearman_y_phi']):+.2f}, ρ_κ={float(rb['spearman_y_kappa']):+.2f}" if col_idx == 0 else f"  |  EuroLLM ρ_φ={float(re['spearman_y_phi']):+.2f}, ρ_κ={float(re['spearman_y_kappa']):+.2f}"),
                fontsize=8.2, color=INK, loc="left",
            )
            if row_idx == 2:
                ax.set_xlabel("φ_L = tok_L / tok_EN" if xkey == "phi" else "κ_L = B_L / B_EN (base nats ratio)")
            if col_idx == 0:
                ax.text(0.02, 0.03, b_txt, transform=ax.transAxes, fontsize=7.0, color=INK2,
                        va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.9))
    fig.suptitle(
        "H2 (FLORES-200, 10 langs / 9 non-EN, S3+s4r32): R tracks φ (tokenization), while ratio_lt4 & fragility track κ (competence)",
        fontsize=10.0, color=INK, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
