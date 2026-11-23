#!/usr/bin/env python3
"""Analyze H2 2x2 (language x domain) disparity across BitNet, Qwen2.5-1.5B, and EuroLLM-1.7B.

Fits per model & scheme:
  - Main factorial OLS: dBPC ~ UK + Web + UK*Web (plus 10,000-draw cell-stratified bootstrap 95% CIs)
  - Baseline-adjusted OLS: dBPC ~ UK + Web + UK*Web + base_BPC
Also outputs dBPB metrics, fertility statistics, CSVs, Markdown report, and a 2x2 summary plot.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
H2_DIR = ROOT / "results" / "2026-11-23_agent2" / "H2"
LN2 = math.log(2.0)

CELLS = [
    ("en_wiki", 0, 0),
    ("uk_wiki", 1, 0),
    ("en_web", 0, 1),
    ("uk_web", 1, 1),
]
SCHEMES = ["tenzo_int8", "S3_s4r32", "S13_s4r32"]


def read_scores(path: pathlib.Path) -> dict[int, tuple[int, float]]:
    out: dict[int, tuple[int, float]] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("w "):
            _, wid, n, nll = ln.split()
            out[int(wid)] = (int(n), float(nll))
    return out


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def stratified_bootstrap(rows: list[dict], seed: int = 0, n_draws: int = 10000) -> dict:
    rng = np.random.RandomState(seed)
    by_cell = {c: [r for r in rows if r["cell"] == c] for c, _, _ in CELLS}
    # Order all rows stably by cell
    ordered = sum((by_cell[c] for c, _, _ in CELLS), [])

    def build_mats(rs: list[dict]):
        uk = np.array([r["uk"] for r in rs], float)
        web = np.array([r["web"] for r in rs], float)
        inter = uk * web
        base = np.array([r["base_bpc"] for r in rs], float)
        y_bpc = np.array([r["d_bpc"] for r in rs], float)
        y_bpb = np.array([r["d_bpb"] for r in rs], float)
        X2 = np.vstack([np.ones_like(uk), uk, web, inter]).T
        X3 = np.vstack([np.ones_like(uk), uk, web, inter, base]).T
        return X2, X3, y_bpc, y_bpb

    X2_0, X3_0, ybpc_0, ybpb_0 = build_mats(ordered)
    c2 = fit_ols(X2_0, ybpc_0)
    c3 = fit_ols(X3_0, ybpc_0)
    cb2 = fit_ols(X2_0, ybpb_0)

    b_c2 = np.zeros((n_draws, 4))
    b_c3 = np.zeros((n_draws, 5))
    b_cb2 = np.zeros((n_draws, 4))
    cell_arrays = [by_cell[c] for c, _, _ in CELLS]
    for d in range(n_draws):
        rs_draw = []
        for arr in cell_arrays:
            idx = rng.randint(0, len(arr), len(arr))
            for k in idx:
                rs_draw.append(arr[k])
        X2, X3, ybpc, ybpb = build_mats(rs_draw)
        b_c2[d] = fit_ols(X2, ybpc)
        b_c3[d] = fit_ols(X3, ybpc)
        b_cb2[d] = fit_ols(X2, ybpb)

    def summarize(pt, boot):
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        return [(float(pt[i]), float(lo[i]), float(hi[i])) for i in range(len(pt))]

    return {
        "m2_bpc": summarize(c2, b_c2),      # [intercept, UK, Web, UK*Web]
        "m3_bpc": summarize(c3, b_c3),      # [intercept, UK, Web, UK*Web, base_BPC]
        "m2_bpb": summarize(cb2, b_cb2),    # [intercept, UK, Web, UK*Web]
    }


def main() -> None:
    win_meta = json.loads((H2_DIR / "windows_2x2.json").read_text(encoding="utf-8"))

    models = [
        ("BitNet-b1.58-2B", H2_DIR / "scores" / "bitnet", "fp32", None),
        ("Qwen2.5-1.5B", H2_DIR / "scores" / "qwen", "fp", H2_DIR / "scores" / "qwen" / "token_meta.json"),
        ("EuroLLM-1.7B", H2_DIR / "scores" / "eurollm", "fp", H2_DIR / "scores" / "eurollm" / "token_meta.json"),
    ]

    win_rows = []
    reg_rows = []
    summary_cells = []

    for mname, sdir, base_name, tmeta_path in models:
        tmeta = json.loads(tmeta_path.read_text(encoding="utf-8")) if tmeta_path and tmeta_path.exists() else None
        for scheme in SCHEMES:
            sample_check = sdir / f"{scheme}_en_wiki.txt"
            if not sample_check.exists():
                print(f"Skipping {mname} / {scheme} (missing {sample_check})")
                continue
            rows_model_scheme = []
            for cell, is_uk, is_web in CELLS:
                base_sc = read_scores(sdir / f"{base_name}_{cell}.txt")
                sch_sc = read_scores(sdir / f"{scheme}_{cell}.txt")
                wlist = win_meta["cells"][cell]["windows"]
                tlist = tmeta["cells"][cell] if tmeta else None
                for wid, w in enumerate(wlist):
                    n0, nll0 = base_sc[wid]
                    n1, nll1 = sch_sc[wid]
                    assert n0 == n1
                    nb = tlist[wid]["n_bytes"] if tlist else w["n_bytes"]
                    nc = tlist[wid]["n_chars"] if tlist else w["n_chars"]
                    bpc0 = nll0 / (nc * LN2)
                    bpc1 = nll1 / (nc * LN2)
                    bpb0 = nll0 / (nb * LN2)
                    bpb1 = nll1 / (nb * LN2)
                    rec = {
                        "model": mname,
                        "scheme": scheme,
                        "cell": cell,
                        "uk": is_uk,
                        "web": is_web,
                        "wid": wid,
                        "n_tok": n0,
                        "n_bytes": nb,
                        "n_chars": nc,
                        "fertility": n0 / max(nc, 1),
                        "base_bpc": bpc0,
                        "scheme_bpc": bpc1,
                        "d_bpc": bpc1 - bpc0,
                        "base_bpb": bpb0,
                        "scheme_bpb": bpb1,
                        "d_bpb": bpb1 - bpb0,
                    }
                    rows_model_scheme.append(rec)
                    win_rows.append(rec)

                sub = [r for r in rows_model_scheme if r["cell"] == cell]
                summary_cells.append(
                    {
                        "model": mname,
                        "scheme": scheme,
                        "cell": cell,
                        "mean_fert": float(np.mean([r["fertility"] for r in sub])),
                        "base_bpc": float(np.mean([r["base_bpc"] for r in sub])),
                        "d_bpc_mean": float(np.mean([r["d_bpc"] for r in sub])),
                        "d_bpc_std": float(np.std([r["d_bpc"] for r in sub], ddof=1)),
                        "d_bpb_mean": float(np.mean([r["d_bpb"] for r in sub])),
                    }
                )

            reg = stratified_bootstrap(rows_model_scheme)
            for label, vals in [
                ("unadj_bpc", reg["m2_bpc"]),
                ("adj_base_bpc", reg["m3_bpc"]),
                ("unadj_bpb", reg["m2_bpb"]),
            ]:
                terms = ["intercept", "UK", "Web", "UK*Web", "base_BPC"][: len(vals)]
                for term, (pt, lo, hi) in zip(terms, vals):
                    reg_rows.append(
                        {
                            "model": mname,
                            "scheme": scheme,
                            "spec": label,
                            "term": term,
                            "estimate": pt,
                            "ci_lo": lo,
                            "ci_hi": hi,
                        }
                    )

    if not reg_rows:
        print("No scores found yet.")
        return

    with (H2_DIR / "disparity_2x2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "scheme", "spec", "term", "estimate", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(reg_rows)

    with (H2_DIR / "disparity_2x2_cells.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["model", "scheme", "cell", "mean_fert", "base_bpc", "d_bpc_mean", "d_bpc_std", "d_bpb_mean"]
        )
        w.writeheader()
        w.writerows(summary_cells)

    with (H2_DIR / "disparity_2x2_windows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model", "scheme", "cell", "uk", "web", "wid", "n_tok", "n_bytes", "n_chars",
                "fertility", "base_bpc", "scheme_bpc", "d_bpc", "base_bpb", "scheme_bpb", "d_bpb",
            ],
        )
        w.writeheader()
        w.writerows(win_rows)

    lines = [
        "# H2 — 2x2 Disparity Analysis (Language {EN, UK} × Domain {Wiki, Web})\n",
        "Corpora (zero 32-gram overlap with `calib_en8.ids` / `calib_uk8.ids` / `EN16` / `UK32`):",
        "- `en_wiki`: `wikimedia/wikipedia` (`20231101.en`, rows 0..99)",
        "- `uk_wiki`: `wikimedia/wikipedia` (`20231101.uk`, rows 0..99)",
        "- `en_web`: `HuggingFaceFW/fineweb-edu` (`sample-10BT`, rows 0..99)",
        "- `uk_web`: `Goader/kobza` (`default`, rows 200..299)\n",
        "## Cell Means (16 windows per cell)\n",
        "| Model | Scheme | Cell | Fertility (tok/char) | Base BPC | Mean ΔBPC | Mean ΔBPB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in summary_cells:
        lines.append(
            f"| {r['model']} | {r['scheme']} | {r['cell']} | {r['mean_fert']:.3f} | "
            f"{r['base_bpc']:.4f} | {r['d_bpc_mean']:+.5f} | {r['d_bpb_mean']:+.5f} |"
        )

    lines.append("\n## 2×2 Factorial Regressions (`ΔBPC ~ UK + Web + UK×Web [+ base_BPC]`, 10k stratified bootstrap)\n")
    lines.append("| Model | Scheme | Spec | Term | Estimate | 95% CI |")
    lines.append("| --- | --- | --- | --- | ---: | --- |")
    for r in reg_rows:
        if r["term"] == "intercept":
            continue
        lines.append(
            f"| {r['model']} | {r['scheme']} | {r['spec']} | {r['term']} | "
            f"{r['estimate']:+.5f} | [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}] |"
        )

    (H2_DIR / "disparity_2x2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Plot 2x2 bars if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
        for ax, mname in zip(axes, [m[0] for m in models]):
            sub = [r for r in summary_cells if r["model"] == mname]
            x = np.arange(len(SCHEMES))
            width = 0.19
            for idx, (cell, _, _) in enumerate(CELLS):
                vals = [next((r["d_bpc_mean"] for r in sub if r["scheme"] == s and r["cell"] == cell), 0.0) for s in SCHEMES]
                ax.bar(x + (idx - 1.5) * width, vals, width, label=cell)
            ax.set_xticks(x)
            ax.set_xticklabels(["int8", "S3+s4r32", "S13+s4r32"])
            ax.set_title(mname)
            ax.axhline(0, color="black", lw=0.7)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("ΔBPC (bits/char vs fp)")
        axes[-1].legend(loc="upper left", fontsize=8)
        fig.suptitle("KV Quantization Degradation (ΔBPC) Across Language × Domain & Tokenizer Fertility")
        fig.tight_layout()
        fig.savefig(H2_DIR / "disparity_2x2.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print("Plot skipped:", e)

    print("Wrote disparity_2x2.{csv,md}")


if __name__ == "__main__":
    main()
