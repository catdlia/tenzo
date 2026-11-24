#!/usr/bin/env python3
"""Analyze H1b (ctx=2048): EN-long, EN-concat, UK-long, UK-concat with 10,000-rep window bootstrap."""

from __future__ import annotations

import csv
import json
import math
import pathlib
import subprocess
import sys
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
H1B_DIR = ROOT / "results" / "2026-11-24_agent2" / "H1b"
SCORES_DIR = H1B_DIR / "scores"
BLOCKS = [(0, 512, "1..512"), (512, 1024, "513..1024"), (1024, 1536, "1025..1536"), (1536, 2047, "1537..2047")]


def read_ids(path: pathlib.Path) -> list[list[int]]:
    return [list(map(int, ln.split())) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def read_lps(path: pathlib.Path) -> np.ndarray:
    rows = [[float(x) for x in ln.split()[1:]] for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("lp ")]
    return np.array(rows, dtype=np.float64)


def ppl_block(lps: np.ndarray, lo: int = 0, hi: int | None = None) -> float:
    sl = lps[:, lo:hi]
    return float(math.exp(-sl.sum() / sl.size))


def boot_ratio_ci(lps_a: np.ndarray, lps_b: np.ndarray, lo: int = 0, hi: int | None = None, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    n_w = min(len(lps_a), len(lps_b))
    a = -lps_a[:n_w, lo:hi].sum(axis=1)
    b = -lps_b[:n_w, lo:hi].sum(axis=1)
    n_tok = lps_a[:n_w, lo:hi].shape[1]
    pt = float(math.exp(a.sum() / (n_w * n_tok)) / math.exp(b.sum() / (n_w * n_tok)))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n_w, size=(n_boot, n_w))
    ra = np.exp(a[idx].sum(axis=1) / (n_w * n_tok))
    rb = np.exp(b[idx].sum(axis=1) / (n_w * n_tok))
    ratios = ra / rb
    return pt, float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def find_boundary_ids(tok_json_path: pathlib.Path) -> set[int]:
    vocab = json.loads(tok_json_path.read_text(encoding="utf-8"))["model"]["vocab"]
    return {int(tid) for s, tid in vocab.items() if "ĊĊ" in s}


def build_features(wins: list[list[int]], b_ids: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_w = len(wins)
    T = len(wins[0]) - 1
    pos_k = np.tile(np.arange(T, dtype=np.float64) / 1000.0, (n_w, 1))
    dist = np.zeros((n_w, T), dtype=np.float64)
    nprev = np.zeros((n_w, T), dtype=np.float64)
    for wi, w in enumerate(wins):
        last_b = 0
        npd = 0
        for t in range(T):
            if w[t] in b_ids:
                last_b = t if t > 0 else 0
                npd += 1
            dist[wi, t] = float(t - last_b)
            nprev[wi, t] = float(npd)
    return pos_k, dist, nprev


def bootstrap_ols(X_list: list[np.ndarray], y_list: list[np.ndarray], n_boot: int = 10000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_w = len(X_list)
    xtx = np.stack([X.T @ X for X in X_list], axis=0)
    xty = np.stack([X.T @ y for X, y in zip(X_list, y_list)], axis=0)
    pt = np.linalg.lstsq(xtx.sum(axis=0), xty.sum(axis=0), rcond=None)[0]
    rng = np.random.RandomState(seed)
    boots = np.empty((n_boot, pt.shape[0]), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n_w, n_w)
        boots[b] = np.linalg.lstsq(xtx[idx].sum(axis=0), xty[idx].sum(axis=0), rcond=None)[0]
    return pt, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def boot_mean(vals: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n_w = len(vals)
    idx = rng.randint(0, n_w, size=(n_boot, n_w))
    m = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    tok_json = ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json"
    b_ids = find_boundary_ids(tok_json)
    win_meta = json.loads((H1B_DIR / "windows_h1b.json").read_text(encoding="utf-8"))

    cells = ["en_long", "en_concat", "uk_long", "uk_concat"]
    modes = [("ternary_rot", "ternary_rot"), ("ternary_rot_alpha", "ternary_rot+alpha_mix")]

    csv_rows: list[dict] = []
    ratio_table: list[str] = []
    ols_table: list[str] = []
    cell_meta_lines: list[str] = []

    plot_rows: dict[tuple[str, str], list[float]] = {}

    for cell in cells:
        wins_all = read_ids(H1B_DIR / f"h1b_{cell}.ids")
        p_base = SCORES_DIR / f"{cell}__int8_fused.txt"
        if not p_base.exists():
            continue
        lp_base = read_lps(p_base)
        n_w = min(len(wins_all), len(lp_base))
        for mkey, _ in modes:
            mp = SCORES_DIR / f"{cell}__{mkey}.txt"
            if mp.exists():
                n_w = min(n_w, len(read_lps(mp)))
        if n_w == 0:
            continue
        wins = wins_all[:n_w]
        lp_base = lp_base[:n_w]
        pos_k, dist, nprev = build_features(wins, b_ids)
        cmeta = win_meta["cells"][cell]
        docs_list = cmeta.get("n_source_docs_per_window", cmeta.get("n_source_docs", [1] * len(wins_all)))[:n_w]
        nl2_list = cmeta["nl2_boundary_counts"][:n_w]
        cell_meta_lines.append(
            f"- `{cell}` (`N={n_w}`): source docs/win = `{docs_list}` (mean `{np.mean(docs_list):.2f}`); "
            f"`ĊĊ` breaks/win = `{nl2_list}` (mean `{np.mean(nl2_list):.1f}`)"
        )

        for mkey, mlabel in modes:
            mp = SCORES_DIR / f"{cell}__{mkey}.txt"
            if not mp.exists():
                continue
            lp_m = read_lps(mp)[:n_w]
            dnll = lp_base - lp_m

            full_r, f_lo, f_hi = boot_ratio_ci(lp_m, lp_base)
            b_ratios = [boot_ratio_ci(lp_m, lp_base, lo, hi)[0] for lo, hi, _ in BLOCKS]
            d_b4_b1, d_lo, d_hi = boot_mean(dnll[:, 1536:2047].mean(axis=1) - dnll[:, 0:512].mean(axis=1))

            X_simp = [np.column_stack([np.ones(2047), pos_k[wi]]) for wi in range(n_w)]
            y_list = [dnll[wi] for wi in range(n_w)]
            b_s, lo_s, hi_s = bootstrap_ols(X_simp, y_list)

            X_full = [np.column_stack([np.ones(2047), pos_k[wi], np.log1p(dist[wi]), nprev[wi]]) for wi in range(n_w)]
            b_f, lo_f, hi_f = bootstrap_ols(X_full, y_list)

            pos_flag = "YES (>0)" if lo_f[1] > 0.0 else ("SIMPLE>0" if lo_s[1] > 0.0 else "no (spans 0)")
            ratio_table.append(
                f"| `{cell}` | `{mlabel}` | {n_w} | {ppl_block(lp_base):.3f} | {ppl_block(lp_m):.3f} | "
                f"×{full_r:.4f} [{f_lo:.4f}, {f_hi:.4f}] | "
                + " | ".join(f"×{br:.4f}" for br in b_ratios)
                + f" | {d_b4_b1:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}] |"
            )
            ols_table.append(
                f"| `{cell}` | `{mlabel}` | {n_w} | {b_s[1]:+.4f} [{lo_s[1]:+.4f}, {hi_s[1]:+.4f}] | "
                f"**{b_f[1]:+.4f} [{lo_f[1]:+.4f}, {hi_f[1]:+.4f}]** | "
                f"{b_f[2]:+.4f} [{lo_f[2]:+.4f}, {hi_f[2]:+.4f}] | "
                f"{b_f[3]:+.4f} [{lo_f[3]:+.4f}, {hi_f[3]:+.4f}] | {pos_flag} |"
            )
            csv_rows.append(
                {
                    "cell": cell,
                    "mode": mlabel,
                    "n_windows": n_w,
                    "ppl_int8": ppl_block(lp_base),
                    "ppl_mode": ppl_block(lp_m),
                    "ratio_full": full_r,
                    "ratio_ci_lo": f_lo,
                    "ratio_ci_hi": f_hi,
                    "ratio_b1": b_ratios[0], "ratio_b2": b_ratios[1], "ratio_b3": b_ratios[2], "ratio_b4": b_ratios[3],
                    "dnll_b4_minus_b1": d_b4_b1, "dnll_b4_b1_lo": d_lo, "dnll_b4_b1_hi": d_hi,
                    "ols_simple_pos": b_s[1], "ols_simple_pos_lo": lo_s[1], "ols_simple_pos_hi": hi_s[1],
                    "ols_full_pos": b_f[1], "ols_full_pos_lo": lo_f[1], "ols_full_pos_hi": hi_f[1],
                    "ols_full_logdist": b_f[2], "ols_full_logdist_lo": lo_f[2], "ols_full_logdist_hi": hi_f[2],
                    "ols_full_nprev": b_f[3], "ols_full_nprev_lo": lo_f[3], "ols_full_nprev_hi": hi_f[3],
                }
            )
            plot_rows[(cell, mkey)] = b_ratios

    if not csv_rows:
        print("No completed H1b score files found.", file=sys.stderr)
        return 1

    with (H1B_DIR / "long_vs_concat_2048.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
        xs = [b[2] for b in BLOCKS]
        cstyles = {
            "en_long": ("EN-long (WT2)", "--^", "#1f77b4"),
            "en_concat": ("EN-concat (Wiki short)", ":v", "#17becf"),
            "uk_long": ("UK-long (1 Wiki art)", "-o", "#d62728"),
            "uk_concat": ("UK-concat (Kobza)", "-s", "#9467bd"),
        }
        for ax, (mk, title) in zip(axes, [("ternary_rot", "ternary_rot (uncalibrated)"), ("ternary_rot_alpha", "ternary_rot + alpha_mix")]):
            for cell, (lbl, fmt, col) in cstyles.items():
                if (cell, mk) in plot_rows:
                    ax.plot(xs, plot_rows[(cell, mk)], fmt, label=lbl, color=col, linewidth=1.9, markersize=5.5)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("Context token block")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8.5, loc="upper left")
        axes[0].set_ylabel("Block PPL Ratio vs int8_fused")
        fig.suptitle("H1b: Long Coherent Documents vs Concatenation Across Context Position (ctx=2048)")
        fig.tight_layout()
        fig.savefig(H1B_DIR / "long_vs_concat_2048.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}", file=sys.stderr)

    # Evaluate pre-registered decision rules cleanly from computed rows
    by_cm = {(r["cell"], r["mode"]): r for r in csv_rows}
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    gov = (pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
           if pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists() else "unknown")

    rule_lines: list[str] = ["### Pre-registered Decision Evaluation\n"]
    for mk_label in ("ternary_rot", "ternary_rot+alpha_mix"):
        r_ukl = by_cm.get(("uk_long", mk_label))
        r_ukc = by_cm.get(("uk_concat", mk_label))
        r_enl = by_cm.get(("en_long", mk_label))
        r_enc = by_cm.get(("en_concat", mk_label))
        if not (r_ukl and r_ukc and r_enl and r_enc):
            continue
        ukl_pos = r_ukl["ols_full_pos_lo"] > 0.0
        ukc_pos = r_ukc["ols_full_pos_lo"] > 0.0
        enl_pos = r_enl["ols_full_pos_lo"] > 0.0
        enc_pos = r_enc["ols_full_pos_lo"] > 0.0
        ukl_simp = r_ukl["ols_simple_pos_lo"] > 0.0
        if not ukl_pos and ukc_pos:
            verdict = "Concatenation explains UK full-OLS positional effect (UK-long spans 0, UK-concat > 0)"
        elif ukl_pos and (not enl_pos) and (not enc_pos):
            verdict = "Positional effect is UK language-specific even in single long documents (UK-long > 0; EN-long/concat span 0)"
        else:
            verdict = f"Mixed / nuanced pattern (UK-long full_CI>0={ukl_pos} [simple_CI>0={ukl_simp}], UK-concat={ukc_pos}, EN-long={enl_pos}, EN-concat={enc_pos})"
        rule_lines.append(
            f"- **`{mk_label}`:** {verdict}. "
            f"(`UK-long` full `pos/1000 = {r_ukl['ols_full_pos']:+.4f} [{r_ukl['ols_full_pos_lo']:+.4f}, {r_ukl['ols_full_pos_hi']:+.4f}]`, "
            f"simple `{r_ukl['ols_simple_pos']:+.4f} [{r_ukl['ols_simple_pos_lo']:+.4f}, {r_ukl['ols_simple_pos_hi']:+.4f}]`; "
            f"`UK-concat` full `{r_ukc['ols_full_pos']:+.4f} [{r_ukc['ols_full_pos_lo']:+.4f}, {r_ukc['ols_full_pos_hi']:+.4f}]`; "
            f"`EN-long` `{r_enl['ols_full_pos']:+.4f} [{r_enl['ols_full_pos_lo']:+.4f}, {r_enl['ols_full_pos_hi']:+.4f}]`; "
            f"`EN-concat` `{r_enc['ols_full_pos']:+.4f} [{r_enc['ols_full_pos_lo']:+.4f}, {r_enc['ols_full_pos_hi']:+.4f}]`)."
        )

    md_path = H1B_DIR / "long_vs_concat_2048.md"
    existing = md_path.read_text(encoding="utf-8")
    plan_end = existing.find("\n## Results")
    plan_part = existing[:plan_end].rstrip() if plan_end != -1 else existing.rstrip()

    out_md = [
        plan_part,
        "",
        "## Results",
        "",
        "### Cell Structure",
        *cell_meta_lines,
        "",
        "### 1. PPL & Block PPL Ratios vs `int8_fused` (`ctx = 2048`, `10,000`-rep window bootstrap)",
        "",
        "| Cell | Mode | N | int8 PPL | Mode PPL | Full Ratio [95% CI] | B1 (1..512) | B2 (513..1024) | B3 (1025..1536) | B4 (1537..2047) | Δ(B4−B1) dNLL [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        *ratio_table,
        "",
        "### 2. Per-Token OLS Positional Slopes (`10,000`-rep window bootstrap, `seed=0`)",
        "",
        "| Cell | Mode | N | Simple `pos/1000` [95% CI] | Full `pos/1000` [95% CI] | Full `log1p(dist)` [95% CI] | Full `n_prev_docs` [95% CI] | Pos Effect |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
        *ols_table,
        "",
        *rule_lines,
        "",
        "### Reproducibility & Metadata",
        "- Prepare windows: `python3 scripts/prepare_h1b_windows.py` -> `results/2026-11-24_agent2/H1b/windows_h1b.json`.",
        "- Evaluate engine: `python3 scripts/run_h1b_ctx2048.py` (`tenzo-inference -c 2048 --prefill batched`, `--kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt`).",
        f"- Analyze: `python3 scripts/analyze_h1b_ctx2048.py` (`commit {commit}`, governor `{gov}`, `n_boot=10000`, `seed=0`).",
    ]
    md_path.write_text("\n".join(out_md) + "\n", encoding="utf-8")
    print(f"Wrote {H1B_DIR / 'long_vs_concat_2048.{csv,md,png}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
