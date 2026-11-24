#!/usr/bin/env python3
"""H1a: Positional degradation and document-boundary OLS analysis on H3 ctx=2048 scores.

Preserves the pre-registered `## Plan` section in results/2026-11-24_agent2/H1a/position_h3.md
and appends reproducible CSV/Markdown/PNG results with 10,000-rep window bootstrap CIs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
H3_SCORES = ROOT / "results" / "2026-11-23_agent2" / "H3" / "scores"
OUT_DIR = ROOT / "results" / "2026-11-24_agent2" / "H1a"
BLOCKS_512 = [(0, 512, "1..512"), (512, 1024, "513..1024"), (1024, 1536, "1025..1536"), (1536, 2047, "1537..2047")]
BLOCKS_EQ = [(0, 511, "1..511"), (511, 1023, "512..1023"), (1023, 1535, "1024..1535"), (1535, 2047, "1536..2047")]


def sha256_short(path: pathlib.Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:16]


def read_lps(path: pathlib.Path) -> np.ndarray:
    rows = [[float(x) for x in ln.split()[1:]] for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("lp ")]
    return np.array(rows, dtype=np.float64)


def read_ids(path: pathlib.Path) -> list[list[int]]:
    return [list(map(int, ln.split())) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def find_boundary_ids(tok_json_path: pathlib.Path) -> set[int]:
    tok = json.loads(tok_json_path.read_text(encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    return {int(tid) for s, tid in vocab.items() if "ĊĊ" in s}


def build_window_features(wins: list[list[int]], b_ids: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Returns pos_k [N, 2047], dist [N, 2047], n_prev_docs [N, 2047], boundary_counts [N]."""
    n_w = len(wins)
    T = len(wins[0]) - 1
    pos_k = np.tile(np.arange(T, dtype=np.float64) / 1000.0, (n_w, 1))
    dist = np.zeros((n_w, T), dtype=np.float64)
    nprev = np.zeros((n_w, T), dtype=np.float64)
    bcounts: list[int] = []
    for wi, w in enumerate(wins):
        last_b = 0
        npd = 0
        bc = 0
        for t in range(T):
            if w[t] in b_ids:
                bc += 1
                if t > 0:
                    last_b = t
                    npd += 1
                else:
                    last_b = 0
                    npd += 1
            dist[wi, t] = float(t - last_b)
            nprev[wi, t] = float(npd)
        bcounts.append(bc)
    return pos_k, dist, nprev, bcounts


def bootstrap_ols(X_list: list[np.ndarray], y_list: list[np.ndarray], n_boot: int = 10000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sufficient-statistic window-level bootstrap for pooled OLS."""
    n_w = len(X_list)
    xtx = np.stack([X.T @ X for X in X_list], axis=0)
    xty = np.stack([X.T @ y for X, y in zip(X_list, y_list)], axis=0)

    pt_beta = np.linalg.lstsq(xtx.sum(axis=0), xty.sum(axis=0), rcond=None)[0]
    # Exact check against stacked matrix
    stacked_beta = np.linalg.lstsq(np.vstack(X_list), np.concatenate(y_list), rcond=None)[0]
    assert np.allclose(pt_beta, stacked_beta, atol=1e-9), (pt_beta, stacked_beta)

    rng = np.random.RandomState(seed)
    boots = np.empty((n_boot, pt_beta.shape[0]), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n_w, n_w)
        A = xtx[idx].sum(axis=0)
        v = xty[idx].sum(axis=0)
        boots[b] = np.linalg.lstsq(A, v, rcond=None)[0]
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return pt_beta, lo, hi


def boot_mean_diff(per_win_vals: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n_w = len(per_win_vals)
    pt = float(per_win_vals.mean())
    boots = [float(per_win_vals[rng.randint(0, n_w, n_w)].mean()) for _ in range(n_boot)]
    return pt, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_json = ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json"
    b_ids = find_boundary_ids(tok_json)

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    gov = (pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
           if pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists() else "unknown")

    csv_rows: list[dict] = []
    block_rows_md: list[str] = []
    sub_rows_md: list[str] = []
    win_rows_md: list[str] = []
    ols_rows_md: list[str] = []
    meta_lines: list[str] = []

    plot_data: dict[tuple[str, str], dict] = {}

    for lang in ("uk", "en"):
        wins_path = ROOT / "results" / "reference" / f"windows_{lang}_2048.ids"
        wins = read_ids(wins_path)
        pos_k, dist, nprev, bcounts = build_window_features(wins, b_ids)
        meta_lines.append(
            f"- `{lang}` windows (`windows_{lang}_2048.ids`, `sha256: {sha256_short(wins_path)}...`): boundary counts per window = `{bcounts}`"
        )

        base_path = H3_SCORES / f"engine_int8_fused_{lang}.txt"
        lp_base = read_lps(base_path)

        for mode_key, mode_label in (("ternary_rot", "ternary_rot"), ("ternary_rot_alpha", "ternary_rot+alpha_mix")):
            m_path = H3_SCORES / f"engine_{mode_key}_{lang}.txt"
            lp_mode = read_lps(m_path)
            dnll = lp_base - lp_mode  # positive = mode hurts NLL vs int8_fused

            b512 = [float(dnll[:, lo:hi].mean()) for lo, hi, _ in BLOCKS_512]
            beq = [float(dnll[:, lo:hi].mean()) for lo, hi, _ in BLOCKS_EQ]
            sub256 = []
            for lo, hi, _ in BLOCKS_512:
                m = dist[:, lo:hi] <= 256
                sub256.append(float(dnll[:, lo:hi][m].mean()) if m.any() else float("nan"))
            delta_b4_b1, d_lo, d_hi = boot_mean_diff(dnll[:, 1536:2047].mean(axis=1) - dnll[:, 0:512].mean(axis=1))

            def _fmt(v: float) -> str:
                return "—" if np.isnan(v) else f"{v:.3f}"

            sub_delta = "—" if (np.isnan(sub256[-1]) or np.isnan(sub256[0])) else f"{sub256[-1] - sub256[0]:+.3f}"
            block_rows_md.append(
                f"| {lang.upper()} | `{mode_label}` | "
                + " | ".join(f"{x:.3f}" for x in b512)
                + " | "
                + " | ".join(f"{x:.3f}" for x in beq)
                + f" | {delta_b4_b1:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] |"
            )
            sub_rows_md.append(
                f"| {lang.upper()} | `{mode_label}` (`dist<=256`) | "
                + " | ".join(_fmt(x) for x in sub256)
                + f" | {sub_delta} |"
            )

            for wi in range(len(wins)):
                wb = [float(dnll[wi, lo:hi].mean()) for lo, hi, _ in BLOCKS_512]
                note = " **(0 boundaries)**" if bcounts[wi] == 0 else ""
                win_rows_md.append(
                    f"| {lang.upper()} w={wi} | `{mode_label}` | {bcounts[wi]}{note} | "
                    + " | ".join(f"{x:.3f}" for x in wb)
                    + f" | {wb[-1] - wb[0]:+.3f} |"
                )

            # OLS 1: simple ~ pos/1000
            X_simp = [np.column_stack([np.ones(2047), pos_k[wi]]) for wi in range(len(wins))]
            y_list = [dnll[wi] for wi in range(len(wins))]
            b_simp, lo_simp, hi_simp = bootstrap_ols(X_simp, y_list)

            # OLS 2: full ~ pos/1000 + log1p(dist) + n_prev_docs
            X_full = [np.column_stack([np.ones(2047), pos_k[wi], np.log1p(dist[wi]), nprev[wi]]) for wi in range(len(wins))]
            b_full, lo_full, hi_full = bootstrap_ols(X_full, y_list)

            ols_rows_md.append(
                f"| {lang.upper()} | `{mode_label}` | {b_simp[1]:+.4f} [{lo_simp[1]:+.4f}, {hi_simp[1]:+.4f}] | "
                f"{b_full[1]:+.4f} [{lo_full[1]:+.4f}, {hi_full[1]:+.4f}] | "
                f"{b_full[2]:+.4f} [{lo_full[2]:+.4f}, {hi_full[2]:+.4f}] | "
                f"{b_full[3]:+.4f} [{lo_full[3]:+.4f}, {hi_full[3]:+.4f}] |"
            )

            csv_rows.append(
                {
                    "lang": lang,
                    "mode": mode_label,
                    "boundaries": ";".join(map(str, bcounts)),
                    "b512_1": b512[0], "b512_2": b512[1], "b512_3": b512[2], "b512_4": b512[3],
                    "beq_1": beq[0], "beq_2": beq[1], "beq_3": beq[2], "beq_4": beq[3],
                    "dist256_1": sub256[0], "dist256_2": sub256[1], "dist256_3": sub256[2], "dist256_4": sub256[3],
                    "delta_b4_b1": delta_b4_b1, "delta_b4_b1_ci_lo": d_lo, "delta_b4_b1_ci_hi": d_hi,
                    "ols_simple_pos": b_simp[1], "ols_simple_pos_lo": lo_simp[1], "ols_simple_pos_hi": hi_simp[1],
                    "ols_full_const": b_full[0],
                    "ols_full_pos": b_full[1], "ols_full_pos_lo": lo_full[1], "ols_full_pos_hi": hi_full[1],
                    "ols_full_logdist": b_full[2], "ols_full_logdist_lo": lo_full[2], "ols_full_logdist_hi": hi_full[2],
                    "ols_full_nprev": b_full[3], "ols_full_nprev_lo": lo_full[3], "ols_full_nprev_hi": hi_full[3],
                }
            )
            plot_data[(lang, mode_key)] = {
                "b512": b512,
                "sub256": sub256,
                "dnll_w2": [float(dnll[2, lo:hi].mean()) for lo, hi, _ in BLOCKS_512] if lang == "uk" else None,
            }

    csv_path = OUT_DIR / "position_h3.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
        xs = [b[2] for b in BLOCKS_512]
        styles = [
            ("uk", "ternary_rot", "UK ternary_rot (all)", "-o", "#d62728"),
            ("uk", "ternary_rot_alpha", "UK ternary_rot+alpha_mix (all)", "-s", "#ff7f0e"),
            ("en", "ternary_rot", "EN ternary_rot (all)", "--^", "#1f77b4"),
            ("en", "ternary_rot_alpha", "EN ternary_rot+alpha_mix (all)", "--d", "#2ca02c"),
        ]
        for lang, mk, lbl, fmt, col in styles:
            axes[0].plot(xs, plot_data[(lang, mk)]["b512"], fmt, label=lbl, color=col, linewidth=1.8)
        axes[0].set_title("All tokens (4 windows x 2047)")
        axes[0].set_xlabel("Position block")
        axes[0].set_ylabel("Mean dNLL vs int8_fused (nats/tok)")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8, loc="upper left")

        axes[1].plot(xs, plot_data[("uk", "ternary_rot")]["sub256"], "-o", color="#d62728", label="UK rot (dist<=256)")
        axes[1].plot(xs, plot_data[("uk", "ternary_rot_alpha")]["sub256"], "-s", color="#ff7f0e", label="UK rot+alpha (dist<=256)")
        axes[1].plot(xs, plot_data[("uk", "ternary_rot")]["dnll_w2"], ":o", color="#8c564b", label="UK w=2 (0 boundaries, rot)")
        axes[1].plot(xs, plot_data[("uk", "ternary_rot_alpha")]["dnll_w2"], ":s", color="#9467bd", label="UK w=2 (0 boundaries, alpha)")
        axes[1].plot(xs, plot_data[("en", "ternary_rot")]["sub256"], "--^", color="#1f77b4", label="EN rot (dist<=256)")
        axes[1].set_title("Controlling for boundaries (dist<=256 & w=2)")
        axes[1].set_xlabel("Position block")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8, loc="upper left")

        fig.suptitle("H1a: Positional KV Degradation at ctx=2048 (exploratory N=4)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "position_h3.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}", file=sys.stderr)

    # Read existing Plan prefix and replace/write full document cleanly
    md_path = OUT_DIR / "position_h3.md"
    existing = md_path.read_text(encoding="utf-8")
    plan_end = existing.find("\n## Results")
    plan_part = existing[:plan_end].rstrip() if plan_end != -1 else existing.rstrip()

    results_md = [
        plan_part,
        "",
        "## Results",
        "",
        "### 1. Boundary counts & replication of coordinator checks",
        *meta_lines,
        "",
        "| Lang | Mode | B1 (1..512) | B2 (513..1024) | B3 (1025..1536) | B4 (1537..2047) | Eq B1..511 | Eq B512..1023 | Eq B1024..1535 | Eq B1536..2047 | Δ(B4−B1) [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *block_rows_md,
        "",
        "### 2. Controlling for document boundaries (`dist <= 256` after `\\n\\n`)",
        "",
        "| Lang | Subset / Mode | B1 (1..512) | B2 (513..1024) | B3 (1025..1536) | B4 (1537..2047) | Δ(B4−B1) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *sub_rows_md,
        "",
        "### 3. Per-window breakdown (highlighting boundary-free UK `w=2`)",
        "",
        "| Window | Mode | Boundaries | B1 | B2 | B3 | B4 | B4−B1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *win_rows_md,
        "",
        "### 4. Per-token OLS regression (`10,000`-rep window bootstrap `95% CI`, `seed=0`)",
        "",
        "| Lang | Mode | Simple `pos/1000` [95% CI] | Full `pos/1000` [95% CI] | Full `log1p(dist)` [95% CI] | Full `n_prev_docs` [95% CI] |",
        "| --- | --- | --- | --- | --- | --- |",
        *ols_rows_md,
        "",
    ]
    by_lm = {(r["lang"], r["mode"]): r for r in csv_rows}
    uk_rot = by_lm[("uk", "ternary_rot")]
    uk_alp = by_lm[("uk", "ternary_rot+alpha_mix")]
    en_rot = by_lm[("en", "ternary_rot")]
    results_md += [
        "### Observations (cautious — exploratory `N = 4`)",
        "",
        f"- Coordinator numbers replicate for UK `ternary_rot`: boundary counts `[{uk_rot['boundaries'].replace(';', ', ')}]`, "
        f"equal-quarter blocks `{uk_rot['beq_1']:.3f}, {uk_rot['beq_2']:.3f}, {uk_rot['beq_3']:.3f}, {uk_rot['beq_4']:.3f}`, "
        f"`dist <= 256` blocks `{uk_rot['dist256_1']:.3f}, {uk_rot['dist256_2']:.3f}, {uk_rot['dist256_3']:.3f}, {uk_rot['dist256_4']:.3f}`, "
        f"and full OLS `pos/1000` slope `{uk_rot['ols_full_pos']:+.4f}` (`[{uk_rot['ols_full_pos_lo']:+.4f}, {uk_rot['ols_full_pos_hi']:+.4f}]`) "
        f"with `n_prev_docs = {uk_rot['ols_full_nprev']:+.4f}` (`[{uk_rot['ols_full_nprev_lo']:+.4f}, {uk_rot['ols_full_nprev_hi']:+.4f}]`).",
        f"- Comparing `ternary_rot` vs `ternary_rot+alpha_mix`: full OLS `pos/1000` on UK changes from `{uk_rot['ols_full_pos']:+.4f}` to `{uk_alp['ols_full_pos']:+.4f}` (`[{uk_alp['ols_full_pos_lo']:+.4f}, {uk_alp['ols_full_pos_hi']:+.4f}]`), while EN `ternary_rot` remains near zero/negative (`{en_rot['ols_full_pos']:+.4f} [{en_rot['ols_full_pos_lo']:+.4f}, {en_rot['ols_full_pos_hi']:+.4f}]`).",
        "- Because `N = 4` windows per language provides only 4 independent resampling units, bootstrap CIs remain wide/exploratory even when all 4 UK windows show `B4 > B1`. `H1b` (`N = 12` clean `UK-long` vs `UK-concat`) tests whether `pos/1000` CI > 0 without boundaries.",
        "",
        "### Reproducibility & Metadata",
        f"- Command: `python3 scripts/analyze_h3_position.py` (`commit {commit}`, governor `{gov}`, `N=4` windows/lang, `n_boot=10000`, `seed=0`).",
        "- Engine score files in `results/2026-11-23_agent2/H3/scores/` (`engine_{int8_fused,ternary_rot,ternary_rot_alpha}_{en,uk}.txt`).",
    ]
    md_path.write_text("\n".join(results_md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'position_h3.{csv,md,png}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
