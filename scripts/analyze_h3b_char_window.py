#!/usr/bin/env python3
"""Analyze H3b (Conditional): S3+s4c120 across 5 FLORES-200 languages vs S3+s4r32."""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
H3_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H3"
H3B_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H3b"
sys.path.insert(0, str(ROOT / "scripts"))
import kv_eval_lib as K

LANG_INFO = [
    ("en", "English", "Latin"),
    ("de", "German", "Latin"),
    ("pl", "Polish", "Latin"),
    ("uk", "Ukrainian", "Cyrillic"),
    ("bg", "Bulgarian", "Cyrillic"),
]


def spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def main() -> int:
    meta = json.loads((H3_DIR / "flores5_meta.json").read_text())
    n_win = int(meta["n_windows"])
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, n_win, size=(10000, n_win))

    rows = []
    win_rows = []
    model_summaries = {}

    for model_key, model_label, base_scheme in [("bitnet", "BitNet-2B-4T", "fp32"), ("eurollm", "EuroLLM-1.7B", "fp")]:
        h3_sc = H3_DIR / "scores" / model_key
        h3b_sc = H3B_DIR / "scores" / model_key
        cov = json.loads((h3b_sc / "coverage_stats.json").read_text())

        fp_nll = {}
        toks = {}
        for short, _, _ in LANG_INFO:
            c_arr, n_arr = K.load_nlls(h3_sc / f"{base_scheme}_{short}.txt", n_win)
            fp_nll[short] = n_arr
            toks[short] = c_arr.astype(np.float64)

        _, r32_en = K.load_nlls(h3_sc / "S3_s4r32_en.txt", n_win)
        _, c120_en = K.load_nlls(h3b_sc / "S3_s4c120_en.txt", n_win)
        d_r32_en = r32_en - fp_nll["en"]
        d_c120_en = c120_en - fp_nll["en"]
        rel_r32_en_b = d_r32_en[boot_idx].sum(axis=1) / fp_nll["en"][boot_idx].sum(axis=1)
        rel_c120_en_b = d_c120_en[boot_idx].sum(axis=1) / fp_nll["en"][boot_idx].sum(axis=1)

        m_rows = []
        for short, lang_name, script in LANG_INFO:
            _, r32_l = K.load_nlls(h3_sc / f"S3_s4r32_{short}.txt", n_win)
            _, c120_l = K.load_nlls(h3b_sc / f"S3_s4c120_{short}.txt", n_win)
            d_r32 = r32_l - fp_nll[short]
            d_c120 = c120_l - fp_nll[short]

            rel_r32_b = d_r32[boot_idx].sum(axis=1) / fp_nll[short][boot_idx].sum(axis=1)
            rel_c120_b = d_c120[boot_idx].sum(axis=1) / fp_nll[short][boot_idx].sum(axis=1)
            rel_r32 = float(d_r32.sum() / fp_nll[short].sum())
            rel_c120 = float(d_c120.sum() / fp_nll[short].sum())

            D_r32_b = (d_r32 - d_r32_en)[boot_idx].mean(axis=1)
            D_c120_b = (d_c120 - d_c120_en)[boot_idx].mean(axis=1)
            D_r32 = float((d_r32 - d_r32_en).mean())
            D_c120 = float((d_c120 - d_c120_en).mean())

            R_r32_b = rel_r32_b - rel_r32_en_b
            R_c120_b = rel_c120_b - rel_c120_en_b
            R_r32 = rel_r32 - float(d_r32_en.sum() / fp_nll["en"].sum())
            R_c120 = rel_c120 - float(d_c120_en.sum() / fp_nll["en"].sum())

            R_red_b = R_r32_b - R_c120_b
            R_red = R_r32 - R_c120
            frac_red = (R_red / R_r32) if abs(R_r32) > 1e-12 else 0.0

            row = {
                "model": model_key,
                "model_label": model_label,
                "lang": short,
                "language": lang_name,
                "script": script,
                "tok_ratio_en": float(toks[short].sum() / toks["en"].sum()),
                "cov_chars_exact": cov[short]["cov_chars_exact"],
                "r_eff": cov[short]["r_eff"],
                "bits_win": cov[short]["bits_win"],
                "bits_at_512": cov[short]["bits_at_512"],
                "rel_r32": rel_r32,
                "rel_c120": rel_c120,
                "rel_c120_ci95_lo": float(np.percentile(rel_c120_b, 2.5)),
                "rel_c120_ci95_hi": float(np.percentile(rel_c120_b, 97.5)),
                "D_r32": D_r32,
                "D_c120": D_c120,
                "D_c120_ci95_lo": float(np.percentile(D_c120_b, 2.5)),
                "D_c120_ci95_hi": float(np.percentile(D_c120_b, 97.5)),
                "R_r32": R_r32,
                "R_r32_ci95_lo": float(np.percentile(R_r32_b, 2.5)),
                "R_r32_ci95_hi": float(np.percentile(R_r32_b, 97.5)),
                "R_c120": R_c120,
                "R_c120_ci95_lo": float(np.percentile(R_c120_b, 2.5)),
                "R_c120_ci95_hi": float(np.percentile(R_c120_b, 97.5)),
                "R_red": R_red,
                "R_red_ci95_lo": float(np.percentile(R_red_b, 2.5)),
                "R_red_ci95_hi": float(np.percentile(R_red_b, 97.5)),
                "frac_R_reduced": frac_red,
            }
            rows.append(row)
            m_rows.append(row)

            for wi in range(n_win):
                win_rows.append({
                    "model": model_key,
                    "lang": short,
                    "wid": wi,
                    "fp_nll": float(fp_nll[short][wi]),
                    "delta_r32": float(d_r32[wi]),
                    "delta_c120": float(d_c120[wi]),
                })

        non_en = [r for r in m_rows if r["lang"] != "en"]
        mean_r_r32 = float(np.mean([r["R_r32"] for r in non_en]))
        mean_r_c120 = float(np.mean([r["R_c120"] for r in non_en]))
        mean_frac_red = 1.0 - mean_r_c120 / mean_r_r32 if abs(mean_r_r32) > 1e-12 else 0.0
        n_ci_includes_zero = sum(1 for r in non_en if r["R_c120_ci95_lo"] <= 0.0 <= r["R_c120_ci95_hi"])
        n_sig_reduced = sum(1 for r in non_en if r["R_red_ci95_lo"] > 0.0)
        tr_all = np.array([r["tok_ratio_en"] for r in m_rows])
        rc_all = np.array([r["R_c120"] for r in m_rows])
        rr_all = np.array([r["R_r32"] for r in m_rows])
        model_summaries[model_key] = {
            "mean_non_en_R_r32_pp": mean_r_r32 * 100.0,
            "mean_non_en_R_c120_pp": mean_r_c120 * 100.0,
            "mean_non_en_frac_red": mean_frac_red,
            "n_ci_zero": n_ci_includes_zero,
            "n_sig_reduced": n_sig_reduced,
            "rho_r32": spearman_rank(tr_all, rr_all),
            "rho_c120": spearman_rank(tr_all, rc_all),
        }

    # Evaluate pre-registered decision rule
    b_sum = model_summaries["bitnet"]
    e_sum = model_summaries["eurollm"]
    if (
        b_sum["mean_non_en_frac_red"] > 2.0 / 3.0
        and e_sum["mean_non_en_frac_red"] > 2.0 / 3.0
        and b_sum["n_ci_zero"] >= 3
        and e_sum["n_ci_zero"] >= 3
    ):
        verdict = "explains"
    elif b_sum["mean_non_en_frac_red"] >= 0.15 and e_sum["mean_non_en_frac_red"] >= 0.15:
        verdict = "partially_explains"
    else:
        verdict = "does_not_explain"

    with (H3B_DIR / "multilang_char_window.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    with (H3B_DIR / "multilang_char_window_windows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
        w.writeheader()
        for r in win_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Plot 2x2: Top R_L(r=32) vs R_L(C*=120); Bottom: r_eff and bits@512
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), dpi=150)
    for col, (model_key, model_label) in enumerate([("bitnet", "BitNet-2B-4T"), ("eurollm", "EuroLLM-1.7B")]):
        sub = sorted([r for r in rows if r["model"] == model_key], key=lambda r: r["tok_ratio_en"])
        x = np.arange(len(sub))
        wb = 0.36

        ax = axes[0, col]
        r32_vals = [r["R_r32"] * 100.0 for r in sub]
        r32_lo = [(r["R_r32"] - r["R_r32_ci95_lo"]) * 100.0 for r in sub]
        r32_hi = [(r["R_r32_ci95_hi"] - r["R_r32"]) * 100.0 for r in sub]
        c120_vals = [r["R_c120"] * 100.0 for r in sub]
        c120_lo = [(r["R_c120"] - r["R_c120_ci95_lo"]) * 100.0 for r in sub]
        c120_hi = [(r["R_c120_ci95_hi"] - r["R_c120"]) * 100.0 for r in sub]
        ax.bar(x - wb / 2, r32_vals, wb, label="S3+s4r32 (32 toks)", color="#c53030", alpha=0.88, yerr=[r32_lo, r32_hi], capsize=3)
        ax.bar(x + wb / 2, c120_vals, wb, label="S3+s4c120 (120 chars)", color="#2b6cb0", alpha=0.88, yerr=[c120_lo, c120_hi], capsize=3)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['lang'].upper()}\n(×{r['tok_ratio_en']:.2f})" for r in sub], fontsize=9)
        ax.set_ylabel("Excess rel loss $R_L = \\Delta\\%_L - \\Delta\\%_{EN}$ (pp)")
        ms = model_summaries[model_key]
        ax.set_title(f"{model_label}: mean non-EN $R$ reduced {ms['mean_non_en_frac_red']*100:.1f}%")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        ax2 = axes[1, col]
        reffs = [r["r_eff"] for r in sub]
        b512 = [r["bits_at_512"] for r in sub]
        bars = ax2.bar(x, reffs, 0.52, color="#319795", alpha=0.88)
        ax2.axhline(32.0, color="#c53030", ls="--", lw=1.2, label="Fixed token window r=32")
        for xi, rv, bv in zip(x, reffs, b512):
            ax2.text(xi, rv + 1.0, f"{rv:.1f}t\n({bv:.2f}b)", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{r['lang'].upper()}\n({r['cov_chars_exact']:.0f}c)" for r in sub], fontsize=9)
        ax2.set_ylabel("Effective INT8 recent window `r_eff` (tokens)")
        ax2.set_ylim(0, max(reffs) + 12.0)
        ax2.set_title(f"{model_label} — Effective Window & KV Bits@512 under C*=120")
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend()

    fig.suptitle(f"H3b: Equal-Character Recent Window (S3+s4c120) Across 5 FLORES-200 Languages ({verdict})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(H3B_DIR / "multilang_char_window.png")
    plt.close(fig)

    # Update multilang_char_window.md keeping ## Plan intact
    md_path = H3B_DIR / "multilang_char_window.md"
    plan_only = md_path.read_text().split("\n## Results")[0].rstrip()
    lines = [plan_only, "", "## Results (`N = 48` Parallel Windows, `S3+s4c120` vs `S3+s4r32`)", ""]

    for model_key, model_label in [("bitnet", "BitNet-2B-4T"), ("eurollm", "EuroLLM-1.7B")]:
        sub = sorted([r for r in rows if r["model"] == model_key], key=lambda r: r["tok_ratio_en"])
        ms = model_summaries[model_key]
        lines.append(f"### {model_label}")
        lines.append("")
        lines.append("| Lang | `tok/EN` | `cov_chars` | `r_eff` | `bits_win` | `bits@512` | `R(r=32)` (pp) | `R(C*=120)` (pp) [95% CI] | `ΔR = r32 - c120` [95% CI] | `frac_red` |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sub:
            fr_s = f"{r['frac_R_reduced']*100:.1f}%" if r["lang"] != "en" else "—"
            lines.append(
                f"| `{r['lang']}` ({r['script'][:3]}) | {r['tok_ratio_en']:.2f}× | {r['cov_chars_exact']:.1f} | {r['r_eff']:.1f} | "
                f"{r['bits_win']:.3f} | {r['bits_at_512']:.3f} | "
                f"`{r['R_r32']*100:+.2f}` | `{r['R_c120']*100:+.2f}` [`{r['R_c120_ci95_lo']*100:+.2f}`, `{r['R_c120_ci95_hi']*100:+.2f}`] | "
                f"`{r['R_red']*100:+.2f}` [`{r['R_red_ci95_lo']*100:+.2f}`, `{r['R_red_ci95_hi']*100:+.2f}`] | {fr_s} |"
            )
        lines.append("")
        lines.append(
            f"- **Aggregate non-EN parity (`de, pl, uk, bg`):** Mean `R_L` shifts `{ms['mean_non_en_R_r32_pp']:+.2f}` -> `{ms['mean_non_en_R_c120_pp']:+.2f}` pp "
            f"(**{ms['mean_non_en_frac_red']*100:.1f}%** mean reduction; `ΔR > 0` significantly on `{ms['n_sig_reduced']}/4` languages; "
            f"`95% CI(R_c120)` spans 0 on `{ms['n_ci_zero']}/4`; Spearman `ρ(tok/EN, R_L)`: `{ms['rho_r32']:.3f}` -> `{ms['rho_c120']:.3f}`)."
        )
        lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(f"- **Pre-registered verdict:** **`{verdict}`**")
    if verdict == "explains":
        interp = (
            f"Equalizing physical character coverage (`~118` chars across all 5 languages) reduces non-EN excess relative loss by "
            f"**{b_sum['mean_non_en_frac_red']*100:.1f}%** (BitNet) and **{e_sum['mean_non_en_frac_red']*100:.1f}%** (EuroLLM-1.7B) "
            f"with CI spanning 0 on `{b_sum['n_ci_zero']}/4` and `{e_sum['n_ci_zero']}/4` non-EN languages."
        )
    elif verdict == "partially_explains":
        interp = (
            f"Equalizing physical character coverage (`~118` chars across all 5 languages) reduces non-EN excess relative loss by "
            f"**{b_sum['mean_non_en_frac_red']*100:.1f}%** (BitNet) and **{e_sum['mean_non_en_frac_red']*100:.1f}%** (EuroLLM-1.7B), "
            f"showing token-count windows account for part of the cross-lingual disparity while residual gaps persist at ternary middle positions."
        )
    else:
        interp = (
            f"Equalizing physical character coverage (`~118` chars) changes mean non-EN excess relative loss by "
            f"**{b_sum['mean_non_en_frac_red']*100:.1f}%** (BitNet) and **{e_sum['mean_non_en_frac_red']*100:.1f}%** (EuroLLM-1.7B), "
            f"failing the pre-registered mitigation thresholds."
        )
    lines.append(f"- **Interpretation:** {interp}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- Run: `scripts/run_h3b_char_window.py` (`BitNet` & `EuroLLM` B=1 parity verified)")
    lines.append("- Analysis: `scripts/analyze_h3b_char_window.py` -> `multilang_char_window.{csv,png}`, `multilang_char_window_windows.csv`")
    lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"H3b analysis written. Verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
