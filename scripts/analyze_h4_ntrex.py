#!/usr/bin/env python3
"""Analyze H4 (T49): NTREX-128 coherent parallel evaluation (ctx=1024, N=24)."""

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
H4_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H4"
sys.path.insert(0, str(ROOT / "scripts"))
import kv_eval_lib as K

SLICES = [
    ("1..128", 0, 128),
    ("129..256", 128, 256),
    ("257..384", 256, 384),
    ("385..512", 384, 512),
    ("513..640", 512, 640),
    ("641..768", 640, 768),
    ("769..896", 768, 896),
    ("897..1023", 896, 1023),
]


def spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def main() -> int:
    meta = json.loads((H4_DIR / "ntrex_meta.json").read_text())
    n_win = int(meta["n_windows"])
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, n_win, size=(10000, n_win))

    rows = []
    win_rows = []
    slice_rows = []
    pos_verdicts = {}

    for model_key, model_label, base_scheme in [("bitnet", "BitNet-2B-4T", "fp32"), ("eurollm", "EuroLLM-1.7B", "fp")]:
        sc_dir = H4_DIR / "scores" / model_key
        lp_fp_en = K.load_lps(sc_dir / f"{base_scheme}_en.txt", n_win)
        lp_fp_uk = K.load_lps(sc_dir / f"{base_scheme}_uk.txt", n_win)
        fp_en = np.array([-lp.sum() for lp in lp_fp_en], dtype=np.float64)
        fp_uk = np.array([-lp.sum() for lp in lp_fp_uk], dtype=np.float64)
        tok_en = np.array([len(lp) for lp in lp_fp_en], dtype=np.float64)
        tok_uk = np.array([len(lp) for lp in lp_fp_uk], dtype=np.float64)
        ch_en = np.array([w["langs"]["en"]["chars"] for w in meta["windows"]], dtype=np.float64)
        ch_uk = np.array([w["langs"]["uk"]["chars"] for w in meta["windows"]], dtype=np.float64)
        by_en = np.array([w["langs"]["en"]["bytes"] for w in meta["windows"]], dtype=np.float64)
        by_uk = np.array([w["langs"]["uk"]["bytes"] for w in meta["windows"]], dtype=np.float64)

        for scheme in ["tenzo_int8", "S3+s4r32", "S3+s4r128"]:
            sname = scheme.replace(":", "_").replace("+", "_")
            lp_q_en = K.load_lps(sc_dir / f"{sname}_en.txt", n_win)
            lp_q_uk = K.load_lps(sc_dir / f"{sname}_uk.txt", n_win)
            q_en = np.array([-lp.sum() for lp in lp_q_en], dtype=np.float64)
            q_uk = np.array([-lp.sum() for lp in lp_q_uk], dtype=np.float64)

            d_en = q_en - fp_en
            d_uk = q_uk - fp_uk
            rel_en_b = d_en[boot_idx].sum(axis=1) / fp_en[boot_idx].sum(axis=1)
            rel_uk_b = d_uk[boot_idx].sum(axis=1) / fp_uk[boot_idx].sum(axis=1)
            rel_en = float(d_en.sum() / fp_en.sum())
            rel_uk = float(d_uk.sum() / fp_uk.sum())
            D_b = (d_uk - d_en)[boot_idx].mean(axis=1)
            R_b = rel_uk_b - rel_en_b
            D_val = float((d_uk - d_en).mean())
            R_val = rel_uk - rel_en

            rows.append({
                "model": model_key,
                "model_label": model_label,
                "scheme": scheme,
                "mean_tok_en": float(tok_en.mean()),
                "mean_tok_uk": float(tok_uk.mean()),
                "base_bpc_en": float(fp_en.sum() / ch_en.sum()),
                "base_bpc_uk": float(fp_uk.sum() / ch_uk.sum()),
                "base_bpb_en": float(fp_en.sum() / by_en.sum()),
                "base_bpb_uk": float(fp_uk.sum() / by_uk.sum()),
                "delta_doc_en": float(d_en.mean()),
                "delta_doc_uk": float(d_uk.mean()),
                "rel_en": rel_en,
                "rel_en_ci95_lo": float(np.percentile(rel_en_b, 2.5)),
                "rel_en_ci95_hi": float(np.percentile(rel_en_b, 97.5)),
                "rel_uk": rel_uk,
                "rel_uk_ci95_lo": float(np.percentile(rel_uk_b, 2.5)),
                "rel_uk_ci95_hi": float(np.percentile(rel_uk_b, 97.5)),
                "D": D_val,
                "D_ci95_lo": float(np.percentile(D_b, 2.5)),
                "D_ci95_hi": float(np.percentile(D_b, 97.5)),
                "R": R_val,
                "R_ci95_lo": float(np.percentile(R_b, 2.5)),
                "R_ci95_hi": float(np.percentile(R_b, 97.5)),
            })

            for wi in range(n_win):
                win_rows.append({
                    "model": model_key,
                    "doc_id": meta["windows"][wi]["doc_id"],
                    "scheme": scheme,
                    "wid": wi,
                    "tok_en": int(tok_en[wi]),
                    "tok_uk": int(tok_uk[wi]),
                    "fp_en": float(fp_en[wi]),
                    "fp_uk": float(fp_uk[wi]),
                    "delta_en": float(d_en[wi]),
                    "delta_uk": float(d_uk[wi]),
                    "D_w": float(d_uk[wi] - d_en[wi]),
                })

            # Positional slice decomposition
            uk_slice_boots = {}
            uk_slice_means = {}
            for lang, lp_fp_list, lp_q_list in [("en", lp_fp_en, lp_q_en), ("uk", lp_fp_uk, lp_q_uk)]:
                for sl_name, s0, s1 in SLICES:
                    cnts = np.zeros(n_win, dtype=np.float64)
                    sums = np.zeros(n_win, dtype=np.float64)
                    bases = np.zeros(n_win, dtype=np.float64)
                    n_docs_with_toks = 0
                    for wi in range(n_win):
                        seg_fp = lp_fp_list[wi][s0:s1]
                        seg_q = lp_q_list[wi][s0:s1]
                        if len(seg_fp) > 0:
                            n_docs_with_toks += 1
                            cnts[wi] = float(len(seg_fp))
                            sums[wi] = float((seg_fp - seg_q).sum())
                            bases[wi] = float((-seg_fp).sum())
                    if cnts.sum() == 0:
                        continue
                    bn = cnts[boot_idx].sum(axis=1)
                    bs = sums[boot_idx].sum(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        btok = np.where(bn > 0, bs / bn, np.nan)
                    valid_btok = btok[~np.isnan(btok)]
                    mean_tok_err = float(sums.sum() / cnts.sum())
                    mean_base_err = float(bases.sum() / cnts.sum())
                    slice_rows.append({
                        "model": model_key,
                        "scheme": scheme,
                        "lang": lang,
                        "slice": sl_name,
                        "s0": s0,
                        "s1": s1,
                        "n_docs": n_docs_with_toks,
                        "n_tokens": int(cnts.sum()),
                        "base_nll_per_tok": mean_base_err,
                        "delta_per_tok": mean_tok_err,
                        "ci95_lo": float(np.percentile(valid_btok, 2.5)),
                        "ci95_hi": float(np.percentile(valid_btok, 97.5)),
                    })
                    if lang == "uk":
                        uk_slice_boots[sl_name] = btok
                        uk_slice_means[sl_name] = mean_tok_err

            if scheme == "S3+s4r32":
                tail_slices = [name for name, _, _ in SLICES[1:] if name in uk_slice_means]
                ys = np.array([uk_slice_means[s] for s in tail_slices])
                xs_pos = np.arange(len(tail_slices), dtype=np.float64)
                rho = spearman_rank(xs_pos, ys)
                diff_tail = uk_slice_boots[tail_slices[-1]] - uk_slice_boots["129..256"]
                valid_diff = diff_tail[~np.isnan(diff_tail)]
                d_mean = float(uk_slice_means[tail_slices[-1]] - uk_slice_means["129..256"])
                d_lo, d_hi = float(np.percentile(valid_diff, 2.5)), float(np.percentile(valid_diff, 97.5))
                verdict = "monotonic_growth" if (rho >= 0.80 and d_lo > 0) else "flat_or_bounded"
                pos_verdicts[model_key] = {
                    "last_slice": tail_slices[-1],
                    "spearman_rho_129_1023": rho,
                    "last_minus_second": d_mean,
                    "ci95_lo": d_lo,
                    "ci95_hi": d_hi,
                    "verdict": verdict,
                }

    # Write CSVs
    with (H4_DIR / "ntrex_long_parallel.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    with (H4_DIR / "ntrex_windows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
        w.writeheader()
        for r in win_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    with (H4_DIR / "ntrex_slices.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(slice_rows[0].keys()))
        w.writeheader()
        for r in slice_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Plot 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), dpi=150)
    for col, (model_key, model_label) in enumerate([("bitnet", "BitNet-2B-4T"), ("eurollm", "EuroLLM-1.7B")]):
        ax = axes[0, col]
        for scheme, lang, clr, ls, lbl in [
            ("S3+s4r32", "en", "#2b6cb0", "-", "EN S3+s4r32"),
            ("S3+s4r32", "uk", "#c53030", "-", "UK S3+s4r32"),
            ("S3+s4r128", "uk", "#2f855a", "--", "UK S3+s4r128"),
        ]:
            s_sub = [r for r in slice_rows if r["model"] == model_key and r["scheme"] == scheme and r["lang"] == lang and r["n_docs"] >= 5]
            xs_idx = np.array([SLICES.index(next(sl for sl in SLICES if sl[0] == r["slice"])) for r in s_sub])
            ys = np.array([r["delta_per_tok"] for r in s_sub])
            ylo = np.array([r["delta_per_tok"] - r["ci95_lo"] for r in s_sub])
            yhi = np.array([r["ci95_hi"] - r["delta_per_tok"] for r in s_sub])
            ax.errorbar(xs_idx, ys, yerr=[ylo, yhi], marker="o", linestyle=ls, capsize=3, color=clr, label=lbl)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(np.arange(len(SLICES)))
        ax.set_xticklabels([s[0] for s in SLICES], rotation=28, ha="right", fontsize=8)
        ax.set_ylabel("Mean ΔNLL / token (nats)")
        pv = pos_verdicts[model_key]
        ax.set_title(f"{model_label} Positional Profile (ρ={pv['spearman_rho_129_1023']:.2f}, {pv['verdict']})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)

        ax2 = axes[1, col]
        sch_list = ["tenzo_int8", "S3+s4r32", "S3+s4r128"]
        x = np.arange(len(sch_list))
        mrows = {r["scheme"]: r for r in rows if r["model"] == model_key}
        vals = [mrows[s]["R"] * 100.0 for s in sch_list]
        elo = [(mrows[s]["R"] - mrows[s]["R_ci95_lo"]) * 100.0 for s in sch_list]
        ehi = [(mrows[s]["R_ci95_hi"] - mrows[s]["R"]) * 100.0 for s in sch_list]
        ax2.bar(x, vals, 0.52, color=["#718096", "#c53030", "#2f855a"], alpha=0.88, yerr=[elo, ehi], capsize=4)
        ax2.axhline(0, color="black", lw=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(sch_list, fontsize=9.5)
        ax2.set_ylabel("Excess rel loss $R = \\Delta\\%_{UK} - \\Delta\\%_{EN}$ (pp)")
        ax2.set_title(f"{model_label} — NTREX-128 Excess Disparity ($N=24$)")
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("H4 (T49): Coherent Single-Document Parallel NTREX-128 (ctx=1024, N=24)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(H4_DIR / "ntrex_long_parallel.png")
    plt.close(fig)

    # Update ntrex_long_parallel.md preserving ## Plan
    md_path = H4_DIR / "ntrex_long_parallel.md"
    plan_only = md_path.read_text().split("\n## Results")[0].rstrip()
    lines = [plan_only, "", "## Results (`N = 24` Single-Document Parallel Articles, `ctx = 1024`)", ""]

    for model_key, model_label in [("bitnet", "BitNet-2B-4T"), ("eurollm", "EuroLLM-1.7B")]:
        mrows = [r for r in rows if r["model"] == model_key]
        lines.append(f"### {model_label} (`mean_tok`: EN `{mrows[0]['mean_tok_en']:.1f}`, UK `{mrows[0]['mean_tok_uk']:.1f}`)")
        lines.append("")
        lines.append("| Scheme | EN `rel_loss` [95% CI] | UK `rel_loss` [95% CI] | `D` (`UK-EN`, nats/doc) | `R` (pp) [95% CI] |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in mrows:
            lines.append(
                f"| `{r['scheme']}` | `{r['rel_en']*100:+.2f}%` [`{r['rel_en_ci95_lo']*100:+.2f}%`, `{r['rel_en_ci95_hi']*100:+.2f}%`] | "
                f"`{r['rel_uk']*100:+.2f}%` [`{r['rel_uk_ci95_lo']*100:+.2f}%`, `{r['rel_uk_ci95_hi']*100:+.2f}%`] | "
                f"`{r['D']:+.2f}` [`{r['D_ci95_lo']:+.2f}`, `{r['D_ci95_hi']:+.2f}`] | "
                f"`{r['R']*100:+.2f}` [`{r['R_ci95_lo']*100:+.2f}`, `{r['R_ci95_hi']*100:+.2f}`] |"
            )
        r32 = next(r for r in mrows if r["scheme"] == "S3+s4r32")
        r128 = next(r for r in mrows if r["scheme"] == "S3+s4r128")
        rec_r = 100.0 * (1.0 - r128["R"] / r32["R"]) if abs(r32["R"]) > 1e-12 else 0.0
        rec_d = 100.0 * (1.0 - r128["D"] / r32["D"]) if abs(r32["D"]) > 1e-12 else 0.0
        pv = pos_verdicts[model_key]
        lines.append("")
        lines.append(
            f"- **`S3+s4r128` recovery:** `R` drops `{r32['R']*100:+.2f}` -> `{r128['R']*100:+.2f}` pp (**{rec_r:.1f}%** reduction; `D`: `{r32['D']:+.2f}` -> `{r128['D']:+.2f}`, **{rec_d:.1f}%** reduction)."
        )
        lines.append(
            f"- **Positional trend (`S3+s4r32`, UK `129..1023`):** Spearman `ρ = {pv['spearman_rho_129_1023']:.3f}`, "
            f"`{pv['last_slice']} minus 129..256 = {pv['last_minus_second']:+.5f}` [`{pv['ci95_lo']:+.5f}`, `{pv['ci95_hi']:+.5f}`] -> **`{pv['verdict']}`**."
        )
        lines.append("")
        lines.append("| Pos Slice | `n_docs` (EN / UK) | EN `S3+s4r32` `Δ/tok` [95% CI] | UK `S3+s4r32` `Δ/tok` [95% CI] | UK `S3+s4r128` `Δ/tok` [95% CI] |")
        lines.append("|---|---:|---:|---:|---:|")
        for sl_name, _, _ in SLICES:
            re_ = next((x for x in slice_rows if x["model"] == model_key and x["scheme"] == "S3+s4r32" and x["lang"] == "en" and x["slice"] == sl_name), None)
            ru32 = next((x for x in slice_rows if x["model"] == model_key and x["scheme"] == "S3+s4r32" and x["lang"] == "uk" and x["slice"] == sl_name), None)
            ru128 = next((x for x in slice_rows if x["model"] == model_key and x["scheme"] == "S3+s4r128" and x["lang"] == "uk" and x["slice"] == sl_name), None)
            nd_e = re_["n_docs"] if re_ else 0
            nd_u = ru32["n_docs"] if ru32 else 0
            en_s = f"`{re_['delta_per_tok']:+.5f}` [`{re_['ci95_lo']:+.5f}`, `{re_['ci95_hi']:+.5f}`]" if re_ and nd_e >= 3 else f"— (n={nd_e})"
            u32_s = f"`{ru32['delta_per_tok']:+.5f}` [`{ru32['ci95_lo']:+.5f}`, `{ru32['ci95_hi']:+.5f}`]" if ru32 else "—"
            u128_s = f"`{ru128['delta_per_tok']:+.5f}` [`{ru128['ci95_lo']:+.5f}`, `{ru128['ci95_hi']:+.5f}`]" if ru128 else "—"
            lines.append(f"| `{sl_name}` | {nd_e} / {nd_u} | {en_s} | {u32_s} | {u128_s} |")
        lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append("- Prep: `scripts/prepare_h4_ntrex.py` -> `results/2026-11-24-b_agent2/H4/ntrex_meta.json` (`0` 32-gram overlap)")
    lines.append("- Run: `scripts/run_h4_ntrex.py` (parity verified `0.000e+00` BitNet, `<= 5.96e-06 < 1e-5` EuroLLM)")
    lines.append("- Analysis: `scripts/analyze_h4_ntrex.py` -> `ntrex_long_parallel.{csv,png}`, `ntrex_windows.csv`, `ntrex_slices.csv`")
    lines.append("")

    md_path.write_text("\n".join(lines))
    print("H4 analysis written.", pos_verdicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
