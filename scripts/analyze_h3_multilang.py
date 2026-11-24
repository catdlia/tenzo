#!/usr/bin/env python3
"""Analyze H3 (T48): 5-language FLORES-200 KV quantization degradation."""

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
sys.path.insert(0, str(ROOT / "scripts"))
import kv_eval_lib as K

LANG_INFO = [
    ("en", "eng_Latn", "English", "Latin"),
    ("de", "deu_Latn", "German", "Latin"),
    ("pl", "pol_Latn", "Polish", "Latin"),
    ("uk", "ukr_Cyrl", "Ukrainian", "Cyrillic"),
    ("bg", "bul_Cyrl", "Bulgarian", "Cyrillic"),
]


def spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    dx = x - x.mean()
    dy = y - y.mean()
    den = math.sqrt(float((dx * dx).sum() * (dy * dy).sum()))
    return float((dx * dy).sum() / den) if den > 0 else 0.0


def main() -> int:
    meta = json.loads((H3_DIR / "flores5_meta.json").read_text())
    n_win = int(meta["n_windows"])
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, n_win, size=(10000, n_win))

    rows = []
    win_rows = []
    corr_summary = []

    for model_key, model_label, base_scheme in [
        ("bitnet", "BitNet-2B-4T", "fp32"),
        ("eurollm", "EuroLLM-1.7B", "fp"),
    ]:
        sc_dir = H3_DIR / "scores" / model_key
        # Load baseline for all 5 languages first
        fp_data = {}
        en_tok_total = None
        en_char_total = None
        for short, _, _, _ in LANG_INFO:
            cnts, nlls = K.load_nlls(sc_dir / f"{base_scheme}_{short}.txt", n_win)
            chars = np.array([w["langs"][short]["chars"] for w in meta["windows"]], dtype=np.float64)
            bytess = np.array([w["langs"][short]["bytes"] for w in meta["windows"]], dtype=np.float64)
            words = np.array([w["langs"][short]["words"] for w in meta["windows"]], dtype=np.float64)
            fp_data[short] = {
                "toks": cnts.astype(np.float64),
                "nlls": nlls,
                "chars": chars,
                "bytes": bytess,
                "words": words,
            }
            if short == "en":
                en_tok_total = float(cnts.sum())
                en_char_total = float(chars.sum())

        for scheme in ["tenzo_int8", "S3+s4r32"]:
            sname = scheme.replace(":", "_").replace("+", "_")
            en_toks = fp_data["en"]["toks"]
            en_fp = fp_data["en"]["nlls"]
            _, en_q = K.load_nlls(sc_dir / f"{sname}_en.txt", n_win)
            en_delta = en_q - en_fp
            en_rel_b = en_delta[boot_idx].sum(axis=1) / en_fp[boot_idx].sum(axis=1)

            for short, full_code, lang_name, script in LANG_INFO:
                d_fp = fp_data[short]
                toks = d_fp["toks"]
                fp_nll = d_fp["nlls"]
                cnts_q, q_nll = K.load_nlls(sc_dir / f"{sname}_{short}.txt", n_win)
                assert np.array_equal(cnts_q.astype(np.float64), toks)
                delta = q_nll - fp_nll

                # Ratios & stats
                tok_ratio = float(toks.sum() / en_tok_total)
                char_ratio = float(d_fp["chars"].sum() / en_char_total)
                fertility = float(toks.sum() / d_fp["words"].sum())
                mean_tok = float(toks.mean())

                base_bpt = float(fp_nll.sum() / toks.sum())
                base_bpc = float(fp_nll.sum() / d_fp["chars"].sum())
                base_bpb = float(fp_nll.sum() / d_fp["bytes"].sum())

                # Paired bootstraps
                delta_doc_b = delta[boot_idx].mean(axis=1)
                delta_doc = float(delta.mean())
                dd_lo, dd_hi = float(np.percentile(delta_doc_b, 2.5)), float(np.percentile(delta_doc_b, 97.5))

                rel_b = delta[boot_idx].sum(axis=1) / fp_nll[boot_idx].sum(axis=1)
                rel_loss = float(delta.sum() / fp_nll.sum())
                rl_lo, rl_hi = float(np.percentile(rel_b, 2.5)), float(np.percentile(rel_b, 97.5))

                dtok_b = delta[boot_idx].sum(axis=1) / toks[boot_idx].sum(axis=1)
                dtok = float(delta.sum() / toks.sum())
                dt_lo, dt_hi = float(np.percentile(dtok_b, 2.5)), float(np.percentile(dtok_b, 97.5))

                D_b = (delta - en_delta)[boot_idx].mean(axis=1)
                D_val = float((delta - en_delta).mean())
                D_lo, D_hi = float(np.percentile(D_b, 2.5)), float(np.percentile(D_b, 97.5))

                R_b = rel_b - en_rel_b
                R_val = rel_loss - float(en_delta.sum() / en_fp.sum())
                R_lo, R_hi = float(np.percentile(R_b, 2.5)), float(np.percentile(R_b, 97.5))

                rows.append({
                    "model": model_key,
                    "model_label": model_label,
                    "scheme": scheme,
                    "lang": short,
                    "flores_code": full_code,
                    "language": lang_name,
                    "script": script,
                    "mean_tok": mean_tok,
                    "tok_ratio_en": tok_ratio,
                    "fertility_tok_per_word": fertility,
                    "char_ratio_en": char_ratio,
                    "base_bpt": base_bpt,
                    "base_bpc": base_bpc,
                    "base_bpb": base_bpb,
                    "delta_doc": delta_doc,
                    "delta_doc_ci95_lo": dd_lo,
                    "delta_doc_ci95_hi": dd_hi,
                    "delta_per_tok": dtok,
                    "delta_per_tok_ci95_lo": dt_lo,
                    "delta_per_tok_ci95_hi": dt_hi,
                    "rel_loss": rel_loss,
                    "rel_loss_ci95_lo": rl_lo,
                    "rel_loss_ci95_hi": rl_hi,
                    "D_vs_en": D_val,
                    "D_ci95_lo": D_lo,
                    "D_ci95_hi": D_hi,
                    "R_vs_en": R_val,
                    "R_ci95_lo": R_lo,
                    "R_ci95_hi": R_hi,
                })

                for wi in range(n_win):
                    win_rows.append({
                        "model": model_key,
                        "scheme": scheme,
                        "lang": short,
                        "wid": wi,
                        "tokens": int(toks[wi]),
                        "chars": int(d_fp["chars"][wi]),
                        "bytes": int(d_fp["bytes"][wi]),
                        "words": int(d_fp["words"][wi]),
                        "nll_fp": float(fp_nll[wi]),
                        "nll_q": float(q_nll[wi]),
                        "delta_nll": float(delta[wi]),
                        "rel_loss_w": float(delta[wi] / fp_nll[wi]),
                        "delta_vs_en_w": float(delta[wi] - en_delta[wi]),
                    })

            sub = [r for r in rows if r["model"] == model_key and r["scheme"] == scheme]
            tr = np.array([r["tok_ratio_en"] for r in sub])
            rl = np.array([r["rel_loss"] * 100.0 for r in sub])
            dl = np.array([r["D_vs_en"] for r in sub])
            slope_r, intercept_r = np.polyfit(tr, rl, 1)
            corr_summary.append({
                "model": model_key,
                "model_label": model_label,
                "scheme": scheme,
                "spearman_tok_vs_R": spearman_rank(tr, rl),
                "pearson_tok_vs_R": pearson_r(tr, rl),
                "spearman_tok_vs_D": spearman_rank(tr, dl),
                "pearson_tok_vs_D": pearson_r(tr, dl),
                "ols_slope_rel_pct_per_tokratio": float(slope_r),
                "ols_intercept": float(intercept_r),
            })

    # Write CSVs
    with (H3_DIR / "multilang_flores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    with (H3_DIR / "multilang_flores_windows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
        w.writeheader()
        for r in win_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Plot 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), dpi=150)
    script_color = {"Latin": "#2b6cb0", "Cyrillic": "#c53030"}

    for col, (model_key, model_label) in enumerate([("bitnet", "BitNet-2B-4T"), ("eurollm", "EuroLLM-1.7B")]):
        # Top: rel_loss (%) vs tok_ratio_en under S3+s4r32
        ax = axes[0, col]
        sub = sorted([r for r in rows if r["model"] == model_key and r["scheme"] == "S3+s4r32"], key=lambda r: r["tok_ratio_en"])
        tr = np.array([r["tok_ratio_en"] for r in sub])
        rl = np.array([r["rel_loss"] * 100.0 for r in sub])
        elo = np.array([(r["rel_loss"] - r["rel_loss_ci95_lo"]) * 100.0 for r in sub])
        ehi = np.array([(r["rel_loss_ci95_hi"] - r["rel_loss"]) * 100.0 for r in sub])
        for script_name in ("Latin", "Cyrillic"):
            idx_s = [i for i, r in enumerate(sub) if r["script"] == script_name]
            ax.errorbar(
                tr[idx_s], rl[idx_s], yerr=[elo[idx_s], ehi[idx_s]],
                fmt="o", markersize=8, capsize=4, color=script_color[script_name], label=script_name,
            )
        ann_offsets = {
            ("bitnet", "de"): (6, -12),
            ("bitnet", "uk"): (-24, 11),
            ("bitnet", "bg"): (-42, 10),
            ("eurollm", "pl"): (6, -14),
            ("eurollm", "de"): (7, 7),
            ("eurollm", "bg"): (-46, 8),
        }
        for r in sub:
            ox, oy = ann_offsets.get((model_key, r["lang"]), (5, 6))
            ax.annotate(f"{r['lang'].upper()} ({r['mean_tok']:.0f}t)", (r["tok_ratio_en"], r["rel_loss"] * 100.0),
                        xytext=(ox, oy), textcoords="offset points", fontsize=9)
        cs = next(c for c in corr_summary if c["model"] == model_key and c["scheme"] == "S3+s4r32")
        xs_line = np.linspace(0.95, max(tr) + 0.08, 50)
        ax.plot(xs_line, cs["ols_slope_rel_pct_per_tokratio"] * xs_line + cs["ols_intercept"], "--", color="gray", alpha=0.7)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel("Token multiplier vs English (`tok_L / tok_EN`)")
        ax.set_ylabel("Relative PPL/NLL loss (%) [S3+s4r32 vs FP]")
        ax.set_title(f"{model_label} — Loss vs Token Fertility (ρ={cs['spearman_tok_vs_R']:.2f}, r={cs['pearson_tok_vs_R']:.2f})")
        ax.grid(alpha=0.3)
        ax.legend()

        # Bottom: R_L (%) bar comparison for tenzo_int8 vs S3+s4r32 sorted by tok_ratio_en
        ax2 = axes[1, col]
        langs_sorted = [r["lang"] for r in sub]
        x = np.arange(len(langs_sorted))
        wb = 0.36
        for off, sch, clr in [(-wb / 2, "tenzo_int8", "#718096"), (wb / 2, "S3+s4r32", "#805ad5")]:
            srows = {r["lang"]: r for r in rows if r["model"] == model_key and r["scheme"] == sch}
            vals = [srows[l]["R_vs_en"] * 100.0 for l in langs_sorted]
            blo = [(srows[l]["R_vs_en"] - srows[l]["R_ci95_lo"]) * 100.0 for l in langs_sorted]
            bhi = [(srows[l]["R_ci95_hi"] - srows[l]["R_vs_en"]) * 100.0 for l in langs_sorted]
            ax2.bar(x + off, vals, wb, label=sch, color=clr, alpha=0.88, yerr=[blo, bhi], capsize=3)
        ax2.axhline(0, color="black", lw=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{l.upper()}\n({[r for r in sub if r['lang']==l][0]['script'][:3]}, ×{[r for r in sub if r['lang']==l][0]['tok_ratio_en']:.2f})" for l in langs_sorted], fontsize=8.5)
        ax2.set_ylabel("Excess rel loss $R_L = \\Delta\\%_L - \\Delta\\%_{EN}$ (pp)")
        ax2.set_title(f"{model_label} — Excess Disparity vs EN (sorted by tok ratio)")
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend()

    fig.suptitle("H3 (T48): 5-Language Parallel FLORES-200 KV Degradation (N=48, Seed=0 Bootstrap)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(H3_DIR / "multilang_flores.png")
    plt.close(fig)

    # Evaluate pre-registered decision rule objectively
    c_bn = next(c for c in corr_summary if c["model"] == "bitnet" and c["scheme"] == "S3+s4r32")
    c_eu = next(c for c in corr_summary if c["model"] == "eurollm" and c["scheme"] == "S3+s4r32")
    bn_map = {r["lang"]: r for r in rows if r["model"] == "bitnet" and r["scheme"] == "S3+s4r32"}
    eu_map = {r["lang"]: r for r in rows if r["model"] == "eurollm" and r["scheme"] == "S3+s4r32"}

    # Check BitNet Polish (Latin, higher tok) vs Ukrainian (Cyrillic, lower tok)
    pl_ge_uk_bn = bn_map["pl"]["R_vs_en"] >= bn_map["uk"]["R_vs_en"] - 0.001
    bn_cyr_excess = (bn_map["uk"]["R_vs_en"] > bn_map["pl"]["R_ci95_hi"])  # Cyrillic exceeds Latin even at fewer tokens
    if c_bn["spearman_tok_vs_R"] >= 0.80 and c_eu["spearman_tok_vs_R"] >= 0.80 and pl_ge_uk_bn:
        verdict = "fertility_dominates"
    elif bn_cyr_excess or (c_bn["spearman_tok_vs_R"] < 0.40 and bn_map["uk"]["R_vs_en"] > bn_map["pl"]["R_vs_en"] and bn_map["bg"]["R_vs_en"] > bn_map["pl"]["R_vs_en"]):
        verdict = "script_effect"
    else:
        verdict = "mixed"

    # Update multilang_flores.md keeping ## Plan intact
    md_path = H3_DIR / "multilang_flores.md"
    plan_only = md_path.read_text().split("\n## Results")[0].rstrip()
    lines = [plan_only, "", "## Results (`N = 48` Parallel Windows, Out-of-Sample `sents 209..633`)", ""]

    for model_key, model_label, base_scheme in [("bitnet", "BitNet-2B-4T", "fp32"), ("eurollm", "EuroLLM-1.7B", "fp")]:
        lines.append(f"### {model_label}")
        lines.append("")
        lines.append("| Lang | Script | `mean_tok` | `tok/EN` | `tok/word` | `base BPC` | `tenzo_int8` `rel_loss` | `S3+s4r32` `rel_loss` [95% CI] | `S3+s4r32` `D_L` (nats/win) | `S3+s4r32` `R_L` (pp) [95% CI] |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        sub_s3 = sorted([r for r in rows if r["model"] == model_key and r["scheme"] == "S3+s4r32"], key=lambda r: r["tok_ratio_en"])
        sub_i8 = {r["lang"]: r for r in rows if r["model"] == model_key and r["scheme"] == "tenzo_int8"}
        for r in sub_s3:
            ri = sub_i8[r["lang"]]
            lines.append(
                f"| `{r['lang']}` ({r['language']}) | {r['script']} | {r['mean_tok']:.1f} | {r['tok_ratio_en']:.2f}× | {r['fertility_tok_per_word']:.2f} | "
                f"{r['base_bpc']:.3f} | `{ri['rel_loss']*100:+.2f}%` | "
                f"`{r['rel_loss']*100:+.2f}%` [`{r['rel_loss_ci95_lo']*100:+.2f}%`, `{r['rel_loss_ci95_hi']*100:+.2f}%`] | "
                f"`{r['D_vs_en']:+.2f}` [`{r['D_ci95_lo']:+.2f}`, `{r['D_ci95_hi']:+.2f}`] | "
                f"`{r['R_vs_en']*100:+.2f}` [`{r['R_ci95_lo']*100:+.2f}`, `{r['R_ci95_hi']*100:+.2f}`] |"
            )
        cs = next(c for c in corr_summary if c["model"] == model_key and c["scheme"] == "S3+s4r32")
        lines.append("")
        lines.append(
            f"- **Fertility-vs-loss correlation (`S3+s4r32`):** Spearman `ρ(tok/EN, R_L) = {cs['spearman_tok_vs_R']:.3f}`, "
            f"Pearson `r = {cs['pearson_tok_vs_R']:.3f}`, Spearman `ρ(tok/EN, D_L) = {cs['spearman_tok_vs_D']:.3f}` "
            f"(OLS slope = `{cs['ols_slope_rel_pct_per_tokratio']:+.2f}` pp per 1.0× EN token ratio)."
        )
        lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(f"- **Pre-registered verdict:** **`{verdict}`**")
    lines.append(
        f"- **BitNet key contrast (`pl` Latin `×{bn_map['pl']['tok_ratio_en']:.2f}` vs `uk` Cyrillic `×{bn_map['uk']['tok_ratio_en']:.2f}`):** "
        f"`pl R_L = {bn_map['pl']['R_vs_en']*100:+.2f}` pp [`{bn_map['pl']['R_ci95_lo']*100:+.2f}`, `{bn_map['pl']['R_ci95_hi']*100:+.2f}`] "
        f"vs `uk R_L = {bn_map['uk']['R_vs_en']*100:+.2f}` pp [`{bn_map['uk']['R_ci95_lo']*100:+.2f}`, `{bn_map['uk']['R_ci95_hi']*100:+.2f}`]; "
        f"`bg` (`×{bn_map['bg']['tok_ratio_en']:.2f}` Cyrillic) `R_bg = {bn_map['bg']['R_vs_en']*100:+.2f}` pp; "
        f"`de` (`×{bn_map['de']['tok_ratio_en']:.2f}` Latin) `R_de = {bn_map['de']['R_vs_en']*100:+.2f}` pp."
    )
    lines.append(
        f"- **EuroLLM-1.7B contrast (`pl` `×{eu_map['pl']['tok_ratio_en']:.2f}`, `de` `×{eu_map['de']['tok_ratio_en']:.2f}`, "
        f"`bg` `×{eu_map['bg']['tok_ratio_en']:.2f}`, `uk` `×{eu_map['uk']['tok_ratio_en']:.2f}`):** "
        f"`R_de = {eu_map['de']['R_vs_en']*100:+.2f}` pp, `R_pl = {eu_map['pl']['R_vs_en']*100:+.2f}` pp, "
        f"`R_bg = {eu_map['bg']['R_vs_en']*100:+.2f}` pp, `R_uk = {eu_map['uk']['R_vs_en']*100:+.2f}` pp."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- Prep: `scripts/prepare_h3_multilang.py` -> `results/2026-11-24-b_agent2/H3/flores5_meta.json` (`0` 32-gram overlap)")
    lines.append("- Run: `scripts/run_h3_multilang.py` (parity verified `max_abs_lp_diff < 1e-6`)")
    lines.append("- Analysis: `scripts/analyze_h3_multilang.py` -> `multilang_flores.{csv,png}`, `multilang_flores_windows.csv`")
    lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"H3 analysis written. Verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
