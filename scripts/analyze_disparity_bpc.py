#!/usr/bin/env python3
"""Language disparity normalized per UTF-8 byte and Unicode character (T40, part 1 / L3).

Analyzes existing score files only (no model runs):
  - Decodes the 511 scored tokens of each window via the corresponding HF tokenizer
    (BitNet Llama-3 tokenizer from local HF cache; Qwen2.5 tokenizer from ~/tenzo-refs/hf/qwen2.5-1.5b-tokenizer).
  - Computes per-window:
      n_bytes, n_chars, fertility = n_scored_tokens / n_chars,
      BPB = sum_nll / (n_bytes * ln(2)), BPC = sum_nll / (n_chars * ln(2)),
      dBPC = BPC(scheme) - BPC(fp), dBPB = BPB(scheme) - BPB(fp), dNLL_char = dBPC * ln(2).
  - Fits within-language OLS (dBPC ~ BPC0) and pooled OLS (dBPC ~ BPC0 + c_UK * UK) with
    10,000-draw within-language bootstrap 95% CIs (seed 0), for both BPC and BPB.
  - Computes within-language correlation (Pearson r + bootstrap 95% CI, OLS slope) between
    per-character dNLL and fertility (tokens/char).

Outputs:
  results/2026-11-23_agent2/L3/disparity_bpc.csv
  results/2026-11-23_agent2/L3/disparity_bpc_windows.csv
  results/2026-11-23_agent2/L3/disparity_bpc.md
  results/2026-11-23_agent2/L3/disparity_bpc_scatter.png
  results/2026-11-23_agent2/L3/fertility_corr.png
"""
import csv
import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "2026-11-23_agent2" / "L3"
LN2 = math.log(2.0)


def rd_scores(path: Path) -> dict[int, tuple[int, float]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "w":
                out[int(parts[1])] = (int(parts[2]), float(parts[3]))
    return out


def load_ids_file(path: Path) -> list[list[int]]:
    return [list(map(int, line.split())) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_window_stats(token_windows: list[list[int]], tok) -> list[dict]:
    stats = []
    for wid, ids in enumerate(token_windows):
        scored_ids = ids[1:]  # target tokens scored by both golden_bitnet.py and second_model_kv.py
        text = tok.decode(scored_ids, clean_up_tokenization_spaces=False)
        n_tok = len(scored_ids)
        n_bytes = len(text.encode("utf-8"))
        n_chars = len(text)
        stats.append({
            "wid": wid,
            "n_tok": n_tok,
            "n_bytes": n_bytes,
            "n_chars": n_chars,
            "fertility": n_tok / max(n_chars, 1),
        })
    return stats


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    denom = max(float(((y - y.mean()) ** 2).sum()), 1e-30)
    r2 = 1.0 - float((res ** 2).sum()) / denom
    return float(coef[0]), float(coef[1]), r2


def pooled_c(x_en: np.ndarray, y_en: np.ndarray, x_uk: np.ndarray, y_uk: np.ndarray) -> np.ndarray:
    x = np.concatenate([x_en, x_uk])
    y = np.concatenate([y_en, y_uk])
    d = np.concatenate([np.zeros(len(x_en)), np.ones(len(x_uk))])
    A = np.vstack([np.ones_like(x), x, d]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float((xc ** 2).sum()) * float((yc ** 2).sum()))
    return float((xc * yc).sum()) / denom if denom > 1e-30 else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    tok_bitnet = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T", local_files_only=True)
    tok_qwen = AutoTokenizer.from_pretrained(str(Path.home() / "tenzo-refs" / "hf" / "qwen2.5-1.5b-tokenizer"))

    # BitNet windows
    bn_en_ids = load_ids_file(ROOT / "results" / "reference" / "windows.ids")
    bn_uk_ids = load_ids_file(ROOT / "results" / "reference" / "windows_uk32.ids")
    bn_en_meta = build_window_stats(bn_en_ids, tok_bitnet)
    bn_uk_meta = build_window_stats(bn_uk_ids, tok_bitnet)

    # Qwen2.5-1.5B windows (reconstructed from windows.json + wt2_test.raw + kobza_first_rows.json)
    qw_meta_json = json.loads((ROOT / "results" / "2026-11-20_overnight" / "part2" / "second_model" / "windows.json").read_text())
    wt2_raw = (Path.home() / "tenzo-refs" / "hf" / "wt2_test.raw").read_text(encoding="utf-8")
    qw_en_all = tok_qwen(wt2_raw, add_special_tokens=False).input_ids
    assert len(qw_en_all) == qw_meta_json["en"]["n_text_tokens"], (len(qw_en_all), qw_meta_json["en"]["n_text_tokens"])
    qw_en_ids = [qw_en_all[s:s + 512] for s in qw_meta_json["en"]["starts"]]

    kobza_rows = json.loads((ROOT / "results" / "reference" / "kobza_first_rows.json").read_text())["rows"]
    qw_uk_all = tok_qwen("\n\n".join(r["row"]["text"] for r in kobza_rows), add_special_tokens=False).input_ids
    assert len(qw_uk_all) == qw_meta_json["uk"]["n_text_tokens"], (len(qw_uk_all), qw_meta_json["uk"]["n_text_tokens"])
    qw_uk_ids = [qw_uk_all[w * 512:(w + 1) * 512] for w in range(16)]
    qw_en_meta = build_window_stats(qw_en_ids, tok_qwen)
    qw_uk_meta = build_window_stats(qw_uk_ids, tok_qwen)

    R19 = ROOT / "results" / "2026-11-19_overnight"
    R20 = ROOT / "results" / "2026-11-20_overnight"
    P2 = R20 / "part2"
    REF_G_EN = ROOT / "results" / "reference" / "golden" / "nll_golden_float32.txt"
    REF_G_UK32 = P2 / "kv_final" / "uk32" / "fp32.txt"

    specs = [
        (
            "BitNet-b1.58-2B-4T", "int8_fused (engine vs fp32)",
            bn_en_meta, bn_uk_meta,
            R19 / "ppl_kv_v2" / "fp32.txt", R19 / "ppl_kv_v2_uk" / "fp32.txt",
            R19 / "ppl_kv_v2" / "int8_fused.txt", R19 / "ppl_kv_v2_uk" / "int8_fused.txt",
        ),
        (
            "BitNet-b1.58-2B-4T", "S3+s4r32 (uncalibrated)",
            bn_en_meta, bn_uk_meta,
            REF_G_EN, REF_G_UK32,
            R19 / "kv_ptq" / "final_en16" / "S3_s4r32.txt", P2 / "kv_final" / "uk32" / "S3_s4r32.txt",
        ),
        (
            "BitNet-b1.58-2B-4T", "S13+s4r32 (K INT4, V ternary)",
            bn_en_meta, bn_uk_meta,
            REF_G_EN, REF_G_UK32,
            P2 / "kv_final" / "en16" / "S13_s4r32.txt", P2 / "kv_final" / "uk32" / "S13_s4r32.txt",
        ),
        (
            "BitNet-b1.58-2B-4T", "S3+s4r32 calibrated on EN",
            bn_en_meta, bn_uk_meta,
            REF_G_EN, REF_G_UK32,
            P2 / "calib" / "eval_en16" / "calib_en.txt", P2 / "calib" / "eval_uk32" / "calib_en.txt",
        ),
        (
            "BitNet-b1.58-2B-4T", "S3+s4r32 calibrated on UK",
            bn_en_meta, bn_uk_meta,
            REF_G_EN, REF_G_UK32,
            P2 / "calib" / "eval_en16" / "calib_uk.txt", P2 / "calib" / "eval_uk32" / "calib_uk.txt",
        ),
        (
            "Qwen2.5-1.5B", "INT8 per token (tenzo_int8)",
            qw_en_meta, qw_uk_meta,
            P2 / "second_model" / "fp_en.txt", P2 / "second_model" / "fp_uk.txt",
            P2 / "second_model" / "tenzo_int8_en.txt", P2 / "second_model" / "tenzo_int8_uk.txt",
        ),
        (
            "Qwen2.5-1.5B", "S3+s4r32",
            qw_en_meta, qw_uk_meta,
            P2 / "second_model" / "fp_en.txt", P2 / "second_model" / "fp_uk.txt",
            P2 / "second_model" / "S3_s4r32_en.txt", P2 / "second_model" / "S3_s4r32_uk.txt",
        ),
    ]

    rs = np.random.RandomState(0)
    summary_rows = []
    window_rows = []

    for model, scheme, en_meta, uk_meta, ge_p, gu_p, ce_p, cu_p in specs:
        GE, GU = rd_scores(ge_p), rd_scores(gu_p)
        CE, CU = rd_scores(ce_p), rd_scores(cu_p)
        ie = sorted(set(GE) & set(CE))
        iu = sorted(set(GU) & set(CU))

        def extract(ids, meta_list, C, G):
            fert = np.array([meta_list[i]["fertility"] for i in ids], dtype=np.float64)
            n_ch = np.array([meta_list[i]["n_chars"] for i in ids], dtype=np.float64)
            n_by = np.array([meta_list[i]["n_bytes"] for i in ids], dtype=np.float64)
            n_tk = np.array([C[i][0] for i in ids], dtype=np.float64)
            for i in ids:
                assert C[i][0] == G[i][0] == meta_list[i]["n_tok"], (i, C[i][0], G[i][0], meta_list[i]["n_tok"])
            c_nll = np.array([C[i][1] for i in ids], dtype=np.float64)
            g_nll = np.array([G[i][1] for i in ids], dtype=np.float64)
            bpc_c = c_nll / (n_ch * LN2)
            bpc_0 = g_nll / (n_ch * LN2)
            bpb_c = c_nll / (n_by * LN2)
            bpb_0 = g_nll / (n_by * LN2)
            return fert, n_tk, n_ch, n_by, bpc_0, bpc_c - bpc_0, bpb_0, bpb_c - bpb_0

        fe, nte, nce, nbe, bpc0_e, dbpc_e, bpb0_e, dbpb_e = extract(ie, en_meta, CE, GE)
        fu, ntu, ncu, nbu, bpc0_u, dbpc_u, bpb0_u, dbpb_u = extract(iu, uk_meta, CU, GU)

        for lang_name, ids, fert, nt, nc, nb, bpc0, dbpc, bpb0, dbpb in (
            ("en", ie, fe, nte, nce, nbe, bpc0_e, dbpc_e, bpb0_e, dbpb_e),
            ("uk", iu, fu, ntu, ncu, nbu, bpc0_u, dbpc_u, bpb0_u, dbpb_u),
        ):
            for k, wid in enumerate(ids):
                window_rows.append({
                    "model": model, "scheme": scheme, "lang": lang_name, "wid": wid,
                    "n_tok": int(nt[k]), "n_chars": int(nc[k]), "n_bytes": int(nb[k]),
                    "fertility": float(fert[k]),
                    "base_bpc": float(bpc0[k]), "dbpc": float(dbpc[k]),
                    "base_bpb": float(bpb0[k]), "dbpb": float(dbpb[k]),
                    "dnll_per_char": float(dbpc[k] * LN2),
                })

        draws = []
        for _ in range(10000):
            se = rs.randint(0, len(ie), len(ie))
            su = rs.randint(0, len(iu), len(iu))
            c_bpc = pooled_c(bpc0_e[se], dbpc_e[se], bpc0_u[su], dbpc_u[su])[2]
            c_bpb = pooled_c(bpb0_e[se], dbpb_e[se], bpb0_u[su], dbpb_u[su])[2]
            r_fe = pearson_r(fe[se], dbpc_e[se] * LN2)
            r_fu = pearson_r(fu[su], dbpc_u[su] * LN2)
            draws.append((dbpc_e[se].mean(), dbpc_u[su].mean(), dbpb_e[se].mean(), dbpb_u[su].mean(), c_bpc, c_bpb, r_fe, r_fu))
        draws = np.array(draws)
        ci = lambda col: (float(np.percentile(draws[:, col], 2.5)), float(np.percentile(draws[:, col], 97.5)))
        coef_bpc = pooled_c(bpc0_e, dbpc_e, bpc0_u, dbpc_u)
        coef_bpb = pooled_c(bpb0_e, dbpb_e, bpb0_u, dbpb_u)
        c_bpc_ci = ci(4)
        c_bpb_ci = ci(5)

        for lang_name, ids, fert, nc, nb, bpc0, dbpc, bpb0, dbpb, col_bpc, col_bpb, col_r in (
            ("en", ie, fe, nce, nbe, bpc0_e, dbpc_e, bpb0_e, dbpb_e, 0, 2, 6),
            ("uk", iu, fu, ncu, nbu, bpc0_u, dbpc_u, bpb0_u, dbpb_u, 1, 3, 7),
        ):
            _, slope_bpc, r2_bpc = ols(bpc0, dbpc)
            _, slope_bpb, r2_bpb = ols(bpb0, dbpb)
            _, slope_fert, r2_fert = ols(fert, dbpc * LN2)
            r_fert = pearson_r(fert, dbpc * LN2)
            bpc_ci = ci(col_bpc)
            bpb_ci = ci(col_bpb)
            r_ci = ci(col_r)
            summary_rows.append({
                "model": model, "scheme": scheme, "lang": lang_name, "windows": len(ids),
                "mean_chars": float(nc.mean()), "mean_bytes": float(nb.mean()),
                "mean_fertility": float(fert.mean()),
                "base_bpc": float(bpc0.mean()), "dbpc": float(dbpc.mean()),
                "dbpc_ci_lo": bpc_ci[0], "dbpc_ci_hi": bpc_ci[1],
                "slope_bpc": slope_bpc, "r2_bpc": r2_bpc,
                "pooled_c_uk_bpc": float(coef_bpc[2]), "pooled_c_uk_bpc_lo": c_bpc_ci[0], "pooled_c_uk_bpc_hi": c_bpc_ci[1],
                "base_bpb": float(bpb0.mean()), "dbpb": float(dbpb.mean()),
                "dbpb_ci_lo": bpb_ci[0], "dbpb_ci_hi": bpb_ci[1],
                "slope_bpb": slope_bpb, "r2_bpb": r2_bpb,
                "pooled_c_uk_bpb": float(coef_bpb[2]), "pooled_c_uk_bpb_lo": c_bpb_ci[0], "pooled_c_uk_bpb_hi": c_bpb_ci[1],
                "r_dnll_char_vs_fertility": r_fert, "r_fert_ci_lo": r_ci[0], "r_fert_ci_hi": r_ci[1],
                "slope_dnll_char_vs_fertility": slope_fert, "r2_fert": r2_fert,
            })

    with (OUT / "disparity_bpc.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with (OUT / "disparity_bpc_windows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
        w.writeheader()
        w.writerows(window_rows)

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    cpu = subprocess.check_output("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2", shell=True, text=True).strip()
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    load = Path("/proc/loadavg").read_text().strip()

    md = [
        "# Language Disparity Normalized per Byte and Character (L3 / T40 part 1)",
        "",
        f"Metadata: commit `{commit}`, CPU `{cpu}`, governor `{gov}`, load `{load}`, offline score analysis (no model runs).",
        "",
        "## 1. BPC and BPB Disparity (`dBPC = BPC(scheme) - BPC(fp)`)",
        "",
        "| model | scheme | lang | win | fertility (tok/char) | base BPC | dBPC [95% CI] | pooled c_UK(BPC) [95% CI] | base BPB | dBPB [95% CI] | pooled c_UK(BPB) [95% CI] |",
        "|---|---|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['model']} | {r['scheme']} | {r['lang']} | {r['windows']} | {r['mean_fertility']:.4f} | "
            f"{r['base_bpc']:.4f} | {r['dbpc']:+.4f} [{r['dbpc_ci_lo']:+.4f}, {r['dbpc_ci_hi']:+.4f}] | "
            f"{r['pooled_c_uk_bpc']:+.4f} [{r['pooled_c_uk_bpc_lo']:+.4f}, {r['pooled_c_uk_bpc_hi']:+.4f}] | "
            f"{r['base_bpb']:.4f} | {r['dbpb']:+.4f} [{r['dbpb_ci_lo']:+.4f}, {r['dbpb_ci_hi']:+.4f}] | "
            f"{r['pooled_c_uk_bpb']:+.4f} [{r['pooled_c_uk_bpb_lo']:+.4f}, {r['pooled_c_uk_bpb_hi']:+.4f}] |"
        )

    md += [
        "",
        "## 2. Within-Language Fertility Correlation (`dNLL/char` vs `tokens/char`)",
        "",
        "| model | scheme | lang | win | fertility (tok/char) | r(dNLL/char, fertility) [95% CI] | slope (nats/char per tok/char) | R2 |",
        "|---|---|---|---:|---:|---|---|---:|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['model']} | {r['scheme']} | {r['lang']} | {r['windows']} | {r['mean_fertility']:.4f} | "
            f"{r['r_dnll_char_vs_fertility']:+.3f} [{r['r_fert_ci_lo']:+.3f}, {r['r_fert_ci_hi']:+.3f}] | "
            f"{r['slope_dnll_char_vs_fertility']:+.4f} | {r['r2_fert']:.3f} |"
        )

    # Plots from window_rows
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = list(dict.fromkeys((r["model"], r["scheme"]) for r in summary_rows))
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2))
    axes_flat = axes.ravel()
    for idx, (m, s) in enumerate(pairs):
        ax = axes_flat[idx]
        for lang, col in (("en", "tab:blue"), ("uk", "tab:orange")):
            sub = [w for w in window_rows if w["model"] == m and w["scheme"] == s and w["lang"] == lang]
            xb = np.array([w["base_bpc"] for w in sub])
            yb = np.array([w["dbpc"] for w in sub])
            ax.scatter(xb, yb, s=16, c=col, label=lang.upper(), alpha=0.85)
            ia, ib, _ = ols(xb, yb)
            xs = np.linspace(xb.min(), xb.max(), 20)
            ax.plot(xs, ia + ib * xs, c=col, lw=1.2)
        ax.set_title(f"{m.split('-')[0]}: {s}", fontsize=8)
        ax.set_xlabel("base BPC (bits/char)", fontsize=7)
        ax.set_ylabel("dBPC (bits/char)", fontsize=7)
        ax.legend(fontsize=7)
    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "disparity_bpc_scatter.png", dpi=130)
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 4, figsize=(15.5, 7.2))
    axes2_flat = axes2.ravel()
    for idx, (m, s) in enumerate(pairs):
        ax = axes2_flat[idx]
        for lang, col in (("en", "tab:blue"), ("uk", "tab:orange")):
            sub = [w for w in window_rows if w["model"] == m and w["scheme"] == s and w["lang"] == lang]
            xf = np.array([w["fertility"] for w in sub])
            yf = np.array([w["dnll_per_char"] for w in sub])
            ax.scatter(xf, yf, s=16, c=col, label=lang.upper(), alpha=0.85)
            ia, ib, _ = ols(xf, yf)
            xs = np.linspace(xf.min(), xf.max(), 20)
            ax.plot(xs, ia + ib * xs, c=col, lw=1.2)
        ax.set_title(f"{m.split('-')[0]}: {s}", fontsize=8)
        ax.set_xlabel("fertility (tok/char)", fontsize=7)
        ax.set_ylabel("dNLL/char (nats/char)", fontsize=7)
        ax.legend(fontsize=7)
    for j in range(len(pairs), len(axes2_flat)):
        axes2_flat[j].axis("off")
    fig2.tight_layout()
    fig2.savefig(OUT / "fertility_corr.png", dpi=130)
    plt.close(fig2)

    (OUT / "disparity_bpc.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
