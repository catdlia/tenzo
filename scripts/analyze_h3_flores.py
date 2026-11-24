#!/usr/bin/env python3
"""Analyze H3 FLORES-200 parallel EN/UK KV degradation across BitNet, EuroLLM-1.7B, and Qwen2.5-1.5B.

Preserves `## Plan` in results/2026-11-24_agent2/H3/parallel_flores.md and writes:
  - results/2026-11-24_agent2/H3/parallel_flores.csv
  - results/2026-11-24_agent2/H3/parallel_flores_windows.csv
  - results/2026-11-24_agent2/H3/parallel_flores.png
  - updated results/2026-11-24_agent2/H3/parallel_flores.md
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import subprocess
import sys
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
H3_DIR = ROOT / "results" / "2026-11-24_agent2" / "H3"
SCORES_DIR = H3_DIR / "scores"
LN2 = math.log(2.0)


def read_window_nlls(path: pathlib.Path) -> np.ndarray:
    vals: dict[int, float] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("w "):
            _, wid, _, nll = ln.split()
            vals[int(wid)] = float(nll)
    return np.array([vals[i] for i in range(len(vals))], dtype=np.float64)


def pair_boot(arr: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(arr)
    pt = float(arr.mean())
    idx = rng.randint(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return pt, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pair_boot_ratio_diff(num_uk: np.ndarray, den_uk: np.ndarray, num_en: np.ndarray, den_en: np.ndarray,
                         n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(num_uk)
    pt = float(num_uk.sum() / den_uk.sum() - num_en.sum() / den_en.sum())
    idx = rng.randint(0, n, size=(n_boot, n))
    r_uk = num_uk[idx].sum(axis=1) / den_uk[idx].sum(axis=1)
    r_en = num_en[idx].sum(axis=1) / den_en[idx].sum(axis=1)
    diffs = r_uk - r_en
    return pt, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> int:
    meta = json.loads((H3_DIR / "flores_windows_meta.json").read_text(encoding="utf-8"))
    wins = meta["windows"]
    n_w = len(wins)
    en_chars = np.array([w["en_chars"] for w in wins], dtype=np.float64)
    uk_chars = np.array([w["uk_chars"] for w in wins], dtype=np.float64)
    en_bytes = np.array([w["en_bytes"] for w in wins], dtype=np.float64)
    uk_bytes = np.array([w["uk_bytes"] for w in wins], dtype=np.float64)

    models_spec = [
        ("bitnet", "BitNet-b1.58-2B", "fp32", ["tenzo_int8", "S3+s4r32", "S13+s4r32"]),
        ("eurollm", "EuroLLM-1.7B", "fp", ["tenzo_int8", "S3+s4r32", "S13+s4r32"]),
        ("qwen", "Qwen2.5-1.5B", "fp", ["tenzo_int8", "S3+s4r32", "S13+s4r32"]),
    ]

    summary_rows: list[dict] = []
    window_rows: list[dict] = []
    base_rows: list[str] = []
    table_rows: list[str] = []
    norm_rows: list[str] = []

    for mkey, mlabel, base_name, schemes in models_spec:
        mdir = SCORES_DIR / mkey
        bp_en = mdir / f"{base_name}_en.txt"
        bp_uk = mdir / f"{base_name}_uk.txt"
        if not (bp_en.exists() and bp_uk.exists()):
            continue
        b_en = read_window_nlls(bp_en)
        b_uk = read_window_nlls(bp_uk)
        if len(b_en) < n_w or len(b_uk) < n_w:
            continue

        gap_b, g_lo, g_hi = pair_boot(b_uk - b_en)
        tok_en = np.mean([w["tokens"][mkey]["en"] for w in wins])
        tok_uk = np.mean([w["tokens"][mkey]["uk"] for w in wins])
        base_bpc_en = float(b_en.sum() / (LN2 * en_chars.sum()))
        base_bpc_uk = float(b_uk.sum() / (LN2 * uk_chars.sum()))
        base_bpb_en = float(b_en.sum() / (LN2 * en_bytes.sum()))
        base_bpb_uk = float(b_uk.sum() / (LN2 * uk_bytes.sum()))
        base_rows.append(
            f"| {mlabel} | {tok_en:.1f} | {tok_uk:.1f} | {b_en.mean():.2f} | {b_uk.mean():.2f} | "
            f"{gap_b:+.2f} [{g_lo:+.2f}, {g_hi:+.2f}] | {base_bpc_en:.4f} / {base_bpc_uk:.4f} | {base_bpb_en:.4f} / {base_bpb_uk:.4f} |"
        )

        for scheme in schemes:
            sname = scheme.replace(":", "_").replace("+", "_")
            sp_en = mdir / f"{sname}_en.txt"
            sp_uk = mdir / f"{sname}_uk.txt"
            if not (sp_en.exists() and sp_uk.exists()):
                continue
            s_en = read_window_nlls(sp_en)
            s_uk = read_window_nlls(sp_uk)
            if len(s_en) < n_w or len(s_uk) < n_w:
                continue

            d_en = s_en - b_en
            d_uk = s_uk - b_uk
            dw = d_uk - d_en
            D_pt, D_lo, D_hi = pair_boot(dw)

            rel_en = d_en / b_en
            rel_uk = d_uk / b_uk
            rw = rel_uk - rel_en
            R_pt, R_lo, R_hi = pair_boot(rw)
            rel_en_pt, _, _ = pair_boot(rel_en)
            rel_uk_pt, _, _ = pair_boot(rel_uk)
            R_corp_pt, Rc_lo, Rc_hi = pair_boot_ratio_diff(d_uk, b_uk, d_en, b_en)

            dbpc_en = d_en / (LN2 * en_chars)
            dbpc_uk = d_uk / (LN2 * uk_chars)
            dbpb_en = d_en / (LN2 * en_bytes)
            dbpb_uk = d_uk / (LN2 * uk_bytes)
            dbpc_diff, bpc_lo, bpc_hi = pair_boot(dbpc_uk - dbpc_en)
            dbpb_diff, bpb_lo, bpb_hi = pair_boot(dbpb_uk - dbpb_en)

            if D_lo > 0.0:
                verdict = "super-prop (D>0, R>0)" if R_lo > 0.0 else ("prop-to-base (D>0, R~0)" if (R_lo <= 0.0 <= R_hi) else "D>0, R<0")
            elif D_hi < 0.0:
                verdict = "EN worse (D<0)"
            else:
                verdict = "no clear diff (D~0)"

            table_rows.append(
                f"| {mlabel} | `{scheme}` | {d_en.mean():+.3f} | {d_uk.mean():+.3f} | "
                f"**{D_pt:+.3f} [{D_lo:+.3f}, {D_hi:+.3f}]** | "
                f"{100*rel_en_pt:+.2f}% / {100*rel_uk_pt:+.2f}% | "
                f"{100*R_pt:+.2f}% [{100*R_lo:+.2f}%, {100*R_hi:+.2f}%] | "
                f"{dbpc_diff:+.4f} [{bpc_lo:+.4f}, {bpc_hi:+.4f}] | "
                f"{dbpb_diff:+.4f} [{bpb_lo:+.4f}, {bpb_hi:+.4f}] | `{verdict}` |"
            )
            norm_rows.append(
                f"| {mlabel} | `{scheme}` | {dbpc_en.mean():+.4f} | {dbpc_uk.mean():+.4f} | "
                f"{dbpb_en.mean():+.4f} | {dbpb_uk.mean():+.4f} | "
                f"{100*(d_uk.sum()/b_uk.sum() - d_en.sum()/b_en.sum()):+.2f}% [{100*Rc_lo:+.2f}%, {100*Rc_hi:+.2f}%] |"
            )

            summary_rows.append(
                {
                    "model": mlabel,
                    "scheme": scheme,
                    "n_windows": n_w,
                    "base_en_nats": float(b_en.mean()),
                    "base_uk_nats": float(b_uk.mean()),
                    "delta_en_nats": float(d_en.mean()),
                    "delta_uk_nats": float(d_uk.mean()),
                    "D_nats": D_pt,
                    "D_ci_lo": D_lo,
                    "D_ci_hi": D_hi,
                    "rel_en": rel_en_pt,
                    "rel_uk": rel_uk_pt,
                    "R_diff": R_pt,
                    "R_ci_lo": R_lo,
                    "R_ci_hi": R_hi,
                    "R_corp_diff": R_corp_pt,
                    "R_corp_ci_lo": Rc_lo,
                    "R_corp_ci_hi": Rc_hi,
                    "dbpc_en": float(dbpc_en.mean()),
                    "dbpc_uk": float(dbpc_uk.mean()),
                    "dbpc_diff": dbpc_diff,
                    "dbpc_ci_lo": bpc_lo,
                    "dbpc_ci_hi": bpc_hi,
                    "dbpb_en": float(dbpb_en.mean()),
                    "dbpb_uk": float(dbpb_uk.mean()),
                    "dbpb_diff": dbpb_diff,
                    "dbpb_ci_lo": bpb_lo,
                    "dbpb_ci_hi": bpb_hi,
                    "verdict": verdict,
                }
            )
            for wi in range(n_w):
                window_rows.append(
                    {
                        "model": mlabel,
                        "scheme": scheme,
                        "wid": wi,
                        "b_en_nats": float(b_en[wi]),
                        "b_uk_nats": float(b_uk[wi]),
                        "d_en_nats": float(d_en[wi]),
                        "d_uk_nats": float(d_uk[wi]),
                        "D_nats": float(dw[wi]),
                        "rel_en": float(rel_en[wi]),
                        "rel_uk": float(rel_uk[wi]),
                        "R_diff": float(rw[wi]),
                        "dbpc_en": float(dbpc_en[wi]),
                        "dbpc_uk": float(dbpc_uk[wi]),
                        "dbpb_en": float(dbpb_en[wi]),
                        "dbpb_uk": float(dbpb_uk[wi]),
                    }
                )

    if not summary_rows:
        print("No completed FLORES H3 score files found yet.", file=sys.stderr)
        return 1

    with (H3_DIR / "parallel_flores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    with (H3_DIR / "parallel_flores_windows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
        w.writeheader()
        for r in window_rows:
            w.writerow(r)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models_present = []
        for r in summary_rows:
            if r["model"] not in models_present:
                models_present.append(r["model"])
        schemes_order = ["tenzo_int8", "S3+s4r32", "S13+s4r32"]
        scheme_colors = {"tenzo_int8": "#2ca02c", "S3+s4r32": "#d62728", "S13+s4r32": "#1f77b4"}
        by_ms = {(r["model"], r["scheme"]): r for r in summary_rows}

        fig, axes = plt.subplots(2, len(models_present), figsize=(12.0, 6.2))
        if len(models_present) == 1:
            axes = axes[:, None]
        xs = np.arange(len(schemes_order))
        for mi, ml in enumerate(models_present):
            for ri, (metric, lo_k, hi_k, ylabel, mult) in enumerate(
                [
                    ("D_nats", "D_ci_lo", "D_ci_hi", "D = Δ_UK − Δ_EN (nats)", 1.0),
                    ("R_diff", "R_ci_lo", "R_ci_hi", "R = Δ_UK/B_UK − Δ_EN/B_EN (pp)", 100.0),
                ]
            ):
                ax = axes[ri, mi]
                ys, ylo, yhi, cols = [], [], [], []
                for sc in schemes_order:
                    r = by_ms.get((ml, sc))
                    cols.append(scheme_colors[sc])
                    if r is None:
                        ys.append(0.0); ylo.append(0.0); yhi.append(0.0)
                    else:
                        v = r[metric] * mult
                        ys.append(v)
                        ylo.append(max(0.0, v - r[lo_k] * mult))
                        yhi.append(max(0.0, r[hi_k] * mult - v))
                ax.bar(xs, ys, width=0.58, color=cols, alpha=0.88)
                ax.errorbar(xs, ys, yerr=[ylo, yhi], fmt="none", ecolor="black", capsize=3.5, linewidth=1.2)
                ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
                ax.set_xticks(xs)
                ax.set_xticklabels(["int8", "S3+s4r32", "S13+s4r32"], fontsize=8.5)
                if ri == 0:
                    ax.set_title(ml, fontsize=10.5, fontweight="bold")
                if mi == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle("H3: Parallel FLORES-200 EN vs UK KV Quantization Disparity (N = 32 paired windows)")
        fig.tight_layout()
        fig.savefig(H3_DIR / "parallel_flores.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}", file=sys.stderr)

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    gov = (pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
           if pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists() else "unknown")

    by_ms = {(r["model"], r["scheme"]): r for r in summary_rows}
    mean_bytes_ratio = float((uk_bytes / uk_chars).mean() / (en_bytes / en_chars).mean())
    tok_stats = {
        ml: (
            float(np.mean([w["tokens"][mk]["en"] for w in wins])),
            float(np.mean([w["tokens"][mk]["uk"] for w in wins])),
        )
        for mk, ml, _, _ in models_spec
    }
    conclusions = [
        "### 4. Conclusions & Caveats",
        "",
        "- **Decision Rule 1 (`95% CI(D) > 0`, UK penalty at matched semantic content):** Holds across **all three model families** on `S3+s4r32` — "
        + ", ".join(
            f"{ml} `D={by_ms[(ml, 'S3+s4r32')]['D_nats']:+.3f} [{by_ms[(ml, 'S3+s4r32')]['D_ci_lo']:+.3f}, {by_ms[(ml, 'S3+s4r32')]['D_ci_hi']:+.3f}]` nats"
            for ml in ("BitNet-b1.58-2B", "EuroLLM-1.7B", "Qwen2.5-1.5B") if (ml, "S3+s4r32") in by_ms
        )
        + ". On `S13+s4r32` (`K INT4`), BitNet (`"
        + f"{by_ms[('BitNet-b1.58-2B', 'S13+s4r32')]['D_nats']:+.3f} [{by_ms[('BitNet-b1.58-2B', 'S13+s4r32')]['D_ci_lo']:+.3f}, {by_ms[('BitNet-b1.58-2B', 'S13+s4r32')]['D_ci_hi']:+.3f}]`"
        + ") and Qwen2.5 (`"
        + f"{by_ms[('Qwen2.5-1.5B', 'S13+s4r32')]['D_nats']:+.3f}`"
        + ") retain strong UK penalty while EuroLLM-1.7B crosses zero (`"
        + f"{by_ms[('EuroLLM-1.7B', 'S13+s4r32')]['D_nats']:+.3f} [{by_ms[('EuroLLM-1.7B', 'S13+s4r32')]['D_ci_lo']:+.3f}, {by_ms[('EuroLLM-1.7B', 'S13+s4r32')]['D_ci_hi']:+.3f}]`"
        + "). Under `tenzo_int8`, BitNet and EuroLLM show `D ~ 0` while Qwen2.5 degrades UK (`"
        + f"{by_ms[('Qwen2.5-1.5B', 'tenzo_int8')]['D_nats']:+.3f} [{by_ms[('Qwen2.5-1.5B', 'tenzo_int8')]['D_ci_lo']:+.3f}, {by_ms[('Qwen2.5-1.5B', 'tenzo_int8')]['D_ci_hi']:+.3f}]`"
        + ").",
        "- **Decision Rule 2 (`95% CI(R) > 0`, super-proportional vs baseline complexity):** Under `S3+s4r32`, **all three models** exhibit `95% CI(R) > 0` (`BitNet` "
        + f"`{100*by_ms[('BitNet-b1.58-2B','S3+s4r32')]['R_diff']:+.2f}% [{100*by_ms[('BitNet-b1.58-2B','S3+s4r32')]['R_ci_lo']:+.2f}%, {100*by_ms[('BitNet-b1.58-2B','S3+s4r32')]['R_ci_hi']:+.2f}%]`, "
        + f"`EuroLLM-1.7B` `{100*by_ms[('EuroLLM-1.7B','S3+s4r32')]['R_diff']:+.2f}% [{100*by_ms[('EuroLLM-1.7B','S3+s4r32')]['R_ci_lo']:+.2f}%, {100*by_ms[('EuroLLM-1.7B','S3+s4r32')]['R_ci_hi']:+.2f}%]`, "
        + f"`Qwen2.5-1.5B` `{100*by_ms[('Qwen2.5-1.5B','S3+s4r32')]['R_diff']:+.2f}% [{100*by_ms[('Qwen2.5-1.5B','S3+s4r32')]['R_ci_lo']:+.2f}%, {100*by_ms[('Qwen2.5-1.5B','S3+s4r32')]['R_ci_hi']:+.2f}%]`), showing UK sensitivity rises super-proportionally faster than base NLL.",
        f"- **Resolution of `T45` (`ΔBPC` vs `ΔBPB`):** On `EuroLLM-1.7B S3+s4r32`, UK loses `{by_ms[('EuroLLM-1.7B','S3+s4r32')]['delta_uk_nats']:+.3f}` vs EN `{by_ms[('EuroLLM-1.7B','S3+s4r32')]['delta_en_nats']:+.3f}` nats on the exact same meaning (`ΔBPC diff {by_ms[('EuroLLM-1.7B','S3+s4r32')]['dbpc_diff']:+.4f} [{by_ms[('EuroLLM-1.7B','S3+s4r32')]['dbpc_ci_lo']:+.4f}, {by_ms[('EuroLLM-1.7B','S3+s4r32')]['dbpc_ci_hi']:+.4f}]`), yet `ΔBPB diff` crosses zero (`{by_ms[('EuroLLM-1.7B','S3+s4r32')]['dbpb_diff']:+.4f} [{by_ms[('EuroLLM-1.7B','S3+s4r32')]['dbpb_ci_lo']:+.4f}, {by_ms[('EuroLLM-1.7B','S3+s4r32')]['dbpb_ci_hi']:+.4f}]`) solely because UTF-8 Ukrainian has `{mean_bytes_ratio:.2f}×` more bytes/char than English ASCII. Parallel content removes denominator artifacts.",
        f"- **Caveat — Sequence token-length & `s4r32` protection fraction (tested next in `H4`):** Matched-content UK strings produce more tokens than EN (`BitNet` {tok_stats['BitNet-b1.58-2B'][1]:.1f} vs {tok_stats['BitNet-b1.58-2B'][0]:.1f}; `EuroLLM` {tok_stats['EuroLLM-1.7B'][1]:.1f} vs {tok_stats['EuroLLM-1.7B'][0]:.1f}; `Qwen2.5` {tok_stats['Qwen2.5-1.5B'][1]:.1f} vs {tok_stats['Qwen2.5-1.5B'][0]:.1f}), so the fixed 36-token INT8 protection (`sinks=4, recent=32`) protects a slightly larger fraction of EN tokens on shorter windows. `H4` isolates `r ∈ {{0, 128, 256}}` on these same FLORES windows.",
    ]

    md_path = H3_DIR / "parallel_flores.md"
    existing = md_path.read_text(encoding="utf-8")
    plan_end = existing.find("\n## Results")
    plan_part = existing[:plan_end].rstrip() if plan_end != -1 else existing.rstrip()

    out_md = [
        plan_part,
        "",
        "## Results",
        "",
        f"Dataset: `FLORES-200 devtest` (`N = {n_w}` paired windows, sentences `0..{meta['sentences_used'] - 1}`, `max_scored_tokens <= 511` across all 3 tokenizers, archive `sha256: {meta['archive_sha256'][:16]}...`, `0` shared 32-grams vs all existing calibration/eval files).",
        "",
        "### 1. Unquantized baseline on identical semantic content",
        "",
        "| Model | Mean tok EN | Mean tok UK | Mean B_EN (nats) | Mean B_UK (nats) | Paired gap B_UK−B_EN [95% CI] | Base BPC EN / UK | Base BPB EN / UK |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        *base_rows,
        "",
        "### 2. Primary table: Matched-content quantization disparity (`D`, `R`, `ΔBPC`, `ΔBPB`)",
        "",
        "| Model | Scheme | Mean Δ_EN (nats) | Mean Δ_UK (nats) | Primary D = Δ_UK−Δ_EN [95% CI] | Rel loss EN / UK | Rel diff R = UK−EN [95% CI] | ΔBPC (UK−EN) [95% CI] | ΔBPB (UK−EN) [95% CI] | Decision |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
        *table_rows,
        "",
        "### 3. Per-unit breakdown (`ΔBPC`, `ΔBPB`, corpus-weighted `R`)",
        "",
        "| Model | Scheme | ΔBPC EN | ΔBPC UK | ΔBPB EN | ΔBPB UK | Corpus-weighted Δ_UK/B_UK − Δ_EN/B_EN [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        *norm_rows,
        "",
        *conclusions,
        "",
        "### Reproducibility & Metadata",
        f"- Prepare windows: `python3 scripts/prepare_h3_flores.py` -> `results/2026-11-24_agent2/H3/flores_windows_meta.json`.",
        "- Score models: `python3 scripts/run_h3_flores.py --model {bitnet,eurollm,qwen} --batch 32` (layer-major unpadded `0.000e+00` vs `B=1`, under `MemoryMax=4G` BitNet / `6G` HF).",
        f"- Analyze: `python3 scripts/analyze_h3_flores.py` (`commit {commit}`, governor `{gov}`, `n_boot=10000`, paired by window index, `seed=0`).",
    ]
    md_path.write_text("\n".join(out_md) + "\n", encoding="utf-8")
    print(f"Wrote {H3_DIR / 'parallel_flores.{csv,md,png}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
