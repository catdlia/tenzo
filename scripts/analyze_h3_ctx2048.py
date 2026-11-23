#!/usr/bin/env python3
"""Analyze H3 (ctx=2048) score files across position blocks:
  1..512, 513..1024, 1025..1536, 1537..2047.

Compares golden (fp32, tenzo_int8, S3+s4r32, S13+s4r32) and Tenzo engine
(int8_fused, ternary_rot, plus optional ternary_rot with calibrated mix alphas).
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
H3_DIR = ROOT / "results" / "2026-11-23_agent2" / "H3"
BLOCKS = [(0, 512, "1..512"), (512, 1024, "513..1024"), (1024, 1536, "1025..1536"), (1536, 2047, "1537..2047")]


def load_lps(path: pathlib.Path) -> list[list[float]]:
    lps = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("lp "):
            lps.append([float(x) for x in ln.split()[1:]])
    return lps


def ppl_from_lps(seqs: list[list[float]], lo: int = 0, hi: int | None = None) -> tuple[float, int]:
    total_nll = 0.0
    total_n = 0
    for seq in seqs:
        sl = seq[lo:hi]
        total_nll -= sum(sl)
        total_n += len(sl)
    return (math.exp(total_nll / total_n) if total_n > 0 else float("nan")), total_n


def boot_ratio_ci(seqs_a: list[list[float]], seqs_b: list[list[float]], lo: int = 0, hi: int | None = None, seed: int = 0):
    rng = np.random.RandomState(seed)
    na = np.array([len(s[lo:hi]) for s in seqs_a], float)
    nb = np.array([len(s[lo:hi]) for s in seqs_b], float)
    la = np.array([-sum(s[lo:hi]) for s in seqs_a], float)
    lb = np.array([-sum(s[lo:hi]) for s in seqs_b], float)
    n_w = min(len(na), len(nb))
    if n_w == 0:
        return float("nan"), float("nan"), float("nan")
    na, nb, la, lb = na[:n_w], nb[:n_w], la[:n_w], lb[:n_w]
    pt = math.exp(la.sum() / na.sum()) / math.exp(lb.sum() / nb.sum())
    vals = []
    for _ in range(10000):
        idx = rng.randint(0, n_w, n_w)
        vals.append(math.exp(la[idx].sum() / na[idx].sum()) / math.exp(lb[idx].sum() / nb[idx].sum()))
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    scores_dir = H3_DIR / "scores"
    specs = [
        ("golden_fp32", "golden", "fp32", "fp32"),
        ("golden_tenzo_int8", "golden", "tenzo_int8", "golden_fp32"),
        ("golden_S3_s4r32", "golden", "S3+s4r32", "golden_fp32"),
        ("golden_S13_s4r32", "golden", "S13+s4r32", "golden_fp32"),
        ("engine_int8_fused", "engine", "int8_fused", "golden_fp32"),
        ("engine_ternary_rot", "engine", "ternary_rot", "engine_int8_fused"),
        ("engine_ternary_rot_alpha", "engine", "ternary_rot+alpha_mix", "engine_int8_fused"),
    ]

    csv_rows = []
    md = [
        "# H3 — Long Context Evaluation (ctx = 2048, 4 EN + 4 UK Windows)\n",
        "All windows are `[BOS] + 2047` BitNet tokens (zero 32-gram overlap with `calib_en8.ids` / `calib_uk8.ids`).\n",
    ]

    for lang in ("en", "uk"):
        md.append(f"## {lang.upper()} (`windows_{lang}_2048.ids`)\n")
        md.append("| Source | Mode | Full PPL | Ratio vs Ref (95% CI) | 1..512 | 513..1024 | 1025..1536 | 1537..2047 |")
        md.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |")
        loaded: dict[str, list[list[float]]] = {}
        for key, _, _, _ in specs:
            p = scores_dir / f"{key}_{lang}.txt"
            if p.exists():
                loaded[key] = load_lps(p)

        block_ratio_rows: list[str] = [
            f"### {lang.upper()} Block Ratios vs Ref\n",
            "| Source | Mode | Ref | Full Ratio | 1..512 | 513..1024 | 1025..1536 | 1537..2047 | Peak RSS |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]

        for key, src, label, ref_key in specs:
            if key not in loaded or len(loaded[key]) < 4:
                continue
            seqs = loaded[key]
            full_ppl, _ = ppl_from_lps(seqs)
            block_ppls = [ppl_from_lps(seqs, lo, hi)[0] for lo, hi, _ in BLOCKS]
            if ref_key in loaded and key != ref_key:
                r, lo, hi = boot_ratio_ci(seqs, loaded[ref_key])
                ratio_str = f"×{r:.4f} [{lo:.4f}, {hi:.4f}]"
            else:
                r, lo, hi = 1.0, 1.0, 1.0
                ratio_str = "×1.0000 (ref)"
            md.append(
                f"| {src} | `{label}` | {full_ppl:.3f} | {ratio_str} | "
                + " | ".join(f"{bp:.3f}" for bp in block_ppls)
                + " |"
            )
            log_path = scores_dir / f"{key}_{lang}.log"
            rss_str = "—"
            if log_path.exists():
                txt = log_path.read_text(encoding="utf-8", errors="replace")
                import re
                m_hwm = re.search(r"PROC_HWM_KB=(\d+)", txt)
                m_rss = re.search(r"peak_rss=([0-9.]+GB)", txt)
                if m_hwm:
                    rss_str = f"{int(m_hwm.group(1)) / 1024.0:.1f} MB"
                elif m_rss:
                    rss_str = m_rss.group(1)

            row = {
                "lang": lang,
                "key": key,
                "source": src,
                "mode": label,
                "ref": ref_key,
                "n_windows": len(seqs),
                "ppl_full": full_ppl,
                "ratio_vs_ref": r,
                "ci_lo": lo,
                "ci_hi": hi,
                "ppl_1_512": block_ppls[0],
                "ppl_513_1024": block_ppls[1],
                "ppl_1025_1536": block_ppls[2],
                "ppl_1537_2047": block_ppls[3],
                "peak_rss": rss_str,
            }
            block_ratios: list[float] = []
            for (blo, bhi, bname), _ in zip(BLOCKS, block_ppls):
                if ref_key in loaded and key != ref_key:
                    br, _, _ = boot_ratio_ci(seqs, loaded[ref_key], blo, bhi)
                else:
                    br = 1.0
                row[f"ratio_{bname}"] = br
                block_ratios.append(br)
            csv_rows.append(row)
            block_ratio_rows.append(
                f"| {src} | `{label}` | `{ref_key}` | ×{r:.4f} | "
                + " | ".join(f"×{br:.4f}" for br in block_ratios)
                + f" | {rss_str} |"
            )
        md.append("")
        md.extend(block_ratio_rows)
        md.append("")

    if not csv_rows:
        print("No H3 score files found yet.")
        return

    # Simulator-vs-engine parity at ctx=2048 (computed via compare_engine_sim.py formula)
    md.append("## Simulator-vs-Engine Parity (ctx = 2048, N = 4 windows)\n")
    md.append("| Lang | Mode Pair | Engine PPL | Sim PPL | Raw Ratio [95% CI] | Adj Ratio [95% CI] | Contains 1 |")
    md.append("| --- | --- | ---: | ---: | --- | --- | --- |")
    pair_specs = [
        ("en", "int8_fused vs tenzo_int8", "engine_int8_fused", "golden_tenzo_int8", "golden_fp32", "golden_fp32"),
        ("en", "ternary_rot vs S3+s4r32", "engine_ternary_rot", "golden_S3_s4r32", "engine_int8_fused", "golden_fp32"),
        ("uk", "int8_fused vs tenzo_int8", "engine_int8_fused", "golden_tenzo_int8", "golden_fp32", "golden_fp32"),
        ("uk", "ternary_rot vs S3+s4r32", "engine_ternary_rot", "golden_S3_s4r32", "engine_int8_fused", "golden_fp32"),
    ]
    for lang, label, ek, sk, ebk, gbk in pair_specs:
        seqs_map = {k: load_lps(scores_dir / f"{k}_{lang}.txt") for k in {ek, sk, ebk, gbk} if (scores_dir / f"{k}_{lang}.txt").exists()}
        if any(k not in seqs_map for k in (ek, sk, ebk, gbk)):
            continue
        ep, _ = ppl_from_lps(seqs_map[ek])
        sp, _ = ppl_from_lps(seqs_map[sk])
        raw_r, raw_lo, raw_hi = boot_ratio_ci(seqs_map[ek], seqs_map[sk])
        # Baseline-adjusted: (engine/engine_base) / (sim/golden_base)
        rng = np.random.RandomState(0)
        n_w = min(len(seqs_map[ek]), len(seqs_map[sk]), len(seqs_map[ebk]), len(seqs_map[gbk]))
        def _arr(seqs):
            return np.array([-sum(s) for s in seqs[:n_w]], float), np.array([len(s) for s in seqs[:n_w]], float)
        el, en = _arr(seqs_map[ek])
        sl, sn = _arr(seqs_map[sk])
        ebl, ebn = _arr(seqs_map[ebk])
        gbl, gbn = _arr(seqs_map[gbk])
        adj_pt = (math.exp(el.sum() / en.sum()) / math.exp(ebl.sum() / ebn.sum())) / (
            math.exp(sl.sum() / sn.sum()) / math.exp(gbl.sum() / gbn.sum())
        )
        adj_vals = []
        for _ in range(10000):
            idx = rng.randint(0, n_w, n_w)
            adj_vals.append(
                (math.exp(el[idx].sum() / en[idx].sum()) / math.exp(ebl[idx].sum() / ebn[idx].sum()))
                / (math.exp(sl[idx].sum() / sn[idx].sum()) / math.exp(gbl[idx].sum() / gbn[idx].sum()))
            )
        adj_lo, adj_hi = float(np.percentile(adj_vals, 2.5)), float(np.percentile(adj_vals, 97.5))
        c1 = (raw_lo <= 1.0 <= raw_hi) and (adj_lo <= 1.0 <= adj_hi)
        md.append(
            f"| {lang} | `{label}` | {ep:.3f} | {sp:.3f} | ×{raw_r:.4f} [{raw_lo:.4f}, {raw_hi:.4f}] | "
            f"×{adj_pt:.4f} [{adj_lo:.4f}, {adj_hi:.4f}] | `{c1}` |"
        )
    md.append("")
    md.append("## Observations & Caveats\n")
    md.append("- **Engine vs Golden parity holds at ctx = 2048**: `int8_fused` (`×0.9998` EN, `×0.9993` UK vs `golden_fp32`) and `ternary_rot` (baseline-adjusted engine/sim ratio `1.0006 [0.9965, 1.0034]` EN, `1.0030 [0.9991, 1.0083]` UK) match the Python golden simulator within statistical noise across full 2048-token contexts.")
    md.append("- **Asymmetric positional drift**: on EN, `engine ternary_rot` vs `int8_fused` stays bounded across position blocks (`×1.0673` -> `×1.0557` -> `×1.0860` -> `×1.0659`), and `S13+s4r32` stays near `×1.000–1.018`. On UK, `engine ternary_rot` degrades monotonically with context position (`×1.1443` in `1..512`, `×1.1650` in `513..1024`, `×1.2432` in `1025..1536`, `×1.3051` in `1537..2047`; `golden S3+s4r32` `×1.1382` -> `×1.3019`), and `S13+s4r32` rises from `×1.0866` (`1..512`) to `×1.1358` (`1537..2047`).")
    md.append("- **Calibrated mix alphas (`alpha_mix.txt`) generalize to ctx = 2048 without overfitting**: calibrated on 512-token EN+UK windows, `ternary_rot+alpha_mix` lowers full PPL ratio from `×1.0687` to `×1.0381` (EN) and from `×1.2127` to `×1.1643` (UK), improving every block (UK `1537..2047` drops from `×1.3051` to `×1.2511`), while not removing the monotonic UK position growth.")
    md.append("- **Memory budget**: Tenzo engine `PROC_HWM_KB` stays below 1 GB at `ctx=2048` (`950.4–950.7 MB` `int8_fused`; `902.3–902.8 MB` `ternary_rot`, ~48 MB smaller than `int8_fused`).")
    md.append("- **Caveat**: each language has `N = 4` long windows (8,188 scored tokens), so 4-window bootstrap CIs are coarser than at ctx=512 (`N = 16` / `32`), though the directional block-by-block EN-vs-UK divergence is consistent across both golden and engine runs.\n")
    md.append("## Reproducibility & Environment\n")
    md.append("- Windows: `python3 scripts/make_windows_2048.py` -> `results/reference/windows_en_2048.ids` (`sha256: 66bc4ca09d6e49d3...`), `results/reference/windows_uk_2048.ids` (`sha256: 403a0f9a71daa621...`).")
    md.append("- Golden: `python3 scripts/run_h3_ctx2048.py --mode golden-all --batch 4 --threads 4` (256-row chunked log-softmax, peak RSS 3.01–3.55 GB under `MemoryMax=4G`).")
    md.append("- Engine: `python3 scripts/run_h3_ctx2048.py --mode engine-all` (`-c 2048 --prefill batched`, with `TENZO_KV_ALPHA=/app/results/2026-11-23_agent2/H1/alpha_mix.txt` for `ternary_rot+alpha_mix`).")
    md.append("- Analysis: `python3 scripts/analyze_h3_ctx2048.py` (CPU: `12th Gen Intel(R) Core(TM) i3-1215U`, governor: `powersave`, binary built on `agent2/2026-11-23`).\n")

    fields = list(csv_rows[0].keys())
    with (H3_DIR / "ctx2048.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
        x = [b[2] for b in BLOCKS]
        plot_modes = [
            ("engine_ternary_rot", "engine ternary_rot (uncal)", "-o", "#d62728"),
            ("engine_ternary_rot_alpha", "engine ternary_rot+alpha_mix", "-s", "#ff7f0e"),
            ("golden_S3_s4r32", "golden S3+s4r32", "--^", "#8c564b"),
            ("golden_S13_s4r32", "golden S13+s4r32", "-d", "#1f77b4"),
            ("engine_int8_fused", "engine int8_fused", ":.", "#2ca02c"),
        ]
        for ax, lang in zip(axes, ("en", "uk")):
            by_key = {r["key"]: r for r in csv_rows if r["lang"] == lang}
            for k, lbl, fmt, col in plot_modes:
                if k not in by_key:
                    continue
                ys = [by_key[k][f"ratio_{b[2]}"] for b in BLOCKS]
                ax.plot(x, ys, fmt, label=lbl, color=col, linewidth=1.8, markersize=5)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(f"{lang.upper()} (4x2048 windows)")
            ax.set_xlabel("Context token position block")
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Block PPL Ratio vs Reference")
        axes[1].legend(fontsize=8, loc="upper left")
        fig.suptitle("BitNet-b1.58-2B-4T KV Quantization Ratio Across Context Position (ctx=2048)")
        fig.tight_layout()
        fig.savefig(H3_DIR / "ctx2048.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}")

    (H3_DIR / "ctx2048.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote ctx2048.{csv,md,png}")


if __name__ == "__main__":
    main()
