#!/usr/bin/env python3
"""Evaluate H2 (2026-11-24): Cross-domain transfer of alpha_mix.txt in TenzoEngine (ctx=512).

Evaluates BitNet 2x2 windows (results/2026-11-23_agent2/H2/bitnet_{en,uk}_{wiki,web}.ids, 16x512 each)
under int8_fused, ternary_rot, and ternary_rot --kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt.
"""

from __future__ import annotations

import csv
import hashlib
import math
import pathlib
import subprocess
import sys
import time
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
OLD_H2_IDS_DIR = ROOT / "results" / "2026-11-23_agent2" / "H2"
H2_DIR = ROOT / "results" / "2026-11-24_agent2" / "H2"
SCORES_DIR = H2_DIR / "scores"
ALPHA_CONTAINER = "/app/results/2026-11-23_agent2/H1/alpha_mix.txt"


def load_lps(path: pathlib.Path) -> np.ndarray:
    rows = [[float(x) for x in ln.split()[1:]] for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("lp ")]
    return np.array(rows, dtype=np.float64)


def ppl_from_lps(lps: np.ndarray) -> float:
    return float(math.exp(-lps.sum() / lps.size))


def paired_ratio_ci(lps_a: np.ndarray, lps_b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    n_w, n_t = lps_a.shape
    a = -lps_a.sum(axis=1)
    b = -lps_b.sum(axis=1)
    pt = float(math.exp(a.sum() / (n_w * n_t)) / math.exp(b.sum() / (n_w * n_t)))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n_w, size=(n_boot, n_w))
    ra = np.exp(a[idx].sum(axis=1) / (n_w * n_t))
    rb = np.exp(b[idx].sum(axis=1) / (n_w * n_t))
    ratios = ra / rb
    return pt, float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def paired_gain_ci(lps_rot: np.ndarray, lps_alpha: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    # ratio_rot / ratio_alpha == ppl(rot) / ppl(alpha)
    return paired_ratio_ci(lps_rot, lps_alpha, n_boot=n_boot, seed=seed)


def run_engine_cell(cell: str, mode_key: str, kv_arg: str, alpha_c: str | None) -> pathlib.Path:
    ids_rel = f"results/2026-11-23_agent2/H2/bitnet_{cell}.ids"
    out_p = SCORES_DIR / f"{cell}__{mode_key}.txt"
    log_p = SCORES_DIR / f"{cell}__{mode_key}.log"
    if out_p.exists() and sum(1 for ln in out_p.read_text(encoding="utf-8").splitlines() if ln.startswith("lp ")) == 16:
        print(f"[skip] {cell} {mode_key} already complete", flush=True)
        return out_p
    cmd = [
        "nice", "-n", "19", "ionice", "-c3",
        "docker", "compose", "run", "--rm",
        "-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread",
        "dev", "/app/cmake-build-debug/tenzo-inference",
        "-m", "/app/models/bitnet-1.58b-v2",
        "--kv-quant", kv_arg,
        "-c", "512",
        "--prefill", "batched",
        "--score", f"/app/{ids_rel}",
        "--score-out", f"/app/{out_p.relative_to(ROOT)}",
    ]
    if alpha_c:
        cmd.extend(["--kv-alpha", alpha_c])
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log_p.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Engine failed for {cell} {mode_key}: rc={proc.returncode}")
    print(f"[done] {cell:8s} {mode_key:18s} in {time.time()-t0:.1f}s", flush=True)
    return out_p


def main() -> int:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    cells_meta = [
        ("en_wiki", "in-domain (WT2 calib ~ wikimedia en_wiki)"),
        ("uk_web", "in-domain (Kobza calib, rows 100-199 vs 200-299)"),
        ("en_web", "out-of-domain (FineWeb-Edu)"),
        ("uk_wiki", "out-of-domain (Wikimedia UK)"),
    ]
    modes = [
        ("int8_fused", "int8_fused", None),
        ("ternary_rot", "ternary_rot", None),
        ("ternary_rot_alpha", "ternary_rot", ALPHA_CONTAINER),
    ]

    for cell, _ in cells_meta:
        for mk, kv_arg, ac in modes:
            run_engine_cell(cell, mk, kv_arg, ac)

    csv_rows: list[dict] = []
    md_rows: list[str] = []
    alpha_headers: list[str] = []

    for cell, domain_status in cells_meta:
        lps_i8 = load_lps(SCORES_DIR / f"{cell}__int8_fused.txt")
        lps_rot = load_lps(SCORES_DIR / f"{cell}__ternary_rot.txt")
        lps_alp = load_lps(SCORES_DIR / f"{cell}__ternary_rot_alpha.txt")
        hdr = [ln for ln in (SCORES_DIR / f"{cell}__ternary_rot_alpha.txt").read_text(encoding="utf-8").splitlines() if ln.startswith("# kv_alpha=")]
        if hdr:
            alpha_headers.append(f"- `{cell}`: `{hdr[0]}`")

        p_i8 = ppl_from_lps(lps_i8)
        p_rot = ppl_from_lps(lps_rot)
        p_alp = ppl_from_lps(lps_alp)
        r_rot, rot_lo, rot_hi = paired_ratio_ci(lps_rot, lps_i8)
        r_alp, alp_lo, alp_hi = paired_ratio_ci(lps_alp, lps_i8)
        gain, g_lo, g_hi = paired_gain_ci(lps_rot, lps_alp)
        transfers = "YES (CI>1)" if g_lo > 1.0 else "NO (CI<=1)"

        md_rows.append(
            f"| `{cell}` | {domain_status} | {p_i8:.3f} | {p_rot:.3f} | ×{r_rot:.4f} [{rot_lo:.4f}, {rot_hi:.4f}] | "
            f"{p_alp:.3f} | ×{r_alp:.4f} [{alp_lo:.4f}, {alp_hi:.4f}] | "
            f"**×{gain:.4f} [{g_lo:.4f}, {g_hi:.4f}]** | {transfers} |"
        )
        csv_rows.append(
            {
                "cell": cell,
                "domain_status": domain_status,
                "n_windows": len(lps_i8),
                "ppl_int8_fused": p_i8,
                "ppl_ternary_rot": p_rot,
                "ratio_rot_vs_int8": r_rot, "rot_ci_lo": rot_lo, "rot_ci_hi": rot_hi,
                "ppl_ternary_rot_alpha": p_alp,
                "ratio_alpha_vs_int8": r_alp, "alpha_ci_lo": alp_lo, "alpha_ci_hi": alp_hi,
                "gain_rot_over_alpha": gain, "gain_ci_lo": g_lo, "gain_ci_hi": g_hi,
                "transfers_ci_gt_1": g_lo > 1.0,
            }
        )

    with (H2_DIR / "alpha_domain_transfer.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        xs = np.arange(len(csv_rows))
        labels = [r["cell"] for r in csv_rows]
        axes[0].bar(xs - 0.18, [r["ratio_rot_vs_int8"] for r in csv_rows], width=0.34, label="ternary_rot", color="#d62728", alpha=0.85)
        axes[0].bar(xs + 0.18, [r["ratio_alpha_vs_int8"] for r in csv_rows], width=0.34, label="ternary_rot + alpha_mix", color="#ff7f0e", alpha=0.85)
        axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.9)
        axes[0].set_xticks(xs); axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("PPL ratio vs int8_fused")
        axes[0].set_title("Engine PPL Ratio by Cell (ctx=512)")
        axes[0].grid(True, axis="y", alpha=0.25); axes[0].legend(fontsize=8.5)

        gains = [r["gain_rot_over_alpha"] for r in csv_rows]
        g_ylo = [r["gain_rot_over_alpha"] - r["gain_ci_lo"] for r in csv_rows]
        g_yhi = [r["gain_ci_hi"] - r["gain_rot_over_alpha"] for r in csv_rows]
        axes[1].bar(xs, gains, width=0.5, color=["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c"], alpha=0.85)
        axes[1].errorbar(xs, gains, yerr=[g_ylo, g_yhi], fmt="none", ecolor="black", capsize=4)
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.9)
        axes[1].set_xticks(xs); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Gain = PPL(rot) / PPL(rot + alpha_mix)")
        axes[1].set_title("Calibrated α Gain (95% paired CI)")
        axes[1].grid(True, axis="y", alpha=0.25)

        fig.suptitle("H2: Cross-Domain Transfer of Calibrated alpha_mix in TenzoEngine")
        fig.tight_layout()
        fig.savefig(H2_DIR / "alpha_domain_transfer.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}", file=sys.stderr)

    ood_ok = all(r["transfers_ci_gt_1"] for r in csv_rows if "out-of-domain" in r["domain_status"])
    all_ok = all(r["transfers_ci_gt_1"] for r in csv_rows)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    gov = (pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
           if pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists() else "unknown")

    md_path = H2_DIR / "alpha_domain_transfer.md"
    existing = md_path.read_text(encoding="utf-8")
    plan_end = existing.find("\n## Results")
    plan_part = existing[:plan_end].rstrip() if plan_end != -1 else existing.rstrip()

    rec_str = (
        "**Recommend making `alpha_mix` available/defaulted for `ternary_rot` (or opt-in with D2 provenance)** — "
        "all out-of-domain cells (`en_web`, `uk_wiki`) show strict improvement (`95% CI > 1.0`)."
        if ood_ok else
        "**Recommend keeping `alpha_mix` opt-in (`--kv-alpha`)** — out-of-domain gain CI does not clear `1.0` across all cells."
    )
    out_md = [
        plan_part,
        "",
        "## Results",
        "",
        "| Cell | Domain vs `calib_mix` | `int8_fused` PPL | `ternary_rot` PPL | `rot / int8` [95% CI] | `rot+alpha` PPL | `alpha / int8` [95% CI] | Gain `rot / alpha` [95% CI] | Transfers? |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |",
        *md_rows,
        "",
        "### Decision & Provenance",
        f"- **Out-of-domain transfer (`en_web` & `uk_wiki` CI > 1):** `{ood_ok}` (all 4 cells > 1: `{all_ok}`).",
        f"- **D2 Recommendation:** {rec_str}",
        *alpha_headers,
        "",
        "### Reproducibility & Metadata",
        "- Command: `python3 scripts/run_h2_alpha_transfer.py` (`tenzo-inference -m /app/models/bitnet-1.58b-v2 -c 512 --prefill batched`, `--kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt`).",
        f"- `commit {commit}`, governor `{gov}`, `n_boot=10000`, paired by window index, `seed=0`.",
    ]
    md_path.write_text("\n".join(out_md) + "\n", encoding="utf-8")
    print(f"Wrote {H2_DIR / 'alpha_domain_transfer.{csv,md,png}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
