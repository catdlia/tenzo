#!/usr/bin/env python3
"""NTREX two-point slot ratio: how large an EN recent window matches UK at r = 128? (coordinator, exploratory)

Re-analysis of EXISTING H4 score files (NTREX-128, 24 coherent articles, ctx <= 1024,
BitNet golden and EuroLLM-1.7B, S3 with r = 32 and r = 128). No model is run.

rel_L(r) = sum_w Delta_w / sum_w B_w (pooled), the H4 convention (H1 uses the per-window mean of ratios). The UK reference loss is t = rel_UK(128).
The EN window with the same loss is found on the chord between r = 32 and r = 128:
  r_EN* = 32 + (rel_EN(32) - t) / (rel_EN(32) - rel_EN(128)) * 96.
Loss falls convexly with r, so the chord lies above the true curve and over-estimates r_EN*;
the slot ratio 128 / r_EN* is therefore conservative (a lower bound under convexity).
If t > rel_EN(32), r_EN* <= 32 ("left", ratio >= 4); if t < rel_EN(128), r_EN* > 128 (censored).
Paired window bootstrap, joint across models (same 24 articles), 10,000 reps, seed 0.
"""
from __future__ import annotations
import csv, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_matched_difficulty import read_scores  # noqa: E402

SRC = ROOT / "results/2026-11-24-b_agent2/H4/scores"
OUT = ROOT / "results/2026-11-25_coordinator/iso_loss_bits"
MODELS = {"bitnet": "fp32", "eurollm": "fp"}
BOOT, SEED = 10_000, 0


def per_window_dB(model, base_name, scheme, lang, ids):
    b = read_scores(SRC / model / f"{base_name}_{lang}.txt")
    s = read_scores(SRC / model / f"{scheme}_{lang}.txt")
    assert sorted(b) == ids and sorted(s) == ids
    return (np.array([(b[w] - s[w]).sum() for w in ids]), np.array([-b[w].sum() for w in ids]))


def r_star(e32, e128, t):
    """Vectorized chord inversion; returns (r, flag) with flag 0 interior, 1 left (<= 32), 2 censored (> 128)."""
    r = 32 + (e32 - t) / (e32 - e128) * 96
    flag = np.where(t > e32, 1, np.where(t < e128, 2, 0))
    r = np.where(flag == 1, 32.0, np.where(flag == 2, np.nan, r))
    return r, flag


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(read_scores(SRC / "bitnet" / "fp32_en.txt"))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), size=(BOOT, len(ids)))
    rows, boots = [], {}
    for m, base in MODELS.items():
        R = {(L, r): per_window_dB(m, base, f"S3_s4r{r}", L, ids) for L in ("en", "uk") for r in (32, 128)}
        pt = {k: D.sum() / B.sum() for k, (D, B) in R.items()}
        bt = {k: D[idx].sum(1) / B[idx].sum(1) for k, (D, B) in R.items()}
        t = pt[("uk", 128)]
        r0, f0 = r_star(np.array(pt[("en", 32)]), np.array(pt[("en", 128)]), np.array(t))
        rb, fb = r_star(bt[("en", 32)], bt[("en", 128)], bt[("uk", 128)])
        ratio_b = 128.0 / rb
        boots[m] = ratio_b
        ok = np.isfinite(ratio_b)
        lo, hi = np.percentile(ratio_b[ok], [2.5, 97.5])
        rows.append(dict(model=m, rel_en_r32=pt[("en", 32)], rel_en_r128=pt[("en", 128)],
                         rel_uk_r32=pt[("uk", 32)], rel_uk_r128=t, r_en_equiv=float(r0),
                         flag=["interior", "left", "censored"][int(f0)], slot_ratio=128.0 / float(r0),
                         slot_ratio_lo=lo, slot_ratio_hi=hi, left_share=float(np.mean(fb == 1)),
                         censored_share=float(np.mean(fb == 2))))
    d = boots["bitnet"] - boots["eurollm"]
    ok = np.isfinite(d)
    diff = dict(model="bitnet-eurollm", slot_ratio=rows[0]["slot_ratio"] - rows[1]["slot_ratio"],
                slot_ratio_lo=np.percentile(d[ok], 2.5), slot_ratio_hi=np.percentile(d[ok], 97.5),
                p_le_0=float(np.mean(d[ok] <= 0)))
    keys = list(rows[0].keys())
    with open(OUT / "ntrex_slot_ratio.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
        w.writerow({k: (f"{diff[k]:.6g}" if isinstance(diff.get(k), (float, np.floating)) else diff.get(k, "")) for k in keys})
    json.dump(dict(script="scripts/analyze_ntrex_slot_ratio.py", source=str(SRC.relative_to(ROOT)), n_windows=len(ids),
                   boot=BOOT, seed=SEED, p_bitnet_minus_eurollm_le_0=diff["p_le_0"]),
              open(OUT / "ntrex_slot_ratio_meta.json", "w"), indent=2)
    for r in rows:
        print(f"{r['model']:8s} EN r32 {100*r['rel_en_r32']:.2f}% r128 {100*r['rel_en_r128']:.2f}% | UK r32 {100*r['rel_uk_r32']:.2f}% "
              f"r128 {100*r['rel_uk_r128']:.2f}% | EN-equivalent r {r['r_en_equiv']:.1f} ({r['flag']}) "
              f"slot ratio {r['slot_ratio']:.2f} [{r['slot_ratio_lo']:.2f}, {r['slot_ratio_hi']:.2f}] "
              f"left {r['left_share']:.2f} cens {r['censored_share']:.2f}")
    print(f"difference bitnet - eurollm: {diff['slot_ratio']:+.2f} [{diff['slot_ratio_lo']:+.2f}, {diff['slot_ratio_hi']:+.2f}], P(<=0) {diff['p_le_0']:.4f}")


if __name__ == "__main__":
    main()
