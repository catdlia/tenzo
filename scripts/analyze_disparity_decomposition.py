#!/usr/bin/env python3
"""Where does a language's extra KV-quantization loss come from? Exact three-way decomposition (coordinator, exploratory).

Re-analysis of EXISTING score files; no model is run. For a parallel corpus (same content in EN and L),
the extra nats per window caused by the KV scheme are

  D = (sum_w Delta_L - sum_w Delta_EN) / W = (N_L * dbar_L - N_EN * dbar_EN) / W

with N the token totals and dbar the mean per-token damage d = lp_base - lp_scheme. Tokens are binned by base NLL
b = -lp_base (the bins of scripts/analyze_matched_difficulty.py); s(k) is the token share and m(k) the mean damage
of bin k. Then, exactly,

  D * W = (N_L - N_EN) * dbar_EN                                  fertility    : more tokens, charged at EN's rate
        + N_L * sum_k (s_L - s_EN) * (m_L + m_EN) / 2             composition  : L's tokens sit in other difficulty bins
        + N_L * sum_k (m_L - m_EN) * (s_L + s_EN) / 2             fragility    : L's tokens lose more at equal difficulty

(midpoint / Kitagawa split, so the result does not depend on which language is the reference for the last two terms).
Variant "lt4" drops tokens with b >= 4 in both languages first (in that tail, quantization noise lowers NLL).
Paired window bootstrap (windows aligned across languages), 10,000 reps, seed 0.
"""
from __future__ import annotations
import argparse, csv, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_matched_difficulty as M  # noqa: E402

# (corpus, base dir, scheme dir, models {name: base file stem}, langs, schemes)
SETS = [
    ("flores32", "results/2026-11-24_agent2/H3/scores", "results/2026-11-24_agent2/H3/scores",
     {"bitnet": "fp32", "eurollm": "fp", "qwen": "fp"}, ["en", "uk"], ["S3_s4r32"]),
    ("flores5", "results/2026-11-24-b_agent2/H3/scores", "results/2026-11-24-b_agent2/H3/scores",
     {"bitnet": "fp32", "eurollm": "fp"}, ["en", "de", "pl", "uk", "bg"], ["S3_s4r32"]),
    ("flores5", "results/2026-11-24-b_agent2/H3/scores", "results/2026-11-24-b_agent2/H3b/scores",
     {"bitnet": "fp32", "eurollm": "fp"}, ["en", "de", "pl", "uk", "bg"], ["S3_s4c120"]),
    ("ntrex", "results/2026-11-24-b_agent2/H4/scores", "results/2026-11-24-b_agent2/H4/scores",
     {"bitnet": "fp32", "eurollm": "fp"}, ["en", "uk"], ["S3_s4r32", "S3_s4r128"]),
]
COMPONENTS = ["fertility", "composition", "fragility"]


def decompose(S_L, C_L, S_E, C_E, upto=None):
    """Inputs: summed bin damage S and counts C, shape [R, K]. Returns dict of [R] arrays (totals, not per window)."""
    sl = slice(0, upto)
    S_L, C_L, S_E, C_E = S_L[:, sl], C_L[:, sl], S_E[:, sl], C_E[:, sl]
    N_L, N_E = C_L.sum(1), C_E.sum(1)
    s_L, s_E = C_L / N_L[:, None], C_E / N_E[:, None]
    m_L = np.where(C_L > 0, S_L / np.maximum(C_L, 1), 0.0)
    m_E = np.where(C_E > 0, S_E / np.maximum(C_E, 1), 0.0)
    dbar_E = S_E.sum(1) / N_E
    out = dict(
        fertility=(N_L - N_E) * dbar_E,
        composition=N_L * ((s_L - s_E) * (m_L + m_E) / 2).sum(1),
        fragility=N_L * ((m_L - m_E) * (s_L + s_E) / 2).sum(1),
        D=S_L.sum(1) - S_E.sum(1),
        phi=N_L / N_E,
    )
    resid = out["D"] - (out["fertility"] + out["composition"] + out["fragility"])
    assert np.allclose(resid, 0, atol=1e-6 * (1 + np.abs(out["D"]).max())), float(np.abs(resid).max())
    return out


def ci(x):
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="results/2026-11-25_coordinator/decomposition")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, idx_cache = [], {}
    for corpus, bdir, sdir, models, langs, schemes in SETS:
        for model, base_name in models.items():
            for scheme in schemes:
                arr = {}
                for L in langs:
                    pb, ps = ROOT / bdir / model / f"{base_name}_{L}.txt", ROOT / sdir / model / f"{scheme}_{L}.txt"
                    if not (pb.exists() and ps.exists()):
                        continue
                    base, sch = M.read_scores(pb), M.read_scores(ps)
                    ids = sorted(base)
                    assert ids == sorted(sch), (pb, ps)
                    S, C, *_ = M.per_window(base, sch, ids)
                    arr[L] = (S, C, ids)
                if "en" not in arr:
                    continue
                W = len(arr["en"][2])
                if (corpus, W) not in idx_cache:
                    idx_cache[(corpus, W)] = rng.integers(0, W, size=(a.boot, W))
                idx = idx_cache[(corpus, W)]
                S_E, C_E, ids_E = arr["en"]
                for L, (S_L, C_L, ids_L) in arr.items():
                    if L == "en":
                        continue
                    assert ids_L == ids_E
                    for variant, upto in (("all", None), ("lt4", M.LT4)):
                        p = decompose(S_L.sum(0)[None], C_L.sum(0)[None], S_E.sum(0)[None], C_E.sum(0)[None], upto)
                        b = decompose(S_L[idx].sum(1), C_L[idx].sum(1), S_E[idx].sum(1), C_E[idx].sum(1), upto)
                        row = dict(corpus=corpus, model=model, scheme=scheme, lang=L, variant=variant, n_windows=W,
                                   phi=float(p["phi"][0]), D_per_window=float(p["D"][0]) / W)
                        row["D_lo"], row["D_hi"] = (v / W for v in ci(b["D"]))
                        for c in COMPONENTS:
                            row[c] = float(p[c][0]) / W
                            row[f"{c}_lo"], row[f"{c}_hi"] = (v / W for v in ci(b[c]))
                            row[f"{c}_share"] = float(p[c][0] / p["D"][0]) if p["D"][0] != 0 else np.nan
                        rows.append(row)
    with open(out / "decomposition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    json.dump(dict(script="scripts/analyze_disparity_decomposition.py", boot=a.boot, seed=0,
                   sets=[s[:2] + (s[2],) for s in SETS]), open(out / "meta.json", "w"), indent=1)
    for r in rows:
        print(f"{r['corpus']:8s} {r['model']:7s} {r['scheme']:10s} {r['lang']} {r['variant']:3s} phi {r['phi']:.2f} "
              f"D {r['D_per_window']:+6.2f} [{r['D_lo']:+.2f},{r['D_hi']:+.2f}] | "
              + " | ".join(f"{c[:4]} {r[c]:+6.2f} [{r[c+'_lo']:+.2f},{r[c+'_hi']:+.2f}] ({100*r[c+'_share']:+4.0f}%)" for c in COMPONENTS))


if __name__ == "__main__":
    main()
