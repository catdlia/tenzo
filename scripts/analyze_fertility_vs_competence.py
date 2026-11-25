#!/usr/bin/env python3
"""Fertility vs competence (T50 part 2): regression of the KV-quantization disparity on log phi and log kappa.

Written by the coordinator (2026-11-25) BEFORE the 11-language data exist, so the analysis is fixed in advance.
The agent only adds the `flores11` set to SETS in scripts/analyze_matched_difficulty.py and runs this script.

Per model, over the non-EN languages L of a parallel set (windows aligned across languages):
  phi_L   = tokens_L / tokens_EN                       (fertility, same content)
  kappa_L = B_L / B_EN                                 (cost of content in nats; competence proxy)
  R_L     = sum D_L / sum B_L - sum D_EN / sum B_EN    (relative-loss disparity, pooled, in pp)
  ratio_L = matched-difficulty damage ratio L / EN, b < 4 (scripts/analyze_matched_difficulty.py), optional response
  frag_L  = fragility term of the exact decomposition (scripts/analyze_disparity_decomposition.py), in nats per
            EN token of the same content, optional response
OLS  y_L = a + b_phi * log phi_L + b_kappa * log kappa_L, standardized coefficients (z-scored across languages).
Paired window bootstrap (10,000, seed 0): every resample recomputes phi, kappa, y for all languages and refits.
Decision rule (pre-registered in the agent prompt, T50):
  "tokenization" if CI(b_phi) > 0 and CI(b_kappa) contains 0; "competence" the reverse; "both" if both CIs > 0;
  "collinear" if VIF > 5 or both CIs contain 0; otherwise "unclear".
With fewer than 6 non-EN languages the fit is reported but flagged "too_few_languages" (no verdict).
"""
from __future__ import annotations
import argparse, csv, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_matched_difficulty as M  # noqa: E402
from analyze_disparity_decomposition import decompose  # noqa: E402


def load(cfg, model, base_name, scheme):
    d0 = ROOT / cfg["dir"]
    arrs, ids_ref = {}, None
    for L in cfg["langs"]:
        pb, ps = d0 / model / f"{base_name}_{L}.txt", d0 / model / f"{scheme}_{L}.txt"
        if not (pb.exists() and ps.exists()):
            continue
        base, sch = M.read_scores(pb), M.read_scores(ps)
        ids = sorted(base)
        assert ids == sorted(sch), (pb, ps)
        ids_ref = ids_ref or ids
        assert ids == ids_ref, ("window ids differ", L)
        S, C, B, D, N = M.per_window(base, sch, ids)
        arrs[L] = dict(S=S, C=C, B=B, D=D, N=N)
    return arrs, ids_ref


def descriptors(arrs, idx, response):
    """Returns phi, kappa, y as arrays [R, L] (R = 1 for the point estimate) for the non-EN languages."""
    P = {L: M.stats(v, idx) for L, v in arrs.items()}
    langs = [L for L in arrs if L != "en"]
    E = P["en"]
    phi = np.stack([P[L]["N"] / E["N"] for L in langs], 1)
    kap = np.stack([P[L]["B"] / E["B"] for L in langs], 1)
    if response == "R":
        y = np.stack([100 * (P[L]["D"] / P[L]["B"] - E["D"] / E["B"]) for L in langs], 1)
    elif response == "fragility":
        y = np.stack([decompose(P[L]["S"], P[L]["C"], E["S"], E["C"])["fragility"] / E["N"] for L in langs], 1)
    else:  # ratio_lt4
        me = M.matched(E["S"], E["C"], E["C"], M.LT4)
        y = np.stack([M.matched(P[L]["S"], P[L]["C"], E["C"], M.LT4) / me for L in langs], 1)
    return langs, np.atleast_2d(phi), np.atleast_2d(kap), np.atleast_2d(y)


def fit(phi, kap, y):
    """Standardized OLS per row. Returns [R, 2] coefficients (log phi, log kappa)."""
    out = np.full((phi.shape[0], 2), np.nan)
    for r in range(phi.shape[0]):
        X = np.stack([np.log(phi[r]), np.log(kap[r])], 1)
        yy = y[r]
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(yy)):
            continue
        sx, sy = X.std(0), yy.std()
        if np.any(sx == 0) or sy == 0:
            continue
        Z = (X - X.mean(0)) / sx
        A = np.column_stack([np.ones(len(yy)), Z])
        beta, *_ = np.linalg.lstsq(A, (yy - yy.mean()) / sy, rcond=None)
        out[r] = beta[1:]
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ci(x):
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def verdict(c_phi, c_kap, vif, n_langs):
    if n_langs < 6:
        return "too_few_languages"
    pos = lambda c: c[0] > 0
    has0 = lambda c: c[0] <= 0 <= c[1]
    if vif > 5 or (has0(c_phi) and has0(c_kap)):
        return "collinear"
    if pos(c_phi) and has0(c_kap):
        return "tokenization"
    if pos(c_kap) and has0(c_phi):
        return "competence"
    if pos(c_phi) and pos(c_kap):
        return "both"
    return "unclear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="flores5", help="key of SETS in analyze_matched_difficulty.py (flores11 for T50)")
    ap.add_argument("--scheme", default="S3_s4r32")
    ap.add_argument("--response", default="R", choices=["R", "ratio_lt4", "fragility"])
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="results/2026-11-25_coordinator/fertility_vs_competence")
    a = ap.parse_args()
    cfg = M.SETS[a.set]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, lang_rows, idx = [], [], None
    for model, base_name in cfg["models"].items():
        arrs, ids = load(cfg, model, base_name, a.scheme)
        if "en" not in arrs or len(arrs) < 3:
            continue
        if idx is None:
            idx = rng.integers(0, len(ids), size=(a.boot, len(ids)))
        langs, phi, kap, y = descriptors(arrs, None, a.response)
        _, phib, kapb, yb = descriptors(arrs, idx, a.response)
        b0 = fit(phi, kap, y)[0]
        bb = fit(phib, kapb, yb)
        lp, lk = np.log(phi[0]), np.log(kap[0])
        r = float(np.corrcoef(lp, lk)[0, 1])
        vif = 1.0 / (1.0 - r * r) if abs(r) < 1 else np.inf
        c_phi, c_kap = ci(bb[:, 0]), ci(bb[:, 1])
        rows.append(dict(set=a.set, model=model, scheme=a.scheme, response=a.response, n_langs=len(langs),
                         n_windows=len(ids), beta_logphi=b0[0], beta_logphi_lo=c_phi[0], beta_logphi_hi=c_phi[1],
                         beta_logkappa=b0[1], beta_logkappa_lo=c_kap[0], beta_logkappa_hi=c_kap[1],
                         corr_logphi_logkappa=r, vif=vif, spearman_y_phi=spearman(y[0], phi[0]),
                         spearman_y_kappa=spearman(y[0], kap[0]), fit_share=float(np.isfinite(bb[:, 0]).mean()),
                         verdict=verdict(c_phi, c_kap, vif, len(langs))))
        for j, L in enumerate(langs):
            lang_rows.append(dict(set=a.set, model=model, lang=L, phi=phi[0, j], kappa=kap[0, j], y=y[0, j],
                                  y_lo=ci(yb[:, j])[0], y_hi=ci(yb[:, j])[1]))
    tag = f"{a.set}_{a.scheme}_{a.response}"
    for name, rr in ((f"regression_{tag}.csv", rows), (f"languages_{tag}.csv", lang_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
            w.writeheader()
            for r in rr:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    json.dump(dict(script="scripts/analyze_fertility_vs_competence.py", args=vars(a), seed=0), open(out / f"meta_{tag}.json", "w"), indent=1)
    for r in rows:
        print(f"{r['model']:8s} n={r['n_langs']} b_logphi {r['beta_logphi']:+.2f} [{r['beta_logphi_lo']:+.2f},{r['beta_logphi_hi']:+.2f}] "
              f"b_logkappa {r['beta_logkappa']:+.2f} [{r['beta_logkappa_lo']:+.2f},{r['beta_logkappa_hi']:+.2f}] corr {r['corr_logphi_logkappa']:+.2f} "
              f"VIF {r['vif']:.1f} rho(y,phi) {r['spearman_y_phi']:+.2f} rho(y,kappa) {r['spearman_y_kappa']:+.2f} -> {r['verdict']}")
    for r in lang_rows:
        print(f"   {r['model']:8s} {r['lang']}: phi {r['phi']:.2f} kappa {r['kappa']:.2f} y {r['y']:+.3f} [{r['y_lo']:+.3f},{r['y_hi']:+.3f}]")


if __name__ == "__main__":
    main()
