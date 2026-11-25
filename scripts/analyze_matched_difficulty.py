#!/usr/bin/env python3
"""Matched-difficulty analysis of KV-quantization damage (T50, part 1).

Coordinator re-analysis of EXISTING score files; no model is run.

Per scored token:
  b = -lp_base               base NLL (difficulty before KV quantization)
  d = lp_base - lp_scheme    extra NLL caused by the KV scheme (>0 = worse)

Tokens are binned by b. For each (corpus, model, scheme, language) we report the
mean d per bin and a "matched" mean: the language's per-bin means reweighted to
the EN bin distribution of the same corpus/model. matched_L - matched_EN is the
extra damage at equal difficulty; matched_L / matched_EN the ratio (reported only
where the EN matched mean is clearly > 0). A b < 4 variant excludes the
high-surprise tail, where ternary noise *lowers* NLL (distribution flattening).

Windows are aligned across languages (parallel corpora) and across models within
a corpus, so the bootstrap resamples window ids jointly (paired), 10,000 reps, seed 0.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EDGES = np.array([0, 0.05, 0.3, 1, 2, 4, 7, np.inf])
LABELS = ["<0.05", "0.05-0.3", "0.3-1", "1-2", "2-4", "4-7", ">=7"]
LT4 = 5  # bins 0..4 are b < 4
SETS = {
    "flores32": dict(dir="results/2026-11-24_agent2/H3/scores",
                     models={"bitnet": "fp32", "eurollm": "fp", "qwen": "fp"},
                     langs=["en", "uk"], schemes=["S3_s4r32", "S13_s4r32", "tenzo_int8"]),
    "flores5": dict(dir="results/2026-11-24-b_agent2/H3/scores",
                    models={"bitnet": "fp32", "eurollm": "fp"},
                    langs=["en", "de", "pl", "uk", "bg"], schemes=["S3_s4r32", "tenzo_int8"]),
    "ntrex": dict(dir="results/2026-11-24-b_agent2/H4/scores",
                  models={"bitnet": "fp32", "eurollm": "fp"},
                  langs=["en", "uk"], schemes=["S3_s4r32", "S3_s4r128", "tenzo_int8"]),
}


def read_scores(p: pathlib.Path) -> dict[int, np.ndarray]:
    out, cur, n = {}, None, None
    with open(p) as f:
        for line in f:
            if line.startswith("w "):
                _, i, n_, _s = line.split()
                cur, n = int(i), int(n_)
            elif line.startswith("lp ") and cur is not None:
                lp = np.array(line.split()[1:], dtype=np.float64)
                assert len(lp) == n, (p, cur, len(lp), n)
                out[cur] = lp
    return out


def per_window(base: dict, sch: dict, ids: list[int]):
    K = len(LABELS)
    S = np.zeros((len(ids), K)); C = np.zeros((len(ids), K))
    B = np.zeros(len(ids)); D = np.zeros(len(ids)); N = np.zeros(len(ids))
    for j, w in enumerate(ids):
        lb, ls = base[w], sch[w]
        b, d = -lb, lb - ls
        k = np.clip(np.searchsorted(EDGES, b, side="right") - 1, 0, K - 1)
        S[j] = np.bincount(k, weights=d, minlength=K)
        C[j] = np.bincount(k, minlength=K)
        B[j], D[j], N[j] = b.sum(), d.sum(), len(b)
    return S, C, B, D, N


def stats(arr, idx):
    """arr: dict of per-window arrays; idx: (R, Nw) resample or None (point estimate)."""
    if idx is None:
        return {k: v.sum(axis=0)[None] for k, v in arr.items()}
    return {k: v[idx].sum(axis=1) for k, v in arr.items()}


def matched(Sl, Cl, Ce, upto=None):
    sl = slice(0, upto)
    m = np.where(Cl[:, sl] > 0, Sl[:, sl] / np.maximum(Cl[:, sl], 1), np.nan)
    w = Ce[:, sl] / Ce[:, sl].sum(axis=1, keepdims=True)
    w = np.where(np.isnan(m), 0.0, w)
    w = w / w.sum(axis=1, keepdims=True)
    return np.nansum(w * np.nan_to_num(m), axis=1)


def ci(x):
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/2026-11-25_coordinator/matched_difficulty")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary, bins_rows, cross_rows, inputs = [], [], [], {}
    point_cache, boot_cache = {}, {}
    for sname, cfg in SETS.items():
        d0 = ROOT / cfg["dir"]
        ids_ref = None
        for model, base_name in cfg["models"].items():
            for scheme in cfg["schemes"]:
                arrs = {}
                for L in cfg["langs"]:
                    pb, ps = d0 / model / f"{base_name}_{L}.txt", d0 / model / f"{scheme}_{L}.txt"
                    if not (pb.exists() and ps.exists()):
                        continue
                    for p in (pb, ps):
                        inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                    base, sch = read_scores(pb), read_scores(ps)
                    ids = sorted(base)
                    assert ids == sorted(sch)
                    if ids_ref is None:
                        ids_ref = ids
                    assert ids == ids_ref, (sname, model, L)
                    S, C, B, D, N = per_window(base, sch, ids)
                    arrs[L] = dict(S=S, C=C, B=B, D=D, N=N)
                if "en" not in arrs:
                    continue
                Nw = len(ids_ref)
                key = (sname, Nw)
                if key not in boot_cache:
                    boot_cache[key] = rng.integers(0, Nw, size=(a.boot, Nw))
                idx = boot_cache[key]
                P = {L: stats(v, None) for L, v in arrs.items()}
                Bt = {L: stats(v, idx) for L, v in arrs.items()}
                for L in arrs:
                    s, c = P[L]["S"][0], P[L]["C"][0]
                    for k in range(len(LABELS)):
                        bins_rows.append(dict(corpus=sname, model=model, scheme=scheme, lang=L, bin=LABELS[k],
                                              n_tok=int(c[k]), share=c[k] / c.sum(),
                                              mean_d=(s[k] / c[k]) if c[k] else np.nan))
                for tag, upto in (("all", None), ("lt4", LT4)):
                    me_p = matched(P["en"]["S"], P["en"]["C"], P["en"]["C"], upto)[0]
                    me_b = matched(Bt["en"]["S"], Bt["en"]["C"], Bt["en"]["C"], upto)
                    en_pos = ci(me_b)[0] > 0
                    for L in arrs:
                        ml_p = matched(P[L]["S"], P[L]["C"], P["en"]["C"], upto)[0]
                        ml_b = matched(Bt[L]["S"], Bt[L]["C"], Bt["en"]["C"], upto)
                        diff_b, ratio_b = ml_b - me_b, ml_b / me_b
                        rel = lambda X: X["D"].ravel() / X["B"].ravel()
                        R_p = rel(P[L])[0] - rel(P["en"])[0]
                        R_b = Bt[L]["D"] / Bt[L]["B"] - Bt["en"]["D"] / Bt["en"]["B"]
                        row = dict(corpus=sname, model=model, scheme=scheme, lang=L, variant=tag, n_windows=Nw,
                                   tok=float(P[L]["N"][0]), phi=float(P[L]["N"][0] / P["en"]["N"][0]),
                                   kappa=float(P[L]["B"][0] / P["en"]["B"][0]),
                                   raw_mean_d=float(P[L]["S"][0].sum() / P[L]["C"][0].sum()),
                                   matched_d=float(ml_p), matched_d_lo=ci(ml_b)[0], matched_d_hi=ci(ml_b)[1],
                                   diff_vs_en=float(ml_p - me_p), diff_lo=ci(diff_b)[0], diff_hi=ci(diff_b)[1],
                                   ratio_vs_en=float(ml_p / me_p) if en_pos else np.nan,
                                   ratio_lo=ci(ratio_b)[0] if en_pos else np.nan,
                                   ratio_hi=ci(ratio_b)[1] if en_pos else np.nan,
                                   en_matched_ci_gt0=bool(en_pos),
                                   R_rel_pp=100 * float(R_p), R_lo=100 * ci(R_b)[0], R_hi=100 * ci(R_b)[1])
                        summary.append(row)
                        point_cache[(sname, model, scheme, L, tag)] = ratio_b if en_pos else None
    # cross-model: ratio(bitnet) - ratio(other), joint resampling (same window ids within a corpus)
    for (sname, model, scheme, L, tag), rb in list(point_cache.items()):
        if model != "bitnet" or L == "en" or rb is None:
            continue
        for other in ("eurollm", "qwen"):
            ro = point_cache.get((sname, other, scheme, L, tag))
            if ro is None:
                continue
            pb = next(r for r in summary if (r["corpus"], r["model"], r["scheme"], r["lang"], r["variant"]) == (sname, "bitnet", scheme, L, tag))
            po = next(r for r in summary if (r["corpus"], r["model"], r["scheme"], r["lang"], r["variant"]) == (sname, other, scheme, L, tag))
            dlt = rb - ro
            cross_rows.append(dict(corpus=sname, scheme=scheme, lang=L, variant=tag, other=other,
                                   ratio_bitnet=pb["ratio_vs_en"], ratio_other=po["ratio_vs_en"],
                                   diff=pb["ratio_vs_en"] - po["ratio_vs_en"], diff_lo=ci(dlt)[0], diff_hi=ci(dlt)[1],
                                   p_le0=float(np.mean(dlt[np.isfinite(dlt)] <= 0))))
    for name, rows in (("matched_difficulty.csv", summary), ("matched_bins.csv", bins_rows), ("matched_cross_model.csv", cross_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out / "meta.json").write_text(json.dumps(dict(script="scripts/analyze_matched_difficulty.py", boot=a.boot, seed=0,
                                                   edges=[float(e) for e in EDGES[:-1]] + ["inf"], inputs_sha256_16=inputs), indent=1))
    print("rows", len(summary), len(bins_rows), len(cross_rows))


if __name__ == "__main__":
    main()
