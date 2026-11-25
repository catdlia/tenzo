#!/usr/bin/env python3
"""Iso-loss KV budget per language (T51 preview, coordinator).

Re-analysis of EXISTING H1 score files (FLORES-200, 32 paired EN/UK windows,
BitNet-b1.58-2B-4T golden fp32, S3 = Hadamard + ternary g32 + 4 sinks + INT8 recent
window). No model is run.

Question: how many INT8 recent slots / bits per KV element does Ukrainian need to
reach the same relative loss as English?

* rel_L = mean_w (Delta_w / B_w)   (extra NLL over base NLL per window, averaged;
  the same convention as rel / R in the H1, H3 and H4 reports)
* x-axis: effective bits/element at a standardized ctx = 512 (`bits_at_512` from
  H1 `char_window_coverage.csv`; kv_ptq.bits constants 8.125 INT8 ring, 2.5 S3),
  plus the mean active INT8 recent slots r_eff and the covered characters.
* Two policy families, each its own curve per language: token window r
  (EN r = 8..32, UK r = 32..96) and character window C = 80..240 (both langs).
* For a target t the needed budget is the first crossing of rel_L(x) below t,
  linear interpolation between adjacent policies. tax = UK - EN.
* Paired window bootstrap (same window ids for EN, UK and all policies),
  10,000 reps, seed 0. A resample whose curve never reaches t is "censored"
  (needs more than the largest tested window) and is excluded from the CI; the
  censored share is reported. A curve already below t at its cheapest policy
  returns that policy's budget (an upper bound) and is counted as "left".
"""
from __future__ import annotations
import csv, hashlib, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_matched_difficulty import read_scores  # noqa: E402

H1 = ROOT / "results/2026-11-24-b_agent2/H1"
OUT = ROOT / "results/2026-11-25_coordinator/iso_loss_bits"
TARGETS = [0.010, 0.0075, 0.005]
BOOT, SEED = 10_000, 0
FAMILIES = {"token_r": "token window r", "char_C": "character window C"}
XKEYS = ["bits_at_512", "r_eff", "cov_chars_exact", "bits_win"]


def policy_file(policy: str, lang: str) -> pathlib.Path:
    return H1 / "scores" / f"{policy.replace('+', '_')}_{lang}.txt"


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def first_crossing(xs: np.ndarray, ys: np.ndarray, t: float):
    """xs ascending. Returns (x*, flag) with flag 0 = interior, 1 = left (already below), 2 = censored."""
    if ys[0] <= t:
        return xs[0], 1
    for i in range(1, len(xs)):
        if ys[i] <= t:
            f = (ys[i - 1] - t) / (ys[i - 1] - ys[i])
            return xs[i - 1] + f * (xs[i] - xs[i - 1]), 0
    return np.nan, 2


def interp_on(xs_src: np.ndarray, xs_dst: np.ndarray, x: float) -> float:
    return float(np.interp(x, xs_src, xs_dst)) if np.isfinite(x) else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cov = [r for r in csv.DictReader(open(H1 / "char_window_coverage.csv")) if r["family"] in FAMILIES]
    base = {L: read_scores(H1 / "scores" / f"fp32_{L}.txt") for L in ("en", "uk")}
    ids = sorted(base["en"])
    assert ids == sorted(base["uk"]), "EN/UK window ids differ"
    Bw = {L: np.array([-base[L][w].sum() for w in ids]) for L in base}
    inputs = {f"fp32_{L}.txt": sha16(H1 / "scores" / f"fp32_{L}.txt") for L in base}

    curves = {}  # (family, lang) -> dict(policies, x arrays, Dw matrix [P, W])
    for fam in FAMILIES:
        for L in ("en", "uk"):
            rows = sorted([r for r in cov if r["family"] == fam and r["lang"] == L], key=lambda r: float(r["bits_at_512"]))
            Dw = []
            for r in rows:
                p = policy_file(r["policy"], L)
                inputs[p.name] = sha16(p)
                s = read_scores(p)
                assert sorted(s) == ids, (p, "window ids differ")
                Dw.append([(base[L][w] - s[w]).sum() for w in ids])
                rel_csv = float(r["rel"])
                rel_me = float(np.mean(np.array(Dw[-1]) / Bw[L]))
                assert abs(rel_me - rel_csv) < 1e-6, (p, rel_me, rel_csv)
            curves[(fam, L)] = dict(policies=[r["policy"] for r in rows],
                                    x={k: np.array([float(r[k]) for r in rows]) for k in XKEYS},
                                    Dw=np.array(Dw))

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), size=(BOOT, len(ids)))

    def rel_curve(fam, L, sel=None):
        Rw = curves[(fam, L)]["Dw"] / Bw[L][None, :]  # [P, W] per-window relative loss
        if sel is None:
            return Rw.mean(1)
        return Rw[:, sel].mean(-1).T  # [BOOT, P]

    curve_rows = []
    for (fam, L), c in curves.items():
        pt = rel_curve(fam, L)
        bt = rel_curve(fam, L, idx)
        lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
        for j, pol in enumerate(c["policies"]):
            curve_rows.append(dict(family=fam, lang=L, policy=pol, bits_at_512=c["x"]["bits_at_512"][j],
                                   bits_win=c["x"]["bits_win"][j], r_eff=c["x"]["r_eff"][j],
                                   cov_chars=c["x"]["cov_chars_exact"][j], rel=pt[j], rel_lo=lo[j], rel_hi=hi[j]))

    iso_rows = []
    for fam in FAMILIES:
        ce, cu = curves[(fam, "en")], curves[(fam, "uk")]
        pe, pu = rel_curve(fam, "en"), rel_curve(fam, "uk")
        be, bu = rel_curve(fam, "en", idx), rel_curve(fam, "uk", idx)
        for t in TARGETS:
            row = dict(family=fam, target_rel=t)
            for L, c, p, b in (("en", ce, pe, be), ("uk", cu, pu, bu)):
                xb = c["x"]["bits_at_512"]
                x0, f0 = first_crossing(xb, p, t)
                row[f"bits_{L}"] = x0
                row[f"flag_{L}"] = ["interior", "left", "censored"][f0]
                row[f"r_eff_{L}"] = interp_on(xb, c["x"]["r_eff"], x0)
                row[f"chars_{L}"] = interp_on(xb, c["x"]["cov_chars_exact"], x0)
                res = np.array([first_crossing(xb, b[k], t) for k in range(BOOT)])
                row[f"_boot_{L}"] = res[:, 0]
                row[f"censored_{L}"] = float(np.mean(res[:, 1] == 2))
                row[f"left_{L}"] = float(np.mean(res[:, 1] == 1))
                ok = np.isfinite(res[:, 0])
                row[f"bits_{L}_lo"], row[f"bits_{L}_hi"] = np.percentile(res[ok, 0], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
            tax_b = row.pop("_boot_uk") - row.pop("_boot_en")
            ok = np.isfinite(tax_b)
            row["tax_bits"] = row["bits_uk"] - row["bits_en"]
            row["tax_bits_lo"], row["tax_bits_hi"] = np.percentile(tax_b[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
            row["tax_share_resamples"] = float(ok.mean())
            row["p_tax_le_0"] = float(np.mean(tax_b[ok] <= 0)) if ok.any() else float("nan")
            row["slots_ratio_uk_en"] = row["r_eff_uk"] / row["r_eff_en"] if row["r_eff_en"] else float("nan")
            row["chars_ratio_uk_en"] = row["chars_uk"] / row["chars_en"] if row["chars_en"] else float("nan")
            iso_rows.append(row)

    def dump(path, rows):
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})

    dump(OUT / "iso_loss_curves.csv", curve_rows)
    dump(OUT / "iso_loss_bits.csv", iso_rows)
    json.dump(dict(script="scripts/analyze_iso_loss_bits.py", source=str(H1.relative_to(ROOT)), n_windows=len(ids),
                   boot=BOOT, seed=SEED, targets=TARGETS, bits_constants="kv_ptq.bits: INT8 ring 8.125, S3 ternary g32 2.5",
                   inputs_sha256_16=inputs), open(OUT / "meta.json", "w"), indent=2)
    for r in iso_rows:
        print(f"{r['family']:8s} t={r['target_rel']*100:.2f}%  EN {r['bits_en']:.3f} [{r['bits_en_lo']:.3f},{r['bits_en_hi']:.3f}] "
              f"r={r['r_eff_en']:.1f} ch={r['chars_en']:.0f} ({r['flag_en']}, cens {r['censored_en']:.2f})  "
              f"UK {r['bits_uk']:.3f} [{r['bits_uk_lo']:.3f},{r['bits_uk_hi']:.3f}] r={r['r_eff_uk']:.1f} ch={r['chars_uk']:.0f} "
              f"({r['flag_uk']}, cens {r['censored_uk']:.2f})  tax {r['tax_bits']:+.3f} [{r['tax_bits_lo']:+.3f},{r['tax_bits_hi']:+.3f}] "
              f"slots x{r['slots_ratio_uk_en']:.2f} chars x{r['chars_ratio_uk_en']:.2f}")


if __name__ == "__main__":
    main()
