#!/usr/bin/env python3
"""Independent verification of PPL and paired ratios directly from score files.
Does NOT import tools/reference/ppl_ci.py.
"""
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def load_windows(rel_path: str) -> dict[int, tuple[int, float]]:
    path = ROOT / rel_path
    windows: dict[int, tuple[int, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "w":
                continue
            wid = int(parts[1])
            n_tok = int(parts[2])
            sum_nll = float(parts[3])
            if wid in windows:
                raise ValueError(f"Duplicate window id {wid} in {rel_path}")
            windows[wid] = (n_tok, sum_nll)
    if not windows:
        raise ValueError(f"No 'w' lines found in {rel_path}")
    return windows


def calc_ppl(windows: dict[int, tuple[int, float]], ids: list[int] | None = None) -> tuple[float, int, int]:
    if ids is None:
        ids = sorted(windows.keys())
    total_n = sum(windows[i][0] for i in ids)
    total_nll = sum(windows[i][1] for i in ids)
    return math.exp(total_nll / total_n), len(ids), total_n


def calc_paired_ratio(cand_path: str, ref_path: str) -> tuple[float, float, float, int]:
    c = load_windows(cand_path)
    r = load_windows(ref_path)
    if set(c.keys()) != set(r.keys()):
        common = sorted(set(c.keys()) & set(r.keys()))
        raise ValueError(
            f"Window mismatch between {cand_path} ({len(c)}) and {ref_path} ({len(r)}), common={len(common)}"
        )
    ids = sorted(c.keys())
    for i in ids:
        if c[i][0] != r[i][0]:
            raise ValueError(f"Token count mismatch at window {i}: {c[i][0]} vs {r[i][0]}")
    ppl_c, n_w, _ = calc_ppl(c, ids)
    ppl_r, _, _ = calc_ppl(r, ids)
    return ppl_c / ppl_r, ppl_c, ppl_r, n_w


def main() -> None:
    checks = [
        ("golden EN fp32", "ppl", "results/reference/golden/nll_golden_float32.txt", None, 19.066),
        ("golden UK8 fp32", "ppl", "results/reference/golden/nll_golden_float32_uk.txt", None, 141.911),
        ("kv_v2 fp32 (EN16)", "ppl", "results/2026-11-19_overnight/ppl_kv_v2/fp32.txt", None, 19.124),
        ("kv_v2 int8_fused (EN16)", "ppl", "results/2026-11-19_overnight/ppl_kv_v2/int8_fused.txt", None, 19.141),
        ("kv_v2 tl1_fused (EN16)", "ppl", "results/2026-11-19_overnight/ppl_kv_v2/tl1_fused.txt", None, 54.94),
        ("kv_v2 popcount_fused (EN16)", "ppl", "results/2026-11-19_overnight/ppl_kv_v2/popcount_fused.txt", None, 207.7),
        ("bitnet_ref EN16", "ppl", "results/2026-11-20_overnight/bitnet_ref/nll_bitnet_rel_en.txt", None, 19.059),
        ("ternary_rot EN16", "ppl", "results/2026-11-20_overnight/part2/ppl_ternary_rot/ternary_rot.txt", None, 20.502),
        (
            "PTQ S13+s4r32 EN16 vs golden",
            "ratio",
            "results/2026-11-20_overnight/part2/kv_final/en16/S13_s4r32.txt",
            "results/reference/golden/nll_golden_float32.txt",
            1.0104,
        ),
        (
            "PTQ S3+s4r32 UK32 vs uk32/fp32",
            "ratio",
            "results/2026-11-20_overnight/part2/kv_final/uk32/S3_s4r32.txt",
            "results/2026-11-20_overnight/part2/kv_final/uk32/fp32.txt",
            1.1512,
        ),
        (
            "PTQ S13+s4r32 UK32 vs uk32/fp32",
            "ratio",
            "results/2026-11-20_overnight/part2/kv_final/uk32/S13_s4r32.txt",
            "results/2026-11-20_overnight/part2/kv_final/uk32/fp32.txt",
            1.0960,
        ),
        (
            "calib_en eval EN16 vs golden",
            "ratio",
            "results/2026-11-20_overnight/part2/calib/eval_en16/calib_en.txt",
            "results/reference/golden/nll_golden_float32.txt",
            1.0399,
        ),
        (
            "calib_uk eval EN16 vs golden",
            "ratio",
            "results/2026-11-20_overnight/part2/calib/eval_en16/calib_uk.txt",
            "results/reference/golden/nll_golden_float32.txt",
            1.0488,
        ),
        (
            "calib_en eval UK32 vs uk32/fp32",
            "ratio",
            "results/2026-11-20_overnight/part2/calib/eval_uk32/calib_en.txt",
            "results/2026-11-20_overnight/part2/kv_final/uk32/fp32.txt",
            1.1261,
        ),
        (
            "calib_uk eval UK32 vs uk32/fp32",
            "ratio",
            "results/2026-11-20_overnight/part2/calib/eval_uk32/calib_uk.txt",
            "results/2026-11-20_overnight/part2/kv_final/uk32/fp32.txt",
            1.1166,
        ),
        (
            "Qwen2.5 tenzo_int8 EN vs fp_en",
            "ratio",
            "results/2026-11-20_overnight/part2/second_model/tenzo_int8_en.txt",
            "results/2026-11-20_overnight/part2/second_model/fp_en.txt",
            1.016,
        ),
        (
            "Qwen2.5 S3+s4r32 EN vs fp_en",
            "ratio",
            "results/2026-11-20_overnight/part2/second_model/S3_s4r32_en.txt",
            "results/2026-11-20_overnight/part2/second_model/fp_en.txt",
            1.956,
        ),
        (
            "Qwen2.5 tenzo_int8 UK vs fp_uk",
            "ratio",
            "results/2026-11-20_overnight/part2/second_model/tenzo_int8_uk.txt",
            "results/2026-11-20_overnight/part2/second_model/fp_uk.txt",
            1.041,
        ),
        (
            "Qwen2.5 S3+s4r32 UK vs fp_uk",
            "ratio",
            "results/2026-11-20_overnight/part2/second_model/S3_s4r32_uk.txt",
            "results/2026-11-20_overnight/part2/second_model/fp_uk.txt",
            2.774,
        ),
    ]

    print(f"{'Item':<34} | {'Kind':<5} | {'Claimed':>9} | {'Recomputed':>10} | {'RelDiff%':>8} | {'Status':<6} | Details")
    print("-" * 115)
    for label, kind, cand, ref, claimed in checks:
        if kind == "ppl":
            w = load_windows(cand)
            val, nw, ntok = calc_ppl(w)
            details = f"windows={nw}, tokens={ntok}"
        else:
            val, ppl_c, ppl_r, nw = calc_paired_ratio(cand, ref)
            details = f"windows={nw}, cand={ppl_c:.4f}, ref={ppl_r:.4f}"
        rel_diff_pct = (val - claimed) / claimed * 100.0
        status = "OK" if abs(rel_diff_pct) <= 0.1 else "DIFF>0.1%"
        print(f"{label:<34} | {kind:<5} | {claimed:>9.4f} | {val:>10.6f} | {rel_diff_pct:>+7.4f}% | {status:<6} | {details}")


if __name__ == "__main__":
    main()
