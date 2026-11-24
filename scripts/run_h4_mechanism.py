#!/usr/bin/env python3
"""H4 (T46): Mechanism ablations and single-pass diagnostics on FLORES-12 (ctx 512).

Evaluates BitNet-b1.58-2B (layer-major unpadded GB.Golden, MemoryMax=4G) on the first
12 paired FLORES windows (results/2026-11-24_agent2/H3/flores_bitnet_{en,uk}.ids[:12]):
  1. Reuse fp32 and S3+s4r32 scores (w=0..11) from H3.
  2. Ablations:
     - K-only (KV8:S3+s4r32) vs V-only (K8V:S3+s4r32)
     - Recent window r in {0, 128, 256} with sinks=4 (S3+s4r0, S3+s4r128, S3+s4r256)
     - Layer subsets (S3+s4r32 in layers 0..9, 10..19, or 20..29 only; fp32 elsewhere)
  3. Single-pass fp32 diagnostic probes across all 30 layers (EN vs UK):
     - Attention mass outside sinks=4 & recent=32 (for query positions t >= 64)
     - Relative L2 quantization error of K_rope and V under S3 (Hadamard + TWN g=32)
     - Attention entropy H(p_fp32) vs H(p_S3_K) & leaked far mass across 128-token blocks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "tools" / "reference"
sys.path.insert(0, str(REF_DIR))

H3_BITNET = ROOT / "results" / "2026-11-24_agent2" / "H3"
H4_DIR = ROOT / "results" / "2026-11-24_agent2" / "H4"
SCORES_DIR = H4_DIR / "scores"


def read_ids(path: pathlib.Path, limit: int = 12) -> list[list[int]]:
    wins = [list(map(int, ln.split())) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return wins[:limit]


def load_nlls_from_txt(path: pathlib.Path, limit: int = 12) -> np.ndarray:
    vals: dict[int, float] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("w "):
            _, wid, _, nll = ln.split()
            vals[int(wid)] = float(nll)
    return np.array([vals[i] for i in range(limit)], dtype=np.float64)


def pair_boot(arr: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(arr)
    idx = rng.randint(0, n, size=(n_boot, n))
    m = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pair_ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    # Total NLL ratio / PPL ratio approximation using ratio of mean NLL vs baseline NLL?
    # Note exact PPL ratio = exp((S.sum() - B.sum()) / total_tokens), here passed separately when needed.
    rng = np.random.RandomState(seed)
    n = len(num)
    pt = float(num.sum() / den.sum())
    idx = rng.randint(0, n, size=(n_boot, n))
    r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return pt, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def ppl_ratio_ci(s_nll: np.ndarray, b_nll: np.ndarray, n_tok: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(s_nll)
    diff = s_nll - b_nll
    pt = float(math.exp(diff.sum() / n_tok.sum()))
    idx = rng.randint(0, n, size=(n_boot, n))
    r = np.exp(diff[idx].sum(axis=1) / n_tok[idx].sum(axis=1))
    return pt, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def run_layer_major(g, seqs: list[list[int]], collect_probes: bool = False) -> tuple[list[list[float]], dict | None]:
    import golden_bitnet as GB
    import kv_ptq

    tmax = max(len(s) for s in seqs)
    cos_all, sin_all = g.rope(tmax)
    inps = [torch.tensor([s], dtype=torch.long) for s in seqs]

    # Diagnostic storage across 30 layers (if collect_probes)
    probe_blocks = [(0, 128, "0..127"), (128, 256, "128..255"), (256, 384, "256..383")]
    layer_stats: list[dict] = []

    with torch.no_grad():
        xs = [g.embed_ids(inp) for inp in inps]
        for l in range(g.L):
            pre = f"model.layers.{l}."
            w_in = g.vec(pre + "input_layernorm.weight")
            wq = g.load_lin(pre + "self_attn.q_proj")
            wk = g.load_lin(pre + "self_attn.k_proj")
            wv = g.load_lin(pre + "self_attn.v_proj")
            w_asn = g.vec(pre + "self_attn.attn_sub_norm.weight")
            wo = g.load_lin(pre + "self_attn.o_proj")
            w_pan = g.vec(pre + "post_attention_layernorm.weight")
            wg = g.load_lin(pre + "mlp.gate_proj")
            wu = g.load_lin(pre + "mlp.up_proj")
            w_fsn = g.vec(pre + "mlp.ffn_sub_norm.weight")
            wd = g.load_lin(pre + "mlp.down_proj")

            k_err_sq, k_norm_sq = 0.0, 0.0
            v_err_sq, v_norm_sq = 0.0, 0.0
            far_mass_fp, far_mass_s3k = [], []
            ent_fp_blk = [[] for _ in probe_blocks]
            ent_s3_blk = [[] for _ in probe_blocks]
            leak_fp_blk = [[] for _ in probe_blocks]
            leak_s3_blk = [[] for _ in probe_blocks]

            for idx_s, x in enumerate(xs):
                B, T, _ = x.shape
                h = g.rmsnorm(x, w_in)
                q = g.bitlinear(h, wq).view(B, T, g.H, g.D).transpose(1, 2)
                k = g.bitlinear(h, wk).view(B, T, g.Hkv, g.D).transpose(1, 2)
                v = g.bitlinear(h, wv).view(B, T, g.Hkv, g.D).transpose(1, 2)
                c, s = cos_all[:T][None, None], sin_all[:T][None, None]
                q = q * c + g.rotate_half(q) * s
                k = k * c + g.rotate_half(k) * s

                if collect_probes:
                    kh_s3 = kv_ptq.tern_g32_rot(k)
                    vh_s3 = kv_ptq.tern_g32_rot(v)
                    k_err_sq += float((k - kh_s3).pow(2).sum())
                    k_norm_sq += float(k.pow(2).sum())
                    v_err_sq += float((v - vh_s3).pow(2).sum())
                    v_norm_sq += float(v.pow(2).sum())

                    # Softmax fp32 vs S3-K+s4r32 (using fp32 q)
                    r_gqa = g.H // g.Hkv
                    causal = torch.ones(T, T, dtype=torch.bool).tril_()
                    wm = kv_ptq.window_mask(T, 4, 32)
                    k_i8 = kv_ptq.int8_absmax(k)

                    kb_fp = k[0].repeat_interleave(r_gqa, dim=0)
                    sc_fp = torch.matmul(q[0], kb_fp.transpose(-1, -2)) * (g.D ** -0.5)
                    sc_fp = sc_fp.masked_fill(~causal, torch.finfo(sc_fp.dtype).min)
                    p_fp = torch.softmax(sc_fp, dim=-1, dtype=torch.float32)

                    kb_s3 = kh_s3[0].repeat_interleave(r_gqa, dim=0)
                    kb_i8 = k_i8[0].repeat_interleave(r_gqa, dim=0)
                    sc_s3 = torch.matmul(q[0], kb_s3.transpose(-1, -2)) * (g.D ** -0.5)
                    sc_i8 = torch.matmul(q[0], kb_i8.transpose(-1, -2)) * (g.D ** -0.5)
                    sc_mix = torch.where(wm, sc_i8, sc_s3).masked_fill(~causal, torch.finfo(sc_fp.dtype).min)
                    p_s3 = torch.softmax(sc_mix, dim=-1, dtype=torch.float32)

                    # Mid/far attention mass outside sinks=4 and recent=32 for queries t >= 64
                    t_idx = torch.arange(T)[:, None]
                    s_idx = torch.arange(T)[None, :]
                    far_m = (s_idx >= 4) & ((t_idx - s_idx) >= 32) & causal
                    if T > 64:
                        mf_q = (p_fp[:, 64:, :] * far_m[None, 64:, :]).sum(dim=-1).mean()
                        ms_q = (p_s3[:, 64:, :] * far_m[None, 64:, :]).sum(dim=-1).mean()
                        far_mass_fp.append(float(mf_q))
                        far_mass_s3k.append(float(ms_q))

                    h_fp = -(p_fp * torch.log(p_fp.clamp(min=1e-12))).sum(dim=-1).mean(dim=0)  # [T]
                    h_s3 = -(p_s3 * torch.log(p_s3.clamp(min=1e-12))).sum(dim=-1).mean(dim=0)  # [T]
                    l_fp = (p_fp * far_m[None, :, :]).sum(dim=-1).mean(dim=0)
                    l_s3 = (p_s3 * far_m[None, :, :]).sum(dim=-1).mean(dim=0)
                    for bi, (blo, bhi, _) in enumerate(probe_blocks):
                        if T > blo:
                            hi_c = min(T, bhi)
                            ent_fp_blk[bi].append(float(h_fp[blo:hi_c].mean()))
                            ent_s3_blk[bi].append(float(h_s3[blo:hi_c].mean()))
                            # For far-mass block >= 64
                            if hi_c > max(blo, 64):
                                lo_f = max(blo, 64)
                                leak_fp_blk[bi].append(float(l_fp[lo_f:hi_c].mean()))
                                leak_s3_blk[bi].append(float(l_s3[lo_f:hi_c].mean()))

                a = g.attention(q, k, v, l)
                a = g.rmsnorm(a, w_asn)
                x = x + g.bitlinear(a, wo)
                h = g.rmsnorm(x, w_pan)
                m = torch.square(F.relu(g.bitlinear(h, wg))) * g.bitlinear(h, wu)
                m = g.rmsnorm(m, w_fsn)
                xs[idx_s] = x + g.bitlinear(m, wd)

            if collect_probes:
                ls = {
                    "layer": l,
                    "k_rel_l2": math.sqrt(k_err_sq / max(k_norm_sq, 1e-12)),
                    "v_rel_l2": math.sqrt(v_err_sq / max(v_norm_sq, 1e-12)),
                    "far_mass_fp": float(np.mean(far_mass_fp)) if far_mass_fp else float("nan"),
                    "far_mass_s3k": float(np.mean(far_mass_s3k)) if far_mass_s3k else float("nan"),
                }
                for bi, (_, _, bname) in enumerate(probe_blocks):
                    ls[f"H_fp_{bname}"] = float(np.mean(ent_fp_blk[bi])) if ent_fp_blk[bi] else float("nan")
                    ls[f"H_s3k_{bname}"] = float(np.mean(ent_s3_blk[bi])) if ent_s3_blk[bi] else float("nan")
                    ls[f"leak_fp_{bname}"] = float(np.mean(leak_fp_blk[bi])) if leak_fp_blk[bi] else float("nan")
                    ls[f"leak_s3k_{bname}"] = float(np.mean(leak_s3_blk[bi])) if leak_s3_blk[bi] else float("nan")
                layer_stats.append(ls)

            del w_in, wq, wk, wv, w_asn, wo, w_pan, wg, wu, w_fsn, wd

        w_fn = g.vec("model.norm.weight")
        out_lps: list[list[float]] = []
        for inp, x in zip(inps, xs):
            hn = g.rmsnorm(x, w_fn)[0, :-1]
            tgt = inp[0, 1:, None]
            lps: list[float] = []
            for r0 in range(0, hn.shape[0], 256):
                lg = g.lm_logits_rows(hn[r0 : r0 + 256])
                lps.extend(torch.log_softmax(lg.double(), -1).gather(1, tgt[r0 : r0 + 256])[:, 0].tolist())
                del lg
            out_lps.append(lps)
        return out_lps, ({"layers": layer_stats} if collect_probes else None)


def save_scores(path: pathlib.Path, header: str, seqs: list[list[int]], lps_list: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for wi, lps in enumerate(lps_list):
            f.write(f"w {wi} {len(lps)} {-sum(lps):.9f}\n")
            f.write("lp " + " ".join(f"{x:.9f}" for x in lps) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    n_w = args.n_windows
    en_wins = read_ids(H3_BITNET / "flores_bitnet_en.ids", n_w)
    uk_wins = read_ids(H3_BITNET / "flores_bitnet_uk.ids", n_w)

    # Copy/reuse fp32 & S3+s4r32 w=0..11 from H3/scores/bitnet
    for base_fn in ("fp32", "S3_s4r32"):
        for lang in ("en", "uk"):
            dst = SCORES_DIR / f"{base_fn}_{lang}.txt"
            if not dst.exists():
                src_lines = (H3_BITNET / "scores" / "bitnet" / f"{base_fn}_{lang}.txt").read_text(encoding="utf-8").splitlines()
                kept = [src_lines[0]]
                keep_next = False
                for ln in src_lines[1:]:
                    if ln.startswith("w "):
                        wid = int(ln.split()[1])
                        keep_next = wid < n_w
                        if keep_next:
                            kept.append(ln)
                    elif ln.startswith("lp ") and keep_next:
                        kept.append(ln)
                        keep_next = False
                dst.write_text("\n".join(kept) + "\n", encoding="utf-8")

    # Register layer-group schemes cleanly inside kv_ptq
    import golden_bitnet as GB
    import kv_ptq

    s3_hook, _ = kv_ptq.make_hook("S3+s4r32")
    layer_group_specs = {
        "S3_s4r32_L0_9": set(range(0, 10)),
        "S3_s4r32_L10_19": set(range(10, 20)),
        "S3_s4r32_L20_29": set(range(20, 30)),
    }

    ablation_defs = [
        ("S3_s4r32", "S3+s4r32", "Full baseline (all 30 layers, s=4 r=32)"),
        ("KV8_S3_s4r32", "KV8:S3+s4r32", "K-only S3 (V=INT8, all layers, s=4 r=32)"),
        ("K8V_S3_s4r32", "K8V:S3+s4r32", "V-only S3 (K=INT8, all layers, s=4 r=32)"),
        ("S3_s4r0", "S3+s4r0", "Recent window r=0 (s=4 sinks INT8 only)"),
        ("S3_s4r128", "S3+s4r128", "Recent window r=128 (s=4 sinks + 128 recent INT8)"),
        ("S3_s4r256", "S3+s4r256", "Recent window r=256 (s=4 sinks + 256 recent INT8)"),
        ("S3_s4r32_L0_9", "L0..9", "S3+s4r32 only in early layers 0..9 (10..29 fp32)"),
        ("S3_s4r32_L10_19", "L10..19", "S3+s4r32 only in mid layers 10..19 (rest fp32)"),
        ("S3_s4r32_L20_29", "L20..29", "S3+s4r32 only in late layers 20..29 (rest fp32)"),
    ]

    probes_path = H4_DIR / "probes_summary.json"
    if not args.analyze_only:
        GB.VERBOSE = False
        g = GB.Golden(GB.SNAP, torch.float32, args.threads)
        t0 = time.time()

        # Run single-pass fp32 diagnostic probes if missing
        if not probes_path.exists():
            GB.kv_quant = GB.kv_quant_identity
            GB.q_quant = None
            print("[run] diagnostic fp32 probe pass EN+UK (n=12)...", flush=True)
            _, probes_en = run_layer_major(g, en_wins, collect_probes=True)
            _, probes_uk = run_layer_major(g, uk_wins, collect_probes=True)
            probes_path.write_text(json.dumps({"en": probes_en, "uk": probes_uk}, indent=1) + "\n", encoding="utf-8")
            print(f"[done] diagnostic probes in {time.time()-t0:.1f}s", flush=True)

        # Run missing ablations
        for fn_key, scheme_str, _ in ablation_defs:
            if fn_key == "S3_s4r32":
                continue
            need_langs = [lg for lg in ("en", "uk") if not (SCORES_DIR / f"{fn_key}_{lg}.txt").exists()]
            if not need_langs:
                print(f"[skip] {fn_key} already complete", flush=True)
                continue
            if fn_key in layer_group_specs:
                active = layer_group_specs[fn_key]
                def _lg_hook(k, v, layer, act=active):
                    return s3_hook(k, v, layer) if layer in act else (k, v)
                GB.kv_quant = _lg_hook
                GB.q_quant = None
            else:
                kv_ptq.install(scheme_str, GB)

            for lang, wins in (("en", en_wins), ("uk", uk_wins)):
                if lang not in need_langs:
                    continue
                lps_out, _ = run_layer_major(g, wins, collect_probes=False)
                save_scores(SCORES_DIR / f"{fn_key}_{lang}.txt", f"# h4 scheme={fn_key} lang={lang} windows={n_w}", wins, lps_out)
                print(f"[done] {fn_key:16s} {lang} t={time.time()-t0:.1f}s", flush=True)

    # Analyze all results cleanly
    b_en = load_nlls_from_txt(SCORES_DIR / "fp32_en.txt", n_w)
    b_uk = load_nlls_from_txt(SCORES_DIR / "fp32_uk.txt", n_w)
    tok_en = np.array([len(w) - 1 for w in en_wins], dtype=np.float64)
    tok_uk = np.array([len(w) - 1 for w in uk_wins], dtype=np.float64)

    # Baseline S3+s4r32 D for % reduction calculation
    s3_d_en = load_nlls_from_txt(SCORES_DIR / "S3_s4r32_en.txt", n_w) - b_en
    s3_d_uk = load_nlls_from_txt(SCORES_DIR / "S3_s4r32_uk.txt", n_w) - b_uk
    base_D_mean = float((s3_d_uk - s3_d_en).mean())

    csv_rows: list[dict] = []
    md_ablation_rows: list[str] = []
    for fn_key, scheme_str, desc in ablation_defs:
        s_en = load_nlls_from_txt(SCORES_DIR / f"{fn_key}_en.txt", n_w)
        s_uk = load_nlls_from_txt(SCORES_DIR / f"{fn_key}_uk.txt", n_w)
        d_en = s_en - b_en
        d_uk = s_uk - b_uk
        dw = d_uk - d_en
        D_pt, D_lo, D_hi = pair_boot(dw)
        r_en, en_lo, en_hi = ppl_ratio_ci(s_en, b_en, tok_en)
        r_uk, uk_lo, uk_hi = ppl_ratio_ci(s_uk, b_uk, tok_uk)
        rw = (d_uk / b_uk) - (d_en / b_en)
        R_pt, R_lo, R_hi = pair_boot(rw)
        pct_of_full = 100.0 * D_pt / base_D_mean
        removed_pct = 100.0 - pct_of_full
        md_ablation_rows.append(
            f"| `{scheme_str}` | {desc} | ×{r_en:.4f} [{en_lo:.4f},{en_hi:.4f}] | ×{r_uk:.4f} [{uk_lo:.4f},{uk_hi:.4f}] | "
            f"{D_pt:+.3f} [{D_lo:+.3f},{D_hi:+.3f}] | {100*R_pt:+.2f}% [{100*R_lo:+.2f}%,{100*R_hi:+.2f}%] | "
            f"{pct_of_full:.1f}% ({removed_pct:+.1f}% removed) |"
        )
        csv_rows.append(
            {
                "key": fn_key, "scheme": scheme_str, "desc": desc, "n_windows": n_w,
                "ppl_ratio_en": r_en, "ppl_ratio_en_lo": en_lo, "ppl_ratio_en_hi": en_hi,
                "ppl_ratio_uk": r_uk, "ppl_ratio_uk_lo": uk_lo, "ppl_ratio_uk_hi": uk_hi,
                "delta_en_nats": float(d_en.mean()), "delta_uk_nats": float(d_uk.mean()),
                "D_nats": D_pt, "D_ci_lo": D_lo, "D_ci_hi": D_hi,
                "R_diff": R_pt, "R_ci_lo": R_lo, "R_ci_hi": R_hi,
                "pct_of_S3_s4r32_D": pct_of_full,
            }
        )

    with (H4_DIR / "mechanism.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    probes = json.loads(probes_path.read_text(encoding="utf-8"))
    pe = probes["en"]["layers"]
    pu = probes["uk"]["layers"]
    with (H4_DIR / "mechanism_probes_layers.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["layer"] + [f"en_{k}" for k in pe[0] if k != "layer"] + [f"uk_{k}" for k in pu[0] if k != "layer"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for le, lu in zip(pe, pu):
            row = {"layer": le["layer"]}
            for k, v in le.items():
                if k != "layer":
                    row[f"en_{k}"] = v
            for k, v in lu.items():
                if k != "layer":
                    row[f"uk_{k}"] = v
            w.writerow(row)

    # Probe summary helpers
    k_l2_en = float(np.mean([r["k_rel_l2"] for r in pe]))
    k_l2_uk = float(np.mean([r["k_rel_l2"] for r in pu]))
    v_l2_en = float(np.mean([r["v_rel_l2"] for r in pe]))
    v_l2_uk = float(np.mean([r["v_rel_l2"] for r in pu]))
    fm_fp_en = float(np.mean([r["far_mass_fp"] for r in pe]))
    fm_fp_uk = float(np.mean([r["far_mass_fp"] for r in pu]))
    fm_s3_en = float(np.mean([r["far_mass_s3k"] for r in pe]))
    fm_s3_uk = float(np.mean([r["far_mass_s3k"] for r in pu]))

    blk_lines = []
    for bname in ("0..127", "128..255", "256..383"):
        he_fp = [r[f"H_fp_{bname}"] for r in pe if not math.isnan(r[f"H_fp_{bname}"])]
        he_s3 = [r[f"H_s3k_{bname}"] for r in pe if not math.isnan(r[f"H_s3k_{bname}"])]
        hu_fp = [r[f"H_fp_{bname}"] for r in pu if not math.isnan(r[f"H_fp_{bname}"])]
        hu_s3 = [r[f"H_s3k_{bname}"] for r in pu if not math.isnan(r[f"H_s3k_{bname}"])]
        lfe = [r[f"leak_fp_{bname}"] for r in pe if not math.isnan(r[f"leak_fp_{bname}"])]
        lse = [r[f"leak_s3k_{bname}"] for r in pe if not math.isnan(r[f"leak_s3k_{bname}"])]
        lfu = [r[f"leak_fp_{bname}"] for r in pu if not math.isnan(r[f"leak_fp_{bname}"])]
        lsu = [r[f"leak_s3k_{bname}"] for r in pu if not math.isnan(r[f"leak_s3k_{bname}"])]
        e_str = f"{np.mean(he_fp):.3f} → {np.mean(he_s3):.3f} (Δ={np.mean(he_s3)-np.mean(he_fp):+.4f})" if he_fp else "—"
        u_str = f"{np.mean(hu_fp):.3f} → {np.mean(hu_s3):.3f} (Δ={np.mean(hu_s3)-np.mean(hu_fp):+.4f})" if hu_fp else "—"
        le_str = f"{100*np.mean(lfe):.2f}% → {100*np.mean(lse):.2f}%" if lfe else "—"
        lu_str = f"{100*np.mean(lfu):.2f}% → {100*np.mean(lsu):.2f}%" if lfu else "—"
        blk_lines.append(f"| `{bname}` | {e_str} | {u_str} | {le_str} | {lu_str} |")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
        lbls = [r["scheme"] for r in csv_rows]
        ds = [r["D_nats"] for r in csv_rows]
        dlo = [max(0.0, r["D_nats"] - r["D_ci_lo"]) for r in csv_rows]
        dhi = [max(0.0, r["D_ci_hi"] - r["D_nats"]) for r in csv_rows]
        y_pos = np.arange(len(lbls))
        axes[0].barh(y_pos, ds, xerr=[dlo, dhi], color="#d62728", alpha=0.82, capsize=3)
        axes[0].axvline(base_D_mean, color="gray", linestyle="--", linewidth=0.9, label="S3+s4r32")
        axes[0].set_yticks(y_pos); axes[0].set_yticklabels(lbls, fontsize=8.5)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("D = Δ_UK − Δ_EN (nats, 95% CI)")
        axes[0].set_title("Ablations on FLORES-12")
        axes[0].grid(True, axis="x", alpha=0.25)

        layers = [r["layer"] for r in pe]
        axes[1].plot(layers, [r["k_rel_l2"] for r in pu], "-o", markersize=3.5, label=f"K rel L2 UK ({k_l2_uk:.3f})", color="#d62728")
        axes[1].plot(layers, [r["k_rel_l2"] for r in pe], "--o", markersize=3.5, label=f"K rel L2 EN ({k_l2_en:.3f})", color="#1f77b4")
        axes[1].plot(layers, [r["v_rel_l2"] for r in pu], "-s", markersize=3.5, label=f"V rel L2 UK ({v_l2_uk:.3f})", color="#ff7f0e")
        axes[1].plot(layers, [r["v_rel_l2"] for r in pe], "--s", markersize=3.5, label=f"V rel L2 EN ({v_l2_en:.3f})", color="#2ca02c")
        axes[1].set_xlabel("Layer (0..29)")
        axes[1].set_ylabel("Relative L2 S3 quantization error")
        axes[1].set_title("Per-layer S3 K/V quantization error")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.suptitle("H4: Mechanism Ablations & Layer-wise Probes (FLORES-12, BitNet-b1.58-2B)")
        fig.tight_layout()
        fig.savefig(H4_DIR / "mechanism.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}", file=sys.stderr)

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    gov = (pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
           if pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists() else "unknown")

    by_key = {r["key"]: r for r in csv_rows}
    kv8 = by_key["KV8_S3_s4r32"]
    k8v = by_key["K8V_S3_s4r32"]
    r0 = by_key["S3_s4r0"]
    r256 = by_key["S3_s4r256"]
    l0 = by_key["S3_s4r32_L0_9"]
    l10 = by_key["S3_s4r32_L10_19"]
    l20 = by_key["S3_s4r32_L20_29"]

    md_path = H4_DIR / "mechanism.md"
    existing = md_path.read_text(encoding="utf-8")
    plan_end = existing.find("\n## Results")
    plan_part = existing[:plan_end].rstrip() if plan_end != -1 else existing.rstrip()

    out_md = [
        plan_part,
        "",
        "## Results",
        "",
        f"Evaluated on first `N = {n_w}` paired FLORES-200 windows (`flores_bitnet_{{en,uk}}.ids[:{n_w}]`). Baseline `S3+s4r32` has `D = {base_D_mean:+.3f}` nats.",
        "",
        "### 1. Ablations (`D = Δ_UK − Δ_EN`, 10,000-rep paired bootstrap `seed=0`)",
        "",
        "| Ablation | Description | EN PPL ratio [95% CI] | UK PPL ratio [95% CI] | D = Δ_UK−Δ_EN [95% CI] | Rel diff R [95% CI] | Share of `S3+s4r32` `D` |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *md_ablation_rows,
        "",
        "### 2. Single-pass `fp32` Diagnostics (30 layers, `N = 12` EN vs UK)",
        "",
        "| Metric | EN mean | UK mean | UK − EN |",
        "| --- | ---: | ---: | ---: |",
        f"| K rel L2 error under `S3` | {k_l2_en:.4f} | {k_l2_uk:.4f} | {k_l2_uk - k_l2_en:+.4f} |",
        f"| V rel L2 error under `S3` | {v_l2_en:.4f} | {v_l2_uk:.4f} | {v_l2_uk - v_l2_en:+.4f} |",
        f"| Mid/far attn mass `fp32` (`4<=s<=t-32`, `t>=64`) | {100*fm_fp_en:.2f}% | {100*fm_fp_uk:.2f}% | {100*(fm_fp_uk - fm_fp_en):+.2f} pp |",
        f"| Mid/far attn mass under `S3-K+s4r32` (`t>=64`) | {100*fm_s3_en:.2f}% | {100*fm_s3_uk:.2f}% | {100*(fm_s3_uk - fm_s3_en):+.2f} pp |",
        "",
        "| Position block | EN Entropy `fp32 → S3-K` (nats) | UK Entropy `fp32 → S3-K` (nats) | EN Mid/far mass `fp32 → S3-K` | UK Mid/far mass `fp32 → S3-K` |",
        "| --- | --- | --- | --- | --- |",
        *blk_lines,
        "",
        "### 3. Hypothesis Evaluation `(a)–(d)`",
        "",
        f"- **K vs V dominance:** Keeping K at INT8 (`K8V:S3+s4r32`, V ternary) gives `D = {k8v['D_nats']:+.3f}` (`{k8v['pct_of_S3_s4r32_D']:.1f}%` of baseline `D`), whereas keeping V at INT8 (`KV8:S3+s4r32`, K ternary) gives `D = {kv8['D_nats']:+.3f}` (`{kv8['pct_of_S3_s4r32_D']:.1f}%`).",
        f"- **Recent-window `r` sensitivity:** `r=0` gives `D = {r0['D_nats']:+.3f}`, baseline `r=32` gives `D = {base_D_mean:+.3f}`, `r=256` leaves `D = {r256['D_nats']:+.3f}` (`{r256['pct_of_S3_s4r32_D']:.1f}%`). Note also the caveat whether shorter EN mean length (`~182` tok vs `~307` UK) shrinks UK loss more cleanly at `r=256`.",
        f"- **Layer sensitivity (`0..9` / `10..19` / `20..29`):** Early `D={l0['D_nats']:+.3f}` (`{l0['pct_of_S3_s4r32_D']:.1f}%`), mid `D={l10['D_nats']:+.3f}` (`{l10['pct_of_S3_s4r32_D']:.1f}%`), late `D={l20['D_nats']:+.3f}` (`{l20['pct_of_S3_s4r32_D']:.1f}%`).",
        f"- **(a) Far-attention weight:** Baseline mid/far mass outside `s4r32` is `{100*fm_fp_uk:.2f}%` (UK) vs `{100*fm_fp_en:.2f}%` (EN) (`Δ={100*(fm_fp_uk-fm_fp_en):+.2f} pp`).",
        f"- **(b) Intrinsic K/V quantizability:** Relative L2 `S3` error is `{k_l2_uk:.4f}` vs `{k_l2_en:.4f}` (K) and `{v_l2_uk:.4f}` vs `{v_l2_en:.4f}` (V).",
        f"- **(d) Attention leakage (`S3-K`):** `S3-K` shifts mid/far mass `{100*fm_fp_uk:.2f}% → {100*fm_s3_uk:.2f}%` on UK (`{100*(fm_s3_uk-fm_fp_uk):+.2f} pp`) vs `{100*fm_fp_en:.2f}% → {100*fm_s3_en:.2f}%` on EN (`{100*(fm_s3_en-fm_fp_en):+.2f} pp`).",
        "",
        "### Reproducibility & Metadata",
        f"- Command: `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- nice -n 19 ionice -c3 python3 scripts/run_h4_mechanism.py --n-windows {n_w}`",
        f"- `commit {commit}`, governor `{gov}`, `n_boot=10000`, paired by window index, `seed=0`.",
    ]
    md_path.write_text("\n".join(out_md) + "\n", encoding="utf-8")
    print(f"Wrote {H4_DIR / 'mechanism.{csv,md,png}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
