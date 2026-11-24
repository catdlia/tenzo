#!/usr/bin/env python3
"""H1 (T47): Token-count vs equal-character recent-window coverage on FLORES-200 (BitNet, N=32).

Runs under MemoryMax=4G using layer-major unpadded GB.Golden (verified 0.000e+00 vs B=1):
  (a) Token recent-window sweep S3+s4r<r>:
      - EN: r in {8, 12, 16, 19, 24} (+ reused r=32)
      - UK: r in {40, 48, 54, 64, 96} (+ reused r=32)
  (a2) Equal-character recent-window policy S3+s4c<C> (sinks=4 tokens; INT8 iff s<4 or s==t or E_t - c_s <= C):
      - C in {80, 120, 160, 240} for BOTH EN and UK
      - V-only check K8V:S3+s4c120 at C*=120 for BOTH EN and UK

Writes results/2026-11-24-b_agent2/H1/{scores/*.txt, char_window_coverage.csv, char_window_coverage.md, char_window_coverage.png}.
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

H3_SCORES = ROOT / "results" / "2026-11-24_agent2" / "H3" / "scores" / "bitnet"
H3_IDS_DIR = ROOT / "results" / "2026-11-24_agent2" / "H3"
TEXTS_JSON = pathlib.Path.home() / "tenzo-refs" / "datasets" / "flores200" / "windows_flores_texts.json"
TOK_JSON = ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json"
H1_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H1"
SCORES_DIR = H1_DIR / "scores"


def read_ids(path: pathlib.Path) -> list[list[int]]:
    return [list(map(int, ln.split())) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_nlls_from_txt(path: pathlib.Path, n_w: int = 32) -> np.ndarray:
    vals: dict[int, float] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("w "):
            _, wid, _, nll = ln.split()
            vals[int(wid)] = float(nll)
    return np.array([vals[i] for i in range(n_w)], dtype=np.float64)


def save_scores(path: pathlib.Path, header: str, lps_list: list[list[float]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(header.rstrip("\n") + "\n")
        for wi, lps in enumerate(lps_list):
            f.write(f"w {wi} {len(lps)} {-sum(lps):.9f}\n")
            f.write("lp " + " ".join(f"{x:.9f}" for x in lps) + "\n")
    tmp.replace(path)


def boot_mean_ci(arr: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(arr)
    idx = rng.randint(0, n, size=(n_boot, n))
    m = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def build_char_offsets(seqs: list[list[int]], texts: list[str]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from tokenizers import Tokenizer as FastTok
    ft = FastTok.from_file(str(TOK_JSON))
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for s_ids, txt in zip(seqs, texts):
        enc = ft.encode(txt, add_special_tokens=False)
        assert enc.ids == s_ids[1:], "Tokenizer mismatch against FLORES .ids"
        T = len(s_ids)
        c_arr = np.zeros(T, dtype=np.int32)
        e_arr = np.zeros(T, dtype=np.int32)
        l_arr = np.zeros(T, dtype=np.int32)
        for idx, (cs, ce) in enumerate(enc.offsets, start=1):
            c_arr[idx] = cs
            e_arr[idx] = ce
            l_arr[idx] = ce - cs
        out.append((c_arr, e_arr, l_arr))
    return out


def char_window_mask(c_arr: np.ndarray, e_arr: np.ndarray, C: int, sinks: int = 4) -> torch.Tensor:
    T = len(c_arr)
    t_idx = np.arange(T)[:, None]
    s_idx = np.arange(T)[None, :]
    m = (s_idx <= t_idx) & ((s_idx < sinks) | (s_idx == t_idx) | ((e_arr[t_idx] - c_arr[s_idx]) <= C))
    return torch.from_numpy(m)


def coverage_stats_token_r(offsets_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]], r: int) -> dict[str, float]:
    exact_spans: list[float] = []
    approx_spans: list[float] = []
    r_effs: list[float] = []
    t_means: list[float] = []
    for c_arr, e_arr, l_arr in offsets_list:
        T = len(l_arr)
        t_means.append(float(T))
        total_chars = float(e_arr[-1])
        approx_spans.append(r * (total_chars / max(1, T - 1)))
        if T > r:
            step_spans = [float(l_arr[max(0, t - r + 1) : t + 1].sum()) for t in range(r, T)]
            exact_spans.append(float(np.mean(step_spans)))
        else:
            exact_spans.append(total_chars)
        # Recent INT8 count beyond sinks=4 at t >= 4
        rc = [min(r, t - 3) for t in range(4, T)]
        r_effs.append(float(np.mean(rc)) if rc else 0.0)
    t_m = float(np.mean(t_means))
    re_m = float(np.mean(r_effs))
    hi_512 = min(512.0, 4.0 + re_m)
    hi_win = min(t_m, 4.0 + re_m)
    i8, ter = 8.0 + 16.0 / 128.0, 2.0 + 16.0 / 32.0
    return {
        "cov_chars_exact": float(np.mean(exact_spans)),
        "cov_chars_approx": float(np.mean(approx_spans)),
        "r_eff": re_m,
        "mean_tok": t_m - 1.0,
        "bits_win": (hi_win * i8 + max(0.0, t_m - hi_win) * ter) / t_m,
        "bits_at_512": (hi_512 * i8 + (512.0 - hi_512) * ter) / 512.0,
    }


def coverage_stats_char_C(offsets_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]], C: int, sinks: int = 4) -> dict[str, float]:
    cov_steady: list[float] = []
    cov_all: list[float] = []
    r_effs: list[float] = []
    r_final: list[float] = []
    t_means: list[float] = []
    for c_arr, e_arr, l_arr in offsets_list:
        T = len(l_arr)
        t_means.append(float(T))
        masks = char_window_mask(c_arr, e_arr, C, sinks=sinks).numpy()
        rc_list = []
        ca_list = []
        cs_list = []
        for t in range(sinks, T):
            recent_mask = masks[t, sinks : t + 1]
            cnt = int(recent_mask.sum())
            span = float(l_arr[sinks : t + 1][recent_mask].sum())
            rc_list.append(cnt)
            ca_list.append(span)
            if (e_arr[t] - c_arr[sinks]) >= C:
                cs_list.append(span)
        r_effs.append(float(np.mean(rc_list)) if rc_list else 0.0)
        r_final.append(float(rc_list[-1]) if rc_list else 0.0)
        cov_all.append(float(np.mean(ca_list)) if ca_list else 0.0)
        cov_steady.append(float(np.mean(cs_list)) if cs_list else (float(np.mean(ca_list)) if ca_list else 0.0))
    t_m = float(np.mean(t_means))
    re_m = float(np.mean(r_effs))
    hi_512 = min(512.0, 4.0 + re_m)
    hi_512_end = min(512.0, 4.0 + float(np.mean(r_final)))
    hi_win = min(t_m, 4.0 + re_m)
    i8, ter = 8.0 + 16.0 / 128.0, 2.0 + 16.0 / 32.0
    return {
        "cov_chars_exact": float(np.mean(cov_steady)),
        "cov_chars_all_t": float(np.mean(cov_all)),
        "r_eff": re_m,
        "r_eff_end": float(np.mean(r_final)),
        "mean_tok": t_m - 1.0,
        "bits_win": (hi_win * i8 + max(0.0, t_m - hi_win) * ter) / t_m,
        "bits_at_512": (hi_512 * i8 + (512.0 - hi_512) * ter) / 512.0,
        "bits_at_512_steady": (hi_512_end * i8 + (512.0 - hi_512_end) * ter) / 512.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    raw_texts = json.loads(TEXTS_JSON.read_text(encoding="utf-8"))["windows"]
    en_wins = read_ids(H3_IDS_DIR / "flores_bitnet_en.ids")
    uk_wins = read_ids(H3_IDS_DIR / "flores_bitnet_uk.ids")
    assert len(en_wins) == 32 and len(uk_wins) == 32 and len(raw_texts) == 32

    en_off = build_char_offsets(en_wins, [w["en_text"] for w in raw_texts])
    uk_off = build_char_offsets(uk_wins, [w["uk_text"] for w in raw_texts])

    # Copy existing fp32 & S3+s4r32 baselines into H1/scores for self-containment
    for fn in ("fp32_en.txt", "fp32_uk.txt", "S3_s4r32_en.txt", "S3_s4r32_uk.txt"):
        dst = SCORES_DIR / fn
        if not dst.exists():
            dst.write_text((H3_SCORES / fn).read_text(encoding="utf-8"), encoding="utf-8")

    # Define all evaluation runs
    en_r_list = [8, 12, 16, 19, 24]
    uk_r_list = [40, 48, 54, 64, 96]
    char_Cs = [80, 120, 160, 240]

    if not args.analyze_only:
        import golden_bitnet as GB
        import kv_ptq

        GB.VERBOSE = False
        g = GB.Golden(GB.SNAP, torch.float32, args.threads)
        quiet = GB.Dumper(None, 0, [], [])
        STATE = {"masks": None, "idx": 0}

        def score_seqs_layer_major(seqs: list[list[int]], seq_masks: list[torch.Tensor] | None = None) -> list[list[float]]:
            STATE["masks"] = seq_masks
            tmax = max(len(s) for s in seqs)
            cos_all, sin_all = g.rope(tmax)
            inps = [torch.tensor([s], dtype=torch.long) for s in seqs]
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
                    for idx_s, x in enumerate(xs):
                        STATE["idx"] = idx_s
                        B, T, _ = x.shape
                        h = g.rmsnorm(x, w_in)
                        q = g.bitlinear(h, wq).view(B, T, g.H, g.D).transpose(1, 2)
                        k = g.bitlinear(h, wk).view(B, T, g.Hkv, g.D).transpose(1, 2)
                        v = g.bitlinear(h, wv).view(B, T, g.Hkv, g.D).transpose(1, 2)
                        c, s = cos_all[:T][None, None], sin_all[:T][None, None]
                        q = q * c + g.rotate_half(q) * s
                        k = k * c + g.rotate_half(k) * s
                        a = g.attention(q, k, v, l)
                        a = g.rmsnorm(a, w_asn)
                        x = x + g.bitlinear(a, wo)
                        h = g.rmsnorm(x, w_pan)
                        m = torch.square(F.relu(g.bitlinear(h, wg))) * g.bitlinear(h, wu)
                        m = g.rmsnorm(m, w_fsn)
                        xs[idx_s] = x + g.bitlinear(m, wd)
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
                return out_lps

        def make_char_hook(v_only: bool = False):
            def _hook(k, v, layer):
                mask = STATE["masks"][STATE["idx"]]
                assert mask.shape[0] == k.shape[2], (mask.shape, k.shape)
                kh = kv_ptq.int8_absmax(k) if v_only else kv_ptq.tern_g32_rot(k)
                vh = kv_ptq.tern_g32_rot(v)
                return {
                    "k": kh, "v": vh,
                    "k_alt": kv_ptq.int8_absmax(k),
                    "v_alt": kv_ptq.int8_absmax(v),
                    "alt_mask": mask,
                }
            return _hook

        # Parity check vs B=1 on S3+s4c120 (en[0], uk[0])
        masks_c120_pair = [char_window_mask(en_off[0][0], en_off[0][1], 120), char_window_mask(uk_off[0][0], uk_off[0][1], 120)]
        GB.kv_quant = make_char_hook(False)
        GB.q_quant = None
        lm_pair = score_seqs_layer_major([en_wins[0], uk_wins[0]], masks_c120_pair)
        for pi, seq in enumerate([en_wins[0], uk_wins[0]]):
            STATE["masks"] = masks_c120_pair
            STATE["idx"] = pi
            inp = torch.tensor([seq], dtype=torch.long)
            with torch.no_grad():
                hn = g.hidden(inp, quiet)[0, :-1]
                tgt = inp[0, 1:, None]
                b1_lps = []
                for r0 in range(0, hn.shape[0], 256):
                    lg = g.lm_logits_rows(hn[r0 : r0 + 256])
                    b1_lps.extend(torch.log_softmax(lg.double(), -1).gather(1, tgt[r0 : r0 + 256])[:, 0].tolist())
                    del lg
            diff = max(abs(a - b) for a, b in zip(lm_pair[pi], b1_lps))
            print(f"[parity] S3+s4c120 seq={pi} max_abs_lp_diff={diff:.3e}", flush=True)
            assert diff < 1e-5, f"Parity failed: {diff}"

        t0 = time.time()
        # (a) Token sweeps
        for lang, wins, r_vals in (("en", en_wins, en_r_list), ("uk", uk_wins, uk_r_list)):
            for r_val in r_vals:
                dst = SCORES_DIR / f"S3_s4r{r_val}_{lang}.txt"
                if dst.exists() and sum(1 for ln in dst.read_text().splitlines() if ln.startswith("w ")) == 32:
                    print(f"[skip] {dst.name}", flush=True)
                    continue
                kv_ptq.install(f"S3+s4r{r_val}", GB)
                lps = score_seqs_layer_major(wins, None)
                save_scores(dst, f"# h1 scheme=S3+s4r{r_val} lang={lang} windows=32", lps)
                print(f"[done] S3+s4r{r_val} {lang} t={time.time()-t0:.1f}s", flush=True)

        # (a2) Char-window sweeps & K8V:S3+s4c120
        for C in char_Cs:
            for lang, wins, offs in (("en", en_wins, en_off), ("uk", uk_wins, uk_off)):
                dst = SCORES_DIR / f"S3_s4c{C}_{lang}.txt"
                if dst.exists() and sum(1 for ln in dst.read_text().splitlines() if ln.startswith("w ")) == 32:
                    print(f"[skip] {dst.name}", flush=True)
                    continue
                masks = [char_window_mask(c, e, C) for c, e, _ in offs]
                GB.kv_quant = make_char_hook(False)
                GB.q_quant = None
                lps = score_seqs_layer_major(wins, masks)
                save_scores(dst, f"# h1 scheme=S3+s4c{C} lang={lang} windows=32", lps)
                print(f"[done] S3+s4c{C} {lang} t={time.time()-t0:.1f}s", flush=True)

        for lang, wins, offs in (("en", en_wins, en_off), ("uk", uk_wins, uk_off)):
            dst = SCORES_DIR / f"K8V_S3_s4c120_{lang}.txt"
            if dst.exists() and sum(1 for ln in dst.read_text().splitlines() if ln.startswith("w ")) == 32:
                print(f"[skip] {dst.name}", flush=True)
                continue
            masks = [char_window_mask(c, e, 120) for c, e, _ in offs]
            GB.kv_quant = make_char_hook(True)
            GB.q_quant = None
            lps = score_seqs_layer_major(wins, masks)
            save_scores(dst, f"# h1 scheme=K8V:S3+s4c120 lang={lang} windows=32", lps)
            print(f"[done] K8V:S3+s4c120 {lang} t={time.time()-t0:.1f}s", flush=True)

    # Analyze all runs cleanly
    b_en = load_nlls_from_txt(SCORES_DIR / "fp32_en.txt", 32)
    b_uk = load_nlls_from_txt(SCORES_DIR / "fp32_uk.txt", 32)

    single_rows: list[dict] = []
    # Token sweeps
    for lang, offs, base, r_vals in (
        ("en", en_off, b_en, en_r_list + [32]),
        ("uk", uk_off, b_uk, [32] + uk_r_list),
    ):
        for r_val in sorted(r_vals):
            s = load_nlls_from_txt(SCORES_DIR / f"S3_s4r{r_val}_{lang}.txt", 32)
            delta = s - base
            rel = delta / base
            d_pt, d_lo, d_hi = boot_mean_ci(delta)
            r_pt, r_lo, r_hi = boot_mean_ci(rel)
            st = coverage_stats_token_r(offs, r_val)
            single_rows.append({
                "family": "token_r", "policy": f"S3+s4r{r_val}", "param": r_val, "lang": lang,
                "cov_chars_exact": st["cov_chars_exact"], "cov_chars_alt": st["cov_chars_approx"],
                "r_eff": st["r_eff"], "bits_win": st["bits_win"], "bits_at_512": st["bits_at_512"],
                "delta_nats": d_pt, "delta_ci_lo": d_lo, "delta_ci_hi": d_hi,
                "rel": r_pt, "rel_ci_lo": r_lo, "rel_ci_hi": r_hi,
            })

    # Char sweeps + K8V:C=120
    paired_rows: list[dict] = []
    pair_specs = [("token_r32", "S3+s4r32", "S3_s4r32", None)] + [
        ("char_C", f"S3+s4c{C}", f"S3_s4c{C}", C) for C in char_Cs
    ] + [("char_C_vonly", "K8V:S3+s4c120", "K8V_S3_s4c120", 120)]

    r32_rw = (load_nlls_from_txt(SCORES_DIR / "S3_s4r32_uk.txt", 32) - b_uk) / b_uk - (load_nlls_from_txt(SCORES_DIR / "S3_s4r32_en.txt", 32) - b_en) / b_en

    for fam, pname, fkey, c_val in pair_specs:
        s_en = load_nlls_from_txt(SCORES_DIR / f"{fkey}_en.txt", 32)
        s_uk = load_nlls_from_txt(SCORES_DIR / f"{fkey}_uk.txt", 32)
        d_en = s_en - b_en
        d_uk = s_uk - b_uk
        rel_en = d_en / b_en
        rel_uk = d_uk / b_uk
        dw = d_uk - d_en
        rw = rel_uk - rel_en

        de_pt, de_lo, de_hi = boot_mean_ci(d_en)
        du_pt, du_lo, du_hi = boot_mean_ci(d_uk)
        re_pt, re_lo, re_hi = boot_mean_ci(rel_en)
        ru_pt, ru_lo, ru_hi = boot_mean_ci(rel_uk)
        D_pt, D_lo, D_hi = boot_mean_ci(dw)
        R_pt, R_lo, R_hi = boot_mean_ci(rw)
        diff_vs_r32_pt, diff_vs_r32_lo, diff_vs_r32_hi = boot_mean_ci(r32_rw - rw)

        if c_val is not None:
            st_en = coverage_stats_char_C(en_off, c_val)
            st_uk = coverage_stats_char_C(uk_off, c_val)
            for lang, st, d_tup, r_tup in (
                ("en", st_en, (de_pt, de_lo, de_hi), (re_pt, re_lo, re_hi)),
                ("uk", st_uk, (du_pt, du_lo, du_hi), (ru_pt, ru_lo, ru_hi)),
            ):
                single_rows.append({
                    "family": fam, "policy": pname, "param": c_val, "lang": lang,
                    "cov_chars_exact": st["cov_chars_exact"], "cov_chars_alt": st["cov_chars_all_t"],
                    "r_eff": st["r_eff"], "bits_win": st["bits_win"], "bits_at_512": st["bits_at_512"],
                    "delta_nats": d_tup[0], "delta_ci_lo": d_tup[1], "delta_ci_hi": d_tup[2],
                    "rel": r_tup[0], "rel_ci_lo": r_tup[1], "rel_ci_hi": r_tup[2],
                })
        else:
            st_en = coverage_stats_token_r(en_off, 32)
            st_uk = coverage_stats_token_r(uk_off, 32)

        paired_rows.append({
            "family": fam, "policy": pname,
            "cov_en": st_en["cov_chars_exact"], "cov_uk": st_uk["cov_chars_exact"],
            "r_eff_en": st_en["r_eff"], "r_eff_uk": st_uk["r_eff"],
            "bits512_en": st_en["bits_at_512"], "bits512_uk": st_uk["bits_at_512"],
            "bits_win_en": st_en["bits_win"], "bits_win_uk": st_uk["bits_win"],
            "delta_en": de_pt, "delta_en_lo": de_lo, "delta_en_hi": de_hi,
            "delta_uk": du_pt, "delta_uk_lo": du_lo, "delta_uk_hi": du_hi,
            "rel_en": re_pt, "rel_en_lo": re_lo, "rel_en_hi": re_hi,
            "rel_uk": ru_pt, "rel_uk_lo": ru_lo, "rel_uk_hi": ru_hi,
            "D": D_pt, "D_lo": D_lo, "D_hi": D_hi,
            "R": R_pt, "R_lo": R_lo, "R_hi": R_hi,
            "R_reduction_vs_r32": diff_vs_r32_pt, "R_red_lo": diff_vs_r32_lo, "R_red_hi": diff_vs_r32_hi,
        })

    # Write CSV (both tables cleanly stacked or separate CSVs)
    with (H1_DIR / "char_window_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(single_rows[0].keys()))
        w.writeheader()
        for r in single_rows:
            w.writerow(r)

    with (H1_DIR / "char_window_paired.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        w.writeheader()
        for r in paired_rows:
            w.writerow(r)

    # Two-panel Plot: (1) rel_L vs covered chars, (2) rel_L vs effective bits@512
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
        for fam, ls, mk, alpha, lbl_sfx in (
            ("token_r", "-", "o", 0.85, "token window r"),
            ("char_C", "--", "s", 0.90, "char window C"),
        ):
            for lang, col, lname in (("en", "#1f77b4", "EN"), ("uk", "#d62728", "UK")):
                sub = [r for r in single_rows if r["family"] == fam and r["lang"] == lang]
                sub.sort(key=lambda z: z["cov_chars_exact"])
                xs_c = [z["cov_chars_exact"] for z in sub]
                xs_b = [z["bits_at_512"] for z in sub]
                ys = [100.0 * z["rel"] for z in sub]
                ylo = [100.0 * (z["rel"] - z["rel_ci_lo"]) for z in sub]
                yhi = [100.0 * (z["rel_ci_hi"] - z["rel"]) for z in sub]
                axes[0].errorbar(xs_c, ys, yerr=[ylo, yhi], fmt=mk + ls, color=col, alpha=alpha, capsize=3, label=f"{lname} ({lbl_sfx})")
                axes[1].errorbar(xs_b, ys, yerr=[ylo, yhi], fmt=mk + ls, color=col, alpha=alpha, capsize=3, label=f"{lname} ({lbl_sfx})")

        axes[0].axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        axes[0].set_xlabel("Covered recent chars (mean over active steps)")
        axes[0].set_ylabel("Relative NLL loss rel_L (%) [95% CI]")
        axes[0].set_title("Loss vs Covered Characters (Equal Coverage Test)")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)

        axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        axes[1].set_xlabel("Effective KV bits/element (@ctx=512)")
        axes[1].set_ylabel("Relative NLL loss rel_L (%) [95% CI]")
        axes[1].set_title("Loss vs Bit Budget (Fairness Cost)")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8)

        fig.suptitle("H1 (T47): Token vs Equal-Character Recent Window on FLORES-200 (BitNet, N=32)")
        fig.tight_layout()
        fig.savefig(H1_DIR / "char_window_coverage.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}", file=sys.stderr)

    # Decision rule at C* = 120 vs r = 32
    r32_row = [r for r in paired_rows if r["policy"] == "S3+s4r32"][0]
    c120_row = [r for r in paired_rows if r["policy"] == "S3+s4c120"][0]

    def judge(row: dict) -> str:
        contains_zero = (row["R_lo"] <= 0.0 <= row["R_hi"])
        less_third = abs(row["R"]) < (abs(r32_row["R"]) / 3.0)
        red_pos = row["R_red_lo"] > 0.0
        if contains_zero and less_third:
            return "EXPLAINS (CI(R) includes 0 & |R| < 1/3 R_r32)"
        if red_pos and row["R_lo"] > 0.0:
            return "PARTIAL (R significantly reduced, CI(R) > 0)"
        if contains_zero or red_pos:
            return "PARTIAL (CI(R) spans 0 or R reduced, |R| >= 1/3 R_r32)"
        return "DOES NOT EXPLAIN (R not smaller at equal chars)"

    c120_verdict = judge(c120_row)

    # Append results to char_window_coverage.md preserving ## Plan
    md_path = H1_DIR / "char_window_coverage.md"
    existing = md_path.read_text(encoding="utf-8")
    plan_end = existing.find("\n## Results")
    plan_part = existing[:plan_end].rstrip() if plan_end != -1 else existing.rstrip()

    tok_md = []
    for r in sorted([z for z in single_rows if z["family"] == "token_r"], key=lambda x: (x["lang"], x["param"])):
        tok_md.append(
            f"| `{r['lang'].upper()}` | `{r['policy']}` | {r['cov_chars_exact']:.1f} | {r['cov_chars_alt']:.1f} | "
            f"{r['r_eff']:.1f} | {r['bits_win']:.3f} / {r['bits_at_512']:.3f} | "
            f"{r['delta_nats']:+.3f} [{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}] | "
            f"**{100*r['rel']:+.2f}% [{100*r['rel_ci_lo']:+.2f}%, {100*r['rel_ci_hi']:+.2f}%]** |"
        )

    pair_md = []
    for r in paired_rows:
        vstr = "baseline" if r["policy"] == "S3+s4r32" else judge(r)
        pair_md.append(
            f"| `{r['policy']}` | {r['cov_en']:.1f} / {r['cov_uk']:.1f} | {r['r_eff_en']:.1f} / {r['r_eff_uk']:.1f} | "
            f"{r['bits512_en']:.3f} / {r['bits512_uk']:.3f} | "
            f"{100*r['rel_en']:+.2f}% | {100*r['rel_uk']:+.2f}% | "
            f"{r['D']:+.3f} [{r['D_lo']:+.3f}, {r['D_hi']:+.3f}] | "
            f"**{100*r['R']:+.2f}% [{100*r['R_lo']:+.2f}%, {100*r['R_hi']:+.2f}%]** | "
            f"{100*r['R_reduction_vs_r32']:+.2f}% [{100*r['R_red_lo']:+.2f}%, {100*r['R_red_hi']:+.2f}%] | {vstr} |"
        )

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    gov = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()

    out_md = [
        plan_part,
        "",
        "## Results",
        "",
        "### 1. Part (a) — Token-Window Sweep (`S3+s4r<r>`, BitNet `FLORES-200`, `N = 32`)",
        "",
        "| Lang | Policy | Exact covered chars | Linear approx `r × ch/tok` | Mean active `r_eff` | Bits (`win` / `@512`) | `Δ` (nats) [95% CI] | `rel_L` [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        *tok_md,
        "",
        "### 2. Part (a2) — Equal-Character Policy (`S3+s4c<C>`) & Contrast vs `S3+s4r32`",
        "",
        "| Policy | Cov chars EN/UK | Mean `r_eff` EN/UK | Bits `@512` EN/UK | `rel_EN` | `rel_UK` | `D` (nats) [95% CI] | `R = UK − EN` [95% CI] | `R(r32) − R(C)` [95% CI] | Verdict |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
        *pair_md,
        "",
        "### 3. Pre-Registered Decision Outcome",
        "",
        f"- **Baseline `S3+s4r32` (equal tokens):** `EN` covers `{r32_row['cov_en']:.1f}` chars/step vs `UK` `{r32_row['cov_uk']:.1f}` chars/step (`{r32_row['cov_uk']/r32_row['cov_en']:.2%}` of EN coverage); `D = {r32_row['D']:+.3f}` nats, **`R = {100*r32_row['R']:+.2f}% [{100*r32_row['R_lo']:+.2f}%, {100*r32_row['R_hi']:+.2f}%]`**.",
        f"- **Primary test at `C* = 120` (`S3+s4c120`):** **`{c120_verdict}`**. At `C = 120`, `R = {100*c120_row['R']:+.2f}% [{100*c120_row['R_lo']:+.2f}%, {100*c120_row['R_hi']:+.2f}%]` (vs threshold `|R| < {100*abs(r32_row['R'])/3:.2f}%`); UK requires `{c120_row['r_eff_uk']:.1f}` INT8 recent slots (`{c120_row['bits512_uk']:.3f}` bits `@512`, `{c120_row['bits_win_uk']:.3f}` bits on FLORES window) vs `{c120_row['r_eff_en']:.1f}` EN slots (`{c120_row['bits512_en']:.3f}` bits `@512`).",
        "",
        "### Reproducibility & Metadata",
        "- Command: `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- nice -n 19 ionice -c3 python3 scripts/run_h1_char_window.py`",
        f"- `commit {commit}`, governor `{gov}`, `n_boot=10000`, paired by window index, `seed=0`.",
    ]
    md_path.write_text("\n".join(out_md) + "\n", encoding="utf-8")
    print(f"Wrote {H1_DIR / 'char_window_coverage.{csv,md,png}'} and char_window_paired.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
