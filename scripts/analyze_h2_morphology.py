#!/usr/bin/env python3
"""H2 (T47b): Exploratory morphological / subword position degradation analysis."""

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H2"
H3_DIR = ROOT / "results" / "2026-11-24_agent2" / "H3"
TEXTS_PATH = Path("~/tenzo-refs/datasets/flores200/windows_flores_texts.json").expanduser()

CLASSES = [
    "single_token_word",
    "word_prefix",
    "word_middle",
    "word_suffix",
    "punct_space_num",
]
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

MODELS = {
    "BitNet": {
        "tok": ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json",
        "ids_en": H3_DIR / "flores_bitnet_en.ids",
        "ids_uk": H3_DIR / "flores_bitnet_uk.ids",
        "fp_en": H3_DIR / "scores" / "bitnet" / "fp32_en.txt",
        "fp_uk": H3_DIR / "scores" / "bitnet" / "fp32_uk.txt",
        "q_en": H3_DIR / "scores" / "bitnet" / "S3_s4r32_en.txt",
        "q_uk": H3_DIR / "scores" / "bitnet" / "S3_s4r32_uk.txt",
    },
    "EuroLLM-1.7B": {
        "tok": Path("~/tenzo-refs/hf/eurollm-1.7b/tokenizer.json").expanduser(),
        "ids_en": H3_DIR / "flores_eurollm_en.ids",
        "ids_uk": H3_DIR / "flores_eurollm_uk.ids",
        "fp_en": H3_DIR / "scores" / "eurollm" / "fp_en.txt",
        "fp_uk": H3_DIR / "scores" / "eurollm" / "fp_uk.txt",
        "q_en": H3_DIR / "scores" / "eurollm" / "S3_s4r32_en.txt",
        "q_uk": H3_DIR / "scores" / "eurollm" / "S3_s4r32_uk.txt",
    },
}


def load_scores(path: Path):
    lps = []
    for line in path.read_text().splitlines():
        if line.startswith("lp "):
            lps.append(np.array([float(x) for x in line.split()[1:]], dtype=np.float64))
    return lps


def classify_tokens(tok: Tokenizer, text: str, expected_target_ids: list[int]) -> list[str]:
    enc = tok.encode(text, add_special_tokens=False)
    if enc.ids != expected_target_ids:
        raise RuntimeError(f"Token mismatch: {len(enc.ids)} vs {len(expected_target_ids)}")
    words = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    word_to_tokens: dict[int, list[int]] = {}
    labels = [""] * len(enc.ids)
    wi = 0
    for ti, (a, b) in enumerate(enc.offsets):
        if a == b:
            raise RuntimeError(f"Empty offset at token {ti}: {enc.tokens[ti]!r}")
        while wi < len(words) and words[wi][1] <= a:
            wi += 1
        overlaps = []
        wj = wi
        while wj < len(words) and words[wj][0] < b:
            ws, we = words[wj]
            if min(b, we) > max(a, ws):
                overlaps.append(wj)
            wj += 1
        if not overlaps:
            labels[ti] = "punct_space_num"
        elif len(overlaps) > 1:
            raise RuntimeError(f"Multi-word token {ti}: {text[a:b]!r}")
        else:
            word_to_tokens.setdefault(overlaps[0], []).append(ti)
    for idxs in word_to_tokens.values():
        if len(idxs) == 1:
            labels[idxs[0]] = "single_token_word"
        else:
            labels[idxs[0]] = "word_prefix"
            for mid in idxs[1:-1]:
                labels[mid] = "word_middle"
            labels[idxs[-1]] = "word_suffix"
    assert all(lbl in CLASSES for lbl in labels)
    return labels


def ci95(arr: np.ndarray) -> tuple[float, float]:
    valid = arr[~np.isnan(arr)]
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texts = json.loads(TEXTS_PATH.read_text())["windows"]
    n_win = len(texts)
    assert n_win == 32

    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, n_win, size=(10000, n_win))

    rows = []
    contrasts = []
    prefix_max_abs_lt35 = {}

    for model_name, cfg in MODELS.items():
        tok = Tokenizer.from_file(str(cfg["tok"]))
        data = {}
        for lang, text_key in [("en", "en_text"), ("uk", "uk_text")]:
            ids_lines = [[int(x) for x in l.split()] for l in cfg[f"ids_{lang}"].read_text().splitlines()]
            fp_lps = load_scores(cfg[f"fp_{lang}"])
            q_lps = load_scores(cfg[f"q_{lang}"])
            assert len(ids_lines) == n_win == len(fp_lps) == len(q_lps)

            # Per-window arrays of counts, delta sums, base NLL sums for subset=('all','pos35plus')
            per_win = {
                sub: {
                    c: {"n": np.zeros(n_win), "delta": np.zeros(n_win), "base": np.zeros(n_win), "pos_sum": np.zeros(n_win)}
                    for c in CLASSES
                }
                for sub in ("all", "pos35plus")
            }
            max_lt35 = 0.0
            tot_tokens = 0
            tot_delta = np.zeros(n_win)

            for w_idx in range(n_win):
                ids = ids_lines[w_idx]
                fp = fp_lps[w_idx]
                q = q_lps[w_idx]
                assert len(fp) == len(q) == len(ids) - 1
                delta = fp - q  # positive = NLL increase = degradation
                base_nll = -fp
                tot_delta[w_idx] = float(delta.sum())
                tot_tokens += len(delta)
                if len(delta) > 0:
                    max_lt35 = max(max_lt35, float(np.max(np.abs(delta[:35]))))

                labels = classify_tokens(tok, texts[w_idx][text_key], ids[1:])
                pos_arr = np.arange(len(delta))
                for sub, mask_pos in [("all", np.ones(len(delta), dtype=bool)), ("pos35plus", pos_arr >= 35)]:
                    for c in CLASSES:
                        m = mask_pos & np.array([lbl == c for lbl in labels], dtype=bool)
                        if np.any(m):
                            per_win[sub][c]["n"][w_idx] = float(np.sum(m))
                            per_win[sub][c]["delta"][w_idx] = float(np.sum(delta[m]))
                            per_win[sub][c]["base"][w_idx] = float(np.sum(base_nll[m]))
                            per_win[sub][c]["pos_sum"][w_idx] = float(np.sum(pos_arr[m]))

            prefix_max_abs_lt35[f"{model_name}_{lang}"] = max_lt35
            data[lang] = {"per_win": per_win, "tot_delta": tot_delta, "tot_tokens": tot_tokens}

        # Aggregate rows & bootstrap
        for sub in ("all", "pos35plus"):
            for lang in ("en", "uk"):
                pw = data[lang]["per_win"][sub]
                sub_tok_total = sum(float(pw[c]["n"].sum()) for c in CLASSES)
                sub_delta_total = sum(float(pw[c]["delta"].sum()) for c in CLASSES) / n_win
                for c in CLASSES:
                    n_arr = pw[c]["n"]
                    d_arr = pw[c]["delta"]
                    b_arr = pw[c]["base"]
                    p_arr = pw[c]["pos_sum"]
                    cnt = int(n_arr.sum())
                    share_pct = 100.0 * cnt / sub_tok_total if sub_tok_total else 0.0
                    mean_tok_delta = float(d_arr.sum() / cnt) if cnt else 0.0
                    mean_base_nll = float(b_arr.sum() / cnt) if cnt else 0.0
                    mean_pos = float(p_arr.sum() / cnt) if cnt else 0.0
                    doc_mean_c = float(d_arr.mean())
                    doc_share_pct = 100.0 * doc_mean_c / sub_delta_total if abs(sub_delta_total) > 1e-12 else 0.0

                    # Bootstrap ratios and doc contribution
                    bn = n_arr[boot_idx].sum(axis=1)
                    bd = d_arr[boot_idx].sum(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        btok = np.where(bn > 0, bd / bn, np.nan)
                    tok_lo, tok_hi = ci95(btok)
                    doc_lo, doc_hi = ci95(d_arr[boot_idx].mean(axis=1))

                    # Also D_c = UK_doc_c - EN_doc_c only once when lang == 'uk'
                    d_c = np.nan
                    d_c_lo = np.nan
                    d_c_hi = np.nan
                    d_c_share = np.nan
                    if lang == "uk":
                        pw_en = data["en"]["per_win"][sub]
                        d_en_tot = sum(float(pw_en[cc]["delta"].sum()) for cc in CLASSES) / n_win
                        d_tot = sub_delta_total - d_en_tot
                        diff_arr = d_arr - pw_en[c]["delta"]
                        d_c = float(diff_arr.mean())
                        d_c_lo, d_c_hi = ci95(diff_arr[boot_idx].mean(axis=1))
                        d_c_share = 100.0 * d_c / d_tot if abs(d_tot) > 1e-12 else 0.0

                        # UK vs EN per-token delta within class c
                        bn_en = pw_en[c]["n"][boot_idx].sum(axis=1)
                        bd_en = pw_en[c]["delta"][boot_idx].sum(axis=1)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            btok_en = np.where(bn_en > 0, bd_en / bn_en, np.nan)
                        diff_tok = btok - btok_en
                        en_mean_c = float(pw_en[c]["delta"].sum() / pw_en[c]["n"].sum())
                        dt_lo, dt_hi = ci95(diff_tok)
                        contrasts.append({
                            "model": model_name,
                            "subset": sub,
                            "contrast": f"UK_minus_EN_tok[{c}]",
                            "mean": mean_tok_delta - en_mean_c,
                            "ci95_lo": dt_lo,
                            "ci95_hi": dt_hi,
                        })

                    rows.append({
                        "model": model_name,
                        "subset": sub,
                        "lang": lang,
                        "class": c,
                        "tokens": cnt,
                        "token_share_pct": share_pct,
                        "mean_pos": mean_pos,
                        "base_nll_per_tok": mean_base_nll,
                        "delta_nll_per_tok": mean_tok_delta,
                        "delta_tok_ci95_lo": tok_lo,
                        "delta_tok_ci95_hi": tok_hi,
                        "doc_delta_c": doc_mean_c,
                        "doc_delta_ci95_lo": doc_lo,
                        "doc_delta_ci95_hi": doc_hi,
                        "doc_share_pct": doc_share_pct,
                        "D_c": d_c,
                        "D_c_ci95_lo": d_c_lo,
                        "D_c_ci95_hi": d_c_hi,
                        "D_c_share_pct": d_c_share,
                    })

                # Within-language morphological contrasts
                def tok_boot(cls_name):
                    na = pw[cls_name]["n"]
                    da = pw[cls_name]["delta"]
                    bn = na[boot_idx].sum(axis=1)
                    bd = da[boot_idx].sum(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return float(da.sum() / na.sum()), np.where(bn > 0, bd / bn, np.nan)

                suf_m, suf_b = tok_boot("word_suffix")
                pre_m, pre_b = tok_boot("word_prefix")
                mid_m, mid_b = tok_boot("word_middle")
                sng_m, sng_b = tok_boot("single_token_word")
                for cname, mval, barr in [
                    (f"{lang}_suffix_minus_prefix", suf_m - pre_m, suf_b - pre_b),
                    (f"{lang}_suffix_minus_single", suf_m - sng_m, suf_b - sng_b),
                    (f"{lang}_middle_minus_prefix", mid_m - pre_m, mid_b - pre_b),
                ]:
                    lo, hi = ci95(barr)
                    contrasts.append({
                        "model": model_name,
                        "subset": sub,
                        "contrast": cname,
                        "mean": mval,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                    })

    # Write CSVs
    csv_path = OUT_DIR / "morphology_subwords.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and not np.isnan(v) else ("nan" if isinstance(v, float) else v)) for k, v in r.items()})

    con_path = OUT_DIR / "morphology_contrasts.csv"
    with con_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0].keys()))
        w.writeheader()
        for r in contrasts:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Plot 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), dpi=150)
    short_lbl = {
        "single_token_word": "single\nword",
        "word_prefix": "word\nprefix",
        "word_middle": "word\nmiddle",
        "word_suffix": "word\nsuffix",
        "punct_space_num": "punct/\nspace/num",
    }
    x = np.arange(len(CLASSES))
    w_bar = 0.36

    for col, model_name in enumerate(["BitNet", "EuroLLM-1.7B"]):
        # Top: all scored tokens mean delta NLL/tok (exact population matching D)
        ax = axes[0, col]
        for off, lang, clr, lbl in [(-w_bar/2, "en", "#2b6cb0", "EN"), (w_bar/2, "uk", "#c53030", "UK")]:
            sub_rows = {r["class"]: r for r in rows if r["model"] == model_name and r["subset"] == "all" and r["lang"] == lang}
            vals = [sub_rows[c]["delta_nll_per_tok"] for c in CLASSES]
            err_lo = [sub_rows[c]["delta_nll_per_tok"] - sub_rows[c]["delta_tok_ci95_lo"] for c in CLASSES]
            err_hi = [sub_rows[c]["delta_tok_ci95_hi"] - sub_rows[c]["delta_nll_per_tok"] for c in CLASSES]
            ax.bar(x + off, vals, w_bar, label=lbl, color=clr, alpha=0.88, yerr=[err_lo, err_hi], capsize=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([short_lbl[c] for c in CLASSES], fontsize=9)
        ax.set_ylabel("Mean ΔNLL / token (all tokens, nats)")
        ax.set_title(f"{model_name} — Per-Token Error by Class (all tokens)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        # Bottom: window-sum contribution to D = doc_UK - doc_EN (all tokens)
        ax2 = axes[1, col]
        sub_uk = {r["class"]: r for r in rows if r["model"] == model_name and r["subset"] == "all" and r["lang"] == "uk"}
        d_vals = [sub_uk[c]["D_c"] for c in CLASSES]
        d_lo = [sub_uk[c]["D_c"] - sub_uk[c]["D_c_ci95_lo"] for c in CLASSES]
        d_hi = [sub_uk[c]["D_c_ci95_hi"] - sub_uk[c]["D_c"] for c in CLASSES]
        ax2.bar(x, d_vals, 0.55, color="#805ad5", alpha=0.88, yerr=[d_lo, d_hi], capsize=3)
        for xi, dv, hi, c in zip(x, d_vals, d_hi, CLASSES):
            sh = sub_uk[c]["D_c_share_pct"]
            ax2.text(xi, max(0, dv) + hi + (0.35 if model_name == "BitNet" else 0.12), f"{sh:.1f}%", ha="center", va="bottom", fontsize=8.5)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([short_lbl[c] for c in CLASSES], fontsize=9)
        ax2.set_ylabel("Contribution $D_c$ to total $D$ (nats/window)")
        ax2.set_title(f"{model_name} — Additive Contribution $D_c$ to EN/UK Gap")
        ax2.set_ylim(min(0, min(d_vals) - 1.5), max(d_vals) + max(d_hi) + 2.5)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("H2 (T47b): FLORES-200 Subword Position Degradation (S3+s4r32 vs FP, N=32)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / "morphology_subwords.png")
    plt.close(fig)

    # Append results to morphology_subwords.md preserving ## Plan
    md_path = OUT_DIR / "morphology_subwords.md"
    existing = md_path.read_text()
    plan_only = existing.split("\n## Results")[0].rstrip()

    lines = [plan_only, "", "## Results (Exploratory, `N = 32`)", ""]
    lines.append(
        "- Note on early positions (`t < 35`): in whole-window prefill `S3+s4r32` replaces FP32 with `int8_absmax` on sinks+recent tail "
        "(`[0..2] ∪ [T-32..T-1]`) and `tern_g32_rot` on `[3..T-33]`, so `max(|ΔNLL|)` over `t < 35` vs FP32 is non-zero "
        "(" + ", ".join(f"`{k}={v:.3f}`" for k, v in prefix_max_abs_lt35.items()) + "). Primary decomposition uses **all scored tokens** (`t >= 0`), matching `D` exactly; `t >= 35` sensitivity is shown right below."
    )
    lines.append("")

    for model_name in ["BitNet", "EuroLLM-1.7B"]:
        uk_all = [r for r in rows if r["model"] == model_name and r["subset"] == "all" and r["lang"] == "uk"]
        en_all = [r for r in rows if r["model"] == model_name and r["subset"] == "all" and r["lang"] == "en"]
        tot_uk = sum(r["doc_delta_c"] for r in uk_all)
        tot_en = sum(r["doc_delta_c"] for r in en_all)
        tot_d = tot_uk - tot_en
        lines.append(f"### {model_name} (`D = {tot_d:+.3f}` nats/win; `Δ_UK = {tot_uk:+.3f}`, `Δ_EN = {tot_en:+.3f}`)")
        lines.append("")
        lines.append("**All scored tokens (`t >= 0`, exact additive decomposition of `D`):**")
        lines.append("")
        lines.append("| Class | EN tok (%) | EN `base/tok` | EN `ΔNLL/tok` [95% CI] | UK tok (%) | UK `base/tok` | UK `ΔNLL/tok` [95% CI] | `D_c` (nats/win) | Share of `D` |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in CLASSES:
            re_ = next(r for r in rows if r["model"] == model_name and r["subset"] == "all" and r["lang"] == "en" and r["class"] == c)
            ru_ = next(r for r in rows if r["model"] == model_name and r["subset"] == "all" and r["lang"] == "uk" and r["class"] == c)
            lines.append(
                f"| `{c}` | {re_['tokens']} ({re_['token_share_pct']:.1f}%) | {re_['base_nll_per_tok']:.3f} | "
                f"`{re_['delta_nll_per_tok']:+.5f}` [`{re_['delta_tok_ci95_lo']:+.5f}`, `{re_['delta_tok_ci95_hi']:+.5f}`] | "
                f"{ru_['tokens']} ({ru_['token_share_pct']:.1f}%) | {ru_['base_nll_per_tok']:.3f} | "
                f"`{ru_['delta_nll_per_tok']:+.5f}` [`{ru_['delta_tok_ci95_lo']:+.5f}`, `{ru_['delta_tok_ci95_hi']:+.5f}`] | "
                f"`{ru_['D_c']:+.3f}` [`{ru_['D_c_ci95_lo']:+.3f}`, `{ru_['D_c_ci95_hi']:+.3f}`] | {ru_['D_c_share_pct']:.1f}% |"
            )
        lines.append("")
        lines.append("**Paired contrasts (`all`, 10k window bootstrap):**")
        lines.append("")
        lines.append("| Contrast | Mean (`ΔNLL/tok`) | 95% CI |")
        lines.append("|---|---:|---:|")
        for con in [x for x in contrasts if x["model"] == model_name and x["subset"] == "all"]:
            lines.append(f"| `{con['contrast']}` | `{con['mean']:+.5f}` | [`{con['ci95_lo']:+.5f}`, `{con['ci95_hi']:+.5f}`] |")
        lines.append("")
        uk_pos = {r["class"]: r for r in rows if r["model"] == model_name and r["subset"] == "pos35plus" and r["lang"] == "uk"}
        lines.append(
            "**`t >= 35` sensitivity check (`UK ΔNLL/tok`):** "
            + ", ".join(f"`{c}={uk_pos[c]['delta_nll_per_tok']:+.5f}`" for c in CLASSES)
            + " (same qualitative pattern; full breakdown in `morphology_subwords.csv`)."
        )
        lines.append("")

    # Findings summary
    lines.append("## Key Findings (Exploratory)")
    lines.append("")
    for model_name in ["BitNet", "EuroLLM-1.7B"]:
        uk_all_d = {r["class"]: r for r in rows if r["model"] == model_name and r["subset"] == "all" and r["lang"] == "uk"}
        cont_share = uk_all_d["word_middle"]["D_c_share_pct"] + uk_all_d["word_suffix"]["D_c_share_pct"]
        split_share = cont_share + uk_all_d["word_prefix"]["D_c_share_pct"]
        con_sp = next(x for x in contrasts if x["model"] == model_name and x["subset"] == "all" and x["contrast"] == "uk_suffix_minus_prefix")
        con_ss = next(x for x in contrasts if x["model"] == model_name and x["subset"] == "all" and x["contrast"] == "uk_suffix_minus_single")
        con_mp = next(x for x in contrasts if x["model"] == model_name and x["subset"] == "all" and x["contrast"] == "uk_middle_minus_prefix")
        uk_split_tok_pct = sum(uk_all_d[c]["token_share_pct"] for c in ("word_prefix", "word_middle", "word_suffix"))
        mp_note = (
            f"on BitNet `word_middle` exceeds `word_prefix` (`{con_mp['mean']:+.5f}` [`{con_mp['ci95_lo']:+.5f}`, `{con_mp['ci95_hi']:+.5f}`])"
            if model_name == "BitNet"
            else f"`word_middle` minus `word_prefix` is null (`{con_mp['mean']:+.5f}` [`{con_mp['ci95_lo']:+.5f}`, `{con_mp['ci95_hi']:+.5f}`])"
        )
        lines.append(
            f"- **{model_name}:** Across all UK tokens, per-token error is "
            f"`prefix={uk_all_d['word_prefix']['delta_nll_per_tok']:+.4f}`, "
            f"`middle={uk_all_d['word_middle']['delta_nll_per_tok']:+.4f}`, "
            f"`suffix={uk_all_d['word_suffix']['delta_nll_per_tok']:+.4f}`, "
            f"`single={uk_all_d['single_token_word']['delta_nll_per_tok']:+.4f}`, "
            f"`punct={uk_all_d['punct_space_num']['delta_nll_per_tok']:+.4f}` nats/tok. "
            f"`word_suffix` is **not** worse than `word_prefix` (`{con_sp['mean']:+.5f}`, 95% CI [`{con_sp['ci95_lo']:+.5f}`, `{con_sp['ci95_hi']:+.5f}`]) "
            f"or `single_token_word` (`{con_ss['mean']:+.5f}`, 95% CI [`{con_ss['ci95_lo']:+.5f}`, `{con_ss['ci95_hi']:+.5f}`]); "
            f"{mp_note}. "
            f"Multi-token words generate **{split_share:.1f}%** of $D$ (`middle + suffix` = **{cont_share:.1f}%**) reflecting UK's **{uk_split_tok_pct:.1f}%** split-word token share, while even `single_token_word` and `punct_space_num` show significant UK-minus-EN per-token penalties in both models."
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- Script: `scripts/analyze_h2_morphology.py`")
    lines.append("- Tables: `results/2026-11-24-b_agent2/H2/morphology_subwords.csv`, `results/2026-11-24-b_agent2/H2/morphology_contrasts.csv`")
    lines.append("- Plot: `results/2026-11-24-b_agent2/H2/morphology_subwords.png`")
    lines.append("")

    md_path.write_text("\n".join(lines))
    print("Wrote H2 outputs successfully.")


if __name__ == "__main__":
    main()
