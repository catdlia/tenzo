#!/usr/bin/env python3
"""H4 cheap pre-check (<1 min): Download ONLY SmolLM2 tokenizer.json and compute phi_UK on FLORES-32 and NTREX-24."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "2026-11-26_agent2" / "H4"
SMOLLM_DIR = pathlib.Path("~/tenzo-refs/hf/smollm2-1.7b").expanduser()
FLORES_META = ROOT / "results" / "2026-11-24_agent2" / "H3" / "flores_windows_meta.json"
FLORES_TEXTS = pathlib.Path("~/tenzo-refs/datasets/flores200/windows_flores_texts.json").expanduser()
NTREX_META = ROOT / "results" / "2026-11-24-b_agent2" / "H4" / "ntrex_meta.json"
NTREX_TEXTS = pathlib.Path("~/tenzo-refs/datasets/ntrex128/ntrex_windows_texts.json").expanduser()


def ensure_tokenizer() -> pathlib.Path:
    free_gb = shutil.disk_usage("/").free / (1024 ** 3)
    assert free_gb >= 6.0, f"Low disk space on /: {free_gb:.2f} GB < 6.0 GB"
    SMOLLM_DIR.mkdir(parents=True, exist_ok=True)
    tok_p = SMOLLM_DIR / "tokenizer.json"
    cfg_p = SMOLLM_DIR / "tokenizer_config.json"
    if not (tok_p.exists() and cfg_p.exists()):
        os.environ["HF_HUB_OFFLINE"] = "0"
        from huggingface_hub import hf_hub_download

        for fname in ("tokenizer.json", "tokenizer_config.json"):
            hf_hub_download(repo_id="HuggingFaceTB/SmolLM2-1.7B", filename=fname, local_dir=str(SMOLLM_DIR))
    return tok_p


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_p = ensure_tokenizer()
    ft = Tokenizer.from_file(str(tok_p))
    free_gb = shutil.disk_usage("/").free / (1024 ** 3)

    f_meta = json.loads(FLORES_META.read_text(encoding="utf-8"))
    f_texts = json.loads(FLORES_TEXTS.read_text(encoding="utf-8"))["windows"]
    n_meta = json.loads(NTREX_META.read_text(encoding="utf-8"))
    n_texts = json.loads(NTREX_TEXTS.read_text(encoding="utf-8"))["windows"]
    assert len(f_texts) == len(f_meta["windows"]) == 32
    assert len(n_texts) == len(n_meta["windows"]) == 24

    rows = []

    def add_corpus(corpus: str, n_win: int, meta_wins: list[dict], texts_list: list[dict], models: list[str]):
        for m in models:
            if m == "smollm2":
                en_len = [len(ft.encode(w["en_text"], add_special_tokens=False).ids) for w in texts_list]
                uk_len = [len(ft.encode(w["uk_text"], add_special_tokens=False).ids) for w in texts_list]
            else:
                en_len = [int(w["tokens"][m]["en"]) for w in meta_wins]
                uk_len = [int(w["tokens"][m]["uk"]) for w in meta_wins]
            tot_en, tot_uk = sum(en_len), sum(uk_len)
            gt_1023 = sum(x > 1023 for x in uk_len)
            gt_2047 = sum(x > 2047 for x in uk_len)
            rows.append(
                dict(
                    corpus=corpus,
                    n_windows=n_win,
                    model=m,
                    tot_en=tot_en,
                    tot_uk=tot_uk,
                    mean_en=f"{tot_en / n_win:.2f}",
                    mean_uk=f"{tot_uk / n_win:.2f}",
                    max_en=max(en_len),
                    max_uk=max(uk_len),
                    phi_uk=f"{tot_uk / tot_en:.4f}",
                    uk_gt_1023=gt_1023,
                    uk_gt_2047=gt_2047,
                )
            )

    add_corpus("flores32", 32, f_meta["windows"], f_texts, ["bitnet", "eurollm", "qwen", "smollm2"])
    add_corpus("ntrex24", 24, n_meta["windows"], n_texts, ["bitnet", "eurollm", "smollm2"])

    csv_p = OUT_DIR / "smollm2_tokenizer_fertility.csv"
    with csv_p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    feas = "1.7B feasible (>= 9.5 GB)" if free_gb >= 9.5 else "use 360M (< 9.5 GB free)"
    md = [
        "# H4 Tokenizer Pre-Check: `HuggingFaceTB/SmolLM2-1.7B` Fertility (`2026-11-26`)",
        "",
        "- **Scope (D3 / H4 cheap step):** Downloaded ONLY `tokenizer.json` and `tokenizer_config.json` from `HuggingFaceTB/SmolLM2-1.7B` into `~/tenzo-refs/hf/smollm2-1.7b/` (no model weights downloaded tonight; status `exploratory` tokenizer check only).",
        f"- **Disk check (`/` free):** `{free_gb:.2f} GiB` free -> **{feas}** (rule: `>= 9.5 GB` for `1.7B`, else `SmolLM2-360M`).",
        "",
        "## `φ_UK = tokens_UK / tokens_EN` Across Models (Identical Content)",
        "",
        "| Corpus | `N` | Tokenizer / Model | Mean tok EN | Mean tok UK | Max tok UK | `φ_UK` | `UK > 1023` | `UK > 2047` |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| `{r['corpus']}` | {r['n_windows']} | `{r['model']}` | {r['mean_en']} | {r['mean_uk']} | "
            f"{r['max_uk']} | **{r['phi_uk']}×** | {r['uk_gt_1023']} | {r['uk_gt_2047']} |"
        )
    md += [
        "",
        "## Implications for Next-Night Full `H4` Run",
        "- `FLORES-32` fits cleanly within `ctx = 1024` if `uk_gt_1023 == 0` (otherwise report exact drops/trimming).",
        "- For `NTREX-24` (`ctx = 2048` budget in full `H4`), any UK article exceeding `2047` scored tokens under SmolLM2 (`uk_gt_2047` above) must be truncated cleanly at an aligned EN/UK sentence boundary and reported explicitly so content remains parallel.",
    ]
    (OUT_DIR / "smollm2_tokenizer_fertility.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {csv_p} and {OUT_DIR/'smollm2_tokenizer_fertility.md'} (disk free={free_gb:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
