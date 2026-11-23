#!/usr/bin/env python3
"""Prepare 2x2 (language x domain) evaluation windows for H2.

Corpora (fetched via datasets-server into results/2026-11-23_agent2/H2/corpora/):
  - en_wiki : wikimedia/wikipedia (20231101.en, rows 0..99)
  - uk_wiki : wikimedia/wikipedia (20231101.uk, rows 0..99)
  - en_web  : HuggingFaceFW/fineweb-edu (sample-10BT, rows 0..99)
  - uk_web  : Goader/kobza (default, rows 200..299; disjoint from rows 0..38 and 100..199)

For each cell, produces 16 fixed-length BitNet windows ([BOS] + 511 BitNet tokens = 512 tokens,
matching golden_bitnet.py nll --batch 8 natively) and stores the decoded UTF-8 text of the 511
scored tokens in windows_2x2.json for cross-tokenizer evaluation (Qwen2.5-1.5B & EuroLLM-1.7B),
following the exact L3 protocol. Checks 0 shared 32-grams against calib_en8.ids, calib_uk8.ids,
windows.ids, and windows_uk32.ids.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from tokenizers import Tokenizer as FastTok  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
H2_DIR = ROOT / "results" / "2026-11-23_agent2" / "H2"
CORP_DIR = H2_DIR / "corpora"
BLOCK = 511


def load_corpus_text(path: pathlib.Path, min_chars: int = 300) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = [
        row.get("row", {}).get("text", "").strip()
        for row in data.get("rows", [])
        if len(row.get("row", {}).get("text", "").strip()) >= min_chars
    ]
    return "\n\n".join(parts)


def ngrams32(seqs: list[list[int]]) -> set[tuple[int, ...]]:
    s: set[tuple[int, ...]] = set()
    for w in seqs:
        start = 1 if len(w) > 0 and w[0] == 128000 else 0
        for i in range(start, len(w) - 32 + 1):
            s.add(tuple(w[i : i + 32]))
    return s


def read_ids(path: pathlib.Path) -> list[list[int]]:
    return [list(map(int, ln.split())) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    tok_bitnet = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T", local_files_only=True)
    tok_qwen = AutoTokenizer.from_pretrained("/home/illia/tenzo-refs/hf/qwen2.5-1.5b-tokenizer", local_files_only=True)
    tok_euro = FastTok.from_file("/home/illia/tenzo-refs/hf/eurollm-1.7b-tokenizer/tokenizer.json")

    cells = {
        "en_wiki": CORP_DIR / "en_wiki_0_99.json",
        "uk_wiki": CORP_DIR / "uk_wiki_0_99.json",
        "en_web": CORP_DIR / "en_web_0_99.json",
        "uk_web": CORP_DIR / "uk_web_200_299.json",
    }

    ref_sets = {
        "calib_en8": ngrams32(read_ids(ROOT / "results" / "reference" / "calib_en8.ids")),
        "calib_uk8": ngrams32(read_ids(ROOT / "results" / "reference" / "calib_uk8.ids")),
        "en16": ngrams32(read_ids(ROOT / "results" / "reference" / "windows.ids")),
        "uk32": ngrams32(read_ids(ROOT / "results" / "reference" / "windows_uk32.ids")),
    }

    out_meta = {"block_tokens": BLOCK, "cells": {}, "overlap_32grams": {}}
    n_win = 16
    for cell, cpath in cells.items():
        text = load_corpus_text(cpath)
        raw_ids = tok_bitnet.encode(text, add_special_tokens=False)
        assert len(raw_ids) >= n_win * BLOCK, f"{cell} only has {len(raw_ids)} tokens"

        bitnet_wins = [
            [tok_bitnet.bos_token_id] + raw_ids[w * BLOCK : (w + 1) * BLOCK]
            for w in range(n_win)
        ]

        g = ngrams32(bitnet_wins)
        overlaps = {rname: len(g & rgrams) for rname, rgrams in ref_sets.items()}
        assert all(v == 0 for v in overlaps.values()), f"Overlap for {cell}: {overlaps}"
        out_meta["overlap_32grams"][cell] = overlaps

        ids_path = H2_DIR / f"bitnet_{cell}.ids"
        ids_path.write_text("\n".join(" ".join(map(str, ids)) for ids in bitnet_wins) + "\n", encoding="utf-8")

        win_records = []
        q_lens, e_lens = [], []
        for wid, ids in enumerate(bitnet_wins):
            scored_ids = ids[1:]
            dec_text = tok_bitnet.decode(scored_ids, clean_up_tokenization_spaces=False)
            q_ids = tok_qwen.encode(dec_text, add_special_tokens=False)
            e_ids = tok_euro.encode(dec_text, add_special_tokens=False).ids
            q_lens.append(len(q_ids))
            e_lens.append(len(e_ids))
            win_records.append(
                {
                    "wid": wid,
                    "n_bytes": len(dec_text.encode("utf-8")),
                    "n_chars": len(dec_text),
                    "n_tok_bitnet": len(scored_ids),
                    "n_tok_qwen": len(q_ids),
                    "n_tok_eurollm": len(e_ids),
                    "text": dec_text,
                }
            )

        out_meta["cells"][cell] = {
            "source_file": str(cpath.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(cpath.read_bytes()).hexdigest(),
            "bitnet_ids_file": str(ids_path.relative_to(ROOT)),
            "n_windows": n_win,
            "bitnet_len_fixed": BLOCK + 1,
            "qwen_tok_min_max": [min(q_lens), max(q_lens)],
            "eurollm_tok_min_max": [min(e_lens), max(e_lens)],
            "windows": win_records,
        }
        print(
            f"{cell:8s}: bitnet={BLOCK+1} (511 scored), "
            f"qwen={min(q_lens)}..{max(q_lens)}, eurollm={min(e_lens)}..{max(e_lens)}, "
            f"overlaps={overlaps}"
        )

    (H2_DIR / "windows_2x2.json").write_text(json.dumps(out_meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Wrote", H2_DIR / "windows_2x2.json")


if __name__ == "__main__":
    main()
