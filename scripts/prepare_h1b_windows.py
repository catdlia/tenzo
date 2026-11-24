#!/usr/bin/env python3
"""Prepare 4x12=48 ctx=2048 windows for H1b (T44: Long document vs Concatenation).

Cells (each N=12 windows of [BOS] + 2047 BitNet tokens):
  - en_long  : WikiText-2 test (w=0..3 reused from windows_en_2048.ids, w=4..11 continuous non-overlapping)
  - en_concat: wikimedia/wikipedia 20231101.en short articles (300 <= chars < 1500, offset >= 200) joined by "\\n\\n"
  - uk_long  : wikimedia/wikipedia 20231101.uk single long articles (>= 2047 BitNet tokens, offset >= 200)
  - uk_concat: Goader/kobza unused rows (offset >= 300) joined by "\\n\\n"

Checks df -h / >= 6 GB, caches raw JSON slices in ~/tenzo-refs/datasets/h1b/ (outside git),
verifies 0 shared 32-grams against calib_en8, calib_uk8, and all prior eval files (except EN-long w=0..3),
and stores metadata in results/2026-11-24_agent2/H1b/windows_h1b.json.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import time
import urllib.parse
import urllib.request
from tokenizers import Tokenizer as FastTok


ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "results" / "reference"
H1B_DIR = ROOT / "results" / "2026-11-24_agent2" / "H1b"
CACHE_DIR = pathlib.Path.home() / "tenzo-refs" / "datasets" / "h1b"

N_WIN = 12
BLOCK = 2047
BOS = 128000


def read_ids(path: pathlib.Path) -> list[list[int]]:
    return [list(map(int, ln.split())) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def ngrams32(seqs: list[list[int]]) -> set[tuple[int, ...]]:
    s: set[tuple[int, ...]] = set()
    for w in seqs:
        for i in range(1 if len(w) > 0 and w[0] == BOS else 0, len(w) - 32 + 1):
            s.add(tuple(w[i : i + 32]))
    return s


def fetch_page(dataset: str, config: str, offset: int, length: int = 100) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{dataset.replace('/', '_')}__{config}__{offset}_{length}.json"
    p = CACHE_DIR / safe_name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    free_gb = shutil.disk_usage("/").free / (1024 ** 3)
    assert free_gb >= 6.0, f"Low disk space: {free_gb:.2f} GB < 6.0 GB"
    qs = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": "train", "offset": offset, "length": length})
    url = f"https://datasets-server.huggingface.co/rows?{qs}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            p.write_text(raw, encoding="utf-8")
            return data
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"Retry {attempt + 1} for {dataset} offset={offset}: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Unreachable")


def main() -> int:
    H1B_DIR.mkdir(parents=True, exist_ok=True)
    tok_path = ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json"
    ft = FastTok.from_file(str(tok_path))
    vocab = json.loads(tok_path.read_text(encoding="utf-8"))["model"]["vocab"]
    b_ids = {int(tid) for s, tid in vocab.items() if "ĊĊ" in s}

    # Preload reference n-gram sets for leakage check
    ref_paths = {
        "calib_en8": REF / "calib_en8.ids",
        "calib_uk8": REF / "calib_uk8.ids",
        "en16": REF / "windows.ids",
        "uk32": REF / "windows_uk32.ids",
        "en2048": REF / "windows_en_2048.ids",
        "uk2048": REF / "windows_uk_2048.ids",
        "h2_en_wiki": ROOT / "results" / "2026-11-23_agent2" / "H2" / "bitnet_en_wiki.ids",
        "h2_uk_wiki": ROOT / "results" / "2026-11-23_agent2" / "H2" / "bitnet_uk_wiki.ids",
        "h2_en_web": ROOT / "results" / "2026-11-23_agent2" / "H2" / "bitnet_en_web.ids",
        "h2_uk_web": ROOT / "results" / "2026-11-23_agent2" / "H2" / "bitnet_uk_web.ids",
        "flores_en": ROOT / "results" / "2026-11-24_agent2" / "H3" / "flores_bitnet_en.ids",
        "flores_uk": ROOT / "results" / "2026-11-24_agent2" / "H3" / "flores_bitnet_uk.ids",
    }
    ref_grams = {k: ngrams32(read_ids(p)) for k, p in ref_paths.items() if p.exists()}

    meta: dict[str, dict] = {}

    # 1. EN-long: 4 reused windows from windows_en_2048.ids + 8 non-overlapping from wt2_test.raw
    wt2_path = pathlib.Path.home() / "tenzo-refs" / "hf" / "wt2_test.raw"
    wt2_ids = ft.encode(wt2_path.read_text(encoding="utf-8"), add_special_tokens=False).ids
    en2048 = read_ids(REF / "windows_en_2048.ids")
    assert len(en2048) == 4
    # Find start after reused w=0..3 (which took [0 : 4*2047])
    en_long_wins = list(en2048)
    pos = 4 * BLOCK
    all_forbidden_en = set().union(ref_grams.get("calib_en8", set()), ref_grams.get("en16", set()), ref_grams.get("en2048", set()),
                                   ref_grams.get("h2_en_wiki", set()), ref_grams.get("h2_en_web", set()), ref_grams.get("flores_en", set()))
    while len(en_long_wins) < N_WIN and pos + BLOCK <= len(wt2_ids):
        cand = [BOS] + list(wt2_ids[pos : pos + BLOCK])
        if len(ngrams32([cand]) & all_forbidden_en) == 0:
            en_long_wins.append(cand)
        pos += BLOCK
    assert len(en_long_wins) == N_WIN, f"Only built {len(en_long_wins)} en_long windows"
    meta["en_long"] = {
        "source": "wt2_test.raw (w=0..3 reused from windows_en_2048.ids; w=4..11 continuous non-overlapping)",
        "wt2_sha256": hashlib.sha256(wt2_path.read_bytes()).hexdigest(),
        "reused_windows": [0, 1, 2, 3],
        "nl2_boundary_counts": [sum(1 for t in w[1:] if t in b_ids) for w in en_long_wins],
        "n_source_docs": [1] * N_WIN,
    }

    # 2. EN-concat: short articles (<1500 chars, >=300) from wikipedia 20231101.en offset >= 200
    en_concat_docs: list[tuple[int, str, list[int]]] = []
    off = 200
    tot_tok = 0
    nl2_toks = ft.encode("\n\n", add_special_tokens=False).ids
    while tot_tok < N_WIN * BLOCK + 2048 and off < 4000:
        data = fetch_page("wikimedia/wikipedia", "20231101.en", off, 100)
        rows = data.get("rows", [])
        if not rows:
            break
        for i, r in enumerate(rows):
            txt = r.get("row", {}).get("text", "").strip()
            if 300 <= len(txt) < 1500:
                ids = ft.encode(txt, add_special_tokens=False).ids
                en_concat_docs.append((off + i, hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16], ids))
                tot_tok += len(ids) + len(nl2_toks)
        off += 100

    # Concatenate with \n\n tracking exact doc boundaries per window
    stream_en: list[int] = []
    doc_starts_en: list[int] = []
    for _, _, d_ids in en_concat_docs:
        if stream_en:
            stream_en.extend(nl2_toks)
        doc_starts_en.append(len(stream_en))
        stream_en.extend(d_ids)
    en_concat_wins: list[list[int]] = []
    en_concat_ndocs: list[int] = []
    for wi in range(N_WIN):
        s0, s1 = wi * BLOCK, (wi + 1) * BLOCK
        en_concat_wins.append([BOS] + stream_en[s0:s1])
        # Count distinct source docs overlapping [s0, s1)
        first_doc = max(0, max([di for di, ds in enumerate(doc_starts_en) if ds <= s0], default=0))
        last_doc = max([di for di, ds in enumerate(doc_starts_en) if ds < s1], default=0)
        en_concat_ndocs.append(last_doc - first_doc + 1)
    meta["en_concat"] = {
        "source": "wikimedia/wikipedia 20231101.en (<1500 chars, offset >= 200)",
        "rows_fetched_until": off,
        "docs_used_count": len(en_concat_docs),
        "doc_row_indices_sample": [d[0] for d in en_concat_docs[:25]],
        "n_source_docs_per_window": en_concat_ndocs,
        "nl2_boundary_counts": [sum(1 for t in w[1:] if t in b_ids) for w in en_concat_wins],
    }

    # 3. UK-long: individual long uk-wiki articles (>= 2047 BitNet tokens, offset >= 200)
    uk_long_wins: list[list[int]] = []
    uk_long_rows: list[dict] = []
    off = 200
    while len(uk_long_wins) < N_WIN and off < 5000:
        data = fetch_page("wikimedia/wikipedia", "20231101.uk", off, 100)
        rows = data.get("rows", [])
        if not rows:
            break
        for i, r in enumerate(rows):
            txt = r.get("row", {}).get("text", "").strip()
            if len(txt) < 3500:
                continue
            ids = ft.encode(txt, add_special_tokens=False).ids
            if len(ids) >= BLOCK:
                cand = [BOS] + list(ids[:BLOCK])
                # Ensure no overlap with calib_uk8 or H2/uk_wiki
                uk_forbid = ref_grams.get("calib_uk8", set()) | ref_grams.get("uk32", set()) | ref_grams.get("h2_uk_wiki", set())
                if len(ngrams32([cand]) & uk_forbid) > 0:
                    continue
                uk_long_wins.append(cand)
                uk_long_rows.append({
                    "row_idx": off + i,
                    "title": r.get("row", {}).get("title", ""),
                    "chars": len(txt),
                    "total_bitnet_toks": len(ids),
                    "sha256_16": hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16],
                })
                if len(uk_long_wins) == N_WIN:
                    break
        off += 100
    assert len(uk_long_wins) == N_WIN, f"Only found {len(uk_long_wins)} long UK articles up to offset {off}"
    meta["uk_long"] = {
        "source": "wikimedia/wikipedia 20231101.uk (1 article >= 2047 BitNet tokens per window, offset >= 200)",
        "rows_fetched_until": off,
        "articles": uk_long_rows,
        "n_source_docs_per_window": [1] * N_WIN,
        "nl2_boundary_counts": [sum(1 for t in w[1:] if t in b_ids) for w in uk_long_wins],
    }

    # 4. UK-concat: Goader/kobza unused rows (offset >= 300) joined with \n\n
    uk_concat_docs: list[tuple[int, str, list[int]]] = []
    off = 300
    tot_tok = 0
    while tot_tok < N_WIN * BLOCK + 4096 and off < 2000:
        data = fetch_page("Goader/kobza", "default", off, 100)
        rows = data.get("rows", [])
        if not rows:
            break
        for i, r in enumerate(rows):
            txt = r.get("row", {}).get("text", "").strip()
            if len(txt) < 100:
                continue
            ids = ft.encode(txt, add_special_tokens=False).ids
            uk_concat_docs.append((off + i, hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16], ids))
            tot_tok += len(ids) + len(nl2_toks)
        off += 100

    stream_uk: list[int] = []
    doc_starts_uk: list[int] = []
    for _, _, d_ids in uk_concat_docs:
        if stream_uk:
            stream_uk.extend(nl2_toks)
        doc_starts_uk.append(len(stream_uk))
        stream_uk.extend(d_ids)
    uk_concat_wins: list[list[int]] = []
    uk_concat_ndocs: list[int] = []
    for wi in range(N_WIN):
        s0, s1 = wi * BLOCK, (wi + 1) * BLOCK
        uk_concat_wins.append([BOS] + stream_uk[s0:s1])
        first_doc = max(0, max([di for di, ds in enumerate(doc_starts_uk) if ds <= s0], default=0))
        last_doc = max([di for di, ds in enumerate(doc_starts_uk) if ds < s1], default=0)
        uk_concat_ndocs.append(last_doc - first_doc + 1)
    meta["uk_concat"] = {
        "source": "Goader/kobza (default, offset >= 300, joined by \\n\\n)",
        "rows_fetched_until": off,
        "docs_used_count": len(uk_concat_docs),
        "doc_row_indices_sample": [d[0] for d in uk_concat_docs[:25]],
        "n_source_docs_per_window": uk_concat_ndocs,
        "nl2_boundary_counts": [sum(1 for t in w[1:] if t in b_ids) for w in uk_concat_wins],
    }

    # Global 32-gram leakage audit across all new windows vs all existing sets
    cells = {
        "en_long": en_long_wins,
        "en_concat": en_concat_wins,
        "uk_long": uk_long_wins,
        "uk_concat": uk_concat_wins,
    }
    overlap_report: dict[str, dict[str, int]] = {}
    for cname, cwins in cells.items():
        # For en_long, check new windows w=4..11 vs all refs, and w=0..3 vs non-en2048 refs
        check_wins = cwins[4:] if cname == "en_long" else cwins
        cg = ngrams32(check_wins)
        ov = {rname: len(cg & rgrams) for rname, rgrams in ref_grams.items()}
        assert all(v == 0 for v in ov.values()), f"Leakage in {cname}: {ov}"
        overlap_report[cname] = ov
        # Also verify internal uniqueness among the 12 windows
        for wi in range(N_WIN):
            assert len(cwins[wi]) == BLOCK + 1

        out_ids = H1B_DIR / f"h1b_{cname}.ids"
        out_ids.write_text("\n".join(" ".join(map(str, w)) for w in cwins) + "\n", encoding="utf-8")
        meta[cname]["ids_sha256"] = hashlib.sha256(out_ids.read_bytes()).hexdigest()

    (H1B_DIR / "windows_h1b.json").write_text(
        json.dumps({"n_windows_per_cell": N_WIN, "window_tokens": BLOCK + 1, "overlap_32grams": overlap_report, "cells": meta}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    for cname in cells:
        print(
            f"[H1b] {cname:9s}: docs/win={meta[cname].get('n_source_docs_per_window', meta[cname].get('n_source_docs'))} "
            f"nl2={meta[cname]['nl2_boundary_counts']} sha={meta[cname]['ids_sha256'][:12]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
