#!/usr/bin/env python3
"""Prepare N=24 coherent document-respecting NTREX-128 parallel EN/UK windows (ctx=1024)."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import urllib.request
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
H4_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H4"
DATA_DIR = pathlib.Path("~/tenzo-refs/datasets/ntrex128").expanduser()
TEXTS_OUT = DATA_DIR / "ntrex_windows_texts.json"

URLS = {
    "DOCUMENT_IDS.tsv": "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/DOCUMENT_IDS.tsv",
    "newstest2019-src.eng.txt": "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-src.eng.txt",
    "newstest2019-ref.ukr.txt": "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-ref.ukr.txt",
}
MODELS = {
    "bitnet": {"tok": ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json", "bos": 128000},
    "eurollm": {"tok": pathlib.Path("~/tenzo-refs/hf/eurollm-1.7b/tokenizer.json").expanduser(), "bos": 1},
}
N_WINDOWS = 24
MAX_SCORED_TOKENS = 1023
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        p = DATA_DIR / name
        if not p.exists():
            urllib.request.urlretrieve(url, p)


def ngrams32(seq: list[int]) -> set[tuple[int, ...]]:
    return {tuple(seq[i : i + 32]) for i in range(len(seq) - 31)} if len(seq) >= 32 else set()


def load_ref_32grams() -> dict[str, set[tuple[int, ...]]]:
    refs: dict[str, set[tuple[int, ...]]] = {"bitnet": set(), "eurollm": set()}
    scan_dirs = [
        ROOT / "results" / "reference",
        ROOT / "results" / "2026-11-23_agent2" / "H2",
        ROOT / "results" / "2026-11-24_agent2" / "H1b",
        ROOT / "results" / "2026-11-24_agent2" / "H3",
        ROOT / "results" / "2026-11-24-b_agent2" / "H3",
    ]
    for d in scan_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.ids")):
            key = "eurollm" if "eurollm" in p.name else "bitnet"
            for ln in p.read_text().splitlines():
                if not ln.strip():
                    continue
                t = [int(x) for x in ln.split()]
                refs[key] |= ngrams32(t)
                if len(t) > 1:
                    refs[key] |= ngrams32(t[1:])
    return refs


def main() -> int:
    H4_DIR.mkdir(parents=True, exist_ok=True)
    ensure_data()
    doc_lines = (DATA_DIR / "DOCUMENT_IDS.tsv").read_text(encoding="utf-8").splitlines()
    en_lines = (DATA_DIR / "newstest2019-src.eng.txt").read_text(encoding="utf-8").splitlines()
    uk_lines = (DATA_DIR / "newstest2019-ref.ukr.txt").read_text(encoding="utf-8").splitlines()
    assert len(doc_lines) == len(en_lines) == len(uk_lines) == 1997

    docs: dict[str, list[int]] = {}
    doc_order: list[str] = []
    for i, raw_did in enumerate(doc_lines):
        did = raw_did.split("\t")[0]
        if did not in docs:
            docs[did] = []
            doc_order.append(did)
        docs[did].append(i)

    toks = {m: Tokenizer.from_file(str(cfg["tok"])) for m, cfg in MODELS.items()}
    refs = load_ref_32grams()

    candidates = []
    for order_idx, did in enumerate(doc_order):
        idxs = docs[did]
        buf_en, buf_uk = [], []
        best = None
        for i in idxs:
            cen = "\n".join(buf_en + [en_lines[i]])
            cuk = "\n".join(buf_uk + [uk_lines[i]])
            lens = {
                ("bitnet", "en"): toks["bitnet"].encode(cen, add_special_tokens=False).ids,
                ("bitnet", "uk"): toks["bitnet"].encode(cuk, add_special_tokens=False).ids,
                ("eurollm", "en"): toks["eurollm"].encode(cen, add_special_tokens=False).ids,
                ("eurollm", "uk"): toks["eurollm"].encode(cuk, add_special_tokens=False).ids,
            }
            if max(len(v) for v in lens.values()) <= MAX_SCORED_TOKENS:
                buf_en.append(en_lines[i])
                buf_uk.append(uk_lines[i])
                best = (cen, cuk, lens, len(buf_en))
            else:
                break
        if best is None:
            continue
        cen, cuk, lens, used_sents = best
        min_tok = min(len(v) for v in lens.values())
        candidates.append((min_tok, -order_idx, did, used_sents, len(idxs), cen, cuk, lens))

    candidates.sort(reverse=True)
    chosen = candidates[:N_WINDOWS]
    assert len(chosen) == N_WINDOWS

    ids_out: dict[tuple[str, str], list[list[int]]] = {(m, l): [] for m in MODELS for l in ("en", "uk")}
    seen_self: dict[tuple[str, str], set[tuple[int, ...]]] = {(m, l): set() for m in MODELS for l in ("en", "uk")}
    windows_meta = []
    windows_text = []

    for wid, (_, _, did, used_sents, total_doc_sents, en_t, uk_t, lens) in enumerate(chosen):
        w_meta = {
            "wid": wid,
            "doc_id": did,
            "sents_used": used_sents,
            "doc_total_sents": total_doc_sents,
            "is_full_doc": (used_sents == total_doc_sents),
            "langs": {
                "en": {"chars": len(en_t), "bytes": len(en_t.encode("utf-8")), "words": len(WORD_RE.findall(en_t))},
                "uk": {"chars": len(uk_t), "bytes": len(uk_t.encode("utf-8")), "words": len(WORD_RE.findall(uk_t))},
            },
            "tokens": {m: {} for m in MODELS},
        }
        for m, cfg in MODELS.items():
            for l in ("en", "uk"):
                raw_ids = lens[(m, l)]
                full_ids = [cfg["bos"]] + raw_ids
                g32 = ngrams32(raw_ids) | ngrams32(full_ids)
                assert len(g32 & refs[m]) == 0, f"Overlap vs prior ids at {did} {m} {l}"
                assert len(g32 & seen_self[(m, l)]) == 0, f"Self overlap at {did} {m} {l}"
                seen_self[(m, l)] |= g32
                ids_out[(m, l)].append(full_ids)
                w_meta["tokens"][m][l] = len(raw_ids)
        windows_meta.append(w_meta)
        windows_text.append({"wid": wid, "doc_id": did, "en_text": en_t, "uk_text": uk_t})

    ids_sha = {}
    for (m, l), seqs in ids_out.items():
        p = H4_DIR / f"ntrex_{m}_{l}.ids"
        p.write_text("\n".join(" ".join(str(x) for x in s) for s in seqs) + "\n")
        ids_sha[p.name] = sha256_file(p)

    TEXTS_OUT.write_text(json.dumps({"windows": windows_text}, ensure_ascii=False, indent=2))
    meta = {
        "dataset": "MicrosoftTranslator/NTREX (NTREX-128 newstest2019 eng/ukr)",
        "source_sha256": {k: sha256_file(DATA_DIR / k) for k in URLS},
        "n_windows": N_WINDOWS,
        "max_scored_tokens": MAX_SCORED_TOKENS,
        "bos_ids": {m: cfg["bos"] for m, cfg in MODELS.items()},
        "overlap_32grams_vs_calib_and_prior_eval": 0,
        "overlap_32grams_self": 0,
        "ids_sha256": ids_sha,
        "summary_stats": {
            m: {
                l: {
                    "total_scored_tokens": sum(w["tokens"][m][l] for w in windows_meta),
                    "mean_scored_tokens": sum(w["tokens"][m][l] for w in windows_meta) / N_WINDOWS,
                    "min_scored_tokens": min(w["tokens"][m][l] for w in windows_meta),
                    "max_scored_tokens": max(w["tokens"][m][l] for w in windows_meta),
                }
                for l in ("en", "uk")
            }
            for m in MODELS
        },
        "windows": windows_meta,
    }
    (H4_DIR / "ntrex_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Prepared {N_WINDOWS} single-doc NTREX-128 parallel windows (0 overlap).")
    for m in MODELS:
        print(m, meta["summary_stats"][m])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
