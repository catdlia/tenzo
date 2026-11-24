#!/usr/bin/env python3
"""Prepare N=32 parallel EN/UK windows from FLORES-200 devtest for H3 (T45).

Downloads flores200_dataset.tar.gz outside the repo (~/tenzo-refs/datasets/flores200/),
verifies free disk >= 6 GB, packs consecutive aligned sentences joined by " " so that
all three tokenizers (BitNet, EuroLLM-1.7B, Qwen2.5-1.5B) fit within <= 511 tokens on UK
(and <= 511 on EN), checks 0 shared 32-grams vs all existing calibration & eval windows,
and stores metadata + token IDs (never raw text) under results/2026-11-24_agent2/H3/.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import tarfile
import urllib.request
from tokenizers import Tokenizer as FastTok


ROOT = pathlib.Path(__file__).resolve().parents[1]
H3_DIR = ROOT / "results" / "2026-11-24_agent2" / "H3"
DS_DIR = pathlib.Path.home() / "tenzo-refs" / "datasets" / "flores200"
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def read_ids(path: pathlib.Path) -> list[list[int]]:
    return [list(map(int, ln.split())) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def ngrams32(seqs: list[list[int]]) -> set[tuple[int, ...]]:
    s: set[tuple[int, ...]] = set()
    for w in seqs:
        start = 1 if len(w) > 0 else 0
        for i in range(start, len(w) - 32 + 1):
            s.add(tuple(w[i : i + 32]))
    return s


def load_bos_id(cfg_path: pathlib.Path, default_id: int) -> int:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    val = cfg.get("bos_token_id")
    return int(val) if val is not None else default_id


def download_and_extract() -> tuple[pathlib.Path, str, list[str], list[str]]:
    DS_DIR.mkdir(parents=True, exist_ok=True)
    tgz = DS_DIR / "flores200_dataset.tar.gz"
    if not tgz.exists():
        free_gb = shutil.disk_usage("/").free / (1024 ** 3)
        assert free_gb >= 6.0, f"Insufficient free space on /: {free_gb:.2f} GB < 6.0 GB"
        print(f"Downloading {FLORES_URL} (disk free {free_gb:.2f} GB)...", flush=True)
        urllib.request.urlretrieve(FLORES_URL, tgz)
    sha = hashlib.sha256(tgz.read_bytes()).hexdigest()
    print(f"Archive sha256: {sha} ({tgz.stat().st_size} bytes)", flush=True)

    en_lines, uk_lines = None, None
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("devtest/eng_Latn.devtest"):
                en_lines = tf.extractfile(m).read().decode("utf-8").splitlines()
                (DS_DIR / "eng_Latn.devtest").write_text("\n".join(en_lines) + "\n", encoding="utf-8")
            elif m.name.endswith("devtest/ukr_Cyrl.devtest"):
                uk_lines = tf.extractfile(m).read().decode("utf-8").splitlines()
                (DS_DIR / "ukr_Cyrl.devtest").write_text("\n".join(uk_lines) + "\n", encoding="utf-8")
    assert en_lines is not None and uk_lines is not None, "Could not locate devtest files in tarball"
    assert len(en_lines) == len(uk_lines), (len(en_lines), len(uk_lines))
    return tgz, sha, en_lines, uk_lines


def main() -> int:
    H3_DIR.mkdir(parents=True, exist_ok=True)
    _, archive_sha, en_lines, uk_lines = download_and_extract()
    print(f"Loaded {len(en_lines)} aligned devtest sentence pairs.", flush=True)

    toks = {
        "bitnet": (
            FastTok.from_file(str(ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json")),
            load_bos_id(ROOT / "models" / "bitnet-1.58b-v2" / "config.json", 128000),
        ),
        "eurollm": (
            FastTok.from_file(str(pathlib.Path.home() / "tenzo-refs" / "hf" / "eurollm-1.7b" / "tokenizer.json")),
            load_bos_id(pathlib.Path.home() / "tenzo-refs" / "hf" / "eurollm-1.7b" / "config.json", 1),
        ),
        "qwen": (
            FastTok.from_file(str(pathlib.Path.home() / "tenzo-refs" / "hf" / "qwen2.5-1.5b" / "tokenizer.json")),
            load_bos_id(pathlib.Path.home() / "tenzo-refs" / "hf" / "qwen2.5-1.5b" / "config.json", 151643),
        ),
    }

    n_target = 32
    max_tok = 511
    windows_text: list[dict] = []
    windows_meta: list[dict] = []
    ids_out: dict[tuple[str, str], list[list[int]]] = {(m, l): [] for m in toks for l in ("en", "uk")}

    idx = 0
    for w_idx in range(n_target):
        cur_en: list[str] = []
        cur_uk: list[str] = []
        s_start = idx
        while idx < len(en_lines):
            cand_en = " ".join(cur_en + [en_lines[idx].strip()])
            cand_uk = " ".join(cur_uk + [uk_lines[idx].strip()])
            ok = True
            for _, (ft, _) in toks.items():
                if len(ft.encode(cand_uk, add_special_tokens=False).ids) > max_tok:
                    ok = False
                    break
                if len(ft.encode(cand_en, add_special_tokens=False).ids) > max_tok:
                    ok = False
                    break
            if not ok:
                assert len(cur_uk) > 0, f"Single sentence {idx} exceeded {max_tok} tokens!"
                break
            cur_en.append(en_lines[idx].strip())
            cur_uk.append(uk_lines[idx].strip())
            idx += 1

        en_str = " ".join(cur_en)
        uk_str = " ".join(cur_uk)
        rec_meta = {
            "wid": w_idx,
            "sent_start": s_start,
            "sent_end": idx,
            "n_sents": idx - s_start,
            "en_chars": len(en_str),
            "en_bytes": len(en_str.encode("utf-8")),
            "en_sha256_16": hashlib.sha256(en_str.encode("utf-8")).hexdigest()[:16],
            "uk_chars": len(uk_str),
            "uk_bytes": len(uk_str.encode("utf-8")),
            "uk_sha256_16": hashlib.sha256(uk_str.encode("utf-8")).hexdigest()[:16],
            "tokens": {},
        }
        for m, (ft, bos) in toks.items():
            e_ids = ft.encode(en_str, add_special_tokens=False).ids
            u_ids = ft.encode(uk_str, add_special_tokens=False).ids
            assert len(e_ids) <= max_tok and len(u_ids) <= max_tok
            ids_out[(m, "en")].append([bos] + list(e_ids))
            ids_out[(m, "uk")].append([bos] + list(u_ids))
            rec_meta["tokens"][m] = {"en": len(e_ids), "uk": len(u_ids)}
        windows_meta.append(rec_meta)
        windows_text.append({"wid": w_idx, "sent_start": s_start, "sent_end": idx, "en_text": en_str, "uk_text": uk_str})

    # Verify 0 shared 32-grams against calibration and all prior eval sets (BitNet tokens)
    ref_files = {
        "calib_en8": ROOT / "results" / "reference" / "calib_en8.ids",
        "calib_uk8": ROOT / "results" / "reference" / "calib_uk8.ids",
        "en16": ROOT / "results" / "reference" / "windows.ids",
        "uk32": ROOT / "results" / "reference" / "windows_uk32.ids",
        "en2048": ROOT / "results" / "reference" / "windows_en_2048.ids",
        "uk2048": ROOT / "results" / "reference" / "windows_uk_2048.ids",
        "h2_en_wiki": ROOT / "results" / "2026-11-23_agent2" / "H2" / "bitnet_en_wiki.ids",
        "h2_uk_wiki": ROOT / "results" / "2026-11-23_agent2" / "H2" / "bitnet_uk_wiki.ids",
        "h2_en_web": ROOT / "results" / "2026-11-23_agent2" / "H2" / "bitnet_en_web.ids",
        "h2_uk_web": ROOT / "results" / "2026-11-23_agent2" / "H2" / "bitnet_uk_web.ids",
    }
    en_grams = ngrams32(ids_out[("bitnet", "en")])
    uk_grams = ngrams32(ids_out[("bitnet", "uk")])
    overlap_report: dict[str, dict[str, int]] = {}
    for rname, rpath in ref_files.items():
        if not rpath.exists():
            continue
        rg = ngrams32(read_ids(rpath))
        o_en = len(en_grams & rg)
        o_uk = len(uk_grams & rg)
        assert o_en == 0 and o_uk == 0, f"Overlap with {rname}: en={o_en} uk={o_uk}"
        overlap_report[rname] = {"en": o_en, "uk": o_uk}

    for (m, lang), seqs in ids_out.items():
        p = H3_DIR / f"flores_{m}_{lang}.ids"
        p.write_text("\n".join(" ".join(map(str, s)) for s in seqs) + "\n", encoding="utf-8")

    (DS_DIR / "windows_flores_texts.json").write_text(
        json.dumps({"windows": windows_text}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    meta_out = {
        "dataset": "facebook/flores200",
        "url": FLORES_URL,
        "license": "CC-BY-SA-4.0",
        "archive_sha256": archive_sha,
        "split": "devtest",
        "n_windows": len(windows_meta),
        "sentences_used": idx,
        "total_devtest_sentences": len(en_lines),
        "max_scored_tokens": max_tok,
        "bos_ids": {m: bos for m, (_, bos) in toks.items()},
        "ids_sha256": {
            f"flores_{m}_{l}.ids": hashlib.sha256((H3_DIR / f"flores_{m}_{l}.ids").read_bytes()).hexdigest()
            for m in toks
            for l in ("en", "uk")
        },
        "overlap_32grams_bitnet": overlap_report,
        "windows": windows_meta,
    }
    (H3_DIR / "flores_windows_meta.json").write_text(json.dumps(meta_out, indent=1) + "\n", encoding="utf-8")
    print(
        f"Prepared {len(windows_meta)} paired windows using {idx}/{len(en_lines)} devtest sentences; "
        f"mean BitNet tokens EN={sum(w['tokens']['bitnet']['en'] for w in windows_meta)/32:.1f} "
        f"UK={sum(w['tokens']['bitnet']['uk'] for w in windows_meta)/32:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
