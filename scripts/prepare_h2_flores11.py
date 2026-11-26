#!/usr/bin/env python3
"""Prepare N=48 11-language parallel FLORES-200 windows for H2 (T50 part 2)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
H3_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H3"
OUT_DIR = ROOT / "results" / "2026-11-26_agent2" / "H2"
DATA_DIR = Path("~/tenzo-refs/datasets/flores200").expanduser()
TEXTS_OUT = DATA_DIR / "flores11_texts.json"

OLD_LANGS = [
    ("en", "eng_Latn"),
    ("de", "deu_Latn"),
    ("pl", "pol_Latn"),
    ("uk", "ukr_Cyrl"),
    ("bg", "bul_Cyrl"),
]
NEW_LANGS = [
    ("fr", "fra_Latn"),
    ("cs", "ces_Latn"),
    ("hu", "hun_Latn"),
    ("el", "ell_Grek"),
    ("es", "spa_Latn"),
    ("fi", "fin_Latn"),
]
ALL_LANGS = OLD_LANGS + NEW_LANGS

MODELS = {
    "bitnet": {
        "tok": ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json",
        "bos": 128000,
        "base": "fp32",
    },
    "eurollm": {
        "tok": Path("~/tenzo-refs/hf/eurollm-1.7b/tokenizer.json").expanduser(),
        "bos": 1,
        "base": "fp",
    },
}
MAX_SCORED_TOKENS = 1023  # ctx <= 1024 including BOS
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_devtest_files() -> None:
    tar_path = DATA_DIR / "flores200_dataset.tar.gz"
    missing = [f for _, f in ALL_LANGS if not (DATA_DIR / f"{f}.devtest").exists()]
    if not missing:
        return
    with tarfile.open(tar_path, "r:gz") as tf:
        by_name = {}
        for m in tf.getmembers():
            for fcode in missing:
                if m.name.endswith(f"/devtest/{fcode}.devtest") or m.name == f"devtest/{fcode}.devtest":
                    by_name[fcode] = m
        for fcode in missing:
            assert fcode in by_name, f"Missing {fcode}.devtest in {tar_path}"
            with tf.extractfile(by_name[fcode]) as src_f:
                assert src_f is not None
                (DATA_DIR / f"{fcode}.devtest").write_bytes(src_f.read())


def ngrams32(seq: list[int]) -> set[tuple[int, ...]]:
    if len(seq) < 32:
        return set()
    return {tuple(seq[i : i + 32]) for i in range(len(seq) - 31)}


def load_ref_32grams() -> dict[str, set[tuple[int, ...]]]:
    ref_sets: dict[str, set[tuple[int, ...]]] = {"bitnet": set(), "eurollm": set()}
    scan_dirs = [
        ROOT / "results" / "reference",
        ROOT / "results" / "2026-11-23_agent2" / "H2",
        ROOT / "results" / "2026-11-24_agent2" / "H1b",
        ROOT / "results" / "2026-11-24_agent2" / "H3",
    ]
    for d in scan_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.ids")):
            key = "eurollm" if "eurollm" in p.name else "bitnet"
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                toks = [int(x) for x in line.split()]
                ref_sets[key] |= ngrams32(toks)
                if len(toks) > 1:
                    ref_sets[key] |= ngrams32(toks[1:])
    return ref_sets


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_devtest_files()

    h5_meta = json.loads((H3_DIR / "flores5_meta.json").read_text())
    win_defs = h5_meta["windows"]
    n_windows = len(win_defs)
    assert n_windows == 48, f"Expected 48 windows, got {n_windows}"

    sents = {
        short: (DATA_DIR / f"{full}.devtest").read_text(encoding="utf-8").splitlines()
        for short, full in ALL_LANGS
    }
    n_sents_total = len(sents["en"])
    assert all(len(v) == n_sents_total for v in sents.values())

    toks = {m: Tokenizer.from_file(str(cfg["tok"])) for m, cfg in MODELS.items()}
    ref_32g = load_ref_32grams()

    ids_out: dict[tuple[str, str], list[list[int]]] = {
        (m, short): [] for m in MODELS for short, _ in ALL_LANGS
    }
    seen_self = {(m, short): set() for m in MODELS for short, _ in ALL_LANGS}
    dropped: dict[tuple[str, str], list[int]] = {
        (m, short): [] for m in MODELS for short, _ in ALL_LANGS
    }
    windows_meta = []
    windows_text = []

    for wdef in win_defs:
        wid = int(wdef["wid"])
        s0, s1 = int(wdef["sent_start"]), int(wdef["sent_end"])
        w_texts = {short: "\n".join(sents[short][s0:s1]) for short, _ in ALL_LANGS}
        w_meta = {
            "wid": wid,
            "sent_start": s0,
            "sent_end": s1,
            "n_sents": int(wdef["n_sents"]),
            "langs": {},
            "tokens": {m: {} for m in MODELS},
        }
        for short, _ in ALL_LANGS:
            txt = w_texts[short]
            w_meta["langs"][short] = {
                "chars": len(txt),
                "bytes": len(txt.encode("utf-8")),
                "words": len(WORD_RE.findall(txt)),
                "sha256_16": hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16],
            }
            for m, cfg in MODELS.items():
                enc = toks[m].encode(txt, add_special_tokens=False)
                full_ids = [cfg["bos"]] + enc.ids
                g32 = ngrams32(enc.ids) | ngrams32(full_ids)
                ov_ref = len(g32 & ref_32g[m])
                ov_self = len(g32 & seen_self[(m, short)])
                if ov_ref != 0 or ov_self != 0:
                    raise RuntimeError(
                        f"32-gram overlap detected wid={wid} model={m} lang={short}: ref={ov_ref} self={ov_self}"
                    )
                seen_self[(m, short)] |= g32
                w_meta["tokens"][m][short] = len(enc.ids)
                if len(enc.ids) > MAX_SCORED_TOKENS:
                    dropped[(m, short)].append(wid)
                else:
                    ids_out[(m, short)].append(full_ids)

        windows_meta.append(w_meta)
        windows_text.append({"wid": wid, "sent_start": s0, "sent_end": s1, **w_texts})

    for (m, short), d_list in dropped.items():
        if d_list:
            raise RuntimeError(f"Unexpected windows > {MAX_SCORED_TOKENS} tokens for {m} {short}: {d_list}")

    ids_sha = {}
    for (m, short), seqs in ids_out.items():
        p = OUT_DIR / f"flores11_{m}_{short}.ids"
        p.write_text("\n".join(" ".join(str(x) for x in s) for s in seqs) + "\n")
        ids_sha[p.name] = sha256_file(p)

    # Hard verification: old 5 languages must be byte-for-byte identical to H3 flores5_*.ids
    old_shorts = {s for s, _ in OLD_LANGS}
    for m in MODELS:
        for short in sorted(old_shorts):
            p_new = OUT_DIR / f"flores11_{m}_{short}.ids"
            p_old = H3_DIR / f"flores5_{m}_{short}.ids"
            assert p_new.read_bytes() == p_old.read_bytes(), f"Mismatch vs flores5 for {m} {short}!"

    # Copy old 5 score files (fp32/fp and S3_s4r32) cleanly and verify sha256
    copied_scores = {}
    for m, cfg in MODELS.items():
        src_dir = H3_DIR / "scores" / m
        dst_dir = OUT_DIR / "scores" / m
        dst_dir.mkdir(parents=True, exist_ok=True)
        for short in sorted(old_shorts):
            for stem in (cfg["base"], "S3_s4r32"):
                fn = f"{stem}_{short}.txt"
                src = src_dir / fn
                dst = dst_dir / fn
                assert src.exists(), f"Missing reused score file: {src}"
                shutil.copy2(src, dst)
                s_src = sha256_file(src)
                s_dst = sha256_file(dst)
                assert s_src == s_dst
                copied_scores[f"{m}/{fn}"] = s_dst

    TEXTS_OUT.write_text(json.dumps({"windows": windows_text}, ensure_ascii=False, indent=2))
    meta = {
        "dataset": "facebook/flores200 (devtest, fixed sent_start..sent_end from flores5_meta.json)",
        "archive_sha256": sha256_file(DATA_DIR / "flores200_dataset.tar.gz"),
        "langs": {short: full for short, full in ALL_LANGS},
        "old_langs": [short for short, _ in OLD_LANGS],
        "new_langs": [short for short, _ in NEW_LANGS],
        "sent_start": int(win_defs[0]["sent_start"]),
        "sent_end": int(win_defs[-1]["sent_end"]),
        "n_windows": n_windows,
        "max_scored_tokens": MAX_SCORED_TOKENS,
        "dropped_windows": {f"{m}/{s}": d for (m, s), d in dropped.items() if d},
        "reused_old5_ids_byte_identical": True,
        "overlap_32grams_vs_calib_and_prior_eval": 0,
        "overlap_32grams_self": 0,
        "ids_sha256": ids_sha,
        "copied_scores_sha256": copied_scores,
        "summary_stats": {
            m: {
                short: {
                    "total_scored_tokens": sum(w["tokens"][m][short] for w in windows_meta),
                    "mean_scored_tokens": sum(w["tokens"][m][short] for w in windows_meta) / n_windows,
                    "min_scored_tokens": min(w["tokens"][m][short] for w in windows_meta),
                    "max_scored_tokens": max(w["tokens"][m][short] for w in windows_meta),
                }
                for short, _ in ALL_LANGS
            }
            for m in MODELS
        },
        "windows": windows_meta,
    }
    (OUT_DIR / "flores11_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Prepared {n_windows} parallel 11-lang windows (0 dropped, 0 overlap, 10/10 old5 .ids byte-identical, 20 reused score files verified).")
    for m in MODELS:
        stats = {s: f"{meta['summary_stats'][m][s]['mean_scored_tokens']:.1f}(max={meta['summary_stats'][m][s]['max_scored_tokens']})" for s, _ in ALL_LANGS}
        print(m, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
