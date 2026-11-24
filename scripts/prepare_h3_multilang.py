#!/usr/bin/env python3
"""Prepare N=48 5-language parallel FLORES-200 windows for H3 (T48) & H3b."""

import hashlib
import json
import re
import tarfile
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H3"
DATA_DIR = Path("~/tenzo-refs/datasets/flores200").expanduser()
TEXTS_OUT = DATA_DIR / "flores5_texts.json"

LANGS = [
    ("en", "eng_Latn"),
    ("uk", "ukr_Cyrl"),
    ("pl", "pol_Latn"),
    ("bg", "bul_Cyrl"),
    ("de", "deu_Latn"),
]
MODELS = {
    "bitnet": {
        "tok": ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json",
        "bos": 128000,
    },
    "eurollm": {
        "tok": Path("~/tenzo-refs/hf/eurollm-1.7b/tokenizer.json").expanduser(),
        "bos": 1,
    },
}
START_SENT = 209
N_WINDOWS = 48
MAX_SCORED_TOKENS = 511
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_devtest_files():
    tar_path = DATA_DIR / "flores200_dataset.tar.gz"
    missing = [f for _, f in LANGS if not (DATA_DIR / f"{f}.devtest").exists()]
    if missing:
        with tarfile.open(tar_path, "r:gz") as tf:
            for fcode in missing:
                member = f"./flores200_dataset/devtest/{fcode}.devtest"
                with tf.extractfile(member) as src_f:
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_devtest_files()
    sents = {
        short: (DATA_DIR / f"{full}.devtest").read_text(encoding="utf-8").splitlines()
        for short, full in LANGS
    }
    n_sents_total = len(sents["en"])
    assert all(len(v) == n_sents_total for v in sents.values())

    toks = {m: Tokenizer.from_file(str(cfg["tok"])) for m, cfg in MODELS.items()}
    ref_32g = load_ref_32grams()

    idx = START_SENT
    windows_meta = []
    windows_text = []
    ids_out: dict[tuple[str, str], list[list[int]]] = {
        (m, short): [] for m in MODELS for short, _ in LANGS
    }
    seen_self = {(m, short): set() for m in MODELS for short, _ in LANGS}

    for wid in range(N_WINDOWS):
        buf = {short: [] for short, _ in LANGS}
        cnt = 0
        while idx + cnt < n_sents_total:
            cand = {short: "\n".join(buf[short] + [sents[short][idx + cnt]]) for short, _ in LANGS}
            ok = True
            for m, t in toks.items():
                for short, _ in LANGS:
                    if len(t.encode(cand[short], add_special_tokens=False).ids) > MAX_SCORED_TOKENS:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                for short, _ in LANGS:
                    buf[short].append(sents[short][idx + cnt])
                cnt += 1
            else:
                break
        if cnt == 0:
            raise RuntimeError(f"Sentence {idx} alone exceeds {MAX_SCORED_TOKENS} tokens")

        sent_start = idx
        sent_end = idx + cnt
        idx = sent_end

        w_texts = {short: "\n".join(buf[short]) for short, _ in LANGS}
        w_meta = {
            "wid": wid,
            "sent_start": sent_start,
            "sent_end": sent_end,
            "n_sents": cnt,
            "langs": {},
            "tokens": {m: {} for m in MODELS},
        }
        for short, _ in LANGS:
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
                        f"32-gram overlap detected at wid={wid} model={m} lang={short}: ref={ov_ref} self={ov_self}"
                    )
                seen_self[(m, short)] |= g32
                ids_out[(m, short)].append(full_ids)
                w_meta["tokens"][m][short] = len(enc.ids)

        windows_meta.append(w_meta)
        windows_text.append({"wid": wid, "sent_start": sent_start, "sent_end": sent_end, **w_texts})

    ids_sha = {}
    for (m, short), seqs in ids_out.items():
        p = OUT_DIR / f"flores5_{m}_{short}.ids"
        p.write_text("\n".join(" ".join(str(x) for x in s) for s in seqs) + "\n")
        ids_sha[p.name] = sha256_file(p)

    TEXTS_OUT.write_text(json.dumps({"windows": windows_text}, ensure_ascii=False, indent=2))
    meta = {
        "dataset": "facebook/flores200 (devtest)",
        "archive_sha256": sha256_file(DATA_DIR / "flores200_dataset.tar.gz"),
        "langs": [full for _, full in LANGS],
        "sent_start": START_SENT,
        "sent_end": idx,
        "n_windows": N_WINDOWS,
        "max_scored_tokens": MAX_SCORED_TOKENS,
        "bos_ids": {m: cfg["bos"] for m, cfg in MODELS.items()},
        "overlap_32grams_vs_calib_and_prior_eval": 0,
        "overlap_32grams_self": 0,
        "ids_sha256": ids_sha,
        "summary_stats": {
            m: {
                short: {
                    "total_scored_tokens": sum(w["tokens"][m][short] for w in windows_meta),
                    "mean_scored_tokens": sum(w["tokens"][m][short] for w in windows_meta) / N_WINDOWS,
                    "min_scored_tokens": min(w["tokens"][m][short] for w in windows_meta),
                    "max_scored_tokens": max(w["tokens"][m][short] for w in windows_meta),
                }
                for short, _ in LANGS
            }
            for m in MODELS
        },
        "windows": windows_meta,
    }
    (OUT_DIR / "flores5_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Prepared {N_WINDOWS} parallel 5-lang windows (sents {START_SENT}..{idx}, 0 overlap).")
    for m in MODELS:
        print(m, {s: f"{meta['summary_stats'][m][s]['mean_scored_tokens']:.1f}" for s, _ in LANGS})


if __name__ == "__main__":
    main()
