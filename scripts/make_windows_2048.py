#!/usr/bin/env python3
"""Build 4 EN + 4 UK 2048-token evaluation windows for H3 (ctx 2048).

Uses ~/tenzo-refs/hf/wt2_test.raw (EN) and results/reference/kobza_first_rows.json (UK),
checking zero 32-gram overlap against calib_en8.ids and calib_uk8.ids.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "results" / "reference"
N_WIN = 4
BLOCK = 2047  # [BOS] + 2047 text tokens = 2048 total tokens


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
    tok = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T", local_files_only=True)
    bos = tok.bos_token_id

    wt2_path = pathlib.Path.home() / "tenzo-refs" / "hf" / "wt2_test.raw"
    wt2_txt = wt2_path.read_text(encoding="utf-8")
    en_ids = tok.encode(wt2_txt, add_special_tokens=False)

    kobza = json.loads((REF / "kobza_first_rows.json").read_text(encoding="utf-8"))["rows"]
    uk_txt = "\n\n".join(r["row"]["text"] for r in kobza)
    uk_ids = tok.encode(uk_txt, add_special_tokens=False)

    assert len(en_ids) >= N_WIN * BLOCK, len(en_ids)
    assert len(uk_ids) >= N_WIN * BLOCK, len(uk_ids)

    en_wins = [[bos] + en_ids[i * BLOCK : (i + 1) * BLOCK] for i in range(N_WIN)]
    uk_wins = [[bos] + uk_ids[i * BLOCK : (i + 1) * BLOCK] for i in range(N_WIN)]

    cal_en = ngrams32(read_ids(REF / "calib_en8.ids"))
    cal_uk = ngrams32(read_ids(REF / "calib_uk8.ids"))

    ov = {
        "en_vs_cal_en": len(ngrams32(en_wins) & cal_en),
        "en_vs_cal_uk": len(ngrams32(en_wins) & cal_uk),
        "uk_vs_cal_en": len(ngrams32(uk_wins) & cal_en),
        "uk_vs_cal_uk": len(ngrams32(uk_wins) & cal_uk),
    }
    assert all(v == 0 for v in ov.values()), f"Overlap detected: {ov}"

    for name, wins, src_meta in [
        ("windows_en_2048", en_wins, {"source": "wt2_test.raw", "sha256": hashlib.sha256(wt2_path.read_bytes()).hexdigest()}),
        ("windows_uk_2048", uk_wins, {"source": "kobza_first_rows.json", "sha256": hashlib.sha256((REF / "kobza_first_rows.json").read_bytes()).hexdigest()}),
    ]:
        ids_p = REF / f"{name}.ids"
        ids_p.write_text("\n".join(" ".join(map(str, w)) for w in wins) + "\n", encoding="utf-8")
        meta = {
            **src_meta,
            "n_windows": N_WIN,
            "window_len": BLOCK + 1,
            "scored_tokens_per_window": BLOCK,
            "bos": bos,
            "overlap_32grams_with_calib": ov,
            "ids_sha256": hashlib.sha256(ids_p.read_bytes()).hexdigest(),
        }
        (REF / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {ids_p} ({N_WIN} x {len(wins[0])}, sha256={meta['ids_sha256'][:16]})")


if __name__ == "__main__":
    main()
