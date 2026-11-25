#!/usr/bin/env python3
"""H3b (Conditional): Run S3+s4c120 across all 5 FLORES-200 languages on BitNet and EuroLLM-1.7B."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
H3_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H3"
H3B_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H3b"
TEXTS_JSON = pathlib.Path("~/tenzo-refs/datasets/flores200/flores5_texts.json").expanduser()
EUROLLM_DIR = pathlib.Path("~/tenzo-refs/hf/eurollm-1.7b").expanduser()
sys.path.insert(0, str(ROOT / "scripts"))
import kv_eval_lib as K


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["bitnet", "eurollm"])
    ap.add_argument("--char-budget", type=int, default=120)
    ap.add_argument("--langs", default="en,uk,pl,bg,de")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    texts = json.loads(TEXTS_JSON.read_text(encoding="utf-8"))["windows"]
    scheme = f"S3+s4c{args.char_budget}"

    tok_json = (
        ROOT / "models" / "bitnet-1.58b-v2" / "tokenizer.json"
        if args.model == "bitnet"
        else EUROLLM_DIR / "tokenizer.json"
    )
    ids_paths = {lang: H3_DIR / f"flores5_{args.model}_{lang}.ids" for lang in langs}
    out_dir = H3B_DIR / "scores" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_by_scheme_lang = {}
    cov_stats = {}
    for lang in langs:
        seqs = K.read_ids(ids_paths[lang])
        offs = K.build_char_offsets(tok_json, seqs, [w[lang] for w in texts])
        masks_by_scheme_lang[(scheme, lang)] = [K.char_window_mask(c, e, args.char_budget, sinks=4) for c, e, _ in offs]
        cov_stats[lang] = K.coverage_stats_char_c(offs, args.char_budget, sinks=4)

    (out_dir / "coverage_stats.json").write_text(json.dumps(cov_stats, indent=2) + "\n")

    if args.model == "bitnet":
        threads = args.threads if args.threads > 0 else 4
        runner = K.BitNetRunner(threads=threads)
        pair_seqs = [K.read_ids(ids_paths[langs[0]])[0], K.read_ids(ids_paths[langs[1]])[0]]
        pair_masks = [masks_by_scheme_lang[(scheme, langs[0])][0], masks_by_scheme_lang[(scheme, langs[1])][0]]
        runner.set_scheme(scheme)
        got_pair = runner.score_seqs(pair_seqs, pair_masks)
        ref0 = runner.score_b1_ref(pair_seqs[0], pair_masks[0])
        md = max(abs(a - b) for a, b in zip(got_pair[0], ref0))
        print(f"[bitnet parity {scheme}] max_abs_lp_diff={md:.3e}", flush=True)
        assert md < 1e-6, f"Parity failed on {scheme}: {md}"
    else:
        threads = args.threads if args.threads > 0 else 6
        runner = K.EuroLLMRunner(EUROLLM_DIR, threads=threads)
        pair_seqs = [K.read_ids(ids_paths[langs[0]])[0], K.read_ids(ids_paths[langs[1]])[0]]
        pair_masks = [masks_by_scheme_lang[(scheme, langs[0])][0], masks_by_scheme_lang[(scheme, langs[1])][0]]
        runner.set_scheme(scheme)
        got_pair = runner.score_seqs(pair_seqs, pair_masks)
        ref0 = runner.score_b1_hf_eager(pair_seqs[0], use_kvq=True, mask=pair_masks[0])
        md = max(abs(a - b) for a, b in zip(got_pair[0], ref0))
        print(f"[eurollm parity {scheme}] max_abs_lp_diff={md:.3e}", flush=True)
        assert md < 1e-5, f"Parity failed on {scheme}: {md}"

    (out_dir / "batch_identity.json").write_text(json.dumps({scheme: md}, indent=1) + "\n")

    K.run_eval_matrix(
        model_name=args.model,
        runner=runner,
        ids_paths=ids_paths,
        schemes=[scheme],
        out_dir=out_dir,
        batch_size=args.batch,
        masks_by_scheme_lang=masks_by_scheme_lang,
        fresh=args.fresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
