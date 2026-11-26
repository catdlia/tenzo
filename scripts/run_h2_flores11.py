#!/usr/bin/env python3
"""H2 (T50 part 2): Run parallel FLORES-200 evaluation on BitNet and EuroLLM-1.7B."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
H2_DIR = ROOT / "results" / "2026-11-26_agent2" / "H2"
EUROLLM_DIR = pathlib.Path("~/tenzo-refs/hf/eurollm-1.7b").expanduser()
sys.path.insert(0, str(ROOT / "scripts"))
import kv_eval_lib as K


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["bitnet", "eurollm"])
    ap.add_argument("--schemes", default="")
    ap.add_argument("--langs", default="fr,cs,hu,el")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    ids_paths = {lang: H2_DIR / f"flores11_{args.model}_{lang}.ids" for lang in langs}
    for lang, p in ids_paths.items():
        assert p.exists(), f"Missing .ids file: {p}"
    out_dir = H2_DIR / "scores" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs0 = K.read_ids(ids_paths[langs[0]])
    pair = [seqs0[0], K.read_ids(ids_paths[langs[1]])[0]] if len(langs) > 1 else [seqs0[0], seqs0[1]]

    if args.model == "bitnet":
        schemes = [s.strip() for s in args.schemes.split(",") if s.strip()] if args.schemes else ["fp32", "S3+s4r32"]
        threads = args.threads if args.threads > 0 else 4
        runner = K.BitNetRunner(threads=threads)
        chk = {}
        for sch in schemes:
            runner.set_scheme(sch)
            got_pair = runner.score_seqs(pair)
            ref0 = runner.score_b1_ref(pair[0])
            md = max(abs(a - b) for a, b in zip(got_pair[0], ref0))
            print(f"[bitnet parity {sch} on {langs[0]} w0] max_abs_lp_diff={md:.3e}", flush=True)
            assert md < 1e-6, f"Parity failed on {sch}: {md}"
            chk[sch] = md
        bid_path = out_dir / "batch_identity.json"
        prev = json.loads(bid_path.read_text()) if bid_path.exists() else {}
        prev[",".join(langs)] = chk
        bid_path.write_text(json.dumps(prev, indent=1) + "\n")
    else:
        schemes = [s.strip() for s in args.schemes.split(",") if s.strip()] if args.schemes else ["fp", "S3+s4r32"]
        threads = args.threads if args.threads > 0 else 6
        runner = K.EuroLLMRunner(EUROLLM_DIR, threads=threads)
        chk = {}
        for sch in schemes:
            runner.set_scheme(sch)
            got_pair = runner.score_seqs(pair)
            ref0 = runner.score_b1_hf_eager(pair[0], use_kvq=(sch != "fp"))
            md = max(abs(a - b) for a, b in zip(got_pair[0], ref0))
            print(f"[eurollm parity {sch} on {langs[0]} w0] max_abs_lp_diff={md:.3e}", flush=True)
            assert md < 1e-5, f"Parity failed on {sch}: {md}"
            chk[sch] = md
        bid_path = out_dir / "batch_identity.json"
        prev = json.loads(bid_path.read_text()) if bid_path.exists() else {}
        prev[",".join(langs)] = chk
        bid_path.write_text(json.dumps(prev, indent=1) + "\n")

    # Evaluate one language at a time across all schemes so each finished language is a complete pair.
    for lang in langs:
        K.run_eval_matrix(
            model_name=args.model,
            runner=runner,
            ids_paths={lang: ids_paths[lang]},
            schemes=schemes,
            out_dir=out_dir,
            batch_size=args.batch,
            fresh=args.fresh,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
