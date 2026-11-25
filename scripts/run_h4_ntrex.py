#!/usr/bin/env python3
"""H4 (T49): Run NTREX-128 coherent parallel evaluation (ctx=1024, N=24) on BitNet and EuroLLM-1.7B."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
H4_DIR = ROOT / "results" / "2026-11-24-b_agent2" / "H4"
EUROLLM_DIR = pathlib.Path("~/tenzo-refs/hf/eurollm-1.7b").expanduser()
sys.path.insert(0, str(ROOT / "scripts"))
import kv_eval_lib as K


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["bitnet", "eurollm"])
    ap.add_argument("--schemes", default="")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    ids_paths = {lang: H4_DIR / f"ntrex_{args.model}_{lang}.ids" for lang in ("en", "uk")}
    out_dir = H4_DIR / "scores" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "bitnet":
        schemes = [s.strip() for s in args.schemes.split(",") if s.strip()] if args.schemes else ["fp32", "tenzo_int8", "S3+s4r32", "S3+s4r128"]
        threads = args.threads if args.threads > 0 else 4
        runner = K.BitNetRunner(threads=threads)
        pair = [K.read_ids(ids_paths["en"])[0], K.read_ids(ids_paths["uk"])[0]]
        chk = {}
        for sch in ("fp32", "S3+s4r32", "S3+s4r128"):
            runner.set_scheme(sch)
            got_pair = runner.score_seqs(pair)
            ref0 = runner.score_b1_ref(pair[0])
            md = max(abs(a - b) for a, b in zip(got_pair[0], ref0))
            print(f"[bitnet parity {sch}] max_abs_lp_diff={md:.3e}", flush=True)
            assert md < 1e-5, f"Parity failed on {sch}: {md}"
            chk[sch] = md
        (out_dir / "batch_identity.json").write_text(json.dumps(chk, indent=1) + "\n")
    else:
        schemes = [s.strip() for s in args.schemes.split(",") if s.strip()] if args.schemes else ["fp", "tenzo_int8", "S3+s4r32", "S3+s4r128"]
        threads = args.threads if args.threads > 0 else 6
        runner = K.EuroLLMRunner(EUROLLM_DIR, threads=threads)
        pair = [K.read_ids(ids_paths["en"])[0], K.read_ids(ids_paths["uk"])[0]]
        chk = {}
        for sch in ("fp", "S3+s4r32", "S3+s4r128"):
            runner.set_scheme(sch)
            got_pair = runner.score_seqs(pair)
            ref0 = runner.score_b1_hf_eager(pair[0], use_kvq=(sch != "fp"))
            md = max(abs(a - b) for a, b in zip(got_pair[0], ref0))
            print(f"[eurollm parity {sch}] max_abs_lp_diff={md:.3e}", flush=True)
            assert md < 1e-5, f"Parity failed on {sch}: {md}"
            chk[sch] = md
        (out_dir / "batch_identity.json").write_text(json.dumps(chk, indent=1) + "\n")

    K.run_eval_matrix(
        model_name=args.model,
        runner=runner,
        ids_paths=ids_paths,
        schemes=schemes,
        out_dir=out_dir,
        batch_size=args.batch,
        fresh=args.fresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
