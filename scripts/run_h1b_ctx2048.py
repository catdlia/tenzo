#!/usr/bin/env python3
"""Run H1b (ctx=2048) TenzoEngine evaluation across en_long, en_concat, uk_long, uk_concat.

Reuses verified w=0..3 of en_long from results/2026-11-23_agent2/H3/scores/engine_*_en.txt,
then evaluates remaining windows in 4-window chunks under int8_fused, ternary_rot, and
ternary_rot --kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt.
Supports incremental resume (skips completed windows).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
H1B_DIR = ROOT / "results" / "2026-11-24_agent2" / "H1b"
SCORES_DIR = H1B_DIR / "scores"
OLD_H3_SCORES = ROOT / "results" / "2026-11-23_agent2" / "H3" / "scores"
ALPHA_CONTAINER = "/app/results/2026-11-23_agent2/H1/alpha_mix.txt"


def load_existing(path: pathlib.Path) -> tuple[list[str], dict[int, tuple[str, str]]]:
    headers: list[str] = []
    done: dict[int, tuple[str, str]] = {}
    if not path.exists():
        return headers, done
    cur = None
    for ln in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if ln.startswith("#"):
            if not done:
                headers.append(ln)
        elif ln.startswith("w "):
            cur = ln
        elif ln.startswith("lp ") and cur is not None and ln.endswith("\n"):
            wid = int(cur.split()[1])
            done[wid] = (cur, ln)
            cur = None
    return headers, done


def write_scores(path: pathlib.Path, headers: list[str], done: dict[int, tuple[str, str]], n_target: int) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for h in headers:
            f.write(h if h.endswith("\n") else h + "\n")
        for wi in range(n_target):
            if wi in done:
                parts_w = done[wi][0].split()
                f.write(f"w {wi} {parts_w[2]} {parts_w[3]}\n")
                f.write(done[wi][1])
    tmp.replace(path)


def seed_en_long_first4(mode_key: str, out_path: pathlib.Path) -> None:
    old_map = {
        "int8_fused": OLD_H3_SCORES / "engine_int8_fused_en.txt",
        "ternary_rot": OLD_H3_SCORES / "engine_ternary_rot_en.txt",
        "ternary_rot_alpha": OLD_H3_SCORES / "engine_ternary_rot_alpha_en.txt",
    }
    old_p = old_map[mode_key]
    if not old_p.exists():
        return
    # Verify exact match of first 4 windows in h1b_en_long.ids vs windows_en_2048.ids
    ref4 = [ln.strip() for ln in (ROOT / "results" / "reference" / "windows_en_2048.ids").read_text().splitlines() if ln.strip()][:4]
    h1b4 = [ln.strip() for ln in (H1B_DIR / "h1b_en_long.ids").read_text().splitlines() if ln.strip()][:4]
    assert ref4 == h1b4, "en_long w=0..3 does not match windows_en_2048.ids!"

    hdrs, existing = load_existing(out_path)
    if all(w in existing for w in range(4)):
        return
    old_hdrs, old_done = load_existing(old_p)
    for w in range(4):
        if w in old_done and w not in existing:
            existing[w] = old_done[w]
    if not hdrs:
        hdrs = old_hdrs
    write_scores(out_path, hdrs, existing, max(existing.keys()) + 1)
    print(f"[seed] copied w=0..3 from {old_p.name} -> {out_path.name}", flush=True)


def run_chunk(ids_chunk_rel: str, kv_arg: str, alpha_container: str | None, chunk_out_rel: str, log_path: pathlib.Path) -> None:
    cmd = [
        "nice", "-n", "19", "ionice", "-c3",
        "docker", "compose", "run", "--rm",
        "-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread",
        "dev", "/app/cmake-build-debug/tenzo-inference",
        "-m", "/app/models/bitnet-1.58b-v2",
        "--kv-quant", kv_arg,
        "-c", "2048",
        "--prefill", "batched",
        "--score", f"/app/{ids_chunk_rel}",
        "--score-out", f"/app/{chunk_out_rel}",
    ]
    if alpha_container:
        cmd.extend(["--kv-alpha", alpha_container])
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== CMD: {' '.join(cmd)} ===\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Engine failed rc={proc.returncode}, see {log_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="en_long,en_concat,uk_long,uk_concat")
    ap.add_argument("--modes", default="int8_fused,ternary_rot,ternary_rot_alpha")
    ap.add_argument("--max-windows", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    mode_cfg = {
        "int8_fused": ("int8_fused", None),
        "ternary_rot": ("ternary_rot", None),
        "ternary_rot_alpha": ("ternary_rot", ALPHA_CONTAINER),
    }

    t0 = time.time()
    for cell in cells:
        all_lines = [ln.strip() for ln in (H1B_DIR / f"h1b_{cell}.ids").read_text(encoding="utf-8").splitlines() if ln.strip()]
        n_target = min(args.max_windows, len(all_lines))
        for mkey in modes:
            kv_arg, alpha_c = mode_cfg[mkey]
            out_path = SCORES_DIR / f"{cell}__{mkey}.txt"
            log_path = SCORES_DIR / f"{cell}__{mkey}.log"
            if cell == "en_long":
                seed_en_long_first4(mkey, out_path)
            hdrs, done = load_existing(out_path)
            todo = [w for w in range(n_target) if w not in done]
            if not todo:
                print(f"[skip] {cell} {mkey}: {len(done)}/{n_target} already complete", flush=True)
                continue
            print(f"[run] {cell} {mkey}: {len(todo)} windows todo ({todo})", flush=True)
            for b0 in range(0, len(todo), args.chunk):
                w_batch = todo[b0 : b0 + args.chunk]
                tmp_ids = SCORES_DIR / f"_tmp_{cell}_{mkey}.ids"
                tmp_out = SCORES_DIR / f"_tmp_{cell}_{mkey}.txt"
                tmp_ids.write_text("\n".join(all_lines[w] for w in w_batch) + "\n", encoding="utf-8")
                if tmp_out.exists():
                    tmp_out.unlink()
                t_b0 = time.time()
                run_chunk(
                    str(tmp_ids.relative_to(ROOT)),
                    kv_arg,
                    alpha_c,
                    str(tmp_out.relative_to(ROOT)),
                    log_path,
                )
                c_hdrs, c_done = load_existing(tmp_out)
                assert len(c_done) == len(w_batch), f"Expected {len(w_batch)} rows, got {len(c_done)}"
                # Preserve freshly produced header (contains # kv_alpha=... sha256=...) when available
                if not hdrs or (mkey == "ternary_rot_alpha" and not any("kv_alpha=" in h for h in hdrs)):
                    hdrs = c_hdrs
                for local_i, orig_w in enumerate(w_batch):
                    done[orig_w] = c_done[local_i]
                write_scores(out_path, hdrs, done, n_target)
                print(
                    f"  done {cell} {mkey} w={w_batch} ({len(done)}/{n_target}) chunk={time.time()-t_b0:.0f}s total={time.time()-t0:.0f}s",
                    flush=True,
                )
    print(f"H1B COMPLETE up to max_windows={args.max_windows} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
