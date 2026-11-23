#!/usr/bin/env python3
"""Run BitNet golden_bitnet.py nll across all 4 H2 2x2 cells cleanly in batched passes."""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
H2_DIR = ROOT / "results" / "2026-11-23_agent2" / "H2"
OUT_DIR = H2_DIR / "scores" / "bitnet"
CELLS = ["en_wiki", "uk_wiki", "en_web", "uk_web"]
SCHEMES = [
    ("fp32", None),
    ("tenzo_int8", "tenzo_int8"),
    ("S3_s4r32", "S3+s4r32"),
    ("S13_s4r32", "S13+s4r32"),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all64 = H2_DIR / "bitnet_2x2_all64.ids"
    lines: list[str] = []
    for c in CELLS:
        c_lines = [ln.strip() for ln in (H2_DIR / f"bitnet_{c}.ids").read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(c_lines) == 16, f"{c} has {len(c_lines)} lines"
        lines.extend(c_lines)
    all64.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for sname, kv_arg in SCHEMES:
        # Check if already split cleanly
        if all((OUT_DIR / f"{sname}_{c}.txt").exists() for c in CELLS):
            print(f"[skip] {sname} already split", flush=True)
            continue
        merged_txt = OUT_DIR / f"_all64_{sname}.txt"
        log_txt = OUT_DIR / f"_all64_{sname}.log"
        done_w = 0
        if merged_txt.exists():
            done_w = sum(1 for ln in merged_txt.read_text(encoding="utf-8").splitlines() if ln.startswith("w "))
        if done_w < 64:
            cmd = [
                "systemd-run", "--user", "--scope",
                "-p", "MemoryMax=4G", "-p", "MemorySwapMax=0", "--",
                "nice", "-n", "19", "ionice", "-c3",
                "python3", "tools/reference/golden_bitnet.py", "nll",
                "--windows", str(all64.relative_to(ROOT)),
                "--batch", "8",
                "--threads", "4",
                "--out", str(merged_txt.relative_to(ROOT)),
            ]
            if kv_arg is not None:
                cmd.extend(["--kv-scheme", kv_arg])
            print(f"[run] {sname}: {' '.join(cmd)}", flush=True)
            with log_txt.open("w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f"[FAIL] {sname} exit={rc}", file=sys.stderr)
                return rc

        # Split merged_txt (w 0..63) into 4 cell files (w 0..15 each)
        records: dict[int, tuple[str, str]] = {}
        cur = None
        for ln in merged_txt.read_text(encoding="utf-8").splitlines():
            if ln.startswith("w "):
                cur = ln
            elif ln.startswith("lp ") and cur is not None:
                parts = cur.split()
                wid = int(parts[1])
                records[wid] = (cur, ln)
                cur = None
        assert len(records) == 64, f"Expected 64 windows in {merged_txt}, got {len(records)}"

        for idx, c in enumerate(CELLS):
            out_p = OUT_DIR / f"{sname}_{c}.txt"
            out_lines = [f"# golden {sname} {c} windows=16"]
            for local_w in range(16):
                w_line, lp_line = records[idx * 16 + local_w]
                parts = w_line.split()
                parts[1] = str(local_w)
                out_lines.append(" ".join(parts))
                out_lines.append(lp_line)
            out_p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"[done] split {sname} -> {CELLS}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
