#!/usr/bin/env python3
"""Parity test (L4): verify C++ tenzo::runtime::Tokenizer matches HF AutoTokenizer.

Generates >= 3000 diverse test strings (WikiText-2 EN, Kobza UK, C++/Python/CMake source,
prompts.json, and adversarial Unicode/whitespace/contraction/digit/special-token edge cases),
pipes them through `tenzo-inference --tokenize-batch` inside the dev Docker container, and asserts:
  1. 100% exact token-ID parity with add_bos=True and add_bos=False.
  2. 100% exact UTF-8 round-trip decode(encode(s, add_bos=False)) == s.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]


def collect_cases(target_min: int = 3200) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []

    def add(cat: str, s: str) -> None:
        cases.append((cat, s))

    # 1. Adversarial edge cases (contractions, digits, whitespace, UTF-8, emojis, CJK, special tokens)
    adversarial = [
        "",
        " ",
        "  ",
        "   ",
        "    ",
        "\t",
        "\n",
        "\r\n",
        "  \n\r\n  foo\r\nbar   \t  baz   ",
        "   !@#\n\n  xyz",
        "a   \r\n   b",
        "  \t \n  a",
        "foo\x1cbar\x1d\x1e\x1fbaz",
        "The capital of France is",
        "Île-de-France",
        "The Île-de-France region",
        "Привіт, світ!",
        "Київ — столиця України. 2026 рік!",
        "Ґанок, єдність, їжак, щирість — ґҐєЄіІїЇ!",
        "It's I's you're YOU'RE we've WE'VE he'm THEY'LL she'd I'LL he'll",
        "'said 'reign don't won't can't shouldn't've",
        "0 1 12 123 1234 12345 1234567 1234567890",
        "2026-11-23T14:15:00+02:00 v1.58b 3.14159265",
        "foo   <|eot_id|>bar",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are Tenzo.<|eot_id|>",
        "<|begin_of_tex hello <|end_of_text|><|eot_id|><|reserved_special_token_0|>",
        "café naïve résumé coöperate jalapeño Zürich Łódź",
        "🚀🔥🇺🇦𝄞𝄞 hello 世界 你好こんにちは안녕하세요 مرحبا بالعالم",
        "a" * 200,
        "!" * 80 + "\r\n",
        "\u00a0\u2003\u202f\u3000mixed\u2002spaces",
        "e\u0301 i\u0308 + combined Ukrainian й ї",
        "   multiple\t\tspaces\u00a0here   and   at end   ",
        "```cpp\nint main() { return x != y ? (a << 32) | b : 0; }\n```\n",
    ]
    for s in adversarial:
        add("adversarial", s)

    # 2. Golden prompts.json
    prompts_path = ROOT / "tools" / "reference" / "prompts.json"
    if prompts_path.exists():
        pdata = json.loads(prompts_path.read_text(encoding="utf-8"))
        items = pdata.values() if isinstance(pdata, dict) else pdata
        for item in items:
            if isinstance(item, dict) and "prompt" in item:
                add("prompts", item["prompt"])
            elif isinstance(item, str):
                add("prompts", item)

    # 3. English WikiText-2 lines (~900 lines)
    wt2_raw = pathlib.Path.home() / "tenzo-refs" / "hf" / "wt2_test.raw"
    if wt2_raw.exists():
        en_count = 0
        for line in wt2_raw.read_text(encoding="utf-8").splitlines():
            if line.strip():
                add("wikitext2_en", line)
                en_count += 1
                if en_count >= 900:
                    break

    # 4. Ukrainian Kobza sentences/lines (~1200 snippets from kobza_first_rows + rows_100_199)
    uk_count = 0
    for kname in ("kobza_first_rows.json", "kobza_rows_100_199.json"):
        kp = ROOT / "results" / "reference" / kname
        if not kp.exists():
            continue
        kdata = json.loads(kp.read_text(encoding="utf-8"))
        for row in kdata.get("rows", []):
            txt = row.get("row", {}).get("text", "")
            for line in txt.splitlines():
                if line.strip():
                    # Include both full lines and shorter clause chunks so we cover >= 1100 UK lines
                    add("kobza_uk", line)
                    uk_count += 1
                    if len(line) > 120:
                        for part in line.split(". "):
                            if part.strip():
                                add("kobza_uk", part.strip() + ".")
                                uk_count += 1
                    if uk_count >= 1300:
                        break
            if uk_count >= 1300:
                break

    # 5. Repository code/docs lines (C++, Python, CMake, Markdown) until >= target_min
    code_files = (
        sorted((ROOT / "src" / "runtime").glob("*.cpp"))
        + sorted((ROOT / "src" / "runtime").glob("*.h"))
        + sorted((ROOT / "scripts").glob("*.py"))
    )
    for cf in code_files:
        for line in cf.read_text(encoding="utf-8", errors="replace").splitlines():
            if len(cases) >= target_min:
                break
            add("code", line)
        if len(cases) >= target_min:
            break

    # Top up with sliding slices from wt2_raw if needed
    if len(cases) < target_min and wt2_raw.exists():
        txt = wt2_raw.read_text(encoding="utf-8")
        step = 137
        pos = 0
        while len(cases) < target_min and pos + step < len(txt):
            add("wikitext2_slice", txt[pos : pos + step])
            pos += step

    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cases", type=int, default=3000)
    ap.add_argument("--model-dir", default="/app/models/bitnet-1.58b-v2")
    ap.add_argument("--binary", default="/app/cmake-build-debug/tenzo-inference")
    ap.add_argument("--report-json", default="")
    args = ap.parse_args()

    cases = collect_cases(max(args.min_cases, 3200))
    if len(cases) < args.min_cases:
        print(f"ERROR: collected only {len(cases)} < {args.min_cases} cases", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T", local_files_only=True)

    payload_lines = [base64.b64encode(s.encode("utf-8")).decode("ascii") for _, s in cases]
    payload = "\n".join(payload_lines) + "\n.\n"

    cmd = [
        "docker", "compose", "run", "--rm", "-T", "dev",
        args.binary, "-m", args.model_dir, "--tokenize-batch",
    ]
    proc = subprocess.run(cmd, input=payload, text=True, capture_output=True, cwd=str(ROOT))
    if proc.returncode != 0:
        print("tenzo-inference failed:", proc.stderr, file=sys.stderr)
        return 1

    out_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("IDS_BOS:")]
    if len(out_lines) != len(cases):
        print(f"ERROR: expected {len(cases)} output lines, got {len(out_lines)}", file=sys.stderr)
        print("stdout head:", proc.stdout[:500], file=sys.stderr)
        print("stderr head:", proc.stderr[:500], file=sys.stderr)
        return 1

    mismatches = []
    rt_failures = []
    by_cat: dict[str, dict[str, int]] = {}

    for idx, ((cat, s), ln) in enumerate(zip(cases, out_lines)):
        st = by_cat.setdefault(cat, {"n": 0, "bos_ok": 0, "raw_ok": 0, "rt_ok": 0})
        st["n"] += 1
        parts = ln.split("\t")
        got_bos = [int(x) for x in parts[0][len("IDS_BOS:") :].strip().split()] if parts[0][len("IDS_BOS:") :].strip() else []
        got_raw = [int(x) for x in parts[1][len("IDS_RAW:") :].strip().split()] if parts[1][len("IDS_RAW:") :].strip() else []
        rt_b64 = parts[2][len("RT:") :]
        got_rt = base64.b64decode(rt_b64).decode("utf-8", errors="surrogateescape")

        exp_bos = tok.encode(s, add_special_tokens=True)
        exp_raw = tok.encode(s, add_special_tokens=False)

        if got_bos == exp_bos:
            st["bos_ok"] += 1
        else:
            if len(mismatches) < 20:
                mismatches.append({"idx": idx, "cat": cat, "mode": "bos", "str": s[:120], "exp": exp_bos[:20], "got": got_bos[:20]})

        if got_raw == exp_raw:
            st["raw_ok"] += 1
        else:
            if len(mismatches) < 20:
                mismatches.append({"idx": idx, "cat": cat, "mode": "raw", "str": s[:120], "exp": exp_raw[:20], "got": got_raw[:20]})

        if got_rt == s:
            st["rt_ok"] += 1
        else:
            if len(rt_failures) < 20:
                rt_failures.append({"idx": idx, "cat": cat, "str": s[:120], "got": got_rt[:120]})

    total_n = len(cases)
    bos_matches = sum(v["bos_ok"] for v in by_cat.values())
    raw_matches = sum(v["raw_ok"] for v in by_cat.values())
    rt_matches = sum(v["rt_ok"] for v in by_cat.values())

    summary = {
        "total_cases": total_n,
        "bos_exact": bos_matches,
        "raw_exact": raw_matches,
        "roundtrip_exact": rt_matches,
        "by_category": by_cat,
        "sample_mismatches": mismatches,
        "sample_rt_failures": rt_failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report_json:
        pathlib.Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0 if (bos_matches == total_n and raw_matches == total_n and rt_matches == total_n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
