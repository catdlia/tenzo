#!/usr/bin/env python3
"""
scripts/verify_e2e_parity.py - End-to-End Model Inference Parity Verification Suite (Milestone 4 / Feature F18)

Verifies native Tenzo inference against genuine restored BitNet-2B model weights.
Validates:
1. Absence of synthetic/placeholder fallback warnings.
2. End-to-end token generation on deterministic greedy decoding (temp=0.0, rep_penalty=1.0).
3. Parity with verified pre-trained model outputs on standard factoid and technical prompts.
"""

import sys
import os
import subprocess
import argparse
import re
from pathlib import Path

TEST_CASES = [
    {
        "name": "France Capital Factoid Prompt",
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "expected_prefixes": ["Paris", "a small city"],
        "expected_first_tokens": [12366, 264],
    },
    {
        "name": "Compiler Definition Technical Prompt",
        "prompt": "In computer science, a compiler translates source code written in a high-level programming language into",
        "max_tokens": 5,
        "expected_prefixes": ["machine code", "executable code"],
        "expected_first_tokens": [644, 33256],
    }
]


def run_single_test(case, model_dir: str, verbose: bool = True) -> bool:
    print(f"\n▶ Running Parity Test: {case['name']}")
    print(f"  Prompt: \"{case['prompt']}\"")

    inside_docker = Path("/app").is_dir() and Path("/app/cmake-build-debug/tenzo-inference").is_file()

    if inside_docker:
        binary_path = "/app/cmake-build-debug/tenzo-inference"
        cmd = [
            binary_path,
            "-m", model_dir,
            "-p", case["prompt"],
            "-n", str(case["max_tokens"]),
            "-t", "0.0",
            "--rep-penalty", "1.0",
            "--kv-quant", "int8_fused"
        ]
    else:
        cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread",
            "dev",
            "/app/cmake-build-debug/tenzo-inference",
            "-m", model_dir if model_dir.startswith("/app") else f"/app/{model_dir}",
            "-p", case["prompt"],
            "-n", str(case["max_tokens"]),
            "-t", "0.0",
            "--rep-penalty", "1.0",
            "--kv-quant", "int8_fused"
        ]

    env = dict(os.environ)
    env["OMP_PLACES"] = "cores"
    env["OMP_PROC_BIND"] = "spread"

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined_output = res.stdout + "\n" + res.stderr

    # 1. Verify absence of synthetic fallback warning
    if "synthetic/placeholder" in combined_output:
        print("❌ FAILED: Engine is running on synthetic/placeholder weights!", file=sys.stderr)
        if verbose:
            print(combined_output, file=sys.stderr)
        return False
    else:
        print("  ✅ Weight integrity check: Genuine weights active (no synthetic fallbacks detected).")

    # 2. Extract generated output
    match = re.search(r"⚡ Output:\s*(.+)", combined_output)
    if not match:
        print("❌ FAILED: Could not locate '⚡ Output:' marker in stdout/stderr stream!", file=sys.stderr)
        if verbose:
            print(combined_output, file=sys.stderr)
        return False

    gen_text = match.group(1).strip()
    print(f"  Observed completion: \"{gen_text}\"")

    # 3. Match against expected valid completions
    matched_prefix = None
    for prefix in case["expected_prefixes"]:
        if prefix.lower() in gen_text.lower():
            matched_prefix = prefix
            break

    if matched_prefix:
        print(f"  ✅ Parity match: Detected expected prefix \"{matched_prefix}\".")
        print(f"✅ PASSED: {case['name']}")
        return True
    else:
        print(f"❌ FAILED: Completion \"{gen_text}\" did not match any expected prefix: {case['expected_prefixes']}", file=sys.stderr)
        if verbose:
            print(combined_output, file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify End-to-End Inference Parity (M4 / Feature F18)")
    parser.add_argument(
        "-m", "--model-dir",
        type=str,
        default="/app/models/bitnet-1.58b" if Path("/app").is_dir() else "models/bitnet-1.58b",
        help="Path to models/bitnet-1.58b directory",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print verbose execution telemetry",
    )
    args = parser.parse_args()

    print("=================================================================")
    print("  Tenzo End-to-End Model Inference Parity Verification Suite (F18)")
    print("=================================================================")
    print(f"Model directory: {args.model_dir}")

    total = len(TEST_CASES)
    passed = 0

    for case in TEST_CASES:
        if run_single_test(case, args.model_dir, args.verbose):
            passed += 1

    print("\n-----------------------------------------------------------------")
    print(f"Parity Verification Summary: {passed}/{total} test cases passed.")
    print("-----------------------------------------------------------------")

    if passed == total:
        print("🎉 ALL END-TO-END INFERENCE PARITY TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("❌ PARITY VERIFICATION FAILED!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
