#!/usr/bin/env python3
"""
Empirical Vulnerability Reproduction Script
Author: Challenger Agent challenger_m1_1

Reproduces two concrete vulnerabilities found during adversarial analysis:
1. Vulnerability 1 (CRITICAL): NaN and Inf embeddings bypass pre-promotion validation in download_model.py
   because float('nan') <= 1e-4 evaluates to False in Python, committing corrupted NaN/Inf weights to production.
2. Vulnerability 2 (MEDIUM): Test suite pollution in scripts/test_downloader_staging.py where tearDown()
   restores download_file_with_progress via self.orig_download, converting it into a bound instance method
   that crashes any subsequent real invocation with TypeError (3 arguments passed instead of 2).
"""

import os
import sys
import shutil
import tempfile
import inspect
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import download_model

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"


def reproduce_nan_inf_bypass():
    print(f"\n{ANSI_BOLD}{ANSI_CYAN}--- Reproducing Vulnerability 1: NaN / Inf Weights Bypass Validation ---{ANSI_RESET}")
    tmp_dir = tempfile.mkdtemp(prefix="repro_nan_")
    target_dir = os.path.join(tmp_dir, "model_target")
    os.makedirs(target_dir, exist_ok=True)

    # Place a sentinel file in target
    sentinel_path = os.path.join(target_dir, "sentinel.txt")
    with open(sentinel_path, "w") as f:
        f.write("ORIGINAL_SENTINEL_DATA")

    def mock_nan_export(safetensors_path, staging_dir, num_layers=30):
        # Write valid-sized weights containing all NaN float32 values
        weights_path = os.path.join(staging_dir, "weights.bin")
        with open(weights_path, "wb") as f:
            f.write(np.full((10, 2560), np.nan, dtype=np.float32).tobytes())
            f.seek(1836113919)
            f.write(b"\x00")
        with open(os.path.join(staging_dir, "model.mlir"), "w") as f:
            f.write("module {}")
        with open(os.path.join(staging_dir, "tokenizer.vocab"), "wb") as f:
            f.write(b"VOCAB")
        with open(os.path.join(staging_dir, "config.json"), "w") as f:
            f.write('{"model": "nan"}')

    orig_download = download_model.download_file_with_progress
    orig_export = download_model.export_safetensors_to_tenzo
    download_model.download_file_with_progress = lambda u, d: open(d, "wb").write(b"raw")
    download_model.export_safetensors_to_tenzo = mock_nan_export

    try:
        success = download_model.download_and_setup(
            repo_id="test/nan-model",
            output_dir=target_dir,
            num_layers=30
        )

        print(f"  download_and_setup result with NaN weights: {ANSI_BOLD}{success}{ANSI_RESET}")
        promoted_weights = os.path.join(target_dir, "weights.bin")
        has_promoted_weights = os.path.isfile(promoted_weights)

        if success and has_promoted_weights:
            print(f"  {ANSI_RED}❌ VULNERABILITY CONFIRMED: download_model.py successfully promoted all-NaN weights to production!{ANSI_RESET}")
            print(f"     Explanation: float(np.std(first_embeds)) is NaN, and (nan <= 1e-4) is False, bypassing the check!")
            return False
        else:
            print(f"  {ANSI_GREEN}✅ Correctly rejected NaN weights.{ANSI_RESET}")
            return True

    finally:
        download_model.download_file_with_progress = orig_download
        download_model.export_safetensors_to_tenzo = orig_export
        shutil.rmtree(tmp_dir, ignore_errors=True)


def reproduce_bound_method_teardown_bug():
    print(f"\n{ANSI_BOLD}{ANSI_CYAN}--- Reproducing Vulnerability 2: Test Suite Bound-Method tearDown Bug ---{ANSI_RESET}")
    import test_downloader_staging as tds

    # Check state before
    fn_before = download_model.download_file_with_progress
    is_method_before = inspect.ismethod(fn_before) or hasattr(fn_before, "__self__")
    print(f"  Before test execution: type={type(fn_before)}, is_method={is_method_before}")

    # Instantiate test case and invoke setUpClass, setUp, tearDown as unittest would
    test_case = tds.TestNegativeControl("test_simulated_download_network_failure")
    tds.DownloaderStagingTestCase.setUpClass()
    test_case.setUp()
    # Mock download during test
    download_model.download_file_with_progress = lambda u, d: None
    test_case.tearDown()

    # Check state after tearDown
    fn_after = download_model.download_file_with_progress
    is_method_after = inspect.ismethod(fn_after) or hasattr(fn_after, "__self__")
    print(f"  After test tearDown(): type={type(fn_after)}, is_method={is_method_after}")

    if is_method_after:
        print(f"  {ANSI_RED}❌ VULNERABILITY CONFIRMED: download_model.download_file_with_progress was restored as a bound method!{ANSI_RESET}")
        print(f"     Bound to instance: {fn_after.__self__}")
        try:
            fn_after("dummy_url", "dummy_path")
        except TypeError as e:
            print(f"     Attempted call with 2 args raised: {ANSI_YELLOW}{e}{ANSI_RESET}")
        return False
    else:
        print(f"  {ANSI_GREEN}✅ Cleanly restored function.{ANSI_RESET}")
        return True


if __name__ == "__main__":
    v1_fixed = reproduce_nan_inf_bypass()
    v2_fixed = reproduce_bound_method_teardown_bug()
    if v1_fixed and v2_fixed:
        print(f"\n{ANSI_BOLD}{ANSI_GREEN}🎉 ALL VULNERABILITIES VERIFIED FIXED!{ANSI_RESET}")
        sys.exit(0)
    else:
        print(f"\n{ANSI_BOLD}{ANSI_RED}❌ VULNERABILITY REPRODUCTION DETECTED UNRESOLVED DEFECTS!{ANSI_RESET}")
        sys.exit(1)
