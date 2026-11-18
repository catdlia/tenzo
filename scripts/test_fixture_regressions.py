#!/usr/bin/env python3
"""
scripts/test_fixture_regressions.py - Automated Regression Test Suite for Fixture Corruption
Milestone M2_continuation: Requirement R2.b

Verifies that:
1. Truncated TL1 fixtures (size < 4096 or size > 4096) fail cleanly and trigger non-zero exit (exit code 1).
2. Corrupted TL1 fixtures (containing packed nibbles > 8) fail cleanly and trigger non-zero exit (exit code 1).
3. Missing TL1 fixtures fail cleanly and trigger non-zero exit (exit code 1).
4. Valid TL1 fixtures conform strictly to size (4096 bytes) and valid nibble range ([0, 8]).
5. Both reference validation and CLI execution (tenzo-cli tl1 <custom_path>) reject invalid fixtures.
"""

import sys
import os
import tempfile
import subprocess
from typing import Tuple, Optional


EXPECTED_TL1_SIZE = 4096


def validate_tl1_fixture_bytes(data: bytes) -> Tuple[bool, Optional[str]]:
    """
    Reference validation of TL1 fixture bytes.
    Matches strict C++ contract in src/tests/TestTL1.cpp:
    - Exactly 4096 bytes
    - Packed nibbles in [0, 8]
    """
    if len(data) != EXPECTED_TL1_SIZE:
        return False, f"Invalid fixture size: {len(data)} bytes, expected exactly {EXPECTED_TL1_SIZE} bytes"

    for i, b in enumerate(data):
        low_nibble = b & 0x0F
        high_nibble = (b >> 4) & 0x0F
        if low_nibble > 8 or high_nibble > 8:
            return False, f"Invalid nibble at byte {i}: low={low_nibble}, high={high_nibble} (max allowable: 8)"

    return True, None


def validate_tl1_fixture_file(path: str) -> Tuple[bool, Optional[str]]:
    """Validates TL1 fixture file on disk."""
    if not os.path.isfile(path):
        return False, f"Fixture file not found: {path}"

    with open(path, "rb") as f:
        data = f.read()

    return validate_tl1_fixture_bytes(data)


def find_cli_binary() -> Optional[str]:
    """Finds tenzo-cli binary in common locations."""
    candidates = [
        "/app/cmake-build-debug/tenzo-cli",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cmake-build-debug", "tenzo-cli"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def run_cli_tl1(binary_path: str, fixture_path: str) -> Tuple[int, str]:
    """Executes tenzo-cli tl1 <fixture_path> and returns (exit_code, output)."""
    cmd = [binary_path, "tl1", fixture_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15)
        return proc.returncode, proc.stdout
    except Exception as e:
        return -1, str(e)


def is_unrebuilt_cli(output: str) -> bool:
    """Detects if tenzo-cli failed because it has not yet been rebuilt with TL1 fixture support."""
    indicators = [
        "could not open input file",
        "/app/test_tl1.mlir",
        "test_tl1.mlir",
        "Failed to load test_tl1.mlir",
        "Unknown command: tl1",
    ]
    return any(ind in output for ind in indicators)


def check_cli_negative_test(cli_binary: str, fixture_path: str, test_num: int, case_desc: str) -> bool:
    """
    Executes the CLI against an invalid fixture. Passing requires ALL of:
      * non-zero exit code,
      * the fixture-validation tag [TL1-FIXTURE-INVALID] in the output,
      * NO MLIR compile/legalisation failure text (a rejection caused by a broken kernel pipeline is the wrong reason
        and previously made every negative case pass vacuously).
    A binary that predates the fixture support is a FAILURE, not a skip.
    """
    rc, out = run_cli_tl1(cli_binary, fixture_path)
    wrong_reason = [m for m in ("MLIR compilation failed", "failed to legalize", "Failed to load or parse") if m in out]
    if is_unrebuilt_cli(out):
        print(f"❌ [FAIL] Test {test_num} (CLI): tenzo-cli predates TL1 fixture support (rebuild it); output: {out.strip()[:160]}")
        return False
    if wrong_reason:
        print(f"❌ [FAIL] Test {test_num} (CLI): {case_desc} was rejected for the WRONG reason ({wrong_reason[0]!r}), not by fixture validation.")
        return False
    if rc != 0 and "[TL1-FIXTURE-INVALID]" in out:
        print(f"✅ [PASS] Test {test_num} (CLI): tenzo-cli rejected {case_desc} via fixture validation [TL1-FIXTURE-INVALID] (code {rc}).")
        return True
    print(f"❌ [FAIL] Test {test_num} (CLI): {case_desc}: rc={rc}, [TL1-FIXTURE-INVALID] {'present' if '[TL1-FIXTURE-INVALID]' in out else 'missing'}.")
    return False


def check_cli_positive_test(cli_binary: str, fixture_path: str, test_num: int) -> bool:
    """
    Executes CLI against a valid fixture.
    Must verify rc == 0 and [PASS] is present.
    If unrebuilt binary, reports SKIP (pending remote compilation).
    Returns True if pass or skipped, False if failed.
    """
    rc, out = run_cli_tl1(cli_binary, fixture_path)
    if rc == 0 and "[PASS]" in out:
        print(f"✅ [PASS] Test {test_num} (CLI): tenzo-cli correctly accepted valid fixture.")
        return True
    else:
        print(f"❌ [FAIL] Test {test_num} (CLI): tenzo-cli failed on valid fixture (code {rc}):\n{out}")
        return False


def create_valid_tl1_fixture() -> bytes:
    """Generates a valid 4096-byte TL1 fixture with all nibbles in [0, 8]."""
    # 4096 bytes = 1 block * 64 k_half * 64 bytes
    # Pack valid ternary codes: (w0+1)*3 + (w1+1) in [0, 8]
    data = bytearray(EXPECTED_TL1_SIZE)
    for i in range(EXPECTED_TL1_SIZE):
        # Alternate valid nibble patterns in 0..8
        low = (i % 9)
        high = ((i * 3 + 1) % 9)
        data[i] = low | (high << 4)
    return bytes(data)


def main():
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║   Tenzo TL1 Fixture Corruption Regression Test Suite                   ║")
    print("║   Milestone 2 Continuation: Requirement R2.b                           ║")
    print("╚════════════════════════════════════════════════════════════════════════╝\n")

    cli_binary = find_cli_binary()
    if cli_binary:
        print(f"ℹ️ Found tenzo-cli binary: {cli_binary}")
    else:
        print("⚠️ tenzo-cli binary not found; running reference validator regressions only.")

    all_passed = True
    test_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        # Case 1: Truncated fixture (256 bytes)
        test_count += 1
        trunc_256 = os.path.join(tmpdir, "trunc_256.bin")
        with open(trunc_256, "wb") as f:
            f.write(b"\x00" * 256)

        ok_ref1, err1 = validate_tl1_fixture_file(trunc_256)
        if ok_ref1:
            print("❌ [FAIL] Test 1: Python validator accepted truncated (256B) fixture!")
            all_passed = False
        else:
            print(f"✅ [PASS] Test 1 (Ref): Truncated (256B) fixture rejected: {err1}")

        if cli_binary:
            if not check_cli_negative_test(cli_binary, trunc_256, 1, "truncated (256B) fixture"):
                all_passed = False

        # Case 2: Truncated by 1 byte (4095 bytes)
        test_count += 1
        trunc_4095 = os.path.join(tmpdir, "trunc_4095.bin")
        with open(trunc_4095, "wb") as f:
            f.write(b"\x00" * 4095)

        ok_ref2, err2 = validate_tl1_fixture_file(trunc_4095)
        if ok_ref2:
            print("❌ [FAIL] Test 2: Python validator accepted 4095B fixture!")
            all_passed = False
        else:
            print(f"✅ [PASS] Test 2 (Ref): Truncated (4095B) fixture rejected: {err2}")

        if cli_binary:
            if not check_cli_negative_test(cli_binary, trunc_4095, 2, "truncated (4095B) fixture"):
                all_passed = False

        # Case 3: Oversized fixture (5000 bytes)
        test_count += 1
        oversized = os.path.join(tmpdir, "oversized_5000.bin")
        with open(oversized, "wb") as f:
            f.write(b"\x00" * 5000)

        ok_ref3, err3 = validate_tl1_fixture_file(oversized)
        if ok_ref3:
            print("❌ [FAIL] Test 3: Python validator accepted oversized (5000B) fixture!")
            all_passed = False
        else:
            print(f"✅ [PASS] Test 3 (Ref): Oversized (5000B) fixture rejected: {err3}")

        if cli_binary:
            if not check_cli_negative_test(cli_binary, oversized, 3, "oversized (5000B) fixture"):
                all_passed = False

        # Case 4: Corrupted nibbles (4096 bytes, low nibble > 8, e.g. 0x09)
        test_count += 1
        corrupt_nibble_low = os.path.join(tmpdir, "corrupt_nibble_low.bin")
        corrupt_data = bytearray(create_valid_tl1_fixture())
        corrupt_data[100] = 0x09  # low nibble is 9 > 8
        with open(corrupt_nibble_low, "wb") as f:
            f.write(corrupt_data)

        ok_ref4, err4 = validate_tl1_fixture_file(corrupt_nibble_low)
        if ok_ref4:
            print("❌ [FAIL] Test 4: Python validator accepted fixture with low nibble=9!")
            all_passed = False
        else:
            print(f"✅ [PASS] Test 4 (Ref): Corrupted low nibble fixture rejected: {err4}")

        if cli_binary:
            if not check_cli_negative_test(cli_binary, corrupt_nibble_low, 4, "corrupted low nibble fixture"):
                all_passed = False

        # Case 5: Corrupted nibbles (4096 bytes, high nibble > 8, e.g. 0xF0)
        test_count += 1
        corrupt_nibble_high = os.path.join(tmpdir, "corrupt_nibble_high.bin")
        corrupt_data2 = bytearray(create_valid_tl1_fixture())
        corrupt_data2[200] = 0x90  # high nibble is 9 > 8
        with open(corrupt_nibble_high, "wb") as f:
            f.write(corrupt_data2)

        ok_ref5, err5 = validate_tl1_fixture_file(corrupt_nibble_high)
        if ok_ref5:
            print("❌ [FAIL] Test 5: Python validator accepted fixture with high nibble=9!")
            all_passed = False
        else:
            print(f"✅ [PASS] Test 5 (Ref): Corrupted high nibble fixture rejected: {err5}")

        if cli_binary:
            if not check_cli_negative_test(cli_binary, corrupt_nibble_high, 5, "corrupted high nibble fixture"):
                all_passed = False

        # Case 6: Missing fixture (nonexistent path)
        test_count += 1
        missing_path = os.path.join(tmpdir, "nonexistent_weights.bin")
        ok_ref6, err6 = validate_tl1_fixture_file(missing_path)
        if ok_ref6:
            print("❌ [FAIL] Test 6: Python validator accepted missing fixture file!")
            all_passed = False
        else:
            print(f"✅ [PASS] Test 6 (Ref): Missing fixture cleanly rejected: {err6}")

        if cli_binary:
            if not check_cli_negative_test(cli_binary, missing_path, 6, "missing fixture"):
                all_passed = False

        # Case 7: Valid fixture positive control
        test_count += 1
        valid_path = os.path.join(tmpdir, "valid_tl1.bin")
        with open(valid_path, "wb") as f:
            f.write(create_valid_tl1_fixture())

        ok_ref7, err7 = validate_tl1_fixture_file(valid_path)
        if not ok_ref7:
            print(f"❌ [FAIL] Test 7: Python validator rejected valid fixture: {err7}")
            all_passed = False
        else:
            print(f"✅ [PASS] Test 7 (Ref): Valid TL1 fixture correctly accepted.")

        if cli_binary:
            if not check_cli_positive_test(cli_binary, valid_path, 7):
                all_passed = False

    print("\n════════════════════════════════════════════════════════════════════════")
    if all_passed:
        print("🎉 ALL TL1 FIXTURE REGRESSION TESTS PASSED SUCCESSFULLY! 🎉")
        sys.exit(0)
    else:
        print("❌ TL1 FIXTURE REGRESSION TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
