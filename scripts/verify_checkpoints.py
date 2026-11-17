#!/usr/bin/env python3
"""
scripts/verify_checkpoints.py - Verify integrity of genuine BitNet checkpoints.

Checks SHA-256 hashes against models/CHECKPOINTS.sha256 and ensures
checkpoint files are protected against accidental overwrites or corruption.
"""

import argparse
import hashlib
import os
import stat
import sys
from pathlib import Path


def compute_sha256(filepath: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def is_readonly(filepath: Path) -> bool:
    mode = filepath.stat().st_mode
    # Check if any write bit is set (user, group, other)
    return (mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)) == 0


def make_readonly(filepath: Path) -> bool:
    try:
        # Explicitly enforce canonical 0444 read-only permissions (r--r--r--)
        os.chmod(filepath, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return True
    except OSError as e:
        print(f"⚠️ Warning: Could not chmod 0444 on {filepath}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify integrity of BitNet model checkpoints")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest file (default: models/CHECKPOINTS.sha256)",
    )
    parser.add_argument(
        "--enforce",
        "--enforce-readonly",
        action="store_true",
        dest="enforce",
        help="Enforce read-only (chmod 0444) permissions on checkpoint files",
    )
    parser.add_argument(
        "--check-readonly",
        action="store_true",
        default=True,
        help="Verify checkpoint files are read-only (default: enabled)",
    )
    parser.add_argument(
        "--no-check-readonly",
        action="store_false",
        dest="check_readonly",
        help="Skip read-only permission verification",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output errors",
    )

    args = parser.parse_args()

    # Determine project root (directory containing this script's parent)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    manifest_path = Path(args.manifest) if args.manifest else project_root / "models" / "CHECKPOINTS.sha256"

    if not manifest_path.is_file():
        print(f"❌ ERROR: Manifest file not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"📋 Verifying checkpoints from manifest: {manifest_path}")

    failed = False
    total = 0
    passed = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"❌ ERROR: Invalid manifest line {line_num}: '{line}'", file=sys.stderr)
                failed = True
                continue

            expected_hash, rel_path = parts[0], parts[1].strip()
            filepath = project_root / rel_path

            total += 1

            if not filepath.is_file():
                print(f"❌ ERROR: Missing checkpoint file: {rel_path} ({filepath})", file=sys.stderr)
                failed = True
                continue

            if args.enforce:
                if not is_readonly(filepath):
                    if make_readonly(filepath):
                        if not args.quiet:
                            print(f"🔒 Enforced read-only permission (0444) on {rel_path}")
                    else:
                        print(f"❌ ERROR: Failed to enforce read-only permission on {rel_path}", file=sys.stderr)
                        failed = True

            if args.check_readonly:
                if not is_readonly(filepath):
                    print(f"❌ ERROR: File {rel_path} is writable (expected 0444 read-only)", file=sys.stderr)
                    failed = True

            if not args.quiet:
                print(f"⏳ Verifying SHA-256 for {rel_path} ({filepath.stat().st_size:,} bytes)...", flush=True)

            actual_hash = compute_sha256(filepath)
            if actual_hash.lower() != expected_hash.lower():
                print(
                    f"❌ FAILED: Checksum mismatch for {rel_path}\n"
                    f"   Expected: {expected_hash}\n"
                    f"   Actual:   {actual_hash}",
                    file=sys.stderr,
                )
                failed = True
            else:
                passed += 1
                if not args.quiet:
                    print(f"✅ PASSED: {rel_path} -> {actual_hash[:16]}... (OK)")

    if not args.quiet:
        print(f"\n📊 Summary: {passed}/{total} checkpoints passed.")

    if total == 0:
        print("❌ ERROR: Manifest file contains 0 valid entries (empty manifest)!", file=sys.stderr)
        sys.exit(1)

    if failed or passed != total:
        print("❌ Checkpoint verification failed!", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("🎉 All checkpoints verified successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
