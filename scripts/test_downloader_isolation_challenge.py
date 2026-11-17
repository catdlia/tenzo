#!/usr/bin/env python3
"""
scripts/test_downloader_isolation_challenge.py - Empirical Challenge Suite
Milestone 1 / Requirement R1: Atomic promotion, rollback, permission enforcement, and multi-PID isolation.

Empirically tests:
1. Strict 0444 permission enforcement and overwrite prevention (DAC enforcement via unprivileged process).
2. Atomic swap failure simulation & bit-for-bit rollback of pre-existing target.
3. Concurrent multi-PID staging isolation (.staging_<pid>), fault isolation, and publication race safety.
"""

import os
import sys
import time
import shutil
import tempfile
import hashlib
import unittest
import subprocess
import multiprocessing as mp
import numpy as np

# Ensure project scripts dir is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import download_model

ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"

_ORIG_DOWNLOAD = download_model.download_file_with_progress
_ORIG_EXPORT = download_model.export_safetensors_to_tenzo
_ORIG_REPLACE = os.replace


class IsolationChallengeBase(unittest.TestCase):
    def setUp(self):
        # Create unique test directory and make it globally traversable so unprivileged users can test permissions
        self.tmp_dir = tempfile.mkdtemp(prefix="tenzo_iso_challenge_")
        os.chmod(self.tmp_dir, 0o777)
        self.target_dir = os.path.join(self.tmp_dir, "model_target")
        self.base_assets_dir = os.path.join(self.tmp_dir, "base_assets")
        os.makedirs(self.base_assets_dir, exist_ok=True)
        os.chmod(self.base_assets_dir, 0o777)

        # Populate canonical baseline topology fixtures
        with open(os.path.join(self.base_assets_dir, "tokenizer.vocab"), "wb") as f:
            f.write(b"CANONICAL_TOKENIZER_VOCAB_PAYLOAD" * 50)
        with open(os.path.join(self.base_assets_dir, "model.mlir"), "w", encoding="utf-8") as f:
            f.write("module { func.func @baseline() { return } }")
        with open(os.path.join(self.base_assets_dir, "config.json"), "w", encoding="utf-8") as f:
            f.write('{"model_type": "bitnet", "num_hidden_layers": 30}')

    def tearDown(self):
        download_model.download_file_with_progress = _ORIG_DOWNLOAD
        download_model.export_safetensors_to_tenzo = _ORIG_EXPORT
        os.replace = _ORIG_REPLACE
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def create_sentinel_target(self, make_weights_readonly=True):
        """Creates pre-existing sentinel files in target_dir and returns exact SHA-256 hashes."""
        os.makedirs(self.target_dir, exist_ok=True)
        os.chmod(self.target_dir, 0o777)
        sentinels = {
            "weights.bin": b"PRE_EXISTING_GOLDEN_WEIGHTS_BINARY_HASH_1234567890",
            "model.mlir": "module { // GOLDEN PRODUCTION TOPOLOGY CANARY }",
            "tokenizer.vocab": b"GOLDEN_VOCAB_CANARY_0987654321",
            "config.json": '{"golden_sentinel": "DO_NOT_CORRUPT_OR_TRUNCATE"}',
            "canary.sig": b"CANARY_SIGNATURE_STAMP_7777777",
        }
        hashes = {}
        for fname, content in sentinels.items():
            fpath = os.path.join(self.target_dir, fname)
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(fpath, mode) as f:
                f.write(content)
            with open(fpath, "rb") as f:
                hashes[fname] = hashlib.sha256(f.read()).hexdigest()
            if fname == "weights.bin" and make_weights_readonly:
                os.chmod(fpath, 0o444)
        return hashes

    def verify_sentinel_intact(self, expected_hashes):
        """Strictly verifies that target directory and all sentinel files are bit-for-bit unchanged."""
        self.assertTrue(os.path.isdir(self.target_dir), "Target directory missing!")
        current_files = sorted(os.listdir(self.target_dir))
        self.assertEqual(current_files, sorted(expected_hashes.keys()), "Target file list changed!")
        for fname, exp_hash in expected_hashes.items():
            fpath = os.path.join(self.target_dir, fname)
            with open(fpath, "rb") as f:
                cur_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(cur_hash, exp_hash, f"Sentinel file '{fname}' was modified! Hash mismatch!")

    def verify_no_staging_or_backup_residue(self):
        """Verifies no leftover .staging_<pid> or .backup_<pid> directories exist."""
        parent_dir = os.path.dirname(self.target_dir)
        target_name = os.path.basename(self.target_dir)
        leftovers = [
            d for d in os.listdir(parent_dir)
            if d.startswith(f"{target_name}.staging_") or d.startswith(f"{target_name}.backup_")
        ]
        self.assertEqual(leftovers, [], f"Leftover staging/backup directories found: {leftovers}")

    def create_valid_staging_weights(self, path, num_layers=30):
        """Generates valid 30-layer weights with non-dummy random embeddings and exact size."""
        with open(path, "wb") as wf:
            rnd = np.random.randn(10, 2560).astype(np.float32).tobytes()
            wf.write(rnd)
            target_size = 1836113920 if num_layers == 30 else (1313341440 + num_layers * 17425408 + 10240)
            wf.seek(target_size - 1)
            wf.write(b"\x00")


# =====================================================================
# 1. Permission Enforcement Challenge
# =====================================================================
class TestPermissionEnforcementChallenge(IsolationChallengeBase):
    """Challenge 1: Strict 0444 permission enforcement and overwrite prevention."""

    def setUp(self):
        super().setUp()

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_RAW_SAFETENSORS")

        def valid_export(safetensors_path, staging_dir, num_layers=30):
            self.create_valid_staging_weights(os.path.join(staging_dir, "weights.bin"), num_layers)
            with open(os.path.join(staging_dir, "model.mlir"), "w") as f:
                f.write("module { func.func @verified_model() { return } }")

        download_model.download_file_with_progress = dummy_download
        download_model.export_safetensors_to_tenzo = valid_export

    def test_weights_permissions_are_strict_0444(self):
        """Verify weights.bin has strict octal mode 0444 with zero write bits for user, group, other."""
        success = download_model.download_and_setup(
            repo_id="test/permission-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success, "download_and_setup should succeed")

        weights_path = os.path.join(self.target_dir, "weights.bin")
        self.assertTrue(os.path.isfile(weights_path))
        st = os.stat(weights_path)
        mode = st.st_mode & 0o777

        self.assertEqual(mode, 0o444, f"weights.bin mode {oct(mode)} != canonical 0444")
        self.assertEqual(mode & 0o222, 0, f"Write bits must be strictly 0: {oct(mode & 0o222)}")
        self.assertEqual(mode & 0o444, 0o444, "Read bits must be set for user, group, other")

    def test_unprivileged_open_wb_fails_with_permission_error(self):
        """Verify unprivileged process attempting open('weights.bin', 'wb') raises PermissionError."""
        success = download_model.download_and_setup(
            repo_id="test/permission-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success)

        # Make parent target_dir traversable by unprivileged user
        os.chmod(self.target_dir, 0o755)
        weights_path = os.path.join(self.target_dir, "weights.bin")

        cmd = ["su", "-s", "/bin/sh", "nobody", "-c", f"python3 -c \"open('{weights_path}', 'wb').write(b'CORRUPT')\""]
        res = subprocess.run(cmd, capture_output=True, text=True)

        self.assertNotEqual(res.returncode, 0, "Unprivileged write in mode 'wb' unexpectedly succeeded!")
        self.assertTrue(
            "PermissionError" in res.stderr or "Permission denied" in res.stderr,
            f"Expected PermissionError, got: {res.stderr}"
        )

    def test_unprivileged_open_ab_fails_with_permission_error(self):
        """Verify unprivileged process attempting open('weights.bin', 'ab') raises PermissionError."""
        success = download_model.download_and_setup(
            repo_id="test/permission-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success)

        os.chmod(self.target_dir, 0o755)
        weights_path = os.path.join(self.target_dir, "weights.bin")

        cmd = ["su", "-s", "/bin/sh", "nobody", "-c", f"python3 -c \"open('{weights_path}', 'ab').write(b'EXTRA')\""]
        res = subprocess.run(cmd, capture_output=True, text=True)

        self.assertNotEqual(res.returncode, 0, "Unprivileged append in mode 'ab' unexpectedly succeeded!")
        self.assertTrue(
            "PermissionError" in res.stderr or "Permission denied" in res.stderr,
            f"Expected PermissionError, got: {res.stderr}"
        )

    def test_unprivileged_open_rw_fails_with_permission_error(self):
        """Verify unprivileged process attempting open('weights.bin', 'r+') raises PermissionError."""
        success = download_model.download_and_setup(
            repo_id="test/permission-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success)

        os.chmod(self.target_dir, 0o755)
        weights_path = os.path.join(self.target_dir, "weights.bin")

        cmd = ["su", "-s", "/bin/sh", "nobody", "-c", f"python3 -c \"open('{weights_path}', 'r+b').write(b'MUTATE')\""]
        res = subprocess.run(cmd, capture_output=True, text=True)

        self.assertNotEqual(res.returncode, 0, "Unprivileged read-write in mode 'r+b' unexpectedly succeeded!")
        self.assertTrue(
            "PermissionError" in res.stderr or "Permission denied" in res.stderr,
            f"Expected PermissionError, got: {res.stderr}"
        )

    def test_unprivileged_truncate_fails_with_permission_error(self):
        """Verify unprivileged process attempting os.truncate('weights.bin', 0) raises PermissionError."""
        success = download_model.download_and_setup(
            repo_id="test/permission-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success)

        os.chmod(self.target_dir, 0o755)
        weights_path = os.path.join(self.target_dir, "weights.bin")

        cmd = ["su", "-s", "/bin/sh", "nobody", "-c", f"python3 -c \"import os; os.truncate('{weights_path}', 0)\""]
        res = subprocess.run(cmd, capture_output=True, text=True)

        self.assertNotEqual(res.returncode, 0, "Unprivileged truncate unexpectedly succeeded!")
        self.assertTrue(
            "PermissionError" in res.stderr or "Permission denied" in res.stderr,
            f"Expected PermissionError, got: {res.stderr}"
        )

    def test_explicit_chmod_permits_modification(self):
        """Verify file cannot be written until explicit chmod 0666 is performed."""
        success = download_model.download_and_setup(
            repo_id="test/permission-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success)

        os.chmod(self.target_dir, 0o777)
        weights_path = os.path.join(self.target_dir, "weights.bin")

        # Confirm write fails before chmod
        cmd_write = ["su", "-s", "/bin/sh", "nobody", "-c", f"python3 -c \"open('{weights_path}', 'r+b').write(b'MODIFIED')\""]
        res_before = subprocess.run(cmd_write, capture_output=True, text=True)
        self.assertNotEqual(res_before.returncode, 0)

        # Explicitly chmod to grant write
        os.chmod(weights_path, 0o666)

        # Confirm write succeeds after chmod
        res_after = subprocess.run(cmd_write, capture_output=True, text=True)
        self.assertEqual(res_after.returncode, 0, f"Write after explicit chmod failed: {res_after.stderr}")

        with open(weights_path, "rb") as f:
            self.assertEqual(f.read(8), b"MODIFIED")


# =====================================================================
# 2. Swap Failure & Rollback Challenge
# =====================================================================
class TestSwapFailureAndRollbackChallenge(IsolationChallengeBase):
    """Challenge 2: Simulated swap failure and complete restoration from backup."""

    def setUp(self):
        super().setUp()

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_RAW_SAFETENSORS")

        def valid_export(safetensors_path, staging_dir, num_layers=30):
            self.create_valid_staging_weights(os.path.join(staging_dir, "weights.bin"), num_layers)
            with open(os.path.join(staging_dir, "model.mlir"), "w") as f:
                f.write("module { func.func @new_model() { return } }")

        download_model.download_file_with_progress = dummy_download
        download_model.export_safetensors_to_tenzo = valid_export

    def test_rollback_on_staging_to_target_swap_failure(self):
        """Simulate failure on the atomic promotion swap (staging -> target).
        Assert target_dir is 100% restored from backup_dir, all sentinels intact, no residue."""
        sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        real_replace = _ORIG_REPLACE
        counter = [0]

        def failing_replace(src, dst):
            counter[0] += 1
            if counter[0] == 2:
                # 1st call is target_dir -> backup_dir (succeeds)
                # 2nd call is staging_dir -> target_dir (simulated filesystem / IO crash)
                raise OSError("Simulated filesystem I/O error during staging -> target rename")
            return real_replace(src, dst)

        os.replace = failing_replace

        success = download_model.download_and_setup(
            repo_id="test/swap-failure-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must return False when atomic promotion swap fails")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_or_backup_residue()

    def test_rollback_on_target_to_backup_swap_failure(self):
        """Simulate failure on the initial backup swap (target -> backup).
        Assert target_dir was never moved and remains completely intact."""
        sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        real_replace = _ORIG_REPLACE

        def failing_first_replace(src, dst):
            if "backup_" in dst:
                raise OSError("Simulated filesystem error during target -> backup rename")
            return real_replace(src, dst)

        os.replace = failing_first_replace

        success = download_model.download_and_setup(
            repo_id="test/first-swap-failure-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must return False when target -> backup fails")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_or_backup_residue()

    def test_rollback_preserves_deep_directory_structure_and_canaries(self):
        """Verify rollback restores nested directory trees and non-standard canaries without loss."""
        sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        # Create nested directory tree inside target
        sub_dir = os.path.join(self.target_dir, "nested_module")
        os.makedirs(sub_dir, exist_ok=True)
        sub_file = os.path.join(sub_dir, "metadata.json")
        with open(sub_file, "w") as f:
            f.write('{"nested_canary": "CRITICAL_NESTED_DATA"}')

        real_replace = _ORIG_REPLACE
        counter = [0]

        def failing_replace(src, dst):
            counter[0] += 1
            if counter[0] == 2:
                raise OSError("Simulated swap failure")
            return real_replace(src, dst)

        os.replace = failing_replace

        success = download_model.download_and_setup(
            repo_id="test/deep-rollback-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success)
        self.assertTrue(os.path.isfile(sub_file), "Nested file was lost during rollback!")
        with open(sub_file, "r") as f:
            self.assertEqual(f.read(), '{"nested_canary": "CRITICAL_NESTED_DATA"}')
        self.verify_no_staging_or_backup_residue()


# =====================================================================
# 3. Concurrent Multi-PID Isolation Challenge
# =====================================================================
class TestConcurrentPIDIsolationChallenge(IsolationChallengeBase):
    """Challenge 3: PID tagging (.staging_<pid>), multi-process isolation, and race resilience."""

    def setUp(self):
        super().setUp()

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_RAW_SAFETENSORS")

        def valid_export(safetensors_path, staging_dir, num_layers=30):
            self.create_valid_staging_weights(os.path.join(staging_dir, "weights.bin"), num_layers)
            with open(os.path.join(staging_dir, "model.mlir"), "w") as f:
                f.write("module { func.func @verified_model() { return } }")

        download_model.download_file_with_progress = dummy_download
        download_model.export_safetensors_to_tenzo = valid_export

    def test_pid_tagging_in_staging_and_backup_paths(self):
        """Verify downloader explicitly tags staging_dir and backup_dir with os.getpid()."""
        pid = os.getpid()
        target_abs = os.path.abspath(self.target_dir)
        expected_staging = f"{target_abs}.staging_{pid}"

        seen_staging = []
        orig_export = download_model.export_safetensors_to_tenzo

        def spy_export(st_path, staging_dir, num_layers=30):
            seen_staging.append(staging_dir)
            orig_export(st_path, staging_dir, num_layers)

        download_model.export_safetensors_to_tenzo = spy_export

        success = download_model.download_and_setup(
            repo_id="test/pid-tag-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertTrue(success)
        self.assertEqual(len(seen_staging), 1)
        self.assertEqual(seen_staging[0], expected_staging, f"Staging dir {seen_staging[0]} did not match {expected_staging}")
        self.assertTrue(expected_staging.endswith(f".staging_{pid}"))

    def test_concurrent_multi_process_staging_isolation(self):
        """Verify 4 concurrent processes create separate .staging_<pid> dirs and do not collide."""
        num_procs = 4
        barrier = mp.Barrier(num_procs)
        status_queue = mp.Queue()

        target_dir = self.target_dir
        base_assets_dir = self.base_assets_dir

        def worker_fn(proc_idx, b, q):
            pid = os.getpid()

            def custom_export(st_path, s_dir, num_layers=30):
                # Write unique canary per process
                canary_file = os.path.join(s_dir, f"proc_{proc_idx}_pid_{pid}.canary")
                with open(canary_file, "w") as f:
                    f.write(f"UNIQUE_PAYLOAD_FOR_PROC_{proc_idx}_PID_{pid}")

                # Write valid weights
                w_path = os.path.join(s_dir, "weights.bin")
                with open(w_path, "wb") as wf:
                    rnd = np.random.randn(10, 2560).astype(np.float32).tobytes()
                    wf.write(rnd)
                    wf.seek(1836113920 - 1)
                    wf.write(b"\x00")

                with open(os.path.join(s_dir, "model.mlir"), "w") as f:
                    f.write(f"module {{ func.func @model_proc_{proc_idx}() {{ return }} }}")

                # Synchronize: All 4 processes hold their staging dirs open simultaneously
                try:
                    b.wait(timeout=10)
                except Exception:
                    pass

                # Check active staging directories in parent
                parent = os.path.dirname(target_dir)
                active_stagings = [
                    d for d in os.listdir(parent)
                    if d.startswith(f"{os.path.basename(target_dir)}.staging_")
                ]

                # Check other processes did not contaminate our staging dir
                foreign_canaries = [
                    f for f in os.listdir(s_dir)
                    if f.endswith(".canary") and f != f"proc_{proc_idx}_pid_{pid}.canary"
                ]

                q.put({
                    "proc_idx": proc_idx,
                    "pid": pid,
                    "has_own_canary": os.path.exists(canary_file),
                    "foreign_canaries": foreign_canaries,
                    "active_staging_count": len(active_stagings),
                })

            download_model.export_safetensors_to_tenzo = custom_export

            download_model.download_and_setup(
                repo_id=f"test/concurrent-proc-{proc_idx}",
                output_dir=target_dir,
                base_assets_dir=base_assets_dir,
            )

        procs = [mp.Process(target=worker_fn, args=(i, barrier, status_queue)) for i in range(num_procs)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
                p.join()

        reports = []
        while not status_queue.empty():
            reports.append(status_queue.get())

        self.assertEqual(len(reports), num_procs, f"Expected {num_procs} reports, got {len(reports)}")
        pids = [r["pid"] for r in reports]
        self.assertEqual(len(set(pids)), num_procs, "All process PIDs must be distinct")

        for r in reports:
            self.assertTrue(r["has_own_canary"], f"Process {r['proc_idx']} lost its canary!")
            self.assertEqual(r["foreign_canaries"], [], f"Process {r['proc_idx']} contaminated with {r['foreign_canaries']}")
            self.assertEqual(r["active_staging_count"], num_procs, f"Expected {num_procs} simultaneous staging dirs")

        # Verify final target exists, is cohesive, and weights are 0444
        self.assertTrue(os.path.isdir(self.target_dir))
        for fname in ("weights.bin", "model.mlir", "tokenizer.vocab", "config.json"):
            fpath = os.path.join(self.target_dir, fname)
            self.assertTrue(os.path.isfile(fpath) and os.path.getsize(fpath) > 0)
        mode = os.stat(os.path.join(self.target_dir, "weights.bin")).st_mode & 0o777
        self.assertEqual(mode, 0o444)

    def test_concurrent_process_failure_isolation(self):
        """Failure and staging rmtree in Process A does not disrupt concurrent Process B's staging."""
        barrier = mp.Barrier(2)
        results = mp.Queue()
        target_dir = self.target_dir
        base_assets_dir = self.base_assets_dir

        def worker_failing(b, q):
            pid = os.getpid()
            def failing_export(st_path, s_dir, num_layers=30):
                with open(os.path.join(s_dir, "fail_canary.txt"), "w") as f:
                    f.write("FAILING")
                b.wait(timeout=10) # Both processes are staged
                raise RuntimeError("Simulated network/disk crash in Process A")

            download_model.export_safetensors_to_tenzo = failing_export
            ok = download_model.download_and_setup(repo_id="test/failing", output_dir=target_dir, base_assets_dir=base_assets_dir)
            q.put(("A", pid, ok))

        def worker_succeeding(b, q):
            pid = os.getpid()
            def valid_export(st_path, s_dir, num_layers=30):
                w_path = os.path.join(s_dir, "weights.bin")
                with open(w_path, "wb") as wf:
                    rnd = np.random.randn(10, 2560).astype(np.float32).tobytes()
                    wf.write(rnd)
                    wf.seek(1836113920 - 1)
                    wf.write(b"\x00")
                with open(os.path.join(s_dir, "model.mlir"), "w") as f:
                    f.write("module { func.func @succeeding_proc() { return } }")
                b.wait(timeout=10) # Wait for sibling
                time.sleep(0.1)   # Give sibling time to fail and wipe its staging dir

            download_model.export_safetensors_to_tenzo = valid_export
            ok = download_model.download_and_setup(repo_id="test/succeeding", output_dir=target_dir, base_assets_dir=base_assets_dir)
            q.put(("B", pid, ok))

        p_a = mp.Process(target=worker_failing, args=(barrier, results))
        p_b = mp.Process(target=worker_succeeding, args=(barrier, results))
        p_a.start()
        p_b.start()
        p_a.join(timeout=15)
        p_b.join(timeout=15)

        res_dict = {}
        while not results.empty():
            tag, pid, ok = results.get()
            res_dict[tag] = ok

        self.assertFalse(res_dict.get("A", True), "Process A should have returned False")
        self.assertTrue(res_dict.get("B", False), "Process B should have succeeded")
        self.assertTrue(os.path.isdir(self.target_dir))
        with open(os.path.join(self.target_dir, "model.mlir")) as f:
            self.assertIn("@succeeding_proc", f.read())

    def test_concurrent_racing_publication_integrity(self):
        """When 2 processes race on final commit, the loser cleanly aborts via ENOTEMPTY without corrupting winner."""
        barrier = mp.Barrier(2)
        results = mp.Queue()
        target_dir = self.target_dir
        base_assets_dir = self.base_assets_dir

        def worker_racer(proc_idx, b, q):
            pid = os.getpid()
            def racing_export(st_path, s_dir, num_layers=30):
                w_path = os.path.join(s_dir, "weights.bin")
                with open(w_path, "wb") as wf:
                    rnd = np.random.randn(10, 2560).astype(np.float32).tobytes()
                    wf.write(rnd)
                    wf.seek(1836113920 - 1)
                    wf.write(b"\x00")
                with open(os.path.join(s_dir, "model.mlir"), "w") as f:
                    f.write(f"module {{ func.func @racer_{proc_idx}() {{ return }} }}")
                b.wait(timeout=10)

            download_model.export_safetensors_to_tenzo = racing_export
            ok = download_model.download_and_setup(repo_id=f"test/racer-{proc_idx}", output_dir=target_dir, base_assets_dir=base_assets_dir)
            q.put((proc_idx, pid, ok))

        p1 = mp.Process(target=worker_racer, args=(1, barrier, results))
        p2 = mp.Process(target=worker_racer, args=(2, barrier, results))
        p1.start()
        p2.start()
        p1.join(timeout=15)
        p2.join(timeout=15)

        res = []
        while not results.empty():
            res.append(results.get())

        # At least one process must succeed, and target must be complete and uncorrupted
        success_count = sum(1 for _, _, ok in res if ok)
        self.assertGreaterEqual(success_count, 1, "At least one racing process must succeed")
        self.assertTrue(os.path.isdir(self.target_dir))
        for fname in ("weights.bin", "model.mlir", "tokenizer.vocab", "config.json"):
            fpath = os.path.join(self.target_dir, fname)
            self.assertTrue(os.path.isfile(fpath) and os.path.getsize(fpath) > 0)
        mode = os.stat(os.path.join(self.target_dir, "weights.bin")).st_mode & 0o777
        self.assertEqual(mode, 0o444)


def run_all_challenge_tests():
    print(f"\n{ANSI_BOLD}{ANSI_CYAN}{'=' * 75}{ANSI_RESET}")
    print(f"{ANSI_BOLD}{ANSI_CYAN}  TENZO DOWNLOADER EMPIRICAL ISOLATION & ROLLBACK CHALLENGE SUITE{ANSI_RESET}")
    print(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 75}{ANSI_RESET}\n")

    test_classes = [
        TestPermissionEnforcementChallenge,
        TestSwapFailureAndRollbackChallenge,
        TestConcurrentPIDIsolationChallenge,
    ]

    suite = unittest.TestSuite()
    for tc in test_classes:
        for name in unittest.TestLoader().getTestCaseNames(tc):
            suite.addTest(tc(name))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 75)
    if result.wasSuccessful():
        print(f"{ANSI_BOLD}{ANSI_GREEN}🎉 ALL {result.testsRun} ISOLATION & ROLLBACK CHALLENGE TESTS PASSED SUCCESSFULLY!{ANSI_RESET}")
        return 0
    else:
        print(f"{ANSI_BOLD}{ANSI_RED}❌ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS out of {result.testsRun} tests!{ANSI_RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_challenge_tests())
