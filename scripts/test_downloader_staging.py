#!/usr/bin/env python3
"""
Comprehensive Unit & Integration Test Suite for Downloader Atomic Staging
Tests Requirement R1: Cohesive atomic staging, pre-promotion verification,
permission enforcement, and rollback mechanics in scripts/download_model.py.
"""

import os
import sys
import shutil
import tempfile
import stat
import hashlib
import unittest
import time
import subprocess
import multiprocessing as mp
import numpy as np

# Add project root and scripts dir to path
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

def print_header(title: str):
    print(f"\n{ANSI_BOLD}{ANSI_CYAN}{'=' * 70}{ANSI_RESET}")
    print(f"{ANSI_BOLD}{ANSI_CYAN}  {title}{ANSI_RESET}")
    print(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 70}{ANSI_RESET}\n")

def print_pass(name: str):
    print(f"  {ANSI_GREEN}✅ [PASS]{ANSI_RESET} {name}")

def print_fail(name: str, detail: str = ""):
    print(f"  {ANSI_RED}❌ [FAIL]{ANSI_RESET} {name} {detail}")


_ORIG_DOWNLOAD = download_model.download_file_with_progress
_ORIG_EXPORT = download_model.export_safetensors_to_tenzo
_ORIG_REPLACE = os.replace


class DownloaderStagingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="tenzo_downloader_test_")
        self.target_dir = os.path.join(self.tmp_dir, "model_target")
        self.base_assets_dir = os.path.join(self.tmp_dir, "base_assets")
        os.makedirs(self.base_assets_dir, exist_ok=True)

        # Create valid base assets fixtures
        self.fixture_vocab = os.path.join(self.base_assets_dir, "tokenizer.vocab")
        with open(self.fixture_vocab, "wb") as f:
            f.write(b"FIXTURE_VOCAB_TOKEN_DATA_" * 100)

        self.fixture_mlir = os.path.join(self.base_assets_dir, "model.mlir")
        with open(self.fixture_mlir, "w", encoding="utf-8") as f:
            f.write("module { func.func @main() { return } }")

        self.fixture_config = os.path.join(self.base_assets_dir, "config.json")
        with open(self.fixture_config, "w", encoding="utf-8") as f:
            f.write('{"model_type": "bitnet", "num_hidden_layers": 30}')

    def tearDown(self):
        # Reset any monkeypatched functions
        download_model.download_file_with_progress = _ORIG_DOWNLOAD
        download_model.export_safetensors_to_tenzo = _ORIG_EXPORT
        if hasattr(os, "replace"):
            os.replace = _ORIG_REPLACE
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def create_sentinel_target(self):
        """Creates pre-existing sentinel model in target_dir to verify it is untouched."""
        os.makedirs(self.target_dir, exist_ok=True)
        sentinels = {
            "weights.bin": b"SENTINEL_ORIGINAL_WEIGHTS_BINARY_DATA_1234567890",
            "model.mlir": "module { // SENTINEL ORIGINAL MLIR }",
            "tokenizer.vocab": b"SENTINEL_ORIGINAL_VOCAB_DATA",
            "config.json": '{"sentinel": "ORIGINAL_CONFIG"}',
        }
        hashes = {}
        for fname, content in sentinels.items():
            fpath = os.path.join(self.target_dir, fname)
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(fpath, mode) as f:
                f.write(content)
            with open(fpath, "rb") as f:
                hashes[fname] = hashlib.sha256(f.read()).hexdigest()
        return hashes

    def verify_sentinel_intact(self, expected_hashes):
        """Verifies that all files in target_dir have identical hashes and no files were added/removed."""
        self.assertTrue(os.path.isdir(self.target_dir), "Target directory was removed!")
        current_files = sorted(os.listdir(self.target_dir))
        self.assertEqual(current_files, sorted(expected_hashes.keys()), "Target directory file list changed!")
        for fname, exp_hash in expected_hashes.items():
            fpath = os.path.join(self.target_dir, fname)
            with open(fpath, "rb") as f:
                cur_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(cur_hash, exp_hash, f"Sentinel file '{fname}' was modified!")

    def verify_no_staging_residue(self):
        """Verifies that no staging or backup directories remain in the parent directory."""
        parent_dir = os.path.dirname(self.target_dir)
        target_name = os.path.basename(self.target_dir)
        leftovers = [
            d for d in os.listdir(parent_dir)
            if d.startswith(f"{target_name}.staging_") or d.startswith(f"{target_name}.backup_")
        ]
        self.assertEqual(leftovers, [], f"Staging or backup directories were not cleaned up: {leftovers}")

    def create_valid_staging_weights(self, path, num_layers=30, uniform=False, size_override=None, corrupt_val=None):
        """Helper to create sparse weights with genuine embeddings and valid size."""
        with open(path, "wb") as wf:
            if corrupt_val is not None:
                bad_arr = np.full((10, 2560), corrupt_val, dtype=np.float32)
                wf.write(bad_arr.tobytes())
            elif uniform:
                # Uniform zero embeddings (std = 0 <= 1e-4)
                wf.write(b"\x00" * (2560 * 4 * 10))
            else:
                # Genuine random float32 embeddings (std ~ 1.0 > 1e-4)
                rnd = np.random.randn(10, 2560).astype(np.float32).tobytes()
                wf.write(rnd)
            target_size = size_override if size_override is not None else 1836113920
            wf.seek(target_size - 1)
            wf.write(b"\x00")


class TestNegativeControl(DownloaderStagingTestCase):
    """Test Suite 1: Negative Control & Failure Resistance"""

    def test_simulated_download_network_failure(self):
        """Simulate network failure during download. Assert target untouched, staging cleaned."""
        sentinel_hashes = self.create_sentinel_target()

        def failing_download(url, dest_path):
            # Create partial file before failing
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"PARTIAL_CHUNK_BEFORE_NETWORK_RESET")
            raise ConnectionResetError("Simulated network drop mid-transfer")

        download_model.download_file_with_progress = failing_download

        success = download_model.download_and_setup(
            repo_id="test/failing-network-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "download_and_setup should return False on network failure")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()

    def test_simulated_export_conversion_failure(self):
        """Simulate exception during SafeTensors weight conversion. Assert target untouched, staging cleaned."""
        sentinel_hashes = self.create_sentinel_target()

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"RAW_SAFETENSORS_HEADER_DATA")

        def failing_export(safetensors_path, output_dir, num_layers=30):
            # Create partial weights before failing
            weights_path = os.path.join(output_dir, "weights.bin")
            with open(weights_path, "wb") as f:
                f.write(b"PARTIAL_CORRUPT_WEIGHTS")
            raise ValueError("Corrupted tensor header in safetensors")

        download_model.download_file_with_progress = dummy_download
        download_model.export_safetensors_to_tenzo = failing_export

        success = download_model.download_and_setup(
            repo_id="test/corrupt-export-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "download_and_setup should return False on export failure")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()

    def test_interruption_handling(self):
        """Simulate KeyboardInterrupt during processing. Assert target untouched, staging cleaned."""
        sentinel_hashes = self.create_sentinel_target()

        def interrupted_download(url, dest_path):
            raise KeyboardInterrupt("Simulated Ctrl-C interruption")

        download_model.download_file_with_progress = interrupted_download

        success = download_model.download_and_setup(
            repo_id="test/interrupted-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "download_and_setup should return False on interruption")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()

    def test_cli_invocation_negative_control(self):
        """Invoke scripts/download_model.py via subprocess CLI with invalid repo. Assert exit code 1 and target intact."""
        import subprocess
        sentinel_hashes = self.create_sentinel_target()

        cli_script = os.path.join(SCRIPTS_DIR, "download_model.py")
        cmd = [sys.executable, cli_script, "invalid-dummy-org/nonexistent-model", self.target_dir]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.assertEqual(proc.returncode, 1, f"CLI expected exit code 1, got {proc.returncode}")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()



class TestCohesiveSetValidation(DownloaderStagingTestCase):
    """Test Suite 2: Cohesive Set Pre-Promotion Validation"""

    def setUp(self):
        super().setUp()
        self.sentinel_hashes = self.create_sentinel_target()

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_ST")
        download_model.download_file_with_progress = dummy_download

    def test_missing_mlir_rejection(self):
        """Pre-promotion verification must reject missing or empty model.mlir."""
        def export_missing_mlir(safetensors_path, staging_dir, num_layers=30):
            # Write valid weights
            self.create_valid_staging_weights(os.path.join(staging_dir, "weights.bin"))
            # Deliberately remove model.mlir from staging
            mlir_p = os.path.join(staging_dir, "model.mlir")
            if os.path.exists(mlir_p):
                os.remove(mlir_p)

        download_model.export_safetensors_to_tenzo = export_missing_mlir

        success = download_model.download_and_setup(
            repo_id="test/missing-mlir",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must reject staging with missing model.mlir")
        self.verify_sentinel_intact(self.sentinel_hashes)
        self.verify_no_staging_residue()

    def test_missing_tokenizer_vocab_rejection(self):
        """Pre-promotion verification must reject missing or 0-byte tokenizer.vocab."""
        def export_empty_vocab(safetensors_path, staging_dir, num_layers=30):
            self.create_valid_staging_weights(os.path.join(staging_dir, "weights.bin"))
            # Truncate tokenizer.vocab to 0 bytes
            vocab_p = os.path.join(staging_dir, "tokenizer.vocab")
            with open(vocab_p, "wb") as f:
                pass  # 0 bytes

        download_model.export_safetensors_to_tenzo = export_empty_vocab

        success = download_model.download_and_setup(
            repo_id="test/empty-vocab",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must reject staging with 0-byte tokenizer.vocab")
        self.verify_sentinel_intact(self.sentinel_hashes)
        self.verify_no_staging_residue()

    def test_missing_config_json_rejection(self):
        """Pre-promotion verification must reject missing config.json."""
        def export_missing_config(safetensors_path, staging_dir, num_layers=30):
            self.create_valid_staging_weights(os.path.join(staging_dir, "weights.bin"))
            cfg_p = os.path.join(staging_dir, "config.json")
            if os.path.exists(cfg_p):
                os.remove(cfg_p)

        download_model.export_safetensors_to_tenzo = export_missing_config

        success = download_model.download_and_setup(
            repo_id="test/missing-config",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must reject staging with missing config.json")
        self.verify_sentinel_intact(self.sentinel_hashes)
        self.verify_no_staging_residue()

    def test_undersized_weights_rejection(self):
        """Pre-promotion verification must reject weights < 1,836,113,920 bytes for 30 layers."""
        def export_undersized(safetensors_path, staging_dir, num_layers=30):
            # Only 50 MB
            self.create_valid_staging_weights(
                os.path.join(staging_dir, "weights.bin"),
                size_override=50 * 1024 * 1024
            )

        download_model.export_safetensors_to_tenzo = export_undersized

        success = download_model.download_and_setup(
            repo_id="test/undersized-weights",
            output_dir=self.target_dir,
            num_layers=30,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must reject undersized weights (< 1,836,113,920 bytes)")
        self.verify_sentinel_intact(self.sentinel_hashes)
        self.verify_no_staging_residue()

    def test_dummy_embeddings_rejection(self):
        """Pre-promotion verification must reject weights with uniform embeddings (std <= 1e-4)."""
        def export_dummy_embeds(safetensors_path, staging_dir, num_layers=30):
            self.create_valid_staging_weights(
                os.path.join(staging_dir, "weights.bin"),
                uniform=True
            )

        download_model.export_safetensors_to_tenzo = export_dummy_embeds

        success = download_model.download_and_setup(
            repo_id="test/dummy-embeddings",
            output_dir=self.target_dir,
            num_layers=30,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must reject weights with dummy/uniform embeddings")
        self.verify_sentinel_intact(self.sentinel_hashes)
        self.verify_no_staging_residue()

    def test_nan_inf_embeddings_rejection(self):
        """Pre-promotion verification must reject weights with NaN or Inf embeddings."""
        for bad_val, name in [(np.nan, "NaN"), (np.inf, "Inf"), (-np.inf, "-Inf")]:
            with self.subTest(val=name):
                def export_corrupt_embeds(safetensors_path, staging_dir, num_layers=30):
                    self.create_valid_staging_weights(
                        os.path.join(staging_dir, "weights.bin"),
                        corrupt_val=bad_val
                    )

                download_model.export_safetensors_to_tenzo = export_corrupt_embeds

                success = download_model.download_and_setup(
                    repo_id=f"test/corrupt-{name.lower()}-embeddings",
                    output_dir=self.target_dir,
                    num_layers=30,
                    base_assets_dir=self.base_assets_dir,
                )

                self.assertFalse(success, f"Must reject weights with {name} embeddings")
                self.verify_sentinel_intact(self.sentinel_hashes)
                self.verify_no_staging_residue()


class TestAtomicPromotionAndRollback(DownloaderStagingTestCase):
    """Test Suite 3: Successful Transaction Atomic Promotion & Rollback Mechanics"""

    def setUp(self):
        super().setUp()
        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_ST")

        def valid_export(safetensors_path, staging_dir, num_layers=30):
            self.create_valid_staging_weights(os.path.join(staging_dir, "weights.bin"))
            # Ensure model.mlir exists in staging
            mlir_p = os.path.join(staging_dir, "model.mlir")
            with open(mlir_p, "w", encoding="utf-8") as f:
                f.write("module { func.func @promoted_model() { return } }")

        download_model.download_file_with_progress = dummy_download
        download_model.export_safetensors_to_tenzo = valid_export

    def test_successful_promotion_over_existing_target(self):
        """Atomically replace pre-existing target. Verify permissions, content, and cleanup."""
        self.create_sentinel_target()

        success = download_model.download_and_setup(
            repo_id="test/valid-promoted-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertTrue(success, "download_and_setup should succeed on valid model")
        self.verify_no_staging_residue()

        # Check cohesive files exist in target
        for fname in ("weights.bin", "model.mlir", "tokenizer.vocab", "config.json"):
            fpath = os.path.join(self.target_dir, fname)
            self.assertTrue(os.path.isfile(fpath), f"File {fname} missing in promoted target!")
            self.assertGreater(os.path.getsize(fpath), 0, f"File {fname} is empty in promoted target!")

        # Verify weights size and permissions
        weights_path = os.path.join(self.target_dir, "weights.bin")
        self.assertEqual(os.path.getsize(weights_path), 1836113920)
        mode = os.stat(weights_path).st_mode & 0o777
        self.assertEqual(mode, 0o444, f"weights.bin permissions {oct(mode)} != canonical 0444")

        # Verify new content replaced sentinel
        with open(os.path.join(self.target_dir, "model.mlir"), "r") as f:
            self.assertIn("@promoted_model", f.read())

    def test_successful_promotion_fresh_target(self):
        """Atomically create target when target directory does not initially exist."""
        self.assertFalse(os.path.exists(self.target_dir))

        success = download_model.download_and_setup(
            repo_id="test/fresh-target-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertTrue(success, "download_and_setup should succeed on fresh target")
        self.assertTrue(os.path.isdir(self.target_dir))
        self.verify_no_staging_residue()

        weights_path = os.path.join(self.target_dir, "weights.bin")
        self.assertTrue(os.path.isfile(weights_path))
        self.assertEqual(os.path.getsize(weights_path), 1836113920)
        mode = os.stat(weights_path).st_mode & 0o777
        self.assertEqual(mode, 0o444)

    def test_atomic_rollback_on_swap_failure(self):
        """Simulate failure during final os.replace. Verify backup is restored to target."""
        sentinel_hashes = self.create_sentinel_target()

        # Mock os.replace to fail on the second call (staging -> target)
        real_replace = _ORIG_REPLACE
        call_count = [0]

        def failing_replace(src, dst):
            call_count[0] += 1
            if call_count[0] == 2:
                # First call is target -> backup, second is staging -> target
                raise OSError("Simulated filesystem error during atomic staging rename")
            return real_replace(src, dst)

        os.replace = failing_replace

        success = download_model.download_and_setup(
            repo_id="test/swap-failure-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must return False when swap fails")
        # Assert backup was rolled back to target_dir and sentinels are completely intact
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()

    def test_rollback_on_initial_backup_replace_failure(self):
        """Simulate failure during initial target -> backup replace. Assert target untouched."""
        sentinel_hashes = self.create_sentinel_target()
        real_replace = _ORIG_REPLACE

        def failing_first_replace(src, dst):
            if "backup_" in dst:
                raise OSError("Simulated filesystem error during target -> backup replace")
            return real_replace(src, dst)

        os.replace = failing_first_replace

        success = download_model.download_and_setup(
            repo_id="test/first-swap-failure-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must return False when first swap fails")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()


class TestPermissionEnforcement(DownloaderStagingTestCase):
    """Test Suite 4: Strict 0444 Permission Enforcement & Overwrite Prevention"""

    def setUp(self):
        super().setUp()
        os.chmod(self.tmp_dir, 0o777)

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_ST")

        def valid_export(safetensors_path, staging_dir, num_layers=30):
            self.create_valid_staging_weights(os.path.join(staging_dir, "weights.bin"))
            mlir_p = os.path.join(staging_dir, "model.mlir")
            with open(mlir_p, "w", encoding="utf-8") as f:
                f.write("module { func.func @promoted_model() { return } }")

        download_model.download_file_with_progress = dummy_download
        download_model.export_safetensors_to_tenzo = valid_export

    def test_weights_mode_is_strict_0444(self):
        """Verify weights.bin has strict mode 0444 and zero write bits."""
        success = download_model.download_and_setup(
            repo_id="test/perm-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success)
        weights_path = os.path.join(self.target_dir, "weights.bin")
        mode = os.stat(weights_path).st_mode & 0o777
        self.assertEqual(mode, 0o444, f"weights.bin permissions {oct(mode)} != 0444")
        self.assertEqual(mode & 0o222, 0, "Write bits (user, group, other) must all be zero")

    def test_unprivileged_overwrite_prevented_with_permission_error(self):
        """Unprivileged process cannot overwrite, append, or truncate weights.bin without chmod."""
        success = download_model.download_and_setup(
            repo_id="test/perm-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success)
        os.chmod(self.target_dir, 0o755)
        weights_path = os.path.join(self.target_dir, "weights.bin")

        for op in [
            f"open('{weights_path}', 'wb').write(b'hack')",
            f"open('{weights_path}', 'ab').write(b'hack')",
            f"open('{weights_path}', 'r+b').write(b'hack')",
            f"import os; os.truncate('{weights_path}', 0)",
        ]:
            cmd = ["su", "-s", "/bin/sh", "nobody", "-c", f"python3 -c \"{op}\""]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0, f"Unprivileged op unexpectedly succeeded: {op}")
            self.assertTrue(
                "PermissionError" in res.stderr or "Permission denied" in res.stderr,
                f"Expected PermissionError for {op}, got: {res.stderr}"
            )

    def test_explicit_chmod_required_for_modification(self):
        """Modification is only allowed after explicit chmod grants write permission."""
        success = download_model.download_and_setup(
            repo_id="test/perm-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success)
        os.chmod(self.target_dir, 0o777)
        weights_path = os.path.join(self.target_dir, "weights.bin")

        # Before chmod: write fails
        cmd = ["su", "-s", "/bin/sh", "nobody", "-c", f"python3 -c \"open('{weights_path}', 'r+b').write(b'PATCHED')\""]
        res_before = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res_before.returncode, 0)

        # Explicit chmod
        os.chmod(weights_path, 0o666)

        # After chmod: write succeeds
        res_after = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res_after.returncode, 0, f"Write after chmod failed: {res_after.stderr}")
        with open(weights_path, "rb") as f:
            self.assertEqual(f.read(7), b"PATCHED")


class TestConcurrentPIDIsolation(DownloaderStagingTestCase):
    """Test Suite 5: Multi-Process Staging Isolation & PID Tagging"""

    def setUp(self):
        super().setUp()

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_ST")

        def valid_export(safetensors_path, staging_dir, num_layers=30):
            self.create_valid_staging_weights(os.path.join(staging_dir, "weights.bin"))
            mlir_p = os.path.join(staging_dir, "model.mlir")
            with open(mlir_p, "w", encoding="utf-8") as f:
                f.write("module { func.func @promoted_model() { return } }")

        download_model.download_file_with_progress = dummy_download
        download_model.export_safetensors_to_tenzo = valid_export

    def test_staging_and_backup_directory_pid_tagging(self):
        """Verify staging and backup directory paths strictly embed current PID."""
        pid = os.getpid()
        expected_staging = f"{os.path.abspath(self.target_dir)}.staging_{pid}"

        observed_staging = []
        orig_export = download_model.export_safetensors_to_tenzo
        def spy_export(st_path, staging_dir, num_layers=30):
            observed_staging.append(staging_dir)
            orig_export(st_path, staging_dir, num_layers)

        download_model.export_safetensors_to_tenzo = spy_export

        success = download_model.download_and_setup(
            repo_id="test/pid-tag-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertTrue(success)
        self.assertEqual(len(observed_staging), 1)
        self.assertEqual(observed_staging[0], expected_staging)

    def test_concurrent_multi_process_staging_isolation(self):
        """Multiple concurrent processes create isolated .staging_<pid> dirs simultaneously without collision."""
        num_procs = 4
        barrier = mp.Barrier(num_procs)
        status_queue = mp.Queue()

        target_dir = self.target_dir
        base_assets_dir = self.base_assets_dir

        def worker_fn(proc_idx, b, q):
            pid = os.getpid()

            def custom_export(st_path, s_dir, num_layers=30):
                canary_file = os.path.join(s_dir, f"canary_proc_{proc_idx}.txt")
                with open(canary_file, "w") as f:
                    f.write(f"PROCESS_{proc_idx}_PID_{pid}")

                w_path = os.path.join(s_dir, "weights.bin")
                with open(w_path, "wb") as wf:
                    rnd = np.random.randn(10, 2560).astype(np.float32).tobytes()
                    wf.write(rnd)
                    wf.seek(1836113920 - 1)
                    wf.write(b"\x00")

                with open(os.path.join(s_dir, "model.mlir"), "w") as f:
                    f.write(f"module {{ func.func @model_proc_{proc_idx}() {{ return }} }}")

                try:
                    b.wait(timeout=10)
                except Exception:
                    pass

                parent = os.path.dirname(target_dir)
                staging_entries = [d for d in os.listdir(parent) if d.startswith(f"{os.path.basename(target_dir)}.staging_")]
                has_own_canary = os.path.exists(canary_file)

                q.put({
                    "proc_idx": proc_idx,
                    "pid": pid,
                    "has_own_canary": has_own_canary,
                    "active_staging_count": len(staging_entries)
                })

            download_model.export_safetensors_to_tenzo = custom_export
            download_model.download_and_setup(
                repo_id=f"test/concurrent-model-{proc_idx}",
                output_dir=target_dir,
                base_assets_dir=base_assets_dir
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

        self.assertEqual(len(reports), num_procs)
        pids = [r["pid"] for r in reports]
        self.assertEqual(len(set(pids)), num_procs, "All process PIDs must be distinct")

        for r in reports:
            self.assertTrue(r["has_own_canary"])
            self.assertEqual(r["active_staging_count"], num_procs)

        self.assertTrue(os.path.isdir(self.target_dir))
        weights_path = os.path.join(self.target_dir, "weights.bin")
        self.assertTrue(os.path.isfile(weights_path))
        self.assertEqual(os.stat(weights_path).st_mode & 0o777, 0o444)

    def test_concurrent_failure_isolation(self):
        """Failure in Process A does not impact concurrent Process B's staging."""
        barrier = mp.Barrier(2)
        results = mp.Queue()
        target_dir = self.target_dir
        base_assets_dir = self.base_assets_dir

        def worker_failing(b, q):
            pid = os.getpid()
            def failing_export(st_path, s_dir, num_layers=30):
                with open(os.path.join(s_dir, "failing_marker.txt"), "w") as f:
                    f.write("FAILING")
                b.wait(timeout=10)
                raise RuntimeError("Simulated failure in Process A")

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
                    f.write("module { func.func @succeeding() { return } }")
                b.wait(timeout=10)
                time.sleep(0.1)

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

        self.assertFalse(res_dict.get("A", True))
        self.assertTrue(res_dict.get("B", False))
        self.assertTrue(os.path.isdir(self.target_dir))
        with open(os.path.join(self.target_dir, "model.mlir")) as f:
            self.assertIn("@succeeding", f.read())


def run_all_tests():
    print_header("Tenzo Downloader Atomic Staging Verification Suite")

    suite = unittest.TestSuite()
    for tc in (
        TestNegativeControl,
        TestCohesiveSetValidation,
        TestAtomicPromotionAndRollback,
        TestPermissionEnforcement,
        TestConcurrentPIDIsolation,
    ):
        for name in unittest.TestLoader().getTestCaseNames(tc):
            suite.addTest(tc(name))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"{ANSI_BOLD}{ANSI_GREEN}🎉 ALL {result.testsRun} DOWNLOADER STAGING TESTS PASSED SUCCESSFULLY!{ANSI_RESET}")
        return 0
    else:
        print(f"{ANSI_BOLD}{ANSI_RED}❌ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS out of {result.testsRun} tests!{ANSI_RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
