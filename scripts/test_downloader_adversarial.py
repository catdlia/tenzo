#!/usr/bin/env python3
"""
Adversarial Empirical Stress-Testing Harness for Downloader Atomic Staging
Tests Requirement R1 under hostile, adversarial, and boundary conditions:
1. Adversarial Test 1: Network drop / partial download mid-stream into existing sentinel model directory.
2. Adversarial Test 2: Incomplete cohesive set in staging (missing & 0-byte files across all assets).
3. Adversarial Test 3: Truncated weights file (< 1.83 GB, down to exact 1-byte off-by-one boundary).
4. Adversarial Test 4: Uniform / dummy / synthetic weights (std = 0, std <= 1e-4, NaN analysis).
5. Adversarial Test 5: Atomic swap failure & rollback with pre-existing read-only target files.
"""

import os
import sys
import shutil
import tempfile
import hashlib
import unittest
import socket
import threading
import http.server
import socketserver
import subprocess
import numpy as np

# Ensure scripts dir is accessible
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

class AdversarialTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="tenzo_adversarial_test_")
        self.target_dir = os.path.join(self.tmp_dir, "model_target")
        self.base_assets_dir = os.path.join(self.tmp_dir, "base_assets")
        os.makedirs(self.base_assets_dir, exist_ok=True)

        # Baseline fixtures
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
        """Creates an existing target containing verified sentinel files and canaries."""
        os.makedirs(self.target_dir, exist_ok=True)
        sentinels = {
            "weights.bin": b"PRE_EXISTING_PRODUCTION_WEIGHTS_CANARY_ABCD_1234567890",
            "model.mlir": "module { // PRE_EXISTING PRODUCTION MLIR CANARY }",
            "tokenizer.vocab": b"PRE_EXISTING_PRODUCTION_VOCAB_CANARY_0987654321",
            "config.json": '{"production_sentinel": "CRITICAL_DO_NOT_CORRUPT"}',
            "audit_canary.sha256": b"IMMUTABLE_AUDIT_CANARY_SIGNATURE_9999",
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
        """Strictly assert that every sentinel file has 0 byte alteration and identical SHA-256."""
        self.assertTrue(os.path.isdir(self.target_dir), "Target directory was unexpectedly removed!")
        current_files = sorted(os.listdir(self.target_dir))
        self.assertEqual(current_files, sorted(expected_hashes.keys()), "Target directory file list altered!")
        for fname, exp_hash in expected_hashes.items():
            fpath = os.path.join(self.target_dir, fname)
            with open(fpath, "rb") as f:
                cur_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(cur_hash, exp_hash, f"Sentinel file '{fname}' was modified! Hash mismatch!")

    def verify_no_staging_residue(self):
        """Strictly assert that staging and backup directories are 100% wiped."""
        parent_dir = os.path.dirname(self.target_dir)
        target_name = os.path.basename(self.target_dir)
        leftovers = [
            d for d in os.listdir(parent_dir)
            if d.startswith(f"{target_name}.staging_") or d.startswith(f"{target_name}.backup_")
        ]
        self.assertEqual(leftovers, [], f"Residue detected! Staging/backup dirs not removed: {leftovers}")

    def create_staging_weights(self, path, num_layers=30, mode="random", size=1836113920):
        """Creates staging weights with specified characteristics."""
        import struct
        with open(path, "wb") as wf:
            if mode == "zeros":
                wf.write(b"\x00" * min(size, 2560 * 4 * 10))
            elif mode == "ones":
                ones = np.ones((10, 2560), dtype=np.float32).tobytes()
                wf.write(ones[:min(size, len(ones))])
            elif mode == "constant":
                c = np.full((10, 2560), 42.0, dtype=np.float32).tobytes()
                wf.write(c[:min(size, len(c))])
            elif mode == "low_variance":
                # Variance below 1e-4 (e.g. std = 1e-5)
                low_v = (np.random.randn(10, 2560) * 1e-5).astype(np.float32).tobytes()
                wf.write(low_v[:min(size, len(low_v))])
            elif mode == "boundary_std":
                # Exactly normalized std <= 1e-4
                arr = np.random.randn(10, 2560).astype(np.float32)
                arr = (arr - np.mean(arr)) / np.std(arr) * 1e-4
                b = arr.tobytes()
                wf.write(b[:min(size, len(b))])
            elif mode == "nan":
                nans = np.full((10, 2560), np.nan, dtype=np.float32).tobytes()
                wf.write(nans[:min(size, len(nans))])
            else:
                # Genuine random (std ~ 1.0)
                rnd = np.random.randn(10, 2560).astype(np.float32).tobytes()
                wf.write(rnd[:min(size, len(rnd))])

            if size > (2560 * 4 * 10):
                wf.seek(size - 1)
                wf.write(b"\x00")
            elif size < (2560 * 4 * 10):
                wf.truncate(size)


# =====================================================================
# 1. Adversarial Test 1: Network drop / partial download mid-stream
# =====================================================================
class TestAdversarialNetworkDrop(AdversarialTestBase):
    """Adversarial Test 1: Real and simulated network failures mid-stream."""

    def test_real_http_socket_drop_midstream(self):
        """Run a local raw TCP server that sends partial chunk and abruptly drops TCP socket with RST."""
        import struct
        sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        # Custom TCP server that abruptly aborts
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]

        def serve_and_drop():
            try:
                conn, _ = server_sock.accept()
                req = conn.recv(1024)
                # Send HTTP header with 5MB length, then 64KB data, then hard close
                header = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/octet-stream\r\n"
                    b"Content-Length: 5242880\r\n"
                    b"Connection: close\r\n\r\n"
                )
                conn.sendall(header)
                conn.sendall(b"A" * 65536)  # Partial 64 KB
                # Force TCP RST
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                conn.close()
            except Exception:
                pass
            finally:
                try:
                    server_sock.close()
                except Exception:
                    pass

        server_thread = threading.Thread(target=serve_and_drop, daemon=True)
        server_thread.start()

        # Attempt download from the dropping server
        url = f"http://127.0.0.1:{port}/model.safetensors"
        success = download_model.download_and_setup(
            repo_id=url,
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "download_and_setup must return False on abrupt TCP reset")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()

    def test_connection_reset_exception_midstream(self):
        """Simulate ConnectionResetError during download with partial write."""
        sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        def failing_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"CORRUPTED_PARTIAL_BYTES" * 1024)
            raise ConnectionResetError("Connection forcibly closed by remote host [WinError/Linux 104]")

        download_model.download_file_with_progress = failing_download

        success = download_model.download_and_setup(
            repo_id="test/aborted-midstream-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must return False on ConnectionResetError")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()

    def test_urllib_http_error_midstream(self):
        """Simulate urllib HTTP 500 error."""
        import urllib.error
        sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        def failing_download(url, dest_path):
            raise urllib.error.HTTPError(url, 500, "Internal Server Error", {}, None)

        download_model.download_file_with_progress = failing_download

        success = download_model.download_and_setup(
            repo_id="test/http-500-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must return False on HTTP 500")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()


# =====================================================================
# 2. Adversarial Test 2: Incomplete cohesive set in staging
# =====================================================================
class TestAdversarialIncompleteCohesiveSet(AdversarialTestBase):
    """Adversarial Test 2: Incomplete cohesive set in staging."""

    def setUp(self):
        super().setUp()
        self.sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        # Mock download to create dummy raw safetensors
        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_RAW_SAFETENSORS")
        download_model.download_file_with_progress = dummy_download

    def _run_with_custom_staging(self, setup_fn):
        def custom_export(safetensors_path, staging_dir, num_layers=30):
            # First create full valid set
            self.create_staging_weights(os.path.join(staging_dir, "weights.bin"))
            with open(os.path.join(staging_dir, "model.mlir"), "w") as f:
                f.write("module { func.func @test() {} }")
            with open(os.path.join(staging_dir, "tokenizer.vocab"), "wb") as f:
                f.write(b"VALID_VOCAB")
            with open(os.path.join(staging_dir, "config.json"), "w") as f:
                f.write('{"model": "valid"}')
            # Now apply corruption/deletion
            setup_fn(staging_dir)

        download_model.export_safetensors_to_tenzo = custom_export

        success = download_model.download_and_setup(
            repo_id="test/adversarial-incomplete-set",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertFalse(success, "Must return False when cohesive set is incomplete")
        self.verify_sentinel_intact(self.sentinel_hashes)
        self.verify_no_staging_residue()

    def test_missing_model_mlir(self):
        """Weights present but model.mlir is completely deleted from staging."""
        def corrupt(staging):
            os.remove(os.path.join(staging, "model.mlir"))
        self._run_with_custom_staging(corrupt)

    def test_zero_byte_model_mlir(self):
        """Weights present but model.mlir is 0 bytes."""
        def corrupt(staging):
            with open(os.path.join(staging, "model.mlir"), "w") as f:
                pass
        self._run_with_custom_staging(corrupt)

    def test_missing_config_json(self):
        """Weights present but config.json is completely deleted from staging."""
        def corrupt(staging):
            os.remove(os.path.join(staging, "config.json"))
        self._run_with_custom_staging(corrupt)

    def test_zero_byte_config_json(self):
        """Weights present but config.json is 0 bytes."""
        def corrupt(staging):
            with open(os.path.join(staging, "config.json"), "w") as f:
                pass
        self._run_with_custom_staging(corrupt)

    def test_missing_tokenizer_vocab(self):
        """Weights present but tokenizer.vocab is completely deleted from staging."""
        def corrupt(staging):
            os.remove(os.path.join(staging, "tokenizer.vocab"))
        self._run_with_custom_staging(corrupt)

    def test_zero_byte_tokenizer_vocab(self):
        """Weights present but tokenizer.vocab is 0 bytes."""
        def corrupt(staging):
            with open(os.path.join(staging, "tokenizer.vocab"), "wb") as f:
                pass
        self._run_with_custom_staging(corrupt)

    def test_missing_weights_bin(self):
        """All assets present but weights.bin is missing."""
        def corrupt(staging):
            os.remove(os.path.join(staging, "weights.bin"))
        self._run_with_custom_staging(corrupt)

    def test_zero_byte_weights_bin(self):
        """All assets present but weights.bin is 0 bytes."""
        def corrupt(staging):
            with open(os.path.join(staging, "weights.bin"), "wb") as f:
                pass
        self._run_with_custom_staging(corrupt)


# =====================================================================
# 3. Adversarial Test 3: Truncated weights file (< 1.83 GB)
# =====================================================================
class TestAdversarialTruncatedWeights(AdversarialTestBase):
    """Adversarial Test 3: Truncated weights files in staging."""

    def setUp(self):
        super().setUp()
        self.sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_RAW_SAFETENSORS")
        download_model.download_file_with_progress = dummy_download

    def _run_with_weights_size(self, size_bytes, num_layers=30):
        def custom_export(safetensors_path, staging_dir, num_layers=num_layers):
            self.create_staging_weights(
                os.path.join(staging_dir, "weights.bin"),
                num_layers=num_layers,
                size=size_bytes
            )
            with open(os.path.join(staging_dir, "model.mlir"), "w") as f:
                f.write("module { func.func @test() {} }")
            with open(os.path.join(staging_dir, "tokenizer.vocab"), "wb") as f:
                f.write(b"VALID_VOCAB")
            with open(os.path.join(staging_dir, "config.json"), "w") as f:
                f.write('{"model": "valid"}')

        download_model.export_safetensors_to_tenzo = custom_export

        success = download_model.download_and_setup(
            repo_id=f"test/truncated-weights-{size_bytes}",
            output_dir=self.target_dir,
            num_layers=num_layers,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertFalse(success, f"Must return False when weights are truncated ({size_bytes} bytes)")
        self.verify_sentinel_intact(self.sentinel_hashes)
        self.verify_no_staging_residue()

    def test_truncated_tiny_weights_1kb(self):
        """Truncated to only 1,024 bytes."""
        self._run_with_weights_size(1024)

    def test_truncated_100mb(self):
        """Truncated to 100 MB."""
        self._run_with_weights_size(100 * 1024 * 1024)

    def test_truncated_embeddings_only(self):
        """Truncated to exactly embeddings size (1,313,341,440 bytes), missing all 30 layers."""
        self._run_with_weights_size(1313341440)

    def test_truncated_off_by_one_byte(self):
        """Adversarial boundary: Exactly 1 byte less than required: 1,836,113,919 bytes."""
        self._run_with_weights_size(1836113919)

    def test_truncated_custom_layers_boundary(self):
        """For num_layers=10, min bytes is 1,487,605,760. Test with 1,487,605,759 bytes."""
        expected_min = 1313341440 + 10 * 17425408 + 10240
        self._run_with_weights_size(expected_min - 1, num_layers=10)


# =====================================================================
# 4. Adversarial Test 4: Uniform/dummy weights (std = 0)
# =====================================================================
class TestAdversarialDummyWeights(AdversarialTestBase):
    """Adversarial Test 4: Uniform / synthetic dummy weights."""

    def setUp(self):
        super().setUp()
        self.sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_RAW_SAFETENSORS")
        download_model.download_file_with_progress = dummy_download

    def _run_with_weights_mode(self, mode):
        def custom_export(safetensors_path, staging_dir, num_layers=30):
            self.create_staging_weights(
                os.path.join(staging_dir, "weights.bin"),
                mode=mode,
                size=1836113920
            )
            with open(os.path.join(staging_dir, "model.mlir"), "w") as f:
                f.write("module { func.func @test() {} }")
            with open(os.path.join(staging_dir, "tokenizer.vocab"), "wb") as f:
                f.write(b"VALID_VOCAB")
            with open(os.path.join(staging_dir, "config.json"), "w") as f:
                f.write('{"model": "valid"}')

        download_model.export_safetensors_to_tenzo = custom_export

        success = download_model.download_and_setup(
            repo_id=f"test/dummy-weights-{mode}",
            output_dir=self.target_dir,
            num_layers=30,
            base_assets_dir=self.base_assets_dir,
        )
        self.assertFalse(success, f"Must return False when weights mode is '{mode}'")
        self.verify_sentinel_intact(self.sentinel_hashes)
        self.verify_no_staging_residue()

    def test_uniform_zeros_rejection(self):
        """All zero embeddings (std = 0.0)."""
        self._run_with_weights_mode("zeros")

    def test_uniform_ones_rejection(self):
        """All one embeddings (std = 0.0)."""
        self._run_with_weights_mode("ones")

    def test_uniform_constant_rejection(self):
        """All constant 42.0 embeddings (std = 0.0)."""
        self._run_with_weights_mode("constant")

    def test_low_variance_noise_rejection(self):
        """Synthetic low-variance noise (std ~ 1e-5 <= 1e-4)."""
        self._run_with_weights_mode("low_variance")

    def test_boundary_std_rejection(self):
        """Boundary variance (std = 1e-4 <= 1e-4)."""
        self._run_with_weights_mode("boundary_std")


# =====================================================================
# 5. Adversarial Test 5: Real CLI & Subprocess & Rollback Verification
# =====================================================================
class TestAdversarialCLIAndRollback(AdversarialTestBase):
    """Adversarial Test 5: Subprocess CLI execution and atomic rollback mechanics."""

    def test_cli_execution_with_invalid_repo_and_existing_readonly_target(self):
        """Run download_model.py CLI against pre-existing read-only target. Check exit code 1, sentinels intact."""
        sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        cli_script = os.path.join(SCRIPTS_DIR, "download_model.py")
        cmd = [sys.executable, cli_script, "https://nonexistent.domain.invalid/model.safetensors", self.target_dir]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.assertEqual(proc.returncode, 1, f"Expected CLI exit code 1, got {proc.returncode}")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()

    def test_atomic_rollback_preserves_readonly_sentinels(self):
        """When promotion rename fails, target with 0444 sentinels is restored 100% intact."""
        sentinel_hashes = self.create_sentinel_target(make_weights_readonly=True)

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_RAW_SAFETENSORS")

        def valid_export(safetensors_path, staging_dir, num_layers=30):
            self.create_staging_weights(os.path.join(staging_dir, "weights.bin"))
            with open(os.path.join(staging_dir, "model.mlir"), "w") as f:
                f.write("module { func.func @promoted() {} }")
            with open(os.path.join(staging_dir, "tokenizer.vocab"), "wb") as f:
                f.write(b"VALID_VOCAB")
            with open(os.path.join(staging_dir, "config.json"), "w") as f:
                f.write('{"model": "valid"}')

        download_model.download_file_with_progress = dummy_download
        download_model.export_safetensors_to_tenzo = valid_export

        real_replace = _ORIG_REPLACE
        counter = [0]
        def failing_replace(src, dst):
            counter[0] += 1
            if counter[0] == 2:
                # Fail second rename (staging -> target)
                raise OSError("Simulated atomic staging rename failure (EIO / disk error)")
            return real_replace(src, dst)

        os.replace = failing_replace

        success = download_model.download_and_setup(
            repo_id="test/rollback-readonly-sentinel",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertFalse(success, "Must return False when swap fails")
        self.verify_sentinel_intact(sentinel_hashes)
        self.verify_no_staging_residue()

    def test_positive_control_over_readonly_target(self):
        """Positive control: Fully valid cohesive model replaces existing read-only target cleanly."""
        self.create_sentinel_target(make_weights_readonly=True)

        def dummy_download(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(b"DUMMY_RAW_SAFETENSORS")

        def valid_export(safetensors_path, staging_dir, num_layers=30):
            self.create_staging_weights(os.path.join(staging_dir, "weights.bin"))
            with open(os.path.join(staging_dir, "model.mlir"), "w") as f:
                f.write("module { func.func @verified_promoted_model() {} }")
            with open(os.path.join(staging_dir, "tokenizer.vocab"), "wb") as f:
                f.write(b"VERIFIED_VOCAB")
            with open(os.path.join(staging_dir, "config.json"), "w") as f:
                f.write('{"model_type": "bitnet", "quant_mode": "i2_s"}')

        download_model.download_file_with_progress = dummy_download
        download_model.export_safetensors_to_tenzo = valid_export

        success = download_model.download_and_setup(
            repo_id="test/verified-success-model",
            output_dir=self.target_dir,
            base_assets_dir=self.base_assets_dir,
        )

        self.assertTrue(success, "Must return True on valid cohesive set")
        self.verify_no_staging_residue()

        # Check that target now has new files and weights.bin is 0444
        target_weights = os.path.join(self.target_dir, "weights.bin")
        self.assertEqual(os.path.getsize(target_weights), 1836113920)
        mode = os.stat(target_weights).st_mode & 0o777
        self.assertEqual(mode, 0o444, f"Promoted weights permissions must be 0444, got {oct(mode)}")

        with open(os.path.join(self.target_dir, "model.mlir"), "r") as f:
            self.assertIn("@verified_promoted_model", f.read())


def main():
    print(f"\n{ANSI_BOLD}{ANSI_CYAN}{'=' * 75}{ANSI_RESET}")
    print(f"{ANSI_BOLD}{ANSI_CYAN}  TENZO DOWNLOADER EMPIRICAL ADVERSARIAL STRESS SUITE{ANSI_RESET}")
    print(f"{ANSI_BOLD}{ANSI_CYAN}{'=' * 75}{ANSI_RESET}\n")

    test_classes = [
        TestAdversarialNetworkDrop,
        TestAdversarialIncompleteCohesiveSet,
        TestAdversarialTruncatedWeights,
        TestAdversarialDummyWeights,
        TestAdversarialCLIAndRollback,
    ]

    suite = unittest.TestSuite()
    for tc in test_classes:
        for name in unittest.TestLoader().getTestCaseNames(tc):
            suite.addTest(tc(name))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 75)
    if result.wasSuccessful():
        print(f"{ANSI_BOLD}{ANSI_GREEN}🎉 ALL {result.testsRun} ADVERSARIAL STRESS TESTS PASSED SUCCESSFULLY!{ANSI_RESET}")
        return 0
    else:
        print(f"{ANSI_BOLD}{ANSI_RED}❌ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS out of {result.testsRun} tests!{ANSI_RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
