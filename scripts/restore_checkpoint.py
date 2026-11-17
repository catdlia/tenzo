#!/usr/bin/env python3
"""
scripts/restore_checkpoint.py - Genuine 30-Layer BitNet Checkpoint Restoration Tool (Milestone 4 / Feature F17)

Restores genuine pre-trained 30-layer weights from BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
into models/bitnet-1.58b/ with full topology conformance and atomic export staging.
"""

import os
import sys
import struct
import shutil
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np

# Map for GGUF 2-bit ternary decoding:
# 0b00 -> -1, 0b01 -> 0, 0b10 -> 1, 0b11 -> 0
MAP2BIT = np.array([-1, 0, 1, 0], dtype=np.int8)

# GGUF constants
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


class GGUFReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, "rb")
        self.metadata = {}
        self.tensors = {}
        self._parse_header()

    def _read_str(self):
        length = struct.unpack("<Q", self.file.read(8))[0]
        return self.file.read(length).decode("utf-8", errors="replace")

    def _read_val(self, vtype):
        if vtype == GGUF_TYPE_UINT8: return struct.unpack("<B", self.file.read(1))[0]
        elif vtype == GGUF_TYPE_INT8: return struct.unpack("<b", self.file.read(1))[0]
        elif vtype == GGUF_TYPE_UINT16: return struct.unpack("<H", self.file.read(2))[0]
        elif vtype == GGUF_TYPE_INT16: return struct.unpack("<h", self.file.read(2))[0]
        elif vtype == GGUF_TYPE_UINT32: return struct.unpack("<I", self.file.read(4))[0]
        elif vtype == GGUF_TYPE_INT32: return struct.unpack("<i", self.file.read(4))[0]
        elif vtype == GGUF_TYPE_FLOAT32: return struct.unpack("<f", self.file.read(4))[0]
        elif vtype == GGUF_TYPE_UINT64: return struct.unpack("<Q", self.file.read(8))[0]
        elif vtype == GGUF_TYPE_INT64: return struct.unpack("<q", self.file.read(8))[0]
        elif vtype == GGUF_TYPE_FLOAT64: return struct.unpack("<d", self.file.read(8))[0]
        elif vtype == GGUF_TYPE_BOOL: return struct.unpack("<?", self.file.read(1))[0]
        elif vtype == GGUF_TYPE_STRING: return self._read_str()
        elif vtype == GGUF_TYPE_ARRAY:
            sub_type = struct.unpack("<I", self.file.read(4))[0]
            arr_len = struct.unpack("<Q", self.file.read(8))[0]
            return [self._read_val(sub_type) for _ in range(arr_len)]
        else:
            raise ValueError(f"Unknown GGUF type: {vtype}")

    def _parse_header(self):
        magic = self.file.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic: {magic}")
        version = struct.unpack("<I", self.file.read(4))[0]
        tensor_count = struct.unpack("<Q", self.file.read(8))[0]
        kv_count = struct.unpack("<Q", self.file.read(8))[0]

        for _ in range(kv_count):
            key = self._read_str()
            vtype = struct.unpack("<I", self.file.read(4))[0]
            val = self._read_val(vtype)
            self.metadata[key] = val

        for _ in range(tensor_count):
            name = self._read_str()
            n_dims = struct.unpack("<I", self.file.read(4))[0]
            dims = [struct.unpack("<Q", self.file.read(8))[0] for _ in range(n_dims)]
            qtype = struct.unpack("<I", self.file.read(4))[0]
            offset = struct.unpack("<Q", self.file.read(8))[0]
            self.tensors[name] = {
                "dims": dims,
                "type": qtype,
                "offset": offset
            }

        alignment = self.metadata.get("general.alignment", 32)
        pos = self.file.tell()
        self.data_offset = (pos + alignment - 1) & ~(alignment - 1)

    def close(self):
        self.file.close()


def dequant_i2_s(raw_bytes: bytes, n_elements: int) -> np.ndarray:
    raw_blocks = np.frombuffer(raw_bytes[:n_elements // 4], dtype=np.uint8).reshape(-1, 32)
    c0 = (raw_blocks >> 6) & 3
    c1 = (raw_blocks >> 4) & 3
    c2 = (raw_blocks >> 2) & 3
    c3 = (raw_blocks >> 0) & 3

    v0 = MAP2BIT[c0]
    v1 = MAP2BIT[c1]
    v2 = MAP2BIT[c2]
    v3 = MAP2BIT[c3]

    W_flat = np.stack([v0, v1, v2, v3], axis=1).reshape(-1)
    return W_flat[:n_elements]


def pack_tenzo_ternary(W: np.ndarray) -> bytes:
    N, K = W.shape
    W_4 = W.reshape(N // 4, 4, K)
    b0 = (W_4[:, 0, :] + 1).astype(np.uint8) & 0x03
    b1 = ((W_4[:, 1, :] + 1).astype(np.uint8) & 0x03) << 2
    b2 = ((W_4[:, 2, :] + 1).astype(np.uint8) & 0x03) << 4
    b3 = ((W_4[:, 3, :] + 1).astype(np.uint8) & 0x03) << 6
    packed = b0 | b1 | b2 | b3
    return packed.tobytes()


def compute_sha256(filepath: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def update_manifest_sha256(manifest_path: Path, rel_file_path: str, new_hash: str):
    if not manifest_path.exists():
        print(f"⚠️ Manifest {manifest_path} not found, skipping manifest update.")
        return

    lines = []
    updated = False
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines.append(line)
                continue
            parts = stripped.split(None, 1)
            if len(parts) == 2 and parts[1].strip() == rel_file_path:
                lines.append(f"{new_hash}  {rel_file_path}\n")
                updated = True
            else:
                lines.append(line)

    if not updated:
        lines.append(f"{new_hash}  {rel_file_path}\n")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"📝 Updated manifest {manifest_path}: {rel_file_path} -> {new_hash}")


def restore_checkpoint(gguf_path: str, target_dir_str: str, manifest_path_str: str = None, num_layers: int = 30):
    project_root = Path(__file__).resolve().parent.parent
    gguf_path = Path(gguf_path)
    if not gguf_path.is_absolute():
        gguf_path = project_root / gguf_path

    target_dir = Path(target_dir_str)
    if not target_dir.is_absolute():
        target_dir = project_root / target_dir

    if manifest_path_str:
        manifest_path = Path(manifest_path_str)
        if not manifest_path.is_absolute():
            manifest_path = project_root / manifest_path
    else:
        manifest_path = project_root / "models" / "CHECKPOINTS.sha256"

    print(f"=== Restoring Genuine BitNet Checkpoint ===")
    print(f"  Source GGUF:     {gguf_path}")
    print(f"  Target Dir:      {target_dir}")
    print(f"  Layers to write: {num_layers}")

    if not gguf_path.exists():
        raise FileNotFoundError(f"Source GGUF not found: {gguf_path}")

    # Set up staging directory
    pid = os.getpid()
    staging_dir = Path(f"{target_dir}.staging_{pid}")
    backup_dir = Path(f"{target_dir}.backup_{pid}")

    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Staging Dir:     {staging_dir}")

    reader = GGUFReader(str(gguf_path))
    staging_weights_path = staging_dir / "weights.bin"

    try:
        # 1. Prepare cohesive set in staging: model.mlir, tokenizer.vocab, config.json
        # Copy existing model.mlir and tokenizer.vocab if present in target_dir
        for required_file in ["model.mlir", "tokenizer.vocab"]:
            src_f = target_dir / required_file
            dst_f = staging_dir / required_file
            if src_f.exists() and src_f.stat().st_size > 0:
                shutil.copy2(src_f, dst_f)
            else:
                raise RuntimeError(f"Required artifact {required_file} missing in {target_dir}!")

        # Write canonical config.json
        config_data = {
            "model_type": "bitnet",
            "architectures": ["BitnetForCausalLM"],
            "num_hidden_layers": num_layers,
            "hidden_size": 2560,
            "intermediate_size": 6912,
            "num_attention_heads": 20,
            "num_key_value_heads": 5,
            "head_dim": 128,
            "vocab_size": 128256,
            "quant_mode": "i2_s"
        }
        with open(staging_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        # 2. Extract genuine weights to staging_dir/weights.bin
        vocab_size = 128256
        hidden_size = 2560
        expected_embed_bytes = vocab_size * hidden_size * 4  # 1,313,341,440
        total_written = 0

        with open(staging_weights_path, "wb") as wf:
            # Step A: Token Embeddings (F16 in GGUF -> F32 in Tenzo)
            print("  ⏳ Converting token embeddings (Float16 -> Float32)...")
            embed_info = reader.tensors["token_embd.weight"]
            reader.file.seek(reader.data_offset + embed_info["offset"])

            chunk_tokens = 4096
            written_embed_bytes = 0
            for start_idx in range(0, vocab_size, chunk_tokens):
                n_tok = min(chunk_tokens, vocab_size - start_idx)
                raw_chunk = reader.file.read(n_tok * hidden_size * 2)
                f32_chunk = np.frombuffer(raw_chunk, dtype=np.float16).astype(np.float32).tobytes()
                wf.write(f32_chunk)
                written_embed_bytes += len(f32_chunk)

            assert written_embed_bytes == expected_embed_bytes, (
                f"Embedding bytes mismatch: {written_embed_bytes} vs {expected_embed_bytes}"
            )
            total_written += written_embed_bytes
            print(f"  ✅ Embeddings written: {written_embed_bytes:,} bytes")

            # Step B: 30 Layers of 11 tensors each
            layer_tensor_specs = [
                ("attn_norm", 2560 * 4, "norm", None),
                ("attn_q", 2560 * 640, "proj", (2560, 2560)),
                ("attn_k", 640 * 640, "proj", (2560, 640)),
                ("attn_v", 640 * 640, "proj", (2560, 640)),
                ("attn_sub_norm", 2560 * 4, "norm", None),
                ("attn_output", 2560 * 640, "proj", (2560, 2560)),
                ("ffn_norm", 2560 * 4, "norm", None),
                ("ffn_gate", 6912 * 640, "proj", (2560, 6912)),
                ("ffn_up", 6912 * 640, "proj", (2560, 6912)),
                ("ffn_sub_norm", 6912 * 4, "norm", None),
                ("ffn_down", 2560 * 1728, "proj", (6912, 2560)),
            ]

            for layer_idx in range(num_layers):
                layer_bytes = 0
                for suffix, exp_sz, t_type, proj_dims in layer_tensor_specs:
                    name = f"blk.{layer_idx}.{suffix}.weight"
                    info = reader.tensors[name]
                    reader.file.seek(reader.data_offset + info["offset"])

                    if t_type == "norm":
                        cnt = info["dims"][0]
                        raw = reader.file.read(cnt * 4)
                        assert len(raw) == exp_sz, f"Size mismatch for {name}: {len(raw)} != {exp_sz}"
                        wf.write(raw)
                        layer_bytes += len(raw)
                    else:  # proj
                        K, N = proj_dims
                        raw = reader.file.read(N * K // 4 + 32)
                        W_flat = dequant_i2_s(raw, N * K)
                        W = W_flat.reshape(N, K)
                        packed = pack_tenzo_ternary(W)
                        assert len(packed) == exp_sz, f"Size mismatch for {name}: {len(packed)} != {exp_sz}"
                        wf.write(packed)
                        layer_bytes += len(packed)

                assert layer_bytes == 17425408, f"Layer {layer_idx} size mismatch: {layer_bytes} != 17425408"
                total_written += layer_bytes
                if (layer_idx + 1) % 5 == 0 or layer_idx == num_layers - 1:
                    print(f"  ✅ Layer {layer_idx + 1}/{num_layers} written (17,425,408 bytes per layer)")

            # Step C: Output Norm (Float32)
            final_norm_info = reader.tensors["output_norm.weight"]
            reader.file.seek(reader.data_offset + final_norm_info["offset"])
            raw_final_norm = reader.file.read(2560 * 4)
            assert len(raw_final_norm) == 10240, f"Final norm size mismatch: {len(raw_final_norm)} != 10240"
            wf.write(raw_final_norm)
            total_written += len(raw_final_norm)
            print(f"  ✅ Final Norm written: {len(raw_final_norm):,} bytes")

        reader.close()

        # 3. Strict Pre-Promotion Verification on Staging
        expected_total = 1836113920
        actual_size = staging_weights_path.stat().st_size
        print(f"\n🔍 Staging Verification:")
        print(f"  Total size written: {actual_size:,} bytes (expected: {expected_total:,})")
        if actual_size != expected_total:
            raise ValueError(f"CRITICAL: Weights size {actual_size} != {expected_total}!")

        # Verify non-dummy data: check embeddings are genuine
        with open(staging_weights_path, "rb") as f:
            first_embeds = np.frombuffer(f.read(2560 * 4 * 10), dtype=np.float32).reshape(10, 2560)
            std_dev = np.std(first_embeds)
            print(f"  Embedding sample std-dev: {std_dev:.6f}")
            if std_dev < 1e-4:
                raise ValueError("CRITICAL: Restored weights appear to be uniform/dummy!")

        # Verify cohesive set
        cohesive_files = ["weights.bin", "model.mlir", "tokenizer.vocab", "config.json"]
        for cf in cohesive_files:
            cf_path = staging_dir / cf
            if not cf_path.is_file() or cf_path.stat().st_size == 0:
                raise RuntimeError(f"Cohesive set verification failed: {cf} missing or empty in staging!")
            print(f"  Cohesive artifact verified: {cf} ({cf_path.stat().st_size:,} bytes)")

        # Enforce canonical read-only permission (0444) on staging weights.bin
        os.chmod(staging_weights_path, 0o444)
        print("  🔒 Read-only permission (0444) set on staging weights.bin")

        # 4. Atomic Promotion
        print(f"\n🚀 Atomically committing restored cohesive set to {target_dir}...")
        if target_dir.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            os.replace(target_dir, backup_dir)

        os.replace(staging_dir, target_dir)

        # Cleanup backup upon successful replace
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        # Re-ensure 0444 on final weights.bin
        final_weights_bin = target_dir / "weights.bin"
        os.chmod(final_weights_bin, 0o444)

        print(f"🎉 Checkpoint successfully restored to {target_dir}!")

        # 5. Compute SHA-256 and update manifest
        print(f"\n⏳ Computing SHA-256 of restored weights.bin...")
        new_sha256 = compute_sha256(final_weights_bin)
        print(f"  SHA-256: {new_sha256}")

        rel_path = str(final_weights_bin.relative_to(project_root))
        update_manifest_sha256(manifest_path, rel_path, new_sha256)

        return new_sha256

    except Exception as e:
        print(f"\n❌ Restoration failed: {e}", file=sys.stderr)
        # Restore backup if target_dir was replaced
        if backup_dir.exists() and not target_dir.exists():
            os.replace(backup_dir, target_dir)
            print(f"  Rolled back target_dir from backup.", file=sys.stderr)
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
            print(f"  Cleaned up staging dir.", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description="Restore genuine 30-layer BitNet checkpoint")
    parser.add_argument(
        "--gguf-path",
        type=str,
        default="BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        help="Path to source ggml-model-i2_s.gguf",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="models/bitnet-1.58b",
        help="Target model directory (default: models/bitnet-1.58b)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest file (default: models/CHECKPOINTS.sha256)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=30,
        help="Number of layers to export (default: 30)",
    )
    args = parser.parse_args()

    restore_checkpoint(
        gguf_path=args.gguf_path,
        target_dir_str=args.target_dir,
        manifest_path_str=args.manifest,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()
