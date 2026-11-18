# Provenance of models/bitnet-1.58b/weights.bin (F17)

sha256 `68c8f32b7f5d905ed7b3aaaf688e8913ed12227c7d9afc1bf3f429e3479400d1` (unchanged before/after this run; matches models/CHECKPOINTS.sha256).

## How it was produced
`scripts/restore_checkpoint.py` (2026-10-08) reads `BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`
(sha256 4221b252..., listed in CHECKPOINTS.sha256), converts f16 embeddings to f32, decodes each i2_s tensor
(ggml group-major order: element = 32*group + byte inside every 128-element block), repacks it as Tenzo ternary
(`pack_tenzo_ternary`, 4 consecutive output rows per byte) and writes embeddings + 30 x 17,425,408-byte layers + final norm.
`model.mlir`, `tokenizer.vocab` come from the earlier HF export (tenzo-frontend/export_bitnet.py, Aug 2026) and carry the
per-tensor scales (210 `arith.constant ... : f32` feeding `tenzo.bitlinear_elut`).

## Independent checks (tools/reference/check_provenance.py -> provenance_check.json, layers 0/15/29 x 7 projections = 21 tensors)
* ternary weights: GGUF i2_s == HF `microsoft/bitnet-b1.58-2B-4T` safetensors **bit-for-bit** in all 21 tensors (mismatch 0.0);
* `weights.bin` bytes == repack(GGUF) == repack(HF) in all 21 tensors;
* re-running the restore code path with a patched copy reproduced the identical SHA-256 (the experiment output was written under /tmp and deleted);
* scales: the GGUF stores an fp32 per-tensor scale in the 32 bytes after the packed data (restore_checkpoint reads and discards it);
  `model.mlir` carries the HF bf16 scale. They agree to bf16 rounding, i.e. a per-tensor relative deviation up to ~0.4%
  (e.g. L0 attn_k: GGUF 1.79406 vs HF/mlir 1.796875, 0.16%; L0 attn_v 0.16%; L0 attn_output 0.08%). Systematic but small.
* the engine therefore runs HF-identical ternary weights with bf16-rounded scales and f16-rounded embeddings.

## Not done / caveats
* Only 3 of 30 layers were compared tensor-by-tensor (all 7 projections each); the remaining layers are only covered by the SHA-256.
* Norm weights, embeddings and the final norm were not compared against HF here.
* Consequence for bitnet.cpp: the GGUF holds the same ternary weights as HF, so the poor quality of the local bitnet.cpp build
  (PPL ~100 vs HF 12-17) comes from that fork's graph/kernels/tokenizer, NOT from different weights (see OVERNIGHT_REPORT.md).
