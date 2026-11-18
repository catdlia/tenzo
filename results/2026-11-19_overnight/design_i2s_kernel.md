# Design note: I2_S-style ternary GEMV with 8-bit activations (not implemented tonight)

Context: fix 2 (`fix2_act_quant.md`) restored HF activation semantics (Qp = 127). The TL1 LUT kernel needs two int8
tables per activation pair (w0·a0 and w1·a1, each ≤ 127) and two `vpshufb` per nibble instead of one. This roughly
doubles the shuffle/convert/add work per weight byte. The measured speed effect is in `speed_after_fix2.md`.

## Proposal: maddubs kernel (the bitnet.cpp I2_S idea)

- **Weights.** Store u = w + 1 ∈ {0, 1, 2} as 2-bit codes, 4 per byte. Unpack 32 bytes into 128 u8 lanes with
  shift + `and 0x03` (4 shifts/ands per 32 bytes).
- **Activations.** Signed int8 q ∈ [-127, 127], exactly as in HF ActQuant (per token, scale 127/max).
- **Dot product.** `_mm256_maddubs_epi16(u8 weights, s8 activations)` gives pairwise Σ u·q in int16.
  - |u·q| ≤ 254 per product and ≤ 508 per pair, so there is no saturation.
  - `_mm256_madd_epi16(x, ones)` widens to int32, and the sums accumulate in int32.
- **Correction.** Σ w·q = Σ (u − 1)·q = Σ u·q − Σ q. Σ q is computed **once per token per input vector**
  (shared by all output rows), so the correction costs one subtraction per output.
- **Scale.** y = (Σ u·q − Σ q) · (max/127) · weight_scale.

## Expected properties

- **Exactness.** Integer arithmetic matches the HF fake-quant result up to float scaling, exactly like the current LUT
  kernel. The `hf-semantics` test (`src/tests/TestHFSemantics.cpp`) would gate it unchanged.
- **Work per 32 weight bytes (128 weights).** About 4 unpack ops, 4 `maddubs`, 4 `madd` and 4 int32 adds. The
  current fixed TL1 kernel needs 2 index extractions, 4 shuffles and 8 sign-extends + 8 adds per 32 bytes (64
  weights). The maddubs path therefore does about 2× fewer µops per weight and no LUT construction
  (the current `build_lut` writes 2 × 32 bytes per activation pair per token).
- **Memory.** Same 2 bits per weight, so bandwidth is unchanged.
- **Cost.** A new weight layout. It needs a re-pack at load time (one pass over weights.bin into a new buffer, about
  the same size) or a new export format. weights.bin itself does not have to change.

## Not done tonight, and why

The task rule says: if decode drops by more than 10%, do not rewrite the kernel tonight; record the numbers and this note.
