# Quantization & Bit-Packing Schemes in Tenzo

Tenzo supports a comprehensive matrix of quantization formats ranging from 1.58-bit ternary representations to standard 4-bit and 8-bit block schemes.

---

## 1. BitNet 1.58-bit (TL1) Quantization

### 1.1 Mathematical Definition
Weights are restricted to ternary values $w \in \{-1, 0, 1\}$:
$$\gamma = \frac{1}{K \times N} \sum_{i,j} |W_{i,j}|, \quad \widetilde{W} = \text{Round}\left(\text{Clamp}\left(\frac{W}{\gamma}, -1, 1\right)\right)$$

### 1.2 Packing Schemes
* **Standard 2-bit Packing:** 4 weights per byte (`val_tern = (raw >> (2 * slot)) & 3 - 1`).
* **TL1 Dual-Element LUT Packing:** Pairs of ternary weights $(w_0, w_1)$ are packed into a single 4-bit nibble:
  $$\text{idx} = (w_0 + 1) + 3 \times (w_1 + 1) \in [0, 8]$$
  Two pairs are packed into an 8-bit byte, enabling simultaneous two-element lookup table addition via AVX2 `vpshufb` (`_mm256_shuffle_epi8`).

---

## 2. GGUF Block Quantization (llama.cpp)

### 2.1 Q4_0 Format
* **Block Size:** 32 weights per block.
* **Layout:** 18 bytes total:
  - `d` (2 bytes): 16-bit float (FP16) block scale factor.
  - `qs` (16 bytes): 32 4-bit nibbles (low nibbles for weights $0..15$, high nibbles for weights $16..31$).
* **Dequantization Formula:**
  $$w_i = (q_i - 8) \times d$$

### 2.2 Q8_0 Format
* **Block Size:** 32 weights per block.
* **Layout:** 34 bytes total (2-byte FP16 scale `d` + 32 signed 8-bit integers `qs`).
* **Dequantization Formula:**
  $$w_i = q_i \times d$$

---

## 3. AutoGPTQ 4-Bit Groupwise Quantization

### 3.1 Format Specification
* **Weight Storage:** 8 4-bit weights packed along input dimension $K$ into a 32-bit integer `qweight`.
* **Zero-Points:** Packed in `qzeros` matrix.
* **Dequantization Formula:**
  $$w_{k, n} = \left(\text{raw\_w}_{k,n} - (\text{zero}_{g,n} + 1)\right) \times \text{scale}_{g,n}$$
  where $g = \lfloor k / \text{group\_size} \rfloor$.

---

## 4. AutoAWQ 4-Bit Column-Interleaved Quantization

### 4.1 Column Interleaving
To optimize SIMD matrix-vector multiplication pipelines, AWQ interleaves column order:
$$\text{Order} = [0, 2, 4, 6, 1, 3, 5, 7]$$
* **Dequantization Formula:**
  $$w_{k + \text{Order}[i], n} = (\text{raw\_w}_{i} - \text{zero}_{g, n}) \times \text{scale}_{g, n}$$

---

## 5. ExLlamaV2 (EXL2) Variable Bitrate

### 5.1 Dynamic Bitrate Groups
EXL2 dynamically assigns bit widths (2.0 to 8.0 bits) across sub-layers and weight matrices based on error gradient sensitivity.
* **Group Descriptors (`qgroups`):**
  - `[bits, k_start, k_len]`
* **Dispatch:**
  - Micro-kernels unpack with variable bit masks $M = 2^{\text{bits}} - 1$ and zero offset $2^{\text{bits}-1}$.
