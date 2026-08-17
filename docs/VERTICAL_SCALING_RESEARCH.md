# Deep Vertical Scaling & Optimization Architecture for Tenzo

## Executive Summary

**Vertical scaling** focuses on maximizing computational density, latency reduction, memory efficiency, and algorithmic intelligence on any target hardware—in contrast to **horizontal scaling**, which focuses on porting to heterogeneous backends (ARM NEON, Vulkan, Apple Metal, WebAssembly).

This document outlines the theoretical foundation, mathematical models, and implementation roadmap for vertical optimizations in the Tenzo AI compiler and inference engine.

---

## 1. Advanced Memory & KV-Cache Hierarchy

### 1.1 PagedAttention & Non-Contiguous Physical Memory
Current fixed contiguous buffers suffer from high fragmentation (up to 60-80% wasted memory when allocating `max_seq_len = 8192`).
* **Virtual Memory Paging:** Partition sequence KV data into fixed-size physical blocks (e.g., block size = 16 or 32 tokens).
* **Block Table Translation:** A virtual-to-physical address lookup table maps `(seq_id, logical_token_idx) -> (physical_block_id, block_offset)`.
* **Zero-Copy Prompt Sharing:** Enables copy-on-write branching for multi-sample generation, beam search, and speculative tree verification with near-zero memory overhead.

### 1.2 Chunked Prefill & FlashDecoding
* **Chunked Prefill:** Long prompt contexts are partitioned into compute chunks ($B_{\text{prefill}} = 512$ tokens), interleaving prompt processing with active token generation to avoid latency spikes and cache blowouts.
* **FlashDecoding:** Standard autoregressive decoding is memory-bandwidth bound. FlashDecoding splits the sequence dimension $T$ across multiple thread blocks/cores, reducing partial softmax accumulators in parallel:
  $$\text{Softmax}(Q K^T) V = \bigoplus_{i=1}^{S} \left( \frac{e^{m_i - m_{\text{global}}}}{l_{\text{global}}} \text{AttnChunk}_i \right)$$
  where $m_i$ and $l_i$ are running max and sum exponentials.

### 1.3 Sub-Byte On-The-Fly KV Quantization
* **INT4 / INT2 Ternary KV:** Quantize Key and Value projections immediately before writing to the Paged KV memory pool.
* **Scale Isolation:** Maintain per-channel FP16/BF16 scale vectors per block, reducing KV RAM from 4 GB down to **256 MB** for 8k context on 2B models.

---

## 2. Speculative Decoding & Predictive Acceleration

```
  ┌──────────────────────────────────────────────────────────┐
  │                 Speculative Execution Loop                │
  └──────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────┐       K Draft Tokens        ┌─────────────────┐
  │  Draft Model    │ ──────────────────────────> │   Target Model  │
  │  (Ternary TL1)  │                             │ (Full BitNet/LLM│
  └─────────────────┘                             └─────────────────┘
                                                           │
                                                           ▼ Parallel Verify
                                                  ┌─────────────────┐
                                                  │ Acceptance Test │
                                                  │ (Greedy / Nucleus)
                                                  └─────────────────┘
                                                           │
                                           ┌───────────────┴───────────────┐
                                           ▼                               ▼
                                   Accepted Tokens                 Reject & Correct
```

### 2.1 Ternary Draft Models
* **Mechanism:** Use a small 1-to-2 layer BitNet ternary draft network (or a pruned 150M TL1 model) to generate $K = 4$ candidate tokens with microsecond latency ($< 0.5\text{ ms/token}$).
* **Single-Pass Parallel Verification:** Run the full 2B/8B model across all $K$ tokens in a single vectorized forward pass:
  $$r = \min\left(1, \frac{P_{\text{target}}(x_{t+k})}{P_{\text{draft}}(x_{t+k})}\right)$$
* **Throughput Gain:** Expected **2.2x to 3.4x wall-clock speedup** on CPU/GPU without any degradation in perplexity or output quality.

### 2.2 Medusa-Style Multi-Head Decoding
* Instead of a separate draft model, attach $M = 3$ shallow linear residual heads to the final transformer layer of Tenzo.
* Each head predicts token $t+1, t+2, t+3$ simultaneously from the hidden state $h_t$.
* Tree-attention verification in the subsequent forward step validates paths simultaneously.

---

## 3. MLIR Graph Rewriting & Kernel Fusion

### 3.1 Super-Fused Transformer Blocks
Currently, intermediate tensors flow through multiple memory roundtrips:
`RMSNorm -> BitLinear -> RoPE -> Attention -> BitLinear -> Add -> RMSNorm -> BitLinear -> Activation -> BitLinear -> Add`.

Tenzo's MLIR transformation pipeline will fuse these into **Two Macro-Kernels per Layer**:
1. **Fused Attn Macro-Kernel:** `InRMSNorm + Multi-Head Q/K/V BitLinear + Fused RoPE + Paged KV Update + FlashAttn + Out_BitLinear + Residual Add`.
2. **Fused FFN Macro-Kernel:** `PostRMSNorm + Gate/Up BitLinear + SwiGLU/ReLU2 + FFN_SubNorm + Down_BitLinear + Residual Add`.

### 3.2 Register-Level Loop Tiling & Multi-Reduction
* Utilize MLIR's `vector.multi_reduction` with 2D register tiling ($M \times N = 6 \times 16$ on AVX2, $8 \times 32$ on AVX-512/ARM SVE).
* Accumulate GEMV in CPU vector registers without memory spills:
  ```mlir
  %acc = vector.contract {indexing_maps = [...], iterator_types = ["parallel", "reduction"]} %act, %weights, %init
  ```

---

## 4. Next-Gen Micro-Scaling Formats (NVFP4 / MXFP4)

* **NVFP4 / MXFP4 Specification:** 4-bit floating point formats (E2M1) with shared scaling factors over blocks of 16 values (E8M0 scales).
* **AVX-VNNI Acceleration:** Leveraging Intel `vpdpbusd` and ARM `SDOT`/`UDOT` instructions:
  - 4x INT8 operations per instruction cycle.
  - Reduces decompression overhead to zero by computing integer dot products directly in vector ALU pipelines.

---

## 5. Implementation Milestones

| Milestone | Target | Description | Expected Gain |
| :--- | :--- | :--- | :--- |
| **V1.0** | PagedAttention Engine | Virtual block-based KV cache allocation & defragmentation | -70% Peak RAM |
| **V1.1** | FlashDecoding MLIR Pass | Parallelized sequence reduction for decodes | +2.0x Long-Ctx Speed |
| **V1.2** | Super-Fused Graph Passes | Macro-kernel fusion eliminating memory roundtrips | +35% Single-Token Speed |
| **V1.3** | Speculative Draft Engine | Integrated draft-target speculative decoding pipeline | +2.5x Generation Speed |
| **V1.4** | VNNI / Microscaling | Native `vpdpbusd` and NEON `SDOT` GEMV paths | +80% Compute Density |
