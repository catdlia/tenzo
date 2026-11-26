# Method and experimental setup — draft 0 (coordinator, 2026-11-25)

_Prose draft for the Method / Setup sections. It only describes procedures that already exist in the repository (paths given) and contains no result numbers. Results enter through `results/paper/claims.md` IDs. `[TODO]` marks experiments still running; `[check]` marks facts to verify against the primary source before submission._

## 3.1 Models

- **BitNet b1.58 2B4T.** Ternary weights, 8-bit per-token absmax activations, 30 layers, 20 query and 5 key-value heads, head dimension 128.
  - We do not evaluate it through an inference engine. Instead we use an independent fp32 re-implementation written from the Hugging Face model code ("golden", `tools/reference/golden_bitnet.py`).
  - The golden reproduces HF's bf16 implementation up to bf16 rounding on first-step logits [claim T28; `[check]` the exact agreement numbers before quoting them].
  - All KV-quantization effects for BitNet are measured inside this reference.
- **Full-precision comparison models.** EuroLLM-1.7B and Qwen2.5-1.5B, run with HF transformers.
  - A patched eager-attention function receives post-RoPE K and V and applies the same quantizers as the golden path (`tools/reference/second_model_kv.py`).
  - Compute is fp32 from the bf16 checkpoints.
  - A check is run that the patch without quantization reproduces HF's own eager attention (`--check-windows`).
  - `[TODO]` SmolLM2-1.7B (`tokenizer.json` fertility check done `[2026-11-26-H4-tok]`; weights run pending — `NTREX-24` will require aligned EN/UK sentence trimming to fit `ctx=2048`).

## 3.2 KV-cache quantization

All schemes act on post-RoPE keys and on values, per layer and KV head, on vectors x ∈ ℝ^128. Quantization of token t uses only token t, so causality is preserved (`tools/reference/kv_ptq.py`).

- **INT8.** Per-token absmax scaling to [−127, 127], rounding half away from zero. 8 bits plus an fp16 scale per vector (8.125 bits per element).
- **S3 (main ternary scheme).**
  - A fixed randomized Hadamard rotation R is applied first, so the reconstruction is x̂ = Q(xR)Rᵀ.
  - Then a ternary code per group of 32 elements: threshold 0.7·mean|x| (TWN), least-squares scale on the support.
  - Stored as 2-bit codes plus an fp16 scale per group: 2.5 bits per element.
- **Cache policy.** Four attention-sink tokens and a recent window are kept in INT8; all other positions use S3. We compare two definitions of the recent window:
  - a *token-defined* window: the last r tokens;
  - a *character-defined* window: all tokens whose text span lies within the last C characters, computed from tokenizer offsets.

  We write S3+s4r32 for 4 sinks and r = 32, and S3+s4cC for the character variant.
- **Comparators and ablations.**
  - S13: K as asymmetric INT4 with g = 32; V as least-squares-optimal ternary with rotation.
  - K-only and V-only variants, in which the other tensor stays INT8.
  - S3 restricted to one third of the layers.
  - `[TODO T51]` V with 2-bit or 4-bit codes while K stays ternary.
- **Effective bits.** Bits per element are (n₈·8.125 + (T − n₈)·b)/T, where n₈ is the number of INT8 positions (sinks plus active recent slots) and b the bits of the main scheme (`[check]` `tools/reference/kv_ptq.py` also includes `fp16` zero-points for asymmetric schemes (`int2_rot`, `int4_g32`, `S13`) and accounts for the trailing incomplete group `incomplete_group_mask`, keeping average `15.5` tokens in INT8). We report them at a standardized context of 512 and asymptotically (T → ∞).
- **Engine check.** The same S3+s4r32 scheme is implemented in the Tenzo C++ runtime (`ternary_rot`). There it reproduces the simulator: the confidence interval of the PPL ratio contains 1 at contexts 512 and 2048 [claims 2026-11-23-H1-eng, 2026-11-23-H3].

## 3.3 Optional clip calibration

Static clip multipliers α ∈ [0.5, 1.5] per layer, KV head and 32-group are fitted layer by layer (`tools/reference/kv_calib.py`).
- The objective is the local attention-output error, with ternary codes fixed (straight-through estimator).
- Model weights are untouched and causality is unchanged.
- We fit one table on a mixed EN+UK calibration set and evaluate it in and out of domain [claims 2026-11-23-H1-sim, 2026-11-24-H2].

## 3.4 Parallel evaluation data

Comparing per-token NLL across languages is confounded by tokenization and by text difficulty. We therefore evaluate on **content-matched** text:

- **FLORES-200 devtest.**
  - Windows are runs of consecutive sentences. Sentences are added while the longest tokenization of the window, across all models of the comparison, stays at or below 511 scored tokens (`[check]` `H2` reuses `flores5_meta.json` boundaries and evaluates `10` languages under `<= 1023` scored tokens, reaching `605` BitNet / `523` EuroLLM tokens on high-fertility languages).
  - Each window has identical content in every language and for every model.
  - Two sets: 32 EN/UK windows (three tokenizers), and 48 windows over sentences 209–633 for en, de, pl, uk, bg (`+ fr, cs, hu, el, es` completed in `[2026-11-26-H2]`; `[TODO]` `+ fi`).
- **NTREX-128** (newstest2019 EN→UK). 24 complete articles of at most 1023 tokens: coherent long-form text rather than sentence concatenations.
- **Leakage control.** Every evaluation window shares no 32-gram with the calibration windows or with any other evaluation set. Raw text is not redistributed; scripts rebuild the windows from the official archives (sha256 recorded).

## 3.5 Metrics

For window w in language L:
- B_w = −Σ log p_base is the base NLL in nats;
- Δ_w = Σ (log p_base − log p_scheme) is the extra NLL caused by the KV scheme.

On parallel text, totals over the same content are directly comparable across languages.

- **Disparity.**
  - D = Δ_UK − Δ_EN: extra nats for identical content.
  - rel_L = ΣΔ / ΣB: relative loss. One earlier analysis averaged per-window ratios; see the note in the iso-loss report.
  - R = rel_UK − rel_EN, in percentage points.
- **Language descriptors** (same content, per model).
  - Fertility φ_L = tokens_L / tokens_EN.
  - Cost of content κ_L = B_L / B_EN: how many more nats the model spends on the same content. We use it as a proxy for the model's competence in L.
- **Damage at matched difficulty** (`scripts/analyze_matched_difficulty.py`).
  - Tokens are binned by base NLL b = −log p_base (edges 0, 0.05, 0.3, 1, 2, 4, 7, ∞).
  - The language's mean Δ per bin is reweighted to the English bin distribution of the same corpus and model.
  - We report the difference and the ratio to English, plus a variant restricted to b < 4. In the highest-surprise tail, quantization noise lowers NLL by flattening the distribution.
- **Iso-loss budget** (`scripts/analyze_iso_loss_bits.py`).
  - For a target relative loss, this is the smallest INT8 window (in tokens, covered characters and effective bits) at which each language reaches the target.
  - It is found by linear interpolation along a family of window sizes. The ratio of UK to EN slots is the transferable quantity.
- **Statistics.**
  - Paired window bootstrap with 10,000 resamples and seed 0; percentile 95% intervals.
  - Windows are aligned across languages, schemes and models within a corpus, so all comparisons resample the same window ids jointly.
  - Decision rules were written down before each run (`## Plan` sections in the run reports). Analyses posed after seeing the data are labelled exploratory.

## 3.6 Scope

- We measure next-token NLL only; no downstream task accuracy (`[TODO]` a long-context multilingual check, e.g. OneRuler `[check]`).
- All quantization is post-training: no KV-aware fine-tuning.
- Evaluated models: one ternary model and two (`[TODO]` three — `SmolLM2-1.7B` weights pending; `[check]` multilingual set is `10` FLORES languages completed in `[2026-11-26-H2]` rather than `5` / `11`) full-precision models of 1.5–2B parameters.
