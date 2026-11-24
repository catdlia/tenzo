# H4 — Mechanism Ablations and Attention/Quantization Diagnostics (`T46`)

## Plan

- **Objective (`T46`):** Identify which component(s) of `S3+s4r32` (K vs V, recent window protection `r`, or early/mid/late transformer layers) drive Ukrainian's disproportionate quantization loss `D_w = Δ_w,UK − Δ_w,EN` (and `R_w = Δ_w,UK/B_UK − Δ_w,EN/B_EN`) on equal parallel content, and test four mechanistic hypotheses `(a)–(d)`.
- **Dataset (`N = 12` paired FLORES-200 windows, `ctx <= 512`, BitNet-b1.58-2B-4T):**
  - First `12` aligned EN/UK windows from `H3`: `results/2026-11-24_agent2/H3/flores_bitnet_{en,uk}.ids[:12]` (`0` shared 32-grams vs all calib/eval sets).
  - Baselines `fp32` and `S3+s4r32` (`w=0..11`) are reused directly from `results/2026-11-24_agent2/H3/scores/bitnet/`.
- **Ablations (`GB.Golden` `fp32` layer-major, `MemoryMax=4G`):**
  1. **K vs V split:** `KV8:S3+s4r32` (K quantized to `S3`, V kept INT8) vs `K8V:S3+s4r32` (K kept INT8, V quantized to `S3`), testing whether Key quantization or Value quantization dominates `D`.
  2. **Recent-window `r` sweep (`sinks=4` fixed):** `r ∈ {0, 128, 256}` (`S3+s4r0`, `S3+s4r128`, `S3+s4r256`, alongside `r=32`), testing how fast `D` shrinks with local INT8 protection (noting EN mean length `~182` tok vs UK `~307` tok at `r=256`).
  3. **Layer-group subsets (`10` active `S3+s4r32` layers, rest `fp32`):** `L0..9` (early), `L10..19` (mid), `L20..29` (late).
- **Single-pass `fp32` hook diagnostics (`30` layers, EN vs UK, `N = 12`):**
  - Relative L2 `S3` quantization error (`Hadamard + TWN g=32`) of post-RoPE `K` and `V` (`k_rel_l2`, `v_rel_l2`).
  - Mid/far attention mass outside `sinks=4` and `recent=32` (`4 <= s <= t - 32` for `t >= 64`) under exact `fp32` KV vs `S3-K+s4r32` KV.
  - Attention entropy `H(p_fp32)` vs `H(p_S3_K)` and leaked far mass across `128`-token query blocks (`0..127`, `128..255`, `256..383`).
- **Pre-registered hypotheses `(a)–(d)` & evaluation criteria (`10,000`-rep paired window bootstrap `seed=0`):**
  - **(a) UK attends farther:** Baseline `fp32` mid/far attention mass outside `s4r32` is substantially higher on UK than EN, and expanding `r` eliminates most of `D`.
  - **(b) UK K/V tensors quantize worse intrinsically:** Mean relative L2 `S3` error (`k_rel_l2` or `v_rel_l2`) is clearly higher on UK than EN (note: check both K and V error alongside K-only vs V-only `D`).
  - **(c) V error accumulates / late-layer compounding:** `K8V` dominates `D`, and/or `L20..29` / `r=0` sensitivity drives the UK penalty.
  - **(d) Attention leakage from noisy `S3-K`:** Quantizing `K` to `S3` (`KV8:S3+s4r32`) drives the majority of `D`, raises attention entropy `H(p_S3_K) > H(p_fp32)`, and leaks attention mass into the ternary mid/far tail `4 <= s <= t - 32`.
  - Primary ablation metric: fraction of baseline `S3+s4r32` `D = Δ_UK − Δ_EN` retained / removed (`% removed = 100% * (1 - D_ablation / D_S3+s4r32)`) and paired 95% CIs for `D` and `R`. If no hypothesis matches cleanly, report honestly without post-hoc overclaiming.
