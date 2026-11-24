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

## Results

Evaluated on first `N = 12` paired FLORES-200 windows (`flores_bitnet_{en,uk}.ids[:12]`). Baseline `S3+s4r32` has `D = +21.989` nats.

### 1. Ablations (`D = Δ_UK − Δ_EN`, 10,000-rep paired bootstrap `seed=0`)

| Ablation | Description | EN PPL ratio [95% CI] | UK PPL ratio [95% CI] | D = Δ_UK−Δ_EN [95% CI] | Rel diff R [95% CI] | Share of `S3+s4r32` `D` |
| --- | --- | --- | --- | --- | --- | --- |
| `S3+s4r32` | Full baseline (all 30 layers, s=4 r=32) | ×1.0015 [0.9867,1.0176] | ×1.0763 [1.0446,1.1103] | +21.989 [+13.290,+30.496] | +1.42% [+0.76%,+2.08%] | 100.0% (+0.0% removed) |
| `KV8:S3+s4r32` | K-only S3 (V=INT8, all layers, s=4 r=32) | ×1.0016 [0.9902,1.0138] | ×1.0297 [1.0079,1.0548] | +8.571 [+2.301,+15.356] | +0.57% [+0.09%,+1.07%] | 39.0% (+61.0% removed) |
| `K8V:S3+s4r32` | V-only S3 (K=INT8, all layers, s=4 r=32) | ×1.0026 [0.9942,1.0110] | ×1.0695 [1.0459,1.0929] | +19.865 [+13.009,+26.221] | +1.24% [+0.73%,+1.72%] | 90.3% (+9.7% removed) |
| `S3+s4r0` | Recent window r=0 (s=4 sinks INT8 only) | ×1.1876 [1.1564,1.2239] | ×1.5312 [1.4263,1.6732] | +98.239 [+76.421,+127.249] | +2.67% [+1.11%,+4.54%] | 446.8% (-346.8% removed) |
| `S3+s4r128` | Recent window r=128 (s=4 sinks + 128 recent INT8) | ×0.9992 [0.9959,1.0025] | ×1.0122 [0.9980,1.0269] | +3.823 [-0.582,+8.109] | +0.27% [-0.05%,+0.57%] | 17.4% (+82.6% removed) |
| `S3+s4r256` | Recent window r=256 (s=4 sinks + 256 recent INT8) | ×1.0005 [0.9971,1.0042] | ×1.0015 [0.9961,1.0074] | +0.370 [-1.434,+2.228] | +0.02% [-0.13%,+0.17%] | 1.7% (+98.3% removed) |
| `L0..9` | S3+s4r32 only in early layers 0..9 (10..29 fp32) | ×0.9972 [0.9878,1.0075] | ×1.0332 [1.0104,1.0574] | +10.387 [+3.713,+17.070] | +0.74% [+0.24%,+1.25%] | 47.2% (+52.8% removed) |
| `L10..19` | S3+s4r32 only in mid layers 10..19 (rest fp32) | ×1.0030 [0.9962,1.0106] | ×1.0049 [0.9971,1.0126] | +0.940 [-0.953,+2.937] | +0.03% [-0.19%,+0.22%] | 4.3% (+95.7% removed) |
| `L20..29` | S3+s4r32 only in late layers 20..29 (rest fp32) | ×0.9998 [0.9958,1.0039] | ×1.0263 [1.0130,1.0403] | +7.898 [+3.942,+12.002] | +0.50% [+0.25%,+0.78%] | 35.9% (+64.1% removed) |

### 2. Single-pass `fp32` Diagnostics (30 layers, `N = 12` EN vs UK)

| Metric | EN mean | UK mean | UK − EN |
| --- | ---: | ---: | ---: |
| K rel L2 error under `S3` | 0.4263 | 0.4259 | -0.0003 |
| V rel L2 error under `S3` | 0.4247 | 0.4264 | +0.0017 |
| Mid/far attn mass `fp32` (`4<=s<=t-32`, `t>=64`) | 43.65% | 44.38% | +0.73 pp |
| Mid/far attn mass under `S3-K+s4r32` (`t>=64`) | 39.42% | 42.33% | +2.91 pp |

| Position block | EN Entropy `fp32 → S3-K` (nats) | UK Entropy `fp32 → S3-K` (nats) | EN Mid/far mass `fp32 → S3-K` | UK Mid/far mass `fp32 → S3-K` |
| --- | --- | --- | --- | --- |
| `0..127` | 1.937 → 2.069 (Δ=+0.1316) | 1.693 → 1.796 (Δ=+0.1027) | 41.66% → 36.99% | 35.67% → 33.12% |
| `128..255` | 2.431 → 2.736 (Δ=+0.3044) | 2.212 → 2.516 (Δ=+0.3035) | 46.43% → 42.73% | 46.50% → 44.58% |
| `256..383` | — | 2.376 → 2.751 (Δ=+0.3749) | — | 51.52% → 49.42% |

### 3. Hypothesis Evaluation `(a)–(d)`

- **K vs V dominance:** Keeping K at INT8 (`K8V:S3+s4r32`, V ternary) gives `D = +19.865` (`90.3%` of baseline `D`), whereas keeping V at INT8 (`KV8:S3+s4r32`, K ternary) gives `D = +8.571` (`39.0%`).
- **Recent-window `r` sensitivity:** `r=0` gives `D = +98.239`, baseline `r=32` gives `D = +21.989`, `r=256` leaves `D = +0.370` (`1.7%`). Note also the caveat whether shorter EN mean length (`~182` tok vs `~307` UK) shrinks UK loss more cleanly at `r=256`.
- **Layer sensitivity (`0..9` / `10..19` / `20..29`):** Early `D=+10.387` (`47.2%`), mid `D=+0.940` (`4.3%`), late `D=+7.898` (`35.9%`).
- **(a) Far-attention weight:** Baseline mid/far mass outside `s4r32` is `44.38%` (UK) vs `43.65%` (EN) (`Δ=+0.73 pp`).
- **(b) Intrinsic K/V quantizability:** Relative L2 `S3` error is `0.4259` vs `0.4263` (K) and `0.4264` vs `0.4247` (V).
- **(d) Attention leakage (`S3-K`):** `S3-K` shifts mid/far mass `44.38% → 42.33%` on UK (`-2.05 pp`) vs `43.65% → 39.42%` on EN (`-4.23 pp`).

### Reproducibility & Metadata
- Command: `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- nice -n 19 ionice -c3 python3 scripts/run_h4_mechanism.py --n-windows 12`
- `commit 8158e04`, governor `powersave`, `n_boot=10000`, paired by window index, `seed=0`.
