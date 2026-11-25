# H3b (Conditional): 5-Language Equal-Character Recent Window (`S3+s4c120`)

## Plan

- **Trigger:** Conditionally activated per coordinator protocol because `H1` (`T47`) concluded **`PARTIAL`** (`~22%` reduction in UK-minus-EN `R`, leaving `R = +0.98% [+0.55%, +1.40%] > 0`).
- **Machine state (`2026-11-25T03:26:16+02:00`):** `loadavg 2.69 3.81 4.01`, `MemAvailable 9.58 GiB`, `/` `8.4G` free, `governor=powersave`, `AC=1`, active `zen-bin` (`tenzo-b1` waiting cleanly -> no idle-yield).
- **Dataset & Reused Scores (`N = 48` parallel 5-language `FLORES-200` windows from `H3`):**
  - Token IDs: `results/2026-11-24-b_agent2/H3/flores5_{bitnet,eurollm}_{en,uk,pl,bg,de}.ids` (`48` windows, `0` 32-gram overlap).
  - Raw character offsets verified `48/48` across all `5` languages × `2` models (`480/480` exact match against `~/tenzo-refs/datasets/flores200/flores5_texts.json`).
  - Baselines reused directly from `results/2026-11-24-b_agent2/H3/scores/{bitnet,eurollm}/` (`fp32`/`fp` and `S3_s4r32`).
- **Character-Window Scheme (`S3+s4c120`, $C^* = 120$):**
  - Keep `4` sink tokens (`0..3`), current query `t`, and all prior non-sink keys/values `s` with character span `E_t - c_s <= 120` in `int8_absmax`; all older non-sink keys/values in `tern_g32_rot`.
  - Run on both `BitNet-2B-4T` (`MemoryMax=4G`, `MemAvailable >= 6 GiB`) and `EuroLLM-1.7B` (`MemoryMax=6G`, `MemAvailable >= 8 GiB`) after B=1 parity gate (`< 1e-5`).
- **Metrics & Bootstrap (`10,000` paired window samples, `seed = 0`):**
  - Effective coverage/budget: `cov_chars_exact`, `r_eff`, `bits_win`, `bits_at_512`.
  - Quality parity per `(model, L)`: `rel_loss`, $D_L(C^*=120)$, $R_L(C^*=120)$ vs $R_L(r=32)$, paired reduction $R_L(r=32) - R_L(C^*=120)$ with 95% CI, fraction reduced `1 - R_L(C*) / R_L(r=32)`, and across-language Spearman $\rho(\text{tok}_L / \text{tok}_{\text{EN}}, R_L)$.
- **Pre-registered Decision Rule:**
  - **`explains`:** Mean non-EN `R_L(C*)` drops by $> 2/3$ relative to `r=32` AND `95% CI(R_L(C*))` spans zero on $\ge 3/4$ non-EN languages in both models.
  - **`partially_explains`:** `C*=120` significantly reduces `R_L` (`R_L(r=32) - R_L(C*) > 0`) on non-EN languages across models (`~15–65%` reduction) while substantial excess non-EN disparity (`R_L > 0`) persists.
  - **`does_not_explain`:** Equal-character coverage fails to consistently reduce `R_L` across the 4 non-EN languages (or degrades overall disparity).
