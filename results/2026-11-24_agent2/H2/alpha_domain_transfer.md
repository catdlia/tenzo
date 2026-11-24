# H2 — Cross-Domain Transfer of `alpha_mix.txt` in `TenzoEngine` (`ctx = 512`)

## Plan

- **Question (`D2` / `H2`):** Does the per-layer/per-head clip calibration table `results/2026-11-23_agent2/H1/alpha_mix.txt` (`sha256: d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb`, calibrated on `calib_en8.ids` WikiText-2 train + `calib_uk8.ids` Kobza rows `100..199`) transfer cleanly outside its calibration domains, or only help in-domain text?
- **Evaluation cells (`results/2026-11-23_agent2/H2/bitnet_{en,uk}_{wiki,web}.ids`, `16 × 511` scored tokens each):**
  - `en_wiki` (`in-domain` proxy — `wikimedia/wikipedia` `20231101.en`, rows `0..99`, encyclopedic text close to WikiText-2 train).
  - `uk_web` (`in-domain` — `Goader/kobza` rows `200..299`, web text from same corpus as `calib_uk8.ids` rows `100..199`).
  - `en_web` (`out-of-domain` — `HuggingFaceFW/fineweb-edu` `sample-10BT`).
  - `uk_wiki` (`out-of-domain` — `wikimedia/wikipedia` `20231101.uk`, rows `0..99`).
- **Engine & modes (`TenzoEngine`, `-c 512 --prefill batched`):**
  - `int8_fused` (baseline)
  - `ternary_rot` (`S3+s4r32`)
  - `ternary_rot --kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt` (verified via `# kv_alpha=... sha256=...` score header)
- **Primary metric & pre-registered decision rule (`10,000`-rep paired window bootstrap, `seed=0`):**
  - Ratios vs `int8_fused` (`ratio_rot`, `ratio_alpha`) and calibrated alpha gain `gain_alpha = PPL(ternary_rot) / PPL(ternary_rot + alpha_mix) = ratio_rot / ratio_alpha` with paired 95% window-bootstrap CI.
  - **"Calibrated `alpha_mix` transfers out-of-domain"** iff `95% CI(gain_alpha) > 1.0` (`lo > 1.0`) on BOTH out-of-domain cells (`en_web` and `uk_wiki`), used to form the `D2` recommendation for `--kv-quant ternary_rot` defaults.

## Results

| Cell | Domain vs `calib_mix` | `int8_fused` PPL | `ternary_rot` PPL | `rot / int8` [95% CI] | `rot+alpha` PPL | `alpha / int8` [95% CI] | Gain `rot / alpha` [95% CI] | Transfers? |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| `en_wiki` | in-domain (WT2 calib ~ wikimedia en_wiki) | 18.534 | 18.850 | ×1.0170 [1.0106, 1.0231] | 18.723 | ×1.0102 [1.0043, 1.0161] | **×1.0068 [1.0032, 1.0106]** | YES (CI>1) |
| `uk_web` | in-domain (Kobza calib, rows 100-199 vs 200-299) | 188.754 | 215.995 | ×1.1443 [1.1238, 1.1663] | 207.838 | ×1.1011 [1.0840, 1.1194] | **×1.0392 [1.0275, 1.0514]** | YES (CI>1) |
| `en_web` | out-of-domain (FineWeb-Edu) | 15.105 | 15.570 | ×1.0308 [1.0199, 1.0427] | 15.396 | ×1.0193 [1.0121, 1.0267] | **×1.0113 [1.0052, 1.0179]** | YES (CI>1) |
| `uk_wiki` | out-of-domain (Wikimedia UK) | 67.474 | 74.358 | ×1.1020 [1.0785, 1.1258] | 72.776 | ×1.0786 [1.0621, 1.0967] | **×1.0217 [1.0121, 1.0308]** | YES (CI>1) |

### Decision & Provenance
- **Out-of-domain transfer (`en_web` & `uk_wiki` CI > 1):** `True` (all 4 cells > 1: `True`).
- **D2 Recommendation:** **Recommend making `alpha_mix` available/defaulted for `ternary_rot` (or opt-in with D2 provenance)** — all out-of-domain cells (`en_web`, `uk_wiki`) show strict improvement (`95% CI > 1.0`).
- `en_wiki`: `# kv_alpha=/app/results/2026-11-23_agent2/H1/alpha_mix.txt sha256=d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb`
- `uk_web`: `# kv_alpha=/app/results/2026-11-23_agent2/H1/alpha_mix.txt sha256=d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb`
- `en_web`: `# kv_alpha=/app/results/2026-11-23_agent2/H1/alpha_mix.txt sha256=d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb`
- `uk_wiki`: `# kv_alpha=/app/results/2026-11-23_agent2/H1/alpha_mix.txt sha256=d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb`

### Reproducibility & Metadata
- Command: `python3 scripts/run_h2_alpha_transfer.py` (`tenzo-inference -m /app/models/bitnet-1.58b-v2 -c 512 --prefill batched`, `--kv-alpha /app/results/2026-11-23_agent2/H1/alpha_mix.txt`).
- `commit c8de26c`, governor `powersave`, `n_boot=10000`, paired by window index, `seed=0`.
