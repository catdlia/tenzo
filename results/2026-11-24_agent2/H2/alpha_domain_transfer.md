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
