# Tenzo Agent 2 Progress Report (2026-11-23)

- **Branch:** `agent2/2026-11-23`
- **Start weights checksum (`models/bitnet-1.58b/weights.bin`):** `68c8f32b7f5d905ed7b3aaaf688e8913ed12227c7d9afc1bf3f429e3479400d1`
- **Initial machine state (`2026-11-23T13:48:30+02:00`):** **User at laptop ("Людина за ноутом")** — `loadavg` `1.69 2.35 2.48`, `MemAvailable` `11,869 MiB`, `/` `5.6G` free (`99%` used), governor `powersave`, active `Telegram`, `zen-bin`, `claude-desktop`, `qemu-system-x86`, `jetbrains-toolbox`. Heavy commands run `nice -n 19 ionice -c3`, builds `-j1`, Python `MemoryMax <= 4G`, benchmarks (`B1`) deferred until clean machine.

## 1. Tasks

| Task | Class | Status | Time | Key Result | Files |
|---|---|---|---|---|---|
| **L1** (T42: `q_dim` fix, warnings, pybind) | L + build | **done** | ~10 min | `attn_sub_norm` & `out_proj` length fixed to `q_dim` across CPU/GPU/CUDA/ROCm/RVV & pybind (`rel_l2=9.91e-8` vs `0.493` bug on `q_dim=256, hidden=128`); zero build warnings; all gates pass (`G6` 5/5 MATCH, `PPL4=16.700700` byte-identical) | `src/runtime/TenzoEngine.cpp`, `src/bindings/PybindModule.cpp`, `src/tests/TestEngineAPI.cpp`, `Makefile`, `results/2026-11-23_agent2/L1/` |
| **L2** (T02: Parity gate v2) | L | **done** | ~8 min | `--gate v2` added (`v1` unchanged); `fp32` & `int8_fused` pass `v2` `max_abs` (7/7), `top-1` (6/7, 7/7), and EN16 PPL CI (`[1.0003,1.0057]`, `[1.0009,1.0070]`), each failing 1 prompt `1−cos` (`p4` for `fp32`, `p2` for `int8_fused`); `ternary_rot` fails `v2` as expected (`p5` `cos=0.9596`, PPL `1.0753`) | `scripts/verify_model_parity.py`, `results/2026-11-23_agent2/L2/parity_v3.md` |
| **L3** (T40 p1: BPC/byte disparity) | L | in progress | — | — | `scripts/analyze_disparity_bpc.py`, `results/2026-11-23_agent2/L3/` |
| **H1** (T39: EN+UK mix calib & engine `TENZO_KV_ALPHA`) | H | in progress | — | — | `results/2026-11-23_agent2/H1/` |
| **H2** (T40 p2: language × domain + 3rd model) | H | pending | — | — | — |
| **L4** (T29: Llama-3 byte-level BPE tokenizer) | L + build | pending | — | — | — |
| **H3** (T30: ctx 2048 PTQ & engine eval) | H | pending | — | — | — |
| **B1** (T38: Clean-machine speed queue) | B | pending (blocked on clean machine) | — | — | — |
| **L5** (End-of-day summary & docs) | L | pending | — | — | — |

## 2. Decision Log

- **D1 (`2026-11-23T13:48` — Owner decisions):** Parity gate v2 spec defined in L2 (v1 preserved unchanged); governor remains `powersave` (recorded in CSV metadata, no `sudo`); disk cleanup left to owner (`df -h /` must have `>= 8 GB` free or `>= 12 GB` for 3rd model before large downloads); EN+UK mix calibration starts from scratch and wires `TENZO_KV_ALPHA` into engine; Qwen2.5-1.5B + one open non-gated <= 2B/4GB UK-capable 3rd model allowed into `~/tenzo-refs/hf/` or `~/.cache/huggingface` (never `/tmp`); BPE tokenizer prioritized while user is at laptop.
- **D2 (`2026-11-23T13:48` — Machine state at start):** User at laptop (`Telegram`, `zen-bin`, `claude-desktop`, `qemu-system-x86`, `load1=1.69`, `MemAvailable=11.6 GiB`, `/=5.6G` free). Policy: `nice -n 19 ionice -c3`, `-j1` builds, Python `MemoryMax <= 4G`, defer `B1`. Note: `/` currently has `5.6G < 8G` free, so full-model downloads in `H2` are blocked unless disk space is freed by owner; small tokenizers (`<= 20 MB`) and text slices (`<= 50 MB`) for `L3`/`H2` corpus prep remain allowed under D5.
- **D3 (`2026-11-23T14:02` — Machine state before L2):** Still "User at laptop" (`loadavg=4.30 4.66 3.83`, `MemAvailable=10,239 MiB`). Ran L2 parity evaluations sequentially with `nice -n 19 ionice -c3`. Preserved exact `v2` gate threshold `1 - cos(Tenzo, golden) <= 1.5 * (1 - cos(HF, golden))` without tuning despite `fp32` (`p4`) and `int8_fused` (`p2`) missing on 1 prompt each; documented for owner review.

## 3. Blockers & Disk Watch

- **Disk space on `/`:** `5.6G` free (`99%` used) is below the `8 GB` (`12 GB` for 3rd model) threshold required by owner decision D3 for downloading model weights in `H2`.
- **Clean-machine state for `B1`:** Laptop currently in active desktop use (`load1=1.69`, `zen-bin`/`Telegram` running).

## 4. What Must NOT Be Claimed

- No post-fix speed claims (clean-machine `B1` queue not yet executed).
- No full `7/7` `v2` parity gate pass for `fp32` or `int8_fused` (each fails `1 - cos <= 1.5 * (1 - cos_ref)` on 1/7 prompts, while passing `max_abs` 7/7, `top-1`, and `EN16 PPL` CI).
- No EN+UK mix calibration result or `TENZO_KV_ALPHA` engine result yet (`H1` running).
- No byte-/character-normalized disparity conclusion yet (`L3` running).
- No `ctx 2048` KV quality result yet (`H3` pending).

## 5. Questions to Owner

1. **L2 Parity Gate `v2` cosine floor:** `fp32` (`p4_ukrainian`: `1−cos = 1.77e-4` vs `1.5×ref = 1.13e-4`) and `int8_fused` (`p2_tech_en`: `3.09e-4` vs `1.96e-4`) pass `v2` `max_abs` (7/7), `top-1`, and `EN16 PPL` CI (`[1.0003, 1.0057]`, `[1.0009, 1.0070]`), failing only because `1−cos_ref` on those specific prompts is tiny (`~1e-4`). Should `v2` keep the strict `1.5×(1−cos_ref)` rule or add a small noise floor (e.g., `max(1.5*(1−cos_ref), 5e-4)`)?
