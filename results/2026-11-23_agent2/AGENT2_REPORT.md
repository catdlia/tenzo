# Tenzo Agent 2 Progress Report (2026-11-23)

- **Branch:** `agent2/2026-11-23`
- **Start weights checksum (`models/bitnet-1.58b/weights.bin`):** `68c8f32b7f5d905ed7b3aaaf688e8913ed12227c7d9afc1bf3f429e3479400d1`
- **Initial machine state (`2026-11-23T13:48:30+02:00`):** **User at laptop ("Людина за ноутом")** — `loadavg` `1.69 2.35 2.48`, `MemAvailable` `11,869 MiB`, `/` `5.6G` free (`99%` used), governor `powersave`, active `Telegram`, `zen-bin`, `claude-desktop`, `qemu-system-x86`, `jetbrains-toolbox`. Heavy commands run `nice -n 19 ionice -c3`, builds `-j1`, Python `MemoryMax <= 4G`, benchmarks (`B1`) deferred until clean machine.

## 1. Tasks

| Task | Class | Status | Time | Key Result | Files |
|---|---|---|---|---|---|
| **L1** (T42: `q_dim` fix, warnings, pybind) | L + build | **done** | ~10 min | `attn_sub_norm` & `out_proj` length fixed to `q_dim` across CPU/GPU/CUDA/ROCm/RVV & pybind (`rel_l2=9.91e-8` vs `0.493` bug on `q_dim=256, hidden=128`); zero build warnings; all gates pass (`G6` 5/5 MATCH, `PPL4=16.700700` byte-identical) | `src/runtime/TenzoEngine.cpp`, `src/bindings/PybindModule.cpp`, `src/tests/TestEngineAPI.cpp`, `Makefile`, `results/2026-11-23_agent2/L1/` |
| **L2** (T02: Parity gate v2) | L | in progress | — | — | `scripts/verify_model_parity.py`, `results/2026-11-23_agent2/L2/` |
| **L3** (T40 p1: BPC/byte disparity) | L | pending | — | — | — |
| **H1** (T39: EN+UK mix calib & engine `TENZO_KV_ALPHA`) | H | pending | — | — | — |
| **H2** (T40 p2: language × domain + 3rd model) | H | pending | — | — | — |
| **L4** (T29: Llama-3 byte-level BPE tokenizer) | L + build | pending | — | — | — |
| **H3** (T30: ctx 2048 PTQ & engine eval) | H | pending | — | — | — |
| **B1** (T38: Clean-machine speed queue) | B | pending (blocked on clean machine) | — | — | — |
| **L5** (End-of-day summary & docs) | L | pending | — | — | — |

## 2. Decision Log

- **D1 (`2026-11-23T13:48` — Owner decisions):** Parity gate v2 spec defined in L2 (v1 preserved unchanged); governor remains `powersave` (recorded in CSV metadata, no `sudo`); disk cleanup left to owner (`df -h /` must have `>= 8 GB` free or `>= 12 GB` for 3rd model before large downloads); EN+UK mix calibration starts from scratch and wires `TENZO_KV_ALPHA` into engine; Qwen2.5-1.5B + one open non-gated <= 2B/4GB UK-capable 3rd model allowed into `~/tenzo-refs/hf/` or `~/.cache/huggingface` (never `/tmp`); BPE tokenizer prioritized while user is at laptop.
- **D2 (`2026-11-23T13:48` — Machine state at start):** User at laptop (`Telegram`, `zen-bin`, `claude-desktop`, `qemu-system-x86`, `load1=1.69`, `MemAvailable=11.6 GiB`, `/=5.6G` free). Policy: `nice -n 19 ionice -c3`, `-j1` builds, Python `MemoryMax <= 4G`, defer `B1`. Note: `/` currently has `5.6G < 8G` free, so full-model downloads in `H2` are blocked unless disk space is freed by owner; small tokenizers (`<= 20 MB`) and text slices (`<= 50 MB`) for `L3`/`H2` corpus prep remain allowed under D5.

## 3. Blockers & Disk Watch

- **Disk space on `/`:** `5.6G` free (`99%` used) is below the `8 GB` (`12 GB` for 3rd model) threshold required by owner decision D3 for downloading model weights in `H2`.
- **Clean-machine state for `B1`:** Laptop currently in active desktop use (`load1=1.69`, `zen-bin`/`Telegram` running).

## 4. What Must NOT Be Claimed

- No post-fix speed claims (clean-machine `B1` queue not yet executed).
- No EN+UK mix calibration result or `TENZO_KV_ALPHA` engine result yet (`H1` pending).
- No `L2` v2 parity gate evaluation yet (`L2` in progress).
- No byte-/character-normalized disparity conclusion yet (`L3` pending).
- No `ctx 2048` KV quality result yet (`H3` pending).

## 5. Questions to Owner

*(None currently open beyond those answered in D1–D6; new questions arising during execution will be appended here.)*
