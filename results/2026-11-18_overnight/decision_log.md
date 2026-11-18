# Decision log (autonomous, no human available)

D1 (02:05) Branch overnight/2026-11-18 cut from recovery (which already contained T01 results de66443).
D2 Builds: Tenzo targets in Docker (cmake-build-debug, Debug config but targets carry -O3 -mavx2 -mfma via TENZO_RUNTIME_FLAGS).
D3 sudo needs a password -> governor stays `powersave` for ALL measurements (flagged in every CSV).
D4 bitnet.cpp reference dumper is built and run on the HOST (BitNet/build was built on the host with host glibc/clang 23;
   the Ubuntu container cannot load those .so files). Only Tenzo code is built in Docker.
D5 Reference validation FIRST: local bitnet.cpp (isHuangXin llama.cpp fork + local commit 0b6459f + 1-line diff in include/bitnet-lut-kernels.h)
   gives WikiText-2 head PPL ~100 and "a small city, ..." for "The capital of France is", while HF transformers
   (microsoft/bitnet-b1.58-2B-4T from the local HF cache, offline) gives PPL ~14-17 and " Paris. Paris is a city that ...".
   => bitnet.cpp in this tree is NOT trusted as ground truth. Ground truth = HF transformers fp32 (tools/reference/hf_ref_dump.py).
   The libllama dumper is still built as requested; its divergence is reported as a finding.
D6 Reference binaries (*.f32) are gitignored (rule 9 spirit: no binaries in git); token ids/greedy/NLL/metrics are text and committed,
   with a sha256 manifest + regeneration commands.
D7 HF ground truth runs in bf16 (checkpoint-native dtype). fp32 attempts: (1) dtype=float32 at load -> OOM-killed (7.2 GB RSS, host has ~7 GB free); (2) load bf16 + model.float() -> transformers raises ".float() is not supported for quantized model". Stopped after 2 approaches. Consequence: the reference itself carries bf16 noise (~3 significant digits in activations); parity thresholds are applied as specified but the report states the reference noise caveat.

## INCIDENT 02:36 (caused by this session) and clock accounting
D8 At 02:36:23 the kernel OOM-killer fired: it killed my HF reference process (5.1 GB RSS), then `claude-desktop`, the user's
   Hyprland session (`wayland-wm@hyprland.desktop.service: Failed with result 'oom-kill'`) and the `systemd-inhibit` sleep process.
   Cause: I started a Docker build of tenzo-cli while the HF bf16 reference (5+ GB) was running with ~2.5 GB MemAvailable, and swap was
   already full, even though I had noted that memory was tight. My Bash call died with exit 137. The session then sat idle until
   the user typed "continue" at 08:33 (6 h later). No data was lost: weights.bin sha256 is unchanged, git was clean up to commit 675a94f.
   Mitigations from here on: HF jobs run under `systemd-run --user --scope -p MemoryMax=7G -p MemorySwapMax=0`, no Docker builds while HF runs
   unless MemAvailable > 8 GB, MemAvailable is checked before heavy steps.
D9 Time budget: wall-clock 02:05 -> 08:33 is not a meaningful measure of work done (about 35 minutes of active work, then ~6 h dead time after the OOM).
   Interpretation chosen: the user's explicit "continue" at 08:33 restarts the run; the 6 h budget is counted in ACTIVE time, so roughly 5 h remain.
   Block order is kept, and the final report states this.
