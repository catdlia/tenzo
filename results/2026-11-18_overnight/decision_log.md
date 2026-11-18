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
