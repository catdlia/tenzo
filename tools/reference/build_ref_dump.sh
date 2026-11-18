#!/bin/bash
# Host build of the libllama reference dumper (BitNet/build was built on the host; see decision log D4).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-/tmp/ov/bitnet_ref_dump}"
mkdir -p "$(dirname "$OUT")"
clang++ -std=c++17 -O2 -o "$OUT" "$ROOT/tools/reference/bitnet_ref_dump.cpp" \
  -I"$ROOT/BitNet/3rdparty/llama.cpp/include" -I"$ROOT/BitNet/3rdparty/llama.cpp/ggml/include" \
  -L"$ROOT/BitNet/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$ROOT/BitNet/build/bin"
echo "built $OUT"
