#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$ROOT/results/2026-11-23_agent2/H1/alpha_mix.txt}"
MODEL_DIR="${2:-$ROOT/models/bitnet-1.58b-v2}"
DST="$MODEL_DIR/kv_alpha_mix.txt"
EXPECTED_SHA256="d67ed67b412ab136c63a75cc7239769726b48717ef8b48fd8b5f4e00ca043eeb"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: source alpha file not found: $SRC" >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: target model directory not found: $MODEL_DIR" >&2
  exit 1
fi

src_sha="$(sha256sum "$SRC" | awk '{print $1}')"
if [[ "$src_sha" != "$EXPECTED_SHA256" ]]; then
  echo "ERROR: source sha256 mismatch: got $src_sha expected $EXPECTED_SHA256" >&2
  exit 1
fi

cp -f "$SRC" "$DST"
dst_sha="$(sha256sum "$DST" | awk '{print $1}')"
if [[ "$dst_sha" != "$EXPECTED_SHA256" ]]; then
  echo "ERROR: installed sha256 mismatch: got $dst_sha expected $EXPECTED_SHA256" >&2
  exit 1
fi

echo "Installed $DST (sha256=$dst_sha)"
