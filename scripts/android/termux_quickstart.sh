#!/bin/bash
# ==============================================================================
# Tenzo: 1-Click Termux Setup & Run
# Run this inside Termux on your Android Phone/Tablet
# ==============================================================================

set -e

echo "⚡ Starting Tenzo 1-Click Setup on Android Termux..."

# 1. Update and install required dependencies in Termux (no libomp)
pkg update -y
pkg install -y git clang cmake ninja make python ndk-sysroot

# 2. OpenMP symlink safeguard
if [ -d "$PREFIX/lib" ]; then
    if [ -f "$PREFIX/lib/libomp.a" ] && [ ! -f "$PREFIX/lib/libomp.so" ]; then
        ln -sf "$PREFIX/lib/libomp.a" "$PREFIX/lib/libomp.so" 2>/dev/null || true
    fi
fi

# 3. Build standalone runtime & diagnostics
if [ -d "build-termux" ]; then
    ninja -C build-termux tenzo-inference tenzo-diag
    ./build-termux/tenzo-diag tenzo-frontend/export_output || true
else
    ./scripts/android/build_termux.sh
fi

# 4. Launch interactive chat
python3 scripts/tenzo_cli.py chat
