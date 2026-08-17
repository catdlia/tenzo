#!/bin/bash
# ==============================================================================
# Tenzo: 1-Click Termux Setup & Run
# Run this inside Termux on your Android Phone/Tablet
# ==============================================================================

set -e

echo "⚡ Starting Tenzo 1-Click Setup on Android Termux..."

# 1. Install dependencies
pkg update -y && pkg install -y git clang cmake ninja make python libomp ndk-sysroot

# 2. Build
if [ ! -f "build-termux/tenzo-inference" ]; then
    ./scripts/android/build_termux.sh
fi

# 3. Launch interactive chat
python3 scripts/tenzo_cli.py chat
