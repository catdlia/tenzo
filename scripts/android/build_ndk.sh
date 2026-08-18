#!/bin/bash
# ==============================================================================
# Tenzo Compiler: Android NDK Cross-Compilation Script
# ==============================================================================

set -e

if [ -z "${ANDROID_NDK_HOME}" ] && [ -z "${ANDROID_NDK}" ]; then
    echo "❌ Error: ANDROID_NDK_HOME or ANDROID_NDK environment variable must be set."
    echo "Example: export ANDROID_NDK_HOME=/opt/android-sdk/ndk/25.2.9519653"
    exit 1
fi

NDK_PATH="${ANDROID_NDK_HOME:-$ANDROID_NDK}"
BUILD_DIR="build-android-ndk"
API_LEVEL=28 # Android 9.0+

echo "🚀 Cross-compiling Tenzo for aarch64-linux-android (API ${API_LEVEL})..."

mkdir -p "${BUILD_DIR}"

cmake -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${NDK_PATH}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM="android-${API_LEVEL}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_STL=c++_static \
    -DANDROID_CPP_FEATURES="rtti exceptions" \
    -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+fp16 -O3 -ffast-math"

ninja -C "${BUILD_DIR}" tenzo_runtime_static tenzo-inference

echo "✅ Android NDK build completed: ${BUILD_DIR}/tenzo-inference"
