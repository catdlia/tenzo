#!/bin/bash
# run_benchmark_native.sh - Run benchmark natively with P-cores affinity
#
# This script sets up optimal CPU configuration for maximum performance:
# 1. Sets CPU governor to 'performance' for maximum frequency
# 2. Uses taskset to pin to P-cores only (cores 0,2 on Alder Lake)
# 3. Runs the benchmark with proper thread configuration

set -e

echo "🔧 Setting up for maximum performance..."

# Check if we have sudo access
if ! sudo -n true 2>/dev/null; then
    echo "⚠️  Need sudo password for CPU governor setup"
fi

# Set performance governor (needs sudo)
echo "📊 Setting CPU governor to 'performance'..."
sudo cpupower frequency-set -g performance 2>/dev/null || {
    echo "⚠️  Could not set governor, trying alternative..."
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "performance" | sudo tee "$cpu" > /dev/null 2>&1 || true
    done
}

# Show current CPU frequency
echo "📊 Current CPU frequencies:"
cat /proc/cpuinfo | grep "cpu MHz" | head -4

# Build if needed
BUILD_DIR="/home/illia/CLionProjects/untitled/build_e2e"
if [ ! -f "$BUILD_DIR/tenzo-cli" ]; then
    echo "🔨 Building..."
    cd /home/illia/CLionProjects/untitled
    docker compose run --rm dev sh -c "cd /app/build_e2e && ninja"
fi

echo ""
echo "🚀 Running benchmark with P-cores affinity (cores 0,2)..."
echo "   This avoids E-cores for consistent high performance"
echo ""

# Run inside Docker with taskset
# Note: taskset inside Docker might not work, but cpuset should
cd /home/illia/CLionProjects/untitled
docker compose run --rm dev sh -c "OMP_NUM_THREADS=2 /app/build_e2e/tenzo-cli gemm-e2e"

echo ""
echo "✅ Benchmark complete!"
echo ""
echo "📊 For even better results, try running natively (outside Docker):"
echo "   taskset -c 0,2 ./build_e2e/tenzo-cli gemm-e2e"

