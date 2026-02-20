#!/bin/bash
# Benchmark runner with JSON output for performance tracking
# Usage: ./scripts/benchmark.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/cmake-build-debug"
BINARY="${BUILD_DIR}/tenzo-cli"
OUTPUT_DIR="${PROJECT_DIR}/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "🔬 Running Tenzo Benchmarks..."
echo ""

# Run CPU benchmark and extract results
echo "⚡ CPU MatMul (512x512)..."
CPU_OUTPUT=$($BINARY cpu 2>&1)

CPU_SCALAR=$(echo "$CPU_OUTPUT" | grep "Scalar:" | grep -oE '[0-9]+' | head -1)
CPU_VECTOR=$(echo "$CPU_OUTPUT" | grep "Vector:" | grep -oE '[0-9]+' | head -1)
CPU_SPEEDUP=$(echo "$CPU_OUTPUT" | grep "Speedup:" | grep -oE '[0-9]+\.[0-9]+' | head -1)

# Run Conv2D benchmark
echo "🎯 Conv2D (32x32x3 → 30x30x64)..."
CONV_OUTPUT=$($BINARY conv2d 2>&1)

CONV_TIME=$(echo "$CONV_OUTPUT" | grep "Per iteration:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
CONV_GFLOPS=$(echo "$CONV_OUTPUT" | grep "Throughput:" | grep -oE '[0-9]+\.[0-9]+' | head -1)

# Get hardware info
HW_CPU=$(echo "$CPU_OUTPUT" | grep "CPU:" | awk '{print $2}')
HW_AVX2=$(echo "$CPU_OUTPUT" | grep "AVX2:" | awk '{print $2}')

# Generate JSON report
JSON_FILE="${OUTPUT_DIR}/benchmark_${TIMESTAMP}.json"

cat > "$JSON_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "version": "0.3.0",
  "hardware": {
    "cpu": "${HW_CPU:-unknown}",
    "avx2": "${HW_AVX2:-unknown}"
  },
  "benchmarks": {
    "matmul_512x512": {
      "scalar_ms": ${CPU_SCALAR:-0},
      "vector_ms": ${CPU_VECTOR:-0},
      "speedup": ${CPU_SPEEDUP:-0}
    },
    "conv2d_32x32": {
      "time_ms": ${CONV_TIME:-0},
      "gflops": ${CONV_GFLOPS:-0}
    }
  }
}
EOF

echo ""
echo "📊 Results saved to: $JSON_FILE"
echo ""

# Print summary
echo "╔════════════════════════════════════════════════════╗"
echo "║              Benchmark Summary                     ║"
echo "╠════════════════════════════════════════════════════╣"
printf "║  CPU: %-43s ║\n" "${HW_CPU:-unknown}"
printf "║  MatMul Speedup: %-32s ║\n" "${CPU_SPEEDUP:-N/A}x"
printf "║  Conv2D Throughput: %-29s ║\n" "${CONV_GFLOPS:-N/A} GFLOPS"
echo "╚════════════════════════════════════════════════════╝"

# Keep latest symlink
ln -sf "benchmark_${TIMESTAMP}.json" "${OUTPUT_DIR}/latest.json"

