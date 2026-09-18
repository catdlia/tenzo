#!/bin/bash
# run_benchmark_simple.sh - Build and run micro_bench benchmark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building micro_bench with MLIR kernel ==="

docker compose -f "$PROJECT_ROOT/docker-compose.yml" run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev bash -c "\
    ninja -C /app/cmake-build-debug micro_bench && \
    echo '=== Running micro_bench ===' && \
    /app/cmake-build-debug/micro_bench"

