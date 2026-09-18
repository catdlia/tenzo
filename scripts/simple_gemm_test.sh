#!/bin/bash
# simple_gemm_test.sh - Test GEMM generation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building and running tenzo-cli gemm..."
docker compose -f "$PROJECT_ROOT/docker-compose.yml" run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev bash -c "\
    ninja -C /app/cmake-build-debug tenzo-cli && \
    echo '==BUILD DONE==' && \
    /app/cmake-build-debug/tenzo-cli gemm" 2>&1 | tee "$PROJECT_ROOT/gemm_test_output.log"

echo ""
echo "Output saved to gemm_test_output.log"
tail -50 "$PROJECT_ROOT/gemm_test_output.log"

