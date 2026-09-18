#!/bin/bash
# run_gemm_test.sh - Build and run GEMM benchmark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f /.dockerenv ]; then
    cd /app/cmake-build-debug
    echo "=== Building tenzo-cli ==="
    ninja tenzo-cli
    if [ $? -eq 0 ]; then
        echo ""
        echo "=== BUILD SUCCESS ==="
        echo ""
        echo "=== Running GEMM benchmark ==="
        echo ""
        ./tenzo-cli gemm
        echo ""
        echo "=== TEST COMPLETE ==="
    else
        echo "=== BUILD FAILED ==="
        exit 1
    fi
else
    echo "=== Building and running GEMM test via Docker ==="
    docker compose -f "$PROJECT_ROOT/docker-compose.yml" run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev bash -c "\
        ninja -C /app/cmake-build-debug tenzo-cli && \
        echo '=== Running GEMM benchmark ===' && \
        /app/cmake-build-debug/tenzo-cli gemm"
fi

