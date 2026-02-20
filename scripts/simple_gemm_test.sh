#!/bin/bash
# simple_gemm_test.sh - Test GEMM generation

echo "Building tenzo-cli..."
docker run --rm -v $(pwd):/app tenzo-dev:latest bash -c "\
    cd /tmp && \
    cmake -B/tmp/b -S/app -GNinja -DCMAKE_BUILD_TYPE=Release && \
    ninja -C /tmp/b tenzo-cli && \
    echo '==BUILD DONE==' && \
    /tmp/b/tenzo-cli gemm" 2>&1 | tee gemm_test_output.log

echo ""
echo "Output saved to gemm_test_output.log"
tail -100 gemm_test_output.log

