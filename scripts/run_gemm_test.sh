#!/bin/bash
# Run inside Docker container

cd /tmp/build_final

echo "=== Building tenzo-cli ==="
ninja tenzo-cli

if [ $? -eq 0 ]; then
    echo ""
    echo "=== BUILD SUCCESS ==="
    echo ""
    echo "=== Running GEMM test ==="
    echo ""
    ./tenzo-cli gemm
    echo ""
    echo "=== TEST COMPLETE ==="
else
    echo "=== BUILD FAILED ==="
    exit 1
fi

