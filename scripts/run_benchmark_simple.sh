#!/bin/bash
# Simple test: compile MLIR kernel and build benchmark

echo "=== Building micro_bench with MLIR kernel ==="

# Step 1: Check if we already have micro_bench
if docker run --rm -v $(pwd):/app tenzo-dev:latest ls /tmp/build/micro_bench 2>/dev/null; then
    echo "✅ micro_bench already exists"

    # Run it
    echo ""
    echo "=== Running micro_bench ==="
    docker run --rm -v $(pwd):/app tenzo-dev:latest /tmp/build/micro_bench 2>&1
else
    echo "❌ micro_bench not found, need to build first"
    echo ""
    echo "Building..."
    docker run --rm -v $(pwd):/app tenzo-dev:latest bash -c "\
        cmake -B/tmp/build -S/app -GNinja -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20 && \
        ninja -C /tmp/build micro_bench 2>&1 | grep -E '(Building|Linking)' && \
        echo 'Build complete' && \
        /tmp/build/micro_bench"
fi

