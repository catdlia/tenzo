#!/bin/bash
# test_packing_compilation.sh - Test if packing kernels compile correctly

echo "╔════════════════════════════════════════════════════════╗"
echo "║  🔧 Testing Packing Kernels Compilation               ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

cd /home/illia/CLionProjects/untitled

echo "Step 1: Build tenzo-cli with packing support..."
docker run --rm -v $(pwd):/app tenzo-dev:latest bash -c "\
    cd /tmp && \
    cmake -B/tmp/build_packing -S/app -GNinja -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -30 && \
    echo '---CMAKE DONE---' && \
    ninja -C /tmp/build_packing tenzo-cli 2>&1 | grep -E '(Building|Linking|error)' | tail -30 && \
    echo '---BUILD COMPLETE---' && \
    ls -lh /tmp/build_packing/tenzo-cli" > packing_build.log 2>&1

if grep -q "BUILD COMPLETE" packing_build.log; then
    echo "✅ Build successful!"
    echo ""
    echo "Step 2: Test packing command..."
    docker run --rm -v $(pwd):/app tenzo-dev:latest /tmp/build_packing/tenzo-cli packing 2>&1 | head -50
else
    echo "❌ Build failed! Check packing_build.log"
    tail -50 packing_build.log
fi

