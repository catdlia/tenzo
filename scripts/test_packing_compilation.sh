#!/bin/bash
# test_packing_compilation.sh - Test if packing kernels compile correctly

echo "╔════════════════════════════════════════════════════════╗"
echo "║  🔧 Testing Packing Kernels Compilation               ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Step 1: Build tenzo-cli with packing support..."
docker compose -f "$PROJECT_ROOT/docker-compose.yml" run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev bash -c "\
    ninja -C /app/cmake-build-debug tenzo-cli 2>&1 | grep -E '(Building|Linking|error)' | tail -30 && \
    echo '---BUILD COMPLETE---' && \
    ls -lh /app/cmake-build-debug/tenzo-cli" > /tmp/packing_build.log 2>&1

if grep -q "BUILD COMPLETE" /tmp/packing_build.log; then
    echo "✅ Build successful!"
    echo ""
    echo "Step 2: Test packing command..."
    docker compose -f "$PROJECT_ROOT/docker-compose.yml" run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli packing 2>&1 | head -50
else
    echo "❌ Build failed! Check /tmp/packing_build.log"
    tail -50 /tmp/packing_build.log
fi

