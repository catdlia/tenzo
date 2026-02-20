#!/bin/bash
# Tenzo Compiler Test Runner
# Usage: ./scripts/run_tests.sh [all|cpu|gpu|conv2d|bench]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/cmake-build-debug"
BINARY="${BUILD_DIR}/tenzo-cli"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  🔧 ${GREEN}Tenzo Compiler Test Suite${NC}          ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    echo ""
}

print_test() {
    echo -e "${YELLOW}▶${NC} Running: $1"
}

print_pass() {
    echo -e "${GREEN}✅ PASSED:${NC} $1"
}

print_fail() {
    echo -e "${RED}❌ FAILED:${NC} $1"
}

check_binary() {
    if [ ! -f "$BINARY" ]; then
        echo -e "${RED}Error: tenzo-cli not found at $BINARY${NC}"
        echo "Please build the project first:"
        echo "  cd $PROJECT_DIR && docker compose run --rm dev ninja -C /app/cmake-build-debug tenzo-cli"
        exit 1
    fi
}

run_cpu_test() {
    print_test "CPU MatMul Benchmark (512x512)"
    if $BINARY cpu 2>&1 | tee /tmp/cpu_test.txt | grep -q "Speedup"; then
        SPEEDUP=$(grep "Speedup" /tmp/cpu_test.txt | head -1)
        print_pass "CPU Benchmark - $SPEEDUP"
        return 0
    else
        print_fail "CPU Benchmark"
        return 1
    fi
}

run_conv2d_test() {
    print_test "Conv2D Test (32x32x3 -> 30x30x64)"
    if $BINARY conv2d 2>&1 | tee /tmp/conv2d_test.txt | grep -q "Conv2D correct"; then
        THROUGHPUT=$(grep "Throughput" /tmp/conv2d_test.txt)
        print_pass "Conv2D - $THROUGHPUT"
        return 0
    else
        print_fail "Conv2D"
        return 1
    fi
}

run_gpu_test() {
    print_test "GPU Pipeline (SPIR-V Generation)"
    if $BINARY gpu 2>&1 | tee /tmp/gpu_test.txt | grep -q "SUCCESS"; then
        print_pass "GPU Pipeline"
        return 0
    else
        print_fail "GPU Pipeline"
        return 1
    fi
}

run_all_tests() {
    local passed=0
    local failed=0

    echo -e "${BLUE}Running all tests...${NC}"
    echo ""

    if run_cpu_test; then ((passed++)); else ((failed++)); fi
    echo ""

    if run_conv2d_test; then ((passed++)); else ((failed++)); fi
    echo ""

    if run_gpu_test; then ((passed++)); else ((failed++)); fi
    echo ""

    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "Tests: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"

    if [ $failed -gt 0 ]; then
        return 1
    fi
    return 0
}

generate_report() {
    echo ""
    echo -e "${BLUE}📊 Generating Performance Report...${NC}"
    echo ""

    REPORT_FILE="${PROJECT_DIR}/benchmark_report.md"

    cat > "$REPORT_FILE" << EOF
# Tenzo Compiler Benchmark Report

Generated: $(date)

## Hardware Info
$(grep -A5 "Hardware Info" /tmp/cpu_test.txt 2>/dev/null || echo "N/A")

## CPU MatMul (512x512)
$(grep -E "Scalar:|Vector:|Speedup" /tmp/cpu_test.txt 2>/dev/null || echo "N/A")

## Conv2D Performance
$(grep -E "Per iteration|Throughput" /tmp/conv2d_test.txt 2>/dev/null || echo "N/A")

## GPU Status
$(grep -E "GPU:|SUCCESS|SPIR-V" /tmp/gpu_test.txt 2>/dev/null || echo "N/A")
EOF

    echo -e "Report saved to: ${GREEN}$REPORT_FILE${NC}"
}

# Main
print_header
check_binary

case "${1:-all}" in
    cpu)
        run_cpu_test
        ;;
    conv2d)
        run_conv2d_test
        ;;
    gpu)
        run_gpu_test
        ;;
    bench)
        run_cpu_test
        echo ""
        run_conv2d_test
        generate_report
        ;;
    all)
        run_all_tests
        generate_report
        ;;
    *)
        echo "Usage: $0 [all|cpu|gpu|conv2d|bench]"
        exit 1
        ;;
esac

