#!/bin/bash
set -e

echo "=== Tenzo Engine Benchmark Baseline ==="
echo ""
echo "1. Clone and compile Microsoft's official BitNet engine:"
echo "   git clone https://github.com/microsoft/BitNet"
echo "   cd BitNet"
echo "   pip install -r requirements.txt"
echo "   python3 utils/get_model.py -m microsoft/bitnet-b1.58-2B-4T"
echo "   cmake -B build -DCMAKE_BUILD_TYPE=Release"
echo "   cmake --build build --config Release"
echo ""
echo "2. Run Microsoft's bitnet.cpp:"
echo "   build/bin/run_generate -m models/bitnet_2b/ -p \"What is AI?\" -t 0.0"
echo ""
echo "3. Executing Tenzo Engine:"
echo "   docker compose run --rm dev /app/cmake-build-debug/tenzo-cli generate -m export_output_bitnet -p \"What is AI?\" -t 0.0"
echo ""
echo "=== Tenzo Engine Output ==="
docker compose run --rm dev /app/cmake-build-debug/tenzo-cli generate -m export_output_bitnet -p "What is AI?" -t 0.0
