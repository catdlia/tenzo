# Tenzo Compiler Makefile
# Convenience wrapper for common operations

.PHONY: all build test clean docker-build docker-run cpu gpu conv2d bench help q

# Default target
all: build

# Quick rebuild and test (fastest iteration)
q:
	@docker compose run --rm dev bash -c "ninja -C /app/cmake-build-debug tenzo-cli && /app/cmake-build-debug/tenzo-cli cpu 2>&1 | tail -8"

# Build inside Docker
build:
	@echo "🔨 Building Tenzo Compiler..."
	docker compose run --rm dev ninja -C /app/cmake-build-debug tenzo-cli

# Configure CMake
configure:
	@echo "⚙️ Configuring CMake..."
	docker compose run --rm dev cmake -B/app/cmake-build-debug -S/app -GNinja -DCMAKE_BUILD_TYPE=Debug

# Clean build
clean:
	@echo "🧹 Cleaning build directory..."
	docker compose run --rm dev rm -rf /app/cmake-build-debug/*

# Run all tests
test: build
	@echo "🧪 Running tests..."
	docker compose run --rm dev /app/cmake-build-debug/tenzo-cli test

# Run CPU benchmark
cpu: build
	@echo "⚡ Running CPU benchmark..."
	docker compose run --rm dev /app/cmake-build-debug/tenzo-cli cpu

# Run large matrix benchmark (768x768)
large: build
	@echo "🔥 Running Large Matrix benchmark..."
	docker compose run --rm dev /app/cmake-build-debug/tenzo-cli large

# Run parallel (multithreaded) benchmark
parallel: build
	@echo "🔀 Running Parallel benchmark..."
	docker compose run --rm dev /app/cmake-build-debug/tenzo-cli parallel

# Run stability test (3 runs)
stability: build
	@echo "📊 Running Stability Test (3 runs)..."
	@for i in 1 2 3; do \
		echo "=== Run $$i ==="; \
		docker compose run --rm dev /app/cmake-build-debug/tenzo-cli cpu 2>&1 | grep -E "Scalar:|Vector:|Speedup:"; \
	done

# Run hardware analysis
hw-analyze:
	@echo "🖥️ Running Hardware Analysis..."
	docker compose run --rm dev python3 /app/scripts/benchmark_hardware.py -b

# Run Conv2D benchmark
conv2d: build
	@echo "🎯 Running Conv2D benchmark..."
	docker compose run --rm dev /app/cmake-build-debug/tenzo-cli conv2d

# Run GPU test
gpu: build
	@echo "🎮 Running GPU test..."
	docker compose run --rm dev /app/cmake-build-debug/tenzo-cli gpu

# Run GPU benchmark
gpu-bench: build
	@echo "🚀 Running GPU benchmark..."
	docker compose run --rm dev /app/cmake-build-debug/tenzo-cli gpu-bench

# Run all benchmarks
bench: build
	@echo "📊 Running all benchmarks..."
	docker compose run --rm dev /app/cmake-build-debug/tenzo-cli all

# Show version
version: build
	docker compose run --rm dev /app/cmake-build-debug/tenzo-cli version

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker compose build dev

# Interactive development shell (fastest for iterating)
dev:
	@echo "🚀 Starting interactive dev shell..."
	@echo "   Inside the shell, use:"
	@echo "   ninja tenzo-cli && ./tenzo-cli cpu"
	@docker compose run --rm dev bash

# Start docker shell (legacy)
docker-shell: dev

# Generate compile_commands.json for IDE
compile-commands: configure
	@echo "📝 Generating compile_commands.json..."
	@cp cmake-build-debug/compile_commands.json . 2>/dev/null || true

# Format code
format:
	@echo "✨ Formatting code..."
	find src -name '*.cpp' -o -name '*.h' | xargs clang-format -i

# Watch mode - rebuild on changes
watch:
	@echo "👀 Watching for changes..."
	@while true; do \
		inotifywait -r -e modify src/ 2>/dev/null && make build; \
	done

# Help
help:
	@echo "╔════════════════════════════════════════════════════╗"
	@echo "║          Tenzo Compiler Build System               ║"
	@echo "╠════════════════════════════════════════════════════╣"
	@echo "║  make build       - Build the compiler             ║"
	@echo "║  make test        - Run all tests                  ║"
	@echo "║  make cpu         - Run CPU MatMul benchmark       ║"
	@echo "║  make large       - Run large matrix (768x768)     ║"
	@echo "║  make parallel    - Run parallel benchmark         ║"
	@echo "║  make stability   - Run 3x stability test          ║"
	@echo "║  make conv2d      - Run Conv2D benchmark           ║"
	@echo "║  make gpu         - Run GPU pipeline test          ║"
	@echo "║  make hw-analyze  - Hardware analysis + benchmark  ║"
	@echo "║  make bench       - Run all benchmarks             ║"
	@echo "║  make clean       - Clean build directory          ║"
	@echo "║  make docker-shell- Start interactive shell        ║"
	@echo "║  make help        - Show this help                 ║"
	@echo "╚════════════════════════════════════════════════════╝"

