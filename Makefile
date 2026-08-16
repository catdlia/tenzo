# Tenzo Compiler Makefile
# Convenience wrapper for common operations

.PHONY: build build-local build-docker all q configure clean test cpu large parallel stability hw-analyze conv2d gpu gpu-bench bench version docker-build dev docker-shell compile-commands format watch help

# Default target
all: build

# ==========================================
# 🚀 BUILD TARGETS
# ==========================================

# Головний білд: використовує Hetzner
build:
	@echo "🚀 Запуск віддаленої компіляції через Hetzner..."
	./remote_build.sh

# Локальний бекап (на випадок відсутності інтернету)
build-local:
	@echo "⚠️ УВАГА: Запуск локальної компіляції (може бути довго)..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev ninja -C /app/cmake-build-debug tenzo-cli tenzo_runtime

# ==========================================
# 🧪 TESTING & RUNNING (via Docker)
# ==========================================

# Quick rebuild and test (fastest iteration)
q: build
	@docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev bash -c "/app/cmake-build-debug/tenzo-cli cpu 2>&1 | tail -8"

# Run all tests
test: build
	@echo "🧪 Running tests..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli test

# Run CPU benchmark
cpu: build
	@echo "⚡ Running CPU benchmark..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli cpu

# Run large matrix benchmark (768x768)
large: build
	@echo "🔥 Running Large Matrix benchmark..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli large

# Run parallel (multithreaded) benchmark
parallel: build
	@echo "🔀 Running Parallel benchmark..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli parallel

# Run stability test (3 runs)
stability: build
	@echo "📊 Running Stability Test (3 runs)..."
	@for i in 1 2 3; do \
		echo "=== Run $$i ==="; \
		docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli cpu 2>&1 | grep -E "Scalar:|Vector:|Speedup:"; \
	done

# Run Conv2D benchmark
conv2d: build
	@echo "🎯 Running Conv2D benchmark..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli conv2d

# Run Dynamic Inference test
dynamic: build
	@echo "🌊 Running Dynamic Inference test..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli dynamic

# Run GPU test
gpu: build
	@echo "🎮 Running GPU test..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli gpu

# Run GPU benchmark
gpu-bench: build
	@echo "🚀 Running GPU benchmark..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli gpu-bench

# Run text generation
# Usage: make generate PROMPT="Your prompt" [TOKENS=50] [TEMP=0.7]
generate:
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli generate -p "$(if $(PROMPT),$(PROMPT),Tenzo Edge AI)" -n $(if $(TOKENS),$(TOKENS),30) -t $(if $(TEMP),$(TEMP),0.7) -m tenzo-frontend/export_output

# Run all benchmarks
bench: build
	@echo "📊 Running all benchmarks..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli all

# ==========================================
# 🐍 ONNX FRONTEND (Python)
# ==========================================

# Convert ONNX model to Tenzo MLIR
# Usage: make onnx-convert MODEL=model.onnx [OUT=output.mlir]
onnx-convert:
	@echo "🔄 Converting ONNX to Tenzo MLIR..."
	@if [ -d ".venv" ]; then \
		.venv/bin/python3 tenzo-frontend/onnx_to_mlir.py $(MODEL) -o $(if $(OUT),$(OUT),output.mlir); \
	else \
		python3 tenzo-frontend/onnx_to_mlir.py $(MODEL) -o $(if $(OUT),$(OUT),output.mlir); \
	fi

# Run ONNX frontend tests
onnx-test:
	@echo "🧪 Running ONNX Frontend test..."
	@if [ -d ".venv" ]; then \
		.venv/bin/python3 tenzo-frontend/test_simple_mlp.py; \
	else \
		python3 tenzo-frontend/test_simple_mlp.py; \
	fi

# Setup Python environment
frontend-setup:
	@echo "📦 Setting up ONNX frontend dependencies..."
	python3 -m venv .venv
	.venv/bin/pip install -r tenzo-frontend/requirements.txt

# ==========================================
# 🛠 UTILITIES
# ==========================================

# Configure CMake
configure:
	@echo "⚙️ Configuring CMake..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev cmake -B/app/cmake-build-debug -S/app -GNinja -DCMAKE_BUILD_TYPE=Debug

# Clean build
clean:
	@echo "🧹 Cleaning build directory..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev rm -rf /app/cmake-build-debug/*

# Run hardware analysis
hw-analyze:
	@echo "🖥️ Running Hardware Analysis..."
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev python3 /app/scripts/benchmark_hardware.py -b

# Show version
version: build
	docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev /app/cmake-build-debug/tenzo-cli version

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker compose build dev

# Interactive development shell
dev:
	@echo "🚀 Starting interactive dev shell..."
	@docker compose run --rm -e OMP_PLACES=cores -e OMP_PROC_BIND=spread dev bash

docker-shell: dev

# Generate compile_commands.json for IDE
compile-commands: configure
	@echo "📝 Generating compile_commands.json..."
	@cp cmake-build-debug/compile_commands.json . 2>/dev/null || true

# Format code
format:
	@echo "✨ Formatting code..."
	find src -name '*.cpp' -o -name '*.h' | xargs clang-format -i

# Watch mode
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
	@echo "║  make build       - Віддалена збірка (Hetzner)║"
	@echo "║  make build-local - Локальна збірка (в Docker)     ║"
	@echo "║  make test        - Run all tests                  ║"
	@echo "║  make cpu         - Run CPU MatMul benchmark       ║"
	@echo "║  make large       - Run large matrix (768x768)     ║"
	@echo "║  make gpu         - Run GPU pipeline test          ║"
	@echo "╚════════════════════════════════════════════════════╝"

stop-cloud:
	hcloud server delete tenzo-build-node