---
description: How to build and run the Tenzo compiler (hybrid cloud + local Docker)
---
// turbo-all

# Tenzo Build & Run Workflow

## ⚠️ Critical Rules
1. **NEVER compile locally** — the host CPU (i3-1215U) is too weak
2. **NEVER run binaries on the host** — it lacks Vulkan/MLIR runtime libs
3. Build directory is always `cmake-build-debug` — never create `build/`

---

## Build (Remote Compilation)

1. Run the build command:
```bash
make build
```
This executes `remote_build.sh` which:
- Provisions a Hetzner cpx62 server (16 cores) if not already running
- Compiles via `distcc` inside Docker → remote server
- Leaves server running for 55 min (auto-shutdown timer)

2. To stop the cloud server manually:
```bash
make stop-cloud
```

---

## Run Tests & Benchmarks (Local Docker)

All execution happens inside the Docker container via `docker compose run`:

```bash
make test        # Quick validation (CPU + Conv2D + GPU)
make cpu         # CPU MatMul benchmark (512×512)
make large       # Large matrix (1024×1024)
make parallel    # Multithreaded benchmark
make gpu         # GPU/Vulkan pipeline test
make gpu-bench   # GPU vs CPU benchmark
make conv2d      # Conv2D benchmark
make bench       # All benchmarks
```

Each target depends on `build` — if code changed, it will recompile first.

---

## Quick Iteration

```bash
make q    # Rebuild + run CPU test, show last 8 lines only
```

---

## Interactive Development

```bash
make dev  # Opens bash shell inside Docker container
```

---

## CMake Configuration

```bash
make configure   # Run cmake inside Docker container
```

The project uses:
- **ccache** for caching compilation results
- **lld** as the linker
- **distcc** is injected dynamically via `CCACHE_PREFIX=distcc` env var (NOT in CMakeLists.txt)

---

## Docker Setup

- `Dockerfile`: Ubuntu 24.04 + LLVM 21 + Vulkan SDK + Intel GPU drivers
- `docker-compose.yml`: Mounts project at `/app`, passes Intel iGPU (`/dev/dri`), pins to P-cores (cpuset 0-3)

To rebuild the Docker image:
```bash
make docker-build
```
