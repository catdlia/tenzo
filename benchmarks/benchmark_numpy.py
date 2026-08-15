#!/usr/bin/env python3
"""
benchmark_numpy.py - NumPy (OpenBLAS/MKL) GEMM Benchmark

This benchmark tests NumPy's matrix multiplication performance.
NumPy uses OpenBLAS or Intel MKL under the hood - libraries written
by the best engineers over 20+ years.

This is the "gold standard" to compare Tenzo against.
"""

import os
import sys

# Show thread configuration BEFORE importing numpy
print(f"--- 🏆 NumPy (OpenBLAS/MKL) Benchmark ---")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set (using all)')}")
print(f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")
print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")

import numpy as np
import time

def get_numpy_config():
    """Print NumPy's BLAS configuration"""
    print(f"\n📦 NumPy Configuration:")
    print(f"   NumPy version: {np.__version__}")

    # Try to get BLAS info
    try:
        config = np.__config__
        if hasattr(config, 'show'):
            # NumPy 1.x
            print("   BLAS info:")
            np.__config__.show()
        elif hasattr(config, 'blas_info'):
            print(f"   BLAS: {config.blas_info}")
    except Exception as e:
        print(f"   (Could not get BLAS config: {e})")

    # Check for specific libraries
    try:
        import numpy.core._multiarray_umath as _mu
        if hasattr(_mu, '__cpu_features__'):
            features = _mu.__cpu_features__
            print(f"   CPU features: AVX2={features.get('AVX2', '?')}, FMA3={features.get('FMA3', '?')}")
    except:
        pass

def benchmark_gemm(size, iterations=20, warmup=5):
    """
    Benchmark GEMM: C = A @ B

    Args:
        size: Matrix dimension (NxN)
        iterations: Number of timed iterations
        warmup: Number of warmup iterations

    Returns:
        GFLOPS achieved
    """
    # Initialize matrices (float32 to match Tenzo)
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    # Warmup (let CPU boost, fill caches)
    for _ in range(warmup):
        C = np.dot(A, B)

    # Timed benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C = np.dot(A, B)
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    ops = 2.0 * size**3  # 2 * N^3 FLOPs for matrix multiply
    gflops = (ops / avg_time) / 1e9

    print(f"   Size: {size:4d}x{size:4d} | Time: {avg_time*1000:7.2f} ms | Speed: {gflops:6.2f} GFLOPS")
    return gflops

def main():
    get_numpy_config()

    print(f"\n📊 GEMM Benchmark (C = A × B, float32):")
    print(f"   Warmup: 5 iterations, Benchmark: 20 iterations")
    print()

    sizes = [256, 512, 768, 1024]
    results = {}

    for size in sizes:
        gflops = benchmark_gemm(size)
        results[size] = gflops

    # Summary
    print(f"\n" + "="*60)
    print(f"📊 SUMMARY")
    print(f"="*60)

    avg_gflops = sum(results.values()) / len(results)
    max_gflops = max(results.values())
    max_size = max(results, key=results.get)

    print(f"   Average: {avg_gflops:.2f} GFLOPS")
    print(f"   Peak:    {max_gflops:.2f} GFLOPS (at {max_size}x{max_size})")

    # Comparison hint
    print(f"\n💡 Compare with Tenzo results:")
    print(f"   Tenzo Single-core: ~30-35 GFLOPS")
    print(f"   Tenzo 8-instance:  ~133 GFLOPS")

    return 0

if __name__ == "__main__":
    sys.exit(main())

