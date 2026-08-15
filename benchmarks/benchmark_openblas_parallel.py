#!/usr/bin/env python3
"""
benchmark_openblas_parallel.py - OpenBLAS parallel instances benchmark
Runs inside Docker to avoid per-container overhead
"""

import numpy as np
import time
import multiprocessing as mp
import os

# Force single-threaded OpenBLAS per process
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def worker(worker_id, size, iterations, result_queue):
    """Single worker running GEMM benchmark"""
    # Each worker gets its own matrices
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    # Warmup
    for _ in range(10):
        C = np.dot(A, B)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C = np.dot(A, B)
    elapsed = time.perf_counter() - start

    avg_time = elapsed / iterations
    gflops = (2.0 * size**3 / avg_time) / 1e9

    result_queue.put((worker_id, gflops))

def run_parallel_benchmark(num_workers, size=512, iterations=50):
    """Run N parallel workers"""
    print(f"\n--- 🚀 Launching {num_workers} parallel worker(s) ---")

    result_queue = mp.Queue()
    processes = []

    start_time = time.time()

    # Start workers
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(i, size, iterations, result_queue))
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()

    # Collect results
    results = []
    while not result_queue.empty():
        worker_id, gflops = result_queue.get()
        results.append(gflops)
        print(f"    Worker {worker_id}: {gflops:.2f} GFLOPS")

    elapsed = time.time() - start_time
    total_gflops = sum(results)
    avg_gflops = total_gflops / num_workers if num_workers > 0 else 0

    print(f"  ✅ Completed in {elapsed:.1f}s")
    print(f"  📊 Total Throughput: {total_gflops:.2f} GFLOPS")
    print(f"  📊 Average per Worker: {avg_gflops:.2f} GFLOPS")

    return total_gflops

def main():
    print("="*70)
    print("🥊 OpenBLAS PARALLEL SCALING TEST")
    print("   Using multiprocessing with single-threaded OpenBLAS per worker")
    print(f"   NumPy version: {np.__version__}")
    print("="*70)

    # Test configurations
    counts = [1, 2, 4, 6, 8]
    history = []

    for count in counts:
        total = run_parallel_benchmark(count, size=512, iterations=50)
        history.append((count, total))

        if count < counts[-1]:
            print(f"\n⏳ Cooling down for 3 seconds...")
            time.sleep(3)

    # Print report
    print("\n\n" + "="*70)
    print("📊 OPENBLAS PARALLEL SCALING REPORT")
    print("="*70)
    print(f"{'Workers':<12} | {'Total GFLOPS':<14} | {'Scaling':<12} | {'Efficiency':<12}")
    print("-" * 70)

    base = history[0][1] if history[0][1] > 0 else 1

    for count, total in history:
        scaling = total / base
        efficiency = (scaling / count) * 100
        print(f"{count:<12} | {total:>12.2f}  | {scaling:>6.2f}x      | {efficiency:>6.1f}%")

    print("-" * 70)

    # Comparison
    print("\n" + "="*70)
    print("🥊 COMPARISON: OpenBLAS vs Tenzo (8 parallel instances)")
    print("="*70)

    openblas_8 = history[-1][1] if len(history) >= 5 else 0
    tenzo_8 = 133.14

    print(f"{'Library':<20} | {'8 Workers GFLOPS':<20}")
    print("-" * 50)
    print(f"{'OpenBLAS':<20} | {openblas_8:>18.2f}")
    print(f"{'Tenzo':<20} | {tenzo_8:>18.2f}")
    print("-" * 50)

    if tenzo_8 > openblas_8 and openblas_8 > 0:
        print(f"\n🏆 TENZO WINS by {tenzo_8/openblas_8:.2f}x!")
    elif openblas_8 > tenzo_8:
        print(f"\n🏆 OpenBLAS WINS by {openblas_8/tenzo_8:.2f}x!")
    else:
        print("\n🤝 TIE!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

