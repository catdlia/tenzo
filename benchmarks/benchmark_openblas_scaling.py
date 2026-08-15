#!/usr/bin/env python3
"""
benchmark_openblas_scaling.py - Test OpenBLAS scaling with parallel instances

Same methodology as benchmark_scaling.py but for OpenBLAS via NumPy.
This gives a fair comparison: Tenzo 8 instances vs OpenBLAS 8 instances.
"""

import subprocess
import time
import re
import sys
import os

def run_numpy_instance(workdir, instance_id):
    """Run a single NumPy GEMM benchmark instance"""

    # Python code to run inside Docker
    python_code = '''
import numpy as np
import time
import sys

size = 512  # Same as Tenzo micro_bench effective size
iterations = 100

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
print(f"GFLOPS:{gflops:.2f}")
'''

    cmd = [
        "docker", "run", "--rm",
        "-e", "OMP_NUM_THREADS=1",
        "-e", "OPENBLAS_NUM_THREADS=1",
        "-v", f"{workdir}:/app",
        "-w", "/app",
        "tenzo-dev:latest",
        "bash", "-c",
        f"apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq python3-pip > /dev/null 2>&1 && pip3 install -q numpy && python3 -c '{python_code}'"
    ]

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def parse_gflops(output):
    """Extract GFLOPS from output"""
    match = re.search(r"GFLOPS:([\d\.]+)", output)
    if match:
        return float(match.group(1))
    return 0.0

def run_parallel_benchmark(num_instances, workdir):
    """Run N parallel OpenBLAS instances"""
    print(f"\n--- 🚀 Launching {num_instances} parallel OpenBLAS instance(s) ---")

    processes = []
    start_time = time.time()

    for i in range(num_instances):
        p = run_numpy_instance(workdir, i)
        processes.append(p)

    print(f"  Waiting for {num_instances} process(es) to complete...")

    results = []
    for i, p in enumerate(processes):
        stdout, stderr = p.communicate()
        gflops = parse_gflops(stdout + stderr)
        results.append(gflops)
        if gflops > 0:
            print(f"    Instance {i+1}: {gflops:.2f} GFLOPS")
        else:
            print(f"    Instance {i+1}: Failed to parse")

    elapsed = time.time() - start_time
    total_gflops = sum(results)
    avg_gflops = total_gflops / num_instances if num_instances > 0 else 0

    print(f"  ✅ Completed in {elapsed:.1f}s")
    print(f"  📊 Total Throughput: {total_gflops:.2f} GFLOPS")
    print(f"  📊 Average per Instance: {avg_gflops:.2f} GFLOPS")

    return total_gflops, results

def main():
    workdir = os.getcwd()

    print("="*70)
    print("🥊 OpenBLAS SCALING TEST (for comparison with Tenzo)")
    print("   Running parallel NumPy instances with single-threaded OpenBLAS")
    print("="*70)

    # Test configurations - same as Tenzo scaling test
    counts = [1, 2, 4, 6, 8]
    history = []

    for count in counts:
        total_gflops, results = run_parallel_benchmark(count, workdir)
        history.append((count, total_gflops, results))

        if count < counts[-1]:
            print(f"\n⏳ Cooling down for 5 seconds...")
            time.sleep(5)

    # Print final report
    print("\n\n" + "="*70)
    print("📊 OPENBLAS SCALING REPORT")
    print("="*70)
    print(f"{'Instances':<12} | {'Total GFLOPS':<14} | {'Scaling':<12} | {'Efficiency':<12}")
    print("-" * 70)

    base_gflops = history[0][1] if history[0][1] > 0 else 1

    for count, total, results in history:
        scaling = total / base_gflops
        efficiency = (scaling / count) * 100
        ideal = f"(Ideal: {count}.0x)"

        print(f"{count:<12} | {total:>12.2f}  | {scaling:>6.2f}x {ideal:<10} | {efficiency:>6.1f}%")

    print("-" * 70)

    # Comparison with Tenzo
    print("\n" + "="*70)
    print("🥊 COMPARISON: OpenBLAS vs Tenzo (8 instances)")
    print("="*70)

    openblas_8 = history[-1][1] if len(history) >= 5 else 0
    tenzo_8 = 133.14  # From previous benchmark

    print(f"{'Library':<20} | {'8 Instances GFLOPS':<20} | {'Winner':<15}")
    print("-" * 70)
    print(f"{'OpenBLAS':<20} | {openblas_8:>18.2f}  | {'🏆' if openblas_8 > tenzo_8 else ''}")
    print(f"{'Tenzo':<20} | {tenzo_8:>18.2f}  | {'🏆' if tenzo_8 > openblas_8 else ''}")
    print("-" * 70)

    if tenzo_8 > openblas_8:
        ratio = tenzo_8 / openblas_8 if openblas_8 > 0 else float('inf')
        print(f"\n🏆 TENZO WINS by {ratio:.2f}x!")
    else:
        ratio = openblas_8 / tenzo_8 if tenzo_8 > 0 else float('inf')
        print(f"\n🏆 OpenBLAS WINS by {ratio:.2f}x!")

    return 0

if __name__ == "__main__":
    sys.exit(main())

