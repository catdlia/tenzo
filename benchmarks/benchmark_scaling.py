#!/usr/bin/env python3
"""
benchmark_scaling.py - Test CPU scaling with parallel GEMM instances

This script launches multiple Docker containers running micro_bench in parallel
to measure how well the system scales with parallel workloads.

Expected results for Intel Alder Lake i3-1215U (2 P-cores + 4 E-cores):
- 1 thread:  ~55 GFLOPS (single P-core at turbo)
- 2 threads: ~70-80 GFLOPS (both P-cores)
- 4 threads: ~80-90 GFLOPS (P-cores + some E-cores)
- 6+ threads: ~80 GFLOPS or less (thermal throttling + memory bottleneck)
"""

import subprocess
import time
import re
import sys
import os

def parse_gflops(output):
    """Extract GFLOPS from benchmark output (prefer MLIR, fallback to Intrinsics)"""
    # Try MLIR-Generated Kernel first
    match = re.search(r"MLIR-Generated Kernel:.*?GFLOPS:\s+([\d\.]+)", output, re.DOTALL)
    if match:
        return float(match.group(1))

    # Fallback to AVX2 Intrinsics
    match = re.search(r"AVX2 Intrinsics.*?GFLOPS:\s+([\d\.]+)", output, re.DOTALL)
    if match:
        return float(match.group(1))

    # Try any GFLOPS pattern
    match = re.search(r"GFLOPS:\s+([\d\.]+)", output)
    if match:
        return float(match.group(1))

    return 0.0

def run_parallel_bench(num_instances, workdir):
    """Run N parallel Docker containers with micro_bench"""
    print(f"\n--- 🚀 Launching {num_instances} parallel instance(s) ---")

    processes = []
    start_time = time.time()

    # Launch N Docker processes in parallel
    for i in range(num_instances):
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{workdir}:/app",
            "-w", "/app",
            "tenzo-dev:latest",
            "./build_e2e/micro_bench"
        ]

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(p)

    print(f"  Waiting for {num_instances} process(es) to complete...")

    # Collect results
    results = []
    for i, p in enumerate(processes):
        stdout, stderr = p.communicate()
        gflops = parse_gflops(stdout + stderr)
        results.append(gflops)
        if gflops > 0:
            print(f"    Instance {i+1}: {gflops:.2f} GFLOPS")
        else:
            print(f"    Instance {i+1}: Failed to parse (check output)")
            # Debug: print first 500 chars
            # print(f"    Output: {(stdout + stderr)[:500]}")

    elapsed = time.time() - start_time
    total_gflops = sum(results)
    avg_gflops = total_gflops / num_instances if num_instances > 0 else 0

    print(f"  ✅ Completed in {elapsed:.1f}s")
    print(f"  📊 Total Throughput: {total_gflops:.2f} GFLOPS")
    print(f"  📊 Average per Instance: {avg_gflops:.2f} GFLOPS")

    return total_gflops, results

def main():
    # Get working directory
    workdir = os.getcwd()
    print(f"Working directory: {workdir}")

    # Check if micro_bench exists
    bench_path = os.path.join(workdir, "build_e2e", "micro_bench")
    if not os.path.exists(bench_path):
        print(f"❌ Error: {bench_path} not found!")
        print("   Run: docker compose run --rm dev sh -c 'cd /app/build_e2e && ninja'")
        sys.exit(1)

    print("\n" + "="*60)
    print("🧪 CPU SCALING TEST")
    print("   Testing parallel throughput on Intel Alder Lake")
    print("   (2 P-cores + 4 E-cores = 6 cores, 8 threads)")
    print("="*60)

    # Test configurations
    counts = [1, 2, 4, 6, 8]
    history = []

    # Set performance governor if possible
    print("\n🔧 Attempting to set CPU governor to 'performance'...")
    try:
        subprocess.run(
            ["sudo", "cpupower", "frequency-set", "-g", "performance"],
            capture_output=True,
            timeout=5
        )
        print("   ✅ Governor set to performance")
    except Exception as e:
        print(f"   ⚠️  Could not set governor: {e}")

    # Run scaling tests
    for count in counts:
        total_gflops, results = run_parallel_bench(count, workdir)
        history.append((count, total_gflops, results))

        # Cool down between tests
        if count < counts[-1]:
            print(f"\n⏳ Cooling down for 5 seconds...")
            time.sleep(5)

    # Print final report
    print("\n\n" + "="*70)
    print("📊 FINAL SCALING REPORT")
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

    # Find optimal configuration
    best_count, best_gflops, _ = max(history, key=lambda x: x[1])
    print(f"\n🏆 OPTIMAL: {best_count} parallel instances = {best_gflops:.2f} GFLOPS")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if best_count <= 2:
        print("   - P-cores only provide best performance")
        print("   - E-cores add overhead without benefit for this workload")
        print("   - Consider using taskset -c 0,2 for native execution")
    elif best_count <= 4:
        print("   - P-cores with HyperThreading provide best scaling")
        print("   - Some E-cores may help with larger workloads")
    else:
        print("   - All cores contribute to throughput")
        print("   - Memory bandwidth may be the limiting factor")

    print("\n✅ Scaling test complete!")

    return 0

if __name__ == "__main__":
    sys.exit(main())

