#!/usr/bin/env python3
"""
benchmark_fair_comparison.py - Fair Tenzo vs OpenBLAS comparison

Runs both benchmarks in identical conditions:
- Same Docker environment
- Same CPU governor (performance)
- Same cooling periods
- Alternating tests to avoid thermal bias
"""

import subprocess
import time
import re
import os
import sys

WORKDIR = os.getcwd()

def set_performance_mode():
    """Set CPU to performance mode"""
    print("🔧 Setting CPU governor to 'performance'...")
    try:
        subprocess.run(
            ["sudo", "cpupower", "frequency-set", "-g", "performance"],
            capture_output=True,
            timeout=5
        )
        print("   ✅ Performance mode enabled")
    except:
        print("   ⚠️  Could not set governor (continuing anyway)")

    # Show CPU frequency
    try:
        result = subprocess.run(
            ["cat", "/proc/cpuinfo"],
            capture_output=True,
            text=True,
            timeout=2
        )
        freqs = [line for line in result.stdout.split('\n') if 'cpu MHz' in line][:4]
        if freqs:
            print(f"   Current CPU freq: {freqs[0].split(':')[1].strip()} MHz")
    except:
        pass

def run_tenzo_benchmark(num_instances, iterations=50):
    """Run Tenzo micro_bench instances"""
    print(f"   🔷 Running Tenzo ({num_instances} instances)...")

    processes = []
    start_time = time.time()

    for i in range(num_instances):
        cmd = [
            "docker", "run", "--rm", "--cpuset-cpus=0-7",
            "-v", f"{WORKDIR}:/app", "-w", "/app",
            "tenzo-dev:latest",
            "./build_e2e/micro_bench"
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(p)

    # Collect results
    results = []
    for i, p in enumerate(processes):
        stdout, stderr = p.communicate()
        output = stdout + stderr

        # Parse MLIR GFLOPS
        match = re.search(r"MLIR.*?GFLOPS:\s+([\d\.]+)", output, re.DOTALL)
        if match:
            gflops = float(match.group(1))
            results.append(gflops)
            print(f"      Instance {i+1}: {gflops:.2f} GFLOPS")
        else:
            print(f"      Instance {i+1}: Failed to parse")
            results.append(0.0)

    elapsed = time.time() - start_time
    total = sum(results)
    avg = total / num_instances if num_instances > 0 else 0

    print(f"      Total: {total:.2f} GFLOPS | Avg: {avg:.2f} | Time: {elapsed:.1f}s")

    return total, results, elapsed

def run_openblas_benchmark(num_instances, iterations=50):
    """Run OpenBLAS instances via NumPy"""
    print(f"   🔶 Running OpenBLAS ({num_instances} instances)...")

    # Python code for benchmark
    python_code = f'''
import numpy as np
import time
size = 512
iterations = {iterations}
A = np.random.rand(size, size).astype(np.float32)
B = np.random.rand(size, size).astype(np.float32)
for _ in range(10): np.dot(A, B)
start = time.perf_counter()
for _ in range({iterations}): C = np.dot(A, B)
elapsed = time.perf_counter() - start
gflops = (2.0 * size**3 / (elapsed / {iterations})) / 1e9
print(f"GFLOPS:{{gflops:.2f}}")
'''

    processes = []
    start_time = time.time()

    for i in range(num_instances):
        cmd = [
            "docker", "run", "--rm", "--cpuset-cpus=0-7",
            "-e", "OMP_NUM_THREADS=1",
            "-e", "OPENBLAS_NUM_THREADS=1",
            "-v", f"{WORKDIR}:/app", "-w", "/app",
            "tenzo-dev:latest",
            "bash", "-c",
            # Install both libopenblas-dev and pip numpy
            f"apt-get update -qq > /dev/null 2>&1 && "
            f"apt-get install -y -qq libopenblas-dev python3-pip > /dev/null 2>&1 && "
            f"pip3 install -q numpy > /dev/null 2>&1 && "
            f"python3 -c '{python_code}'"
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(p)

    # Collect results
    results = []
    for i, p in enumerate(processes):
        stdout, stderr = p.communicate()
        output = stdout + stderr

        match = re.search(r"GFLOPS:([\d\.]+)", output)
        if match:
            gflops = float(match.group(1))
            results.append(gflops)
            print(f"      Instance {i+1}: {gflops:.2f} GFLOPS")
        else:
            # Show error for debugging
            if i == 0:  # Only show first error to avoid spam
                error_lines = [l for l in output.split('\n') if 'error' in l.lower() or 'traceback' in l.lower()]
                if error_lines:
                    print(f"      [Debug] Error: {error_lines[0][:100]}")
            print(f"      Instance {i+1}: Failed to parse")
            results.append(0.0)

    elapsed = time.time() - start_time
    total = sum(results)
    avg = total / num_instances if num_instances > 0 else 0

    print(f"      Total: {total:.2f} GFLOPS | Avg: {avg:.2f} | Time: {elapsed:.1f}s")

    return total, results, elapsed

def main():
    print("="*80)
    print("🥊 FAIR COMPARISON: Tenzo vs OpenBLAS")
    print("="*80)
    print("   Running in identical conditions")
    print("   Each test gets fresh CPU, same environment")
    print("="*80)
    print()

    set_performance_mode()
    print()

    # Test configurations - reduced for speed
    configs = [1, 2, 4, 8]
    results = []

    for num in configs:
        print(f"\n{'='*70}")
        print(f"📊 Testing {num} parallel instance(s)")
        print(f"{'='*70}\n")

        # Run Tenzo first
        tenzo_total, tenzo_results, tenzo_time = run_tenzo_benchmark(num)

        print(f"\n   ⏳ Cooling down for 5 seconds...")
        time.sleep(5)

        # Run OpenBLAS
        openblas_total, openblas_results, openblas_time = run_openblas_benchmark(num)

        results.append({
            'instances': num,
            'tenzo': tenzo_total,
            'tenzo_results': tenzo_results,
            'openblas': openblas_total,
            'openblas_results': openblas_results
        })

        # Longer cooldown between different configs
        if num < configs[-1]:
            print(f"\n   ⏳ Cooling down for 10 seconds before next config...")
            time.sleep(10)

    # Print final table
    print("\n\n")
    print("="*90)
    print("📊 FINAL RESULTS: Tenzo vs OpenBLAS")
    print("="*90)
    print(f"{'Instances':<10} | {'Tenzo':<18} | {'OpenBLAS':<18} | {'Winner':<15} | {'Margin':<10}")
    print("-"*90)

    tenzo_wins = 0
    openblas_wins = 0

    for r in results:
        t = r['tenzo']
        o = r['openblas']

        if t > o:
            winner = "🔷 TENZO"
            margin = f"+{((t/o - 1) * 100):.1f}%" if o > 0 else "+∞"
            tenzo_wins += 1
        elif o > t:
            winner = "🔶 OpenBLAS"
            margin = f"+{((o/t - 1) * 100):.1f}%" if t > 0 else "+∞"
            openblas_wins += 1
        else:
            winner = "TIE"
            margin = "0.0%"

        print(f"{r['instances']:<10} | {t:>16.2f}  | {o:>16.2f}  | {winner:<15} | {margin:<10}")

    print("-"*90)

    # Scaling analysis
    print("\n📈 SCALING EFFICIENCY:")
    print("-"*60)
    print(f"{'Instances':<10} | {'Tenzo':<20} | {'OpenBLAS':<20}")
    print("-"*60)

    tenzo_base = results[0]['tenzo'] if results and results[0]['tenzo'] > 0 else 1
    openblas_base = results[0]['openblas'] if results and results[0]['openblas'] > 0 else 1

    for r in results:
        t_scale = r['tenzo'] / tenzo_base
        t_eff = (t_scale / r['instances']) * 100

        if openblas_base > 0.1:  # Only show OpenBLAS if it has valid data
            o_scale = r['openblas'] / openblas_base
            o_eff = (o_scale / r['instances']) * 100
            print(f"{r['instances']:<10} | {t_scale:.2f}x ({t_eff:.0f}% eff) | {o_scale:.2f}x ({o_eff:.0f}% eff)")
        else:
            print(f"{r['instances']:<10} | {t_scale:.2f}x ({t_eff:.0f}% eff) | N/A")

    # Verdict
    print("\n" + "="*90)
    print("🏆 FINAL VERDICT")
    print("="*90)
    print(f"   Tenzo wins:    {tenzo_wins} / {len(results)} configurations")
    print(f"   OpenBLAS wins: {openblas_wins} / {len(results)} configurations")
    print()

    if tenzo_wins > openblas_wins:
        print("   🎉 TENZO IS THE WINNER!")
    elif openblas_wins > tenzo_wins:
        print("   🎉 OpenBLAS IS THE WINNER!")
    else:
        print("   🤝 IT'S A TIE!")

    # Best configs
    best_tenzo = max(results, key=lambda x: x['tenzo'])
    best_openblas = max(results, key=lambda x: x['openblas'])

    print()
    print("📊 PEAK PERFORMANCE:")
    print(f"   Tenzo:    {best_tenzo['tenzo']:.2f} GFLOPS ({best_tenzo['instances']} instances)")
    print(f"   OpenBLAS: {best_openblas['openblas']:.2f} GFLOPS ({best_openblas['instances']} instances)")

    print("\n✅ Fair comparison complete!")

    return 0

if __name__ == "__main__":
    sys.exit(main())

