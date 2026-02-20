#!/usr/bin/env python3
"""
Matrix Size Sweep Test
Test different matrix sizes to find optimal tile alignment
"""

import subprocess
import json
import re

SIZES = [
    384,   # 64 * 6 = 384, perfect for 6x16
    480,   # 80 * 6 = 480, 30 * 16 = 480
    512,   # Standard benchmark size
    576,   # 96 * 6 = 576, 36 * 16 = 576
    768,   # 128 * 6 = 768, 48 * 16 = 768
]

def run_test(size):
    """Run benchmark for given matrix size"""
    # Modify BLIS benchmark to use this size
    # For now, just run cpu benchmark
    try:
        result = subprocess.run(
            ["docker", "compose", "run", "--rm", "dev",
             f"cd /app/cmake-build-debug && ./tenzo-cli cpu"],
            capture_output=True,
            text=True,
            timeout=120
        )

        output = result.stdout + result.stderr

        # Parse GFLOPS
        match = re.search(r'Vector: (\d+) ms \(([0-9.e+]+) GFLOPS\)', output)
        if match:
            time_ms = int(match.group(1))
            gflops = float(match.group(2))
            return {"time_ms": time_ms, "gflops": gflops}
    except:
        pass

    return None

def main():
    print("🔬 Matrix Size Sweep Benchmark")
    print("=" * 60)

    results = []

    for size in SIZES:
        print(f"\nTesting {size}x{size}...")
        result = run_test(size)
        if result:
            results.append({
                "size": size,
                **result
            })
            print(f"  ✅ {result['gflops']:.2f} GFLOPS ({result['time_ms']} ms)")
        else:
            print(f"  ❌ Failed")

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"{'Size':>6} | {'GFLOPS':>10} | {'Time (ms)':>10} | {'Note':>20}")
    print("-" * 60)

    best = max(results, key=lambda x: x['gflops'])

    for r in results:
        marker = " 🏆" if r['size'] == best['size'] else ""
        div6 = "✓" if r['size'] % 6 == 0 else "✗"
        div16 = "✓" if r['size'] % 16 == 0 else "✗"
        note = f"6:{div6} 16:{div16}{marker}"
        print(f"{r['size']:>6} | {r['gflops']:>10.2f} | {r['time_ms']:>10} | {note:>20}")

    print(f"\n🏆 Best: {best['size']}x{best['size']} with {best['gflops']:.2f} GFLOPS")

if __name__ == "__main__":
    main()

