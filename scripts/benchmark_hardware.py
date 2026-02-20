#!/usr/bin/env python3
"""
Tenzo Hardware Analyzer & Benchmark Tool
=========================================
Detects CPU capabilities and calculates theoretical vs actual performance.

Usage:
    python3 scripts/benchmark_hardware.py [--run-benchmark]
"""

import subprocess
import re
import os
import sys
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

@dataclass
class CPUInfo:
    """CPU Hardware Information"""
    model_name: str = "Unknown"
    physical_cores: int = 1
    logical_cores: int = 1
    p_cores: int = 0  # Performance cores (Intel hybrid)
    e_cores: int = 0  # Efficiency cores (Intel hybrid)
    base_freq_ghz: float = 1.0
    turbo_freq_ghz: float = 1.0

    # Instruction sets
    has_avx2: bool = False
    has_avx512: bool = False
    has_vnni: bool = False
    has_amx: bool = False
    has_fma: bool = False
    has_neon: bool = False  # ARM

    # Cache sizes (KB)
    l1_cache_kb: int = 32
    l2_cache_kb: int = 256
    l3_cache_kb: int = 0

@dataclass
class BenchmarkResult:
    """Benchmark execution result"""
    matrix_size: int
    iterations: int
    scalar_time_ms: float
    vector_time_ms: float
    gflops_scalar: float
    gflops_vector: float
    speedup: float

def detect_cpu_info() -> CPUInfo:
    """Detect CPU information from /proc/cpuinfo and lscpu"""
    info = CPUInfo()

    # Read /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()

        # Model name
        match = re.search(r'model name\s*:\s*(.+)', cpuinfo)
        if match:
            info.model_name = match.group(1).strip()

        # Count logical cores
        info.logical_cores = cpuinfo.count('processor\t:')

        # CPU flags
        match = re.search(r'flags\s*:\s*(.+)', cpuinfo)
        if match:
            flags = match.group(1).lower()
            info.has_avx2 = 'avx2' in flags
            info.has_avx512 = 'avx512f' in flags
            info.has_vnni = 'avx512_vnni' in flags or 'avx_vnni' in flags
            info.has_amx = 'amx_tile' in flags
            info.has_fma = 'fma' in flags

    except FileNotFoundError:
        pass  # Not Linux

    # Use lscpu for more details (try both JSON and text parsing)
    try:
        # First try JSON output (works on English systems)
        result = subprocess.run(['lscpu', '-J'], capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for item in data.get('lscpu', []):
                field = item.get('field', '').rstrip(':')
                value = item.get('data', '')

                if field == 'CPU(s)':
                    info.logical_cores = int(value)
                elif field == 'Core(s) per socket':
                    info.physical_cores = int(value)
                elif field == 'CPU max MHz':
                    info.turbo_freq_ghz = float(value) / 1000
                elif field == 'CPU MHz' or field == 'CPU min MHz':
                    if info.base_freq_ghz == 1.0:
                        info.base_freq_ghz = float(value) / 1000
                elif field == 'L1d cache':
                    match = re.search(r'(\d+)', value)
                    if match:
                        info.l1_cache_kb = int(match.group(1))
                elif field == 'L2 cache':
                    match = re.search(r'(\d+)', value)
                    if match:
                        val = int(match.group(1))
                        if 'MiB' in value:
                            val *= 1024
                        info.l2_cache_kb = val
                elif field == 'L3 cache':
                    match = re.search(r'(\d+)', value)
                    if match:
                        val = int(match.group(1))
                        if 'MiB' in value:
                            val *= 1024
                        info.l3_cache_kb = val
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback: parse plain text lscpu output (handles localized systems)
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                # Look for max frequency (handles multiple languages)
                if 'max' in line.lower() and ('mhz' in line.lower() or 'МГц' in line.lower()):
                    match = re.search(r'(\d+)[,.]?\d*', line)
                    if match:
                        freq = float(match.group(1))
                        if freq > 1000:  # Already in MHz
                            info.turbo_freq_ghz = freq / 1000
                        elif freq < 10:  # Already in GHz
                            info.turbo_freq_ghz = freq
                # Look for core count
                if ('core' in line.lower() or 'ядер' in line.lower()) and 'socket' in line.lower():
                    match = re.search(r'(\d+)', line)
                    if match:
                        info.physical_cores = int(match.group(1))
    except FileNotFoundError:
        pass

    # Detect Intel hybrid architecture (P-cores vs E-cores)
    # 12th/13th gen Intel Alder Lake / Raptor Lake
    if 'i3-12' in info.model_name or 'i5-12' in info.model_name or 'i7-12' in info.model_name or \
       'i3-13' in info.model_name or 'i5-13' in info.model_name or 'i7-13' in info.model_name:
        # i3-1215U: 2 P-cores (2 threads each) + 4 E-cores (1 thread each) = 6 cores, 8 threads
        if 'i3-1215U' in info.model_name:
            info.p_cores = 2
            info.e_cores = 4
            info.physical_cores = 6  # 2P + 4E = 6 physical cores
            # Known frequencies for i3-1215U
            if info.base_freq_ghz <= 1.0:
                info.base_freq_ghz = 1.2
            if info.turbo_freq_ghz <= 1.0:
                info.turbo_freq_ghz = 4.4
        elif 'i5-1235U' in info.model_name or 'i5-1240P' in info.model_name:
            info.p_cores = 2
            info.e_cores = 8
            info.physical_cores = 10
            if info.turbo_freq_ghz <= 1.0:
                info.turbo_freq_ghz = 4.4
        elif 'i7-1255U' in info.model_name or 'i7-1260P' in info.model_name:
            info.p_cores = 2
            info.e_cores = 8
            info.physical_cores = 10
        elif 'i5-12' in info.model_name:
            info.p_cores = 4
            info.e_cores = 8
            info.physical_cores = 12
        elif 'i7-12' in info.model_name:
            info.p_cores = 8
            info.e_cores = 8
            info.physical_cores = 16
        else:
            # Generic estimation for unknown 12th gen
            info.p_cores = max(2, info.logical_cores // 4)
            info.e_cores = max(0, info.logical_cores - info.p_cores * 2)
            info.physical_cores = info.p_cores + info.e_cores
    elif info.physical_cores <= 1 and info.logical_cores > 1:
        # Fallback: estimate physical cores from logical
        # For non-hybrid CPUs with hyperthreading
        info.physical_cores = max(1, info.logical_cores // 2)
        info.p_cores = info.physical_cores
        info.e_cores = 0
    else:
        # Non-hybrid architecture (older Intel, AMD, etc.)
        info.p_cores = info.physical_cores
        info.e_cores = 0

    # Check for ARM NEON
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'neon' in f.read().lower():
                info.has_neon = True
    except:
        pass

    return info


def calculate_theoretical_peak(info: CPUInfo) -> Tuple[float, str]:
    """
    Calculate theoretical peak GFLOPS (Rpeak)

    Formula: Cores × Frequency × OpsPerCycle × 2 (FMA = multiply + add)

    AVX2:   256-bit / 32-bit = 8 floats × 2 (FMA) = 16 ops/cycle
    AVX-512: 512-bit / 32-bit = 16 floats × 2 (FMA) = 32 ops/cycle
    NEON:   128-bit / 32-bit = 4 floats × 2 (FMA) = 8 ops/cycle

    For hybrid CPUs (Alder Lake):
    - P-cores: Full AVX2 performance
    - E-cores: ~60-70% of P-core performance (no hyperthreading, lower clock)
    """

    if info.has_avx512:
        ops_per_cycle = 32
        instruction_set = "AVX-512"
    elif info.has_avx2 and info.has_fma:
        ops_per_cycle = 16
        instruction_set = "AVX2 + FMA"
    elif info.has_avx2:
        ops_per_cycle = 8
        instruction_set = "AVX2"
    elif info.has_neon:
        ops_per_cycle = 8
        instruction_set = "NEON"
    else:
        ops_per_cycle = 2  # Scalar FMA
        instruction_set = "Scalar"

    freq = info.turbo_freq_ghz if info.turbo_freq_ghz > 1 else info.base_freq_ghz

    # For hybrid architectures, calculate separately
    if info.p_cores > 0 and info.e_cores > 0:
        # P-cores: full performance at turbo frequency
        p_core_gflops = info.p_cores * freq * ops_per_cycle
        # E-cores: ~75% clock speed of P-cores, and no hyperthreading benefit
        # i3-1215U: P-core turbo 4.4 GHz, E-core max ~3.3 GHz
        e_core_freq = freq * 0.75  # Approximate E-core max frequency
        e_core_gflops = info.e_cores * e_core_freq * ops_per_cycle
        rpeak = p_core_gflops + e_core_gflops
        instruction_set += f" (Hybrid: {info.p_cores}P+{info.e_cores}E)"
    else:
        # Traditional symmetric cores
        rpeak = info.physical_cores * freq * ops_per_cycle

    return rpeak, instruction_set


def get_optimal_tile_sizes(info: CPUInfo) -> Tuple[int, int, int]:
    """
    Calculate optimal tile sizes based on hardware

    Goal: Fit working set in L1 cache while maximizing register utilization
    L1 is typically 32KB, we want ~16KB for data (A, B, C tiles)
    """

    if info.has_avx512:
        # 16 floats per register, 32 ZMM registers
        return (16, 32, 8)
    elif info.has_avx2:
        # 8 floats per register, 16 YMM registers
        return (8, 16, 4)
    elif info.has_neon:
        # 4 floats per register
        return (4, 8, 4)
    else:
        return (4, 4, 4)


def parse_benchmark_output(output: str, matrix_size: int) -> Optional[BenchmarkResult]:
    """Parse tenzo-cli benchmark output"""

    result = BenchmarkResult(
        matrix_size=matrix_size,
        iterations=0,
        scalar_time_ms=0,
        vector_time_ms=0,
        gflops_scalar=0,
        gflops_vector=0,
        speedup=1.0
    )

    # Parse iterations
    match = re.search(r'(\d+)\s*(?:iterations|runs|times)', output, re.IGNORECASE)
    if match:
        result.iterations = int(match.group(1))

    # Parse scalar time
    match = re.search(r'Scalar[:\s]+(\d+(?:\.\d+)?)\s*ms', output, re.IGNORECASE)
    if match:
        result.scalar_time_ms = float(match.group(1))

    # Parse vector time
    match = re.search(r'Vector[:\s]+(\d+(?:\.\d+)?)\s*ms', output, re.IGNORECASE)
    if match:
        result.vector_time_ms = float(match.group(1))

    # Parse speedup
    match = re.search(r'Speedup[:\s]+(\d+(?:\.\d+)?(?:e[+-]?\d+)?)', output, re.IGNORECASE)
    if match:
        result.speedup = float(match.group(1))

    # Calculate GFLOPS
    # MatMul FLOPS = 2 * M * N * K (multiply + add per element)
    if result.iterations > 0 and result.scalar_time_ms > 0:
        total_flops = 2 * (matrix_size ** 3) * result.iterations
        result.gflops_scalar = total_flops / (result.scalar_time_ms * 1e6)  # ms -> s, flops -> gflops

    if result.iterations > 0 and result.vector_time_ms > 0:
        total_flops = 2 * (matrix_size ** 3) * result.iterations
        result.gflops_vector = total_flops / (result.vector_time_ms * 1e6)

    return result


def run_tenzo_benchmark(tenzo_path: str) -> Optional[BenchmarkResult]:
    """Run tenzo-cli and parse results. Uses Docker if local binary doesn't work."""

    print("🔄 Running Tenzo benchmark...")

    # First try local execution
    local_works = False
    if os.path.exists(tenzo_path):
        try:
            # Quick test if binary works
            test = subprocess.run([tenzo_path, 'version'], capture_output=True, timeout=5)
            if test.returncode == 0:
                local_works = True
        except:
            pass

    try:
        if local_works:
            cmd = [tenzo_path, 'cpu']
        else:
            # Use Docker
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cmd = [
                'docker', 'run', '--rm',
                '-v', f'{project_dir}:/app',
                '-w', '/app',
                'tenzo-dev:latest',
                '/app/cmake-build-debug/tenzo-cli', 'cpu'
            ]
            print("   (Using Docker container)")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )

        output = result.stdout + result.stderr
        print(output)

        # Default matrix size (from PipelineTests.cpp)
        matrix_size = 512

        return parse_benchmark_output(output, matrix_size)

    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out")
        return None
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        return None


def print_hardware_report(info: CPUInfo):
    """Print detailed hardware report"""

    rpeak, instr_set = calculate_theoretical_peak(info)
    tile_m, tile_n, tile_k = get_optimal_tile_sizes(info)

    print("\n" + "="*60)
    print("🖥️  TENZO HARDWARE ANALYSIS REPORT")
    print("="*60)

    print(f"\n📊 CPU: {info.model_name}")
    print(f"   Physical Cores: {info.physical_cores}")
    if info.p_cores > 0:
        print(f"   ├─ P-Cores (Performance): {info.p_cores}")
        print(f"   └─ E-Cores (Efficiency):  {info.e_cores}")
    print(f"   Logical Threads: {info.logical_cores}")
    print(f"   Base Frequency: {info.base_freq_ghz:.2f} GHz")
    print(f"   Turbo Frequency: {info.turbo_freq_ghz:.2f} GHz")

    print(f"\n💾 Cache Hierarchy:")
    print(f"   L1 Data: {info.l1_cache_kb} KB")
    print(f"   L2: {info.l2_cache_kb} KB")
    if info.l3_cache_kb > 0:
        print(f"   L3: {info.l3_cache_kb} KB ({info.l3_cache_kb // 1024} MB)")

    print(f"\n⚡ Instruction Set Support:")
    print(f"   AVX2:     {'✅ Yes' if info.has_avx2 else '❌ No'}")
    print(f"   AVX-512:  {'✅ Yes' if info.has_avx512 else '❌ No'}")
    print(f"   FMA:      {'✅ Yes' if info.has_fma else '❌ No'}")
    print(f"   VNNI:     {'✅ Yes' if info.has_vnni else '❌ No'}")
    print(f"   AMX:      {'✅ Yes' if info.has_amx else '❌ No'}")
    if info.has_neon:
        print(f"   NEON:     ✅ Yes")

    print(f"\n🎯 Theoretical Peak Performance:")
    print(f"   Instruction Set: {instr_set}")
    print(f"   Rpeak (single core): {rpeak / info.p_cores:.1f} GFLOPS")
    print(f"   Rpeak (all P-cores): {rpeak:.1f} GFLOPS")

    print(f"\n📐 Optimal Tile Sizes for {instr_set}:")
    print(f"   M × N × K = {tile_m} × {tile_n} × {tile_k}")

    return rpeak


def print_efficiency_report(rpeak: float, benchmark: Optional[BenchmarkResult]):
    """Print efficiency analysis"""

    print("\n" + "="*60)
    print("📈 PERFORMANCE EFFICIENCY ANALYSIS")
    print("="*60)

    if benchmark is None:
        print("\n⚠️  No benchmark results available")
        print("   Run with: python3 benchmark_hardware.py --run-benchmark")
        return

    print(f"\n🧪 Benchmark: MatMul {benchmark.matrix_size}×{benchmark.matrix_size}")
    print(f"   Iterations: {benchmark.iterations}")

    print(f"\n⏱️  Execution Time:")
    print(f"   Scalar (LLVM -O3): {benchmark.scalar_time_ms:.1f} ms")
    print(f"   Vector (Tenzo):    {benchmark.vector_time_ms:.1f} ms")
    print(f"   Speedup:           {benchmark.speedup:.2f}x")

    print(f"\n🚀 Throughput:")
    print(f"   Scalar: {benchmark.gflops_scalar:.2f} GFLOPS")
    print(f"   Vector: {benchmark.gflops_vector:.2f} GFLOPS")

    # Efficiency = Rmax / Rpeak
    efficiency_scalar = (benchmark.gflops_scalar / rpeak) * 100 if rpeak > 0 else 0
    efficiency_vector = (benchmark.gflops_vector / rpeak) * 100 if rpeak > 0 else 0

    # Single-core efficiency (more relevant for single-threaded code)
    # Single P-core peak = freq * 16 ops (AVX2 FMA)
    single_core_peak = 4.4 * 16  # ~70 GFLOPS for P-core at turbo
    single_core_eff = (benchmark.gflops_vector / single_core_peak) * 100 if single_core_peak > 0 else 0

    print(f"\n📊 Hardware Utilization:")
    print(f"   vs All Cores ({rpeak:.1f} GFLOPS): {efficiency_vector:.1f}%")
    print(f"   vs Single P-core ({single_core_peak:.1f} GFLOPS): {single_core_eff:.1f}% ⭐")

    # Recommendations
    print("\n" + "="*60)
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    # Use single-core efficiency for recommendations (we're single-threaded)
    if single_core_eff >= 25:
        print("\n🟢 EXCELLENT: Single-core efficiency above 25%!")
        print("   Next steps to reach 50%+:")
        print("   - Enable multithreading (use all 6 cores)")
        print("   - Memory packing for better cache utilization")
        print("   - Register blocking for FMA optimization")

    elif single_core_eff >= 15:
        print("\n🟡 GOOD: Single-core efficiency 15-25%")
        print("   Recommendations:")
        print("   - Fine-tune tile sizes for your cache hierarchy")
        print("   - Enable loop unrolling with larger factors")
        print("   - Check for memory alignment issues")

    elif single_core_eff >= 5:
        print("\n🟠 MODERATE: Single-core efficiency 5-15%")
        print("   Recommendations:")
        print("   - Enable Affine vectorization")
        print("   - Add cache-level tiling")
        print("   - Check vector.contract generation")

    else:
        print("\n🏆 EXCELLENT: Good hardware utilization!")
        print("   You're close to the metal. Consider:")
        print("   - Testing on larger matrices")
        print("   - Trying different tile configurations")

    # Gap analysis
    gap = rpeak - benchmark.gflops_vector
    print(f"\n📉 Performance Gap: {gap:.1f} GFLOPS left on the table")
    print(f"   You're using {efficiency_vector:.1f}% of available compute")


def main():
    """Main entry point"""

    run_benchmark = '--run-benchmark' in sys.argv or '-b' in sys.argv

    # Detect hardware
    info = detect_cpu_info()
    rpeak = print_hardware_report(info)

    # Run benchmark if requested
    benchmark = None
    if run_benchmark:
        # Try to find working tenzo-cli
        # Check Docker first (more reliable), then local
        possible_paths = [
            '/app/cmake-build-debug/tenzo-cli',  # Docker path first
            './cmake-build-debug/tenzo-cli',
            './cmake-build-release/tenzo-cli',
            './build/tenzo-cli',
        ]

        tenzo_path = None
        for path in possible_paths:
            if os.path.exists(path):
                # Quick check if binary works
                try:
                    test = subprocess.run([path, 'version'], capture_output=True, timeout=5)
                    if test.returncode == 0:
                        tenzo_path = path
                        break
                except:
                    continue

        # If no working local binary, use Docker path
        if tenzo_path is None:
            tenzo_path = '/app/cmake-build-debug/tenzo-cli'  # Will trigger Docker

        if tenzo_path:
            benchmark = run_tenzo_benchmark(tenzo_path)
        else:
            print("\n⚠️  tenzo-cli not found. Build first with: ninja -C build tenzo-cli")

    print_efficiency_report(rpeak, benchmark)

    # Export data as JSON
    output_data = {
        "cpu": {
            "model": info.model_name,
            "physical_cores": info.physical_cores,
            "logical_cores": info.logical_cores,
            "p_cores": info.p_cores,
            "e_cores": info.e_cores,
            "base_freq_ghz": info.base_freq_ghz,
            "turbo_freq_ghz": info.turbo_freq_ghz,
            "avx2": info.has_avx2,
            "avx512": info.has_avx512,
            "fma": info.has_fma,
            "vnni": info.has_vnni,
        },
        "theoretical_peak_gflops": rpeak,
        "optimal_tiles": get_optimal_tile_sizes(info),
    }

    if benchmark:
        output_data["benchmark"] = {
            "matrix_size": benchmark.matrix_size,
            "gflops_scalar": benchmark.gflops_scalar,
            "gflops_vector": benchmark.gflops_vector,
            "speedup": benchmark.speedup,
            "efficiency_percent": (benchmark.gflops_vector / rpeak) * 100 if rpeak > 0 else 0,
        }

    # Save JSON report
    report_path = Path(__file__).parent.parent / "benchmark_results" / "hardware_report.json"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()

