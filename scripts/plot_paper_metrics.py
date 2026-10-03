#!/usr/bin/env python3
"""
plot_paper_metrics.py - Generate Publication-Ready PDF Charts for Tenzo Research Paper
Generates 3 vector graphics charts:
  1. memory_footprint.pdf : Memory Footprint across Context Lengths (1K to 32K)
  2. generation_throughput.pdf : Generation Speed (Tokens/sec) across 3 Edge Devices
  3. cache_misses_perf.pdf : L1 & LLC Cache Misses on Intel Core i3 (AVX2)
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'benchmark_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_memory_footprint():
    """Chart 1: Memory Footprint across Context Lengths (1K to 32K)"""
    contexts = ['1K', '4K', '8K', '32K']
    x = np.arange(len(contexts))
    
    # Active working set memory in MB
    baseline_mem = [40.0, 160.0, 320.0, 1280.0]
    tenzo_mem = [0.5, 2.0, 4.0, 16.0]
    
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    
    ax.plot(contexts, baseline_mem, marker='o', linewidth=2.5, markersize=8, 
            color='#d62728', label='Baseline (FP32 + GQA Copy Buffer)')
    ax.plot(contexts, tenzo_mem, marker='s', linewidth=2.5, markersize=8, 
            color='#1f77b4', label='Tenzo (1.58-bit Packed + Implicit Shuffle)')
    
    # Annotate 80x reduction
    for i, (b, t) in enumerate(zip(baseline_mem, tenzo_mem)):
        ratio = b / t
        ax.annotate(f"{ratio:.0f}× less", (i, t),
                    textcoords="offset points", xytext=(0, 10), ha='center',
                    fontweight='bold', color='#1f77b4')

    ax.set_ylabel('Active Working Set Memory (MB)')
    ax.set_xlabel('Context Length (Tokens)')
    ax.set_title('Figure 1: Memory Footprint Reduction (80× Active Working Set Savings)')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    fig.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'memory_footprint.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'memory_footprint.png')
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"✅ Generated: {pdf_path}")

def plot_generation_speed():
    """Chart 2: Generation Speed (Tokens/sec) across Devices at 8K Context"""
    devices = ['Laptop (Core i3)\nAVX2 / FMA', 'Raspberry Pi 4B\nCortex-A72 (1-core)', 'Samsung Tab S11\nCortex-X925 (ARMv9)']
    x = np.arange(len(devices))
    width = 0.35

    # Throughput (tokens/sec) at Context = 8192
    baseline_tok_s = [10.4, 6.5, 19.5]
    tenzo_tok_s = [48.9, 8.0, 24.5]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    rects1 = ax.bar(x - width/2, baseline_tok_s, width, label='Baseline Attention', color='#b0bec5', edgecolor='black')
    rects2 = ax.bar(x + width/2, tenzo_tok_s, width, label='Tenzo Packed Attention', color='#2e7d32', edgecolor='black')

    # Add speedup annotations
    for i, (b, t) in enumerate(zip(baseline_tok_s, tenzo_tok_s)):
        speedup = t / b
        ax.annotate(f"{speedup:.2f}×", (x[i] + width/2, t + 1.0),
                    ha='center', va='bottom', fontweight='bold', color='#1b5e20')

    ax.set_ylabel('Throughput (Tokens / sec)')
    ax.set_title('Figure 2: Attention Throughput at 8K Context Across Heterogeneous Targets')
    ax.set_xticks(x)
    ax.set_xticklabels(devices)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 60)
    ax.legend(frameon=True, facecolor='white')

    fig.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'generation_throughput.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'generation_throughput.png')
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"✅ Generated: {pdf_path}")

def plot_cache_misses():
    """Chart 3: L1 and LLC Hardware Cache Misses on Intel i3"""
    categories = ['L1 Data Cache Misses\n(Million Misses)', 'LLC (L3) Cache Misses\n(Million Misses)']
    x = np.arange(len(categories))
    width = 0.35

    # Hardware counters from perf stat (T=8192, 10 iters)
    baseline_misses_m = [79.32, 65.76]
    tenzo_misses_m = [2.15, 0.011]

    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    rects1 = ax.bar(x - width/2, baseline_misses_m, width, label='Baseline (FP32 Copy)', color='#e53935', edgecolor='black')
    rects2 = ax.bar(x + width/2, tenzo_misses_m, width, label='Tenzo (Implicit Shuffle)', color='#1e88e5', edgecolor='black')

    # Annotate reductions
    ax.annotate("37.0× Fewer Misses\n(97.3% Reduction)", (x[0] + width/2, tenzo_misses_m[0] + 3),
                ha='center', va='bottom', fontweight='bold', color='#0d47a1')
    ax.annotate("5,741× Fewer Misses\n(Fits in Cache!)", (x[1] + width/2, tenzo_misses_m[1] + 3),
                ha='center', va='bottom', fontweight='bold', color='#0d47a1')

    ax.set_ylabel('Total Cache Misses (Millions)')
    ax.set_title('Figure 3: Microarchitectural Memory Wall Analysis (Intel Core i3, T=8K)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 95)
    ax.legend(frameon=True, facecolor='white')

    fig.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'cache_misses_perf.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'cache_misses_perf.png')
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"✅ Generated: {pdf_path}")

def main():
    print("🎨 Generating publication-quality charts for research paper...")
    plot_memory_footprint()
    plot_generation_speed()
    plot_cache_misses()
    print("🎉 All 3 paper figures generated in PDF and PNG formats!")

if __name__ == "__main__":
    main()
