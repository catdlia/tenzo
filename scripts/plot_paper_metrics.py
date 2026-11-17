#!/usr/bin/env python3
"""Plot measured native C++ attention latency from explicitly supplied CSVs.

Example:
  python scripts/plot_paper_metrics.py --csv 'Tab S11=tab_s11_metrics.csv' --output tab.png
The legacy CSV memory and bandwidth columns are deliberately ignored: their
formulas were incorrect. These measurements are attention calls, not generation.
Historical files without build/hardware provenance cannot establish new claims.
"""
import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_csv(path):
    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        raise ValueError(f'Empty measurement file: {path}')
    data = []
    for row in rows:
        context = int(row['context_length'])
        baseline = float(row['baseline_latency_ms'])
        packed = float(row['tenzo_latency_ms'])
        if context <= 0 or not all(math.isfinite(x) and x > 0 for x in (baseline, packed)):
            raise ValueError(f'Invalid measurement in {path}: {row}')
        data.append((context, baseline, packed))
    if len({r[0] for r in data}) != len(data):
        raise ValueError(f'Duplicate context in {path}')
    return sorted(data)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, len(args.csv), figsize=(5 * len(args.csv), 4), squeeze=False)
    for ax, spec in zip(axes[0], args.csv):
        label, path = spec.split('=', 1)
        rows = load_csv(path)
        x = [r[0] for r in rows]
        ax.plot(x, [1000/r[1] for r in rows], 'o-', label='FP32 + per-head copy')
        ax.plot(x, [1000/r[2] for r in rows], 's-', label='Packed + direct indexing')
        ax.set(xlabel='Context length', ylabel='Attention invocations / s', title=label, xscale='log', yscale='log')
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Native C++ synthetic attention — supplied historical CSVs')
    fig.tight_layout(); fig.savefig(args.output, dpi=180); plt.close(fig)

if __name__ == '__main__':
    main()
