#!/usr/bin/env python3
"""PPL with 95% bootstrap CI (windows resampled, seed 0, 10000 draws) for score files, and paired PPL ratio vs a
baseline score file. Usage: ppl_ci.py BASE.txt [OTHER.txt ...]  (same window ids required for pairing)."""
import math, sys
import numpy as np


def rd(p):
    W = {}; c = None
    for l in open(p):
        if l.startswith('w '):
            _, i, n, nll = l.split(); c = int(i); W[c] = (int(n), float(nll))
    return W


def main():
    files = sys.argv[1:]
    base = rd(files[0])
    rs = np.random.RandomState(0)
    ids = sorted(base)
    draws = [rs.randint(0, len(ids), len(ids)) for _ in range(10000)]
    print("| file | windows | PPL | 95%% CI | ratio vs %s | 95%% CI of ratio |" % files[0].split('/')[-1])
    print("|---|---|---|---|---|---|")
    for f in files:
        W = rd(f)
        com = [i for i in ids if i in W]
        n = np.array([W[i][0] for i in com]); nl = np.array([W[i][1] for i in com])
        bn = np.array([base[i][0] for i in com]); bl = np.array([base[i][1] for i in com])
        p = math.exp(nl.sum() / n.sum()); pb = math.exp(bl.sum() / bn.sum())
        ps, rr = [], []
        for s in draws:
            s = s[s < len(com)] if len(com) < len(ids) else s
            if len(s) == 0: continue
            a = math.exp(nl[s].sum() / n[s].sum()); b = math.exp(bl[s].sum() / bn[s].sum())
            ps.append(a); rr.append(a / b)
        print("| %s | %d | %.3f | [%.3f, %.3f] | %.4f | [%.4f, %.4f] |" % (f.split('/')[-1], len(com), p,
              np.percentile(ps, 2.5), np.percentile(ps, 97.5), p / pb, np.percentile(rr, 2.5), np.percentile(rr, 97.5)))


if __name__ == "__main__":
    main()
