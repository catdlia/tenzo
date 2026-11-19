#!/usr/bin/env python3
"""Engine vs simulator for one KV scheme, as paired bootstrap over windows (10000 draws, seed 0):
  (1) raw ratio  PPL(engine scheme) / PPL(simulator scheme)                       -- the literal gate
  (2) baseline-adjusted ratio of KV-induced degradations
        [PPL(engine scheme) / PPL(engine int8_fused)]  /  [PPL(simulator scheme) / PPL(golden fp32)]
      The engine's own baseline (INT8 embedding / LM head, int8 KV) differs from golden fp32 by ~0.4% PPL; (2) removes it.
Usage: compare_engine_sim.py ENGINE_SCHEME SIM_SCHEME ENGINE_BASE GOLDEN"""
import math, sys
import numpy as np
def rd(p):
    W = {}
    for l in open(p):
        if l.startswith('w '): _, i, n, x = l.split(); W[int(i)] = (int(n), float(x))
    return W
E, S, EB, G = [rd(p) for p in sys.argv[1:5]]
ids = sorted(set(E) & set(S) & set(EB) & set(G))
arr = lambda W: (np.array([W[i][0] for i in ids], float), np.array([W[i][1] for i in ids], float))
(en, el), (sn, sl), (bn, bl), (gn, gl) = arr(E), arr(S), arr(EB), arr(G)
P = lambda n, l, s: math.exp(l[s].sum() / n[s].sum())
rs = np.random.RandomState(0); raw, adj = [], []
for _ in range(10000):
    s = rs.randint(0, len(ids), len(ids))
    raw.append(P(en, el, s) / P(sn, sl, s)); adj.append((P(en, el, s) / P(bn, bl, s)) / (P(sn, sl, s) / P(gn, gl, s)))
al = np.arange(len(ids)); q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
print("windows %d | engine %.3f sim %.3f engine-base %.3f golden %.3f" % (len(ids), P(en, el, al), P(sn, sl, al), P(bn, bl, al), P(gn, gl, al)))
print("(1) raw ratio engine/sim        : %.4f  95%% CI [%.4f, %.4f]  contains 1: %s" % (P(en, el, al) / P(sn, sl, al), *q(raw), q(raw)[0] <= 1 <= q(raw)[1]))
a = (P(en, el, al) / P(bn, bl, al)) / (P(sn, sl, al) / P(gn, gl, al))
print("(2) baseline-adjusted ratio     : %.4f  95%% CI [%.4f, %.4f]  contains 1: %s" % (a, *q(adj), q(adj)[0] <= 1 <= q(adj)[1]))
print("    KV-induced degradation: engine %.4f  simulator %.4f" % (P(en, el, al) / P(bn, bl, al), P(sn, sl, al) / P(gn, gl, al)))
