#!/usr/bin/env python3
"""Pybind tenzo_runtime vs HF semantics (2026-11-19 fixes): classic_tl1 linear uses HF ActQuant (Qp=127, half-to-even,
floor 1e-5) and KVCache/ExecutionContext take an explicit rope_theta (> 0). Run inside the dev container:
  docker compose run --rm dev python3 /app/scripts/test_pybind_hf_semantics.py
Exit code != 0 on divergence (tolerance 1e-4) or if the old 31-level semantics would also pass (non-discriminating)."""
import sys
sys.path.insert(0, "/app/cmake-build-debug")
import numpy as np
import tenzo_runtime as tr

TOL = 1e-4
ok = True
rng = np.random.RandomState(0)
K, N, scale = 256, 128, 1.5
W = rng.randint(-1, 2, size=(N, K)).astype(np.int8)
x = (rng.uniform(-2, 2, size=(1, K))).astype(np.float32)
x[0, K // 3] = 2.5


def ref(x, qp):
    m = np.maximum(np.abs(x).max(-1, keepdims=True), np.float32(1e-5)).astype(np.float32)
    s = np.float32(qp) / m
    q = np.clip(np.round(x * s), -128, 127)            # np.round = half to even, like torch.round
    return (q.astype(np.int64) @ W.T.astype(np.int64)).astype(np.float32) * (np.float32(1) / s) * np.float32(scale)


got = np.asarray(tr.dispatch_linear_cxx(x, W, None, "classic_tl1", scale), dtype=np.float32).reshape(1, N)
r, bug = ref(x, 127.0), ref(x, 31.0)
e = np.linalg.norm(got - r) / np.linalg.norm(r); eb = np.linalg.norm(bug - r) / np.linalg.norm(r)
print("classic_tl1 linear: rel_l2 vs HF ref %.3e (31 levels would be %.3e)" % (e, eb))
ok &= bool(e <= TOL) and bool(eb > 100 * TOL)

kv = tr.KVCache(num_layers=1, num_q_heads=2, num_kv_heads=1, head_dim=64, max_seq_len=8, kv_mode="fp32", rope_theta=500000.0)
try:
    tr.KVCache(num_layers=1, num_q_heads=2, num_kv_heads=1, head_dim=64, max_seq_len=8, kv_mode="fp32", rope_theta=0.0)
    print("rope_theta=0 accepted (expected rejection)"); ok = False
except Exception as ex:
    print("rope_theta=0 rejected:", type(ex).__name__)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
