#!/usr/bin/env python3
"""KV-cache bytes per token for BitNet-b1.58-2B-4T (30 layers, 5 KV heads, head_dim 128), from the engine's actual layouts
(src/runtime/TenzoEngine.h FusedKVCache). Prints a markdown table; the fixed part is independent of the context length."""
L, H, D = 30, 5, 128
per = {
    "fp32": (2 * D * 4, 0),
    "int8_fused": (2 * (D + 4), 0),                       # int8 values + fp32 scale per token and head, K and V
    "tl1_fused": (2 * (D // 4 + 4), 0),
    "popcount_fused": ((32 + 4) + (D // 4 + 4), 0),        # K bitplanes (32 B) + scale, V TL1 + scale
    "ternary_rot": (2 * (D // 4 + (D // 32) * 4), 36 * 2 * (D + 4)),   # 2-bit codes + 4 fp32 group scales; fixed INT8 ring (4 sinks + 32 recent)
}
print("| kv mode | bytes/token (all layers) | fixed bytes (all layers) | MB at ctx 2048 | MB at ctx 8192 |\n|---|---|---|---|---|")
for k, (v, fixed) in per.items():
    bpt, fx = v * H * L, fixed * H * L
    print("| %s | %d | %d | %.1f | %.1f |" % (k, bpt, fx, (bpt * 2048 + fx) / 2**20, (bpt * 8192 + fx) / 2**20))
