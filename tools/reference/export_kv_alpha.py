#!/usr/bin/env python3
"""alphas.json (tools/reference/kv_calib.py) -> text table read by the engine (env TENZO_KV_ALPHA, kv_mode ternary_rot):
one line per (layer head group): "<layer> <head> <group> <alpha_k> <alpha_v>". Usage: export_kv_alpha.py alphas.json out.txt"""
import json, sys
d = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as f:
    f.write("# kv alpha table from %s (windows %s, steps %s)\n" % (sys.argv[1], d.get("windows"), d.get("steps")))
    for l, (ak, av) in enumerate(zip(d["alpha_k"], d["alpha_v"])):
        for h, (rk, rv) in enumerate(zip(ak, av)):
            for g, (a, b) in enumerate(zip(rk, rv)):
                f.write("%d %d %d %.6f %.6f\n" % (l, h, g, a, b))
print("wrote", sys.argv[2], len(d["alpha_k"]), "layers")
