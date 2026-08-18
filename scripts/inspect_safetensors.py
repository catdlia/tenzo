#!/usr/bin/env python3
import json
import sys
import os

path = sys.argv[1] if len(sys.argv) > 1 else "tenzo-frontend/export_output/model.safetensors"
if not os.path.exists(path):
    print(f"File not found: {path}")
    sys.exit(1)

with open(path, "rb") as f:
    h_len = int.from_bytes(f.read(8), "little")
    h = json.loads(f.read(h_len).decode("utf-8"))
    print(f"Total Tensors: {len(h)}")
    for k in sorted(h.keys()):
        if k != "__metadata__":
            print(f"{k:<55} | {str(h[k]['shape']):<20} | {h[k]['dtype']}")
