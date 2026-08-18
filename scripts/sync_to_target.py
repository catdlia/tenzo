#!/usr/bin/env python3
import os
import sys
import tarfile
import io
import base64
from device_runner import exec_remote

def sync(target):
    print(f"📦 Packing Tenzo source tree for {target}...")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for folder in ["src", "include", "shaders", "examples", "scripts"]:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    if any(p in root for p in ["__pycache__", ".git", ".idea"]):
                        continue
                    for f in files:
                        if f.endswith((".cpp", ".h", ".hpp", ".txt", ".sh", ".py", ".comp", ".vocab")):
                            full_p = os.path.join(root, f)
                            tar.add(full_p, arcname=full_p)
        for f in ["CMakeLists.txt", "Makefile"]:
            if os.path.exists(f):
                tar.add(f, arcname=f)

    tar_bytes = buf.getvalue()
    b64 = base64.b64encode(tar_bytes).decode('ascii')
    print(f"🚀 Transferring archive ({len(tar_bytes)/1024:.1f} KB) to {target}...")
    
    cmd = f"mkdir -p ~/tenzo && cd ~/tenzo && echo '{b64}' | base64 -d | tar -xzf -"
    res = exec_remote(target, cmd, timeout=60)
    print(f"✅ Sync complete on {target}!")
    return res

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tablet"
    sync(target)
