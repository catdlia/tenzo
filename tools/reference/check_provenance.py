#!/usr/bin/env python3
"""F17 provenance check: is models/bitnet-1.58b/weights.bin consistent with BOTH sources?
  GGUF i2_s (BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf)  and  HF safetensors (microsoft/bitnet-b1.58-2B-4T, local cache),
  and does model.mlir carry the HF per-tensor scales?   Read-only; writes only the JSON given with --out.
"""
import argparse, glob, json, re, struct, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from restore_checkpoint import GGUFReader, dequant_i2_s, pack_tenzo_ternary
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="0,15,29"); ap.add_argument("--out", default=None)
a = ap.parse_args()
layers = [int(x) for x in a.layers.split(",")]
gg = GGUFReader(str(ROOT / "BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"))
sf = safe_open(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/*/model.safetensors"))[0], "pt")
wbin = np.memmap(ROOT / "models/bitnet-1.58b/weights.bin", dtype=np.uint8, mode="r")
mlir = (ROOT / "models/bitnet-1.58b/model.mlir").read_text()
mlir_scales = [float(x) for x in re.findall(r"arith\.constant\s+([0-9.eE+-]+)\s*:\s*f32\s*\n\s*%\w+\s*=\s*\"tenzo\.bitlinear_elut\"", mlir)]

specs = [("attn_q", "self_attn.q_proj", (2560, 2560)), ("attn_k", "self_attn.k_proj", (2560, 640)), ("attn_v", "self_attn.v_proj", (2560, 640)),
         ("attn_output", "self_attn.o_proj", (2560, 2560)), ("ffn_gate", "mlp.gate_proj", (2560, 6912)), ("ffn_up", "mlp.up_proj", (2560, 6912)),
         ("ffn_down", "mlp.down_proj", (6912, 2560))]
order = ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_sub_norm", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_sub_norm", "ffn_down"]
sizes = {"attn_norm": 10240, "attn_q": 2560 * 640, "attn_k": 640 * 640, "attn_v": 640 * 640, "attn_sub_norm": 10240, "attn_output": 2560 * 640,
         "ffn_norm": 10240, "ffn_gate": 6912 * 640, "ffn_up": 6912 * 640, "ffn_sub_norm": 6912 * 4, "ffn_down": 2560 * 1728}
EMBED = 128256 * 2560 * 4; LAYER = 17425408
rows = []
for L in layers:
    off = EMBED + L * LAYER
    cursor = {}
    for n in order:
        cursor[n] = off; off += sizes[n]
    for gname, hname, (K, N) in specs:
        info = gg.tensors[f"blk.{L}.{gname}.weight"]
        gg.file.seek(gg.data_offset + info["offset"]); raw = gg.file.read(N * K // 4 + 32)
        W_gguf = dequant_i2_s(raw, N * K).reshape(N, K)
        gguf_scale = struct.unpack("<f", raw[N * K // 4: N * K // 4 + 4])[0]
        packed_hf = sf.get_tensor(f"model.layers.{L}.{hname}.weight").numpy()               # [N/4, K]
        # HF/transformers unpack_weights convention: the packed tensor [N/4, K] holds row j*(N/4)+r in bit-pair j of byte (r, c)
        W_hf = np.concatenate([((packed_hf >> (2 * j)) & 3).astype(np.int8) - 1 for j in range(4)], axis=0)
        hf_scale = float(sf.get_tensor(f"model.layers.{L}.{hname}.weight_scale").float().item())
        mine = bytes(wbin[cursor[gname]: cursor[gname] + sizes[gname]])
        rows.append({"layer": L, "tensor": gname, "ternary_equal_gguf_vs_hf": bool((W_gguf == W_hf).all()),
                     "mismatch_frac_gguf_vs_hf": float((W_gguf != W_hf).mean()),
                     "weights_bin_equals_pack_of_gguf": mine == pack_tenzo_ternary(W_gguf),
                     "weights_bin_equals_pack_of_hf": mine == pack_tenzo_ternary(W_hf),
                     "gguf_embedded_scale_f32": gguf_scale, "hf_weight_scale": hf_scale, "ratio_gguf_over_hf": gguf_scale / hf_scale if hf_scale else None})
n_expected = 30 * 7
out = {"n_mlir_scales": len(mlir_scales), "n_expected": n_expected, "rows": rows,
       "mlir_scales_first7_layer0": mlir_scales[:7]}
for r in rows:
    print(f"L{r['layer']:>2} {r['tensor']:<12} gguf==hf:{r['ternary_equal_gguf_vs_hf']!s:<5} (mismatch {r['mismatch_frac_gguf_vs_hf']:.2e}) "
          f"bin==pack(gguf):{r['weights_bin_equals_pack_of_gguf']!s:<5} bin==pack(hf):{r['weights_bin_equals_pack_of_hf']!s:<5} "
          f"scale gguf={r['gguf_embedded_scale_f32']:.6g} hf={r['hf_weight_scale']:.6g}")
print("model.mlir bitlinear scales found:", len(mlir_scales), "expected", n_expected, "first 7:", mlir_scales[:7])
if a.out: Path(a.out).write_text(json.dumps(out, indent=1))
