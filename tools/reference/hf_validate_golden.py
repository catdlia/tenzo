#!/usr/bin/env python3
"""Validation of golden_bitnet.py against HF transformers itself (same dtype): run HF BitNetForCausalLM on prompt id
files with forward hooks on selected layers, write intermediates with the golden naming (<L>.<name>_p<pos>.f32) and
first-step logits, so tools/reference/compare_golden_hf.py can diff tensor by tensor.
Run under: systemd-run --user --scope -p MemoryMax=7G -p MemorySwapMax=0 (needs MemAvailable >= 9 GB)."""
import argparse, json, os
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--prompt-ids", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--layers", default="0,29")
ap.add_argument("--pos", default="0,5")
ap.add_argument("--dump-prompt", type=int, default=0)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T", dtype=getattr(torch, a.dtype),
                                             low_cpu_mem_usage=True, attn_implementation=a.attn).eval()
layers = [int(x) for x in a.layers.split(",")]
pos = [int(x) for x in a.pos.split(",")]
state = {"on": False}
manifest = []


def save(name, layer, t):
    if not state["on"]:
        return
    x = t[0]
    for p in pos:
        if p < x.shape[0]:
            fn = "L%d.%s_p%d.f32" % (layer, name, p)
            x[p].detach().float().contiguous().numpy().tofile(os.path.join(a.out, fn))
            manifest.append({"name": name, "layer": layer, "pos": p, "file": fn})


for l in layers:
    L = model.model.layers[l]
    sa, mlp = L.self_attn, L.mlp
    hooks = [(L.input_layernorm, "input_norm", "out"), (sa.q_proj, "q", "out"), (sa.k_proj, "k", "out"),
             (sa.v_proj, "v", "out"), (sa.attn_sub_norm, "attn_out", "in"), (sa.attn_sub_norm, "attn_sub_norm", "out"),
             (sa.o_proj, "o_proj", "out"), (L.post_attention_layernorm, "resid_attn", "in"),
             (L.post_attention_layernorm, "post_attn_norm", "out"), (mlp.gate_proj, "gate", "out"),
             (mlp.up_proj, "up", "out"), (mlp.ffn_sub_norm, "act", "in"), (mlp.ffn_sub_norm, "ffn_sub_norm", "out"),
             (mlp.down_proj, "down", "out"), (L, "resid_ffn", "out")]
    for mod, name, which in hooks:
        def h(m, inp, out, name=name, which=which, l=l):
            t = inp[0] if which == "in" else (out[0] if isinstance(out, tuple) else out)
            save(name, l, t)
        mod.register_forward_hook(h)

for i, pf in enumerate(a.prompt_ids):
    ids = torch.tensor([[int(x) for x in open(pf).read().split()]])
    state["on"] = (i == a.dump_prompt)
    with torch.no_grad():
        lg = model(ids).logits[0, -1].float().numpy().astype(np.float32)
    name = os.path.basename(pf).replace(".ids.txt", "")
    lg.tofile(os.path.join(a.out, name + ".logits.f32"))
    print(name, ids.shape[1], "top1", int(lg.argmax()), flush=True)
json.dump({"source": "hf-%s-%s" % (a.dtype, a.attn), "tensors": manifest}, open(os.path.join(a.out, "manifest.json"), "w"))
