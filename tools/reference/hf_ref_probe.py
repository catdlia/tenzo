#!/usr/bin/env python3
"""Probe: load microsoft/bitnet-b1.58-2B-4T from the local HF cache with transformers (offline)
and report greedy continuation + short-window PPL. Used to adjudicate Tenzo vs bitnet.cpp."""
import os, sys, time, math
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(4)
repo = "microsoft/bitnet-b1.58-2B-4T"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
print("loaded in %.1fs" % (time.time() - t0), type(model).__name__, flush=True)
ids = tok("The capital of France is", return_tensors="pt").input_ids
print("prompt ids", ids[0].tolist())
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=24, do_sample=False)
print("greedy:", repr(tok.decode(out[0][ids.shape[1]:])))
txt = open("/tmp/ov/data/wt2_head.txt", encoding="utf-8").read()
enc = tok(txt, return_tensors="pt").input_ids[0]
print("wt2_head tokens", len(enc), "first ids", enc[:8].tolist())
nll, n = 0.0, 0
for w in range(2):
    win = enc[w*512:(w+1)*512].unsqueeze(0)
    with torch.no_grad():
        lg = model(win).logits.float()
    lp = torch.log_softmax(lg[0, :-1], -1)
    tgt = win[0, 1:]
    nll += -lp.gather(1, tgt[:, None]).sum().item(); n += tgt.numel()
    print("window", w, "ppl so far %.2f" % math.exp(nll / n), flush=True)
