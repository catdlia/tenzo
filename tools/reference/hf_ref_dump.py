#!/usr/bin/env python3
"""Ground-truth reference: HF transformers BitNetForCausalLM (bf16 = checkpoint-native dtype, offline local cache of
microsoft/bitnet-b1.58-2B-4T).  Modes:
  prompts : tokenise prompts.json -> ids, first-step logits (float32, V=128256), 32 greedy tokens.
  nll     : teacher-forced per-token log-probs for windows.ids (same text format as `tenzo-inference --score-out`).
"""
import argparse, json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["prompts", "nll"])
ap.add_argument("--prompts", default="tools/reference/prompts.json")
ap.add_argument("--windows", default="results/reference/windows.ids")
ap.add_argument("--out", default="results/reference/hf")
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
repo = "microsoft/bitnet-b1.58-2B-4T"
tok = AutoTokenizer.from_pretrained(repo)
t0 = time.time()
# bf16 is the checkpoint's native dtype. Two fp32 routes were tried and failed on this 19 GB host (see decision log D7):
# (1) dtype=float32 at load: OOM-killed at 7.2 GB anon RSS; (2) load bf16 then model.float(): HF refuses (.float() unsupported on quantized models).
model = AutoModelForCausalLM.from_pretrained(repo, dtype=getattr(torch, a.dtype), low_cpu_mem_usage=True).eval()
print("model loaded %.1fs dtype=%s" % (time.time() - t0, a.dtype), flush=True)

if a.mode == "prompts":
    summary = {}
    for p in json.load(open(a.prompts)):
        ids = tok(p["text"], return_tensors="pt").input_ids          # includes BOS
        with torch.no_grad():
            lg = model(ids).logits[0, -1].float().numpy().astype(np.float32)
            gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.steps, min_new_tokens=a.steps,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        greedy = gen[0, ids.shape[1]:].tolist()
        lg.tofile(os.path.join(a.out, p["id"] + ".logits.f32"))
        open(os.path.join(a.out, p["id"] + ".ids.txt"), "w").write(" ".join(map(str, ids[0].tolist())) + "\n")
        open(os.path.join(a.out, p["id"] + ".greedy.txt"), "w").write(" ".join(map(str, greedy)) + "\n")
        summary[p["id"]] = {"n_prompt_tokens": ids.shape[1], "top1": int(lg.argmax()), "greedy_text": tok.decode(greedy)}
        print(p["id"], ids.shape[1], repr(tok.decode(greedy)[:80]), flush=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
else:
    wins = [list(map(int, l.split())) for l in open(a.windows) if l.strip()]
    outp = os.path.join(a.out, "nll_hf_%s.txt" % a.dtype)
    with open(outp, "w") as f:
        f.write("# hf transformers dtype=%s windows=%d\n" % (a.dtype, len(wins)))
        for w, ids in enumerate(wins):
            t = torch.tensor([ids])
            with torch.no_grad():
                lg = model(t).logits[0, :-1].double()
            lp = torch.log_softmax(lg, -1).gather(1, t[0, 1:, None])[:, 0].tolist()
            f.write("w %d %d %.9f\n" % (w, len(lp), -sum(lp)))
            f.write("lp " + " ".join("%.9f" % x for x in lp) + "\n"); f.flush()
            print("window", w, "ppl %.3f" % np.exp(-np.mean(lp)), flush=True)
