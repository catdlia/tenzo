#!/usr/bin/env python3
"""KV-cache quantization on a second model (Qwen2.5-1.5B, Llama-style GQA, different tokenizer) with HF transformers.

The attention function replaces transformers' module-level `eager_attention_forward` of modeling_qwen2 (the documented "patch eager"
route; Qwen2Attention calls it with post-RoPE K and V): it receives post-RoPE K and V
([B, Hkv, T, 128]) and applies the same fake-quant hooks as the BitNet golden path (tools/reference/kv_ptq.py: INT8 per
token, ternary g32, ternary g32 + Hadamard, S3 + 4 sinks + 32 recent tokens in INT8, ...). Hadamard schemes are simulated
exactly like the golden (k_hat = deq(Q(k R)) R^T). Compute is fp32 from the bf16 checkpoint (exact upcast of every weight
at use), so the model needs ~3.1 GB for weights and no fp32 copy.

Windows (same texts as the BitNet study, Qwen tokenizer, 512 tokens, 511 scored predictions each):
  EN: WikiText-2 raw test, 16 non-overlapping windows at RandomState(0) start offsets (protocol of make_windows.py)
  UK: Kobza first-rows documents joined with a blank line, the first N consecutive windows
Outputs one score file per (scheme, language) in the score-file format of the other tools (w idx n nll / lp ...).

Check: scheme "eager" (HF's own eager attention) vs scheme "fp" (this module's attention without quantization) on the first
--check-windows windows of both languages: mean |dlogp| and PPL must agree (the patch must not change the model).
"""
import argparse, json, math, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2.modeling_qwen2 as MQ

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_ptq  # noqa: E402

STATE = {"hook": None}


def kvq_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """query [B,H,T,D], key/value [B,Hkv,T,D] (post-RoPE). Returns ([B,T,H,D], None) like eager_attention_forward."""
    B, H, T, D = query.shape
    r = H // key.shape[1]
    k, v, alt = key, value, None
    if STATE["hook"] is not None:
        hk = STATE["hook"](key, value, module.layer_idx)
        if isinstance(hk, dict):
            k, v = hk["k"], hk["v"]
            if hk.get("alt_mask") is not None:
                alt = (hk["k_alt"], hk["v_alt"], hk["alt_mask"])
        else:
            k, v = hk
    causal = torch.ones(T, T, dtype=torch.bool, device=query.device).tril_()
    sc_scale = scaling if scaling is not None else D ** -0.5
    outs = []
    for b in range(B):
        kb, vb = k[b].repeat_interleave(r, 0), v[b].repeat_interleave(r, 0)
        sc = torch.matmul(query[b], kb.transpose(-1, -2)) * sc_scale
        if alt is not None:
            sca = torch.matmul(query[b], alt[0][b].repeat_interleave(r, 0).transpose(-1, -2)) * sc_scale
            sc = torch.where(alt[2], sca, sc)
        p = torch.softmax(sc.masked_fill(~causal, float("-inf")), -1, dtype=torch.float32).to(query.dtype)
        if alt is not None:
            m = alt[2].to(p.dtype)
            o = torch.matmul(p * (1 - m), vb) + torch.matmul(p * m, alt[1][b].repeat_interleave(r, 0))
        else:
            o = torch.matmul(p, vb)
        outs.append(o.transpose(0, 1))      # [T, H, D]
    return torch.stack(outs), None


ORIG_EAGER = MQ.eager_attention_forward   # HF reference implementation (scheme "eager")


class UpLinear(nn.Module):
    """bf16 weights, fp32 compute (exact upcast per call)."""
    def __init__(self, lin):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias

    def forward(self, x):
        return F.linear(x, self.weight.float(), None if self.bias is None else self.bias.float())


def to_fp32_compute(model):
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname != "lm_head":
                setattr(mod, cname, UpLinear(child))
    model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.float())
    return model


def windows_en(tok, raw_path, n=16, length=512, seed=0):
    ids = tok(open(raw_path, encoding="utf-8").read(), add_special_tokens=False).input_ids
    rs = np.random.RandomState(seed)
    starts = []
    while len(starts) < n:
        s = int(rs.randint(0, len(ids) - length))
        if all(abs(s - t) >= length for t in starts):
            starts.append(s)
    starts.sort()
    return [ids[s:s + length] for s in starts], {"n_text_tokens": len(ids), "starts": starts}


def windows_uk(tok, kobza_json, n=16, length=512):
    rows = json.load(open(kobza_json))["rows"]
    ids = tok("\n\n".join(r["row"]["text"] for r in rows), add_special_tokens=False).input_ids
    assert len(ids) >= n * length, (len(ids), n * length)
    return [ids[w * length:(w + 1) * length] for w in range(n)], {"n_text_tokens": len(ids)}


@torch.no_grad()
def window_logprobs(model, ids, chunk=16384):
    x = torch.tensor([ids])
    h = model.model(input_ids=x).last_hidden_state[0, :-1]          # [T-1, C] fp32
    tgt = torch.tensor(ids[1:])
    W = model.lm_head.weight
    lse = None
    tl = torch.empty(h.shape[0])
    for v0 in range(0, W.shape[0], chunk):
        lg = F.linear(h, W[v0:v0 + chunk].float()).double()
        part = torch.logsumexp(lg, -1)
        lse = part if lse is None else torch.logaddexp(lse, part)
        inside = (tgt >= v0) & (tgt < v0 + chunk)
        if inside.any():
            idx = (tgt[inside] - v0)
            tl[inside] = lg[inside.nonzero()[:, 0], idx].float()
    return (tl.double() - lse).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/qwen2.5-1.5b")
    ap.add_argument("--wt2-raw", default="/tmp/wt2test/wt2_test.raw")
    ap.add_argument("--kobza", default="results/reference/kobza_first_rows.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schemes", default="fp,tenzo_int8,S2,S3,S3+s4r32")
    ap.add_argument("--en-windows", type=int, default=16)
    ap.add_argument("--uk-windows", type=int, default=16)
    ap.add_argument("--check-windows", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    en, meta_en = windows_en(tok, a.wt2_raw, a.en_windows)
    uk, meta_uk = windows_uk(tok, a.kobza, a.uk_windows)
    json.dump({"model": a.model, "en": meta_en, "uk": meta_uk, "window_tokens": 512,
               "tokenizer": "Qwen2.5 (no BOS added)"}, open(os.path.join(a.out, "windows.json"), "w"))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 attn_implementation="eager").eval()
    to_fp32_compute(model)
    print("loaded %.0fs" % (time.time() - t0), flush=True)

    def run(scheme, lang, wins, nmax=None):
        path = os.path.join(a.out, "%s_%s.txt" % (scheme.replace(":", "_").replace("+", "_"), lang))
        done = {}
        if os.path.exists(path):
            cur = None
            for line in open(path):
                if line.startswith("w "): cur = line
                elif line.startswith("lp ") and cur and line.endswith("\n"): done[int(cur.split()[1])] = (cur, line); cur = None
        MQ.eager_attention_forward = ORIG_EAGER if scheme == "eager" else kvq_attention
        STATE["hook"] = None if scheme in ("fp", "eager") else kv_ptq.make_hook(scheme)[0]
        sel = list(range(len(wins)))[:nmax]
        with open(path, "w") as f:
            f.write("# %s %s windows=%d\n" % (scheme, lang, len(sel)))
            for w in sorted(done):
                f.write(done[w][0]); f.write(done[w][1])
            for w in sel:
                if w in done: continue
                lp = window_logprobs(model, wins[w])
                f.write("w %d %d %.9f\n" % (w, len(lp), -sum(lp)))
                f.write("lp " + " ".join("%.9f" % x for x in lp) + "\n"); f.flush()
                print("%s %s window %d ppl %.3f  %.0fs" % (scheme, lang, w, math.exp(-sum(lp) / len(lp)), time.time() - t0), flush=True)
        return path

    # patch-equivalence reference first: HF's own eager attention on the first windows (compared with scheme "fp" afterwards)
    for lang, wins in (("en", en), ("uk", uk)):
        run("eager", lang, wins, a.check_windows)
    for scheme in a.schemes.split(","):
        for lang, wins in (("en", en), ("uk", uk)):
            run(scheme, lang, wins)
    print("done %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
