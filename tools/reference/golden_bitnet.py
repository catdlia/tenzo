#!/usr/bin/env python3
"""Layer-major "golden" reference forward for microsoft/bitnet-b1.58-2B-4T (BitNetForCausalLM).

Written strictly from the HF code (transformers/models/bitnet/modeling_bitnet.py, transformers/integrations/bitnet.py,
transformers 4.57.6) and config.json of the offline-cached checkpoint.  The Tenzo engine sources were NOT consulted.

Semantics reproduced (HF line refs are to transformers 4.57.6):
  * embed_tokens (bf16 checkpoint values)                                  modeling_bitnet.py BitNetModel.forward
  * BitNetRMSNorm: x.float(); x*rsqrt(mean(x^2)+eps); weight * x.to(in_dtype)      BitNetRMSNorm.forward
  * AutoBitLinear (offline): ActQuant (in fp32, result cast back) = per-token absmax, scale=127/max.clamp(1e-5),
    round (torch.round = half-to-even), clamp(-128,127), divide by scale (fake-quant);
    F.linear with ternary weight (unpack_weights), then output * weight_scale      integrations/bitnet.py AutoBitLinear
  * RoPE: inv_freq = 1/theta^(2i/D), emb = cat(freqs, freqs), rotate_half pairing (i, i+D/2),
    cos/sin computed in fp32 then cast to the activation dtype                     BitNetRotaryEmbedding / apply_rotary_pos_emb
  * GQA: repeat_kv -> q head h uses kv head h // n_rep                             repeat_kv
  * scores * head_dim^-0.5, causal mask, softmax in fp32 then cast to dtype        eager_attention_forward
  * attn_sub_norm before o_proj                                                    BitNetAttention.forward
  * MLP: down(ffn_sub_norm(relu(gate)^2 * up))                                     BitNetMLP.forward
  * final norm, tied lm_head = embed_tokens (unquantized)                         BitNetForCausalLM

Execution is layer-major: all sequences go through layer L before layer L+1; layer weights are read with safe_open
and dropped after the layer; the lm head is evaluated per sequence in chunks and reduced immediately to NLL.

Modes:
  nll      teacher-forced per-token log-probs (same text format as `tenzo-inference --score-out` / hf_ref_dump.py)
  prompts  first-step logits (.f32, V) for prompts given as HF id files
Dumps (--dump-dir): named intermediates for --dump-seq, --dump-layers, --dump-pos  -> <name>.f32 + manifest.json.
KV hook: `kv_quant(k_rope, v, layer)` (module-level, default identity) may return (k_hat, v_hat) or a dict
  {k, v, k_alt, v_alt, alt_mask}: keys/values at (query t, key s) come from *_alt where alt_mask[t, s] is True.
"""
import argparse, json, math, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/"
                          "04c3b9ad9361b824064a1f25ea60a8be9599b127")

try:  # HF unpacking (allowed to import); unwrap torch.compile to avoid a compile step
    from transformers.integrations.bitnet import unpack_weights as _hf_unpack
    _hf_unpack = getattr(_hf_unpack, "_torchdynamo_orig_callable", getattr(_hf_unpack, "__wrapped__", _hf_unpack))
except Exception as e:  # pragma: no cover
    print("cannot import HF unpack_weights:", e, file=sys.stderr); raise


def kv_quant_identity(k, v, layer):
    return k, v


kv_quant = kv_quant_identity  # replaced by tools/reference/kv_ptq.py
q_quant = None                # optional q hook (post-RoPE q [B,H,T,D]) used only to replicate engine KV modes

# DIAGNOSTIC emulation switches (default None = HF semantics). Used only to attribute the Tenzo gap; never a reference.
EMULATE_ACT_QP = None
EMULATE_ROPE_THETA = None
VERBOSE = os.environ.get("GOLDEN_QUIET") is None


class Dumper:
    def __init__(self, out, seq, layers, pos):
        self.out, self.seq, self.layers, self.pos = out, seq, set(layers), list(pos)
        self.manifest = []
        if out:
            os.makedirs(out, exist_ok=True)

    def on(self, layer):
        return self.out is not None and (layer is None or layer in self.layers)

    def __call__(self, name, layer, t, seq_axis_first=True):
        """t: [B, T, ...] activation (batch-major). Dumps t[seq, pos]."""
        if not self.on(layer):
            return
        x = t[self.seq]
        for p in self.pos:
            if p >= x.shape[0]:
                continue
            a = x[p].detach().to(torch.float32).contiguous().numpy()
            fn = "%s%s_p%d.f32" % ("" if layer is None else "L%d." % layer, name, p)
            a.tofile(os.path.join(self.out, fn))
            self.manifest.append({"name": name, "layer": layer, "pos": p, "shape": list(a.shape), "file": fn})

    def close(self):
        if self.out:
            json.dump({"source": "golden", "tensors": self.manifest}, open(os.path.join(self.out, "manifest.json"), "w"),
                      indent=0)


class Golden:
    def __init__(self, snap, dtype, threads):
        torch.set_num_threads(threads)
        self.cfg = json.load(open(os.path.join(snap, "config.json")))
        self.f = safe_open(os.path.join(snap, "model.safetensors"), "pt")
        self.dt = dtype
        c = self.cfg
        self.H, self.Hkv, self.L = c["num_attention_heads"], c["num_key_value_heads"], c["num_hidden_layers"]
        self.D = c["hidden_size"] // self.H
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_theta"] if EMULATE_ROPE_THETA is None else EMULATE_ROPE_THETA
        assert c["hidden_act"] == "relu2" and c["tie_word_embeddings"]
        assert c["quantization_config"]["linear_class"] == "autobitlinear"
        assert c["quantization_config"]["quantization_mode"] == "offline"
        self.embed = self.f.get_tensor("model.embed_tokens.weight")  # bf16, kept in bf16 (656 MB)
        self.wcache = None

    # --- primitives -------------------------------------------------------------------------------------------
    def rmsnorm(self, x, w):
        in_dt = x.dtype
        h = x.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.eps)
        return w * h.to(in_dt)

    def act_quant(self, x):
        # integrations/bitnet.py ActQuant.forward: computed in fp32, dequantized value cast back to the input dtype
        dtype = x.dtype
        x = x.float()
        if EMULATE_ACT_QP is not None:  # DIAGNOSTIC ONLY: Tenzo build_lut (qp levels, half-away round, 1e-6 floor)
            mx = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-6)
            y = x * (EMULATE_ACT_QP / mx)
            return (torch.sign(y) * torch.floor(y.abs() + 0.5) * (mx / EMULATE_ACT_QP)).to(dtype)
        scale = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return ((x * scale).round().clamp(-128, 127) / scale).to(dtype)

    def bitlinear(self, x, wq):
        w, s = wq
        return F.linear(self.act_quant(x), w) * s

    def load_lin(self, name):
        # optional cache of the unpacked ternary weights as int8 (exact; ~2.4 GB) for repeated full forwards (greedy)
        if self.wcache is not None and name in self.wcache:
            w8, s = self.wcache[name]
            return (w8.to(self.dt), s)
        p = self.f.get_tensor(name + ".weight")
        w = _hf_unpack(p, dtype=self.dt)
        s = self.f.get_tensor(name + ".weight_scale").to(self.dt)
        if self.wcache is not None:
            self.wcache[name] = (w.to(torch.int8), s)
        return (w, s)

    def vec(self, name):
        return self.f.get_tensor(name).to(self.dt)

    def rope(self, T):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.D, 2, dtype=torch.int64).float() / self.D))
        pos = torch.arange(T, dtype=torch.float32)
        freqs = torch.outer(pos, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dt), emb.sin().to(self.dt)

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    # --- forward ----------------------------------------------------------------------------------------------
    def embed_ids(self, ids):  # ids [B, T] -> [B, T, C] in dt, converted in chunks
        out = torch.empty(ids.shape[0], ids.shape[1], self.embed.shape[1], dtype=self.dt)
        for b in range(ids.shape[0]):
            out[b] = self.embed[ids[b]].to(self.dt)
        return out

    def attention(self, q, k, v, layer):
        """q [B,H,T,D], k/v [B,Hkv,T,D] (post-RoPE k). Returns [B,T,H*D]."""
        B, H, T, D = q.shape
        r = self.H // self.Hkv
        if q_quant is not None:
            q = q_quant(q, layer)
        hk = kv_quant(k, v, layer)
        alt = None
        if isinstance(hk, dict):
            k, v = hk["k"], hk["v"]
            if hk.get("alt_mask") is not None:
                alt = (hk["k_alt"], hk["v_alt"], hk["alt_mask"])
        else:
            k, v = hk
        causal = torch.ones(T, T, dtype=torch.bool).tril_()
        outs = []
        for b in range(B):  # per sequence keeps the [H,T,T] score tensor small
            kb = k[b].repeat_interleave(r, dim=0)  # == repeat_kv: head h -> kv head h // r
            vb = v[b].repeat_interleave(r, dim=0)
            sc = torch.matmul(q[b], kb.transpose(-1, -2)) * (D ** -0.5)
            if alt is not None:
                ka = alt[0][b].repeat_interleave(r, dim=0)
                sca = torch.matmul(q[b], ka.transpose(-1, -2)) * (D ** -0.5)
                sc = torch.where(alt[2], sca, sc)
            sc = sc.masked_fill(~causal, torch.finfo(sc.dtype).min)
            p = torch.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
            if alt is not None:
                va = alt[1][b].repeat_interleave(r, dim=0)
                m = alt[2].to(p.dtype)
                o = torch.matmul(p * (1 - m), vb) + torch.matmul(p * m, va)
            else:
                o = torch.matmul(p, vb)
            outs.append(o.transpose(0, 1).reshape(T, H * D))
        return torch.stack(outs)

    def layer(self, x, l, cos, sin, dump):
        pre = "model.layers.%d." % l
        B, T, C = x.shape
        with torch.no_grad():
            h = self.rmsnorm(x, self.vec(pre + "input_layernorm.weight")); dump("input_norm", l, h)
            q = self.bitlinear(h, self.load_lin(pre + "self_attn.q_proj")); dump("q", l, q)
            k = self.bitlinear(h, self.load_lin(pre + "self_attn.k_proj")); dump("k", l, k)
            v = self.bitlinear(h, self.load_lin(pre + "self_attn.v_proj")); dump("v", l, v)
            q = q.view(B, T, self.H, self.D).transpose(1, 2)
            k = k.view(B, T, self.Hkv, self.D).transpose(1, 2)
            v = v.view(B, T, self.Hkv, self.D).transpose(1, 2)
            c, s = cos[None, None], sin[None, None]
            q = q * c + self.rotate_half(q) * s
            k = k * c + self.rotate_half(k) * s
            dump("q_rope", l, q.transpose(1, 2)); dump("k_rope", l, k.transpose(1, 2))
            a = self.attention(q, k, v, l); dump("attn_out", l, a)
            a = self.rmsnorm(a, self.vec(pre + "self_attn.attn_sub_norm.weight")); dump("attn_sub_norm", l, a)
            o = self.bitlinear(a, self.load_lin(pre + "self_attn.o_proj")); dump("o_proj", l, o)
            x = x + o; dump("resid_attn", l, x)
            h = self.rmsnorm(x, self.vec(pre + "post_attention_layernorm.weight")); dump("post_attn_norm", l, h)
            g = self.bitlinear(h, self.load_lin(pre + "mlp.gate_proj")); dump("gate", l, g)
            u = self.bitlinear(h, self.load_lin(pre + "mlp.up_proj")); dump("up", l, u)
            m = torch.square(F.relu(g)) * u; dump("act", l, m)
            m = self.rmsnorm(m, self.vec(pre + "mlp.ffn_sub_norm.weight")); dump("ffn_sub_norm", l, m)
            d = self.bitlinear(m, self.load_lin(pre + "mlp.down_proj")); dump("down", l, d)
            x = x + d; dump("resid_ffn", l, x)
        return x

    def hidden(self, ids, dump):
        B, T = ids.shape
        x = self.embed_ids(ids); dump("embed", None, x)
        cos, sin = self.rope(T)
        for l in range(self.L):
            t0 = time.time()
            x = self.layer(x, l, cos, sin, dump)
            if VERBOSE:
                print("  layer %2d %.1fs rss=%.2fGB" % (l, time.time() - t0, rss_gb()), flush=True)
        x = self.rmsnorm(x, self.vec("model.norm.weight")); dump("final_norm", None, x)
        return x

    def lm_logits_rows(self, h):  # h [n, C] -> logits [n, V] float32 (tied head, computed in self.dt)
        V = self.embed.shape[0]
        out = torch.empty(h.shape[0], V, dtype=torch.float32)
        step = 16384
        for v0 in range(0, V, step):
            e = self.embed[v0:v0 + step].to(self.dt)
            out[:, v0:v0 + step] = F.linear(h, e).float()
        return out


def rss_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1e6
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["nll", "prompts"])
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--windows", default="results/reference/windows.ids")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--first-window", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8, help="sequences per layer-major pass")
    ap.add_argument("--prompt-ids", nargs="*", default=[], help="files with space-separated HF ids (prompts mode)")
    ap.add_argument("--greedy-steps", type=int, default=0, help="prompts mode: also write <name>.greedy.txt")
    ap.add_argument("--out", required=True, help="nll: output score file; prompts: output dir")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--dump-seq", type=int, default=0)
    ap.add_argument("--dump-layers", default="0,1,15,29")
    ap.add_argument("--dump-pos", default="0,1,2,5,50,200,511")
    ap.add_argument("--kv-scheme", default=None, help="kv_ptq.py scheme name (installs the KV hook)")
    ap.add_argument("--emulate-act-qp", type=float, default=None, help="DIAGNOSTIC: Tenzo-style activation quant levels")
    ap.add_argument("--emulate-rope-theta", type=float, default=None, help="DIAGNOSTIC: override rope_theta")
    a = ap.parse_args()
    global EMULATE_ACT_QP, EMULATE_ROPE_THETA
    EMULATE_ACT_QP, EMULATE_ROPE_THETA = a.emulate_act_qp, a.emulate_rope_theta
    if a.kv_scheme:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import kv_ptq
        kv_ptq.install(a.kv_scheme, sys.modules[__name__])
    dt = getattr(torch, a.dtype)
    g = Golden(SNAP, dt, a.threads)
    dump = Dumper(a.dump_dir, a.dump_seq, [int(x) for x in a.dump_layers.split(",") if x],
                  [int(x) for x in a.dump_pos.split(",") if x])
    t0 = time.time()
    if a.mode == "prompts":
        os.makedirs(a.out, exist_ok=True)
        if a.greedy_steps:
            g.wcache = {}
        for i, pf in enumerate(a.prompt_ids):
            ids = torch.tensor([[int(x) for x in open(pf).read().split()]])
            d = dump if i == a.dump_seq else Dumper(None, 0, [], [])
            h = g.hidden(ids, d)
            lg = g.lm_logits_rows(h[0])
            if i == a.dump_seq:
                dump("logits", None, lg[None])
            name = os.path.basename(pf).replace(".ids.txt", "")
            lg[-1].numpy().astype(np.float32).tofile(os.path.join(a.out, name + ".logits.f32"))
            print(name, ids.shape[1], "top1", int(lg[-1].argmax()), "%.0fs" % (time.time() - t0), flush=True)
            if a.greedy_steps:
                # greedy continuation, full recompute per step (no KV cache: the reference stays trivially correct)
                open(os.path.join(a.out, name + ".ids.txt"), "w").write(" ".join(map(str, ids[0].tolist())) + "\n")
                seq = ids[0].tolist(); greedy = [int(lg[-1].argmax())]
                quiet = Dumper(None, 0, [], [])
                while len(greedy) < a.greedy_steps:
                    h = g.hidden(torch.tensor([seq + greedy]), quiet)
                    greedy.append(int(g.lm_logits_rows(h[0, -1:])[0].argmax()))
                open(os.path.join(a.out, name + ".greedy.txt"), "w").write(" ".join(map(str, greedy)) + "\n")
                print(name, "greedy", greedy[:8], "%.0fs" % (time.time() - t0), flush=True)
    else:
        wins = [list(map(int, l.split())) for l in open(a.windows) if l.strip()]
        sel = list(range(a.first_window, len(wins)))
        if a.max_windows:
            sel = sel[:a.max_windows]
        with open(a.out, "w") as f:
            f.write("# golden dtype=%s kv=%s windows=%d emulate_act_qp=%s emulate_rope_theta=%s\n" % (
                a.dtype, a.kv_scheme or "none", len(sel), a.emulate_act_qp, a.emulate_rope_theta))
            for b0 in range(0, len(sel), a.batch):
                idx = sel[b0:b0 + a.batch]
                ids = torch.tensor([wins[w] for w in idx])
                d = dump if b0 == 0 else Dumper(None, 0, [], [])
                h = g.hidden(ids, d)
                for j, w in enumerate(idx):
                    lg = g.lm_logits_rows(h[j, :-1])
                    if b0 == 0 and j == a.dump_seq:
                        dump("logits", None, lg[None])
                    lp = torch.log_softmax(lg.double(), -1).gather(1, ids[j, 1:, None])[:, 0].tolist()
                    del lg
                    f.write("w %d %d %.9f\n" % (w, len(lp), -sum(lp)))
                    f.write("lp " + " ".join("%.9f" % x for x in lp) + "\n"); f.flush()
                    print("window", w, "ppl %.3f" % math.exp(-sum(lp) / len(lp)), "%.0fs" % (time.time() - t0),
                          flush=True)
    dump.close()
    print("done %.0fs peak rss %.2fGB" % (time.time() - t0, peak_gb()), flush=True)


def peak_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1e6
    return 0.0


if __name__ == "__main__":
    main()
