#!/usr/bin/env python3
"""Layer-wise calibrated ternary KV (round 3, T30) on top of the golden fp32 reference.

Scheme being calibrated: S3+s4r32 (tools/reference/kv_ptq.py): K (post-RoPE) and V rotated by the fixed randomized
Hadamard R (seed 0), ternary per group of 32 along D (TWN threshold 0.7*mean|x|, LS scale on the support), 4 sink
tokens and the 32 most recent tokens served from per-token INT8.
Learned (static, so causality is unchanged; model weights untouched): clip multipliers alpha in [0.5, 1.5] for K and V,
per layer, KV head and 32-group:  x_hat = (alpha * s * t) R^T, with the ternary codes t (threshold) fixed per
forward (straight-through: gradients reach alpha only through the scale).
Objective per layer l (GPTQ-like, local): || Attn(Q, K_hat, V_hat) - Attn(Q, K, V) ||^2 on the calibration windows,
output taken before o_proj, causal mask, same sinks/recent INT8 path. Adam, `--steps` per layer, minibatches of
`--mb` windows. Layers are calibrated sequentially: the input of layer l+1 comes from the model in which layers <= l
already use their calibrated quantizer (error accumulation is seen by later layers).

  calibrate: kv_calib.py calibrate --windows a.ids [b.ids] --out alphas.json [--steps 100]
  evaluate : golden_bitnet.py nll ... --kv-scheme calib:<alphas.json>
"""
import argparse, json, math, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_bitnet as G  # noqa: E402
import kv_ptq as K  # noqa: E402

GS = 32


def quant_alpha(x, alpha):
    """x [B,H,T,D] -> S3 ternary reconstruction with per (head, group) multiplier alpha [H, D/GS]."""
    R = K.hadamard().to(x.dtype)
    y = x @ R
    B, H, T, D = y.shape
    yg = y.reshape(B, H, T, D // GS, GS)
    with torch.no_grad():
        m = yg.abs().mean(-1, keepdim=True)
        nz = yg.abs() > 0.7 * m
        cnt = nz.sum(-1, keepdim=True)
        s = torch.where(cnt > 0, (yg.abs() * nz).sum(-1, keepdim=True) / cnt.clamp(min=1), m + 1e-6)
        t = torch.sign(yg) * nz
    a = alpha.clamp(0.5, 1.5)[None, :, None, :, None]
    return ((a * s) * t).reshape(B, H, T, D) @ R.T


def windows_mask(T):
    return K.window_mask(T, 4, 32)


def attn(q, k, v, k_alt=None, v_alt=None, mask=None):
    """q [B,H,T,D], k/v [B,Hkv,T,D]; alt (INT8) keys/values where mask[t,s]. Returns [B,H,T,D] (before o_proj)."""
    B, H, T, D = q.shape
    r = H // k.shape[1]
    kk, vv = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    sc = q @ kk.transpose(-1, -2) * D ** -0.5
    if mask is not None:
        sca = q @ k_alt.repeat_interleave(r, 1).transpose(-1, -2) * D ** -0.5
        sc = torch.where(mask, sca, sc)
    causal = torch.ones(T, T, dtype=torch.bool).tril_()
    p = torch.softmax(sc.masked_fill(~causal, float("-inf")), -1)
    if mask is not None:
        m = mask.to(p.dtype)
        return (p * (1 - m)) @ vv + (p * m) @ v_alt.repeat_interleave(r, 1)
    return p @ vv


def calibrate(a):
    torch.manual_seed(0)
    g = G.Golden(G.SNAP, torch.float32, a.threads)
    wins = []
    for f in a.windows:
        ws = [list(map(int, l.split())) for l in open(f) if l.strip()]
        wins += ws[:a.per_file]
    ids = torch.tensor(wins)
    Bn, T = ids.shape
    mask = windows_mask(T)
    out = {"scheme": "S3+s4r32 calibrated", "windows": a.windows, "per_file": a.per_file, "steps": a.steps, "mb": a.mb,
           "lr": a.lr, "alpha_k": [], "alpha_v": [], "loss": []}
    calibrated = {}

    def hook(k, v, layer):  # used while advancing the hidden state: calibrated layers use their alphas
        if layer not in calibrated:
            return k, v
        ak, av = calibrated[layer]
        with torch.no_grad():
            return {"k": quant_alpha(k, ak), "v": quant_alpha(v, av), "k_alt": K.int8_absmax(k), "v_alt": K.int8_absmax(v),
                    "alt_mask": mask}

    G.kv_quant = hook
    x = g.embed_ids(ids)
    cos, sin = g.rope(T)
    t0 = time.time()
    for l in range(g.L):
        pre = "model.layers.%d." % l
        with torch.no_grad():
            h = g.rmsnorm(x, g.vec(pre + "input_layernorm.weight"))
            q = g.bitlinear(h, g.load_lin(pre + "self_attn.q_proj")).view(Bn, T, g.H, g.D).transpose(1, 2)
            k = g.bitlinear(h, g.load_lin(pre + "self_attn.k_proj")).view(Bn, T, g.Hkv, g.D).transpose(1, 2)
            v = g.bitlinear(h, g.load_lin(pre + "self_attn.v_proj")).view(Bn, T, g.Hkv, g.D).transpose(1, 2)
            c, s = cos[None, None], sin[None, None]
            q = q * c + g.rotate_half(q) * s
            k = k * c + g.rotate_half(k) * s
            ka, va = K.int8_absmax(k), K.int8_absmax(v)
            target = torch.cat([attn(q[i:i + 1], k[i:i + 1], v[i:i + 1]) for i in range(Bn)])
        ak = torch.ones(g.Hkv, g.D // GS, requires_grad=True)
        av = torch.ones(g.Hkv, g.D // GS, requires_grad=True)
        opt = torch.optim.Adam([ak, av], lr=a.lr)

        def loss_on(sel, ak_, av_):
            kh, vh = quant_alpha(k[sel], ak_), quant_alpha(v[sel], av_)
            o = attn(q[sel], kh, vh, ka[sel], va[sel], mask)
            return (o - target[sel]).pow(2).mean() / target[sel].pow(2).mean()

        with torch.no_grad():
            base = float(sum(loss_on(slice(i, i + 1), torch.ones_like(ak), torch.ones_like(av)) for i in range(Bn)) / Bn)
        hist = []
        for st in range(a.steps):
            sel = torch.randperm(Bn)[:a.mb]
            loss = loss_on(sel, ak, av)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                ak.clamp_(0.5, 1.5); av.clamp_(0.5, 1.5)
            if st % 10 == 0 or st == a.steps - 1:
                hist.append([st, loss.item()])
        with torch.no_grad():
            final = float(sum(loss_on(slice(i, i + 1), ak, av) for i in range(Bn)) / Bn)
        calibrated[l] = (ak.detach().clone(), av.detach().clone())
        out["alpha_k"].append(ak.detach().tolist()); out["alpha_v"].append(av.detach().tolist())
        out["loss"].append({"layer": l, "rel_mse_alpha1": base, "rel_mse_calibrated": final, "trace": hist})
        print("layer %2d rel_mse %.5f -> %.5f  alpha_k [%.2f,%.2f] alpha_v [%.2f,%.2f]  %.0fs rss %.2fGB" % (
            l, base, final, float(ak.min()), float(ak.max()), float(av.min()), float(av.max()), time.time() - t0, G.rss_gb()),
            flush=True)
        del q, k, v, ka, va, target
        x = g.layer(x, l, cos, sin, G.Dumper(None, 0, [], []))   # advance with the calibrated quantizer of layer l
        json.dump(out, open(a.out, "w"))
    print("done %.0fs peak %.2fGB" % (time.time() - t0, G.peak_gb()), flush=True)


def install(path, golden_module):
    d = json.load(open(path))
    al = [(torch.tensor(k_), torch.tensor(v_)) for k_, v_ in zip(d["alpha_k"], d["alpha_v"])]

    def hook(k, v, layer):
        T = k.shape[2]
        ak, av = al[layer]
        return {"k": quant_alpha(k, ak), "v": quant_alpha(v, av), "k_alt": K.int8_absmax(k), "v_alt": K.int8_absmax(v),
                "alt_mask": windows_mask(T)}

    golden_module.kv_quant = hook
    print("[kv_calib] installed alphas from", path, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate"])
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--per-file", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    with torch.no_grad():
        pass
    calibrate(a)
