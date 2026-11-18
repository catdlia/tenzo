#!/usr/bin/env python3
"""KV-cache fake-quant schemes for the golden reference (tools/reference/golden_bitnet.py --kv-scheme NAME).

All schemes act on post-RoPE K and on V, shape [B, Hkv, T, D] (the engine also quantizes K after RoPE:
FusedKVCache::forward_attention_raw applies RoPE in place, then appends to the cache). Quantization of token t only
uses token t itself (per-token schemes) or tokens <= t: per-channel K over token groups (KIVI-style) quantizes a group
only once it is complete; until then the keys of that group are served from INT8 via the golden's alt-mask path.

Engine replicas (read from src/runtime/TenzoEngine.cpp, FusedKVCache):
  tenzo_int8 : per token/head absmax/127, std::round (half away), clamp; q stays fp32.
  tenzo_tl1  : per token/head th = 0.5*mean|x|, ternary sign(x)*[|x|>th], scale = mean |x| over |x|>th
               (mean+1e-6 if none), for K and V; q quantized per head to int (absmax/127, half away).
Study schemes (S1..S9) are documented in SCHEMES below; `bits(name, ctx)` gives effective bits per K/V element
including scales (fp16) and windows, with ternary stored as 2-bit codes.
"""
import math
import torch

D = 128
SCALE_BITS = 16  # fp16 scale (and fp16 zero-point for asymmetric schemes)


def round_half_away(x):
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


# ---------------------------------------------------------------- primitives (last dim = quantization group)
def int8_absmax(x):
    a = x.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    s = a / 127.0
    return round_half_away(x / s).clamp(-128, 127) * s


def ternary(x, delta_mult):
    """TWN-style ternary over the last dim: threshold delta_mult*mean|x|, LS-optimal scale for that support."""
    m = x.abs().mean(-1, keepdim=True)
    th = delta_mult * m
    nz = x.abs() > th
    cnt = nz.sum(-1, keepdim=True)
    s = torch.where(cnt > 0, (x.abs() * nz).sum(-1, keepdim=True) / cnt.clamp(min=1), m + 1e-6)
    return torch.sign(x) * nz * s


def uniform_asym(x, bits):
    lo = x.amin(-1, keepdim=True); hi = x.amax(-1, keepdim=True)
    n = 2 ** bits - 1
    s = ((hi - lo) / n).clamp(min=1e-8)
    return torch.round((x - lo) / s).clamp(0, n) * s + lo


def grouped(fn, x, g):
    sh = x.shape
    return fn(x.reshape(*sh[:-1], sh[-1] // g, g)).reshape(sh)


_H = None


def hadamard(seed=0):
    """Orthonormal randomized Hadamard R = H_D * diag(signs) / sqrt(D), fixed seed."""
    global _H
    if _H is None:
        h = torch.ones(1, 1)
        while h.shape[0] < D:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        gen = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (D,), generator=gen).float() * 2 - 1
        _H = (h * signs[None, :]) / math.sqrt(D)
    return _H


def rotated(fn, x):
    R = hadamard().to(x.dtype)
    return fn(x @ R) @ R.T


def per_channel_token_groups(fn, k, g=32):
    """Quantize K per channel over complete groups of g tokens: k [B,H,T,D] -> groups along T."""
    B, H, T, Dd = k.shape
    full = (T // g) * g
    out = k.clone()
    if full:
        kk = k[:, :, :full].reshape(B, H, full // g, g, Dd).transpose(-1, -2)  # [..., D, g]
        out[:, :, :full] = fn(kk).transpose(-1, -2).reshape(B, H, full, Dd)
    return out


def incomplete_group_mask(T, g=32):
    """alt_mask[t, s] True where key s's group is not yet complete at query t (served from INT8)."""
    s = torch.arange(T)
    gend = (s // g) * g + (g - 1)
    return gend[None, :] > torch.arange(T)[:, None]


def window_mask(T, sinks, recent):
    t = torch.arange(T)[:, None]; s = torch.arange(T)[None, :]
    m = torch.zeros(T, T, dtype=torch.bool)
    if sinks:
        m |= s < sinks
    if recent:
        m |= (t - s) < recent
    return m


# ---------------------------------------------------------------- schemes
def tern_tok(x): return ternary(x, 0.7)                                   # S1
def tern_g32(x): return grouped(lambda y: ternary(y, 0.7), x, 32)         # S2
def tern_g32_rot(x): return rotated(tern_g32, x)                         # S3
def tl1_engine(x): return ternary(x, 0.5)                                 # engine tl1_fused


def int2_rot(x): return rotated(lambda y: grouped(lambda z: uniform_asym(z, 2), y, 32), x)   # S8
def int4_g32(x): return grouped(lambda z: uniform_asym(z, 4), x, 32)                          # S9


SCHEMES = {
    # name: (k_fn, v_fn, k_per_channel_groups, sinks, recent, description)
    "fp32":        (None, None, False, 0, 0, "no KV quantization"),
    "tenzo_int8":  (int8_absmax, int8_absmax, False, 0, 0, "replica of engine int8_fused"),
    "tenzo_tl1":   (tl1_engine, tl1_engine, False, 0, 0, "replica of engine tl1_fused (+ q int quant)"),
    "S1":          (tern_tok, tern_tok, False, 0, 0, "ternary per token/head, TWN 0.7, LS scale"),
    "S2":          (tern_g32, tern_g32, False, 0, 0, "ternary g=32 along D"),
    "S3":          (tern_g32_rot, tern_g32_rot, False, 0, 0, "S2 + randomized Hadamard (D=128)"),
    "S4":          ("kivi", tern_g32, True, 0, 0, "K per-channel ternary over 32-token groups, V per-token g=32"),
    "S8":          (int2_rot, int2_rot, False, 0, 0, "2-bit asym g=32 + Hadamard (comparator)"),
    "S9":          (int4_g32, int4_g32, False, 0, 0, "INT4 asym g=32 (comparator)"),
}


def resolve(name):
    """NAME | NAME+s<S>r<R> (sinks/recent INT8 window) | K8V:<base> (K INT8, V from base) | KV8:<base> (K from base, V INT8)."""
    if name in SCHEMES:
        return SCHEMES[name]
    if "+" in name:
        base, w = name.split("+", 1)
        sinks = int(w[1:w.index("r")]); recent = int(w[w.index("r") + 1:])
        kf, vf, pc, _, _, desc = resolve(base)
        return (kf, vf, pc, sinks, recent, desc + " + sinks %d + recent %d INT8" % (sinks, recent))
    if name.startswith("K8V:"):
        _, vf, _, s_, r_, desc = resolve(name[4:])
        return (int8_absmax, vf, False, s_, r_, "K INT8, V from " + desc)
    if name.startswith("KV8:"):
        kf, _, pc, s_, r_, desc = resolve(name[4:])
        return (kf, int8_absmax, pc, s_, r_, "V INT8, K from " + desc)
    raise KeyError(name)


def kivi_k(k):
    return per_channel_token_groups(lambda y: ternary(y, 0.7), k, 32)


def make_variant(base, sinks=0, recent=0, k_override=None, v_override=None, name=None):
    kf, vf, pc, _, _, desc = SCHEMES[base]
    if k_override is not None:
        kf, pc = k_override
    if v_override is not None:
        vf = v_override
    return (kf, vf, pc, sinks, recent, desc + (" + sinks %d + recent %d INT8" % (sinks, recent) if sinks or recent else ""))


def register(name, spec):
    SCHEMES[name] = spec


def _apply(fn, x):
    if fn is None:
        return x
    if fn == "kivi":
        return kivi_k(x)
    return fn(x)


def install(name, golden_module):
    """Install hooks into the golden module. name may be NAME or NAME+s4r32 style handled by callers via register()."""
    kf, vf, pc, sinks, recent, desc = resolve(name)

    def hook(k, v, layer):
        T = k.shape[2]
        kh = _apply(kf, k)
        vh = _apply(vf, v)
        mask = None
        if pc or kf == "kivi":
            mask = incomplete_group_mask(T, 32)
        if sinks or recent:
            wm = window_mask(T, sinks, recent)
            mask = wm if mask is None else (mask | wm)
        if mask is None:
            return kh, vh
        return {"k": kh, "v": vh, "k_alt": int8_absmax(k), "v_alt": int8_absmax(v), "alt_mask": mask}

    golden_module.kv_quant = hook
    if name == "tenzo_tl1":
        def qq(q, layer):
            a = q.abs().amax(-1, keepdim=True)
            s = torch.where(a > 0, a / 127.0, torch.ones_like(a))
            return round_half_away(q / s) * s
        golden_module.q_quant = qq
    print("[kv_ptq] installed %s: %s" % (name, desc), flush=True)


# ---------------------------------------------------------------- effective bits per element (K and V averaged)
def _bits_fn(fn, pc):
    if fn is None:
        return 32.0
    if fn is int2_rot:
        return 2 + 2 * SCALE_BITS / 32   # scale + zero point per 32
    if fn is int4_g32:
        return 4 + 2 * SCALE_BITS / 32
    if fn is int8_absmax:
        return 8 + SCALE_BITS / D
    if fn in (tern_tok, tl1_engine):
        return 2 + SCALE_BITS / D
    if fn in (tern_g32, tern_g32_rot):
        return 2 + SCALE_BITS / 32
    if fn == "kivi":
        return 2 + SCALE_BITS / 32
    return None


def bits(name, ctx=None, base_bits=None):
    """Average bits per K/V element at context ctx (None = asymptotic). base_bits overrides per-scheme lookup."""
    kf, vf, pc, sinks, recent, _ = resolve(name)
    if base_bits is None:
        kb, vb = _bits_fn(kf, pc), _bits_fn(vf, False)
    else:
        kb, vb = base_bits
    i8 = 8 + SCALE_BITS / D
    if ctx is None:
        return (kb + vb) / 2
    hi = min(ctx, sinks + recent)
    kpart = (hi * i8 + (ctx - hi) * kb) / ctx
    if pc or kf == "kivi":  # incomplete group (avg 15.5 tokens) in INT8
        extra = min(ctx - hi, 15.5)
        kpart += extra * (i8 - kb) / ctx
    vpart = (hi * i8 + (ctx - hi) * vb) / ctx
    return (kpart + vpart) / 2
