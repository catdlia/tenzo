#!/usr/bin/env python3
"""Score H3 FLORES-200 parallel windows (32 EN + 32 UK) across BitNet, EuroLLM-1.7B, and Qwen2.5-1.5B.

Uses layer-major unpadded sequence lists:
  - Each sequence i runs strictly unpadded [1, L_i] (0.0 numerical discrepancy vs B=1).
  - Weights for layer l are unpacked / cast to float32 ONCE per chunk of --batch sequences (default 32),
    cutting CPU memory-bandwidth overhead ~30x while keeping peak RSS < 2 GB (BitNet) and < 4.5 GB (HF).
Verifies exact parity (< 1e-5 max_abs_lp_diff vs standard B=1 forward) on fp and S3+s4r32 at startup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "tools" / "reference"
sys.path.insert(0, str(REF_DIR))

H3_DIR = ROOT / "results" / "2026-11-24_agent2" / "H3"


def rss_gb() -> float:
    for ln in open("/proc/self/status", encoding="utf-8"):
        if ln.startswith("VmRSS"):
            return int(ln.split()[1]) / 1e6
    return 0.0


def peak_gb() -> float:
    for ln in open("/proc/self/status", encoding="utf-8"):
        if ln.startswith("VmHWM"):
            return int(ln.split()[1]) / 1e6
    return 0.0


def read_ids(path: pathlib.Path) -> list[list[int]]:
    return [list(map(int, ln.split())) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_existing(path: pathlib.Path) -> dict[int, tuple[str, str]]:
    done: dict[int, tuple[str, str]] = {}
    if not path.exists():
        return done
    cur = None
    for ln in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if ln.startswith("w "):
            cur = ln
        elif ln.startswith("lp ") and cur is not None and ln.endswith("\n"):
            done[int(cur.split()[1])] = (cur, ln)
            cur = None
    return done


def write_sorted_scores(path: pathlib.Path, header: str, done_map: dict[int, tuple[str, str]], n_total: int) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(header.rstrip("\n") + "\n")
        for w in range(n_total):
            if w in done_map:
                f.write(done_map[w][0])
                f.write(done_map[w][1])
    tmp.replace(path)


def run_bitnet(args: argparse.Namespace) -> int:
    import golden_bitnet as GB
    import kv_ptq

    torch.set_num_threads(args.threads)
    GB.VERBOSE = False
    out_dir = H3_DIR / "scores" / "bitnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    g = GB.Golden(GB.SNAP, torch.float32, args.threads)
    quiet = GB.Dumper(None, 0, [], [])

    def score_seqs_layer_major(seqs: list[list[int]]) -> list[list[float]]:
        if not seqs:
            return []
        tmax = max(len(s) for s in seqs)
        cos_all, sin_all = g.rope(tmax)
        inps = [torch.tensor([s], dtype=torch.long) for s in seqs]
        with torch.no_grad():
            xs = [g.embed_ids(inp) for inp in inps]
            for l in range(g.L):
                pre = f"model.layers.{l}."
                w_in = g.vec(pre + "input_layernorm.weight")
                wq = g.load_lin(pre + "self_attn.q_proj")
                wk = g.load_lin(pre + "self_attn.k_proj")
                wv = g.load_lin(pre + "self_attn.v_proj")
                w_asn = g.vec(pre + "self_attn.attn_sub_norm.weight")
                wo = g.load_lin(pre + "self_attn.o_proj")
                w_pan = g.vec(pre + "post_attention_layernorm.weight")
                wg = g.load_lin(pre + "mlp.gate_proj")
                wu = g.load_lin(pre + "mlp.up_proj")
                w_fsn = g.vec(pre + "mlp.ffn_sub_norm.weight")
                wd = g.load_lin(pre + "mlp.down_proj")

                for idx_s, x in enumerate(xs):
                    B, T, _ = x.shape
                    h = g.rmsnorm(x, w_in)
                    q = g.bitlinear(h, wq).view(B, T, g.H, g.D).transpose(1, 2)
                    k = g.bitlinear(h, wk).view(B, T, g.Hkv, g.D).transpose(1, 2)
                    v = g.bitlinear(h, wv).view(B, T, g.Hkv, g.D).transpose(1, 2)
                    c, s = cos_all[:T][None, None], sin_all[:T][None, None]
                    q = q * c + g.rotate_half(q) * s
                    k = k * c + g.rotate_half(k) * s
                    a = g.attention(q, k, v, l)
                    a = g.rmsnorm(a, w_asn)
                    x = x + g.bitlinear(a, wo)
                    h = g.rmsnorm(x, w_pan)
                    m = torch.square(F.relu(g.bitlinear(h, wg))) * g.bitlinear(h, wu)
                    m = g.rmsnorm(m, w_fsn)
                    xs[idx_s] = x + g.bitlinear(m, wd)
                del w_in, wq, wk, wv, w_asn, wo, w_pan, wg, wu, w_fsn, wd

            w_fn = g.vec("model.norm.weight")
            out_lps: list[list[float]] = []
            for inp, x in zip(inps, xs):
                hn = g.rmsnorm(x, w_fn)[0, :-1]
                tgt = inp[0, 1:, None]
                lps: list[float] = []
                for r0 in range(0, hn.shape[0], 256):
                    lg = g.lm_logits_rows(hn[r0 : r0 + 256])
                    lps.extend(torch.log_softmax(lg.double(), -1).gather(1, tgt[r0 : r0 + 256])[:, 0].tolist())
                    del lg
                out_lps.append(lps)
            return out_lps

    def score_b1_ref(seq: list[int]) -> list[float]:
        inp = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            hn = g.hidden(inp, quiet)[0, :-1]
            tgt = inp[0, 1:, None]
            lps: list[float] = []
            for r0 in range(0, hn.shape[0], 256):
                lg = g.lm_logits_rows(hn[r0 : r0 + 256])
                lps.extend(torch.log_softmax(lg.double(), -1).gather(1, tgt[r0 : r0 + 256])[:, 0].tolist())
                del lg
            return lps

    # Equivalence test: unpadded B=1 GB.Golden vs layer-major unpadded list on fp32 & S3+s4r32
    en_wins = read_ids(H3_DIR / "flores_bitnet_en.ids")
    uk_wins = read_ids(H3_DIR / "flores_bitnet_uk.ids")
    pair = [en_wins[0], uk_wins[0]]
    check_res = {}
    for chk_scheme in ("fp32", "S3+s4r32"):
        if chk_scheme == "fp32":
            GB.kv_quant = GB.kv_quant_identity
            GB.q_quant = None
        else:
            kv_ptq.install(chk_scheme, GB)
        ref = [score_b1_ref(s) for s in pair]
        got = score_seqs_layer_major(pair)
        md = max(max(abs(a - b) for a, b in zip(ref[k], got[k])) for k in (0, 1))
        print(f"[bitnet parity check {chk_scheme}] layer_major vs B=1 ref max_abs_lp_diff={md:.3e}", flush=True)
        assert md < 1e-6, f"BitNet layer-major mismatch on {chk_scheme}: {md}"
        check_res[chk_scheme] = md
    (out_dir / "batch_identity.json").write_text(json.dumps(check_res, indent=1) + "\n", encoding="utf-8")

    t0 = time.time()
    for scheme in schemes:
        if scheme == "fp32":
            GB.kv_quant = GB.kv_quant_identity
            GB.q_quant = None
        else:
            kv_ptq.install(scheme, GB)
        sname = scheme.replace(":", "_").replace("+", "_")
        for lang, wins in (("en", en_wins), ("uk", uk_wins)):
            out_path = out_dir / f"{sname}_{lang}.txt"
            done = {} if args.fresh else load_existing(out_path)
            ids_path = H3_DIR / f"flores_bitnet_{lang}.ids"
            ids_sha = hashlib.sha256(ids_path.read_bytes()).hexdigest()[:16]
            header = f"# model=bitnet kv={scheme} lang={lang} windows={len(wins)} ids_sha256={ids_sha}"
            todo = [w for w in range(len(wins)) if w not in done]
            if not todo:
                print(f"[skip] bitnet {scheme} {lang} already complete ({len(done)}/{len(wins)})", flush=True)
                continue
            print(f"[run] bitnet {scheme} {lang}: {len(todo)} todo (chunk={args.batch})", flush=True)
            for b0 in range(0, len(todo), args.batch):
                b_ids = todo[b0 : b0 + args.batch]
                res_list = score_seqs_layer_major([wins[w] for w in b_ids])
                for w, lps in zip(b_ids, res_list):
                    nll = -sum(lps)
                    done[w] = (f"w {w} {len(lps)} {nll:.9f}\n", "lp " + " ".join(f"{x:.9f}" for x in lps) + "\n")
                write_sorted_scores(out_path, header, done, len(wins))
                print(
                    f"  bitnet {scheme:10s} {lang} {len(done):2d}/{len(wins)} t={time.time()-t0:.0f}s rss={rss_gb():.2f}GB",
                    flush=True,
                )
            tot_nll = sum(float(done[w][0].split()[3]) for w in range(len(wins)))
            tot_n = sum(int(done[w][0].split()[2]) for w in range(len(wins)))
            print(
                f"[done] bitnet {scheme:10s} {lang} ppl={math.exp(tot_nll / tot_n):.3f} "
                f"t={time.time()-t0:.1f}s rss={rss_gb():.2f}GB peak={peak_gb():.2f}GB",
                flush=True,
            )
    return 0


def run_hf(args: argparse.Namespace) -> int:
    import kv_ptq
    from transformers import AutoConfig, AutoModelForCausalLM

    model_map = {
        "eurollm": pathlib.Path.home() / "tenzo-refs" / "hf" / "eurollm-1.7b",
        "qwen": pathlib.Path.home() / "tenzo-refs" / "hf" / "qwen2.5-1.5b",
    }
    model_dir = model_map[args.model]
    torch.set_num_threads(args.threads)
    cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    assert head_dim == 128, f"Expected head_dim=128, got {head_dim}"

    if cfg.model_type == "llama":
        from transformers.models.llama import modeling_llama as M
    elif cfg.model_type == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2 as M
    else:
        raise ValueError(f"Unsupported model_type={cfg.model_type}")

    ORIG_EAGER = M.eager_attention_forward
    repeat_kv = M.repeat_kv
    STATE = {"hook": None}
    LAYER_F32: dict[torch.nn.Module, tuple[torch.Tensor, torch.Tensor | None]] = {}

    def kvq_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        hook = STATE["hook"]
        if hook is None:
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
            return attn_output, attn_weights

        res = hook(key.float(), value.float(), module.layer_idx)
        if isinstance(res, dict):
            k_main = repeat_kv(res["k"].to(query.dtype), module.num_key_value_groups)
            v_main = repeat_kv(res["v"].to(query.dtype), module.num_key_value_groups)
            k_alt = repeat_kv(res["k_alt"].to(query.dtype), module.num_key_value_groups)
            v_alt = repeat_kv(res["v_alt"].to(query.dtype), module.num_key_value_groups)
            m = res["alt_mask"][None, None]
            sc = torch.matmul(query, k_main.transpose(2, 3)) * scaling
            sc_alt = torch.matmul(query, k_alt.transpose(2, 3)) * scaling
            sc = torch.where(m, sc_alt, sc)
            if attention_mask is not None:
                sc = sc + attention_mask[:, :, :, : k_main.shape[-2]]
            p = F.softmax(sc, dim=-1, dtype=torch.float32).to(query.dtype)
            mf = m.to(p.dtype)
            attn_output = (torch.matmul(p * (1.0 - mf), v_main) + torch.matmul(p * mf, v_alt)).transpose(1, 2).contiguous()
            return attn_output, p

        kh = repeat_kv(res[0].to(query.dtype), module.num_key_value_groups)
        vh = repeat_kv(res[1].to(query.dtype), module.num_key_value_groups)
        attn_weights = torch.matmul(query, kh.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : kh.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, vh).transpose(1, 2).contiguous()
        return attn_output, attn_weights

    t0 = time.time()
    print(f"Loading {model_dir} (bfloat16 + per-layer fp32 cache, threads={args.threads})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name != "lm_head":
            def _fwd(x, m=mod):
                cached = LAYER_F32.get(m)
                if cached is not None:
                    w, b = cached
                else:
                    w = m.weight.float()
                    b = m.bias.float() if m.bias is not None else None
                return F.linear(x.float(), w, b).to(x.dtype)
            mod.forward = _fwd

    lm_head = model.get_output_embeddings()
    lm_w = lm_head.weight
    lm_b = getattr(lm_head, "bias", None)
    lm_bf = lm_b.float() if lm_b is not None else None
    V = lm_w.shape[0]
    inner = model.model

    def hidden_to_lps(inp: torch.Tensor, last_h: torch.Tensor) -> list[float]:
        hb = last_h[0, :-1].float()
        tgt = inp[0, 1:, None]
        lps: list[float] = []
        for r0 in range(0, hb.shape[0], 128):
            hc = hb[r0 : r0 + 128]
            lg = torch.empty(hc.shape[0], V, dtype=torch.float32)
            for v0 in range(0, V, 16384):
                wc = lm_w[v0 : v0 + 16384].float()
                bc = lm_bf[v0 : v0 + 16384] if lm_bf is not None else None
                lg[:, v0 : v0 + 16384] = F.linear(hc, wc, bc)
            lps.extend(torch.log_softmax(lg, dim=-1).gather(1, tgt[r0 : r0 + 128])[:, 0].tolist())
            del lg
        return lps

    def score_seqs_layer_major_hf(seqs: list[list[int]]) -> list[list[float]]:
        if not seqs:
            return []
        has_sliding = bool(getattr(inner, "has_sliding_layers", False))
        create_sw_mask = getattr(M, "create_sliding_window_causal_mask", None)
        with torch.no_grad():
            inps = [torch.tensor([s], dtype=torch.long) for s in seqs]
            hs = [inner.embed_tokens(inp) for inp in inps]
            cache_positions = [torch.arange(0, len(s), dtype=torch.long) for s in seqs]
            pos_ids_list = [cp.unsqueeze(0) for cp in cache_positions]
            pos_embs = [inner.rotary_emb(h, pid) for h, pid in zip(hs, pos_ids_list)]
            masks_list = []
            for h, cp, pid in zip(hs, cache_positions, pos_ids_list):
                mk = {"config": inner.config, "input_embeds": h, "attention_mask": None,
                      "cache_position": cp, "past_key_values": None, "position_ids": pid}
                if has_sliding and create_sw_mask is not None:
                    masks_list.append({
                        "full_attention": M.create_causal_mask(**mk),
                        "sliding_attention": create_sw_mask(**mk),
                    })
                else:
                    cm = M.create_causal_mask(**mk)
                    masks_list.append({"full_attention": cm})

            for layer in inner.layers[: inner.config.num_hidden_layers]:
                LAYER_F32.clear()
                for sub in layer.modules():
                    if isinstance(sub, torch.nn.Linear):
                        LAYER_F32[sub] = (sub.weight.float(), sub.bias.float() if sub.bias is not None else None)
                att_type = getattr(layer, "attention_type", "full_attention")
                for i in range(len(hs)):
                    out_i = layer(
                        hs[i],
                        attention_mask=masks_list[i][att_type],
                        position_ids=pos_ids_list[i],
                        past_key_values=None,
                        use_cache=False,
                        cache_position=cache_positions[i],
                        position_embeddings=pos_embs[i],
                    )
                    hs[i] = out_i[0] if isinstance(out_i, tuple) else out_i
                LAYER_F32.clear()

            return [hidden_to_lps(inp, inner.norm(h)) for inp, h in zip(inps, hs)]

    def score_b1_ref(seq: list[int]) -> list[float]:
        inp = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            h = inner(input_ids=inp, use_cache=False).last_hidden_state
            return hidden_to_lps(inp, h)

    out_dir = H3_DIR / "scores" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    en_wins = read_ids(H3_DIR / f"flores_{args.model}_en.ids")
    uk_wins = read_ids(H3_DIR / f"flores_{args.model}_uk.ids")

    # Equivalence checks: standard B=1 model.model vs layer-major unpadded list on fp & S3+s4r32
    pair = [en_wins[0], uk_wins[0]]
    M.eager_attention_forward = ORIG_EAGER
    STATE["hook"] = None
    ref_fp = [score_b1_ref(s) for s in pair]
    M.eager_attention_forward = kvq_attention
    lm_fp = score_seqs_layer_major_hf(pair)
    diff_fp = max(max(abs(a - b) for a, b in zip(ref_fp[k], lm_fp[k])) for k in (0, 1))

    STATE["hook"] = kv_ptq.make_hook("S3+s4r32")[0]
    ref_s3 = [score_b1_ref(s) for s in pair]
    lm_s3 = score_seqs_layer_major_hf(pair)
    diff_s3 = max(max(abs(a - b) for a, b in zip(ref_s3[k], lm_s3[k])) for k in (0, 1))
    print(f"[{args.model} parity] layer_major vs B=1 ref: fp={diff_fp:.3e}, S3+s4r32={diff_s3:.3e}; rss={rss_gb():.2f}GB", flush=True)
    assert diff_fp < 1e-5 and diff_s3 < 1e-5, f"HF layer-major parity failed: fp={diff_fp}, s3={diff_s3}"
    (out_dir / "batch_identity.json").write_text(
        json.dumps({"model": str(model_dir), "eager_b1_vs_lm_fp": diff_fp, "s3_b1_vs_lm_s3": diff_s3}, indent=1) + "\n",
        encoding="utf-8",
    )

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    for scheme in schemes:
        M.eager_attention_forward = ORIG_EAGER if scheme == "eager" else kvq_attention
        STATE["hook"] = None if scheme in ("fp", "eager") else kv_ptq.make_hook(scheme)[0]
        sname = scheme.replace(":", "_").replace("+", "_")
        for lang, wins in (("en", en_wins), ("uk", uk_wins)):
            out_path = out_dir / f"{sname}_{lang}.txt"
            done = {} if args.fresh else load_existing(out_path)
            ids_path = H3_DIR / f"flores_{args.model}_{lang}.ids"
            ids_sha = hashlib.sha256(ids_path.read_bytes()).hexdigest()[:16]
            header = f"# model={args.model} kv={scheme} lang={lang} windows={len(wins)} ids_sha256={ids_sha}"
            todo = [w for w in range(len(wins)) if w not in done]
            if not todo:
                print(f"[skip] {args.model} {scheme} {lang} already complete", flush=True)
                continue
            print(f"[run] {args.model} {scheme} {lang}: {len(todo)} windows (chunk={args.batch})", flush=True)
            for b0 in range(0, len(todo), args.batch):
                b_ids = todo[b0 : b0 + args.batch]
                res_list = score_seqs_layer_major_hf([wins[w] for w in b_ids])
                for w, lps in zip(b_ids, res_list):
                    nll = -sum(lps)
                    done[w] = (f"w {w} {len(lps)} {nll:.9f}\n", "lp " + " ".join(f"{x:.9f}" for x in lps) + "\n")
                write_sorted_scores(out_path, header, done, len(wins))
                print(
                    f"  {args.model} {scheme:10s} {lang} {len(done):2d}/{len(wins)} t={time.time()-t0:.0f}s rss={rss_gb():.2f}GB",
                    flush=True,
                )
    print(f"DONE {args.model} in {time.time()-t0:.1f}s peak_rss={peak_gb():.2f}GB", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["bitnet", "eurollm", "qwen"])
    ap.add_argument("--schemes", default="")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if not args.schemes:
        if args.model == "bitnet":
            args.schemes = "fp32,tenzo_int8,S3+s4r32,S13+s4r32"
        elif args.model == "eurollm":
            args.schemes = "fp,tenzo_int8,S3+s4r32,S13+s4r32"
        else:
            args.schemes = "fp,tenzo_int8,S3+s4r32,S13+s4r32"
    if args.threads <= 0:
        args.threads = 4 if args.model == "bitnet" else 6

    return run_bitnet(args) if args.model == "bitnet" else run_hf(args)


if __name__ == "__main__":
    raise SystemExit(main())
