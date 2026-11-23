#!/usr/bin/env python3
"""Evaluate HF LlamaForCausalLM / Qwen2ForCausalLM models across H2 2x2 windows under kv_ptq schemes.

Stores weights in bfloat16 (fits MemoryMax=4G) while executing nn.Linear in float32 via on-the-fly
weight casting (3-4x faster on CPUs without AVX-512 BF16) and token/vocab-chunked lm_head (<180 MB peak).
Verifies runtime patch equivalence (ORIG_EAGER vs patched fp on the first --check-windows windows).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REF_DIR = SCRIPT_DIR.parent / "tools" / "reference"
sys.path.insert(0, str(REF_DIR))

import kv_ptq  # noqa: E402
from tokenizers import Tokenizer as FastTok  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402


STATE = {"hook": None}


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


def install_fp32_linear_cast(model) -> int:
    """Wrap transformer nn.Linear modules to compute F.linear in float32 on CPU."""
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name != "lm_head":
            def _fwd(x, m=mod):
                b = m.bias.float() if m.bias is not None else None
                return torch.nn.functional.linear(x.float(), m.weight.float(), b).to(x.dtype)
            mod.forward = _fwd
            count += 1
    return count


def window_logprobs(model, ids_list: list[int]) -> list[float]:
    ids = torch.tensor([ids_list], dtype=torch.long)
    lm_head = model.get_output_embeddings()
    lm_w = lm_head.weight
    lm_b = getattr(lm_head, "bias", None)
    lm_bf = lm_b.float() if lm_b is not None else None
    V = lm_w.shape[0]
    targets = ids[0, 1:, None]
    out_lps: list[float] = []
    with torch.no_grad():
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1].float()
        T_out = h.shape[0]
        for t0 in range(0, T_out, 128):
            hc = h[t0 : t0 + 128]
            lg = torch.empty(hc.shape[0], V, dtype=torch.float32)
            for v0 in range(0, V, 16384):
                w_chunk = lm_w[v0 : v0 + 16384].float()
                b_chunk = lm_bf[v0 : v0 + 16384] if lm_bf is not None else None
                lg[:, v0 : v0 + 16384] = torch.nn.functional.linear(hc, w_chunk, b_chunk)
            lp = torch.log_softmax(lg, dim=-1).gather(1, targets[t0 : t0 + 128])[:, 0].tolist()
            out_lps.extend(lp)
    return out_lps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--windows-json", default="results/2026-11-23_agent2/H2/windows_2x2.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schemes", default="fp,tenzo_int8,S3+s4r32,S13+s4r32")
    ap.add_argument("--cells", default="en_wiki,uk_wiki,en_web,uk_web")
    ap.add_argument("--check-windows", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--fresh", action="store_true", help="Ignore partial score files and overwrite from scratch")
    ap.add_argument("--no-fp32-cast", action="store_true", help="Disable float32 nn.Linear cast")
    args = ap.parse_args()

    t0 = time.time()
    torch.set_num_threads(args.threads)
    model_dir = pathlib.Path(args.model_dir)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    assert head_dim == 128, f"kv_ptq requires head_dim=128, got {head_dim}"

    if cfg.model_type == "llama":
        from transformers.models.llama import modeling_llama as M
    elif cfg.model_type == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2 as M
    else:
        raise ValueError(f"Unsupported model_type={cfg.model_type}")

    ORIG_EAGER = M.eager_attention_forward
    repeat_kv = M.repeat_kv

    def kvq_attention(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        hook = STATE["hook"]
        if hook is None:
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
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
                causal_mask = attention_mask[:, :, :, : k_main.shape[-2]]
                sc = sc + causal_mask
            p = torch.nn.functional.softmax(sc, dim=-1, dtype=torch.float32).to(query.dtype)
            mf = m.to(p.dtype)
            attn_output = torch.matmul(p * (1.0 - mf), v_main) + torch.matmul(p * mf, v_alt)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, p

        kh = repeat_kv(res[0].to(query.dtype), module.num_key_value_groups)
        vh = repeat_kv(res[1].to(query.dtype), module.num_key_value_groups)
        attn_weights = torch.matmul(query, kh.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : kh.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, vh).transpose(1, 2).contiguous()
        return attn_output, attn_weights

    tok = FastTok.from_file(str(model_dir / "tokenizer.json"))
    bos_id = cfg.bos_token_id if getattr(cfg, "bos_token_id", None) is not None else tok.token_to_id("<s>")

    meta = json.loads(pathlib.Path(args.windows_json).read_text(encoding="utf-8"))
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    cell_inputs: dict[str, list[list[int]]] = {}
    scored_meta: dict[str, list[dict]] = {}
    for c in cells:
        seq_list, m_list = [], []
        for w in meta["cells"][c]["windows"]:
            raw_ids = tok.encode(w["text"], add_special_tokens=False).ids
            if bos_id is not None:
                ids = [int(bos_id)] + list(raw_ids)
                scored_ids = list(raw_ids)
                scored_text = w["text"]
            else:
                ids = list(raw_ids)
                scored_ids = list(raw_ids[1:])
                scored_text = tok.decode(scored_ids, skip_special_tokens=False)
            seq_list.append(ids)
            m_list.append(
                {
                    "wid": w["wid"],
                    "n_tok": len(scored_ids),
                    "n_bytes": len(scored_text.encode("utf-8")),
                    "n_chars": len(scored_text),
                    "orig_bytes": w["n_bytes"],
                    "orig_chars": w["n_chars"],
                }
            )
        cell_inputs[c] = seq_list
        scored_meta[c] = m_list
    (out_dir / "token_meta.json").write_text(
        json.dumps({"model": str(model_dir), "model_type": cfg.model_type, "bos_id": bos_id, "cells": scored_meta}, indent=1)
        + "\n",
        encoding="utf-8",
    )

    print(f"Loading {model_dir} ({cfg.model_type}, bfloat16, eager, threads={args.threads})... rss={rss_gb():.2f}GB", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    if not args.no_fp32_cast:
        n_cast = install_fp32_linear_cast(model)
        print(f"Installed float32 linear cast on {n_cast} layers; rss={rss_gb():.2f}GB", flush=True)

    # Also patch attn_weights float32 stability check vs ORIG_EAGER identically!
    def run_scheme(scheme: str, cell: str, nmax: int | None = None) -> tuple[pathlib.Path, dict[int, list[float]]]:
        sname = scheme.replace(":", "_").replace("+", "_")
        out_path = out_dir / f"{sname}_{cell}.txt"
        done = {} if args.fresh else load_existing(out_path)
        M.eager_attention_forward = ORIG_EAGER if scheme == "eager" else kvq_attention
        STATE["hook"] = None if scheme in ("fp", "eager") else kv_ptq.make_hook(scheme)[0]

        seqs = cell_inputs[cell]
        sel = list(range(len(seqs)))[:nmax]
        lps_map: dict[int, list[float]] = {}
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"# {scheme} {cell} windows={len(sel)}\n")
            for w in sel:
                if w in done:
                    f.write(done[w][0])
                    f.write(done[w][1])
                    lps_map[w] = [float(x) for x in done[w][1].split()[1:]]
                    continue
                lp = window_logprobs(model, seqs[w])
                lps_map[w] = lp
                nll = -sum(lp)
                f.write(f"w {w} {len(lp)} {nll:.9f}\n")
                f.write("lp " + " ".join(f"{x:.9f}" for x in lp) + "\n")
                f.flush()
                print(
                    f"{scheme:10s} {cell:8s} w={w:2d} n={len(lp)} ppl={math.exp(nll / len(lp)):7.3f} "
                    f"t={time.time()-t0:.0f}s rss={rss_gb():.2f}GB",
                    flush=True,
                )
        return out_path, lps_map

    check_cells = [c for c in ("en_wiki", "uk_wiki") if c in cell_inputs]
    eager_lps = {}
    if args.check_windows > 0:
        for c in check_cells:
            _, eager_lps[c] = run_scheme("eager", c, nmax=args.check_windows)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    max_patch_diff = 0.0
    patch_checked = False
    for scheme in schemes:
        for c in cells:
            _, lps_map = run_scheme(scheme, c)
            if scheme == "fp" and args.check_windows > 0 and c in eager_lps:
                for w in range(args.check_windows):
                    diffs = [abs(a - b) for a, b in zip(eager_lps[c][w], lps_map[w])]
                    max_patch_diff = max(max_patch_diff, max(diffs))
        if scheme == "fp" and args.check_windows > 0 and not patch_checked:
            patch_checked = True
            print(f"PATCH_IDENTITY eager vs fp max_abs_lp_diff={max_patch_diff:.3e}", flush=True)
            (out_dir / "patch_identity.json").write_text(
                json.dumps({"model": str(model_dir), "max_abs_lp_diff": max_patch_diff, "check_windows": args.check_windows}, indent=1)
                + "\n",
                encoding="utf-8",
            )
            assert max_patch_diff < 1e-5, f"Patch identity failed: {max_patch_diff}"

    print(f"DONE {time.time() - t0:.1f}s peak_rss={peak_gb():.2f}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
