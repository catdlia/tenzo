#!/usr/bin/env python3
"""Shared layer-major FP32 evaluation & bootstrap utilities for Tenzo KV studies."""

from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import sys
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "tools" / "reference"
if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))


def read_ids(path: pathlib.Path) -> list[list[int]]:
    return [list(map(int, ln.split())) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_existing(path: pathlib.Path) -> dict[int, tuple[str, str]]:
    done: dict[int, tuple[str, str]] = {}
    if not path.exists():
        return done
    lines = path.read_text(encoding="utf-8").splitlines(True)
    i = 0
    while i < len(lines):
        if lines[i].startswith("w ") and i + 1 < len(lines) and lines[i + 1].startswith("lp "):
            wid = int(lines[i].split()[1])
            done[wid] = (lines[i], lines[i + 1])
            i += 2
        else:
            i += 1
    return done


def write_sorted_scores(path: pathlib.Path, header: str, done_map: dict[int, tuple[str, str]], n_total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(header.rstrip("\n") + "\n")
        for w in range(n_total):
            if w in done_map:
                f.write(done_map[w][0])
                f.write(done_map[w][1])
    tmp.replace(path)


def load_nlls(path: pathlib.Path, n_expected: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (tok_counts, nlls) ordered by window ID."""
    counts: dict[int, int] = {}
    nlls: dict[int, float] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("w "):
            _, wid_s, n_s, nll_s = ln.split()
            wid = int(wid_s)
            counts[wid] = int(n_s)
            nlls[wid] = float(nll_s)
    keys = sorted(nlls.keys())
    if n_expected is not None:
        assert len(keys) == n_expected and keys == list(range(n_expected)), (
            f"Incomplete score file {path}: {len(keys)} vs {n_expected}"
        )
    return (
        np.array([counts[k] for k in keys], dtype=np.int64),
        np.array([nlls[k] for k in keys], dtype=np.float64),
    )


def load_lps(path: pathlib.Path, n_expected: int | None = None) -> list[np.ndarray]:
    lps: dict[int, np.ndarray] = {}
    cur_wid = None
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("w "):
            cur_wid = int(ln.split()[1])
        elif ln.startswith("lp ") and cur_wid is not None:
            lps[cur_wid] = np.array([float(x) for x in ln.split()[1:]], dtype=np.float64)
            cur_wid = None
    keys = sorted(lps.keys())
    if n_expected is not None:
        assert len(keys) == n_expected and keys == list(range(n_expected)), (
            f"Incomplete score file {path}: {len(keys)} vs {n_expected}"
        )
    return [lps[k] for k in keys]


def rss_gb() -> float:
    for ln in pathlib.Path("/proc/self/status").read_text().splitlines():
        if ln.startswith("VmRSS:"):
            return int(ln.split()[1]) / 1048576.0
    return 0.0


def peak_gb() -> float:
    for ln in pathlib.Path("/proc/self/status").read_text().splitlines():
        if ln.startswith("VmHWM:"):
            return int(ln.split()[1]) / 1048576.0
    return 0.0


def boot_mean_ci(arr: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    m = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def build_char_offsets(tok_json: pathlib.Path, seqs: list[list[int]], texts: list[str]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from tokenizers import Tokenizer as FastTok

    ft = FastTok.from_file(str(tok_json))
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for s_ids, txt in zip(seqs, texts):
        enc = ft.encode(txt, add_special_tokens=False)
        assert enc.ids == s_ids[1:], f"Tokenizer mismatch vs .ids: {len(enc.ids)} vs {len(s_ids)-1}"
        t_len = len(s_ids)
        c_arr = np.zeros(t_len, dtype=np.int32)
        e_arr = np.zeros(t_len, dtype=np.int32)
        l_arr = np.zeros(t_len, dtype=np.int32)
        for idx, (cs, ce) in enumerate(enc.offsets, start=1):
            c_arr[idx] = cs
            e_arr[idx] = ce
            l_arr[idx] = ce - cs
        out.append((c_arr, e_arr, l_arr))
    return out


def char_window_mask(c_arr: np.ndarray, e_arr: np.ndarray, char_budget: int, sinks: int = 4) -> torch.Tensor:
    t_len = len(c_arr)
    t_idx = np.arange(t_len)[:, None]
    s_idx = np.arange(t_len)[None, :]
    m = (s_idx <= t_idx) & ((s_idx < sinks) | (s_idx == t_idx) | ((e_arr[t_idx] - c_arr[s_idx]) <= char_budget))
    return torch.from_numpy(m)


def coverage_stats_char_c(
    offsets_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    char_budget: int,
    sinks: int = 4,
) -> dict[str, float]:
    cov_steady, cov_all, r_effs, r_final, tok_lens = [], [], [], [], []
    for c_arr, e_arr, _ in offsets_list:
        t_len = len(c_arr)
        tok_lens.append(float(t_len))
        m = char_window_mask(c_arr, e_arr, char_budget, sinks=sinks).numpy()
        for t in range(1, t_len):
            non_sink = np.where((np.arange(t_len) >= sinks) & (np.arange(t_len) <= t) & m[t])[0]
            if len(non_sink):
                s0 = int(non_sink[0])
                span = float(e_arr[t] - c_arr[s0])
                cov_all.append(span)
                if s0 > sinks:
                    cov_steady.append(span)
                r_effs.append(float(t - s0 + 1))
            else:
                r_effs.append(float(t + 1))
        non_sink_last = np.where((np.arange(t_len) >= sinks) & m[t_len - 1])[0]
        r_final.append(float((t_len - 1) - int(non_sink_last[0]) + 1) if len(non_sink_last) else 0.0)
    t_m = float(np.mean(tok_lens))
    re_m = float(np.mean(r_effs))
    hi_512 = min(512.0, float(sinks) + re_m)
    hi_win = min(t_m, float(sinks) + re_m)
    i8, ter = 8.0 + 16.0 / 128.0, 2.0 + 16.0 / 32.0
    return {
        "cov_chars_exact": float(np.mean(cov_steady)) if cov_steady else float(np.mean(cov_all)),
        "cov_chars_all_t": float(np.mean(cov_all)),
        "r_eff": re_m,
        "r_eff_end": float(np.mean(r_final)),
        "mean_tok": t_m - 1.0,
        "bits_win": (hi_win * i8 + max(0.0, t_m - hi_win) * ter) / t_m,
        "bits_at_512": (hi_512 * i8 + (512.0 - hi_512) * ter) / 512.0,
    }


def make_custom_char_hook(v_only: bool, state: dict):
    import kv_ptq

    def hook(k: torch.Tensor, v: torch.Tensor, layer: int):
        mask = state["masks"][state["idx"]]
        assert mask.shape[0] == k.shape[2], (mask.shape, k.shape)
        kh = kv_ptq.int8_absmax(k) if v_only else kv_ptq.tern_g32_rot(k)
        vh = kv_ptq.tern_g32_rot(v)
        return {
            "k": kh,
            "v": vh,
            "k_alt": kv_ptq.int8_absmax(k),
            "v_alt": kv_ptq.int8_absmax(v),
            "alt_mask": mask,
        }

    return hook


class BitNetRunner:
    def __init__(self, threads: int = 4):
        import golden_bitnet as GB
        import kv_ptq

        GB.VERBOSE = False
        self.GB = GB
        self.kv_ptq = kv_ptq
        self.g = GB.Golden(GB.SNAP, torch.float32, threads)
        self.quiet = GB.Dumper(None, 0, [], [])
        self.state = {"masks": None, "idx": 0}

    def set_scheme(self, scheme: str) -> None:
        GB = self.GB
        kv_ptq = self.kv_ptq
        GB.q_quant = None
        if scheme == "fp32":
            GB.kv_quant = GB.kv_quant_identity
            self.state["masks"] = None
        elif scheme.startswith("S3+s4c") or scheme.startswith("K8V:S3+s4c"):
            GB.kv_quant = make_custom_char_hook(scheme.startswith("K8V:"), self.state)
        else:
            self.state["masks"] = None
            kv_ptq.install(scheme, GB)

    def score_b1_ref(self, seq: list[int], mask: torch.Tensor | None = None) -> list[float]:
        self.state["masks"] = [mask] if mask is not None else None
        self.state["idx"] = 0
        inp = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            hn = self.g.hidden(inp, self.quiet)[0, :-1]
            tgt = inp[0, 1:, None]
            lps: list[float] = []
            for r0 in range(0, hn.shape[0], 256):
                lg = self.g.lm_logits_rows(hn[r0 : r0 + 256])
                lps.extend(torch.log_softmax(lg.double(), -1).gather(1, tgt[r0 : r0 + 256])[:, 0].tolist())
                del lg
        return lps

    def score_seqs(self, seqs: list[list[int]], seq_masks: list[torch.Tensor] | None = None) -> list[list[float]]:
        g = self.g
        self.state["masks"] = seq_masks
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
                    self.state["idx"] = idx_s
                    bsz, t_len, _ = x.shape
                    h = g.rmsnorm(x, w_in)
                    q = g.bitlinear(h, wq).view(bsz, t_len, g.H, g.D).transpose(1, 2)
                    k = g.bitlinear(h, wk).view(bsz, t_len, g.Hkv, g.D).transpose(1, 2)
                    v = g.bitlinear(h, wv).view(bsz, t_len, g.Hkv, g.D).transpose(1, 2)
                    c, s = cos_all[:t_len][None, None], sin_all[:t_len][None, None]
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
                out_lps.append(lps)
        return out_lps


class EuroLLMRunner:
    def __init__(self, model_dir: pathlib.Path, threads: int = 6):
        import kv_ptq
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.llama import modeling_llama as M

        torch.set_num_threads(threads)
        self.kv_ptq = kv_ptq
        self.M = M
        self.orig_eager = M.eager_attention_forward
        self.repeat_kv = M.repeat_kv
        self.state = {"hook": None, "masks": None, "idx": 0}

        print(f"Loading {model_dir} (bfloat16 + per-layer fp32 linear cache, threads={threads})... rss={rss_gb():.2f}GB", flush=True)
        self.cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model.eval()
        self._install_fp32_linear_cast()

        runner = self

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
            hook = runner.state["hook"]
            if hook is None:
                key_states = runner.repeat_kv(key, module.num_key_value_groups)
                value_states = runner.repeat_kv(value, module.num_key_value_groups)
                attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
                attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
                return attn_output, attn_weights

            res = hook(key.float(), value.float(), module.layer_idx)
            if isinstance(res, dict):
                k_main = runner.repeat_kv(res["k"].to(query.dtype), module.num_key_value_groups)
                v_main = runner.repeat_kv(res["v"].to(query.dtype), module.num_key_value_groups)
                k_alt = runner.repeat_kv(res["k_alt"].to(query.dtype), module.num_key_value_groups)
                v_alt = runner.repeat_kv(res["v_alt"].to(query.dtype), module.num_key_value_groups)
                m = res["alt_mask"][None, None]
                sc = torch.matmul(query, k_main.transpose(2, 3)) * scaling
                sc_alt = torch.matmul(query, k_alt.transpose(2, 3)) * scaling
                sc = torch.where(m, sc_alt, sc)
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : k_main.shape[-2]]
                    sc = sc + causal_mask
                p = F.softmax(sc, dim=-1, dtype=torch.float32).to(query.dtype)
                mf = m.to(p.dtype)
                attn_output = torch.matmul(p * (1.0 - mf), v_main) + torch.matmul(p * mf, v_alt)
                return attn_output.transpose(1, 2).contiguous(), p

            kh = runner.repeat_kv(res[0].to(query.dtype), module.num_key_value_groups)
            vh = runner.repeat_kv(res[1].to(query.dtype), module.num_key_value_groups)
            attn_weights = torch.matmul(query, kh.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : kh.shape[-2]]
                attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_output = torch.matmul(attn_weights, vh).transpose(1, 2).contiguous()
            return attn_output, attn_weights

        self.kvq_attention = kvq_attention

    def _install_fp32_linear_cast(self) -> None:
        lm_head = self.model.lm_head

        def wrap_linear(mod: torch.nn.Linear, cast_back: bool):
            def _fwd(x: torch.Tensor) -> torch.Tensor:
                w = getattr(mod, "_cached_w_f32", None)
                if w is None:
                    w = mod.weight.float()
                b = getattr(mod, "_cached_b_f32", None)
                if b is None and mod.bias is not None:
                    b = mod.bias.float()
                out = F.linear(x.float(), w, b)
                return out.to(x.dtype) if cast_back else out

            mod.forward = _fwd

        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                wrap_linear(m, cast_back=(m is not lm_head))

    def set_scheme(self, scheme: str) -> None:
        self.M.eager_attention_forward = self.orig_eager if scheme == "eager" else self.kvq_attention
        if scheme in ("fp", "fp32", "eager"):
            self.state["hook"] = None
            self.state["masks"] = None
        elif scheme.startswith("S3+s4c") or scheme.startswith("K8V:S3+s4c"):
            self.state["hook"] = make_custom_char_hook(scheme.startswith("K8V:"), self.state)
        else:
            self.state["hook"] = self.kv_ptq.make_hook(scheme)[0]
            self.state["masks"] = None

    def score_seqs(self, seqs: list[list[int]], seq_masks: list[torch.Tensor] | None = None) -> list[list[float]]:
        self.state["masks"] = seq_masks
        inps = [torch.tensor([s], dtype=torch.long) for s in seqs]
        tmax = max(len(s) for s in seqs)
        pos_all = torch.arange(tmax, dtype=torch.long)[None, :]
        mask_full = torch.full((tmax, tmax), float("-inf"), dtype=torch.bfloat16).triu(1)
        with torch.no_grad():
            xs = [self.model.model.embed_tokens(inp) for inp in inps]
            dummy = torch.zeros((1, tmax, self.cfg.hidden_size), dtype=torch.bfloat16)
            cos_all, sin_all = self.model.model.rotary_emb(dummy, pos_all)
            for layer in self.model.model.layers:
                mods = [m for m in layer.modules() if isinstance(m, torch.nn.Linear)]
                for m in mods:
                    m._cached_w_f32 = m.weight.detach().float()
                    if m.bias is not None:
                        m._cached_b_f32 = m.bias.detach().float()
                try:
                    for idx_s, x in enumerate(xs):
                        self.state["idx"] = idx_s
                        t_len = x.shape[1]
                        am = mask_full[:t_len, :t_len][None, None]
                        pe = (cos_all[:, :t_len, :], sin_all[:, :t_len, :])
                        pid = pos_all[:, :t_len]
                        xs[idx_s] = layer(x, attention_mask=am, position_ids=pid, position_embeddings=pe)
                finally:
                    for m in mods:
                        m._cached_w_f32 = None
                        m._cached_b_f32 = None

            lm = self.model.lm_head
            lm._cached_w_f32 = lm.weight.detach().float()
            out_lps: list[list[float]] = []
            try:
                for inp, x in zip(inps, xs):
                    hn = self.model.model.norm(x)[0, :-1]
                    tgt = inp[0, 1:, None]
                    lps: list[float] = []
                    for r0 in range(0, hn.shape[0], 256):
                        lg = lm(hn[r0 : r0 + 256])
                        lps.extend(torch.log_softmax(lg.float(), -1).gather(1, tgt[r0 : r0 + 256])[:, 0].tolist())
                    out_lps.append(lps)
            finally:
                lm._cached_w_f32 = None
        return out_lps

    def score_b1_hf_eager(self, seq: list[int], use_kvq: bool = False, mask: torch.Tensor | None = None) -> list[float]:
        """Full HuggingFace LlamaForCausalLM forward on one sequence."""
        self.state["masks"] = [mask] if mask is not None else None
        self.state["idx"] = 0
        prev_attn = self.M.eager_attention_forward
        self.M.eager_attention_forward = self.kvq_attention if use_kvq else self.orig_eager
        inp = torch.tensor([seq], dtype=torch.long)
        try:
            with torch.no_grad():
                logits = self.model(inp, use_cache=False).logits[0, :-1].float()
                lp = torch.log_softmax(logits, dim=-1)
                tgt = inp[0, 1:, None]
                return lp.gather(1, tgt)[:, 0].tolist()
        finally:
            self.M.eager_attention_forward = prev_attn


def run_eval_matrix(
    model_name: str,
    runner,
    ids_paths: dict[str, pathlib.Path],
    schemes: list[str],
    out_dir: pathlib.Path,
    batch_size: int = 48,
    masks_by_scheme_lang: dict[tuple[str, str], list[torch.Tensor]] | None = None,
    fresh: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for scheme in schemes:
        runner.set_scheme(scheme)
        sname = scheme.replace(":", "_").replace("+", "_")
        for lang, ids_path in ids_paths.items():
            wins = read_ids(ids_path)
            ids_sha = hashlib.sha256(ids_path.read_bytes()).hexdigest()[:16]
            out_path = out_dir / f"{sname}_{lang}.txt"
            done = {} if fresh else load_existing(out_path)
            header = f"# model={model_name} kv={scheme} lang={lang} windows={len(wins)} ids_sha256={ids_sha}"
            todo = [w for w in range(len(wins)) if w not in done]
            if not todo:
                print(f"[skip] {model_name} {scheme} {lang} already complete ({len(done)}/{len(wins)})", flush=True)
                continue
            masks_all = None if masks_by_scheme_lang is None else masks_by_scheme_lang.get((scheme, lang))
            print(f"[run] {model_name} {scheme} {lang}: {len(todo)} windows (chunk={batch_size})", flush=True)
            for b0 in range(0, len(todo), batch_size):
                b_ids = todo[b0 : b0 + batch_size]
                b_seqs = [wins[w] for w in b_ids]
                b_masks = [masks_all[w] for w in b_ids] if masks_all is not None else None
                res_list = runner.score_seqs(b_seqs, b_masks)
                for w, lps in zip(b_ids, res_list):
                    nll = -sum(lps)
                    done[w] = (f"w {w} {len(lps)} {nll:.9f}\n", "lp " + " ".join(f"{x:.9f}" for x in lps) + "\n")
                write_sorted_scores(out_path, header, done, len(wins))
                print(
                    f"  {model_name} {scheme:10s} {lang} {len(done):2d}/{len(wins)} t={time.time()-t0:.0f}s rss={rss_gb():.2f}GB",
                    flush=True,
                )
            tot_nll = sum(float(done[w][0].split()[3]) for w in range(len(wins)))
            tot_tok = sum(int(done[w][0].split()[2]) for w in range(len(wins)))
            print(
                f"[done] {model_name} {scheme:10s} {lang} nll={tot_nll:.3f} tok={tot_tok} "
                f"t={time.time()-t0:.1f}s rss={rss_gb():.2f}GB peak={peak_gb():.2f}GB",
                flush=True,
            )
