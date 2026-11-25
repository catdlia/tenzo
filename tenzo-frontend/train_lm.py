#!/usr/bin/env python3
"""Small ternary causal LM trainer with exact checkpoint/resume (T18) and optional KV-aware QAT (RQ3).

Coordinator skeleton, 2026-11-25. Pure PyTorch, no Tenzo engine. What it is for:
  * T18 — checkpoint/resume: weights + optimizer + LR schedule + RNG states + data cursor, written atomically
    (tmp file + os.replace) every --save-every steps; --resume continues bit-exactly on CPU (tested by
    tenzo-frontend/test_train_resume.py, which SIGKILLs a run mid-way and compares the loss curve).
  * T19 — smoke runs: per-step loss and tokens/s in <out>/log.jsonl.
  * RQ3 — KV-aware QAT: --kv {none, ternary, s3}. "s3" mirrors tools/reference/kv_ptq.py S3+s4r32: randomized
    Hadamard rotation, ternary groups of 32 (TWN threshold 0.7*mean|x|, LS scale), keys/values of the 4 sink tokens
    and of the last --kv-recent tokens (relative to each query) in INT8. Straight-through gradients.

Model: BitLinear (ternary weights, tenzo-frontend/qat.py) everywhere except embeddings / LM head; RMSNorm,
multi-head causal attention with RoPE, ReLU^2 MLP (as in BitNet b1.58). Data: a uint16/uint32 token file (--data
*.bin, see --dtype) or, for tests, raw bytes of any file (--data file --bytes).

Optional: --hub-repo USER/NAME uploads the checkpoint folder with huggingface_hub after each save (needs a token in
the environment; failures are logged, never fatal).
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qat import BitLinear  # noqa: E402


# ------------------------------------------------------------------ KV fake quantization (STE)
def _ste(x, q):
    return x + (q - x).detach()


def _int8(x):
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.sign(x / s) * torch.floor((x / s).abs() + 0.5).clamp(max=127) * s


def _tern_g32(x, g=32):
    sh = x.shape
    y = x.reshape(*sh[:-1], sh[-1] // g, g)
    th = 0.7 * y.abs().mean(-1, keepdim=True)
    nz = (y.abs() > th).to(y.dtype)
    cnt = nz.sum(-1, keepdim=True)
    s = torch.where(cnt > 0, (y.abs() * nz).sum(-1, keepdim=True) / cnt.clamp(min=1), th / 0.7 + 1e-6)
    return (torch.sign(y) * nz * s).reshape(sh)


def hadamard(d, seed=0):
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (d,), generator=g).float() * 2 - 1
    return h * signs[None, :] / math.sqrt(d)


# ------------------------------------------------------------------ model
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope(x, base=10000.0):
    T, D = x.shape[-2], x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, D, 2, dtype=torch.float32) / D)
    f = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Attention(nn.Module):
    def __init__(self, d, heads, kv, sinks, recent):
        super().__init__()
        self.h, self.hd, self.kv, self.sinks, self.recent = heads, d // heads, kv, sinks, recent
        self.qkv = BitLinear(d, 3 * d, bias=False)
        self.o = BitLinear(d, d, bias=False)
        self.sub_norm = RMSNorm(d)
        if kv == "s3":
            self.register_buffer("R", hadamard(self.hd), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        if self.kv == "none":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.kv == "ternary":  # per-token TWN ternary, no window (the old qat.py behaviour, but causal)
            out = F.scaled_dot_product_attention(q, _ste(k, _tern_g32(k, self.hd)), _ste(v, _tern_g32(v, self.hd)), is_causal=True)
        else:  # s3: rotation + ternary g32; sinks and the recent window from INT8
            R = self.R.to(k.dtype)
            k_t, v_t = _ste(k, _tern_g32(k @ R) @ R.T), _ste(v, _tern_g32(v @ R) @ R.T)
            k_8, v_8 = _ste(k, _int8(k)), _ste(v, _int8(v))
            t = torch.arange(T)[:, None]; s = torch.arange(T)[None, :]
            hp = (s < self.sinks) | ((t - s) < self.recent)            # positions served from INT8
            sc_t = (q @ k_t.transpose(-1, -2)) / math.sqrt(self.hd)
            sc_8 = (q @ k_8.transpose(-1, -2)) / math.sqrt(self.hd)
            sc = torch.where(hp, sc_8, sc_t).masked_fill(~causal, float("-inf"))
            p = sc.softmax(-1)
            out = (p * hp) @ v_8 + (p * ~hp) @ v_t
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o(self.sub_norm(out))


class Block(nn.Module):
    def __init__(self, d, heads, ff, kv, sinks, recent):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, heads, kv, sinks, recent)
        self.up, self.down = BitLinear(d, ff, bias=False), BitLinear(ff, d, bias=False)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.down(F.relu(self.up(self.n2(x))) ** 2)


class LM(nn.Module):
    def __init__(self, vocab, d, layers, heads, ff, kv, sinks, recent):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, kv, sinks, recent) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, std=0.02)   # tied head: keep initial logits small (loss ~ ln vocab)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


# ------------------------------------------------------------------ data with a resumable cursor
class Stream:
    """Contiguous windows over a token array, visited in a seeded random order per epoch; state = (epoch, pos)."""

    def __init__(self, tokens: np.ndarray, seq: int, batch: int, seed: int):
        self.t, self.seq, self.batch, self.seed = tokens, seq, batch, seed
        self.n = (len(tokens) - 1) // seq
        assert self.n >= batch, "data too small for one batch"
        self.epoch, self.pos = 0, 0
        self._order()

    def _order(self):
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(self.n)

    def next(self):
        if self.pos + self.batch > self.n:
            self.epoch += 1; self.pos = 0; self._order()
        ids = self.order[self.pos: self.pos + self.batch]
        self.pos += self.batch
        x = np.stack([self.t[i * self.seq: i * self.seq + self.seq + 1] for i in ids]).astype(np.int64)
        return torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])

    def state(self):
        return dict(epoch=self.epoch, pos=self.pos)

    def load(self, st):
        self.epoch, self.pos = st["epoch"], st["pos"]
        self._order()


# ------------------------------------------------------------------ checkpointing
def save_ckpt(out: pathlib.Path, step, model, opt, sched, stream, keep=2):
    st = dict(step=step, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), data=stream.state(),
              rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate()))
    tmp = out / f".ckpt_{step:08d}.pt.tmp"
    torch.save(st, tmp)
    os.replace(tmp, out / f"ckpt_{step:08d}.pt")             # atomic on POSIX
    (out / "latest.tmp").write_text(f"ckpt_{step:08d}.pt\n")
    os.replace(out / "latest.tmp", out / "latest")
    for old in sorted(out.glob("ckpt_*.pt"))[:-keep]:
        old.unlink()


def load_ckpt(out: pathlib.Path, model, opt, sched, stream):
    p = out / "latest"
    if not p.exists():
        return 0
    st = torch.load(out / p.read_text().strip(), weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    stream.load(st["data"])
    torch.set_rng_state(st["rng"]["torch"]); np.random.set_state(st["rng"]["numpy"]); random.setstate(st["rng"]["python"])
    return st["step"]


def hub_upload(out, repo):
    try:
        from huggingface_hub import upload_folder
        upload_folder(repo_id=repo, folder_path=str(out), allow_patterns=["ckpt_*.pt", "latest", "config.json"],
                      commit_message="checkpoint")
    except Exception as e:  # never fatal
        print(f"[hub] upload failed: {e}", flush=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--bytes", action="store_true", help="treat --data as raw bytes (vocab 256), for tests")
    ap.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ff", type=int, default=384)
    ap.add_argument("--kv", default="none", choices=["none", "ternary", "s3"])
    ap.add_argument("--kv-sinks", type=int, default=4); ap.add_argument("--kv-recent", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50); ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--hub-repo", default="")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tokens = (np.frombuffer(pathlib.Path(a.data).read_bytes(), dtype=np.uint8) if a.bytes
              else np.memmap(a.data, dtype=a.dtype, mode="r"))
    vocab = 256 if a.bytes else a.vocab
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))

    model = LM(vocab, a.d, a.layers, a.heads, a.ff, a.kv, a.kv_sinks, a.kv_recent)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    stream = Stream(tokens, a.seq, a.batch, a.seed)
    step = load_ckpt(out, model, opt, sched, stream) if a.resume else 0
    if step:
        print(f"resumed at step {step}", flush=True)
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    while step < a.steps:
        x, y = stream.next()
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        dt = time.time() - t0; t0 = time.time()
        log.write(json.dumps(dict(step=step, loss=float(loss), loss_hex=float(loss).hex(), lr=sched.get_last_lr()[0],
                                  tok_s=a.batch * a.seq / max(dt, 1e-9))) + "\n"); log.flush()
        if step % a.save_every == 0 or step == a.steps:
            save_ckpt(out, step, model, opt, sched, stream)
            if a.hub_repo:
                hub_upload(out, a.hub_repo)
    print(f"done at step {step}, last loss {float(loss):.4f}", flush=True)


if __name__ == "__main__":
    main()
