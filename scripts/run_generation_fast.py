#!/usr/bin/env python3
import os
import sys
import time
import re
import argparse
import numpy as np

# Ensure tenzo_runtime can be imported from build directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BUILD_DIR = os.path.join(PROJECT_ROOT, "cmake-build-debug")
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

try:
    import tenzo_runtime
except ImportError as e:
    print(f"❌ Failed to import tenzo_runtime: {e}")
    sys.exit(1)


class Tokenizer:
    def __init__(self, vocab_path):
        self.id_to_token = {}
        self.token_to_id = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for rank, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    tok_id = int(parts[0])
                    tok_str = parts[1].replace("Ġ", " ")
                    self.id_to_token[tok_id] = tok_str
                    self.token_to_id[tok_str] = tok_id

    def encode(self, text):
        tokens = [128000] if 128000 in self.id_to_token else []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(32, len(text) - i), 0, -1):
                sub = text[i:i+length]
                if sub in self.token_to_id:
                    tokens.append(self.token_to_id[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                i += 1
        return tokens

    def decode(self, token_ids):
        return "".join([self.id_to_token.get(t, "") for t in token_ids])


def unpack_2bit_weights(raw, out_dim, in_dim):
    c0 = (raw & 0x03).astype(np.int8) - 1
    c1 = ((raw >> 2) & 0x03).astype(np.int8) - 1
    c2 = ((raw >> 4) & 0x03).astype(np.int8) - 1
    c3 = ((raw >> 6) & 0x03).astype(np.int8) - 1
    
    tern = np.zeros((out_dim, in_dim), dtype=np.int8)
    tern[:, 0::4] = c0
    tern[:, 1::4] = c1
    tern[:, 2::4] = c2
    tern[:, 3::4] = c3
    return tern


def parse_mlir_and_load_weights(mlir_path, weights_path, max_seq_len=8192, kv_mode="int8_fused"):
    print(f"📖 Parsing MLIR graph: {mlir_path}")
    with open(mlir_path, "r") as f:
        mlir_text = f.read()

    print(f"📦 Mapping binary weights: {weights_path}")
    raw_bytes = np.fromfile(weights_path, dtype=np.uint8)

    vocab_size = 128256
    hidden_size = 2560
    ff_dim = 6912
    num_q_heads = 20
    num_kv_heads = 5
    head_dim = 128
    num_layers = 30

    # 1. Embeddings: 128256 x 2560 float32
    embed_bytes = vocab_size * hidden_size * 4
    embed_tokens_w = raw_bytes[0:embed_bytes].view(np.float32).reshape(vocab_size, hidden_size)

    pattern = re.compile(r'(%\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*index\s*\n\s*(%\w+)\s*=\s*memref\.view\s+%\w+\[\1\]\[\]\s*:\s*memref<\?xi8>\s+to\s+memref<([^>]+)>')
    all_views = []
    for m in pattern.finditer(mlir_text):
        offset = int(m.group(2))
        shape_str = m.group(4)
        all_views.append((offset, shape_str))

    scale_pattern = re.compile(r'(%\w+)\s*=\s*arith\.constant\s+([0-9\.e\+\-]+)\s*:\s*f32\s*\n\s*%\w+\s*=\s*"tenzo\.bitlinear_elut"\([^,]+,\s*[^,]+,\s*\1\)')
    all_scales = [float(m.group(2)) for m in scale_pattern.finditer(mlir_text)]

    print(f"  • Model: Hidden Size={hidden_size}, Vocab Size={vocab_size}, Layers={num_layers}")
    print(f"  • KV Cache: Mode={kv_mode.upper()}, Max Sequence Length={max_seq_len}")
    print(f"  • Found {len(all_views)} memory views and {len(all_scales)} BitLinear scales")

    ctx = tenzo_runtime.ExecutionContext(
        hidden_size, num_q_heads, num_kv_heads, head_dim, num_layers, max_seq_len, "classic_tl1", kv_mode
    )

    v_idx = 1  # all_views[0] is embed_tokens.weight
    s_idx = 0

    for l in range(num_layers):
        # 1. in_norm (2560 f32)
        off, _ = all_views[v_idx]; v_idx += 1
        in_norm = raw_bytes[off : off + hidden_size * 4].view(np.float32)

        # 2. q_proj (2560x640 packed i8)
        off, _ = all_views[v_idx]; v_idx += 1
        q_raw = raw_bytes[off : off + 2560 * 640].reshape(2560, 640)
        q_tern = unpack_2bit_weights(q_raw, 2560, 2560)
        q_scale = all_scales[s_idx]; s_idx += 1

        # 3. k_proj (640x640 packed i8)
        off, _ = all_views[v_idx]; v_idx += 1
        k_raw = raw_bytes[off : off + 640 * 640].reshape(640, 640)
        k_tern = unpack_2bit_weights(k_raw, 640, 2560)
        k_scale = all_scales[s_idx]; s_idx += 1

        # 4. v_proj (640x640 packed i8)
        off, _ = all_views[v_idx]; v_idx += 1
        v_raw = raw_bytes[off : off + 640 * 640].reshape(640, 640)
        v_tern = unpack_2bit_weights(v_raw, 640, 2560)
        v_scale = all_scales[s_idx]; s_idx += 1

        # 5. attn_sub_norm (2560 f32)
        off, _ = all_views[v_idx]; v_idx += 1
        attn_sub_norm = raw_bytes[off : off + hidden_size * 4].view(np.float32)

        # 6. o_proj (2560x640 packed i8)
        off, _ = all_views[v_idx]; v_idx += 1
        o_raw = raw_bytes[off : off + 2560 * 640].reshape(2560, 640)
        o_tern = unpack_2bit_weights(o_raw, 2560, 2560)
        o_scale = all_scales[s_idx]; s_idx += 1

        # 7. post_norm (2560 f32)
        off, _ = all_views[v_idx]; v_idx += 1
        post_norm = raw_bytes[off : off + hidden_size * 4].view(np.float32)

        # 8. gate_proj (6912x640 packed i8)
        off, _ = all_views[v_idx]; v_idx += 1
        gate_raw = raw_bytes[off : off + ff_dim * 640].reshape(ff_dim, 640)
        gate_tern = unpack_2bit_weights(gate_raw, ff_dim, 2560)
        gate_scale = all_scales[s_idx]; s_idx += 1

        # 9. up_proj (6912x640 packed i8)
        off, _ = all_views[v_idx]; v_idx += 1
        up_raw = raw_bytes[off : off + ff_dim * 640].reshape(ff_dim, 640)
        up_tern = unpack_2bit_weights(up_raw, ff_dim, 2560)
        up_scale = all_scales[s_idx]; s_idx += 1

        # 10. ffn_sub_norm (6912 f32)
        off, _ = all_views[v_idx]; v_idx += 1
        ffn_sub_norm = raw_bytes[off : off + ff_dim * 4].view(np.float32)

        # 11. down_proj (2560x1728 packed i8)
        off, _ = all_views[v_idx]; v_idx += 1
        down_raw = raw_bytes[off : off + 2560 * 1728].reshape(2560, 1728)
        down_tern = unpack_2bit_weights(down_raw, 2560, ff_dim)
        down_scale = all_scales[s_idx]; s_idx += 1

        ctx.set_layer_full(
            l,
            q_tern, q_scale,
            k_tern, k_scale,
            v_tern, v_scale,
            o_tern, o_scale,
            gate_tern, gate_scale,
            up_tern, up_scale,
            down_tern, down_scale,
            in_norm,
            attn_sub_norm,
            post_norm,
            ffn_sub_norm
        )

    # Final norm
    if v_idx < len(all_views):
        off, _ = all_views[v_idx]
        final_norm = raw_bytes[off : off + hidden_size * 4].view(np.float32)
        ctx.set_final_norm(final_norm)

    print(f"✅ All {num_layers} layers packed into AVX2 registers successfully!")
    return ctx, embed_tokens_w


def sample_token(logits, past_tokens, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.15):
    logits = logits.copy()
    
    # Repetition penalty
    if repetition_penalty != 1.0 and len(past_tokens) > 0:
        for t in set(past_tokens):
            if logits[t] > 0:
                logits[t] /= repetition_penalty
            else:
                logits[t] *= repetition_penalty

    if temperature <= 0.0:
        return int(np.argmax(logits))

    # Top-K
    if top_k > 0 and top_k < len(logits):
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        min_top_k = np.min(logits[top_k_indices])
        logits[logits < min_top_k] = -1e9

    # Temperature scaling
    logits = logits / temperature
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)

    # Top-P
    if top_p > 0.0 and top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_index = np.searchsorted(cumulative_probs, top_p)
        sorted_probs[cutoff_index + 1:] = 0.0
        probs_sum = np.sum(sorted_probs)
        if probs_sum > 0:
            sorted_probs /= probs_sum
        return int(np.random.choice(sorted_indices, p=sorted_probs))

    return int(np.random.choice(len(probs), p=probs))


def generate_text(prompt, max_tokens=50, temp=0.7, repetition_penalty=1.15, kv_mode="int8_fused"):
    model_dir = os.path.join(PROJECT_ROOT, "tenzo-frontend", "export_output")
    mlir_path = os.path.join(model_dir, "model.mlir")
    weights_path = os.path.join(model_dir, "weights.bin")
    vocab_path = os.path.join(model_dir, "tokenizer.vocab")

    tokenizer = Tokenizer(vocab_path)
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [128000, 1, 2]

    safe_max_seq_len = max(8192, len(prompt_tokens) + max_tokens + 256)
    ctx, embed_w = parse_mlir_and_load_weights(mlir_path, weights_path, max_seq_len=safe_max_seq_len, kv_mode=kv_mode)

    print(f"\n💬 [Tenzo Engine] Output Stream: {prompt}", end="", flush=True)

    generated_tokens = []
    all_tokens = list(prompt_tokens)

    # 1. Prefill prompt tokens
    t_start = time.perf_counter()
    cur_x = None
    for tok in prompt_tokens:
        tok_embed = ctx.embedding_lookup(tok, embed_w)
        cur_x = ctx.forward_step(tok_embed)

    ttft = time.perf_counter() - t_start

    # 2. Autoregressive Decode
    t_decode_start = time.perf_counter()
    for step in range(max_tokens):
        logits = ctx.compute_logits(cur_x, embed_w).reshape(-1)
        next_tok = sample_token(logits, all_tokens, temperature=temp, repetition_penalty=repetition_penalty)
        
        all_tokens.append(next_tok)
        generated_tokens.append(next_tok)

        tok_str = tokenizer.decode([next_tok])
        if tok_str in ["<|endoftext|>", "<|eot_id|>", "</s>", "<eos>"]:
            print("\n[EOS reached]", flush=True)
            break

        print(tok_str, end="", flush=True)

        tok_embed = ctx.embedding_lookup(next_tok, embed_w)
        cur_x = ctx.forward_step(tok_embed)

    decode_time = time.perf_counter() - t_decode_start
    n_gen = len(generated_tokens)
    speed = n_gen / decode_time if decode_time > 0 else 0

    kv_desc = "INT8 Fused (4x compressed)" if kv_mode == "int8_fused" else "FP32 Standard"

    print(f"\n\n╔════════════════════════════════════════════════════════╗")
    print(f"║             Tenzo Engine Profiling Summary             ║")
    print(f"╠════════════════════════════════════════════════════════╣")
    print(f"║ KV Cache Mode:           {kv_desc:<29} ║")
    print(f"║ Prompt Tokens:           {len(prompt_tokens):<29} ║")
    print(f"║ Generated Tokens:        {n_gen:<29} ║")
    print(f"║ Time To First Token:     {ttft*1000:<26.2f} ms ║")
    print(f"║ Total Decode Time:       {decode_time*1000:<26.2f} ms ║")
    print(f"║ Decode Speed:            {speed:<26.2f} tok/sec ║")
    print(f"╚════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tenzo Ultra-Fast 1.58-bit BitNet Text Generator")
    parser.add_argument("-p", "--prompt", type=str, default="Explain the importance of compilers in computer science", help="Input prompt")
    parser.add_argument("-n", "--max-tokens", type=int, default=50, help="Maximum generated tokens")
    parser.add_argument("-t", "--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-r", "--repetition-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--kv-mode", type=str, default="int8_fused", choices=["fp32", "int8_fused"], help="KV Cache compression mode")
    args = parser.parse_args()

    generate_text(args.prompt, args.max_tokens, args.temp, args.repetition_penalty, args.kv_mode)
