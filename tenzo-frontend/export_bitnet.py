import os
import sys
import argparse
import struct
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

def export_bitnet_model(model_name="microsoft/bitnet-b1.58-2B-4T", output_dir="export_output_bitnet", num_layers=2, quant_mode="i2_s"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Hey! Exporting {model_name} ({num_layers} layers, quant_mode={quant_mode})...")

    # 1. Download & Extract Tokenizer Vocabulary
    print("[Export BitNet] Extracting Tokenizer Vocabulary...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        vocab_dict = tokenizer.get_vocab()
    except Exception:
        try:
            from transformers import LlamaTokenizerFast
            tokenizer = LlamaTokenizerFast.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
            vocab_dict = tokenizer.get_vocab()
        except Exception:
            vocab_dict = {f"token_{i}": i for i in range(32000)}
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])

    vocab_path = os.path.join(output_dir, "tokenizer.vocab")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token_str, token_id in sorted_vocab:
            # Replace newlines/spaces for clean single-line reading
            cleaned_str = token_str.replace("\n", "\\n").replace("\r", "\\r")
            f.write(f"{token_id} {cleaned_str}\n")
    print(f"  ✅ Saved vocabulary ({len(sorted_vocab)} tokens) to {vocab_path}")

    # 2. Download Safetensors
    print("[Export BitNet] Loading Safetensors weights...")
    local_st = os.path.join(model_name, "model.safetensors") if os.path.isdir(model_name) else None
    if local_st and os.path.exists(local_st):
        safetensors_path = local_st
    else:
        safetensors_path = hf_hub_download(model_name, "model.safetensors")
    st = safe_open(safetensors_path, framework="pt")

    weights_bin_path = os.path.join(output_dir, "weights.bin")
    weights_file = open(weights_bin_path, "wb")
    current_offset = 0
    weight_metadata = {}

    def write_tensor(name, tensor_data, scale_val=1.0):
        nonlocal current_offset
        # Ensure contiguous bytes
        if tensor_data.dtype == torch.uint8:
            raw_bytes = tensor_data.numpy().tobytes()
            shape = list(tensor_data.shape)
            # Reconstruct original unquantized shape [out_features, in_features]
            orig_shape = [shape[0] * 4, shape[1]]
        elif tensor_data.dtype in (torch.float32, torch.float16, torch.bfloat16):
            raw_bytes = tensor_data.to(torch.float32).numpy().tobytes()
            orig_shape = list(tensor_data.shape)
        else:
            raw_bytes = tensor_data.numpy().tobytes()
            orig_shape = list(tensor_data.shape)

        size_bytes = len(raw_bytes)
        weights_file.write(raw_bytes)
        weight_metadata[name] = {
            "offset": current_offset,
            "size_bytes": size_bytes,
            "shape": orig_shape,
            "scale": scale_val
        }
        current_offset += size_bytes
        return weight_metadata[name]

    # Write Embeddings
    print("  -> Exporting Token Embeddings...")
    embed_w = st.get_tensor("model.embed_tokens.weight")
    embed_w_f32 = embed_w.to(torch.float32).numpy()
    scales = np.max(np.abs(embed_w_f32), axis=1) / 127.0
    scales[scales == 0] = 1.0
    q_weight = np.round(embed_w_f32 / scales[:, None]).astype(np.int8)
    
    write_tensor("model.embed_tokens.weight", torch.from_numpy(q_weight))
    write_tensor("model.embed_tokens.scales", torch.from_numpy(scales.astype(np.float32)))    # Read config if present
    config_path = os.path.join(model_name, "config.json") if os.path.isdir(model_name) else None
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            cfg = json.load(f)
        hidden_size = cfg.get("hidden_size", 2560)
        ff_dim = cfg.get("intermediate_size", 6912)
        n_kv_heads = cfg.get("num_key_value_heads", 5)
        n_q_heads = cfg.get("num_attention_heads", 20)
        head_dim = cfg.get("head_dim", hidden_size // n_q_heads)
    else:
        hidden_size = 2560
        ff_dim = 6912
        n_kv_heads = 5
        n_q_heads = 20
        head_dim = 128

    vocab_size = len(sorted_vocab)

    # Build dynamic function signature for num_layers KV caches
    kv_args_in = []
    kv_types_in = []
    for l in range(num_layers):
        kv_args_in.append(f"%arg{2 + 2*l}: tensor<1x{n_kv_heads}x1024x{head_dim}xf32>")
        kv_args_in.append(f"%arg{3 + 2*l}: tensor<1x{n_kv_heads}x1024x{head_dim}xf32>")
        kv_types_in.append(f"tensor<1x{n_kv_heads}x1024x{head_dim}xf32>")
        kv_types_in.append(f"tensor<1x{n_kv_heads}x1024x{head_dim}xf32>")
    
    seq_pos_arg_idx = 2 + 2 * num_layers
    seq_pos_var = f"%arg{seq_pos_arg_idx}"
    
    fn_inputs_str = f"%arg0: tensor<1x1xi32>, %arg1: memref<?xi8>, " + ", ".join(kv_args_in) + f", {seq_pos_var}: tensor<1xi32>"
    fn_outputs_str = f"tensor<1x1x{vocab_size}xf32>, " + ", ".join(kv_types_in)

    mlir_lines = [
        "// Generated by Tenzo BitNet 2B Exporter",
        "module {",
        f"  func.func @main({fn_inputs_str}) -> ({fn_outputs_str}) attributes {{llvm.emit_c_interface}} {{"
    ]

    var_counter = 0
    def get_var(t="tensor<*xf32>"):
        nonlocal var_counter
        v = f"%v{var_counter}"
        var_counter += 1
        return v

    c_w_off = get_var("index")
    mlir_lines.append(f'    {c_w_off} = arith.constant 0 : index')
    w_mem = get_var(f"memref<{vocab_size}x{hidden_size}xi8>")
    mlir_lines.append(f'    {w_mem} = memref.view %arg1[{c_w_off}][] : memref<?xi8> to memref<{vocab_size}x{hidden_size}xi8>')
    w_tens = get_var(f"tensor<{vocab_size}x{hidden_size}xi8>")
    mlir_lines.append(f'    {w_tens} = bufferization.to_tensor {w_mem} restrict : memref<{vocab_size}x{hidden_size}xi8> to tensor<{vocab_size}x{hidden_size}xi8>')

    scale_meta = weight_metadata["model.embed_tokens.scales"]
    c_scale_off = get_var("index")
    mlir_lines.append(f'    {c_scale_off} = arith.constant {scale_meta["offset"]} : index')
    scale_mem = get_var(f"memref<{vocab_size}xf32>")
    mlir_lines.append(f'    {scale_mem} = memref.view %arg1[{c_scale_off}][] : memref<?xi8> to memref<{vocab_size}xf32>')
    scale_tens = get_var(f"tensor<{vocab_size}xf32>")
    mlir_lines.append(f'    {scale_tens} = bufferization.to_tensor {scale_mem} restrict : memref<{vocab_size}xf32> to tensor<{vocab_size}xf32>')

    cur_x = get_var(f"tensor<1x1x{hidden_size}xf32>")
    mlir_lines.append(f'    {cur_x} = "tenzo.embedding"(%arg0, {w_tens}, {scale_tens}) : (tensor<1x1xi32>, tensor<{vocab_size}x{hidden_size}xi8>, tensor<{vocab_size}xf32>) -> tensor<1x1x{hidden_size}xf32>')

    updated_kv_list = []

    # Export Transformer Layers
    for layer_idx in range(num_layers):
        print(f"  -> Exporting Layer {layer_idx}/{num_layers}...")
        prefix = f"model.layers.{layer_idx}"
        curr_k = f"%arg{2 + 2*layer_idx}"
        curr_v = f"%arg{3 + 2*layer_idx}"

        # 1. Input RMSNorm
        in_norm_w = st.get_tensor(f"{prefix}.input_layernorm.weight")
        norm_meta = write_tensor(f"{prefix}.input_layernorm.weight", in_norm_w)
        
        c_off = get_var("index")
        mlir_lines.append(f'    {c_off} = arith.constant {norm_meta["offset"]} : index')
        w_mem = get_var(f"memref<{hidden_size}xf32>")
        mlir_lines.append(f'    {w_mem} = memref.view %arg1[{c_off}][] : memref<?xi8> to memref<{hidden_size}xf32>')
        w_norm_tens = get_var(f"tensor<{hidden_size}xf32>")
        mlir_lines.append(f'    {w_norm_tens} = bufferization.to_tensor {w_mem} restrict : memref<{hidden_size}xf32> to tensor<{hidden_size}xf32>')

        norm_x = get_var(f"tensor<1x1x{hidden_size}xf32>")
        mlir_lines.append(f'    {norm_x} = "tenzo.rmsnorm"({cur_x}, {w_norm_tens}) {{eps = 1.00000000e-05 : f32}} : (tensor<1x1x{hidden_size}xf32>, tensor<{hidden_size}xf32>) -> tensor<1x1x{hidden_size}xf32>')

        # 2. Self Attention Linear projections (q_proj, k_proj, v_proj, o_proj)
        def export_bitlinear(weight_key, input_var, out_features, in_features):
            w = st.get_tensor(weight_key)
            if f"{weight_key}_scale" in st.keys():
                s_tensor = st.get_tensor(f"{weight_key}_scale")
                scale_val = float(s_tensor.to(torch.float32).item()) if s_tensor.numel() == 1 else 1.0
            else:
                scale_val = float(w.to(torch.float32).abs().mean().item())
                if scale_val < 1e-8: scale_val = 1.0

            if w.dtype == torch.uint8:
                raw = w.contiguous().numpy().astype(np.uint8)          # [N/4, in_features]
                N_quarter, K = raw.shape
                tern = np.zeros((out_features, in_features), dtype=np.int8)
                for slot in range(4):
                    mask = 3 << (2 * slot)
                    val = (raw & mask) >> (2 * slot)
                    val_tern = val.astype(np.int8) - 1
                    tern[slot * N_quarter:(slot + 1) * N_quarter, :] = val_tern
            else:
                w_f32 = w.to(torch.float32)
                scaled_w = w_f32 / scale_val
                tern = torch.clamp(torch.round(scaled_w), -1.0, 1.0).to(torch.int8).numpy()

            if quant_mode == "tl1_pack":
                # Microsoft TL1 dense mapping: (w0+1)*3 + (w1+1)
                w_even = tern[:, 0::2]  # [N, K/2]
                w_odd = tern[:, 1::2]   # [N, K/2]
                
                # N = out_features
                bm = 64
                n_blocks = out_features // bm
                K_half = in_features // 2
                w_even = tern[:, 0::2]
                w_odd = tern[:, 1::2]
                idx_matrix = (w_even + 1) | ((w_odd + 1) << 2)
                
                idx_blocked = idx_matrix.reshape(n_blocks, bm, K_half)
                idx_T = idx_blocked.transpose(0, 2, 1)
                n_low_pack = idx_T[:, :, 0:32]
                n_high_pack = idx_T[:, :, 32:64]
                packed = (n_low_pack.astype(np.uint8)) | (n_high_pack.astype(np.uint8) << 4)
                
                packed_t = torch.from_numpy(packed.astype(np.int8))
                meta = write_tensor(weight_key, packed_t, scale_val)

                k_pack = K_half
                c_off = get_var("index")
                mlir_lines.append(f'    {c_off} = arith.constant {meta["offset"]} : index')
                w_mem = get_var(f"memref<{n_blocks}x{k_pack}x32xi8>")
                mlir_lines.append(f'    {w_mem} = memref.view %arg1[{c_off}][] : memref<?xi8> to memref<{n_blocks}x{k_pack}x32xi8>')
                w_tens = get_var(f"tensor<{n_blocks}x{k_pack}x32xi8>")
                mlir_lines.append(f'    {w_tens} = bufferization.to_tensor {w_mem} restrict : memref<{n_blocks}x{k_pack}x32xi8> to tensor<{n_blocks}x{k_pack}x32xi8>')

                sc_var = get_var("f32")
                mlir_lines.append(f'    {sc_var} = arith.constant {scale_val:.8e} : f32')

                out_v = get_var(f"tensor<1x1x{out_features}xf32>")
                mlir_lines.append(
                    f'    {out_v} = "tenzo.bitlinear_tl1_pack"({input_var}, {w_tens}, {sc_var}) '
                    f'{{bit_width = 2 : i32, quant_scheme = "ternary"}} : '
                    f'(tensor<1x1x{in_features}xf32>, tensor<{n_blocks}x{k_pack}x32xi8>, f32) -> '
                    f'tensor<1x1x{out_features}xf32>'
                )
                return out_v

            elif quant_mode == "tl1":
                # TL1 (g=2) N-major outer-product packing with blocks of 64
                w_even = tern[:, 0::2]  # [N, K/2]
                w_odd = tern[:, 1::2]   # [N, K/2]
                # Bitwise packing: w_even+1 in bits 0-1, w_odd+1 in bits 2-3
                idx_matrix = (w_even + 1) | ((w_odd + 1) << 2)  # [N, K/2] uint8, values 0..10
                
                N, K_half = idx_matrix.shape
                assert N % 64 == 0, "N must be divisible by 64"
                n_blocks = N // 64
                
                # Reshape N into blocks of 64, then transpose K to be the middle dimension
                idx_blocked = idx_matrix.reshape(n_blocks, 64, K_half)
                idx_T = idx_blocked.transpose(0, 2, 1)  # [N/64, K/2, 64]
                
                # Pack N pairs into 1 byte: channels 0..31 in lower nibble, 32..63 in upper nibble
                n_low_pack = idx_T[:, :, 0:32]    # [N/64, K/2, 32]
                n_high_pack = idx_T[:, :, 32:64] # [N/64, K/2, 32]
                packed = (n_low_pack.astype(np.uint8)) | (n_high_pack.astype(np.uint8) << 4)
                
                packed_t = torch.from_numpy(packed.astype(np.int8))
                meta = write_tensor(weight_key, packed_t, scale_val)

                k_pack = K_half
                c_off = get_var("index")
                mlir_lines.append(f'    {c_off} = arith.constant {meta["offset"]} : index')
                w_mem = get_var(f"memref<{n_blocks}x{k_pack}x32xi8>")
                mlir_lines.append(f'    {w_mem} = memref.view %arg1[{c_off}][] : memref<?xi8> to memref<{n_blocks}x{k_pack}x32xi8>')
                w_tens = get_var(f"tensor<{n_blocks}x{k_pack}x32xi8>")
                mlir_lines.append(f'    {w_tens} = bufferization.to_tensor {w_mem} restrict : memref<{n_blocks}x{k_pack}x32xi8> to tensor<{n_blocks}x{k_pack}x32xi8>')

                sc_var = get_var("f32")
                mlir_lines.append(f'    {sc_var} = arith.constant {scale_val:.8e} : f32')

                out_v = get_var(f"tensor<1x1x{out_features}xf32>")
                mlir_lines.append(
                    f'    {out_v} = "tenzo.bitlinear_tl1"({input_var}, {w_tens}, {sc_var}) '
                    f'{{bit_width = 2 : i32, quant_scheme = "ternary"}} : '
                    f'(tensor<1x1x{in_features}xf32>, tensor<{n_blocks}x{k_pack}x32xi8>, f32) -> '
                    f'tensor<1x1x{out_features}xf32>'
                )
                return out_v

            elif quant_mode == "i4_s":
                # INT4: 2 4-bit nibbles per byte along K axis [N, K/2]
                w_f32 = tern.astype(np.float32)
                amax = np.max(np.abs(w_f32), axis=1) / 7.0
                amax[amax == 0] = 1.0
                scale_val = np.mean(amax)
                w_i4 = np.clip(np.round(w_f32 / amax[:, None]), -8, 7).astype(np.int8)
                w_u4 = (w_i4 & 0x0F).astype(np.uint8)
                c0 = w_u4[:, 0::2]
                c1 = w_u4[:, 1::2]
                packed = c0 | (c1 << 4)

                packed_t = torch.from_numpy(packed.astype(np.int8))
                meta = write_tensor(weight_key, packed_t, scale_val)

                k_pack = in_features // 2
                c_off = get_var("index")
                mlir_lines.append(f'    {c_off} = arith.constant {meta["offset"]} : index')
                w_mem = get_var(f"memref<{out_features}x{k_pack}xi8>")
                mlir_lines.append(f'    {w_mem} = memref.view %arg1[{c_off}][] : memref<?xi8> to memref<{out_features}x{k_pack}xi8>')
                w_tens = get_var(f"tensor<{out_features}x{k_pack}xi8>")
                mlir_lines.append(f'    {w_tens} = bufferization.to_tensor {w_mem} restrict : memref<{out_features}x{k_pack}xi8> to tensor<{out_features}x{k_pack}xi8>')

                sc_var = get_var("f32")
                mlir_lines.append(f'    {sc_var} = arith.constant {scale_val:.8e} : f32')

                out_v = get_var(f"tensor<1x1x{out_features}xf32>")
                mlir_lines.append(
                    f'    {out_v} = "tenzo.bitlinear_int4"({input_var}, {w_tens}, {sc_var}) '
                    f'{{bit_width = 4 : i32, quant_scheme = "int4"}} : '
                    f'(tensor<1x1x{in_features}xf32>, tensor<{out_features}x{k_pack}xi8>, f32) -> '
                    f'tensor<1x1x{out_features}xf32>'
                )
                return out_v

            elif quant_mode == "i3_s":
                # INT3: 8 3-bit weights into 3 bytes along K axis [N, 3*K/8]
                w_f32 = tern.astype(np.float32)
                amax = np.max(np.abs(w_f32), axis=1) / 3.5
                amax[amax == 0] = 1.0
                scale_val = np.mean(amax)
                w_i3 = np.clip(np.round(w_f32 / amax[:, None]) + 4, 0, 7).astype(np.uint8)
                
                N, K = w_i3.shape
                packed = np.zeros((N, (K // 8) * 3), dtype=np.uint8)
                for b in range(K // 8):
                    w0, w1, w2, w3 = w_i3[:, b*8+0], w_i3[:, b*8+1], w_i3[:, b*8+2], w_i3[:, b*8+3]
                    w4, w5, w6, w7 = w_i3[:, b*8+4], w_i3[:, b*8+5], w_i3[:, b*8+6], w_i3[:, b*8+7]
                    packed[:, b*3+0] = w0 | (w1 << 3) | ((w2 & 0x03) << 6)
                    packed[:, b*3+1] = (w2 >> 2) | (w3 << 1) | (w4 << 4) | ((w5 & 0x01) << 7)
                    packed[:, b*3+2] = (w5 >> 1) | (w6 << 2) | (w7 << 5)

                packed_t = torch.from_numpy(packed.astype(np.int8))
                meta = write_tensor(weight_key, packed_t, scale_val)

                k_pack = (in_features // 8) * 3
                c_off = get_var("index")
                mlir_lines.append(f'    {c_off} = arith.constant {meta["offset"]} : index')
                w_mem = get_var(f"memref<{out_features}x{k_pack}xi8>")
                mlir_lines.append(f'    {w_mem} = memref.view %arg1[{c_off}][] : memref<?xi8> to memref<{out_features}x{k_pack}xi8>')
                w_tens = get_var(f"tensor<{out_features}x{k_pack}xi8>")
                mlir_lines.append(f'    {w_tens} = bufferization.to_tensor {w_mem} restrict : memref<{out_features}x{k_pack}xi8> to tensor<{out_features}x{k_pack}xi8>')

                sc_var = get_var("f32")
                mlir_lines.append(f'    {sc_var} = arith.constant {scale_val:.8e} : f32')

                out_v = get_var(f"tensor<1x1x{out_features}xf32>")
                mlir_lines.append(
                    f'    {out_v} = "tenzo.bitlinear_int3"({input_var}, {w_tens}, {sc_var}) '
                    f'{{bit_width = 3 : i32, quant_scheme = "int3"}} : '
                    f'(tensor<1x1x{in_features}xf32>, tensor<{out_features}x{k_pack}xi8>, f32) -> '
                    f'tensor<1x1x{out_features}xf32>'
                )
                return out_v

            else:
                # Default i2_s: Repack 4 contiguous ternary weights per byte along K axis
                codes = (tern + 1).astype(np.uint8)                    # {-1,0,1} -> {0,1,2}
                c0 = codes[:, 0::4]                                    # cols 0, 4, 8, ...
                c1 = codes[:, 1::4]                                    # cols 1, 5, 9, ...
                c2 = codes[:, 2::4]                                    # cols 2, 6, 10, ...
                c3 = codes[:, 3::4]                                    # cols 3, 7, 11, ...
                packed = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)       # [N, K/4] uint8

                packed_t = torch.from_numpy(packed.astype(np.int8))
                meta = write_tensor(weight_key, packed_t, scale_val)

                k_pack = in_features // 4
                c_off = get_var("index")
                mlir_lines.append(f'    {c_off} = arith.constant {meta["offset"]} : index')
                w_mem = get_var(f"memref<{out_features}x{k_pack}xi8>")
                mlir_lines.append(f'    {w_mem} = memref.view %arg1[{c_off}][] : memref<?xi8> to memref<{out_features}x{k_pack}xi8>')
                w_tens = get_var(f"tensor<{out_features}x{k_pack}xi8>")
                mlir_lines.append(f'    {w_tens} = bufferization.to_tensor {w_mem} restrict : memref<{out_features}x{k_pack}xi8> to tensor<{out_features}x{k_pack}xi8>')

                sc_var = get_var("f32")
                mlir_lines.append(f'    {sc_var} = arith.constant {scale_val:.8e} : f32')

                out_v = get_var(f"tensor<1x1x{out_features}xf32>")
                mlir_lines.append(
                    f'    {out_v} = "tenzo.bitlinear_elut"({input_var}, {w_tens}, {sc_var}) '
                    f'{{bit_width = 2 : i32, quant_scheme = "ternary"}} : '
                    f'(tensor<1x1x{in_features}xf32>, tensor<{out_features}x{k_pack}xi8>, f32) -> '
                    f'tensor<1x1x{out_features}xf32>'
                )
                return out_v

        q_out = export_bitlinear(f"{prefix}.self_attn.q_proj.weight", norm_x, 2560, 2560)
        k_out = export_bitlinear(f"{prefix}.self_attn.k_proj.weight", norm_x, 640, 2560)
        v_out = export_bitlinear(f"{prefix}.self_attn.v_proj.weight", norm_x, 640, 2560)

        # 4D Reshape for Multi-Head Attention (Q: 20 heads x 128, K/V: 5 heads x 128)
        q_exp = get_var("tensor<1x1x20x128xf32>")
        mlir_lines.append(f'    {q_exp} = tensor.expand_shape {q_out} [[0], [1], [2, 3]] output_shape [1, 1, 20, 128] : tensor<1x1x2560xf32> into tensor<1x1x20x128xf32>')
        empty_q = get_var("tensor<1x20x1x128xf32>")
        mlir_lines.append(f'    {empty_q} = tensor.empty() : tensor<1x20x1x128xf32>')
        q_4d = get_var("tensor<1x20x1x128xf32>")
        mlir_lines.append(f'    {q_4d} = linalg.transpose ins({q_exp} : tensor<1x1x20x128xf32>) outs({empty_q} : tensor<1x20x1x128xf32>) permutation = [0, 2, 1, 3]')

        k_exp = get_var("tensor<1x1x5x128xf32>")
        mlir_lines.append(f'    {k_exp} = tensor.expand_shape {k_out} [[0], [1], [2, 3]] output_shape [1, 1, 5, 128] : tensor<1x1x640xf32> into tensor<1x1x5x128xf32>')
        empty_k = get_var("tensor<1x5x1x128xf32>")
        mlir_lines.append(f'    {empty_k} = tensor.empty() : tensor<1x5x1x128xf32>')
        k_4d = get_var("tensor<1x5x1x128xf32>")
        mlir_lines.append(f'    {k_4d} = linalg.transpose ins({k_exp} : tensor<1x1x5x128xf32>) outs({empty_k} : tensor<1x5x1x128xf32>) permutation = [0, 2, 1, 3]')

        v_exp = get_var("tensor<1x1x5x128xf32>")
        mlir_lines.append(f'    {v_exp} = tensor.expand_shape {v_out} [[0], [1], [2, 3]] output_shape [1, 1, 5, 128] : tensor<1x1x640xf32> into tensor<1x1x5x128xf32>')
        empty_v = get_var("tensor<1x5x1x128xf32>")
        mlir_lines.append(f'    {empty_v} = tensor.empty() : tensor<1x5x1x128xf32>')
        v_4d = get_var("tensor<1x5x1x128xf32>")
        mlir_lines.append(f'    {v_4d} = linalg.transpose ins({v_exp} : tensor<1x1x5x128xf32>) outs({empty_v} : tensor<1x5x1x128xf32>) permutation = [0, 2, 1, 3]')

        # Apply RoPE in 4D
        rope_q = get_var("tensor<1x20x1x128xf32>")
        mlir_lines.append(f'    {rope_q} = "tenzo.rope"({q_4d}, {seq_pos_var}) : (tensor<1x20x1x128xf32>, tensor<1xi32>) -> tensor<1x20x1x128xf32>')
        rope_k = get_var("tensor<1x5x1x128xf32>")
        mlir_lines.append(f'    {rope_k} = "tenzo.rope"({k_4d}, {seq_pos_var}) : (tensor<1x5x1x128xf32>, tensor<1xi32>) -> tensor<1x5x1x128xf32>')

        # KV Cache Update
        up_k = get_var("tensor<1x5x1024x128xf32>")
        up_v = get_var("tensor<1x5x1024x128xf32>")
        mlir_lines.append(
            f'    {up_k}, {up_v} = "tenzo.kv_cache_update"({curr_k}, {curr_v}, {rope_k}, {v_4d}, {seq_pos_var}) : '
            f'(tensor<1x5x1024x128xf32>, tensor<1x5x1024x128xf32>, tensor<1x5x1x128xf32>, tensor<1x5x1x128xf32>, tensor<1xi32>) -> (tensor<1x5x1024x128xf32>, tensor<1x5x1024x128xf32>)'
        )
        updated_kv_list.append(up_k)
        updated_kv_list.append(up_v)

        # Attention
        attn_4d = get_var("tensor<1x20x1x128xf32>")
        mlir_lines.append(
            f'    {attn_4d} = "tenzo.attention"({rope_q}, {up_k}, {up_v}, {seq_pos_var}) : '
            f'(tensor<1x20x1x128xf32>, tensor<1x5x1024x128xf32>, tensor<1x5x1024x128xf32>, tensor<1xi32>) -> tensor<1x20x1x128xf32>'
        )

        empty_attn = get_var("tensor<1x1x20x128xf32>")
        mlir_lines.append(f'    {empty_attn} = tensor.empty() : tensor<1x1x20x128xf32>')
        attn_trans = get_var("tensor<1x1x20x128xf32>")
        mlir_lines.append(f'    {attn_trans} = linalg.transpose ins({attn_4d} : tensor<1x20x1x128xf32>) outs({empty_attn} : tensor<1x1x20x128xf32>) permutation = [0, 2, 1, 3]')

        attn_3d = get_var("tensor<1x1x2560xf32>")
        mlir_lines.append(f'    {attn_3d} = tensor.collapse_shape {attn_trans} [[0], [1], [2, 3]] : tensor<1x1x20x128xf32> into tensor<1x1x2560xf32>')

        # Helper for tensor retrieval
        def get_tensor_safe(keys, default_shape):
            for k in keys:
                if k in st.keys():
                    return st.get_tensor(k)
            return torch.ones(default_shape, dtype=torch.float32)

        # Attention Sub-LayerNorm (attn_sub_norm / inner_attn_ln)
        attn_sub_w = get_tensor_safe([f"{prefix}.self_attn.attn_sub_norm.weight", f"{prefix}.self_attn.inner_attn_ln.weight"], [hidden_size])
        attn_sub_meta = write_tensor(f"{prefix}.self_attn.attn_sub_norm.weight", attn_sub_w)
        attn_sub_off = get_var("index")
        mlir_lines.append(f'    {attn_sub_off} = arith.constant {attn_sub_meta["offset"]} : index')
        w_attn_sub_mem = get_var(f"memref<{hidden_size}xf32>")
        mlir_lines.append(f'    {w_attn_sub_mem} = memref.view %arg1[{attn_sub_off}][] : memref<?xi8> to memref<{hidden_size}xf32>')
        w_attn_sub_tens = get_var(f"tensor<{hidden_size}xf32>")
        mlir_lines.append(f'    {w_attn_sub_tens} = bufferization.to_tensor {w_attn_sub_mem} restrict : memref<{hidden_size}xf32> to tensor<{hidden_size}xf32>')

        attn_sub_out = get_var(f"tensor<1x1x{hidden_size}xf32>")
        mlir_lines.append(f'    {attn_sub_out} = "tenzo.rmsnorm"({attn_3d}, {w_attn_sub_tens}) {{eps = 1.00000000e-05 : f32}} : (tensor<1x1x{hidden_size}xf32>, tensor<{hidden_size}xf32>) -> tensor<1x1x{hidden_size}xf32>')

        o_out = export_bitlinear(f"{prefix}.self_attn.o_proj.weight", attn_sub_out, hidden_size, hidden_size)
        h1 = get_var(f"tensor<1x1x{hidden_size}xf32>")
        mlir_lines.append(f'    {h1} = "tenzo.add"({cur_x}, {o_out}) : (tensor<1x1x{hidden_size}xf32>, tensor<1x1x{hidden_size}xf32>) -> tensor<1x1x{hidden_size}xf32>')

        # 3. Post Attention RMSNorm
        post_norm_w = get_tensor_safe([f"{prefix}.post_attention_layernorm.weight", f"{prefix}.post_attn_norm.weight"], [hidden_size])
        p_norm_meta = write_tensor(f"{prefix}.post_attention_layernorm.weight", post_norm_w)
        c_off = get_var("index")
        mlir_lines.append(f'    {c_off} = arith.constant {p_norm_meta["offset"]} : index')
        w_mem = get_var(f"memref<{hidden_size}xf32>")
        mlir_lines.append(f'    {w_mem} = memref.view %arg1[{c_off}][] : memref<?xi8> to memref<{hidden_size}xf32>')
        w_pnorm_tens = get_var(f"tensor<{hidden_size}xf32>")
        mlir_lines.append(f'    {w_pnorm_tens} = bufferization.to_tensor {w_mem} restrict : memref<{hidden_size}xf32> to tensor<{hidden_size}xf32>')

        post_norm_h = get_var(f"tensor<1x1x{hidden_size}xf32>")
        mlir_lines.append(f'    {post_norm_h} = "tenzo.rmsnorm"({h1}, {w_pnorm_tens}) {{eps = 1.00000000e-05 : f32}} : (tensor<1x1x{hidden_size}xf32>, tensor<{hidden_size}xf32>) -> tensor<1x1x{hidden_size}xf32>')

        # 4. FFN: relu2(gate_proj(x)) * up_proj(x), then ffn_sub_norm, then down_proj
        gate_out = export_bitlinear(f"{prefix}.mlp.gate_proj.weight", post_norm_h, ff_dim, hidden_size)
        relu2_out = get_var(f"tensor<1x1x{ff_dim}xf32>")
        mlir_lines.append(f'    {relu2_out} = "tenzo.relu2"({gate_out}) : (tensor<1x1x{ff_dim}xf32>) -> tensor<1x1x{ff_dim}xf32>')

        up_out = export_bitlinear(f"{prefix}.mlp.up_proj.weight", post_norm_h, ff_dim, hidden_size)

        act_mult_out = get_var(f"tensor<1x1x{ff_dim}xf32>")
        mlir_lines.append(f'    {act_mult_out} = "tenzo.mul"({relu2_out}, {up_out}) : (tensor<1x1x{ff_dim}xf32>, tensor<1x1x{ff_dim}xf32>) -> tensor<1x1x{ff_dim}xf32>')

        ffn_sub_w = get_tensor_safe([f"{prefix}.mlp.ffn_sub_norm.weight", f"{prefix}.mlp.ffn_layernorm.weight"], [ff_dim])
        ffn_sub_meta = write_tensor(f"{prefix}.mlp.ffn_sub_norm.weight", ffn_sub_w)
        ffn_sub_off = get_var("index")
        mlir_lines.append(f'    {ffn_sub_off} = arith.constant {ffn_sub_meta["offset"]} : index')
        w_ffn_sub_mem = get_var(f"memref<{ff_dim}xf32>")
        mlir_lines.append(f'    {w_ffn_sub_mem} = memref.view %arg1[{ffn_sub_off}][] : memref<?xi8> to memref<{ff_dim}xf32>')
        w_ffn_sub_tens = get_var(f"tensor<{ff_dim}xf32>")
        mlir_lines.append(f'    {w_ffn_sub_tens} = bufferization.to_tensor {w_ffn_sub_mem} restrict : memref<{ff_dim}xf32> to tensor<{ff_dim}xf32>')

        ffn_sub_out = get_var(f"tensor<1x1x{ff_dim}xf32>")
        mlir_lines.append(f'    {ffn_sub_out} = "tenzo.rmsnorm"({act_mult_out}, {w_ffn_sub_tens}) {{eps = 1.00000000e-05 : f32}} : (tensor<1x1x{ff_dim}xf32>, tensor<{ff_dim}xf32>) -> tensor<1x1x{ff_dim}xf32>')

        down_out = export_bitlinear(f"{prefix}.mlp.down_proj.weight", ffn_sub_out, hidden_size, ff_dim)
        cur_x = get_var(f"tensor<1x1x{hidden_size}xf32>")
        mlir_lines.append(f'    {cur_x} = "tenzo.add"({h1}, {down_out}) : (tensor<1x1x{hidden_size}xf32>, tensor<1x1x{hidden_size}xf32>) -> tensor<1x1x{hidden_size}xf32>')

    # Final RMSNorm (model.norm) before LM Head
    final_norm_w = get_tensor_safe(["model.norm.weight", "model.layernorm.weight"], [hidden_size])
    final_norm_meta = write_tensor("model.norm.weight", final_norm_w)
    final_norm_off = get_var("index")
    mlir_lines.append(f'    {final_norm_off} = arith.constant {final_norm_meta["offset"]} : index')
    final_norm_mem = get_var(f"memref<{hidden_size}xf32>")
    mlir_lines.append(f'    {final_norm_mem} = memref.view %arg1[{final_norm_off}][] : memref<?xi8> to memref<{hidden_size}xf32>')
    final_norm_w_tens = get_var(f"tensor<{hidden_size}xf32>")
    mlir_lines.append(f'    {final_norm_w_tens} = bufferization.to_tensor {final_norm_mem} restrict : memref<{hidden_size}xf32> to tensor<{hidden_size}xf32>')

    final_norm_x = get_var(f"tensor<1x1x{hidden_size}xf32>")
    mlir_lines.append(f'    {final_norm_x} = "tenzo.rmsnorm"({cur_x}, {final_norm_w_tens}) {{eps = 1.00000000e-05 : f32}} : (tensor<1x1x{hidden_size}xf32>, tensor<{hidden_size}xf32>) -> tensor<1x1x{hidden_size}xf32>')

    # Final LM Head (tied with embed_tokens)
    lm_head_out = get_var(f"tensor<1x1x{vocab_size}xf32>")
    mlir_lines.append(f'    {lm_head_out} = "tenzo.matmul_q8"({final_norm_x}, {w_tens}, {scale_tens}) : (tensor<1x1x{hidden_size}xf32>, tensor<{vocab_size}x{hidden_size}xi8>, tensor<{vocab_size}xf32>) -> tensor<1x1x{vocab_size}xf32>')
    
    ret_vars_str = f"{lm_head_out}, " + ", ".join(updated_kv_list)
    ret_types_str = f"tensor<1x1x{vocab_size}xf32>, " + ", ".join(kv_types_in)
    mlir_lines.append(f'    return {ret_vars_str} : {ret_types_str}')
    mlir_lines.append("  }")
    mlir_lines.append("}")

    weights_file.close()

    mlir_path = os.path.join(output_dir, "model.mlir")
    with open(mlir_path, "w") as f:
        f.write("\n".join(mlir_lines))

    print(f"\n🎉 Successfully exported BitNet 2B model to '{output_dir}'")
    print(f"  - Model MLIR: {mlir_path}")
    print(f"  - Binary Weights: {weights_bin_path} ({current_offset} bytes)")
    print(f"  - Vocabulary: {vocab_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Microsoft BitNet 2B model to Tenzo MLIR format")
    parser.add_argument("--model-name", type=str, default="microsoft/bitnet-b1.58-2B-4T", help="Hugging Face model ID")
    parser.add_argument("--output-dir", type=str, default="export_output_bitnet", help="Directory to save exported files")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer layers to export (default: 2)")
    parser.add_argument("--quant-mode", type=str, choices=["i2_s", "i4_s", "i3_s", "tl1", "tl1_pack"], default="i2_s", help="Quantization mode: i2_s (1.58b), i4_s (INT4), i3_s (INT3), tl1, tl1_pack")
    args = parser.parse_args()

    export_bitnet_model(model_name=args.model_name, output_dir=args.output_dir, num_layers=args.num_layers, quant_mode=args.quant_mode)
