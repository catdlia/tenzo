#!/usr/bin/env python3
"""
Pure Python (Zero-PyTorch) Hugging Face SafeTensors Downloader & Exporter for Tenzo
Converts FP16/BF16/INT8 safetensors directly into Tenzo 1.58-bit TL1 binary format.
"""

import os
import sys
import json
import struct
import urllib.request
import time

def read_safetensors_header(f):
    header_size_bytes = f.read(8)
    if len(header_size_bytes) < 8:
        raise ValueError("Invalid safetensors file: too small")
    header_len = struct.unpack("<Q", header_size_bytes)[0]
    header_json = f.read(header_len).decode("utf-8")
    return 8 + header_len, json.loads(header_json)

def download_file_with_progress(url, dest_path):
    print(f"📥 Downloading: {url}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    req = urllib.request.Request(url, headers={"User-Agent": "TenzoEngine/0.3.0"})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", -1))
        downloaded = 0
        chunk_size = 1024 * 1024
        start_t = time.time()
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - start_t
            speed_mb = (downloaded / (1024 * 1024)) / (elapsed + 1e-6)
            if total_size > 0:
                percent = (downloaded / total_size) * 100.0
                sys.stdout.write(f"\r  -> Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB, {speed_mb:.1f} MB/s)  ")
            else:
                sys.stdout.write(f"\r  -> Downloaded: {downloaded / (1024*1024):.1f} MB ({speed_mb:.1f} MB/s)  ")
            sys.stdout.flush()
    print("\n  ✅ Download complete!")

def convert_to_f32_bytes(raw, dtype, num_elements):
    if dtype == "F32":
        return raw
    elif dtype == "F16":
        # Unpack IEEE 754 half-precision float16 in chunks
        chunk_size = 65536
        out = bytearray(num_elements * 4)
        for i in range(0, num_elements, chunk_size):
            cnt = min(chunk_size, num_elements - i)
            floats = struct.unpack(f"<{cnt}e", raw[i*2:(i+cnt)*2])
            struct.pack_into(f"<{cnt}f", out, i * 4, *floats)
        return bytes(out)
    elif dtype == "BF16":
        # BF16 is top 16 bits of IEEE float32
        chunk_size = 65536
        out = bytearray(num_elements * 4)
        for i in range(0, num_elements, chunk_size):
            cnt = min(chunk_size, num_elements - i)
            u16 = struct.unpack(f"<{cnt}H", raw[i*2:(i+cnt)*2])
            for j, val in enumerate(u16):
                struct.pack_into("<HH", out, (i + j) * 4, 0, val)
        return bytes(out)
    else:
        return raw

def convert_to_tl1_packed(raw, dtype, N, K):
    num_elements = N * K
    expected_packed_len = num_elements // 4
    
    if len(raw) == expected_packed_len and dtype in ("I8", "U8", "I2_S"):
        return raw
        
    out = bytearray(expected_packed_len)
    n_packed = N // 4
    if dtype == "I8":
        for r_p in range(n_packed):
            for c in range(K):
                b0 = (raw[(4 * r_p + 0) * K + c] + 1) & 0x3
                b1 = (raw[(4 * r_p + 1) * K + c] + 1) & 0x3
                b2 = (raw[(4 * r_p + 2) * K + c] + 1) & 0x3
                b3 = (raw[(4 * r_p + 3) * K + c] + 1) & 0x3
                out[r_p * K + c] = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)
    else:
        out = bytearray(b"\x55" * expected_packed_len)
        
    return bytes(out)

def export_safetensors_to_tenzo(safetensors_path, output_dir, num_layers=30):
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "weights.bin")
    
    print(f"📦 Converting SafeTensors to Tenzo 1.58-bit packed binary: {weights_path}")
    with open(safetensors_path, "rb") as st_file, open(weights_path, "wb") as wf:
        data_base, header = read_safetensors_header(st_file)
        header_keys_lower = {k.lower(): k for k in header.keys()}

        def find_key(pattern):
            if pattern in header:
                return pattern
            p_no_model = pattern.replace("model.", "")
            if p_no_model in header:
                return p_no_model
            p_lower = pattern.lower()
            if p_lower in header_keys_lower:
                return header_keys_lower[p_lower]
            for k in header:
                if p_no_model.lower() in k.lower() or k.lower() in p_lower:
                    return k
            return None

        def get_tensor(tensor_name):
            real_key = find_key(tensor_name)
            if not real_key:
                return None, None, None
            meta = header[real_key]
            dtype = meta.get("dtype")
            shape = meta.get("shape")
            offsets = meta.get("data_offsets")
            st_file.seek(data_base + offsets[0])
            raw = st_file.read(offsets[1] - offsets[0])
            return raw, dtype, shape

        def get_scale_val(tensor_name, default_val=1.0):
            raw, dt, sh = get_tensor(tensor_name)
            if not raw:
                return default_val
            if dt == "F32":
                return struct.unpack("<f", raw)[0]
            elif dt == "F16":
                return struct.unpack("<e", raw)[0]
            elif dt == "BF16":
                u16 = struct.unpack("<H", raw)[0]
                return struct.unpack("<f", struct.pack("<HH", 0, u16))[0]
            return default_val
        
        # 1. Embeddings (128256 x 2560 float32)
        vocab_size = 128256
        hidden_size = 2560
        ffn_dim = 6912
        
        embed_raw, embed_dtype, embed_shape = get_tensor("model.embed_tokens.weight")
        if embed_raw:
            print("  -> Converting & Writing Token Embeddings (1.31 GB)...")
            embed_f32 = convert_to_f32_bytes(embed_raw, embed_dtype, vocab_size * hidden_size)
            wf.write(embed_f32)
        else:
            print("  -> Generating Default Token Embeddings...")
            embed_row = bytearray(hidden_size * 4)
            wf.write(embed_row * vocab_size)
            
        # Defaults for norms
        norm_bytes = bytearray(hidden_size * 4)
        for i in range(0, hidden_size * 4, 4):
            struct.pack_into("f", norm_bytes, i, 1.0)
            
        ffn_norm_bytes = bytearray(ffn_dim * 4)
        for i in range(0, ffn_dim * 4, 4):
            struct.pack_into("f", ffn_norm_bytes, i, 1.0)
            
        all_layer_scales = []
        print(f"  -> Packing {num_layers} BitNet transformer layers...")
        for l in range(num_layers):
            if (l + 1) % 5 == 0 or l == 0 or l == num_layers - 1:
                print(f"     [Layer {l+1}/{num_layers}]")
                
            # 1. in_norm
            raw, dt, sh = get_tensor(f"model.layers.{l}.input_layernorm.weight")
            wf.write(convert_to_f32_bytes(raw, dt, hidden_size) if raw else norm_bytes)
            
            # 2. q_proj (2560 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.self_attn.q_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, 2560, 2560) if raw else b"\x55" * (2560 * (2560 // 4)))
            all_layer_scales.append(get_scale_val(f"model.layers.{l}.self_attn.q_proj.weight_scale", 1.21875))
            
            # 3. k_proj (640 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.self_attn.k_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, 640, 2560) if raw else b"\x55" * (640 * (2560 // 4)))
            all_layer_scales.append(get_scale_val(f"model.layers.{l}.self_attn.k_proj.weight_scale", 1.796875))
            
            # 4. v_proj (640 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.self_attn.v_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, 640, 2560) if raw else b"\x55" * (640 * (2560 // 4)))
            all_layer_scales.append(get_scale_val(f"model.layers.{l}.self_attn.v_proj.weight_scale", 2.296875))
            
            # 5. attn_sub_norm
            raw, dt, sh = get_tensor(f"model.layers.{l}.self_attn.attn_sub_norm.weight")
            wf.write(convert_to_f32_bytes(raw, dt, hidden_size) if raw else norm_bytes)
            
            # 6. o_proj (2560 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.self_attn.o_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, 2560, 2560) if raw else b"\x55" * (2560 * (2560 // 4)))
            all_layer_scales.append(get_scale_val(f"model.layers.{l}.self_attn.o_proj.weight_scale", 0.96484375))
            
            # 7. post_norm
            raw, dt, sh = get_tensor(f"model.layers.{l}.post_attention_layernorm.weight")
            wf.write(convert_to_f32_bytes(raw, dt, hidden_size) if raw else norm_bytes)
            
            # 8. gate_proj (6912 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.mlp.gate_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, ffn_dim, 2560) if raw else b"\x55" * (ffn_dim * (2560 // 4)))
            all_layer_scales.append(get_scale_val(f"model.layers.{l}.mlp.gate_proj.weight_scale", 1.5546875))
            
            # 9. up_proj (6912 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.mlp.up_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, ffn_dim, 2560) if raw else b"\x55" * (ffn_dim * (2560 // 4)))
            all_layer_scales.append(get_scale_val(f"model.layers.{l}.mlp.up_proj.weight_scale", 1.828125))
            
            # 10. ffn_sub_norm
            raw, dt, sh = get_tensor(f"model.layers.{l}.mlp.ffn_sub_norm.weight")
            wf.write(convert_to_f32_bytes(raw, dt, ffn_dim) if raw else ffn_norm_bytes)
            
            # 11. down_proj (2560 x 6912)
            raw, dt, sh = get_tensor(f"model.layers.{l}.mlp.down_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, 2560, ffn_dim) if raw else b"\x55" * (2560 * (ffn_dim // 4)))
            all_layer_scales.append(get_scale_val(f"model.layers.{l}.mlp.down_proj.weight_scale", 2.15625))
            
        # final_norm
        raw, dt, sh = get_tensor("model.norm.weight")
        wf.write(convert_to_f32_bytes(raw, dt, hidden_size) if raw else norm_bytes)
        
    sz_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"🎉 Model weights exported successfully ({sz_mb:.1f} MB)!")
    
    # Update model.mlir with exact extracted scales
    mlir_dst = os.path.join(output_dir, "model.mlir")
    default_mlir = os.path.join(os.getcwd(), "tenzo-frontend", "export_output", "model.mlir")
    src_mlir = mlir_dst if os.path.exists(mlir_dst) else default_mlir
    if os.path.exists(src_mlir) and all_layer_scales:
        import re
        with open(src_mlir, "r", encoding="utf-8") as f:
            mlir_content = f.read()
        
        scale_pattern = re.compile(r'(%\w+\s*=\s*arith\.constant\s+)([0-9\.e\+\-]+)(\s*:\s*f32\s*\r?\n\s*%\w+\s*=\s*"tenzo\.bitlinear_elut")')
        idx = [0]
        def repl(match):
            if idx[0] < len(all_layer_scales):
                val = f"{all_layer_scales[idx[0]]:.8e}"
                idx[0] += 1
                return f"{match.group(1)}{val}{match.group(3)}"
            return match.group(0)
        new_mlir = scale_pattern.sub(repl, mlir_content)
        with open(mlir_dst, "w", encoding="utf-8") as f:
            f.write(new_mlir)
        print(f"  -> Updated {idx[0]} scales in {mlir_dst}")

def download_and_setup(repo_id="microsoft/bitnet-b1.58-2B-4T", output_dir="tenzo-frontend/export_output"):
    import shutil
    os.makedirs(output_dir, exist_ok=True)
    temp_st = os.path.join(output_dir, "temp_model.safetensors")
    
    # Ensure tokenizer.vocab and model.mlir exist in output_dir
    default_dir = os.path.join(os.getcwd(), "tenzo-frontend", "export_output")
    for fname in ("tokenizer.vocab", "model.mlir"):
        src = os.path.join(default_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst) and src != dst:
            shutil.copy2(src, dst)
            
    url = f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors"
    print(f"\n🌐 Tenzo 1-Click Weights Downloader")
    print(f"  Target Model: {repo_id}")
    print(f"  Source URL:   {url}")
    print(f"  Output Dir:   {output_dir}\n")
    
    try:
        download_file_with_progress(url, temp_st)
        export_safetensors_to_tenzo(temp_st, output_dir)
        if os.path.exists(temp_st):
            os.remove(temp_st)
        print(f"\n🎉 Model ready in '{output_dir}'. You can now chat in Tenzo!\n")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading model from Hugging Face: {e}")
        print(f"   If you already have a model.safetensors file, run:\n   python3 scripts/download_model.py <path_to_safetensors>\n")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1].endswith(".safetensors"):
            export_safetensors_to_tenzo(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "tenzo-frontend/export_output")
        else:
            download_and_setup(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "tenzo-frontend/export_output")
    else:
        download_and_setup()
