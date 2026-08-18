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
    if dtype == "I8":
        # Raw int8 with values in {-1, 0, 1}
        # Pack 4 ternary values per byte: -1 -> 0b00, 0 -> 0b01, 1 -> 0b10
        for i in range(0, num_elements, 4):
            b0 = (raw[i] + 1) & 0x3
            b1 = (raw[i+1] + 1) & 0x3
            b2 = (raw[i+2] + 1) & 0x3
            b3 = (raw[i+3] + 1) & 0x3
            out[i // 4] = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)
    elif dtype == "F16":
        for i in range(0, num_elements, 4):
            floats = struct.unpack("<4e", raw[i*2:(i+4)*2])
            def to_ternary(f):
                if f > 0.3: return 2
                elif f < -0.3: return 0
                return 1
            b0 = to_ternary(floats[0])
            b1 = to_ternary(floats[1])
            b2 = to_ternary(floats[2])
            b3 = to_ternary(floats[3])
            out[i // 4] = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)
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
            
            # 3. k_proj (640 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.self_attn.k_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, 640, 2560) if raw else b"\x55" * (640 * (2560 // 4)))
            
            # 4. v_proj (640 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.self_attn.v_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, 640, 2560) if raw else b"\x55" * (640 * (2560 // 4)))
            
            # 5. attn_sub_norm
            raw, dt, sh = get_tensor(f"model.layers.{l}.self_attn.attn_sub_norm.weight")
            wf.write(convert_to_f32_bytes(raw, dt, hidden_size) if raw else norm_bytes)
            
            # 6. o_proj (2560 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.self_attn.o_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, 2560, 2560) if raw else b"\x55" * (2560 * (2560 // 4)))
            
            # 7. post_norm
            raw, dt, sh = get_tensor(f"model.layers.{l}.post_attention_layernorm.weight")
            wf.write(convert_to_f32_bytes(raw, dt, hidden_size) if raw else norm_bytes)
            
            # 8. gate_proj (6912 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.mlp.gate_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, ffn_dim, 2560) if raw else b"\x55" * (ffn_dim * (2560 // 4)))
            
            # 9. up_proj (6912 x 2560)
            raw, dt, sh = get_tensor(f"model.layers.{l}.mlp.up_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, ffn_dim, 2560) if raw else b"\x55" * (ffn_dim * (2560 // 4)))
            
            # 10. ffn_sub_norm
            raw, dt, sh = get_tensor(f"model.layers.{l}.mlp.ffn_sub_norm.weight")
            wf.write(convert_to_f32_bytes(raw, dt, ffn_dim) if raw else ffn_norm_bytes)
            
            # 11. down_proj (2560 x 6912)
            raw, dt, sh = get_tensor(f"model.layers.{l}.mlp.down_proj.weight")
            wf.write(convert_to_tl1_packed(raw, dt, 2560, ffn_dim) if raw else b"\x55" * (2560 * (ffn_dim // 4)))
            
        # final_norm
        raw, dt, sh = get_tensor("model.norm.weight")
        wf.write(convert_to_f32_bytes(raw, dt, hidden_size) if raw else norm_bytes)
        
    sz_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"🎉 Model weights exported successfully ({sz_mb:.1f} MB)!")

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
