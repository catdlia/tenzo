import struct
import math

def check():
    with open("tenzo-frontend/export_output/weights.bin", "rb") as f:
        # 1. Embeddings
        embed_raw = f.read(128256 * 2560 * 4)
        print(f"Read {len(embed_raw)} bytes of embeddings")
        
        # Check token 0, token 128000, etc.
        f0 = struct.unpack("<10f", embed_raw[:40])
        print("Token 0 first 10 floats:", f0)
        
        # Check token "Hello" (token id 9906 or 15339)
        # Let's find token for 'Hello' in tokenizer.vocab
        # 2. Layer 0
        in_norm = struct.unpack("<10f", f.read(2560 * 4)[:40])
        print("Layer 0 in_norm first 10 floats:", in_norm)
        
        q_w_raw = f.read(2560 * (2560 // 4))
        print(f"Layer 0 q_w read {len(q_w_raw)} bytes, first 16 bytes:", list(q_w_raw[:16]))

if __name__ == "__main__":
    check()
