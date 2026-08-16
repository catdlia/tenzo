import numpy as np
import sys
import os

# Add the build directory to sys.path so we can import the pybind11 module
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cmake-build-debug"))

try:
    import tenzo_runtime
    HAS_TENZO = True
except ImportError:
    print("⚠️  Warning: tenzo_runtime module not found. Did you run `make build-local`?")
    HAS_TENZO = False

class KVCacheManager:
    def __init__(self, max_batch, max_seq, num_heads, head_dim, dtype=np.float32):
        self.cache_k = np.zeros((max_batch, num_heads, max_seq, head_dim), dtype=dtype)
        self.cache_v = np.zeros((max_batch, num_heads, max_seq, head_dim), dtype=dtype)
        self.current_seq_len = 0

    def update(self, new_k, new_v):
        batch, heads, seq_len, dim = new_k.shape
        self.cache_k[:, :, self.current_seq_len:self.current_seq_len+seq_len, :] = new_k
        self.cache_v[:, :, self.current_seq_len:self.current_seq_len+seq_len, :] = new_v
        self.current_seq_len += seq_len
        return (
            self.cache_k[:, :, :self.current_seq_len, :],
            self.cache_v[:, :, :self.current_seq_len, :]
        )

def dispatch_linear(x, weight, bias=None, quant_scheme="fp32"):
    """
    Universal dispatch for Linear operations.
    If a scheme lacks a C++ MLIR kernel, fallback to numpy simulation.
    """
    if quant_scheme == "fp32" or quant_scheme == "fp16":
        res = np.dot(x, weight.T)
        if bias is not None:
            res += bias
        return res
    elif quant_scheme == "classic_tl1" or quant_scheme == "classic_tl2":
        if HAS_TENZO:
            print(f"[Tenzo C++ Call] Executing native {quant_scheme} BitLinear kernel")
            # Call the C++ extension
            res_np = tenzo_runtime.dispatch_linear_cxx(x, weight)
            if bias is not None:
                res_np += bias
            return res_np
        else:
            print(f"[Numpy Fallback] Simulating {quant_scheme} BitLinear kernel (tenzo_runtime missing)")
            res = np.dot(x, weight.T)
            if bias is not None:
                res += bias
            return res
    elif quant_scheme == "int4":
        print("[Numpy Fallback] Simulating INT4 Linear")
        res = np.dot(x, weight.T)
        if bias is not None:
            res += bias
        return res
    else:
        raise ValueError(f"Unknown quant_scheme: {quant_scheme}")

def generate_token(input_ids, kv_cache, config):
    print("--- Decoding Step ---")
    x = np.random.randn(1, 1, config["hidden_size"]).astype(np.float32)
    
    # Q, K, V Projections
    q_w = np.random.randn(config["hidden_size"], config["hidden_size"]).astype(np.float32)
    k_w = np.random.randn(config["hidden_size"], config["hidden_size"]).astype(np.float32)
    v_w = np.random.randn(config["hidden_size"], config["hidden_size"]).astype(np.float32)
    
    q = dispatch_linear(x, q_w, quant_scheme=config["q_scheme"])
    k = dispatch_linear(x, k_w, quant_scheme=config["q_scheme"])
    v = dispatch_linear(x, v_w, quant_scheme=config["q_scheme"])
    
    # Reshape for Attention
    q = q.reshape(1, 1, config["num_heads"], config["head_dim"]).transpose(0, 2, 1, 3)
    k = k.reshape(1, 1, config["num_heads"], config["head_dim"]).transpose(0, 2, 1, 3)
    v = v.reshape(1, 1, config["num_heads"], config["head_dim"]).transpose(0, 2, 1, 3)
    
    # Update KV Cache
    past_k, past_v = kv_cache.update(k, v)
    
    # Attention
    attn_weights = np.matmul(q, past_k.transpose(0, 1, 3, 2)) / (config["head_dim"] ** 0.5)
    # Simple softmax
    attn_weights = np.exp(attn_weights - np.max(attn_weights, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    
    attn_output = np.matmul(attn_weights, past_v)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(1, 1, config["hidden_size"])
    
    # Output Projection
    out_w = np.random.randn(config["hidden_size"], config["hidden_size"]).astype(np.float32)
    output = dispatch_linear(attn_output, out_w, quant_scheme=config["q_scheme"])
    
    return output

if __name__ == "__main__":
    print("🚀 Starting Universal Inference Mockup")
    config = {
        "hidden_size": 128,
        "num_heads": 4,
        "head_dim": 32,
        "q_scheme": "classic_tl1" # Test different schemes: fp32, classic_tl1, int4
    }
    
    kv_cache = KVCacheManager(max_batch=1, max_seq=1024, num_heads=config["num_heads"], head_dim=config["head_dim"])
    
    input_ids = np.array([[1, 2, 3]])
    
    # Prefill mock
    print("--- Prefill Step ---")
    kv_cache.update(
        np.random.randn(1, config["num_heads"], 3, config["head_dim"]).astype(np.float32),
        np.random.randn(1, config["num_heads"], 3, config["head_dim"]).astype(np.float32)
    )
    
    # Decode 3 tokens
    for _ in range(3):
        generate_token(None, kv_cache, config)

