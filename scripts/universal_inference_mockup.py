import numpy as np
import time
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

def verify_numerical_correctness():
    print("\n=======================================================")
    print("🔍 [TEST 1] Numerical Accuracy & AVX2 Kernel Validation")
    print("=======================================================")

    if not HAS_TENZO:
        print("❌ Cannot test native kernels: tenzo_runtime missing")
        return

    # 1. FP32 GEMM Test
    M, K, N = 1, 128, 64
    x_fp32 = np.random.randn(M, K).astype(np.float32)
    w_fp32 = np.random.randn(N, K).astype(np.float32)
    bias_fp32 = np.random.randn(N).astype(np.float32)

    ref_fp32 = np.dot(x_fp32, w_fp32.T) + bias_fp32
    cxx_fp32 = tenzo_runtime.dispatch_linear_fp32(x_fp32, w_fp32, bias_fp32)

    diff_fp32 = np.max(np.abs(ref_fp32 - cxx_fp32))
    print(f"  • FP32 BLIS 6x16 AVX2 GEMM max diff vs NumPy: {diff_fp32:.2e}")
    assert diff_fp32 < 1e-4, f"FP32 GEMM mismatch! Diff: {diff_fp32}"
    print("  ✅ FP32 GEMM: PASSED (100% Match)")

    # 2. BitLinear TL1 Ternary SIMD Test
    # Generate random ternary weights {-1, 0, 1}
    w_tern = np.random.choice([-1, 0, 1], size=(N, K)).astype(np.int8)
    packed_w = tenzo_runtime.pack_ternary_weights(w_tern)
    
    # Reference calculation for quantized BitLinear
    max_abs = np.max(np.abs(x_fp32))
    scale_act = 31.0 / max(max_abs, 1e-6)
    x_q16 = np.round(x_fp32 * scale_act).astype(np.int16)
    
    ref_tl1_int = np.dot(x_q16.astype(np.float32), w_tern.T.astype(np.float32))
    ref_tl1 = ref_tl1_int * (max_abs / 31.0) * 1.0 # scale = 1.0
    
    cxx_tl1 = tenzo_runtime.dispatch_bitlinear_tl1(x_fp32, packed_w, 1.0)
    
    diff_tl1 = np.max(np.abs(ref_tl1 - cxx_tl1))
    print(f"  • BitLinear TL1 AVX2 LUT kernel max diff vs Quant Ref: {diff_tl1:.2e}")
    assert diff_tl1 < 1e-3, f"BitLinear TL1 mismatch! Diff: {diff_tl1}"
    print("  ✅ BitLinear TL1 AVX2 Kernel: PASSED (100% Match)")
    print("=======================================================\n")

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
    if quant_scheme == "fp32" or quant_scheme == "fp16":
        if HAS_TENZO:
            return tenzo_runtime.dispatch_linear_fp32(x, weight, bias)
        else:
            res = np.dot(x, weight.T)
            if bias is not None:
                res += bias
            return res
    elif quant_scheme == "classic_tl1" or quant_scheme == "classic_tl2":
        if HAS_TENZO:
            return tenzo_runtime.dispatch_linear_cxx(x, weight, bias, quant_scheme=quant_scheme)
        else:
            res = np.dot(x, weight.T)
            if bias is not None:
                res += bias
            return res
    else:
        raise ValueError(f"Unknown quant_scheme: {quant_scheme}")

def generate_token(input_ids, kv_cache, config, weights):
    x = np.random.randn(1, 1, config["hidden_size"]).astype(np.float32)
    
    # Q, K, V Projections
    q = dispatch_linear(x, weights["q_w"], quant_scheme=config["q_scheme"])
    k = dispatch_linear(x, weights["k_w"], quant_scheme=config["q_scheme"])
    v = dispatch_linear(x, weights["v_w"], quant_scheme=config["q_scheme"])
    
    # Reshape for Attention
    q = q.reshape(1, 1, config["num_heads"], config["head_dim"]).transpose(0, 2, 1, 3)
    k = k.reshape(1, 1, config["num_heads"], config["head_dim"]).transpose(0, 2, 1, 3)
    v = v.reshape(1, 1, config["num_heads"], config["head_dim"]).transpose(0, 2, 1, 3)
    
    # Update KV Cache
    past_k, past_v = kv_cache.update(k, v)
    
    # Scaled Dot-Product Attention
    attn_weights = np.matmul(q, past_k.transpose(0, 1, 3, 2)) / (config["head_dim"] ** 0.5)
    attn_weights = np.exp(attn_weights - np.max(attn_weights, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    
    attn_output = np.matmul(attn_weights, past_v)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(1, 1, config["hidden_size"])
    
    # Output Projection
    output = dispatch_linear(attn_output, weights["out_w"], quant_scheme=config["q_scheme"])
    return output

def run_e2e_simulation(scheme="classic_tl1"):
    print(f"\n🚀 Running End-to-End Simulation with scheme: [{scheme}]")
    config = {
        "hidden_size": 2048,
        "num_heads": 16,
        "head_dim": 128,
        "q_scheme": scheme
    }

    # Initialize weights
    if scheme == "classic_tl1":
        q_w = np.random.choice([-1, 0, 1], size=(config["hidden_size"], config["hidden_size"])).astype(np.int8)
        k_w = np.random.choice([-1, 0, 1], size=(config["hidden_size"], config["hidden_size"])).astype(np.int8)
        v_w = np.random.choice([-1, 0, 1], size=(config["hidden_size"], config["hidden_size"])).astype(np.int8)
        out_w = np.random.choice([-1, 0, 1], size=(config["hidden_size"], config["hidden_size"])).astype(np.int8)
        if HAS_TENZO:
            q_w = tenzo_runtime.pack_ternary_weights(q_w)
            k_w = tenzo_runtime.pack_ternary_weights(k_w)
            v_w = tenzo_runtime.pack_ternary_weights(v_w)
            out_w = tenzo_runtime.pack_ternary_weights(out_w)
    else:
        q_w = np.random.randn(config["hidden_size"], config["hidden_size"]).astype(np.float32)
        k_w = np.random.randn(config["hidden_size"], config["hidden_size"]).astype(np.float32)
        v_w = np.random.randn(config["hidden_size"], config["hidden_size"]).astype(np.float32)
        out_w = np.random.randn(config["hidden_size"], config["hidden_size"]).astype(np.float32)

    weights = {"q_w": q_w, "k_w": k_w, "v_w": v_w, "out_w": out_w}
    kv_cache = KVCacheManager(max_batch=1, max_seq=1024, num_heads=config["num_heads"], head_dim=config["head_dim"])

    # Warmup
    for _ in range(2):
        generate_token(None, kv_cache, config, weights)

    # Benchmark 10 tokens
    n_tokens = 10
    t0 = time.perf_counter()
    for step in range(n_tokens):
        generate_token(None, kv_cache, config, weights)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    tok_per_sec = n_tokens / elapsed
    print(f"  • Generated {n_tokens} tokens in {elapsed:.3f} s ({tok_per_sec:.2f} tok/sec per layer)")

if __name__ == "__main__":
    verify_numerical_correctness()
    run_e2e_simulation(scheme="fp32")
    run_e2e_simulation(scheme="classic_tl1")
