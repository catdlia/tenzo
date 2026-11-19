// Engine C API argument validation (T34): unknown kv_mode / TENZO_GEMV / rope_theta must make tenzo_create_engine fail
// instead of silently falling back; every supported kv_mode must be accepted. Tiny geometry, so no large allocations.
#include "tenzo.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

namespace tenzo {

bool runEngineAPITests() {
    llvm::outs() << "--- Engine API validation ---\n";
    bool ok = true;
    auto tiny = [](const char* kv) {
        tenzo_config_t c = tenzo_default_config();
        c.hidden_size = 128; c.num_q_heads = 2; c.num_kv_heads = 1; c.head_dim = 64; c.num_layers = 1;
        c.ffn_dim = 128; c.vocab_size = 32; c.max_seq_len = 16; c.kv_mode = kv;
        return c;
    };
    for (const char* kv : {"fp32", "int8_fused", "tl1_fused", "popcount_fused", "paged_int8"}) {
        tenzo_config_t c = tiny(kv);
        tenzo_engine_t e = tenzo_create_engine(&c);
        llvm::outs() << "  kv_mode=" << kv << ": " << (e ? "accepted" : "REJECTED") << "\n";
        ok &= (e != nullptr);
        if (e) tenzo_destroy_engine(e);
    }
    for (const char* kv : {"bogus", "int8", ""}) {
        tenzo_config_t c = tiny(kv);
        tenzo_engine_t e = tenzo_create_engine(&c);
        llvm::outs() << "  kv_mode='" << kv << "': " << (e ? "ACCEPTED (bug)" : "rejected") << "\n";
        ok &= (e == nullptr);
        if (e) tenzo_destroy_engine(e);
    }
    {
        tenzo_config_t c = tiny("fp32");
        c.rope_theta = 0.0f;
        tenzo_engine_t e = tenzo_create_engine(&c);
        llvm::outs() << "  rope_theta=0: " << (e ? "ACCEPTED (bug)" : "rejected") << "\n";
        ok &= (e == nullptr);
        if (e) tenzo_destroy_engine(e);
    }
    {
        const char* old = std::getenv("TENZO_GEMV");
        std::string saved = old ? old : "";
        setenv("TENZO_GEMV", "bogus", 1);
        tenzo_config_t c = tiny("fp32");
        tenzo_engine_t e = tenzo_create_engine(&c);
        llvm::outs() << "  TENZO_GEMV=bogus: " << (e ? "ACCEPTED (bug)" : "rejected") << "\n";
        ok &= (e == nullptr);
        if (e) tenzo_destroy_engine(e);
        if (old) setenv("TENZO_GEMV", saved.c_str(), 1); else unsetenv("TENZO_GEMV");
    }
    llvm::outs() << (ok ? "✅ [PASS] Engine API validation\n" : "❌ [FAIL] Engine API validation\n");
    return ok;
}

} // namespace tenzo
