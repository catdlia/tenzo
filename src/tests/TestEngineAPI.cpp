// Engine C API argument validation (T34): unknown kv_mode / TENZO_GEMV / rope_theta must make tenzo_create_engine fail
// instead of silently falling back; every supported kv_mode must be accepted. Tiny geometry, so no large allocations.
#include "tenzo.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>
#include <string>

namespace tenzo {

bool runKVRotTests();
bool runBatchedPrefillTests();
bool runQDimNeHiddenTests();

bool runEngineAPITests() {
    llvm::outs() << "--- Engine API validation ---\n";
    bool ok = true;
    auto tiny = [](const char* kv) {
        tenzo_config_t c = tenzo_default_config();
        c.hidden_size = 128; c.num_q_heads = 2; c.num_kv_heads = 1; c.head_dim = 128; c.num_layers = 1;
        c.ffn_dim = 128; c.vocab_size = 32; c.max_seq_len = 16; c.kv_mode = kv;
        return c;
    };
    for (const char* kv : {"fp32", "int8_fused", "tl1_fused", "popcount_fused", "paged_int8", "ternary_rot"}) {
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
    const bool kv_ok = runKVRotTests();
    const bool bp_ok = runBatchedPrefillTests();
    const bool qd_ok = runQDimNeHiddenTests();
    return ok && kv_ok && bp_ok && qd_ok;
}

} // namespace tenzo

// ------------------------------------------------------------------------------------------------------------------
// ternary_rot KV mode: AVX2 FWHT against the dense matrix R, and the full cache (append + attention) against an
// independent double-precision reference of S3+s4r32 written with dense matrices (quantize in the rotated domain,
// rotate back, plain softmax attention over the dequantized keys / values).
#include "runtime/TenzoEngine.h"
#include "runtime/Hadamard.h"
#include <cmath>
#include <vector>

namespace tenzo {
namespace {

struct DenseR {
    double m[128][128];
    DenseR() {
        for (int i = 0; i < 128; ++i)
            for (int j = 0; j < 128; ++j)
                m[i][j] = ((__builtin_popcount(i & j) & 1) ? -1.0 : 1.0) * kHadamardSigns128[j] / std::sqrt(128.0);
    }
    void rot(const double* x, double* y) const {      // y = x R
        for (int j = 0; j < 128; ++j) { double s = 0; for (int i = 0; i < 128; ++i) s += x[i] * m[i][j]; y[j] = s; }
    }
    void unrot(const double* y, double* x) const {    // x = y R^T
        for (int i = 0; i < 128; ++i) { double s = 0; for (int j = 0; j < 128; ++j) s += y[j] * m[i][j]; x[i] = s; }
    }
};

void ref_ternary_g32(const double* y, double* out) {
    for (int g = 0; g < 4; ++g) {
        double mean = 0; for (int i = 0; i < 32; ++i) mean += std::abs(y[32 * g + i]); mean /= 32;
        const double th = 0.7 * mean; double sum = 0; int cnt = 0;
        for (int i = 0; i < 32; ++i) if (std::abs(y[32 * g + i]) > th) { sum += std::abs(y[32 * g + i]); ++cnt; }
        const double s = cnt ? sum / cnt : mean + 1e-6;
        for (int i = 0; i < 32; ++i) { const double v = y[32 * g + i]; out[32 * g + i] = std::abs(v) > th ? (v > 0 ? s : -s) : 0.0; }
    }
}

void ref_int8(const double* x, double* out) {
    double amax = 1e-8; for (int d = 0; d < 128; ++d) amax = std::max(amax, std::abs(x[d]));
    const double s = amax / 127.0;
    for (int d = 0; d < 128; ++d) out[d] = std::min(127.0, std::max(-128.0, std::round(x[d] / s))) * s;
}

} // namespace

// ------------------------------------------------------------------------------------------------------------------
// Batched prefill vs token-by-token prefill on a tiny random ternary model (public C API + KV cache internals): the next-token
// logits and every byte of the KV cache must be IDENTICAL, for chunk sizes with and without a remainder.
namespace {
std::vector<int8_t> bp_pack_tl1(const std::vector<int8_t>& W, int N, int K) {
    std::vector<int8_t> out(size_t(N / 64) * (K / 2) * 32);
    for (int b = 0; b < N / 64; ++b)
        for (int k = 0; k < K / 2; ++k)
            for (int ch = 0; ch < 32; ++ch) {
                auto nib = [&](int row) { return (W[size_t(row) * K + 2 * k] + 1) | ((W[size_t(row) * K + 2 * k + 1] + 1) << 2); };
                out[(size_t(b) * (K / 2) + k) * 32 + ch] = static_cast<int8_t>(nib(b * 64 + ch) | (nib(b * 64 + ch + 32) << 4));
            }
    return out;
}
} // namespace

bool runBatchedPrefillTests() {
    llvm::outs() << "--- batched prefill vs token-by-token (tiny model, bit-identity) ---\n";
    const int Hd = 256, NQ = 2, NKV = 1, HD = 128, F = 384, V = 32, T = 37, NL = 2;   // q_dim == hidden_size as in BitNet; F != hidden; 2 layers so the residual chain is exercised
    unsigned st = 99;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) & 0xFFFF; };
    auto mat = [&](int n, int k) { std::vector<int8_t> m(size_t(n) * k); for (auto& w : m) w = int8_t(int(rnd() % 3) - 1); return m; };
    auto vec = [&](int n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = lo + (hi - lo) * (rnd() / 65536.0f); return v; };
    auto pq = bp_pack_tl1(mat(NQ * HD, Hd), NQ * HD, Hd), pk = bp_pack_tl1(mat(NKV * HD, Hd), NKV * HD, Hd), pv = bp_pack_tl1(mat(NKV * HD, Hd), NKV * HD, Hd),
         po = bp_pack_tl1(mat(Hd, NQ * HD), Hd, NQ * HD), pg = bp_pack_tl1(mat(F, Hd), F, Hd), pu = bp_pack_tl1(mat(F, Hd), F, Hd), pd = bp_pack_tl1(mat(Hd, F), Hd, F);
    auto n_in = vec(Hd, 0.5f, 1.5f), n_attn = vec(Hd, 0.5f, 1.5f), n_post = vec(Hd, 0.5f, 1.5f), n_ffn = vec(F, 0.5f, 1.5f), n_final = vec(Hd, 0.5f, 1.5f);
    auto E = vec(V * Hd, -1.0f, 1.0f);
    std::vector<int> toks; for (int i = 0; i < T; ++i) toks.push_back(int(rnd() % V));
    bool ok = true;
    auto make = [&](const char* kv) {
        tenzo_config_t c = tenzo_default_config();
        c.hidden_size = Hd; c.num_q_heads = NQ; c.num_kv_heads = NKV; c.head_dim = HD; c.num_layers = NL; c.ffn_dim = F;
        c.vocab_size = V; c.max_seq_len = 64; c.kv_mode = kv;
        tenzo_engine_t e = tenzo_create_engine(&c);
        tenzo_layer_weights_t lw{};
        lw.q_w = pq.data(); lw.q_scale = 0.05f; lw.k_w = pk.data(); lw.k_scale = 0.05f; lw.v_w = pv.data(); lw.v_scale = 0.08f;
        lw.out_w = po.data(); lw.out_scale = 0.06f; lw.gate_w = pg.data(); lw.gate_scale = 0.07f; lw.up_w = pu.data(); lw.up_scale = 0.07f;
        lw.down_w = pd.data(); lw.down_scale = 0.05f; lw.in_norm = n_in.data(); lw.attn_sub_norm = n_attn.data(); lw.post_norm = n_post.data();
        lw.ffn_sub_norm = n_ffn.data();
        if (!e) return e;
        for (int l = 0; l < NL; ++l) tenzo_set_layer_weights(e, l, &lw);
        tenzo_set_final_norm(e, n_final.data()); tenzo_set_lm_head_f32(e, E.data(), V, Hd);
        tenzo_reset(e);
        return e;
    };
    for (const char* kv : {"fp32", "int8_fused", "ternary_rot"}) {
        for (int chunk : {8, 16, 32, 64}) {
            tenzo_engine_t a = make(kv), b = make(kv), c2 = make(kv);
            if (!a || !b || !c2) { llvm::errs() << "  ❌ engine creation failed (" << kv << ")\n"; return false; }
            for (int i = 0; i < T - 1; ++i) { tenzo_prefill_token(a, toks[i]); tenzo_prefill_token(c2, toks[i]); }
            tenzo_status_t rc = tenzo_prefill_tokens(b, toks.data(), T - 1, chunk);
            std::vector<float> la(V), lb(V), lc(V);
            tenzo_forward_logits(a, toks[T - 1], la.data(), V);
            tenzo_forward_logits(b, toks[T - 1], lb.data(), V);
            tenzo_forward_logits(c2, toks[T - 1], lc.data(), V);
            const auto& ka = a->impl->kv_cache; const auto& kb = b->impl->kv_cache;
            double dl = 0, dc = 0; for (int i = 0; i < V; ++i) { dl = std::max(dl, (double)std::abs(la[i] - lb[i])); dc = std::max(dc, (double)std::abs(la[i] - lc[i])); }
            bool same_logits = std::memcmp(la.data(), lb.data(), V * sizeof(float)) == 0;
            bool same_kv = ka.cur_seq_len == kb.cur_seq_len && ka.fp32_k == kb.fp32_k && ka.fp32_v == kb.fp32_v && ka.int8_k == kb.int8_k && ka.int8_v == kb.int8_v &&
                        ka.int8_k_scales == kb.int8_k_scales && ka.int8_v_scales == kb.int8_v_scales && ka.tr_k_codes == kb.tr_k_codes &&
                        ka.tr_v_codes == kb.tr_v_codes && ka.tr_k_scales == kb.tr_k_scales && ka.tr_v_scales == kb.tr_v_scales &&
                        ka.tr_k_i8 == kb.tr_k_i8 && ka.tr_v_i8 == kb.tr_v_i8;
            bool same = rc == TENZO_SUCCESS && same_logits && same_kv;
            llvm::outs() << "  kv=" << kv << " chunk=" << chunk << ": " << (same ? "identical" : "DIFFERENT") << " (rc=" << rc << ", logits "
                         << (same_logits ? "same" : "differ, max|d|=") ;
            if (!same_logits) llvm::outs() << dl;
            llvm::outs() << ", kv " << (same_kv ? "same" : "differ") << "; sequential-vs-sequential max|d|=" << dc << ")\n";
            ok &= same;
            tenzo_destroy_engine(c2);
            tenzo_destroy_engine(a); tenzo_destroy_engine(b);
        }
    }
    llvm::outs() << (ok ? "✅ [PASS] batched prefill bit-identity\n" : "❌ [FAIL] batched prefill bit-identity\n");
    return ok;
}

bool runKVRotTests() {
    llvm::outs() << "--- ternary_rot KV: FWHT and attention vs independent references ---\n";
    bool ok = true;
    static const DenseR R;
    // 1. FWHT
    {
        std::vector<float> x(128); unsigned s = 12345;
        for (auto& v : x) { s = s * 1664525u + 1013904223u; v = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        std::vector<float> y(128), back(128);
        hadamard::rotate(x.data(), y.data());
        hadamard::unrotate(y.data(), back.data());
        std::vector<double> xd(x.begin(), x.end()), yd(128);
        R.rot(xd.data(), yd.data());
        double e = 0, n = 0, e2 = 0, n2 = 0;
        for (int i = 0; i < 128; ++i) { e += (y[i] - yd[i]) * (y[i] - yd[i]); n += yd[i] * yd[i]; e2 += (back[i] - x[i]) * (back[i] - x[i]); n2 += x[i] * x[i]; }
        const double r1 = std::sqrt(e / n), r2 = std::sqrt(e2 / n2);
        llvm::outs() << "  FWHT vs dense x@R: rel_l2 " << r1 << "; unrotate(rotate(x)) vs x: " << r2 << "\n";
        ok &= (r1 < 1e-6) && (r2 < 1e-6);
    }
    // 2. cache attention vs reference (80 steps: sinks, ring wrap-around and ternary region all exercised)
    {
        const int T = 80, HQ = 2, HKV = 1, D = 128;
        FusedKVCache cache(1, HQ, HKV, D, 128, "ternary_rot");
        cache.rope_theta = 500000.0f;
        unsigned s = 777;
        auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; };
        std::vector<std::vector<double>> K, V;
        double worst = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<float> q(HQ * D), k(HKV * D), v(HKV * D), out(HQ * D);
            for (auto& x : q) x = 2.0f * rnd();
            for (auto& x : k) x = 2.0f * rnd() * (1.0f + (t % 7));   // varying magnitudes
            for (auto& x : v) x = rnd();
            for (int d = 0; d < D; d += 17) k[d] *= 6.0f;           // outlier channels
            cache.forward_attention_raw(0, q.data(), k.data(), v.data(), out.data());
            cache.increment_seq_len(1);
            // q and k now hold the post-RoPE vectors
            K.emplace_back(k.begin(), k.end()); V.emplace_back(v.begin(), v.end());
            std::vector<double> o_ref(HQ * D, 0.0);
            for (int h = 0; h < HQ; ++h) {
                std::vector<double> sc(t + 1), qd(q.begin() + h * D, q.begin() + (h + 1) * D);
                double mx = -1e300;
                std::vector<std::vector<double>> vh(t + 1, std::vector<double>(D));
                for (int j = 0; j <= t; ++j) {
                    std::vector<double> kh(D), y(D), yt(D);
                    if (j < 4 || t - j < 32) {
                        ref_int8(K[j].data(), kh.data()); ref_int8(V[j].data(), vh[j].data());
                    } else {
                        R.rot(K[j].data(), y.data()); ref_ternary_g32(y.data(), yt.data()); R.unrot(yt.data(), kh.data());
                        R.rot(V[j].data(), y.data()); ref_ternary_g32(y.data(), yt.data()); R.unrot(yt.data(), vh[j].data());
                    }
                    double d = 0; for (int i = 0; i < D; ++i) d += qd[i] * kh[i];
                    sc[j] = d / std::sqrt((double)D); mx = std::max(mx, sc[j]);
                }
                double z = 0; for (auto& x : sc) { x = std::exp(x - mx); z += x; }
                for (int j = 0; j <= t; ++j) for (int i = 0; i < D; ++i) o_ref[h * D + i] += sc[j] / z * vh[j][i];
            }
            double e = 0, n = 0;
            for (int i = 0; i < HQ * D; ++i) { e += (out[i] - o_ref[i]) * (out[i] - o_ref[i]); n += o_ref[i] * o_ref[i]; }
            worst = std::max(worst, std::sqrt(e / std::max(n, 1e-30)));
        }
        llvm::outs() << "  ternary_rot attention, " << T << " steps (sinks 4, recent 32, ternary beyond): worst rel_l2 vs reference " << worst << "\n";
        ok &= (worst < 1e-4);
    }
    llvm::outs() << (ok ? "✅ [PASS] ternary_rot KV\n" : "❌ [FAIL] ternary_rot KV\n");
    return ok;
}

// ------------------------------------------------------------------------------------------------------------------
// q_dim != hidden_size regression (T42): attn_sub_norm and out_proj input length must be q_dim, not hidden_size.
// Tested on a tiny 1-layer model (hidden=128, q_dim=2*128=256) across decode (i2s, lut) and batched prefill (i2s).
bool runQDimNeHiddenTests() {
    llvm::outs() << "--- q_dim != hidden_size (attn_sub_norm + out_proj length) ---\n";
    const int Hd = 128, NQ = 2, NKV = 1, HD = 128, QD = NQ * HD, F = 128, V = 32, T = 6;
    const double theta = 500000.0;
    unsigned st = 4242;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) & 0xFFFF; };
    auto mat = [&](int n, int k) { std::vector<int8_t> m(size_t(n) * k); for (auto& w : m) w = int8_t(int(rnd() % 3) - 1); return m; };
    auto vec = [&](int n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = lo + (hi - lo) * (rnd() / 65536.0f); return v; };

    auto Wq = mat(QD, Hd), Wk = mat(NKV * HD, Hd), Wv = mat(NKV * HD, Hd), Wo = mat(Hd, QD);
    auto Wg = mat(F, Hd), Wu = mat(F, Hd), Wd = mat(Hd, F);
    const float sq = 0.05f, sk = 0.05f, sv = 0.08f, so = 0.06f, sg = 0.07f, su = 0.07f, sd = 0.05f;
    auto n_in = vec(Hd, 0.5f, 1.5f), n_attn = vec(QD, 0.5f, 1.5f), n_post = vec(Hd, 0.5f, 1.5f), n_ffn = vec(F, 0.5f, 1.5f),
         n_final = vec(Hd, 0.5f, 1.5f), E = vec(V * Hd, -1.0f, 1.0f);
    std::vector<int> toks;
    for (int t = 0; t < T; ++t) toks.push_back(int(rnd() % V));

    auto rms = [](const std::vector<float>& x, const std::vector<float>& w, int dim) {
        double ss = 0;
        for (int i = 0; i < dim; ++i) ss += double(x[i]) * x[i];
        const float inv = 1.0f / std::sqrt(float(ss / dim) + 1e-5f);
        std::vector<float> y(x.size(), 0.0f);
        for (int i = 0; i < dim; ++i) y[i] = x[i] * inv * w[i];
        return y;
    };
    auto bitlin = [](const std::vector<float>& x, const std::vector<int8_t>& W, int N, int K, float ws) {
        float m = 0.0f;
        for (float v : x) m = std::max(m, std::abs(v));
        m = std::max(m, 1e-5f);
        const float sc = 127.0f / m, inv = 1.0f / sc;
        std::vector<int> q(K);
        for (int k = 0; k < K; ++k) q[k] = static_cast<int>(std::clamp(std::nearbyint(x[k] * sc), -128.0f, 127.0f));
        std::vector<float> y(N);
        for (int n = 0; n < N; ++n) {
            int64_t acc = 0;
            for (int k = 0; k < K; ++k) acc += int64_t(W[size_t(n) * K + k]) * q[k];
            y[n] = static_cast<float>(acc) * (inv * ws);
        }
        return y;
    };
    auto rope = [&](float* h, int pos) {
        for (int i = 0; i < HD / 2; ++i) {
            double inv = 1.0 / std::pow(theta, (2.0 * i) / HD);
            double c = std::cos(pos * inv), s = std::sin(pos * inv);
            double x1 = h[i], x2 = h[i + HD / 2];
            h[i] = static_cast<float>(x1 * c - x2 * s);
            h[i + HD / 2] = static_cast<float>(x1 * s + x2 * c);
        }
    };
    auto run_ref = [&](int attn_norm_dim) {
        std::vector<std::vector<float>> K, Vv;
        std::vector<float> last_logits(V);
        for (int t = 0; t < T; ++t) {
            std::vector<float> x(E.begin() + size_t(toks[t]) * Hd, E.begin() + size_t(toks[t] + 1) * Hd);
            auto h = rms(x, n_in, Hd);
            auto q = bitlin(h, Wq, QD, Hd, sq);
            auto k = bitlin(h, Wk, NKV * HD, Hd, sk);
            auto v = bitlin(h, Wv, NKV * HD, Hd, sv);
            for (int hh = 0; hh < NQ; ++hh) rope(q.data() + hh * HD, t);
            for (int hh = 0; hh < NKV; ++hh) rope(k.data() + hh * HD, t);
            K.push_back(k); Vv.push_back(v);
            std::vector<float> a(QD, 0.0f);
            for (int hh = 0; hh < NQ; ++hh) {
                std::vector<double> sc(t + 1);
                double mx = -1e300;
                for (int s = 0; s <= t; ++s) {
                    double d = 0;
                    for (int i = 0; i < HD; ++i) d += double(q[hh * HD + i]) * K[s][i];
                    sc[s] = d / std::sqrt(double(HD)); mx = std::max(mx, sc[s]);
                }
                double z = 0;
                for (auto& s : sc) { s = std::exp(s - mx); z += s; }
                for (int s = 0; s <= t; ++s)
                    for (int i = 0; i < HD; ++i) a[hh * HD + i] += float(sc[s] / z * Vv[s][i]);
            }
            auto o = bitlin(rms(a, n_attn, attn_norm_dim), Wo, Hd, QD, so);
            for (int i = 0; i < Hd; ++i) x[i] += o[i];
            auto p = rms(x, n_post, Hd);
            auto gt = bitlin(p, Wg, F, Hd, sg), up = bitlin(p, Wu, F, Hd, su);
            std::vector<float> act(F);
            for (int i = 0; i < F; ++i) { float r = std::max(0.0f, gt[i]); act[i] = r * r * up[i]; }
            auto d = bitlin(rms(act, n_ffn, F), Wd, Hd, F, sd);
            for (int i = 0; i < Hd; ++i) x[i] += d[i];
            auto fn = rms(x, n_final, Hd);
            for (int vv = 0; vv < V; ++vv) {
                double acc = 0;
                for (int i = 0; i < Hd; ++i) acc += double(fn[i]) * E[size_t(vv) * Hd + i];
                last_logits[vv] = float(acc);
            }
        }
        return last_logits;
    };
    auto rel_l2 = [](const std::vector<float>& a, const std::vector<float>& b) {
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) { double d = double(a[i]) - b[i]; num += d * d; den += double(b[i]) * b[i]; }
        return std::sqrt(num / std::max(den, 1e-300));
    };

    const auto ref_ok = run_ref(QD);
    const auto ref_bug = run_ref(Hd);
    const double bug_diff = rel_l2(ref_bug, ref_ok);

    auto pq = bp_pack_tl1(Wq, QD, Hd), pk = bp_pack_tl1(Wk, NKV * HD, Hd), pv = bp_pack_tl1(Wv, NKV * HD, Hd),
         po = bp_pack_tl1(Wo, Hd, QD), pg = bp_pack_tl1(Wg, F, Hd), pu = bp_pack_tl1(Wu, F, Hd), pd = bp_pack_tl1(Wd, Hd, F);

    const char* old_gemv = std::getenv("TENZO_GEMV");
    std::string saved_gemv = old_gemv ? old_gemv : "";

    auto run_engine = [&](const char* gemv, bool batched) {
        setenv("TENZO_GEMV", gemv, 1);
        tenzo_config_t c = tenzo_default_config();
        c.hidden_size = Hd; c.num_q_heads = NQ; c.num_kv_heads = NKV; c.head_dim = HD; c.num_layers = 1;
        c.ffn_dim = F; c.vocab_size = V; c.max_seq_len = 16; c.kv_mode = "fp32"; c.rope_theta = float(theta);
        tenzo_engine_t e = tenzo_create_engine(&c);
        std::vector<float> logits(V);
        if (!e) return logits;
        tenzo_layer_weights_t lw{};
        lw.q_w = pq.data(); lw.q_scale = sq; lw.k_w = pk.data(); lw.k_scale = sk; lw.v_w = pv.data(); lw.v_scale = sv;
        lw.out_w = po.data(); lw.out_scale = so; lw.gate_w = pg.data(); lw.gate_scale = sg;
        lw.up_w = pu.data(); lw.up_scale = su; lw.down_w = pd.data(); lw.down_scale = sd;
        lw.in_norm = n_in.data(); lw.attn_sub_norm = n_attn.data(); lw.post_norm = n_post.data(); lw.ffn_sub_norm = n_ffn.data();
        tenzo_set_layer_weights(e, 0, &lw);
        tenzo_set_final_norm(e, n_final.data());
        tenzo_set_lm_head_f32(e, E.data(), V, Hd);
        tenzo_reset(e);
        if (batched) {
            tenzo_prefill_tokens(e, toks.data(), T - 1, 3);
        } else {
            for (int t = 0; t < T - 1; ++t) tenzo_prefill_token(e, toks[t]);
        }
        tenzo_forward_logits(e, toks[T - 1], logits.data(), V);
        tenzo_destroy_engine(e);
        return logits;
    };

    const auto out_i2s_seq = run_engine("i2s", false);
    const auto out_i2s_bat = run_engine("i2s", true);
    const auto out_lut_seq = run_engine("lut", false);

    if (old_gemv) setenv("TENZO_GEMV", saved_gemv.c_str(), 1); else unsetenv("TENZO_GEMV");

    const double e_i2s = rel_l2(out_i2s_seq, ref_ok);
    const double e_bat = rel_l2(out_i2s_bat, ref_ok);
    const double e_lut = rel_l2(out_lut_seq, ref_ok);
    const bool eq_bat = std::memcmp(out_i2s_seq.data(), out_i2s_bat.data(), V * sizeof(float)) == 0;
    const bool eq_lut = std::memcmp(out_i2s_seq.data(), out_lut_seq.data(), V * sizeof(float)) == 0;
    llvm::outs() << "  hidden=" << Hd << " q_dim=" << QD << ": rel_l2 i2s_seq=" << e_i2s << " i2s_batched=" << e_bat
                 << " lut_seq=" << e_lut << " (old hidden_size bug would be " << bug_diff << "), bit_eq(bat=" << eq_bat
                 << ", lut=" << eq_lut << ")\n";
    bool ok = (e_i2s < 1e-4) && (e_bat < 1e-4) && (e_lut < 1e-4) && (bug_diff > 1e-2) && eq_bat && eq_lut;
    llvm::outs() << (ok ? "✅ [PASS] q_dim != hidden_size\n" : "❌ [FAIL] q_dim != hidden_size\n");
    return ok;
}

} // namespace tenzo
