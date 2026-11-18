// Regression tests for the two semantic bugs found on 2026-11-19 against HF transformers BitNet
// (modeling_bitnet.py / integrations/bitnet.py):
//   (1) RoPE base must be the model's rope_theta (500000 for BitNet-b1.58-2B-4T), not a hard-coded 10000;
//   (2) BitLinear activations are quantized like HF ActQuant: scale = 127 / max(|x|).clamp(1e-5),
//       round half-to-even, clamp(-128, 127)  (the kernels used 31 or 63 levels before).
// Each check compares against an independent reference written from the HF formulas, and additionally verifies that
// the *buggy* semantics (theta 10000, 31 levels) would be rejected by the same tolerance (the test discriminates).
// Paths covered: MLIR lowering of tenzo.rope / tenzo.bitlinear_tl1_pack / tenzo.bitlinear_tl1, and the standalone
// runtime engine (TenzoEngine, public C API) on a 1-layer model with fp32 KV.

#include "context/TenzoContext.h"
#include "runtime/ExecutionContext.h"
#include "passes/Passes.h"
#include "tenzo.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace mlir;

namespace tenzo {
namespace {

constexpr double kTol = 1e-4; // project-wide numerical divergence tolerance

struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<uint32_t>(s >> 33); }
    float uniform() { return (next() & 0xFFFFFF) / float(0x1000000); } // [0, 1)
    float sym() { return 2.0f * uniform() - 1.0f; }
    int tern() { return static_cast<int>(next() % 3) - 1; }
};

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { double d = double(a[i]) - b[i]; num += d * d; den += double(b[i]) * b[i]; }
    return std::sqrt(num / std::max(den, 1e-300));
}

// HF ActQuant (fp32): returns integer codes and the dequant factor max/127. qp = 31 / 63 reproduce the old level
// counts; trunc = true reproduces the old MLIR lowerings (FPToSI = truncation toward zero, no rounding).
std::vector<int> act_quant(const std::vector<float>& x, float& inv_scale, float qp = 127.0f, bool trunc = false) {
    float m = 0.0f;
    for (float v : x) m = std::max(m, std::abs(v));
    m = std::max(m, 1e-5f);
    const float scale = qp / m;
    inv_scale = 1.0f / scale;
    std::vector<int> q(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        q[i] = static_cast<int>(std::clamp(trunc ? std::trunc(x[i] * scale) : std::nearbyint(x[i] * scale), -128.0f, 127.0f));
    return q;
}

// Ternary matmul with HF fake-quant activations: y[n] = sum_k W[n][k] * q[k] * inv_scale * w_scale
std::vector<float> bitlinear_ref(const std::vector<float>& x, const std::vector<int8_t>& W, int N, int K, float w_scale,
                                 float qp = 127.0f, bool trunc = false) {
    float inv = 0;
    auto q = act_quant(x, inv, qp, trunc);
    std::vector<float> y(N);
    for (int n = 0; n < N; ++n) {
        int64_t acc = 0;
        for (int k = 0; k < K; ++k) acc += int64_t(W[size_t(n) * K + k]) * q[k];
        y[n] = static_cast<float>(acc) * (inv * w_scale);
    }
    return y;
}

// TL1 pack layout shared by the runtime (pack_ternary_weights_raw) and tenzo.bitlinear_tl1_pack:
// [N/64][K/2][32] bytes; byte ch of (block b, pair k): low nibble -> row b*64+ch, high nibble -> row b*64+ch+32;
// nibble = (w(row, 2k) + 1) | ((w(row, 2k+1) + 1) << 2).
std::vector<int8_t> pack_tl1(const std::vector<int8_t>& W, int N, int K) {
    std::vector<int8_t> out(size_t(N / 64) * (K / 2) * 32);
    for (int b = 0; b < N / 64; ++b)
        for (int k = 0; k < K / 2; ++k)
            for (int ch = 0; ch < 32; ++ch) {
                auto nib = [&](int row) {
                    return (W[size_t(row) * K + 2 * k] + 1) | ((W[size_t(row) * K + 2 * k + 1] + 1) << 2);
                };
                out[(size_t(b) * (K / 2) + k) * 32 + ch] = static_cast<int8_t>(nib(b * 64 + ch) | (nib(b * 64 + ch + 32) << 4));
            }
    return out;
}

// HF apply_rotary_pos_emb with rotate_half pairing (i, i + D/2), inv_freq = theta^(-2i/D), double precision.
void rope_ref(float* h, int D, int pos, double theta) {
    for (int i = 0; i < D / 2; ++i) {
        double inv = 1.0 / std::pow(theta, (2.0 * i) / D);
        double c = std::cos(pos * inv), s = std::sin(pos * inv);
        double x1 = h[i], x2 = h[i + D / 2];
        h[i] = static_cast<float>(x1 * c - x2 * s);
        h[i + D / 2] = static_cast<float>(x1 * s + x2 * c);
    }
}

OwningOpRef<ModuleOp> compile(MLIRContext& ctx, const std::string& src, bool expect_ok = true) {
    llvm::SourceMgr sm;
    sm.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(src), llvm::SMLoc());
    auto module = parseSourceFile<ModuleOp>(sm, &ctx);
    if (!module) { llvm::errs() << "  parse failed\n"; return nullptr; }
    PassManager pm(&ctx);
    addTenzoToLinalgPass(pm);
    addTenzoBufferizationPasses(pm);
    addTenzoToLLVMPasses(pm, false, {8, 16, 4}, true);
    if (failed(pm.run(module.get()))) {
        if (expect_ok) llvm::errs() << "  MLIR compilation failed\n";
        return nullptr;
    }
    return module;
}

// ------------------------------------------------------------------------------------------- compiler: tenzo.rope
bool test_rope_lowering(MLIRContext& ctx) {
    const int H = 2, D = 128, pos = 37;
    const double theta = 500000.0;
    std::string src = R"(
module {
  func.func @main(%arg0: tensor<1x2x1x128xf32>) -> tensor<1x2x1x128xf32> attributes {llvm.emit_c_interface} {
    %pos = arith.constant dense<37> : tensor<1xi32>
    %r = "tenzo.rope"(%arg0, %pos) {theta = 5.000000e+05 : f64} : (tensor<1x2x1x128xf32>, tensor<1xi32>) -> tensor<1x2x1x128xf32>
    return %r : tensor<1x2x1x128xf32>
  }
}
)";
    auto module = compile(ctx, src);
    if (!module) return false;
    Lcg g(7);
    runtime::Tensor in({1, H, 1, D}), out({1, H, 1, D});
    std::vector<float> x(H * D);
    for (auto& v : x) v = g.sym() * 3.0f;
    std::memcpy(in.data, x.data(), x.size() * sizeof(float));
    runtime::ExecutionContext ec(ctx, module.get());
    std::vector<runtime::Tensor*> ins = {&in}, outs = {&out};
    ec.forward(ins, outs);
    std::vector<float> got(out.data, out.data + H * D), ref = x, bug = x;
    for (int h = 0; h < H; ++h) { rope_ref(ref.data() + h * D, D, pos, theta); rope_ref(bug.data() + h * D, D, pos, 10000.0); }
    const double e = rel_l2(got, ref), e_bug = rel_l2(bug, ref);
    llvm::outs() << "  tenzo.rope theta=5e5 pos=37: rel_l2 vs HF ref " << e << " (theta=1e4 would be " << e_bug << ")\n";
    if (!(e <= kTol)) { llvm::errs() << "  ❌ tenzo.rope diverges from HF RoPE\n"; return false; }
    if (!(e_bug > 100 * kTol)) { llvm::errs() << "  ❌ test does not discriminate theta\n"; return false; }

    // Without a theta attribute the lowering must refuse (no silent default base).
    std::string no_theta = src;
    no_theta.replace(no_theta.find(" {theta = 5.000000e+05 : f64}"), std::string(" {theta = 5.000000e+05 : f64}").size(), "");
    if (compile(ctx, no_theta, /*expect_ok=*/false)) {
        llvm::errs() << "  ❌ tenzo.rope without theta compiled (expected a lowering error)\n";
        return false;
    }
    llvm::outs() << "  tenzo.rope without theta: lowering rejected as expected\n";
    return true;
}

// ---------------------------------------------------------------------------- compiler: tenzo.bitlinear_tl1_pack
std::vector<float> test_activations(int K, uint64_t seed) {
    // Spread values plus an outlier so that most codes have |q| > 31 at 127 levels.
    Lcg g(seed);
    std::vector<float> x(K);
    for (auto& v : x) v = g.sym() * 2.0f;
    x[K / 3] = 2.5f;
    return x;
}

bool test_tl1_pack_lowering(MLIRContext& ctx) {
    const int K = 256, N = 128;
    const float w_scale = 1.5f;
    Lcg g(11);
    std::vector<int8_t> W(size_t(N) * K);
    for (auto& w : W) w = static_cast<int8_t>(g.tern());
    auto packed = pack_tl1(W, N, K);
    auto x = test_activations(K, 3);
    float inv = 0;
    auto q = act_quant(x, inv);
    int big = 0;
    for (int v : q) big += std::abs(v) > 31;
    std::string src = R"(
module {
  func.func @main(%arg0: tensor<1x1x256xf32>, %arg1: memref<2x128x32xi8>) -> tensor<1x1x128xf32> attributes {llvm.emit_c_interface} {
    %scale = arith.constant 1.5 : f32
    %w = bufferization.to_tensor %arg1 restrict : memref<2x128x32xi8> to tensor<2x128x32xi8>
    %out = "tenzo.bitlinear_tl1_pack"(%arg0, %w, %scale) : (tensor<1x1x256xf32>, tensor<2x128x32xi8>, f32) -> tensor<1x1x128xf32>
    return %out : tensor<1x1x128xf32>
  }
}
)";
    auto module = compile(ctx, src);
    if (!module) return false;
    runtime::Tensor in({1, 1, K}), wt({2, 128, 32}), out({1, 1, N});
    std::memcpy(in.data, x.data(), K * sizeof(float));
    std::memcpy(reinterpret_cast<int8_t*>(wt.data), packed.data(), packed.size());
    runtime::ExecutionContext ec(ctx, module.get());
    std::vector<runtime::Tensor*> ins = {&in, &wt}, outs = {&out};
    ec.forward(ins, outs);
    std::vector<float> got(out.data, out.data + N);
    auto ref = bitlinear_ref(x, W, N, K, w_scale), bug = bitlinear_ref(x, W, N, K, w_scale, 31.0f);
    const double e = rel_l2(got, ref), e_bug = rel_l2(bug, ref);
    llvm::outs() << "  tenzo.bitlinear_tl1_pack K=256 (" << big << "/" << K << " codes with |q|>31): rel_l2 vs HF ref " << e
                 << " (31 levels would be " << e_bug << ")\n";
    if (!(e <= kTol)) { llvm::errs() << "  ❌ tenzo.bitlinear_tl1_pack diverges from HF ActQuant semantics\n"; return false; }
    if (!(e_bug > 100 * kTol)) { llvm::errs() << "  ❌ test does not discriminate activation levels\n"; return false; }
    return true;
}

// ---------------------------------------------------------------------------------- compiler: tenzo.bitlinear_tl1
// All weights = +1 (every 2-bit code = 2, byte 0xAA): y[n] = scale * inv * sum_k q_k for every row, independent of
// the op's row/column interleaving, so only the activation quantization is under test.
bool test_tl1_lowering(MLIRContext& ctx) {
    const int K = 128, N = 128;
    const float w_scale = 1.5f;
    // With all-(+1) weights quantization errors of mixed-sign inputs average out, so use positive activations in
    // [0.2, 0.4] plus one outlier: a level-count or rounding-mode error then shifts every code the same way.
    Lcg g(5);
    std::vector<float> x(K);
    for (auto& v : x) v = 0.2f + 0.2f * g.uniform();
    x[K / 3] = 2.5f;
    std::vector<int8_t> W(size_t(N) * K, 1);
    std::string src = R"(
module {
  func.func @main(%arg0: tensor<1x1x128xf32>, %arg1: memref<2x64x32xi8>) -> tensor<1x1x128xf32> attributes {llvm.emit_c_interface} {
    %scale = arith.constant 1.5 : f32
    %w = bufferization.to_tensor %arg1 restrict : memref<2x64x32xi8> to tensor<2x64x32xi8>
    %out = "tenzo.bitlinear_tl1"(%arg0, %w, %scale) : (tensor<1x1x128xf32>, tensor<2x64x32xi8>, f32) -> tensor<1x1x128xf32>
    return %out : tensor<1x1x128xf32>
  }
}
)";
    auto module = compile(ctx, src);
    if (!module) return false;
    runtime::Tensor in({1, 1, K}), wt({2, 64, 32}), out({1, 1, N});
    std::memcpy(in.data, x.data(), K * sizeof(float));
    std::memset(wt.data, 0xAA, 4096);
    runtime::ExecutionContext ec(ctx, module.get());
    std::vector<runtime::Tensor*> ins = {&in, &wt}, outs = {&out};
    ec.forward(ins, outs);
    std::vector<float> got(out.data, out.data + N);
    auto ref = bitlinear_ref(x, W, N, K, w_scale);
    auto bug = bitlinear_ref(x, W, N, K, w_scale, 63.0f, /*trunc=*/true); // semantics before 2026-11-19
    const double e = rel_l2(got, ref), e_bug = rel_l2(bug, ref);
    llvm::outs() << "  tenzo.bitlinear_tl1 K=128 (all +1 weights): rel_l2 vs HF ref " << e
                 << " (old 63-level truncation would be " << e_bug << ")\n";
    if (!(e <= kTol)) { llvm::errs() << "  ❌ tenzo.bitlinear_tl1 diverges from HF ActQuant semantics\n"; return false; }
    if (!(e_bug > 100 * kTol)) { llvm::errs() << "  ❌ test does not discriminate activation levels\n"; return false; }
    return true;
}

// ------------------------------------------------------------------------------- runtime engine (public C API)
struct TinyModel {
    static constexpr int Hd = 128, NQ = 2, NKV = 1, HD = 64, F = 128, V = 32, T = 8;
    std::vector<int8_t> Wq, Wk, Wv, Wo, Wg, Wu, Wd;
    float sq, sk, sv, so, sg, su, sd;
    std::vector<float> n_in, n_attn, n_post, n_ffn, n_final, E;
    std::vector<int> tokens;

    TinyModel() {
        Lcg g(2026);
        auto mat = [&](int n, int k) { std::vector<int8_t> m(size_t(n) * k); for (auto& w : m) w = int8_t(g.tern()); return m; };
        auto vec = [&](int n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = lo + (hi - lo) * g.uniform(); return v; };
        Wq = mat(NQ * HD, Hd); Wk = mat(NKV * HD, Hd); Wv = mat(NKV * HD, Hd); Wo = mat(Hd, NQ * HD);
        Wg = mat(F, Hd); Wu = mat(F, Hd); Wd = mat(Hd, F);
        sq = 0.05f; sk = 0.05f; sv = 0.08f; so = 0.06f; sg = 0.07f; su = 0.07f; sd = 0.05f;
        n_in = vec(Hd, 0.5f, 1.5f); n_attn = vec(Hd, 0.5f, 1.5f); n_post = vec(Hd, 0.5f, 1.5f);
        n_ffn = vec(F, 0.5f, 1.5f); n_final = vec(Hd, 0.5f, 1.5f);
        E = vec(V * Hd, -1.0f, 1.0f);
        for (int t = 0; t < T; ++t) tokens.push_back(int(g.next() % V));
    }
};

std::vector<float> rms(const std::vector<float>& x, const std::vector<float>& w) {
    double ss = 0;
    for (float v : x) ss += double(v) * v;
    const float inv = 1.0f / std::sqrt(float(ss / x.size()) + 1e-5f);
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * w[i];
    return y;
}

// Reference forward (HF BitNet semantics) over the token sequence; returns logits per position.
std::vector<std::vector<float>> tiny_ref(const TinyModel& m, double theta, float qp) {
    using M = TinyModel;
    std::vector<std::vector<float>> K, Vv, out;
    for (int t = 0; t < M::T; ++t) {
        std::vector<float> x(m.E.begin() + size_t(m.tokens[t]) * M::Hd, m.E.begin() + size_t(m.tokens[t] + 1) * M::Hd);
        auto h = rms(x, m.n_in);
        auto q = bitlinear_ref(h, m.Wq, M::NQ * M::HD, M::Hd, m.sq, qp);
        auto k = bitlinear_ref(h, m.Wk, M::NKV * M::HD, M::Hd, m.sk, qp);
        auto v = bitlinear_ref(h, m.Wv, M::NKV * M::HD, M::Hd, m.sv, qp);
        for (int hh = 0; hh < M::NQ; ++hh) rope_ref(q.data() + hh * M::HD, M::HD, t, theta);
        for (int hh = 0; hh < M::NKV; ++hh) rope_ref(k.data() + hh * M::HD, M::HD, t, theta);
        K.push_back(k); Vv.push_back(v);
        std::vector<float> a(M::NQ * M::HD, 0.0f);
        for (int hh = 0; hh < M::NQ; ++hh) {
            const int kv = hh / (M::NQ / M::NKV); // repeat_kv
            std::vector<double> sc(t + 1);
            double mx = -1e300;
            for (int s = 0; s <= t; ++s) {
                double d = 0;
                for (int i = 0; i < M::HD; ++i) d += double(q[hh * M::HD + i]) * K[s][kv * M::HD + i];
                sc[s] = d / std::sqrt(double(M::HD)); mx = std::max(mx, sc[s]);
            }
            double z = 0;
            for (auto& s : sc) { s = std::exp(s - mx); z += s; }
            for (int s = 0; s <= t; ++s)
                for (int i = 0; i < M::HD; ++i) a[hh * M::HD + i] += float(sc[s] / z * Vv[s][kv * M::HD + i]);
        }
        auto o = bitlinear_ref(rms(a, m.n_attn), m.Wo, M::Hd, M::NQ * M::HD, m.so, qp);
        for (int i = 0; i < M::Hd; ++i) x[i] += o[i];
        auto p = rms(x, m.n_post);
        auto gt = bitlinear_ref(p, m.Wg, M::F, M::Hd, m.sg, qp);
        auto up = bitlinear_ref(p, m.Wu, M::F, M::Hd, m.su, qp);
        std::vector<float> act(M::F);
        for (int i = 0; i < M::F; ++i) { float r = std::max(0.0f, gt[i]); act[i] = r * r * up[i]; }
        auto d = bitlinear_ref(rms(act, m.n_ffn), m.Wd, M::Hd, M::F, m.sd, qp);
        for (int i = 0; i < M::Hd; ++i) x[i] += d[i];
        auto f = rms(x, m.n_final);
        std::vector<float> lg(M::V);
        for (int vv = 0; vv < M::V; ++vv) {
            double acc = 0;
            for (int i = 0; i < M::Hd; ++i) acc += double(f[i]) * m.E[size_t(vv) * M::Hd + i];
            lg[vv] = float(acc);
        }
        out.push_back(lg);
    }
    return out;
}

bool test_engine_tiny_model() {
    using M = TinyModel;
    TinyModel m;
    tenzo_config_t cfg = tenzo_default_config();
    cfg.hidden_size = M::Hd; cfg.num_q_heads = M::NQ; cfg.num_kv_heads = M::NKV; cfg.head_dim = M::HD;
    cfg.num_layers = 1; cfg.ffn_dim = M::F; cfg.vocab_size = M::V; cfg.max_seq_len = 16; cfg.kv_mode = "fp32";
    cfg.rope_theta = 500000.0f;
    tenzo_engine_t eng = tenzo_create_engine(&cfg);
    if (!eng) { llvm::errs() << "  ❌ engine creation failed\n"; return false; }
    auto pq = pack_tl1(m.Wq, M::NQ * M::HD, M::Hd), pk = pack_tl1(m.Wk, M::NKV * M::HD, M::Hd),
         pv = pack_tl1(m.Wv, M::NKV * M::HD, M::Hd), po = pack_tl1(m.Wo, M::Hd, M::NQ * M::HD),
         pg = pack_tl1(m.Wg, M::F, M::Hd), pu = pack_tl1(m.Wu, M::F, M::Hd), pd = pack_tl1(m.Wd, M::Hd, M::F);
    tenzo_layer_weights_t lw{};
    lw.q_w = pq.data(); lw.q_scale = m.sq; lw.k_w = pk.data(); lw.k_scale = m.sk; lw.v_w = pv.data(); lw.v_scale = m.sv;
    lw.out_w = po.data(); lw.out_scale = m.so; lw.gate_w = pg.data(); lw.gate_scale = m.sg;
    lw.up_w = pu.data(); lw.up_scale = m.su; lw.down_w = pd.data(); lw.down_scale = m.sd;
    lw.in_norm = m.n_in.data(); lw.attn_sub_norm = m.n_attn.data(); lw.post_norm = m.n_post.data();
    lw.ffn_sub_norm = m.n_ffn.data();
    bool ok = tenzo_set_layer_weights(eng, 0, &lw) == TENZO_SUCCESS && tenzo_set_final_norm(eng, m.n_final.data()) == TENZO_SUCCESS &&
              tenzo_set_lm_head_f32(eng, m.E.data(), M::V, M::Hd) == TENZO_SUCCESS;
    if (!ok) { llvm::errs() << "  ❌ engine weight setup failed\n"; tenzo_destroy_engine(eng); return false; }
    tenzo_reset(eng);
    std::vector<std::vector<float>> got;
    for (int t = 0; t < M::T; ++t) {
        std::vector<float> lg(M::V);
        if (tenzo_forward_logits(eng, m.tokens[t], lg.data(), M::V) != TENZO_SUCCESS) {
            llvm::errs() << "  ❌ tenzo_forward_logits failed\n"; tenzo_destroy_engine(eng); return false;
        }
        got.push_back(lg);
    }
    tenzo_destroy_engine(eng);
    auto ref = tiny_ref(m, 500000.0, 127.0f), bug_theta = tiny_ref(m, 10000.0, 127.0f), bug_act = tiny_ref(m, 500000.0, 31.0f);
    double worst = 0, bt = 0, ba = 0;
    for (int t = 0; t < M::T; ++t) {
        worst = std::max(worst, rel_l2(got[t], ref[t]));
        bt = std::max(bt, rel_l2(bug_theta[t], ref[t]));
        ba = std::max(ba, rel_l2(bug_act[t], ref[t]));
    }
    llvm::outs() << "  engine 1-layer fp32-KV, 8 positions: max rel_l2(logits) vs HF ref " << worst << " (theta=1e4 would be " << bt
                 << ", 31-level activations " << ba << ")\n";
    if (!(worst <= kTol)) { llvm::errs() << "  ❌ engine diverges from HF semantics\n"; return false; }
    if (!(bt > 100 * kTol) || !(ba > 100 * kTol)) { llvm::errs() << "  ❌ test does not discriminate\n"; return false; }
    return true;
}

} // namespace

bool runHFSemanticsTests(MLIRContext& context) {
    llvm::outs() << "--- HF semantics regressions (rope_theta, 8-bit ActQuant) ---\n";
    bool ok = true;
    ok &= test_rope_lowering(context);
    ok &= test_tl1_pack_lowering(context);
    ok &= test_tl1_lowering(context);
    ok &= test_engine_tiny_model();
    llvm::outs() << (ok ? "✅ [PASS] HF semantics regressions\n" : "❌ [FAIL] HF semantics regressions\n");
    return ok;
}

} // namespace tenzo
