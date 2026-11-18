// bitnet_ref_dump.cpp - reference dumper on top of libllama (BitNet fork in BitNet/3rdparty/llama.cpp).
// Modes (all token-id driven; the tokenizer is only used by `tokenize`):
//   tokenize <model.gguf> <text>                       -> prints "TOKENS: id id ..." (BOS added, special parsing off)
//   prompt   <model.gguf> <ids_file> <out_prefix> [steps=32]
//                                                      -> <prefix>.logits.f32 (V floats, last prompt position),
//                                                         <prefix>.greedy.txt (argmax, no stop tokens, no penalties)
//   nll      <model.gguf> <windows_file> <out_file>    -> same text format as `tenzo-inference --score-out`
// Build: tools/reference/build_ref_dump.sh   (host build, links BitNet/build/bin/libllama.so)
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

static std::vector<std::vector<int>> read_id_lines(const char* path) {
    std::vector<std::vector<int>> out; std::ifstream f(path); std::string line;
    while (std::getline(f, line)) { std::istringstream iss(line); std::vector<int> ids; int v;
        while (iss >> v) ids.push_back(v); if (!ids.empty()) out.push_back(ids); }
    return out;
}
static double log_prob_of(const float* lg, int V, int target) {
    float mx = lg[0]; for (int i = 1; i < V; ++i) mx = std::max(mx, lg[i]);
    double s = 0; for (int i = 0; i < V; ++i) s += std::exp((double)lg[i] - mx);
    return ((double)lg[target] - mx) - std::log(s);
}
static int argmax(const float* lg, int V) { int b = 0; for (int i = 1; i < V; ++i) if (lg[i] > lg[b]) b = i; return b; }

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: see header comment\n"); return 2; }
    std::string mode = argv[1]; const char* model_path = argv[2];
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int V = llama_vocab_n_tokens(vocab);

    if (mode == "tokenize") {
        std::string text = argv[3];
        std::vector<llama_token> toks(text.size() + 8);
        int n = llama_tokenize(vocab, text.c_str(), (int)text.size(), toks.data(), (int)toks.size(), /*add_special=*/true, /*parse_special=*/false);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        printf("TOKENS:"); for (int i = 0; i < n; ++i) printf(" %d", toks[i]); printf("\n");
        return 0;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 1024; cp.n_threads = 4; cp.n_threads_batch = 4;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    auto forward_all = [&](std::vector<int>& ids, bool all_logits_out, std::vector<std::vector<float>>* all, std::vector<float>* last) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_init((int)ids.size(), 0, 1);
        b.n_tokens = (int)ids.size();
        for (int i = 0; i < b.n_tokens; ++i) { b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (all_logits_out || i == b.n_tokens - 1) ? 1 : 0; }
        int rc = llama_decode(ctx, b);
        if (rc != 0) { fprintf(stderr, "llama_decode rc=%d\n", rc); exit(1); }
        if (all_logits_out) { for (int i = 0; i < b.n_tokens; ++i) all->emplace_back(llama_get_logits_ith(ctx, i), llama_get_logits_ith(ctx, i) + V); }
        else { const float* l = llama_get_logits_ith(ctx, b.n_tokens - 1); last->assign(l, l + V); }
        llama_batch_free(b);
    };

    if (mode == "prompt") {
        auto seqs = read_id_lines(argv[3]); std::string prefix = argv[4]; int steps = argc > 5 ? atoi(argv[5]) : 32;
        std::vector<int> ids = seqs.at(0); std::vector<float> last;
        forward_all(ids, false, nullptr, &last);
        { std::ofstream f(prefix + ".logits.f32", std::ios::binary); f.write((const char*)last.data(), sizeof(float) * V); }
        std::ofstream g(prefix + ".greedy.txt");
        for (int k = 0; k < steps; ++k) {      // no KV tricks: re-run the full sequence each step (slow but unambiguous)
            int nxt = argmax(last.data(), V); g << nxt << (k + 1 < steps ? " " : "\n");
            if (k + 1 < steps) { ids.push_back(nxt); forward_all(ids, false, nullptr, &last); }
        }
        printf("DUMP_OK prefix=%s prompt_tokens=%zu\n", prefix.c_str(), seqs[0].size());
    } else if (mode == "nll") {
        auto wins = read_id_lines(argv[3]); std::ofstream of(argv[4]); of.precision(9);
        of << "# bitnet.cpp(libllama) i2_s windows=" << wins.size() << "\n";
        for (size_t w = 0; w < wins.size(); ++w) {
            std::vector<std::vector<float>> all; forward_all(wins[w], true, &all, nullptr);
            std::vector<double> lps; double nll = 0;
            for (size_t i = 0; i + 1 < wins[w].size(); ++i) { double lp = log_prob_of(all[i].data(), V, wins[w][i + 1]); lps.push_back(lp); nll -= lp; }
            of << "w " << w << " " << lps.size() << " " << nll << "\nlp"; for (double lp : lps) of << " " << lp; of << "\n"; of.flush();
            printf("window %zu ppl %.3f\n", w, std::exp(nll / lps.size())); fflush(stdout);
        }
    } else { fprintf(stderr, "unknown mode\n"); return 2; }
    llama_free(ctx); llama_model_free(model);
    return 0;
}
