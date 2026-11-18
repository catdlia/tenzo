# KV-cache mode quality: teacher-forced perplexity, BitNet-b1.58-2B-4T (generated 2026-11-18)

Protocol (identical for every engine, token ids supplied by the HF tokenizer so tokenisation cannot confound): each window is
`[BOS] + 511 text tokens`, all 511 next-token predictions are scored, PPL = exp(sum NLL / sum tokens), log-softmax in float64.
Tenzo: `tenzo-inference --score` at ctx 512 through the normal decode path, one token at a time, selected `--kv-quant`.
Reference rows: HF transformers bf16 (ground truth; fp32 compute was impossible on this host, see decision log D7) and the
local bitnet.cpp/libllama i2_s build (which is NOT trustworthy: see OVERNIGHT_REPORT.md). Files: ppl_kv.csv, ppl_kv_uk.csv, ppl_kv/*.txt, ppl_kv_uk/*.txt.

## English: WikiText-2
16 fixed windows x 512 tokens (BOS + 511 text tokens, 511 scored each); dataset: WikiText-2 (raw test), 16 windows chosen with numpy RandomState(0). 95% CI = bootstrap over windows (10000 draws, seed 0).

| engine / kv mode | PPL | 95% CI | dPPL vs tenzo fp32 | 95% CI of d | mean abs dlogp vs HF | KV bytes/token | RSS KB (ctx 8192) |
|---|---|---|---|---|---|---|---|
| hf_bf16 | 19.091 | [16.178, 22.467] | -4.040 | [-5.064, -3.164] | - | - | - |
| bitnet_cpp_i2s | 102.051 | [86.759, 117.960] | 78.920 | [66.297, 91.617] | 2.0733 | - | - |
| fp32 | 23.131 | [19.555, 27.219] | 0.000 | [0.000, 0.000] | 0.5574 | 153600 | 2405136 |
| int8_fused | 23.221 | [19.580, 27.403] | 0.089 | [-0.094, 0.296] | 0.5624 | 39600 | 1493628 |
| popcount_fused | 174.977 | [146.046, 208.029] | 151.846 | [125.441, 182.386] | 2.4937 | 10800 | 1263104 |
| tl1_fused | 59.420 | [50.120, 70.386] | 36.288 | [30.445, 43.256] | 1.4641 | 10800 | 1263104 |

## Ukrainian: Kobza slice (8 windows)
8 fixed windows x 512 tokens (BOS + 511 text tokens, 511 scored each); dataset: Goader/kobza first 39 documents (datasets-server first-rows), consecutive windows. 95% CI = bootstrap over windows (10000 draws, seed 0).

| engine / kv mode | PPL | 95% CI | dPPL vs tenzo fp32 | 95% CI of d | mean abs dlogp vs HF | KV bytes/token | RSS KB (ctx 8192) |
|---|---|---|---|---|---|---|---|
| hf_bf16 | 142.232 | [112.080, 176.042] | -67.125 | [-91.368, -47.918] | - | - | - |
| fp32 | 209.357 | [159.924, 266.216] | 0.000 | [0.000, 0.000] | 0.9710 | 153600 | 2405136 |
| int8_fused | 206.815 | [158.412, 262.867] | -2.543 | [-6.036, 0.488] | 0.9529 | 39600 | 1493628 |
| popcount_fused | 9075.716 | [5927.721, 13001.283] | 8866.358 | [5766.358, 12752.377] | 4.5438 | 10800 | 1263104 |
| tl1_fused | 2186.047 | [1632.286, 2820.575] | 1976.690 | [1469.135, 2562.826] | 3.2099 | 10800 | 1263104 |

## How to read this
* `RSS KB (ctx 8192)` is VmRSS at the end of a 16-token run with `-c 8192` (ppl_kv/rss_ctx8192.txt; KV buffers are zero-filled at allocation so the whole allocation counts);
  the same process also holds ~1.2 GB of weights/embeddings, so only differences between rows mean anything. `KV bytes/token` is computed from the engine's
  buffer layout (30 layers x 5 KV heads, K and V, scales included).
* int8_fused is statistically indistinguishable from fp32 KV on both sets (CI of the difference contains 0).
* tl1_fused (ternary KV, 14x smaller than fp32) multiplies PPL by about 2.6x on English and about 10x on Ukrainian; popcount_fused is worse (7.6x / 43x).
* Even with an fp32 KV cache Tenzo is above the HF reference (+21% on WikiText-2, +47% on the Ukrainian slice): see ppl_experiment/README.md for what was ruled out.
