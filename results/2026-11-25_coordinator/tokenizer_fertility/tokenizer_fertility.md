# Tokenizer fertility on Ukrainian vs English (T20, coordinator, 2026-11-25)

`scripts/tokenizer_fertility.py` → `tokenizer_fertility.csv`, `meta.json`. Status: exploratory; ran in a container in ~7 s.

**Texts.** The committed H2 corpora:
- Wikipedia en/uk, held-out rows 80–99;
- FineWeb-Edu (en);
- Kobza (uk).

The texts are not parallel, so the UK/EN ratios are indicative; FLORES gives content-matched values.

**Tokenizers.**
- What is available locally: Llama-3 128k (BitNet-2B-4T) and Llama-2 32k (TinyLlama).
- A byte-level BPE with 32k vocabulary, trained here on only 160 Wikipedia articles (Wikipedia rows 0–79 per language) as a stand-in for "our own" vocabulary.
- The Kobza / Modern-LiBERTa tokenizer could not be compared: Hugging Face is not reachable from the coordinator's sandboxes. It is worth adding on the laptop.

| Tokenizer | uk wiki tok/word | en wiki tok/word | uk web (Kobza) | en web (FineWeb-Edu) | uk wiki chars/tok |
|---|---|---|---|---|---|
| Llama-3, 128k | 2.57 | 1.39 | 2.41 | 1.29 | 2.90 |
| Llama-2, 32k | 3.07 | 1.67 | 2.96 | 1.51 | 2.43 |
| own BPE, 32k (160 articles) | **2.15** | 1.51 | **2.24** | 1.46 | **3.47** |

## Reading

- **Ukrainian.** Even this tiny 32k vocabulary needs 16% fewer tokens than Llama-3's 128k vocabulary on held-out Ukrainian Wikipedia, and 7% fewer on Kobza web text. It does so with a quarter of the embedding rows. On English it is 9–13% worse than Llama-3. A vocabulary trained on real Kobza + FineWeb data (GBs, not MBs) should do better on both.
- **What this means for T20 and T21.** A dedicated uk+en vocabulary of ~32–64k is the natural choice for a small (100M) model:
  - the embedding matrix stays small, which matters a lot at 100M parameters;
  - Ukrainian fertility drops below Llama-3's.
- **Link to the KV paper.** A lower Ukrainian fertility also shrinks the token-defined part of the KV-cache disparity, the fertility term in the decomposition.
- **Next step (laptop, HF reachable):** add the Kobza/Modern-LiBERTa tokenizer and a BPE trained on ~1 GB of Kobza + FineWeb-Edu, and measure on FLORES devtest (parallel) rather than Wikipedia.
