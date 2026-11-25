#!/usr/bin/env python3
"""Tokenizer fertility on Ukrainian vs English text (T20, coordinator, 2026-11-25).

Compares the tokenizers available locally in models/ plus a small byte-level BPE trained here as a stand-in for
"our own" Ukrainian+English vocabulary. Texts: the committed H2 corpora (Wikipedia en/uk, FineWeb-Edu en,
Kobza uk). The own BPE is trained on Wikipedia rows 0-79 of each language and evaluated on the held-out rows 80-99
and on the web corpora, so its numbers are not measured on training text.

Metrics per (tokenizer, corpus): tokens per whitespace word, characters per token, and the UK/EN token ratio on
Wikipedia (not parallel text, so this ratio is indicative only; FLORES gives the content-matched value).

Usage: python3 scripts/tokenizer_fertility.py [--vocab 32000] [--out results/2026-11-25_coordinator/tokenizer_fertility]
Needs: pip package `tokenizers`.
"""
from __future__ import annotations
import argparse, csv, json, pathlib
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORP = ROOT / "results/2026-11-23_agent2/H2/corpora"
TOKENIZERS = {
    "llama3_128k (BitNet-2B-4T)": ROOT / "models/bitnet-1.58b-v2/tokenizer.json",
    "llama2_32k (TinyLlama)": ROOT / "models/gptq_tinyllama/tokenizer.json",
}


def rows(name):
    d = json.load(open(CORP / name, encoding="utf-8"))
    return [r["row"]["text"] for r in d["rows"]]


def stats(tok, texts):
    n_tok = sum(len(tok.encode(t, add_special_tokens=False).ids) for t in texts)
    n_word = sum(len(t.split()) for t in texts)
    n_char = sum(len(t) for t in texts)
    return n_tok, n_word, n_char


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--out", default="results/2026-11-25_coordinator/tokenizer_fertility")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    uk_w, en_w = rows("uk_wiki_0_99.json"), rows("en_wiki_0_99.json")
    evals = {"uk_wiki_80_99": uk_w[80:], "en_wiki_80_99": en_w[80:],
             "uk_web (Kobza)": rows("uk_web_200_299.json"), "en_web (FineWeb-Edu)": rows("en_web_0_99.json")}

    toks = {name: Tokenizer.from_file(str(p)) for name, p in TOKENIZERS.items() if p.exists()}
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, min_frequency=2, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    bpe.train_from_iterator(uk_w[:80] + en_w[:80], trainer=trainer)
    toks[f"own_bpe_{a.vocab // 1000}k (uk+en wiki, 160 articles)"] = bpe

    rows_out = []
    for tname, tok in toks.items():
        per = {}
        for cname, texts in evals.items():
            n_tok, n_word, n_char = stats(tok, texts)
            per[cname] = n_tok
            rows_out.append(dict(tokenizer=tname, vocab=tok.get_vocab_size(), corpus=cname, tokens=n_tok, words=n_word,
                                 chars=n_char, tokens_per_word=round(n_tok / n_word, 3), chars_per_token=round(n_char / n_tok, 3)))
    with open(out / "tokenizer_fertility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    json.dump(dict(script="scripts/tokenizer_fertility.py", vocab=a.vocab, own_bpe_train="uk_wiki rows 0-79 + en_wiki rows 0-79",
                   tokenizers={k: str(v.relative_to(ROOT)) for k, v in TOKENIZERS.items()}),
              open(out / "meta.json", "w"), indent=1)
    for r in rows_out:
        print(f"{r['tokenizer']:44s} {r['corpus']:22s} tok/word {r['tokens_per_word']:5.2f}  chars/tok {r['chars_per_token']:5.2f}")


if __name__ == "__main__":
    main()
