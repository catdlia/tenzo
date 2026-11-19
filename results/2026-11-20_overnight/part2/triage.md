# Part 2 triage: what part 1 left unfinished (state at 02:27)

Part-2 letters refer to the part-2 prompt's list (A = I2_S, B = gate v2, C = bitnet.cpp baseline, D = e2e v3, E = PTQ r2, F = tokenizer).
The part-1 prompt as received was truncated in block A (part-1 decision log D1), so B (gate v2) and F (tokenizer) were never specified to this run.

| item | status after part 1 | why | part-2 action |
|---|---|---|---|
| A I2_S kernel, correctness | done: bit-identical to LUT on logits, greedy, scores, G6; hf-semantics ok | – | none |
| A I2_S speed A/B | not measured | clean-machine rule failed 3/3 (browsers, load 4-13) | pending queue (`scripts/pending_benchmarks.sh`) |
| C bitnet.cpp baseline | done: c17d1c5 + llama.cpp 5eb47b7, PPL within CI of golden EN/UK, greedy == HF | – | none |
| D e2e v3 vs bitnet.cpp | not measured | clean-machine rule | pending queue |
| E PTQ round 2 | screening done (4 EN + 4 UK windows) | finalists on 16 EN + 32 UK and ctx 2048 not run (time; part 2 arrived) | 0c: finalists on 16 EN + 32 UK (needs 32 UK eval windows: only 8 exist, see D8) |
| B gate v2 | not started | spec not received (truncated prompt) | skipped, recorded as blocker |
| F tokenizer | not started | spec not received; lower priority | skipped unless time remains |
