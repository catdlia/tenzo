#!/usr/bin/env python3
"""Markdown summary of verify_model_parity.py JSON files (one per kv mode). Usage: summarize_parity.py OUT.md A.json [B.json ...]"""
import json, sys
out, files = sys.argv[1], sys.argv[2:]
L = ["# M4 parity v2: Tenzo (after fixes 1+2) vs golden fp32 and HF bf16", "",
     "First-step logits on 7 prompts, Tenzo driven with the reference token ids. Gates unchanged (scripts/verify_model_parity.py):",
     "cosine > 0.999, max_abs <= 0.25, top-1 equal. greedy = matching prefix length of the 32-token greedy continuation.", "",
     "| kv mode | prompt | ref | cosine | max_abs | top-1 | top-5 | KL(ref||tenzo) | greedy | gate |", "|---|---|---|---|---|---|---|---|---|---|"]
summary = {}
for f in files:
    d = json.load(open(f)); kv = d["kv_quant"]
    for pid, pr in d["prompts"].items():
        for src, m in pr.items():
            if not isinstance(m, dict) or "cosine_similarity" not in m:
                continue
            L.append("| %s | %s | %s | %.5f | %.3f | %s | %.1f | %.5f | %d/%d | %s |" % (kv, pid, src, m["cosine_similarity"],
                     m["max_abs_diff"], m["top1_match"], m["top5_agreement"], m["kl_ref_to_tenzo"], m["greedy_prefix_match"],
                     m["greedy_len"], "PASS" if m["gate_pass"] else "FAIL"))
            s = summary.setdefault((kv, src), [0, 0, 0, 0])
            s[0] += 1; s[1] += m["gate_pass"]; s[2] += m["top1_match"]; s[3] += m["gate_cosine"]
L += ["", "| kv mode | ref | prompts | gate PASS | top-1 match | cosine gate |", "|---|---|---|---|---|---|"]
for (kv, src), s in summary.items():
    L.append("| %s | %s | %d | %d | %d | %d |" % (kv, src, *s))
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L[-len(summary) - 2:]))
