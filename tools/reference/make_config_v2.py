#!/usr/bin/env python3
"""models/bitnet-1.58b-v2/config.json = models/bitnet-1.58b/config.json + rope_theta / rms_norm_eps / hidden_act /
tie_word_embeddings copied from the HF config.json of microsoft/bitnet-b1.58-2B-4T (offline cache). Weights are
symlinked, not copied or modified."""
import json, os
src = json.load(open("models/bitnet-1.58b/config.json"))
hf = json.load(open(os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/"
                                       "04c3b9ad9361b824064a1f25ea60a8be9599b127/config.json")))
for k in ("rope_theta", "rms_norm_eps", "hidden_act", "tie_word_embeddings", "max_position_embeddings"):
    src[k] = hf[k]
src["config_source"] = "models/bitnet-1.58b/config.json + HF microsoft/bitnet-b1.58-2B-4T@04c3b9ad config.json"
json.dump(src, open("models/bitnet-1.58b-v2/config.json", "w"), indent=2)
print(json.dumps(src, indent=1))
