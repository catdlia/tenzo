# BitNet-b1.58-2B-4T Tokenizer (`tokenizer.json`)

`tools/tokenizer/bitnet_b158_tokenizer.json` (8.7 MB) was removed from git tracking in task **L1 / T43** to comply with the repository `< 5 MB` tracked-file policy and to avoid redistributing third-party Llama-3-derived tokenizer assets directly in an Apache-2.0 source tree.

## Upstream Source

- **HuggingFace repository:** `microsoft/bitnet-b1.58-2B-4T`
- **Revision (commit SHA):** `04c3b9ad9361b824064a1f25ea60a8be9599b127`
- **File:** `tokenizer.json`
- **SHA-256:** `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`

## Staging for Local Engine Runs

`tenzo-inference` automatically loads `tokenizer.json` from the model directory (or via `TENZO_TOKENIZER_JSON`). Copy it from the local HuggingFace cache into `models/bitnet-1.58b-v2/`:

```bash
cp ~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/04c3b9ad9361b824064a1f25ea60a8be9599b127/tokenizer.json \
   models/bitnet-1.58b-v2/tokenizer.json
sha256sum models/bitnet-1.58b-v2/tokenizer.json
# Expected: e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7
```

Or download directly via `huggingface_hub`:

```bash
python3 -c '
import shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download("microsoft/bitnet-b1.58-2B-4T", "tokenizer.json",
                    revision="04c3b9ad9361b824064a1f25ea60a8be9599b127")
shutil.copyfile(p, "models/bitnet-1.58b-v2/tokenizer.json")
'
```
