#!/usr/bin/env python3
"""2026-11-20 windows: 32 UK eval windows (superset of windows_uk.ids, same procedure on the same 39 Kobza first-rows
documents) and calibration windows that do not overlap any eval set:
  calib_en8.ids  WikiText-2 raw TRAIN rows 0-599 (datasets-server rows API), concatenated, first 8 x 511 tokens
  calib_uk8.ids  Kobza train rows 100-199 (datasets-server rows API; eval uses rows 0-38), first 8 x 511 tokens
Windows = [BOS] + 511 text tokens (HF tokenizer of microsoft/bitnet-b1.58-2B-4T, no special tokens).
No-overlap proof written to calib_windows.json: per-window sha256 of the id sequence, and the number of shared
32-token n-grams between every calibration set and every eval set (must be 0)."""
import glob, hashlib, json, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
R = "results/reference"
tok = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
B = 511


def windows(ids, n, start=0):
    assert len(ids) >= start + n * B, (len(ids), n)
    return [[tok.bos_token_id] + ids[start + w * B:start + (w + 1) * B] for w in range(n)]


def write(name, ws):
    with open(os.path.join(R, name), "w") as f:
        for w in ws: f.write(" ".join(map(str, w)) + "\n")


def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


kobza = json.load(open(os.path.join(R, "kobza_first_rows.json")))["rows"]
uk_ids = tok("\n\n".join(r["row"]["text"] for r in kobza), add_special_tokens=False).input_ids
uk32 = windows(uk_ids, 32)
old8 = [list(map(int, l.split())) for l in open(os.path.join(R, "windows_uk.ids")) if l.strip()]
assert uk32[:8] == old8, "first 8 UK eval windows must equal windows_uk.ids"
write("windows_uk32.ids", uk32)

kc = json.load(open(os.path.join(R, "kobza_rows_100_199.json")))["rows"]
assert min(r["row_idx"] for r in kc) == 100 and max(r["row_idx"] for r in kc) == 199
cal_uk = windows(tok("\n\n".join(r["row"]["text"] for r in kc), add_special_tokens=False).input_ids, 8)
write("calib_uk8.ids", cal_uk)

wt_rows = []
for f in sorted(glob.glob(os.path.join(R, "wt2_train_rows", "wt2_train_*.json")), key=lambda p: int(p.rsplit("_", 1)[1][:-5])):
    wt_rows += json.load(open(f))["rows"]
wt_rows.sort(key=lambda r: r["row_idx"])
cal_en = windows(tok("".join(r["row"]["text"] for r in wt_rows), add_special_tokens=False).input_ids, 8)
write("calib_en8.ids", cal_en)

en16 = [list(map(int, l.split())) for l in open(os.path.join(R, "windows.ids")) if l.strip()]


def grams(ws, n=32):
    s = set()
    for w in ws:
        for i in range(1, len(w) - n + 1): s.add(tuple(w[i:i + n]))
    return s


evals = {"windows.ids(EN16)": en16, "windows_uk32.ids(UK32)": uk32}
cals = {"calib_en8.ids": cal_en, "calib_uk8.ids": cal_uk}
overlap = {c: {e: len(grams(cw) & grams(ew)) for e, ew in evals.items()} for c, cw in cals.items()}
hw = lambda ws: [hashlib.sha256(" ".join(map(str, w)).encode()).hexdigest()[:16] for w in ws]
meta = {"kobza_first_rows_sha256": sha(os.path.join(R, "kobza_first_rows.json")),
        "kobza_rows_100_199_sha256": sha(os.path.join(R, "kobza_rows_100_199.json")),
        "wt2_train_rows_sha256": {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(R, "wt2_train_rows", "*.json")))},
        "uk_eval_text_tokens": len(uk_ids), "uk32_first8_equal_windows_uk": True,
        "files_sha256": {n: sha(os.path.join(R, n)) for n in ("windows.ids", "windows_uk.ids", "windows_uk32.ids", "calib_en8.ids", "calib_uk8.ids")},
        "window_sha16": {"EN16": hw(en16), "UK32": hw(uk32), "CAL_EN8": hw(cal_en), "CAL_UK8": hw(cal_uk)},
        "shared_32gram_counts": overlap}
ws_all = sum(meta["window_sha16"].values(), [])
meta["all_window_hashes_distinct"] = len(ws_all) == len(set(ws_all))
json.dump(meta, open(os.path.join(R, "calib_windows.json"), "w"), indent=1)
print(json.dumps({"shared_32gram_counts": overlap, "distinct": meta["all_window_hashes_distinct"], "uk_tokens": len(uk_ids)}, indent=1))
