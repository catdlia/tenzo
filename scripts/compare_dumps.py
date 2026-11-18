#!/usr/bin/env python3
"""Layer-by-layer divergence: compare an engine dump dir (TENZO_DUMP_DIR) against a golden dump dir
(tools/reference/golden_bitnet.py --dump-dir). Both carry manifest.json with {name, layer, pos, file}.

For every tensor present in both, in computation order: rel L2, max_abs, cosine. Noise floor = median rel L2 of
layer-0 input_norm/q/k/v (the earliest tensors). Reports the first tensor whose rel L2 exceeds the floor by >= 10x.
Outputs <out>.md (table), <out>.csv and <out>.png (error vs computation step, one line per position).
"""
import argparse, csv, json, os
import numpy as np

ORDER = ["embed", "input_norm", "q", "k", "v", "q_rope", "k_rope", "attn_out", "attn_sub_norm", "o_proj", "resid_attn",
         "post_attn_norm", "gate", "up", "act", "ffn_sub_norm", "down", "resid_ffn", "final_norm", "logits"]


def load(d):
    m = json.load(open(os.path.join(d, "manifest.json")))["tensors"]
    return {(t["layer"], t["name"], t["pos"]): os.path.join(d, t["file"]) for t in m}


def step_key(k):
    layer, name, pos = k
    if name == "embed":
        return (-1, 0)
    if layer is None:
        return (10 ** 6, ORDER.index(name))
    return (layer, ORDER.index(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("golden")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    E, G = load(a.engine), load(a.golden)
    keys = sorted([k for k in E if k in G], key=lambda k: (step_key(k), k[2]))
    rows = []
    for k in keys:
        x = np.fromfile(E[k], np.float32).astype(np.float64)
        y = np.fromfile(G[k], np.float32).astype(np.float64)
        if x.size != y.size:
            rows.append(dict(layer=k[0], name=k[1], pos=k[2], rel_l2=float("nan"), max_abs=float("nan"),
                             cos=float("nan"), note="size %d vs %d" % (x.size, y.size)))
            continue
        rel = np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30)
        cos = x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30)
        rows.append(dict(layer=k[0], name=k[1], pos=k[2], rel_l2=rel, max_abs=float(abs(x - y).max()), cos=cos, note=""))
    early = [r["rel_l2"] for r in rows if r["layer"] == 0 and r["name"] in ("input_norm", "q", "k", "v")]
    floor = float(np.median(early)) if early else float("nan")
    first = None
    for r in rows:
        if np.isfinite(r["rel_l2"]) and floor > 0 and r["rel_l2"] >= 10 * floor:
            first = r; break
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(a.out + ".md", "w") as f:
        f.write("# %s\n\nengine: `%s`  golden: `%s`\n\n" % (a.title or "dump comparison", a.engine, a.golden))
        f.write("noise floor (median rel L2 of L0 input_norm/q/k/v): %.3e\n\n" % floor)
        if first:
            f.write("first tensor with rel L2 >= 10x floor: **L%s.%s pos %d** (rel L2 %.3e)\n\n" % (
                first["layer"], first["name"], first["pos"], first["rel_l2"]))
        else:
            f.write("no tensor exceeds 10x the noise floor\n\n")
        f.write("| layer | tensor | pos | rel_l2 | max_abs | cos | note |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| %s | %s | %d | %.3e | %.3e | %.6f | %s |\n" % (
                "-" if r["layer"] is None else r["layer"], r["name"], r["pos"], r["rel_l2"], r["max_abs"], r["cos"], r["note"]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = sorted({step_key((r["layer"], r["name"], r["pos"])) for r in rows})
        idx = {s: i for i, s in enumerate(steps)}
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for p in sorted({r["pos"] for r in rows}):
            pr = [r for r in rows if r["pos"] == p and np.isfinite(r["rel_l2"])]
            ax.semilogy([idx[step_key((r["layer"], r["name"], r["pos"]))] for r in pr], [max(r["rel_l2"], 1e-9) for r in pr],
                        marker=".", lw=1, label="pos %d" % p)
        lbl = []
        for s in steps:
            lbl.append("emb" if s[0] == -1 else ("head:" + ORDER[s[1]] if s[0] == 10 ** 6 else "L%d.%s" % (s[0], ORDER[s[1]])))
        ax.set_xticks(range(len(steps))); ax.set_xticklabels(lbl, rotation=90, fontsize=6)
        ax.axhline(floor, color="gray", ls="--", lw=0.8); ax.set_ylabel("rel L2 vs golden")
        ax.set_title(a.title); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(a.out + ".png", dpi=110)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print("floor %.3e first %s" % (floor, None if not first else "L%s.%s p%d %.3e" % (first["layer"], first["name"],
                                                                                         first["pos"], first["rel_l2"])))


if __name__ == "__main__":
    main()
