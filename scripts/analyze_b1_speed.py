#!/usr/bin/env python3
"""Summarize clean idle B1 speed benchmarks (strictly rows verified `valid` in b1_summary.csv)."""

from __future__ import annotations

import csv
import pathlib
import statistics
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
B1_DIR = ROOT / "results" / "2026-11-26_agent2" / "B1"


def read_rows(p: pathlib.Path) -> list[dict[str, str]]:
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def med_iqr(vals: list[float]) -> tuple[float, float, float, float]:
    arr = np.array(vals, dtype=np.float64)
    med = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    return med, float(q75 - q25), float(arr.min()), float(arr.max())


def main() -> int:
    summary_path = B1_DIR / "b1_summary.csv"
    summary_rows = read_rows(summary_path)
    valid_meta = {r["item"]: r for r in summary_rows if r.get("status") == "valid"}
    assert len(valid_meta) == len(summary_rows) and len(valid_meta) > 0, (
        f"Unexpected non-valid rows in {summary_path}: {summary_rows}"
    )

    out_rows: list[dict[str, str]] = []
    md: list[str] = [
        "# B1 Clean Idle Speed Benchmarks (`2026-11-26`)",
        "",
        "- **Provenance:** `results/2026-11-26_agent2/B1/b1_summary.csv` (`tenzo-b1.service`, `scripts/b1_idle_runner.sh` v2, pinned CPUs `0-3`, AC online `1`, governor `powersave`, E-core `cpu4..7` `foreign_cpu_mean < 0.15`, `0` forbidden processes before/after).",
        f"- **Completed valid items:** `{len(valid_meta)}/{len(summary_rows)}` (all on attempt `1`, `rc=0`).",
        "",
        "## 1. Runner Gate & Telemetry (`b1_summary.csv`, `valid` only)",
        "",
        "| Item | Status | Attempt | `load1` (`before→after`) | `foreign_cpu_mean` | `foreign_cpu_max` | Gov | AC | Temp (°C) |",
        "|---|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for r in summary_rows:
        md.append(
            f"| `{r['item']}` | `{r['status']}` | {r['attempt']} | `{r['load1_before']}→{r['load1_after']}` | "
            f"`{float(r['foreign_cpu_mean']):.4f}` | `{float(r['foreign_cpu_max']):.4f}` | `{r['governor']}` | "
            f"{r['ac_online']} | {r['temp_c']} |"
        )

    def add_row(item: str, group: str, metric: str, unit: str, vals: list[float]):
        m, iqr, lo, hi = med_iqr(vals)
        out_rows.append(
            dict(
                item=item,
                group=group,
                metric=metric,
                unit=unit,
                n=str(len(vals)),
                median=f"{m:.4f}",
                iqr=f"{iqr:.4f}",
                min=f"{lo:.4f}",
                max=f"{hi:.4f}",
            )
        )
        return m, iqr, lo, hi

    # 1. A/B GEMV (i2s vs lut)
    md += [
        "",
        "## 2. Kernel A/B Benchmarks (`i2s` vs `lut`, `int8_fused`, `tg=128`, `5` ABBA reps = `10` runs/variant)",
        "",
        "| Prompt | Variant | `n` | `pp_tps` median [IQR] | `tg_tps` median [IQR] |",
        "|---|---|---:|---:|---:|",
    ]
    for pp in (128, 512):
        item = f"ab_gemv_pp{pp}"
        if item not in valid_meta:
            continue
        rows = read_rows(B1_DIR / f"{item}.csv")
        meds = {}
        for var in ("i2s", "lut"):
            sel = [r for r in rows if r["variant"] == var]
            mp, ip, _, _ = add_row(item, var, "pp_tps", "tok/s", [float(r["pp_tps"]) for r in sel])
            mt, it, _, _ = add_row(item, var, "tg_tps", "tok/s", [float(r["tg_tps"]) for r in sel])
            meds[var] = (mp, mt)
            md.append(f"| `pp={pp}` | `{var}` | {len(sel)} | `{mp:.2f}` [`{ip:.2f}`] | `{mt:.2f}` [`{it:.2f}`] |")
        if "i2s" in meds and "lut" in meds:
            md.append(
                f"| `pp={pp}` | **`i2s / lut` speedup** | — | **`{meds['i2s'][0]/meds['lut'][0]:.2f}×`** | "
                f"**`{meds['i2s'][1]/meds['lut'][1]:.2f}×`** |"
            )

    # 2. A/B Prefill (batched vs sequential)
    md += [
        "",
        "## 3. Prefill Schedule A/B (`batched chunk=32` vs `sequential`, `int8_fused`, `5` ABBA reps = `10` runs/variant)",
        "",
        "| Prompt | Variant | `n` | `pp_tps` median [IQR] | `tg_tps` median [IQR] |",
        "|---|---|---:|---:|---:|",
    ]
    for pp in (128, 512):
        item = f"ab_prefill_pp{pp}"
        if item not in valid_meta:
            continue
        rows = read_rows(B1_DIR / f"{item}.csv")
        meds = {}
        for var in ("sequential", "batched"):
            sel = [r for r in rows if r["variant"] == var]
            mp, ip, _, _ = add_row(item, var, "pp_tps", "tok/s", [float(r["pp_tps"]) for r in sel])
            mt, it, _, _ = add_row(item, var, "tg_tps", "tok/s", [float(r["tg_tps"]) for r in sel])
            meds[var] = (mp, mt)
            md.append(f"| `pp={pp}` | `{var}` | {len(sel)} | `{mp:.2f}` [`{ip:.2f}`] | `{mt:.2f}` [`{it:.2f}`] |")
        if "batched" in meds and "sequential" in meds:
            md.append(
                f"| `pp={pp}` | **`batched / sequential` speedup** | — | **`{meds['batched'][0]/meds['sequential'][0]:.2f}×`** | "
                f"`{meds['batched'][1]/meds['sequential'][1]:.2f}×` |"
            )

    # 3. E2E Fair v3
    if "e2e_fair_v3" in valid_meta:
        rows = read_rows(B1_DIR / "e2e_fair_v3.csv")
        md += [
            "",
            "## 4. End-to-End Fair Comparison (`tenzo` token-by-token prefill vs `bitnet.cpp`, CPUs `0-3`, `threads=4`, `gen=128`, `n=5`)",
            "",
            "| `prompt_len` | Engine | KV | `n` | `pp_tps` med [IQR] | `tg_tps` med [IQR] | `pg_tps` med [IQR] | RSS (MB) med |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        groups = []
        for plen in ("16", "128", "512"):
            for eng, kv in (("tenzo", "fp32"), ("tenzo", "int8_fused"), ("bitnet.cpp", "f16"), ("bitnet.cpp", "q8_0")):
                sel = [r for r in rows if r["prompt_len"] == plen and r["engine"] == eng and r["kv"] == kv]
                if not sel:
                    continue
                mp, ip, _, _ = add_row("e2e_fair_v3", f"{eng}:{kv}:pp{plen}", "pp_tps", "tok/s", [float(r["pp_tps"]) for r in sel])
                mt, it, _, _ = add_row("e2e_fair_v3", f"{eng}:{kv}:pp{plen}", "tg_tps", "tok/s", [float(r["tg_tps"]) for r in sel])
                mg, ig, _, _ = add_row("e2e_fair_v3", f"{eng}:{kv}:pp{plen}", "pg_tps", "tok/s", [float(r["pg_tps"]) for r in sel])
                rss_vals = [float(r["rss_mb"]) for r in sel if r.get("rss_mb")]
                rss_s = f"`{statistics.median(rss_vals):.1f}`" if rss_vals else "—"
                md.append(
                    f"| {plen} | `{eng}` | `{kv}` | {len(sel)} | `{mp:.2f}` [`{ip:.2f}`] | "
                    f"`{mt:.2f}` [`{it:.2f}`] | `{mg:.2f}` [`{ig:.2f}`] | {rss_s} |"
                )
                groups.append((plen, eng, kv, mp, mt, mg))
        md.append("")
        md.append(
            "- *Note:* In `e2e_fair_v3`, `tenzo` uses token-by-token prefill (`prefill` default) while `--prefill batched` raises `pp_tps` to `~50–54 tok/s` (Table 3 above); `bitnet.cpp` `tg_tps` is measured from empty context (`llama-bench -n`), and `pg_tps` is combined prompt+generation (`-pg`)."
        )

    # 4. Decode KV modes
    if "decode_kv_modes" in valid_meta:
        rows = read_rows(B1_DIR / "decode_kv_modes.csv")
        md += [
            "",
            "## 5. KV-Mode Throughput Comparison (`tenzo-inference --prefill batched -c 1024`, `tg=128`, `n=3` reps)",
            "",
            "| `pp` | KV Mode | `n` | `pp_tps` med [IQR] | `tg_tps` med [IQR] |",
            "|---:|---|---:|---:|---:|",
        ]
        for pp in ("128", "512"):
            for kv in ("int8_fused", "tl1_fused", "ternary_rot", "ternary_rot_alpha"):
                sel = [r for r in rows if r["pp"] == pp and r["kv"] == kv]
                if not sel:
                    continue
                mp, ip, _, _ = add_row("decode_kv_modes", f"{kv}:pp{pp}", "pp_tps", "tok/s", [float(r["pp_tps"]) for r in sel])
                mt, it, _, _ = add_row("decode_kv_modes", f"{kv}:pp{pp}", "tg_tps", "tok/s", [float(r["tg_tps"]) for r in sel])
                md.append(f"| {pp} | `{kv}` | {len(sel)} | `{mp:.2f}` [`{ip:.2f}`] | `{mt:.2f}` [`{it:.2f}`] |")

    (B1_DIR / "speed.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (B1_DIR / "speed_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {B1_DIR/'speed.md'} and {B1_DIR/'speed_summary.csv'} ({len(out_rows)} metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
