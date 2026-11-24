# B0 — Fixing `scripts/b1_idle_runner.sh` (`T38`) Validity Criteria

## 1. Root Cause of Previous Bug

In the previous runner (`agent2/2026-11-24@b683412`), `scripts/b1_idle_runner.sh` checked `/proc/loadavg` `load1 > 1.5` every 5 seconds *while* executing the benchmark item (`DURING_LOAD_THRESH=1.5`). Because `tenzo-inference` (via `docker-compose.yml` `cpuset: "0-3"`) and `llama-bench` (`-t 4` pinned to `0-3`) run with 4 runnable threads, the benchmark's own work pushes 1-minute `load1` up to `~3–4` on a 100% idle laptop, causing false `invalid_load` rejections across 3 retries and premature `gave_up` termination on item 1.

## 2. Verified CPU Topology & Benchmark Pinning (`i3-1215U`)

Verified via `lscpu -e`:
- **P-cores (logical CPUs `0–3`):** CPUs `0, 1` (Physical Core `0`, `MAXMHZ = 4400.0 MHz`) and CPUs `2, 3` (Physical Core `1`, `MAXMHZ = 4400.0 MHz`).
- **E-cores (logical CPUs `4–7`):** CPUs `4, 5, 6, 7` (Physical Cores `2, 3, 4, 5`, `MAXMHZ = 3300.0 MHz`).

Pinning verification across all queue harnesses:
- `docker-compose.yml` configures `cpuset: "0-3"` on service `dev`. This confines:
  - `scripts/ab_gemv_speed.sh` (`docker compose run --rm ... dev /app/cmake-build-debug/tenzo-inference`)
  - `scripts/ab_prefill_speed.sh` (`docker compose run --rm ... dev /app/cmake-build-debug/tenzo-inference`)
  - `scripts/bench_decode_modes.sh` (`docker compose run --rm ... dev /app/cmake-build-debug/tenzo-inference`)
  - `scripts/run_e2e_fair.py` (`run_tenzo` and `--bench-in-container` `run_llama`)
- `scripts/run_e2e_fair.py` host fallback (`--bench-in-container` unset) also uses `taskset -c 0-3 ... -t 4`.
- In addition, `scripts/b1_idle_runner.sh` now wraps each executed command string in `taskset -c 0-3` so host driver scripts (`bash`, `python3`, `docker compose` CLI) are also confined to CPUs `0–3` and leave E-cores `4–7` strictly for measuring unrelated background/foreign activity.

## 3. New Validity Criterion & Queue Logic

1. **Idle pre-gate (`wait_until_idle`, unchanged):**
   - Polls every `30s` until `20` consecutive checks (`10 min`) simultaneously pass: AC online (`1`), `load1 < 1.0`, and zero forbidden processes (`firefox|zen-bin|neko|steam|qemu-system|clion|idea|java`).
2. **Foreign CPU sampling during execution (`2s` interval over `/proc/stat` `cpu4..cpu7`):**
   - For each interval $k$ between consecutive 2-second `/proc/stat` reads summed over `cpu4`, `cpu5`, `cpu6`, `cpu7`:
     $$\text{total} = \text{user} + \text{nice} + \text{system} + \text{idle} + \text{iowait} + \text{irq} + \text{softirq} + \text{steal}, \quad \text{idle\_all} = \text{idle} + \text{iowait}$$
     $$u_k = \frac{\Delta\text{total}_{4..7} - \Delta\text{idle\_all}_{4..7}}{\Delta\text{total}_{4..7}}$$
   - Aggregate metrics recorded per attempt: `foreign_cpu_mean` and `foreign_cpu_max`.
   - `load1_before` and `load1_after` are logged purely as provenance metadata (never used to invalidate).
3. **Attempt validity:**
   - An attempt is **`valid`** iff:
     - `rc == 0`, AND
     - `foreign_cpu_mean < 0.15`, AND
     - zero forbidden processes present immediately before AND immediately after execution (`proc_before == 0 && proc_after == 0`).
   - Otherwise the attempt is recorded as `invalid_foreign_cpu` (if `foreign_cpu_mean >= 0.15`), `invalid_proc` (if forbidden processes detected), or `failed` (`rc != 0`).
4. **Retry & queue progression:**
   - Each invalid/failed attempt returns to `wait_until_idle` before retrying (no immediate back-to-back retry), up to `3` attempts per queue item (`--smoke` runs `1` attempt without `wait_until_idle`).
   - If an item exhausts all `3` attempts, it is recorded as `failed`, marked `item_failed_<item>.fail`, and the runner advances to the **next item** rather than aborting the queue.
   - Final `STATUS` is `done <timestamp>` (zero item failures) or `done_with_failures <timestamp> failed=<n>` (one or more items failed after 3 attempts). Completed items (`item_done_<item>.ok`) are skipped cleanly on restart.
5. **CSV schema (`b1_summary.csv`):**
   `item,status,attempt,load1_before,load1_after,foreign_cpu_mean,foreign_cpu_max,governor,ac_online,temp_c,out_csv,rc`
