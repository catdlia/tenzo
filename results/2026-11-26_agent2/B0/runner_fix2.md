# B0 Report — `b1_runner_v2.patch` & Live `tenzo-b1` Launch (`2026-11-26`)

## 1. Previous Run Archive
- Checked `systemctl --user status tenzo-b1` before changes: unit was not loaded (`Unit tenzo-b1.service could not be found`).
- Archived `results/2026-11-24-b_B1/{STATUS,b1_summary.csv}` and `runner.log` (`1` line: `waiting 2026-11-24T12:41:18+02:00`, `0` benchmark runs, `0` summary rows) into `results/2026-11-26_agent2/B0/previous_run/`.

## 2. Patch Application & Verification
- Applied `results/2026-11-25_coordinator/b1_runner_v2.patch` cleanly to `scripts/b1_idle_runner.sh`.
- Verified `bash -n scripts/b1_idle_runner.sh` (`exit 0`) and inspected `git diff scripts/b1_idle_runner.sh` (`TIMEOUT_HOURS=168`, `CONSEC_NEXT=4`, `BLOCK_LOG_SEC=900`, `--forbidden-regex`, `probe_inhibit`, `forbidden_procs_pids`, short re-gate `< 300s` after valid item).
- Ignored runtime output directories in `.gitignore`:
  - `results/2026-11-26_B1/`
  - `results/2026-11-26_agent2/B0/dryrun_out/`
  - `results/2026-11-26_agent2/B0/smoke_out/`

## 3. Unit-Context `systemd-inhibit` Probe & Forbidden Process Scan
- **Unit probe command:**
  `systemd-run --user --wait --pipe systemd-inhibit --what=idle:sleep:handle-lid-switch --why=probe true; echo inhibit_lid_rc=$?`
  **Result:** `inhibit_lid_rc=0` (`Finished with result: success`, `status=0/SUCCESS`, `27ms`).
- **Forbidden process scan (`^(firefox|zen-bin|neko|steam|qemu-system|clion|idea|java)`):**
  `ps -eo pid,comm,args | ...` -> **`NONE`** (`0` matching processes, so default `--forbidden-regex` was kept unchanged).

## 4. Dry-Run & Real Smoke Verification
- **`--dry-run` (`--consec 2 --interval 1 --out results/2026-11-26_agent2/B0/dryrun_out`):**
  - Logged `blocked: ac=1 load1=2.55 (need < 1.0) procs=[dry-run: not checked]` immediately, waited until `load1=0.94`, passed 2/2 checks, ran `dryrun_item`:
    `OK dryrun_item attempt=1 foreign_cpu_mean=0.0372 foreign_cpu_max=0.0372 load1=0.94->0.94 temp=50.0C` (`STATUS: done 2026-11-26T01:33:02+02:00 dry_run=1 rc=0`).
- **Real `--smoke` (`--out results/2026-11-26_agent2/B0/smoke_out`):**
  - `runner.log`: `[2026-11-26T01:33:02+02:00] inhibit=idle:sleep:handle-lid-switch`
  - `smoke_ab_gemv_pp16_tg16`: `attempt=1 foreign_cpu_mean=0.1031 foreign_cpu_max=0.1534 load1=0.94->1.25 temp=52.0C rc=0`
  - `STATUS`: `done 2026-11-26T01:33:23+02:00 smoke=1 valid=1` (`valid=1` confirmed!).

## 5. Live `tenzo-b1.service` Launch
- **Command:**
  `systemd-run --user --unit=tenzo-b1 -p MemoryMax=4G -p MemorySwapMax=0 -p Nice=19 -- /home/illia/CLionProjects/untitled/scripts/b1_idle_runner.sh --out /home/illia/CLionProjects/untitled/results/2026-11-26_B1`
- **State (`2026-11-26T01:33:37+02:00`):**
  - `STATUS`: `waiting 2026-11-26T01:33:32+02:00`
  - `runner.log`:
    - `[2026-11-26T01:33:32+02:00] inhibit=idle:sleep:handle-lid-switch`
    - `[2026-11-26T01:33:32+02:00] blocked: ac=1 load1=1.35 (need < 1.0) procs=[]`
- `procs=[]` and `ac=1` (no blocking desktop processes; only `load1` decaying below `1.0` after smoke check). Entering quiet wait (`≤ 120 min`, checking every `10 min`) with zero heavy workloads so `B1` can drain all 6 items cleanly.
