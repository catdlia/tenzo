# B1 runner v2 patch (coordinator, 2026-11-25)

`b1_runner_v2.patch` changes `scripts/b1_idle_runner.sh` and nothing else. The script in the repo was **not** modified: the `tenzo-b1` unit is still running it, and bash reads the file while it executes. Apply the patch only after stopping the unit:

```bash
systemctl --user stop tenzo-b1
git apply results/2026-11-25_coordinator/b1_runner_v2.patch
bash -n scripts/b1_idle_runner.sh
```

## Changes

| Problem (found 25.11) | Change |
|---|---|
| No log of why the machine is "not idle" | While waiting, the runner logs `blocked: ac=… load1=… (need < 1.0) procs=[comm(pid) …]` immediately and then every `BLOCK_LOG_SEC` (default 900 s; `--block-log-sec`). |
| `TIMEOUT_HOURS=48`: the current run gives up on 26.11 around 12:41 | Default timeout 168 h. |
| `systemd-inhibit --what=idle:sleep:handle-lid-switch` was never exercised (smoke skipped it), and polkit may refuse `handle-lid-switch` to a user unit | At start (not in `--dry-run`) the runner probes `idle:sleep:handle-lid-switch`, then `idle:sleep`, then none, and logs `inhibit=…`. Every run, smoke included, uses the probed setting. |
| 10 min of idle before **each** of the 6 items (~1 h of waiting) | After a valid item that ended < 5 min ago, the next item needs only `CONSEC_NEXT` checks (default 4 = 2 min; `--consec-next`). The first item and every retry after an invalid one still need the full `CONSEC_REQ`. |
| Forbidden-process list hard-coded twice | One `FORBIDDEN_RE` variable (`--forbidden-regex`); same default list. |

## Tests run by the coordinator

The tests ran in a Linux VM (2 CPUs, no logind) on copies of the script, never on the repo file:

1. `bash -n`.
2. `--dry-run --consec 2 --interval 1`: two idle checks, fake item `valid`, `STATUS done`.
3. `--dry-run --load-threshold 0 --block-log-sec 3`: `blocked:` lines appear, and the first one appears immediately.
4. A copy whose 6 queue commands were replaced by `sleep 1`, run on the real (non-dry) path with `--consec 3 --consec-next 1`:
   - `inhibit=none (all probes failed; running without inhibit)` — the VM has no logind;
   - the first item waited 3 checks, the next five took the short re-gate;
   - all 6 items `valid`, `STATUS done`.
5. `git apply --check` on the repo: applies cleanly.

**Still to check on the laptop** (the agent, B0):
- What the inhibit probe gives inside the user unit: `systemd-run --user --wait --pipe systemd-inhibit --what=idle:sleep:handle-lid-switch --why=probe true; echo rc=$?`.
- A real `--smoke`.
- Whether `java` in `FORBIDDEN_RE` matches an always-running background process.
