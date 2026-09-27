# GATO timing handoff for the overnight coordinator

Prepared 2026-09-27. This document prepares work; it does **not** declare a quiet
window or launch a job. Run only in the exclusive slot assigned by the user.
Repository on the shared box: `/home/plancher/Desktop/GATO`.

## Next slot: isolate the benchmark seed change

Prepared source `d91ab5a`, receipt commit `3b1a051`: 356 passed / zero skips,
38 unchanged goldens. Both zero-tail and hold policies passed 400-step finite,
bitwise raw/controller trajectory and PCG-count parity at B1/8/128. Native
modules are rebuilt; require green CI and record the actual launch HEAD.
Subsequent documentation-only commits need no new receipt fingerprint.

The original bundle completed (checkpoint 127.4 s, compile 64.3 s, calibration
45.3 s). Do not repeat it blindly. The [review](checkpoint-2026-09-27.md) found
that the historical Fig-3 sweep used a zero-tail initial guess while the
September checkpoint used a hold. The next requested suite is **seed-ab**:

- **Launcher:** `GATO_QUIET_WINDOW=1 bash examples/benchmarks/run_timing_handoff.sh --suite seed-ab`.
- **Working directory:** `/home/plancher/Desktop/GATO`.
- **Reservation:** provisionally **15–30 minutes**, releasing early. Six short
  sweeps plus quiet-settling/preflight, no builds or Fig-7; not a hard timeout.
- **Prerequisites:** explicit exclusive CPU/GPU slot, clean source, updated
  receipt/CI and prebuilt default iiwa14 N64. Same cooperative lock and tool
  requirements as below. Correctness-only raw/controller parity must pass first.
- **Outputs:** unique `examples/benchmarks/night_logs/handoff_*` with six CSVs,
  `.runs.jsonl` provenance, frozen goal `.npy`, per-leg logs and snapshots.
- **Stop/resume:** `STOP` in that exact handoff directory stops before the next
  sweep. Preserve partial results; rerun the incomplete seed-ab suite into a
  fresh directory in another assigned slot. No process suspension or in-place resume.

Three repeats per seed alternate order. Every sweep after the first explicitly
loads the first sweep's saved reference. Compare seed effects on this one
source/binary, not historical-versus-current CUDA speed. Check hashes and all
repeats before interpreting a residual gap. `all` still means the original
checkpoint/compile/calibration bundle and deliberately does NOT include seed-ab.

Safe preview: `bash examples/benchmarks/run_timing_handoff.sh --suite seed-ab --dry-run`.
The original intake/reservations below are retained as historical runbook context;
they are superseded by this section for the next slot.

## Coordinator intake: the six required fields

1. **Launcher:** `GATO_QUIET_WINDOW=1 bash examples/benchmarks/run_timing_handoff.sh --suite all`.
   Use `--suite checkpoint`, `compile`, or `calibrate` to split the reservation.
   The persistent `nohup` launch recipe below records the launcher PID and log.
   Preparation only: append `--dry-run` (no quiet-window declaration needed).
2. **Working directory:** `/home/plancher/Desktop/GATO`.
3. **Estimated runtime / reservation:** allow **120 minutes total**: checkpoint
   60, compile 30, optional calibration 30. These are unmeasured scheduling
   allowances, not guaranteed runtimes or kill deadlines. Release early.
4. **Prerequisites:** user-assigned exclusive CPU/GPU slot; cooperating agents
   honor `/tmp/a2rlab-timing.lock`; clean tracked source, initialized pinned
   submodules, valid `gpu-proof.json`, and green receipt/CPU CI. Use the existing
   `.venv/bin/python` with GATO, numpy, pinocchio, matplotlib and pybind11.
   Prebuilt default modules: iiwa14 N16/N64 for checkpoint and indy7/iiwa14 N64
   for calibration. Required host tools: bash, git, flock, rg, nvidia-smi, nvcc,
   cmake and standard GNU utilities; compile also needs GNU make, a compatible
   C++ toolchain, `/usr/bin/time`, and a working `systemd-run --user --scope`.
   This compile recipe targets the current **sm120** box; do not transplant it
   blindly. Require ≥40 GiB available RAM before each build; enforced cap is
   36 GiB, no swap, two jobs. Dependencies/modules are already prepared here;
   if something is missing, report it and prepare outside the timing slot.
5. **Output location:** launcher log `/tmp/gato-timing-launch.XXXXXXXX.log`;
   bundle logs/results under
   `/home/plancher/Desktop/GATO/examples/benchmarks/night_logs/handoff_*`;
   nested runtime outputs under the sibling `merge_checkpoint_*` directory.
   Both exact directory names are announced in logs. Fig-7 also writes uniquely
   tagged `data/<tag>.pkl`, `<tag>_cdf.png`, and `<tag>_table_I.txt` under
   `/home/plancher/Desktop/GATO/examples/paper-figures/`.
6. **Stop/resume:** create `STOP` in the exact announced `handoff_*` directory
   to stop before the next outer leg; this does **not** interrupt the current
   leg or the nested checkpoint. There is **no in-place resume**. After the
   old job/descendants have exited and the box is released, launch only the
   unfinished suite in a newly assigned slot; it gets fresh output paths.
   Preserve old outputs and label interrupted results incomplete. Detailed
   rules and exit codes are in “Stop and resume” below.

## Reservation and scope

Request **two hours for the complete GATO bundle**, releasing the box as soon as
it finishes. If tonight is crowded, allocate **one hour for `checkpoint` first**
and defer compilation/calibration. These are conservative scheduling allowances,
not measured runtimes or hard deadlines. Revise them from the first run's logs;
do not promise a fixed completion time. Do not start a long leg near another
project's slot. A ten-scenario sample may still be slow.

| Suite | Work | Priority / provisional allowance |
| --- | --- | --- |
| `checkpoint` | Three iiwa14 N64 repeats, B=1,8,128, 400 solves each; then ten seeded B128 Fig-7 scenarios | First; reserve up to 60 min |
| `compile` | Fresh Release/sm120 builds of indy7 N16 and iiwa14 N64; cold/no-op wall time, peak process RSS, module bytes | Second; allow 30 min |
| `calibrate` | Indy7/iiwa14 N64 kicked-fig8 PCG/BDSV probes, fitted policy, auto validation when selected | Optional third; allow 30 min |
| `all` | The above in order, never concurrently | Reserve 120 min |

Not included: a full Fig-7/Table-I refresh, all other paper figures, competitor
benchmarks, Go2 tuning, stream/graph experiments, or hardware trials. This runner
owns GATO only. Obtain other projects' exact commands and estimates from their
agents; do not invent them or use GATO's older broad overnight runners.

## State to preserve

`cleanup-modernization` and `modernizing-tests` are synchronized development
branch names, not two implementations. Both were `bfb1023` before this harness
addition. `main` is separate and must not be merged or rewritten tonight.
Receipt at source `78fc5d4`: 341 passed, zero skips, all 38 goldens passed;
bounded AL/barrier/ADMM contact racechecks: zero hazards. The harness-only
addition does not change receipt-fingerprinted sources. Record the actual HEAD
used, and require the existing receipt/CPU CI to remain valid.

No pulls, pin bumps, code edits, regeneration, production rebuilds, receipts,
dependency installations, tuning-default changes, or website updates during the
slot. The pre-existing untracked sqpcpu submodule content is unrelated; leave it
alone. Read `CLAUDE.md` for project conventions if doing anything beyond running
and reporting this bundle.

## Launch and shared-box coordination

Safe preparation, anytime (no GPU queries, builds, locks or timing):

```bash
cd /home/plancher/Desktop/GATO
bash examples/benchmarks/run_timing_handoff.sh --suite all --dry-run
```

Only after the user assigns the quiet window, launch once in a persistent
terminal or under `nohup`. Use a fresh launcher log:

```bash
cd /home/plancher/Desktop/GATO
launch_log=$(mktemp /tmp/gato-timing-launch.XXXXXXXX.log)
GATO_QUIET_WINDOW=1 nohup bash examples/benchmarks/run_timing_handoff.sh --suite all > "$launch_log" 2>&1 &
launch_pid=$!
printf 'launcher PID=%s log=%s\n' "$launch_pid" "$launch_log"
```

Use `--suite checkpoint` for the shorter first-priority slot; `compile` and
`calibrate` can run later without repeating the runtime checkpoint. Do not set
`FORCE=1` or pass busy-GPU overrides. GPU must have zero compute processes and
utilization ≤5%; integer one-minute CPU load must be ≤2 at boundaries. Compile
measurements also require an exclusive CPU/RAM slot, not just a free GPU.

The runner acquires `/tmp/a2rlab-timing.lock` using `flock -n`; exit 75 means
another cooperating runner holds it. **Every participating project must use
that same lock** for its timing command. For another project's approved command:

```bash
flock -n /tmp/a2rlab-timing.lock /absolute/path/to/that-project-approved-runner
```

Do not wrap GATO in a second flock on the same file: it locks internally. Never
delete the lock file to "unlock" it. The lock is advisory: it cannot stop agents
that ignore it, correctness jobs, compiles, or unrelated GPU activity. Coordinate
the slot explicitly with all agents; idle preflight alone is not permission.

Watch the launcher log and the announced `SUMMARY.txt`; actual progress goes
to each leg's log. A checkpoint has its own nested summary/output directory,
announced in `checkpoint.log`. While a leg runs, do not launch tests/builds as
"health checks."

## Stop and resume

- **Graceful stop:** create an empty `STOP` file in the exact `handoff_*`
  directory printed by the current runner. Do not use a glob that could select
  another run. It prevents the next outer leg from starting; it does not pause
  or interrupt the current process. The entire runtime checkpoint is one outer
  leg, so it can finish all three repeats and ten scenarios before honoring
  the request. If no outer legs remain, the run finishes normally.
- **Urgent stop / hang:** no automatic timeout or process-tree cancellation is
  implemented. Coordinate first, identify the exact launcher and descendants
  (including any compile systemd scope), and stop only that verified GATO tree.
  Do not kill by broad process name, delete the shared lock, or assume the
  parent's exit means its children exited. Do not SIGSTOP/SIGCONT a timing run:
  suspension would contaminate measurements and retain the reservation.
- **Release:** inspect logs and confirm all GATO descendants, compiler scopes
  and GPU compute contexts from the run have ended. Only then return the slot.
  A released advisory lock alone is not proof that every child is gone.
- **Resume:** first inspect `SUMMARY.txt`, leg logs and provenance. Keep valid
  completed suites; launch the unfinished suite using the same launcher with
  `--suite checkpoint`, `--suite compile`, or `--suite calibrate` in a new
  exclusive slot. Do not rerun `all` unless intentionally repeating everything.
  There is no per-repeat, per-scenario, per-target or per-plant resume switch:
  an interrupted suite is rerun in full with new outputs. In particular, never
  call a resumed partially built tree a cold compile measurement. Preserve the
  old partial results as incomplete; do not append them to the new result pool.
  Record both source/binary provenances if anything changed between slots.
- **Exit interpretation:** `0` plus final `DONE` means all selected legs finished
  (still subject to contamination/quality review); `3` plus `DEFERRED` means the
  STOP request was honored; `75` plus the lock-refusal message means no work
  started because another runner holds the lock. Other nonzero exits indicate
  refusal/failure; inspect the message rather than inferring from the number
  alone (child commands propagate their own codes). A launcher log may be the
  only artifact if refusal occurred before output-directory creation.

## Outputs and acceptance

All new compile/tuning outputs live under unique
`examples/benchmarks/night_logs/handoff_*` directories. The checkpoint creates
its own `merge_checkpoint_*` directory and uniquely tagged Fig-7 data/plots in
the existing paper-figure output directories (paths are printed in its log).
Preserve these outputs; do not overwrite old pools or baseline CSVs.

- Record HEAD, recursive pins, receipt, production module hashes, environment,
  commands, wall duration per leg, exit status and any contention. The runner
  captures these and checks that source, receipt, module hashes and production
  CMake cache stay unchanged across outer legs.
- Boundary checks do not detect every transient workload. Any observed overlap
  invalidates the affected timing; report and reschedule it, never silently
  accept partial CSVs or stop the competing job.
- Fig-3 CSVs contain internal solver latency and per-trajectory cost, not full
  Python/controller latency or independent-plant tracking error. Report each
  repeat and variation, and compare only matching prior iiwa14 fig8 cells.
  Do not use the archived Indy7 point-to-point baseline.
- The ten-scenario Fig-7 sample checks regression behavior and task quality; it
  is not a replacement paper success-rate estimate. Keep protocol/seed/tag and
  failures. Full Fig-7 refresh remains deferred.
- Compile output is isolated from production `build/` and `python/gato`.
  Report GNU-time wall and maximum RSS separately for each cold/no-op target,
  and module sizes. "Cold" means a fresh CMake/object tree, not cleared OS disk
  caches. RSS is GNU time's process statistic, not aggregate cgroup peak memory.
  One observation per target is diagnostic, not a speedup claim.
- Calibration writes only the new `linsys-tuning.json` in the results directory.
  Keep PCG/BDSV quality/latency diagnostics and conditional auto validation;
  do not install it as a production default. The Fig-3 checkpoint pins PCG and
  does not validate the tuned controller. Autotuner `--dry-run` still TIMES:
  only the top-level handoff runner's `--dry-run` is a preparation-only command.

Return a short report: completed/deferred/failed legs, result directory paths,
latency/variation and task-quality findings, compile wall/RSS/bytes, proposed
calibration policy, observed contamination, and total reservation consumed.
No merge, release, or public performance claim follows automatically.
