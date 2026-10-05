# Paper experiments: protocols and reproduction status

These scripts evaluate experiments associated with the [GATO paper](https://arxiv.org/abs/2510.07625).
Runnable does not mean numerically reproduced. Keep published results, historical
datasets and fresh evaluations of the current code distinct. A correctness
receipt is not a timing report.

Current-code numbers: [docs/figure-refresh-2026-10-01.md](../../docs/figure-refresh-2026-10-01.md).

## Setup and modes

Use the project `.venv`; `./tools/install.sh --examples` supplies simulation AND
plotting dependencies (`--test` alone does not include matplotlib). Build with
`JOBS=2 ./tools/build.sh --profile receipt`, or select only the needed modules.
Scripts resolve inputs independently of cwd; run the commands below from the repo root.

```bash
# No GPU or result writes: preview the focused checkpoint.
examples/benchmarks/run_merge_checkpoint.sh --dry-run
# Render trusted saved data; does not evaluate the current solver.
.venv/bin/python examples/paper-figures/reproduce_fig4_hparam.py --replot
# Small correctness smoke, NOT paper numbers (uses GPU).
.venv/bin/python examples/paper-figures/make_all.py --only fig4,fig5 --quick
```

- Fig-3 assembles saved CSVs by default. Timing lanes need `--run-gato` or
  `--run-bt` and an exclusive quiet window; MPCGPU cells are imported from that
  repository's own timing run with `--mpcgpu-timing-dir`.
- Fig-4/5/7 regenerate on the GPU by default. `--replot` loads existing data;
  `--quick` is a plumbing smoke, not a statistical reproduction.
- Pickles are trusted local artifacts; never load arbitrary downloaded pickles.
- Generated pools/plots need not exist in a fresh clone. Fig-4 has a bundled
  fallback; other replot commands require their inputs first.

## Figure scope

| Script | Modules | Meaning and current limitation |
|---|---|---|
| `reproduce_fig3_fair.py` | iiwa14 N64 (left); N8/16/32/64/128 (heatmap); `--robot indy7` for the paper's arm (GATO + CPU lanes) | Matched three-solver fig8 benchmark on an identical problem; ratios are not the paper's (different robot, harness, timing boundary) |
| `reproduce_fig4_hparam.py` | iiwa14 N64 | Normalized merit vs SQP iteration; recovered grid differs from paper text |
| `reproduce_fig5_disturbance.py` | indy7 N64 | Fixed-pacing disturbance rejection; does not reproduce latency-induced degradation |
| `reproduce_fig7_pickplace.py` | iiwa14 N16 | Pick-place success vs batch size on two task settings (stop: 15→100 %; pass-through: 26→97 %), refreshed Oct 3 |

Fig-6 is a simulation snapshot, not a numerical regression target. Fig-8 /
Table-II require physical hardware; preserve them as published results unless
new hardware experiments are performed.

### Fig-3: matched comparison, different robot

The current harness uses iiwa14; the published scalability figure used Indy7.
`benchmarks/iiwa_fig8_shared.py` defines the common trajectory, EE frame, costs
and budget (SQP=1, PCG cap 200 / relative tolerance 1e-4, rho=0.01). Every lane
takes `--robot {iiwa14,indy7}` (iiwa14 default, names unchanged). The Indy7 lane
runs the SAME fig8 (A = 0.15 m, period 6 s, center = EE at `INDY7_START_CONFIGS["ready"]`)
for GATO and the CPU solver only — MPCGPU has no Indy7 build or goal file, so the
Indy7 goal is synthesized from the formula — and writes `*_indy7` CSVs and figures
(`sweep_fig8_gato_indy7.csv`, `fig3_fair_scalability_indy7.png`, …).

- GATO: `../benchmarks/sweep_batch_iiwa_fig8.py` (N and batch sweeps).
- CPU: `../benchmarks/baselines/track_iiwa_fig8_bt.py`, multi-threaded C++ QDLDL-based CPU solver (OSQP with QDLDL, `pysqpcpu`)
  built by `build_cpu_baseline.sh`.
- MPCGPU: that repository's `tools/timing.py` figure-eight plan (same costs and
  budget), imported with `--mpcgpu-timing-dir`; it has no batch axis, so the
  displayed baseline is sequential B × single-solve.

The assembler consumes `../benchmarks/data/sweep_fig8_{gato,bt,mpcgpu}[_indy7].csv`.
Pre-July-30 data used the wrong terminal frame and the June point-to-point data a
different task; neither is a baseline for these sweeps.

The GATO sweep records internal solver duration, not Python or controller
latency; the paper timed around wrappers. Match boundaries before quoting
speedups. The sweep's warm-start seed is explicit (`--initial-guess zero-tail`,
the historical choice; `hold` for a controlled comparison — the two differ by
up to 7%, see the refresh document). Each CSV has a `.runs.jsonl` companion and a
hashed frozen reference `.npy`; never mix seeds. `--check-only` verifies
raw/controller bitwise parity without timing.

### Fig-4: choose and name the grid

Defaults evaluate 50 random targets × 24 Q/R combinations, matching the
recovered dataset. The paper text describes 100 runs × 81 cost choices.
`--num-targets` changes the target count, not the missing cost-grid definition.
Document the chosen grid before refreshing the figure; do not claim exact
paper reproduction without reconciling that difference.

### Fig-5: quality and latency are different experiments

The script uses `pace_by_solve_time=False`. Its force sweep and 50 N trajectories
measure control quality at fixed simulation pacing, useful even on a shared GPU.
They do not model additional actuation delay from larger batches. The paper's
latency-induced degradation needs a separately specified delay/pacing experiment
on a quiet box. Replotting historical data is not a refresh; data predating the
July force/frame fixes is unsuitable for current validation.

### Fig-7 / Table I: refreshed October 3, 2026 (corrected simulator, identified-weight batch)

**Simulator correction (September 27):** the old initialization wrote an axis-angle vector
into quaternion xyz with w=0 and placed robot velocities at wrong offsets in the augmented
state. Corrected runs use Pinocchio manifold integration and are labeled
`unit-quaternion-pendulum-v2`. Historical pools (and the paper) are not comparable.

**Two task settings (October 3):** `--task stop` (default) tests arm arrival with a dwell,
not payload placement — the EE reference ramps to each goal over 1.5 s (minimum jerk) and the success gates
must hold 100 ms — run with the paper's exploration sampler: 15 / 100 / 99 / 98 % at
B = 1 / 8 / 32 / 128 on 100 seeded scenarios. `--task pass-through` uses step references
and an instantaneous gate in the corrected simulator, with the identified-weight sampler
(`IdentifiedWrenchSampler`, `--estimator wid`): 26 / 83 / 97 / 95 %. Batching is the lever in
both; the batch is filled differently because a swinging load must not be chased when the arm
has to stop, while knowing the load pays when flying through. The paper's `ForceEstimator`
never identifies a 15 kg payload (vertical estimate ≈ −17 N against 147 N) — on the
pass-through task it gives 6 / 62 / 84 / 82 %. `--success-plot` renders
`fig7_success_vs_batch.png`; numbers, diagnosis and the secondary rows are in
`docs/figure-refresh-2026-10-01.md`. `--estimator`, `--goal-ramp`, `--settle-time`,
`--qd-cost`, `--pend-mass` override a preset; tag every variant apart.

For shared-box GPU correctness, `check_pickplace.py --scenarios 1,2 --out <new-dir>
[--estimator fe|wid]` replays seeded cases with fixed pacing, checks finite/deterministic
traces and records goal gates; it retains no latency measurements. Pools are fixed-paced
and deterministic, so they need no quiet window (they do disturb anyone else's timing).

These are exploratory results: the seed-0 pool was reused during tuning, the stop
pool records a dirty source tree, and neither task enforces actuator limits or
payload settling. Never overwrite an old pool: tag every protocol change.

### Held-out Fig-7 correctness validation

`fig7_validation_protocol.json` freezes 100 separate seed-20261004 scenarios,
four batch sizes and three arms: stop/FE, pass-through/identified-weight, and
stop/FE with a torque penalty plus a hard simulator torque clamp. No tuning is
allowed on this holdout. All arms record actual applied torque, joint-position
and velocity violations, and payload swing; an arm-arrival success is not a
payload-placement or safe-robotics claim. The clamp enforces torque only, not
position or velocity constraints.

From a clean, committed checkout with the current iiwa14 N16 module built:

```bash
.venv/bin/python examples/paper-figures/validate_fig7.py docs/open-tasks/fig7-heldout
.venv/bin/python examples/paper-figures/summarize_fig7_validation.py docs/open-tasks/fig7-heldout
```

This is fixed-pacing correctness work, not timing. Coordinate GPU sharing first.
`--limit 1` runs a partial smoke; rerun without that option to finish the same pool.
Create `STOP` in the output directory to pause between episodes; remove it to
resume with the same command, source commit, protocol and binary. Completed
episodes are not overwritten. Changing any identity requires a new directory.
Only a complete 1,200-episode pool supports the full protocol summary.

## Merge checkpoint versus research refresh

The [focused runner](../benchmarks/run_merge_checkpoint.sh) samples Fig-3 at
N64 / B=1,8,128 with three process repeats and the Fig-7 stop task at B128 /
ten seeded scenarios: a regression sample, not a refresh. Compiler measurements
and autotuning are opt-in legs of `../benchmarks/run_timing_handoff.sh`. Still
open for a research refresh: Fig-4's exact cost grid and Fig-5's latency
experiment. `../benchmarks/run_timing_night.sh` is the legacy broad run (sibling
repository work, full Fig-7); do not launch it blindly on a shared machine.

## Provenance and interpretation

The paper used an RTX 4090 / CUDA 12.6 setup. Different hardware, compilers,
models and budgets can change absolute times AND relative speedups. Re-run all
compared lanes with matched accuracy before making a new performance claim.

Retain source/submodule SHAs, binary/configuration provenance, hardware/toolchain,
seeds, pacing, sample counts, metric definitions and exact commands. Record
missing provenance rather than mixing pools. `_common.load_data` prefers a
regenerated local pickle over its bundled fallback and prints the selected
source: check that output. [Archaeology](../../docs/archaeology.md) is a dated
record of recovered data, not the current reproduction verdict.
