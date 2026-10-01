# Paper experiments: protocols and reproduction status

These scripts evaluate experiments associated with the [GATO paper](https://arxiv.org/abs/2510.07625).
Runnable does not mean numerically reproduced. Keep published results, historical
datasets and fresh evaluations of the current code distinct. A correctness
receipt is not a timing report.

The latest current-code refresh and its numbers: [docs/figure-refresh-2026-10-01.md](../../docs/figure-refresh-2026-10-01.md).

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

- Fig-3 assembles saved CSVs by default. Timing lanes need `--run-gato`,
  `--run-bt` or `--run-mpcgpu` and an exclusive quiet window.
- Fig-4/5/7 regenerate on the GPU by default. `--replot` loads existing data;
  `--quick` is a plumbing smoke, not a statistical reproduction.
- Pickles are trusted local artifacts; never load arbitrary downloaded pickles.
- Generated pools/plots need not exist in a fresh clone. Fig-4 has a bundled
  fallback; other replot commands require their inputs first.

## Figure scope

| Script | Modules | Meaning and current limitation |
|---|---|---|
| `reproduce_fig3_fair.py` | iiwa14 N64 (left); N8/16/32/64/128 (heatmap) | Matched iiwa14 fig8 benchmark, not the published Indy7 task; September sample needs seed-policy A/B |
| `reproduce_fig4_hparam.py` | iiwa14 N64 | Normalized merit vs SQP iteration; recovered grid differs from paper text |
| `reproduce_fig5_disturbance.py` | indy7 N64 | Fixed-pacing disturbance rejection; does not reproduce latency-induced degradation |
| `reproduce_fig7_pickplace.py` | iiwa14 N16 | Pick-place success and physical task-completion time; success gap unresolved; full refresh deferred |

Fig-6 is a simulation snapshot, not a numerical regression target. Fig-8 /
Table-II require physical hardware; preserve them as published results unless
new hardware experiments are performed.

### Fig-3: matched comparison, different robot

The current harness uses iiwa14; the published scalability figure used Indy7.
`benchmarks/iiwa_fig8_shared.py` defines the common trajectory, EE frame, costs
and budget (SQP=1, PCG cap 200 / relative tolerance 1e-4, rho=0.01).

- GATO: `../benchmarks/sweep_batch_iiwa_fig8.py` (N and batch sweeps).
- CPU: `../benchmarks/baselines/track_iiwa_fig8_bt.py`, multi-threaded C++ QDLDL-based CPU solver (OSQP with QDLDL, `pysqpcpu`)
  built by `build_cpu_baseline.sh`.
- MPCGPU: sibling repository's `tools/time_persolve.sh`, matched configuration;
  no batch axis, so the displayed baseline is sequential B × single-solve.

The assembler consumes `../benchmarks/data/sweep_fig8_{gato,bt,mpcgpu}.csv`.
Old Indy7 point-to-point data is NOT a baseline for the iiwa14 fig8 sweep.
Pre-July-30 data used the wrong terminal frame; do not mix it with named-EE
results. [Archived June scripts](../archive/README.md) preserve history, not
interchangeable benchmark implementations.

The current GATO sweep records internal solver duration, not full Python or
controller latency. The paper describes timing around wrappers. Match these
boundaries before quoting speedups; record both when adding an end-to-end lane.
Historical local iiwa14 numbers are not measurements of the current commit.

The September 27 checkpoint exposed seed drift: the historical raw sweep used
zero states beyond knot zero; the migrated controller seeded a hold. The sweep
now defaults explicitly to `--initial-guess zero-tail`, with `hold` available
for a controlled comparison. Each new CSV has a `.runs.jsonl` companion and
hashed frozen reference `.npy`; do not combine seeds or silently use the last
CSV row as a matched baseline. `--check-only` verifies raw/controller bitwise
trajectory and iteration-count parity without reporting or saving timing.

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

### Fig-7 / Table I: preserve the unresolved gap

**September 27 simulator correction:** the old initialization wrote an
axis-angle vector into quaternion xyz with w=0, violating the spherical-joint
unit-quaternion requirement. Nonzero robot velocities also landed at incorrect
offsets in the augmented state. Corrected runs use Pinocchio manifold integration
and are labeled `unit-quaternion-pendulum-v2`. Preserve historical pools, including
the overnight 8/10 sample, but do not mix their metrics with corrected runs or
claim the old simulator was a valid paper protocol. This bug predates the
overnight checkpoint; its discovery does not establish the cause of every
historical-versus-current difference.

For shared-box GPU correctness, `check_pickplace.py --scenarios 1,2 --out <new-dir>`
replays selected seeded cases with fixed pacing, checks finite/deterministic
traces, and records goal completion/timeout position and velocity gates. It
does not retain latency measurements. The two overnight failures reproduce
with the old initialization and both reach all goals with the corrected one.

A historical local 100-scenario `fig7_paper_ready` table exists, so “full sweep
never run” is outdated. It reports 83% episode success at B128 versus 99.2% in
the paper; it is NOT a result for the current revision. Reconcile success
aggregation, velocity norm, initial conditions, force-estimator configuration
and pacing before attributing this to a solver regression or calling the task
“strictly harder.” The current script uses fixed simulation pacing and a
Euclidean velocity norm; paper wording and recovered implementations need an
explicit protocol decision. Timing alone cannot resolve this question.

The merge checkpoint samples ten seeded B128 scenarios: a regression sample,
not a replacement success-rate estimate or CDF. Full Fig-7 refresh and estimator
research are deferred; never overwrite the old pool.

## Merge checkpoint versus research refresh

The [focused runner](../benchmarks/run_merge_checkpoint.sh) samples Fig-3 at
N64 / B=1,8,128 with three process repeats and Fig-7 at B128 / ten seeded
scenarios. It does NOT refresh the full Fig-3 grid, Fig-4, Fig-5 or full Fig-7.
Compiler measurements and autotuning are separate opt-in legs in the
[merge checklist](../../docs/merge-readiness.md).

Next research refresh: Fig-3's matched full sweep, Fig-4's explicitly chosen
grid, and Fig-5's quality and separately specified latency experiment. The legacy
`../benchmarks/run_timing_night.sh` also runs sibling-repo work and full Fig-7.
Its historical 6–7 hour estimate is not a promise for the current code; do not
launch it blindly for the focused checkpoint.

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
