# September 27 checkpoint review

The overnight coordinator measured source `3e29653` on the RTX 5090 / CUDA
13.2 box. Collection succeeded with unchanged source, receipt and module
fingerprints. Boundary checks and coordinator monitoring found no contention;
they are not continuous proof against every transient workload. The existing
receipt was verified, not rerun overnight (341 tests, 38 bitwise goldens).

## Measured checkpoint, before the follow-up fixes

| iiwa14 N64 batch | Median range across three processes (ms) | Latest saved August median (ms) |
| ---: | ---: | ---: |
| 1 | 0.6345–0.6360 | 0.4490 |
| 8 | 0.9530–0.9530 | 0.6980 |
| 128 | 5.1985–5.2005 | 4.5950 |

Each process used 400 solves, discarding ten. These are internal solver times,
not end-to-end controller latency. Relative to the August cells, medians were
approximately 41%, 37%, and 13% higher. This is a diagnostic signal, not an
attributed CUDA regression: the migrated benchmark changed its initial guess.

Fresh isolated builds: indy7 N16 took 26.05 s, iiwa14 N64 32.68 s; both no-op
builds took 0.04 s. GNU time reported maximum process RSS 1,302,832 / 1,308,864
KiB and module sizes were 3,777,424 / 4,334,480 bytes. These are single
observations with fresh object trees, not cold OS caches or aggregate peak
memory. Production binaries were unchanged.

Calibration proposed PCG for indy7 and auto tau=0.3151144875 for iiwa14.
The iiwa14 mean was 0.396 ms PCG versus 0.377 ms auto on one seeded six-second
probe. Proposals remain isolated; no defaults were installed and this is not a
general speedup claim.

Suite durations were 127.4 s checkpoint, 64.3 s compile, 45.3 s calibration.
The original two-hour reservation was an unmeasured allowance, not actual demand.

## Correctness findings and follow-up

### Benchmark seed drift

The historical raw batch sweep seeded only knot zero, leaving the rest of the
trajectory zero. Its migration to `MPCController.reset(x0)` silently changed
that to a hold at x0. Both guesses are legal for these fixed-base models but
can produce different optimization paths beyond the ten discarded solves.

`MPCController.reset(x0, xu_warm=...)` now supports explicit initial guesses
without private-buffer access; the default remains a hold. The batch sweep
defaults explicitly to historical `zero-tail`, with `--initial-guess hold`
retained for comparison. New timing outputs preserve the actual reference
array, its hash, URDF hash, seed policy, solver configuration and source stamp.
`--check-only` compares raw/controller trajectories and PCG iteration counts
without emitting or saving latency results.

**Timing effect remains unmeasured.** Next run `--suite seed-ab` in an assigned
quiet window: three repeats per seed, alternating order, one frozen reference,
same source and binaries. This isolates seed policy, not old/new kernel speed.
If a residual gap remains, prepare a separate source A/B before attributing it.

### Pick-place simulation initialization

The overnight sample succeeded in 8/10 episodes; scenarios 1 and 2 timed out
at goal four. The historical `fig7_paper_ready` pool contains identical first
ten scenario parameters and succeeded in all ten, although implementations and
environment were not matched.

Review found a pre-existing simulator bug: a three-element axis-angle vector
was copied into the xyz entries of a spherical-joint quaternion, leaving w=0.
Its norm was not one. Robot velocities were also copied into the augmented
configuration/velocity layout at incorrect offsets when nonzero. These are
invalid initial conditions, not merely a different task difficulty.

Initialization now uses Pinocchio's manifold integration from neutral and
separate robot q/v slices, shared by goal and fig8 loops. New Fig-7 pools are
labeled `unit-quaternion-pendulum-v2`. Preserve the old pools, but do not mix
their success rates or completion times with v2 or the paper. Goal-event
telemetry records distance and velocity at each completion/timeout, including
the final tick. `check_pickplace.py` saves fixed-pacing correctness traces and
checks repeat determinism without retaining performance measurements.

Correctness-only causal replay on September 27: restoring ONLY the old
initialization reproduces both fourth-goal timeouts (quaternion norms 0.36166
and 0.23006; terminal velocity norms 2.434 and 2.199 rad/s). Corrected scenarios
1/2 reach all goals, twice bitwise deterministically. The complete corrected
ten-scenario sample is still **8/10**, now failing scenarios **3 and 9 at goal
five**. Their timeout distance/velocity pairs are approximately 0.06521 m /
2.06796 rad/s and 0.07313 m / 3.41300 rad/s. All ten traces repeat exactly.
These failures remain visible; no gates, timeouts, costs or estimator settings
were tuned to eliminate them. This is not a full Fig-7 refresh or a new paper
success-rate claim.

Both seed policies also passed 400-step bitwise raw/controller trajectory and
PCG-count parity at B1/8/128, with finite outputs. Neither correctness experiment
reports or saves solver latency. Full traces are in the local working notes:
`docs/open-tasks/pickplace_v2_ten_scenario_2026-09-27/`.

### Floating effort weights

The Python contract specified `n_actuated` weights (12 for Go2), but the native
binding and upload required `NUM_JOINTS` (19). Both now use `ACTUATED_SIZE`.
The cost kernels already read actuator-indexed weights; contact-force weights
remain separate. KKT gates cover default/fc modules, rejected nq-width input,
and bitwise restoration of the scalar path.

## Remaining decisions

Fresh-checkout follow-up also found that the lean install lacked SciPy, causing
trimesh voxelization to fall back to bounding-box collision covers (Indy7 20
versus 29 spheres; iiwa14 32 versus 44). GATO now declares SciPy as a base
codegen dependency, rejects voxelization fallback warnings, and invalidates
older cache keys. An injected missing-dependency gate must fail before writing
headers. No GRiD source change is involved.

Validation complete on source `d91ab5a` / receipt `3b1a051`: **356 passed,
zero skips, 38 unchanged bitwise goldens**. Effort-vector memcheck: two passed,
zero errors. The repaired fresh checkout's lean install (no Pinocchio) regenerates
all robot headers/registry byte-for-byte, builds indy7 N64 with two jobs under
the 36 GiB cap, and runs examples 01/02. Fifteen synthetic timing-runner tests
pass with fake GPU/compiler/lock commands; the new timing suite remains unrun.

1. Require green receipt/CPU CI for the validated source and preserve this evidence.
2. Review the seed A/B once a quiet slot is assigned; do not run timing on the
   shared box or install the calibration recommendations automatically.
3. Finish latest-main integration review and final
   candidate verification. Citation metadata now matches the README's ICRA
   entry; temporary branch URLs stay until the API is actually on main.
   The current development candidate's fresh-checkout onboarding passed; repeat
   if integration changes its install/build behavior.
4. Keep full paper refreshes and Go2 walking separate. Render the prepared
   website quick-start/link update before any authorized deployment.

## Artifact locations on the collection machine

Under `examples/benchmarks/night_logs/`: `handoff_JK5C4Oqv` (checkpoint),
`merge_checkpoint_SQbu4XFG` (runtime details), `handoff_PqUuN3Mz` (compile),
`handoff_Uxw8UKPt` (calibration). The original Fig-7 pool/plot/table use tag
`merge_checkpoint_SQbu4XFG` under `examples/paper-figures/`. These local outputs
are not all distributed with Git; the tracked August comparison CSV is
`examples/benchmarks/data/sweep_fig8_gato.csv` (last occurrence per cell).
