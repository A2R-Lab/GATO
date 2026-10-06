# Development and validation

`./tools/install.sh --test --dev` creates the project-local `.venv`, initializes the
`external/GRiD` and `external/GLASS` submodules, installs the `[test]` + `[dev]` extras
(pinocchio, mujoco, scipy, gymnasium, pytest, pytest-gpu-proof) and regenerates the
vendored GRiD headers. That `.venv` is the only Python for this repo: receipts, tests and
timing all run from it. Never import models, environments or generated files from a
sibling checkout.

GRiD and GLASS are developed in their own repositories and pinned here as gitlinks
(`external/`), read-only except for pin bumps. Codegen changes travel
GRiDCodeGenerator → GRiD → GATO; after a GRiD bump run `.venv/bin/python tools/regen_grid.py`
and `pytest test/test_codegen.py` (the vendored `gato/dynamics/<robot>/grid.cuh` must
regenerate byte-identically — never edit it by hand).
The top-level GLASS gitlink must match GRiD's nested GLASS pin; the codegen gate
checks this before comparing generated headers. A dependency update requires
rebuilding the receipt profile and passing the existing numerical goldens.
Previously collected timings remain evidence for their recorded pins, not the
new checkout.

## Build

Modules are compiled per `(plant, knot_points[, variant])` into `python/gato/bsqpN{N}_{plant}[_fc|_eh].so`;
the batch size is a runtime argument. `tools/build.sh` configures and builds incrementally
in `build/`:

```bash
MODULES="indy7:8,16;go2:16:fc" JOBS=1 ./tools/build.sh   # explicit plant:N[:variant] list
./tools/build.sh --profile receipt                       # exactly test/receipt_modules.txt
./tools/build.sh --variant fc                            # fc (or eh) variants of PLANT x KNOTS
./tools/build.sh --clean                                 # wipe build/ and reconfigure
```

`PLANT`, `KNOTS`, `ARCH` and `JOBS` (default 4) are environment overrides. `ARCH` is the CUDA
architecture (default `native`); set it explicitly, e.g. `ARCH=120` on the lab box, whenever
native detection could pick another target (worktree builds) — the symptom of a wrong
architecture is frozen solves, not a build error. Every translation unit pulls the whole generated
`grid.cuh`, so one go2 compile peaks near 7 GB of RAM; `gato.build(...)` and
`examples/05_build_your_robot.py` reconfigure `build/` for their own module request, so
re-run your usual configure afterwards.

On the shared lab box every build or test command runs in a memory-capped scope with one
compiler job, and the GPU is left to whoever is timing:

```bash
JOBS=1 systemd-run --user --scope -p MemoryMax=36G -p MemorySwapMax=0 --same-dir --quiet \
  ./tools/build.sh --profile receipt
systemd-run --user --scope -p MemoryMax=36G -p MemorySwapMax=0 --same-dir --quiet \
  .venv/bin/python -m pytest test/ -q
```

Correctness runs may share the GPU; timing may not (below). Do not kill processes you
did not start.

## Tests

`test/` is the pytest suite; markers select the tier:

```bash
pytest -m "not gpu"           # host-only: packaging, math, codegen determinism
pytest -m "gpu and not slow"  # smoke solves, bit-determinism, shapes, controller, constraint gates
pytest                        # + slow: codegen diff for every robot, gato.build dogfood, kernel harnesses
```

- `test/test_parity_golden.py` is the bit-parity gate for kernel changes: for every module
  in `test/receipt_modules.txt` it solves fixed problems on the pcg and bdsv paths (plus the
  constrained cases: limit ADMM + EE row, collision AL/ADMM, masked CONTACT_POS AL/ADMM) and
  compares `xu`, merits and iteration counts bitwise against `test/golden/*.npz`. A golden
  that moves means something changed, never how much (the solves are deliberately
  unconverged). Attribute the drift with the referee gates (`test_dynamics_fingerprint.py`,
  `test_f_ext.py`, `test_exact_hessian.py`, `test_rowgroups.py`) and
  `compute-sanitizer --tool memcheck` on the cheapest module first; only then
  `GATO_GOLDEN_REBASELINE=1 pytest test/test_parity_golden.py -k <case>` and say why in the
  commit message.
- `test/test_kernel_gates.py` builds and runs the five standalone harnesses in `test/cuda/`
  with `nvcc -DNDEBUG -arch=native` (slow). Shared-memory changes also get
  `compute-sanitizer --tool racecheck build/bsqp` (the C++ demo, built by default).
- A missing dependency is a broken environment, never a skip: `test/expected_skips.txt` is
  empty and CI fails on any host-tier skip.

## Signed receipt

GPU CI is a signed [pytest-gpu-proof](https://github.com/A2R-Lab/pytest-gpu-proof) receipt
(`gpu-proof.json` at the repo root, schema 3, plugin 0.4.0) produced on the lab GPU and
verified CPU-only by the `verify-gpu-proof` workflow against the signer's GitHub keys.
The receipt attests the full suite on exactly the `test/receipt_modules.txt` module set;
the fingerprint covers native sources and bindings, Python, tests, examples,
build tools/configuration, CI and dependency gitlinks (the exact scope is in
`pyproject.toml [tool.gpu_proof]` and `test/gpu-proof-policy.yaml`). Changes in
that scope require a fresh receipt before an authorized push. A missing receipt
fails CI; `test/expected_tests.txt` enforces the complete collected suite.

1. Commit the source (and any submodule pin bump); the tree must be clean.
2. Configure and build the receipt profile: `./tools/build.sh --profile receipt` (capped,
   as above). Each module embeds a content identity of its actual native inputs.
   The signer refuses missing or stale/unidentified binaries; reconfigure after
   native edits so CMake updates the identity and rebuilds affected modules.
3. `PYTHON=.venv/bin/python ./test/run_gpu_proof.sh` on the GPU box. It refuses a dirty
   tree, test-selection arguments, `PYTEST_ADDOPTS`, a wrong plugin version and
   a Python without the `[test]` deps. It installs nothing, runs the whole suite
   and writes `gpu-proof.json`.
4. Verify locally the way CI does:
   ```bash
   .venv/bin/gpu-proof verify --receipt gpu-proof.json --repo . \
     --policy test/gpu-proof-policy.yaml --expected-skips test/expected_skips.txt --require-gpu
   ```
5. `git add gpu-proof.json` and commit it on its own ("attest ..."), one SHA per receipt.
   Push only when authorized, then check the workflow run.

Never edit a signed receipt; a dirty-tree local pass is not a receipt.

## Timing

Timing needs an otherwise idle machine and an explicitly declared quiet window; nothing in
the test suite or the build measures performance. The coordinator entry point is
`examples/benchmarks/run_timing_handoff.sh`, which refuses to run without
`GATO_QUIET_WINDOW=1`, a clean tree, a committed receipt and the cooperative
`/tmp/a2rlab-timing.lock`:

```bash
examples/benchmarks/run_timing_handoff.sh --dry-run                         # prints the plan, touches nothing
GATO_QUIET_WINDOW=1 examples/benchmarks/run_timing_handoff.sh --suite checkpoint
GATO_QUIET_WINDOW=1 examples/benchmarks/run_timing_handoff.sh --suite all   # checkpoint, compile, calibrate
```

`--suite` is one of `checkpoint` (three iiwa14 N64 B1/8/128 repeats plus ten B128 Fig-7
scenarios), `seed-ab`, `boundaries`, `compile`, `calibrate` or `all`. The `boundaries`
suite pairs `MPCController.step` wall time with internal solver time on Indy7 and
iiwa14, N64, B1/8/128, three repeats. It excludes sensing, reference preparation,
simulation and actuator I/O. `all` selects checkpoint, compile and calibrate;
it does not include boundaries or seed-ab. Logs land in
`examples/benchmarks/night_logs/handoff_*/` with source, submodule, receipt and module
hashes recorded before and after. The narrower `run_merge_checkpoint.sh` takes the same
`--dry-run` / `GATO_QUIET_WINDOW=1` gate. Do not run heavy CPU work or compiles while a
timing leg is in flight, and do not quote numbers from a contended box. The paper-figure
scripts (`examples/paper-figures/`) consume the resulting CSVs; see
[figure protocols](../examples/paper-figures/README.md).
