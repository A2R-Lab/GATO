# Development and validation

`./tools/install.sh --test --dev` creates the project-local `.venv`, installs the
pinned `grid-rbd` wheel with the `[test]` + `[dev]` extras (pinocchio, mujoco, scipy,
gymnasium, pytest, pytest-gpu-proof) and regenerates the vendored GRiD headers. That `.venv` is the only Python for this repo: receipts, tests and
timing all run from it. Never import models, environments or generated files from a
sibling checkout.

## Installing from a wheel: register_robot and the per-host cache

`pip install gato-mpc` (or `pip install .` from a checkout) ships GATO's native
sources under `gato/_native` and no compiled module. `gato.register_robot(name,
urdf, N=[...])` runs codegen into the cache (`GATO_CACHE_DIR`, default the
platform user cache, e.g. `~/.cache/gato`): `robots/<codegen_key>/<name>/` holds
`grid.cuh` + `limits.cuh`, `manifest.json` maps names to entries. The first
`BSQP(..., plant_type=name)` for a horizon compiles `bsqpN{N}_{name}` into
`modules/<name>/<module_key>/` (CMake on the bundled tree with
`GATO_PLANT_ROOT` pointing at the cache, `GATO_MODULE_OUTPUT_DIR` at the entry;
build tree removed on success; `build.log` and `meta.json` kept) and logs the
one-time cost. The module key hashes the robot's codegen key, N, variant, the
native source identity (`tools/native_identity.py`, the same content hash every
module embeds) and the toolchain (CUDA arch from `nvidia-smi`, nvcc, host
compiler, CMake, Python ABI tag, pybind11) — any change is a cache miss, never a
stale binary; `register_robot(..., build=True)` compiles up front. The vendored
robots are pre-registered from the bundled headers (`_cache.resolve`). Package
modules in `python/gato/` (the receipt profile) win over cache modules with the
same name; `gato.available()` lists both. `test/test_register_robot.py` is the
gate (its slow tier compiles into a temporary cache).

## Releasing to PyPI

`.github/workflows/publish.yml` is the only upload path (PyPI trusted
publishing, OIDC; no token is stored anywhere). It builds the universal wheel
and sdist, installs the wheel outside the checkout and runs
`tools/check_installed_package.py` without a GPU, then uploads to TestPyPI and,
for a `v*` tag, to PyPI — each upload behind a protected environment whose
reviewer approves by hand. One-time set-up, by a maintainer:

1. On pypi.org and test.pypi.org, add a *pending* trusted publisher for the
   project `gato-mpc`: owner `A2R-Lab`, repository `GATO`, workflow
   `publish.yml`, environment `pypi` (resp. `testpypi`). The project is
   created by its first upload.
2. In the GitHub repository settings create environments `testpypi` and
   `pypi`, each with a required reviewer; restrict `pypi` to tags `v*`.

Per release: the signed receipt must be valid at the release commit; bump
`version` in `pyproject.toml` and `python/gato/__init__.py`, date the CHANGELOG
entry, merge to `main`. Rehearse with `workflow_dispatch` (approve `testpypi`),
then on a GPU box: `pip install -i https://test.pypi.org/simple/
--extra-index-url https://pypi.org/simple gato-mpc` into a clean venv and run
`python -I tools/check_installed_package.py --work /tmp/gato-smoke --gpu`.
Release: push the tag `v<version>` (the workflow refuses a tag that does not
match `pyproject.toml`), approve `testpypi`, re-check, approve `pypi`; write
the GitHub release from `CHANGELOG.md` and attach `SHA256SUMS.txt` from the
build artifact. Wheels never contain compiled solver modules.

GRiD and GLASS are developed in their own repositories; GATO consumes both through
the released `grid-rbd` wheel, pinned to an exact version in `pyproject.toml` and
`python/gato/_grid.py` (which also pins the GLASS commit the wheel must bundle —
`gato._grid.require()` refuses any other install, at codegen, configure and test
time). The wheel supplies the code generator, URDFParser, the global `glass::`
headers and `grid_collision_geometry.cuh`; CMake asks `python/gato/_grid.py` for
the include directories. After a grid-rbd bump run `.venv/bin/python tools/regen_grid.py`
and `pytest test/test_codegen.py` (the vendored `gato/dynamics/<robot>/grid.cuh` must
regenerate byte-identically — never edit it by hand). A dependency update requires
rebuilding the receipt profile (every module embeds a content identity of the
dependency headers it was built from) and passing the existing numerical goldens.
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
`grid.cuh`, so one go2 compile peaks near 7 GB of RAM; `gato.build(...)` reconfigures
`build/` for its own module request, so re-run your usual configure afterwards
(`gato.register_robot` / example 05 build in the per-host cache and leave `build/` alone).

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

1. Commit the source (and any grid-rbd pin bump); the tree must be clean.
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
`examples/benchmarks/night_logs/handoff_*/` with source, dependency, receipt and module
hashes recorded before and after. The narrower `run_merge_checkpoint.sh` takes the same
`--dry-run` / `GATO_QUIET_WINDOW=1` gate. Do not run heavy CPU work or compiles while a
timing leg is in flight, and do not quote numbers from a contended box. The paper-figure
scripts (`examples/paper-figures/`) consume the resulting CSVs; see
[figure protocols](../examples/paper-figures/README.md).
