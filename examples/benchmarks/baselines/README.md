# Optional CPU comparison baseline

The pinned `sqpcpu` submodule supplies the multi-threaded OSQP/QDLDL comparison
for Fig-3. It is not a GATO runtime dependency. It requires Pinocchio 3.8 headers;
GATO's installed Pinocchio 4.x does not supply the old aligned-vector header.
Keep the environments separate. Do not borrow a sibling project's environment.

From the GATO root on Linux/Python 3.12, with a C++ compiler, CMake 3.24 or newer,
Git, Eigen and urdfdom development packages installed:

```bash
python3 -m venv .venv-cpu-baseline
.venv-cpu-baseline/bin/pip install -r examples/benchmarks/baselines/requirements.txt
JOBS=1 systemd-run --user --scope -p MemoryMax=36G -p MemorySwapMax=0 --same-dir \
  bash examples/benchmarks/baselines/build_cpu_baseline.sh
source examples/benchmarks/baselines/sqpcpu_env.sh
"$GATO_CPU_PYTHON" examples/benchmarks/baselines/check_cpu_baseline.py
```

The helper finds the actual `cmeel.prefix`, refreshes CMake's dependency cache,
builds with one job by default (maximum two), and prints an import check. The
generated environment file selects the matching Python and native libraries.
The correctness check tests FK against Pinocchio and finite, repeatable batched
solves for both arms. It collects no timings.

The older cached baseline on the lab box linked to GRiD's environment. That is
not a fresh-install recipe. Keep its historical measurements, but record the
new baseline's environment and library hashes when recollecting comparisons.

`track_iiwa_fig8_bt.py` measures performance: run it only in an assigned exclusive
CPU/GPU window, using `"$GATO_CPU_PYTHON"` after sourcing the environment file.
Do not mix a rebuilt baseline into a previously collected figure without a
fresh, labeled comparison. The numerical protocol is documented in the
[paper-figure guide](../../paper-figures/README.md).
