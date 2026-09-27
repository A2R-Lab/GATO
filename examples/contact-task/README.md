# Contact-wipe experiment

The iiwa14 wipe task compares position tracking (`pos`), a frozen torque-space
friction cone (`ucone`), and optimized contact-force slots (`fc`) on the same
24-scenario pool. Runs use fixed simulation pacing; solve-time statistics collected
on a shared GPU are not performance measurements.

From the repository root, install the example dependencies and build both iiwa14
N16 variants (or the full receipt profile):

```bash
./tools/install.sh --examples --dev
./tools/build.sh --profile receipt
.venv/bin/python examples/contact-task/run_wipe_cell.py --arm pos --scenarios 0 --depth 0.002 --out /tmp/gato-wipe-smoke
.venv/bin/python examples/contact-task/run_wipe_cell.py --arm fc --scenarios 0 --out /tmp/gato-wipe-smoke
.venv/bin/python examples/contact-task/summarize_wipe.py /tmp/gato-wipe-smoke
```

`fc` loads the named `_fc` module alongside the ordinary module. The position and
torque-cone arms use a press-depth parameter; the contact-force arm tracks its
force reference without that parameter. `--calibrate` sweeps press depths for the
baseline arms; inspect the reported force before choosing a depth for a full pool.

`bash examples/contact-task/run_wipe_pool.sh 0.002` runs all three arms and writes
a dated directory under `data/`. Each cell records its scenario, solver parameters,
metrics and simulator trace. These generated pickle pools and logs are local,
gitignored artifacts and are not included in a fresh clone. The summary compares
paired scenarios; use only trusted pickle files. `wipe_paper_assets.py` generates
tables and figures from an existing pool (see its command-line help).

See [the constraint documentation](../../docs/constraints.md) for force-slot,
friction-cone and mechanism conventions.
