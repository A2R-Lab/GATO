"""Timing inventory and configuration provenance, not performance thresholds."""
import hashlib
import itertools
import json
import math
import subprocess


def test_timing_launcher_isolates_results_and_calibration(repo_root):
    launcher = repo_root / "examples/benchmarks/run_timing_w2a.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text()
    assert 'QUICKQ=(--out "$LOGDIR/results.jsonl")' in text
    assert 'export GATO_LINSYS_TUNING=$LOGDIR/tuning-proposed.json' in text
    assert 'fatal "commit tracked changes before timing"' in text
    assert 'fence-identity' in text and 'fence-gpu' in text
    assert '|| exit 1' in text


def test_oct19_timing_inventory_and_effective_configuration(repo_root):
    root = repo_root / "examples/benchmarks/data/admm_time_matrix"
    raw = root / "20261019.jsonl"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == "8758fbac1d0950a12a4a31cfee7dc02afb76348d7d7b2671b49601001c81b59c"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len(rows) == 92
    assert all(r["sha"] == "34a4d30" and r["quotable"] for r in rows)
    matrix = [r for r in rows if r["leg"] == "matrix"]
    cells = ("free", "fixed10-pcg", "fixed10-fac", "exit-pcg", "exit-fac", "exit-auto",
             "exit-fac-ce0", "exit-fac-ce2", "exit3-fac", "default")
    assert len(matrix) == 80
    assert {(r["plant"], r["N"], r["kick_every"], r["cell"]) for r in matrix} == set(
        itertools.product(("indy7", "iiwa14"), (32, 64), (0, 25), cells))
    for r in matrix:
        assert r["steps"] == 600
        assert all(math.isfinite(v) and v >= 0 for v in r["solve_ms"].values())
        assert r["effective_limits"] == ("" if r["cell"] == "free" else "admm")
        if r["cell"] in ("free", "fixed10-pcg", "fixed10-fac"):
            assert r["effective_admm_exit"] == [0, 0, 1]
        if r["cell"].startswith("fixed10"):
            assert r["inner_it_per_sqp_it"] == 10
    fit = [r for r in rows if r["leg"] == "fit"]
    assert len(fit) == 4 and all(r["rc"] == 0 for r in fit)
    assert {(r["plant"], r["N"]) for r in fit} == set(itertools.product(("indy7", "iiwa14"), (32, 64)))
    deadline = [r for r in rows if r["leg"] == "deadline"]
    assert len(deadline) == 8
    assert {(r["plant"], r.get("budget_frac")) for r in deadline} == set(
        itertools.product(("indy7", "iiwa14"), (None, 1., .8, .6)))
    for r in deadline:
        assert r["steps"] == 600
        assert sum(r["iters_hist"].values()) == r["steps"]
        assert all(math.isfinite(v) and v >= 0 for v in r["wall_ms"].values())
    tuning = root / "tuning_20261019.json"
    assert hashlib.sha256(tuning.read_bytes()).hexdigest() == "a0af16538dd5a9c1395f8de068a296bdcf1686f6669e1057361e7618202d8aca"
    entries = json.loads(tuning.read_text())
    assert len(entries) == 4
    assert all(v["admm_provenance"]["sha"] == "34a4d30" for v in entries.values())
