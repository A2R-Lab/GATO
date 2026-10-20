"""Timing inventory and configuration provenance, not performance thresholds."""
import hashlib
import itertools
import json
import math
import subprocess

import numpy as np


def test_timing_launcher_isolates_results_and_calibration(repo_root):
    launcher = repo_root / "examples/benchmarks/run_timing_w2a.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text()
    assert 'QUICKQ=(--out "$LOGDIR/results.jsonl")' in text
    assert 'export GATO_LINSYS_TUNING=$LOGDIR/tuning-proposed.json' in text
    assert 'fatal "commit tracked changes before timing"' in text
    assert 'fence-identity' in text and 'fence-gpu' in text
    assert '|| exit 1' in text


def test_oct20_controller_replay_evidence(repo_root):
    root = repo_root / "examples/benchmarks/data/controller_replay"
    hashes = (
        "cc10c0ad6cece668d79cda2ab2561cc0eb20aec82bd711fa8d9eb29b3578896a",
        "cef4ef58747af39b473603b06942fd7640c37b2712b810ac0b348616cf2a18a0",
    )
    plants = ("indy7", "iiwa14", "fr3", "gen3", "ur5e")
    kinds = ("numpy", "torch", "jax")
    expected = set(itertools.product(plants, (1, 8), ("off", "admm"), range(3), kinds))
    plans, inputs, identities = {}, {}, {}
    for run, digest in enumerate(hashes, 1):
        raw = root / f"20261020_run{run}.jsonl"
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == digest
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        assert len(rows) == len(expected) == 180
        assert {(r["plant"], r["batch"], r["limits"], r["repeat"], r["kind"])
                for r in rows} == expected
        for r in rows:
            assert r["mode"] == "timing" and r["quotable"] and r["parity"] == "PASS"
            assert r["provenance"]["sha"] == "519c57a2b1a9f819ed06d89198b966c479d2075a"
            assert not r["provenance"]["dirty"]
            assert r["N"] == 64 and r["dt"] == .01
            assert r["samples"] == 200 and r["warmup"] == 20
            assert r["versions"] == dict(torch="2.14.1", jax="0.11.2",
                                         numpy="2.4.6", **{"grid-rbd": "0.5.3"})
            order = r["repeat"]
            assert r["order"] == list(kinds[order:] + kinds[:order])
            assert r["effective_limits"] == ("" if r["limits"] == "off" else "admm")
            assert r["admm_exit"] == [0, 0 if r["limits"] == "off" else .01, 1]
            assert r["linsys"] == "auto" and r["bdsv_threshold"] == .1
            assert r["admm_linsys"] == "bdsv_factor"
            assert r["iterations"] == [1] * 220
            assert len(r["pred_err"]) == 220 and np.isfinite(r["pred_err"]).all()
            assert math.isfinite(r["row_violation_max"])
            for fence in (r["box_before"], r["box_after"]):
                assert not fence["foreign_pids"] and fence["load1"] < 1
                assert "RTX 5090" in fence["gpu"] and "615.71.09" in fence["gpu"]
            cell = r["plant"], r["batch"], r["limits"]
            assert plans.setdefault(cell, r["plan_telemetry_sha256"]) == r["plan_telemetry_sha256"]
            assert inputs.setdefault(r["plant"], r["input_sha256"]) == r["input_sha256"]
            assert identities.setdefault(r["plant"], r["native_source_id"]) == r["native_source_id"]
            for boundary in ("call_complete", "native_solve"):
                samples = np.asarray(r[f"{boundary}_samples_ms"])
                assert samples.shape == (200,) and np.isfinite(samples).all()
                assert (samples >= 0).all()
                stats = r[f"{boundary}_ms"]
                values = [*np.percentile(samples, [50, 90, 99]), samples.max(), samples.mean()]
                np.testing.assert_allclose([stats[k] for k in ("p50", "p90", "p99", "max", "mean")],
                                           values, rtol=1e-12, atol=1e-12)


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
