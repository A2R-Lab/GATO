"""gato.tasks on every vendored arm (host tier, pinocchio only): the registered
end-effector frame resolves, the figure-8 generator lands its speed ratio from
the URDF velocity limits, and the pick-and-place generator draws reachable
goals inside the position limits."""
import hashlib
import itertools
import json
import math

import numpy as np
import pytest

import gato
from gato import tasks

pin = pytest.importorskip("pinocchio")
ARMS = ("indy7", "iiwa14", "fr3", "gen3", "ur5e")


def test_new_arm_holdout_inventory_and_provenance(repo_root):
    """Dataset integrity, not a rerun or a claim that every episode is feasible."""
    path = repo_root / "examples/benchmarks/data/limits_holdout/new_arms_20261018.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "12995516eb1f81507cbbec312b899fedb269870042d3a5dfaaba89aea6279e3e")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = set(itertools.product(("fr3", "gen3", "ur5e"), ("fig8", "pickplace"),
                                     range(100), ("none", "barrier", "al", "admm")))
    keys = [(r["plant"], r["task"], r["scenario"], r["mech"]) for r in rows]
    assert len(rows) == len(set(keys)) == 2400 and set(keys) == expected

    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return math.isfinite(value) if isinstance(value, (int, float)) else True

    scenarios = {}
    for row in rows:
        assert finite(row) and row["steps"] > 0
        assert row["sha"] == "cf605df" and row["seed"] == 20261018
        assert row["N"] == 64 and row["set"] == "holdout"
        assert row["quotable_times"] is False
        assert row["config"] == {"limits": "telemetry" if row["mech"] == "none" else row["mech"]}
        key = row["plant"], row["task"], row["scenario"]
        assert scenarios.setdefault(key, row["scenario_def"]) == row["scenario_def"]
        if row["task"] == "pickplace":
            assert 0 <= row["reached"] <= 3
            assert (row["time_to_all"] is not None) == (row["reached"] == 3)


@pytest.mark.parametrize("plant", ARMS)
def test_generators_on_every_arm(plant):
    model = pin.buildModelFromUrdf(str(gato.robot_urdf(plant)))
    ee = tasks.ee_frame_of(plant)
    assert ee == gato.robot_info(plant)["ee_frame"] and model.getFrameId(ee) < model.nframes
    assert tasks.within_q_limits(model, tasks.READY[plant], margin=tasks.Q_MARGIN)
    lim = tasks.urdf_limits(model)
    scs = tasks.fig8_scenarios(model, plant, 2, seed=1, ratio_range=(0.8, 1.2))
    assert len(scs) == 2
    for sc in scs:
        assert sc["plant"] == plant and len(sc["q0"]) == model.nq
        assert abs(sc["ratio_est"] / sc["ratio"] - 1.0) <= 0.15
        assert np.all(sc["q0"] > lim["q_lo"]) and np.all(sc["q0"] < lim["q_hi"])
        traj, sim_time = tasks.fig8_reference(sc, 64, periods=1)
        assert traj.shape[0] % 6 == 0 and traj.shape[0] // 6 >= 64 + int(sim_time / sc["dt"])
    pp = tasks.pickplace_scenarios(model, plant, 1, seed=1, n_goals=2, ramps=(0.0,))
    assert len(pp) == 1 and len(pp[0]["goals"]) == 2
    lo, hi = tasks.PICKPLACE_BOX[plant]
    for g in pp[0]["goals"]:
        assert np.all(np.asarray(g) >= lo - 1e-9) and np.all(np.asarray(g) <= hi + 1e-9)
