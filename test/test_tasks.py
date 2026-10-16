"""gato.tasks on every vendored arm (host tier, pinocchio only): the registered
end-effector frame resolves, the figure-8 generator lands its speed ratio from
the URDF velocity limits, and the pick-and-place generator draws reachable
goals inside the position limits."""
import numpy as np
import pytest

import gato
from gato import tasks

pin = pytest.importorskip("pinocchio")
ARMS = ("indy7", "iiwa14", "fr3", "gen3", "ur5e")


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
