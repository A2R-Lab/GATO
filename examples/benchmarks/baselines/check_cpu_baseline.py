"""Small optional CPU-baseline correctness gate; no timing collection."""
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import pysqpcpu

ROOT = Path(__file__).resolve().parents[3]
registry = json.loads((ROOT/'python/gato/_registry.json').read_text())
for plant in ('indy7','iiwa14'):
    metadata = registry[plant]
    path = str(ROOT/metadata['urdf'])
    model = pin.buildModelFromUrdf(path)
    q = np.zeros(model.nq)
    q[1] = .3
    data = model.createData()
    pin.framesForwardKinematics(model,data,q)
    expected = data.oMf[model.getFrameId(metadata['ee_frame'])].translation
    results = []
    for _ in range(2):
        solver = pysqpcpu.BatchThneed(urdf_filename=path,eepos_frame_name=metadata['ee_frame'],
                                     batch_size=2,num_threads=1,N=8,max_qp_iters=1)
        np.testing.assert_allclose(solver.eepos(q),expected,rtol=0,atol=1e-10)
        solver.sqp(np.r_[q,np.zeros(model.nv)],np.tile(expected,8))
        result = np.asarray(solver.get_results())
        assert result.shape == (2,solver.traj_len) and np.isfinite(result).all()
        results.append(result.copy())
    # External OSQP/CPU allocation paths differ at ~1e-9 on these fixtures;
    # this is a numerical repeatability gate, not a bitwise golden contract.
    np.testing.assert_allclose(results[0],results[1],rtol=1e-6,atol=1e-8)
    print(plant, 'FK/finite-solve/repeatability passed')
