"""Frozen scenario generation and observational instrumentation contracts."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/'examples/paper-figures'


def harness():
    sys.path.insert(0,str(HERE))
    try:
        spec = importlib.util.spec_from_file_location('fig7_validation',HERE/'validate_fig7.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(HERE))


def test_fig7_holdout_is_frozen_and_separate():
    m = harness()
    protocol = json.loads(m.PROTOCOL.read_text())
    assert protocol['seed'] != 0 and protocol['scenarios'] == 100
    a = m.scenarios(protocol)
    np.random.seed(12345)  # independent of ambient RNG changes
    assert a == m.scenarios(protocol) and len(a) == 100
    assert a != m.scenarios({**protocol,'seed':0})
    assert protocol['arms']['stop-fe-bounded']['clip_torque']


def test_fig7_summary_separates_success_feasibility_and_completion(tmp_path):
    spec = importlib.util.spec_from_file_location('fig7_summary', HERE/'summarize_fig7_validation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(source='test',protocol_sha256='test',
        protocol=dict(arms={'test-arm':{}},batch_sizes=[8],scenarios=2))))
    telemetry = dict(command_effort_ratio=.5,applied_effort_ratio=.5,velocity_ratio=2.,
                     position_violation_rad=0.,payload_swing_max_rad=.2,
                     payload_relative_speed_max_rad_s=.1)
    episode = dict(arm='test-arm',batch=8,scenario=0,goal_outcomes=['reached']*5,telemetry=telemetry)
    (tmp_path/'test-arm-B8-000.json').write_text(json.dumps(episode))
    assert not module.summarize(tmp_path)['complete']
    episode = {**episode,'scenario':1,'goal_outcomes':['reached']*4+['timeout'],
               'telemetry':{**telemetry,'velocity_ratio':.5}}
    (tmp_path/'test-arm-B8-001.json').write_text(json.dumps(episode))
    summary = module.summarize(tmp_path)
    assert summary['complete']
    row, = summary['rows']
    assert row['episodes'] == 2 and row['successes'] == row['feasible_episodes'] == 1
    assert row['successful_and_feasible'] == 0 and row['maxima']['velocity_ratio'] == 2.
    (tmp_path/'duplicate-B8-001.json').write_text(json.dumps(episode))
    with pytest.raises(ValueError,match='duplicate'):
        module.summarize(tmp_path)


@pytest.mark.gpu
def test_fig7_instrumentation_preserves_unclipped_trajectory():
    m = harness()
    urdf, _, model = m.C.resolve_model('iiwa14')
    x = np.r_[m.IIWA14_START_CONFIGS['ready'],np.zeros(7)]
    pendulum = m.scenarios(json.loads(m.PROTOCOL.read_text()))[0]
    outputs = []
    for cls in (m.MPC_GATO,m.ObservedDriver):
        driver = cls(model,model_path=urdf,N=16,dt=.01,batch_size=1,plant_type='iiwa14',
                     pendulum_config=pendulum,params=m.PICKPLACE_SOLVER_PARAMS,linsys='pcg')
        _, stats = driver.run_mpc_goals(x,[m.PICKPLACE_DEFAULT_GOALS[0]],goal_timeout=.1,
                                       pace_by_solve_time=False,velocity_norm=2)
        outputs.append(stats)
    for field in ('joint_positions','joint_velocities','goal_distances'):
        np.testing.assert_array_equal(outputs[0][field],outputs[1][field])
    assert driver.audit['samples'] >= 100 and driver.audit['clipped_samples'] == 0


@pytest.mark.gpu
def test_fig7_identifier_observes_clamped_torque():
    m = harness()
    urdf, _, model = m.C.resolve_model('iiwa14')
    driver = m.ObservedDriver(model,model_path=urdf,N=16,dt=.01,batch_size=1,plant_type='iiwa14',
        pendulum_config=m.scenarios(json.loads(m.PROTOCOL.read_text()))[0],
        params=m.PICKPLACE_SOLVER_PARAMS,linsys='pcg',estimator='wid',clip_torque=True)
    observed = []
    driver.wrench_identifier.identify = lambda **kw: observed.append(kw['tau_applied'].copy())
    q,dq = driver._initial_sim_state(np.r_[m.IIWA14_START_CONFIGS['ready'],np.zeros(7)])
    requested = 2 * driver.effort
    q1,dq1 = driver.world.step(q,dq,requested,.001)
    driver._observe_substep(q,dq,q1,dq1,requested,.001)
    np.testing.assert_array_equal(observed[0],driver.effort)
    assert driver.audit['applied_effort_ratio'] == 1
    assert driver.audit['command_effort_ratio'] == 2
    assert driver.audit['clipped_samples'] == 1
