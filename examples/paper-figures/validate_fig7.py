#!/usr/bin/env python3
"""Resumable held-out correctness pool. No performance results are collected.

The frozen JSON protocol is reviewed before collection. Each completed episode
is saved separately; STOP in the output directory pauses between episodes.
Resume uses the same source/protocol/binary identities. No existing result is
overwritten. Keep outputs under docs/open-tasks/ until a reviewed summary is ready.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import _common as C
from _pickplace_runner import PICKPLACE_DEFAULT_GOALS, PICKPLACE_SOLVER_PARAMS
from gato.config import IIWA14_START_CONFIGS
from gato.mpc_gato import MPC_GATO

PROTOCOL = Path(__file__).with_name('fig7_validation_protocol.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenarios(protocol):
    rng = np.random.default_rng(protocol['seed'])
    out = []
    for _ in range(protocol['scenarios']):
        length = rng.uniform(*protocol['length_m'])
        damping = rng.uniform(*protocol['damping'])
        magnitude = rng.uniform(*protocol['initial_angle_rad'])
        direction = rng.normal(size=3)
        out.append(dict(mass=protocol['mass_kg'], length=length, damping=damping,
                        initial_angle=(direction / np.linalg.norm(direction) * magnitude).tolist()))
    return out


class ObservedDriver(MPC_GATO):
    """Experiment-only instrumentation; the production controller is unchanged."""
    def __init__(self, *args, clip_torque=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit = dict(samples=0, command_effort_ratio=0., applied_effort_ratio=0.,
                          velocity_ratio=0., position_violation_rad=0.,
                          payload_swing_max_rad=0., payload_swing_final_rad=0.,
                          payload_relative_speed_max_rad_s=0., clipped_samples=0)
        # Use the supplied robot-only model's URDF limits, not the appended payload.
        robot = self.solver_model
        self.effort = np.asarray(robot.effortLimit)
        self.velocity = np.asarray(robot.velocityLimit)
        self.lower, self.upper = np.asarray(robot.lowerPositionLimit), np.asarray(robot.upperPositionLimit)
        self.audit_data = self.model.createData()
        self.last_command = self.last_applied = None
        original_step = self.world.step

        def step(q, dq, u, dt):
            self.last_command = np.asarray(u).copy()
            self.last_applied = np.clip(u, -self.effort, self.effort) if clip_torque else np.asarray(u)
            return original_step(q, dq, self.last_applied, dt)
        self.world.step = step

    def _observe_substep(self, q_pre, dq_pre, q_post, dq_post, tau, sim_dt):
        # Identifier must see the torque actually applied, including the safety clamp.
        super()._observe_substep(q_pre, dq_pre, q_post, dq_post, self.last_applied, sim_dt)
        import pinocchio as pin
        if not all(np.isfinite(v).all() for v in (q_post, dq_post, self.last_applied)):
            raise FloatingPointError('Nonfinite simulated state/control')
        a = self.audit
        a['samples'] += 1
        a['clipped_samples'] += int(np.any(self.last_applied != self.last_command))
        a['command_effort_ratio'] = max(a['command_effort_ratio'], float(np.max(np.abs(self.last_command)/self.effort)))
        a['applied_effort_ratio'] = max(a['applied_effort_ratio'], float(np.max(np.abs(self.last_applied)/self.effort)))
        a['velocity_ratio'] = max(a['velocity_ratio'], float(np.max(np.abs(dq_post[:self.nv_robot])/self.velocity)))
        q = q_post[:self.nq_robot]
        a['position_violation_rad'] = max(a['position_violation_rad'], float(np.max(np.maximum(self.lower-q,q-self.upper))), 0.)
        pin.forwardKinematics(self.model, self.audit_data, q_post)
        # Bob lies on the pendulum joint's -Z axis; angle to world gravity-down.
        a['payload_swing_final_rad'] = float(np.arccos(np.clip(self.audit_data.oMi[-1].rotation[2,2],-1,1)))
        a['payload_swing_max_rad'] = max(a['payload_swing_max_rad'], a['payload_swing_final_rad'])
        a['payload_relative_speed_max_rad_s'] = max(a['payload_relative_speed_max_rad_s'], float(np.linalg.norm(dq_post[self.nv_robot:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--arms', nargs='+')
    parser.add_argument('--limit', type=int, help='first k held-out scenarios; partial evidence, never relabel as the full pool')
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    arms = args.arms or list(protocol['arms'])
    if set(arms) - protocol['arms'].keys(): parser.error('unknown arm')
    if args.limit is not None and not 1 <= args.limit <= protocol['scenarios']: parser.error('invalid limit')
    root = Path(C.REPO)
    status = subprocess.check_output(['git','status','--porcelain','--ignore-submodules=untracked'],cwd=root,text=True)
    if status.strip(): raise RuntimeError('Commit reviewed sources before held-out collection')
    sys.path.insert(0,str(root/'tools'))
    from native_identity import native_identity
    import gato.bsqpN16_iiwa14 as native
    if native.NATIVE_SOURCE_ID != native_identity(root,'iiwa14'): raise RuntimeError('Stale native module')
    identity = dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                    protocol_sha256=sha(PROTOCOL), binary_sha256=sha(native.__file__),
                    protocol=protocol, scenarios=scenarios(protocol))
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = args.output/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity: raise RuntimeError('Resume identity mismatch; use a new directory')
    else:
        with manifest.open('x') as f: json.dump(identity,f,indent=2)
    urdf, _, model = C.resolve_model('iiwa14')
    x0 = np.r_[IIWA14_START_CONFIGS['ready'], np.zeros(model.nv)]
    for index, pendulum in enumerate(identity['scenarios'][:args.limit]):
        for arm in arms:
            config = protocol['arms'][arm]
            for batch in protocol['batch_sizes']:
                if (args.output/'STOP').exists(): print('Paused at episode boundary',flush=True); return
                path = args.output/f'{arm}-B{batch}-{index:03}.json'
                if path.exists(): continue
                with (args.output/'run.log').open('a') as log, contextlib.redirect_stdout(log):
                    driver = ObservedDriver(model, model_path=urdf, N=16, dt=.01, batch_size=batch,
                        pendulum_config=pendulum, plant_type='iiwa14', linsys='pcg',
                        params={**PICKPLACE_SOLVER_PARAMS,'ctrl_lim_cost':config['ctrl_lim_cost']},
                        estimator=config['estimator'], clip_torque=config['clip_torque'])
                    _, stats = driver.run_mpc_goals(x0,PICKPLACE_DEFAULT_GOALS,sim_dt=.001,
                        goal_timeout=5.,goal_threshold=.05,velocity_threshold=1.,velocity_norm=2,
                        pace_by_solve_time=False,settle_time=config['dwell_s'],goal_ramp=config['ramp_s'])
                result = dict(scenario=index,arm=arm,batch=batch,goal_outcomes=stats['goal_outcomes'],
                    completion_sim_s=stats['time_to_all_reached'],goal_events=stats['goal_events'],telemetry=driver.audit)
                with path.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
                print(path.name, stats['goal_outcomes'],flush=True)
    print('Requested held-out episodes complete; inspect task success AND feasibility',flush=True)


if __name__ == '__main__': main()
