"""Fixed-pacing pick-place correctness traces, never performance measurements.

Run selected one-based scenarios with the paper harness's seed/distribution and
unchanged goal gates. Each repeat must agree on trajectories and outcomes. Keep
both successes and failures; outputs contain physical simulation time, not solve
latency. Example (GPU correctness; shared-box coordination still applies):

  .venv/bin/python examples/paper-figures/check_pickplace.py --scenarios 1,2 \
      --out /tmp/gato-pickplace-check
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np

import _common as C
from _pickplace_runner import (PICKPLACE_DEFAULT_GOALS, PICKPLACE_MPC_DEFAULTS,
                               PICKPLACE_SOLVER_PARAMS, sample_pendulum_params)
from gato.config import IIWA14_START_CONFIGS
from gato.mpc_gato import MPC_GATO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenarios', default='1,2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--estimator', default='fe', choices=['fe', 'wid'])
    p.add_argument('--out', type=Path, required=True, help='new directory; never overwritten')
    args = p.parse_args()
    ids = [int(i) for i in args.scenarios.split(',')]
    if not ids or min(ids) < 1 or args.repeats < 1 or args.batch < 1:
        p.error('scenario ids, repeats and batch must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    scenarios = [sample_pendulum_params() for _ in range(max(ids))]
    urdf, _, model = C.resolve_model('iiwa14')
    x = np.concatenate([IIWA14_START_CONFIGS['ready'], np.zeros(model.nv)])
    summary = dict(source=C.bench.git_provenance(), seed=args.seed, batch=args.batch,
                   protocol='unit-quaternion-pendulum-v2', timing=False,
                   solver_params=PICKPLACE_SOLVER_PARAMS, gates=PICKPLACE_MPC_DEFAULTS, estimator=args.estimator,
                   urdf_sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest(), results=[])
    for scenario in ids:
        previous = None
        for repeat in range(args.repeats):
            mpc = MPC_GATO(model, urdf, N=16, dt=0.01, batch_size=args.batch,
                           params=PICKPLACE_SOLVER_PARAMS, linsys='pcg',
                           pendulum_config=scenarios[scenario - 1], track_full_stats=True,
                           estimator=args.estimator)
            # The legacy driver prints durations even in fixed-pacing mode;
            # discard that display, never use or persist its timing fields.
            with contextlib.redirect_stdout(io.StringIO()):
                _, stats = mpc.run_mpc_goals(x, PICKPLACE_DEFAULT_GOALS, sim_dt=0.001,
                                            **PICKPLACE_MPC_DEFAULTS)
            trace = {k: np.asarray(stats[k]) for k in
                     ('timestamps', 'goal_distances', 'ee_actual', 'joint_positions',
                      'joint_velocities', 'best_trajectory_id', 'sqp_iters', 'pcg_iters',
                      'pred_errors', 'force_estimates')}
            for key, values in trace.items():
                if not np.isfinite(values).all():
                    raise AssertionError(f'scenario {scenario}: nonfinite {key}')
                if previous is not None:
                    np.testing.assert_array_equal(values, previous[0][key])
            if previous is not None:
                assert stats['goal_events'] == previous[1]
            previous = trace, stats['goal_events']
            np.savez_compressed(args.out / f'scenario-{scenario}-repeat-{repeat + 1}.npz', **trace)
            result = dict(scenario=scenario, repeat=repeat + 1,
                          outcomes=stats['goal_outcomes'], events=stats['goal_events'],
                          time_to_all_reached=stats['time_to_all_reached'])
            summary['results'].append(result)
            print(json.dumps(result), flush=True)
        del mpc
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Finite/deterministic correctness collection complete: {args.out}; review task outcomes separately.')


if __name__ == '__main__':
    main()
