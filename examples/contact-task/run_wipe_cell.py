"""Run contact-wipe cells for ONE arm (one process = one module family).

    python run_wipe_cell.py --arm pos   --depth 0.002 --out data/wipe
    python run_wipe_cell.py --arm ucone --depth 0.002 --out data/wipe
    python run_wipe_cell.py --arm fc    --out data/wipe          # fc modules swapped in!
    python run_wipe_cell.py --arm pos --calibrate                # depth sweep, no pkl

The fc arm REQUIRES the build_fc modules on the package path (run_wipe_pool.sh
does the swap dance); pos/ucone run on default modules. Scenario selection via
--scenarios "0,1,2" (default: all 24).
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "python"))  # the repo package
sys.path.insert(0, _HERE)
import wipe_common as W


def build_mpc(arm, scenario, depth):
    import pinocchio as pin
    from gato.mpc_gato import MPC_GATO
    from gato.worlds import MuJoCoWorld
    from gato.config import DEFAULT_SOLVER_PARAMS

    model, q0, ee, tip, z_s, center = W.ready_press_geometry()
    world = MuJoCoWorld(W.URDF, plane=W.plane_cfg(center, scenario["stroke"]),
                        timestep=W.SIM_DT, record_contact=True)
    mpc = MPC_GATO(pin.buildModelFromUrdf(W.URDF), W.URDF, N=W.N_KNOTS,
                   dt=W.DT, batch_size=1, world=world,
                   # pin pcg: the committed n=24 pool + quiet quotes were
                   # measured under it (controller default is "auto" since 08-12)
                   solver_params={"linsys": "pcg"})
    s = mpc.solver

    if arm == "fc":
        if s.n_fc == 0:
            raise RuntimeError("fc arm needs the build_fc modules on the package "
                               "path (run via run_wipe_pool.sh)")
        s.set_fc_cost(1e-2)
        s.set_fc_ref(W.fc_reference())
        s.enable_u_cone(W.fc_cone_rows(s.nu, s.n_actuated),
                        mech="admm", rho=0.01)
    elif arm == "ucone":
        if s.n_fc != 0:
            raise RuntimeError("ucone arm must run on DEFAULT modules")
        C, d = W.frozen_pinv_cone_rows(model, q0, s.nu)
        s.enable_u_cone(C, d, mech="admm", rho=0.01)
    elif arm == "pos":
        if s.n_fc != 0:
            raise RuntimeError("pos arm must run on DEFAULT modules")
    else:
        raise ValueError(arm)

    x0 = np.concatenate([q0, np.zeros(7)])
    return mpc, model, q0, x0, ee, tip, center, DEFAULT_SOLVER_PARAMS


def run_cell(arm, scenario, depth, outdir=None):
    depth = 0.0 if arm == "fc" else depth  # fc has no depth knob: that's the claim
    mpc, model, q0, x0, ee, tip, center, params = build_mpc(arm, scenario, depth)
    traj, t_wipe_start, t_end = W.wipe_reference(
        center, ee, tip, scenario["theta"], scenario["stroke"], depth)
    _, stats = mpc.run_mpc_fig8(x0, traj, sim_dt=W.SIM_DT,
                                sim_time=t_end + W.SIM_DT,
                                pace_by_solve_time=False)
    m = W.compute_metrics(stats, mpc.world, traj, t_wipe_start, t_end)
    if outdir:
        proto = W.protocol_stamp(arm, scenario, depth, mpc.world, params,
                                 extra={"tip_drop": tip})
        W.save_cell(os.path.join(outdir, f"{arm}_s{scenario['id']:02d}.pkl"),
                    proto, m, stats, mpc.world)
    return m


def calibrate(arm, depths):
    """Press-hold force response at theta=0: pick the depth whose settled fn is
    closest to F_SET (the pos/ucone force knob; fc needs none)."""
    sc = dict(W.SCENARIOS[0])
    sc["stroke"] = 0.0  # press-hold only: hold at the press point
    rows = []
    for depth in depths:
        m = run_cell(arm, sc, depth)
        rows.append((depth, m["fn_mean"], m["fn_rms_err"]))
        print(f"  depth {depth*1000:5.1f} mm -> fn {m['fn_mean']:7.2f} N "
              f"(rms err {m['fn_rms_err']:6.2f})")
    best = min(rows, key=lambda r: abs(r[1] - W.F_SET))
    print(f"CALIBRATED depth = {best[0]*1000:.1f} mm (fn {best[1]:.2f} N, target {W.F_SET})")
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pos", "ucone", "fc"])
    ap.add_argument("--depth", type=float, default=0.0,
                    help="press depth below the surface [m] (pos/ucone)")
    ap.add_argument("--scenarios", default="",
                    help="comma-separated scenario ids (default all)")
    ap.add_argument("--out", default="", help="output dir for per-cell pkls")
    ap.add_argument("--calibrate", action="store_true",
                    help="depth sweep instead of the scenario pool")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.arm, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        return

    ids = ([int(x) for x in args.scenarios.split(",") if x != ""]
           if args.scenarios else [s["id"] for s in W.SCENARIOS])
    for sid in ids:
        sc = W.SCENARIOS[sid]
        m = run_cell(args.arm, sc, args.depth, outdir=args.out or None)
        print(f"[{args.arm} s{sid:02d} th={np.rad2deg(sc['theta']):5.1f} "
              f"L={sc['stroke']*100:.0f}cm] fn_rms {m['fn_rms_err']:6.2f} N | "
              f"path_rms {m['path_rms']*1000:6.2f} mm | "
              f"cone_viol {m['cone_viol_mean']:6.3f} N | "
              f"loss {m['contact_loss_frac']*100:5.1f}% | "
              f"solve {m['solve_ms_mean']:.2f} ms")


if __name__ == "__main__":
    main()
