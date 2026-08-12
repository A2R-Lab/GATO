"""Regenerate Fig-5 (Case Study 2): fixed-disturbance rejection.

Paper IV-D: a 6-DoF Indy7 tracks the figure-8 EE trajectory while an unmodeled
constant external force is applied at the end effector in the -Z direction. GATO
runs an "online hypothesize-and-test" batch: M trajectory-optimization problems
differing only in the assumed external force (sampled on a sphere around the prior
estimate), and applies the control from the hypothesis whose predicted motion best
matches the observed state. This batch force-estimation lives in MPC_GATO
(``update_force_batch`` + ``evaluate_best_trajectory``, auto-enabled for M>3).

We sweep disturbance magnitude (20..80 N) x batch size and record the steady-state
tracking error and total joint velocity (Fig-5 left), plus the realized EE
trajectories at 50 N for M in {1,32,128} (Fig-5 right) — modest batch sizes (~32)
reject the disturbance best before added latency dominates.

Examples::
    python examples/paper-figures/reproduce_fig5_disturbance.py            # full sweep
    python examples/paper-figures/reproduce_fig5_disturbance.py --quick    # fast smoke
    python examples/paper-figures/reproduce_fig5_disturbance.py --replot   # plot saved data
"""
import argparse
import numpy as np

import _common as C

N = 64
DT = 0.01
SIM_TIME = 6.0          # a couple figure-8 cycles for a steady-state estimate
STEADY_FRAC = 0.5       # average the back half of the run for "steady-state"


def _run_one(model, urdf, M, force_N, sim_time):
    from gato.mpc_gato import MPC_GATO
    from gato.common import figure8
    from gato.config import FIG8_DEFAULT_PARAMS, INDY7_START_CONFIGS
    f_ext = np.array([0.0, 0.0, -float(force_N), 0.0, 0.0, 0.0])
    mpc = MPC_GATO(model, model_path=urdf, N=N, dt=DT, batch_size=M, plant_type="indy7",
                   constant_f_ext=f_ext, track_full_stats=False,
                   # paper numbers were measured under the pcg path (controller
                   # default is "auto" since 08-12)
                   solver_params={"linsys": "pcg"})
    fig8 = figure8(DT, **FIG8_DEFAULT_PARAMS)
    x0 = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(model.nv)))
    _, st = mpc.run_mpc_fig8(x0, fig8, sim_dt=0.001, sim_time=sim_time, pace_by_solve_time=False)
    err = np.asarray(st["goal_distances_knot0"])
    vel = np.asarray([np.sum(np.abs(v)) for v in st["joint_velocities"]])  # total joint vel / step
    k = int(len(err) * STEADY_FRAC)
    return {"track_err": float(np.mean(err[k:])), "joint_vel": float(np.mean(vel[k:])),
            "ee_actual": np.asarray(st["ee_actual"]), "fig8": fig8}


def sweep(forces, batch_sizes, sim_time):
    urdf, _, model = C.resolve_model("indy7")
    C.require_module("indy7", N)
    data = {"forces": forces, "batch_sizes": batch_sizes, "grid": {}, "traj": {}}
    for F in forces:
        data["grid"][F] = {}
        for M in batch_sizes:
            print(f"  disturbance {F} N, batch {M}")
            r = _run_one(model, urdf, M, F, sim_time)
            data["grid"][F][M] = {"track_err": r["track_err"], "joint_vel": r["joint_vel"]}
    return data


def collect_traj(traj_force, traj_batches, sim_time):
    """EE trajectories at a representative disturbance for the Fig-5 right panel."""
    urdf, _, model = C.resolve_model("indy7")
    out = {"force": traj_force, "ee": {}, "fig8": None}
    for M in traj_batches:
        print(f"  EE-traj: {traj_force} N, batch {M}")
        r = _run_one(model, urdf, M, traj_force, sim_time)
        out["ee"][M] = r["ee_actual"]
        out["fig8"] = r["fig8"]
    return out


def plot_sweep(data):
    plt = C.set_paper_rcParams()
    forces, batch_sizes = data["forces"], data["batch_sizes"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for M in batch_sizes:
        err = [data["grid"][F][M]["track_err"] for F in forces]
        vel = [data["grid"][F][M]["joint_vel"] for F in forces]
        c = C.batch_color(M)
        ax1.plot(forces, err, "o-", color=c, label=f"M={M}")
        ax2.plot(forces, vel, "o-", color=c, label=f"M={M}")
    ax1.set_xlabel("Disturbance Force (N)"); ax1.set_ylabel("Tracking Error (m)"); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Disturbance Force (N)"); ax2.set_ylabel("Total Joint Velocity (rad/s)"); ax2.grid(True, alpha=0.3)
    ax1.legend(title="Batch Size", fontsize=9)
    plt.tight_layout()
    C.savefig(fig, "fig5_disturbance_sweep")


def plot_traj(traj):
    plt = C.set_paper_rcParams()
    fig8 = np.asarray(traj["fig8"]).reshape(-1, 6)
    Ms = sorted(traj["ee"])
    fig, axes = plt.subplots(1, len(Ms), figsize=(5 * len(Ms), 5), squeeze=False)
    for ax, M in zip(axes[0], Ms):
        ax.plot(fig8[:, 0], fig8[:, 2], "k--", lw=1, alpha=0.6, label="Reference")
        ee = np.asarray(traj["ee"][M])
        ax.plot(ee[:, 0], ee[:, 2], color=C.batch_color(M), lw=1.5, label=f"M={M}")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title(f"Batch Size = {M}")
        ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(f"EE trajectory under {traj['force']:.0f} N disturbance")
    plt.tight_layout()
    C.savefig(fig, "fig5_ee_traj")


def main():
    p = argparse.ArgumentParser(description="Regenerate Fig-5 (CS2 disturbance rejection).")
    C.add_repro_args(p)
    p.add_argument("--forces", default="20,30,40,50,60,70,80")
    p.add_argument("--batch-sizes", default="1,4,8,16,32,64,128")
    p.add_argument("--traj-force", type=float, default=50.0)
    p.add_argument("--traj-batches", default="1,32,128")
    p.add_argument("--sim-time", type=float, default=SIM_TIME)
    args = p.parse_args()
    np.random.seed(args.seed)

    if args.replot:
        data = C.load_data("fig5_disturbance")
        traj = C.load_data("fig5_traj")
    else:
        forces = [float(x) for x in args.forces.split(",")]
        batch_sizes = C.parse_int_list(args.batch_sizes)
        traj_batches = C.parse_int_list(args.traj_batches)
        sim_time = args.sim_time
        if args.quick:
            forces, batch_sizes, traj_batches, sim_time = [50.0], [1, 8], [1, 8], 1.5
            print("[quick] tiny subset — NOT paper numbers")
        data = sweep(forces, batch_sizes, sim_time)
        traj = collect_traj(args.traj_force if not args.quick else 50.0, traj_batches, sim_time)
        C.save_data(data, "fig5_disturbance")
        C.save_data(traj, "fig5_traj")

    plot_sweep(data)
    plot_traj(traj)


if __name__ == "__main__":
    main()
