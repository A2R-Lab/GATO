"""Regenerate Fig-4 (Case Study 1): online hyperparameter (rho) optimization.

Paper IV-C: a 7-DoF KUKA iiwa14, horizon N=64, h=0.05 s, 100 SQP iterations from
zero-initialized state/control on random workspace goals. The batched solver
samples the damping parameter rho (log-spaced 1e-8..1e1, adapted each SQP iter via
the line search) across the batch; a single-solve baseline uses rho=1e-1. We plot
the average normalized best-merit vs SQP iteration per batch size — larger batches
converge faster (Fig-4).

DEFAULT = REGENERATE on the GPU (this reproduces our bundled
``examples/gato_hparam_batch_results.pkl``: 24 cost-config combos x 50 random
targets). Use ``--replot`` to skip the GPU and render from that bundled data.

NOTE on paper fidelity: the paper text states "100 runs each with 81 different
values for Q and R". Our recovered/bundled data (the actual published figure)
used 50 targets x a 24-combo Q/R grid; the defaults here reproduce THAT. Override
with --num-targets / --max-iters; the exact paper Q/R grid is a backlog item
pending the student's confirmation.

Examples::
    python examples/paper-figures/reproduce_fig4_hparam.py            # full regen (GPU)
    python examples/paper-figures/reproduce_fig4_hparam.py --quick    # fast smoke
    python examples/paper-figures/reproduce_fig4_hparam.py --replot   # no GPU, bundled data
"""
import argparse
import numpy as np

import _common as C

# Fig-4-specific label palette (batch-size labels + the single-solve rho baseline)
COLOR_PALETTE = {
    "1 (ρ=1e-1)": "#003192", "1 (ρ=1e-3)": "#56B4E9",
    "2": "#8B4513", "4": "#747474", "8": "#7030A0",
    "16": "#F19759", "32": "#00693E", "64": "#00693E", "128": "#FF6600",
}

# fixed solver settings (match the notebook / recovered data)
N = 64
DT = 0.05
MAX_PCG_ITERS = 200
PCG_TOL = 1e-3
MU = 1.0
Q_LIM_COST = VEL_LIM_COST = CTRL_LIM_COST = 0.0
RHO_MIN_EXP, RHO_MAX_EXP = -8, 1
# the 2x3x2x2 = 24-combo Q/R cost grid that produced the bundled figure
Q_LIST = [10.0, 1.0]
QD_LIST = [1e-1, 1e-3, 1e-5]
U_LIST = [1e-6, 1e-7]
N_LIST = [100.0, 10.0]

RECOVERED = "examples/gato_hparam_batch_results.pkl"


def build_solver(urdf, B, q_cost, qd_cost, u_cost, N_cost, max_iters, *, rho_batch=None, rho=1e-3):
    from bsqp.interface import BSQP
    return BSQP(
        model_path=urdf, batch_size=B, N=N, dt=DT,
        max_sqp_iters=max_iters, kkt_tol=0.0, max_pcg_iters=MAX_PCG_ITERS,
        pcg_tol=PCG_TOL, solve_ratio=1.0, mu=MU,
        q_cost=q_cost, qd_cost=qd_cost, u_cost=u_cost, N_cost=N_cost,
        q_lim_cost=Q_LIM_COST, vel_lim_cost=VEL_LIM_COST, ctrl_lim_cost=CTRL_LIM_COST,
        rho=rho, rho_batch=rho_batch, adapt_rho=True, plant_type="iiwa14",
    )


def sample_goal():
    return np.array([np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8),
                     np.random.uniform(0.2, 0.8), 0.0, 0.0, 0.0], dtype=np.float32)


def _best_curve_from_stats(stats):
    best = stats.get("best_merit_per_iter", None)
    if best is None:
        ls = np.asarray(stats.get("min_merit", None))
        best = np.min(ls, axis=1) if ls.ndim == 2 else ls
    return np.asarray(best, dtype=np.float32).reshape(-1)


def _curve_for_B(urdf, B, costs, goal, max_iters):
    """Batched rho sweep for batch size B -> normalized best-merit curve."""
    q_cost, qd_cost, u_cost, N_cost = costs
    fractions = np.arange(1, B + 1) / (B + 1)
    rho_vals = np.power(10, RHO_MIN_EXP + fractions * (RHO_MAX_EXP - RHO_MIN_EXP)).astype(np.float32)
    solver = build_solver(urdf, B, q_cost, qd_cost, u_cost, N_cost, max_iters, rho_batch=rho_vals)
    nx, nu = solver.nx, solver.nu
    x0_B = np.tile(np.zeros(nx, dtype=np.float32), (B, 1))
    ref_B = np.tile(np.tile(goal, N).astype(np.float32), (B, 1))
    XU_B = np.zeros((B, solver.N * (nx + nu) - nu), dtype=np.float32)
    XU_B[:, :nx] = x0_B
    solver.solve(x0_B, ref_B, XU_B)
    stats = solver.get_stats()
    denom = float(stats.get("best_initial_merit", np.nan))
    curve = _best_curve_from_stats(stats)
    curve = curve / denom if (denom == denom and denom != 0) else curve
    return np.r_[1.0, curve]


def _curve_adaptive(urdf, costs, goal, rho, max_iters):
    """Single-solve adaptive-rho baseline -> normalized best-merit curve."""
    q_cost, qd_cost, u_cost, N_cost = costs
    solver = build_solver(urdf, 1, q_cost, qd_cost, u_cost, N_cost, max_iters, rho_batch=None, rho=rho)
    nx, nu = solver.nx, solver.nu
    x1 = np.zeros((1, nx), dtype=np.float32)
    ee1 = np.tile(goal, N).astype(np.float32).reshape(1, -1)
    XU_B = np.zeros((1, solver.N * (nx + nu) - nu), dtype=np.float32)
    XU_B[:, :nx] = x1
    solver.solve(x1, ee1, XU_B)
    stats = solver.get_stats()
    denom = float(stats.get("best_initial_merit", np.nan))
    curve = _best_curve_from_stats(stats)
    curve = curve / denom if (denom == denom and denom != 0) else curve
    return np.r_[1.0, curve]


def run_config(urdf, costs, B_list, num_targets, max_iters):
    """Average normalized merit curves per label over `num_targets` random goals."""
    labels = {"1 (ρ=1e-1)": []}
    labels.update({f"{B}": [] for B in B_list})
    for _ in range(num_targets):
        goal = sample_goal()
        raw = {}
        for B in B_list:
            try:
                raw[f"{B}"] = _curve_for_B(urdf, B, costs, goal, max_iters)
            except Exception as e:
                print(f"  skip B={B}: {e}")
        try:
            raw["1 (ρ=1e-1)"] = _curve_adaptive(urdf, costs, goal, 1e-1, max_iters)
        except Exception as e:
            print(f"  skip adaptive baseline: {e}")
        valid = [v for v in raw.values() if v is not None and len(v) > 0]
        if not valid:
            continue
        Kmin = min(len(v) for v in valid)
        denom = float(np.max(np.vstack([v[:Kmin] for v in valid])[:, 0])) or 1.0
        for label, curve in raw.items():
            if curve is not None and len(curve) > 0:
                labels[label].append((curve[:Kmin] / denom).astype(np.float32))
    out = {}
    for label, curves in labels.items():
        if curves:
            Kmin = min(len(c) for c in curves)
            out[label] = np.mean(np.vstack([c[:Kmin] for c in curves]), axis=0)
    return out


def regenerate(num_targets, max_iters, B_list, cost_grid):
    urdf, _, _ = C.resolve_model("iiwa14")
    C.require_module("iiwa14", N)
    agg = {}
    for costs in cost_grid:
        print(f"cost q={costs[0]} qd={costs[1]} u={costs[2]} N={costs[3]}")
        res = run_config(urdf, costs, B_list, num_targets, max_iters)
        for label, curve in res.items():
            agg.setdefault(label, []).append(curve)
    return agg


def aggregate_final(agg):
    final = {}
    for label, curves in agg.items():
        if not curves:
            continue
        Kmin = min(len(c) for c in curves)
        final[label] = np.mean(np.vstack([c[:Kmin] for c in curves]), axis=0)
    return final


def plot(final, out_name):
    plt = C.set_paper_rcParams()
    Kcommon = min(len(v) for v in final.values())
    # plot batch labels in numeric order, baseline first
    def _key(lbl):
        return -1 if lbl.startswith("1 ") else int(lbl)
    fig = plt.figure(figsize=(8, 5))
    for label in sorted(final, key=_key):
        plt.plot(np.arange(Kcommon), final[label][:Kcommon],
                 label=label, color=COLOR_PALETTE.get(label))
    plt.xlabel("SQP iterations")
    plt.ylabel("Relative merit")
    plt.grid(True)
    plt.legend(title="Batch Size")
    plt.tight_layout()
    C.savefig(fig, out_name)


def main():
    p = argparse.ArgumentParser(description="Regenerate Fig-4 (CS1 hyperparameter / rho).")
    C.add_repro_args(p)
    p.add_argument("--num-targets", type=int, default=50, help="random goals per cost-config (paper used 100)")
    p.add_argument("--max-iters", type=int, default=100, help="SQP iterations")
    p.add_argument("--batch-sizes", default="2,4,8,16,32,64,128")
    args = p.parse_args()
    np.random.seed(args.seed)

    if args.replot:
        agg = C.load_data("fig4_hparam_results", recovered=RECOVERED)
    else:
        num_targets, max_iters = args.num_targets, args.max_iters
        B_list = C.parse_int_list(args.batch_sizes)
        cost_grid = [(q, qd, u, n) for q in Q_LIST for qd in QD_LIST for u in U_LIST for n in N_LIST]
        if args.quick:
            num_targets, max_iters, B_list = 3, 20, [2, 32, 128]
            cost_grid = [cost_grid[0]]
            print("[quick] tiny subset — NOT paper numbers")
        agg = regenerate(num_targets, max_iters, B_list, cost_grid)
        C.save_data(agg, "fig4_hparam_results")

    plot(aggregate_final(agg), "fig4_hparam_convergence")


if __name__ == "__main__":
    main()
