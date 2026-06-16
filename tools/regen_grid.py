#!/usr/bin/env python3
"""Regenerate the vendored GRiD CUDA headers for GATO's robots.

Writes:
    gato/dynamics/iiwa14/grid.cuh
    gato/dynamics/indy7/grid.cuh

Drives the GRiD code generator vendored at external/GRiD (no robot_descriptions
package needed — the local GATO URDFs are the source of truth). Run from the
GATO repo root:

    python tools/regen_grid.py              # both robots
    python tools/regen_grid.py --robot indy7
    python tools/regen_grid.py --list       # use an explicit algorithm list

Requires the GRiD submodule initialized:
    git submodule update --init --recursive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_ROOT = REPO_ROOT / "external" / "GRiD"

# Make the GRiD top-level packages (URDFParser, GRiDCodeGenerator) importable.
sys.path.insert(0, str(GRID_ROOT))

# robot id -> (URDF path, fixed EE joint name, output header)
ROBOTS = {
    "iiwa14": dict(
        urdf=REPO_ROOT / "examples" / "iiwa_description" / "iiwa14.urdf",
        fixed_target_name="EE",
        out=REPO_ROOT / "gato" / "dynamics" / "iiwa14" / "grid.cuh",
    ),
    "indy7": dict(
        urdf=REPO_ROOT / "examples" / "indy7_description" / "indy7.urdf",
        fixed_target_name="EE",
        out=REPO_ROOT / "gato" / "dynamics" / "indy7" / "grid.cuh",
    ),
}

# Explicit list covering everything GATO's plant.cuh needs + integrators + EE
# pose/gradient. profile="all" is the simpler, recommended default (it also
# emits grid_plant incl. plant_step_hessian); this list documents intent and is
# used with --list. Add "fdsva_so" if plant_step_hessian is wanted with --list.
ALGORITHM_LIST = [
    "forward_dynamics", "inverse_dynamics",
    "forward_dynamics_gradient", "inverse_dynamics_gradient",
    "minv",
    "end_effector_pose", "end_effector_pose_gradient",
    "integrator", "integrator_gradient", "integrator_with_gradient",
]


def regen(robot_id: str, *, use_all_profile: bool) -> None:
    from URDFParser import URDFParser
    from GRiDCodeGenerator import GRiDCodeGenerator

    spec = ROBOTS[robot_id]
    urdf = spec["urdf"]
    out = spec["out"]
    out.parent.mkdir(parents=True, exist_ok=True)

    if not urdf.exists():
        sys.exit(f"[{robot_id}] URDF not found: {urdf}")

    print(f"[{robot_id}] parsing {urdf}")
    robot = URDFParser().parse(str(urdf), floating_base=False)       # fixed base
    print(f"[{robot_id}] EE target joint = '{spec['fixed_target_name']}'")

    codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True,
                                FILE_NAMESPACE="grid")
    kwargs = dict(
        include_homogenous_transforms=True,            # required for EE pose + gradient
        fixed_target_name=spec["fixed_target_name"],
        output_path=str(out),
    )
    if use_all_profile:
        kwargs["codegen_profile"] = "all"
    else:
        kwargs["algorithm_list"] = ALGORITHM_LIST
    codegen.gen_all_code(**kwargs)
    print(f"[{robot_id}] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate GATO's vendored GRiD headers.")
    ap.add_argument("--robot", choices=list(ROBOTS) + ["all"], default="all")
    ap.add_argument("--list", dest="use_list", action="store_true",
                    help="Use the explicit algorithm list instead of profile='all'.")
    args = ap.parse_args()

    if not GRID_ROOT.exists():
        sys.exit(f"GRiD not found at {GRID_ROOT}. Run: "
                 f"git submodule update --init --recursive")

    targets = list(ROBOTS) if args.robot == "all" else [args.robot]
    for rid in targets:
        regen(rid, use_all_profile=not args.use_list)
    print("Done. NOTE: indy7_plant.cuh must use grid::end_effector_pose_device "
          "(not end_effector_positions_device) to match the regenerated header.")


if __name__ == "__main__":
    main()
