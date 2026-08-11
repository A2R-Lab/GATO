#!/usr/bin/env python3
"""Regenerate the vendored GRiD headers for GATO's robots.

Thin CLI over gato.build.codegen (the single codegen path — gato.build() uses
the same code). Writes gato/dynamics/<robot>/{grid.cuh, limits.cuh} and updates
python/gato/_registry.json. Run from the GATO repo root:

    python tools/regen_grid.py              # both vendored robots
    python tools/regen_grid.py --robot indy7
    python tools/regen_grid.py --list       # use the explicit algorithm list

Requires the GRiD submodule initialized:
    git submodule update --init --recursive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from gato.builder import codegen  # noqa: E402

# robot id -> (URDF path, EE target fixed-joint name, collision/contact bake).
# collision_res 0.15 keeps the sphere set row-group sized (iiwa14 44 / indy7 29
# spheres, rmax ~0.13 m inflation — conservative covering spheres, CL-2);
# contact_frames bakes the EE wrench map so CL-3 starts regen-free.
ROBOTS = {
    "iiwa14": dict(urdf=REPO_ROOT / "examples" / "iiwa_description" / "iiwa14.urdf",
                   ee_frame="EE", collision_res=0.15, contact_frames=["EE"]),
    "indy7": dict(urdf=REPO_ROOT / "examples" / "indy7_description" / "indy7.urdf",
                  ee_frame="EE", collision_res=0.15, contact_frames=["EE"]),
    # floating base (CL-3): EE = the trunk imu (base-position cost target),
    # contact frames = the four feet (fc-on-feet wave); codegen detects the
    # free-flyer from the URDF root and emits gato_abi.cuh alongside.
    "go2": dict(urdf=REPO_ROOT / "external" / "GRiD" / "config" / "robot_assets" / "go2.urdf",
                ee_frame="imu_joint", collision_res=0.15,
                contact_frames=["FR_foot_joint", "FL_foot_joint",
                                "RR_foot_joint", "RL_foot_joint"]),
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate GATO's vendored GRiD headers.")
    ap.add_argument("--robot", choices=list(ROBOTS) + ["all"], default="all")
    ap.add_argument("--list", dest="use_list", action="store_true",
                    help="Use the explicit algorithm list instead of profile='all'.")
    args = ap.parse_args()

    targets = list(ROBOTS) if args.robot == "all" else [args.robot]
    for rid in targets:
        spec = ROBOTS[rid]
        print(f"[{rid}] parsing {spec['urdf']} (EE target joint = {spec['ee_frame']!r})")
        meta = codegen(spec["urdf"], rid, ee_frame=spec["ee_frame"],
                       algorithm_list=ALGORITHM_LIST if args.use_list else None,
                       collision_res=spec.get("collision_res"),
                       contact_frames=spec.get("contact_frames"))
        print(f"[{rid}] wrote gato/dynamics/{rid}/{{grid.cuh, limits.cuh}}  meta={meta}")
    print("Done.")


if __name__ == "__main__":
    main()
