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

from gato.build import codegen  # noqa: E402

# robot id -> (URDF path, EE target fixed-joint name)
ROBOTS = {
    "iiwa14": dict(urdf=REPO_ROOT / "examples" / "iiwa_description" / "iiwa14.urdf",
                   ee_frame="EE"),
    "indy7": dict(urdf=REPO_ROOT / "examples" / "indy7_description" / "indy7.urdf",
                  ee_frame="EE"),
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
                       algorithm_list=ALGORITHM_LIST if args.use_list else None)
        print(f"[{rid}] wrote gato/dynamics/{rid}/{{grid.cuh, limits.cuh}}  meta={meta}")
    print("Done.")


if __name__ == "__main__":
    main()
