"""CPU-only packaging hook: stage GATO's native sources into the wheel.

A non-editable build copies the CMake tree solver modules compile from
(CMakeLists.txt, gato/ headers incl. the vendored robots, the bindings TU and
tools/native_identity.py) to gato/_native inside the package, so
gato.register_robot can compile on the user's machine. Editable installs keep
using the checkout; nothing is ever compiled here.
"""
from pathlib import Path
import shutil

from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
NATIVE_SUFFIXES = (".cuh", ".h", ".hpp", ".cu")


def native_files():
    files = {p.relative_to(ROOT).as_posix(): p for p in ("CMakeLists.txt", "python/bindings.cu", "tools/native_identity.py")
             for p in [ROOT / p]}
    for p in (ROOT / "gato").rglob("*"):
        if p.is_file() and p.suffix in NATIVE_SUFFIXES:
            files[p.relative_to(ROOT).as_posix()] = p
    # the vendored robots' URDFs at their registry paths (gato.robot_urdf); meshes
    # are not needed — their headers are pre-generated
    import json
    for meta in json.loads((ROOT / "python/gato/_registry.json").read_text()).values():
        rel = meta["urdf"]
        if (ROOT / rel).is_file():
            files[rel] = ROOT / rel
    return files


class BuildPy(build_py):
    def run(self):
        super().run()
        if self.editable_mode:
            return
        dest = Path(self.build_lib) / "gato" / "_native"
        shutil.rmtree(dest, ignore_errors=True)
        for rel, src in native_files().items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
