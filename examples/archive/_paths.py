"""Import helpers for the archived scripts (this directory is not a package and
the live helpers it reaches — examples/benchmarks/_bench.py,
examples/paper-figures/_common.py, _pickplace_runner.py — are not either).
Loads them by file path; nothing edits the import path."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
_DIRS = {"_bench": "benchmarks", "_common": "paper-figures",
         "_pickplace_runner": "paper-figures"}


def load(name):
    """examples/<dir>/<name>.py as a module (idempotent via sys.modules)."""
    if name in sys.modules:
        return sys.modules[name]
    if name == "_pickplace_runner":
        load("_common")   # it imports _common by name
    path = os.path.join(EXAMPLES, _DIRS[name], f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
