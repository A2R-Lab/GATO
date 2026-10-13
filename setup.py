"""Build shim: all metadata lives in pyproject.toml; this only registers the
build_py hook that stages GATO's native sources into non-editable wheels."""
from setuptools import setup
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _build_resources import BuildPy

setup(cmdclass={"build_py": BuildPy})
