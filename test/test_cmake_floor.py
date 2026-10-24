"""The CMake floor: every command CMakeLists.txt uses exists in the declared minimum.

Issue #23 (Felix Mueh, JetPack 6): the module loop used block()/endblock(), a 3.25
command, under cmake_minimum_required(VERSION 3.22), so the distribution CMake of
Ubuntu 22.04 failed at configure time. This test keeps the declared floor honest
statically; tools/check_cmake_floor.sh configures the tree with a pinned 3.22.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# command -> the CMake release that introduced it (commands newer than 3.22 that a
# tree like this one is likely to reach for; extend when a new one appears)
INTRODUCED = {
    "block": (3, 25), "endblock": (3, 25),
    "cmake_language": (3, 18), "cmake_path": (3, 20),
    "return_propagate": (3, 25),
}
NEWER_SIGNATURES = [
    # (regex on one line, version, what)
    (re.compile(r"\breturn\s*\(\s*PROPAGATE\b"), (3, 25), "return(PROPAGATE)"),
    (re.compile(r"\bFILE_SET\b"), (3, 23), "target_sources(FILE_SET)"),
    (re.compile(r"\bPATH_EQUAL\b"), (3, 24), "if(PATH_EQUAL)"),
    (re.compile(r"\bCOMPILE_ONLY\b"), (3, 27), "$<COMPILE_ONLY>"),
]


def _declared_floor(text):
    m = re.search(r"cmake_minimum_required\s*\(\s*VERSION\s+(\d+)\.(\d+)", text)
    assert m, "CMakeLists.txt declares no minimum version"
    return int(m.group(1)), int(m.group(2))


def _commands(text):
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m:
            yield m.group(1).lower(), line


# the tree's own CMake files (the baselines under examples/benchmarks/baselines carry
# third-party trees with their own floors)
TRACKED = [p.relative_to(REPO) for p in [REPO / "CMakeLists.txt", *sorted((REPO / "examples" / "cpp").rglob("CMakeLists.txt"))]
           if p.exists()]


@pytest.mark.parametrize("cmakelists", TRACKED, ids=str)
def test_commands_exist_in_the_declared_minimum(cmakelists):
    text = (REPO / cmakelists).read_text()
    floor = _declared_floor(text)
    too_new = []
    for cmd, line in _commands(text):
        since = INTRODUCED.get(cmd)
        if since and since > floor:
            too_new.append(f"{cmd}() needs CMake {since[0]}.{since[1]}: {line.strip()}")
    for rx, since, what in NEWER_SIGNATURES:
        if since > floor:
            for line in text.splitlines():
                if rx.search(line.split("#", 1)[0]):
                    too_new.append(f"{what} needs CMake {since[0]}.{since[1]}: {line.strip()}")
    assert not too_new, f"{cmakelists} declares {floor[0]}.{floor[1]} but uses:\n" + "\n".join(too_new)


def test_cmake_executable_prefers_the_interpreter_s_own(tmp_path, monkeypatch):
    """The cmake wheel installs next to the interpreter; that binary wins over PATH."""
    import os
    import sys
    from gato import _cache
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python").write_text("")
    monkeypatch.setattr(sys, "executable", str(fake_bin / "python"))
    monkeypatch.setenv("PATH", str(tmp_path / "elsewhere"))
    assert _cache.cmake_executable() == "cmake"            # nothing anywhere: the bare name
    venv_cmake = fake_bin / "cmake"
    venv_cmake.write_text("#!/bin/sh\n")
    venv_cmake.chmod(0o755)
    assert _cache.cmake_executable() == str(venv_cmake)
    assert os.access(_cache.cmake_executable(), os.X_OK)


def test_native_arch_is_guarded_by_version():
    """CMAKE_CUDA_ARCHITECTURES=native is a 3.24 feature behind an explicit version check."""
    text = (REPO / "CMakeLists.txt").read_text()
    assert 'VERSION_GREATER_EQUAL "3.24"' in text
    guard = text.index('VERSION_GREATER_EQUAL "3.24"')
    native = text.index("set(CMAKE_CUDA_ARCHITECTURES native)")
    assert guard < native
