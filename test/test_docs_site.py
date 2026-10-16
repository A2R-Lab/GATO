"""The documentation tier (docs/site): the assembler rewrites every link of the
repository's markdown to a site page or an existing GitHub path (a dangling
link is an error), generates the receipt page from gpu-proof.json, and Sphinx
builds the result with warnings as errors. Host tier: no GPU, no CUDA build;
the docs stack is a [test] dependency so this never skips."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "docs" / "site" / "build.py"
pytestmark = pytest.mark.slow


def _load():
    spec = importlib.util.spec_from_file_location("gato_docs_build", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_links_are_rewritten_and_checked(tmp_path):
    mod = _load()
    text = ("[contract](consumer_contract.md#4-solve-interface) [ex](../examples/README.md) "
            "[file](../test/receipt_modules.txt) [dir](../examples/cpp/) [web](https://x.y/z) [here](#anchor)")
    out = mod.rewrite_links(text, "docs/status.md")
    assert "(consumer_contract.md#4-solve-interface)" in out and "(examples.md)" in out
    assert "(https://github.com/A2R-Lab/GATO/blob/main/test/receipt_modules.txt)" in out
    assert "(https://github.com/A2R-Lab/GATO/tree/main/examples/cpp)" in out
    assert "(https://x.y/z)" in out and "(#anchor)" in out
    with pytest.raises(SystemExit, match="does not exist"):
        mod.rewrite_links("[x](../nope/missing.md)", "docs/status.md")
    for rel in mod.PAGES:
        assert (REPO / rel).is_file(), rel


def test_receipt_page_reflects_the_committed_receipt():
    import json
    mod = _load()
    page = mod.receipt_page()
    r = json.loads((REPO / "gpu-proof.json").read_text())
    assert r["repo"]["commit_sha"] in page and r["signer"]["github_user"] in page
    assert f"{len(r['tests'])} recorded" in page
    for line in (REPO / "test/receipt_modules.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            assert f"`{line.split()[0]}` | {line.split()[1]}" in page


def test_site_builds_without_warnings(tmp_path):
    out = tmp_path / "html"
    r = subprocess.run([sys.executable, str(BUILD), "--out", str(out)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    for page in ("index", "readme", "examples", "consumer_contract", "status", "constraints", "receipt", "api",
                 "development", "changelog"):
        assert (out / f"{page}.html").is_file(), page
    api = (out / "api.html").read_text()
    for name in ("BSQP", "MPCController", "MPCPolicy", "SolverParams", "register_robot", "canonicalize_inertials"):
        assert name in api, name
