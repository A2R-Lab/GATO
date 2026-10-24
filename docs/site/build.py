#!/usr/bin/env python3
"""Assemble and build the GATO documentation tier.

The site's pages are the repository's own markdown, copied into ``_src/`` with
their links rewritten (page-to-page links become site links, links to other
repository files become GitHub links, every target is checked), plus a receipt
page generated from ``gpu-proof.json``. ``index.md``, ``api.md`` and ``conf.py``
live here and are copied as they are. Then Sphinx builds ``_build/html``.

    python docs/site/build.py            # assemble + build (warnings are errors)
    python docs/site/build.py --assemble # assemble only (what the host-tier test checks)

No GPU, no CUDA build; the python package is imported numpy-only for the API
reference (heavy dependencies are mocked in conf.py).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = HERE / "_src"
OUT = HERE / "_build" / "html"
GITHUB = "https://github.com/A2R-Lab/GATO/blob/main/"
GITHUB_TREE = "https://github.com/A2R-Lab/GATO/tree/main/"

# repository markdown -> site page (the page's title is the file's own heading)
PAGES = {
    "README.md": "readme",
    "docs/README.md": "guide",
    "docs/status.md": "status",
    "docs/consumer_contract.md": "consumer_contract",
    "docs/constraints.md": "constraints",
    "docs/cost_menu.md": "cost_menu",
    "docs/merit_timing.md": "merit_timing",
    "docs/development.md": "development",
    "docs/figure-refresh-2026-10-01.md": "figure_refresh",
    "docs/archaeology.md": "archaeology",
    "docs/website.md": "website",
    "examples/README.md": "examples",
    "examples/paper-figures/README.md": "paper_figures",
    "gato/dynamics/README.md": "dynamics",
    "CHANGELOG.md": "changelog",
}
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)(\s+\"[^\"]*\")?\)")


def rewrite_links(text, source_rel, assets=None):
    """Rewrite the markdown links of one source file; raise on a dangling target."""
    base = (REPO / source_rel).parent

    def repl(m):
        bang, label, target, title = m.groups()
        title = title or ""
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            return m.group(0)                                   # URL, mailto, or in-page anchor
        path, _, anchor = target.partition("#")
        resolved = (base / path).resolve()
        try:
            rel = resolved.relative_to(REPO).as_posix()
        except ValueError:
            raise SystemExit(f"{source_rel}: link {target!r} leaves the repository")
        if rel in PAGES:
            return f"{bang}[{label}]({PAGES[rel]}.md{'#' + anchor if anchor else ''}{title})"
        if not resolved.exists():
            raise SystemExit(f"{source_rel}: link {target!r} -> {rel} does not exist")
        if bang:
            if not resolved.is_file():
                raise SystemExit(f"{source_rel}: image {target!r} is not a file")
            if assets is not None:
                assets[rel] = resolved
            return f"![{label}](_assets/{rel}{'#' + anchor if anchor else ''}{title})"
        url = (GITHUB_TREE if resolved.is_dir() else GITHUB) + rel + ("#" + anchor if anchor else "")
        return f"{bang}[{label}]({url}{title})"

    return LINK.sub(repl, text)


def committed_receipt():
    """The receipt the site documents is the COMMITTED one: ``git show HEAD:gpu-proof.json``
    in a checkout (a receipt run removes the working-tree file while it records, and an
    uncommitted receipt attests nothing); the file itself outside git (an sdist)."""
    try:
        out = subprocess.run(["git", "show", "HEAD:gpu-proof.json"], cwd=REPO, capture_output=True, text=True)
        if out.returncode == 0:
            return json.loads(out.stdout)
    except OSError:
        pass
    return json.loads((REPO / "gpu-proof.json").read_text())


def receipt_page():
    """The receipt page: what the committed gpu-proof.json says, in words, and how to check it."""
    r = committed_receipt()
    env, repo, sess, signer, fp = r["environment"], r["repo"], r["session"], r["signer"], r["fingerprint"]
    tests = r["tests"]
    by = {}
    for t in tests:
        by[t["outcome"]] = by.get(t["outcome"], 0) + 1
    modules = [line.split() for line in (REPO / "test/receipt_modules.txt").read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    gpu = env.get("gpu_info", {})
    gpu_name = gpu.get("name") or gpu.get("gpu_name") or ", ".join(f"{k}: {v}" for k, v in gpu.items())
    rows = "\n".join(f"| `{p}` | {n} | {v[0] if v else 'default'} |" for p, n, *v in modules)
    files = ", ".join(f"`{p}`" for p in fp.get("included_paths", []))
    outcome = sess["outcome"]
    lines = [
        "# Signed receipt",
        "",
        "GPU correctness is attested by a signed [pytest-gpu-proof](https://github.com/A2R-Lab/pytest-gpu-proof)",
        "receipt committed at the repository root (`gpu-proof.json`) and verified without a GPU by the",
        "`verify-gpu-proof` workflow on every push. This page is generated from the committed receipt at",
        "documentation build time. [What the receipt does and does not attest](consumer_contract.md#5-what-the-gpu-proof-receipt-does-and-does-not-attest)",
        "and [how one is produced](development.md#signed-receipt) are in the reference pages.",
        "",
        "## Current receipt",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Outcome | **{outcome}**: {by.get('passed', 0)} passed, {by.get('failed', 0)} failed, {by.get('skipped', 0)} skipped ({len(tests)} recorded) |",
        f"| Commit | `{repo['commit_sha']}` on `{repo['branch']}` (tree {'dirty' if repo['dirty'] else 'clean'}) |",
        f"| Run | {sess['started_at']} to {sess['ended_at']} (UTC) |",
        f"| GPU | {gpu_name} |",
        f"| Platform | {env['platform']}, Python {env['python_version']}, pytest {env['pytest_version']}, plugin {env['plugin_version']}, schema {r['schema_version']} |",
        f"| Signer | GitHub user `{signer['github_user']}`, {signer['algorithm']} key `{signer['key_fingerprint']}` |",
        f"| Source fingerprint | {fp['algorithm']}, {fp['file_count']} files over {files}; digest `{fp['digest'][:16]}…` |",
        "",
        "The verifier recomputes the source fingerprint from the checkout, checks the commit's ancestry, the",
        "signature against the signer's public GitHub keys, the receipt's age and the policy in",
        "[`test/gpu-proof-policy.yaml`](https://github.com/A2R-Lab/GATO/blob/main/test/gpu-proof-policy.yaml);",
        "the collected suite must equal [`test/expected_tests.txt`](https://github.com/A2R-Lab/GATO/blob/main/test/expected_tests.txt)",
        "and no test may skip.",
        "",
        "## Module set the receipt attests",
        "",
        "The suite runs on exactly these solver modules",
        "([`test/receipt_modules.txt`](https://github.com/A2R-Lab/GATO/blob/main/test/receipt_modules.txt)); a missing one",
        "refuses the run. Variants: `fc` = contact-force controls, `eh` = exact Hessian.",
        "",
        "| Robot | Horizon N | Variant |",
        "|---|---|---|",
        rows,
        "",
        "## Verify it yourself",
        "",
        "```bash",
        "pip install pytest-gpu-proof==" + env["plugin_version"] + " pyyaml",
        "gpu-proof verify --receipt gpu-proof.json --repo . \\",
        "    --policy test/gpu-proof-policy.yaml --expected-skips test/expected_skips.txt --require-gpu",
        "```",
        "",
        "A receipt attests the recorded tests at the fingerprinted sources on the lab GPU. It does not",
        "certify runtime speed, other robots or settings, or safety on physical hardware.",
    ]
    return "\n".join(lines) + "\n"


def assemble():
    if SRC.exists():
        shutil.rmtree(SRC)
    SRC.mkdir()
    assets = {}
    for rel, page in PAGES.items():
        text = (REPO / rel).read_text()
        (SRC / f"{page}.md").write_text(rewrite_links(text, rel, assets))
    for rel, source in assets.items():
        target = SRC / "_assets" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (SRC / "receipt.md").write_text(receipt_page())
    for name in ("conf.py", "index.md", "api.md"):
        shutil.copyfile(HERE / name, SRC / name)
    return SRC


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assemble", action="store_true", help="assemble _src only, do not run Sphinx")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    src = assemble()
    print(f"assembled {len(list(src.glob('*.md')))} pages in {src}")
    if args.assemble:
        return
    cmd = [sys.executable, "-m", "sphinx", "-b", "html", "-W", "--keep-going", "-q", str(src), args.out]
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
