"""Sphinx configuration for the GATO documentation tier (https://a2r-lab.org/GATO/docs/).

The pages are assembled by build.py from the repository's own markdown (README,
docs/*.md, the examples and dynamics READMEs, the CHANGELOG) and a receipt page
generated from gpu-proof.json; the API reference is autodoc over the python
package. No CUDA build runs here: heavy dependencies are mocked.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import gato  # noqa: E402  (numpy-only import)

project = "GATO"
copyright = "2026, A²R Lab"
author = "A²R Lab"
release = gato.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "gh-pages-workflow.yml"]

myst_enable_extensions = ["colon_fence", "fieldlist", "attrs_inline"]
myst_heading_anchors = 4
suppress_warnings = ["myst.header"]      # the assembled pages keep their source heading levels

autodoc_mock_imports = ["pinocchio", "mujoco", "gymnasium", "torch", "jax", "jaxlib", "scipy"]
autodoc_default_options = {"members": True, "member-order": "bysource", "undoc-members": False,
                           "show-inheritance": True}
autodoc_typehints = "description"
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "pydata_sphinx_theme"
html_title = f"GATO {release}"
html_static_path = []
html_show_sourcelink = False
html_theme_options = {
    "github_url": "https://github.com/A2R-Lab/GATO",
    "external_links": [
        {"name": "Project page", "url": "https://a2r-lab.org/GATO/"},
        {"name": "PyPI", "url": "https://pypi.org/project/gato-mpc/"},
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": [],
}
html_context = {"default_mode": "light"}
