# The documentation tier

The public site has two tiers. The project page at <https://a2r-lab.org/GATO/> (the
`gh-pages` branch: `index.html`, `static/`, the paper's figures and the dated current-code
results) presents the work. This directory builds the reference tier served under
<https://a2r-lab.org/GATO/docs/>: the repository's own markdown (README, `docs/*.md`, the
examples, paper-figures and dynamics READMEs, the CHANGELOG), a receipt page generated from
`gpu-proof.json`, and the API reference rendered from the docstrings.

- `build.py` assembles the pages into `_src/` with every link rewritten (page to page, or to
  the file on GitHub; a dangling link fails the build), writes `receipt.md`, and runs Sphinx
  with warnings as errors into `_build/html`. `python docs/site/build.py` builds locally with
  the `[docs]` extra installed; `test/test_docs_site.py` runs the same build in the host tier.
- `conf.py`, `index.md` and `api.md` are the site's own sources. Everything else is edited in
  place in the repository, never here.
- `.github/workflows/docs.yml` builds on pull requests and, on `main`, deploys the assembled
  site (landing from `gh-pages`, this tier under `/docs/`). `gh-pages-workflow.yml` is the
  twin for the `gh-pages` branch, so a project-page edit republishes the same assembly;
  the landing's navigation links to `docs/`.

Both workflows deploy through the `github-pages` environment; publishing is a push to `main`
or `gh-pages` by a maintainer.
