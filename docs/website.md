# GATO website

The site has two tiers. The project page (<https://a2r-lab.org/GATO/>) lives on the `gh-pages`
branch: `index.html`, `static/`, the paper's figures and the dated current-code results. It
stays off `main` because its media (a 19 MB video) would enlarge every code clone. The
documentation tier (<https://a2r-lab.org/GATO/docs/>) is built from `main` by
`docs/site/build.py` and deployed by `.github/workflows/docs.yml`; see
[docs/site/README.md](site/README.md) and [development.md](development.md#documentation-site).
Either branch's workflow republishes the whole assembly (landing plus `/docs/`).

## Project page

Every push to `gh-pages` republishes the site.

To update it:

```bash
git worktree add /tmp/gato-site origin/gh-pages -b site-edit
# edit /tmp/gato-site/index.html or static/; preview with
python3 -m http.server 8766 --directory /tmp/gato-site
cd /tmp/gato-site && git commit -am "Describe the change" && git push origin site-edit:gh-pages
git worktree remove /tmp/gato-site
```

The October 1, 2026 update labeled the paper's figures as published results, added a
"Results from the Latest Release" section (`static/demos/current/`, see
[figure-refresh-2026-10-01.md](figure-refresh-2026-10-01.md)) and pointed the quick start at `main`.
