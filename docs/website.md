# GATO project page

The page lives on the `gh-pages` branch and is served at http://a2r-lab.org/GATO/. GitHub Pages
deploys that branch automatically: every push to `gh-pages` republishes the site. It stays off
`main` because its media (a 19 MB video) would enlarge every code clone.

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
