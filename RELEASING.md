# Releasing xtim publicly

The public repo (`public` = `github.com/ikim-quantum/xtim`) carries a
**fresh-history** artifact: one orphan, parentless commit per release, pushed
as its `main`, with the release tag placed on that orphan commit.

Current state: **4.2.1** (cut 2026-11-26; once pushed, public `main` = the 4.2.1
orphan, tag `v4.2.1`). Previous releases, each reachable by
its own tag on the public remote: 4.2.0 = orphan `07ac0bd`, 4.1.2 = orphan `b6dc030`, 4.1.1 = `ef0525a`,
4.1.0 = `3fcc56b`, 4.0.1 = `39e070c`, 4.0.0 = `83ad531`, 3.2.2 = `edf0dd0`,
3.2.1 = `af1ad3b`, 3.2.0 = `afddb35`, 3.1.13 = `4b4d048`, 3.1.12 = `a6d86ec`,
3.1.11 = `8bbbe0a`, 3.1.10 = `e4f2265`, 3.1.9 = `b5f4f1d`, 3.1.8 = `171d805`,
3.1.7 = `b7102d7`, 3.1.6 = `de74b9c`, 3.1.5 = `98293ad`, 3.1.4 = `2945407`,
3.1.3 = `89c550f`, 3.1.2 = `d889447`, 3.0.1 = `a5e1949` (tags `vX.Y.Z`).

## Decisions

- **License** — MIT (changed from Apache-2.0 at 4.1.2; releases up to 4.1.1
  were Apache-2.0). `LICENSE` carries the MIT text; vendored third-party
  circuits keep their licenses (`THIRD_PARTY_NOTICES.md`, Apache text in
  `LICENSES/Apache-2.0.txt`); pyproject references them via `license-files`
  plus the `License :: OSI Approved :: MIT License` classifier (the SPDX
  `license` field stays intentionally omitted — `packaging.licenses` is
  missing in some isolated build runners, which fails the build).
- **Citation** — `CITATION.cff` cites the software release (version and
  date-released are bumped at every release).
- **History** — fresh-history cut, every release: the orphan branch
  `public-main-X.Y.Z` holds exactly the release tree as a single parentless commit,
  verified against the release commit modulo the documented exclusions below.
- **CI wheels / PyPI** — from 4.2.1 the public tree carries
  `.github/workflows/wheels.yml`: cibuildwheel wheels (Linux x86_64
  manylinux_2_28, macOS arm64 11.0+; CPython 3.11-3.13) plus the sdist, published
  to PyPI by trusted publishing (see "PyPI" below). Up to 4.2.0 `.github/` was
  excluded and no wheels were published.

## Documented exclusions (public tree ≠ release tree)

The orphan tree is the release tree **minus**:

- `benchmarks/ledger/` and `benchmarks/fastpath_replay/` — the internal
  performance ledger: machine-local store paths, internal sign-off records and
  workload names for fixtures that are not distributed (the git-ignored
  local fixture directory under `tests/` is never in the tree). The rest of
  `benchmarks/` ships (tests read `benchmarks/*.stim`).
- `docs/superpowers/` and `docs/repin/` (from 4.2.0) — internal design plans
  and byte re-pin sign-off manifests.
- `docs/dem_export_readout_noise.md` (from 4.2.0) — an internal incident note.
- `tools/fx_leaves_b9c2e92.json` and `tools/check_docs_postF.sh` (from 4.2.0)
  — an unreferenced generated dump and an internal docs gate.

Everything else — sources, tests, docs, benchmarks, examples, packaging
metadata, and (from 4.2.1) `.github/` — ships verbatim. `MANIFEST.in` excludes the
same paths from the sdist, and `.github/` too (`prune .github`: CI files are not
build inputs).

## Cut checklist

1. Land the release content on the release branch; bump `pyproject.toml`
   `version` (and the `xtim/__init__.py` dev fallback), add the CHANGELOG
   section, refresh `CITATION.cff` (version + date-released), and update the
   **"Current state"** line at the top of this file.
   The three version strings are pinned against each other by
   `tests/test_version_consistency.py`, so step 2 catches a partial bump — but
   nothing can catch a stale "Current state" line except this step.
2. Suite green from source: `pytest tests/`.
3. Verify downstream consumers against the edited engine before tagging (their
   byte gates must not move a pinned stream).
4. Commit; local tag `vX.Y.Z` on the release commit.
5. Build the wheel: `python -m build --wheel --no-isolation`.
6. Verify the wheel in a FRESH environment from a NEUTRAL cwd: install it, run
   `pytest` against the INSTALLED package, check `xtim.__version__` and
   `import xtim, xtim.port`; the wheel must contain no `_xtim_testapi` (its tests
   SKIP with the reason in tests/conftest.py). **This `pytest` run is a GATE: it
   must exit 0 (every test PASSED or SKIPPED, zero FAILED/ERRORed) before step
   7.** A fresh env has no optional dev extras (scipy, stim, pymatching,
   matplotlib, ...) pre-installed unless this checklist installed them, so
   failures here usually mean a test imports an optional third-party package at
   module level instead of guarding it with `pytest.importorskip` — fix the
   test, do not wave the run through.
7. Build the orphan commit and push (recipe below). Before pushing, run the
   documented-exclusions check: `git diff --name-status vX.Y.Z public-main-X.Y.Z`
   must show deletions under the paths listed in "Documented exclusions" and
   NOTHING else (from 4.2.1 nothing under `.github/`: the workflow ships), and
   `git rev-list --count public-main-X.Y.Z` must be 1.
8. Publish to PyPI (section "PyPI" below).

## Push recipe

The public artifact is the orphan branch `public-main-X.Y.Z`: a single parentless
commit whose tree is the release tree minus the documented exclusions.

```bash
# build the orphan from the release commit (once per release)
git checkout --orphan public-main-X.Y.Z vX.Y.Z
git rm -r -q --cached benchmarks/ledger benchmarks/fastpath_replay docs/superpowers docs/repin docs/dem_export_readout_noise.md tools/fx_leaves_b9c2e92.json tools/check_docs_postF.sh   # documented exclusions
rm -rf benchmarks/ledger benchmarks/fastpath_replay docs/superpowers docs/repin docs/dem_export_readout_noise.md tools/fx_leaves_b9c2e92.json tools/check_docs_postF.sh
git commit -m "xtim vX.Y.Z — public release"

# publish
git remote add public git@github.com:ikim-quantum/xtim.git    # once
# public main = the orphan commit.  --force-with-lease IS REQUIRED: each
# release is a fresh PARENTLESS commit, so it can never fast-forward over the
# previous one.  Pass the CURRENT public main sha as the lease so the push
# refuses if the remote moved.
git push --force-with-lease=main:<current-public-main-sha> public public-main-X.Y.Z:main
# the tag: from 4.2.1 NOT pushed by hand. Publishing the GitHub release vX.Y.Z
# on main creates tag vX.Y.Z at main = the orphan commit (section "PyPI", step 3).
```

Nothing is lost by that force: every prior release stays reachable by ITS OWN
tag on the public remote, which is the point of tagging the orphan directly
(up to 4.2.0 by `git push public public-main-X.Y.Z:refs/tags/vX.Y.Z`; from
4.2.1 by publishing the GitHub release).
Read the current sha first with `git ls-remote public main` and pass it as the
lease.

**CAUTION: never `git push public vX.Y.Z`.** The local tag points at the
release commit, which carries the full development history; pushing it would
drag that history to the public remote. The public tag is created only by
publishing the GitHub release on main (the orphan commit).

Pushing an orphan that contains workflow files (`.github/workflows/`, from
4.2.1) needs a token with the `workflow` scope; a token without it is refused
by GitHub.

## PyPI

From 4.2.1, `.github/workflows/wheels.yml` builds the wheels and the sdist and
publishes them by trusted publishing (OIDC; no secrets in the repo; PyPI
attestations on by default).

One-time setup, before the first run (`xtim` does not exist yet on either index):
- a **pending** trusted publisher on PyPI AND one on TestPyPI (separate accounts),
  each naming owner `ikim-quantum`, repository `xtim`, workflow `wheels.yml` and
  its environment (`pypi` on PyPI, `testpypi` on TestPyPI);
- the two GitHub environments `testpypi` and `pypi` in the repository settings,
  `pypi` with a required reviewer.

1. Push the orphan as public `main` (push recipe above; no tag push).
2. Run the workflow by hand (Actions -> wheels -> Run workflow, on `main`): it
   builds everything and publishes to **TestPyPI** (job `publish-testpypi`).
   Verify a fresh install from a neutral cwd in a new venv:
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ xtim==X.Y.Z`,
   then `import xtim; xtim.__version__` and the README quick start.
3. Publish the GitHub release `vX.Y.Z` on `main` (= the orphan commit); this
   creates tag `vX.Y.Z` there (the only way the tag is made). The
   `release: published` event runs the workflow again and REBUILDS the wheels
   and sdist: PyPI gets a fresh build of the same tree, not the files tested on
   TestPyPI. Its `publish-pypi` job checks that every dist file carries version
   X.Y.Z (the tag minus `v`), waits for the `pypi` environment approval, then
   uploads to **PyPI**.
