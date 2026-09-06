# Releasing xtim publicly

The private repo (`origin` = `github.com/ikim-quantum/xtim-dev`) carries the
full history. The public repo (`public` = `github.com/ikim-quantum/xtim`)
carries a **fresh-history** artifact: one orphan, parentless commit per release,
pushed as its `main`, with the release tag placed on that orphan commit.

Current state: **3.1.0 is the released public version** (cut from `main`,
2026-09-05; public `main` = the 3.1.0 orphan, tag `v3.1.0`; previous: 3.0.1 =
orphan `a5e1949`). The 2.7.0 cut was
made on a side branch (`release/v2.7-public`)
because the in-flight `xtim.port` module had to be stripped from the package;
3.0.0 ships the port, so the release is cut from `main` directly and no side
branch is needed.

## Owner decisions — resolved

- [x] **License** — DONE. Apache-2.0. `LICENSE` carries the standard Apache-2.0
      text with the copyright line `Copyright 2026 Isaac Kim`; pyproject
      references it via `license-files = ["LICENSE"]` plus the
      `License :: OSI Approved :: Apache Software License` classifier (the SPDX
      `license` field stays intentionally omitted — `packaging.licenses` is
      missing in some isolated build runners, which fails the build).
- [x] **Citation** — DONE, pending preprint. `CITATION.cff` cites the software
      release (author Isaac H. Kim, title xtim, version 3.0.0, repo
      github.com/ikim-quantum/xtim). It will be updated to point at the
      accompanying preprint (`preferred-citation`) when that preprint appears.
- [x] **History** — fresh-history cut, every release. The public repo does NOT
      carry the private history: the orphan branch `public-main` holds exactly
      the release tree as a single parentless commit, verified against the
      release commit modulo the documented exclusions below. No history audit is
      therefore required.
- [ ] **CI wheels** — SKIPPED (owner decision, standing). No CI wheel builds;
      wheels are built locally. `.github/workflows/` is therefore EXCLUDED from
      the public tree (see "Documented exclusions"). Revisit if demand appears.
- [x] **README support expectations** — DONE. README has a "Support
      expectations" section (research software, best-effort support, the
      supported `v3.x` line, platform matrix, orchestration layer out of scope),
      a fast-sampling quick-start documenting the once-per-sampler selfcheck
      diagnostic on the first run, and an `xtim.port` quick-start.

## Documented exclusions (public tree ≠ release tree)

The orphan tree is the release tree **minus**:

- `.github/` — the CI wheel workflow. Excluded at 2.7.0 and again at 3.0.0:
  CI wheels are a skipped decision, and pushing workflow files requires a
  `workflow`-scoped token, which the release push is not given. Verification
  therefore asserts `git diff <tag> public-main` shows `.github/**` deletions
  and NOTHING else.

Everything else — sources, tests, docs, benchmarks, examples, packaging
metadata — ships verbatim.

## What the 3.0 cut guarantees

- Latest engine; no orchestration modules in the package.
- `xtim.port` ships (the declared-consumption segment surface promised in the
  2.7.0 notes), with the two dead parameters removed — see CHANGELOG [3.0.0]
  BREAKING.
- The selfcheck oracle window runs once per sampler lifetime (first call after
  compilation), never re-arms on `set_seed`, and is stream-neutral (oracle:
  `tests/test_selfcheck_once.py`).
- Full test suite green from source AND against the installed wheel in a clean
  environment from a neutral cwd; speed verified against the current engine
  class (see CHANGELOG [3.0.0] "Performance (release verification)").

## Cut checklist

1. Land the release content on `main`; bump `pyproject.toml` `version` (and the
   `xtim/__init__.py` dev fallback), add the CHANGELOG section, refresh
   `CITATION.cff` (version + date-released), and update the **"Current state"**
   line at the top of this file.
   The three version strings are pinned against each other by
   `tests/test_version_consistency.py`, so step 2 catches a partial bump — but
   nothing can catch a stale "Current state" line except this step. It went
   stale at 3.0.1 (the very commit that corrected the push recipe below missed
   it, which is why it is now called out here explicitly).
2. Suite green from source: `pytest tests/`.
3. Verify the downstream consumer against the edited engine before tagging
   (its oracle suites + byte gates must not move a pinned stream).
4. Commit; local tag `vX.Y.Z` on the release commit.
5. Build the wheel: `python -m build --wheel --no-isolation`.
6. Verify the wheel in a FRESH conda env from a NEUTRAL cwd: install it, run
   `pytest` against the INSTALLED package, check `xtim.__version__` and
   `import xtim, xtim.port`.
7. Build the orphan commit and push (recipe below). Before pushing, run the
   documented-exclusions check: `git diff --name-status vX.Y.Z public-main`
   must show the `.github/**` deletions and NOTHING else, and
   `git rev-list --count public-main` must be 1. This verification is stated
   under "Documented exclusions" above but appeared in no step, so nothing
   prompted anyone to run it.

## Push recipe (owner)

The public artifact is the orphan branch `public-main`: a single parentless
commit whose tree is the release tree minus the documented exclusions.

```bash
# build the orphan from the release commit (once per release)
git checkout --orphan public-main vX.Y.Z
git rm -r --cached .github            # documented exclusion
rm -rf .github
git commit -m "xtim vX.Y.Z — public release"

# publish
git remote add public git@github.com:ikim-quantum/xtim.git    # once
# public main = the orphan commit.  --force-with-lease IS REQUIRED from the
# SECOND release onward: each release is a fresh PARENTLESS commit, so it can
# never fast-forward over the previous one.  A plain push is rejected
# non-fast-forward (observed cutting 3.0.1 over 3.0.0).  Pass the CURRENT
# public main sha as the lease so the push refuses if the remote moved.
git push --force-with-lease=main:<current-public-main-sha> public public-main:main
git push public public-main:refs/tags/vX.Y.Z     # tag AT the orphan commit
```

Nothing is lost by that force: every prior release stays reachable by ITS OWN
tag on the public remote (v2.7.0 -> 4548a01, v3.0.0 -> 13444fb, ...), which is
the point of tagging the orphan directly. Read the current sha first with
`git ls-remote public main` and pass it as the lease.

Then publish the private history too:

```bash
git push origin main
git push origin vX.Y.Z          # the PRIVATE tag goes ONLY to origin
```

**CAUTION: never `git push public vX.Y.Z`.** The local tag points at the
release commit on `main`, which carries the full private history; pushing it
would drag that history to the public remote. The private branches and their
history must never reach `public`. The second push line tags the orphan commit
directly, which is the only correct form.

## Student / consumer migration note

The private repo was renamed `xtim` → `xtim-dev` when the public repo took the
`xtim` name (2.7.0). Anyone with an old clone of the private repo needs:

```bash
git remote set-url origin git@github.com:ikim-quantum/xtim-dev.git
```

GitHub redirects the old path, so an un-updated remote keeps working, but it
now silently resolves through the rename — update it.

For 3.0.0 specifically: consumers on the unpublished 2.8.0-class dev line must
drop `frame_in=` / `input_bits=` from their `Segment.run` calls (including test
doubles that mirror the old signature) — see CHANGELOG [3.0.0] BREAKING.
