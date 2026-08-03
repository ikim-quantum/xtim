# Releasing xtim publicly — owner decisions (resolved 2026-08)

`release/v2.7-public` is the finalized public cut. All previously pending
owner decisions are resolved below. Nothing is pushed from this machine by
automation; the owner pushes (recipe at the bottom).

## Owner decisions — resolved

- [x] **License** — DONE. Apache-2.0 confirmed. `LICENSE` carries the standard
      Apache-2.0 text with the copyright line `Copyright 2026 Isaac Kim`;
      pyproject references it via `license-files = ["LICENSE"]` plus the
      `License :: OSI Approved :: Apache Software License` classifier (the
      SPDX `license` field stays intentionally omitted — `packaging.licenses`
      is missing in some isolated build runners).
- [x] **Citation** — DONE, pending preprint. `CITATION.cff` cites the software
      release (author Isaac H. Kim, title xtim, version 2.7.0, repo
      github.com/ikim-quantum/xtim). It will be updated to point at the
      accompanying preprint (`preferred-citation`) when that preprint appears.
- [x] **History** — fresh-history cut. The public repo does NOT carry the
      private history: the orphan branch `public-main` contains exactly the
      release tree as a single commit ("xtim v2.7.0 — initial public
      release"), verified tree-identical to `release/v2.7-public` and
      parentless. No history audit is therefore required.
- [ ] **CI wheels** — SKIPPED (owner decision). No CI wheel builds for the
      first public tag; wheels are built locally. Revisit post-release if
      demand appears.
- [x] **README support expectations** — DONE. README has a "Support
      expectations" section (research software, best-effort support, v2.x
      maintained until v3.0, platform matrix, orchestration layer out of
      scope) and a fast-usage quick-start documenting the once-per-sampler
      selfcheck diagnostic on the first run.

## What the v2.7 cut guarantees

- Latest engine, no orchestration modules in the package.
- The in-flight v3 `xtim.port` surface is removed (ships at v3.0); the
  compact-record C++ internals remain (default-off, inert).
- The selfcheck oracle window runs once per sampler lifetime (first call
  after compilation), never re-arms on `set_seed`, and is stream-neutral
  (oracle: `tests/test_selfcheck_once.py`).
- Full test suite green from source and against the installed wheel in a
  clean environment; C++ seed-parity test green; speed verified against the
  current engine class (see CHANGELOG [2.7.0]).

## Push recipe (owner)

The public artifact is the orphan branch `public-main` (single parentless
commit, tree-identical to `release/v2.7-public` at tag `v2.7.0`). To publish
it as the public repo's `main` with the release tag:

```bash
# from this clone
git remote add public git@github.com:ikim-quantum/xtim.git   # once
git push public public-main:main             # the public repo's main = the orphan commit
git push public public-main:refs/tags/v2.7.0 # tag v2.7.0 there, AT the orphan commit
```

CAUTION: do NOT `git push public v2.7.0` — the local `v2.7.0` tag points at
the private `release/v2.7-public` head and pushing it would drag the private
history to the public remote. The private branch and its history must never
be pushed there; the second line above tags the orphan commit directly.
