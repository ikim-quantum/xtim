# xtim documentation map

Pick your entry point by what you're trying to do. (Everything here assumes the
one-paragraph pitch from the top-level README: xtim is a Stim-shaped simulator that
exactly computes the magic-state expectation values Stim cannot.)

## I'm new — show me the tool
- **[`../examples/xtim_tutorial.ipynb`](../examples/xtim_tutorial.ipynb)** — builds up
  from a single `T` gate. Start here if magic states / χ / post-selection are new.
- **[`xtim_tour.md`](xtim_tour.md)** — the lean end-to-end workflow on a real protocol:
  load → `diagnose()` → sample → decode → post-select → fidelity. Every code block is
  executed by the test suite, so it cannot rot.

## Can xtim simulate *my* circuit?
- **[`xtim_simulable_class.md`](xtim_simulable_class.md)** — the simulable class in one
  condition (non-Clifford gates must fold into one commuting π/8 layer), what the χ cost
  knob means, the controlled-Hadamard / H-eigenstate-cultivation story, and a
  reject-message → what-to-do table. **Read this before writing a new protocol.**
- **[`xtim_dialect.md`](xtim_dialect.md)** — the circuit language: full Stim grammar +
  the level-3 gates (`T`, `CS`, `CCZ`, `CH`) + `PAULI_EXPECTATION`. The reference for
  "does this line parse and what does it mean".
- **[`STIM_GAP_AUDIT.md`](STIM_GAP_AUDIT.md)** — feature-by-feature parity audit against
  Stim (what's supported, desugared, or deferred).

## I have a protocol — score it
- **[`xtim_scoring.md`](xtim_scoring.md)** — start here: the full `diagnose()`
  walkthrough, the fidelity arithmetic, and its four pitfalls (copied targets,
  byproduct frames, completeness, the F>1 guard), plus the `collect` shortcut.
- **[`xtim_postselected_logical_error_rate.md`](xtim_postselected_logical_error_rate.md)**
  — the one-call post-selected 1−F / logical-error-rate scorer (`target_k` + `collect`
  give you a plot-ready curve).
- **[`xtim_logical_fidelity_helpers.md`](xtim_logical_fidelity_helpers.md)** — opt-in
  helpers (`extract_frame`, `fidelity_from_logicals`) for when you need the frame
  arithmetic by hand.

## I want a decoder in the loop
- **[`xtim_dem_reject.md`](xtim_dem_reject.md)** — exporting an honest detector error
  model for magic-state prep: the clean Pauli-decodable part, plus the flagged faults a
  Pauli decoder *can't* correct (post-select or budget them). Includes what to do when
  the DEM genuinely refuses (post-selection-only protocols; CH-class circuits).

## Reference
- **[`xtim_practical_notes.md`](xtim_practical_notes.md)** — the command line, the
  reference cache, programmatic circuit building, reproducible seeding, scope in
  practice (χ as a cost knob; when to use Stim), and what each reject means.
- **[`../CHANGELOG.md`](../CHANGELOG.md)** — release notes; byte-stream stability policy
  per release.
- **[`../examples/README.md`](../examples/README.md)** — the bundled protocol circuits
  (`xtim.list_examples()`), each with its faithful/rate variants and measured
  throughput.
