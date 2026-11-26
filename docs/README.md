# xtim documentation

- [usage.md](usage.md): the API in one page (sampling, post-selection, fidelity, closing frames, output state,
  decoding, xdem, refusals, circuit language).
- [xtim_simulable_class.md](xtim_simulable_class.md): which circuits xtim can simulate.
- [xtim_postselected_logical_error_rate.md](xtim_postselected_logical_error_rate.md): the one-call post-selected
  logical error rate on `OBSERVABLE_INCLUDE`s.
- [xdem.md](xdem.md): the xdem route, file-only sampling with decoder feedback (an extended detector error model with
  diagonal-Clifford fault parts, `xtim.XdemSampler`, exact per-shot scoring of a decoder's correction).
- [performance.md](performance.md): measured speed against other simulators of noisy non-Clifford circuits on public
  workloads (per-shot throughput, time to a logical error rate, exact output fidelity), and compile and DEM export
  time on large circuits.
- [../examples/](../examples/): ten runnable scripts and the bundled circuits;
  [../examples/xdem/](../examples/xdem/): circuit and xdem file pairs.

For contributors: [twirl_record_sampler.md](twirl_record_sampler.md) (engine contract),
[branch_frames.md](branch_frames.md), [frame_cosets_cpp.md](frame_cosets_cpp.md),
[xtim_forced_faults.md](xtim_forced_faults.md) (test API).
