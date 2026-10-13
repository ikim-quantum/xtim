"""The twirl fast sampler: 9-28x faster detector/observable sampling (v2.3.0, opt-in).

`compile_detector_sampler(engine="auto")` uses the stabilizer-twirl record engine when
the circuit is eligible and the exact engine otherwise. It samples the SAME
deterministic detector and observable statistics as the exact engine -- including their
joint, so post-selected / decoded logical error rates are exact -- by compiling each
noise pattern into a cached "plan" and emitting record bits from closed-form channel
algebra instead of evolving the state.

Exact (5-sigma-gated vs the exact engine and vs Stim on every bundled circuit):
  * every deterministic detector frequency, jointly (post-selection exact);
  * logical-observable statistics including their joint with the detectors.
Different: raw byte streams (same seed -> different bytes; distributions match and are
gated) and gauge detectors (declared fair coins; auto routes such circuits to exact).

Run:  python twirl_fast_sampling.py
"""
import time

import numpy as np

import xtim

c = xtim.load_example("cultivation_d5")
SHOTS = 200_000

# Exact engine (the default) -- unchanged from previous releases.
s_exact = c.compile_detector_sampler(seed=7, engine="exact")
t0 = time.perf_counter()
dets_e, _ = s_exact.sample(SHOTS, separate_observables=True)
t_exact = time.perf_counter() - t0

# Twirl engine via auto (routes to exact if the circuit is ineligible).
s_auto = c.compile_detector_sampler(seed=7, engine="auto")
print("engine:", s_auto.engine_report()["engine"],
      "--", s_auto.engine_report()["reason"])
t0 = time.perf_counter()
dets_t, _ = s_auto.sample(SHOTS, separate_observables=True)
t_twirl = time.perf_counter() - t0

# Same physics: per-detector frequencies agree at 5 sigma.
fe = np.asarray(dets_e, dtype=np.float64).mean(axis=0)
ft = np.asarray(dets_t, dtype=np.float64).mean(axis=0)
se = np.sqrt(fe * (1 - fe) / SHOTS + ft * (1 - ft) / SHOTS) + 1e-12
worst = float(np.max(np.abs(fe - ft) / se))
print(f"detectors: {c.num_detectors} columns, worst |z| = {worst:.2f} (must be < 5)")
assert worst < 5.0

print(f"exact:  {t_exact * 1e6 / SHOTS:.3f} us/shot")
print(f"twirl:  {t_twirl * 1e6 / SHOTS:.3f} us/shot")
print("(the twirl's first call includes plan compilation; repeat sample() calls and the "
      "optional disk_cache= amortize it, and the warm per-shot cost is what scales.)")
