"""The twirl engine's diagnostics, raised in the Python layer (diagnostics policy, design §5.0).

The C++ engine writes no stderr line for these events (stage S0d). Construction and ``run()``
append them to ``TwirlRecordSampler::take_diagnostics()``; the bindings call
:func:`emit_twirl` with the GIL held AFTER the run (or the construction) returns — never from
the shot loop, which only counts (the bindings release the GIL around ``run()``, and under
``-W error`` a warning raised mid-loop would raise inside C++). Each event is one tuple
``(kind, name, count, shots, detail, value)`` (``value`` is a float; 0.0 unless stated):

==================  =============================  ==============================================
kind                channel                        fields
==================  =============================  ==============================================
exact_fallback      ``XtimPerformanceWarning``     name = ``exact_*`` route (``record_report()``
                                                   key), count = its shots in this run, shots =
                                                   the run size. One per trigger per run.
selfcheck_window    ``XtimVerificationWarning``    count = oracle window size. Once per process.
disk_load           ``logging`` INFO, ``xtim.twirl``  name = LOADED | absent/rebuild, count =
                                                   plans, detail = plan-cache file path.
disk_save           ``logging`` INFO, ``xtim.twirl``  J-b1 P2 (family samplers): name = SAVED, count = plans,
                                                   detail = plan-cache file path.
oracle_failure      ``XtimCorrectnessWarning``     name = plan_key | local_index, count = shot,
                                                   detail = message; the call then raises the
                                                   oracle-abort ``RuntimeError``.
born_node           ``logging`` INFO, ``xtim.twirl``  S2a, owner decision 7 (large chi): at most ONE
                                                   per call, queued at the call's FIRST Born-trie
                                                   node evaluation; count = chi (bare free-register
                                                   support), shots = kappa_rec of that plan, value
                                                   = that node's evaluation time in SECONDS (raw;
                                                   formatted here, '%.1f' us). No threshold, no
                                                   cap; none when no node is evaluated (warm
                                                   tries, kappa = 0).
channels            ``logging`` INFO, ``xtim.twirl``  S7: one per compile; detail = the "[twirl_records]
                                                   circuit channels: ..." text, count = deterministic
                                                   channels.
==================  =============================  ==============================================

The texts are the former stderr lines verbatim. Silencing is by ``logging`` levels and ``warnings`` filters (no
environment variable). An unknown kind is an engine/binding skew and
raises (never silently dropped). Each call's events are emitted with that call (also when the call
fails); nothing is queued across calls. Outcome rules: :func:`emit_twirl`.
"""
from __future__ import annotations

import logging
import warnings

from .errors import XtimCorrectnessWarning, XtimPerformanceWarning, XtimVerificationWarning

# Library logging convention: the "xtim" logger carries a NullHandler (silent unless the
# application configures logging). Installed here, where the first xtim log record is raised,
# so importing xtim costs no `logging` import.
logging.getLogger("xtim").addHandler(logging.NullHandler())
twirl_log = logging.getLogger("xtim.twirl")


def _emit_one(kind, name, count, shots, detail, value) -> None:
    # stacklevel=3: _emit_one <- emit_twirl <- (C++ binding, no frame) <- the sampling call site
    if kind == "exact_fallback":
        warnings.warn(
            f"[twirl_records] exact fallback '{name}': {count} of {shots} shots in this run "
            f"computed by the per-shot exact engine", XtimPerformanceWarning, stacklevel=3)
    elif kind == "selfcheck_window":
        warnings.warn(
            f"[xtim] selfcheck={count}: oracle window active on the fast path (first "
            f"run of this sampler only; slower; diagnostic)", XtimVerificationWarning,
            stacklevel=3)
    elif kind == "disk_load":
        twirl_log.info("[twirl_records] disk '%s': %s (%d plans; keys only, payloads built "
                       "on first use)", detail, name, count)
    elif kind == "disk_save":
        twirl_log.info("[twirl_records] disk '%s': %s (%d plans)", detail, name, count)
    elif kind == "oracle_failure":
        warnings.warn(detail, XtimCorrectnessWarning, stacklevel=3)
    elif kind == "born_node":
        # The engine passes the measured seconds RAW (R-spread, S2b: no formatting inside run());
        # '%.1f' of value*1e6 here is the same correctly rounded text glibc's snprintf produced.
        twirl_log.info("[twirl_records] Born-trie node evaluation: chi=%d, kappa_rec=%d, first node "
                       "of this call %.1f us (node work grows with chi*j; counters in "
                       "record_report()['born_trie'])", count, shots, value * 1e6)
    elif kind == "channels":
        twirl_log.info("%s", detail)
    else:
        raise RuntimeError(f"xtim: unknown twirl diagnostic kind {kind!r} (engine/binding skew)")


def emit_twirl(events, abort_msg=None, collect=False):
    """Emit ONE call's twirl-engine events, then produce the call's ONE outcome.

    Every event is emitted, in queue order, each under ``except Exception`` (so a warning that
    an "error" filter turns into an exception never hides a later event; KeyboardInterrupt /
    SystemExit propagate immediately); ``first`` is the first such exception. Then:

    * ``collect=True`` (the bindings are propagating a C++ exception, which stays the outcome):
      return ``first`` (or None) without raising;
    * ``abort_msg`` given (the run's oracle failed): ``raise RuntimeError(abort_msg) from first``;
    * else ``first`` is raised if any warning raised; else return None.
    """
    first = None
    for ev in events:
        try:
            _emit_one(*ev)
        except Exception as e:      # noqa: BLE001 — every Exception is collected, never dropped
            if first is None:
                first = e
    if collect:
        return first
    if abort_msg is not None:
        if first is not None:
            raise RuntimeError(abort_msg) from first
        raise RuntimeError(abort_msg)
    if first is not None:
        raise first
    return None
