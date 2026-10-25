"""S0d — the twirl engine's S0 diagnostics leave stderr (diagnostics policy, design §5.0).

Converted lines (each formerly a C++ stderr fprintf) and their channels:
  * "[twirl_records] exact fallback '<route>': n of N shots ..."  -> XtimPerformanceWarning, one per
    fired trigger per run (incl. the formerly SILENT non-commuting PPR routes);
  * "[xtim] selfcheck=<n>: oracle window active ..."                    -> XtimVerificationWarning, once
    per process;
  * "twirl_records PLAN KEY FAIL ..." / "... LOCAL INDEX ORACLE FAIL ..." -> XtimCorrectnessWarning
    (an actual engine-bug mismatch), then the oracle-abort RuntimeError;
  * "[twirl_records] disk '<path>': LOADED|absent/rebuild ..."     -> logging INFO on xtim.twirl.

DEFERRED EMISSION: the bindings release the GIL around run(); the shot loop only counts (the
exact_* route counters of record_report()) and keeps first-occurrence payloads; the Python layer
raises after run() returns. Pinned here: both polarities on every trigger fixture, absence from
fd 2, the warning raising only AFTER a completed run under an "error" filter (and a message
carrying the WHOLE run's count — a mid-loop emission could not know it), and category routing.
"""
import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import xtim
import xtim._diagnostics as diag
import xtim._xtim as _x
from test_record_report import M3_DET, M3_OBS, PPR23, Q3_DEC

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = Path(__file__).resolve().parent
_EXAMPLES = Path(__file__).resolve().parent / "data"
_CULT = (_EXAMPLES / "cultivation_d5.stim").read_text()


def _sampler(text):
    return _x.TwirlSampler(text, 1.0, "", 0)


def _run(s, mode, shots, seed):
    if mode == "sample":
        s.sample(shots, seed=seed)
    else:
        s.sample_barrier(shots, seed=seed, compact=True)


def _recorded(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return rec


def _exact_counts(s):
    return {k: v for k, v in s.record_report()["route_counts"].items()
            if k.startswith("exact") and v > 0}


# ── categories ───────────────────────────────────────────────────────────────────────────────
def test_categories_share_one_exported_base():
    for c in (xtim.XtimPerformanceWarning, xtim.XtimVerificationWarning,
              xtim.XtimCorrectnessWarning, xtim.XtimCacheWarning, xtim.XtimDemWarning):
        assert issubclass(c, xtim.XtimWarning) and issubclass(c, UserWarning)
        assert c.__name__ in xtim.__all__ and c.__doc__
    assert "XtimWarning" in xtim.__all__
    assert len({xtim.XtimPerformanceWarning, xtim.XtimVerificationWarning,
                xtim.XtimCorrectnessWarning}) == 3


def test_xtim_logger_has_a_null_handler():
    # installed where the records are raised (xtim._diagnostics, imported above)
    assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger("xtim").handlers)


# ── exact fallback -> XtimPerformanceWarning ─────────────────────────────────────────────────
# S2b: Born observables and Born decisions are record rows — their kappa>0 plans no longer take an
# exact route (the three exact_born_* / exact_obs_guard triggers are unreachable; negative below).
@pytest.mark.parametrize("text,mode", [(M3_OBS, "sample"), (M3_OBS, "barrier"), (Q3_DEC, "sample")],
                         ids=["born_obs_kpos", "born_obs_kpos_barrier", "born_dec_kpos"])
def test_born_record_routes_raise_no_exact_fallback_warning(text, mode):
    s = _sampler(text)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _run(s, mode, 3000, 11)
    assert _exact_counts(s) == {}
    assert not [w for w in rec if issubclass(w.category, xtim.XtimPerformanceWarning)]


@pytest.mark.parametrize("text,mode", [(M3_DET, "sample"), (M3_DET, "barrier"), (_CULT, "sample")],
                         ids=["kpos_record", "kpos_retention", "cult_d5"])
def test_no_performance_warning_without_an_exact_fallback(text, mode):
    s = _sampler(text)
    rec = _recorded(lambda: _run(s, mode, 2000, 11))
    assert _exact_counts(s) == {}
    assert not [w for w in rec if issubclass(w.category, xtim.XtimPerformanceWarning)]


# test_command_line_error_filter_and_forced_lever_subprocess: RETIRED at J-b1 F with the forced exact lever
# (QEC_TW_FORCE_RECORD_FALLBACK, plan §1.4 A P6 / §5.3: production reads no environment lever and has no exact route).
# The in-process error-filter rule stays pinned above on the legacy sampler.


# ── selfcheck window -> XtimVerificationWarning (once per process) ───────────────────────────
_SELFCHECK = (
    "import json, sys, warnings, xtim\n"
    "text = sys.stdin.read()\n"
    "out = []\n"
    "for sc in %s:\n"
    "    s = xtim.compile_twirl_sampler(text, selfcheck=sc)\n"
    "    with warnings.catch_warnings(record=True) as rec:\n"
    "        warnings.simplefilter('always')\n"
    "        s.sample(300, seed=2); s.sample(300, seed=3)\n"
    "    out.append([[w.category.__name__, str(w.message), w.filename] for w in rec])\n"
    "print(json.dumps(out))\n")


def _selfcheck_run(scs, quiet=False):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT)
    if quiet:
        env["XTIM_QUIET"] = "1"         # S7: removed; must have no effect
    p = subprocess.run([sys.executable, "-c", _SELFCHECK % (scs,)], input=_CULT,
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.splitlines()[-1]), p.stderr


def _window(ws):
    return [m for c, m, _ in ws if c == "XtimVerificationWarning" and "selfcheck=" in m]


def test_selfcheck_window_warns_once_per_process():
    out, err = _selfcheck_run([0, 2000, 2000])
    assert _window(out[0]) == []                                   # selfcheck=0: no window
    assert _window(out[1]) == ["[xtim] selfcheck=2000: oracle window active on the fast "
                               "path (first run of this sampler only; slower; diagnostic)"]
    assert [f for c, m, f in out[1] if c == "XtimVerificationWarning"] == ["<string>"]
    assert _window(out[2]) == []                                   # once per process
    assert "selfcheck=" not in err


def test_selfcheck_window_ignores_removed_xtim_quiet():
    """S7 removed XTIM_QUIET outright: setting it no longer silences the notice (filter it with warnings)."""
    out, err = _selfcheck_run([2000], quiet=True)
    assert _window(out[0]) == ["[xtim] selfcheck=2000: oracle window active on the fast "
                               "path (first run of this sampler only; slower; diagnostic)"]
    assert "selfcheck=" not in err


# ── plan-cache disk load -> logging INFO on xtim.twirl ───────────────────────────────────────
def test_disk_load_is_an_info_record_on_xtim_twirl(tmp_path, caplog, capfd):
    path = str(tmp_path / "m3.twpl")
    caplog.set_level(logging.INFO, logger="xtim.twirl")
    capfd.readouterr()
    s = _x.TwirlSampler(M3_DET, 1.0, path, 0, "")
    recs = [r for r in caplog.records if r.name == "xtim.twirl" and "disk '" in r.getMessage()]
    assert [r.levelno for r in recs] == [logging.INFO]
    assert recs[0].getMessage() == (f"[twirl_records] disk '{path}': absent/rebuild (0 plans; "
                                    "keys only, payloads built on first use)")
    _recorded(lambda: s.sample(500, seed=1))                     # saves the cache file
    caplog.clear()
    _x.TwirlSampler(M3_DET, 1.0, path, 0, "")
    recs = [r.getMessage() for r in caplog.records if r.name == "xtim.twirl" and "disk '" in r.getMessage()]
    assert len(recs) == 1 and f"disk '{path}': LOADED (" in recs[0], recs
    err = capfd.readouterr().err
    assert "absent/rebuild" not in err and "LOADED" not in err      # gone from fd 2
    caplog.clear()
    _recorded(lambda: _x.TwirlSampler(M3_DET, 1.0, "", 0, "").sample(200, seed=1))
    assert not [r for r in caplog.records                              # no cache, no disk record
                if r.name == "xtim.twirl" and "disk '" in r.getMessage()]


# ── oracle failures -> XtimCorrectnessWarning (category routing) ─────────────────────────────
def test_event_routing_per_kind(caplog, monkeypatch):
    """The one emitter maps each engine event kind to its channel. The oracle-failure kinds have
    no in-tree trigger (they report an engine bug; the end-to-end check is a mutant build, S0d
    report), so their routing is pinned here on synthetic events; an unknown kind is loud."""
    caplog.set_level(logging.INFO, logger="xtim.twirl")
    cases = [
        (("exact_fallback", "exact_chain_guard", 3, 10, "", 0.0), xtim.XtimPerformanceWarning,
         "[twirl_records] exact fallback 'exact_chain_guard': 3 of 10 shots in this run computed "
         "by the per-shot exact engine"),
        (("selfcheck_window", "", 7, 0, "", 0.0), xtim.XtimVerificationWarning,
         "[xtim] selfcheck=7: oracle window active on the fast path (first run of this "
         "sampler only; slower; diagnostic)"),
        (("oracle_failure", "plan_key", 4, 0, "twirl_records PLAN KEY FAIL shot=4: x", 0.0),
         xtim.XtimCorrectnessWarning, "twirl_records PLAN KEY FAIL shot=4: x"),
        (("oracle_failure", "local_index", 9, 0, "twirl_records LOCAL INDEX ORACLE FAIL shot=9", 0.0),
         xtim.XtimCorrectnessWarning, "twirl_records LOCAL INDEX ORACLE FAIL shot=9"),
    ]
    for ev, cat, text in cases:
        rec = _recorded(lambda: diag.emit_twirl([ev]))
        assert [(w.category, str(w.message)) for w in rec] == [(cat, text)]
        assert not caplog.records
    rec = _recorded(lambda: diag.emit_twirl([("disk_load", "LOADED", 2, 0, "/p.twpl", 0.0)]))
    assert rec == [] and [r.getMessage() for r in caplog.records] == [
        "[twirl_records] disk '/p.twpl': LOADED (2 plans; keys only, payloads built on first use)"]
    caplog.clear()
    rec = _recorded(lambda: diag.emit_twirl([("channels", "", 3, 0, "[twirl_records] circuit channels: x", 0.0)]))
    assert rec == [] and [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, "[twirl_records] circuit channels: x")]
    with pytest.raises(RuntimeError, match="unknown twirl diagnostic kind"):
        diag.emit_twirl([("bogus", "", 0, 0, "", 0.0)])


def test_emit_outcome_rules():
    perf = ("exact_fallback", "exact_chain_guard", 1, 2, "", 0.0)
    corr = ("oracle_failure", "plan_key", 4, 0, "twirl_records PLAN KEY FAIL shot=4: x", 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", xtim.XtimWarning)
        # abort: RuntimeError raised FROM the first warning that raised; all events emitted
        with pytest.raises(RuntimeError, match="^abort text$") as ei:
            diag.emit_twirl([perf, corr], "abort text")
        assert isinstance(ei.value.__cause__, xtim.XtimPerformanceWarning)
        # collect: nothing raised, the first exception returned
        first = diag.emit_twirl([corr, perf], None, True)
        assert isinstance(first, xtim.XtimCorrectnessWarning)
        # no abort: the first warning is the outcome
        with pytest.raises(xtim.XtimCorrectnessWarning):
            diag.emit_twirl([corr, perf])
    with pytest.raises(RuntimeError, match="^abort text$") as ei:
        _recorded(lambda: diag.emit_twirl([perf], "abort text"))
    assert ei.value.__cause__ is None
    assert _recorded(lambda: diag.emit_twirl([perf], None, True)) and \
        diag.emit_twirl([], None, True) is None


def test_base_exceptions_propagate_immediately():
    """Only Exception is collected: a KeyboardInterrupt raised while emitting one event stops the
    batch at once (the later warning is NOT emitted)."""
    class Boom(logging.Handler):
        def emit(self, record):
            raise KeyboardInterrupt
    lg = logging.getLogger("xtim.twirl")
    h, old = Boom(), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(KeyboardInterrupt):
                diag.emit_twirl([("disk_load", "LOADED", 1, 0, "/p", 0.0),
                                 ("exact_fallback", "exact_chain_guard", 1, 2, "", 0.0)])
        assert rec == []
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


def test_born_node_value_is_formatted_like_glibc():
    """R-spread (S2b): the born_node event's measured time travels as a raw double and Python's
    '%.1f' formats it. The former engine text was glibc snprintf("%.1f", seconds * 1e6); both are
    correctly rounded (ties to even on the exact binary value), so the text is identical for every
    double. Checked against libc's snprintf over random doubles and exact ties."""
    import ctypes
    import ctypes.util
    import random
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    buf = ctypes.create_string_buffer(64)
    rng = random.Random(20261016)
    vals = [rng.uniform(0.0, 1.0) * 10 ** rng.randint(-9, 2) for _ in range(100_000)]
    vals += [0.25, 0.75, 2.5, 0.05, 0.15, 1e6 * 2 ** -20, 1e6 * 2 ** -21, 0.0]
    for sec in vals:
        us = sec * 1e6
        libc.snprintf(buf, 64, b"%.1f", ctypes.c_double(us))
        assert buf.value.decode() == "%.1f" % us, sec


def test_born_node_event_routes_the_raw_value(caplog):
    caplog.set_level(logging.INFO, logger="xtim.twirl")
    rec = _recorded(lambda: diag.emit_twirl([("born_node", "", 16, 2, "", 3.25e-6)]))
    assert rec == []
    assert [r.getMessage() for r in caplog.records] == [
        "[twirl_records] Born-trie node evaluation: chi=16, kappa_rec=2, first node of this call "
        "3.2 us (node work grows with chi*j; counters in record_report()['born_trie'])"]


def test_fast_out_makes_no_python_call_on_a_clean_run_public(monkeypatch):
    """J-b2 J1b (plan r2.1 J1b): the production half of test_fast_out_makes_no_python_call_on_a_clean_run — a warm,
    oracle-clean run queues no event, so the engine makes no Python call at all. (Production has no exact-fallback
    trigger; the multi-event emission rules are pinned on synthetic events by test_emit_outcome_rules.)"""
    calls = []
    real = diag.emit_twirl
    monkeypatch.setattr(diag, "emit_twirl", lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    s = _sampler(M3_DET)
    s.sample(2000, seed=11)
    s.sample_barrier(500, seed=3)
    calls.clear()
    s.sample(2000, seed=11)
    s.sample_barrier(500, seed=3)
    assert s.record_report()["born_trie"]["node_evals"] == 0
    assert calls == []
