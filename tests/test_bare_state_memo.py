"""The process-level framed bare-state memo (cpp/src/bare_state_memo.cpp, perf/dem-share-bare).

A compile (exact or twirl engine), a DEM export, detector_determinism and expectation_frame_check
on the same circuit all build the same bare state; the memo serves the later callers from the
first build. Contract: a hit is byte-identical to a fresh build (XTIM_BARE_MEMO=verify rebuilds
and compares on EVERY hit — a difference raises), every emitted stream / DEM text is identical with
the memo on and off, the memo is bounded (count AND bytes) and lives OUTSIDE the state budget.
The levers are read once per process, so each case runs in a child interpreter.
"""
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import xtim
from xtim import _xtim
from _orig_examples import ORIG_EXAMPLES  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = sorted(n + ".stim" for n in ORIG_EXAMPLES)   # the original bundled examples (tests/data)
assert len(EXAMPLES) >= 8, EXAMPLES


def _child(code, env_extra=None, timeout=900):
    env = dict(os.environ)
    for k in ("XTIM_BARE_MEMO", "XTIM_BARE_MEMO_MB", "XTIM_BARE_MEMO_ENTRIES"):
        env.pop(k, None)
    env.update(env_extra or {})
    pre = textwrap.dedent(f"""
        import json, sys, warnings, tempfile
        warnings.simplefilter("ignore")
        import xtim
        from xtim import _xtim
        xtim.cache_dir = tempfile.mkdtemp(prefix="xtim_memo_test_")
        ROOT = {str(ROOT)!r}
        EXAMPLES = {EXAMPLES!r}
        def text_of(name):
            return open(f"{{ROOT}}/tests/data/{{name}}").read()
    """)
    out = subprocess.run([sys.executable, "-c", pre + textwrap.dedent(code)], env=env,
                         capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


# Every consumer path on one circuit, in the order a user runs them: exact compile, DEM, DEM
# (approx), twirl compile, determinism, frame check, DEM again. Returns the streams/texts as hashes.
CONSUMERS = textwrap.dedent("""
    import hashlib, numpy as np
    def H(a):
        return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
    def leg(fn):
        try:
            return fn()
        except Exception as e:
            return "REFUSED " + type(e).__name__ + ": " + " ".join(str(e).split())[:200]
    def consume(text):
        c = xtim.Circuit(text)
        r = {}
        r["det_exact"] = leg(lambda: H(c.compile_detector_sampler(seed=3, engine="exact").sample(300)))
        r["dem"] = leg(lambda: hashlib.sha256(c.detector_error_model_text().encode()).hexdigest()[:16])
        r["dem_approx"] = leg(lambda: hashlib.sha256(
            c.detector_error_model_text(approximate_disjoint_errors=True).encode()).hexdigest()[:16])
        r["det_twirl"] = leg(lambda: H(c.compile_detector_sampler(seed=6, engine="twirl").sample(1000)))
        r["determinism"] = leg(lambda: hashlib.sha256(
            json.dumps(_xtim.detector_determinism(text), sort_keys=True, default=str).encode()).hexdigest()[:16])
        r["dem2"] = leg(lambda: hashlib.sha256(c.detector_error_model_text().encode()).hexdigest()[:16])
        return r
""")


def _run_consumers(env_extra):
    code = CONSUMERS + textwrap.dedent("""
        res, stats = {}, {}
        prev = _xtim.bare_state_memo_stats()
        for name in EXAMPLES:
            res[name] = consume(text_of(name))
            cur = _xtim.bare_state_memo_stats()
            stats[name] = {k: cur[k] - prev[k] for k in ("hits", "misses", "stores")}
            prev = cur
        print(json.dumps(dict(res=res, stats=stats, final=_xtim.bare_state_memo_stats())))
    """)
    return _child(code, env_extra)


def test_verify_mode_every_hit_rebuilt_and_identical():
    """XTIM_BARE_MEMO=verify: every memo hit is followed by a fresh build and a field-by-field
    (bit-level coefficient) comparison that raises on any difference. Over the bundled examples
    and every consumer path no hit differs, and hits do occur (the DEM after a compile)."""
    out, err = _run_consumers({"XTIM_BARE_MEMO": "verify"})
    assert out["final"]["verify"] is True and out["final"]["enabled"] is True
    # per circuit: the first path misses, every later path hits (ch_cultivation: its DEM refuses
    # before the build, so only the twirl compile + determinism hit); the `_rate` examples share
    # their gate stream with the `_faithful` ones (noise-only difference) and hit from the start.
    assert out["final"]["hits"] >= 4 * len(EXAMPLES), out["final"]
    for name, s in out["stats"].items():
        assert s["hits"] >= 2, (name, s)
        assert s["misses"] <= 1, (name, s)
    assert "memo verify" not in err


def test_streams_identical_with_memo_on_and_off():
    """The memo changes no bytes: DEM texts, detector streams (exact + twirl engines), determinism
    dicts are identical between XTIM_BARE_MEMO=0 (every call builds fresh) and the default."""
    off, _ = _run_consumers({"XTIM_BARE_MEMO": "0"})
    on, _ = _run_consumers({})
    assert off["final"]["enabled"] is False and off["final"]["hits"] == 0
    assert off["final"]["bypassed"] > 0 and off["final"]["entries"] == 0
    assert on["final"]["hits"] > 0
    assert off["res"] == on["res"]


def test_twirl_compile_then_dem_hits():
    """The twirl engine's compile (compile_detector_sampler default) and a DEM export after it
    share one build; a second DEM hits again; a compile after a DEM hits."""
    out, _ = _child(textwrap.dedent("""
        text = text_of("cultivation_d3_faithful.stim")
        c = xtim.Circuit(text)
        _xtim.bare_state_memo_reset_stats()
        s = c.compile_detector_sampler(seed=1, engine="twirl")
        a = dict(_xtim.bare_state_memo_stats())
        c.detector_error_model_text(approximate_disjoint_errors=True)
        b = dict(_xtim.bare_state_memo_stats())
        c.detector_error_model_text(approximate_disjoint_errors=True)
        d = dict(_xtim.bare_state_memo_stats())
        e2 = xtim.Circuit(text).compile_detector_sampler(seed=2, engine="exact")
        e = dict(_xtim.bare_state_memo_stats())
        print(json.dumps(dict(a=a, b=b, d=d, e=e)))
    """))
    assert out["a"]["stores"] == 1 and out["a"]["hits"] == 0
    assert out["b"]["hits"] == 1 and out["b"]["stores"] == 1
    assert out["d"]["hits"] == 2
    assert out["e"]["hits"] == 3 and out["e"]["entries"] == 1


def test_bounds_entries_and_bytes():
    """XTIM_BARE_MEMO_ENTRIES=1 keeps at most one entry (LRU eviction); XTIM_BARE_MEMO_MB=0 stores
    nothing (every build skipped, zero bytes held) and still serves every call correctly."""
    code = CONSUMERS + textwrap.dedent("""
        res = {n: consume(text_of(n)) for n in EXAMPLES[:4]}
        print(json.dumps(dict(res=res, stats=_xtim.bare_state_memo_stats())))
    """)
    one, _ = _child(code, {"XTIM_BARE_MEMO_ENTRIES": "1"})
    assert one["stats"]["cap_entries"] == 1
    # EXAMPLES[:4] hold 3 distinct gate streams (code_switching_rate == _faithful up to noise)
    assert one["stats"]["entries"] == 1 and one["stats"]["stores"] == 3, one["stats"]
    assert one["stats"]["evictions"] == 2, one["stats"]
    zero, _ = _child(code, {"XTIM_BARE_MEMO_MB": "0"})
    assert zero["stats"]["cap_bytes"] == 0
    assert zero["stats"]["entries"] == 0 and zero["stats"]["bytes"] == 0
    assert zero["stats"]["stores"] == 0 and zero["stats"]["skipped"] > 0 and zero["stats"]["hits"] == 0
    off, _ = _child(code, {"XTIM_BARE_MEMO": "0"})
    assert one["res"] == zero["res"] == off["res"]


def test_memo_is_outside_the_state_budget():
    """A held memo entry is not chi-scaled state storage: after a compile + DEM (entry held) and
    dropping every Python object, live_state_bytes() is back to its baseline (the 3.1.10 contract)
    while the memo still holds the entry."""
    out, _ = _child(textwrap.dedent("""
        import gc
        text = text_of("cultivation_d3_faithful.stim")
        gc.collect(); _xtim._release_thread_state_scratch()
        live0 = xtim.live_state_bytes()
        c = xtim.Circuit(text)
        s = c.compile_detector_sampler(seed=3, engine="exact"); s.sample(50)
        c.detector_error_model_text(approximate_disjoint_errors=True)
        peak = xtim.live_state_bytes()
        del s, c
        gc.collect(); _xtim._release_thread_state_scratch()
        st = _xtim.bare_state_memo_stats()
        print(json.dumps(dict(live0=live0, peak=peak, end=xtim.live_state_bytes(), st=st)))
    """))
    assert out["peak"] > out["live0"]
    assert out["end"] == out["live0"], out
    assert out["st"]["entries"] >= 1 and out["st"]["bytes"] > 0 and out["st"]["hits"] >= 1


def test_default_caps_and_stats_surface():
    st = _xtim.bare_state_memo_stats()
    for k in ("hits", "misses", "stores", "skipped", "evictions", "bypassed", "entries", "bytes",
              "cap_entries", "cap_bytes", "cap_entry_bytes", "enabled", "verify"):
        assert k in st
    if "XTIM_BARE_MEMO_ENTRIES" not in os.environ:
        assert st["cap_entries"] == 8
    if "XTIM_BARE_MEMO_MB" not in os.environ:
        assert st["cap_bytes"] == 256 << 20
    if "XTIM_BARE_MEMO_ENTRY_MB" not in os.environ:
        assert st["cap_entry_bytes"] == 16 << 20
    assert st["bytes"] <= st["cap_bytes"] and st["entries"] <= st["cap_entries"]


def test_large_chi_state_is_never_captured():
    """A state over the per-entry cap (default 16 MiB; a chi = 2^20 BH state is ~97 MB) is skipped
    before any copy: a cold DEM on it costs exactly the build, holds nothing, and a second DEM
    rebuilds (never a hit). With the cap raised the same state is stored and hit."""
    qs = " ".join(map(str, range(20)))
    bh20 = f"RX {qs}\nT {qs}\nCX 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19\nMX {qs}\n"
    out, _ = _child(textwrap.dedent(f"""
        c = xtim.Circuit({bh20!r})
        _xtim.bare_state_memo_reset_stats()
        c.detector_error_model_text()
        a = dict(_xtim.bare_state_memo_stats())
        c.detector_error_model_text()
        b = dict(_xtim.bare_state_memo_stats())
        _xtim.bare_state_memo_set_entry_cap(1 << 30)
        c.detector_error_model_text()
        c.detector_error_model_text()
        d = dict(_xtim.bare_state_memo_stats())
        print(json.dumps(dict(a=a, b=b, d=d)))
    """))
    assert out["a"]["skipped"] == 1 and out["a"]["entries"] == 0 and out["a"]["bytes"] == 0, out["a"]
    assert out["b"]["skipped"] == 2 and out["b"]["hits"] == 0 and out["b"]["misses"] == 2, out["b"]
    assert out["d"]["stores"] == 1 and out["d"]["hits"] == 1, out["d"]
    assert out["d"]["bytes"] > 64 << 20


@pytest.mark.parametrize("var,value", [("XTIM_BARE_MEMO", "bogus"), ("XTIM_BARE_MEMO", "trace"),
                                       ("XTIM_BARE_MEMO_MB", "12x"), ("XTIM_BARE_MEMO_ENTRIES", "-1"),
                                       ("XTIM_BARE_MEMO_ENTRY_MB", "")])
def test_malformed_lever_is_refused_by_name(var, value):
    """S7 fold (#2): a malformed lever value is a NAMED refusal (ValueError naming the variable and the value) on the
    first memo use — no stderr line, no silent default; the trace mode is gone (``trace`` is refused). An EMPTY value
    is "unset" (the default), as before."""
    env = dict(os.environ)
    for k in ("XTIM_BARE_MEMO", "XTIM_BARE_MEMO_MB", "XTIM_BARE_MEMO_ENTRIES", "XTIM_BARE_MEMO_ENTRY_MB"):
        env.pop(k, None)
    env[var] = value
    code = textwrap.dedent("""
        import json, xtim
        from xtim import _xtim
        out = {}
        for leg, fn in (("stats", _xtim.bare_state_memo_stats),
                        ("dem", lambda: xtim.Circuit("R 0\\nX_ERROR(0.1) 0\\nM 0\\nDETECTOR rec[-1]\\n")
                                             .detector_error_model_text())):
            try:
                fn(); out[leg] = "ok"
            except ValueError as e:
                out[leg] = "ValueError: " + str(e)
        print(json.dumps(out))
    """)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    if value == "":
        assert got == {"stats": "ok", "dem": "ok"}
    else:
        for leg in ("stats", "dem"):
            assert got[leg].startswith("ValueError: ") and f"{var}='{value}'" in got[leg], got
    assert "[xtim]" not in proc.stderr and "bare-state memo" not in proc.stderr, proc.stderr
