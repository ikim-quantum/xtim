"""Process-wide budget on live chi-scaled state storage (3.1.9).

Every container whose size scales with chi = 2^r (bare-state branch lists, FramedSuperposition
amplitude entries, parsed reference branches) allocates through StateAllocator, which charges one
shared counter on allocation and releases exactly that much on deallocation; transient chi-scaled
build tables are charged by RAII tickets.  An allocation that would push the live total over the
budget (2 GiB, or XTIM_MAX_STATE_BYTES, read once per process) raises xtim.XtimCapacityError BEFORE
it happens.  Budget variants run in subprocesses (the variable is read once).
"""
import gc
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

import xtim
from xtim import _xtim

ROOT = pathlib.Path(__file__).resolve().parents[1]
BH12 = """RX 0 1 2 3 4 5 6 7 8 9 10 11
T 0 1 2 3 4 5 6 7 8 9 10 11
CX 0 1 2 3 4 5 6 7 8 9 10 11
MX 0 1 2 3 4 5 6 7 8 9 10 11
"""


def _all_t(n):
    qs = " ".join(map(str, range(n)))
    return f"RX {qs}\nT {qs}\nMX {qs}\n"


def _child(code, budget=None):
    """Run `code` in a fresh interpreter (the budget is read once per process); it prints one JSON
    line.  Returns the parsed dict."""
    env = dict(os.environ)
    env.pop("XTIM_MAX_STATE_BYTES", None)
    if budget is not None:
        env["XTIM_MAX_STATE_BYTES"] = str(budget)
    pre = textwrap.dedent("""
        import json, resource, sys, warnings
        warnings.simplefilter("ignore")
        import xtim
        def rss_kb():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    """)
    out = subprocess.run([sys.executable, "-c", pre + textwrap.dedent(code)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


REFUSE = textwrap.dedent("""
    try:
        st = xtim.bare_state_of(TEXT)
        res = {"refused": False}
    except xtim.XtimCapacityError as e:
        res = {"refused": True, "msg": str(e), "mem": isinstance(e, MemoryError)}
    res["live"] = xtim.live_state_bytes()
    try:
        res["budget"] = xtim.state_budget_bytes()
    except xtim.XtimCapacityError:
        res["budget"] = None
    print(json.dumps(res))
""")


def _needs(msg):
    return int(re.search(r"needs (\d+) bytes", msg).group(1))


def test_default_budget_is_2gib_and_benchmarks_fit():
    assert xtim.state_budget_bytes() == 2 << 30 or "XTIM_MAX_STATE_BYTES" in os.environ
    live0 = xtim.live_state_bytes()
    s = xtim.bare_state_of(BH12)
    assert 0 < xtim.live_state_bytes() - live0 < 2 << 20      # chi = 4096: well under a MiB-ish
    del s


def test_one_state_over_budget_refuses_before_allocating():
    """r = 24 all-T: chi = 2^24, ~3 GiB at the build peak > 2 GiB.  Refused up front: nothing
    chi-sized is allocated (peak RSS stays near the interpreter's), nothing stays charged."""
    res = _child("TEXT = " + repr(_all_t(24)) + "\nbase = rss_kb()\n" + REFUSE
                 .replace("print(json.dumps(res))", "res['rss_growth_kb'] = rss_kb() - base\nprint(json.dumps(res))"))
    assert res["refused"] and res["mem"], res
    msg = res["msg"]
    for frag in ("needs", "already live", "the budget is 2147483648 bytes", "chi = 16777216",
                 "r = 24", "XTIM_MAX_STATE_BYTES", "Refused before allocating"):
        assert frag in msg, (frag, msg)
    assert _needs(msg) > 2 << 30
    assert res["live"] == 0
    assert res["rss_growth_kb"] < 64 * 1024, res          # no multi-GiB allocation was attempted


def test_boundary_exact_and_override():
    """The build peak is charged exactly as estimated: budget == need passes, need - 1 refuses.
    The override both lowers and raises the limit (and accepts K/M/G suffixes)."""
    text = _all_t(10)
    probe = _child("TEXT = " + repr(text) + REFUSE, budget=1)
    assert probe["refused"] and probe["budget"] == 1
    need = _needs(probe["msg"])
    under = _child("TEXT = " + repr(text) + REFUSE, budget=need)
    over = _child("TEXT = " + repr(text) + REFUSE, budget=need - 1)
    assert not under["refused"] and under["live"] > 0, under
    assert over["refused"] and over["live"] == 0, over
    raised = _child("TEXT = " + repr(text) + REFUSE, budget="1G")
    assert not raised["refused"] and raised["budget"] == 1 << 30


def test_bad_override_is_named():
    res = _child("TEXT = " + repr(_all_t(3)) + REFUSE, budget="12Q")
    assert res["refused"] and "XTIM_MAX_STATE_BYTES='12Q'" in res["msg"], res


def test_copies_cross_the_budget_and_dropping_frees_it():
    """Several copies that fit one by one but not together: the copy that would cross refuses;
    dropping a copy frees exactly its bytes and the next copy fits again."""
    live0 = xtim.live_state_bytes()
    s = xtim.bare_state_of(BH12)
    one = xtim.live_state_bytes() - live0
    del s
    budget = int(4.5 * one)
    res = _child(f"""
        s = xtim.bare_state_of({BH12!r})
        one = xtim.live_state_bytes()
        copies, refused_at, msg = [], None, ""
        for i in range(8):
            try:
                copies.append(s.copy())
            except xtim.XtimCapacityError as e:
                refused_at, msg = i, str(e)
                break
        live_full = xtim.live_state_bytes()
        copies.pop()
        live_after_drop = xtim.live_state_bytes()
        copies.append(s.copy())
        del copies, s
        print(json.dumps(dict(one=one, refused_at=refused_at, msg=msg, live_full=live_full,
                              live_after_drop=live_after_drop, live_end=xtim.live_state_bytes())))
    """, budget=budget)
    assert res["one"] == one
    assert res["refused_at"] == 3, res                     # s + 3 copies = 4 states <= 4.5; a 5th > 4.5
    assert "needs" in res["msg"] and str(budget) in res["msg"]
    assert res["live_full"] == 4 * one
    assert res["live_after_drop"] == 3 * one
    assert res["live_end"] == 0


def test_live_total_returns_to_baseline():
    """Build, copy, compile (twirl and exact), sample, materialize barrier states, export branch
    frames — then drop everything: the live total is back to its baseline (exact release)."""
    gc.collect()
    _xtim._release_thread_state_scratch()
    live0 = xtim.live_state_bytes()
    text = (ROOT / "examples" / "cultivation_d3_faithful.stim").read_text()
    objs = []
    s = xtim.bare_state_of(BH12)
    objs += [s, s.copy(), s.branch_frames()]
    c = xtim.Circuit(text)
    for eng in ("twirl", "exact"):
        smp = c.compile_detector_sampler(seed=3, engine=eng)
        smp.sample(200)
        objs.append(smp)
    ts = _xtim.TwirlSampler("R 0 1 2\nH 0\nT 0\nCX 0 1\nCX 0 2\nCX 1 2\nM 2\nDETECTOR rec[-1]\n"
                            "OUTPUT_QUBITS out 0 1\n", 0.0, "", 0, False, "")
    buf = ts.sample_barrier(16, 5)
    objs += [ts, buf] + [buf.materialize(i) for i in range(16)]
    assert xtim.live_state_bytes() > live0
    del objs, s, c, smp, ts, buf
    gc.collect()
    _xtim._release_thread_state_scratch()
    assert xtim.live_state_bytes() == live0
