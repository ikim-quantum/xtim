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
    # S2k re-pin: an entry is charged 24 B (packed key), so one state is ~0.1 MB while the build
    # peak (the CanonicalStabSum list + the entry list + the build tables) is ~6.9 states; 7.5
    # states keeps the build under the budget and still crosses it with copies.
    budget = int(7.5 * one)
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
    assert res["refused_at"] == 6, res                     # s + 6 copies = 7 states <= 7.5; an 8th > 7.5
    assert "needs" in res["msg"] and str(budget) in res["msg"]
    assert res["live_full"] == 7 * one
    assert res["live_after_drop"] == 6 * one
    assert res["live_end"] == 0


def test_live_total_returns_to_baseline():
    """Build, copy, compile (twirl and exact), sample, materialize barrier states, export branch
    frames — then drop everything: the live total is back to its baseline (exact release)."""
    gc.collect()
    _xtim._release_thread_state_scratch()
    live0 = xtim.live_state_bytes()
    text = (ROOT / "tests" / "data" / "cultivation_d3_faithful.stim").read_text()
    objs = []
    s = xtim.bare_state_of(BH12)
    objs += [s, s.copy(), s.branch_frames()]
    c = xtim.Circuit(text)
    for eng in ("twirl", "exact"):
        smp = c.compile_detector_sampler(seed=3, engine=eng)
        smp.sample(200)
        objs.append(smp)
    ts = _xtim.TwirlSampler("R 0 1 2\nH 0\nT 0\nCX 0 1\nCX 0 2\nCX 1 2\nM 2\nDETECTOR rec[-1]\n"
                            "OUTPUT_QUBITS out 0 1\n", 0.0, "", 0, "")
    buf = ts.sample_barrier(16, 5)
    objs += [ts, buf] + [buf.materialize(i) for i in range(16)]
    assert xtim.live_state_bytes() > live0
    del objs, s, c, smp, ts, buf
    gc.collect()
    _xtim._release_thread_state_scratch()
    assert xtim.live_state_bytes() == live0


# ---------------------------------------------------------------------------------------------
# 3.1.10: the budget gaps of 3.1.9 — thread scratch, Python exports, automatic release, a refusal
# in the middle of an operation, the sigma heap block for r = 25..30.
# ---------------------------------------------------------------------------------------------
CULT = (ROOT / "tests" / "data" / "cultivation_d3_faithful.stim").read_text()
SWITCH = (ROOT / "tests" / "data" / "code_switching_faithful.stim").read_text()


def test_measurement_scratch_is_charged_and_released():
    """(a) The chi-scaled measurement scratch (packed sign rows, per-branch Pauli images, partner
    hash, ...) is charged while it is held. TLC: it is owned by the exact engine for ONE compile / run,
    not by the thread, so nothing of it is retained once the compile or a run returns (live after
    sample() == live after compile). Its size (> 100 B per branch) and its charge-before-growth are
    pinned in C++ (test_tlc_scratch: test_scratch_charge)."""
    res = _child(f"""
        smp = xtim.Circuit({BH12!r}).compile_detector_sampler(seed=3, engine="exact")
        live_c = xtim.live_state_bytes()
        smp.sample(10)
        print(json.dumps(dict(live_c=live_c, live_s=xtim.live_state_bytes())))
    """)
    assert res["live_s"] == res["live_c"], res


def test_measurement_scratch_not_retained_across_live_samplers():
    """(a, TLC) Scratch lives for one compile / run, not for the sampler's lifetime: k exact samplers
    held alive at once, each compiled and sampled in turn, hold exactly k times one sampler's states
    (a scratch held per live sampler would add k scratches)."""
    k = 4
    probe = _child(f"""
        smp = xtim.Circuit({BH12!r}).compile_detector_sampler(seed=3, engine="exact")
        print(json.dumps(dict(live=xtim.live_state_bytes())))
    """)
    res = _child(f"""
        held = []
        for i in range({k}):
            smp = xtim.Circuit({BH12!r}).compile_detector_sampler(seed=3 + i, engine="exact")
            smp.sample(10)
            held.append(smp)
        print(json.dumps(dict(live=xtim.live_state_bytes())))
    """)
    assert res["live"] == k * probe["live"], (res, probe)


def test_measurement_scratch_refused_before_allocating():
    """(a) A chi = 4096 state (build peak ~0.68 MB; the state itself 0.1 MB since S2k) plus its
    branch_frames export (0.41 MB): the Born/expectation kernel's scratch (~0.39 MB) no longer fits
    under a 0.8 MB budget and is refused before allocating, naming the scratch; nothing stays
    charged and the state stays usable."""
    res = _child(f"""
        from xtim import _xtim
        s = xtim.bare_state_of({BH12!r})
        bf = s.branch_frames()
        live0 = xtim.live_state_bytes()
        try:
            s.pauli_expectation([0, 1], [], 0)
            res = dict(refused=False)
        except xtim.XtimCapacityError as e:
            res = dict(refused=True, msg=str(e))
        res["same"] = xtim.live_state_bytes() == live0
        del bf
        live1 = xtim.live_state_bytes()
        res["after"] = abs(s.pauli_expectation([0, 1], [], 0)) <= 1.0
        res["released"] = xtim.live_state_bytes() == live1   # TLC: the call's scratch went with the call
        print(json.dumps(res))
    """, budget=800_000)
    assert res["refused"] and "Born-probability scratch" in res["msg"], res
    assert "Refused before allocating" in res["msg"] and res["same"], res
    assert res["after"] and res["released"], res


def test_branch_frames_arrays_are_charged_while_alive():
    """(b) branch_frames() exports chi x n numpy arrays: charged while any of them (or a view) is
    alive, released exactly when the last reference goes."""
    gc.collect()
    s = xtim.bare_state_of(BH12)
    chi, n, k = 1 << s.k, s.n, s.k
    live0 = xtim.live_state_bytes()
    bf = s.branch_frames()
    assert xtim.live_state_bytes() - live0 == chi * (k + 16 + 2 * n + 1)
    del bf["branch_x"]
    assert xtim.live_state_bytes() - live0 == chi * (k + 16 + n + 1)
    view = bf["coeff"][::2]
    del bf
    gc.collect()
    assert xtim.live_state_bytes() - live0 == chi * 16          # the view keeps coeff alive
    del view
    gc.collect()
    assert xtim.live_state_bytes() == live0


def test_branch_frames_export_refused_before_allocating():
    probe = _child(f"""
        s = xtim.bare_state_of({BH12!r})
        print(json.dumps(dict(live=xtim.live_state_bytes())))
    """)
    res = _child(f"""
        s = xtim.bare_state_of({BH12!r})
        pad = [s.copy() for _ in range(4)]          # 0.39 MB (S2k: 0.1 MB per state): the 0.41 MB export no longer fits
        live0 = xtim.live_state_bytes()
        try:
            s.branch_frames(); res = dict(refused=False)
        except xtim.XtimCapacityError as e:
            res = dict(refused=True, msg=str(e))
        res["same"] = xtim.live_state_bytes() == live0
        del pad
        res["ok_after"] = abs(s.pauli_expectation_x(0)) <= 1.0     # the state stays usable
        print(json.dumps(res))
    """, budget=800_000)                        # the 0.68 MB build peak fits
    assert res["refused"] and "branch_frames export" in res["msg"] and res["same"], res
    assert res["ok_after"], res


def test_live_total_returns_to_baseline_automatically():
    """(c) No explicit release: once every Python-held xtim object is dropped, the live total is
    back to its baseline (the calling thread's scratch is released with the last owner)."""
    res = _child(f"""
        import gc
        from xtim import _xtim
        live0 = xtim.live_state_bytes()
        objs = []
        s = xtim.bare_state_of({BH12!r})
        objs += [s, s.copy(), s.branch_frames()]
        c = xtim.Circuit({CULT!r})
        for eng in ("twirl", "exact"):
            smp = c.compile_detector_sampler(seed=3, engine=eng)
            smp.sample(200)
            objs.append(smp)
        e2 = xtim.Circuit({BH12!r}).compile_detector_sampler(seed=3, engine="exact")
        e2.sample(10); objs.append(e2)
        ts = _xtim.TwirlSampler("R 0 1 2\\nH 0\\nT 0\\nCX 0 1\\nCX 0 2\\nCX 1 2\\nM 2\\nDETECTOR rec[-1]\\n"
                                "OUTPUT_QUBITS out 0 1\\n", 0.0, "", 0, "")
        buf = ts.sample_barrier(16, 5)
        objs += [ts, buf] + [buf.materialize(i) for i in range(16)]
        peak, scr = xtim.live_state_bytes(), _xtim._thread_scratch_bytes()
        del objs, s, c, smp, e2, ts, buf
        gc.collect()
        mid = xtim.live_state_bytes()
        # a module-level call with no owner alive also leaves nothing behind
        xtim.Circuit({CULT!r}).detector_error_model_text(approximate_disjoint_errors=True)  # 3.1.12: per-shot sign laws
        print(json.dumps(dict(live0=live0, peak=peak, scr=scr, end=mid, after_dem=xtim.live_state_bytes(),
                              scr_end=_xtim._thread_scratch_bytes(), owners=_xtim._scratch_owner_count())))
    """)
    # (TLC: the framed engine's scratch is caller-owned, released with its compile / run / call; the
    # thread scratch that remains — the CanonicalStabSum oracle's slots — is released as before)
    assert res["peak"] > res["live0"], res
    assert res["end"] == res["live0"] and res["after_dem"] == res["live0"], res
    assert res["scr_end"] == 0 and res["owners"] == 0, res


def _compile_live(text, eng="exact"):
    return _child(f"""
        smp = xtim.Circuit({text!r}).compile_detector_sampler(seed=3, engine={eng!r})
        print(json.dumps(dict(live=xtim.live_state_bytes())))
    """)["live"]


def test_refusal_mid_sampling_invalidates_the_exact_sampler():
    """(d) cultivation-d3 on the exact engine grows its working states while sampling: with the
    budget at the post-compile total, sample() is refused mid-run; the sampler then marks itself
    invalid and every later call raises a clear error (not undefined results)."""
    budget = _compile_live(CULT)
    res = _child(f"""
        smp = xtim.Circuit({CULT!r}).compile_detector_sampler(seed=3, engine="exact")
        out = {{}}
        try:
            smp.sample(3000); out["first"] = "ok"
        except xtim.XtimCapacityError as e:
            out["first"] = str(e)
        try:
            smp.sample(5); out["second"] = "ok"
        except xtim.XtimCapacityError as e:
            out["second"] = str(e)
        print(json.dumps(out))
    """, budget=budget + 512)
    assert "Refused before allocating" in res["first"] and "now INVALID" in res["first"], res
    assert "is no longer usable" in res["second"], res


def test_refusal_while_compiling_is_a_capacity_error_not_a_rejection():
    """(d) A budget refusal inside the exact sampler's setup surfaces as XtimCapacityError; 3.1.9's
    setup guard turned it into an out-of-class rejection message."""
    budget = _compile_live(SWITCH)          # the setup's peak is above its post-compile total
    res = _child(f"""
        try:
            xtim.Circuit({SWITCH!r}).compile_detector_sampler(seed=3, engine="exact")
            out = dict(kind="ok")
        except Exception as e:
            out = dict(kind=type(e).__name__, msg=str(e)[:400])
        print(json.dumps(out))
    """, budget=budget)
    assert res["kind"] == "XtimCapacityError", res


def test_refused_state_measurement_invalidates_the_state():
    """(d) A FramedSuperposition whose measurement is refused marks itself invalid."""
    res = _child(f"""
        s = xtim.bare_state_of({BH12!r})
        bf = s.branch_frames()                     # 0.41 MB pad
        pad = [s.copy() for _ in range(2)]         # + 0.2 MB (S2k: 0.1 MB per state): a 0.1 MB clone no longer fits
        out = dict(refused=None)
        for q in range(s.n):
            for b in (2, 0, 1):
                try:
                    s.measure_pauli(b, q, 0.3)
                except xtim.XtimCapacityError as e:
                    out["refused"] = str(e); break
            if out["refused"]:
                break
        del bf, pad
        try:
            s.pauli_expectation_x(0); out["after"] = "ok"
        except xtim.XtimCapacityError as e:
            out["after"] = str(e)
        print(json.dumps(out))
    """, budget=800_000)
    assert res["refused"] and "now INVALID" in res["refused"], res
    assert "is no longer usable" in res["after"], res


def test_refused_materialize_leaves_the_buffer_valid():
    """(d) BarrierBuffer.materialize is read-only on the buffer: a refused materialize leaves it
    valid; once memory is freed the same shot materializes to exactly the unconstrained state.
    J-b1 F (fixplan-F-materialize-c0hat r2): the layout-3 materialize returns the FramedSuperposition handle on the
    gate-level route (a bare copy, the Clifford on its frame, the records forced in place), so the r6.6 pin below
    holds again unchanged: the pre-r2 F re-pin (OD-2026-10-20-materialize-peak, a 6.875-state transient of the C0̂
    route's bare rebuild) is withdrawn."""
    qs = " ".join(map(str, range(12)))
    text = (f"RX {qs}\nT {qs}\nCX 0 1 2 3 4 5 6 7 8 9 10 11\nR 12\nM 12\nDETECTOR rec[-1]\n"
            f"OUTPUT_QUBITS out {qs}\n")              # the chi = 4096 magic register is the output
    code = f"""
        import hashlib, numpy as np, gc
        from xtim import _xtim
        ts = _xtim.TwirlSampler({text!r}, 1.0, "", 0, "")
        buf = ts.sample_barrier(8, 5)
        pad = [buf.materialize(i) for i in (0, 1, 2, 4, 5, 6)]   # six states (S2k: 0.1 MB each): the build peak stays under
        gc.collect()
        live1 = xtim.live_state_bytes()
        out = dict(live1=live1, pad=None)
        try:
            buf.materialize(3); out["first"] = "ok"
        except xtim.XtimCapacityError as e:
            out["first"] = str(e)
        del pad
        gc.collect()
        before = xtim.live_state_bytes()
        st = buf.materialize(3)
        out["m"] = xtim.live_state_bytes() - before        # what one materialize adds
        bf = st.branch_frames()
        out["h"] = hashlib.sha256(b"".join(np.ascontiguousarray(bf[k]).tobytes()
                                           for k in sorted(bf) if k != "n")).hexdigest()
        print(json.dumps(out))
    """
    ref = _child(code)
    assert ref["first"] == "ok"
    # live1 holds the pad (six materialized states of m bytes each); S2b's forced-projection
    # materialize peaks at ONE state (the bare copy it projects in place), so with the pad it needs
    # m more than live1 and is refused, and without the pad it fits
    m = ref["m"]
    res = _child(code, budget=ref["live1"] + m - 1)
    assert "Refused before allocating" in res["first"], (res, ref)
    assert res["h"] == ref["h"], (res, ref)
    # r6.6 (owner-signed 2026-10-17): the CHARGED peak is exactly one state — at live1 + m the first
    # materialize fits (a regression back to a two-state peak is red here); this is the budget
    # accounting, not RSS
    ok = _child(code, budget=ref["live1"] + m)
    assert ok["first"] == "ok", (ok, ref)
    assert ok["h"] == ref["h"], (ok, ref)


def test_sigma_heap_block_charged_for_every_supported_rank():
    """(e) A FramedSuperposition entry's sigma is a packed 64-bit key since S2k (no heap block of
    its own): the per-entry charge is exactly sizeof(entry) = 24 B (a copy of a chi = 4096 state
    charges 4096 x 24 bytes; 3.1.10..S2a charged sizeof(entry) + 48 = 88 B for the byte-vector key).
    A CanonicalStabSum branch still carries a byte sigma (r <= 30 bytes: one glibc chunk of 32 B for
    r <= 24, 48 B for r = 25..40) and is still charged sizeof(branch) + 48. Expectation reads on the
    copy keep no key index: nothing but the thread scratch stays charged after them."""
    s = xtim.bare_state_of(BH12)
    live0 = xtim.live_state_bytes()
    c = s.copy()
    assert xtim.live_state_bytes() - live0 == (1 << s.k) * 24
    scr0 = _xtim._thread_scratch_bytes()
    for q in range(s.n):
        c.pauli_expectation_x(q)
    # what stays charged beyond the copy is the (separately accounted) thread scratch only
    assert xtim.live_state_bytes() - live0 - (_xtim._thread_scratch_bytes() - scr0) == (1 << s.k) * 24
    del c
