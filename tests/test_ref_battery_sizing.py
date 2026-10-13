"""perf/ref-compile-battery: the work-sized noiseless-invariants battery + the failed-TreePlan memo.

* Verification never costs more than compilation: clamp(W_compile / w1, 1, ceiling) shots. Wherever
  noiseless shots replay a TreePlan (w1 = 0) the battery is the former FIXED one — 500 at compile
  (ref_compile_text), the caller's 200 at a cache hit (ref_verify_cheap) — pinned on every bundled
  example. The factory class (no TreePlan, live-block shots, w1 > 0) is pinned in CI time on a
  4-qubit circuit at QEC_TREEPLAN_CAP=2 (w1 = 42, W_compile = 214 => 5 shots).
* A failed TreePlan extraction is recorded once and never repeated for the same block; the R>0
  channel rebuild after it is SKIPPED (named in engine_report / plan_report) — and where the old
  rebuild refused on a tight state budget, the circuit now samples exactly (pinned below).
(The rule, the one-sequence continuation, the wobble check, the work credit and the memo key are
mutation-tested in cpp/tests/test_ref_battery.cpp.)
"""
import glob
import os
import subprocess
import sys
import textwrap

import pytest

import xtim
from xtim import _xtim

EXAMPLES = sorted(glob.glob(os.path.join(os.path.dirname(xtim.__file__), "..", "examples", "*.stim")))


@pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.basename(p) for p in EXAMPLES])
def test_cheap_circuits_keep_the_full_battery(path):
    text = open(path).read()
    r = _xtim.ref_compile_text(text)
    assert r["ok"], r["error"]
    assert r["invariants_shots"] == 500
    assert r["invariants_ceiling"] == 500
    v = _xtim.ref_verify_cheap(text, r["ref_text"])
    assert v["ok"], v["error"]
    assert v["invariants_shots"] == 200
    assert v["invariants_ceiling"] == 200


@pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.basename(p) for p in EXAMPLES])
def test_full_battery_is_silent(path, tmp_path, monkeypatch):
    """The XtimVerificationWarning fires only for a reduced battery: every bundled example runs
    the full 500 / 200 (w1 = 0) through the Python compile paths — cache miss and cache hit —
    and must emit none."""
    import warnings
    monkeypatch.setattr(xtim, "cache_dir", str(tmp_path))
    text = open(path).read()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(2):   # miss (compile battery + store), then hit (cheap verify)
            xtim.Circuit(text)._resolve_reference()
    assert not [w for w in rec if issubclass(w.category, xtim.XtimVerificationWarning)], \
        [str(w.message) for w in rec]


M3 = ("R 0 1 2 3\nH 0 1 2 3\nT 0 1 2\nCX 0 1 1 2 2 3\nH 0 1 2\nM 0 1 2\n"
      "DETECTOR rec[-1]\nPAULI_EXPECTATION(0) X3\n")


def _run(code, **env_over):
    env = dict(os.environ)
    env.update(env_over)
    env["XTIM_QUIET"] = "1"
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True,
                          text=True, env=env, check=True).stdout.split()


def test_factory_class_battery_is_work_sized():
    out = _run(f"""
        from xtim import _xtim as X
        r = X.ref_compile_text({M3!r})
        v = X.ref_verify_cheap({M3!r}, r["ref_text"])
        print(r["ok"], r["invariants_shot_work"], r["invariants_compile_work"],
              r["invariants_sized_shots"], r["invariants_shots"] >= r["invariants_sized_shots"],
              v["ok"], v["invariants_sized_shots"])
    """, QEC_TREEPLAN_CAP="2")
    assert out == ["True", "42", "214", "5", "True", "True", "5"], out


def test_reduced_battery_warns_once_per_compile_with_its_numbers():
    """3.1.13 owner rule ("put out a warning only"): a battery that runs fewer shots than its
    ceiling emits ONE XtimVerificationWarning per compile through the Python compile paths —
    shots run vs ceiling, W_compile and w1, and the (1-q)^N miss bound. The cap-2 circuit
    (w1 = 42, W_compile = 214 => ns = 5) at a reference compile (ceiling 500) and at the cache-hit
    verify (ceiling 200); a second compile of the same Circuit object reuses its resolved
    reference and stays silent."""
    out = _run(f"""
        import json, tempfile, warnings
        import xtim
        from xtim import _xtim as X
        r = X.ref_compile_text({M3!r})
        xtim.cache_dir = tempfile.mkdtemp()
        res = []
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            c = xtim.Circuit({M3!r})
            c.compile_detector_sampler()          # miss: compile battery
            c.compile_detector_sampler()          # same object: memoised, no new battery
            res.append([str(w.message) for w in rec
                        if issubclass(w.category, xtim.XtimVerificationWarning)])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            xtim.Circuit({M3!r}).compile_detector_sampler()   # hit: cheap verify
            res.append([str(w.message) for w in rec
                        if issubclass(w.category, xtim.XtimVerificationWarning)])
        v = X.ref_verify_cheap({M3!r}, r["ref_text"])
        print(json.dumps([res, r["invariants_shots"], v["invariants_shots"]]))
    """, QEC_TREEPLAN_CAP="2")
    import json
    (miss, hit), n_compile, n_verify = json.loads(" ".join(out))
    assert 5 <= n_compile < 500 and 5 <= n_verify < 200
    assert len(miss) == 1 and len(hit) == 1, (miss, hit)
    for msg, n, ceiling, where in ((miss[0], n_compile, 500, "compile"),
                                   (hit[0], n_verify, 200, "cache-hit verify")):
        assert f"xtim reference {where}:" in msg
        assert f"ran {n} of {ceiling} shots" in msg
        assert "W_compile = 214, w1 = 42 work units" in msg
        assert f"clamp(W_compile / w1, 1, {ceiling}) = 5" in msg
        assert f"missed with probability (1-q)^{n}" in msg
        assert f"(full battery: (1-q)^{ceiling})" in msg


def test_error_filter_still_caches_the_reference():
    """Under warnings.simplefilter('error', XtimVerificationWarning) the cache MISS stores the
    (valid) reference before the warning propagates, so the next compile is a cache HIT (whose
    warning also propagates) — never a re-compile on every call."""
    out = _run(f"""
        import glob, os, tempfile, warnings
        import xtim
        xtim.cache_dir = d = tempfile.mkdtemp()
        warnings.simplefilter("error", xtim.XtimVerificationWarning)
        res = []
        for _ in range(2):
            try:
                xtim.Circuit({M3!r}).compile_detector_sampler()
                res.append("none")
            except xtim.XtimVerificationWarning as e:
                res.append(str(e).split(":")[0].replace(" ", "_"))
            res.append(len(glob.glob(os.path.join(d, "*.ref"))))
        print(*res)
    """, QEC_TREEPLAN_CAP="2")
    assert out == ["xtim_reference_compile", "1", "xtim_reference_cache-hit_verify", "1"], out


# The reviewer's repro (perf/ref-compile-battery review I-2): compile + sample at cap 1, then
# rebind at cap 2 — the cap-1 program and its samples are still live while the cap-2 program
# compiles, so ~11.7 KB of the 12000-byte budget is in use. 3.1.12 refused the cap-2 compile there
# (XtimCapacityError: its channel rebuild captured leaf states of a tree that is then discarded).
_SEQ = """
    import hashlib, os
    from xtim import _xtim as X
    H = lambda r: hashlib.sha256(repr(sorted((k, bytes(memoryview(v)) if hasattr(v, "tobytes")
                                              else v) for k, v in r.items())).encode()).hexdigest()[:10]
    out = []
    for cap in ("1", "2"):
        os.environ["QEC_TREEPLAN_CAP"] = cap
        p = X.CompiledProgram({t!r}, "")
        r = p.sample(50, 3)
        out += [H(r), p.plan_report()["expectation_channel"].split(":")[0]]
    print(*out)
"""


def test_tight_budget_channel_skip_samples_exactly():
    tight = _run(_SEQ.format(t=M3), XTIM_MAX_STATE_BYTES="12000")
    roomy = _run(_SEQ.format(t=M3), XTIM_MAX_STATE_BYTES="40000")
    assert tight == roomy == ["fb343e5314", "skipped", "fb343e5314", "skipped"], (tight, roomy)


def test_failed_extraction_is_memoised_and_byte_identical():
    code = textwrap.dedent("""
        import hashlib, os
        os.environ["QEC_TREEPLAN_CAP"] = "2"
        os.environ["XTIM_QUIET"] = "1"
        from xtim import _xtim as X
        T = ("R 0 1 2 3\\nH 0 1 2 3\\nT 0 1 2\\nCX 0 1 1 2 2 3\\nH 0 1 2\\nM 0 1 2\\n"
             "DETECTOR rec[-1]\\nPAULI_EXPECTATION(0) X3\\n")
        def run():
            p = X.CompiledProgram(T, "")
            r = p.sample(200, 11)
            return hashlib.sha256(repr(sorted((k, bytes(memoryview(v)) if hasattr(v, "tobytes") else v)
                                              for k, v in r.items())).encode()).hexdigest()
        X._tree_fail_memo_clear()
        a = run(); s1 = X._tree_fail_memo_stats()
        b = run(); s2 = X._tree_fail_memo_stats()
        X._tree_fail_memo_clear()
        c = run()
        print(a == b == c, s1["stores"], s1["misses"], s1["hits"], s2["hits"], s2["misses"])
    """)
    env = dict(os.environ)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         check=True).stdout.split()
    # run 1: zero-error extraction fails + stored, channel rebuild skipped (no memo access);
    # run 2: answered by the memo (1 hit); run 3 after clear: fresh again — same bytes throughout.
    assert out == ["True", "1", "1", "0", "1", "1"], out
