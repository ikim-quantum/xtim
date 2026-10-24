"""J-b1 F (fixplan-F-materialize-c0hat r2): the public value readers of the layout-3 BarrierBuffer.

Gates:
  * materialize(i) / state(i) return the extension's FramedSuperposition HANDLE (as b9c2e92), branch_frames(i) is
    materialize(i).branch_frames();
  * value: on cultivation_d5 (seed 7, shots 0-9) every shot's materialized state equals the legacy (b9c2e92 law)
    sampler's materialize on the same shots: pauli_expectations_all on the record-free random Paulis, 1e-12;
  * speed: cultivation_d5 seed 7, every shot of a 10-shot barrier, materialize and pauli_expectations_all, run in a
    child with a bounded timeout; each call < 50 ms (b9c2e92: 0.1-0.5 ms). The C0̂ route (MUT-MAT-C0HAT: the value
    readers back on C0̂) is RED here: shots 5 / 6 are refused by C0̂'s capacity bound (or, without it, never finish);
  * capacity: the C0̂ amplitude map (the frame-preserving branch-export route; test API materialize(tableau=False))
    refuses by name, BEFORE any work, a plan whose support bound c * 2^d (d = the rank of the term images' X-parts, c <= chi the cosets of their span met by the bare's keys) exceeds the state budget — never a hang.
The ledger twins are api/materialize_{d5,fac} and api/pe_all_{d5,fac} (HARD, <= +5% vs b9c2e92).
"""
import json
import os
import subprocess
import sys
import textwrap
import time
import warnings

import numpy as np
import pytest

import xtim
from xtim import _xtim

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D5 = os.path.join(ROOT, "tests", "data", "cultivation_d5.stim")


def _prod(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xtim.compile_twirl_sampler(text, selfcheck=0)


def test_materialize_returns_the_handle():
    s = _prod(open(D5).read())
    b = s.sample_barrier(4, seed=7)
    st = b.materialize(1)
    assert type(st) is _xtim.FramedSuperposition
    assert type(b.state(1)) is _xtim.FramedSuperposition
    bf, ref = b.branch_frames(1), st.branch_frames()
    assert sorted(bf) == sorted(ref)
    for k in ref:
        assert np.array_equal(np.asarray(bf[k]), np.asarray(ref[k])), k
    with pytest.raises(IndexError):
        b.materialize(4)


def _paulis(n, rng, k=24):
    out = []
    for _ in range(k):
        x = np.flatnonzero(rng.random(n) < 0.02).tolist()
        z = np.flatnonzero(rng.random(n) < 0.02).tolist()
        out.append((x, z, len(set(x) & set(z)) % 2))
    out.append(([], [0], 0))
    return out


def test_cult_d5_materialize_speed_bounded():
    code = textwrap.dedent(f"""
        import json, time, warnings
        warnings.simplefilter("ignore")
        import xtim
        s = xtim.compile_twirl_sampler(open({D5!r}).read(), selfcheck=0)
        b = s.sample_barrier(10, seed=7)
        worst = 0.0
        for i in range(10):
            for _ in range(2):
                t = time.perf_counter(); b.materialize(i); worst = max(worst, time.perf_counter() - t)
        t = time.perf_counter(); b.pauli_expectations_all([([], [0], 0)]); pe = (time.perf_counter() - t) / 10
        print(json.dumps(dict(worst=worst, pe=pe)))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["worst"] < 0.05 and r["pe"] < 0.05, r


def test_c0hat_capacity_refusal_is_named_and_immediate():
    from xtim import _xtim_testapi as TA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = TA.compile_family_sampler(open(D5).read(), selfcheck=0)
    d, o, dc, R = s.sample_records(10, seed=7)
    t = time.perf_counter()
    with pytest.raises(xtim.XtimCapacityError, match=r"C0 hat.*support bound c \* 2\^d"):
        s.materialize(R, 5, False)
    assert time.perf_counter() - t < 5.0
    s.materialize(R, 5, True)                             # the gate-level route serves the same record


def test_cult_d5_values_equal_the_pinned_legacy_law():
    """J-b2 J1b (plan r2.1 J1b): the legacy law's values on the same shots and Paulis, pinned at b6826c1
    (tests/data/jb2_cult_d5_pe_all.json: the legacy sampler's pauli_expectations_all, equal to the public reader there
    to 1e-12), against the public reader and materialize(i) itself; the legacy sampler retires at J2."""
    import json
    pin = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jb2_cult_d5_pe_all.json")))
    r = np.asarray(pin["re"]) + 1j * np.asarray(pin["im"])
    text = open(D5).read()
    b = _prod(text).sample_barrier(10, seed=7)
    n = b.materialize(0).n
    P = _paulis(n, np.random.default_rng(3))
    a = np.asarray(b.pauli_expectations_all(P))
    assert a.shape == r.shape == (10, len(P))
    assert float(np.max(np.abs(a - r))) < 1e-12
    for i in range(10):
        st = b.materialize(i)
        for j, (x, z, _ph) in enumerate(P[:6]):
            assert abs(st.pauli_expectation_xz(x, z) - r[i, j].real) < 1e-12, (i, j)
