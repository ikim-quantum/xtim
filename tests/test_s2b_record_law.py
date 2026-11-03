"""S2b gate L-1: the emitted joint law of every record column equals the exact law.

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

S2b emits every diagonal record (IN_GROUP channels, the Born observable, Born decisions) from the
plan's record table: base XOR coins XOR the record trie's Born directions XOR t(P). This gate samples
the engine on `sample()` and on `sample_barrier()` and compares the joint law of the emitted
columns (detectors and observables relative to the noiseless reference, decisions raw) with the
exact law of the ORIGINAL circuit text on dense states (tests/_record_oracle.py, no xtim code):
every cell at |z| < 5, and zero shots in a zero-probability cell.

Fixtures: the FX-born set (Born observables, Born decisions, unread directions, the A4 sign case,
the sigma-4 conditioning fixtures) and the S1 random family (diag / mirror / a4 flavours, no port).
"""
import functools
import os
import sys

import numpy as np
import pytest

import xtim

sys.path.insert(0, os.path.dirname(__file__))
from _record_law_x import emitted_law_x, oracle, random_circuit  # noqa: E402
from _record_oracle import emitted_columns, emitted_law, max_z  # noqa: E402
from test_s2b_record_table import FIXTURES  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

N = 40_000


def _unpack(a, k):
    a = np.asarray(a, np.uint8)
    if k == 0:
        return np.zeros((a.shape[0], 0), np.uint8)
    return np.unpackbits(a, axis=1, bitorder="little")[:, :k]


def _sample(s, n, seed):
    r = s.sample(n, seed=seed)
    if not hasattr(r, "decisions"):          # the engine object: (dets, obs[, decisions])
        return np.asarray(r[0]), np.asarray(r[1]), (np.asarray(r[2]) if len(r) > 2 else None)
    return np.asarray(r.dets), np.asarray(r.obs), np.asarray(r.decisions)


def _emitted(s, mode, seed):
    nd, no, nde = s.num_detectors, s.num_observables, s.num_decisions
    if mode == "sample":
        d, o, dec = _sample(s, N, seed)
    else:
        b = s.sample_barrier(N, seed=seed)
        d, o, dec = b.dets(), b.obs(), (b.decisions() if nde else None)
    cols = [_unpack(d, nd), _unpack(o, no)]
    if nde:
        cols.append(_unpack(dec, nde))
    return np.concatenate(cols, 1)


def _family_raw():
    """(name, raw generator text): x = the S1 random circuits without a port, f = family_circuit,
    d = dressed_circuit."""
    from test_s2b_record_table import dressed_circuit, family_circuit
    out = []
    for seed in range(300):
        text, frames = random_circuit(seed)
        if not frames and "INPUT_QUBITS" not in text:
            out.append(("x%d" % seed, text))
        out.append(("f%d" % seed, family_circuit(seed)))
    for seed in range(200):
        out.append(("d%d" % seed, dressed_circuit(seed)))
    return out


@functools.lru_cache(maxsize=1)
@functools.lru_cache(maxsize=1)
def _family():
    """The fuzz family: each raw circuit restricted to its S2b record family (code review MINOR-7: the out-of-family
    records were removed instead of the circuit being skipped). J-b2 J2: the restriction was defined by the legacy
    sampler's refusals; the restricted family is FROZEN as it was at b6826c1 (tests/data/jb2_s2b_family.json.gz,
    written by this function's pre-J2 form), so the gates below keep the identical corpus."""
    import gzip
    import json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jb2_s2b_family.json.gz")
    with gzip.open(path, "rt") as fh:
        return tuple((n, t) for n, t in json.load(fh)["family"])


# In-family floors (code review MINOR-7), fixed before measuring a mutant. Each family must keep at
# least FLOOR of its raw members after the restriction (no family skips vacuously), most members
# must carry a record beyond the appended deterministic detector, and the f / d families must carry
# negative-sign generator preparations (the fold bit beta' = 1, M-BETA). A failure means: fix the
# generators, never lower the bar.
_FLOOR = {"x": 0.9, "f": 0.95, "d": 0.95}
_NONTRIVIAL_FLOOR = 0.9
_SIGNED_FLOOR = {"f": 0.3, "d": 0.3}


def _signed(name, text):
    from test_s2b_record_table import sign_preps
    seed = int(name[1:])
    if name[0] == "f":
        n = len(text.splitlines()[0].split()) - 1               # the leading "RX 0 .. n-1"
        return bool(sign_preps(seed, range(n)))
    if name[0] == "d":
        nb = sum(1 for ln in text.splitlines() if ln.startswith("CCZ"))
        return bool(sign_preps(seed, [3 * b + 1 for b in range(nb)], p=0.5))
    return False


def _sigma_fixtures():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "s2b"))
    from test_sigma_joint_law import FIXTURES as SF
    return sorted((k, v) for k, v in SF.items() if k not in FIXTURES)


# ── J-b2 J1b: the PRODUCTION leg (the public sampler; plan r2.1 J1b). The same joint record law against the same
# independent dense oracles (_record_oracle / _record_law_x), on the same fixtures and the same fuzz family, on both
# surfaces. Production's barrier serves CH/PPR-class members too (the kind-2 route: no refusal), so every member runs
# both surfaces here. In-family membership reads the record order of the production compile (tests/_public.py).
from _public import public_sampler, rec_order  # noqa: E402


def _in_family_public(text):
    from test_s2b_record_table import KIND
    try:
        order = rec_order(text)
    except Exception:
        return False
    return sorted(order) == sorted((KIND[k], i) for k, i, _ in oracle(text).columns())


def _check_public(text, mode, seed=7, selfcheck=0, x=False):
    law = emitted_law_x(oracle(text)) if x else emitted_law(text)
    s = public_sampler(text, selfcheck=selfcheck)
    ncols = len(emitted_columns(text))
    M = _emitted(s, mode, seed)
    assert M.shape[1] == ncols, (M.shape, ncols)
    z, imp = max_z(M, law)
    assert imp == 0 and z < 5, (z, imp, law)
    return s


@pytest.mark.parametrize("mode", ["sample", "barrier"])
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_law_is_exact_public(name, mode):
    s = _check_public(FIXTURES[name], mode)
    rc = s.record_report()["route_counts"]
    assert not any(k.startswith("exact") for k in rc), rc       # production has no exact route


def test_family_floors_public():
    """MINOR-7 floors on the production compile (membership by its record order); the barrier floor is 1.0 on every
    family (production's barrier refuses no member)."""
    raw = {}
    for name, _ in _family_raw():
        raw[name[0]] = raw.get(name[0], 0) + 1
    got, nontriv, signed = {}, {}, {}
    for name, text in _family():
        k = name[0]
        assert _in_family_public(text), name
        got[k] = got.get(k, 0) + 1
        recs = [ln for ln in text.splitlines() if ln.startswith(("DETECTOR", "OBSERVABLE", "DECISION"))]
        nontriv[k] = nontriv.get(k, 0) + (len(recs) >= 2)
        signed[k] = signed.get(k, 0) + _signed(name, text)
    for k, fl in _FLOOR.items():
        assert raw.get(k, 0) > 0 and got.get(k, 0) >= fl * raw[k], (k, got.get(k), raw.get(k))
        assert nontriv[k] >= _NONTRIVIAL_FLOOR * got[k], (k, nontriv[k], got[k])
    for k, fl in _SIGNED_FLOOR.items():
        assert signed[k] >= fl * got[k], (k, signed[k], got[k])


@pytest.mark.parametrize("name,text", _family(), ids=lambda v: v if isinstance(v, str) and len(v) < 8 else "")
def test_random_family_law_is_exact_public(name, text):
    assert _in_family_public(text)
    seed = int(name[1:])
    _check_public(text, "sample", seed=seed + 1, x=True)
    _check_public(text, "barrier", seed=seed + 1, x=True)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_selfcheck_window_is_stream_neutral_public(name):
    """Q6 on production: the selfcheck window (the semantic oracle overlay) is stream-neutral."""
    text = FIXTURES[name]
    a = _sample(public_sampler(text, selfcheck=0), 5000, 3)
    b = _sample(public_sampler(text, selfcheck=5000), 5000, 3)
    for x, y in zip(a, b):
        assert (x is None and y is None) or np.array_equal(x, y)


@pytest.mark.parametrize("mode", ["sample", "barrier"])
@pytest.mark.parametrize("name,text", _sigma_fixtures(), ids=lambda v: v if isinstance(v, str) and len(v) < 24 else "")
def test_sigma_fixture_record_law_is_exact_public(name, text, mode):
    """The sigma-4 fixture set's record law on production (the fixtures, not sigma: no sigma surface since F)."""
    if not _in_family_public(text):
        pytest.skip("columns outside the S2b record family")
    _check_public(text, mode, x=True)
