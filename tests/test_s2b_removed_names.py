"""S2b loudness gate (design: every removed or renamed name raises): the S2a coin-record surface is
gone — AttributeError for the removed methods, KeyError for the removed keys — and the renamed
accessors are the only spelling. Mutation: re-adding ``coins`` as an alias of ``outcomes`` turns
test_removed_attributes red.

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).
"""
import numpy as np
import pytest

import xtim
import xtim._xtim as _x
from xtim import port
from _public import public_engine  # noqa: E402

P2 = ("RX 0\nT 0\nRX 2\nT 2\nX_ERROR(0.2) 1\nM 1\nMY 0\nMX 2\nDETECTOR rec[-3]\n"
      "OBSERVABLE_INCLUDE(0) rec[-2]\nDECISION(0) rec[-1]\n")
SEG = "RX 0\nT 0\nX_ERROR(0.2) 1\nM 1\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n"


@pytest.fixture(scope="module")
def objs():
    s = public_engine(P2, 1.0, "", 0, True)          # J-b2 J1b: the PUBLIC sampler (the names stay gone there)
    b = s.sample_barrier(20, seed=1)
    seg = port.compile(SEG, port.Consume(dets=True, groups=True, residual_law=True))
    res = seg.run(shots=50, seed=2)
    return s, b, seg, res


@pytest.mark.parametrize("which,name", [
    ("b", "coins"), ("b", "coin_layout"), ("s", "coin_layout"), ("seg", "coin_layout"),
    ("res", "group_coins"),
])
def test_removed_attributes(objs, which, name):
    s, b, seg, res = objs
    obj = {"s": s, "b": b, "seg": seg, "res": res}[which]
    with pytest.raises(AttributeError):
        getattr(obj, name)
    assert not hasattr(_x, "CoinLayout")


def test_renamed_accessors_exist(objs):
    s, b, seg, res = objs
    assert len(b.outcomes(0)) >= 0 and isinstance(res.group_outcomes, tuple)
    assert np.asarray(b.record_outcomes(0)).shape == (len(s.record_operators()),)


@pytest.mark.parametrize("key", ["coin_born_obs_at", "coin_born_dec_at", "coin_len"])
def test_removed_plan_structure_keys(objs, key):
    s, b, seg, res = objs
    with pytest.raises(KeyError):
        b.plan_structure(0)[key]


@pytest.mark.parametrize("key", ["classes", "has_born_obs", "nborn", "born_dec_index", "fair", "kernel",
                                 "born_obs", "born_dec", "prefix", "born_obs_at", "born_dec_at", "len"])
def test_removed_record_layout_v1_keys(objs, key):
    s, b, seg, res = objs
    assert b.record_layout()["layout"] == s.record_layout()["layout"] == 3
    with pytest.raises(KeyError):
        b.record_layout()[key]
    with pytest.raises(KeyError):
        s.record_layout()[key]


def test_removed_frames_and_report_keys(objs):
    s, b, seg, res = objs
    F = b.frames()                                   # layout 3: no allow_* flags (nothing is refused)
    with pytest.raises(KeyError):
        F["born_obs_raw"]
    rep = s.record_report()
    with pytest.raises(KeyError):
        rep["memo_nodes"]
    for k in ("exact_born_obs_kappa_pos", "exact_born_dec_kappa_pos", "exact_obs_guard",
              "exact_chain_guard"):
        with pytest.raises(KeyError):
            rep["route_counts"][k]
    with pytest.raises(KeyError):                 # r6.5: the production counter is gone
        rep["sigma_aux"]["channel_violations"]


# Code review NITs (r6.8): no docstring describes a removed mechanism as live. A docstring may name a
# removed spelling only to say it was removed or renamed ("renamed from coins", "removed").
_STALE = ["twirl_collapse", "kernel-chain", "kernel chain", "DEC word", "OBS word 0 … DEC",
          "coin_layout()", ".coins(", "born_obs_raw", "fair coins then", "chain outcomes"]


def _docstrings():
    import inspect
    import xtim.frames
    import xtim.port
    import xtim.record_fidelity
    import xtim.twirl
    seen = []
    for mod in (_x, xtim, xtim.frames, xtim.port, xtim.record_fidelity, xtim.twirl):
        for name, obj in vars(mod).items():
            if name.startswith("__"):
                continue
            doc = inspect.getdoc(obj) if (inspect.isclass(obj) or callable(obj)) else None
            if doc:
                seen.append((f"{mod.__name__}.{name}", doc))
            if inspect.isclass(obj) and getattr(obj, "__module__", "").startswith("xtim"):
                for an, av in vars(obj).items():
                    d = getattr(av, "__doc__", None)
                    if isinstance(d, str) and d and not an.startswith("__"):
                        seen.append((f"{mod.__name__}.{name}.{an}", d))
    return seen


def test_docstrings_name_no_removed_api():
    bad = []
    for where, doc in _docstrings():
        for ln in doc.splitlines():
            low = ln.lower()
            if "removed" in low or "renamed" in low or "deleted" in low or "retired" in low:
                continue
            for tok in _STALE:
                if tok in ln:
                    bad.append((where, tok, ln.strip()[:120]))
    assert not bad, bad
