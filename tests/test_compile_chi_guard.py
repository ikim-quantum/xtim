"""Bare-state branch-rank guards (3.1.8 fix of the 3.1.7 `1 << r` overflow).

In 3.1.7 a bare state whose magic branch rank r exceeded 30 skipped FastTODD and then formed
`chi = 1 << r` unchecked: undefined behaviour that wraps mod 32 on x86. Switch-3 (bt27, raw rank
33) built a truncated chi = 2 state of norm^2 2^-32, which engine="auto" routed to the exact
engine with a "sampling stays exact" message: at p = 0 24 detectors fired 50% of the time.

Contract pinned here: a circuit is either sampled CORRECTLY (zero detector firings at p = 0) or
refused LOUDLY with a typed error naming the rank and the limit. Never a wrong state. Since the
FastTODD lift (word-vector residue masks + symbolic even-residue recovery) bt27 reduces to
chi = 2^9 and samples correctly; the all-T family keeps rank n after reduction and refuses.
"""
import pathlib

import numpy as np
import pytest

import xtim

DATA = pathlib.Path(__file__).parent / "data"


def _all_t(n: int) -> str:
    """|+>^n then T on every qubit: n independent magic columns, T-count n is optimal, so the
    branch rank stays n after any T-count reduction (chi = 2^n)."""
    qs = " ".join(map(str, range(n)))
    return f"RX {qs}\nT {qs}\nMX {qs}\n"


@pytest.mark.parametrize("n", [31, 33, 40])
def test_rank_above_limit_refuses_loudly(n):
    text = _all_t(n)
    with pytest.raises(xtim.XtimCapacityError) as ei:
        xtim.Circuit(text).compile_detector_sampler(seed=1, engine="exact")
    msg = str(ei.value)
    assert f"2^{n}" in msg and "2^30" in msg, msg
    # Typed as a MemoryError, NOT ValueError/RuntimeError: engine="auto" must not reroute it.
    assert isinstance(ei.value, MemoryError)
    assert not isinstance(ei.value, (ValueError, RuntimeError))


@pytest.mark.parametrize("n", [31, 33])
def test_rank_above_limit_refuses_on_every_path(n):
    text = _all_t(n)
    with pytest.raises(xtim.XtimCapacityError):
        xtim.Circuit(text).compile_detector_sampler(seed=1)          # auto
    with pytest.raises(xtim.XtimCapacityError):
        xtim.compile_twirl_sampler(text)
    with pytest.raises(xtim.XtimCapacityError):
        xtim.bare_state_of(text)


def test_rank_at_small_n_still_builds():
    # Positive control on the same family: rank 6, chi = 64 builds and samples.
    s = xtim.Circuit(_all_t(6)).compile_sampler(seed=3)
    m = np.asarray(s.sample(16))
    assert m.shape == (16, 6)


def test_bt27_p0_reduces_and_samples_correctly():
    """Switch-3 at p = 0: every detector is deterministic, so a correct state fires none.
    3.1.8 runs FastTODD at raw rank 33 (word-vector residue masks) and the symbolic even-residue
    recovery: the state reduces to chi = 2^9 and samples exactly, on BOTH engines."""
    text = (DATA / "merlin_switch3_bt27_p0.stim").read_text()
    c = xtim.Circuit(text)
    s = c.compile_detector_sampler(seed=12345)
    assert s.engine_report()["engine"] == "twirl", s.engine_report()
    d = np.asarray(s.sample(500))
    assert d.shape == (500, 72)
    assert int(d.sum()) == 0, f"{int(d.sum())} detector firings at p = 0 (wrong bare state)"
    e = np.asarray(c.compile_detector_sampler(seed=7, engine="exact").sample(200))
    assert int(e.sum()) == 0, f"exact engine: {int(e.sum())} detector firings at p = 0"
    bf = xtim.bare_state_of(text).branch_frames()
    assert len(bf["coeff"]) == 512
    assert abs(float(np.sum(np.abs(bf["coeff"]) ** 2)) - 1.0) < 1e-9


def test_reference_fallback_refuses_a_non_normalised_state(monkeypatch, tmp_path):
    """3.1.7 downgraded the reference compile's "not a state" verdict to the benign
    "sampling stays exact" fallback — onto the SAME (wrong) deduced construction."""
    import xtim.reference as R

    def fake_compile(text, source="xtim-circuit"):
        raise xtim.XtimReferenceError(
            "reference compile failed: round-trip load failed: coefficient norm^2 = "
            "2.3283064365387082e-10 != 1 (not a state)")

    monkeypatch.setattr(R, "compile_reference", fake_compile)
    monkeypatch.setattr(xtim, "cache_dir", str(tmp_path))
    with pytest.raises(xtim.XtimInvariantError):
        R.resolve_reference("R 0\nM 0\n")
