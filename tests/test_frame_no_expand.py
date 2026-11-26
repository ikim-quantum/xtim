"""S2-T4b pin: compile and DEM export never expand the bare state's compact Frame.

The bare-state build emits the frame in its compact shape (S2-T3). compile_detector_sampler / compile_twirl_sampler
(construction), the default sampling path (threads 1 and > 1) and DEM export (stim and xdem, include_expectations on
and off) read it only through compact-safe queries (Frame::certified_rows / conjugate / forward_image / sweep), so the
process-wide count of compact -> expanded transitions (_xtim._frame_expansions) does not move. Products that edit
the frame's rows (the exact framed sampler, ports, bare_state_of, RecordFidelity, the selfcheck window) still expand
once at their own construction; the last tests pin that the counter sees such an expansion.
S2-CLEAN (S2-T4b review M3): a bundled magic example (cultivation_d3_faithful: a compact frame with a nonempty block and
a π that is not the identity) runs every test too — the inverse-image cache's dense branch, its ski rental and the
expectation readers on a magic frame — and the selfcheck window of compile_twirl_sampler(...).sample (on
code_switching_faithful), the documented dense-frame product, moves the counter.
"""
import pytest
import stim

import xtim
from xtim import _xtim


def surface(d):
    return str(stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                      after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                      after_reset_flip_probability=0.001))


def chain(n):
    return "".join(f"R {q}\nX_ERROR(0.01) {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(n))


# A Clifford circuit with a PAULI_EXPECTATION column (include_expectations on / off differ on it), next to a chain.
SMALL = "R 0 1\nH 0\nCX 0 1\nZ_ERROR(0.2) 0\nMX 0\nPAULI_EXPECTATION(0) X1 rec[-1]\n"
EXPECT = chain(300) + SMALL.replace(" 0", " 900").replace(" 1", " 901").replace("X1", "X901")

MAGIC = open(xtim.example_path("cultivation_d3_faithful")).read()
CASES = {"surface_d5": surface(5), "surface_d11": surface(11), "chain_300": chain(300),
         "cultivation_d3_faithful": MAGIC}


def expansions(fn):
    a = _xtim._frame_expansions()
    out = fn()
    return _xtim._frame_expansions() - a, out


@pytest.mark.parametrize("name", sorted(CASES))
def test_compile_and_sample_do_not_expand(name):
    c = xtim.Circuit(CASES[name])
    d, s = expansions(lambda: c.compile_detector_sampler(seed=1))
    assert d == 0, f"{name}: compile_detector_sampler expanded the frame {d}x"
    d, _ = expansions(lambda: s.sample(1000))
    assert d == 0, f"{name}: sample(1000) expanded the frame {d}x"
    d, _ = expansions(lambda: s.sample(1000, threads=3))
    assert d == 0, f"{name}: sample(1000, threads=3) expanded the frame {d}x"


@pytest.mark.parametrize("name", sorted(CASES))
def test_compile_twirl_sampler_ctor_does_not_expand(name):
    d, _ = expansions(lambda: xtim.compile_twirl_sampler(CASES[name]))
    assert d == 0, f"{name}: compile_twirl_sampler expanded the frame {d}x"


@pytest.mark.parametrize("name", sorted(CASES) + ["expect"])
@pytest.mark.parametrize("fmt", ["stim", "xdem"])
@pytest.mark.parametrize("inc", [True, False])
def test_dem_does_not_expand(name, fmt, inc):
    c = xtim.Circuit(CASES.get(name, EXPECT))
    if fmt == "xdem" and inc and name == "expect":
        pytest.skip("xdem refuses include_expectations=True with a PAULI_EXPECTATION (documented)")
    d, _ = expansions(lambda: c.detector_error_model_text(format=fmt, include_expectations=inc))
    assert d == 0, f"{name}: DEM ({fmt}, include_expectations={inc}) expanded the frame {d}x"


def test_counter_sees_a_dense_product():
    # bare_state_of returns the dense frame by API contract (E): the counter moves, so a 0 above is not vacuous.
    d, _ = expansions(lambda: xtim.bare_state_of(CASES["surface_d5"]))
    assert d >= 1


def test_counter_sees_the_selfcheck_window():
    # the selfcheck window copies the bare state per shot (dense_bare_, "E"): on code_switching_faithful the first
    # sample() expands the shared frame on the home, with selfcheck=0 the same sample() does not (on
    # cultivation_d3_faithful no sample() expands it, measured, so it cannot show the window)
    text = open(xtim.example_path("code_switching_faithful")).read()
    s0 = xtim.compile_twirl_sampler(text, selfcheck=0)
    d, _ = expansions(lambda: s0.sample(200, seed=1))
    assert d == 0
    s = xtim.compile_twirl_sampler(text, selfcheck=50)
    d, _ = expansions(lambda: s.sample(200, seed=1))
    assert d >= 1
