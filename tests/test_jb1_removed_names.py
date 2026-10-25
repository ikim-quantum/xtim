"""J-b1 F loudness gate (plan §3.7, §4.1, r3.4): every name the flip removed or renamed raises BY NAME —
AttributeError for a removed attribute, TypeError for a removed keyword / positional form, KeyError (absence) for a
removed key, ValueError for engine="auto" and sample_barrier(compact=False). The renamed spellings are the only ones.
Pattern: tests/test_s2b_removed_names.py.
"""
import numpy as np
import pytest

import xtim
import xtim._xtim as _x
import xtim.frames
from xtim import port

pytestmark = pytest.mark.filterwarnings("ignore")

P2 = ("RX 0\nT 0\nRX 2\nT 2\nX_ERROR(0.2) 1\nM 1\nMY 0\nMX 2\nDETECTOR rec[-3]\n"
      "OBSERVABLE_INCLUDE(0) rec[-2]\nDECISION(0) rec[-1]\n")
SEG = "RX 0\nT 0\nX_ERROR(0.2) 1\nM 1\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n"
IF_TEXT = "INPUT_BITS b 1\nR 0\nIF b[0] {\nX 0\n}\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n"
PORT = "INPUT_QUBITS in 0\nR 1\nX_ERROR(0.1) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"


@pytest.fixture(scope="module")
def objs():
    s = _x.TwirlSampler(P2, 1.0, "", 0)
    b = s.sample_barrier(20, seed=1)
    seg = port.compile(SEG, port.Consume(dets=True, groups=True, residual_law=True))
    res = seg.run(shots=50, seed=2)
    return s, b, seg, res


@pytest.mark.parametrize("which,name", [
    # sigma retired (D1)
    ("b", "sigmas"), ("b", "sig_bits"), ("b", "sigma"), ("s", "sig_bits"),
    ("res", "sig_bits"), ("res", "group_sigmas"), ("seg", "sigma_law"), ("seg", "sig_bits"), ("seg", "gw"),
    # the layout-3 record surface
    ("b", "plan_key"), ("res", "group_plans"), ("res", "group_exact"), ("res", "group_plans_or_none"),
    ("b", "exact_flags"), ("b", "frame_dressing"), ("b", "input_frame_dressing"), ("b", "input_frame_advance"),
    ("b", "record_tables_held"), ("res", "frame_dressing"), ("seg", "frame_dressing_report"),
    # channel_report and the legacy-only accessors (jb2 ruling C2)
    ("s", "channel_report"), ("s", "frame_dressing_report"), ("s", "record_channels"),
    ("s", "_plan_born_directions"),
])
def test_removed_attributes(objs, which, name):
    s, b, seg, res = objs
    obj = {"s": s, "b": b, "seg": seg, "res": res}[which]
    with pytest.raises(AttributeError):
        getattr(obj, name)


def test_removed_module_names():
    for mod, name in ((xtim, "compile_twirl_sampler_from_state"), (xtim, "_project_bare_onto_syndrome"),
                      (_x, "_project_bare_onto_syndrome"), (xtim.frames, "require_kappa_zero"),
                      (xtim.twirl, "compile_twirl_sampler_from_state"), (xtim.twirl, "_compose_input_stage_text")):
        with pytest.raises(AttributeError):
            getattr(mod, name)
    from xtim import _xtim_testapi as TA
    with pytest.raises(AttributeError):
        TA.compile_family_partitioned
    with pytest.raises(AttributeError):
        xtim.bare_state_of("R 0\nM 0\n").port_sigma_law


def test_removed_keywords_and_forms(objs):
    s, b, seg, res = objs
    with pytest.raises(TypeError):
        xtim.compile_twirl_sampler(P2, skip_refused_observables=True)
    with pytest.raises(TypeError):
        _x.TwirlSampler(P2, 1.0, "", 0, True)                # the old 5th positional (skip_refused_observables)
    st = xtim.bare_state_of(P2)
    with pytest.raises(TypeError):
        _x.TwirlSampler(st, P2)                              # the positional state constructors
    with pytest.raises(TypeError):
        _x.TwirlSampler(st, [0], P2)
    with pytest.raises(TypeError):
        _x.TwirlSampler.from_ideal_state(xtim.compile_twirl_sampler(SEG).ideal_state(), PORT)   # keyword-only
    with pytest.raises(TypeError):
        s.sample_barrier(4, seed=1, frame_dressing=True)
    with pytest.raises(ValueError, match="compact=False"):
        s.sample_barrier(4, seed=1, compact=False)
    with pytest.raises(TypeError):
        b.frames(allow_kappa=True)
    with pytest.raises(TypeError):
        port.Consume(dets=True, frame_dressing=True)
    with pytest.raises(TypeError):
        xtim.twirl.PartitionedTwirlSampler(IF_TEXT, 1.0, None, 0, False, _make_inner=None)
    inp = xtim.input_state_of("R 0\nX 0\nOUTPUT_QUBITS out 0\n")
    with pytest.raises(TypeError):
        xtim.compile_twirl_sampler_from_ideal_state(PORT, ideal_state=inp, inject=True)
    with pytest.raises(TypeError):
        xtim.compile_twirl_sampler_from_ideal_state(PORT, ideal_state=inp.state)   # no wire map: refused by name


_STRICT = "RX 0\nT 0\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n"


def test_record_fidelity_renamed(objs):
    s = _x.TwirlSampler(_STRICT, 1.0, "", 0)
    b = s.sample_barrier(20, seed=1)
    rf = xtim.RecordFidelity(s)
    with pytest.raises(AttributeError):
        rf.fidelities
    with pytest.raises(TypeError):
        rf.evaluate(b, flips=np.zeros((b.num_shots, 0), np.uint8))
    with pytest.raises(KeyError):
        rf.stats()["exact_shots"]
    assert rf.evaluate(b).shape == (b.num_shots,)


def test_engine_auto_removed():
    with pytest.raises(ValueError, match="engine='auto' was removed at xtim-engine-13"):
        xtim.Circuit(SEG).compile_detector_sampler(engine="auto")
    with pytest.raises(AttributeError):
        xtim.Circuit(SEG).compile_detector_sampler().channel_report


@pytest.mark.parametrize("key", ["r", "kappa", "fallback", "coin_masks", "kernel_masks", "kernel_pre", "kernel_base",
                                 "kernel_foldable", "kernel_xz", "kernel_phase", "prefix_xz"])
def test_removed_plan_structure_keys(objs, key):
    s, b, seg, res = objs
    with pytest.raises(KeyError):
        b.plan_structure(0)[key]
    with pytest.raises(KeyError):
        s.plan_structure(b.plan_content(0)[0])[key]


@pytest.mark.parametrize("key", ["r", "kappa", "fallback", "plan_id", "exact"])
def test_removed_frames_fields(objs, key):
    s, b, seg, res = objs
    with pytest.raises(KeyError):
        b.frames()[key]


def test_removed_report_keys(objs):
    s, b, seg, res = objs
    s.sample(100, seed=3)
    rep = s.record_report()
    assert not [k for k in rep["route_counts"] if k.startswith("exact")]
    for key in ("ppr_plans", "ppr_plan_misses", "sigma_aux", "frame_dressed_shots", "exact_shots"):
        with pytest.raises(KeyError):
            rep[key]
    for key in ("ops_plane", "ops_pivot_reduce", "ops_greduce", "ops_echelon"):
        with pytest.raises(KeyError):
            rep["rec"][key]
    assert "exact_y" not in str(s.record_layout())


def test_renamed_spellings_exist(objs):
    s, b, seg, res = objs
    assert callable(xtim.compile_twirl_sampler_from_ideal_state)
    assert callable(xtim.frames.require_kappa_rec_zero)
    assert isinstance(res.group_plan_content, tuple) and callable(b.plan_content)
    assert callable(xtim.RecordFidelity(_x.TwirlSampler(_STRICT, 1.0, "", 0)).evaluate)
