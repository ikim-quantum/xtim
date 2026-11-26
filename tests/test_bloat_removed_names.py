"""bloat-cut (owner 2026-11-01): removed options and names fail loudly (TypeError / AttributeError / ImportError),
never silently ignored.

  C  the TWPL disk plan cache: compile_twirl_sampler(disk_cache=), compile_twirl_sampler_from_ideal_state(disk_cache=),
     _xtim.TwirlSampler(disk_cache=, disk_cache_auto_dir=) and its old positional form, port.compile(plan_cache=)
     (tests/test_port_groups.py PG-09), xtim.twirl.twirl_cache_dir, record_report()["disk_path"], the test API's
     compile_family_sampler(disk_path=) / take_disk_events.
  D  unused bindings and back-compat aliases: FramedSuperposition.port_signature / port_signature_struct,
     _xtim.bare_state_memo_clear / bare_state_memo_set_caps, the _bare_state_of / _input_state_of /
     _port_contract_of_state aliases (_xtim and xtim / xtim.twirl).
"""
import warnings

import pytest

import xtim
import xtim.twirl
from xtim import _xtim as _x

pytestmark = pytest.mark.filterwarnings("ignore")

DET = "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n"
PORT = "INPUT_QUBITS in 0\nR 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1]\n"


# ── C: the TWPL disk plan cache ───────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("value", [True, "/tmp/never.twpl"])
def test_c_compile_twirl_sampler_disk_cache_refused(value):
    with pytest.raises(TypeError):
        xtim.compile_twirl_sampler(DET, disk_cache=value)


def test_c_from_ideal_state_disk_cache_refused():
    ideal = xtim.input_state_of("R 0\nH 0\nOUTPUT_QUBITS out 0\n")
    with pytest.raises(TypeError):
        xtim.compile_twirl_sampler_from_ideal_state(PORT, ideal_state=ideal, disk_cache=True)


def test_c_engine_constructor_disk_args_refused():
    with pytest.raises(TypeError):
        _x.TwirlSampler(DET, disk_cache="/tmp/never.twpl")
    with pytest.raises(TypeError):
        _x.TwirlSampler(DET, disk_cache_auto_dir="/tmp")
    with pytest.raises(TypeError):
        _x.TwirlSampler(DET, 1.0, "", 0)                     # the old positional form (disk_cache third)
    ideal = _x.ideal_state_of_state(xtim.input_state_of("R 0\nH 0\nOUTPUT_QUBITS out 0\n").state, [0])
    with pytest.raises(TypeError):
        _x.TwirlSampler.from_ideal_state(ideal_state=ideal, circuit_text=PORT, disk_cache="/tmp/never.twpl")


def test_c_names_gone():
    assert not hasattr(xtim.twirl, "twirl_cache_dir")
    s = _x.TwirlSampler(DET, 1.0, 0)
    s.sample(10, seed=1)
    assert "disk_path" not in s.record_report()


def test_c_test_api_disk_surface_gone():
    from xtim import _xtim_testapi as TA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError):
            TA.compile_family_sampler(DET, selfcheck=0, disk_path="/tmp/never.twpl")
        s = TA.compile_family_sampler(DET, selfcheck=0)
    assert not hasattr(s, "take_disk_events")


# ── D: unused bindings and back-compat aliases ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["bare_state_memo_clear", "bare_state_memo_set_caps", "_bare_state_of",
                                  "_input_state_of", "_port_contract_of_state"])
def test_d_module_bindings_gone(name):
    assert not hasattr(_x, name)


@pytest.mark.parametrize("name", ["_bare_state_of", "_input_state_of"])
def test_d_python_aliases_gone(name):
    assert not hasattr(xtim, name) and not hasattr(xtim.twirl, name)
    with pytest.raises(ImportError):
        exec(f"from xtim import {name}", {})


@pytest.mark.parametrize("name", ["port_signature", "port_signature_struct"])
def test_d_state_methods_gone(name):
    st = xtim.bare_state_of(DET)
    assert not hasattr(st, name)
    assert hasattr(st, "port_orbit_operators")   # kept: adaptq calls it


def test_d_public_names_kept():
    st, w = _x.input_state_of("R 0\nH 0\nOUTPUT_QUBITS out 0\n")
    assert callable(_x.port_contract_of_state) and _x.port_contract_of_state(st, w)["product"]
    assert xtim.input_state_of is xtim.twirl.input_state_of and xtim.bare_state_of is xtim.twirl.bare_state_of


# ── review F1: parameters after p_factor are keyword-only (an old positional call cannot rebind silently) ─────
DEC = "RX 0\nT 0\nMX 0\nDECISION(0) rec[-1]\n" + DET
IF_TEXT = "INPUT_BITS b 1\nR 0\nIF b[0] {\nX 0\n}\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n"


@pytest.mark.parametrize("args", [(1.0, False, 0), (1.0, True, 500), (1.0, None, 0), (1.0, "/tmp/x.twpl", 0),
                                  (1.0, 0)])
def test_f1_compile_twirl_sampler_old_positional_forms_refused(args):
    # old (circuit_text, p_factor, disk_cache, selfcheck, ...): False/True would have rebound to selfcheck/decisions
    with pytest.raises(TypeError):
        xtim.compile_twirl_sampler(DEC, *args)
    s = xtim.compile_twirl_sampler(DEC, 1.0, selfcheck=0)      # the keyword form
    assert s.num_decisions == 1


def test_f1_partitioned_sampler_positional_refused():
    with pytest.raises(TypeError):
        xtim.twirl.PartitionedTwirlSampler(IF_TEXT, 1.0, None, 0)
    with pytest.raises(TypeError):
        xtim.twirl.PartitionedTwirlSampler(IF_TEXT, 1.0, 0)
    assert isinstance(xtim.compile_twirl_sampler(IF_TEXT, 1.0, selfcheck=0), xtim.twirl.PartitionedTwirlSampler)
