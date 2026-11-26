"""IF blocks on INPUT_BITS are refused BY NAME on every surface that receives no bit values.

Owner rule: exact or refuse by name. An IF body runs for a shot iff its INPUT_BITS bit matches; a
surface that takes no bit values cannot evaluate it, so it must raise a ValueError naming the IF
condition (never run every body unconditionally, never fail with an internal error first). The
empty-body case (``IF d[0] { TICK }``) is refused the same way.
"""
import pytest

import xtim
from xtim import _xtim
from xtim._exact_scope import explicit_exact

_BODY = "INPUT_BITS d 1\nR 0 1\nIF d[0] {\n  %s\n}\nX_ERROR(0.01) 0\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
IF_X = _BODY % "X 0"
IF_EMPTY = _BODY % "TICK"
_OUT = "INPUT_BITS d 1\nR 0 1\nIF d[0] {\n  %s\n}\nH 1\nOUTPUT_QUBITS out 1\n"
OUT_IF_X = _OUT % "X 0"
OUT_IF_EMPTY = _OUT % "TICK"

# Header spellings the parser accepts (stim_parse.cpp: keyword in any case, bit read by stoll).
_SPELLINGS = [("if d[0] {", "lower_if"), ("IF d[+0] {", "plus_bit"), ("iF\td[-0]\t{", "mixed_case_tab_minus0")]

CASES = [pytest.param(IF_X, OUT_IF_X, id="x_body"), pytest.param(IF_EMPTY, OUT_IF_EMPTY, id="empty_body")] + [
    pytest.param(IF_X.replace("IF d[0] {", h), OUT_IF_X.replace("IF d[0] {", h), id=i) for h, i in _SPELLINGS]

_ESCAPE = "resolve_branches_text"


def _surfaces(text, out_text):
    c = xtim.Circuit(text)
    oc = xtim.Circuit(out_text)
    return [
        # guarded before this change
        ("compile_detector_sampler", lambda: c.compile_detector_sampler()),
        ("detector_error_model", lambda: c.detector_error_model_text()),
        ("detector_error_model", lambda: c.detector_error_model()),
        ("detector_error_model", lambda: c.detector_error_model_with_reject()),
        ("output_candidates", lambda: xtim.output_candidates(oc)),
        ("output_is_deterministic", lambda: xtim.output_is_deterministic(oc, [1])),
        ("output_state", lambda: xtim.output_state(oc, [1])),
        ("compile_twirl_sampler_from_ideal_state",
         lambda: xtim.compile_twirl_sampler_from_ideal_state(
             text, ideal_state=xtim.input_state_of("R 0\nH 0\nOUTPUT_QUBITS out 0\n"))),
        # newly guarded
        ("postselected_logical_error_rate", lambda: xtim.postselected_logical_error_rate(c, shots=16)),
        ("postselected_logical_error_rate", lambda: c.postselected_logical_error_rate(shots=16)),
        ("bare_state_of", lambda: xtim.bare_state_of(text)),
        ("input_state_of", lambda: xtim.input_state_of(out_text)),
        ("extract_frame", lambda: xtim.extract_frame(c, "Z1")),
        ("extract_frame", lambda: c.extract_frame("Z1")),
    ]


@pytest.mark.parametrize("text,out_text", CASES)
def test_public_surfaces_refuse_if_by_name(text, out_text):
    for surface, call in _surfaces(text, out_text):
        with pytest.raises(ValueError) as ei:
            call()
        msg = str(ei.value)
        assert msg.startswith(surface + ":"), (surface, msg)
        assert "IF d[0]" in msg, (surface, msg)
        assert "INPUT_BITS" in msg and _ESCAPE in msg, (surface, msg)


@pytest.mark.parametrize("text,out_text", CASES)
def test_raw_bindings_refuse_if_by_name(text, out_text):
    # The engine's own refusal (the deferral pass): ValueError, user-facing, no "pipeline bug".
    raw = [
        lambda: _xtim.export_dem_text(text),
        lambda: _xtim.TwirlSampler(text, 1.0, 0, True),
        lambda: _xtim.bare_state_of(text),
        lambda: _xtim.input_state_of(out_text),
        lambda: _xtim.expectation_frames(text),
        lambda: _xtim.detector_determinism(text),
    ]
    for call in raw:
        # explicit_exact: the raw DEM export is an exact entry point (the counter's contract, conftest.py).
        with explicit_exact(), pytest.raises(ValueError) as ei:
            call()
        msg = str(ei.value)
        assert "the circuit has IF blocks on INPUT_BITS (IF " in msg, msg
        assert _ESCAPE in msg and "PartitionedTwirlSampler" in msg, msg
        assert "pipeline bug" not in msg, msg


def test_raw_binding_names_the_port_when_kept():
    # The TwirlSampler front end keeps the parsed INPUT_BITS ports: the condition is named in full.
    with pytest.raises(ValueError, match=r"\(IF d\[0\]\)"):
        _xtim.TwirlSampler(IF_X, 1.0, 0, True)


def test_resolved_branches_are_accepted():
    # The escape the message names works: each resolved text is IF-free and compiles.
    for bits in ([0], [1]):
        resolved = _xtim.resolve_branches_text(IF_X, bits)
        assert "IF" not in resolved
        xtim.Circuit(resolved).detector_error_model_text()


def test_spellings_are_ifs_to_the_parser():
    # The spelling cases above are real IF blocks (the engine sees a driving bit), not parse errors.
    for h, _ in _SPELLINGS:
        assert _xtim.if_driving_bits(IF_X.replace("IF d[0] {", h)) == [(0, 0)], h


def test_guard_refuses_when_no_header_is_recognised(monkeypatch):
    # Belt and braces: an IF the engine sees but the header regex misses is still refused,
    # named by the engine's (port, bit).
    import re
    import xtim.circuit as xc
    monkeypatch.setattr(xc, "_IF_COND_RE", re.compile(r"(?!)"))
    assert xc._unresolved_if_conditions(IF_X) == ["<INPUT_BITS port #0>[0]"]
    with pytest.raises(ValueError, match=r"^bare_state_of: .*IF <INPUT_BITS port #0>\[0\]"):
        xtim.bare_state_of(IF_X)


_NO_IF = "INPUT_BITS d 1\nR 0 1\nX_ERROR(0.01) 0\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
_NO_IF_OUT = "INPUT_BITS d 1\nR 0 1\nH 1\nOUTPUT_QUBITS out 1\n"


def test_input_bits_without_if_is_accepted():
    # Declaring INPUT_BITS alone is not an IF: every guarded surface serves the circuit (as at 38ac487).
    c = xtim.Circuit(_NO_IF)
    assert _xtim.if_driving_bits(_NO_IF) == []
    xtim.bare_state_of(_NO_IF)
    xtim.input_state_of(_NO_IF_OUT)
    assert c.extract_frame("Z1") == []
    with pytest.warns(UserWarning):
        r = c.postselected_logical_error_rate(shots=16)
    assert r.shots == 16
    dets, obs = c.compile_detector_sampler(seed=1).sample(4, separate_observables=True)
    assert dets.shape == (4, 1) and obs.shape == (4, 1)
