"""Public-API contract tests for the distributed `xtim` package.

These are NOT the engine's correctness suite (that lives upstream: byte-parity vs
the CLI, exact oracles, stream pins). They verify the lighter downstream contract:
the installed package's public surface behaves as documented, on tiny circuits with
EXACT known physics. Fast, dependency-light, and they double as usage examples.
"""
import math

import numpy as np
import pytest

import xtim


def test_version_constants_exposed():
    assert isinstance(xtim.ENGINE_VERSION, str) and xtim.ENGINE_VERSION
    assert isinstance(xtim.REF_FORMAT_VERSION, int)


def test_clifford_measurement_deterministic():
    # |1> prepared, Z-measured -> every shot is 1.
    c = xtim.Circuit("R 0\nX 0\nM 0\n")
    m = c.compile_sampler(seed=1).sample(256)
    assert m.shape == (256, 1)
    assert bool(m.all())


def test_clifford_expectation_exact():
    # S|+> = |+i> is the +1 eigenstate of Y -> <Y> = +1 exactly (a chi=1 Clifford).
    c = xtim.Circuit("RX 0\nS 0\nPAULI_EXPECTATION(0) Y0\n")
    dets, exps = c.compile_detector_sampler(seed=2).sample(
        512, return_expectations=True)
    assert exps.shape == (512, 1)
    assert np.allclose(exps, 1.0, atol=1e-9), np.unique(exps)


def test_magic_expectation_exact():
    # |T> = T|+>: <X> = cos(pi/4) = 1/sqrt(2), emitted as a RAW per-shot value.
    # Noiseless => deterministic, so every shot equals 1/sqrt(2).
    c = xtim.Circuit("RX 0\nT 0\nPAULI_EXPECTATION(0) X0\n")
    dets, exps = c.compile_detector_sampler(seed=3).sample(
        2000, return_expectations=True)
    assert exps.shape == (2000, 1)
    assert np.allclose(exps, 2.0 ** -0.5, atol=1e-9), np.unique(exps)


def test_expectation_columns_ordering():
    # No observables, two PAULI_EXPECTATION declarations -> columns [0, 1].
    c = xtim.Circuit("RX 0\nT 0\nPAULI_EXPECTATION(0) X0\nPAULI_EXPECTATION(1) Z0\n")
    assert c.num_observables == 0
    assert c.num_expectations == 2
    assert c.expectation_columns == [0, 1]


def test_detector_sampler_one_run_multiple_records():
    # All requested records come from ONE engine run; shapes line up.
    c = xtim.Circuit("RX 0\nT 0\nPAULI_EXPECTATION(0) X0\n")
    dets, exps, meas = c.compile_detector_sampler(seed=4).sample(
        100, return_expectations=True, return_measurements=True)
    assert dets.shape == (100, c.num_detectors)
    assert exps.shape == (100, c.num_expectations)
    assert meas.shape == (100, c.num_measurements)


def test_parse_error_is_typed():
    from xtim.errors import XtimParseError
    with pytest.raises(XtimParseError):
        xtim.Circuit("NOT_A_REAL_GATE 0\n")


def test_detector_error_model_is_stim_object():
    stim = pytest.importorskip("stim")
    # A noisy Clifford circuit -> a real stim.DetectorErrorModel.
    c = xtim.Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n")
    dem = c.detector_error_model()
    assert isinstance(dem, stim.DetectorErrorModel)
    assert dem.num_detectors == c.num_detectors == 1


def test_dem_with_reject_pauli_circuit_is_plain_dem():
    stim = pytest.importorskip("stim")
    # A Pauli (rep-code) circuit has an EMPTY reject region and its decode DEM is
    # byte-identical to the plain export — the additive feature is a no-op here.
    c = xtim.Circuit("R 0 1 2\nX_ERROR(0.05) 0 1 2\nM 0 1 2\n"
                     "DETECTOR rec[-3] rec[-2]\nDETECTOR rec[-2] rec[-1]\n")
    res = c.detector_error_model_with_reject()
    assert isinstance(res, xtim.DemWithReject)
    assert isinstance(res.dem, stim.DetectorErrorModel)
    assert res.reject_detectors == [] and res.postselect_faults == []
    assert str(res.dem) == str(c.detector_error_model())


def test_dem_with_reject_cultivation():
    pytest.importorskip("stim")
    from pathlib import Path
    path = (Path(__file__).resolve().parent.parent / "examples"
            / "cultivation_d3_faithful.stim")
    if not path.is_file():
        pytest.skip("examples/cultivation_d3_faithful.stim not bundled (monorepo run)")
    res = xtim.Circuit.from_file(path).detector_error_model_with_reject()
    assert isinstance(res, xtim.DemWithReject)
    # postselect_faults is the PRIMARY output: each flags a not-Pauli-correctable fault
    # (sorted detector signature + positive probability) for the user to post-select/budget.
    assert res.postselect_faults  # cultivation has faults to flag
    for f in res.postselect_faults:
        assert isinstance(f, xtim.PostselectFault)
        assert f.detectors == sorted(f.detectors)
        assert 0.0 < f.probability
    # reject_detectors is the derived sorted, de-duplicated union of the signatures.
    union = sorted({d for f in res.postselect_faults for d in f.detectors})
    assert res.reject_detectors == union
    assert all(isinstance(d, int) for d in res.reject_detectors)
    # the decode DEM builds a matcher and decode_batch runs on the kept (post-selected) shots.
    pm = pytest.importorskip("pymatching")
    m = pm.Matching.from_detector_error_model(res.dem)
    dets = xtim.Circuit.from_file(path).compile_detector_sampler(seed=1).sample(
        500, separate_observables=True)[0]
    keep = res.keep_mask(dets)
    assert m.decode_batch(dets[keep]).shape[0] == int(keep.sum())


def test_collect_runs_serial_two_tasks():
    c = xtim.Circuit("RX 0\nT 0\nPAULI_EXPECTATION(0) X0\n")
    rows = xtim.collect([
        xtim.Task(circuit=c, shots=200, seed=1, metadata={"k": 1}),
        xtim.Task(circuit=c, shots=200, seed=2, metadata={"k": 2}),
    ])
    assert len(rows) == 2
    for row in rows:
        assert row["shots"] == 200
        # no keep/decoder -> value is the raw mean expectation = 1/sqrt(2).
        assert np.allclose(row["value"], 2.0 ** -0.5, atol=1e-9)
    assert [r["metadata"]["k"] for r in rows] == [1, 2]


def test_programmatic_build_samples_like_text():
    """append / + / * build a circuit that samples identically to the same text."""
    built = xtim.Circuit("")
    built.append("X", 0)
    built.append("X", 5)
    built.append("M", [0, 1, 5])
    text = xtim.Circuit("X 0\nX 5\nM 0 1 5\n")
    assert built.text == text.text
    a = built.compile_sampler(seed=4).sample(8, bit_packed=True)
    b = text.compile_sampler(seed=4).sample(8, bit_packed=True)
    assert np.array_equal(a, b)


def test_repeat_and_concat_semantics():
    rep = xtim.Circuit("X 0\nM 0") * 3
    flat = xtim.Circuit("X 0\nM 0\nX 0\nM 0\nX 0\nM 0")
    assert rep.text.startswith("REPEAT 3 {")
    assert rep.num_measurements == flat.num_measurements == 3
    assert len(rep) == 1  # a REPEAT block is one top-level instruction
    assert (xtim.Circuit("H 0") + xtim.Circuit("M 0")).text == "H 0\nM 0"


def test_to_file_round_trips(tmp_path):
    c = xtim.Circuit("X 0\nM 0\n")
    path = tmp_path / "c.stim"
    c.to_file(path)
    assert xtim.Circuit.from_file(path).text == c.text


def test_invalid_append_raises_and_preserves_circuit():
    c = xtim.Circuit("H 0")
    with pytest.raises(xtim.XtimParseError):
        c.append("NOT_A_REAL_GATE", 0)
    assert c.text == "H 0"  # atomic: a failed append leaves the circuit untouched


def test_load_example_bundled_circuits():
    names = xtim.list_examples()
    assert "cube_ccz" in names and "miniature_oracle" in names
    # load_example returns a ready Circuit; example_path resolves a real file.
    c = xtim.load_example("cube_ccz")
    assert isinstance(c, xtim.Circuit) and c.num_qubits == 12
    # accepts the ".stim" suffix too, and resolves to an existing file
    import os
    assert os.path.exists(xtim.example_path("cube_ccz.stim"))
    # every advertised name resolves
    for n in names:
        assert os.path.exists(xtim.example_path(n))
    # unknown name -> ValueError that names the available examples
    with pytest.raises(ValueError, match="unknown example"):
        xtim.load_example("does_not_exist")


def test_seed_validation_clean_errors():
    c = xtim.Circuit("X 0\nM 0")
    with pytest.raises(ValueError, match="seed must be in"):
        c.compile_sampler(seed=-1).sample(4)
    with pytest.raises(TypeError, match="seed must be an int"):
        c.compile_sampler(seed=1.5).sample(4)
    # valid seeds still work and are deterministic
    import numpy as np
    a = c.compile_sampler(seed=0).sample(8)
    b = c.compile_sampler(seed=0).sample(8)
    assert np.array_equal(a, b)


def test_scale_noise_zero_p0_clean_error():
    with pytest.raises(ValueError, match="p0"):
        xtim.scale_noise("X_ERROR(0.001) 0\n", p=0.01, p0=0.0)


def test_pathological_cache_dir_falls_back_gracefully():
    # An embedded NUL byte makes the filesystem reject the path; the cache must
    # warn + fall back to the (exact) deduced bare state, not leak a raw error.
    import warnings
    old = xtim.cache_dir
    try:
        xtim.cache_dir = "/tmp/xtim\x00bad"
        c = xtim.Circuit("RX 0\nT 0\nPAULI_EXPECTATION(0) X0\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", xtim.XtimCacheWarning)
            exps = c.compile_detector_sampler(seed=0).sample(
                4000, return_expectations=True)[1]
        assert abs(exps.mean() - 0.70710678) < 0.05   # still exact via fallback
    finally:
        xtim.cache_dir = old


# ── fidelity / figure-of-merit (Task.target_k), v0.5.9 ───────────────────────

# |T> = (|0> + e^{iπ/4}|1>)/√2 has <X>=<Y>=1/√2: the two nonzero-beta columns.
_T_STATE = "R 0\nH 0\nT 0\nPAULI_EXPECTATION(0) X0\nPAULI_EXPECTATION(1) Y0\n"
# |T>⊗|T>: the COMPLETE 8-Pauli support of a 2-qubit product magic state (k=2).
_TT_STATE = (
    "R 0 1\nH 0 1\nT 0 1\n"
    "PAULI_EXPECTATION(0) X0\nPAULI_EXPECTATION(1) Y0\n"
    "PAULI_EXPECTATION(2) X1\nPAULI_EXPECTATION(3) Y1\n"
    "PAULI_EXPECTATION(4) X0*X1\nPAULI_EXPECTATION(5) X0*Y1\n"
    "PAULI_EXPECTATION(6) Y0*X1\nPAULI_EXPECTATION(7) Y0*Y1\n"
)


def test_fidelity_noiseless_T_state_is_one():
    row = xtim.collect([xtim.Task(circuit=xtim.Circuit(_T_STATE),
                                  shots=20000, seed=1, target_k=1)])[0]
    assert row["target_k"] == 1
    np.testing.assert_allclose(row["target_beta"], [2.0 ** -0.5] * 2, atol=1e-9)
    assert abs(row["fidelity"] - 1.0) < 1e-9
    assert math.isclose(row["infidelity"], 1.0 - row["fidelity"])  # exact, unclamped


def test_fidelity_keys_absent_without_target_k():
    row = xtim.collect([xtim.Task(circuit=xtim.Circuit(_T_STATE),
                                  shots=200, seed=1)])[0]
    for key in ("fidelity", "infidelity", "infidelity_sem", "target_k", "target_beta"):
        assert key not in row


def test_fidelity_target_k_must_be_positive_int():
    c = xtim.Circuit(_T_STATE)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            xtim.Task(circuit=c, shots=10, target_k=bad)


def test_fidelity_complete_support_k2_is_one():
    row = xtim.collect([xtim.Task(circuit=xtim.Circuit(_TT_STATE),
                                  shots=20000, seed=2, target_k=2)])[0]
    assert abs(row["fidelity"] - 1.0) < 1e-6  # exercises the 1/2**k for k>1


def test_fidelity_over_one_warns_on_wrong_target_k():
    # |T>⊗|T> scored with target_k=1 gives F≈2 -> physically impossible -> warns.
    with pytest.warns(UserWarning):
        row = xtim.collect([xtim.Task(circuit=xtim.Circuit(_TT_STATE),
                                      shots=4000, seed=2, target_k=1)])[0]
    assert row["fidelity"] > 1.5  # value left unclamped


def test_fidelity_incomplete_support_warns_via_ceiling():
    # The symmetric guard: |T> scored at target_k=2 can never reach F=1 (noiseless
    # ceiling 0.5) -> a UserWarning flags the incomplete set / too-large target_k.
    with pytest.warns(UserWarning, match="cannot reach fidelity 1"):
        row = xtim.collect([xtim.Task(circuit=xtim.Circuit(_T_STATE),
                                      shots=4000, seed=1, target_k=2)])[0]
    assert abs(row["fidelity"] - 0.5) < 1e-2  # value left unchanged


def test_fidelity_no_columns_refuses():
    c = xtim.Circuit("R 0\nH 0\nT 0\n")  # magic but no PAULI_EXPECTATION
    with pytest.raises(xtim.XtimError):
        xtim.collect([xtim.Task(circuit=c, shots=10, target_k=1)])


# ── resolve_branches_text round-trip regression (GH fix 2026-07-31) ───────────
# Bug: the serializer emitted `PAULI_EXPECTATION 0 X0` (space form) but the
# parser requires `PAULI_EXPECTATION(0) X0` (paren form), breaking round-trips
# on any payload-carrying circuit (adaptq's exact tier hits this).

def test_resolve_branches_text_pauli_expectation_roundtrip():
    """resolve_branches_text must emit the parenthesised form so the output
    re-parses cleanly (round-trip).  The serializer previously emitted a
    space-separated label that the parser rejected."""
    from xtim import _xtim

    # Minimal circuit: a T-magic gate behind an IF branch with PAULI_EXPECTATION.
    text = (
        "INPUT_BITS drv 1\n"
        "RX 0\n"
        "T 0\n"
        "IF drv[0] {\n"
        "  PAULI_EXPECTATION(0) X0\n"
        "}\n"
    )

    # Branch not taken: PAULI_EXPECTATION is dropped, output is plain.
    out0 = _xtim.resolve_branches_text(text, [0])
    c0 = xtim.Circuit(out0)          # must parse without error
    assert "PAULI_EXPECTATION" not in c0.text

    # Branch taken: PAULI_EXPECTATION must survive in the parenthesised form.
    out1 = _xtim.resolve_branches_text(text, [1])
    assert "PAULI_EXPECTATION(0)" in out1, (
        f"serializer must use paren form; got: {out1!r}"
    )
    # Re-parse the resolved text -- this is the round-trip check.
    c1 = xtim.Circuit(out1)          # raises ValueError on pre-fix space form
    assert c1.text == out1           # exact equality: no content lost or changed

    # Sampler still executes correctly on the re-parsed circuit.
    _, exps = c1.compile_detector_sampler(seed=7).sample(
        200, return_expectations=True
    )
    assert exps.shape == (200, 1)
    assert np.allclose(exps, 2.0 ** -0.5, atol=1e-9)


def test_resolve_branches_text_pauli_expectation_roundtrip_with_frame():
    """round-trip also holds for a PAULI_EXPECTATION with an obs_frame record
    and a multi-qubit Pauli product."""
    from xtim import _xtim

    text = (
        "INPUT_BITS drv 1\n"
        "RX 0\n"
        "RX 1\n"
        "M 0\n"
        "IF drv[0] {\n"
        "  PAULI_EXPECTATION(0) X0*X1 rec[-1]\n"
        "}\n"
    )

    out1 = _xtim.resolve_branches_text(text, [1])
    assert "PAULI_EXPECTATION(0)" in out1
    c1 = xtim.Circuit(out1)
    assert c1.text == out1
