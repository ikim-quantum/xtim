"""EXP-fix E6 (owner 2026-10-21): the beta-extraction fidelity workflow is removed — diagnose, the collect
beta fidelity (Task.target_k), ler's fidelity mode and fidelity_from_logicals. PAULI_EXPECTATION itself stays
(raw exact-engine columns, DEM L-columns); direct fidelity estimation (OUTPUT_QUBITS + RecordFidelity) is the
fidelity route. Mutations: ME-NAME (re-export diagnose), ME-LER (drop ler's PE refusal), ME-DET (ler's keep mask
from all detectors) each turn a test here red."""
import os
import subprocess
import sys

import numpy as np
import pytest

import xtim
from _orig_examples import ORIG_EXAMPLES, load_orig  # noqa: E402,F401
from xtim._exact_scope import explicit_exact


@pytest.mark.parametrize("name", ["diagnose", "Diagnosis", "ExpectationInfo", "OutputStateInfo",
                                  "fidelity_from_logicals"])
def test_removed_module_names(name):
    with pytest.raises(AttributeError):
        getattr(xtim, name)
    assert name not in xtim.__all__


@pytest.mark.parametrize("name", ["diagnose", "fidelity_from_logicals"])
def test_removed_circuit_methods(name):
    assert not hasattr(xtim.Circuit("M 0\n"), name)


def test_diagnose_module_gone():
    # (checked on the package directory: an editable install of another checkout can extend the import path)
    assert not os.path.exists(os.path.join(os.path.dirname(xtim.__file__), "diagnose.py"))


def test_task_target_k_removed():
    with pytest.raises(TypeError):
        xtim.Task(circuit=xtim.Circuit("M 0\n"), shots=1, target_k=1)


def test_cli_diagnose_verb_unknown(tmp_path):
    f = tmp_path / "c.stim"
    f.write_text("M 0\n")
    r = subprocess.run([sys.executable, "-m", "xtim", "diagnose", str(f)], capture_output=True, text=True)
    assert r.returncode != 0 and "invalid choice" in r.stderr, (r.returncode, r.stderr[-300:])


def test_ler_refuses_pauli_expectation_circuits_by_name():
    c = load_orig("cultivation_d3_faithful")        # PE + OBSERVABLE_INCLUDE
    with pytest.raises(xtim.XtimError, match="declares 2 PAULI_EXPECTATION column"):
        c.postselected_logical_error_rate(p=1e-3, shots=100)
    with pytest.raises(xtim.XtimError, match="RecordFidelity"):
        load_orig("distillation_15_1_3").postselected_logical_error_rate(shots=100)


def test_ler_target_k_removed():
    with pytest.raises(TypeError):
        xtim.Circuit("M 0\n").postselected_logical_error_rate(shots=10, target_k=1)


# A gauge detector (random noiselessly) next to a deterministic one carrying the observable: the keep mask must use
# the deterministic detector only (ME-DET: a mask over ALL detectors halves the acceptance).
_OBS = ("RX 0\nM 0\nDETECTOR rec[-1]\n"
        "R 1 2\nX_ERROR(0.01) 1 2\nCX 1 2\nM 1 2\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")


def test_ler_observable_mode_equals_the_manual_workflow():
    c = xtim.Circuit(_OBS)
    r = c.postselected_logical_error_rate(p=0.05, p0=0.01, shots=40_000, seed=3)
    with explicit_exact():
        det = list(xtim._xtim.detector_determinism(_OBS)["deterministic_detectors"])
    assert det == [1]
    s = xtim.Circuit(xtim.scale_noise(_OBS, 0.05, 0.01)).compile_detector_sampler(seed=3)
    dets, obs = s.sample(40_000, separate_observables=True)
    keep = ~dets[:, det].any(axis=1)
    assert r.kept == int(keep.sum())
    assert r.value == float(obs.any(axis=1)[keep].mean())
    assert r.mode == "observable" and 0.9 < r.acceptance < 0.97
