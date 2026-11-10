"""exact-out T5 (owner ruling 2026-10-30): the exact engine's PUBLIC entry points are removed; the engine itself stays
as the test oracle (xtim._xtim_testapi + tests/_exact_engine.py). Every removed name fails loudly (AttributeError /
TypeError / argparse error) — no compat shims. Re-adding any of them turns a test here red.

Removed: Circuit.compile_detector_sampler(engine=...) (the argument), Circuit.compile_sampler / CompiledSampler,
CompiledDetectorSampler, the `xtim sample` CLI verb, `xtim detect --exp_out / --meas_out`, Circuit.compile_exact_branches
/ ExactBranches / ExactBranchSimulator / ExpandedLawTooLarge (module xtim.forced), collect / Task (module xtim.collect;
scale_noise stays), Circuit.compile_reference / reference_info / Reference / cache_dir / REF_FORMAT_VERSION (module
xtim.reference), XtimReferenceError / XtimCacheWarning, the `xtim state compile|verify` CLI verbs, and the _xtim
bindings CompiledProgram / ExactBranchProgram / ExactLawExpander / ref_info / ref_verify_cheap / ref_compile_text /
deferred_signature / _tree_fail_memo_stats / _tree_fail_memo_clear / EXACT_LAW_WORK_DEFAULT / REF_FORMAT_VERSION
(moved to xtim._xtim_testapi).

exact-out T6: the plan_frame_cosets binding + FRAME_COSETS_ABI (-> xtim._xtim_testapi) and xtim/frames.py
(-> tests/_frames.py; it was never exported from `xtim`)."""
import os
import subprocess
import sys

import numpy as np
import pytest

import xtim
from xtim import _xtim
from xtim import _xtim_testapi as TA

MODULE_NAMES = ["CompiledSampler", "CompiledDetectorSampler", "Task", "collect", "Reference", "ExactBranches",
                "ExactBranchSimulator", "ExpandedLawTooLarge", "REF_FORMAT_VERSION", "cache_dir",
                "XtimReferenceError", "XtimCacheWarning"]
CIRCUIT_METHODS = ["compile_sampler", "compile_exact_branches", "compile_reference", "reference_info",
                   "_resolve_reference"]
BINDINGS = ["CompiledProgram", "ExactBranchProgram", "ExactLawExpander", "ref_info", "ref_verify_cheap",
            "ref_compile_text", "deferred_signature", "_tree_fail_memo_stats", "_tree_fail_memo_clear",
            "EXACT_LAW_WORK_DEFAULT", "REF_FORMAT_VERSION",
            "plan_frame_cosets", "FRAME_COSETS_ABI"]                       # T6
TEXT = "R 0\nX_ERROR(0.2) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_removed_module_names(name):
    with pytest.raises(AttributeError):
        getattr(xtim, name)
    assert name not in xtim.__all__


@pytest.mark.parametrize("name", ["XtimReferenceError", "XtimCacheWarning"])
def test_removed_error_classes(name):
    assert not hasattr(xtim.errors, name)


@pytest.mark.parametrize("name", CIRCUIT_METHODS)
def test_removed_circuit_methods(name):
    assert not hasattr(xtim.Circuit(TEXT), name)


@pytest.mark.parametrize("module", ["forced.py", "reference.py", "collect.py", "frames.py"])
def test_removed_modules(module):
    # (checked on the package directory: an editable install of another checkout can extend the import path)
    assert not os.path.exists(os.path.join(os.path.dirname(xtim.__file__), module))


@pytest.mark.parametrize("engine", ["exact", "twirl", "auto"])
def test_engine_argument_removed(engine):
    with pytest.raises(TypeError, match="engine"):
        xtim.Circuit(TEXT).compile_detector_sampler(seed=1, engine=engine)


def test_detector_sampler_is_the_twirl_engine():
    s = xtim.Circuit(TEXT).compile_detector_sampler(seed=1)
    assert isinstance(s, xtim.TwirlDetectorSampler) and s.engine_report() == {"engine": "twirl"}


@pytest.mark.parametrize("name", BINDINGS)
def test_exact_bindings_moved_to_the_test_api(name):
    assert not hasattr(_xtim, name), name
    assert hasattr(TA, name), name


def test_scale_noise_stays():
    assert "scale_noise" in xtim.__all__
    assert xtim.scale_noise("X_ERROR(0.001) 0\n", 0.01) == "X_ERROR(0.01) 0\n"


def _cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "xtim", *args], capture_output=True, text=True, cwd=cwd)


@pytest.mark.parametrize("verb", [["sample"], ["state", "compile"], ["state", "verify"]])
def test_cli_removed_verbs(verb, tmp_path):
    f = tmp_path / "c.stim"
    f.write_text(TEXT)
    r = _cli(*verb, "--in", str(f), "--shots", "4") if verb == ["sample"] else _cli(*verb, str(f), str(tmp_path / "x"))
    assert r.returncode != 0 and "invalid choice" in r.stderr, (r.returncode, r.stderr[-300:])


@pytest.mark.parametrize("flag", ["--exp_out", "--meas_out"])
def test_cli_detect_removed_flags(flag, tmp_path):
    f = tmp_path / "c.stim"
    f.write_text(TEXT)
    r = _cli("detect", "--in", str(f), "--shots", "4", flag, str(tmp_path / "o"))
    assert r.returncode != 0 and "unrecognized arguments" in r.stderr, (r.returncode, r.stderr[-300:])


def test_cli_detect_is_the_twirl_sampler(tmp_path):
    f = tmp_path / "c.stim"
    f.write_text(TEXT)
    out, obs = tmp_path / "d.b8", tmp_path / "o.b8"
    r = _cli("detect", "--in", str(f), "--shots", "300", "--seed", "7", "--out_format", "b8",
             "--out", str(out), "--obs_out", str(obs))
    assert r.returncode == 0, r.stderr
    d, o = xtim.Circuit(TEXT).compile_detector_sampler(seed=7).sample(300, separate_observables=True, bit_packed=True)
    assert out.read_bytes() == np.asarray(d).tobytes() and obs.read_bytes() == np.asarray(o).tobytes()


def test_cli_detect_refusal_is_by_name(tmp_path):
    f = tmp_path / "c.stim"
    f.write_text("RX 0\nT 0\nH 0\nT 0\nMX 0\nDETECTOR rec[-1]\n")      # outside T-depth 1
    r = _cli("detect", "--in", str(f), "--shots", "4")
    assert r.returncode == 1 and "outside xtim's model" in r.stderr, (r.returncode, r.stderr[-300:])


def test_frames_module_is_test_api():
    """T6: the frame algebra is tests/_frames.py on the test-API table (xtim/frames.py is gone: test_removed_modules;
    an import check would be fooled by an editable install of another checkout on the import path)."""
    import _frames
    fn, why = _frames._resolve_compiled_table()
    assert fn is TA.plan_frame_cosets and why is None
