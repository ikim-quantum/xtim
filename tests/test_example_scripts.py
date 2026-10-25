"""Every examples/*.py script runs standalone and ends with its noiseless F == 1 assert (README section 6)."""
import os
import pathlib
import subprocess
import sys

import pytest

import xtim

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "examples").glob("*.py"))


def test_scripts_present():
    assert len(SCRIPTS) >= 8, SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_example_script_runs(script, tmp_path):
    if "decoded" in script.name:
        pytest.importorskip("pymatching")
        pytest.importorskip("stim")
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(xtim.__file__).resolve().parents[1]), MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "noiseless check: F = 1 on every shot" in r.stdout or "07_refusals" in script.name, r.stdout
