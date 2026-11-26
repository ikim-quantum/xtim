"""Peak memory of the bare-state build behind compile_detector_sampler (Stage-1 T4: frame moves, no placeholder frames).

On an N-measurement R / X_ERROR / M / DETECTOR chain the deferred circuit has n' = 3N wires and every frame tableau
(forward + inverse rows) is n'^2 bytes. Before T4 three such tableaux were alive at the build's peak (two identity
placeholders beside the one being built) plus a second inverse tableau during each dual rebuild: N = 8000 peaked at
3.2 GB. After T4 it peaks at 1.5 GB. The bound below sits between the two (about 1.6x the measured value), so it
catches a returned placeholder or copy without being sensitive to allocator noise.

Measured in a fresh subprocess (ru_maxrss is a process-lifetime maximum). Marked slow: about 35 s.
"""
import os
import subprocess
import sys

import pytest

_WORKER = r"""
import resource, sys, xtim
N = int(sys.argv[1])
t = "".join(f"R {q}\nX_ERROR(0.01) {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(N))
xtim.Circuit(t).compile_detector_sampler(seed=1)
print("MAXRSS_KB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""


def _peak_rss_mb(n):
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", _WORKER, str(n)], env=env, capture_output=True, text=True,
                         check=True).stdout
    line = [ln for ln in out.splitlines() if ln.startswith("MAXRSS_KB")][-1]
    return int(line.split()[1]) / 1024.0


@pytest.mark.slow
def test_chain_8000_compile_peak_rss():
    mb = _peak_rss_mb(8000)
    assert mb < 2400, f"compile_detector_sampler on the N=8000 chain peaked at {mb:.0f} MB (T4: ~1500 MB, was ~3200 MB)"
