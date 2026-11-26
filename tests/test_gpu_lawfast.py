"""The one-word law's two implementations (xdem_kernels.h, xk_law_fast): the register form (XK_CAP_RHO <= 32: the coin
span's echelon from the generators g_i, 32-bit words, every local array indexed by a loop counter) and post_fast's
own construction (XK_CAP_RHO > 32). Both compute post_fast's unique objects (pivot columns, private-pivot kernel basis,
omega, C's canonical key) from the same coin word, so every output byte must agree: the reference build's variant 2
(XK_CAP_RHO = 32) against its variant 0 (64) on every one-word example and on random one-word files with rho up to
32 and a large beta_L (many keys of dim C >= 2), and Metal (compiled at the file's rho: the register form whenever
rho <= 32) against variant 0.
"""
import random
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim.gpu.backend import GpuBackend

from _gpu_ref import RefLauncher
from _xdem_synth import random_text
from test_gpu_postselect import _mask

ROOT = Path(__file__).parent.parent
EXAMPLES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "examples" / "xdem").glob("*.xdem"))
# random one-word files: (seed, D, O, r, m, A, L, tc); rho 16, 22 and 32, rhoL 6, 14 and 26
SYNTH = {"syn_rho16": (11, 10, 6, 20, 1, 40, 160, 80), "syn_rho22": (12, 8, 14, 28, 1, 50, 220, 110),
         "syn_rho32": (14, 6, 26, 40, 1, 70, 300, 150)}
SHOTS, SEED = 3000, 77


def _text(name):
    if name in SYNTH:
        seed, *kw = SYNTH[name]
        D, O, r, m, A, L, tc = kw
        return random_text(random.Random(seed), D=D, O=O, r=r, m=m, A=A, L=L, tc=tc)
    return (ROOT / name).read_text()


def _outputs(text, launcher):
    """sample, sample_ps (half the detectors) and estimate(None), each from a fresh sampler"""
    out = []
    s = xtim.XdemSampler(text)
    out.append(GpuBackend(s._s, launcher()).sample(SHOTS, SEED))
    s = xtim.XdemSampler(text)
    d, o, c, counts = GpuBackend(s._s, launcher()).sample_ps(SHOTS, SEED, _mask(s.num_detectors, "half"))
    out.append((d, o, c, sorted(counts.items())))
    s = xtim.XdemSampler(text)
    out.append(GpuBackend(s._s, launcher()).estimate(SHOTS, SEED, None))
    return out


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, (tuple, list)):
            _same(x, y)
        elif isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
        else:
            assert x == y


def _one_word_rho(text):
    t = xtim.XdemSampler(text)._s.device_tables()
    return int(t["rho"]) if t["fast"] and int(t["A"]) else None


CASES = [p for p in EXAMPLES if _one_word_rho(_text(p)) is not None] + list(SYNTH)


@pytest.mark.parametrize("name", CASES)
def test_register_form_equals_post_fast_construction(name):
    text = _text(name)
    rho = _one_word_rho(text)
    assert rho is not None and rho <= 32
    _same(_outputs(text, lambda: RefLauncher(variant=2)), _outputs(text, lambda: RefLauncher(variant=0)))


def test_synthetic_files_reach_the_register_form_hard_cases():
    """not vacuous: the random files have rho past 15 (cultivation_d5's) up to 32, a large beta_L, and many shots
    with dim C >= 2 (keys of several vectors), dim C varying from shot to shot"""
    for name in SYNTH:
        text = _text(name)
        t = xtim.XdemSampler(text)._s.device_tables()
        assert t["fast"] and 15 < int(t["rho"]) <= 32 and int(t["rhoL"]) >= 4, (name, t["rho"], t["rhoL"])
        s = xtim.XdemSampler(text)
        _, _, cs = GpuBackend(s._s, RefLauncher(variant=2)).sample(SHOTS, SEED)
        table = s._s.cspaces(0, int(cs.max()) + 1)   # C id -> its rows (dim C of them)
        dims = np.array([len(table[int(c)]) for c in cs])
        assert (dims >= 2).mean() > 0.05 and len(np.unique(dims)) >= 2, name


@pytest.mark.metal
@pytest.mark.parametrize("name", CASES)
def test_metal_equals_post_fast_construction(name):
    from xtim.gpu.metal import MetalLauncher
    text = _text(name)
    _same(_outputs(text, MetalLauncher), _outputs(text, lambda: RefLauncher(variant=0)))
