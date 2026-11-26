"""The structured multi-word law (xtim/gpu/structure.py on the host, xk_law_struct_mw in xdem_kernels.h).

The one-word law's tests (test_gpu_structure.py) on multi-word files (coord_words >= 2: rho > 64). Files:
Bravyi-Haah-like texts generated here (tests/_xdem_synth.bh_like_text, rho = k + 3): k = 70 (rho = 73, two slot words,
like merlin_bh70) and k = 100 (rho = 103, like merlin_bh100), k = 130 (rho = 133, three slot words), with signs or
without. Nothing depends on the merlin_bh* benchmark circuits (arXiv:2610.06811, not in this repository), except the
tests marked as such, which run on merlin_bh70 / bh100 from $XTIM_MERLIN_CIRCUITS (no default) and skip when it is
unset or the files are absent.

Detector: accepts the generated files (slot words = ceil(rho / 64)), refuses boundary_r65 (multi-word, no structure)
and a file past XK_MAX_ST_W slot words; a structured file whose T is made non-linear at the tables level is refused;
each refusal runs xk_law_multi. law_path / law_path_reason, and the structured law's queue (defer is ignored).

Exact: the structured law gives xk_law_multi's C per shot (the cspace ids of `sample`: dim C and key) and the zero
decoder's verdict on its own rows (estimate's counts = sample's then score); the renumbering leaves xk_law_multi's
outputs byte-identical; Metal equals the reference byte for byte. Statistics: the structured sampler vs the CPU
sampler (test_xdem_cross_route.py's z-scores, Z_MAX), rows being equal to xk_law_multi's only up to the coin shift.
"""
from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim.gpu import structure as gs
from xtim.gpu import tables as gt
from xtim.gpu.backend import DEFAULT_BLOCK, GpuBackend
from xtim.xdem import XdemStatus

import _gpu_compare as C
from _fake_cupy import cxx, cxx_syntax_check
from _gpu_ref import RefLauncher
from _xdem_synth import bh_like_text
from test_gpu_backend import stat_compare
from test_gpu_estimate import _hist_of_scores
from test_gpu_postselect import _mask
from test_xdem_cross_route import Z_MAX

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
B = DEFAULT_BLOCK   # the backend's block (K1's shots per group)
_MERLIN = os.environ.get("XTIM_MERLIN_CIRCUITS")
PAPER = Path(_MERLIN).expanduser() if _MERLIN else None
GEN = [(70, False), (70, True), (100, False), (130, True)]
GEN_IDS = [f"bh_like{k}{'_signs' if sg else ''}" for k, sg in GEN]


def _gen(k, signs):
    return bh_like_text(k, 300, seed=k + 7 * signs, signs=signs)


def _metal():
    try:
        from xtim.gpu.metal import MetalLauncher
        return MetalLauncher()
    except Exception:
        return None


def _fast():
    """Metal where there is one (the sampling tests' shots), else the reference build"""
    return _metal() or RefLauncher()


def _paper(name):
    if PAPER is None:
        pytest.skip("the merlin_bh* benchmark circuits are not configured (set XTIM_MERLIN_CIRCUITS)")
    p = PAPER / f"merlin_{name}_dep_p1e-3.stim"
    if not p.exists():
        pytest.skip(f"the merlin_bh* benchmark circuits are absent ({PAPER})")
    return xtim.Circuit(p.read_text()).detector_error_model_text(format="xdem")


class _Tables:
    """a sampler whose device tables are replaced (everything else delegated)"""

    def __init__(self, cpu, t):
        self._cpu, self._t = cpu, t

    def device_tables(self):
        return self._t

    def __getattr__(self, attr):
        return getattr(self._cpu, attr)


# ------------------------------------------------------------------------------------------------ detector, plan


@pytest.mark.parametrize("k, signs", GEN, ids=GEN_IDS)
def test_detector_accepts_multi_word_bh_like(k, signs):
    t = xtim.XdemSampler(_gen(k, signs))._s.device_tables()
    rho, cw = int(t["rho"]), int(t["coord_words"])
    assert rho == k + 3 and cw >= 2
    d = gs.detect(t)
    assert d["structure"], d["reason"]
    assert d["rD"] == 3 and len(d["core_gauge"]) == 3 and len(d["leaves"]) >= k - 1
    assert all(g == f for g, f in d["leaves"])
    assert gs.certified(t)
    p, why = gs.plan(t)
    assert p is not None, why
    assert p.sw == (rho + 63) // 64 <= gs.ST_MAX_W == gt.XK_MAX_ST_W
    assert (p.nG, p.rD, len(p.nonleaf)) == (3, 3, 8)
    # a leaf type's run of slots is cut at each slot word's end: at least one record per word it spans
    assert len(p.records) >= (8 if k == 70 else 9) + 2 * signs
    st = gs.st_table(p)
    assert int(st[3]) == int(signs) and int(st[4]) == p.sw
    assert sorted(p.perm) == list(range(int(t["A"])))
    at = len(p.nonleaf)
    for off, n, *_ in sorted(p.records):
        assert off == at
        at += n
    # the observable-coordinate table: sw words per entry
    stab, _ = gt.obs_table(t, p.sw)
    assert len(stab) % p.sw == 0


def test_detector_refuses_unstructured_and_too_wide():
    # boundary_r65: a multi-word file without the structure
    r65 = xtim.XdemSampler((ROOT / "tests" / "data" / "xdem" / "synth" / "boundary_r65.xdem").read_text())._s
    t = r65.device_tables()
    assert int(t["coord_words"]) >= 2
    p, why = gs.plan(t)
    assert p is None and why.startswith("no structure"), why
    b = GpuBackend(r65, RefLauncher(), structured=True)
    assert (b.law_path, b.law_path_reason) == ("multi-word", why) and b._plan is None and b._mw > 0
    # rho = 257: five slot words, past XK_MAX_ST_W
    t = xtim.XdemSampler(bh_like_text(254, 300, seed=254))._s.device_tables()
    p, why = gs.plan(t)
    assert p is None and why == f"rho = 257: 5 slot words (at most {gt.XK_MAX_ST_W})", why


def test_uncertified_multi_word_falls_back():
    """T made non-linear in kappa (one leaf axis's gamma bit 0 flipped in GcW; a tables-level change): the structure is
    still detected, the certificate fails, the backend runs xk_law_multi"""
    s = xtim.XdemSampler(_gen(70, False))
    t = dict(s._s.device_tables())
    p, _ = gs.plan(t)
    a = p.perm.index(p.records[0][0])   # the original index of the first record's first axis
    cw = int(t["coord_words"])
    G = np.array(t["GcW"]).copy()
    G[a * cw] ^= np.array(1, G.dtype)
    t["GcW"] = G
    assert gs.detect(t)["structure"] and not gs.certified(t)
    q, why = gs.plan(t)
    assert q is None and why.startswith("not certified"), why
    b = GpuBackend(_Tables(s._s, t), RefLauncher(), structured=True)
    assert (b.law_path, b.law_path_reason) == ("multi-word", why) and b._plan is None


def test_multi_word_law_path_and_defaults():
    s = xtim.XdemSampler(_gen(70, False))._s
    b = GpuBackend(s, RefLauncher())
    assert (b.law_path, b.law_path_reason) == ("multi-word structured", "")
    assert b._law is False and b._defer and b._mw == 0   # the queue, no law table, no mws scratch
    from xtim.gpu import backend as GB
    assert b._block == GB.DEFAULT_BLOCK
    # the structured multi-word law runs from K2b's queue only: defer=False is ignored
    b = GpuBackend(s, RefLauncher(), defer=False)
    assert b.law_path == "multi-word structured" and b._defer
    b = GpuBackend(s, RefLauncher(), structured=False)
    assert (b.law_path, b.law_path_reason) == ("multi-word", "structured=False") and not b._defer and b._mw > 0
    b = GpuBackend(s, RefLauncher(), law_table=True)
    assert (b.law_path, b.law_path_reason) == ("multi-word", "law_table=True")
    with pytest.raises(ValueError, match="law table"):
        GpuBackend(s, RefLauncher(), structured=True, law_table=True)


# ------------------------------------------------------------------------------------------------ exact


@pytest.mark.parametrize("k, signs", GEN[:2], ids=GEN_IDS[:2])
def test_renumbering_keeps_the_multi_word_law(k, signs):
    """xk_law_multi on the renumbered tables: every output byte the same"""
    s = xtim.XdemSampler(_gen(k, signs))
    t = s._s.device_tables()
    p, _ = gs.plan(t)
    flags = _mask(s.num_detectors, "half")
    a = GpuBackend(s._s, RefLauncher(), structured=False)
    b = GpuBackend(_Tables(s._s, gs.renumber(t, p)), RefLauncher(), structured=False)
    for x, y in zip(a.sample(2000, 4), b.sample(2000, 4)):
        assert np.array_equal(x, y)
    for x, y in zip(a.sample_ps(2000, 5, flags), b.sample_ps(2000, 5, flags)):
        assert (x == y) if isinstance(x, dict) else np.array_equal(x, y)


def _same_c(text, n, seed, launcher_new=_fast):
    """structured vs xk_law_multi (forced) on the same shots: the cspace ids (dim C and C's key) equal per shot"""
    s = xtim.XdemSampler(text)
    bs = GpuBackend(s._s, launcher_new(), structured=True)
    bg = GpuBackend(s._s, launcher_new(), structured=False, block=bs._block)   # the same stream (blocks of shots)
    assert bs.law_path == "multi-word structured" and bg.law_path == "multi-word"
    ds, os_, cs = bs.sample(n, seed)
    dg, og, cg = bg.sample(n, seed)
    assert np.array_equal(cs, cg), np.flatnonzero(cs != cg)[:5]
    assert (cs != 0).mean() > 0.2   # a nontrivial C on many shots
    return (ds, os_), (dg, og)


@pytest.mark.parametrize("k, signs", GEN, ids=GEN_IDS)
def test_multi_word_structured_same_c(k, signs):
    rs, rg = _same_c(_gen(k, signs), 50_000 if _metal() else 3000, 11)
    # the rows: equal up to the coin shift (most shots equal; the distribution is tested against the CPU below)
    eq = (np.all(rs[0] == rg[0], axis=1) & np.all(rs[1] == rg[1], axis=1)).mean()
    assert eq > 0.5


@pytest.mark.parametrize("k, signs", [(70, True), (130, True)], ids=["70_signs", "130_signs"])
@pytest.mark.parametrize("kind", [None, "half"])
def test_multi_word_structured_estimate_equals_sample_then_score(k, signs, kind):
    """the zero decoder's verdict on the structured law's own rows: estimate's counts (NOKEY, the membership test in
    W') = sample's / sample_ps's then score (the key, on the host)"""
    s = xtim.XdemSampler(_gen(k, signs))
    O, n = s.num_observables, 6000   # enough zero-decoder passes, of several dims C, under the half mask
    flags = None if kind is None else _mask(s.num_detectors, kind)
    b = GpuBackend(s._s, RefLauncher(), structured=True)
    assert b.law_path == "multi-word structured"
    if flags is None:
        _, obs, cs = b.sample(n, 9, first_block=2)
        passed = n
    else:
        _, obs, cs, counts = b.sample_ps(n, 9, flags, first_block=2)
        passed = counts["passed"]
    st = XdemStatus(obs, cs, None, s, s._s.num_cspaces())
    p = s.score(st, np.zeros((len(cs), O), bool))
    want = _hist_of_scores(p)
    assert want.sum() > 0 and len(set(np.flatnonzero(want))) > 1
    att, got, h = GpuBackend(s._s, RefLauncher(), structured=True).estimate(n, 9, flags, first_block=2)
    assert (att, got) == (n, passed)
    assert len(h) >= len(want) and not h[len(want):].any()   # dim C up to rho here: the counts past 64 are 0
    assert h[:len(want)].tolist() == want.tolist()
    assert sum(Fraction(int(c), 2**j) for j, c in enumerate(h)) == sum(Fraction(x) for x in p)


@pytest.mark.parametrize("k, signs", [(70, True), (130, True)], ids=["70_signs", "130_signs"])
@pytest.mark.parametrize("kind", [None, "half"])
def test_multi_word_structured_metal_equals_reference(k, signs, kind):
    if _metal() is None:
        pytest.skip("no Metal device")
    from xtim.gpu.metal import MetalLauncher
    text = _gen(k, signs)
    ref = C.entry_outputs(text, kind, RefLauncher, 3000, 21, structured=True)
    got = C.entry_outputs(text, kind, MetalLauncher, 3000, 21, structured=True)
    assert ref.keys() == got.keys()
    for name in ref:
        assert ref[name].tobytes() == got[name].tobytes(), name


@pytest.mark.parametrize("k, signs", [(70, True), (130, True)], ids=["70_signs", "130_signs"])
def test_multi_word_structured_statistics_vs_cpu(k, signs):
    shots = 1_000_000 if _metal() else 50_000
    r = stat_compare(_gen(k, signs), shots, 83_000 + k, 84_000 + k, launcher=_fast(), structured=True)
    assert r["n_tests"] >= r["D"]
    assert r["max_abs_z"] <= Z_MAX, r


def test_multi_word_structured_cuda_program_compiles_as_host_cxx_syntax():
    """the CUDA program with the structured multi-word law compiled in (see test_gpu_cuda.py's syntax check)"""
    if cxx() is None:
        pytest.skip("no c++ on PATH")
    from xtim.gpu import cuda
    s = xtim.XdemSampler(_gen(70, True))
    b = GpuBackend(s._s, RefLauncher())
    text = cuda.program(HEADER.read_text(), gt.capacities(b.tables, B, b._pair_cap, struct=b._plan))
    assert "#define XK_CAP_ST_W" in text
    ok, log = cxx_syntax_check(text)
    assert ok, log


# ------------------------------------------------------------------------------------------- the merlin_bh* files


@pytest.mark.parametrize("name", ["bh70", pytest.param("bh100", marks=pytest.mark.slow)])
def test_paper_bh_multi_word_structured_same_c_and_statistics(name):
    x = _paper(name)
    t = xtim.XdemSampler(x)._s.device_tables()
    p, why = gs.plan(t)
    assert p is not None and p.sw == 2, why
    assert (p.nG, p.rD, len(p.nonleaf)) == (3, 3, 8)
    _same_c(x, 200_000 if _metal() else 1000, 13)
    if _metal() is not None:
        r = stat_compare(x, 1_000_000, 93_000, 94_000, launcher=_fast(), structured=True)
        assert r["max_abs_z"] <= Z_MAX, r


# ------------------------------------------------------------------------------------------------ no non-leaf axes

_GUARDED_SCRIPT = r"""
import ctypes, mmap, sys
import numpy as np
# -I drops PYTHONPATH and the script's directory: put the tests and the repo root (xtim run from the source
# tree, not installed) on the path explicitly
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import xtim
from xtim.gpu import structure as gs, tables as gt
import _gpu_compare as C
from _gpu_ref import RefLauncher
from _xdem_synth import bh_like_text

class Guarded(RefLauncher):
    # tab64 is copied to end exactly at the end of the structured table, right before a PROT_NONE page: a read past
    # the table is a SIGSEGV
    def upload(self, buffers):
        super().upload(buffers)
        d = self._buf["dims"]
        end = int(d[gt.XK_DIM_OFF_ST]) + len(gs.st_table(PLAN))
        n = len(self._buf["tab64"])
        assert end == n - 1, (end, n)   # the table is the arena's last (the arena's spare word is cut off)
        pages = -(-8 * end // mmap.PAGESIZE)
        self.mm = mmap.mmap(-1, (pages + 1) * mmap.PAGESIZE)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        libc = ctypes.CDLL(None)
        assert libc.mprotect(ctypes.c_void_p(base + pages * mmap.PAGESIZE), mmap.PAGESIZE, 0) == 0
        a = np.frombuffer(self.mm, np.uint64, end, pages * mmap.PAGESIZE - 8 * end)
        a[:] = self._buf["tab64"][:end]
        self._buf["tab64"] = a

s = xtim.XdemSampler(bh_like_text(70, 300, seed=70))
t = s._s.device_tables()
plan, why = gs.plan(t)
t = gs.renumber(t, plan)
PLAN = plan._replace(nonleaf=[])   # no non-leaf axes: dims[XK_DIM_ST_NN] = 0
E = C.pair_cap(t)
shots = 256
buf = gt.pack(t, block=C.B, pair_cap=E, shots=shots, seed=3, struct=PLAN,
              mode=gt.XK_MODE_EMIT | gt.XK_MODE_DEFER)
assert buf["dims"][gt.XK_DIM_ST_NN] == 0 and gt.capacities(t, C.B, E, struct=PLAN)["XK_CAP_ST_N"] == 1
out = C.run_kernels(Guarded(), t, buf, E, shots, PLAN)
assert "resolve_law" in out["_kernels"] and out["lawq"][0] > 0   # the multi-word structured law ran
print("ok")
"""


def test_structured_mw_law_reads_nothing_past_the_table_without_nonleaf_axes():
    """A multi-word structured table with no non-leaf axes (XK_CAP_ST_N is 1 there): the law's loops over the
    non-leaf axes read nothing past the table, which is the arena's last. The reference build runs on a copy of the
    arena that ends exactly at the table, with a PROT_NONE page after it, in a subprocess (a stray read kills it)."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-I", "-c", _GUARDED_SCRIPT, str(Path(__file__).resolve().parent),
                        str(Path(__file__).resolve().parents[1])],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip() == "ok", (r.returncode, r.stdout[-500:], r.stderr[-1500:])
