"""The structured one-word law (xtim/gpu/structure.py on the host, xk_law_struct in xdem_kernels.h).

Files: Bravyi-Haah-like texts generated here (tests/_xdem_synth.bh_like_text: the merlin_bh* files' axis pattern for
any number of leaves k, so both generic forms are covered: k = 12 gives rho = 15, the register form; k = 36 gives
rho = 39, the elimination form), with signs (the phase's linear term; one type split into several records) or without.
No fixture is committed and nothing depends on the merlin_bh* benchmark circuits (arXiv:2610.06811, not in this
repository), except the tests marked as such, which run on merlin_bh12 / bh36 (and bh50, slow) from
$XTIM_MERLIN_CIRCUITS (no default) and skip when it is unset or the files are absent.

Detector: accepts the generated and the merlin bh files, refuses cultivation_d5 and a random (unstructured) file; a
structured file whose T is made non-linear at the tables level (no valid xdem text is uncertified: §3.5(b)) is refused
and runs the generic law. The path chosen (GpuBackend.law_path) and its defaults.

Exact: the structured law gives the generic law's C per shot (the cspace ids of `sample`, so dim C and key) and the
zero decoder's verdict on its own rows (estimate's counts = sample's then score, kernel flag bit 2 = score > 0); the
renumbering leaves the generic law's outputs byte-identical; Metal equals the reference byte for byte. Statistics: the
structured sampler vs the CPU sampler (the z-scores of test_xdem_cross_route.py, Z_MAX), rows being equal to the
generic law's only up to the coin shift.
"""
from __future__ import annotations

import os
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim.gpu import structure as gs
from xtim.gpu import tables as gt
from xtim.gpu.backend import DEFAULT_BLOCK, STRUCT_MIN_RHO, GpuBackend
from xtim.xdem import XdemStatus

import _gpu_compare as C
from _fake_cupy import cxx, cxx_syntax_check
from _gpu_ref import RefLauncher
from _xdem_synth import bh_like_text, random_text
from test_gpu_backend import stat_compare
from test_gpu_estimate import _check_flags_vs_score, _hist_of_scores
from test_gpu_postselect import _mask
from test_xdem_cross_route import Z_MAX

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "xtim" / "gpu" / "xdem_kernels.h"
B = DEFAULT_BLOCK   # the backend's block (K1's shots per group)
_MERLIN = os.environ.get("XTIM_MERLIN_CIRCUITS")
PAPER = Path(_MERLIN).expanduser() if _MERLIN else None
GEN = [(12, False), (12, True), (36, False), (36, True)]
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
def test_detector_accepts_bh_like(k, signs):
    t = xtim.XdemSampler(_gen(k, signs))._s.device_tables()
    d = gs.detect(t)
    assert d["structure"], d["reason"]
    assert d["rD"] == 3 and len(d["core_gauge"]) == 3 and len(d["leaves"]) >= k - 1
    assert all(g == f for g, f in d["leaves"])
    assert gs.certified(t)
    p, why = gs.plan(t)
    assert p is not None, why
    assert (p.nG, p.rD, len(p.nonleaf)) == (3, 3, 8)
    assert len(p.records) == (10 if signs else 8)
    st = gs.st_table(p)
    assert int(st[3]) == int(signs) and int(st[0]) == p.Lall
    # the renumbering: non-leaf axes first, each record a run of consecutive new axes at consecutive slots
    assert sorted(p.perm) == list(range(int(t["A"])))
    at = len(p.nonleaf)
    for off, n, *_ in sorted(p.records):
        assert off == at
        at += n


@pytest.mark.parametrize("name", ["bh12", "bh36"])
def test_detector_accepts_paper_bh(name):
    t = xtim.XdemSampler(_paper(name))._s.device_tables()
    p, why = gs.plan(t)
    assert p is not None, why
    assert (p.nG, p.rD, len(p.nonleaf), len(p.records)) == (3, 3, 8, 8)


def test_detector_refuses_cultivation_and_random():
    t = xtim.XdemSampler((ROOT / "examples" / "xdem" / "cultivation_d5.xdem").read_text())._s.device_tables()
    assert t["fast"] and int(t["A"])
    p, why = gs.plan(t)
    assert p is None and why.startswith("no structure"), why
    # a random valid one-word file (img linear in kappa: certified, but no leaves)
    x = random_text(random.Random(5), D=24, O=3, r=12, m=6, A=40, L=300)
    t = xtim.XdemSampler(x)._s.device_tables()
    assert t["fast"] and int(t["A"]) and int(t["coord_words"]) == 1
    assert gs.certified(t)
    p, why = gs.plan(t)
    assert p is None and why.startswith("no structure"), why
    b = GpuBackend(xtim.XdemSampler(x)._s, RefLauncher(), structured=True)
    assert b.law_path == "generic" and b.law_path_reason == why


def test_uncertified_falls_back_to_generic():
    """T made non-linear in kappa (one leaf axis's beta_D part flipped; a tables-level change, no valid text has it):
    the structure is still detected, the certificate fails, the backend runs the generic law"""
    s = xtim.XdemSampler(_gen(12, False))
    t = dict(s._s.device_tables())
    p, _ = gs.plan(t)
    a = p.perm.index(p.records[0][0])   # the original index of the first record's first axis
    Tc = np.array(t["Tc"]).copy()
    Tc[a] ^= np.array(1, Tc.dtype)
    t["Tc"] = Tc
    assert gs.detect(t)["structure"] and not gs.certified(t)
    q, why = gs.plan(t)
    assert q is None and why.startswith("not certified"), why
    b = GpuBackend(_Tables(s._s, t), RefLauncher(), structured=True)
    assert b.law_path == "generic" and b.law_path_reason == why and b._plan is None


def test_law_path_and_defaults():
    s12 = xtim.XdemSampler(_gen(12, False))._s
    s36 = xtim.XdemSampler(_gen(36, False))._s
    assert int(s12.device_tables()["rho"]) <= STRUCT_MIN_RHO < int(s36.device_tables()["rho"])
    b = GpuBackend(s12, RefLauncher())
    assert (b.law_path, b.law_path_reason) == ("generic", f"rho <= {STRUCT_MIN_RHO}")
    b = GpuBackend(s12, RefLauncher(), structured=True)
    assert (b.law_path, b.law_path_reason) == ("structured", "")
    b = GpuBackend(s36, RefLauncher())
    assert b.law_path == "structured"
    assert b._law is False and b._defer   # the structured law's defaults: no law table, the queue
    # K1's block: DEFAULT_BLOCK on every law path
    from xtim.gpu import backend as GB
    assert b._block == GB.DEFAULT_BLOCK
    assert GpuBackend(s36, RefLauncher(), structured=False)._block == GB.DEFAULT_BLOCK
    assert GpuBackend(s36, RefLauncher(), block=48)._block == 48
    b = GpuBackend(s36, RefLauncher(), defer=False)
    assert b.law_path == "structured" and b._law is False and not b._defer
    # the law table holds the generic law's laws: law_table=True means the generic law
    b = GpuBackend(s36, RefLauncher(), law_table=True)
    assert (b.law_path, b.law_path_reason) == ("generic", "law_table=True") and b._law is not False
    with pytest.raises(ValueError, match="law table"):
        GpuBackend(s36, RefLauncher(), structured=True, law_table=True)
    b = GpuBackend(s36, RefLauncher(), structured=False)
    assert (b.law_path, b.law_path_reason) == ("generic", "structured=False")
    r65 = xtim.XdemSampler((ROOT / "tests" / "data" / "xdem" / "synth" / "boundary_r65.xdem").read_text())._s
    assert GpuBackend(r65, RefLauncher(), structured=True).law_path == "multi-word"


# ------------------------------------------------------------------------------------------------ exact


@pytest.mark.parametrize("k, signs", GEN, ids=GEN_IDS)
def test_renumbering_keeps_the_generic_law(k, signs):
    """the generic law on the renumbered tables: every output byte the same"""
    s = xtim.XdemSampler(_gen(k, signs))
    t = s._s.device_tables()
    p, _ = gs.plan(t)
    flags = _mask(s.num_detectors, "half")
    a = GpuBackend(s._s, RefLauncher(), structured=False)
    b = GpuBackend(_Tables(s._s, gs.renumber(t, p)), RefLauncher(), structured=False)
    for x, y in zip(a.sample(3000, 4), b.sample(3000, 4)):
        assert np.array_equal(x, y)
    for x, y in zip(a.sample_ps(3000, 5, flags), b.sample_ps(3000, 5, flags)):
        assert (x == y) if isinstance(x, dict) else np.array_equal(x, y)


def _same_c(text, n, seed, launcher_new=_fast, **kw):
    """structured vs generic (forced) on the same shots: the cspace ids (dim C and C's key) equal per shot"""
    s = xtim.XdemSampler(text)
    bs = GpuBackend(s._s, launcher_new(), structured=True, **kw)
    bg = GpuBackend(s._s, launcher_new(), structured=False, block=bs._block)   # the same stream (blocks of shots)
    assert bs.law_path == "structured" and bg.law_path == "generic"
    ds, os_, cs = bs.sample(n, seed)
    dg, og, cg = bg.sample(n, seed)
    assert np.array_equal(cs, cg), np.flatnonzero(cs != cg)[:5]
    assert (cs != 0).mean() > 0.2   # a nontrivial C on many shots
    return (ds, os_), (dg, og)


@pytest.mark.parametrize("k, signs", GEN, ids=GEN_IDS)
@pytest.mark.parametrize("defer", [True, False])
def test_structured_same_c_as_generic(k, signs, defer):
    rs, rg = _same_c(_gen(k, signs), 50_000 if _metal() else 4000, 11, defer=defer)
    # the rows: equal up to the coin shift (most shots equal; the distribution is tested against the CPU below)
    eq = (np.all(rs[0] == rg[0], axis=1) & np.all(rs[1] == rg[1], axis=1)).mean()
    assert eq > 0.5


@pytest.mark.parametrize("k, signs", GEN, ids=GEN_IDS)
@pytest.mark.parametrize("kind", [None, "half"])
def test_structured_estimate_equals_sample_then_score(k, signs, kind):
    """the zero decoder's verdict on the structured law's own rows: estimate's counts (NOKEY, the membership test in
    W') = sample's / sample_ps's then score (the key, on the host)"""
    s = xtim.XdemSampler(_gen(k, signs))
    O, n = s.num_observables, 4000
    flags = None if kind is None else _mask(s.num_detectors, kind)
    b = GpuBackend(s._s, RefLauncher(), structured=True)
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
    for defer in (True, False):
        att, got, h = GpuBackend(s._s, RefLauncher(), structured=True, defer=defer).estimate(n, 9, flags,
                                                                                              first_block=2)
        assert (att, got) == (n, passed)
        assert h.tolist() == want.tolist()
        assert sum(Fraction(int(c), 2**j) for j, c in enumerate(h)) == sum(Fraction(x) for x in p)


@pytest.mark.parametrize("k, signs", GEN, ids=GEN_IDS)
def test_structured_zero_flag_matches_score(k, signs):
    """kernel level (walk, resolve on the reference build): flag bit 2 == score(zeros) > 0, the passing score 2^-cdim"""
    s = xtim.XdemSampler(_gen(k, signs))
    b = GpuBackend(s._s, RefLauncher(), structured=True)
    t, n, E = b.tables, 20 * B - 11, b._pair_cap
    buf = gt.pack(t, block=B, pair_cap=E, shots=n, seed=41, struct=b._plan)
    L = RefLauncher()
    L.compile(HEADER.read_text(), gt.capacities(t, B, E, struct=b._plan))
    L.upload(buf)
    L.run("walk", int(buf["dims"][gt.XK_DIM_NBLOCKS]))
    L.run("resolve", n)
    p, cdim = _check_flags_vs_score(s, b, L.download("rows"), L.download("ckey"), L.download("cdim"),
                                    L.download("flag"))
    assert (p > 0).any() and len(set(cdim.tolist())) > 1


@pytest.mark.parametrize("k, signs", GEN, ids=GEN_IDS)
@pytest.mark.parametrize("kind", [None, "half"])
def test_structured_metal_equals_reference(k, signs, kind):
    if _metal() is None:
        pytest.skip("no Metal device")
    from xtim.gpu.metal import MetalLauncher
    text = _gen(k, signs)
    first = None
    for defer in (True, False):   # K2's queue: not an output byte changed either
        ref = C.entry_outputs(text, kind, RefLauncher, 3000, 21, structured=True, defer=defer)
        got = C.entry_outputs(text, kind, MetalLauncher, 3000, 21, structured=True, defer=defer)
        first = first or ref
        assert ref.keys() == got.keys() == first.keys()
        for name in ref:
            assert ref[name].tobytes() == got[name].tobytes() == first[name].tobytes(), (name, defer)


@pytest.mark.parametrize("k, signs", [(12, True), (36, False), (36, True)], ids=["12_signs", "36", "36_signs"])
def test_structured_statistics_vs_cpu(k, signs):
    shots = 1_000_000 if _metal() else 100_000
    r = stat_compare(_gen(k, signs), shots, 81_000 + k, 82_000 + k, launcher=_fast(), structured=True)
    assert r["n_tests"] >= r["D"]
    assert r["max_abs_z"] <= Z_MAX, r


def test_structured_cuda_program_compiles_as_host_cxx_syntax():
    """the CUDA program with the structured law compiled in (see test_gpu_cuda.py's syntax check)"""
    if cxx() is None:
        pytest.skip("no c++ on PATH")
    from xtim.gpu import cuda
    s = xtim.XdemSampler(_gen(36, True))
    b = GpuBackend(s._s, RefLauncher())
    text = cuda.program(HEADER.read_text(), gt.capacities(b.tables, B, b._pair_cap, struct=b._plan))
    assert "#define XK_CAP_ST_T" in text
    ok, log = cxx_syntax_check(text)
    assert ok, log


# ------------------------------------------------------------------------------------------- the merlin_bh* files


@pytest.mark.parametrize("name", ["bh12", "bh36", pytest.param("bh50", marks=pytest.mark.slow)])
def test_paper_bh_structured_same_c_and_statistics(name):
    x = _paper(name)
    _same_c(x, 200_000 if _metal() else 2000, 13)
    if _metal() is not None:
        r = stat_compare(x, 1_000_000, 91_000, 92_000, launcher=_fast(), structured=True)
        assert r["max_abs_z"] <= Z_MAX, r


# ------------------------------------------------------------------------------------------------ the device C-id table


class _KeyTap(RefLauncher):
    """a reference launcher that keeps each batch's per-shot C keys and dim C (the stream's order)"""

    def __init__(self):
        super().__init__()
        self.keys, self.cdims = [], []

    def run_batch(self, steps):
        super().run_batch(steps)
        n = int(self._buf["dims"][gt.XK_DIM_NSHOTS])
        kw = len(self._buf["ckey"]) // len(self._buf["cdim"]) if len(self._buf["cdim"]) else 1
        self.keys.append(self._buf["ckey"][:n * kw].reshape(n, kw).copy())
        self.cdims.append(self._buf["cdim"][:n].copy())


def test_c_id_table_fills_and_overflows_with_the_same_ids():
    """A Bravyi-Haah-like file with 36 observables has a distinct C on almost every shot: the backend's device table of
    known C keys (gt.CSPACE_TABLE_MAX_KEYS = 4096) fills and the shots past it are resolved on the host. The ids equal
    those of a run with one block per batch (the table grows each batch) and of a repeat, the table holds no key
    twice, and each shot's C (the sampler's cspaces[id]) equals what a fresh CPU sampler makes of the shot's key."""
    text = bh_like_text(36, 300, seed=36)
    shots = 20_000

    def run(**kw):
        s = xtim.XdemSampler(text)
        tap = _KeyTap()
        b = GpuBackend(s._s, tap, **kw)
        d, o, c = b.sample(shots, 7)
        return s, b, tap, d, o, c

    s1, b1, tap1, d1, o1, c1 = run(max_batch_bytes=1)
    assert b1.batches_run == -(-shots // B)   # one block per batch
    assert len(b1._ckeys) == len(b1._cids) == gt.CSPACE_TABLE_MAX_KEYS   # full
    assert len(np.unique(c1)) > gt.CSPACE_TABLE_MAX_KEYS + 1000          # and well past it
    assert len(np.unique(b1._ckeys, axis=0)) == len(b1._ckeys) and len(np.unique(b1._cids)) == len(b1._cids)
    for other in (run(max_batch_bytes=1), run(), run(max_batch_bytes=1 << 20)):
        s2, b2, _, d2, o2, c2 = other
        assert np.array_equal(c1, c2) and np.array_equal(d1, d2) and np.array_equal(o1, o2)
        assert np.array_equal(b1._ckeys, b2._ckeys) and np.array_equal(b1._cids, b2._cids)
    # the same sampler again: the ids it already made
    assert np.array_equal(GpuBackend(s1._s, RefLauncher(), max_batch_bytes=1).sample(shots, 7)[2], c1)
    # a fresh CPU sampler's C of every shot's key
    keys, cdim = np.concatenate(tap1.keys), np.concatenate(tap1.cdims)
    assert np.array_equal(cdim > 0, c1 > 0) and cdim.max() <= gt.dimc_of(s1._s.device_tables())
    fresh = xtim.XdemSampler(text)
    nz = np.flatnonzero(cdim)
    uk, inv = np.unique(keys[nz], axis=0, return_inverse=True)
    fid = np.asarray(fresh._s.cspace_of_keys(np.ascontiguousarray(uk, np.uint64)), np.int64)[inv.reshape(-1)]
    t_ids = np.asarray(fresh._s.cspace_of_keys(np.ascontiguousarray(b1._ckeys, np.uint64)), np.int64)
    ours, theirs = s1._s.cspaces(0, s1._s.num_cspaces()), fresh._s.cspaces(0, fresh._s.num_cspaces())
    for i in nz[::7]:   # every 7th shot with a C (a few thousand)
        assert np.array_equal(ours[c1[i]], theirs[fid[np.searchsorted(nz, i)]]), i
    # the table's keys and ids: each id is the sampler's id of that key's C
    for i in range(0, len(t_ids), 13):
        assert np.array_equal(theirs[t_ids[i]], ours[b1._cids[i]]), i
