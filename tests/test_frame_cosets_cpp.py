"""The compiled sector table (3.1.3, `_xtim.plan_frame_cosets`, cpp/include/qeccore/
frame_cosets.hpp) == the pure-Python table (`xtim.frames.plan_frame_cosets_numpy`, the 3.1.2
production path) — same sectors, same first-encounter order, same fields and dtypes EXACTLY, and
`weight` within 1e-12 relative (EXP-fix #5, owner Q4 2026-10-21: the compiled table sums each coset
in closed form — an exact Z4 Gauss sum over the 2^rank cosets — instead of the 2^|supp a| walk, so
its weight is no longer the oracle's fixed-order float sum; sector-skip flips at w < 1e-12 are
reported, not pinned) — on

  * every plan of every retained shot of the multi-sector fixtures (test_branch_frames),
  * the synthetic masks |supp a| = 1..9 on the 10-qubit magic and stabilizer references,
  * the 287 REAL a-masks of adaptq's u2 unit (qrm magic producer, tests/data/
    u2_qrm_magic_amasks.npy, captured 2026-09-14 from a 20 000-shot batch at p = 1e-3) on
    the fixture's own bare state (the same reference adaptq builds: verified equal),
  * a seeded random sweep of masks on the three data fixtures' bare states,

and both against the reference oracle `plan_frame_cosets` (tolerance, as the 3.1.2 pin).
Plus: the two refusal paths raise the SAME exception type on both backends, the compiled
path is deterministic (byte-identical across calls), the dispatcher's backend selection
(a stale extension is REFUSED, never silently served by the pure-Python path; the backend is a
test-only argument, S7: no environment lever), and the sampling HOT PATH untouched (the record stream of the
three data fixtures hashes to the values captured on the 3.1.2 build before this change).

The numpy-vs-numpy and cpp-vs-cpp comparisons stay BIT FOR BIT (the same arithmetic in a fixed
order, review MAJOR-2).

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).
"""
import hashlib
import os
import struct

import numpy as np
import pytest

from xtim import _xtim, compile_twirl_sampler, frames

from test_branch_frames import (MULTI_SECTOR_CASES, _SYNTH_REFS, _assert_cosets_equal,
                                _bare_copy, _text)

DATA = os.path.join(os.path.dirname(__file__), "data")


def _fixture_text(name):
    with open(os.path.join(DATA, name + ".stim")) as fh:
        return fh.read()


def _bits(x):
    return struct.pack("<d", float(x))


def _assert_bitwise(cpp, npy, ctx):
    """cpp == numpy path: structure, dtypes, ints and the weight's BITS."""
    assert isinstance(cpp, list) and len(cpp) == len(npy), (ctx, len(cpp), len(npy))
    for c_c, c_n in zip(cpp, npy):
        assert list(c_c.keys()) == list(c_n.keys()) == ["vz", "fx", "fz", "logical_flip", "weight", "frame_copy"], ctx
        for key in ("vz", "fx", "fz"):
            a, b = c_c[key], c_n[key]
            assert isinstance(a, np.ndarray) and a.dtype == np.uint8 and a.shape == b.shape, (ctx, key)
            assert a.tobytes() == np.asarray(b, np.uint8).tobytes(), (ctx, key)
        assert type(c_c["frame_copy"]) is bool and c_c["frame_copy"] == c_n["frame_copy"], ctx
        assert c_c["logical_flip"] == c_n["logical_flip"], ctx
        if c_n["logical_flip"] is not None:
            xb, zb = c_c["logical_flip"]
            assert isinstance(c_c["logical_flip"], tuple) and isinstance(xb, list) and isinstance(zb, list), ctx
            assert all(type(v) is int for v in xb + zb), ctx
        assert type(c_c["weight"]) is float, ctx
        assert _bits(c_c["weight"]) == _bits(c_n["weight"]), (ctx, c_c["weight"].hex(), c_n["weight"].hex())


SKIP_FLIPS = []   # (ctx, weight) of sectors present on one side only, all below the 1e-12 skip threshold's reach


def _assert_close(cpp, npy, ctx, rtol=1e-12):
    """cpp == numpy path: structure, dtypes and ints exact; weight within rtol (Q4). A sector that one side
    skips at its w < 1e-12 test may be present on the other with a weight of that order: reported."""
    keys_c = [c["vz"].tobytes() for c in cpp]
    keys_n = [np.asarray(c["vz"], np.uint8).tobytes() for c in npy]
    if keys_c != keys_n:
        common = set(keys_c) & set(keys_n)
        for c in [c for c, k in zip(cpp, keys_c) if k not in common] + \
                 [c for c, k in zip(npy, keys_n) if k not in common]:
            assert c["weight"] < 1e-10, (ctx, "sector set differs beyond a skip flip", c["weight"])
            SKIP_FLIPS.append((ctx, c["weight"]))
        cpp = [c for c, k in zip(cpp, keys_c) if k in common]
        npy = [c for c, k in zip(npy, keys_n) if k in common]
    assert isinstance(cpp, list) and len(cpp) == len(npy), (ctx, len(cpp), len(npy))
    for c_c, c_n in zip(cpp, npy):
        assert list(c_c.keys()) == list(c_n.keys()) == ["vz", "fx", "fz", "logical_flip", "weight", "frame_copy"], ctx
        for key in ("vz", "fx", "fz"):
            a, b = c_c[key], c_n[key]
            assert isinstance(a, np.ndarray) and a.dtype == np.uint8 and a.shape == b.shape, (ctx, key)
            assert a.tobytes() == np.asarray(b, np.uint8).tobytes(), (ctx, key)
        assert type(c_c["frame_copy"]) is bool and c_c["frame_copy"] == c_n["frame_copy"], ctx
        assert c_c["logical_flip"] == c_n["logical_flip"], ctx
        assert type(c_c["weight"]) is float, ctx
        w_c, w_n = c_c["weight"], c_n["weight"]
        assert abs(w_c - w_n) <= rtol * abs(w_n) + 1e-15, (ctx, w_c, w_n)


def test_extension_has_the_compiled_table():
    assert hasattr(_xtim, "plan_frame_cosets")
    assert frames._CPP_PLAN_FRAME_COSETS is not None


def _both(fr_ref, a, ctx, *, reference=True):
    cpp = frames.plan_frame_cosets_fast(fr_ref, a, backend="cpp")
    npy = frames.plan_frame_cosets_fast(fr_ref, a, backend="numpy")
    _assert_close(cpp, npy, ctx)
    if reference:
        ref_table = frames.plan_frame_cosets(fr_ref, a)
        _assert_cosets_equal(cpp, ref_table, ctx)             # the 3.1.2 tolerance leg (1e-12)
        _assert_bitwise(npy, ref_table, (ctx, "reference"))   # the two Python oracles: same arithmetic → same bits
    return cpp


@pytest.mark.parametrize("name", sorted(_SYNTH_REFS))
def test_cpp_equals_numpy_on_synthetic_masks(name):
    lines, n_data = _SYNTH_REFS[name]
    fr_ref = _bare_copy(_text(lines, n_data)).branch_frames()
    n = int(fr_ref["n"])
    rng = np.random.default_rng(20260908)
    masks = [np.zeros(n, np.uint8)]
    for size in range(1, 10):
        m = np.zeros(n, np.uint8); m[:size] = 1
        masks.append(m)
        for _ in range(2):
            m2 = np.zeros(n, np.uint8)
            m2[rng.choice(n_data, size=size, replace=False)] = 1
            masks.append(m2)
    for j, a in enumerate(masks):
        _both(fr_ref, a, (name, j, int(a.sum())))


def test_cpp_equals_numpy_on_the_287_real_u2_plans():
    masks = np.load(os.path.join(DATA, "u2_qrm_magic_amasks.npy"))
    fr_ref = _bare_copy(_fixture_text("adaptq_qrm_magic_producer")).branch_frames()
    assert masks.shape == (287, int(fr_ref["n"])) and int(fr_ref["n"]) == 78
    assert len(fr_ref["coeff"]) == 2 and len(fr_ref["free"]) == 1
    sizes = masks.sum(axis=1)
    assert sizes.max() == 9 and len({int(s) for s in sizes}) == 10
    n_multi = 0
    for j, a in enumerate(masks):
        tab = _both(fr_ref, a, ("u2", j, int(a.sum())), reference=int(a.sum()) <= 7)
        assert all(c["frame_copy"] for c in tab), j          # adaptq's refusal never fires on u2
        n_multi += len(tab) > 1
    assert n_multi > 100                                       # the coherent plans are the point


def test_cpp_equals_numpy_random_sweep_on_data_fixtures():
    rng = np.random.default_rng(20260914)
    for name in ("adaptq_born5_dec_enum_producer", "adaptq_qrm_magic_producer", "adaptq_steane_h_producer"):
        fr_ref = _bare_copy(_fixture_text(name)).branch_frames()
        n = int(fr_ref["n"])
        for j in range(60):
            size = int(rng.integers(0, 10))
            a = np.zeros(n, np.uint8)
            if size:
                a[rng.choice(n, size=size, replace=False)] = 1
            _both(fr_ref, a, (name, j, size), reference=(j % 6 == 0))


def test_cpp_is_deterministic_and_the_default_backend():
    fr_ref = _bare_copy(_text(*_SYNTH_REFS["magic10"])).branch_frames()
    a = np.zeros(int(fr_ref["n"]), np.uint8); a[[0, 2, 3, 7]] = 1
    t1 = frames.plan_frame_cosets_fast(fr_ref, a, backend="cpp")
    t2 = frames.plan_frame_cosets_fast(fr_ref, a, backend="cpp")
    _assert_bitwise(t1, t2, "det")
    _assert_bitwise(frames.plan_frame_cosets_fast(fr_ref, a), t1, "auto→cpp")
    # the numpy path is selectable (explicitly) and is the 3.1.2 body with the portable
    # weight arithmetic (review MAJOR-2)
    _assert_close(t1, frames.plan_frame_cosets_numpy(fr_ref, a), "numpy body")
    _assert_close(t1, frames.plan_frame_cosets_fast(fr_ref, a, backend="numpy"), "backend=numpy")
    with pytest.raises(ValueError, match="unknown backend"):
        frames.plan_frame_cosets_fast(fr_ref, a, backend="fortran")
    # the memoizing wrapper serves the compiled table (adaptq's call surface is unchanged)
    cache = {}
    got = frames.plan_frame_cosets_cached(fr_ref, a, cache)
    _assert_bitwise(got, t1, "cached")
    assert frames.plan_frame_cosets_cached(fr_ref, a, cache) is got


def test_stale_extension_is_refused_never_served_by_the_numpy_path(monkeypatch):
    """Review MAJOR-1: a `.so` without the compiled table (built before 3.1.3) or at another
    FRAME_COSETS_ABI is a STALE build; `auto` (the default) and `cpp` both raise the named
    StaleExtensionError (a RuntimeError naming the rebuild), the pure-Python path is served
    only on an EXPLICIT backend="numpy".  A silent 20× fallback is not a null option."""
    fr_ref = _bare_copy(_text(*_SYNTH_REFS["stab10"])).branch_frames()
    a = np.zeros(int(fr_ref["n"]), np.uint8); a[0] = 1
    assert _xtim.FRAME_COSETS_ABI == frames.FRAME_COSETS_ABI
    ref_table = frames.plan_frame_cosets_numpy(fr_ref, a)
    # (1) the symbol is missing from the extension
    monkeypatch.delattr(_xtim, "plan_frame_cosets")
    fn, why = frames._resolve_compiled_table()
    assert fn is None and "no plan_frame_cosets" in why and "stale" in why
    monkeypatch.setattr(frames, "_CPP_PLAN_FRAME_COSETS", fn)
    monkeypatch.setattr(frames, "_CPP_UNAVAILABLE_REASON", why)
    for backend in ("auto", "cpp"):
        with pytest.raises(frames.StaleExtensionError, match="no plan_frame_cosets.*stale.*rebuild") as ei:
            frames.plan_frame_cosets_fast(fr_ref, a, backend=backend)
        assert isinstance(ei.value, RuntimeError)
    _assert_bitwise(frames.plan_frame_cosets_fast(fr_ref, a, backend="numpy"), ref_table, "explicit numpy")
    monkeypatch.undo()
    # (2) the symbol is there but at another ABI
    monkeypatch.setattr(_xtim, "FRAME_COSETS_ABI", frames.FRAME_COSETS_ABI + 1)
    fn, why = frames._resolve_compiled_table()
    assert fn is None and "FRAME_COSETS_ABI" in why and "stale" in why
    monkeypatch.setattr(frames, "_CPP_PLAN_FRAME_COSETS", fn)
    monkeypatch.setattr(frames, "_CPP_UNAVAILABLE_REASON", why)
    with pytest.raises(frames.StaleExtensionError, match="expects %d" % frames.FRAME_COSETS_ABI):
        frames.plan_frame_cosets_fast(fr_ref, a)
    _assert_bitwise(frames.plan_frame_cosets_fast(fr_ref, a, backend="numpy"), ref_table, "explicit numpy")
    monkeypatch.undo()
    # (3) the live extension resolves and the two backends agree
    fn, why = frames._resolve_compiled_table()
    assert fn is _xtim.plan_frame_cosets and why is None
    _assert_close(frames.plan_frame_cosets_fast(fr_ref, a), ref_table, "auto == numpy")


def _raises_same(fr_bad, a, exc):
    with pytest.raises(exc):
        frames.plan_frame_cosets_fast(fr_bad, a, backend="numpy")
    with pytest.raises(exc):
        frames.plan_frame_cosets_fast(fr_bad, a, backend="cpp")


def test_refusals_match_the_numpy_path():
    """(1) a null-space basis read that is not ±1 → ValueError on both; (2) an export that is
    not a full frame → AssertionError on both.  Corruptions are applied to a COPY of a real
    export (magic10) and checked to actually fire on the numpy path first."""
    fr_ref = _bare_copy(_text(*_SYNTH_REFS["magic10"])).branch_frames()
    n = int(fr_ref["n"])
    a = np.zeros(n, np.uint8); a[1] = 1                    # Z_1 commutes with the GHZ Z-generators
    # (1) put an i on the generator that Z_1's null-space read multiplies: the read becomes ±i
    bad = dict(fr_ref)
    sp = np.array(fr_ref["stab_phase"], np.int8).copy()
    sx = np.asarray(fr_ref["stab_x"], np.uint8); dx = np.asarray(fr_ref["destab_x"], np.uint8)
    dz = np.asarray(fr_ref["destab_z"], np.uint8)
    # the generators whose destabilizer anticommutes with Z_1 form the product R; flip one phase
    hit = [g for g in range(n) if int(dx[g][1]) & 1]
    assert hit
    sp[hit[0]] = (sp[hit[0]] + 1) & 3
    bad["stab_phase"] = sp
    _raises_same(bad, a, ValueError)
    # (2) drop the destabilizer that produces Z_1 so R != P: the export is not a full frame
    bad2 = dict(fr_ref)
    dx2 = dx.copy(); dz2 = dz.copy()
    dx2[hit[0]] = 0; dz2[hit[0]] = 0
    bad2["destab_x"] = dx2; bad2["destab_z"] = dz2
    _raises_same(bad2, a, AssertionError)
    del sx


# J-b2 J1b: the PRODUCTION leg (plan r2.1 J1b). The same three fixtures' sampled stream on the PUBLIC sampler's
# layout-3 barrier (dets, obs, decisions, record_keys, record_group_ids, per-shot outcomes and plan contents, and
# branch_frames of every 64th retained state), pinned at b6826c1 (xtim-engine-13). The legacy pins above (the sigma-era
# frames export) retire with the legacy sampler at J2.
_STREAM_HASHES_L3 = {
    "adaptq_born5_dec_enum_producer": "a20b95180ee3d5677adfcee14eb4e4da2bb07fe09c5b997d36b7c414d2ed3e62",
    "adaptq_qrm_magic_producer": "c3e25f8088c886acb205ffa3416aaf97493d92aedbc36bf386f6a657fb5c950a",
    "adaptq_steane_h_producer": "46fc554d79f6ebb9135a7be8acb5311be2a18f1efee07b6307f8f5821755ef8c",
}


@pytest.mark.parametrize("name", sorted(_STREAM_HASHES_L3))
def test_sampling_stream_pinned_public(name):
    from _public import public_sampler
    s = public_sampler(_fixture_text(name))
    h = hashlib.sha256()
    for seed in (7, 11):
        buf = s.sample_barrier(256, seed=seed)
        for a in (buf.dets(), buf.obs(), buf.decisions(), buf.record_keys(), buf.record_group_ids()):
            arr = np.ascontiguousarray(np.asarray(a))
            h.update(str(arr.dtype).encode()); h.update(str(arr.shape).encode()); h.update(arr.tobytes())
        for i in range(256):
            h.update(bytes(np.asarray(buf.outcomes(i), np.uint8))); h.update(repr(buf.plan_content(i)).encode())
        for i in range(0, 256, 64):
            bf = buf.branch_frames(i)
            for k in sorted(bf):
                arr = np.ascontiguousarray(np.asarray(bf[k])); h.update(k.encode()); h.update(arr.tobytes())
    assert h.hexdigest() == _STREAM_HASHES_L3[name]
