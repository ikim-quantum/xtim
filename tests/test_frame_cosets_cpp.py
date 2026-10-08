"""The compiled sector table (3.1.3, `_xtim.plan_frame_cosets`, cpp/include/qeccore/
frame_cosets.hpp) == the pure-Python table (`xtim.frames.plan_frame_cosets_numpy`, the 3.1.2
production path), BIT FOR BIT — same sectors, same first-encounter order, same fields, same
dtypes, the same `weight` double — on

  * every plan of every retained shot of the multi-sector fixtures (test_branch_frames),
  * the synthetic masks |supp a| = 1..9 on the 10-qubit magic and stabilizer references,
  * the 287 REAL a-masks of adaptq's u2 unit (qrm magic producer, tests/data/
    u2_qrm_magic_amasks.npy, captured 2026-09-14 from a 20 000-shot batch at p = 1e-3) on
    the fixture's own bare state (the same reference adaptq builds: verified equal),
  * a seeded random sweep of masks on the three data fixtures' bare states,

and both against the reference oracle `plan_frame_cosets` (tolerance, as the 3.1.2 pin).
Plus: the two refusal paths raise the SAME exception type on both backends, the compiled
path is deterministic (byte-identical across calls), the dispatcher's backend selection
(a stale extension is REFUSED, never silently served by the pure-Python path; the env var
exercised in a fresh process), and the sampling HOT PATH untouched (the record stream of the
three data fixtures hashes to the values captured on the 3.1.2 build before this change).

The `weight` bit pin is a claim about IEEE basic operations in a fixed order (re·re + im·im
per branch, sequential sums — review MAJOR-2), not about numpy's or CPython's reduction
kernels: it holds on every interpreter / numpy / CPU dispatch the package declares.
"""
import hashlib
import os
import struct

import numpy as np
import pytest

from xtim import _xtim, compile_twirl_sampler, frames

from test_branch_frames import (MULTI_SECTOR_CASES, _SYNTH_REFS, _assert_cosets_equal,
                                _bare_copy, _sampler, _text)

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


def test_extension_has_the_compiled_table():
    assert hasattr(_xtim, "plan_frame_cosets")
    assert frames._CPP_PLAN_FRAME_COSETS is not None


def _both(fr_ref, a, ctx, *, reference=True):
    cpp = frames.plan_frame_cosets_fast(fr_ref, a, backend="cpp")
    npy = frames.plan_frame_cosets_fast(fr_ref, a, backend="numpy")
    _assert_bitwise(cpp, npy, ctx)
    if reference:
        ref_table = frames.plan_frame_cosets(fr_ref, a)
        _assert_cosets_equal(cpp, ref_table, ctx)             # the 3.1.2 tolerance leg (1e-12)
        _assert_bitwise(cpp, ref_table, (ctx, "reference"))   # 3.1.3: same arithmetic → same bits
    return cpp


@pytest.mark.parametrize("name", sorted(MULTI_SECTOR_CASES))
def test_cpp_equals_numpy_on_multi_sector_fixtures(name):
    lines, n_data, _ = MULTI_SECTOR_CASES[name]
    text = _text(lines, n_data)
    fr = _sampler(text).sample_barrier(12, 11).frames()
    fr_ref = _bare_copy(text).branch_frames()
    seen = 0
    for pid in sorted(set(int(p) for p in np.asarray(fr["plan_id"]))):
        _both(fr_ref, np.asarray(fr["a"])[pid], (name, pid))
        seen += 1
    assert seen >= 1


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
    assert frames.FRAME_COSETS_BACKEND == "auto"
    _assert_bitwise(frames.plan_frame_cosets_fast(fr_ref, a), t1, "auto→cpp")
    # the numpy path is selectable (explicitly) and is the 3.1.2 body with the portable
    # weight arithmetic (review MAJOR-2)
    _assert_bitwise(t1, frames.plan_frame_cosets_numpy(fr_ref, a), "numpy body")
    _assert_bitwise(t1, frames.plan_frame_cosets_fast(fr_ref, a, backend="numpy"), "backend=numpy")
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
    for backend in ("auto", "cpp", None):
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
    _assert_bitwise(frames.plan_frame_cosets_fast(fr_ref, a), ref_table, "auto == numpy")


_ENV_PROBE = r"""
import json, sys
import numpy as np
import xtim.frames as frames
ref = __import__("pickle").load(open(sys.argv[1], "rb"))
a = np.zeros(int(ref["n"]), np.uint8); a[0] = 1
called = []
orig = frames._CPP_PLAN_FRAME_COSETS
frames._CPP_PLAN_FRAME_COSETS = lambda *args: called.append(1) or orig(*args)
t = frames.plan_frame_cosets_fast(ref, a)
print(json.dumps({"backend": frames.FRAME_COSETS_BACKEND, "compiled_calls": len(called),
                  "weight": [c["weight"].hex() for c in t]}))
"""


def _run_env_probe(tmp_path, env_value):
    import pickle, subprocess, sys
    fr_ref = _bare_copy(_text(*_SYNTH_REFS["stab10"])).branch_frames()
    p = tmp_path / "ref.pkl"
    with open(p, "wb") as fh:
        pickle.dump({k: (np.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in fr_ref.items()}, fh)
    env = dict(os.environ); env["XTIM_FRAME_COSETS_BACKEND"] = env_value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run([sys.executable, "-c", _ENV_PROBE, str(p)], env=env,
                          capture_output=True, text=True, cwd=str(tmp_path))
    a = np.zeros(int(fr_ref["n"]), np.uint8); a[0] = 1
    return proc, fr_ref, a


def test_env_var_selects_the_backend_in_a_fresh_process(tmp_path):
    """Review MINOR-4: XTIM_FRAME_COSETS_BACKEND is read once at import — exercised in a
    fresh subprocess: `numpy` serves the pure-Python path (0 compiled calls, same weight
    bits); `cpp` and unset serve the compiled table; an unknown value is refused AT IMPORT."""
    import json
    proc, fr_ref, a = _run_env_probe(tmp_path, "numpy")
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["backend"] == "numpy" and got["compiled_calls"] == 0
    assert got["weight"] == [c["weight"].hex() for c in frames.plan_frame_cosets_numpy(fr_ref, a)]
    proc, _, _ = _run_env_probe(tmp_path, "cpp")
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["backend"] == "cpp" and got["compiled_calls"] == 1
    proc, _, _ = _run_env_probe(tmp_path, "fortran")
    assert proc.returncode != 0
    assert "XTIM_FRAME_COSETS_BACKEND='fortran': unknown backend" in proc.stderr
    assert "ValueError" in proc.stderr


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


# ── the sampling HOT PATH is untouched: record-stream hashes of the 3.1.2 build ───────────
# Captured 2026-09-14 on the 3.1.2 extension (main 7258c58) BEFORE the compiled table was
# added, by scratch script stream_hash.py: two 256-shot barriers (seeds 7, 11) per fixture;
# sha256 over frames() {prefix_words, plan_id, a, r, kappa, fallback} (+dtype, shape),
# record_keys, record_group_ids and branch_frames(i) for i in 0, 64, 128, 192.
_STREAM_HASHES = {
    "adaptq_born5_dec_enum_producer": "6f648e4e2c972a0590819b2e7942946d2c3b3fc22440e4cad0a88140e18082c0",
    "adaptq_qrm_magic_producer": "1eed2c693f079afd216aea4e0607e5cb45516f3c8874530daba7fc2763c77aa6",
    "adaptq_steane_h_producer": "f67b43289055655427174aed838aa2a6aaea4532d2457c2ef956b3a644587060",
}


@pytest.mark.parametrize("name", sorted(_STREAM_HASHES))
def test_sampling_stream_unchanged_since_3_1_2(name):
    s = compile_twirl_sampler(_fixture_text(name))
    h = hashlib.sha256()
    for seed in (7, 11):
        buf = s.sample_barrier(256, seed)
        fr = buf.frames()
        for k in ("prefix_words", "plan_id", "a", "r", "kappa", "fallback"):
            arr = np.ascontiguousarray(np.asarray(fr[k]))
            h.update(k.encode()); h.update(str(arr.dtype).encode()); h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        rk = buf.record_keys()
        if isinstance(rk, np.ndarray):
            h.update(np.ascontiguousarray(rk).tobytes())
        else:
            h.update(repr([bytes(x) if not isinstance(x, (bytes, str)) else x for x in rk]).encode())
        h.update(np.asarray(buf.record_group_ids()).tobytes())
        for i in range(0, 256, 64):
            bf = buf.branch_frames(i)
            for k in sorted(bf):
                arr = np.ascontiguousarray(np.asarray(bf[k])); h.update(k.encode()); h.update(arr.tobytes())
    assert h.hexdigest() == _STREAM_HASHES[name]
