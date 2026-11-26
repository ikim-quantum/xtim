"""Exact oracle tests for the xtim.port groups/residual_law record route (port-v3 T2).

REWRITTEN at J-b1 F (xtim-engine-13; plan §5.3 "rewritten for sigma retirement", refusal -> exactness at PG-08):
the record route reads the LAYOUT-3 BarrierBuffer (sigma retired; keys = [kind][len] content || P_rec || y).
Every port output is checked byte-exactly against the same-seed raw buffer surface:
  PG-01  group_id / group_keys / group_first_shot == record_group_ids / record_groups (first appearance);
         group dets / decisions == the per-shot gather at the first shots
  PG-02  group_plan_content / group_outcomes / group_prefix / group_signs == the raw per-shot accessors
  PG-04  the read model (born_dec, plan_structure(content), output_wires, record_layout) == the raw sampler's
  PG-07  undeclared-not-computed: groups without residual_law never calls the read-model fetch (spy)
  PG-08  CH / PPR circuits are SERVED on the record route (formerly refused): kind-2 plan contents
  PG-09  (deleted, bloat-cut C: the plan_cache compile option is gone; compile(..., plan_cache=) is a TypeError)
  PG-11  residual_law-only: read model present, no group arrays
  PG-12  zero-shot record run: valid empty shapes
  PG-13  determinism at scale (20k shots, same seed -> identical keys and streams)
The original sigma-era header follows for the record.


ZERO statistical assertions. Every test is a byte-exact comparison against the
CURRENT (legacy) record surface, or a structural/mutation probe:

  PG-01  group partition + keys + gids byte-equal to record_groups()  (per workload,
         EVERY group, EVERY key byte — array_equal over the full matrices)
  PG-02  compact sample_barrier store byte-equal to the legacy store at the raw
         surface (record_groups + record_keys + per-shot plan_key/coins/sigma)
  PG-03  dets/decisions through the port byte-equal to the record surface
  PG-04  σ-law / born_dec / plan_structure reads through the port == the raw
         accessors, byte-equal (every distinct plan of the draw)
  PG-05  per-group arrays match a per-shot gather cross-check
  PG-06  corruption probe: corrupt ONE compact identity → partition/keys change
         (the compact fields are load-bearing — mutation-verified, non-vacuous)
  PG-07  undeclared-not-computed: groups declared WITHOUT residual_law → the
         σ-law fetch seam is NEVER called (spy)
  PG-08  CH/PPR circuits refuse groups/residual_law on the record route with the
         physics message (extends the T1 compile-time test to this route)
  PG-09  (deleted, bloat-cut C)
  PG-10  compact materialize(i) reconstructs the exact legacy state (approx_equal)
  PG-11  residual_law-only declaration: read model present, group arrays absent
  PG-12  zero-shot record run: valid empty shapes

Workloads (the plan's three + the born-decision class):
  steane-class     tests/data/adaptq_steane_h_producer.stim      (2048 shots)
  qrm k_port=1     tests/data/adaptq_qrm_magic_producer.stim     (2048 shots)
  cultivation_d5   $CULTIVATION_BENCHMARK or the frozen oracle checkout
                   (2000 shots; skipped if the benchmark is absent)
  born5            tests/data/adaptq_born5_dec_enum_producer.stim (1024 shots;
                   exercises the born-memo + compact interplay: plan bytes are
                   still serialized engine-side for the memo, the record store
                   stays compact)
"""
import os
import pathlib
import re

import numpy as np
import pytest

import xtim
import xtim.port as port
from xtim.port import Consume

_DATA = pathlib.Path(__file__).parent / "data"

# cultivation-d5 workload location (port-v3 T3 fold-in): default to THIS repo's
# shipped example (byte-identical to the frozen oracle benchmark — verified at
# T3), overridable via $CULTIVATION_BENCHMARK.  The old default hard-coded the
# oracle checkout's absolute path, which breaks on any other checkout; the
# in-repo example makes the test self-contained under the adaptq-eng binding.
# Resolve through the IMPORTED package, not this file's location.  The old
# form (tests/../xtim/_examples/...) only existed when the tests sat inside the
# source checkout, so RELEASING.md's step 6 -- "run pytest against the INSTALLED
# package from a neutral cwd" -- silently skipped this leg, the heaviest
# workload here, at every release.  xtim/_examples IS shipped in the wheel (9
# circuits), and under an editable install this still lands in the source tree,
# so one path serves both.
_CULT_DEFAULT = str(pathlib.Path(__file__).resolve().parent / "data" / "cultivation_d5.stim")
_REC_RE = re.compile(r"rec\[-?\d+\]")


def _strip_metadata(text: str) -> str:
    """Same stripping convention as adaptq's cultivation harness."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        head = s.split()[0]
        if head.startswith("QUBIT_COORDS") or head == "TICK":
            continue
        if head.startswith("OBSERVABLE_INCLUDE"):
            continue
        if head.startswith("DETECTOR"):
            recs = _REC_RE.findall(s)
            out.append("DETECTOR " + " ".join(recs) if recs else "DETECTOR")
            continue
        out.append(s)
    return "\n".join(out) + "\n"


def _cultivation_text():
    p = pathlib.Path(os.environ.get("CULTIVATION_BENCHMARK", _CULT_DEFAULT))
    if not p.exists():
        pytest.skip(f"cultivation benchmark not found: {p}")
    return _strip_metadata(p.read_text())


# (name, text getter, shots)
_WORKLOADS = [
    ("steane_h", lambda: (_DATA / "adaptq_steane_h_producer.stim").read_text(), 2048),
    ("qrm_magic", lambda: (_DATA / "adaptq_qrm_magic_producer.stim").read_text(), 2048),
    ("cultivation_d5", _cultivation_text, 2000),
    ("born5", lambda: (_DATA / "adaptq_born5_dec_enum_producer.stim").read_text(), 1024),
]
_SEED = 20260731


def _raw_sampler(text):
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    return s._inner if hasattr(s, "_inner") else s


def _dict_equal(d1, d2):
    """Recursive structural equality over dicts / lists / tuples / numpy arrays / scalars."""
    if isinstance(d1, dict) or isinstance(d2, dict):
        if not (isinstance(d1, dict) and isinstance(d2, dict)) or sorted(d1) != sorted(d2):
            return False
        return all(_dict_equal(d1[k], d2[k]) for k in d1)
    if isinstance(d1, np.ndarray) or isinstance(d2, np.ndarray):
        return np.array_equal(np.asarray(d1), np.asarray(d2))
    if isinstance(d1, (list, tuple)) and isinstance(d2, (list, tuple)):
        return len(d1) == len(d2) and all(_dict_equal(a, b) for a, b in zip(d1, d2))
    return d1 == d2


def _dict_identical(d1, d2):
    """_dict_equal, stricter: the same container / scalar types, dict key ORDER, and per array the same dtype, shape and
    C-contiguity (what a consumer of the plan_structure format can observe)."""
    if type(d1) is not type(d2):
        return False
    if isinstance(d1, dict):
        return list(d1) == list(d2) and all(_dict_identical(d1[k], d2[k]) for k in d1)
    if isinstance(d1, np.ndarray):
        return (d1.dtype == d2.dtype and d1.dtype.str == d2.dtype.str and d1.shape == d2.shape
                and d1.flags.c_contiguous == d2.flags.c_contiguous and d1.flags.writeable == d2.flags.writeable
                and np.array_equal(d1, d2))
    if isinstance(d1, (list, tuple)):
        return len(d1) == len(d2) and all(_dict_identical(a, b) for a, b in zip(d1, d2))
    return d1 == d2


def _unpack(b8, ncols, shots):
    b8 = np.asarray(b8, np.uint8)
    if ncols == 0:
        return np.zeros((shots, 0), dtype=bool)
    return np.unpackbits(b8, axis=1, bitorder="little")[:, :ncols].astype(bool)


# ── PG-01 / PG-02: the port's group arrays == the raw layout-3 buffer, exhaustive ──────────────
@pytest.mark.parametrize("name,get_text,shots", _WORKLOADS, ids=[w[0] for w in _WORKLOADS])
def test_pg01_groups_byte_equal_the_raw_buffer(name, get_text, shots):
    text = get_text()
    seg = port.compile(text, Consume(dets=True, decisions=True, groups=True))
    res = seg.run(shots, _SEED)
    buf = _raw_sampler(text).sample_barrier(shots, seed=_SEED)
    gid = np.asarray(buf.record_group_ids(), np.int32)
    keys, counts = buf.record_groups()
    assert np.array_equal(res.group_id, gid)
    assert list(res.group_keys) == [bytes(k) for k in keys]
    first = res.group_first_shot
    assert len(first) == res.n_groups == len(keys)
    assert all(gid[first[g]] == g for g in range(res.n_groups))
    assert np.array_equal(np.bincount(gid, minlength=res.n_groups), np.asarray(counts))
    nd, ndec = res.dets.shape[1], res.decisions.shape[1]
    assert np.array_equal(res.dets, _unpack(buf.dets(), nd, shots))
    assert np.array_equal(res.decisions, _unpack(buf.decisions(), ndec, shots))
    assert np.array_equal(res.group_dets, res.dets[first])
    assert np.array_equal(res.group_decisions, res.decisions[first])
    # every shot of a group carries the group's key (the retained state's sufficient statistic)
    rk = [bytes(k) for k in buf.record_keys()]
    assert all(rk[i] == res.group_keys[gid[i]] for i in range(shots))


@pytest.mark.parametrize("name,get_text,shots", _WORKLOADS, ids=[w[0] for w in _WORKLOADS])
def test_pg02_group_readers_equal_the_raw_accessors(name, get_text, shots):
    text = get_text()
    seg = port.compile(text, Consume(dets=True, groups=True))
    res = seg.run(shots, _SEED)
    buf = _raw_sampler(text).sample_barrier(shots, seed=_SEED)
    first = [int(i) for i in res.group_first_shot]
    for g, i in enumerate(first):
        kind, content, digest = res.group_plan_content[g]
        pc, dg = buf.plan_content(i)
        assert bytes(pc) == bytes([kind]) + len(content).to_bytes(4, "little") + content and bytes(dg) == digest
        assert np.array_equal(res.group_outcomes[g], np.asarray(buf.outcomes(i)))
        plan = res.group_plan(g)
        assert plan["kind"] == kind and plan["content"] == content
    lay = buf.record_layout()
    W = int(lay["prefix_words"])
    for g, k in enumerate(res.group_keys):
        L = int.from_bytes(k[1:5], "little")
        assert np.array_equal(res.group_prefix[g], np.frombuffer(k[5 + L:5 + L + 8 * W], "<u8"))
    n = int(lay["n"])
    paulis = []
    for q in range(min(n, 6)):
        z = np.zeros(n, np.uint8); z[q] = 1
        paulis.append((np.zeros(n, np.uint8), z, 0))
    sg = np.asarray(buf.signs(paulis))
    assert np.array_equal(res.group_signs(paulis), sg[first])
    assert np.array_equal(res.signs(paulis), sg)


# ── PG-04: the Segment read model == the raw sampler's accessors ─────────────────────────────
def test_pg04_read_model_equals_raw_accessors():
    text = (_DATA / "adaptq_born5_dec_enum_producer.stim").read_text()
    seg = port.compile(text, Consume(dets=True, groups=True, residual_law=True))
    raw = _raw_sampler(text)
    assert _dict_equal(seg.born_dec, raw.born_dec())
    assert seg.output_wires == tuple(int(w) for w in raw.output_wires())
    assert _dict_equal(seg.record_layout(), raw.record_layout())
    res = seg.run(512, _SEED)
    buf = raw.sample_barrier(512, seed=_SEED)
    for g, i in enumerate(res.group_first_shot[:40]):
        pc = bytes(buf.plan_content(int(i))[0])
        assert _dict_equal(seg.plan_structure(pc), raw.plan_structure(pc))
        assert _dict_equal(seg.plan_structure(pc), buf.plan_structure(int(i)))
    # memoized (FFS ruling 2026-11-24: bit-packed memo; repeated calls return EQUAL dicts, identity not guaranteed)
    assert _dict_identical(seg.plan_structure(pc), seg.plan_structure(pc))


def test_pg07_groups_without_residual_law_no_read_model_fetch(monkeypatch):
    """PG-07 (undeclared-not-computed, spy): a groups-only compile never calls the read-model seam; a residual_law
    compile calls it exactly once (compile-cached thereafter)."""
    calls = []
    real = port._fetch_read_model

    def spy(sampler):
        calls.append(1)
        return real(sampler)

    monkeypatch.setattr(port, "_fetch_read_model", spy)
    text = ("R 0 1\nH 0\nCX 0 1\nX_ERROR(0.125) 0\n"
            "R 2\nCX 0 2\nCX 1 2\nM 2\nDECISION(0) rec[-1]\n"
            "OUTPUT_QUBITS out 0 1\n# PG-07 unique (J-b1 F)\n")
    seg = port.compile(text, Consume(dets=True, groups=True))
    seg.run(64, 7)
    assert calls == []
    for attr in ("born_dec", "output_wires"):
        with pytest.raises(AttributeError, match="residual_law"):
            getattr(seg, attr)
    seg2 = port.compile(text, Consume(dets=True, groups=True, residual_law=True))
    assert len(calls) == 1 and isinstance(seg2.born_dec, dict)


def test_pg08_ch_ppr_record_route_served():
    """PG-08, RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness (plan §5.3 test_port_groups.py:319-338;
    M-DECPPR): CH / PPR circuits are SERVED on the record route — their CH / PPR shots are kind-2 records — and the
    port equals the raw buffer."""
    text = (pathlib.Path(__file__).resolve().parent / "data" / "ch_cultivation.stim").read_text()
    seg = port.compile(text, Consume(dets=True, decisions=True, groups=True, residual_law=True))
    res = seg.run(2000, 3)
    kinds = {k for k, _c, _d in res.group_plan_content}
    assert 2 in kinds, kinds
    buf = _raw_sampler(text).sample_barrier(2000, seed=3)
    assert list(res.group_keys) == [bytes(k) for k in buf.record_groups()[0]]
    assert np.array_equal(res.dets, _unpack(buf.dets(), res.dets.shape[1], 2000))


def test_pg09_plan_cache_option_removed():
    """PG-09 (bloat-cut C): the plan_cache compile option is deleted with the TWPL disk layer; passing it is a
    TypeError (never silently ignored)."""
    text = (_DATA / "adaptq_steane_h_producer.stim").read_text()
    with pytest.raises(TypeError):
        port.compile(text, Consume(dets=True, groups=True), plan_cache=True)


def test_pg11_residual_law_only_read_model_without_group_arrays():
    """PG-11: residual_law without groups -> record route, Segment read model present, Result carries NO group
    arrays (nothing undeclared computed)."""
    text = (_DATA / "adaptq_steane_h_producer.stim").read_text()
    seg = port.compile(text, Consume(dets=True, residual_law=True))
    assert isinstance(seg.born_dec, dict)
    res = seg.run(128, _SEED)
    for attr in ("group_id", "group_keys", "group_outcomes", "group_plan_content"):
        with pytest.raises(AttributeError, match="groups"):
            getattr(res, attr)
    buf = _raw_sampler(text).sample_barrier(128, seed=_SEED)
    assert np.array_equal(res.dets, _unpack(buf.dets(), res.dets.shape[1], 128))


def test_pg12_zero_shot_record_run():
    """PG-12: a zero-shot record run returns valid empty shapes."""
    text = (_DATA / "adaptq_steane_h_producer.stim").read_text()
    seg = port.compile(text, Consume(dets=True, decisions=True, groups=True))
    res = seg.run(0, 1)
    assert res.dets.shape[0] == 0
    assert res.group_id.shape == (0,)
    assert res.n_groups == 0
    assert res.group_keys == ()
    assert res.group_outcomes == ()
    assert res.group_plan_content == ()


@pytest.mark.parametrize("name,get_text", [
    ("steane_h", lambda: (_DATA / "adaptq_steane_h_producer.stim").read_text()),
    ("cultivation_d5", _cultivation_text),
], ids=["steane_h", "cultivation_d5"])
def test_pg13_scaled_determinism_20k(name, get_text):
    """PG-13: 20 000 shots per workload, twice at the same seed on fresh Segments: identical keys, groups and
    streams (the layout-3 store has one form; the sigma-era compact/legacy comparison is gone)."""
    shots = 20_000
    text = get_text()
    a = _raw_sampler(text).sample_barrier(shots, seed=_SEED)
    b = _raw_sampler(text).sample_barrier(shots, seed=_SEED)
    assert [bytes(k) for k in a.record_keys()] == [bytes(k) for k in b.record_keys()]
    assert np.array_equal(np.asarray(a.dets()), np.asarray(b.dets()))
    assert np.array_equal(np.asarray(a.decisions()), np.asarray(b.decisions()))


# ── FFS (2026-11-24): the FF route's speed-ups, each against an independent oracle ─────────────────────────────
_FFS_WORKLOADS = _WORKLOADS + [
    ("ch_cultivation", lambda: (_DATA / "ch_cultivation.stim").read_text(), 1500),
]


def _key_partition(keys):
    """The d002345 partition, in Python: a group per distinct layout-3 key, first-appearance order."""
    ix, gid, first, gkeys = {}, [], [], []
    for i, k in enumerate(keys):
        g = ix.get(k)
        if g is None:
            g = ix[k] = len(gkeys)
            gkeys.append(k)
            first.append(i)
        gid.append(g)
    return np.asarray(gid, np.int32), gkeys, np.asarray(first, np.int64)


@pytest.mark.parametrize("name,get_text,shots", _FFS_WORKLOADS, ids=[w[0] for w in _FFS_WORKLOADS])
def test_pg14_compact_grouping_equals_the_key_partition(name, get_text, shots):
    """PG-14: the engine groups on the compact identity (content handle, P_rec, y) and materializes a key per group;
    the partition, first-appearance order, first shots, counts and key bytes equal grouping the per-shot keys."""
    buf = _raw_sampler(get_text()).sample_barrier(shots, seed=_SEED)
    gid, gkeys, first = _key_partition([bytes(k) for k in buf.record_keys()])
    keys, counts = buf.record_groups()
    assert np.array_equal(np.asarray(buf.record_group_ids()), gid)
    assert [bytes(k) for k in keys] == gkeys
    assert np.array_equal(np.asarray(counts), np.bincount(gid, minlength=len(gkeys)))
    f = np.asarray(buf._record_group_first())
    assert f.dtype == np.int64 and np.array_equal(f, first)
    res = port.compile(get_text(), Consume(groups=True)).run(shots, _SEED)
    assert res.group_first_shot.dtype == np.int64 and np.array_equal(res.group_first_shot, first)


@pytest.mark.parametrize("name,get_text,shots", _FFS_WORKLOADS, ids=[w[0] for w in _FFS_WORKLOADS])
def test_pg15_batched_group_readers_equal_the_per_row_accessors(name, get_text, shots):
    """PG-15: the Result's per-group readers (one batched engine call each, kept on the Result) equal the per-row
    accessors at every group's first shot, on the first and on a repeated read; group_plan(g) equals the d002345
    parse of the group's content; group_signs equals signs() at the first shots for two Pauli lists read in turn
    (the cross-call sign-rule memo switches lists) and equals a fresh sampler's signs."""
    text = get_text()
    seg = port.compile(text, Consume(dets=True, groups=True))
    res = seg.run(shots, _SEED)
    buf = _raw_sampler(text).sample_barrier(shots, seed=_SEED)
    first = [int(i) for i in res.group_first_shot]
    n = int(buf.record_layout()["n"])
    for rep in range(2):
        pcs, outs, pre = res.group_plan_content, res.group_outcomes, res.group_prefix
        assert len(pcs) == len(outs) == pre.shape[0] == res.n_groups and pre.dtype == np.uint64
        for g, i in enumerate(first):
            pc, dg = buf.plan_content(i)
            kind = pc[0]
            L = int.from_bytes(pc[1:5], "little")
            assert pcs[g] == (kind, bytes(pc[5:5 + L]), bytes(dg))
            assert type(pcs[g][0]) is int and type(pcs[g][1]) is bytes and type(pcs[g][2]) is bytes
            o = np.asarray(buf.outcomes(i))
            assert outs[g].dtype == o.dtype and np.array_equal(outs[g], o)
            k = res.group_keys[g]
            assert np.array_equal(pre[g], np.frombuffer(k[5 + L:5 + L + 8 * pre.shape[1]], "<u8"))
    for g in (0, res.n_groups // 2, res.n_groups - 1):
        kind, content, digest = res.group_plan_content[g]
        d = {"kind": kind, "content": content, "digest": digest}
        if kind == 0:
            d["a"] = [q for q in range(n) if content[q]]
            d["cz"] = [(content[j] | (content[j + 1] << 8), content[j + 2] | (content[j + 3] << 8))
                       for j in range(n + 1, len(content) - 3, 4)]
        assert res.group_plan(g) == d
    lists = []
    for which in range(2):
        ps = []
        for q in range(min(n, 6)):
            x, z = np.zeros(n, np.uint8), np.zeros(n, np.uint8)
            if which:
                x[(7 * q) % n] = 1
            z[q] = 1
            ps.append((x, z, int(x @ z) & 3))
        lists.append(ps)
    fresh = [np.asarray(_raw_sampler(text).sample_barrier(shots, seed=_SEED).signs(ps)) for ps in lists]
    for ps, f in [(lists[0], fresh[0]), (lists[1], fresh[1]), (lists[0], fresh[0])]:
        assert np.array_equal(res.signs(ps), f)
        gs = res.group_signs(ps)
        assert gs.dtype == np.int8 and np.array_equal(gs, f[first])


def test_pg16_plan_structure_packed_memo_round_trip(monkeypatch):
    """PG-16: the plan_structure memo holds the engine's compact blob per content; every later call (the unpacked-dict
    cache disabled, so each call rebuilds from the blob) returns a dict identical in keys, key order, types, dtypes,
    shapes and values to the engine's plan_structure; a malformed blob is refused (ValueError), never read past."""
    monkeypatch.setattr(port, "_PS_HOT", 0)
    for name, get_text, shots in _FFS_WORKLOADS:
        if name == "cultivation_d5":
            shots = 600
        text = get_text()
        port._cached_build.cache_clear()
        seg = port.compile(text, Consume(groups=True, residual_law=True))
        raw = _raw_sampler(text)
        res = seg.run(shots, _SEED)
        pcs = list(dict.fromkeys(bytes([k]) + len(c).to_bytes(4, "little") + c for k, c, _ in res.group_plan_content))
        for pc in pcs:
            assert _dict_identical(seg.plan_structure(pc), raw.plan_structure(pc)), (name, pc[:8])   # cold
        for pc in pcs:
            assert _dict_identical(seg.plan_structure(pc), raw.plan_structure(pc)), (name, pc[:8])   # from the blob
            assert type(seg._ps_memo[pc]) is bytes
    blob = seg._ps_memo[pcs[-1]]
    for bad in (b"", b"junk", blob[:-1], blob + b"\0", b"\0" + blob[1:]):
        with pytest.raises(ValueError, match="plan_structure blob"):
            xtim._xtim._plan_structure_unpack(bad)
    port._cached_build.cache_clear()


def test_pg18_plan_structure_cache_eviction(monkeypatch):
    """PG-18: with a small unpacked-dict cache (size 3 > 0), plan_structure is driven through hits, evictions and
    rebuilds from the blob (forward, reverse and shuffled passes over more contents than the cache holds): every
    returned dict is identical to a fresh sampler's plan_structure, the cache never exceeds its size, and an immediate
    repeat is a hot hit (the cached object)."""
    monkeypatch.setattr(port, "_PS_HOT", 3)
    text = (_DATA / "adaptq_qrm_magic_producer.stim").read_text()
    port._cached_build.cache_clear()
    seg = port.compile(text, Consume(groups=True, residual_law=True))
    res = seg.run(2048, _SEED)
    pcs = list(dict.fromkeys(bytes([k]) + len(c).to_bytes(4, "little") + c for k, c, _ in res.group_plan_content))
    assert len(pcs) > 3
    fresh = _raw_sampler(text)
    ref = {pc: fresh.plan_structure(pc) for pc in pcs}
    order = pcs + pcs[::-1] + list(np.random.default_rng(3).permutation(len(pcs)))
    for item in order:
        pc = pcs[item] if not isinstance(item, bytes) else item
        d = seg.plan_structure(pc)
        assert _dict_identical(d, ref[pc])
        assert seg.plan_structure(pc) is d             # hot hit
        assert len(seg._ps_hot) <= 3
    assert len(seg._ps_memo) == len(pcs)
    port._cached_build.cache_clear()


def test_pg17_bool_unpack_equals_the_d002345_unpack():
    """PG-17: the port's b8 unpack (count=, bool view) equals the slice + astype form in values, dtype and shape."""
    rng = np.random.default_rng(5)
    for shots, ncols in [(0, 5), (1, 1), (7, 8), (33, 9), (100, 107), (5, 0)]:
        b8 = rng.integers(0, 256, (shots, (ncols + 7) // 8), dtype=np.uint8)
        a = port._unpack_b8(b8, ncols, shots)
        ref = _unpack(b8, ncols, shots)
        assert a.dtype == ref.dtype == np.bool_ and a.shape == ref.shape and a.flags.c_contiguous
        assert a.tobytes() == ref.tobytes()
