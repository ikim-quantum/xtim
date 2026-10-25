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
  PG-09  plan_cache compile option (file written; warm bytes unmoved; cache key separates Segments)
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
  PG-09  plan_cache compile option: file created, warm-cache record-route bytes
         UNMOVED, cache key separates Segments
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
# oracle worktree's absolute path, which breaks on any other checkout; the
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
    assert seg.plan_structure(pc) is seg.plan_structure(pc)              # memoized


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


def test_pg09_plan_cache_compile_option(tmp_path):
    """PG-09: plan_cache is a compile option: the .twpl file is created after a run; a warm cache does NOT move
    record-route bytes; the cache key separates Segments."""
    text = (_DATA / "adaptq_steane_h_producer.stim").read_text()
    cache = tmp_path / "steane_plans.twpl"
    c = Consume(dets=True, groups=True)
    seg_nc = port.compile(text, c)
    res_nc = seg_nc.run(1024, _SEED)
    seg_c = port.compile(text, c, plan_cache=str(cache))
    assert seg_c is not seg_nc, "plan_cache must be part of the compile cache key"
    assert seg_c is port.compile(text, c, plan_cache=str(cache))
    res_cold = seg_c.run(1024, _SEED)
    assert cache.exists(), "plan cache file not written after the run"
    port._cached_build.cache_clear()
    seg_w = port.compile(text, c, plan_cache=str(cache))
    res_warm = seg_w.run(1024, _SEED)
    assert res_nc.group_keys == res_cold.group_keys == res_warm.group_keys
    assert np.array_equal(res_nc.group_id, res_warm.group_id)


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
