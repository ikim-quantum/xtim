"""Exact oracle tests for the xtim.port groups/residual_law record route (port-v3 T2).

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
_CULT_DEFAULT = str(pathlib.Path(xtim.__file__).resolve().parent
                    / "_examples" / "cultivation_d5.stim")
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
    if sorted(d1.keys()) != sorted(d2.keys()):
        return False
    for k in d1:
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            if not np.array_equal(np.asarray(v1), np.asarray(v2)):
                return False
        elif isinstance(v1, list):
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if not np.array_equal(np.asarray(a), np.asarray(b)):
                    return False
        elif v1 != v2:
            return False
    return True


def _unpack(b8, ncols, shots):
    b8 = np.asarray(b8, np.uint8)
    if ncols == 0:
        return np.zeros((shots, 0), dtype=bool)
    return np.unpackbits(b8, axis=1, bitorder="little")[:, :ncols].astype(bool)


# ── PG-01/03/05: port record route vs the current surface, exhaustive ─────────

@pytest.mark.parametrize("name,get_text,shots", _WORKLOADS,
                         ids=[w[0] for w in _WORKLOADS])
def test_pg01_group_partition_keys_gids_byte_equal(name, get_text, shots):
    """PG-01 + PG-03 + PG-05: the port's group partition, EVERY group key byte,
    the gids, dets/decisions, and the per-group arrays are byte-equal to the
    current record_groups() surface at the same seed."""
    text = get_text()
    seg = port.compile(text, Consume(dets=True, decisions=True, groups=True))
    res = seg.run(shots, _SEED)

    s = _raw_sampler(text)
    buf = s.sample_barrier(shots, _SEED)                 # LEGACY store
    gids0, fo0, keys0 = buf.record_groups()
    gids0 = np.asarray(gids0, np.int32)
    fo0 = np.asarray(fo0, np.int32)
    keys0 = np.asarray(keys0, np.uint8)

    # partition + first-occurrence order + every key byte
    assert np.array_equal(res.group_id, gids0)
    assert np.array_equal(res.group_first_shot, fo0)
    assert res.group_keys.shape == keys0.shape
    assert np.array_equal(res.group_keys, keys0)
    assert res.n_groups == fo0.shape[0]

    # PG-03: per-shot streams byte-equal to the record surface
    n_det = int(s.num_detectors)
    n_dec = int(s.num_decisions)
    assert np.array_equal(res.dets, _unpack(buf.dets(), n_det, shots))
    assert np.array_equal(res.decisions, _unpack(buf.decisions(), n_dec, shots))

    # PG-05: per-group arrays == per-shot gather cross-check
    sig0 = np.asarray(buf.sigmas(), np.uint64)
    assert np.array_equal(res.group_sigmas, sig0[fo0])
    assert res.sig_bits == int(buf.sig_bits)
    assert np.array_equal(res.group_dets, res.dets[fo0])
    assert np.array_equal(res.group_decisions, res.decisions[fo0])
    coins = res.group_coins
    plans = res.group_plans
    assert len(coins) == res.n_groups and len(plans) == res.n_groups
    for g in range(res.n_groups):
        i = int(fo0[g])
        assert coins[g] == bytes(buf.coins(i))
        assert plans[g] == bytes(buf.plan_key(i))


@pytest.mark.parametrize("name,get_text,shots", _WORKLOADS,
                         ids=[w[0] for w in _WORKLOADS])
def test_pg02_compact_store_byte_equal_raw_surface(name, get_text, shots):
    """PG-02: sample_barrier(compact=True) is byte-equal to the legacy store on
    EVERY read surface: record_groups (3-tuple), record_keys, record_group_ids,
    and the per-shot plan_key/coins/sigma accessors."""
    text = get_text()
    s1 = _raw_sampler(text)
    s2 = _raw_sampler(text)
    b1 = s1.sample_barrier(shots, _SEED)                 # legacy
    b2 = s2.sample_barrier(shots, _SEED, None, True)     # compact

    g1 = b1.record_groups()
    g2 = b2.record_groups()
    for a, b in zip(g1, g2):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(b1.record_keys()), np.asarray(b2.record_keys()))
    assert np.array_equal(np.asarray(b1.record_group_ids()),
                          np.asarray(b2.record_group_ids()))
    assert np.array_equal(np.asarray(b1.sigmas()), np.asarray(b2.sigmas()))
    assert np.array_equal(np.asarray(b1.dets()), np.asarray(b2.dets()))
    assert np.array_equal(np.asarray(b1.decisions()), np.asarray(b2.decisions()))
    for i in range(shots):
        assert bytes(b1.plan_key(i)) == bytes(b2.plan_key(i))
        assert bytes(b1.coins(i)) == bytes(b2.coins(i))


def test_pg04_sigma_law_reads_byte_equal_raw_accessors():
    """PG-04: the Segment read model (sigma_law / born_dec / plan_structure)
    is byte-equal to the raw accessors, on the k_port=1 and born-dec workloads,
    for EVERY distinct plan of the draw."""
    for fname, shots in [("adaptq_qrm_magic_producer.stim", 1024),
                         ("adaptq_born5_dec_enum_producer.stim", 1024)]:
        text = (_DATA / fname).read_text()
        seg = port.compile(text, Consume(dets=True, groups=True, residual_law=True))
        res = seg.run(shots, _SEED)

        s = _raw_sampler(text)
        buf = s.sample_barrier(shots, _SEED)
        st = xtim._xtim.bare_state_of(text)
        wires = list(s.output_wires())

        # σ-law: byte-equal to the raw port_sigma_law on the same wires
        assert list(seg.output_wires) == wires
        assert _dict_equal(seg.sigma_law, st.port_sigma_law(wires))
        # born_dec: byte-equal to the raw buffer accessor
        assert _dict_equal(seg.born_dec, buf.born_dec())
        # sig widths coherent with the sampled surface
        assert seg.sig_bits == int(buf.sig_bits) == int(seg.sigma_law["ngens"])
        assert seg.gw == res.group_sigmas.shape[1]
        # plan_structure: every distinct plan of this draw, byte-equal
        for pk in sorted(set(res.group_plans)):
            assert _dict_equal(seg.plan_structure(pk), buf.plan_structure(pk))
        # memoized: identity-equal on a repeat call
        pk0 = res.group_plans[0]
        assert seg.plan_structure(pk0) is seg.plan_structure(pk0)


def test_pg06_corruption_probe_partition_changes():
    """PG-06 (mutation gate): corrupting ONE shot's stored compact identity
    (prefix word) must change the reconstructed partition/keys — proving the
    compact fields are load-bearing, not a decorative copy."""
    text = (_DATA / "adaptq_qrm_magic_producer.stim").read_text()
    s = _raw_sampler(text)
    buf = s.sample_barrier(2048, _SEED, None, True)
    g0 = [np.asarray(a).copy() for a in buf.record_groups()]
    # pick a shot WITH a plan (plan_id > 0) so the corrupted word is part of
    # the legacy key bytes too
    target = next(i for i in range(2048) if len(buf.plan_key(i)) > 0)
    buf._debug_corrupt_compact(target)
    g1 = [np.asarray(a) for a in buf.record_groups()]
    changed = any(not np.array_equal(a, b) for a, b in zip(g0, g1))
    assert changed, "corrupted compact identity did NOT change the partition/keys"
    # the corrupted shot's reconstructed plan bytes must differ as well
    # (bit 0 of prefix word 0 flipped)
    assert bytes(buf.plan_key(target)) != bytes(_raw_sampler(text).sample_barrier(
        2048, _SEED, None, True).plan_key(target))
    # legacy-mode buffers refuse the probe (loud, never silent)
    bl = _raw_sampler(text).sample_barrier(4, _SEED)
    with pytest.raises(RuntimeError, match="compact"):
        bl._debug_corrupt_compact(0)


def test_pg06b_plan_id_corruption_probe():
    """PG-06b (port-v3 T3 fold-in): corrupting a shot's INTERNED plan_id (reset
    to the empty plan) must change the partition AND the reconstructed legacy
    plan bytes — the plan_id component of the compact identity is load-bearing,
    independently of the prefix words (which PG-06 covers)."""
    text = (_DATA / "adaptq_qrm_magic_producer.stim").read_text()
    s = _raw_sampler(text)
    buf = s.sample_barrier(2048, _SEED, None, True)
    g0 = [np.asarray(a).copy() for a in buf.record_groups()]
    target = next(i for i in range(2048) if len(buf.plan_key(i)) > 0)
    pk_before = bytes(buf.plan_key(target))
    buf._debug_corrupt_compact(target, "plan_id")
    g1 = [np.asarray(a) for a in buf.record_groups()]
    changed = any(not np.array_equal(a, b) for a, b in zip(g0, g1))
    assert changed, "corrupted plan_id did NOT change the partition/keys"
    # the shot's reconstructed plan bytes must now be the EMPTY plan
    assert bytes(buf.plan_key(target)) == b"" != pk_before
    # a shot with NO plan refuses the plan_id probe (loud, never silent)
    buf2 = _raw_sampler(text).sample_barrier(2048, _SEED, None, True)
    clean = next(i for i in range(2048) if len(buf2.plan_key(i)) == 0)
    with pytest.raises(RuntimeError, match="plan_id == 0"):
        buf2._debug_corrupt_compact(clean, "plan_id")
    # unknown field name refuses
    with pytest.raises(RuntimeError, match="unknown field"):
        buf2._debug_corrupt_compact(0, "sigma")


def test_pg07_groups_without_residual_law_no_sigma_law_fetch(monkeypatch):
    """PG-07 (undeclared-not-computed, spy): a groups-only compile must never
    call the σ-law fetch seam; a residual_law compile must call it exactly once
    (compile-cached thereafter)."""
    calls = []
    real = port._fetch_sigma_law

    def spy(text, sampler):
        calls.append(text)
        return real(text, sampler)

    monkeypatch.setattr(port, "_fetch_sigma_law", spy)
    # unique circuit text so the lru_cache cannot serve a pre-spy Segment
    text = ("R 0 1\nH 0\nCX 0 1\nX_ERROR(0.125) 0\n"
            "R 2\nCX 0 2\nCX 1 2\nM 2\nDECISION(0) rec[-1]\n"
            "OUTPUT_QUBITS out 0 1\n# PG-07 unique\n")
    seg = port.compile(text, Consume(dets=True, groups=True))
    seg.run(64, 7)
    assert calls == [], "groups-only compile fetched the σ-law (undeclared computed)"
    with pytest.raises(AttributeError, match="residual_law"):
        seg.sigma_law
    with pytest.raises(AttributeError, match="residual_law"):
        seg.born_dec
    seg2 = port.compile(text, Consume(dets=True, groups=True, residual_law=True))
    assert len(calls) == 1
    assert isinstance(seg2.sigma_law, dict)


def test_pg08_ch_ppr_record_route_refusal():
    """PG-08: CH/PPR circuits refuse groups AND residual_law with the spec §2
    physics message naming both sides — the record route never engages."""
    text = (pathlib.Path(xtim.__file__).parent / "_examples" /
            "ch_cultivation.stim").read_text()
    for consume, decl in [
        (Consume(dets=True, groups=True), "groups"),
        (Consume(dets=True, residual_law=True), "residual_law"),
        (Consume(dets=True, decisions=True, groups=True, residual_law=True),
         "groups and residual_law"),
    ]:
        with pytest.raises(RuntimeError) as ei:
            port.compile(text, consume)
        msg = str(ei.value)
        assert "CH/PPR" in msg
        assert decl in msg
        assert "diagonal" in msg or "retention" in msg
    # ... and still SERVES dets/decisions (bare route)
    seg = port.compile(text, Consume(dets=True, decisions=True))
    res = seg.run(50, 3)
    assert res.dets.shape[0] == 50


def test_pg09_plan_cache_compile_option(tmp_path):
    """PG-09: plan_cache is a compile option: the .twpl file is created after a
    run; a warm cache does NOT move record-route bytes; the cache key separates
    Segments."""
    text = (_DATA / "adaptq_steane_h_producer.stim").read_text()
    cache = tmp_path / "steane_plans.twpl"
    c = Consume(dets=True, groups=True)

    seg_nc = port.compile(text, c)                       # no cache
    res_nc = seg_nc.run(1024, _SEED)

    seg_c = port.compile(text, c, plan_cache=str(cache))
    assert seg_c is not seg_nc, "plan_cache must be part of the compile cache key"
    assert seg_c is port.compile(text, c, plan_cache=str(cache))
    res_cold = seg_c.run(1024, _SEED)
    assert cache.exists(), "plan cache file not written after the run"

    # A FRESH compile that pre-loads the now-warm cache: record-route bytes
    # must be UNMOVED (retention path is plan-warmth independent).
    port._cached_build.cache_clear()
    seg_w = port.compile(text, c, plan_cache=str(cache))
    res_warm = seg_w.run(1024, _SEED)
    for a, b in [(res_nc.group_keys, res_cold.group_keys),
                 (res_cold.group_keys, res_warm.group_keys)]:
        assert np.array_equal(a, b)
    assert np.array_equal(res_nc.group_id, res_warm.group_id)


def test_pg10_compact_materialize_exact():
    """PG-10: materialize(i) from the COMPACT store reconstructs the exact
    collapsed state the legacy store produces (approx_equal, tol 1e-12), for
    every group representative of a small draw."""
    text = (_DATA / "adaptq_qrm_magic_producer.stim").read_text()
    b1 = _raw_sampler(text).sample_barrier(256, _SEED)
    b2 = _raw_sampler(text).sample_barrier(256, _SEED, None, True)
    fo = np.asarray(b1.record_groups()[1])
    for i in fo:
        assert b1.materialize(int(i)).approx_equal(b2.materialize(int(i)))


def test_pg11_residual_law_only_read_model_without_group_arrays():
    """PG-11: residual_law without groups → record route, Segment read model
    present, Result carries NO group arrays (nothing undeclared computed)."""
    text = (_DATA / "adaptq_steane_h_producer.stim").read_text()
    seg = port.compile(text, Consume(dets=True, residual_law=True))
    assert isinstance(seg.sigma_law, dict)
    assert isinstance(seg.born_dec, dict)
    assert seg.sig_bits > 0
    res = seg.run(128, _SEED)
    assert res.dets.shape == (128, port.compile(
        text, Consume(dets=True, residual_law=True))._n_det)
    for attr in ("group_id", "group_keys", "group_sigmas", "group_coins"):
        with pytest.raises(AttributeError, match="groups"):
            getattr(res, attr)
    # streams equal the record surface (residual_law routes to sample_barrier)
    buf = _raw_sampler(text).sample_barrier(128, _SEED)
    assert np.array_equal(res.dets, _unpack(buf.dets(), res.dets.shape[1], 128))


def test_pg12_zero_shot_record_run():
    """PG-12: a zero-shot record run returns valid empty shapes."""
    text = (_DATA / "adaptq_steane_h_producer.stim").read_text()
    seg = port.compile(text, Consume(dets=True, decisions=True, groups=True))
    res = seg.run(0, 1)
    assert res.dets.shape[0] == 0
    assert res.group_id.shape == (0,)
    assert res.n_groups == 0
    assert res.group_keys.shape[0] == 0
    assert res.group_coins == ()
    assert res.group_plans == ()


# ── PG-13: SCALED byte-identity leg (port-v3 T3 fold-in) ──────────────────────

@pytest.mark.parametrize("name,get_text", [
    ("steane_h", lambda: (_DATA / "adaptq_steane_h_producer.stim").read_text()),
    ("cultivation_d5", _cultivation_text),
], ids=["steane_h", "cultivation_d5"])
def test_pg13_scaled_byte_identity_20k(name, get_text):
    """PG-13: compact-store byte identity at SCALE — 20 000 shots per workload
    (the T2 report's 100k re-verify enrolled as a real, always-run test): the
    full record_groups tuple, the σ words, and the per-shot dets/decisions
    streams are array-equal between the compact and legacy stores."""
    shots = 20_000
    text = get_text()
    b_leg = _raw_sampler(text).sample_barrier(shots, _SEED)
    b_cmp = _raw_sampler(text).sample_barrier(shots, _SEED, None, True)
    for a, b in zip(b_leg.record_groups(), b_cmp.record_groups()):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(b_leg.sigmas()), np.asarray(b_cmp.sigmas()))
    assert np.array_equal(np.asarray(b_leg.dets()), np.asarray(b_cmp.dets()))
    assert np.array_equal(np.asarray(b_leg.decisions()),
                          np.asarray(b_cmp.decisions()))
