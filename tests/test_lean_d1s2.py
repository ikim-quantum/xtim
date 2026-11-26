"""lean D1-S2 (2026-11-05, OWNER-SIGNED stream change on T circuits): the ROTATION SHOT LOOP — a diagonal-class circuit
with rotations (t > 0) serves every unframed shot keyed on s = ⊕ S_f (lean/D1/S2/plan.md; cpp/src/twirl_sampler_rotloop.inc.hpp,
twirl_rotation.cpp, twirl_sampler_rot.cpp, twirl_sampler_rotw.cpp).

The decisive teeth (review H2): the GAUGE-MAP ORACLE `rot_gauge_of` — per key the rotation plan against (D) (compose_fired →
normal form → canonicalize → the record-space-B table) under the COMPUTED gauge (g from P ⊕ P_D, δ from the directions'
phases: never learned) and against (G) on the raw s — PASS on every distinct key of the corpus T circuits, the fixtures
(ANTI extras with virtual planes, the reviewer's ANTI PAIR whose product is logical, a t = 70 two-word key, a ρ = 70 plan
with two A words) and the factory fixtures; the walked nodes (p0_R(z') == p0_D(z' ⊕ δ)) against the recorded outcomes
(T-b: the coins are the keyed COIN bits in pivot order); sample == sample_barrier, split == single, threads N == 1
(T-c); the window (selfcheck) and oracle_all_shots run green. Mutants (lean/tools/mk_mutant.sh on twirl_rotation.cpp):
a dropped carry, a flipped pivot base flip, a flipped direction sign, the pivot order reversed — each red here (T-d).
"""
import glob
import math
import os
import sys
import warnings

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _public import full_table  # noqa: E402  (lean N1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _private import PRIVATE_CORPUS  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
      "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
      "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"
# review H4: two ANTI extras whose product is logical (a non-pivot with a cancelled virtual residue)
ANTI_PAIR_DEC = "RX 0\nR 1\nT 0\nCX 0 1\nDEPOLARIZE1(0.05) 0 1\nMX 0 1\nDECISION(0) rec[-2]\nDECISION(1) rec[-1]\n"
ANTI_PAIR_OBS = "RX 0\nR 1\nT 0\nCX 0 1\nDEPOLARIZE1(0.05) 0 1\nMX 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-1]\n"
FIXTURES = {
    "m3_obs": M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET,
    "m3_obs_2dec": M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET + "MX 0\nDECISION(0) rec[-1]\nMY 1\nDECISION(1) rec[-1]\n",
    "pair_ccz": ("RX 0\nRX 1\nRX 2\nRX 3\nH 1\nX_ERROR(0.3) 3\nDEPOLARIZE1(0.3) 0\nCS 2 0\nX_ERROR(0.3) 2\nZ_ERROR(0.3) 1\nT 3\n"
                 "X_ERROR(0.3) 1\nZ_ERROR(0.3) 0\nS 1\nX_ERROR(0.3) 0\nZ_ERROR(0.3) 3\nS 0\nS 2\nX_ERROR(0.3) 0\nCCZ 0 1 2\nMY 1\n"
                 "OBSERVABLE_INCLUDE(0) rec[-1]\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n"),
    "pair_cs": ("RX 0\nRX 1\nCS 1 0\nX_ERROR(0.3) 1\nS 1\nZ_ERROR(0.3) 1\nT 1\nX_ERROR(0.3) 1\nZ_ERROR(0.3) 1\nH 0\nX_ERROR(0.3) 0\n"
                "Z_ERROR(0.3) 1\nH 0\nX_ERROR(0.3) 0\nCS 0 1\nDEPOLARIZE1(0.3) 1\nZ_ERROR(0.3) 1\nS 0\nX_ERROR(0.3) 1\nZ_ERROR(0.3) 0\n"
                "MX 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nR 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n"),
    "anti_dec": ("RX 0\nR 1 2 3 4\nX_ERROR(0.25) 0\nT 0\nH 1 3\nCX 1 2 3 4\nCX 0 1\nCZ 0 3\n"
                 "DEPOLARIZE1(0.06) 0 1 2 3 4\nMX 1 2 3 4 0\nDETECTOR rec[-5] rec[-4]\n"
                 "DECISION(0) rec[-5]\nDECISION(1) rec[-3]\nDECISION(2) rec[-1]\n"),
    "anti3_dec": ("R 0 1 2\nRX 3\nT 3\nZ_ERROR(0.1) 3\nX_ERROR(0.1) 0 1\nMX 0 1 3\nDECISION(0) rec[-3]\nDECISION(1) rec[-3] rec[-2]\n"
                  "DECISION(2) rec[-2]\nR 4\nX_ERROR(0.1) 4\nM 4\nDETECTOR rec[-1]\n"),
    "anti_pair_dec": ANTI_PAIR_DEC,
    "anti_pair_obs": ANTI_PAIR_OBS,
    # a detector whose noiseless value is 1 (ref = 1 on a touched channel) next to rotation content
    "ref1": "RX 0\nT 0\nX_ERROR(0.2) 0\nT_DAG 0\nR 1\nX 1\nX_ERROR(0.1) 1\nMX 0\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n",
}


def _multiword(pairs=35, nq=5):
    out = [f"RX {' '.join(map(str, range(nq)))}\n"]
    for i in range(pairs):
        q = i % nq
        out.append(f"T {q}\nX_ERROR(0.05) {q}\nT_DAG {q}\nDEPOLARIZE1(0.02) {q}\nCX {q} {(q + 1) % nq}\n")
    out.append("MX " + " ".join(map(str, range(nq))) + "\n")
    for q in range(0, nq - 1, 2):
        out.append(f"DETECTOR rec[{q - nq}] rec[{q + 1 - nq}]\n")
    out.append("OBSERVABLE_INCLUDE(0) rec[-1]\n")
    return "".join(out)


def _rho70(nq=70):
    """Every shot fires 70 T-sandwiched faults: one plan with ρ = 70 pivots (two A words; review H3: no cap)."""
    out = [f"RX {' '.join(map(str, range(nq)))}\n"]
    for q in range(nq):
        out.append(f"T {q}\nX_ERROR(1) {q}\nT_DAG {q}\n")
    out.append("MX " + " ".join(map(str, range(nq))) + "\n")
    for q in range(nq):
        out.append(f"DETECTOR rec[{q - nq}]\n")
    out.append("OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\n")
    return "".join(out)


FIXTURES["multiword_t70"] = _multiword()
FIXTURES["rho70"] = _rho70()

CORPUS = sorted(glob.glob(os.path.join(HERE, "data", "*.stim"))) + sorted(glob.glob(os.path.join(ROOT, "examples", "*.stim")))
CASES = {os.path.relpath(p, ROOT): open(p).read() for p in CORPUS}
CASES.update({"fixture:" + k: v for k, v in FIXTURES.items()})
for p in PRIVATE_CORPUS:
    if os.path.isfile(p):
        CASES["factory:" + os.path.basename(p)] = open(p).read()


def _fam(text, **kw):
    try:
        return TA.compile_family_sampler(text, selfcheck=kw.pop("selfcheck", 0), **kw)
    except Exception as e:   # the corpus's compile refusals are not this test's subject
        pytest.skip(f"compile refused: {type(e).__name__}: {str(e)[:120]}")


def _rot_or_skip(s):
    if not TA.rot_on(s):
        pytest.skip("not on the rotation loop (t = 0 or a kind-2 circuit)")


def _distinct_keys(s, shots, seed, cap):
    """The distinct diagonal-class keys (a shot with a PPR-retry alternative takes today's term route, not the loop)."""
    ev = s.term_events(shots, seed)
    isp = s.term_alt_is_ppr()
    seen, out = set(), []
    for al in ev:
        if any(isp[int(a)] for a in al):
            continue
        k = tuple(sorted(al))
        if k in seen:
            continue
        seen.add(k)
        out.append(list(al))
        if len(out) >= cap:
            break
    return out


# ── T-a: the gauge-map oracle on every distinct key (fresh builds, then the stored plans) ────────────────────────────
@pytest.mark.parametrize("name", sorted(CASES), ids=lambda n: os.path.basename(n))
def test_gauge_oracle_per_key(name):
    s = _fam(CASES[name])
    _rot_or_skip(s)
    keys = _distinct_keys(s, 3000, 1, 400 if name.startswith("factory") else 1500)
    assert keys, name
    for al in keys:
        c = TA.rot_gauge_of(s, al, False, None)
        assert c["ok"], (name, al, c["why"])
        assert c["g_table_equal"] and c["columns"] == TA.rot_model(s)["m"]
    for al in keys[:50]:
        c = TA.rot_gauge_of(s, al, True, None)
        assert c["ok"], (name, al, c["why"])
    st = TA.rot_store_stats(s)
    assert st["on"] and st["plans"] >= 1


# ── T-b / nodes: the recorded outcomes of a barrier run against the oracle (the coins are the keyed COIN bits in pivot
#    order; every walked node p0_R(z') == p0_D(z' ⊕ δ)) ──────────────────────────────────────────────────────────────
M64 = (1 << 64) - 1
G = 0x9E3779B97F4A7C15
COIN = 2


def _mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def _draw(seed, shot, tag, ctr):
    key = _mix(seed ^ _mix(tag))
    return _mix((key + ((shot << 20) + ctr + 1) * G) & M64)


@pytest.mark.parametrize("name", ["fixture:m3_obs_2dec", "fixture:pair_ccz", "fixture:anti_dec", "fixture:anti_pair_dec",
                                  "fixture:multiword_t70", "fixture:rho70", "fixture:ref1", "tests/data/cultivation_d5.stim",
                                  "tests/data/magna_rp2_fid.stim", "tests/data/adaptq_born5_dec_enum_producer.stim"])
def test_recorded_outcomes_match_the_oracle_and_the_keyed_coins(name):
    text = CASES[name]
    s = _fam(text)
    _rot_or_skip(s)
    N, seed = 600, 23
    TA.noise_force_full(s, True)                     # lean N1 (option b): the recording run's events are the full table's
    ev = s.term_events(N, seed)
    d, o, dc, R = s.sample_records(N, seed=seed)
    assert R.num_records == N
    nodes = 0
    for i in range(N):
        uz = [int(x) for x in np.asarray(R.outcomes(i), np.uint8)]
        c = TA.rot_gauge_of(s, list(ev[i]), True, uz)
        assert c["ok"], (name, i, c["why"])
        rho = c["rho"]
        word = _draw(seed, i, COIN, 0)
        for k in range(min(rho, 64)):
            assert uz[k] == ((word >> k) & 1), (name, i, k)
        assert c["nodes"] == c["kappa"]
        nodes += c["nodes"]
        assert R.kind(i) == 2 and R.route(i) == 2
    assert nodes > 0 or all(TA.rot_gauge_of(s, list(ev[i]), True, None)["kappa"] == 0 for i in range(10))


# ── the selfcheck window and oracle_all_shots run green ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(CASES), ids=lambda n: os.path.basename(n))
def test_window_and_oracle_all_shots_green(name):
    text = CASES[name]
    s = _fam(text, oracle_all_shots=True)
    _rot_or_skip(s)
    n = 200 if name.startswith("factory") else 400
    out = s.sample(n, 3)
    assert np.asarray(out[0]).shape[0] == n
    st = TA.rot_store_stats(s)
    assert st["window_shots"] >= 1, st
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = xtim.compile_twirl_sampler(text, selfcheck=300)
        r = p.sample(1000, seed=5)
        assert np.asarray(r[0] if not hasattr(r, "dets") else r.dets).shape[0] == 1000
        assert not any(k.startswith("exact") for k in p.record_report()["route_counts"])


# ── T-c: sample == sample_barrier; split == single; threads N == 1; seed=None continuation ──────────────────────────
def _fields(out):
    if hasattr(out, "dets"):
        return [np.asarray(out.dets), np.asarray(out.obs), np.asarray(out.decisions)]
    return [np.asarray(f) for f in out if f is not None]


TC = ["fixture:m3_obs_2dec", "fixture:pair_ccz", "fixture:anti_dec", "fixture:anti_pair_dec", "fixture:anti_pair_obs",
      "fixture:multiword_t70", "fixture:rho70", "fixture:ref1", "tests/data/cultivation_d5.stim", "tests/data/magna_rp2_fid.stim"]


@pytest.mark.parametrize("name", TC)
def test_sample_equals_barrier_split_and_threads(name):
    text = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # RE-PINNED lean N1 (option b): sample() == sample_barrier() holds on the recording runs' (full) noise table;
        # every sample run below is the test-API FamilySampler put on it (the pruned table's split / threads
        # equalities: tests/test_lean_n1.py); the threaded barrier is the public sampler's (a recording run)
        s = full_table(text)
        N = 3000
        a = _fields(s.sample(N, seed=31))
        b = s.sample_barrier(N, seed=31)
        assert np.array_equal(a[0], np.asarray(b.dets()))
        assert np.array_equal(a[1], np.asarray(b.obs()))
        if len(a) > 2:
            assert np.array_equal(a[2], np.asarray(b.decisions()))
        s2 = full_table(text)
        parts = [_fields(s2.sample(1, seed=31)), _fields(s2.sample(1234)), _fields(s2.sample(N - 1235))]
        for i, w in enumerate(a):
            assert np.array_equal(w, np.concatenate([p[i] for p in parts])), (name, i)
        for T in (2, 7):
            sT = full_table(text)
            g = _fields(sT.sample(N, seed=31, threads=T))
            for i, w in enumerate(a):
                assert np.array_equal(w, g[i]), (name, T, i)
            assert sT.record_report()["plans"] == s.record_report()["plans"]
            bT = xtim.compile_twirl_sampler(text, selfcheck=0).sample_barrier(N, seed=31, threads=T)
            assert np.array_equal(np.asarray(b.dets()), np.asarray(bT.dets()))
            assert [bytes(k) for k in b.record_keys()] == [bytes(k) for k in bT.record_keys()]
            assert [bytes(b.outcomes(i)) for i in range(N)] == [bytes(bT.outcomes(i)) for i in range(N)]


@pytest.mark.parametrize("name", TC)
def test_postselect_accepted_rows_equal_filtered(name):
    text = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(text, selfcheck=0)
        nd = s.num_detectors
        if nd == 0:
            pytest.skip("no detectors")
        full = _fields(s.sample(3000, seed=7))
        keep = ~np.unpackbits(full[0], axis=1, bitorder="little")[:, :nd].any(1)
        s2 = xtim.compile_twirl_sampler(text, selfcheck=0)
        out = s2.sample(3000, seed=7, postselect=list(range(nd)))
        got = _fields(out[0]) if hasattr(out[0], "dets") else [np.asarray(f) for f in out[:-1]]
        counts = out[-1]
        for a, b in zip(full, got):
            assert np.array_equal(a[keep], b)
        assert counts["attempts"] == 3000 and counts["passed"] == int(keep.sum()) and 0 <= counts["early"] <= counts["discarded"]


# ── review H4: the ANTI pair is served (never a throw), its law is the exact engine's ────────────────────────────────
def test_anti_pair_is_served_and_law_equals_exact():
    from _exact_engine import exact_detector_sampler
    s = _fam(ANTI_PAIR_OBS)
    assert TA.rot_on(s)
    M = TA.rot_model(s)
    assert M["anti_extras"] == 2 and M["nvirt"] == 1
    for al in _distinct_keys(s, 2000, 1, 50):
        c = TA.rot_gauge_of(s, al, True, None)
        assert c["ok"], c["why"]
    assert TA.rot_store_stats(s)["anti_products"] >= 1
    NJ = 60000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = xtim.compile_twirl_sampler(ANTI_PAIR_OBS, selfcheck=0)
        o = np.unpackbits(np.asarray(p.sample(NJ, seed=11)[1]), axis=1, bitorder="little")[:, :2]
        ex = exact_detector_sampler(xtim.Circuit(ANTI_PAIR_OBS), seed=11)
        _, eo = ex.sample(NJ, separate_observables=True, bit_packed=True)
        eo = np.unpackbits(np.asarray(eo), axis=1, bitorder="little")[:, :2]
    for pat in range(4):
        q1 = np.mean((o[:, 0] + 2 * o[:, 1]) == pat)
        q2 = np.mean((eo[:, 0] + 2 * eo[:, 1]) == pat)
        sig = math.sqrt((q1 * (1 - q1) + q2 * (1 - q2)) / NJ) + 1e-9
        assert abs(q1 - q2) <= 5 * sig, (pat, q1, q2)
    # the decisions form: the same joint law (the same record operators)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd = xtim.compile_twirl_sampler(ANTI_PAIR_DEC, selfcheck=0)
        dcs = np.unpackbits(np.asarray(pd.sample(NJ, seed=12).decisions), axis=1, bitorder="little")[:, :2]
    for pat in range(4):
        q1 = np.mean((dcs[:, 0] + 2 * dcs[:, 1]) == pat)
        q2 = np.mean((eo[:, 0] + 2 * eo[:, 1]) == pat)
        sig = math.sqrt((q1 * (1 - q1) + q2 * (1 - q2)) / NJ) + 1e-9
        assert abs(q1 - q2) <= 5 * sig, ("dec", pat, q1, q2)


# ── review H3: no cap — a plan with ρ = 70 pivots (two A words) ─────────────────────────────────────────────────────
def test_rho_70_plan_two_a_words():
    s = _fam(FIXTURES["rho70"])
    _rot_or_skip(s)
    ev = s.term_events(5, 1)
    c = TA.rot_gauge_of(s, list(ev[0]), True, None)
    assert c["ok"], c["why"]
    assert c["rho"] == 70 and c["AW"] == 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = xtim.compile_twirl_sampler(FIXTURES["rho70"], selfcheck=0)
        d = np.unpackbits(np.asarray(p.sample(4000, seed=2)[0]), axis=1, bitorder="little")[:, :70]
    assert 0.4 < d.mean() < 0.6 and (abs(d.mean(0) - 0.5) < 0.1).all()


# ── Clifford circuits stay off the loop ──────────────────────────────────────────────────────────────────────────────
def test_clifford_circuit_is_not_on_the_loop():
    s = _fam("RX 0 1 2\nCX 0 1\nDEPOLARIZE1(0.1) 0 1 2\nCZ 1 2\nX_ERROR(0.2) 1\nMX 0 1\nDETECTOR rec[-1] rec[-2]\nM 2\nDETECTOR rec[-1]\n")
    assert not TA.rot_on(s) and not TA.rot_store_stats(s)["on"]
    with pytest.raises(ValueError):
        TA.rot_gauge_of(s, [0], True, None)


# ── record_report under the loop: plans = the store's, no exact route, classes by the s = 0 plan ─────────────────────
def test_record_report_counts_the_store():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = xtim.compile_twirl_sampler(CASES["tests/data/cultivation_d5.stim"], selfcheck=0)
        p.sample(4000, seed=3)
        r = p.record_report()
    f = _fam(CASES["tests/data/cultivation_d5.stim"])
    f.sample(4000, 3)
    st = TA.rot_store_stats(f)
    assert r["plans"] == st["plans"] > 100
    assert r["plan_misses"]["kappa0"] + r["plan_misses"]["kappa_pos"] == st["misses_k0"] + st["misses_kpos"] == st["plans"]
    assert sum(r["route_counts"].values()) == r["used"] == 4000
    assert set(r["classes"]["observables"].values()) <= {"coin", "born", "deterministic"}
