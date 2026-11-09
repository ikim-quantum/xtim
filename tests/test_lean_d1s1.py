"""lean D1-S1 (2026-11-05): the rotation model and the compact per-alternative rotation records, compiled once at the
sampler's construction (qeccore/twirl_rotation.hpp, cpp/src/twirl_rotation.cpp, twirl_sampler_rot.cpp; lean/D1/DESIGN.md
§1 items 1–3). Byte-identical: nothing sampled reads them yet (the byte gate is the proof; `_ctor_digest` does not hash
them — its move is signed with S2's stream change, tests/test_lean_l6.py's pins stay).

Oracle: cpp/testapi/rotation_py.cpp `rot_oracle_model` / `rot_oracle_records` — the P2/D1 probe's own construction
(per-pair Pauli anticommutation for every row, its own RREFs, free_dir_of on its own Γ_n basis; the records from the
record operators' own transpose ⟨P_f, M_c⟩, never the sampler's prefix rows). The production objects must equal it
field by field on every public corpus circuit with T content, on the pair-term / Born / ANTI-decision fixtures (virtual
planes) and on a t > 64 (two-word) fixture. The records' construction-time decode check is `rot_check` (its named
refusal: cpp/tests/test_rotation_verify.cpp, the corruption seam).
"""
import glob
import os

import numpy as np
import pytest

from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# the small fixtures of lean P2 / D1 (tools/probe_p2_small.py / probe_d1_small.py on probe/pbc)
M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
      "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
      "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"
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
    # a Clifford circuit (t = 0): the model is empty, the records carry the channel words only
    "clifford": "RX 0 1 2\nCX 0 1\nDEPOLARIZE1(0.1) 0 1 2\nCZ 1 2\nX_ERROR(0.2) 1\nMX 0 1\nDETECTOR rec[-1] rec[-2]\nM 2\nDETECTOR rec[-1]\n",
}


def _multiword(pairs=35, nq=5):
    """t = 70 > 64 rotations (two sign words) at chi = 1: T / T_DAG pairs with noise between them (the cultivation
    pattern: the noiseless pair cancels, a fault between the two makes the rotation content), Cliffords between the pairs
    so the pushed axes differ, X readouts with detectors on pairs."""
    out = [f"RX {' '.join(map(str, range(nq)))}\n"]
    for i in range(pairs):
        q = i % nq
        out.append(f"T {q}\nX_ERROR(0.05) {q}\nT_DAG {q}\nDEPOLARIZE1(0.02) {q}\nCX {q} {(q + 1) % nq}\n")
    out.append("MX " + " ".join(map(str, range(nq))) + "\n")
    for q in range(0, nq - 1, 2):
        out.append(f"DETECTOR rec[{q - nq}] rec[{q + 1 - nq}]\n")
    out.append("OBSERVABLE_INCLUDE(0) rec[-1]\n")
    return "".join(out)


FIXTURES["multiword_t70"] = _multiword()

CORPUS = sorted(glob.glob(os.path.join(HERE, "data", "*.stim"))) + sorted(glob.glob(os.path.join(ROOT, "examples", "*.stim")))
CASES = {os.path.relpath(p, ROOT): open(p).read() for p in CORPUS}
CASES.update({"fixture:" + k: v for k, v in FIXTURES.items()})


def _compile(text):
    try:
        return TA.compile_family_sampler(text, selfcheck=0)
    except Exception as e:   # the corpus's compile refusals (unchanged by S1) are not this test's subject
        pytest.skip(f"compile refused: {type(e).__name__}: {str(e)[:120]}")


def _model_or_skip(s):
    M = TA.rot_model(s)
    assert M["built"], "every compiled sampler has a rotation model (lean D1-S3: any commuting axes)"
    if M["t"] == 0:
        pytest.skip("no T content")
    return M


def _assert_model_equal(M, O):
    scalars = ("built", "kind2", "t", "TW", "k", "KW", "m", "nch", "ng", "phi_rank", "gz_dim", "nx", "nvirt", "TXW", "anti_extras")
    for k in scalars:
        assert M[k] == O[k], (k, M[k], O[k])
    arrays = ("rho", "scan", "rank_of", "cols_off", "cols", "eta", "live", "cone_gen", "cone_row", "gn_rows", "gn_piv",
              "gn_col_to_row", "gn_a", "gn_b", "gn_phase", "gn_par", "x_T0", "x_logical")
    for k in arrays:
        a, b = np.asarray(M[k]), np.asarray(O[k])
        assert a.shape == b.shape and np.array_equal(a, b), (k, a.shape, b.shape)
    assert M["x_fd"] == O["x_fd"]


def _assert_records_equal(R, O):
    for k in ("TW", "XR", "CW", "alts"):
        assert R[k] == O[k], (k, R[k], O[k])
    assert np.array_equal(np.asarray(R["off"]), np.asarray(O["off"]))
    assert np.array_equal(np.asarray(R["rec"]), np.asarray(O["rec"]))


@pytest.mark.parametrize("name", sorted(CASES), ids=lambda n: os.path.basename(n))
def test_model_equals_oracle(name):
    s = _compile(CASES[name])
    M = _model_or_skip(s)
    O = TA.rot_oracle_model(s)
    _assert_model_equal(M, O)
    # structural sanity: every column's row bits are the inverted index, the cone rows annihilate Γ_n
    t, TW, m = M["t"], M["TW"], M["m"]
    rho = np.asarray(M["rho"]).reshape(m, TW)
    cols_off, cols = np.asarray(M["cols_off"]), np.asarray(M["cols"])
    for j in range(t):
        cs = cols[cols_off[j]:cols_off[j + 1]]
        assert np.all(np.diff(cs) > 0)
        assert set(cs.tolist()) == {c for c in range(m) if (int(rho[c, j >> 6]) >> (j & 63)) & 1}
    live = np.asarray(M["live"])
    for j in range(t):
        assert ((int(live[j >> 6]) >> (j & 63)) & 1) == int(cols_off[j + 1] > cols_off[j])
    gn = np.asarray(M["gn_rows"]).reshape(-1, TW)
    cone = np.asarray(M["cone_row"]).reshape(-1, TW)
    for r in gn:
        for g in cone:
            assert sum(bin(int(a & b)).count("1") for a, b in zip(r, g)) % 2 == 0
    assert len(M["gn_piv"]) == t - M["phi_rank"]
    assert M["gz_dim"] <= len(M["gn_piv"])


@pytest.mark.parametrize("name", sorted(CASES), ids=lambda n: os.path.basename(n))
def test_records_equal_oracle(name):
    s = _compile(CASES[name])
    M = TA.rot_model(s)
    if not M["built"]:
        pytest.skip("kind-2 circuit: no records")
    R = TA.rot_records(s)
    O = TA.rot_oracle_records(s)
    _assert_records_equal(R, O)
    assert R["alts"] == len(s.term_alt_faults())
    assert TA.rot_check(s) == (R["alts"], 0)
    # the layout: every record = [2·TW sign][2 token][2·XR extras][nz][3·nz], ascending nonzero entries
    rec, off = np.asarray(R["rec"]), np.asarray(R["off"])
    TW, XR, CW = R["TW"], R["XR"], R["CW"]
    hdr = 2 * TW + 2 + 2 * XR + 1
    for a in range(R["alts"]):
        r = rec[off[a]:off[a + 1]]
        nz = int(r[hdr - 1])
        assert len(r) == hdr + 3 * nz
        ws = r[hdr::3][:nz]
        assert np.all(np.diff(ws) > 0) and (nz == 0 or ws.max() < CW)
        assert all((int(r[hdr + 3 * i + 1]) | (int(r[hdr + 3 * i + 2]) << 32)) != 0 for i in range(nz))


def test_kind2_circuit_is_flagged_and_has_a_model():
    """A CH circuit: the term view has an X-content axis — flagged kind 2 (information). RE-PINNED lean D1-S3 (owner-signed):
    the rotation model covers any commuting axes, so it has a model and records, both equal to the independent oracle's."""
    s = _compile(CASES["tests/data/ch_cultivation.stim"])
    M = TA.rot_model(s)
    assert M["kind2"] and M["built"] and M["t"] > 0
    R = TA.rot_records(s)
    assert R["alts"] > 0 and TA.rot_check(s) == (R["alts"], 0)
    _assert_model_equal(M, TA.rot_oracle_model(s))


def test_multiword_t_over_64():
    s = _compile(FIXTURES["multiword_t70"])
    M = _model_or_skip(s)
    assert M["t"] == 70 and M["TW"] == 2
    _assert_model_equal(M, TA.rot_oracle_model(s))
    R = TA.rot_records(s)
    assert R["TW"] == 2
    _assert_records_equal(R, TA.rot_oracle_records(s))
    # some alternative sets a rotation in the second word
    rec, off = np.asarray(R["rec"]), np.asarray(R["off"])
    hi = [int(rec[off[a] + 2]) | (int(rec[off[a] + 3]) << 32) for a in range(R["alts"])]
    assert any(h != 0 for h in hi)


@pytest.mark.parametrize("name, nvirt, anti", [("pair_ccz", 1, 1), ("anti_dec", 3, 3), ("pair_cs", 0, 0), ("m3_obs", 0, 0)])
def test_virtual_planes_of_anti_extras(name, nvirt, anti):
    """ANTI extras carry a virtual coordinate each (bits [t, t + nvirt) of the extras' words): the P2/D1 fixtures'
    counts (pair_ccz: 1 virtual plane; the keyed-streams ANTI-decision fixture: 3)."""
    s = _compile(FIXTURES[name])
    M = _model_or_skip(s)
    assert (M["nvirt"], M["anti_extras"]) == (nvirt, anti)
    T0 = np.asarray(M["x_T0"]).reshape(M["nx"], M["TXW"])
    t = M["t"]
    virt = [[(int(T0[e, (t + v) >> 6]) >> ((t + v) & 63)) & 1 for v in range(nvirt)] for e in range(M["nx"])]
    for e in range(M["nx"]):
        assert (sum(virt[e]) > 0) == (not M["x_logical"][e]), (e, virt[e], M["x_logical"][e])
        assert (M["x_fd"][e] is None) == (not M["x_logical"][e])
    _assert_model_equal(M, TA.rot_oracle_model(s))


def test_clifford_circuit_model_is_empty_and_records_are_channel_words():
    s = _compile(FIXTURES["clifford"])
    M = TA.rot_model(s)
    assert M["built"] and not M["kind2"] and M["t"] == 0 and M["TW"] == 0
    assert len(M["gn_piv"]) == 0 and M["phi_rank"] == 0 and len(M["cone_gen"]) == 0
    _assert_model_equal(M, TA.rot_oracle_model(s))
    R = TA.rot_records(s)
    _assert_records_equal(R, TA.rot_oracle_records(s))
    rec, off = np.asarray(R["rec"]), np.asarray(R["off"])
    for a in range(R["alts"]):
        assert int(rec[off[a]]) == 0 and int(rec[off[a] + 1]) == 0   # token of the empty key = 0
    assert TA.rot_check(s) == (R["alts"], 0)


def test_term_view_holds_no_alt_table_and_footprint_reports_the_model():
    """The term view stays (axes, kernel: small) without its A × NW per-alternative table (the records are built from a
    chunked propagation, D1 review M2); the rotation group is reported by the footprint."""
    s = _compile(open(os.path.join(ROOT, "examples", "cultivation_d5.stim")).read())
    fp = {k: int(v) for k, v in s._footprint().items()}
    assert "rotation" in fp and "term_view" in fp
    R = TA.rot_records(s)
    assert fp["rotation"] >= 4 * len(R["rec"])
    # the per-alternative table would be 2 × A × NW × 8 bytes (NW = 5 here: 80 B per alternative); the retained view
    # (axes, ops, kernel, term rows) is far below that
    A = R["alts"]
    assert fp["term_view"] < 16 * A, (fp["term_view"], A)
    # the test API re-arms on demand (the oracle reads the alts): the model is still equal afterwards
    _assert_model_equal(TA.rot_model(s), TA.rot_oracle_model(s))


# ── a consumer compiled from a producer's ideal state (the carried-compose fixtures of test_jb1_carried_compose): the
# bare frame maps Z-strings of Γ_n to X content on the free register — the direction is (a, b, phase, par), not a Z
# direction + a sign bit (the P2 probe's pure-Z assumption holds on the canaries only) ──
MERGE_P3 = ("R 0\nRX 1\nT 1\nH 1\nCX 0 1\nCX 1 0\nOUTPUT_QUBITS out 0 1\n",
            "INPUT_QUBITS in 0 1\nDEPOLARIZE1(0.1) 0 1\nT 0\nCX 0 1\nS 0\nCZ 1 0\nCZ 0 1\nS 1\nT 0\nCZ 1 0\nM 0 1\n")


def test_consumer_from_ideal_state_has_x_content_directions():
    p = _compile(MERGE_P3[0])
    c = TA.compile_family_sampler_from_ideal_state(p.ideal_state(), MERGE_P3[1])
    M = _model_or_skip(c)
    assert M["t"] == 2 and M["k"] == 1
    assert int((np.asarray(M["gn_a"]) != 0).sum()) > 0, "the fixture's point: X free content on a Γ_n direction"
    _assert_model_equal(M, TA.rot_oracle_model(c))
    R = TA.rot_records(c)
    assert R["alts"] > 0
    _assert_records_equal(R, TA.rot_oracle_records(c))
    assert TA.rot_check(c) == (R["alts"], 0)


# ── S1 review H1: the extras' virtual coordinates. Three ANTI extras X0, X0X1, X1 (t = 1): their residual planes are
# x, x + y, y — X0X1's reduces PARTLY against X0's. The word of an extra that opens a coordinate is that coordinate alone
# (its plane is the basis vector); the echelon row carries new ⊕ used. Expected words 0x2 / 0x4 / 0x6 (bit 0 = the
# rotation); the old construction gave 0x2 / 0x6 / 0x6 (X1 a copy of X0X1 although their product X0 is ANTI). ──
ANTI3 = ("R 0 1 2\nRX 3\nT 3\nZ_ERROR(0.1) 3\nX_ERROR(0.1) 0 1\nMX 0 1 3\n"
         "DECISION(0) rec[-3]\nDECISION(1) rec[-3] rec[-2]\nDECISION(2) rec[-2]\n"
         "R 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\n")
ANTI3_OBS = ANTI3.replace("DECISION(0) rec[-3]\nDECISION(1) rec[-3] rec[-2]\nDECISION(2) rec[-2]\n",
                          "OBSERVABLE_INCLUDE(0) rec[-3]\nOBSERVABLE_INCLUDE(1) rec[-3] rec[-2]\nOBSERVABLE_INCLUDE(2) rec[-2]\n")


@pytest.mark.parametrize("text", [ANTI3, ANTI3_OBS], ids=["decisions", "observables"])
def test_anti_extras_partial_reduction_words(text):
    s = _compile(text)
    M = _model_or_skip(s)
    assert (M["t"], M["nx"], M["nvirt"], M["anti_extras"]) == (1, 3, 2, 3)
    words = [int(x) for x in np.asarray(M["x_T0"]).ravel()]
    assert words == [0x2, 0x4, 0x6], [hex(w) for w in words]
    O = TA.rot_oracle_model(s)
    assert [int(x) for x in np.asarray(O["x_T0"]).ravel()] == [0x2, 0x4, 0x6]
    _assert_model_equal(M, O)
    assert TA.rot_virtual_check(s) == (7, 0)   # every nonempty subset of the 3 extras


@pytest.mark.parametrize("name", sorted(CASES), ids=lambda n: os.path.basename(n))
def test_virtual_coordinates_independent_check(name):
    """Shares nothing with the echelon construction: for subsets of extras, virtual XOR = 0 <=> the product of their
    operators (times Z_{T0}) commutes with the certified group (G.reduce)."""
    s = _compile(CASES[name])
    M = TA.rot_model(s)
    if not M["built"] or M["nx"] == 0:
        pytest.skip("no model / no extras")
    checked, bad = TA.rot_virtual_check(s, 256, 3)
    assert checked > 0 and bad == 0, (checked, bad)


# the S1 review's random-circuit generator (/tmp/d1s1rev/fuzz.py): n 2-8, T / T† / CS / CCZ / CH at six densities,
# M / MX / MY readouts, detectors, observables and decisions as extras
_G1 = ["H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_Y_DAG", "H_YZ", "C_XYZ"]
_G2 = ["CX", "CY", "CZ", "SWAP", "ISWAP", "SQRT_XX", "SQRT_ZZ", "XCZ", "CXSWAP"]


def _fuzz_circuit(seed):
    import random
    r = random.Random(seed)
    n = r.randint(2, 8)
    L = [r.choice(["R", "RX", "RY"]) + " " + " ".join(map(str, range(n)))]
    depth = r.randint(6, 30)
    nct = {0: 0, 1: 2, 2: 6, 3: 1, 4: depth, 5: 3}[seed % 6]
    tpos = {r.randrange(depth) for _ in range(nct)}

    def noise():
        k = r.randrange(4)
        qs = r.sample(range(n), r.randint(1, n))
        return ["DEPOLARIZE1(0.02) ", "X_ERROR(0.03) ", "Z_ERROR(0.03) ", "PAULI_CHANNEL_1(0.01,0.02,0.005) "][k] + " ".join(map(str, qs))
    for d in range(depth):
        if r.random() < 0.6:
            L.append(noise())
        if d in tpos:
            x = r.random()
            if x < 0.5 or n < 2:
                L.append(r.choice(["T", "T_DAG"]) + f" {r.randrange(n)}")
            elif x < 0.8:
                a, b = r.sample(range(n), 2); L.append(r.choice(["CS", "CS_DAG"]) + f" {a} {b}")
            elif x < 0.93 and n >= 3:
                a, b, c = r.sample(range(n), 3); L.append(f"CCZ {a} {b} {c}")
            else:
                a, b = r.sample(range(n), 2); L.append(f"CH {a} {b}")
        else:
            k = r.random()
            if k < 0.4:
                qs = r.sample(range(n), r.randint(1, n)); L.append(r.choice(_G1) + " " + " ".join(map(str, qs)))
            elif k < 0.8 and n >= 2:
                m = r.randint(1, n // 2); qs = r.sample(range(n), 2 * m); L.append(r.choice(_G2) + " " + " ".join(map(str, qs)))
    bases = [r.choice(["M", "MX", "MY"]) for _ in range(n)]
    for q in range(n):
        L.append(f"{bases[q]} {q}")
    for _ in range(r.randint(0, 2)):
        ks = r.sample(range(1, n + 1), r.randint(1, min(2, n))); L.append("DETECTOR " + " ".join(f"rec[-{k}]" for k in ks))
    for i in range(r.randint(0, 3)):
        ks = r.sample(range(1, n + 1), r.randint(1, min(3, n))); L.append(f"OBSERVABLE_INCLUDE({i}) " + " ".join(f"rec[-{k}]" for k in ks))
    if r.random() < 0.4:
        for i in range(r.randint(1, 3)):
            ks = r.sample(range(1, n + 1), r.randint(1, min(2, n))); L.append(f"DECISION({i}) " + " ".join(f"rec[-{k}]" for k in ks))
    return "\n".join(L) + "\n"


@pytest.mark.parametrize("lo", list(range(0, 240, 40)))
def test_virtual_coordinates_fuzz(lo):
    """Random circuits (the review's generator): on every compiled model with extras the independent virtual check holds,
    the model equals the oracle and the records decode; at least a few models with ANTI extras / nvirt >= 2 are reached."""
    anti = nvirt2 = built = 0
    for seed in range(lo, lo + 40):
        try:
            s = TA.compile_family_sampler(_fuzz_circuit(seed), selfcheck=0)
        except Exception:
            continue   # a compile refusal (unchanged by S1) is not this test's subject
        M = TA.rot_model(s)
        if not M["built"]:
            assert M["kind2"]
            continue
        built += 1
        _assert_model_equal(M, TA.rot_oracle_model(s))
        R = TA.rot_records(s)
        _assert_records_equal(R, TA.rot_oracle_records(s))
        assert TA.rot_check(s) == (R["alts"], 0)
        if M["nx"]:
            checked, bad = TA.rot_virtual_check(s, 128, seed)
            assert bad == 0, (seed, checked, bad)
        anti += M["anti_extras"] > 0
        nvirt2 += M["nvirt"] >= 2
    assert built >= 5 and anti >= 1, (built, anti, nvirt2)


def test_screened_term_view_is_completed_on_demand():
    """Review M1: an unarmed Z-only sampler keeps a z-screened term view (axes + ops: what the model and the records
    read; no KerPhi); the test API's accessors complete it on demand (TermResidual::arm) and read its kernel."""
    s = _compile(open(os.path.join(ROOT, "examples", "cultivation_d5.stim")).read())
    M = _model_or_skip(s)
    assert s.term_merge_stats()["terms"] == M["t"]          # arms (completes) the screened view
    assert len(s.term_kernel()) == M["t"] - s.term_rank()   # KerPhi built on demand
    _assert_model_equal(TA.rot_model(s), TA.rot_oracle_model(s))
