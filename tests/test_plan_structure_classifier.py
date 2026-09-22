"""E3 (exact-residual arc, Task 3): plan-structure exposure + the per-(check, plan)
exact indefiniteness classifier PROTOTYPE composing it.

Exposed primitives under test (ADDITIVE, pure data — no new engine computation):
  * FramedSuperposition.certified_symplectic() — the certified generators as full
    n-qubit symplectic rows + exact phases (row b == sigmas() bit b).
  * BarrierBuffer.plan_structure(plan_key_bytes) — a plan key parsed into the residual
    normal form (prefix_xz, a, cz) + the plan's ShotLaw dump (r, kappa, det_signs,
    coin/kernel masks, kernel logicals with exact phases) via the IDENTICAL
    build_shot_law call materialize(i) makes.

The classifier prototype (to be productized in adaptq, Task 5) implements the
per-(check, plan) classification of the exact-residual engine-surface memo (section 3):
conjugate the check through the plan residual O' = (P.CZ.S^a)^dag O (P.CZ.S^a), span-test
O' against <bare certified generators  U  plan kernel logicals> (born-measured decision
operators would extend the basis; none of the in-suite workloads declares Born-class
decisions), then
  * in span            -> DEFINITE: sign = plan-constant phase bit XOR the selected
                          kernel outcome bits (record-linear; the fair coins never enter
                          — the collapsed state is coin-independent);
  * out of span, anticommuting with a basis element -> e = 0 (balanced twirl-split);
  * out of span, commuting -> plan-constant Born data from bare expectations
                          <O' L_S>_bare over kernel-outcome subsets S; per-shot
                          e = sum_S (-1)^{S.k} <O' L_S> / sum_S (-1)^{S.k} <L_S>,
                          branch weights (1 +- e)/2.

ORACLE (exact, zero statistical assertions): the materialized per-shot collapsed state.
  * Every materialized port-signature check must classify DEFINITE and its predicted
    sign must equal the materialized sign — exact integer equality, every shot.
  * The T2 span accounting must be reproduced exactly: the qrm magic producer's
    34 gap (shot, check) occurrences all classify out-of-span(bare certified), every
    covered check classifies in-span; cat_err and steane_h have ZERO gaps (no-gap
    controls).
  * The carried logical triple (La, Lb, La.Lb — k_port==1) must reproduce the
    materialized expectation per shot: definite values exactly (+-1); genuinely
    indefinite values (|e| < 1, e.g. the carried magic +-1/sqrt(2)) to <= 1e-12 —
    both sides are DETERMINISTIC float evaluations that differ only in ulp-level
    rounding path (bare-composition vs materialized read); nothing statistical.

T5 step 0 (born-op exposure): ``BarrierBuffer.born_dec()`` exposes the Born-measured
decision operators (LOGICAL/ANTI-class DECISIONs), and the classifier's span basis is
extended with the PLAN-CONJUGATED born ops, whose per-shot eigenvalue is the RAW outcome
bit coins[r+kappa+j].  Exercised by the vendored born-branching workload
``adaptq_born5_dec_enum_producer.stim`` (the adaptq dec-enum class: five independent
heavily-biased born decisions, joint rare branch p ~ 6.7e-5) — see the ``born5`` tests
at the bottom: classifier expectations vs materialized reads, exact.

Honest-coverage note (reported, not hidden): no in-suite workload produces a plan with
kappa > 0 (verified per plan below), nor is one reachable synthetically from these bare
states by single-a / single-cz contents (searched). The kernel-logical FIELDS of
plan_structure are therefore covered structurally (shapes/dtypes/empty contract) and by
pass-through from the engine's ShotLaw (whose kappa>0 behaviour is gated by the twirl
kernel suite); the classifier's kappa>0 composition path is exercised only vacuously
(the born block of the measured-op machinery is the non-vacuous sibling, born5 tests).
"""
import pathlib

import numpy as np
import pytest

import xtim
import xtim._xtim as _xtim

_DATA = pathlib.Path(__file__).parent / "data"

_CAT_ERR = ("R 0 1\nH 0\nCX 0 1\nX_ERROR(0.25) 0\n"
            "R 2\nCX 0 2\nCX 1 2\nM 2\nDECISION(0) rec[-1]\n"
            "OUTPUT_QUBITS out 0 1\n")

_SHOTS = 256
_SEED = 20260730

# (name, circuit-text getter, expected k_port, expected T2 gap (shot,check) occurrences)
_WORKLOADS = [
    ("cat_err", lambda: _CAT_ERR, 0, 0),
    ("steane_h", lambda: (_DATA / "adaptq_steane_h_producer.stim").read_text(), 0, 0),
    ("qrm_magic", lambda: (_DATA / "adaptq_qrm_magic_producer.stim").read_text(), 1, 34),
]

_SUMMARY = {}


def _compile(text):
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    return s._inner if hasattr(s, "_inner") else s


# ---------------------------------------------------------------------------
# Exact Pauli algebra: (x bits, z bits, phase mod 4), operator = i^phase X^x Z^z.
# Same convention as the engine's Pauli (pauli.hpp): multiply phase rule
# (pa + pb + 2*(a.z & b.x)) mod 4; Hermitian support rep has phase = |x & z| mod 4
# (port_signature's herm convention: sign bit d = (phase - xz_overlap) mod 4 -> 0/2).
# ---------------------------------------------------------------------------

def _pmul(a, b):
    ax, az, ap = a
    bx, bz, bp = b
    ph = (ap + bp + 2 * int(np.sum(az & bx) & 1)) % 4
    return (ax ^ bx, az ^ bz, ph)


def _sym_prod(a, b):
    return int((np.sum(a[0] & b[1]) + np.sum(a[1] & b[0])) & 1)


def _herm(xbits, zbits):
    return (xbits.copy(), zbits.copy(), int(np.sum(xbits & zbits)) % 4)


def _key_to_pauli(key, wires, W, n):
    xb = np.zeros(n, dtype=np.uint8)
    zb = np.zeros(n, dtype=np.uint8)
    for i, w in enumerate(wires):
        xb[w] = key[i]
        zb[w] = key[W + i]
    return _herm(xb, zb)


def _conjugate_through_plan(O, plan):
    """O' = (P.CZ.S^a)^dag O (P.CZ.S^a): conjugate by P (anticommutation sign only —
    P's global phase is excluded from the key and cancels), then by CZ
    (z_j ^= x_l, z_l ^= x_j, phase += 2*(x_j & x_l)), then by S^dag on the a-support
    (x_q=1: z_q ^= 1, phase += 3; from S^dag X S = -Y = i^3 X Z)."""
    x, z, ph = O[0].copy(), O[1].copy(), O[2]
    n = len(np.asarray(plan["a"]))
    pxz = np.asarray(plan["prefix_xz"]).astype(np.uint8)
    ph = (ph + 2 * _sym_prod((x, z, 0), (pxz[:n], pxz[n:], 0))) % 4
    for j, l in np.asarray(plan["cz"]).reshape(-1, 2):
        xj, xl = int(x[j]), int(x[l])
        z[j] ^= xl
        z[l] ^= xj
        ph = (ph + 2 * (xj & xl)) % 4
    a = np.asarray(plan["a"]).astype(np.uint8)
    for q in np.nonzero(a & x)[0]:
        z[q] ^= 1
        ph = (ph + 3) % 4
    return (x, z, ph)


class _Gf2Solver:
    """GF(2) RREF span solver with combination tracking over the given basis rows."""

    def __init__(self, rows):
        self.m = len(rows)
        self.red, self.redc, self.piv = [], [], []
        eye = np.eye(self.m, dtype=np.uint8) if self.m else np.zeros((0, 0), np.uint8)
        for i, row in enumerate(rows):
            r = row.copy()
            c = eye[i].copy()
            for pr, pc, pcol in zip(self.red, self.redc, self.piv):
                if r[pcol]:
                    r ^= pr
                    c ^= pc
            nz = np.nonzero(r)[0]
            if len(nz) == 0:
                continue
            lead = int(nz[0])
            for k in range(len(self.red)):
                if self.red[k][lead]:
                    self.red[k] ^= r
                    self.redc[k] ^= c
            self.red.append(r)
            self.redc.append(c)
            self.piv.append(lead)

    def solve(self, v):
        """Combination uint8[m] with XOR-sum(rows[comb]) == v, or None if out of span."""
        v = v.copy()
        c = np.zeros(self.m, dtype=np.uint8)
        for pr, pc, pcol in zip(self.red, self.redc, self.piv):
            if v[pcol]:
                v ^= pr
                c ^= pc
        return c if not v.any() else None


def _make_exp(make_state):
    """Exact <i^p X^x Z^z> evaluator on fresh states via the S-rotation composition:
    apply S on every Y site of a FRESH state copy — S (XZ) S^dag = iX clears the
    overlap — then one pauli_expectation_xz call; i^{p+|Y|} must be real (+-1)."""

    def pexp(T):
        x, z, p = T
        ys = np.nonzero(x & z)[0]
        st = make_state()
        for q in ys:
            st.apply_clifford(1, int(q), 0)      # kind 1 = S
        p2 = (p + len(ys)) % 4
        z2 = z.copy()
        z2[ys] ^= 1
        val = st.pauli_expectation_xz([int(q) for q in np.nonzero(x)[0]],
                                      [int(q) for q in np.nonzero(z2)[0]])
        assert p2 in (0, 2), "anti-Hermitian expectation requested (phase %d)" % p2
        return val if p2 == 0 else -val

    return pexp


def _classify(check_key, plan, gens, gsolver, bare_exp, wires, W, n, borns=()):
    """Per-(check, plan) exact classification (memoize on the plan-key bytes + check).

    ``borns`` (T5 step 0) is the buffer's born-decision operator list (n-wide Pauli
    triples, PHYSICAL frame — conjugated through the plan here, exactly like the
    check); the measured-operator basis is <certified gens U plan kernel logicals U
    plan-conjugated born ops>, with per-shot outcome bits coins[r:r+kappa] (kernels)
    then coins[r+kappa:r+kappa+nborn] (raw born outcomes).

    Returns dict:
      in_span_bare : unconjugated check in span(certified) — the T2 accounting notion
      definite     : conjugated O' in span(certified U kernels U conj born ops)
      const_bit    : plan-constant sign bit (definite)
      measured_sel : uint8[kappa + nborn] — measured-op outcome bits entering the
                     sign (definite; kernel block first, then born block)
      branch       : indefinite plan-constant data ({'zero': True} for the balanced
                     split; else the <O' M_S>/<M_S> bare moments per measured-op
                     subset, M ranging over kernels then conj born ops)
    """
    O = _key_to_pauli(check_key, wires, W, n)
    in_span_bare = gsolver.solve(np.concatenate([O[0], O[1]])) is not None

    Op = _conjugate_through_plan(O, plan)
    kappa = int(plan["kappa"])
    kxz = np.asarray(plan["kernel_xz"]).reshape(kappa, 2 * n).astype(np.uint8)
    kph = np.asarray(plan["kernel_phase"]).astype(np.uint8)
    kerns = [(kxz[j][:n], kxz[j][n:], int(kph[j])) for j in range(kappa)]
    bconj = [_conjugate_through_plan(B, plan) for B in borns]
    meas = kerns + bconj                     # measured ops, outcome-bit order
    nmeas = len(meas)

    if nmeas:
        # Soundness of the per-factor eigenvalue composition below: every measured
        # op must commute with every certified generator and with the other
        # measured ops (else an earlier eigenvalue is destroyed by a later
        # projection).  Holds for every in-suite workload; a violation is a loud
        # failure, never a silent misprediction.
        for m in meas:
            assert not any(_sym_prod(m, g) for g in gens), \
                "measured op anticommutes with a certified generator"
        for a in range(nmeas):
            for b in range(a + 1, nmeas):
                assert _sym_prod(meas[a], meas[b]) == 0, \
                    "non-commuting measured ops: composition unsupported"
        basis = [np.concatenate([g[0], g[1]]) for g in gens] + \
                [np.concatenate([m[0], m[1]]) for m in meas]
        solver = _Gf2Solver(basis)
    else:
        solver = gsolver
    comb = solver.solve(np.concatenate([Op[0], Op[1]]))
    if comb is not None:
        Q = (np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8), 0)
        for i in np.nonzero(comb[:len(gens)])[0]:
            Q = _pmul(Q, gens[i])
        msel = np.zeros(nmeas, dtype=np.uint8)          # T3 review Minor: plain zeros
        for j in range(nmeas):
            if comb[len(gens) + j]:
                msel[j] = 1
                Q = _pmul(Q, meas[j])
        assert Q[0].tobytes() == Op[0].tobytes() and Q[1].tobytes() == Op[1].tobytes()
        delta = (Op[2] - Q[2]) % 4
        assert delta in (0, 2), "non-Hermitian decomposition (delta=%d)" % delta
        return dict(in_span_bare=in_span_bare, definite=True,
                    const_bit=delta // 2, measured_sel=msel)
    if any(_sym_prod(Op, g) for g in gens) or any(_sym_prod(Op, m) for m in meas):
        return dict(in_span_bare=in_span_bare, definite=False, branch=dict(zero=True))
    moments = {}
    for S in range(1 << nmeas):
        L = (np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8), 0)
        for j in range(nmeas):
            if (S >> j) & 1:
                L = _pmul(L, meas[j])
        moments[S] = (bare_exp(_pmul(Op, L)), bare_exp(L))
    return dict(in_span_bare=in_span_bare, definite=False,
                branch=dict(zero=False, moments=moments, nmeas=nmeas))


def _expectation_of(cls, mbits):
    """Per-shot expectation from the plan-constant classification + the shot's
    measured-op outcome bits ``mbits = coins[r : r + kappa + nborn]`` (kernel
    outcomes then raw born outcomes)."""
    if cls["definite"]:
        s = cls["const_bit"]
        for j, b in enumerate(mbits):
            if b and cls["measured_sel"][j]:
                s ^= 1
        return -1.0 if s else 1.0
    br = cls["branch"]
    if br["zero"]:
        return 0.0
    num = den = 0.0
    for S, (m_ol, m_l) in br["moments"].items():
        par = 0
        for j in range(br["nmeas"]):
            if (S >> j) & 1 and mbits[j]:
                par ^= 1
        f = -1.0 if par else 1.0
        num += f * m_ol
        den += f * m_l
    return num / den


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=_WORKLOADS, ids=[w[0] for w in _WORKLOADS])
def workload(request):
    name, get_text, k_port_expected, gaps_expected = request.param
    text = get_text()
    inner = _compile(text)
    wires = list(inner.output_wires())
    buf = inner.sample_barrier(_SHOTS, _SEED)
    bare = _xtim.bare_state_of(text)
    n = bare.n
    cs = bare.certified_symplectic()
    ngens = int(cs["ngens"])
    xz = np.asarray(cs["xz"]).astype(np.uint8)
    phs = np.asarray(cs["phase"]).astype(np.uint8)
    gens = [(xz[b][:n], xz[b][n:], int(phs[b])) for b in range(ngens)]
    gsolver = _Gf2Solver([np.concatenate([g[0], g[1]]) for g in gens])
    law = bare.port_sigma_law(wires)
    stab = np.asarray(law["stab"])
    law_keys = {bytes(bytearray(stab[j])) for j in range(int(law["ns"]))}
    plans = {}
    for i in range(_SHOTS):
        pk = bytes(bytearray(buf.plan_key(i)))
        if pk not in plans:
            plans[pk] = buf.plan_structure(pk)
    return dict(name=name, text=text, wires=wires, W=len(wires), buf=buf, bare=bare,
                n=n, gens=gens, gsolver=gsolver, law_keys=law_keys, plans=plans,
                k_port_expected=k_port_expected, gaps_expected=gaps_expected,
                bare_exp=_make_exp(lambda: _xtim.bare_state_of(text)))


# ---------------------------------------------------------------------------
# Exposure contracts: certified_symplectic
# ---------------------------------------------------------------------------

def test_certified_symplectic_contract(workload):
    """Shapes/dtypes; row order == sigma bit order (ngens == buf.sig_bits); every
    generator operator has EXACT +1 eigenvalue on the bare state."""
    bare, buf, n = workload["bare"], workload["buf"], workload["n"]
    cs = bare.certified_symplectic()
    ngens = int(cs["ngens"])
    xz = np.asarray(cs["xz"])
    phs = np.asarray(cs["phase"])
    assert int(cs["n"]) == n
    assert xz.shape == (ngens, 2 * n) and xz.dtype == np.uint8
    assert phs.shape == (ngens,) and phs.dtype == np.uint8
    assert ngens == int(buf.sig_bits), "row order contract: ngens must equal sigma width"
    bare_exp = workload["bare_exp"]
    for g in workload["gens"]:
        assert bare_exp(g) == 1.0, "certified generator not a +1 stabilizer of bare"


# ---------------------------------------------------------------------------
# Exposure contracts: plan_structure
# ---------------------------------------------------------------------------

def _serialize_plan(plan):
    """Re-serialize a parsed plan to key bytes per the documented format
    (prefix x/z support words LE || a-mask || 0xFF || cz u16-LE pairs)."""
    n = int(plan["n"])
    NW = (n + 63) // 64
    pxz = np.asarray(plan["prefix_xz"]).astype(np.uint8)

    def words_bytes(bits):
        w = np.zeros(NW, dtype=np.uint64)
        for q in np.nonzero(bits)[0]:
            w[q >> 6] |= np.uint64(1) << np.uint64(q & 63)
        return w.astype("<u8").tobytes()

    out = words_bytes(pxz[:n]) + words_bytes(pxz[n:])
    out += bytes(int(b) & 1 for b in np.asarray(plan["a"]))
    out += b"\xff"
    for f, s in np.asarray(plan["cz"]).reshape(-1, 2):
        out += bytes([f & 255, (f >> 8) & 255, s & 255, (s >> 8) & 255])
    return out


def test_plan_structure_parse_roundtrip(workload):
    """serialize(parse(key)) must be byte-identical to the recorded key for every
    distinct non-empty plan; the empty key must parse to the identity residual."""
    for pk, plan in workload["plans"].items():
        if pk == b"":
            assert not np.asarray(plan["prefix_xz"]).any()
            assert not np.asarray(plan["a"]).any()
            assert np.asarray(plan["cz"]).size == 0
            assert int(plan["r"]) == 0 and int(plan["kappa"]) == 0
            continue
        assert _serialize_plan(plan) == pk, "plan-key parse/serialize round-trip broken"


def test_plan_structure_shapes_and_record_widths(workload):
    """Structural pins: gw == sigmas() columns; r + kappa == len(coins(i)) for every
    shot (no Born decisions declared in these workloads — pinned via born_dec());
    mask/kernel array shapes."""
    buf = workload["buf"]
    assert int(buf.born_dec()["nborn"]) == 0, (
        "these workloads declare no Born-class decision (born5 is the born fixture)")
    gw_cols = buf.sigmas().shape[1]
    n = workload["n"]
    kappa_seen = 0
    for i in range(_SHOTS):
        pk = bytes(bytearray(buf.plan_key(i)))
        plan = workload["plans"][pk]
        r, kappa = int(plan["r"]), int(plan["kappa"])
        kappa_seen = max(kappa_seen, kappa)
        assert int(plan["n"]) == n
        assert int(plan["gw"]) == gw_cols
        assert len(bytearray(buf.coins(i))) == r + kappa
        assert np.asarray(plan["det_signs"]).shape == (gw_cols,)
        assert np.asarray(plan["coin_masks"]).shape == (r, gw_cols)
        assert np.asarray(plan["kernel_masks"]).shape == (kappa, gw_cols)
        assert np.asarray(plan["kernel_xz"]).shape == (kappa, 2 * n)
        for name in ("kernel_base", "kernel_foldable", "kernel_phase"):
            assert np.asarray(plan[name]).shape == (kappa,)
        assert not bool(plan["fallback"])
    # Honest coverage: report (never hide) that kappa>0 does not occur in-suite.
    print("[plan_structure] %s: max kappa over %d shots = %d"
          % (workload["name"], _SHOTS, kappa_seen))


# ---------------------------------------------------------------------------
# THE acceptance oracle: classifier vs materialized states
# ---------------------------------------------------------------------------

def test_classifier_signature_checks_exact(workload):
    """Every materialized port-signature check, every shot: classification must be
    DEFINITE with the exact materialized sign; the T2 span accounting must be
    reproduced (covered <-> in span(bare certified), gaps <-> out of span, with the
    pinned per-workload gap totals: qrm_magic 34, controls 0)."""
    buf, wires, W, n = (workload[k] for k in ("buf", "wires", "W", "n"))
    gens, gsolver, bare_exp = (workload[k] for k in ("gens", "gsolver", "bare_exp"))
    law_keys = workload["law_keys"]
    cls_memo = {}
    covered = gap = checked = 0
    errors = []
    for i in range(_SHOTS):
        pk = bytes(bytearray(buf.plan_key(i)))
        plan = workload["plans"][pk]
        r = int(plan["r"])
        coins = list(bytearray(buf.coins(i)))
        mbits = np.array(coins[r:], dtype=np.uint8)
        st = buf.materialize(i)
        keys = {bytes(k): int(v) for k, v in
                st.port_signature_struct(wires)["keys"].items()}
        for kk, sgn in keys.items():
            ck = (pk, kk)
            if ck not in cls_memo:
                cls_memo[ck] = _classify(np.frombuffer(kk, dtype=np.uint8), plan,
                                         gens, gsolver, bare_exp, wires, W, n)
            cls = cls_memo[ck]
            checked += 1
            is_gap = kk not in law_keys
            if is_gap:
                gap += 1
                if cls["in_span_bare"]:
                    errors.append((i, kk.hex(), "T2 gap check classified IN bare span"))
            else:
                covered += 1
                if not cls["in_span_bare"]:
                    errors.append((i, kk.hex(), "covered check classified OUT of bare span"))
            if not cls["definite"]:
                errors.append((i, kk.hex(), "materialized stabilizer classified indefinite"))
                continue
            pred = _expectation_of(cls, mbits)
            got = -1.0 if sgn else 1.0
            if pred != got:                         # exact: both are +-1.0
                errors.append((i, kk.hex(), "sign %r != materialized %r" % (pred, got)))
    assert not errors, ("classifier vs materialization oracle: %d error(s); first 10: %r"
                        % (len(errors), errors[:10]))
    assert checked > 0
    assert gap == workload["gaps_expected"], (
        "T2 gap accounting moved: expected %d gap occurrences, saw %d"
        % (workload["gaps_expected"], gap))
    _SUMMARY[workload["name"]] = dict(checked=checked, covered=covered, gap=gap,
                                      plans=len(workload["plans"]))
    print("[classifier] %s: %d (shot, check) pairs exact (%d covered, %d gap), "
          "%d distinct plans" % (workload["name"], checked, covered, gap,
                                 len(workload["plans"])))


def test_classifier_logical_triple_expectations(workload):
    """k_port==1 only: the carried logical triple (La, Lb, La.Lb) per shot must
    reproduce the materialized expectation from plan-constant data + record bits:
    definite -> exact +-1 equality; indefinite -> equality to <= 1e-12 (deterministic
    ulp-path difference only), with branch weights (1 +- e)/2. Requires at least one
    genuinely indefinite 0 < |e| < 1 case and one balanced e == 0 case."""
    if workload["k_port_expected"] == 0:
        pytest.skip("no carried logical (k_port == 0)")
    buf, wires, W, n = (workload[k] for k in ("buf", "wires", "W", "n"))
    gens, gsolver, bare_exp = (workload[k] for k in ("gens", "gsolver", "bare_exp"))
    orbit = workload["bare"].port_orbit_operators(wires)
    assert bool(orbit["ok"])
    La, Lb = [np.asarray(a).astype(np.uint8) for a in orbit["logicals"]]
    triple = [La, Lb, La ^ Lb]
    cls_memo = {}
    errors = []
    frac_seen = zero_seen = False
    for i in range(_SHOTS):
        pk = bytes(bytearray(buf.plan_key(i)))
        plan = workload["plans"][pk]
        r = int(plan["r"])
        coins = list(bytearray(buf.coins(i)))
        mbits = np.array(coins[r:], dtype=np.uint8)
        mat_exp = _make_exp(lambda: buf.materialize(i))
        for li, lkey in enumerate(triple):
            ck = (pk, li)
            if ck not in cls_memo:
                cls_memo[ck] = _classify(lkey, plan, gens, gsolver, bare_exp,
                                         wires, W, n)
            cls = cls_memo[ck]
            pred = _expectation_of(cls, mbits)
            got = mat_exp(_key_to_pauli(lkey, wires, W, n))
            if cls["definite"]:
                if pred != got:
                    errors.append((i, li, "definite", pred, got))
            else:
                if abs(pred - got) > 1e-12:
                    errors.append((i, li, "indefinite", pred, got))
                w_plus, w_minus = (1.0 + pred) / 2.0, (1.0 - pred) / 2.0
                if not (0.0 <= w_plus <= 1.0 and 0.0 <= w_minus <= 1.0):
                    errors.append((i, li, "weights out of range", w_plus, w_minus))
                if cls["branch"]["zero"]:
                    zero_seen = True
                elif 1e-6 < abs(pred) < 1.0 - 1e-6:
                    frac_seen = True
    assert not errors, ("logical-triple expectation oracle: %d error(s); first 10: %r"
                        % (len(errors), errors[:10]))
    assert frac_seen, "no genuinely indefinite 0 < |e| < 1 case seen (magic carrier?)"
    assert zero_seen, "no balanced (anticommuting, e == 0) twirl-split case seen"


# ---------------------------------------------------------------------------
# Sensitivity probe: a corrupted plan must be detected by the oracle
# ---------------------------------------------------------------------------

def test_corruption_probe_prefix_flip(workload):
    """Flip ONE prefix z-bit (on a port wire where some materialized check has x
    support) of one shot's plan: at least one definite sign prediction must now
    MISMATCH the materialized oracle — the classifier is prefix-sensitive."""
    buf, wires, W, n = (workload[k] for k in ("buf", "wires", "W", "n"))
    gens, gsolver, bare_exp = (workload[k] for k in ("gens", "gsolver", "bare_exp"))
    detected = False
    for i in range(_SHOTS):
        pk = bytes(bytearray(buf.plan_key(i)))
        plan = workload["plans"][pk]
        r = int(plan["r"])
        coins = list(bytearray(buf.coins(i)))
        mbits = np.array(coins[r:], dtype=np.uint8)
        st = buf.materialize(i)
        keys = {bytes(k): int(v) for k, v in
                st.port_signature_struct(wires)["keys"].items()}
        # corrupt: flip prefix z on the first port wire with x-support in some key
        for kk, sgn in keys.items():
            key = np.frombuffer(kk, dtype=np.uint8)
            xw = np.nonzero(key[:W])[0]
            if len(xw) == 0:
                continue
            wq = wires[int(xw[0])]
            bad = dict(plan)
            pxz = np.asarray(plan["prefix_xz"]).astype(np.uint8).copy()
            pxz[n + wq] ^= 1
            bad["prefix_xz"] = pxz
            cls = _classify(key, bad, gens, gsolver, bare_exp, wires, W, n)
            if not cls["definite"]:
                continue
            pred = _expectation_of(cls, mbits)
            got = -1.0 if sgn else 1.0
            if pred != got:
                detected = True
                break
        if detected:
            break
    assert detected, ("prefix z-bit corruption went undetected on every probed check "
                      "— the classifier is not prefix-sensitive")


# ---------------------------------------------------------------------------
# T5 step 0: born-op exposure + born-extended classifier vs materialization
# (vendored adaptq dec-enum class: adaptq_born5_dec_enum_producer.stim — five
# independent heavily-biased born decisions, joint rare branch p ~ 6.7e-5)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def born5():
    text = (_DATA / "adaptq_born5_dec_enum_producer.stim").read_text()
    inner = _compile(text)
    wires = list(inner.output_wires())
    buf = inner.sample_barrier(_SHOTS, _SEED)
    bare = _xtim.bare_state_of(text)
    n = bare.n
    cs = bare.certified_symplectic()
    ngens = int(cs["ngens"])
    xz = np.asarray(cs["xz"]).astype(np.uint8)
    phs = np.asarray(cs["phase"]).astype(np.uint8)
    gens = [(xz[b][:n], xz[b][n:], int(phs[b])) for b in range(ngens)]
    gsolver = _Gf2Solver([np.concatenate([g[0], g[1]]) for g in gens])
    bd = buf.born_dec()
    nborn = int(bd["nborn"])
    bxz = np.asarray(bd["xz"]).astype(np.uint8).reshape(nborn, 2 * n)
    bph = np.asarray(bd["phase"]).astype(np.uint8)
    borns = [(bxz[j][:n], bxz[j][n:], int(bph[j])) for j in range(nborn)]
    plans = {}
    for i in range(_SHOTS):
        pk = bytes(bytearray(buf.plan_key(i)))
        if pk not in plans:
            plans[pk] = buf.plan_structure(pk)
    return dict(text=text, wires=wires, W=len(wires), buf=buf, bare=bare, n=n,
                gens=gens, gsolver=gsolver, bd=bd, borns=borns, nborn=nborn,
                plans=plans, bare_exp=_make_exp(lambda: _xtim.bare_state_of(text)))


def test_born_dec_contract(born5):
    """born_dec() pure-data contract on the dec-enum workload: five born decisions
    (indices 0..4, inv 0, phase 0, single-Z supports off the port); each operator's
    composed bare expectation is the biased ⟨Z⟩ = 1/sqrt(2) of the H·T·H qubit; the
    coin record is [r fair ‖ kappa ‖ nborn raw] for every shot; and (this workload
    carries no readout-flip noise or input frame, inv = 0) the EMITTED decision
    record equals the raw born coins bit-for-bit."""
    buf, bd, nborn, n = born5["buf"], born5["bd"], born5["nborn"], born5["n"]
    assert nborn == 5
    assert int(bd["n"]) == n
    assert np.asarray(bd["dec_index"]).tolist() == [0, 1, 2, 3, 4]
    assert np.asarray(bd["inv"]).tolist() == [0] * 5
    assert np.asarray(bd["phase"]).tolist() == [0] * 5
    bare_exp = born5["bare_exp"]
    for j, B in enumerate(born5["borns"]):
        assert not B[0].any(), "born op %d is not Z-type" % j
        assert int(B[1].sum()) == 1, "born op %d is not single-qubit" % j
        assert abs(bare_exp(B) - 2.0 ** -0.5) < 1e-12, (
            "born op %d bare expectation is not the H·T·H ⟨Z⟩ = 1/sqrt(2)" % j)
    n_dec = 5
    decs = np.unpackbits(np.asarray(buf.decisions(), np.uint8), axis=1,
                         bitorder="little")[:, :n_dec]
    for i in range(_SHOTS):
        pk = bytes(bytearray(buf.plan_key(i)))
        plan = born5["plans"][pk]
        r, kappa = int(plan["r"]), int(plan["kappa"])
        coins = list(bytearray(buf.coins(i)))
        assert len(coins) == r + kappa + nborn, (
            "coin record is not [r ‖ kappa ‖ nborn] at shot %d" % i)
        assert decs[i].tolist() == coins[r + kappa:], (
            "emitted decision record != raw born coins at shot %d "
            "(inv=0, no rf/frame in this workload)" % i)


def test_born5_classifier_expectations_exact(born5):
    """Classifier vs materialization on the born-branching workload: for every shot
    and every probe operator, the born-extended classification must reproduce the
    materialized expectation — definite values exactly (±1 float equality),
    indefinite to ≤ 1e-12 (deterministic ulp-path difference only).

    Probes: Z_anc(j) per port wire (certified ZZ ⊗ born-op decomposition: sign =
    plan-const prefix bit ⊕ raw born coin j), the pair Z_anc(0)·Z_anc(1), and
    X_anc(0) (anticommutes with the certified Z·Z generator → balanced e = 0).
    Requires: every Z probe DEFINITE with born_sel selecting exactly born op j;
    at least one shot with a NON-empty plan whose prefix flips a predicted sign
    (the prefix ⊗ born composition is live, not vacuous)."""
    buf, wires, W, n = born5["buf"], born5["wires"], born5["W"], born5["n"]
    gens, gsolver, borns = born5["gens"], born5["gsolver"], born5["borns"]
    bare_exp = born5["bare_exp"]
    probes = []
    for j in range(W):
        key = np.zeros(2 * W, np.uint8)
        key[W + j] = 1
        probes.append(("Z%d" % j, key, j))
    pair = np.zeros(2 * W, np.uint8)
    pair[W + 0] = 1
    pair[W + 1] = 1
    probes.append(("Z0Z1", pair, None))
    x0 = np.zeros(2 * W, np.uint8)
    x0[0] = 1
    probes.append(("X0", x0, None))

    cls_memo = {}
    errors = []
    nonempty_plan_flip = False
    for i in range(_SHOTS):
        pk = bytes(bytearray(buf.plan_key(i)))
        plan = born5["plans"][pk]
        r = int(plan["r"])
        coins = list(bytearray(buf.coins(i)))
        mbits = np.array(coins[r:], dtype=np.uint8)
        mat_exp = _make_exp(lambda: buf.materialize(i))
        for name, key, bj in probes:
            ck = (pk, name)
            if ck not in cls_memo:
                cls_memo[ck] = _classify(key, plan, gens, gsolver, bare_exp,
                                         wires, W, n, borns=borns)
            cls = cls_memo[ck]
            pred = _expectation_of(cls, mbits)
            got = mat_exp(_key_to_pauli(key, wires, W, n))
            if name.startswith("Z") and bj is not None:
                if not cls["definite"]:
                    errors.append((i, name, "Z probe classified indefinite"))
                    continue
                kappa = int(plan["kappa"])
                sel = cls["measured_sel"]
                want = np.zeros(kappa + len(borns), np.uint8)
                want[kappa + bj] = 1
                if sel.tolist() != want.tolist():
                    errors.append((i, name, "born_sel %r != expected %r"
                                   % (sel.tolist(), want.tolist())))
                if cls["const_bit"] == 1 and pred == got:
                    nonempty_plan_flip = True   # prefix flipped AND still exact
            if cls["definite"]:
                if pred != got:
                    errors.append((i, name, "definite %r != materialized %r"
                                   % (pred, got)))
            else:
                if abs(pred - got) > 1e-12:
                    errors.append((i, name, "indefinite %r != materialized %r"
                                   % (pred, got)))
        if len(errors) > 10:
            break
    assert not errors, ("born5 classifier oracle: %d error(s); first 10: %r"
                        % (len(errors), errors[:10]))
    # X0 must classify balanced (e = 0) under the clean plan.
    clean = born5["plans"].get(b"")
    assert clean is not None, "no clean-plan shot in the born5 draw"
    cx0 = cls_memo.get((b"", "X0")) or _classify(x0, clean, gens, gsolver,
                                                 bare_exp, wires, W, n,
                                                 borns=borns)
    assert not cx0["definite"] and cx0["branch"]["zero"], (
        "X_anc(0) must be the balanced (anticommuting, e = 0) class")
    assert nonempty_plan_flip, (
        "no noise-flipped (const_bit = 1) Z probe was exercised — the prefix ⊗ born "
        "composition ran vacuously; raise the ancilla X_ERROR rate or the shot count")


def test_born5_rare_branch_covered_structurally(born5):
    """The dec-enum point, engine-side: the classification is per (check, plan) and
    RECORD-LINEAR — it never enumerates observed branches — so the joint all-ones
    born branch (p ≈ 0.1464^5 ≈ 6.7e-5), which a 256-shot draw does not contain
    (asserted), is served by the SAME memoized verdict: evaluating the Z_anc(j)
    classifications at the all-ones born vector predicts sign −1 for every check
    with ZERO additional draws.  (The retired discovery-draw pattern could not see
    this branch; the classifier cannot miss it.)"""
    buf, wires, W, n = born5["buf"], born5["wires"], born5["W"], born5["n"]
    gens, gsolver, borns = born5["gens"], born5["gsolver"], born5["borns"]
    decs = np.unpackbits(np.asarray(buf.decisions(), np.uint8), axis=1,
                         bitorder="little")[:, :5]
    assert not (decs.sum(axis=1) == 5).any(), (
        "the rare all-ones branch appeared in %d shots — the structural-coverage "
        "assertion below would be vacuous; change _SEED" % _SHOTS)
    clean = born5["plans"].get(b"")
    assert clean is not None
    kappa = int(clean["kappa"])
    rare = np.ones(kappa + len(borns), np.uint8)
    for j in range(W):
        key = np.zeros(2 * W, np.uint8)
        key[W + j] = 1
        cls = _classify(key, clean, gens, gsolver, born5["bare_exp"],
                        wires, W, n, borns=borns)
        assert cls["definite"]
        assert _expectation_of(cls, rare) == -1.0, (
            "rare-branch sign of Z_anc(%d) is not −1 under the clean plan" % j)


def test_zz_summary():
    """Every declared workload ran through the signature-check oracle."""
    missing = [w[0] for w in _WORKLOADS if w[0] not in _SUMMARY]
    assert not missing, "workloads missing from the classifier oracle: %r" % missing
    for name, s in _SUMMARY.items():
        print("[E3 summary] %s: checked=%d covered=%d gap=%d plans=%d"
              % (name, s["checked"], s["covered"], s["gap"], s["plans"]))
