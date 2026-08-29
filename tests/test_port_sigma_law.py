"""Exact oracle test for FramedSuperposition.port_sigma_law (E2, exact-residual arc).

The σ-law contract, for the port-restricted decomposition of the BARE state:

    sign_j(shot) = ref_signs[j] XOR parity(comb[j] AND sigma_shot)     (exact, GF(2))

for every port stabilizer check j the law covers, on every sampled shot, where
sigma_shot is the shot's certified-generator syndrome row (BarrierBuffer.sigmas())
and the materialized collapsed state's port_signature is the ground truth.

Checks are matched BY SUPPORT between the law's sorted stab rows and the
materialized signature keys.  IMPORTANT: the materialized signature may contain
checks the bare-generator law does NOT cover (per-plan / collapsed-kernel cases
— E3/Task-3 scope).  This test therefore
  (a) asserts EXACT sign equality on every covered check — zero tolerance,
      zero statistical assertions anywhere;
  (b) MEASURES and REPORTS the coverage fraction per workload honestly (gaps
      are reported, not hidden; only ZERO coverage fails — that would mean the
      law is miswired).

Workloads (adaptq battery classes; the flattened producer texts under
tests/data/ were generated once from adaptq's builtin gates — see the header
comment inside each file):
  - cat_err-class  (tiny definite producer, k_port=0)
  - steane-class   (noisy Steane |0> prep + noisy logical H, k_port=0)
  - qrm magic producer (steane-class magic carrier, k_port=1 — the k_port>=1 case)
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

# (name, circuit-text getter, expected k_port)
_WORKLOADS = [
    ("cat_err", lambda: _CAT_ERR, 0),
    ("steane_h", lambda: (_DATA / "adaptq_steane_h_producer.stim").read_text(), 0),
    ("qrm_magic", lambda: (_DATA / "adaptq_qrm_magic_producer.stim").read_text(), 1),
]

# Per-workload coverage record, printed by the final summary test.
_COVERAGE = {}


def _compile(text):
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    return s._inner if hasattr(s, "_inner") else s


def _parity_rows(sig_rows, comb):
    """GF(2) parity(comb[j] & sigma_i) for all shots i, checks j — exact.

    sig_rows: (shots, gw) uint64; comb: (ns, gw) uint64 -> (shots, ns) uint8.
    """
    v = sig_rows[:, None, :] & comb[None, :, :]          # (shots, ns, gw)
    bits = np.unpackbits(v.view(np.uint8), axis=-1)      # (shots, ns, gw*64)
    return (bits.sum(axis=-1) & 1).astype(np.uint8)


def _gf2_rank(M):
    """GF(2) row rank of integer/uint8 matrix M (rows x cols)."""
    M = (np.asarray(M, dtype=np.uint8) & 1).copy()
    rows, cols = M.shape
    pivot_row = 0
    for col in range(cols):
        found = next((r for r in range(pivot_row, rows) if M[r, col]), -1)
        if found == -1:
            continue
        M[[pivot_row, found]] = M[[found, pivot_row]]
        for r in range(rows):
            if r != pivot_row and M[r, col]:
                M[r] ^= M[pivot_row]
        pivot_row += 1
    return pivot_row


def _in_gf2_span(basis, v):
    """True iff row vector v is in the GF(2) row span of basis."""
    r_basis = _gf2_rank(basis)
    r_aug = _gf2_rank(np.vstack([basis, v]))
    return r_aug == r_basis


@pytest.fixture(scope="module", params=_WORKLOADS, ids=[w[0] for w in _WORKLOADS])
def workload(request):
    name, get_text, k_port_expected = request.param
    text = get_text()
    inner = _compile(text)
    wires = list(inner.output_wires())
    buf = inner.sample_barrier(_SHOTS, _SEED)
    bare = _xtim.bare_state_of(text)
    law = bare.port_sigma_law(wires)
    return dict(name=name, text=text, wires=wires, buf=buf, bare=bare, law=law,
                k_port_expected=k_port_expected)


# ---------------------------------------------------------------------------
# Structural exactness on the bare state (order, ref signs, logicals, packing)
# ---------------------------------------------------------------------------

def test_law_shape_and_kport(workload):
    law = workload["law"]
    W = len(workload["wires"])
    ns, gw, ngens = int(law["ns"]), int(law["gw"]), int(law["ngens"])
    assert law["W"] == W
    assert int(law["k_port"]) == workload["k_port_expected"]
    assert gw == (ngens + 63) // 64
    stab = np.asarray(law["stab"])
    comb = np.asarray(law["comb"])
    ref = np.asarray(law["ref_signs"])
    assert stab.shape == (ns, 2 * W) and stab.dtype == np.uint8
    assert comb.shape == (ns, gw) and comb.dtype == np.uint64
    assert ref.shape == (ns,) and ref.dtype == np.uint8


def test_comb_tail_bits_zero(workload):
    """Combination words must use ONLY bits 0..ngens-1 (the σ word layout), so
    (comb & sigmas()) needs no masking."""
    law = workload["law"]
    ngens, gw = int(law["ngens"]), int(law["gw"])
    comb = np.asarray(law["comb"])
    tail = ngens % 64
    if gw == 0 or tail == 0:
        return
    mask = np.uint64(~((np.uint64(1) << np.uint64(tail)) - np.uint64(1)))
    assert np.all((comb[:, gw - 1] & mask) == np.uint64(0)), (
        "comb rows set bits beyond ngens — σ word-layout contract broken")


def test_order_and_ref_signs_match_bare_port_signature(workload):
    """The law's sorted stab rows must be IDENTICAL (order and content) to the
    bare state's port_signature keys, and ref_signs must equal its sign values
    (σ_bare = 0, so the law's prediction on the bare state IS ref_signs)."""
    law = workload["law"]
    bare = workload["bare"]
    wires = workload["wires"]
    ps = bare.port_signature_struct(wires)
    assert int(ps["ns"]) == int(law["ns"])
    stab = np.asarray(law["stab"])
    ref = np.asarray(law["ref_signs"])
    sig_keys = [bytes(k) for k in ps["keys"].keys()]      # emission (sorted) order
    sig_signs = [int(v) for v in ps["keys"].values()]
    law_keys = [bytes(bytearray(stab[j])) for j in range(int(law["ns"]))]
    assert law_keys == sig_keys, "law stab order differs from port_signature order"
    assert list(ref) == sig_signs, "ref_signs differ from bare port_signature signs"


def test_logicals_match_port_orbit_operators(workload):
    """The carried logical pair rows (M·P law) must be byte-identical to
    port_orbit_operators' La/Lb; k_port==0 must emit none."""
    law = workload["law"]
    orbit = workload["bare"].port_orbit_operators(workload["wires"])
    assert bool(law["ok"]) == bool(orbit["ok"])
    logs = [np.asarray(a) for a in law["logicals"]]
    orbit_logs = [np.asarray(a) for a in orbit.get("logicals", [])]
    assert len(logs) == len(orbit_logs)
    for a, b in zip(logs, orbit_logs):
        assert np.array_equal(a, b)
    if workload["k_port_expected"] == 0:
        assert logs == []
    else:
        assert len(logs) == 2, "k_port>=1 case must carry the (La, Lb) pair"


# ---------------------------------------------------------------------------
# THE oracle: per-shot exact sign law vs materialized port_signature
# ---------------------------------------------------------------------------

def test_sigma_law_exact_vs_materialized(workload):
    """For EVERY shot and EVERY law-covered check (matched by support):
        ref_j XOR parity(c_j AND sigma_i) == materialized port_signature sign.
    Zero tolerance.  Coverage is measured and reported; only zero coverage
    fails (a miswired law)."""
    law = workload["law"]
    buf = workload["buf"]
    wires = workload["wires"]
    ns, gw = int(law["ns"]), int(law["gw"])
    stab = np.asarray(law["stab"])
    comb = np.asarray(law["comb"])
    ref = np.asarray(law["ref_signs"])
    law_keys = [bytes(bytearray(stab[j])) for j in range(ns)]
    law_key_set = set(law_keys)

    S = buf.sigmas()
    assert S.shape[0] == _SHOTS
    pred = (ref[None, :] ^ _parity_rows(S, comb)).astype(np.uint8)  # (shots, ns)

    matched = 0          # (shot, check) pairs matched by support
    total = _SHOTS * ns  # (shot, check) pairs the law emits
    mat_total = 0        # materialized keys seen (all shots)
    mat_covered = 0      # ... of which the law covers by support
    mismatches = []
    for i in range(_SHOTS):
        st = buf.materialize(i)
        keys = {bytes(k): int(v) for k, v in
                st.port_signature_struct(wires)["keys"].items()}
        mat_total += len(keys)
        mat_covered += sum(1 for k in keys if k in law_key_set)
        for j in range(ns):
            got = keys.get(law_keys[j])
            if got is None:
                continue                       # coverage gap — reported, not failed
            matched += 1
            if got != int(pred[i, j]):
                mismatches.append((i, j, int(pred[i, j]), got))

    assert not mismatches, (
        "EXACT sign law violated on %d covered (shot, check) pairs; first 10: %r"
        % (len(mismatches), mismatches[:10]))
    assert matched > 0, (
        "ZERO coverage: no law check matched any materialized signature key "
        "by support — the law is miswired")

    _COVERAGE[workload["name"]] = dict(
        matched=matched, total=total,
        law_coverage=matched / total if total else float("nan"),
        mat_covered=mat_covered, mat_total=mat_total,
        materialized_coverage=mat_covered / mat_total if mat_total else float("nan"),
        shots=_SHOTS, ns=ns, k_port=int(law["k_port"]))
    print("\n[port_sigma_law coverage] %s: law-checks matched %d/%d (%.4f); "
          "materialized keys covered %d/%d (%.4f); ns=%d k_port=%d shots=%d"
          % (workload["name"], matched, total, matched / total,
             mat_covered, mat_total, mat_covered / mat_total,
             ns, int(law["k_port"]), _SHOTS))


# ---------------------------------------------------------------------------
# In-suite corruption probe (T2 review Minor 1)
# Verify the oracle is sensitive: a single bit-flip in either comb or ref_signs
# must be detected as a sign mismatch on at least one covered (shot, check) pair.
# The probe is deterministic: we pick a comb bit on a column that FIRES (σ varies
# there across the batch), so the flip is guaranteed to change at least one pred.
# ---------------------------------------------------------------------------

def test_corruption_probe_comb_and_ref(workload):
    """Single comb-bit and single ref-sign corruptions must both be detected."""
    law = workload["law"]
    buf = workload["buf"]
    wires = workload["wires"]
    ns, gw, ngens = int(law["ns"]), int(law["gw"]), int(law["ngens"])
    stab = np.asarray(law["stab"])
    comb = np.asarray(law["comb"])
    ref = np.asarray(law["ref_signs"])
    law_keys = [bytes(bytearray(stab[j])) for j in range(ns)]

    S = buf.sigmas()                                       # (shots, gw) uint64

    # Unpack sigma and comb to bit arrays (unpackbits order: MSB-first per byte,
    # little-endian byte order within each uint64 word).
    S_bits = np.unpackbits(S.view(np.uint8), axis=-1)[:, :ngens]   # (shots, ngens)
    comb_full = np.unpackbits(comb.view(np.uint8), axis=-1)         # (ns, gw*64)
    comb_bits = comb_full[:, :ngens]                                 # (ns, ngens)

    # Find a (j, ucol) pair: ucol fires in sigma AND comb[j, ucol] == 1.
    firing_cols = np.where(S_bits.any(axis=0))[0]
    assert len(firing_cols) > 0, (
        "No sigma column fires across %d shots — workload has no stochastic noise "
        "(cannot run comb-bit corruption probe for %s)" % (_SHOTS, workload["name"]))
    probe_j, probe_ucol = None, None
    for ucol in firing_cols:
        for j in range(ns):
            if comb_bits[j, ucol]:
                probe_j, probe_ucol = int(j), int(ucol)
                break
        if probe_j is not None:
            break
    assert probe_j is not None, (
        "No comb bit is set on any firing sigma column for %s — "
        "law has no σ-sensitivity (comb probe cannot be constructed)"
        % workload["name"])

    # --- Probe 1: flip ONE comb bit (unpackbits col probe_ucol, row probe_j) ---
    comb_bits_c = comb_full.copy()
    comb_bits_c[probe_j, probe_ucol] ^= 1
    comb_corrupt = np.packbits(comb_bits_c, axis=-1).view(np.uint64)
    pred_c = (ref[None, :] ^ _parity_rows(S, comb_corrupt)).astype(np.uint8)

    mm_comb = []
    for i in range(_SHOTS):
        st = buf.materialize(i)
        keys = {bytes(k): int(v) for k, v in
                st.port_signature_struct(wires)["keys"].items()}
        got = keys.get(law_keys[probe_j])
        if got is not None and got != int(pred_c[i, probe_j]):
            mm_comb.append(i)
    assert mm_comb, (
        "Comb-bit corruption at (stab[%d], sigma_col=%d) went undetected across "
        "%d shots for workload %s — oracle is insensitive to this mutation"
        % (probe_j, probe_ucol, _SHOTS, workload["name"]))

    # --- Probe 2: flip ONE ref sign (always row 0; all shots are covered) ---
    ref_c = ref.copy()
    ref_c[0] ^= np.uint8(1)
    pred_r = (ref_c[None, :] ^ _parity_rows(S, comb)).astype(np.uint8)

    mm_ref = []
    for i in range(_SHOTS):
        st = buf.materialize(i)
        keys = {bytes(k): int(v) for k, v in
                st.port_signature_struct(wires)["keys"].items()}
        got = keys.get(law_keys[0])
        if got is not None and got != int(pred_r[i, 0]):
            mm_ref.append(i)
    assert mm_ref, (
        "ref_signs[0] flip went undetected across %d shots for workload %s "
        "— oracle is insensitive to ref-sign mutations"
        % (_SHOTS, workload["name"]))


# ---------------------------------------------------------------------------
# Self-certifying gap accounting (T2 review Minor 2)
# Every materialized-signature support NOT matched by the law must lie OUTSIDE
# the GF(2) row span of the law's stabilizer supports.  If a gap key were in
# span, the law should have covered it — its absence would be a silent law bug.
# ---------------------------------------------------------------------------

def test_gap_keys_are_out_of_gf2_span(workload):
    """Coverage gaps must be genuinely out-of-span, not silent law bugs."""
    law = workload["law"]
    buf = workload["buf"]
    wires = workload["wires"]
    ns = int(law["ns"])
    stab = np.asarray(law["stab"])                        # (ns, 2W) uint8 binary
    law_keys_set = {bytes(bytearray(stab[j])) for j in range(ns)}

    in_span_gaps = []
    for i in range(_SHOTS):
        st = buf.materialize(i)
        for raw_k, _ in st.port_signature_struct(wires)["keys"].items():
            k = bytes(raw_k)
            if k not in law_keys_set:
                v = np.frombuffer(k, dtype=np.uint8).astype(np.int8)
                if _in_gf2_span(stab.astype(np.int8), v):
                    in_span_gaps.append((i, list(v)))
    assert not in_span_gaps, (
        "SILENT LAW BUG: %d gap key(s) ARE in span(law stab rows) for workload %s "
        "— the law should have covered them; first 5: %r"
        % (len(in_span_gaps), workload["name"], in_span_gaps[:5]))


# ---------------------------------------------------------------------------
# Honest coverage summary (runs last within this module)
# ---------------------------------------------------------------------------

def test_zz_coverage_summary():
    """Report per-workload coverage fractions (no thresholds beyond non-zero,
    already asserted above).  Ensures every declared workload actually ran."""
    missing = [n for n, _, _ in _WORKLOADS if n not in _COVERAGE]
    assert not missing, "workloads missing from the oracle run: %r" % missing
    for name, cov in _COVERAGE.items():
        print("[port_sigma_law summary] %s: law %.4f (%d/%d), materialized %.4f "
              "(%d/%d), k_port=%d"
              % (name, cov["law_coverage"], cov["matched"], cov["total"],
                 cov["materialized_coverage"], cov["mat_covered"],
                 cov["mat_total"], cov["k_port"]))
