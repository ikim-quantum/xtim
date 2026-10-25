"""Independent composer for the BarrierBuffer.frames() STATE CONTRACT (fix/frames-born-dec).

    |psi_i> ∝ M_dec(i) · Pi_W(i) · Pi_c (1 + (-1)^{c·sigma(i)} G_c)/2 · P · S^a · CZ · |bare>

with every outcome RAW (docs/branch_frames.md §7). S2b: the W and decision outcomes are the
record values frames()['y'] (record_outcomes); the channel signs c·sigma(i) equal y's channel
columns (the sigma-4 channel relation; checked by the selfcheck window's named throw). Built from xtim primitives only: the bare
reference (`bare_state_of`), engine Clifford gates for CZ / S^a / the X-Z prefix, and a FORCED
`measure_pauli` per projection factor (Clifford gates map the operator to one wire's Z, the
measurement is forced to the wanted outcome, the gates are undone). A zero-probability forced
outcome is a contract failure (ContractError), never skipped.

Plus the G-frames commutation assertion (fixplan r2-2), from exported data only:
  * `record_reads()` maps distinct records to distinct wires, or to equal bases on a shared wire;
  * every factor operator (record-channel products, the Born observable W, the born_dec() D_j)
    is supported on read wires in their read basis, and every pair has symplectic product 0.
"""
import numpy as np

from xtim import _xtim

_BASIS = {0: (1, 0), 1: (1, 1), 2: (0, 1)}       # read basis -> (x, z) bit on its wire


class ContractError(AssertionError):
    pass


def pmul(x1, z1, p1, x2, z2, p2):
    """(i^p1 X^x1 Z^z1)(i^p2 X^x2 Z^z2) with exact phase."""
    return x1 ^ x2, z1 ^ z2, (p1 + p2 + 2 * (int(np.dot(z1.astype(np.int64), x2.astype(np.int64))) & 1)) % 4


def project_onto(st, x, z, ph, s):
    """In place: project st onto the s = ±1 eigenspace of the Hermitian i^ph·X^x·Z^z, by a forced
    measure_pauli. Raises ContractError if that outcome has probability 0."""
    x = np.array(x, dtype=np.uint8); z = np.array(z, dtype=np.uint8); ph = int(ph)
    # already an eigenstate: the projector is the identity (s) or annihilates the state (-s)
    e = st.pauli_expectation([int(q) for q in np.nonzero(x)[0]], [int(q) for q in np.nonzero(z)[0]], ph)
    if abs(abs(e) - 1.0) < 1e-12:
        if abs(e - s) > 1e-12:
            raise ContractError(f"forced outcome {s:+d} has probability 0 (state is a {-s:+d} eigenstate)")
        return 1.0
    gates = []
    for q in (int(q) for q in np.nonzero(x)[0]):
        if z[q]:                                    # S: X·Z -> i·X  (so i^ph X Z -> i^{ph+1} X)
            st.apply_clifford(1, q, 0); gates.append((2, q, 0)); ph = (ph + 1) % 4; z[q] = 0
        st.apply_clifford(0, q, 0); gates.append((0, q, 0)); z[q] = 1   # H: X -> Z
    sup = [int(q) for q in np.nonzero(z)[0]]
    if not sup:
        raise ContractError("identity operator in a projection factor")
    piv = sup[0]
    for q in sup[1:]:                               # CX q->piv: Z_q·Z_piv -> Z_piv
        st.apply_clifford(6, q, piv); gates.append((6, q, piv))
    if ph not in (0, 2):
        raise ContractError(f"non-Hermitian projection factor (phase {ph})")
    want = s if ph == 0 else -s                     # eigenvalue of Z_piv
    pplus = (1.0 + st.pauli_expectation_z(piv)) / 2.0
    p = pplus if want == 1 else 1.0 - pplus
    if p <= 1e-12:
        raise ContractError(f"forced outcome {s:+d} has probability {p:.3g}")
    got = st.measure_pauli(2, piv, pplus / 2.0 if want == 1 else (1.0 + pplus) / 2.0)
    if got != want:
        raise ContractError("measure_pauli did not take the forced outcome")
    for k, a, b in reversed(gates):
        st.apply_clifford(k, a, b)
    return p


def channel_ops(text, sampler):
    """G_c per record channel: (mask words, x, z, phase), the product of the certified generators."""
    cs = _xtim.bare_state_of(text).certified_symplectic()
    n = int(cs["n"])
    G, PH = np.asarray(cs["xz"]).astype(np.uint8), np.asarray(cs["phase"])
    ops = []
    for m in np.asarray(sampler.record_channels()):
        x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8); ph = 0
        for b in range(G.shape[0]):
            if (int(m[b >> 6]) >> (b & 63)) & 1:
                x, z, ph = pmul(x, z, ph, G[b, :n], G[b, n:], int(PH[b]))
        ops.append((m, x, z, ph))
    return ops, n


def channel_sign(sig_bits, m):
    return -1 if (sum(int(sig_bits[b]) for b in range(len(sig_bits))
                      if (int(m[b >> 6]) >> (b & 63)) & 1) & 1) else 1


def unpack(packed):
    """Stim-b8 (LE) packed bit rows (dets() / obs() / decisions()) -> uint8[shots, 8*words]."""
    a = np.ascontiguousarray(np.asarray(packed, dtype=np.uint8))
    return np.unpackbits(a, axis=1, bitorder="little") if a.size else a.reshape(a.shape[0], 0)


def frame_state(text, fr, i, bare=None):
    """P·S^a·CZ·|bare> (no projection) from the frames() export (bare: a cached bare_state_of)."""
    from xtim.frames import unpack_prefix
    st = _xtim.bare_state_of(text) if bare is None else bare.copy()
    n = int(fr["n"]); k = int(fr["plan_id"][i])
    for (f, s) in np.asarray(fr["cz"][k]).reshape(-1, 2):
        st.apply_clifford(7, int(f), int(s))
    for q in np.nonzero(np.asarray(fr["a"])[k])[0]:
        st.apply_clifford(1, int(q), 0)
    px, pz = unpack_prefix(np.asarray(fr["prefix_words"])[i:i + 1], n)
    for q in np.nonzero(px[0])[0]:
        st.apply_clifford(3, int(q), 0)
    for q in np.nonzero(pz[0])[0]:
        st.apply_clifford(5, int(q), 0)
    return st


def born_dec_ops(sampler_or_buf):
    bd = sampler_or_buf.born_dec()
    nb = int(bd["nborn"]); n = int(bd["n"])
    xz = np.asarray(bd["xz"]).astype(np.uint8)
    return [(int(bd["dec_index"][j]), xz[j, :n], xz[j, n:], int(bd["phase"][j]), int(bd["inv"][j]))
            for j in range(nb)]


def compose(text, sampler, buf, fr, i, *, chan=None, bare=None, decisions=True, dec_bits="raw",
            channels=True):
    """The contract state of non-exact shot i. r1 composer = channels (+ W); then M_dec.
    Mutation knobs: decisions=False omits M_dec (f1); dec_bits="emitted" projects each D_j at the
    EMITTED decisions() bit instead of the raw coin (RAW-vs-EMITTED); channels=False skips the
    record-channel projection Pi_c (f5, the OLD unprojected reading)."""
    if chan is None:
        chan, _ = channel_ops(text, sampler)
    st = frame_state(text, fr, i, bare)
    sig = buf.sigma(i)
    for m, x, z, ph in (chan if channels else ()):
        project_onto(st, x, z, ph, channel_sign(sig, m))
    y = np.asarray(fr["y"])[i]                      # S2b: every record's RAW value
    if int(fr["born_obs_col"]) >= 0:
        n = int(fr["n"]); w = np.asarray(fr["born_obs_xz"]).astype(np.uint8)
        project_onto(st, w[:n], w[n:], int(fr["born_obs_phase"]),
                     -1 if int(y[int(fr["born_obs_col"])]) else 1)
    if decisions:
        ops = born_dec_ops(buf)
        if ops:
            col = dec_columns(sampler)
            dec = unpack(buf.decisions())[i]
            for j, (di, x, z, ph, _inv) in enumerate(ops):
                bit = int(y[col[di]]) if dec_bits == "raw" else int(dec[di])
                project_onto(st, x, z, ph, -1 if bit else 1)
    return st


def dec_columns(sampler):
    """S2b: DECISION index -> its column of frames()['y'] (record storage order)."""
    order = sampler.plan_structure(b"")["rec_order"]
    return {int(o) & ((1 << 24) - 1): c for c, o in enumerate(order) if (int(o) >> 24) == 2}


def assert_reads_one_basis(reads):
    """Distinct records -> distinct wires, or equal bases on a shared wire."""
    basis_of = {}
    for k, (q, b) in enumerate(zip(np.asarray(reads["wire"]), np.asarray(reads["basis"]))):
        q, b = int(q), int(b)
        if basis_of.setdefault(q, (b, k))[0] != b:
            raise ContractError(f"wire {q} read in two bases (records {basis_of[q][1]} and {k})")
    return {q: b for q, (b, _k) in basis_of.items()}


def record_pauli(reads, k, n):
    x = np.zeros(n, np.uint8); z = np.zeros(n, np.uint8)
    q = int(np.asarray(reads["wire"])[k]); x[q], z[q] = _BASIS[int(np.asarray(reads["basis"])[k])]
    return x, z


def symp(x1, z1, x2, z2):
    return (int(np.dot(x1.astype(np.int64), z2.astype(np.int64)))
            + int(np.dot(z1.astype(np.int64), x2.astype(np.int64)))) & 1


def assert_rows_commute(rows):
    """Every pair of (name, x, z) rows has symplectic product 0."""
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            if symp(rows[a][1], rows[a][2], rows[b][1], rows[b][2]):
                raise ContractError(f"factors {rows[a][0]} and {rows[b][0]} anticommute")


def factor_rows(text, sampler, fr=None):
    """Every contract factor operator as (name, x, z): record channels, W (if any), D_j."""
    chan, n = channel_ops(text, sampler)
    rows = [(f"G{c}", x, z) for c, (_m, x, z, _ph) in enumerate(chan)]
    if fr is not None and "born_obs_xz" in fr:
        w = np.asarray(fr["born_obs_xz"]).astype(np.uint8)
        if w.size and w.any():
            rows.append(("W", w[:n], w[n:]))
    for j, (di, x, z, _ph, _inv) in enumerate(born_dec_ops(sampler)):
        rows.append((f"D{j}(dec {di})", x, z))
    return rows, n


def assert_g_frames(text, sampler, fr=None):
    """The G-frames commutation assertion from exported data: one read basis per wire, every factor
    supported on read wires in their read basis, and all factors pairwise commuting."""
    reads = sampler.record_reads()
    basis = assert_reads_one_basis(reads)
    rows, n = factor_rows(text, sampler, fr)
    for name, x, z in rows:
        for q in np.flatnonzero(x | z):
            q = int(q)
            if q not in basis or (int(x[q]), int(z[q])) != _BASIS[basis[q]]:
                raise ContractError(f"factor {name} acts on wire {q} outside its read basis")
    assert_rows_commute(rows)
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# G-frames oracle corpus (fixplan r2 "G-frames r2" + r2-2): fixtures whose fault set is known
# exactly on both sides. A stochastic fixture expands into its forced single-fault family: every
# noise-site atom forced once (the law: exact_branches([(site, target, pauli)]); the twirl side:
# the same Pauli as a probability-1 channel at that site, every other noise line removed — a
# probability-1 channel, not a gate, so the noiseless REFERENCE that detectors and observables are
# relative to stays the original circuit's), plus the fault-free member;
# a readout flip M(p) is forced as {"record": j} (law) / M(1) (twirl). Probes: PAULI_EXPECTATION
# on the OUTPUT qubits only (never read, so they commute with every unrecorded read).
# ─────────────────────────────────────────────────────────────────────────────

_M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
       "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
       "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
_DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"

_F1 = ("RX 3\nT 3\nRX 4\nT_DAG 4\nX_ERROR(0.5) 2\nCS_DAG 4 3\nCCZ 4 3 2\nR 10\nM 10\nDETECTOR rec[-1]\n")
_F2 = ("RX 3\nT 3\nRX 4\nT_DAG 4\nX_ERROR(0.5) 2\nCS_DAG 4 3\nCCZ 4 3 2\nR 5 6\nH 5\nCX 5 6\nRX 7\nT 7\n"
       "R 8\nX_ERROR(0.5) 8\nCCZ 7 5 8\nR 9\nH 9\nCX 9 5\nCX 9 6\nH 9\nM 9\nDETECTOR rec[-1]\n")

GF_FIXTURES = {
    # one Born decision on an MX record entangled (CZ) with the output qubit
    "bd1": "RX 0 1\nT 0\nCZ 0 1\nMX 0\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 1\n",
    # the same with an inverted record: emitted = raw XOR 1 (RAW-vs-EMITTED goes red here)
    "bd1_inv": "RX 0 1\nT 0\nCZ 0 1\nMX !0\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 1\n",
    # readout flip M(p): the forced-flip member has emitted != raw (RAW-vs-EMITTED red)
    "bd1_mp": "RX 0 1\nT 0\nCZ 0 1\nMX(0.25) 0\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 1\n",
    # Pauli noise before the read, plus a noisy deterministic detector
    "bd1_noisy": ("RX 0 1\nT 0\nCZ 0 1\nDEPOLARIZE1(0.2) 0 1\nMX 0\nDECISION(0) rec[-1]\n"
                  "R 9\nX_ERROR(0.1) 9\nM 9\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 1\n"),
    # two Born decisions on two magic wires, both entangled with the output
    "bd2": ("RX 0 1 2\nT 0\nT_DAG 1\nCZ 0 2\nCZ 1 2\nMX 0 1\nDECISION(0) rec[-2]\n"
            "DECISION(1) rec[-1]\nOUTPUT_QUBITS out 2\n"),
    # CZ on record wires (an X fault before a CCZ): the channel projection is not the identity
    "cz_records": ("R 0\nRX 1 2 3\nH 3\nT 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nMX 1 2\nM 0\n"
                   "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 3\n"),
    # CZ on record wires + a Born decision on the output's magic partner
    "cz_records_bd": ("R 0\nRX 1 2 3 4\nT 4\nCZ 4 3\nX_ERROR(0.5) 0\nCCZ 0 1 2\nMX 1 2\nM 0\n"
                      "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nMX 4\nDECISION(0) rec[-1]\n"
                      "OUTPUT_QUBITS out 3\n"),
    "m3_obs": _M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET + "OUTPUT_QUBITS out 0 1 2\n",
    "m3_obs_dec": (_M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET
                   + "MX 0\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 1 2\n"),
    "m3_obs_2dec": (_M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + _DET
                    + "MX 0\nDECISION(0) rec[-1]\nMY 1\nDECISION(1) rec[-1]\nOUTPUT_QUBITS out 2\n"),
    "q3_dec": ("R 5\nX_ERROR(0.2) 5\nM 5\nDETECTOR rec[-1]\nMY 0\nMPP Z0\nMPP Z0\nH 3\nT 3\n"
               "X_ERROR(0.5) 2\nCCZ 0 3 2\nDECISION(0) rec[-3]\nDECISION(1) rec[-2]\n"
               "DECISION(2) rec[-1]\nOUTPUT_QUBITS out 2 3\n"),
    "m3_obs_x3": (_M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\nR 5\nX_ERROR(0.3) 5\nM 5\nDETECTOR rec[-1]\n"
                  "OUTPUT_QUBITS out 0 1 2\n"),
    # adv-A ANTI trio (<S>/unify/adv-A/antidec.py CIRC): ANTI/LOGICAL decisions with noise
    "q3dec_noisyT": ("RX 0\nR 1 2\nX_ERROR(0.2) 0\nT 0\nH 1\nCX 1 2\nCX 0 1\nDEPOLARIZE1(0.05) 1 2\n"
                     "MX 1 2 0\nDETECTOR rec[-3] rec[-2]\nDECISION(0) rec[-3]\nDECISION(1) rec[-1]\n"),
    "bell2_3dec": ("RX 0\nR 1 2 3 4\nX_ERROR(0.25) 0\nT 0\nH 1 3\nCX 1 2 3 4\nCX 0 1\nCZ 0 3\n"
                   "DEPOLARIZE1(0.06) 0 1 2 3 4\nMX 1 2 3 4 0\nDETECTOR rec[-5] rec[-4]\nDECISION(0) rec[-5]\n"
                   "DECISION(1) rec[-3]\nDECISION(2) rec[-1]\n"),
    "teleport_dec": ("RX 0\nR 1 2 3\nX_ERROR(0.2) 0\nT 0\nH 1\nCX 1 2\nCX 0 1\nH 0\nDEPOLARIZE1(0.05) 0 1 2 3\n"
                     "M 0 1\nMX 2\nM 3\nDETECTOR rec[-1]\nDECISION(0) rec[-4]\nDECISION(1) rec[-3]\n"
                     "DECISION(2) rec[-2]\n"),
    # decision ⟂ unread-kappa hazard family (<S>/ndhaz/{F1,F1o,F1x,F2,F2o,F2x}.stim; closed in 355093e)
    "F1": _F1 + "MX 4\nDECISION(0) rec[-1]\n",
    "F1o": _F1 + "OUTPUT_QUBITS out 4\n",
    "F2": _F2 + "MX 4\nDECISION(0) rec[-1]\nMX 7\nDECISION(1) rec[-1]\nMX 3\nDECISION(2) rec[-1]\n",
    "F2o": _F2 + "OUTPUT_QUBITS out 3 4 7\n",
    "ccz_obs": ("RX 0 1 2\nX_ERROR(0.3) 0\nCCZ 0 1 2\nRX 3\nT 3\nCX 3 1\nMX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
                + _DET + "OUTPUT_QUBITS out 0 1 2\n"),
}

# F1x / F2x: their observables are REFUSED by the sampler (channel_report()['refused_observables']:
# emitted 0, never projected), while the exact law conditions on them, so they are not golden-law
# fixtures; they run through the commutation assertion + composer == materialize instead.
NONGOLDEN_FIXTURES = {
    "F1x": _F1 + "MX 4\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "F2x": (_F2 + "MX 4\nOBSERVABLE_INCLUDE(0) rec[-1]\nMX 7\nOBSERVABLE_INCLUDE(1) rec[-1]\nMX 3\n"
            "OBSERVABLE_INCLUDE(2) rec[-1]\n"),
}

_ONE_Q = {"X_ERROR": "X", "Y_ERROR": "Y", "Z_ERROR": "Z", "DEPOLARIZE1": "XYZ"}


def _noise_line(line):
    """(name, args, targets) for a noise instruction line, else None."""
    import re
    m = re.match(r"^(X_ERROR|Y_ERROR|Z_ERROR|DEPOLARIZE1|PAULI_CHANNEL_1)\(([^)]*)\)\s+(.*)$", line.strip())
    if not m:
        return None
    return m.group(1), [float(a) for a in m.group(2).split(",")], m.group(3).split()


def _meas_flip(line):
    """(gate, p, targets) for a measurement line with a readout-flip argument, else None."""
    import re
    m = re.match(r"^(M|MX|MY|MR|MRX|MRY)\(([^)]*)\)\s+(.*)$", line.strip())
    return None if not m else (m.group(1), float(m.group(2)), m.group(3).split())


def single_fault_family(text):
    """[(tag, twirl_text, law_faults)]: the fault-free member, every single noise-site atom, and
    every single readout flip. Noise lines must be single-qubit channels (asserted)."""
    lines = text.splitlines()
    noise = [(li, _noise_line(l)) for li, l in enumerate(lines) if _noise_line(l)]
    flips = [(li, _meas_flip(l)) for li, l in enumerate(lines) if _meas_flip(l)]

    def render(gate_at=None, flip_at=None):
        out = []
        for li, l in enumerate(lines):
            if _noise_line(l):
                if gate_at is not None and gate_at[0] == li:
                    out.append(f"{gate_at[1]}_ERROR(1) {gate_at[2]}")
                continue
            mf = _meas_flip(l)
            if mf:
                g, _p, tg = mf
                for t in tg:
                    out.append(f"{g}(1) {t}" if flip_at == (li, t) else f"{g} {t}")
                continue
            out.append(l)
        return "\n".join(out) + "\n"

    fam = [("clean", render(), [])]
    for site, (li, (name, args, tg)) in enumerate(noise):
        if name == "PAULI_CHANNEL_1":
            paulis = "".join(p for p, a in zip("XYZ", args) if a > 0)
        else:
            paulis = _ONE_Q[name]
        for t, q in enumerate(tg):
            for P in paulis:
                fam.append((f"site{site}.{t}.{P}", render(gate_at=(li, P, q)), [(site, t, P)]))
    # absolute record index of each flip-carrying measurement target
    rec = 0; rec_of = {}
    for li, l in enumerate(lines):
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        head = s.split()[0].split("(")[0]
        if head in ("M", "MX", "MY", "MR", "MRX", "MRY", "MPP"):
            for t in s.split()[1:]:
                if head == "MPP" and "*" in t:
                    pass
                rec_of[(li, t)] = rec; rec += 1
    for li, (g, p, tg) in flips:
        for t in tg:
            fam.append((f"flip{rec_of[(li, t)]}", render(flip_at=(li, t)), [{"record": rec_of[(li, t)]}]))
    return fam


def output_qubits(text):
    for l in text.splitlines():
        if l.strip().startswith("OUTPUT_QUBITS"):
            return [int(t) for t in l.split()[2:]]
    return []


_NOT_TOUCHING = {"DETECTOR", "OBSERVABLE_INCLUDE", "DECISION", "PAULI_EXPECTATION", "TICK",
                 "OUTPUT_QUBITS", "INPUT_QUBITS", "QUBIT_COORDS", "SHIFT_COORDS"}
_MEAS = {"M": "Z", "MX": "X", "MY": "Y", "MZ": "Z", "MR": None, "MRX": None, "MRY": None, "MRZ": None,
         "MPP": None}


def probe_records(text):
    """[(record index k, qubit q, basis letter, '!' inverted)] for every single-qubit measurement record that is
    the LAST operation touching its qubit (no later gate, noise, reset or measurement): the
    qubit's end-of-circuit read-basis Pauli is then the record wire's read-basis Pauli in the
    deferred picture (m1: the probes that let the golden law see Pi_c)."""
    rows = []
    for l in text.splitlines():
        t = l.split("#")[0].strip()
        if not t:
            continue
        toks = t.split()
        head = toks[0].split("(")[0]
        rows.append((head, toks[1:]))
    last = {}
    for li, (head, tg) in enumerate(rows):
        if head in _NOT_TOUCHING:
            continue
        for tok in tg:
            if tok.startswith(("rec", "sweep")):
                continue
            for piece in tok.lstrip("!").split("*"):
                piece = piece.lstrip("!").lstrip("XYZ")
                if piece.isdigit():
                    last[int(piece)] = li
    out = []; rec = 0
    for li, (head, tg) in enumerate(rows):
        if head not in _MEAS:
            continue
        for tok in tg:
            if head != "MPP":
                q = int(tok.lstrip("!"))
                if _MEAS[head] and last.get(q) == li and [x.lstrip("!") for x in tg].count(str(q)) == 1:
                    out.append((rec, q, _MEAS[head], tok.startswith("!")))
            rec += 1
    return out


def probes_for(text):
    """PAULI_EXPECTATION probes: every single-qubit X/Y/Z on the OUTPUT qubits and, for >= 2
    outputs, every two-qubit XX/ZZ/XZ/ZX product (label 'X1', ...); plus every probe-able record
    wire's read-basis Pauli (label 'r<k>:<P><q>'; probe_records). Returns [(label, {qubit: P})]."""
    outs = output_qubits(text)
    pr = [(f"{P}{q}", {q: P}) for q in outs for P in "XYZ"]
    pr += [(f"r{k}{'!' if bang else ''}:{P}{q}", {q: P}) for k, q, P, bang in probe_records(text)]
    for a in range(len(outs)):
        for b in range(a + 1, len(outs)):
            for P, Q in ("XX", "ZZ", "XZ", "ZX"):
                pr.append((f"{P}{outs[a]}{Q}{outs[b]}", {outs[a]: P, outs[b]: Q}))
    return pr


def law_text(text, probes):
    body = "".join(f"PAULI_EXPECTATION({k}) " + "*".join(f"{P}{q}" for q, P in sorted(d.items())) + "\n"
                   for k, (_lab, d) in enumerate(probes))
    return text + body


def law_table(text, faults, probes):
    """Expanded exact law, keyed 'dets|obs|decs' -> [probability, [conditional <P> per probe]]."""
    import xtim
    sim = xtim.Circuit(law_text(text, probes)).compile_exact_branches()
    ft = [{"record": f["record"]} if isinstance(f, dict) else tuple(f) for f in faults]
    law = sim.exact_branches(ft)
    out = {}
    for r in range(len(law.probabilities)):
        k = ("".join(map(str, law.detectors[r].astype(int))) + "|"
             + "".join(map(str, law.observables[r].astype(int))) + "|"
             + "".join(map(str, law.decisions[r].astype(int))))
        if k in out:
            raise ContractError(f"duplicate law key {k}")
        out[k] = [float(law.probabilities[r]), [float(e) for e in np.asarray(law.expectations[r])]]
    return out


def probe_wires(label, d, sampler, outs):
    """The deferred-wire form of a law probe: output qubits via output_wires(); a record probe
    'r<k>:<P><q>' (law: the measured Pauli P on qubit q after its last operation, = (-1)^raw of
    record k) becomes record k's DEFERRED read-basis Pauli on its wire (record_reads(); deferral
    may rewrite the basis, e.g. 'H 0; M 0' reads X on the wire) — the same raw-outcome operator.
    The deferred read can also carry a basis-rotation SIGN: record_reads()['invert'] is the '!' bit
    XOR the rotation's sign (e.g. MY deferred as an X read with invert 1), so the wire operator is
    (-1)^{invert XOR '!'} B_k (key '_neg')."""
    if label.startswith("r"):
        head = label[1:label.index(":")]
        bang = head.endswith("!")
        k = int(head.rstrip("!"))
        rr = sampler.record_reads()
        out = {int(np.asarray(rr["wire"])[k]): "XYZ"[int(np.asarray(rr["basis"])[k])]}
        if int(np.asarray(rr["invert"])[k]) ^ int(bang):
            out["_neg"] = True
        return out
    wires = dict(zip(outs, sampler.output_wires()))
    return {int(wires[q]): P for q, P in d.items()}


def probe_value(st, wires, d):
    """<P> on a composed state, P = tensor of single-qubit X/Y/Z on deferred wires (wires=None:
    the keys of d already are deferred wires)."""
    xs, zs, ph = [], [], 0
    if d.get("_neg"):
        ph = 2
    for q, P in d.items():
        if q == "_neg":
            continue
        w = int(q if wires is None else wires[q])
        if P in "XY":
            xs.append(w)
        if P in "YZ":
            zs.append(w)
        if P == "Y":
            ph += 1                               # Y = i X Z
    return st.pauli_expectation(xs, zs, ph % 4).real


def shot_key(dets, obs, decs, i):
    return ("".join(map(str, dets[i])) + "|" + "".join(map(str, obs[i])) + "|"
            + "".join(map(str, decs[i])))


def random_circuit(rng):
    """A seeded random diagonal-class circuit (n <= 10 wires before deferral): magic on RX qubits,
    diagonal entanglers, Pauli noise, Born decisions (some read with ! or M(p)), sometimes a
    Born observable, a noisy deterministic detector, and sometimes the CZ-on-record-wires gadget
    (an X fault on the |0> control of a CCZ whose targets' MX records feed detectors). At least
    one data qubit stays unmeasured (OUTPUT_QUBITS)."""
    nd = int(rng.integers(2, 6))
    L = ["RX " + " ".join(map(str, range(nd)))]
    for _ in range(int(rng.integers(2, 7))):
        k = int(rng.integers(0, 7))
        a, b = (int(x) for x in rng.choice(nd, 2, replace=False))
        if k <= 1:
            L.append(f"{['T', 'T_DAG'][k]} {a}")
        elif k == 2:
            L.append(f"CZ {a} {b}")
        elif k == 3:
            L.append(f"CS {a} {b}")
        elif k == 4 and nd >= 3:
            c = int(rng.choice([q for q in range(nd) if q not in (a, b)]))
            L.append(f"CCZ {a} {b} {c}")
        elif k == 5:
            L.append(f"{['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1'][int(rng.integers(0, 3))]}"
                     f"({float(rng.choice([0.1, 0.3, 1.0]))}) {a}")
        else:
            L.append(f"S {a}")
    nmeas = int(rng.integers(1, nd))                     # >= 1 data qubit left unmeasured
    meas = [int(q) for q in rng.choice(nd, nmeas, replace=False)]
    outs = [q for q in range(nd) if q not in meas]
    nrec = 0; ndec = 0; has_obs = False
    for q in meas:
        style = int(rng.integers(0, 4))
        L.append({0: f"MX {q}", 1: f"MX !{q}", 2: f"MX({float(rng.choice([0.1, 0.5]))}) {q}",
                  3: f"MY {q}"}[style])
        nrec += 1
        if not has_obs and rng.random() < 0.25:
            L.append("OBSERVABLE_INCLUDE(0) rec[-1]"); has_obs = True
        else:
            L.append(f"DECISION({ndec}) rec[-1]"); ndec += 1
    if rng.random() < 0.4:                               # CZ-on-record-wires gadget on fresh wires
        L += ["R 6", "RX 7 8", f"X_ERROR({float(rng.choice([0.5, 1.0]))}) 6", "CCZ 6 7 8",
              "MX 7 8", "M 6", "DETECTOR rec[-3]", "DETECTOR rec[-2]", "DETECTOR rec[-1]"]
    L += ["R 9", f"X_ERROR({float(rng.choice([0.0, 0.1]))}) 9", "M 9", "DETECTOR rec[-1]",
          "OUTPUT_QUBITS out " + " ".join(map(str, outs))]
    return "\n".join(L) + "\n"
