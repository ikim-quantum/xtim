"""Pauli rec-feedback AFTER a non-Clifford gate on the twirl / retained-state path.

GAP (through integ/3.1.12): the TwirlSampler family coherentizes EVERY feedback
(``normalize_for_twirl``: ``coherentize_all``).  A feedback driven by an X- or
Y-basis read was emitted AFTER its source measurement, so deferral copied the
control out through a basis rotation (``H``) that the coherent gate then left in
the bulk of the control wire.  When that wire carries a fault's
Pauli-times-diagonal residual — any fault BEFORE a T/CS/CCZ on it, e.g.
``DEPOLARIZE1`` on a magic wire that is then read in X and drives ``CZ rec[-k]``
— the residual meets that ``H`` and leaves the diagonal class (``H S H`` is not
diagonal).  ``sample()`` survived on the PPR route, but the state-retaining
``sample_barrier`` (``xtim.port`` groups / residual_law, adaptq's segment
harness) refused: ``twirl_records: state retention / Born decisions unsupported
for CH/PPR circuits``.  adaptq worked around it by dropping the Pauli from the
text and applying it as an external frame.

FIX: ``NormalizePolicy::hoist_feedback_on_class_exit`` — when the default form
leaves the diagonal class, the feedbacks of X/Y sources are emitted just BEFORE
their source measurement (they are diagonal in its basis, so they commute with
it), written with the basis rotation on the TARGET.  The control wire then ends
at its own measurement, exactly as in the feedback-free circuit, and the fault
residual reaches a terminal read instead of a Hadamard.  WHICH feedbacks may move is
derived from the deferred-measurement theorem (coherentize_feedback.cpp): conjugate
the feedback Pauli P back through its window W; it moves, as the b-controlled
W^dag P W, iff that stays a Pauli commuting with every measurement / reset /
expectation it passes and with the source's basis on the control (Pauli noise and
other feedbacks in the window are transparent).

ORACLES (no xtim code): ``_record_oracle.record_law`` (exact record law by
dense branch enumeration, feedback included) for every record statistic, and a
dense state vector projected onto each shot's own records for the retained
state, shot by shot.  The same circuits with the feedback replaced by the
deferred-measurement form (a coherent controlled-Pauli from the measured qubit,
before its measurement) are the second reference.
"""
import itertools
import random
import re

import numpy as np
import pytest

import xtim
import xtim._xtim as _x
from xtim.circuit import Circuit

from _record_oracle import _P1, _ONE, _apply1, _bits, _project, max_z, parities, record_law

_BASIS = {"M": "Z", "MX": "X", "MY": "Y"}


# ── dense oracle: the state conditioned on a shot's records ───────────────────

def _conditioned_state(text, outcomes):
    """Dense run of ``text`` with every measurement PROJECTED onto the recorded bit
    ``outcomes[k]`` and every ``rec``-driven Pauli applied from those bits.  Noise is
    deterministic here (``*_ERROR(1)`` fires, ``(0)`` never does).  Returns (psi, n,
    output wires)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = max(int(t.lstrip("!")) for ln in lines for t in ln.split()[1:]
            if t.lstrip("!").isdigit()) + 1
    zb = _bits(n)
    psi = np.zeros(1 << n, complex)
    psi[0] = 1.0
    rec, out = [], None
    for ln in lines:
        tok = ln.split()
        name = re.sub(r"\(.*\)", "", tok[0])
        arg = re.findall(r"\(([^)]*)\)", tok[0])
        tg = tok[1:]
        if name in ("DETECTOR", "DECISION", "OBSERVABLE_INCLUDE"):
            continue
        if name == "OUTPUT_QUBITS":
            out = [int(t) for t in tg if t.isdigit()]
        elif name in ("R", "RX"):
            for t in tg:                       # every wire is fresh in these circuits
                if name == "RX":
                    psi = _apply1(psi, n, _ONE["H"], int(t))
        elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            p = float(arg[0])
            assert p in (0.0, 1.0), "per-shot oracle needs deterministic faults"
            if p == 1.0:
                for t in tg:
                    psi = _apply1(psi, n, _P1[name[0]], int(t))
        elif name in _ONE:
            for t in tg:
                psi = _apply1(psi, n, _ONE[name], int(t))
        elif name in ("CX", "CY", "CZ") and tg[0].startswith("rec["):
            for c, t in zip(tg[::2], tg[1::2]):
                k = int(re.match(r"rec\[-(\d+)\]", c).group(1))
                if rec[-k]:
                    psi = _apply1(psi, n, _P1[name[1]], int(t))
        elif name in ("CX", "CY"):
            for c, t in zip(tg[::2], tg[1::2]):
                on = _project(psi, n, [(int(c), "Z")], -1)
                psi = psi - on + _apply1(on, n, _P1[name[1]], int(t))
        elif name in ("CZ", "CS", "CCZ"):
            ar = 3 if name == "CCZ" else 2
            ph = 1j if name == "CS" else -1.0
            for i in range(0, len(tg), ar):
                mask = np.ones(1 << n, bool)
                for q in tg[i:i + ar]:
                    mask &= zb[int(q)] == 1
                psi = psi.copy()
                psi[mask] *= ph
        elif name in _BASIS:
            for t in tg:
                bit = int(outcomes[len(rec)])           # the RECORDED bit
                inv = t.startswith("!")                 # `!q`: outcome = recorded ^ 1
                psi = _project(psi, n, [(int(t.lstrip("!")), _BASIS[name])],
                               -1 if bit ^ inv else +1)
                nrm = np.linalg.norm(psi)
                assert nrm > 1e-9, "the engine recorded a zero-probability outcome"
                psi = psi / nrm
                rec.append(bit)
        else:
            raise ValueError(f"oracle: unsupported {name!r}")
    return psi, n, out


def _dense_expect(psi, n, qs, word):
    v = psi
    for q, p in zip(qs, word):
        if p != "I":
            v = _apply1(v, n, _P1[p], q)
    return np.vdot(psi, v)


def _engine_reads(ow, word):
    xs = [w for w, p in zip(ow, word) if p in "XY"]
    zs = [w for w, p in zip(ow, word) if p in "ZY"]
    return (xs, zs, sum(p == "Y" for p in word) % 4)    # Y = i·X·Z


def _decision_bits(buf, shots, ndec):
    d = np.asarray(buf.decisions(), np.uint8).reshape(shots, -1)
    return np.unpackbits(d, axis=1, bitorder="little")[:, :ndec]


def _sampler(text):
    return _x.TwirlSampler(text, 1.0, "", 0, "")


def _check_retained(text, shots, seed):
    """Every measurement carries its own ``DECISION(k) rec[-1]`` (k in record order), so
    the decisions ARE the shot's records.  Per shot: every Pauli expectation of the
    retained output state == the dense state conditioned on those records."""
    S = _sampler(text)
    buf = S.sample_barrier(shots, seed)
    ow = list(S.output_wires())
    ndec = len(parities(text)[2])
    recs = _decision_bits(buf, shots, ndec)
    words = list(itertools.product("IXYZ", repeat=len(ow)))
    eng = np.asarray(buf.pauli_expectations_all([_engine_reads(ow, w) for w in words]))
    for i in range(shots):
        psi, n, out = _conditioned_state(text, recs[i])
        want = np.array([_dense_expect(psi, n, out, w) for w in words])
        bad = np.flatnonzero(np.abs(eng[i] - want) > 1e-9)
        assert bad.size == 0, (i, recs[i].tolist(),
                               [("".join(words[b]), eng[i][b], want[b]) for b in bad[:6]])
    return S, recs


def _hoisted(S):
    return int(S.record_report()["feedback_hoisted"])


# ── circuits ─────────────────────────────────────────────────────────────────

_ANCHOR = "R 9\nM 9\nDECISION(0) rec[-1]\nDETECTOR rec[-1]\n"   # one deterministic channel


def _teleport_like(fault, read="MX", fb="CZ", tgate="T", tail=""):
    """Magic wire 0 (|+>, fault, then ``tgate``) entangled with target 1, read in
    ``read`` and driving ``fb rec[-1] 1``: the adaptq segment shape in miniature."""
    gate = {"T": "T 0\n", "T_DAG": "T_DAG 0\n", "CS": "RX 2\nCS 0 2\n",
            "CCZ": "RX 2 3\nCCZ 0 2 3\n"}[tgate]
    ent = "RX 1\nCZ 0 1\n" if fb == "CZ" else "R 1\nCX 0 1\n" if fb == "CX" else \
        "R 1\nCX 0 1\nCZ 0 1\n"
    return (_ANCHOR + "RX 0\n" + fault + gate + ent + f"{read} 0\nDECISION(1) rec[-1]\n"
            + f"{fb} rec[-1] 1\n" + tail + "OUTPUT_QUBITS out 1\n")


# Measured on integ/3.1.12 (839e452): every case below REFUSED sample_barrier.  The
# Z_ERROR cases were refused only because the propagation table built an X atom for a
# pure-Z channel; since fix/z-channel-atoms (8bf7c8b) they are in class in the DEFAULT form
# and nothing is moved (hoisted == 0), so the X/Y-fault cases carry the hoist path.
FIXED = [(f, r, g) for f in ("X_ERROR(1) 0\n", "Y_ERROR(1) 0\n", "Z_ERROR(1) 0\n")
         for r in ("MX", "MY") for g in ("T", "T_DAG", "CS")]
# ... and these still do, now BY NAME: X/Y-type feedback driven by an X/Y read of a
# fault-dressed wire (the coherent G conjugates the S into a Z_c*P_t rotation).
NAMED = [(f, r, fb, g) for f in ("X_ERROR(1) 0\n", "DEPOLARIZE1(0.1) 0\n")
         for r in ("MX", "MY") for fb in ("CX", "CY") for g in ("T", "CS")]


@pytest.mark.parametrize("fault,read,gate", FIXED)
def test_retained_state_after_feedback_matches_dense_per_shot(fault, read, gate):
    text = _teleport_like(fault, read, "CZ", gate)
    S, recs = _check_retained(text, 48, 3)      # per-shot dense match, either form
    # X/Y fault: the default form is out of class and the fallback moves the feedback;
    # Z fault (Z commutes with T): the default form is in class, nothing is moved.
    assert _hoisted(S) == (0 if fault.startswith("Z_ERROR") else 1)
    assert 0 < recs[:, 1].sum() < 48            # the control record is a genuine coin


@pytest.mark.parametrize("read", ["MX", "MY"])
def test_inverted_source_record(read):
    """``MX !0``: the feedback fires on the +1 outcome (the hoisted form's unconditional
    Pauli on the target)."""
    text = _teleport_like("X_ERROR(1) 0\n", read, "CZ", "T").replace(f"{read} 0\n", f"{read} !0\n")
    S, _ = _check_retained(text, 48, 17)
    assert _hoisted(S) == 1


def test_fault_free_teleport_keeps_default_form_byte_for_byte():
    """No fault before the T: the default coherent form is in class, nothing is hoisted,
    and the retained states match the dense oracle as before."""
    for fb in ("CZ", "CX", "CY"):
        S, _ = _check_retained(_teleport_like("", "MX", fb), 32, 5)
        assert _hoisted(S) == 0


def test_feedback_onto_a_wire_that_later_goes_through_a_t():
    """``CZ rec`` onto the target, then a T on the target (the byproduct commutes with
    it but the coherentizer's cone test flags it as crossing), then the target is read
    in X through an ancilla-free MX."""
    text = _teleport_like("X_ERROR(1) 0\n", "MX", "CZ", "T", tail="T 1\n")
    S, _ = _check_retained(text, 48, 7)
    assert _hoisted(S) == 1


def test_two_feedbacks_from_two_magic_reads_onto_one_target():
    text = (_ANCHOR + "RX 0 2\nX_ERROR(1) 0\nY_ERROR(1) 2\nT 0\nT_DAG 2\nRX 1\nCZ 0 1\n"
            "CZ 2 1\nMX 0\nDECISION(1) rec[-1]\nMX 2\nDECISION(2) rec[-1]\n"
            "CZ rec[-2] 1\nCZ rec[-1] 1\nOUTPUT_QUBITS out 1\n")
    S, _ = _check_retained(text, 48, 11)
    assert _hoisted(S) == 2


def test_hoist_is_decided_per_feedback():
    """Two feedbacks from ONE MX record.  The second's window holds a T on its target
    where the moved Pauli is X: W^dag X W is not a Pauli, so (deferred-measurement
    theorem) it cannot move; the first still can. RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal ->
    exactness: the kept second feedback leaves the diagonal class, which the legacy retained engine refused by name
    (naming feedback #1 and the blocking T); the family law serves it exactly (term route)."""
    text = (_ANCHOR + "RX 0\nX_ERROR(1) 0\nT 0\nRX 1 2\nCZ 0 1\nCZ 0 2\nMX 0\n"
            "DECISION(1) rec[-1]\nCZ rec[-1] 1\nT 2\nCX rec[-1] 2\nOUTPUT_QUBITS out 1 2\n")
    _check_conditional(text, shots=6000, seed=3)


def test_hoist_conjugates_through_a_clifford_on_the_target():
    """``CX 5 1`` between the read and ``CZ rec[-1] 1``: W^dag Z_1 W = Z_1 Z_5, so the
    feedback moves to its source as the X-controlled (Z_1 Z_5) — not as Z_1 (that would
    be in class but WRONG).  Per-shot retained states match the dense oracle."""
    text = (_ANCHOR + "RX 0 5\nX_ERROR(1) 0\nT 0\nRX 1\nCZ 0 1\nMX 0\nDECISION(1) rec[-1]\n"
            "CX 5 1\nCZ rec[-1] 1\nOUTPUT_QUBITS out 1 5\n")
    S, recs = _check_retained(text, 48, 1)
    assert _hoisted(S) == 1
    assert 0 < recs[:, 1].sum() < 48


@pytest.mark.parametrize("fault,read,fb,gate", NAMED)
def test_xy_feedback_from_dressed_xy_read_is_served_exactly(fault, read, fb, gate):
    """RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness (plan §5.3): the X/Y-type feedback from a
    dressed X/Y read, moved to its source, met the residual's S as an (I + iX)-type rotation on the kept target —
    outside the LEGACY retained engine's Pauli x diagonal-Clifford representation, refused by name. The family law
    serves it (the term route, kind 2): the retained state of every record class equals the dense branch oracle."""
    _check_conditional(_teleport_like(fault, read, fb, gate), shots=6000, seed=3)


@pytest.mark.parametrize("read,fb", [("MX", "CX"), ("MY", "CY"), ("MX", "CY"), ("MY", "CX")])
def test_xy_feedback_onto_a_target_read_right_after_is_accepted(read, fb):
    """The same feedback with the target READ instead of retained: the hoisted form's
    target rotation folds into that read, so it stays in class — record law exact."""
    text = (_teleport_like("X_ERROR(1) 0\n", read, fb, "T").replace("OUTPUT_QUBITS out 1\n", "")
            + "MX 1\nDECISION(2) rec[-1]\n")
    S = _sampler(text)
    assert _hoisted(S) == 1
    N = 20000
    cols = _dec_columns(text)
    z, imp = max_z(_decision_bits(S.sample_barrier(N, 5), N, len(cols)), _law(text, cols))
    assert imp == 0 and z < 5.5, (z, imp)


# ── record statistics: twirl vs exact record law vs deferred-measurement rewrite ──

def _deferred_rewrite(text):
    """Replace each ``C<P> rec[-k] t`` by the deferred-measurement form: a coherent
    controlled-P from the measured qubit, in its measured basis, emitted right BEFORE
    that measurement (the generator keeps the control idle and the target untouched
    in between).  A user-level coherent circuit: no feedback anywhere."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    meas_line, out = [], list(lines)
    for i, ln in enumerate(lines):
        name = re.sub(r"\(.*\)", "", ln.split()[0])
        if name in _BASIS:
            meas_line += [(i, int(t), _BASIS[name]) for t in ln.split()[1:]]
    inserts = {}
    nrec = 0
    for i, ln in enumerate(lines):
        tok = ln.split()
        name = re.sub(r"\(.*\)", "", tok[0])
        if name in _BASIS:
            nrec += len(tok) - 1
            continue
        if name in ("CX", "CY", "CZ") and tok[1].startswith("rec["):
            k = int(re.match(r"rec\[-(\d+)\]", tok[1]).group(1))
            mi, c, b = meas_line[nrec - k]
            t = int(tok[2])
            rot = {"Z": ([], []), "X": (["H"], ["H"]), "Y": (["S_DAG", "H"], ["H", "S"])}[b]
            g = [f"{r} {c}" for r in rot[0]] + [f"{name} {c} {t}"] + [f"{r} {c}" for r in rot[1]]
            inserts.setdefault(mi, []).extend(g)
            out[i] = None
    res = []
    for i, ln in enumerate(out):
        res.extend(inserts.get(i, []))
        if ln is not None:
            res.append(ln)
    return "\n".join(res) + "\n"


def _random_circuit(rng, noisy_p=0.1):
    """Random small feedback-after-non-Clifford circuit, every record a DECISION.

    Magic wires 0..2 (|+>, noise, T/T_DAG/CS/CCZ), Clifford entanglers onto targets
    3..4, X/Y reads of the magic wires driving CX/CY/CZ feedback onto the targets,
    optionally a T on a CZ-fed target, then the targets read in a random basis."""
    p = noisy_p
    L = ["R 9", "M 9", "DECISION(0) rec[-1]", "DETECTOR rec[-1]", "RX 0 1 2"]
    for q in rng.sample(range(3), 2):           # two noisy wires keep the dense law cheap
        L.append(f"PAULI_CHANNEL_1({p / 3},{p / 3},{p / 3}) {q}")
    L.append(rng.choice(["T 0", "T_DAG 0"]))
    L.append(rng.choice(["CS 1 2", "CCZ 0 1 2", "T 1\nT_DAG 2"]))
    L.append(rng.choice(["R 3 4", "RX 3 4", "R 3\nRX 4"]))
    for q in range(3):
        L.append(f"{rng.choice(['CZ', 'CX'])} {q} {rng.choice([3, 4])}")
    reads, fbs = [], []
    for q in range(3):
        rb = rng.choice(["MX", "MY", "MX", "M"])
        L.append(f"{rb}({p / 2}) {q}" if rng.random() < 0.3 else f"{rb} {q}")
        reads.append(rb)
    k_dec = 1
    for j in range(3):
        L.append(f"DECISION({k_dec}) rec[-{3 - j}]")
        k_dec += 1
    t_after = set()
    for j, rb in enumerate(reads):
        # X/Y-type feedback from an X/Y read of a dressed wire is outside the retained
        # class (test_xy_feedback_from_dressed_xy_read_is_refused_by_name)
        P = rng.choice(["CX", "CY", "CZ"]) if rb == "M" else "CZ"
        t = rng.choice([3, 4])
        L.append(f"{P} rec[-{3 - j}] {t}")
        fbs.append((P, t))
    for t in (3, 4):
        if all(P == "CZ" for P, tt in fbs if tt == t) and rng.random() < 0.5:
            L.append(f"T {t}")
            t_after.add(t)
    for t in (3, 4):
        L.append(f"{rng.choice(['M', 'MX', 'MY'])} {t}")
        L.append(f"DECISION({k_dec}) rec[-1]")
        k_dec += 1
    return "\n".join(L) + "\n"


def _fold_readout(text):
    """The oracle's spelling of ``M[X|Y](p) q`` on a never-again-observed wire: an
    anticommuting error before a clean read (``record_law`` ignores readout args).
    Exact for these generators, whose read wires are abandoned at their read."""
    return re.sub(r"(M[XY]?)\(([^)]*)\) (\d+)",
                  lambda m: f"{'X' if m.group(1) == 'M' else 'Z'}_ERROR({m.group(2)}) "
                            f"{m.group(3)}\n{m.group(1)} {m.group(3)}", text)


def _law(text, cols):
    out = {}
    for rec, pr in record_law(text, max_branches=1 << 20).items():
        key = tuple(sum(rec[i] for i in col) & 1 for col in cols)
        out[key] = out.get(key, 0.0) + pr
    return out


def _dec_columns(text):
    decs = parities(text)[2]
    return [decs[k] for k in sorted(decs)]


@pytest.mark.parametrize("seed", range(10))
def test_random_feedback_after_nonclifford_record_law(seed):
    """Joint law of every record (as DECISIONs) on the RETAINED path == the exact dense
    record law; the deferred-measurement rewrite has the same law (oracle sanity) and the
    exact engine agrees on it.  Circuits outside the simulable class (refused by every
    engine) are skipped by name; the run must stay non-vacuous."""
    rng = random.Random(1000 + seed)
    text = _random_circuit(rng)
    cols = _dec_columns(text)
    oracle_text = _fold_readout(text)
    law = _law(oracle_text, cols)
    rew = _deferred_rewrite(oracle_text)
    law_rew = _law(rew, cols)
    assert max(abs(law.get(k, 0) - law_rew.get(k, 0)) for k in set(law) | set(law_rew)) < 1e-12
    try:
        S = _sampler(text)
    except ValueError as e:
        assert "non-commuting magic" in str(e), e
        pytest.skip("outside the simulable class for every engine: " + str(e)[:80])
    N = 20000
    buf = S.sample_barrier(N, 7 + seed)
    z, imp = max_z(_decision_bits(buf, N, len(cols)), law)
    assert imp == 0 and z < 5.5, (z, imp, text)
    # the deferred-measurement rewrite through the exact engine: raw records vs the law
    m = np.asarray(Circuit(rew).compile_detector_sampler(seed=3, engine="exact")
                   .sample(N, return_measurements=True)[-1], np.uint8)
    rec_cols = np.stack([m[:, c].sum(axis=1) & 1 for c in cols], axis=1)
    z2, imp2 = max_z(rec_cols, law)
    assert imp2 == 0 and z2 < 5.5, (z2, imp2)


def test_random_sweep_is_non_vacuous():
    """Most generated circuits are accepted, and the fallback engages on them."""
    accepted = hoisted = 0
    for seed in range(10):
        text = _random_circuit(random.Random(1000 + seed))
        try:
            S = _sampler(text)
        except ValueError:
            continue
        accepted += 1
        hoisted += _hoisted(S) > 0
        S.sample_barrier(8, 1)                  # retention accepted on every one
    assert accepted == 10 and hoisted >= 6, (accepted, hoisted)


@pytest.mark.parametrize("seed", range(4))
def test_random_retained_state_per_shot(seed):
    """Per-shot retained-state oracle on random deterministic-fault circuits (same
    generator shape, faults fixed to p = 1 or 0, targets kept as the output port)."""
    rng = random.Random(77 + seed)
    text = _random_circuit(rng, noisy_p=0.0)
    text = re.sub(r"PAULI_CHANNEL_1\([^)]*\) (\d)",
                  lambda m: f"{rng.choice(['X', 'Y', 'Z'])}_ERROR(1) {m.group(1)}", text)
    text = re.sub(r"(M[XY]?)\([^)]*\)", r"\1", text)
    # keep the targets as the output port: drop their final reads
    lines = text.splitlines()
    while not lines[-1].startswith("CX rec") and not lines[-1].startswith("CY rec") \
            and not lines[-1].startswith("CZ rec") and not lines[-1].startswith("T "):
        lines.pop()
    text = "\n".join(lines) + "\nOUTPUT_QUBITS out 3 4\n"
    try:
        _sampler(text)
    except ValueError as e:
        assert "non-commuting magic" in str(e), e
        pytest.skip("outside the simulable class for every engine")
    _check_retained(text, 24, 5 + seed)


def test_port_segment_groups_accept_the_adaptq_shape():
    """The xtim.port record surface (groups / residual_law) compiles the shape that
    adaptq had to rewrite into an external frame."""
    text = _teleport_like("X_ERROR(1) 0\n").replace("OUTPUT_QUBITS", "DEPOLARIZE1(0.01) 1\nOUTPUT_QUBITS")
    seg = xtim.compile_segment(text, xtim.Consume(dets=True, groups=True, residual_law=True))
    assert seg is not None


# ── deferred-measurement theorem: the window rule (review I-1 / M-4) ─────────
# Oracle: ``_branch_oracle`` (dense branch enumeration with Stim semantics — M(p) flips
# the RECORD only, MR*/MPP/REPEAT/tags/DEPOLARIZE1/2).  Per circuit: the joint DECISION
# law, and for every record class the mean of the engine's per-shot retained-state
# Pauli expectations (all 4^|out| words) against the oracle's conditional expectation;
# deterministic classes to 1e-6.

import _branch_oracle as _bo  # noqa: E402


def _check_conditional(text, shots=6000, seed=1):
    S = _sampler(text)
    buf = S.sample_barrier(shots, seed)
    cols = _bo.decisions(text)
    dec = _decision_bits(buf, shots, len(cols))
    ow = list(S.output_wires())
    words = list(itertools.product("IXYZ", repeat=len(ow))) if ow else [()]
    tab = _bo.cond_table(text, words)
    keys, cnt = np.unique(dec, axis=0, return_counts=True)
    emp = {tuple(int(x) for x in k): int(c) for k, c in zip(keys, cnt)}
    for k in set(emp) | set(tab):
        p = tab.get(k, (0.0, None))[0]
        c = emp.get(k, 0)
        assert p > 1e-13 or c == 0, ("impossible record class sampled", k, c)
        if 1e-13 < p < 1:
            z = abs(c - shots * p) / np.sqrt(shots * p * (1 - p))
            assert z < 5.5, ("record law", k, c, shots * p)
    if ow:
        eng = np.asarray(buf.pauli_expectations_all([_engine_reads(ow, w) for w in words])).real
        for k, (p, e) in tab.items():
            idx = np.flatnonzero((dec == np.array(k, np.uint8)).all(axis=1))
            if len(idx) < 30:
                continue
            m, sd = eng[idx].mean(axis=0), eng[idx].std(axis=0)
            dev = np.abs(m - e)
            # A class looks deterministic when every sampled shot agrees; that is evidence only
            # with enough shots (a 10% minority outcome is absent from 35 shots w.p. 2.5%).
            det = (sd < 1e-12) & (len(idx) >= 300)
            assert not det.any() or dev[det].max() < 1e-6, ("deterministic class", k)
            # Standard error: per-shot values lie in [-1, 1], so their spread around the oracle
            # mean e is at most sqrt(1 - e^2); the sample sd can undershoot it in small classes.
            spread = np.maximum(np.maximum(sd, np.sqrt(np.clip(1 - e * e, 0, None))), 1e-3)
            zx = dev / (spread / np.sqrt(len(idx)))
            assert zx.max() < 7.5, ("conditional expectation", k, float(zx.max()))
    return S


def _win(read, window, fb, post="", out="1", fault="X_ERROR(1) 0\n"):
    """Magic wire 0 (|+>, fault, T) entangled with 1 (and 1 with 5); ``read`` of 0, then
    ``window``, then ``fb``.  Wires 5, 6 are spectators the window may use."""
    return (_ANCHOR + "RX 0 1 5 6\n" + fault + "T 0\nCZ 0 1\nCX 1 5\n" + read +
            "\nDECISION(1) rec[-1]\n" + window + fb + "\n" + post + "OUTPUT_QUBITS out " + out + "\n")


MOVABLE = {
    # sources: measure-reset (the reset is after the read: in no window), p = 0 and p > 0
    "MRX": _win("MRX 0", "", "CZ rec[-1] 1"),
    "MRY": _win("MRY 0", "", "CZ rec[-1] 1"),
    "MRX(p)": _win("MRX(0.2) 0", "", "CZ rec[-1] 1"),
    "MRY(p)": _win("MRY(0.2) 0", "", "CZ rec[-1] 1"),
    "MY(p)!": _win("MY(0.2) !0", "", "CZ rec[-1] 1"),
    # MPP sources (desugared to an ancilla gadget; the feedback reads its MX)
    "MPP_X": (_ANCHOR + "RX 0 1 2\nX_ERROR(1) 0\nT 0\nCZ 0 1\nCX 2 1\nMPP X0*X2\n"
              "DECISION(1) rec[-1]\nCZ rec[-1] 1\nOUTPUT_QUBITS out 1\n"),
    "MPP_Y": (_ANCHOR + "RX 0 1 2\nX_ERROR(1) 0\nT 0\nCZ 0 1\nCX 2 1\nMPP Y0*X2\n"
              "DECISION(1) rec[-1]\nCZ rec[-1] 1\nOUTPUT_QUBITS out 1\n"),
    # Pauli noise in the window: on the target, two-qubit, on the control after its read
    "Z_ERROR_on_t": _win("MX 0", "Z_ERROR(0.3) 1\n", "CZ rec[-1] 1", out="1 5"),
    "DEPOLARIZE2_on_t": _win("MX 0", "DEPOLARIZE2(0.2) 1 6\n", "CZ rec[-1] 1", out="1 5"),
    "Z_ERROR_on_c": _win("MX 0", "Z_ERROR(1) 0\n", "CZ rec[-1] 1"),
    "X_ERROR_on_c": _win("MX 0", "X_ERROR(1) 0\n", "CZ rec[-1] 1"),
    # handled by conjugation: an MPP on the target (through its desugar gadget), a Clifford
    # (t stays observed — out 1 5 — so rule 0 does not apply and Q = X_anc Z_1 is exercised)
    "MPP_on_t": _win("MX 0", "MPP X1*Z6\nDECISION(2) rec[-1]\n", "CZ rec[-2] 1", out="1 5"),
    "MPP_Y_on_t": _win("MX 0", "MPP Y1*X6\nDECISION(2) rec[-1]\n", "CZ rec[-2] 1", out="1 5"),
    "CX_spectator_into_t": _win("MX 0", "CX 6 1\n", "CZ rec[-1] 1", out="1 6"),
    # commuting: a PAULI_EXPECTATION of Z on the target.  Only the MOVE is checked here: the
    # hoist runs only on the TwirlSampler family, which reports no PAULI_EXPECTATION column
    # (sample() returns an empty obs block for it, sample_barrier exposes OUTPUT_QUBITS words
    # only), and the engines that do report PE values never take this path.
    "PE_Z_on_t": _win("MX 0", "PAULI_EXPECTATION(0) Z1\n", "CZ rec[-1] 1"),
    # a reset in the window that the moved Pauli survives (RX = R; H: X -> Z, Z fixes |0>)
    "RX_on_t_CX": _win("MX 0", "RX 1\n", "CX rec[-1] 1", out="1 5"),
    # chains: the moved feedback's window holds another source and its feedback
    "chain_src_in_window": _win("MX 0", "MX 6\nDECISION(2) rec[-1]\nCZ rec[-1] 5\n",
                                "CZ rec[-2] 1", out="1 5"),
    "chain_target_is_other_src": _win("MX 0", "MX 6\nDECISION(2) rec[-1]\n",
                                      "CZ rec[-2] 6\nCZ rec[-1] 5", out="1 5"),
    # tags
    "tags": _win("MX[foo] 0", "", "CZ[bar] rec[-1] 1"),
    # REPEAT around the window (two rounds, each hoisted) / inside the window
    "REPEAT_around": (_ANCHOR + "RX 1\nREPEAT 2 {\nRX 0\nX_ERROR(0.3) 0\nT 0\nCZ 0 1\nMX 0\n"
                      "CZ rec[-1] 1\n}\nDECISION(1) rec[-2]\nDECISION(2) rec[-1]\n"
                      "OUTPUT_QUBITS out 1\n"),
    "REPEAT_inside": (_ANCHOR + "RX 0 1 5\nX_ERROR(1) 0\nT 0\nCZ 0 1\nMX 0\nDECISION(1) rec[-1]\n"
                      "REPEAT 2 {\nDEPOLARIZE1(0.1) 5\nH 5\n}\nCZ rec[-1] 1\n"
                      "OUTPUT_QUBITS out 1 5\n"),
    # another feedback in the window with an anticommuting Pauli on the same target
    "other_fb_noncommuting": _win("MX 0", "M 6\nDECISION(2) rec[-1]\nCX rec[-1] 1\n",
                                  "CZ rec[-2] 1"),
    # a feedback onto a measured, never-observed wire (rule 0: the identity)
    "fb_onto_dead_wire": _win("MX 0", "CZ rec[-1] 1\nMX 6\nDECISION(2) rec[-1]\n",
                              "CX rec[-1] 0"),
    # a hoisted source next to a clean noisy-readout control that is REUSED (held fold + undo)
    "M(p)_reused": _win("CX 1 6\nMX 0", "", "CZ rec[-1] 1\nM(0.2) 6\nDECISION(2) rec[-1]\n"
                        "CZ rec[-1] 5\nX_ERROR(0.1) 6\nM 6\nDECISION(3) rec[-1]", out="1 5"),
    "MRX(p)_reused": _win("CX 1 6\nMX 0", "", "CZ rec[-1] 1\nMRX(0.2) 6\nDECISION(2) rec[-1]\n"
                          "CZ rec[-1] 5\nMX 6\nDECISION(3) rec[-1]", out="1 5"),
}


@pytest.mark.parametrize("name", sorted(MOVABLE))
def test_window_rule_moves_and_is_exact(name):
    """Refused on integ/3.1.12 (sample_barrier) except MPP_X/MPP_Y/tags/REPEAT_*, which
    the previous rule already moved: now every one is moved and exact."""
    S = _check_conditional(MOVABLE[name])
    assert _hoisted(S) >= 1


STILL_REFUSED = {
    # an anticommuting PAULI_EXPECTATION on the target in the window
    "PE_X_on_t": (_win("MX 0", "PAULI_EXPECTATION(0) X1\n", "CZ rec[-1] 1"),
                  r"feedback #0 \(CZ rec onto qubit 1, reading record #1 \(MX of qubit 0\)\) "
                  r"is NOT MOVABLE.*its window contains a "
                  r"PAULI_EXPECTATION that anticommutes with the moved Pauli"),
    # an anticommuting measurement of the target in the window (target observed later)
    "MX_on_t": (_win("MX 0", "MX 1\nDECISION(2) rec[-1]\n", "CZ rec[-2] 1",
                     "H 1\nM 1\nDECISION(3) rec[-1]\n", out="5"),
                r"its window contains a X-measurement of qubit 1 that anticommutes"),
    # a T on the target where the moved Pauli is X
    "T_on_t": (_win("MX 0", "T 1\n", "CX rec[-1] 1"),
               r"its window contains a non-Clifford T on qubit 1 where the moved Pauli has an "
               r"X/Y component"),
    # cancelling non-Clifford gates: the check is per instruction (sufficient, not necessary)
    "T_TDAG_on_t": (_win("MX 0", "T 1\nT_DAG 1\n", "CX rec[-1] 1", out="1 5"),
                    r"its window contains a non-Clifford T on qubit 1 where the moved Pauli has "
                    r"an X/Y component \(the instruction-by-instruction conjugation leaves the "
                    r"Pauli group there\)"),
    # a reset of the target where the moved Pauli is X
    "reset_t": (_win("MX 0", "RX 1\n", "CZ rec[-1] 1"),
                r"its window contains a reset of qubit 1 where the moved Pauli is X/Y-type"),
}


@pytest.mark.parametrize("name", sorted(STILL_REFUSED))
def test_window_blocked_feedback_is_served_exactly(name):
    """Not movable by the theorem: the default (coherent) form is kept, out of the diagonal class. RE-PINNED
    2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness: the legacy retained engine refused it naming the
    blocking window instruction; the family law serves it (term route) and the retained state equals the dense
    branch oracle on every record class."""
    text, _why = STILL_REFUSED[name]
    _check_conditional(text, shots=6000, seed=3)


def test_moved_but_out_of_class_is_served_exactly():
    """``H 1; CX rec[-1] 1``: W^dag X_1 W = Z_1, so the feedback moves (as an X-controlled Z_1); that gate couples the
    control's S residual onto wire 1, where the window's H then acts — a non-diagonal residual on a kept qubit, which
    the legacy engine refused naming the representation limit. RE-PINNED 2026-10-20 (J-b1), refusal -> exactness."""
    _check_conditional(_win("MX 0", "H 1\n", "CX rec[-1] 1", out="1 5"), shots=6000, seed=3)


# ── refusal attribution in mixed circuits (re-review Important-1) ─────────────
_REAL = "RX 0 1\nX_ERROR(1) 0\nT 0\nCZ 0 1\nMX 0\nDECISION(1) rec[-1]\nCZ rec[-1] 1\nH 1\n"
_CLEAN_XY = "RX 3 4\nCZ 3 4\nMX 3\nDECISION(2) rec[-1]\nCX rec[-1] 4\n"
_CLEAN_BLK = "RX 3 4\nCZ 3 4\nMX 3\nDECISION(2) rec[-1]\nT 4\nCX rec[-1] 4\n"


@pytest.mark.parametrize("clean", [_CLEAN_XY, _CLEAN_BLK])
def test_mixed_circuit_is_served_exactly(clean):
    """A clean feedback next to the one that leaves the diagonal class: refused by name on the legacy engine (naming
    only feedback #0). RE-PINNED 2026-10-20 (J-b1), refusal -> exactness: served, dense-exact per record class."""
    assert _sampler(_ANCHOR + clean + "OUTPUT_QUBITS out 4\n").sample_barrier(8, 1) is not None
    _check_conditional(_ANCHOR + _REAL + clean + "OUTPUT_QUBITS out 1 4\n", shots=6000, seed=3)


def test_mpp_source_is_served_exactly():
    """An MPP source with a not-movable X feedback (a T in its window): refused by name on the legacy engine.
    RE-PINNED 2026-10-20 (J-b1), refusal -> exactness."""
    text = (_ANCHOR + "RX 0 1 2\nX_ERROR(1) 0\nT 0\nCZ 0 1\nCX 2 1\nMPP X0*X2\n"
            "DECISION(1) rec[-1]\nT 1\nCX rec[-1] 1\nOUTPUT_QUBITS out 1\n")
    _check_conditional(text, shots=6000, seed=3)


# ── composition with a NOT-moved in-place feedback (re-review Important-2) ───
# n (source M/MX 2, record #2) stays in place — its window holds a T on its X-type target;
# h (source MX 5) moves, and its moved Pauli lands on n's control wire 2 INSIDE n's read
# window.  n's own deferred-measurement precondition (wire 2 untouched between its
# measurement and its read) then fails in the emitted stream, so n must read a copy of 2.
# With that guard disabled both circuits are accepted and silently wrong (deviation 2.0).
_TRIG = "RX 0 1\nX_ERROR(1) 0\nT 0\nCZ 0 1\nMX 0\nDECISION(1) rec[-1]\nCZ rec[-1] 1\n"
GUARD = {
    "h_X_on_Zread_n": _ANCHOR + _TRIG + "RX 2 3 5\nCZ 5 3\nM 2\nDECISION(2) rec[-1]\nMX 5\n"
                      "DECISION(3) rec[-1]\nT 3\nCX rec[-2] 3\nCX rec[-1] 2\nMX 2\n"
                      "DECISION(4) rec[-1]\nOUTPUT_QUBITS out 1 3\n",
    "h_Z_on_Xread_n": _ANCHOR + _TRIG + "RX 2 3 5\nCZ 5 3\nMX 2\nDECISION(2) rec[-1]\nMX 5\n"
                      "DECISION(3) rec[-1]\nT 3\nCX rec[-2] 3\nCZ rec[-1] 2\nM 2\n"
                      "DECISION(4) rec[-1]\nOUTPUT_QUBITS out 1 3\n",
}


@pytest.mark.parametrize("name", sorted(GUARD))
def test_moved_gate_on_an_in_place_control_forces_its_copy(name):
    S = _check_conditional(GUARD[name], shots=8000, seed=3)
    assert _hoisted(S) == 2


_JOINT = ("R 9\nM 9\nDETECTOR rec[-1]\nDECISION(0) rec[-1]\nRX 0 1 2\nRX 3\nR 4\nR 5 6\n"
          "{fault}\nPAULI_CHANNEL_1(0.05,0.1,0.05) 2\nT 0\nT 1\nCX 0 3\nCZ 1 3\nCZ 2 3\n"
          "CX 0 5\nMX(0.1) 0\nDECISION(1) rec[-1]\nCZ rec[-1] 3\nMX(0.25) 1\nDECISION(2) rec[-1]\n"
          "CZ rec[-1] 5\nMRX(0.1) !2\nDECISION(3) rec[-1]\nCX rec[-1] 3\nOUTPUT_QUBITS out 3 4\n")


def test_jointly_fatal_feedbacks_are_served_exactly():
    """Fuzz seed 3076 with its fault made X-type: two moved feedbacks together leave the diagonal class (refused by
    name on the legacy engine, naming the shared wire). RE-PINNED 2026-10-20 (J-b1), refusal -> exactness."""
    _check_conditional(_JOINT.format(fault="X_ERROR(1) 0"), shots=8000, seed=5)


def test_jointly_fatal_shape_with_a_z_fault_is_accepted_and_exact():
    """The original seed-3076 circuit has Z_ERROR(1) before the T: since fix/z-channel-atoms a
    pure-Z channel builds only its Z atom (Z commutes with T), the default form is in class,
    nothing is moved, and the retained states match the dense branch oracle."""
    S = _check_conditional(_JOINT.format(fault="Z_ERROR(1) 0"), shots=8000, seed=5)
    assert _hoisted(S) == 0
