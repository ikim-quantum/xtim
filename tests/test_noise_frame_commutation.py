"""A pending single-qubit Clifford frame commutes THROUGH a Pauli noise channel (fix/noncommuting-
magic-readout, 2026-10-13).

eliminate_hadamards used to FLUSH a pending frame at every noise instruction, so a CCZ-factory
port's X readout ``H q; DEPOLARIZE1(p) q; M q`` kept an explicit H mid-stream: the H;M -> MX fold
never happened, a fault before a CCZ (whose residual carries CZ arms) met the explicit H and left
the diagonal class — the exact sampler refused the circuit ('non-commuting magic' hint) and the
twirl sampler routed every such shot through PPR plans that refused into the exact fallback
(~2 s/shot on the f3 factory). Now ``F ; N == N' ; F`` with ``N' = F^dag N F``: DEPOLARIZE1/2 are
invariant, X/Y/Z_ERROR and PAULI_CHANNEL_1/2 are relabelled per qubit (an instruction is split
where its qubits' relabelled channels differ), and the forced-fault program maps a user Pauli at
the site through the same relabelling (Instr::origin_pos / origin_letter).

Oracles (no xtim code): tests/_dense_oracle.py (noisy density matrix, record laws).
xtim side: the exact forced-fault law summed over every fault configuration — the exact noisy
law of the SAME compiled program the sampler runs — plus a sampled-law check.
"""
from __future__ import annotations

import collections
import itertools
import random

import numpy as np
import pytest

stim = pytest.importorskip("stim")

import xtim  # noqa: E402
from xtim import _xtim  # noqa: E402
from _dense_oracle import Oracle  # noqa: E402

# ── the 24 single-qubit Cliffords as gate words (and their inverses) ────────────────────────────
_INV = {"H": "H", "S": "S_DAG", "X": "X"}


def _frames():
    seen, out, frontier = set(), [], [()]
    while frontier:
        nxt = []
        for w in frontier:
            t = stim.Tableau(1)
            for g in w:
                t = t.then(stim.Tableau.from_named_gate(g))
            key = str(t)
            if key in seen:
                continue
            seen.add(key)
            out.append(w)
            nxt += [w + (g,) for g in ("H", "S", "X")]
        frontier = nxt
    assert len(out) == 24
    return out


FRAMES = _frames()


def _word(w, q):
    return "".join(f"{g} {q}\n" for g in w)


def _inv_word(w, q):
    return "".join(f"{_INV[g]} {q}\n" for g in reversed(w))


# ── exact noisy record law, two independent ways ────────────────────────────────────────────────
def _with_decisions(text):
    nrec = sum(len(ln.split()) - 1 for ln in text.splitlines()
               if ln.split() and ln.split()[0] in ("M", "MX", "MY"))
    return text + "".join(f"DECISION({i}) rec[-{nrec - i}]\n" for i in range(nrec)), nrec


def _site_options(s):
    """Per target of one site: [(pauli or None, prob)]; targets are independent."""
    ch, qs, ps = s["channel"], s["qubits"], s["probs"]
    if ch in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        one = [(None, 1 - ps[0]), (ch[0], ps[0])]
        return [one] * len(qs)
    if ch == "DEPOLARIZE1":
        return [[(None, 1 - ps[0])] + [(c, ps[0] / 3) for c in "XYZ"]] * len(qs)
    if ch == "PAULI_CHANNEL_1":
        return [[(None, 1 - sum(ps))] + list(zip("XYZ", ps))] * len(qs)
    if ch == "CORRELATED_ERROR":                 # one target: the whole chain support
        q, rem, dist = [], 1.0, {}
        for p, w in zip(ps, s["paulis"]):
            q.append(p * rem)
            rem *= 1 - p
        dist = [(None, 1 - sum(q))]
        for qt, w in zip(q, s["paulis"]):
            dist.append((None if set(w) == {"I"} else w, qt))
        return [dist]
    pairs = ["".join(p) for p in itertools.product("IXYZ", repeat=2)][1:]
    if ch == "DEPOLARIZE2":
        dist = [(None, 1 - ps[0])] + [(p, ps[0] / 15) for p in pairs]
    else:
        dist = [(None, 1 - sum(ps))] + list(zip(pairs, ps))
    return [dist] * (len(qs) // 2)


def xtim_noisy_law(text):
    text, nrec = _with_decisions(text)
    sim = xtim.Circuit(text).compile_exact_branches()
    assert sim.noise_refusal == "", sim.noise_refusal
    targets = []                                   # (site, target, options)
    for si, s in enumerate(sim.noise_sites()):
        for t, opts in enumerate(_site_options(s)):
            targets.append((si, t, [o for o in opts if o[1] > 0]))
    law = collections.defaultdict(float)
    for combo in itertools.product(*[o for _, _, o in targets]):
        w = float(np.prod([p for _, p in combo]))
        faults = [(si, t, pa) for (si, t, _), (pa, _) in zip(targets, combo) if pa is not None]
        br = sim.exact_branches(faults)
        for p, d in zip(br.probabilities, br.decisions):
            law[tuple(int(b) for b in d)] += w * float(p)
    return dict(law)


def oracle_law(text):
    O = Oracle(text)
    O.run()
    return {tuple(k): float(v) for k, v in O.record_dist().items() if abs(v) > 1e-15}


def assert_laws_equal(a, b, tol=1e-10):
    keys = set(a) | set(b)
    bad = {k: (a.get(k, 0.0), b.get(k, 0.0)) for k in keys if abs(a.get(k, 0.0) - b.get(k, 0.0)) > tol}
    assert not bad, bad


def _sampled_law(text, shots, seed):
    m = np.asarray(xtim.Circuit(text).compile_sampler(seed=seed).sample(shots), dtype=np.uint8)
    c = collections.Counter(map(tuple, m.tolist()))
    return {k: v / shots for k, v in c.items()}


def assert_sampled_law(text, exact, shots=20000, seed=11):
    """The SAMPLER's own noise path (the relabelled channel letters, not the forced provenance)
    against the oracle law, 6 sigma per outcome."""
    got = _sampled_law(text, shots, seed)
    for k in set(exact) | set(got):
        p = exact.get(k, 0.0)
        sd = max(np.sqrt(p * (1 - p) / shots), 1.0 / shots)
        assert abs(got.get(k, 0.0) - p) < 6 * sd, (text, k, got.get(k, 0.0), p)


def _signature_has_gate(text, gate_code):
    sig = _xtim.deferred_signature(text)
    return any(item.startswith(f"0:{gate_code},") for item in sig.split(";"))


H_CODE = 7   # GateKind::H in the signature's gate enumeration
MAGIC_CODES = (8, 9, 10, 11)   # T, CS, CCZ, CH


def _h_after_magic(text):
    """An explicit H in the normalized stream after the last non-Clifford gate."""
    items = _xtim.deferred_signature(text).split(";")
    last = max((i for i, it in enumerate(items)
                if any(it.startswith(f"0:{g},") for g in MAGIC_CODES)), default=-1)
    return any(it.startswith(f"0:{H_CODE},") for it in items[last + 1:])


# ── 1. every frame x every channel: exact relabelling (law) and the frame absorbed (structure) ──
ONE_Q = ["X_ERROR(0.11)", "Y_ERROR(0.13)", "Z_ERROR(0.17)", "DEPOLARIZE1(0.19)",
         "PAULI_CHANNEL_1(0.05,0.11,0.23)"]
MAGIC_PREP = "R {q}\nH {q}\nT {q}\nH {q}\n"        # <Z> = 1/sqrt2: a letter error moves the law


@pytest.mark.parametrize("ch", ONE_Q)
def test_every_frame_through_every_one_qubit_channel(ch):
    for w in FRAMES:
        text = (MAGIC_PREP.format(q=0) + _word(w, 0) + f"{ch} 0\n" + _inv_word(w, 0)
                + "M 0\n")
        exact = oracle_law(text)
        assert_laws_equal(xtim_noisy_law(text), exact)
        assert_sampled_law(text, exact)
        if w:                                      # the frame never survives as an explicit H
            core = "R 0\n" + _word(w, 0) + f"{ch} 0\nM 0\n"
            assert not _signature_has_gate(core, H_CODE), (w, ch, _xtim.deferred_signature(core))


PC2 = "PAULI_CHANNEL_2(" + ",".join(f"{0.004 * (t + 1):.3f}" for t in range(15)) + ")"


@pytest.mark.parametrize("ch", ["DEPOLARIZE2(0.2)", PC2])
def test_frame_pairs_through_two_qubit_channels(ch):
    rng = random.Random(7)
    pairs = [(a, b) for a in FRAMES for b in FRAMES]
    for wa, wb in [(FRAMES[0], FRAMES[5])] + rng.sample(pairs, 30):
        text = (MAGIC_PREP.format(q=0) + MAGIC_PREP.format(q=1) + _word(wa, 0) + _word(wb, 1)
                + f"{ch} 0 1\n" + _inv_word(wa, 0) + _inv_word(wb, 1) + "M 0 1\n")
        exact = oracle_law(text)
        assert_laws_equal(xtim_noisy_law(text), exact)
        assert_sampled_law(text, exact)


def test_multi_target_channel_splits_by_relabelled_letter():
    """X_ERROR over three wires carrying three different pending frames (the prep's trailing H
    cancels on 0, stays on 1, and composes with S on 2): one instruction becomes three
    (X->X, X->Z, X->Y), qubit order kept; the forced program addresses each user target."""
    text = (MAGIC_PREP.format(q=0) + MAGIC_PREP.format(q=1) + MAGIC_PREP.format(q=2)
            + "H 0\nS 2\nX_ERROR(0.2) 0 1 2\nH 0\nS_DAG 2\nM 0 1 2\n")
    sig = _xtim.deferred_signature(text)
    noise = [it for it in sig.split(";") if it.startswith("1:")]
    assert [it.split(",")[0] for it in noise] == ["1:0", "1:2", "1:1"], sig   # X, Z, Y _ERROR
    exact = oracle_law(text)
    assert_laws_equal(xtim_noisy_law(text), exact)
    assert_sampled_law(text, exact)


# ── 1b. E / ELSE_CORRELATED_ERROR chains: every term relabelled factor by factor, ONE site ─────
CHAIN = ("E(0.1) X0 Z1\nELSE_CORRELATED_ERROR(0.15) Y0 Y2\n"
         "ELSE_CORRELATED_ERROR(0.2) Z0 X1 X2\nELSE_CORRELATED_ERROR(0.25) X2\n")


def test_frames_through_correlated_chain_law_and_structure():
    """A pending frame on each wire of a 3-qubit chain commutes through it (chain conjugation:
    corr'[t][i] = preimage_i(corr[t][i]), sequential probabilities untouched): the chain stays
    ONE noise instruction, no explicit H survives into the read, and the exact forced-fault law
    (every user word mapped through the per-qubit letter provenance) and the sampler's own law
    both equal the dense oracle."""
    rng = random.Random(1013)
    for _ in range(24):
        ws = [rng.choice(FRAMES) for _ in range(3)]
        text = ("".join(MAGIC_PREP.format(q=q) for q in range(3))
                + "".join(_word(w, q) for q, w in enumerate(ws)) + CHAIN
                + "".join(_inv_word(w, q) for q, w in enumerate(ws)) + "M 0 1 2\n")
        exact = oracle_law(text)
        assert_laws_equal(xtim_noisy_law(text), exact)
        assert_sampled_law(text, exact)
        core = ("R 0 1 2\n" + "".join(_word(w, q) for q, w in enumerate(ws)) + CHAIN
                + "M 0 1 2\n")
        sig = _xtim.deferred_signature(core)
        assert not _signature_has_gate(core, H_CODE), (ws, sig)
        assert len([it for it in sig.split(";") if it.startswith("1:")]) == 1, sig


def test_frame_through_chain_dem_matches_stim():
    """Clifford, stim-checkable: `H 0; S 1; chain; ...; MX 0; M...` — the DEM xtim exports from the
    commuted chain equals stim's (mechanisms and values at 1e-12)."""
    text = ("RX 0\nR 1 2\nH 0\nS 1\nH 2\n" + CHAIN + "H 2\nS_DAG 1\nM 0 1 2\n"
            "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")
    got = stim.DetectorErrorModel(str(xtim.Circuit(text).detector_error_model(
        approximate_disjoint_errors=True)))
    want = stim.Circuit(text).detector_error_model(approximate_disjoint_errors=True)

    def canon(dem):
        out = {}
        for ins in dem.flattened():
            if ins.type == "error":
                k = tuple(sorted(str(t) for t in ins.targets_copy()))
                q, p = out.get(k, 0.0), ins.args_copy()[0]
                out[k] = q + p - 2 * q * p
        return out
    g, w = canon(got), canon(want)
    assert set(g) == set(w), (g, w)
    for k in w:
        assert abs(g[k] - w[k]) < 1e-12, (k, g[k], w[k])
    assert not _signature_has_gate(text, H_CODE), _xtim.deferred_signature(text)


# ── 2. the factory pattern: X fault before a CCZ, then `H q; DEPOLARIZE1 q; M q` ────────────────
MINI_REPRO = """RX 0 1 2
R 3
M 3
X_ERROR(0.1) 0
CCZ 0 1 2
CZ rec[-1] 0
H 0 1 2
DEPOLARIZE1(0.01) 0 1 2
M 0 1 2
"""


def test_minimized_repro_accepted_and_exact():
    """Refused before the fix (deferred-stream gate index 7, the 'non-commuting magic' hint): the
    explicit H on the CCZ wire met the X fault's CZ arm. Now accepted, and its law is exact."""
    sampler = xtim.Circuit(MINI_REPRO).compile_sampler(seed=3)
    sampler.sample(16)
    assert not _h_after_magic(MINI_REPRO)
    assert_laws_equal(xtim_noisy_law(MINI_REPRO), oracle_law(MINI_REPRO))


def test_minimized_repro_sampled_law_matches_oracle():
    assert_sampled_law(MINI_REPRO, oracle_law(MINI_REPRO), shots=40000)


# Deterministic detectors: |CCZ> sandwich with a Z-copy check; X faults before the CCZ; the
# data read as `H; noise; M`. Twirl sample / sample_barrier and DEM must agree with the oracle.
DET_CIRCUIT = """RX 0 1 2
R 3 4
X_ERROR(0.05) 0 1
CCZ 0 1 2
CX 0 3
CX 0 4
H 0 1 2
DEPOLARIZE1(0.02) 0 1 2
X_ERROR(0.03) 3
M 0 1 2 3 4
DETECTOR rec[-2] rec[-1]
"""


def _oracle_det_law(text):
    O = Oracle(text)
    O.run()
    law = collections.defaultdict(float)
    for r, p in O.record_dist().items():
        law[(r[3] ^ r[4],)] += p
    return dict(law)


def test_twirl_and_dem_on_the_factory_pattern():
    exact = _oracle_det_law(DET_CIRCUIT)
    p1 = exact.get((1,), 0.0)
    dem = xtim.Circuit(DET_CIRCUIT).detector_error_model()
    ps = [inst.args_copy()[0] for inst in stim.DetectorErrorModel(str(dem)).flattened()
          if inst.type == "error"]
    q = 0.0
    for p in ps:
        q = q * (1 - p) + (1 - q) * p
    assert abs(q - p1) < 1e-12, (q, p1)
    shots = 200000
    T = _xtim.TwirlSampler(DET_CIRCUIT, 1.0, "", 0)
    det = np.unpackbits(np.asarray(T.sample(shots, seed=5)[0]), axis=1, bitorder="little")
    assert not [k for k in T.record_report()["route_counts"] if k.startswith("exact")]   # J-b1 F: M-API
    B = _xtim.TwirlSampler(DET_CIRCUIT, 1.0, "", 0)
    det_b = np.unpackbits(np.asarray(B.sample_barrier(shots, seed=5).dets()), axis=1,
                          bitorder="little")
    for d in (det, det_b):
        rate = d[:shots, 0].mean()
        assert abs(rate - p1) < 6 * np.sqrt(p1 * (1 - p1) / shots), (rate, p1)


# ── 3. random circuits: frames + noise + magic, exact law vs the dense oracle ──────────────────
def _random_circuit(rng, n):
    lines = [f"{rng.choice(['R', 'RX'])} {q}" for q in range(n)]
    magic = 0
    one_q = ["H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "H_YZ"]
    for _ in range(rng.randint(8, 14)):
        r = rng.random()
        if r < 0.35:
            q = rng.randrange(n)
            lines.append(f"{rng.choice(one_q)} {q}")
        elif r < 0.5:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(['CX', 'CZ'])} {a} {b}")
        elif r < 0.6 and magic < 2:
            magic += 1
            if n >= 3 and rng.random() < 0.4:
                lines.append("CCZ " + " ".join(map(str, rng.sample(range(n), 3))))
            else:
                lines.append(f"{rng.choice(['T', 'T_DAG'])} {rng.randrange(n)}")
        else:
            k = rng.choice(["X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "PAULI_CHANNEL_1",
                            "DEPOLARIZE2", "PAULI_CHANNEL_2"])
            if k in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
                a, b = rng.sample(range(n), 2)
                if k == "DEPOLARIZE2":
                    lines.append(f"DEPOLARIZE2(0.06) {a} {b}")
                else:
                    ps = [0.0] * 15
                    for t in rng.sample(range(15), 3):
                        ps[t] = round(rng.uniform(0.01, 0.05), 3)
                    lines.append(f"PAULI_CHANNEL_2({','.join(map(str, ps))}) {a} {b}")
            else:
                qs = rng.sample(range(n), rng.randint(1, min(2, n)))
                arg = ("0.02,0.05,0.08" if k == "PAULI_CHANNEL_1" else "0.07")
                lines.append(f"{k}({arg}) " + " ".join(map(str, qs)))
            # a frame right before more noise on the same wire (the pattern under test)
            q = rng.randrange(n)
            lines.append(f"{rng.choice(one_q)} {q}")
    lines.append(f"{rng.choice(one_q)} {rng.randrange(n)}")
    lines.append(f"DEPOLARIZE1(0.04) {rng.randrange(n)}")
    lines.append(" ".join([rng.choice(["M", "MX", "MY"])] + [str(q) for q in range(n)]))
    return "\n".join(lines) + "\n"


def _noise_targets(text):
    tot = 1
    for s in xtim.Circuit(_with_decisions(text)[0]).compile_exact_branches().noise_sites():
        for opts in _site_options(s):
            tot *= len([o for o in opts if o[1] > 0])
    return tot


def test_random_circuits_frames_noise_magic_exact_law():
    rng = random.Random(20261013)
    checked = 0
    for _ in range(400):
        n = rng.randint(2, 4)
        text = _random_circuit(rng, n)
        try:
            if _noise_targets(text) > 3000:
                continue
            law = xtim_noisy_law(text)
        except xtim.XtimRejectError:
            continue                               # outside the class (sequential magic etc.)
        assert_laws_equal(law, oracle_law(text))
        checked += 1
        if checked >= 60:
            break
    assert checked >= 40, checked


# ── 4. a PAULI_EXPECTATION does not block the read fold (review F1, 2026-10-14) ────────────────
# Deferral places the terminal reads AFTER every PAULI_EXPECTATION. push_frames used to flush ALL
# frames at one, so `H q; noise q; M q` never folded once any PAULI_EXPECTATION existed and the
# DEM refused the factory pattern (accepted on d93f3b1 via the deleted DEM-only fold). Now only
# the observable's own wires flush.
def _pe_variants():
    return {"repro": DET_CIRCUIT + "PAULI_EXPECTATION(0) Z3\n",
            "unrelated_wire": DET_CIRCUIT.replace("R 3 4\n", "R 3 4 5\n") + "PAULI_EXPECTATION(0) Z5\n"}


@pytest.mark.parametrize("name", ["repro", "unrelated_wire"])
def test_pauli_expectation_elsewhere_keeps_the_read_fold(name):
    text = _pe_variants()[name]
    assert not _h_after_magic(text), _xtim.deferred_signature(text)
    dem = xtim.Circuit(text).detector_error_model()          # refused on 8daadec
    ps = [inst.args_copy()[0] for inst in stim.DetectorErrorModel(str(dem)).flattened()
          if inst.type == "error"]
    q = 0.0
    for p in ps:
        q = q * (1 - p) + (1 - q) * p
    assert abs(q - _oracle_det_law(DET_CIRCUIT).get((1,), 0.0)) < 1e-12
    exact = oracle_law(text)
    assert_laws_equal(xtim_noisy_law(text), exact)
    assert_sampled_law(text, exact)


def test_pauli_expectation_on_the_frame_wire_stays_exact():
    """The PAULI_EXPECTATION reads the H'd wire itself, between its noise and its read: that frame
    must flush there (the expectation is of the H'd wire), and both the expectation value and the
    records' law must be the oracle's."""
    text = DET_CIRCUIT.replace("DEPOLARIZE1(0.02) 0 1 2\n",
                               "DEPOLARIZE1(0.02) 0 1 2\nPAULI_EXPECTATION(0) Z1\n")
    exact = oracle_law(text)
    assert_laws_equal(xtim_noisy_law(text), exact)
    assert_sampled_law(text, exact)
    O = Oracle(text)
    O.run()
    want = sum(v[0] for v in O._exp_vals[0].values()).real
    assert abs(want) > 0.1                                   # a non-trivial value
    text_d, _ = _with_decisions(text)
    sim = xtim.Circuit(text_d).compile_exact_branches()
    got = 0.0
    targets = []
    for si, s_ in enumerate(sim.noise_sites()):
        for t, opts in enumerate(_site_options(s_)):
            targets.append((si, t, [o for o in opts if o[1] > 0]))
    for combo in itertools.product(*[o for _, _, o in targets]):
        w = float(np.prod([p for _, p in combo]))
        faults = [(si, t, pa) for (si, t, _), (pa, _) in zip(targets, combo) if pa is not None]
        br = sim.exact_branches(faults)
        got += w * float(np.dot(br.probabilities, br.expectations[:, 0]))
    assert abs(got - want) < 1e-10, (got, want)
