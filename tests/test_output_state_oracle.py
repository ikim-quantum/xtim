"""xtim.output_state against independent oracles (PLAN.md Task 3)."""
import dataclasses
import pathlib
import re
import sys

import numpy as np
import pytest
import stim

import xtim
import _output_state_oracle as O

ONE = ["H", "S", "S_DAG", "SQRT_X"]


@pytest.fixture(scope="module", autouse=True)
def _report_routes():                                           # Task 2 review I1: how often the fallback ran
    yield
    sys.__stderr__.write(f"\n[output_state oracle] clifford_paulis routes (this process): {O.ROUTES}\n")


def _hop(q, a, b, reps):
    body = (f"R {a} {b}\nH {a}\nCX {a} {b}\nCX {q} {a}\nH {q}\nM {q} {a}\nCX rec[-1] {b}\nCZ rec[-2] {b}\nSWAP {q} {b}")
    return body if reps == 1 else f"REPEAT {reps} {{\n{body}\n}}"


def known_circuit(seed, n=4, straddle=False, rest_magic=False):
    """Outputs 0..n-1, a rest qubit n (with magic of its own when rest_magic), ancillas n+1, n+2 for teleports.
    State: C · prod_k exp(-i s_k pi/8 P_k) |0>, P_k = X_{q_k} Z(m_k) (q_k distinct, m_k off the q's): commuting axes,
    independent columns. C = a random Clifford (stim); then a closed-frame teleport (sometimes in a REPEAT), an MPP of
    an expected stabilizer, and ancilla resets — all the identity on the output. With straddle, the axis X_{q_0} X_rest
    (independent column) couples output and rest: Lemma 4a's "not deterministic". Returns (text, expected | None)."""
    rng = np.random.default_rng(seed)
    r, a, b = n, n + 1, n + 2
    k = int(rng.integers(1, n))
    qs = [int(x) for x in rng.choice(n, k, replace=False)]
    others = [q for q in range(n) if q not in qs]
    axes, L = [], []
    for q in qs:
        m = [int(x) for x in others if rng.random() < 0.5]
        axes.append(("*".join([f"X{q}"] + [f"Z{x}" for x in m]), int(rng.choice([1, -1]))))
    if straddle:
        axes.append((f"X{qs[0]}*X{r}", 1))
    L.append(O.rot_text(axes, undo=False))
    if rest_magic:
        L.append(f"H {r}\nT {r}")
    cliff = stim.Circuit()
    for _ in range(int(rng.integers(4, 12))):
        if rng.random() < 0.5:
            cliff.append(str(rng.choice(ONE)), [int(rng.integers(n))])
        else:
            c, t = (int(x) for x in rng.choice(n, 2, replace=False))
            cliff.append(str(rng.choice(["CX", "CZ"])), [c, t])
    L.append(str(cliff))
    L.append(_hop(int(rng.integers(n)), a, b, int(rng.integers(1, 3))))
    if straddle:
        return "\n".join(L) + "\n", None
    padded = cliff.copy()
    padded.append("I", [n - 1])                                 # the tableau spans all n outputs
    T = stim.Tableau.from_circuit(padded)
    exp_axes = [T(O.ps_of(p, n)) * s for p, s in axes]
    exp_ps = [T(O.ps_of(f"Z{j}", n)) for j in others]           # stab(|0>) ∩ C(axes) = <Z_j : j not in qs>
    exp_all = [T(O.ps_of(f"Z{j}", n)) for j in range(n)]        # stab(|S>), |S> = C|0>
    if exp_ps:
        st0 = exp_ps[0]
        L.append("MPP " + ("!" if st0.sign.real < 0 else "") + "*".join(f"{'_XYZ'[st0[q]]}{q}" for q in range(n) if st0[q]))
    L.append(f"R {a} {b}")
    return "\n".join(L) + "\n", dict(axes=exp_axes, ps=exp_ps, all=exp_all)


@pytest.mark.parametrize("seed", range(40))
def test_known_by_construction(seed):
    text, exp = known_circuit(seed, rest_magic=(seed % 2 == 1))
    qs, n = list(range(4)), 4
    assert xtim.output_is_deterministic(text, qs) is True
    st = xtim.output_state(text, qs)
    assert O.SignedGroup(st.pauli_stabilizers, n) == O.SignedGroup(exp["ps"], n)
    _assert_same_state(st, exp, n)
    O.assert_state(text, st)


def _conj_quarter(q, a, s):
    """exp(-i s pi/4 A) Q exp(+i s pi/4 A) = Q if they commute, else -i s A Q."""
    return q if q.commutes(a) else a * q * (-1j * s)


def _assert_same_state(st, exp, n):
    """R|S> (the helper's) equals the construction's state exactly. The split is deterministic but not unique (M6): an axis
    may come out times a Pauli stabilizer g (exp(-i t A g) R'|S> = exp(-i t A) R'|S>, as g|S> = |S> commutes with
    every axis), and with its sign flipped, the difference exp(-i s pi/4 A) moved into |S>. So: the axes match the
    construction's one to one modulo the signed Pauli stabilizer group, and stab(S) = K stab(S_construction) K^dag
    with K the product of the moved pi/4 rotations. No freedom beyond that is allowed."""
    psg = O.SignedGroup(st.pauli_stabilizers, n)
    assert len(st.rotations) == len(exp["axes"])
    used, moved = set(), []
    for A_e in exp["axes"]:
        s_e = int(A_e.sign.real)
        A_e = A_e * s_e
        hits = []
        for j, (p, s_h) in enumerate(st.rotations):
            pr = O.ps_of(p, n) * A_e                            # = c g with g in the signed group
            c = psg.member_sign(pr)                             # pr in the group (+1), -pr in it (-1), or neither
            if c is not None:
                hits.append((j, c * s_h))                       # the helper's sign on A_e itself
        assert len(hits) == 1 and hits[0][0] not in used, (str(A_e), hits, st.rotations)
        used.add(hits[0][0])
        if hits[0][1] != s_e:
            moved.append((A_e, s_e))
    want = []
    for g in exp["all"]:
        for a, s in moved:
            g = _conj_quarter(g, a, s)
        want.append(g)
    assert O.SignedGroup(st.pauli_stabilizers + st.clifford_paulis, n) == O.SignedGroup(want, n)


@pytest.mark.parametrize("seed", range(10))
def test_straddling_axis_is_not_deterministic(seed):           # Lemma 4a (probe: 300/300)
    text, _ = known_circuit(100 + seed, straddle=True)
    ok, why = xtim.output_is_deterministic(text, list(range(4)), explain=True)
    assert not ok and why.startswith("RF_OUTPUT_NOT_DETERMINISTIC:") and "rotation couples" in why


def test_missing_feedback_refusal_branch_is_hit():              # M5
    refused = 0
    for seed in range(20):
        text, _ = known_circuit(200 + seed)
        dropped = re.sub(r"CZ rec\[-2\] \d+\n", "", text, count=1)  # every known_circuit has one hop
        if xtim.output_is_deterministic(dropped, list(range(4))):
            O.assert_state(dropped, xtim.output_state(dropped, list(range(4))))
        else:
            with pytest.raises(xtim.XtimFidelityRefusal) as e:
                xtim.output_state(dropped, list(range(4)))
            assert e.value.code == "RF_OUTPUT_NOT_DETERMINISTIC"
            refused += 1
    assert refused >= 1


CODE = [(0, 3, 7, 10), (3, 7, 9, 12), (7, 10, 12, 13)]
MOVE = dict(zip((0, 3, 7, 9, 10, 12, 13), range(21, 28)))


def test_cultivation_d3_faithful_acceptance():
    c = xtim.load_example("cultivation_d3_faithful")
    out = list(range(21, 28))
    assert xtim.output_candidates(c) == [0, 3, 7, 9, 10, 12, 13] + out       # I1
    st = xtim.output_state(c, out)                                            # the acceptance subset: 21..27
    n = 28
    code = [("*".join(f"{L}{MOVE[q]}" for q in s), 1) for s in CODE for L in "XZ"]
    g = O.SignedGroup(code, n)
    assert O.SignedGroup(st.pauli_stabilizers, n) == g                        # the code's six checks, all +1
    assert len(st.rotations) == 1                                             # the one rotation a logical T undoes
    zl = O.ps_of("*".join(f"Z{q}" for q in out), n)
    pr = O.ps_of(st.rotations[0][0], n) * zl
    assert g.member_sign(pr) is not None                                        # axis ≡ ±Z^7 modulo the code
    assert len(st.clifford_paulis) == 1                                       # a logical Pauli (decision 2: X or Y)
    cp = O.ps_of(st.clifford_paulis[0][0], n)
    assert not cp.commutes(zl) and all(cp.commutes(O.ps_of(p, n)) for p, _ in code)
    O.assert_state(c, st)
    # completeness, non-dense (M8): a Pauli stabilizer commutes with the code, so it is a stabilizer times I/X_L/Y_L/Z_L;
    # none of the three nontrivial logicals has |<L>| = 1.
    state, wires = xtim._xtim.input_state_of(c.without_noise().text)
    for xs, zs, ph in ((wires, [], 0), ([], wires, 0), (wires, wires, 1)):
        assert abs(state.pauli_expectation(xs, zs, ph)) < 1 - 1e-9
    # the 14-qubit candidate list is deterministic too, with the old data wires at +Z
    st14 = xtim.output_state(c, [0, 3, 7, 9, 10, 12, 13] + out)
    g14 = O.SignedGroup(st14.pauli_stabilizers, n)
    for q in (0, 3, 7, 9, 10, 12, 13):
        assert g14.member_sign(O.ps_of(f"Z{q}", n)) == 1


def _rejected(pairs, ref, n):
    """The signed-span check refuses pairs as generators of ref: a bad generator set, or a different signed group."""
    try:
        return O.SignedGroup(pairs, n) != ref
    except AssertionError:
        return True


def _mutations(pairs, qubits=()):
    """Every sign flip, drop, duplicate and, on qubits, single-letter change (one qubit's letter to each other of
    I/X/Y/Z). A letter change multiplies a generator by a one-qubit Pauli, so it keeps the group only if that Pauli is
    in it: never for a distance-3 code, possible for small random groups (so known_circuit checks no letter changes)."""
    for i, (p, s) in enumerate(pairs):
        yield pairs[:i] + [(p, -s)] + pairs[i + 1:]
        yield pairs[:i] + pairs[i + 1:]
        yield pairs[:i + 1] + [(p, s)] + pairs[i + 1:]
        f = dict((q, L) for q, L in O.parse(p))
        for q in qubits:
            for L in "IXYZ":
                if f.get(q, "I") == L:
                    continue
                g = dict(f)
                g.pop(q) if L == "I" else g.__setitem__(q, L)
                if g:
                    yield pairs[:i] + [("*".join(f"{g[k]}{k}" for k in sorted(g)), s)] + pairs[i + 1:]


def test_signed_group_check_rejects_mutations():                # N2: the polynomial check is at least as strong
    c = xtim.load_example("cultivation_d3_faithful")
    out, n = list(range(21, 28)), 28
    st = xtim.output_state(c, out)
    ref = O.SignedGroup(st.pauli_stabilizers, n)
    code = [("*".join(f"{L}{MOVE[q]}" for q in s), 1) for s in CODE for L in "XZ"]
    assert O.SignedGroup(code, n) == ref and O.SignedGroup(list(reversed(code)), n) == ref
    muts = list(_mutations(list(st.pauli_stabilizers), out))
    assert len(muts) > 100 and all(_rejected(m, ref, n) for m in muts)
    for seed in range(10):                                        # known_circuit: the exact-state check fails too
        text, exp = known_circuit(seed)
        st = xtim.output_state(text, list(range(4)))
        ref = O.SignedGroup(exp["ps"], 4)
        assert all(_rejected(m, ref, 4) for m in _mutations(list(st.pauli_stabilizers)))
        for i, (p, s) in enumerate(st.rotations):
            bad = st.rotations[:i] + ((p, -s),) + st.rotations[i + 1:]
            with pytest.raises(AssertionError):
                _assert_same_state(dataclasses.replace(st, rotations=bad), exp, 4)
        for i, (p, s) in enumerate(st.clifford_paulis):
            bad = st.clifford_paulis[:i] + ((p, -s),) + st.clifford_paulis[i + 1:]
            with pytest.raises(AssertionError):
                _assert_same_state(dataclasses.replace(st, clifford_paulis=bad), exp, 4)


_PORT = re.compile(r"^\s*OUTPUT_QUBITS\s+\S+\s+(.*)$", re.M)


def _old_strict_accepts(text):
    try:
        xtim.RecordFidelity(xtim.compile_twirl_sampler(text, selfcheck=0))
        return True
    except xtim.XtimFidelityRefusal as e:
        if e.code == "RF_OUTPUT_NOT_STRICT":
            return False
        raise


def _check_old_implies_new(text):
    ports = _PORT.findall(text)
    if len(ports) != 1:
        pytest.skip("not exactly one OUTPUT_QUBITS port")
    try:
        old = _old_strict_accepts(text)
    except xtim.XtimTwirlRefusal:
        pytest.skip("outside the sampler's model")
    if old:
        assert xtim.output_is_deterministic(text, [int(t) for t in ports[0].split()]) is True


@pytest.mark.parametrize("name", xtim.list_examples())
def test_old_check_accepts_implies_new_accepts(name):           # A1 cross-check (one way)
    _check_old_implies_new(xtim.load_example(name).text)


_REPO = pathlib.Path(__file__).resolve().parent.parent
_PORT_FILES = sorted(str(f.relative_to(_REPO)) for d in ("tests/data", "examples", "benchmarks", "xtim/_examples")
                     for f in (_REPO / d).rglob("*.stim")
                     if not f.relative_to(_REPO).as_posix().startswith("examples/xdem/")   # the pairs duplicate covered circuits; own test
                     and len(_PORT.findall(f.read_text())) == 1)


def test_in_repo_port_circuits_are_found():                     # final review N3: the sweep is not vacuous
    assert len(_PORT_FILES) >= 26


@pytest.mark.parametrize("path", _PORT_FILES)
def test_old_check_accepts_implies_new_accepts_in_repo(path):   # final review N3: every in-repo single-port circuit
    _check_old_implies_new((_REPO / path).read_text())


def test_new_check_is_more_permissive():                        # I2's example: old refuses, new accepts
    text = "H 0 1\nT 0 1\nM 1\nOUTPUT_QUBITS o 0\n"
    assert _old_strict_accepts(text) is False
    assert xtim.output_is_deterministic(text, [0]) is True


def test_pure_stabilizer_output_matches_stim():
    text = "H 0\nCX 0 1\nS 1\n"
    st = xtim.output_state(text, [0, 1])
    assert st.rotations == () and st.clifford_paulis == ()
    sim = stim.TableauSimulator(seed=1)
    sim.do(stim.Circuit(text))
    for p, s in st.pauli_stabilizers:
        assert sim.peek_observable_expectation(O.ps_of(p, 2)) == s


# Task 2 review I1's confirmed example: DET with one rotation whose localized axis makes "circuit + undo" leave xtim's
# class (non-commuting magic), so the oracle checks the clifford_paulis by the Heisenberg expansion.
I1_UNDO_REFUSED = ("S 0\nS 4\nCX 3 4\nS_DAG 3\nH 2\nH_YZ 1\nCS_DAG 4 1\nT_DAG 1\nT 3\nCS 3 2\nH_YZ 4\nCZ 3 1\nS 3\n"
                   "CZ 4 0\nH 2\nCZ 3 4\nCX 2 4\nS_DAG 2\n")


def test_undo_refused_example_uses_the_fallback():
    st = xtim.output_state(I1_UNDO_REFUSED, [0, 1, 2, 3])
    assert len(st.rotations) == 1
    assert not O.assert_measures(O.base_text(I1_UNDO_REFUSED) + O.rot_text(list(st.rotations), undo=True),
                                 list(st.clifford_paulis))      # xtim refuses the undo circuit
    assert O.assert_state(I1_UNDO_REFUSED, st) == "heisenberg"


@pytest.mark.parametrize("seed", range(6))
def test_heisenberg_route_agrees_and_catches_a_wrong_sign(seed):   # the fallback is itself checked
    text, _ = known_circuit(seed)
    st = xtim.output_state(text, list(range(4)))
    rot, cp = list(st.rotations), list(st.clifford_paulis)
    assert O.assert_measures(O.base_text(text) + O.rot_text(rot, undo=True), cp)    # the undo route accepts
    O.assert_heisenberg(O.base_text(text), rot, cp, 4)                               # and so does the expansion
    for i in range(len(cp)):
        bad = cp[:i] + [(cp[i][0], -cp[i][1])] + cp[i + 1:]
        with pytest.raises(AssertionError):
            O.assert_heisenberg(O.base_text(text), rot, bad, 4)
    if rot:                                                                          # a wrong rotation sign too
        with pytest.raises(AssertionError):
            O.assert_heisenberg(O.base_text(text), [(rot[0][0], -rot[0][1])] + rot[1:], cp, 4)
