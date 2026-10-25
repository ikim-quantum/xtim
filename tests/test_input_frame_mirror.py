"""fix/input-frame-mirror — an input_pauli frame DRESSED by the shot's residual.

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

A port frame component F whose anchor image P' = U F U† is a Pauli (a "clean" component) can still
anticommute with an odd-rotation axis b_j of a fault atom: a T…T_DAG mirror the frame crosses twice
leaves P' Pauli while a fault between the two terms puts rot(b_j) into the residual R. The anchor
state is R P'|B⟩ = (R P' R†)·R|B⟩ = P'·Q·R|B⟩ (R Clifford), so the records' law is the frame-free
law XOR ⟨P'·Q, M_c⟩ — the static relabel ⟨P', M_c⟩ is wrong whenever ⟨Q, M_c⟩ ≠ 0 (pre-fix:
silently, exact_shots = 0, no warning).

Oracles:
  * the EXACT engine (engine="exact") with the frame injected as a probability-1 Pauli fault at
    t = 0 (X_ERROR(1) etc.: a fault never moves the noiseless reference, so the detectors stay
    relative to the frame-free reference, exactly the twirl relabel's convention) — statistical,
    on the minimized repro and on mirror circuits whose pre-fix error is large;
  * deterministic self-consistency of the export: framed ⊕ frame-free ⊕ the exported per-shot
    dressing rows must be the SAME (static) row on every shot — byte-exact, sample_barrier with
    the same seed (a frame consumes no randomness).
Mutation: dropping the dressing rows in apply_input_frame turns every test here red.
"""
import io
import contextlib
import random

import numpy as np
import pytest

import xtim

# Minimized repro (n = 3): the port qubit 1 is crossed by a T…T_DAG mirror on the axis Z0Z1
# (CX sandwiches); one certain X fault between the two terms; the Born observable MX 0 reads the
# magic qubit 0. Truth (exact engine, X on the port at t = 0): P(obs) = 0.146; pre-fix twirl: 0.852.
L4 = """RX 0
R 2
T 0
CX 0 1
T 1
CX 0 1
X_ERROR(1) 0
CX 0 1
T_DAG 1
CX 0 1
DEPOLARIZE1(0.001) 2
MX 0
M 2
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-2]
"""


def _quiet(fn, *a, **k):
    with contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def _frame_rows(ports, frame, N):
    pnq = len(ports)
    fr = np.zeros((N, 2 * pnq), np.uint8)
    pre = ""
    for i, q in enumerate(ports):
        P = frame.get(q, "I")
        if P in "XY":
            fr[:, i] = 1
        if P in "ZY":
            fr[:, pnq + i] = 1
        if P != "I":
            pre += f"{P}_ERROR(1) {q}\n"
    return fr, pre


def _exact(body, ports, pre, N, seed):
    c = xtim.Circuit("R " + " ".join(map(str, ports)) + "\n" + pre + body)
    d, o = _quiet(c.compile_detector_sampler(seed=seed, engine="exact").sample, N,
                  separate_observables=True)
    return np.asarray(d, float), np.asarray(o, float)


def _unpack(a, k):
    return np.unpackbits(np.asarray(a), axis=1, bitorder="little")[:, :k].astype(float)


def _maxz(T, X):
    N = T.shape[0]
    zs = []
    for i in range(T.shape[1]):
        zs.append(abs(T[:, i].mean() - X[:, i].mean()) /
                  np.sqrt((T[:, i].var() + X[:, i].var()) / N + 1e-12))
        for j in range(i + 1, T.shape[1]):
            a, b = T[:, i] * T[:, j], X[:, i] * X[:, j]
            zs.append(abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / N + 1e-12))
    return max(zs)


# Magic-weighted mirror family (scratch fuzz: 300 trials sample + 300 barrier, 0 mismatches on the
# fix; 12/300 each mismatched pre-fix). Same generator as the scratch fuzz.
_INV = {"S": "S_DAG", "CX": "CX", "CZ": "CZ"}


def _rdiag(n, d, r):
    g = []
    for _ in range(d):
        k = r.randrange(3)
        if k == 0:
            g.append(("S", [r.randrange(n)]))
        else:
            a, b = r.sample(range(n), 2)
            g.append(("CX" if k == 1 else "CZ", [a, b]))
    return g


def _magic_mirror(seed, n=4):
    r = random.Random(seed)
    ports = sorted(r.sample(range(1, n), r.randrange(1, 3)))
    others = [q for q in range(1, n) if q not in ports]
    p = r.choice([0.02, 0.05, 0.1, 0.2])
    D = _rdiag(n, r.randrange(2, 7), r)
    E = _rdiag(n, r.randrange(0, 5), r)
    Q = r.sample(range(n), r.randrange(1, n + 1))
    dag = lambda gs: [(_INV[a], q) for a, q in reversed(gs)]
    L = ["RX 0"] + (["R " + " ".join(map(str, others))] if others else []) + ["T 0"]
    L += [f"{a} " + " ".join(map(str, q)) for a, q in D]
    L += [f"T {q}" for q in Q]
    noise = r.choice(["DEPOLARIZE1", "X_ERROR", "Y_ERROR"])
    L.append(f"{noise}({p}) " + " ".join(map(str, range(n))))
    L += [f"{a} " + " ".join(map(str, q)) for a, q in E]
    if r.random() < 0.5:
        L.append(f"DEPOLARIZE2({p}) 0 {r.randrange(1, n)}")
    L += [f"{a} " + " ".join(map(str, q)) for a, q in dag(E)]
    L += [f"T_DAG {q}" for q in Q]
    L += [f"{a} " + " ".join(map(str, q)) for a, q in dag(D)]
    L.append(f"DEPOLARIZE1({p / 4}) " + " ".join(map(str, range(1, n))))
    L += ["MX 0", "M " + " ".join(map(str, range(1, n)))]
    L += [f"DETECTOR rec[-{i + 1}]" for i in range(n - 1)]
    L.append("OBSERVABLE_INCLUDE(0) rec[-%d]" % n)
    frames = [{q: r.choice("XYZ") for q in ports} for _ in range(2)]
    return "\n".join(L) + "\n", ports, frames


# Adversary A R4-1: the plan key's product-wire strip drops S/CZ legs on Z-eigenstate wires; a
# dressed frame with X on such a wire flips the eigenstate, so a dropped CZ(q, r) acts as Z_r. The
# correction here is computed per shot from the UNSTRIPPED fault atoms, so it is exact (pre-fix
# and under any per-plan rule on the stripped key: D2 = 0.0001; truth 0.5).
STRIP = """RX 1 2
CCZ 0 1 2
X_ERROR(0.5) 1
CCZ 0 1 2
DEPOLARIZE1(0.0001) 1 2
MX 1 2
DETECTOR rec[-2]
DETECTOR rec[-1]
"""


# The SOFT-poison class (non-Pauli anchor image support-disjoint from every record) was ALSO
# silently wrong: T 0 makes X_0's image non-Pauli (SOFT, "exact no-op" pre-fix), yet the CCZ
# mirror's fault leg CZ(0,2) turns its X-part into Z_2 on the faulty shots. Truth D2 = 0.5; pre-fix 0.
SOFT = "T 0\n" + STRIP


# test_port_record_route_exports_frame_dressing: retired at J-b1 F (port Consume(frame_dressing=) removed, plan §4.1;
# the family law folds a supplied frame into the record's P_rec — tests/test_ja_frames.py, test_jb1_port_reference.py).


# ── J-b2 J1b: the PRODUCTION legs (plan r2.1 J1b). The same law against the same exact-engine oracle (the frame as a
# probability-1 Pauli fault at t = 0) on the PUBLIC sampler, which folds a supplied frame into the record's P_rec
# (the frames class) instead of the legacy static relabel + dressing export. The legacy export's own surface
# (frame_dressing_report, input_frame_dressing / advance, the opt-in) retires with it at J2 (jb2 mapping).
from _public import compile_public_twirl_sampler  # noqa: E402


def _twirl_public(body, ports):
    return _quiet(compile_public_twirl_sampler, "INPUT_QUBITS in " + " ".join(map(str, ports)) + "\n" + body)


def _run_public(s, route, N, seed, fr):
    inner = getattr(s, "_inner", s)
    if route == "sample":
        d, o = inner.sample(N, seed=seed, input_pauli=fr)[:2]
    else:
        b = inner.sample_barrier(N, seed, fr)
        d, o = b.dets(), b.obs()
    return d, o


@pytest.mark.parametrize("route", ["sample", "barrier"])
def test_minimized_mirror_repro_matches_exact_public(route):
    N = 40000
    s = _twirl_public(L4, [1])
    fr, pre = _frame_rows([1], {1: "X"}, N)
    d, o = _run_public(s, route, N, 3, fr)
    ed, eo = _exact(L4, [1], pre, N, 99)
    assert abs(_unpack(o, 1)[:, 0].mean() - 0.146) < 0.02          # pre-fix: 0.852
    assert _maxz(np.concatenate([_unpack(d, 1), _unpack(o, 1)], 1), np.concatenate([ed, eo], 1)) < 6


@pytest.mark.parametrize("seed", [30, 49])
@pytest.mark.parametrize("route", ["sample", "barrier"])
def test_magic_mirror_family_matches_exact_public(seed, route):
    N = 20000
    body, ports, frames = _magic_mirror(seed)
    s = _twirl_public(body, ports)
    for fi, frame in enumerate(frames):
        fr, pre = _frame_rows(ports, frame, N)
        d, o = _run_public(s, route, N, 11 + fi, fr)
        ed, eo = _exact(body, ports, pre, N, 99 + fi)
        T = np.concatenate([_unpack(d, 3), _unpack(o, 1)], 1)
        X = np.concatenate([ed[:, :3], eo[:, :1]], 1)
        assert _maxz(T, X) < 6, (seed, frame)


@pytest.mark.parametrize("route", ["sample", "barrier"])
def test_product_wire_strip_repro_matches_exact_public(route):
    N = 20000
    s = _twirl_public(STRIP, [0])
    fr, pre = _frame_rows([0], {0: "X"}, N)
    d, _ = _run_public(s, route, N, 1, fr)
    ed, _ = _exact(STRIP, [0], pre, N, 7)
    T = _unpack(d, 2)
    assert abs(T[:, 1].mean() - 0.5) < 0.02
    assert _maxz(T, ed[:, :2]) < 6


@pytest.mark.parametrize("route", ["sample", "barrier"])
@pytest.mark.parametrize("P", ["X", "Y", "Z"])
def test_soft_poison_dressing_matches_exact_public(route, P):
    N = 20000
    s = _twirl_public(SOFT, [0])
    fr, pre = _frame_rows([0], {0: P}, N)
    d, _ = _run_public(s, route, N, 1, fr)
    ed, _ = _exact(SOFT, [0], pre, N, 7)
    assert _maxz(_unpack(d, 2), ed[:, :2]) < 6


def test_frame_free_stream_unchanged_by_a_zero_frame_public():
    """The frame-free stream is unchanged by an all-zero supplied frame (a frame consumes no randomness)."""
    s = _twirl_public(STRIP, [0])
    inner = getattr(s, "_inner", s)
    a = inner.sample(4000, seed=5)
    b = inner.sample(4000, seed=5, input_pauli=np.zeros((4000, 2), np.uint8))
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
