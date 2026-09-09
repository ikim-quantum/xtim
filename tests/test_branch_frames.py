"""FramedSuperposition.branch_frames() (3.1.2) — the state exported as
|psi> = sum_i c_i D_i |ref> (docs/branch_frames.md).

Oracles (no engine call in the checks):
  * DENSE — small circuits with a T gate and injected COHERENT Pauli faults: the
    statevector rebuilt from the export equals the dense simulation of the same
    circuit up to global phase.
  * CONVENTION — xtim.frames.expectation on the full branch set equals the
    engine's own pauli_expectation for random Paulis, Y-type included (the
    export carries the phases; pauli_expectation_xz drops them).
  * BRANCH RESTRICTION — on a coherent state the restricted expectation on ONE
    branch is definite where the full one vanishes (the consumer's use).
  * DETERMINISM — identical arrays across materializations and batches.
"""
import numpy as np
import pytest

from xtim import _xtim, frames

PY = None  # noqa: N816 (tests run in-process)


def _sampler(text: str):
    from xtim import compile_twirl_sampler
    return compile_twirl_sampler(text)


def _frames_and_map(text: str, shot: int = 0, seed: int = 0, shots: int = 1):
    s = _sampler(text)
    buf = s.sample_barrier(shots, seed)
    fr = buf.branch_frames(shot)
    out = list(s.output_wires())          # circuit OUTPUT_QUBITS order -> state wire index
    return fr, out, buf


# ── dense reference simulator (qubit 0 = most significant, matching frames.dense_state) ──
def _dense_sim(n: int, ops):
    """ops: list of (gate, qubits). Gates: H,S,SDG,T,X,Y,Z,CX,CZ. Returns the statevector."""
    dim = 1 << n
    psi = np.zeros(dim, complex); psi[0] = 1.0
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    S = np.diag([1, 1j]); SDG = np.diag([1, -1j]); T = np.diag([1, np.exp(1j * np.pi / 4)])
    TDG = np.diag([1, np.exp(-1j * np.pi / 4)])
    X = np.array([[0, 1], [1, 0]], complex); Z = np.diag([1, -1]).astype(complex); Y = 1j * X @ Z
    one = {"H": H, "S": S, "SDG": SDG, "T": T, "TDG": TDG, "X": X, "Y": Y, "Z": Z}

    def apply1(psi, g, q):
        v = psi.reshape([2] * n)
        v = np.moveaxis(v, q, 0)
        v = np.tensordot(g, v, axes=([1], [0]))
        v = np.moveaxis(v, 0, q)
        return v.reshape(dim)

    def apply_cx(psi, c, t):
        v = psi.reshape([2] * n).copy()
        idx1 = [slice(None)] * n; idx1[c] = 1
        sub = v[tuple(idx1)]                 # control = 1 block, axes shifted
        tt = t - 1 if t > c else t
        sub = np.flip(sub, axis=tt)
        v[tuple(idx1)] = sub
        return v.reshape(dim)

    def apply_cz(psi, c, t):
        v = psi.reshape([2] * n).copy()
        idx = [slice(None)] * n; idx[c] = 1; idx[t] = 1
        v[tuple(idx)] *= -1
        return v.reshape(dim)

    for g, qs in ops:
        if g == "CX":
            psi = apply_cx(psi, *qs)
        elif g == "CZ":
            psi = apply_cz(psi, *qs)
        else:
            psi = apply1(psi, one[g], qs[0])
    return psi


def _overlap_mod(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


# Circuits: every data qubit is an OUTPUT so output_wires() gives the full q -> wire map.
# The engine's twirl sampler wants at least one record; a deterministic ancilla M provides it.
def _text(lines, n_data):
    body = "\n".join(lines)
    anc = n_data
    return (f"{body}\nR {anc}\nM {anc}\nDECISION(0) rec[-1]\n"
            f"OUTPUT_QUBITS out {' '.join(str(q) for q in range(n_data))}\n")


CASES = {
    # |T>-like magic state on one qubit + a coherent X fault BEFORE the T: X then T = superposition
    "t_after_x": (["H 0", "X_ERROR(1) 0", "T 0"], 1,
                  [("H", [0]), ("X", [0]), ("T", [0])]),
    # two-qubit: entangle, T on one, Y fault after -> coherent in the frame
    "bell_t_y": (["H 0", "CX 0 1", "T 1", "Y_ERROR(1) 1"], 2,
                 [("H", [0]), ("CX", [0, 1]), ("T", [1]), ("Y", [1])]),
    # three-qubit repetition-like with T and two faults
    "rep3_t": (["H 0", "CX 0 1", "CX 0 2", "T 0", "X_ERROR(1) 1", "S 2", "Z_ERROR(1) 0"], 3,
               [("H", [0]), ("CX", [0, 1]), ("CX", [0, 2]), ("T", [0]), ("X", [1]),
                ("S", [2]), ("Z", [0])]),
    # stabilizer only (chi = 1): a single branch
    "ghz": (["H 0", "CX 0 1", "CX 1 2", "Z_ERROR(1) 2"], 3,
            [("H", [0]), ("CX", [0, 1]), ("CX", [1, 2]), ("Z", [2])]),
}


def _remap_dense(psi_circ, n_data, wire_of, n_state):
    """Embed the dense circuit state (qubits 0..n_data-1) into the state's wire space:
    the state has n_state wires; circuit qubit q sits at wire wire_of[q]; other wires (the
    measured ancilla, deterministic |0>) are |0>."""
    dim = 1 << n_state
    out = np.zeros(dim, complex)
    for idx in range(1 << n_data):
        bits = [(idx >> (n_data - 1 - q)) & 1 for q in range(n_data)]
        widx = 0
        for q in range(n_data):
            if bits[q]:
                widx |= 1 << (n_state - 1 - wire_of[q])
        out[widx] = psi_circ[idx]
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_dense_reconstruction_matches_statevector(name):
    lines, n_data, ops = CASES[name]
    fr, wires, _ = _frames_and_map(_text(lines, n_data))
    n_state = int(fr["n"])
    psi_rec = frames.dense_state(fr)
    psi_circ = _dense_sim(n_data, ops)
    # ancilla wires (not in `wires`) must be |0>: project the reconstruction there implicitly
    psi_ref = _remap_dense(psi_circ, n_data, wires, n_state)
    ov = _overlap_mod(psi_rec, psi_ref)
    assert ov == pytest.approx(1.0, abs=1e-9), (name, ov)
    # the export's own norm is 1
    assert np.sum(np.abs(np.asarray(fr["coeff"])) ** 2) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", sorted(CASES))
def test_convention_expectation_matches_engine_incl_y(name):
    lines, n_data, ops = CASES[name]
    text = _text(lines, n_data)
    s = _sampler(text)
    buf = s.sample_barrier(1, 0)
    st = buf.materialize(0)
    fr = buf.branch_frames(0)
    n = int(fr["n"])
    rng = np.random.default_rng(7)
    checked_y = 0
    for _ in range(40):
        # a random HERMITIAN Pauli: each site X, Y, Z or I
        site = rng.integers(0, 4, size=n)          # 0:I 1:X 2:Y 3:Z
        xs = [q for q in range(n) if site[q] == 1]
        ys = [q for q in range(n) if site[q] == 2]
        zs = [q for q in range(n) if site[q] == 3]
        px, pz, pp = frames.pauli_from_sites(n, xs, zs, ys)
        got = frames.expectation(fr, px, pz, pp)
        # engine oracle for a Hermitian operator with Y sites: the engine's xz read drops the
        # i-phase (it returns Re<X^x Z^z> = 0 for an odd Y count), so — exactly as adaptq's
        # _check_expectation_inplace does — rotate each Y site to Z (Sdg then H) on a copy,
        # read <X^xs Z^(zs ∪ ys)>, and compare.  The frames route needs no rotation.
        st2 = buf.materialize(0)
        for q in ys:
            st2.apply_clifford(2, q, 0); st2.apply_clifford(0, q, 0)
        want = st2.pauli_expectation_xz(sorted(xs), sorted(zs + ys))
        assert abs(got.imag) < 1e-12, (name, got)
        assert got.real == pytest.approx(want, abs=1e-12), (name, xs, zs, ys, got, want)
        checked_y += bool(ys)
    assert checked_y > 0
    # the frames route needs NO rotation and the phase-dropping xz read is NOT the oracle for Y:
    if n >= 1:
        px, pz, pp = frames.pauli_from_sites(n, ys=[0])
        assert abs(frames.expectation(fr, px, pz, pp)) <= 1.0 + 1e-12
    _ = st


def test_branch_restriction_is_definite_where_the_full_state_is_coherent():
    # t_after_x: (|0>+|1>)/√2 --X--> same, --T--> (|0> + e^{iπ/4}|1>)/√2 : <X> = cos(π/4)=1/√2.
    # Use bell_t_y instead: after T on qubit 1 and a Y fault, the state is a coherent
    # superposition of two Pauli frames on the engine's free generator; project each.
    lines, n_data, _ = CASES["bell_t_y"]
    fr, wires, _ = _frames_and_map(_text(lines, n_data))
    chi = int(np.asarray(fr["coeff"]).shape[0])
    assert chi >= 2, "fixture must be a genuine superposition"
    n = int(fr["n"])
    # For each free generator g (a signed stabilizer of |ref> that the branches flip), the
    # full-state expectation is strictly between -1 and 1 while each single branch reads ±1.
    found = False
    for d in range(len(fr["free"])):
        a = int(fr["free"][d])
        px, pz = np.asarray(fr["stab_x"][a]), np.asarray(fr["stab_z"][a])
        pp = int(fr["stab_phase"][a])
        full = frames.expectation(fr, px, pz, pp)
        signs = [frames.branch_sign(fr, i, px, pz, pp) for i in range(chi)]
        assert all(abs(abs(sg) - 1.0) < 1e-12 for sg in signs), signs
        if abs(abs(full) - 1.0) > 1e-9:
            found = True
            # restricted expectation on one branch is that branch's definite sign
            for i in range(chi):
                mask = np.zeros(chi, bool); mask[i] = True
                r = frames.expectation(fr, px, pz, pp, mask=mask)
                assert r == pytest.approx(signs[i], abs=1e-12)
    assert found, "no free generator with an indefinite full expectation — fixture too weak"


def test_determinism_across_materializations_and_batches():
    lines, n_data, _ = CASES["rep3_t"]
    text = _text(lines, n_data)
    s = _sampler(text)
    b1 = s.sample_barrier(3, 11)
    f1 = b1.branch_frames(1)
    f2 = b1.branch_frames(1)
    s2 = _sampler(text)
    b2 = s2.sample_barrier(3, 11)
    f3 = b2.branch_frames(1)
    for k in f1:
        a1 = np.asarray(f1[k]); a2 = np.asarray(f2[k]); a3 = np.asarray(f3[k])
        assert np.array_equal(a1, a2), k
        assert np.array_equal(a1, a3), k
    # and the state-object route is the same export
    f4 = b1.materialize(1).branch_frames()
    for k in f1:
        assert np.array_equal(np.asarray(f1[k]), np.asarray(f4[k])), k


def test_binding_exists_and_shapes():
    lines, n_data, _ = CASES["ghz"]
    fr, wires, buf = _frames_and_map(_text(lines, n_data))
    n = int(fr["n"]); k = len(fr["free"]); chi = len(fr["coeff"])
    assert hasattr(_xtim.FramedSuperposition, "branch_frames")
    assert hasattr(_xtim.BarrierBuffer, "branch_frames")
    assert np.asarray(fr["stab_x"]).shape == (n, n)
    assert np.asarray(fr["sigma"]).shape == (chi, k)
    assert np.asarray(fr["branch_x"]).shape == (chi, n)
    assert chi == 1 and k == 0                # a stabilizer state: one branch, no free rows
    assert int(fr["branch_phase"][0]) == 0 and not np.asarray(fr["branch_x"]).any()


def _dense_expect(psi, n, px, pz, pp):
    X = np.array([[0, 1], [1, 0]], complex); Z = np.diag([1, -1]).astype(complex); I2 = np.eye(2, dtype=complex)
    M = np.array([[1.0 + 0j]])
    for q in range(n):
        g = I2
        if px[q] and pz[q]: g = X @ Z
        elif px[q]: g = X
        elif pz[q]: g = Z
        M = np.kron(M, g)
    M = (1j ** int(pp)) * M
    return np.vdot(psi, M @ psi) / np.vdot(psi, psi)


def test_projected_expectation_matches_dense_sector_projection():
    """The consumer's use: a coherent state is a superposition over an operator g
    that is NOT its stabilizer (an X-type check on a qubit that took a coherent
    fault); the Born sectors (I ± g)/2 |psi> are the classifier's branches and
    their charge is the projected expectation.  Oracle: the dense statevector
    rebuilt from the export, projected explicitly."""
    lines, n_data, _ = CASES["bell_t_y"]
    fr, wires, _ = _frames_and_map(_text(lines, n_data))
    n = int(fr["n"])
    psi = frames.dense_state(fr)
    # candidate sector operators: single-qubit X on each data wire (Hermitian)
    found = 0
    for w in wires:
        gx, gz, gp = frames.pauli_from_sites(n, xs=[w])
        full = frames.expectation(fr, gx, gz, gp)
        if abs(abs(full) - 1.0) < 1e-9:
            continue                                   # g is a stabilizer here: no split
        for s in (+1, -1):
            # P must commute with g: pick Z on the OTHER data wire
            other = [v for v in wires if v != w][0]
            px, pz, pp = frames.pauli_from_sites(n, zs=[other])
            val, weight = frames.projected_expectation(fr, px, pz, pp, sectors=[((gx, gz, gp), s)])
            # dense: project psi onto (I + s g)/2
            G = _dense_expect  # reuse builder below
            X = np.array([[0, 1], [1, 0]], complex); I2 = np.eye(2, dtype=complex)
            Gm = np.array([[1.0 + 0j]])
            for q in range(n):
                Gm = np.kron(Gm, X if q == w else I2)
            proj = (np.eye(1 << n) + s * Gm) / 2
            phi = proj @ psi
            wd = np.vdot(phi, phi).real
            assert weight.real == pytest.approx(wd, abs=1e-12)
            if wd > 1e-9:
                want = _dense_expect(phi, n, px, pz, pp)
                assert val == pytest.approx(want, abs=1e-12), (w, s, val, want)
                found += 1
    assert found > 0, "fixture has no non-stabilizer X to split on"


# ─────────────────────────────────────────────────────────────────────────────
# 3.1.2 "frames at the port": BarrierBuffer.frames() — the stored per-shot prefix P
# and per-plan S-layer mask reproduce materialize(i) EXACTLY (oracle 1), determinism
# (oracle 3), the kappa > 0 refusal (oracle 4), and the sector-charge algebra against
# projected_expectation on the materialized export (oracle 2, dense fixture).
# ─────────────────────────────────────────────────────────────────────────────

def _bare_copy(text):
    from xtim import _xtim as _x
    return _x._bare_state_of(text)


def _apply_frame(st, fr, i):
    n = int(fr["n"]); pnw = int(fr["pnw"])
    px, pz = frames.unpack_prefix(np.asarray(fr["prefix_words"])[i:i + 1], n)
    pid = int(fr["plan_id"][i])
    a = np.asarray(fr["a"])[pid]
    cz = np.asarray(fr["cz"][pid]).reshape(-1, 2)
    for q in range(n):
        if a[q]: st.apply_clifford(1, q, 0)          # S
    for f, t in cz:
        st.apply_clifford(7, int(f), int(t))         # CZ
    for q in range(n):
        if px[0, q]: st.apply_clifford(3, q, 0)      # X
        if pz[0, q]: st.apply_clifford(5, q, 0)      # Z
    return st


@pytest.mark.parametrize("name", sorted(CASES))
def test_frames_export_reproduces_materialize(name):
    """Oracle 1: for EVERY retained shot, the frame-derived state P·S^a·CZ·|bare> equals
    materialize(i) up to global phase (|overlap| = 1 to 1e-9) — the export carries the whole
    state for kappa = 0 plans, so the replay is redundant."""
    lines, n_data, _ = CASES[name]
    text = _text(lines, n_data)
    s = _sampler(text)
    shots = 12
    buf = s.sample_barrier(shots, 5)
    fr = buf.frames(allow_kappa=True)
    assert fr["shots"] == shots
    assert np.asarray(fr["prefix_words"]).shape == (shots, 2 * int(fr["pnw"]))
    assert np.asarray(fr["a"]).shape == (int(fr["n_plans"]) + 1, int(fr["n"]))
    assert not np.asarray(fr["a"])[0].any() and int(fr["kappa"][0]) == 0
    for i in range(shots):
        if int(fr["kappa"][int(fr["plan_id"][i])]) > 0:
            continue                                  # not a frame coset (see the refusal test)
        st = _apply_frame(_bare_copy(text), fr, i)
        mat = buf.materialize(i)
        assert st.approx_equal(mat, 1e-9), (name, i)


def test_frames_determinism_and_identity_rows():
    lines, n_data, _ = CASES["rep3_t"]
    text = _text(lines, n_data)
    s = _sampler(text)
    b1 = s.sample_barrier(6, 21); b2 = _sampler(text).sample_barrier(6, 21)
    f1 = b1.frames(allow_kappa=True); f2 = b2.frames(allow_kappa=True)
    for k in ("prefix_words", "plan_id", "a", "r", "kappa"):
        assert np.array_equal(np.asarray(f1[k]), np.asarray(f2[k])), k
    # an identity-plan shot (plan_id 0) has an all-zero prefix row
    pid = np.asarray(f1["plan_id"])
    if (pid == 0).any():
        assert not np.asarray(f1["prefix_words"])[pid == 0].any()
    # every non-identity plan is referenced by at least one shot
    assert set(np.unique(pid)) - {0} == set(range(1, int(f1["n_plans"]) + 1))


@pytest.mark.parametrize("name", sorted(CASES))
def test_frames_plan_sector_table_is_a_frame_partition(name):
    """Oracle 2 (in-suite half): for every retained shot the plan's sector table
    (plan_frame_cosets on the bare reference) is a probability partition into FRAME COPIES
    of the reference, and a shot with no S-layer has exactly one sector (the identity
    frame).  The charge half of oracle 2 — sector charges against projected_expectation
    on the materialized export and against today's expectation read — needs split checks
    that are CERTIFIED generators of a transversal-T code, which no dense fixture has; it
    is run on the downstream u2 unit (see the report: 0 mismatches on 1 200 definite
    groups and 900 coherent sector rows, weights equal to the Born weights)."""
    lines, n_data, _ = CASES[name]
    text = _text(lines, n_data)
    s = _sampler(text)
    buf = s.sample_barrier(6, 3)
    fr = buf.frames(allow_kappa=True)
    ref = _bare_copy(text).branch_frames()
    seen_a = False
    for i in range(6):
        pid = int(fr["plan_id"][i])
        if int(fr["kappa"][pid]) > 0:
            continue
        a = np.asarray(fr["a"])[pid]
        cosets = frames.plan_frame_cosets(ref, a)
        assert cosets
        assert abs(sum(c["weight"] for c in cosets) - 1.0) < 1e-12
        assert all(c["frame_copy"] for c in cosets), (name, i, cosets)
        assert len(cosets) <= 1 << int(a.sum())
        if not a.any():
            assert len(cosets) == 1 and not cosets[0]["fx"].any() and not cosets[0]["fz"].any()
        else:
            seen_a = True
    if name == "t_after_x":
        assert seen_a, "the X-before-T fixture must produce an S-layer plan"


def test_frames_refuses_kappa_plans_loudly_inspection_only():
    """Oracle 4: with kappa > 0 plans present, frames() raises naming the plan ids unless
    allow_kappa=True.  The fixture is searched for a kappa > 0 plan; if none of the dense
    circuits produces one the refusal is exercised through the identity-prefix parser on
    a synthetic plan key."""
    from xtim import _xtim as _x
    found = None
    for name, (lines, n_data, _) in sorted(CASES.items()):
        text = _text(lines, n_data)
        s = _sampler(text)
        buf = s.sample_barrier(16, 9)
        fr = buf.frames(allow_kappa=True)
        if (np.asarray(fr["kappa"]) > 0).any():
            found = (text, buf); break
    if found is None:
        pytest.skip("no dense fixture produces a kappa > 0 / fallback plan: the C++ refusal branch is "
                    "verified by inspection only; the consumer twin is exercised in "
                    "test_frames_exports_fallback_and_twin_refuses_it")
    text, buf = found
    with pytest.raises(ValueError, match="kappa > 0"):
        buf.frames()
    with pytest.raises(ValueError, match="plan ids"):
        buf.frames(allow_kappa=False)


def test_require_kappa_zero_refuses_synthetic_kappa_plan():
    """Oracle 4 (consumer-side twin of the export's refusal): a frames() dict carrying a
    kappa > 0 plan is refused by name.  No small dense fixture produces a kappa > 0 plan
    (an S-layer on a normaliser-not-group direction of a NON-trivial certified group), so
    the C++ refusal in BarrierBuffer.frames() is exercised only by inspection here and on
    the downstream unit's report; this pins the rule's text and the id naming."""
    lines, n_data, _ = CASES["ghz"]
    fr = _sampler(_text(lines, n_data)).sample_barrier(2, 1).frames()
    frames.require_kappa_zero(fr)                       # kappa = 0 everywhere: passes
    bad = dict(fr); bad["kappa"] = np.array([0, 2], np.int32)
    with pytest.raises(ValueError, match="kappa > 0.*plan ids 1"):
        frames.require_kappa_zero(bad)


# ─────────────────────────────────────────────────────────────────────────────
# Review MAJOR-1 (frames at the port): in-suite MULTI-SECTOR charge oracle.
#
# Every dense fixture above is a single-sector plan (the S-layer's Z's are stabilizers of
# the frame up to a logical, so the Z^v cosets collapse).  A fixture whose S-layer Z lands
# INSIDE an X-type check of the reference splits into genuine Born sectors:
#   * GHZ+T: H 0; CX 0 1; CX 0 2 (XXX, ZZI, IZZ stabilizers), X_ERROR(1) 0 BEFORE T 0 — the
#     residual normal form puts an S on qubit 0 (a = {0}) whose Z_0 anticommutes with XXX,
#     the free (indefinite) generator of the |T>-like reference -> 2 sectors (m = 1);
#   * two magic qubits (H 0; CX 0 1; T 0; X fault) x (H 2; CX 2 3; T 2; X fault) -> two anti
#     checks XX(01), XX(23) -> 4 sectors (m = 2).
# For EVERY retained shot and EVERY sector: (a) the sector weight == the Born weight <Π>
# read on materialize(i); (b) frame_charge via sector_coset == the sign of
# projected_expectation on materialize(i) for O in {X_L, Y_L} (the free generator and its
# Y-partner, the recipe operators of a magic reference); (c) every sector is a frame copy;
# (d) the frame-derived state still equals materialize(i) (dense cross-check n <= 4).
# ─────────────────────────────────────────────────────────────────────────────

MULTI_SECTOR_CASES = {
    # GHZ (XXX, ZZI, IZZ) with T 0; X fault; T_DAG 0: the reference is the GHZ STABILIZER
    # state again (T·T† = I), while the fault is conjugated to T† X T ∝ X·S† on qubit 0 -> the
    # residual normal form carries an S-layer on qubit 0 whose Z_0 anticommutes with the
    # DEFINITE X-type check XXX -> 2 Born sectors of XXX (m = 1).  This is the dense analogue
    # of an idle fault between transversal T's on a code with an X-type stabilizer through
    # the fault site (the u2 unit's coherent groups).
    "ghz_t_x_tdag": (["H 0", "CX 0 1", "CX 0 2", "T 0", "X_ERROR(1) 0", "T_DAG 0"], 3,
                     [("H", [0]), ("CX", [0, 1]), ("CX", [0, 2]), ("T", [0]), ("X", [0]),
                      ("TDG", [0])]),
    # two such pairs -> two anti checks XX(01), XX(23) -> 4 sectors (m = 2)
    "two_pairs_m2": (["H 0", "CX 0 1", "H 2", "CX 2 3", "T 0", "T 2", "X_ERROR(1) 0",
                      "X_ERROR(1) 2", "T_DAG 0", "T_DAG 2"], 4,
                     [("H", [0]), ("CX", [0, 1]), ("H", [2]), ("CX", [2, 3]), ("T", [0]),
                      ("T", [2]), ("X", [0]), ("X", [2]), ("TDG", [0]), ("TDG", [2])]),
}


def _recipe_ops(fr_ref, anti_checks):
    """The operators whose sector charge the consumer reads: every Hermitian Pauli in the
    group generated by the reference frame's rows (definite generators, free generators and
    free destabilizers — the logical Pauli group over the free rows) that commutes with every
    split check and has a NON-ZERO reference expectation (a definite charge on every frame
    copy).  Enumerated over single rows and pairwise products (enough for k <= 2 free rows)."""
    n = int(fr_ref["n"])
    rows = []
    for a in range(n):
        rows.append((np.asarray(fr_ref["stab_x"][a], np.uint8), np.asarray(fr_ref["stab_z"][a], np.uint8),
                     int(fr_ref["stab_phase"][a])))
    for f in fr_ref["free"]:
        f = int(f)
        rows.append((np.asarray(fr_ref["destab_x"][f], np.uint8), np.asarray(fr_ref["destab_z"][f], np.uint8),
                     int(fr_ref["destab_phase"][f])))
    cands = list(rows)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            cands.append(frames._mul(*rows[i], *rows[j]))
    ops = []; seen = set()
    for (ox, oz, op) in cands:
        # make it Hermitian: i^op X^ox Z^oz is Hermitian iff op == |ox ∧ oz| (mod 2) parity match
        herm = (int(np.count_nonzero(ox & oz)) - int(op)) % 2 == 0
        if not herm:
            op = (op + 1) & 3
        key = (ox.tobytes(), oz.tobytes())
        if key in seen or not (ox.any() or oz.any()):
            continue
        if any(frames._anticommute(ox, oz, gx, gz) for gx, gz, _gp in anti_checks):
            continue
        v = frames.expectation(fr_ref, ox, oz, op)
        if abs(v.imag) > 1e-9:                    # wrong Hermitian phase branch: rotate by i
            op = (op + 2) & 3; v = frames.expectation(fr_ref, ox, oz, op)
        if abs(v.real) < 1e-9:
            continue
        seen.add(key); ops.append(("O%d" % len(ops), (ox, oz, op)))
    return ops


@pytest.mark.parametrize("name", sorted(MULTI_SECTOR_CASES))
def test_multi_sector_charge_oracle(name):
    lines, n_data, dense_ops = MULTI_SECTOR_CASES[name]
    text = _text(lines, n_data)
    s = _sampler(text)
    shots = 10
    buf = s.sample_barrier(shots, 7)
    fr = buf.frames()                                   # kappa = 0 everywhere: no refusal
    frames.require_kappa_zero(fr)
    n = int(fr["n"])
    ref_st = _bare_copy(text)
    fr_ref = ref_st.branch_frames()
    free = [int(f) for f in fr_ref["free"]]
    checks_all = [(np.asarray(fr_ref["stab_x"][a], np.uint8), np.asarray(fr_ref["stab_z"][a], np.uint8),
                   int(fr_ref["stab_phase"][a])) for a in range(n) if a not in set(free)]
    ops = None; ref_sign = None
    PX, PZ = frames.unpack_prefix(np.asarray(fr["prefix_words"]), n)
    cache = {}
    multi = 0; rows = 0
    for i in range(shots):
        pid = int(fr["plan_id"][i])
        a = np.asarray(fr["a"])[pid]
        cos = frames.plan_frame_cosets_cached(fr_ref, a, cache)
        assert cos and abs(sum(c["weight"] for c in cos) - 1.0) < 1e-12
        assert all(c["frame_copy"] for c in cos), (name, i)
        # (d) the frame-derived state equals the replay
        assert _apply_frame(_bare_copy(text), fr, i).approx_equal(buf.materialize(i), 1e-9), (name, i)
        if len(cos) < 2:
            continue
        multi += 1
        ex = buf.branch_frames(i)
        gvals = [frames.expectation(ex, *c).real for c in checks_all]
        anti = [k for k, v in enumerate(gvals) if abs(abs(v) - 1.0) > 1e-9]
        assert len(anti) >= 1, (name, i, gvals)
        checks = [(checks_all[k][0], checks_all[k][1]) for k in anti]
        ref_signs = [int(np.sign(frames.branch_sign(fr_ref, 0, *checks_all[k]).real)) for k in anti]
        if ops is None:
            ops = _recipe_ops(fr_ref, [checks_all[k] for k in anti])
            assert len(ops) >= 2, "fixture must expose at least two readable recipe operators"
            ref_sign = {nm: int(np.sign(frames.expectation(fr_ref, *O).real)) for nm, O in ops}
        seen = set()
        for signs_mask in range(1 << len(anti)):
            signs = [(-1 if (signs_mask >> j) & 1 else 1) * ref_signs[j] for j in range(len(anti))]
            sec = [((checks_all[k][0], checks_all[k][1], checks_all[k][2]), int(sg))
                   for k, sg in zip(anti, signs)]
            try:
                _val, w = frames.projected_expectation(ex, np.zeros(n, np.uint8), np.zeros(n, np.uint8), 0, sectors=sec)
            except ValueError:
                continue                                 # empty sector (<Π> = 0)
            c = frames.sector_coset(cos, PX[i], PZ[i], checks, signs, ref_signs)
            seen.add(id(c))
            # (a) weight == Born weight
            assert abs(c["weight"] - w.real) < 1e-9, (name, i, signs, c["weight"], w)
            # (b) every recipe operator's charge == the sign of the projected expectation
            for nm, O in ops:
                val, _w = frames.projected_expectation(ex, *O, sectors=sec)
                assert abs(abs(val.real) - abs(frames.expectation(fr_ref, *O).real)) < 1e-9, (nm, val)
                want = 1 if val.real * ref_sign[nm] < 0 else 0      # the FLIP convention
                got = int(frames.frame_charge(PX[i], PZ[i], c["fz"], O[0], O[1], 0, fx=c["fx"])[0])
                assert got == want, (name, i, nm, signs, got, want)
                rows += 1
        assert len(seen) == len(cos), (name, i, len(seen), len(cos))
    assert multi >= 1, "the fixture must produce at least one multi-sector shot"
    assert rows >= 4


def test_sector_coset_refuses_non_frame_copy_sectors():
    """Review MINOR-3: a sector flagged frame_copy=False has no single frame — refuse."""
    n = 2
    z = np.zeros(n, np.uint8)
    cos = [{"vz": z, "fx": z, "fz": z, "weight": 1.0, "frame_copy": False, "logical_flip": None}]
    with pytest.raises(ValueError, match="not a frame copy"):
        frames.sector_coset(cos, z, z, [], [], [])


def test_frames_exports_fallback_and_twin_refuses_it():
    """Review MINOR-1/MINOR-6: `fallback` is exported (all zero here) and the consumer twin
    refuses a synthetic fallback plan by id — the C++ refusal on kappa > 0 / fallback plans is
    reached only by inspection (no small dense fixture produces such a plan; see
    test_frames_refuses_kappa_plans_loudly_inspection_only)."""
    lines, n_data, _ = MULTI_SECTOR_CASES["ghz_t_x_tdag"]
    fr = _sampler(_text(lines, n_data)).sample_barrier(4, 2).frames()
    fb = np.asarray(fr["fallback"])
    assert fb.shape == (int(fr["n_plans"]) + 1,) and not fb.any()
    bad = dict(fr); bad["fallback"] = fb.copy(); bad["fallback"][-1] = 1
    with pytest.raises(ValueError, match="fallback law.*plan ids %d" % (int(fr["n_plans"]))):
        frames.require_kappa_zero(bad)


def test_frames_repeated_calls_are_memoized_and_identical():
    """Review MINOR-5: a second frames() call on the same buffer returns identical arrays
    (the per-buffer plan_structure memo is a pure cache)."""
    lines, n_data, _ = MULTI_SECTOR_CASES["two_pairs_m2"]
    buf = _sampler(_text(lines, n_data)).sample_barrier(8, 4)
    f1 = buf.frames(); f2 = buf.frames()
    for k in ("prefix_words", "plan_id", "a", "r", "kappa", "fallback"):
        assert np.array_equal(np.asarray(f1[k]), np.asarray(f2[k])), k


# ─────────────────────────────────────────────────────────────────────────────
# plan_frame_cosets_fast == plan_frame_cosets (the production table vs the oracle).
# The fast table replaces one ref_expectation per v (2^{|a|} reads) by |a| reads on a
# null-space basis (λ is a GF(2) linear functional — proof in lambda_functional) and
# vectorizes the sums; the suite pins the two EQUAL field by field, same order, on the
# multi-sector fixtures (every plan of every retained shot) and on synthetic masks up to
# |supp a| = 9 on a 10-qubit magic reference (2^9 terms) and a chi = 1 stabilizer reference.
# ─────────────────────────────────────────────────────────────────────────────

def _assert_cosets_equal(fast, ref_tab, ctx):
    assert len(fast) == len(ref_tab), (ctx, len(fast), len(ref_tab))
    for c_f, c_r in zip(fast, ref_tab):
        for key in ("vz", "fx", "fz"):
            assert np.array_equal(np.asarray(c_f[key], np.uint8), np.asarray(c_r[key], np.uint8)), (ctx, key)
        assert c_f["frame_copy"] == c_r["frame_copy"], ctx
        assert c_f["logical_flip"] == c_r["logical_flip"], ctx
        assert abs(float(c_f["weight"]) - float(c_r["weight"])) < 1e-12, (ctx, c_f["weight"], c_r["weight"])


@pytest.mark.parametrize("name", sorted(MULTI_SECTOR_CASES))
def test_plan_frame_cosets_fast_equals_reference_on_fixtures(name):
    lines, n_data, _ = MULTI_SECTOR_CASES[name]
    text = _text(lines, n_data)
    fr = _sampler(text).sample_barrier(12, 11).frames()
    fr_ref = _bare_copy(text).branch_frames()
    seen = 0
    for pid in sorted(set(int(p) for p in np.asarray(fr["plan_id"]))):
        a = np.asarray(fr["a"])[pid]
        fast = frames.plan_frame_cosets_fast(fr_ref, a)
        ref_tab = frames.plan_frame_cosets(fr_ref, a)
        _assert_cosets_equal(fast, ref_tab, (name, pid))
        seen += 1
    assert seen >= 1


_SYNTH_REFS = {
    # 10-qubit magic reference: GHZ_10 with a T on the logical — chi = 2, one free row
    "magic10": (["H 0"] + ["CX 0 %d" % q for q in range(1, 10)] + ["T 0"], 10),
    # 10-qubit stabilizer reference (chi = 1): every sector table is a single frame
    "stab10": (["H 0"] + ["CX 0 %d" % q for q in range(1, 10)], 10),
}


@pytest.mark.parametrize("name", sorted(_SYNTH_REFS))
def test_plan_frame_cosets_fast_equals_reference_synthetic_masks(name):
    lines, n_data = _SYNTH_REFS[name]
    text = _text(lines, n_data)
    fr_ref = _bare_copy(text).branch_frames()
    n = int(fr_ref["n"])
    rng = np.random.default_rng(20260908)
    masks = []
    for size in range(1, 10):                       # |supp a| = 1 .. 9
        m = np.zeros(n, np.uint8); m[:size] = 1     # the first `size` data wires
        masks.append(m)
        for _ in range(2):                          # two random supports of that size
            m2 = np.zeros(n, np.uint8)
            m2[rng.choice(n_data, size=size, replace=False)] = 1
            masks.append(m2)
    for j, a in enumerate(masks):
        fast = frames.plan_frame_cosets_fast(fr_ref, a)
        ref_tab = frames.plan_frame_cosets(fr_ref, a)
        _assert_cosets_equal(fast, ref_tab, (name, j, int(a.sum())))


def test_plan_frame_cosets_cached_serves_fast_and_reference():
    lines, n_data, _ = MULTI_SECTOR_CASES["ghz_t_x_tdag"]
    text = _text(lines, n_data)
    fr = _sampler(text).sample_barrier(4, 5).frames()
    fr_ref = _bare_copy(text).branch_frames()
    a = np.asarray(fr["a"])[int(fr["plan_id"][0])]
    c1, c2 = {}, {}
    fast = frames.plan_frame_cosets_cached(fr_ref, a, c1)
    ref_tab = frames.plan_frame_cosets_cached(fr_ref, a, c2, reference=True)
    _assert_cosets_equal(fast, ref_tab, "cached")
    assert frames.plan_frame_cosets_cached(fr_ref, a, c1) is fast   # memoized


def test_lambda_functional_is_linear_on_the_null_space():
    """The claim behind the speed-up, checked directly: for random pairs w1, w2 in K,
    <r|Z^{w1 ⊕ w2}|r> == <r|Z^{w1}|r> <r|Z^{w2}|r> (both read by ref_expectation)."""
    lines, n_data = _SYNTH_REFS["magic10"]
    fr_ref = _bare_copy(_text(lines, n_data)).branch_frames()
    n = int(fr_ref["n"])
    supp = list(range(n_data))
    free_cols, lb = frames.lambda_functional(fr_ref, supp)
    assert len(free_cols) >= 1
    sx = np.asarray(fr_ref["stab_x"], np.uint8)
    rng = np.random.default_rng(3)
    zero = np.zeros(n, np.uint8)
    checked = 0
    for _ in range(200):
        w1 = rng.integers(0, 2, size=n, dtype=np.uint8); w1[n_data:] = 0
        w2 = rng.integers(0, 2, size=n, dtype=np.uint8); w2[n_data:] = 0
        ok = lambda w: not (((sx.astype(np.int64) @ w.astype(np.int64)) & 1).any())
        if not (ok(w1) and ok(w2)):
            continue
        e1 = frames.ref_expectation(fr_ref, zero, w1, 0).real
        e2 = frames.ref_expectation(fr_ref, zero, w2, 0).real
        e12 = frames.ref_expectation(fr_ref, zero, w1 ^ w2, 0).real
        assert abs(e12 - e1 * e2) < 1e-9
        # and the functional reproduces both reads from their free-column coordinates
        for w, e in ((w1, e1), (w2, e2), (w1 ^ w2, e12)):
            bit = int((w[supp][free_cols].astype(np.int64) @ lb.astype(np.int64)) & 1)
            assert (1.0 - 2.0 * bit) == e
        checked += 1
    assert checked >= 20
