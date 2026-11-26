"""J-b1 P9b parts 3-4: the consumer branch law (design §2.10 Lemma R3; plan §2.1 P9b; rulings on fixplan-p9b-xq9 /
fixplan-p9b-frame).

A registered producer branch (add_branch, EAGER: every per-E step and guard there, never inside run()) is put on its
frame — the ideal's rows with the free register F₀ (E_min = {}) or F₀ ∪ E_min (the E rows VARIABLES) — its constant
promoted rows folded, its certified sign differences split off as the Pauli D = Π Xrow_a (on the port wires a frames
row, elsewhere the shot's P). Its consumer bare is built by the SAME construction as branch 0's (tensor + port
Cliffords, or the XQ9 (A) composition):
  * E_min = {}: on this sampler — branch 0's plans (plans never read α), the branch's own Born trie per plan (the
    sparse (plan, branch) map); branch-0 shots never probe it;
  * E_min != {}: on its E-class's per-E consumer (same text and options, its ideal = the branch on the per-E frame,
    its own plan table); its shots are run there and scattered back, relative to the ideal's reference (A1 §2);
  * D6: the payload (bare, tries) is evictable; a miss is re-supplied inside run() by add_branch's pure function.

Gates (the end-to-end chain law):
  * fixed chains (proto3's CE2-chain and two variants; a teleport-in chain; a producer whose branches carry an eps
    delta on the port wires; the per-E fixture producer with covered consumers; CE1 MX0) and a random chained family
    (dressed mirror pairs included): the producer's FAM records → branches(records) → the consumer sampled per shot
    with input_branch = the shot's branch and input_pauli = its frame P_rec·W(u) — the joint (producer ‖ consumer)
    detector law equals an INDEPENDENT statevector trajectory sampler on the joint closed circuit (any T-depth:
    tests/_traj_oracle.py, reference-relative), every marginal and pairwise cell at z < 5 (one rerun at twice the
    shots);
  * non-vacuous: Born directions walked on branch tries, branches with an eps delta on the ports, E_min != {} chains;
  * MUT-BR-TRIE-SHARED (a branch shot walks branch 0's trie), MUT-BR-NO-DELTA (the eps delta dropped), MUT-E-CONST
    (the E rows reduced as constants), MUT-XQ9-LAZY (the per-E steps deferred to the run), MUT-BR-RESUPPLY-STALE (a
    re-supply returns branch 0's amplitudes) -> red;
  * D6: re-supplied payloads give the same bits as resident ones at the same seed;
  * the plan's per-E fixture (consumer T 0 / MX 0) is refused by name (carried_branch_x: an X-part on the E row);
    xstage F1 / F2 / F4 are declared refusals (carried_branch_x / carried_free_x);
  * readers of branch rows: materialize serves them; frames / signs / fidelity refuse by name; branch_handles on a
    branch row (a 3-stage re-export) is the named law limitation LAW_LIMITATION_CARRIED_AXIS / branch_frame; per-E rows
    are sampled but not recorded (refused by name).
"""
import os
import random
import re
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
import _traj_oracle as T  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402
from xtim.errors import TwirlClass, XtimTwirlRefusal  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

CE2 = ("R 0 1 2\nH 0\nCX 0 1\nDEPOLARIZE1(0.3) 0\nT 0\nT_DAG 1\nH 2\nCX 2 0\nCX 2 1\nH 2\nM 2\nDETECTOR rec[-1]\n"
       "OUTPUT_QUBITS out 0 1\n")
FIXED = {
    "ce2_htH_t": (CE2, "H 0\nT 0\nH 0\nT 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n", [0, 1]),
    "ce2_cx_t": (CE2, "CX 0 1\nT 1\nH 0\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n", [0, 1]),
    "ce2_fresh": (CE2, "R 3\nCX 0 3\nT 3\nH 3\nM 3\nDETECTOR rec[-1]\nMX 1\nDETECTOR rec[-1]\n", [0, 1]),
    # teleport-in: the producer's Bell half is read in X (a coin), the consumer reads the carried half
    "bell_teleport": ("RX 0\nR 1\nCX 0 1\nT 0\nZ_ERROR(0.2) 0\nMX 0\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 1\n",
                      "T 0\nH 0\nM 0\nDETECTOR rec[-1]\n", [1]),
    # a producer whose branches carry a nonzero eps delta on the port wires (and Born directions on branch tries)
    "port_delta": ("RX 0 1\nR 2\nX_ERROR(0.2) 1\nS 1\nH 0\nH 0\nCX 0 1\nS 0\nT 1\nCZ 0 2\nCX 1 2\nM 2\n"
                   "DETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1\n",
                   "H 1\nH 1\nS 1\nH 1\nT 0\nMX 0\nDETECTOR rec[-1]\nM 1\nDETECTOR rec[-1]\n", [0, 1]),
}


def fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def _joint(ptext, cbody, outs, nports, shift):
    """The joint closed circuit: the producer (OUTPUT_QUBITS dropped), then the consumer body with its port j on the
    producer's output qubit outs[j] and its fresh wire w on shift + w."""
    def mp(m):
        q = int(m.group(0))
        return str(outs[q]) if q < nports else str(shift + q)
    body = []
    for line in cbody.splitlines():
        if line.startswith(("DETECTOR", "OBSERVABLE_INCLUDE")):
            body.append(line)
        else:
            body.append(re.sub(r"(?<![\[\-\d(.])\b\d+\b", mp, line))
    return re.sub(r"OUTPUT_QUBITS.*\n", "", ptext) + "\n".join(body) + "\n"


def chain_samples(ptext, cbody, outs, N, seed):
    """(FAM chain samples, trajectory samples, consumer) or None when a branch is refused by name."""
    nports = len(outs)
    p = fam(ptext)
    d, o, dc, R = p.sample_records(N, seed=seed)
    H, fr = p.branches(R)
    ctext = "INPUT_QUBITS in " + " ".join(map(str, range(nports))) + "\n" + cbody
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = TA.compile_family_sampler_from_ideal_state(p.ideal_state(), ctext)
    ids = {}
    ib = np.zeros(N, np.int32)
    for i, h in enumerate(H):
        k = (bytes(h["digest"]), bytes(h["z"]))
        if k not in ids:
            try:
                b = c.add_branch(h)
            except XtimTwirlRefusal:
                return None
            ids[k] = b
        ib[i] = ids[k]
    ip = np.concatenate([fr[:, 0, :], fr[:, 1, :]], 1).astype(np.uint8)
    dcn, ocn, _ = c.sample(N, seed=seed + 1, input_pauli=ip, input_branch=ib)
    M = np.concatenate([F.unpack(d, p.num_detectors), F.unpack(o, p.num_observables),
                        F.unpack(dcn, c.num_detectors), F.unpack(ocn, c.num_observables)], 1)
    shift = 1 + max(int(x) for x in re.findall(r"(?<![\[\-\d(.])\b\d+\b", re.sub(r"rec\[-\d+\]", "", ptext)))
    Ed, Eo = T.sample(_joint(ptext, cbody, outs, nports, shift), N, seed=seed + 7)
    return M, np.concatenate([Ed, Eo], 1), c


def chain_z(ptext, cbody, outs, N=30000, seed=5):
    r = chain_samples(ptext, cbody, outs, N, seed)
    if r is None:
        return None, None
    M, E, c = r
    z = F.two_sample_z(M, E)
    if z >= 5:                                         # the recurrence rule: one rerun at twice the shots
        M, E, c = chain_samples(ptext, cbody, outs, 2 * N, seed + 100)
        z = F.two_sample_z(M, E)
    return z, c


@pytest.mark.parametrize("name", sorted(FIXED))
def test_fixed_chains_equal_the_trajectory_oracle(name):
    z, c = chain_z(*FIXED[name])
    assert z is not None, "a fixture branch was refused or has E_min != {}"
    assert z < 5, z
    bc = c.branch_counters()
    assert bc["branch_shots"] > 0
    if name == "ce2_htH_t":
        assert bc["branch_trie_node_evals"] > 0           # Born directions walked on branch tries (non-vacuous)
    if name == "port_delta":
        assert bc["branches_with_port_delta"] > 0          # the eps delta reaches the port (MUT-BR-NO-DELTA -> red)
        assert bc["branch_trie_node_evals"] > 0


def _gen(rng):
    nq = rng.choice([1, 2, 2, 3]); na = rng.choice([0, 1, 1])
    L = ["RX " + " ".join(str(q) for q in range(nq) if rng.random() < 0.6),
         "R " + " ".join(str(nq + a) for a in range(na))]
    for _ in range(rng.randint(2, 7)):
        r = rng.random(); q = rng.randrange(nq)
        if r < 0.25: L.append(f"T {q}")
        elif r < 0.33: L.append(f"T_DAG {q}")
        elif r < 0.45: L.append(f"H {q}")
        elif r < 0.55: L.append(f"S {q}")
        elif r < 0.7 and nq > 1:
            a, b = rng.sample(range(nq), 2); L.append(f"CX {a} {b}")
        else: L.append(f"{rng.choice(['X_ERROR', 'Z_ERROR', 'Y_ERROR', 'DEPOLARIZE1'])}(0.2) {q}")
    if rng.random() < 0.4:                              # a dressed mirror pair (the E_min != {} source)
        q = rng.randrange(nq)
        L += [f"T {q}", f"{rng.choice(['X_ERROR', 'Y_ERROR'])}(0.3) {q}", f"T_DAG {q}"]
    for a in range(na):
        for q in rng.sample(range(nq), rng.randint(1, nq)):
            L.append(f"CX {q} {nq + a}" if rng.random() < 0.6 else f"CZ {q} {nq + a}")
        L.append(f"M {nq + a}"); L.append("DETECTOR rec[-1]")
    L.append("OUTPUT_QUBITS out " + " ".join(map(str, range(nq))))
    ptext = "\n".join(l for l in L if l.strip() not in ("RX", "R", "")) + "\n"
    nf = rng.choice([0, 1])
    nc = nq + nf
    C = []
    for _ in range(rng.randint(1, 5)):
        r = rng.random(); q = rng.randrange(nc)
        if r < 0.3: C.append(f"T {q}")
        elif r < 0.5: C.append(f"H {q}")
        elif r < 0.6: C.append(f"S {q}")
        elif nc > 1:
            a, b = rng.sample(range(nc), 2); C.append(f"{rng.choice(['CX', 'CZ'])} {a} {b}")
    for q in range(nc):
        C.append(f"{rng.choice(['M', 'MX'])} {q}"); C.append("DETECTOR rec[-1]")
    if nf:
        C.insert(0, f"R {nq}")
    return ptext, "\n".join(C) + "\n", list(range(nq))


@pytest.mark.parametrize("seed", [1, 2])
def test_random_chained_family_equals_the_trajectory_oracle(seed):
    rng = random.Random(seed)
    st = dict(chains=0, skipped=0, born=0, delta=0, perE=0, worst=0.0)
    for it in range(60):
        ptext, cbody, outs = _gen(rng)
        try:
            z, c = chain_z(ptext, cbody, outs, N=12000, seed=it + 1)
        except (ValueError, RuntimeError):
            st["skipped"] += 1
            continue
        if z is None:
            st["skipped"] += 1
            continue
        assert z < 5, (z, ptext, cbody)
        st["chains"] += 1
        st["worst"] = max(st["worst"], round(z, 2))
        bc = c.branch_counters()
        st["born"] += bc["branch_trie_node_evals"] > 0
        st["delta"] += bc["branches_with_port_delta"] > 0
        st["perE"] += bc["distinct_emin"] > 1
    print("branch-law chained family", seed, st)
    assert st["chains"] >= 10 and st["born"] > 0 and st["delta"] > 0 and st["perE"] > 0, st


def test_branch0_workload_never_probes():
    p = fam(CE2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = TA.compile_family_sampler_from_ideal_state(
            p.ideal_state(), "INPUT_QUBITS in 0 1\nH 0\nT 0\nH 0\nM 0 1\nDETECTOR rec[-1]\n")
    d, o, dc, R = p.sample_records(200, seed=1)
    for h in p.branch_handles(R):
        c.add_branch(h)
    a = c.sample(3000, seed=7)
    b = c.sample(3000, seed=7, input_branch=np.zeros(3000, np.int32))
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
    assert c.branch_counters()["branch_map_probes"] == 0 and c.branch_counters()["map_probes"] == 0


def test_branch_rows_readers():
    p = fam(CE2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = TA.compile_family_sampler_from_ideal_state(
            p.ideal_state(), "INPUT_QUBITS in 0 1\nT 1\nM 1\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n")
    d, o, dc, R = p.sample_records(300, seed=1)
    H, fr = p.branches(R)
    ids = {}
    ib = np.array([ids.setdefault((bytes(h["digest"]), bytes(h["z"])), c.add_branch(h)) for h in H], np.int32)
    ip = np.concatenate([fr[:, 0, :], fr[:, 1, :]], 1).astype(np.uint8)
    *_, Rc = c.sample_records(300, seed=3, input_pauli=ip, input_branch=ib)
    rows = [i for i in range(300) if Rc.branch(i) != 0]
    assert rows
    st = c.materialize(Rc, rows[0])                         # served: the branch's own consumer bare
    assert int(st["n"]) > 0
    with pytest.raises(ValueError, match="carried-branch row"):
        c.frames(Rc)
    with pytest.raises(ValueError, match="carried-branch row"):
        c.signs(Rc, [])
    with pytest.raises(XtimTwirlRefusal) as ei:
        c.branch_handles(Rc)
    assert ei.value.twirl_class is TwirlClass.LAW_LIMITATION_CARRIED_AXIS and ei.value.sub_reason == "branch_frame"


# ── the per-E law (E_min != {}) ──────────────────────────────────────────────────────────────────────────────────
# PER_E: the plan's per-E fixture producer (review fixplan-p9b-xq9 §3, pinned): the dressed branch (X between T and
# T_DAG) has E_min = {the X(q0) row} with amplitude in both sectors.
PER_E = "RX 0\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nOUTPUT_QUBITS out 0\n"
CE1_MX0 = "RX 0\nR 1\nT 0\nX_ERROR(0.3) 0\nT_DAG 0\nCX 0 1\nMX 0\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 1\n"
PER_E_CHAINS = {
    "per_e_htH": (PER_E, "H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n", [0]),
    "per_e_htH_mx": (PER_E, "H 0\nT 0\nH 0\nMX 0\nDETECTOR rec[-1]\n", [0]),
    "ce1_mx0_htH": (CE1_MX0, "H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n", [1]),
}


@pytest.mark.parametrize("name", sorted(PER_E_CHAINS))
def test_per_e_chains_equal_the_trajectory_oracle(name):
    """The E rows are VARIABLES of the per-E consumer (its own plan table): MUT-E-CONST (the E rows reduced as
    constants) -> red here."""
    z, c = chain_z(*PER_E_CHAINS[name])
    assert z is not None and z < 5, z
    assert c.branch_counters()["distinct_emin"] >= 2                # an E_min != {} branch was sampled


def test_pinned_per_e_fixture_is_refused_by_name():
    """The plan's per-E fixture as the review reads it (consumer T 0 / MX 0): the axis Z(q0) has its X-part on the E
    row X(q0) — class (iii), a law limitation (T-depth 1; served after J-c). Reported, not reshaped."""
    p = fam(PER_E)
    d, o, dc, R = p.sample_records(400, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = TA.compile_family_sampler_from_ideal_state(p.ideal_state(), "INPUT_QUBITS in 0\nT 0\nMX 0\nDETECTOR rec[-1]\n")
    refused = 0
    for h in p.branch_handles(R):
        try:
            c.add_branch(h)
        except XtimTwirlRefusal as e:
            assert e.twirl_class is TwirlClass.LAW_LIMITATION_CARRIED_AXIS and e.sub_reason == "carried_branch_x"
            assert "T-depth" not in str(e)
            refused += 1
    assert refused >= 1


def test_per_e_rows_are_not_recorded():
    p = fam(PER_E)
    d, o, dc, R = p.sample_records(400, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = TA.compile_family_sampler_from_ideal_state(
            p.ideal_state(), "INPUT_QUBITS in 0\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n")
    H, fr = p.branches(R)
    ids = {}
    ib = np.array([ids.setdefault((bytes(h["digest"]), bytes(h["z"])), c.add_branch(h)) for h in H], np.int32)
    assert any(c.branch_emin(b) for b in set(ib.tolist()))
    c.sample(400, seed=2, input_branch=ib)                          # served
    with pytest.raises(ValueError, match="not recorded"):
        c.sample_records(400, seed=2, input_branch=ib)


# ── D6: the branch payload is evictable; a miss is re-supplied inside run() by add_branch's pure function ─────────
# a Born producer whose z = 1 branch's amplitudes differ from the ideal's, read by a consumer Born direction (the
# re-supplied α is what the branch's trie evaluates: MUT-BR-RESUPPLY-STALE flips these bits)
D6_BORN = ("RX 0 1\nT 0\nCX 1 0\nZ_ERROR(0.1) 0\nMX 1\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n",
           "T 0\nMX 0\nDETECTOR rec[-1]\n", [0])


@pytest.mark.parametrize("name", ["ce2_htH_t", "port_delta", "per_e_htH", "d6_born"])
def test_d6_resupply_is_bit_identical(name):
    """Budget 0 evicts every other branch payload at each branch shot, and every payload is evicted before the run:
    the re-supplied payloads give the SAME bits as resident ones at the same seed (MUT-BR-RESUPPLY-STALE -> red)."""
    ptext, cbody, outs = {**FIXED, **PER_E_CHAINS, "d6_born": D6_BORN}[name]
    p = fam(ptext)
    d, o, dc, R = p.sample_records(4000, seed=3)
    H, fr = p.branches(R)
    ctext = "INPUT_QUBITS in " + " ".join(map(str, range(len(outs)))) + "\n" + cbody
    out = []
    for budget in (None, 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = TA.compile_family_sampler_from_ideal_state(p.ideal_state(), ctext)
        ids = {}
        ib = np.array([ids.setdefault((bytes(h["digest"]), bytes(h["z"])), c.add_branch(h)) for h in H], np.int32)
        c.set_branch_budget(budget)
        if budget == 0:
            c.evict_branches()
        ip = np.concatenate([fr[:, 0, :], fr[:, 1, :]], 1).astype(np.uint8)
        out.append([np.asarray(x) for x in c.sample(4000, seed=11, input_pauli=ip, input_branch=ib)])
        bc = c.branch_counters()
        if budget == 0:
            assert bc["branch_resupplies"] + bc["perE_resupplies"] > 0
            assert bc["branch_evictions"] + bc["perE_evictions"] > 0
            if name == "d6_born":
                assert bc["branch_trie_node_evals"] > 0           # the re-supplied α is read (non-vacuous)
    assert all(np.array_equal(a, b) for a, b in zip(*out))


# ── declared refusals (rulings on fixplan-p9b-frame / fixplan-p9b-xq9; listed in the manifest) ───────────────────
# xstage F1 (an S-dressed carried wire, then H·T·H): the dressed branch's E row is X-flipped (class iii) →  [fixplan-F-branch-readers F-3: NOT
# T-depth 1 — the fault blocks the T·T merge (end axes Z, Z, X anticommute): the fault-dependent class; the branch law meets it first]
# carried_branch_x at add_branch; xstage F2 / F4 (a later CCZ after H on a magic-carrying wire; F4 = the toy CCZ factory
# teleport-out): a consumer axis with its X-part on the ideal's free rows → carried_free_x at compile.
DECLARED = {
    "xstage_f1": ("RX 0\nT 0\nX_ERROR(0.3) 0\nT 0\nOUTPUT_QUBITS out 0\n", "H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n",
                  "carried_branch_x"),
    "xstage_f2": ("RX 0 1 2\nX_ERROR(0.1) 0\nX_ERROR(0.1) 1\nZ_ERROR(0.1) 0\nCCZ 0 1 2\nX_ERROR(0.1) 2\nRX 3\nCX 3 2\n"
                  "MX 3\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0 1 2\n",
                  "H 1\nS 0\nCCZ 0 1 2\nM 0\nDETECTOR rec[-1]\nMX 1\nDETECTOR rec[-1]\nM 2\nDETECTOR rec[-1]\n",
                  "carried_free_x"),
    "xstage_f4": ("RX 0 1 2\nX_ERROR(0.1) 0\nY_ERROR(0.1) 1\nCCZ 0 1 2\nRX 3 4 5\nCX 3 0\nCX 4 1\nCX 5 2\nM 0 1 2\n"
                  "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 3 4 5\n",
                  "H 0\nCCZ 0 1 2\nMX 0\nDETECTOR rec[-1]\nM 1\nDETECTOR rec[-1]\nM 2\nDETECTOR rec[-1]\n",
                  "carried_free_x"),
}


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_declared_refusals(name):
    ptext, cbody, sub = DECLARED[name]
    p = fam(ptext)
    nports = len(p.ideal_state().wires)
    seen = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = TA.compile_family_sampler_from_ideal_state(
                p.ideal_state(), "INPUT_QUBITS in " + " ".join(map(str, range(nports))) + "\n" + cbody)
        R = p.sample_records(400, seed=1)[-1]
        for h in p.branch_handles(R):
            c.add_branch(h)
    except XtimTwirlRefusal as e:
        seen = e
    assert seen is not None and seen.twirl_class is TwirlClass.LAW_LIMITATION_CARRIED_AXIS
    assert seen.sub_reason == sub and "T-depth" not in str(seen)
