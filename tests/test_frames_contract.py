"""G-frames: the frames() STATE CONTRACT against a formula-independent oracle (fixplan r2 "G-frames
r2" + r2-2, fix/frames-born-dec).

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

    |psi_i> ∝ M_dec(i) · Pi_W(i) · Pi_c (1 + (-1)^{c·sigma(i)} G_c)/2 · P · S^a · CZ · |bare> ,  RAW outcomes

(1) The commutation assertion, from exported data, before any state comparison, on every fixture
    and random circuit: record_reads() gives every wire one read basis; every factor (record-channel
    products, the Born observable W, the born_dec() operators) lives on read wires in their read
    basis; all factors pairwise commute (so their order in the contract is immaterial).
(2) The oracle: xtim's EXACT per-shot engine law (compile_exact_branches().exact_branches(faults)),
    pinned ONCE from 4c0c26b as golden data (tests/data/frames_contract_golden.json, sha256 per
    member), so a change to the shared exact layer cannot move both sides together. For every
    non-exact shot, <P> on the composed contract state AND on materialize(i) equal the golden row
    matched by the shot's emitted (detectors, observables, decisions), to 1e-12.
(3) Mutations that must go red: f1 (decision rows omitted from the composer), RAW-vs-EMITTED (the
    decisions projected at the emitted bits: !M / M(p) fixtures), and a synthetic reads table with
    one wire in two bases (the commutation check).

Regenerate the golden ONLY as a declared change:  python -I tests/test_frames_contract.py <xtim root>
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

if __name__ == "__main__":                      # golden generator: python -I <this> <xtim root>
    sys.path.insert(0, sys.argv[1])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtim                                      # noqa: E402
import _frames_contract as fc                    # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frames_contract_golden.json")
SHOTS, SEED = 64, 3
ALL = dict(allow_kappa=True, allow_exact=True, allow_born_obs=True, allow_born_dec=True)


def _law_sha(law):
    return hashlib.sha256(json.dumps(law, sort_keys=True).encode()).hexdigest()


def _load():
    with open(GOLDEN) as fh:
        return json.load(fh)


def _targets(s, b):
    return (fc.unpack(b.dets())[:, :s.num_detectors], fc.unpack(b.obs())[:, :s.num_observables],
            fc.unpack(b.decisions())[:, :s.num_decisions])


def test_golden_provenance():
    g = _load()
    assert g["commit"] == "4c0c26b"
    assert sorted(g["fixtures"]) == sorted(fc.GF_FIXTURES)
    for name, f in g["fixtures"].items():
        assert f["text"] == fc.GF_FIXTURES[name]
        fam = fc.single_fault_family(f["text"])
        assert sorted(f["members"]) == sorted(t for t, _x, _f in fam)
        for tag, m in f["members"].items():
            assert _law_sha(m["law"]) == m["sha256"], (name, tag)


@pytest.mark.parametrize("name", sorted(fc.GF_FIXTURES))
def test_current_exact_law_equals_golden(name):
    g = _load()["fixtures"][name]
    probes = [(lab, {int(q): P for q, P in d.items()}) for lab, d in g["probes"]]
    for tag, _ttxt, faults in fc.single_fault_family(g["text"]):
        cur = fc.law_table(g["text"], faults, probes)
        gold = g["members"][tag]["law"]
        assert sorted(cur) == sorted(gold), (tag, sorted(cur), sorted(gold))
        for k in gold:
            assert abs(cur[k][0] - gold[k][0]) < 1e-12
            assert np.allclose(cur[k][1], gold[k][1], atol=1e-12, rtol=0)


# ── mutations: each must go red against the golden law ────────────────────────
_DEC_FIXTURES = ["bd1", "bd1_noisy", "bd2", "cz_records_bd", "q3_dec", "q3dec_noisyT", "bell2_3dec",
                 "teleport_dec", "F1", "F2"]


# ── port fixtures (reviewer, <S>/frames/rv/port_c.py): INPUT_QUBITS, a Born decision coupled to
#    the input wire, random input_pauli frames — the contract's "RAW, before the input-frame
#    relabel" clause: composer (raw coins) == materialize(i), and decisions() carries the relabel.
PORT_FIXTURES = {
    "port_bd": ("INPUT_QUBITS in 0\nRX 1\nT 1\nCZ 0 1\nMX 1\nDECISION(0) rec[-1]\nR 9\nX_ERROR(0.1) 9\nM 9\n"
                "DETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n"),
    "port_bd2": ("INPUT_QUBITS in 0 2\nRX 1\nT 1\nCZ 0 1\nCX 2 1\nMX 1\nDECISION(0) rec[-1]\nMX 2\n"
                 "DECISION(1) rec[-1]\nOUTPUT_QUBITS out 0\n"),
}


def test_mutation_one_wire_two_bases_breaks_commutation():
    reads = {"wire": np.array([0, 0], np.int32), "basis": np.array([0, 2], np.int32)}
    with pytest.raises(fc.ContractError, match="read in two bases"):
        fc.assert_reads_one_basis(reads)
    rows = [(f"rec{k}",) + fc.record_pauli(reads, k, 1) for k in range(2)]
    with pytest.raises(fc.ContractError, match="anticommute"):
        fc.assert_rows_commute(rows)
    # the same records on one basis commute
    ok = {"wire": np.array([0, 0], np.int32), "basis": np.array([2, 2], np.int32)}
    fc.assert_reads_one_basis(ok)
    fc.assert_rows_commute([(f"rec{k}",) + fc.record_pauli(ok, k, 1) for k in range(2)])


# ── golden generator (declared change only) ───────────────────────────────────
if __name__ == "__main__":
    import subprocess
    root = sys.argv[1]
    assert xtim.__file__.startswith(root), xtim.__file__
    commit = subprocess.check_output(["git", "-C", root, "rev-parse", "--short=7", "HEAD"]).decode().strip()
    out = {"commit": commit, "generator": "tests/test_frames_contract.py __main__ (fc.law_table)",
           "fixtures": {}}
    for name, text in sorted(fc.GF_FIXTURES.items()):
        probes = fc.probes_for(text)
        mem = {}
        for tag, _ttxt, faults in fc.single_fault_family(text):
            law = fc.law_table(text, faults, probes)
            mem[tag] = {"faults": faults, "law": law, "sha256": _law_sha(law)}
        out["fixtures"][name] = {"text": text, "probes": [[lab, {str(q): P for q, P in d.items()}]
                                                          for lab, d in probes], "members": mem}
    with open(GOLDEN, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote", GOLDEN, "commit", commit)


# ── J-b2 J1b: the PRODUCTION leg (plan r2.1 J1b). The golden exact law (pinned at 4c0c26b from the exact engine,
# independent of the twirl engine) against the PUBLIC sampler's retained state: for EVERY shot of every forced
# single-fault member, materialize(i)'s probes equal the golden row matched by the shot's emitted (detectors,
# observables, decisions), to 1e-12. Production serves every shot by the law (no exact shot to skip), and serves the
# golden fixtures' observables (no refusal). The sigma-era composer (fc.compose over frames(allow_exact=...)) and its
# mutations are the legacy export's own contract and retire with it at J2 (jb2 mapping).
from _public import public_sampler  # noqa: E402


def _check_member_public(text, member, probes, outs):
    s = public_sampler(text)
    b = s.sample_barrier(SHOTS, seed=SEED)
    D, O, C = _targets(s, b)
    pw = [fc.probe_wires(lab, d, s, outs) for lab, d in probes]
    law = member["law"]
    bad = 0
    for i in range(SHOTS):
        k = fc.shot_key(D, O, C, i)
        assert k in law and law[k][0] > 1e-15, (k, sorted(law))   # the record is possible
        m = b.materialize(i)
        bad += max(abs(fc.probe_value(m, None, w) - g) for w, g in zip(pw, law[k][1])) > 1e-12
    return SHOTS, bad


@pytest.mark.parametrize("name", sorted(fc.GF_FIXTURES))
def test_materialize_equals_golden_exact_law_public(name):
    g = _load()["fixtures"][name]
    probes = [(lab, {int(q): P for q, P in d.items()}) for lab, d in g["probes"]]
    outs = fc.output_qubits(g["text"])
    for tag, ttxt, _faults in fc.single_fault_family(g["text"]):
        n, bad = _check_member_public(ttxt, g["members"][tag], probes, outs)
        assert bad == 0, (tag, n, bad)


def test_materialize_golden_mutation_is_red():
    """Non-vacuity of the production leg: the golden rows of ANOTHER member of the same fixture are rejected."""
    name = "bd1_noisy" if "bd1_noisy" in fc.GF_FIXTURES else sorted(fc.GF_FIXTURES)[0]
    g = _load()["fixtures"][name]
    probes = [(lab, {int(q): P for q, P in d.items()}) for lab, d in g["probes"]]
    outs = fc.output_qubits(g["text"])
    fam = fc.single_fault_family(g["text"])
    red = 0
    for (tag, ttxt, _f), (tag2, _t2, _f2) in zip(fam, fam[1:] + fam[:1]):
        if tag == tag2:
            continue
        try:
            red += _check_member_public(ttxt, g["members"][tag2], probes, outs)[1] > 0
        except AssertionError:
            red += 1                                  # an impossible record under the other member's law
    assert red > 0
