"""lean D1-S3 (2026-11-06, OWNER-SIGNED stream change on CH / PPR circuits): the rotation loop serves CH / PPR circuits —
any pairwise-commuting term axes (X content included) and every shot, PPR-retry shots included (lean/D1/S3/plan.md §1:
Claim 1 holds for any Pauli axis; ρ_c and Φ are symplectic; the Γ_n directions are free_dir_of of the literal product of the
positive Hermitian forms B̂_j, the axis signs η = (phase − |x ∧ z|) / 2).

Teeth (the S2 pattern): `rot_gauge_of` PASS on every distinct key — the rotation plan against the BASE route's exact table
(an all-diagonal shot: (D); a shot with a PPR-retry alternative: the term route's reduced (P, S), kind 0 by its canonical
normal form or (G)) under the COMPUTED gauge, and against (G) on the raw s bit for bit; the selfcheck window (the gauge oracle
plus, on a PPR shot, the state oracle P · Π_{j∈s} rot(B_j)|bare⟩ by gates) green; the model equal to the independent oracle
construction on every CH / PPR circuit. Mutants on twirl_rotation.cpp (ρ without the x part, Φ without the Z block, B̂ without
the Y phase, η from the phase alone) are red here.
"""
import os
import sys
import warnings

import pytest

from xtim import _xtim_testapi as TA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_ja_retention as R  # noqa: E402
from _private import private_text  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

CASES = {name: open(os.path.join(HERE, "data", name)).read()
         for name in ("ch_cultivation.stim", "ja_seed23.stim", "adaptq_born5_dec_enum_producer.stim")}
CASES.update({"ppr:" + k: v for k, v in sorted(R.PPRS.items()) if isinstance(v, str)})
if private_text("ccz_factory_d.stim") is not None:        # a factory whose term axes carry X content (private)
    CASES["factory:d"] = private_text("ccz_factory_d.stim")
# S3 review M1: the class "Z-only axes + PPR-retry alternatives" (kind2 = false, on the loop since S2; its PPR shots joined
# the loop at S3 step 1, so its stream moved under the owner's step-1 signature). The review's fuzz finds (seeds outside the
# J-a suite): the open biased record fuzz 317 and the plain record fuzz 402.
from test_ja_ppr_records import biased_fuzz_records, fuzz_records  # noqa: E402
MIXED = {"mixed:bfz317_open": biased_fuzz_records(317, open_magic=True), "mixed:fz402": fuzz_records(402)}
CASES.update(MIXED)


def _fam(text, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=kw.pop("selfcheck", 0), **kw)


def _shots(name):
    return 400 if name.startswith("factory") else 3000


@pytest.mark.parametrize("name", sorted(CASES))
def test_gauge_oracle_every_key(name):
    s = _fam(CASES[name])
    # an X-content circuit (or, M1, a Z-only circuit with PPR-retry alternatives), served on the loop
    assert TA.rot_on(s) and TA.rot_model(s)["kind2"] == (name not in MIXED), name
    s.sample(10, seed=1)                                       # arms the term view (the PPR shots' base comparator)
    TA.noise_force_full(s, True)   # RE-PINNED lean N1: every alternative's key (the full table: nulls included), as before
    isp = s.term_alt_is_ppr()
    seen, ppr, bad = set(), 0, []
    for al in s.term_events(_shots(name), 7):
        k = tuple(sorted(int(a) for a in al))
        if k in seen:
            continue
        seen.add(k)
        ppr += any(isp[a] for a in k)
        c = TA.rot_gauge_of(s, list(al), True, None)
        if not (c["ok"] and c["g_table_equal"]):
            bad.append((k, c["why"]))
    assert not bad, (name, len(bad), bad[:3])
    assert len(seen) > 1, name
    if name in MIXED:
        assert ppr > 0, name                                   # the class's PPR-retry keys are exercised


@pytest.mark.parametrize("name", sorted(MIXED))
def test_mixed_sample_equals_barrier_split_threads_and_postselect(name, monkeypatch):
    """M1: the D1-S2 T-c checks (sample == sample_barrier, split == single, threads N == 1 incl. the barrier's keys and
    outcomes, postselected rows == the filtered rows) on the Z-only + PPR-retry class."""
    import test_lean_d1s2 as S2
    monkeypatch.setitem(S2.CASES, name, MIXED[name])
    S2.test_sample_equals_barrier_split_and_threads(name)
    S2.test_postselect_accepted_rows_equal_filtered(name)


@pytest.mark.parametrize("name", sorted(CASES))
def test_window_green(name):
    n = _shots(name)
    s = _fam(CASES[name], selfcheck=n)
    s.sample(n, seed=3)
    assert TA.rot_store_stats(s)["window_shots"] == n, TA.rot_store_stats(s)


@pytest.mark.parametrize("name", sorted(CASES))
def test_model_equals_oracle_ch(name):
    from test_lean_d1s1 import _assert_model_equal
    s = _fam(CASES[name])
    _assert_model_equal(TA.rot_model(s), TA.rot_oracle_model(s))
    assert TA.rot_check(s)[1] == 0


# ── step 2 (the deletions): the Clifford path of a circuit WITHOUT rotations (t = 0) ─────────────────────────────────────
# Its fired shot is the identity plan's record rows ⊕ t(P), t(P) read off the rotation records. Teeth: the window's CLIFFORD
# ORACLE on every window shot ((D): the canonical key is the identity's, P == the canonical prefix, t(P) == t(prefix) over the
# channels and the extras), the semantic oracle on every shot (oracle_all_shots: the emitted records projected on P|bare>),
# and the window's stream neutrality. A hot-path mutant (a record word dropped) is red here.
CLIFF = {name: open(os.path.join(HERE, "data", name)).read()
         for name in ("cultivation_d3_rate.stim", "adaptq_steane_h_producer.stim")}
CLIFF["decision_gauge"] = ("RX 0 1\nCX 0 2\nDEPOLARIZE1(0.05) 0 1 2\nMX 0 1\nM 2\nDETECTOR rec[-1]\n"
                           "DECISION(0) rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-2]\n")


@pytest.mark.parametrize("name", sorted(CLIFF))
def test_clifford_path_window_oracles_and_stream_neutrality(name):
    import numpy as np
    text = CLIFF[name]
    s = _fam(text)
    assert not TA.rot_on(s) and TA.rot_model(s)["t"] == 0, name
    w = _fam(text, selfcheck=3000)
    got = w.sample(3000, seed=5)                              # the Clifford oracle on every window shot (raises on a trip)
    ref = s.sample(3000, seed=5)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(got, ref)), name
    o = _fam(text, oracle_all_shots=True)
    o.sample(2000, seed=7)                                    # the semantic oracle on every shot
    assert o.oracle_counts()["shots"] == 2000
