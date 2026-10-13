"""DEM export's diagonal-class fallback normalization (Unreleased).

DEM export propagates every fault in the diagonal class (Pauli x diagonal Clifford) with no PPR
retry. Two ways of WRITING the same circuit could take a T-gate fault's diagonal residual out of
that class although the sampler (PPR retry) accepts the circuit:
  (a) a live-wire MX/MY deferred as `H q; CX q a; H q` (a Hadamard on the data wire);
  (b) a user `H q; <Pauli noise on q>; M q` on an abandoned wire (the noise blocks the H;M fold).
When the default normalization leaves the class, DEM export retries with the diagonal-copy
deferral (fresh |+> ancilla, controlled-X/Y onto q, terminal X read) and the noisy H;M fold
(noise conjugated by H, read in X). Both are exact rewrites; anything still outside the class is
rejected exactly as before. Every circuit the default accepts is byte-identical (the fallback is
consulted only on a reject).

Oracles: xtim's exact forced-fault law (`compile_exact_branches`, the sampler's own propagation
incl. PPR residuals) per single Pauli fault, and the independent statevector `exact_sv` oracle.
"""
import numpy as np
import pytest

import xtim
import exact_sv as E


def export(text, approx):
    return xtim._xtim.export_dem_text(text, True, False, False, True, [], [], approx)


# (a) MX on a LIVE wire (read twice) after a fault between T and T_DAG: residual X.S-type on q0.
LIVE_MX = """
RX 0
T 0
X_ERROR(0.1) 0
T_DAG 0
MX 0
DETECTOR rec[-1]
MX 0
DETECTOR rec[-1] rec[-2]
"""

# (a) the same with a Y read (|+i> = S|+>), exercising the controlled-Y copy.
LIVE_MY = """
RX 0
S 0
T 0
DEPOLARIZE1(0.1) 0
T_DAG 0
MY 0
DETECTOR rec[-1]
MY 0
DETECTOR rec[-1] rec[-2]
"""

# (b) an X-check of a magic-dressed wire written with H's, noise between the last H and M.
NOISY_H_READ = """
RX 0
T 0
DEPOLARIZE1(0.1) 0
T_DAG 0
R 1
H 1
CX 1 0
H 1
X_ERROR(0.1) 1
DEPOLARIZE1(0.1) 1
M 1
DETECTOR rec[-1]
MX 0
DETECTOR rec[-1]
"""

# (a)+(b) together with a coherent parity copy + CS_DAG fix-up shape (the E2 gadget in small).
COHERENT_FIXUP = """
RX 0
T 0
DEPOLARIZE1(0.1) 0
T_DAG 0
RX 1
CX 1 0
DEPOLARIZE2(0.1) 1 0
MX 0
MX 0
DETECTOR rec[-1] rec[-2]
RX 2
CX 2 0
CS_DAG 2 1
DEPOLARIZE1(0.1) 1
CS 2 1
MX 2
DETECTOR rec[-1] rec[-2]
MX 1
DETECTOR rec[-1]
"""

CASES = {"live_mx": LIVE_MX, "live_my": LIVE_MY, "noisy_h_read": NOISY_H_READ,
         "coherent_fixup": COHERENT_FIXUP}


def _forced_law(sim, D, keep_obs, faults):
    br = sim.exact_branches(faults, expectations=[])
    out = {}
    for k in range(len(br)):
        s = frozenset([int(d) for d in np.flatnonzero(br.detectors[k])]
                      + [D + o for o in keep_obs if br.observables[k, o]])
        out[s] = out.get(s, 0.0) + float(br.probabilities[k])
    return out


def _one_hot_full(lines, noise_li, li, repl):
    """The full circuit, every noise instruction kept IN PLACE at zero probability except `li`
    (replaced by the one-hot `repl`): the export sees the circuit's own structure."""
    out = []
    for i, l in enumerate(lines):
        if i == li:
            out.append(repl)
        elif i in noise_li:
            name, arg, tg = E._split(l)
            k = len(arg.split(",")) if arg else 1
            out.append(f"{name}({','.join(['0'] * k)}) " + " ".join(tg))
        else:
            out.append(l)
    return "\n".join(out) + "\n"


def check_vs_forced(text):
    """Per single Pauli fault: the one-hot DEM's exact law == xtim's forced-fault exact law
    (or, for a non-factorable law, a refusal + exact first-order mass under approximation)."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    noise_li = [i for i, l in enumerate(lines) if E._split(l)[0] in E.NOISE]
    sim = xtim.Circuit(text).compile_exact_branches()
    b0 = sim.exact_branches([], expectations=[])
    D, O = b0.detectors.shape[1], b0.observables.shape[1]
    assert not b0.detectors.any()
    keep_obs = [o for o in range(O) if not b0.observables[:, o].any()]
    keep = frozenset(list(range(D)) + [D + o for o in keep_obs])
    n = 0
    for site, li in enumerate(noise_li):
        name, arg, tg = E._split(lines[li])
        for pos, (qs, alts) in enumerate(E.channel_alternatives(name, arg, tg)):
            for p, d in alts:
                if p <= 0:
                    continue
                pstr = "".join(d.get(q, "I") for q in qs)
                law = _forced_law(sim, D, keep_obs, [(site, pos, pstr)])
                want = {frozenset(): 0.9}
                for s, v in law.items():
                    want[s] = want.get(s, 0.0) + 0.1 * v
                r = export(_one_hot_full(lines, set(noise_li), li, E.one_hot(d, qs, 0.1)), 0.0)
                assert not r["rejected"] and not r["errors"], (lines[li], d)
                n += 1
                if r["ok"]:
                    got = E.mech_dist(E.dem_mechs(r["dem"], D), keep)
                    for k in set(got) | set(want):
                        assert abs(got.get(k, 0.0) - want.get(k, 0.0)) <= 1e-12, (
                            lines[li], d, sorted(k), got.get(k), want.get(k))
                else:
                    assert "negative probability" in r["error"] or "over-mixing" in r["error"]
                    assert not E.factorable(want), (lines[li], d)
                    ra = export(_one_hot_full(lines, set(noise_li), li, E.one_hot(d, qs, 0.1)), 1.0)
                    assert ra["ok"], ra["error"]
                    mech = {k & keep: v for k, v in E.dem_mechs(ra["dem"], D).items()}
                    for s, v in law.items():
                        if s:
                            assert abs(mech.get(s, 0.0) - 0.1 * v) <= 1e-12, (lines[li], d, sorted(s))
    return n


@pytest.mark.parametrize("name", sorted(CASES))
def test_fallback_case_accepted_and_sampler_class(name):
    text = CASES[name]
    r = export(text, 1.0)
    assert r["ok"] and not r["rejected"], (r["rejected"], r["error"])
    xtim.Circuit(text).compile_sampler()          # the sampler's class accepts it too


@pytest.mark.parametrize("name", sorted(CASES))
def test_fallback_case_exact_vs_forced_branches(name):
    assert check_vs_forced(CASES[name]) > 0


@pytest.mark.parametrize("name", sorted(set(CASES) - {"coherent_fixup"}))   # exact_sv has no CS
def test_fallback_case_exact_vs_statevector(name):
    E.check_circuit(CASES[name], export)


def test_fallback_still_rejects_genuine_non_diagonal_residual():
    """A Hadamard on the magic-dressed wire followed by more entangling gates is a genuinely
    general Clifford residual: no rewrite applies, the reject (and its index) is unchanged."""
    text = """
RX 0
T 0
X_ERROR(0.1) 0
T_DAG 0
H 0
CX 0 1
M 0 1
"""
    r = export(text, 1.0)
    assert r["rejected"]
    assert r["reject_gate_index"] >= 0


def test_noisy_h_fold_conjugates_the_noise():
    """X_ERROR between the H and the Z read flips the record; after the fold it must become a
    Z_ERROR before the X read (still a record flip). A fold that dropped the conjugation would
    emit no mechanism for it."""
    r = export(NOISY_H_READ, 1.0)
    assert r["ok"], r["error"]
    mech = E.dem_mechs(r["dem"], 2)
    # X_ERROR(0.1) (conjugated: Z before MX, a flip) XOR DEPOLARIZE1(0.1) (flips on 2 of 3)
    want = 0.1 + 0.2 / 3 - 2 * 0.1 * 0.2 / 3
    assert abs(mech[frozenset({0})] - want) < 1e-15, mech
