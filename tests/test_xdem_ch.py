"""CH in the xdem export (Unreleased). The writer rewrites CH(c,t) as two pi/8 rotations about Y_t around a CZ,
(W T W^dag) CZ(c,t) (W T_DAG W^dag) with W = S H (exactly: the T and T_DAG phases cancel): two T-type writer groups on {t}. A circuit whose
rotation axes do not commute (an unaligned T-state fed into a CH) is still refused by the diag check, naming the CH's
rotation, and CH-class residuals with classical feedback stay the DEM export's named refusal.

Oracles (xtim is the oracle; no statevector, no dense route):
  * per fault subset: the file's law of the lines carrying the faults vs xtim's exact engine on the ORIGINAL CH
    circuit (test_xdem_oracle_configs.Gate / compare: exact over GF(2), mixtures counted and failing);
  * the whole outcome law (small files): the file's exact law vs xtim_exact_law of the original circuit, TV <= 1e-12;
  * the native file vs the file of the hand-expanded circuit (the same Clifford+T text): equal law;
  * sampling: XdemSampler on the file vs compile_detector_sampler on the circuit, a two-proportion z per outcome
    pattern.
"""
import itertools
import math
import random
import re
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim import _xtim
from xtim.errors import XtimDemError

from _exact_law import xtim_exact_law
from _xdem_model import XdemModel, exact_file_law
from test_xdem_oracle_configs import Gate, compare, file_laws

ROOT = Path(__file__).resolve().parents[1]
DC = ROOT / "tests/data/dem_ch"


def no_expectations(text):
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("PAULI_EXPECTATION")) + "\n"


H_PREP = "H {q}\nT {q}\nH {q}\nS {q}\nH {q}\n"          # |H> = (H S H) T H |0>, examples/ch_cultivation

# cases the fixtures and generate_ch do not pin: a multi-pair CH line, a control with a lower index than its target,
# T / noise / CH / T_DAG on the control, and a multi-pair line in a REPEAT block
HAND = {
    "ch_multi_pair": H_PREP.format(q=0) + H_PREP.format(q=1) + """RX 2 3
X_ERROR(0.02) 0 2
Z_ERROR(0.03) 1 3
CH 2 0 3 1
DEPOLARIZE1(0.01) 2
MX 2 3
DETECTOR rec[-2]
DETECTOR rec[-1]
""",
    "ch_ctrl_low": H_PREP.format(q=3) + """RX 0
X_ERROR(0.02) 0
DEPOLARIZE1(0.03) 3
CH 0 3
MX 0
DETECTOR rec[-1]
H 3
S_DAG 3
H 3
T_DAG 3
H 3
X_ERROR(0.01) 3
M 3
OBSERVABLE_INCLUDE(0) rec[-1]
""",
    "ch_t_on_control": H_PREP.format(q=1) + """RX 0
T 0
X_ERROR(0.05) 0
CH 0 1
T_DAG 0
Z_ERROR(0.02) 1
MX 0
DETECTOR rec[-1]
""",
    "ch_repeat_multi_pair": H_PREP.format(q=0) + H_PREP.format(q=1) + """REPEAT 2 {
    RX 2 3
    X_ERROR(0.02) 2 0
    CH 2 0 3 1
    Z_ERROR(0.01) 3
    MX 2 3
    DETECTOR rec[-2]
    DETECTOR rec[-1]
}
""",
}


def read(name):
    if name in HAND:
        return HAND[name]
    if name == "ch_cultivation":
        return (ROOT / "examples/ch_cultivation.stim").read_text()
    if name == "ch_cultivation_data":        # tests/data's copy: PAULI_EXPECTATION columns instead of OUTPUT_QUBITS
        return no_expectations((ROOT / "tests/data/ch_cultivation.stim").read_text())
    return no_expectations((DC / f"{name}.stim").read_text())


# the CH circuits that export an extended file (the PAULI_EXPECTATION columns stripped: include_expectations=False)
EXPORTING = ["ch_cultivation", "ch_cultivation_data", "ch_stab", "ch_obs", "ch_feedback", "ch_depol2", "ch_expect",
             "ch_frame"] + list(HAND)


def expand_ch(text):
    """Every `CH c t` pair as S_DAG H T_DAG H S; CZ c t; S_DAG H T H S on t (stim time order)."""
    out = []
    for l in text.splitlines():
        s = l.split("#")[0].split()
        if not s or s[0] != "CH":
            out.append(l)
            continue
        qs = s[1:]
        for c, t in zip(qs[0::2], qs[1::2]):
            out += [f"S_DAG {t}", f"H {t}", f"T_DAG {t}", f"H {t}", f"S {t}", f"CZ {c} {t}",
                    f"S_DAG {t}", f"H {t}", f"T {t}", f"H {t}", f"S {t}"]
    return "\n".join(out) + "\n"


def xdem(text):
    return xtim.Circuit(text).detector_error_model_text(format="xdem")


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def all_subsets_or_sample(G, seed=7):
    n = len(G.atoms)
    if n <= 12:
        return [list(S) for k in range(n + 1) for S in itertools.combinations(range(n), k)]
    rnd = random.Random(seed)
    out = [[i] for i in range(n)] + [list(S) for S in itertools.combinations(range(n), 2)]
    out += [rnd.sample(range(n), rnd.randint(3, min(8, n))) for _ in range(2000)]
    return out


@pytest.mark.parametrize("name", EXPORTING)
def test_ch_exports_an_extended_file(name):
    x = xdem(read(name))
    assert x.startswith("xdem 1")
    assert " x " in x or " rot " in x or " late " in x


@pytest.mark.parametrize("name", EXPORTING)
def test_law_per_fault_subset_against_the_exact_engine_of_the_ch_circuit(name):
    G = Gate(name, read(name))
    cfgs = [c for c in all_subsets_or_sample(G) if G.ok_slots(c)]
    agree, dis, mix, ex = compare(G, cfgs)
    print(f"{name}: atoms={len(G.atoms)} lines={G.nlines} subsets={len(cfgs)} agree={agree} dis={dis} mix={mix}")
    assert len(cfgs) > 0 and dis == 0 and mix == 0, ex


@pytest.mark.parametrize("name", EXPORTING)
def test_whole_law_against_xtim_exact_law(name):
    text = read(name)
    x = xdem(text)
    if len(XdemModel.from_text(x).lines) > 14:
        pytest.skip("file too large for the full enumeration (the per-subset gate covers it)")
    law, tail = xtim_exact_law(text)
    assert tail == 0.0
    d = tv(exact_file_law(x), law)
    print(f"{name}: TV = {d:.3e}")
    assert d <= 1e-12


@pytest.mark.parametrize("name", EXPORTING)
def test_native_file_has_the_law_of_the_hand_expanded_circuit(name):
    """Same model: every fault subset gives the same law on both files (the expanded circuit's atoms are the native
    one's; the files' line ids are matched through the shared atom map), and the bytes are equal when the writer
    sees the same Clifford skeleton up to the T_DAG suffix pair (reported, not required)."""
    text = read(name)
    G = Gate(name, text)
    E = Gate(name, expand_ch(text))
    key = lambda at: repr(at[0])                                           # noqa: E731
    at_e = {key(at): i for i, at in enumerate(E.atoms)}
    assert sorted(at_e) == sorted(key(at) for at in G.atoms)             # noise sites and records are untouched
    cfgs = [c for c in all_subsets_or_sample(G) if G.ok_slots(c)]
    a = file_laws(G.dem, [G.lines_of(c) for c in cfgs])
    b = file_laws(E.dem, [E.lines_of([at_e[key(G.atoms[i])] for i in c]) for c in cfgs])
    assert a == b
    print(f"{name}: {len(cfgs)} subsets equal; bytes {'equal' if G.dem == E.dem else 'differ'}")


def test_ch_rotation_named_in_the_diag_refusal():
    text = (DC / "ch_unaligned.stim").read_text()
    with pytest.raises(XtimDemError, match="does not stay diagonal") as e:
        xdem(text)
    assert "(CH, rotation 1/2)" in str(e.value), str(e.value)


@pytest.mark.parametrize("ie", [True, False])
def test_ch_class_residual_with_feedback_still_refused(ie):
    text = (DC / "ch_resfeedback.stim").read_text()
    with pytest.raises(XtimDemError, match="CH-class fault residuals.*classical feedback"):
        xtim.Circuit(text).detector_error_model_text(format="xdem", include_expectations=ie)


def test_ch_with_expectations_refused_by_name():
    text = (DC / "ch_expect.stim").read_text()
    with pytest.raises(XtimDemError, match='format="xdem" with include_expectations=True'):
        xdem(text)


def _patterns(packed, D):
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :D].astype(np.int64)
    return np.bincount(bits @ (1 << np.arange(D, dtype=np.int64)), minlength=1 << D)


@pytest.mark.parametrize("name,shots,seed", [
    ("ch_cultivation", 400_000, 93_000), ("ch_depol2", 400_000, 93_100), ("ch_feedback", 400_000, 93_200),
    ("ch_obs", 400_000, 93_300),
    pytest.param("ch_cultivation", 8_000_000, 93_400, marks=pytest.mark.slow),
    pytest.param("ch_depol2", 8_000_000, 93_500, marks=pytest.mark.slow)])
def test_cross_route_patterns(name, shots, seed):
    """XdemSampler on the file vs compile_detector_sampler on the circuit: a pooled two-proportion z per joint
    detector+observable pattern; |z| <= 5 (<= 2^(D+O) tests per circuit)."""
    text = read(name)
    S = _xtim._XdemSampler(xdem(text), 65536)
    D, O = S.num_detectors, S.num_observables
    d, o, _ = S.sample(shots, seed, 4, 0, False)
    xd, xo = xtim.Circuit(text).compile_detector_sampler(seed=seed + 1).sample(shots, separate_observables=True,
                                                                               bit_packed=True)
    if O:
        def joint(dp, op):
            db = np.unpackbits(dp, axis=1, bitorder="little")[:, :D]
            ob = np.unpackbits(op, axis=1, bitorder="little")[:, :O]
            return _patterns(np.packbits(np.concatenate([db, ob], 1), axis=1, bitorder="little"), D + O)
        A, B = joint(d, o), joint(xd, xo)
    else:
        A, B = _patterns(d, D), _patterns(xd, D)
    zs = []
    for k1, k2 in zip(A, B):
        p = (k1 + k2) / (2 * shots)
        zs.append(0.0 if p in (0, 1) else (k1 - k2) / shots / math.sqrt(p * (1 - p) * 2 / shots))
    print(f"{name}: {len(zs)} patterns, max|z| {max(map(abs, zs)):.2f}, xdem {A.tolist()} xtim {B.tolist()}")
    assert max(map(abs, zs)) <= 5.0
