"""DEM export: a gate fault that acts on a PAULI_EXPECTATION column THROUGH its declared
byproduct frame (the `rec[-k]` targets) is an error mechanism on that column.

Defect (3.1.8/3.1.9, fixed in 3.1.10): the gate-noise path folded only the conjugation of the
declared Pauli P itself (E^dag P E = +-P), never the flips of the frame records. So

    R 0 1 / H 0 / CX 0 1 / Z_ERROR(0.2) 0 / MX 0 / PAULI_EXPECTATION(0) X1 rec[-1]

exported `logical_observable L0` (no error) while the sampler flips <X1 . frame> in 20% of
shots. The readout twin `MX(0.2) 0` was already exported (3.1.4). The magic twin (Bell pair,
T (x) T^dag, X fault on the measured qubit) takes |<X1>| from 1 to 0 with no detector firing and
was neither emitted nor refused.

The column the sampler reports is (-1)^{frame records} <P>; in deferred space that is the
expectation of the frame-folded operator P' = P . prod_{j in frame} R_j (R_j the measured
Pauli of terminal read j), so every DEM rule (sign flip, magnitude classification, the
completeness gate) is applied to P'.

Oracles (independent of the exporter):
  * Clifford circuits: Stim. Each `PAULI_EXPECTATION P rec[..]` becomes `MPP P` +
    `OBSERVABLE_INCLUDE(O+r) <that record> rec[..]` and Stim's DEM is compared mechanism by
    mechanism (L(O+r) is the documented column of expectation r).
  * Any circuit (incl. magic): single-fault enumeration on the EXACT engine. Every channel is
    isolated (one noise instruction, one target / pair); each of its Pauli alternatives is
    inserted as a deterministic gate into the noiseless circuit and the exact sampler gives the
    raw records and the exact per-shot expectations. The distribution of the signature
    (detector / observable / decision record parities vs the noiseless parities; expectation
    sign vs the noiseless value) must equal the distribution the exported single-channel DEM
    induces. A magnitude change must be refused (no detector) or flagged (detector fires).
"""
import itertools
import math
import random
import re

import numpy as np
import pytest

import xtim
from xtim.errors import XtimDemError

stim = pytest.importorskip("stim")

MEAS_1 = {"M", "MX", "MY", "MR", "MRX", "MRY", "MZ", "MRZ"}
MEAS_2 = {"MXX", "MYY", "MZZ"}
NOISE = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2",
         "PAULI_CHANNEL_1", "PAULI_CHANNEL_2"}
RECLINES = {"DETECTOR", "OBSERVABLE_INCLUDE", "DECISION", "PAULI_EXPECTATION"}


# ---------------------------------------------------------------------------------------
# a tiny flat-circuit parser (absolute record indices)
# ---------------------------------------------------------------------------------------
def _split(line):
    m = re.match(r"^([A-Z_0-9]+)(?:\(([^)]*)\))?\s*(.*)$", line.strip())
    return m.group(1), m.group(2), m.group(3).split()


def _num_records(name, targets):
    if name in MEAS_1:
        return len(targets)
    if name in MEAS_2:
        return len(targets) // 2
    if name == "MPP":
        return len(targets)
    return 0


def parse_records(text):
    """[(kind, label, [abs rec]), ...] for DETECTOR/OBSERVABLE_INCLUDE/DECISION/
    PAULI_EXPECTATION, in order, plus the total record count."""
    out, nrec = [], 0
    for line in text.strip().splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        name, arg, tg = _split(line)
        if name in RECLINES:
            recs = [nrec + int(t[4:-1]) for t in tg if t.startswith("rec[")]
            out.append((name, arg, recs))
        nrec += _num_records(name, tg)
    return out, nrec


# ---------------------------------------------------------------------------------------
# DEM canonicalisation + the distribution a DEM induces on signatures
# ---------------------------------------------------------------------------------------
def canon(dem):
    errors = {}
    for ins in dem.flattened():
        if ins.type == "error":
            key = []
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    key.append(("D", t.val))
                elif t.is_logical_observable_id():
                    key.append(("L", t.val))
            key = tuple(sorted(key))
            assert key not in errors, f"duplicate mechanism {key}"
            errors[key] = ins.args_copy()[0]
    return errors


def assert_dem_equal(got, want, rel=1e-9, abs_=1e-15):
    ge, we = canon(got), canon(want)
    ge = {k: v for k, v in ge.items() if v > abs_}
    we = {k: v for k, v in we.items() if v > abs_}
    assert set(ge) == set(we), (
        f"mechanism sets differ:\n  only xtim: {sorted(set(ge) - set(we))}\n"
        f"  only oracle: {sorted(set(we) - set(ge))}\nxtim:\n{got}\noracle:\n{want}")
    for k in we:
        assert math.isclose(ge[k], we[k], rel_tol=rel, abs_tol=abs_), (k, ge[k], we[k])


def dem_distribution(dem, D):
    """{frozenset(target ids): prob} induced by independent mechanisms (Dk -> k, Lk -> D+k)."""
    mechs = [(frozenset(v if t == "D" else D + v for t, v in k), p)
             for k, p in canon(dem).items()]
    dist = {frozenset(): 1.0}
    for sig, p in mechs:
        nd = {}
        for s, q in dist.items():
            nd[s] = nd.get(s, 0.0) + q * (1 - p)
            t = s ^ sig
            nd[t] = nd.get(t, 0.0) + q * p
        dist = nd
    return dist


# ---------------------------------------------------------------------------------------
# Stim oracle for Clifford circuits
# ---------------------------------------------------------------------------------------
def to_stim(text):
    """PAULI_EXPECTATION(lbl) P rec[..]  ->  MPP P ; OBSERVABLE_INCLUDE(O+r) rec[-1] rec[..]."""
    recs, _ = parse_records(text)
    O = 1 + max([int(a) for k, a, _ in recs if k == "OBSERVABLE_INCLUDE"] or [-1])
    out, x_n, s_n, r = [], 0, 0, 0
    xmap = {}                                   # xtim abs record -> stim abs record
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        name, arg, tg = _split(line)
        if name == "PAULI_EXPECTATION":
            P = [t for t in tg if not t.startswith("rec[")]
            frame = [x_n + int(t[4:-1]) for t in tg if t.startswith("rec[")]
            out.append("MPP " + " ".join(P))
            s_n += 1
            tgts = [s_n - 1] + [xmap[a] for a in frame]
            out.append(f"OBSERVABLE_INCLUDE({O + r}) " + " ".join(
                f"rec[{a - s_n}]" for a in tgts))
            r += 1
            continue
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            abs_ = [x_n + int(t[4:-1]) for t in tg if t.startswith("rec[")]
            head = name + (f"({arg})" if arg is not None else "")
            out.append(head + " " + " ".join(f"rec[{xmap[a] - s_n}]" for a in abs_))
            continue
        k = _num_records(name, tg)
        for i in range(k):
            xmap[x_n + i] = s_n + i
        x_n += k
        s_n += k
        out.append(line)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------
# single-fault enumeration on the exact engine
# ---------------------------------------------------------------------------------------
_PC1 = ["X", "Y", "Z"]
_PC2 = [(a, b) for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I")]


def channel_alternatives(name, arg, targets):
    """[(targets_of_this_channel, [(prob, {q: pauli})...]), ...] — one entry per qubit / pair."""
    ps = [float(x) for x in arg.split(",")] if arg else []
    qs = [int(t) for t in targets]
    out = []
    if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        for q in qs:
            out.append(([q], [(ps[0], {q: name[0]})]))
    elif name in ("DEPOLARIZE1", "PAULI_CHANNEL_1"):
        for q in qs:
            pr = [ps[0] / 3] * 3 if name == "DEPOLARIZE1" else ps
            out.append(([q], [(pr[i], {q: _PC1[i]}) for i in range(3)]))
    elif name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
        for a, b in zip(qs[0::2], qs[1::2]):
            pr = [ps[0] / 15] * 15 if name == "DEPOLARIZE2" else ps
            alts = []
            for i, (pa, pb) in enumerate(_PC2):
                d = {}
                if pa != "I":
                    d[a] = pa
                if pb != "I":
                    d[b] = pb
                alts.append((pr[i], d))
            out.append(([a, b], alts))
    return out


def _run_exact(text, shots, seed):
    c = xtim.Circuit(text)
    s = c.compile_detector_sampler(seed=seed, engine="exact")
    R = c.num_expectations
    res = s.sample(shots, return_measurements=True, return_expectations=R > 0)
    res = res if isinstance(res, tuple) else (res,)
    meas = np.asarray(res[-1], dtype=np.uint8)
    exps = np.asarray(res[-2]) if R > 0 else np.zeros((shots, 0))
    return meas, exps


def _signatures(text, meas, exps, ref_par, ref_exp):
    """Per shot: (frozenset of flipped targets in DEM numbering, magnitude_changed)."""
    recs, _ = parse_records(text)
    D = sum(1 for k, *_ in recs if k == "DETECTOR")
    O = 1 + max([int(a) for k, a, _ in recs if k == "OBSERVABLE_INCLUDE"] or [-1])
    R = sum(1 for k, *_ in recs if k == "PAULI_EXPECTATION")
    par = _parities(recs, meas, D, O)
    sigs = []
    for i in range(meas.shape[0]):
        s = set(np.flatnonzero(par[i] ^ ref_par).tolist())
        mag = False
        for r in range(R):
            v, v0 = exps[i, r], ref_exp[r]
            if abs(v0) < 1e-9:              # sign undefined on a magnitude-0 column
                continue
            if abs(abs(v) - abs(v0)) > 1e-9:
                mag = True
            elif (v < 0) != (v0 < 0):
                s.add(D + O + r)
        sigs.append((frozenset(s), mag))
    return sigs


def _parities(recs, meas, D, O):
    """bool[shots, D+O] (detectors, then observables by label)."""
    cols = np.zeros((meas.shape[0], D + O), dtype=np.uint8)
    d = 0
    for kind, arg, rr in recs:
        if kind == "DETECTOR":
            for a in rr:
                cols[:, d] ^= meas[:, a]
            d += 1
        elif kind == "OBSERVABLE_INCLUDE":
            for a in rr:
                cols[:, D + int(arg)] ^= meas[:, a]
    return cols


def _one_hot_channel(d, qs, p):
    """The single Pauli alternative `d` ({q: P}) as a one-hot PAULI_CHANNEL_1/2 on `qs`."""
    if len(qs) == 1:
        ps = [0.0, 0.0, 0.0]
        ps[_PC1.index(d[qs[0]])] = p
        return f"PAULI_CHANNEL_1({','.join(repr(x) for x in ps)}) {qs[0]}"
    a, b = qs
    ps = [0.0] * 15
    ps[_PC2.index((d.get(a, "I"), d.get(b, "I")))] = p
    return f"PAULI_CHANNEL_2({','.join(repr(x) for x in ps)}) {a} {b}"


def single_fault_check(text, shots=1200, seed=3):
    """Exact single-fault enumeration against the exporter.

    Per Pauli ALTERNATIVE (isolated as a one-hot PAULI_CHANNEL_1/2, p = 0.1):
      * oracle sees a per-shot magnitude change with no detector firing on some shot
        -> the exporter must refuse (XtimDemError "non-fault-tolerant");
      * exporter flags it (postselect_faults) -> sound only if the oracle shows that on every
        shot where no detector fires, every column equals its noiseless value;
      * otherwise the exported DEM distribution == the oracle distribution.
    Per CHANNEL (the authored instruction on one qubit / pair), when none of its alternatives
    is refused or flagged: exported DEM distribution == oracle mixture (checks the exact
    within-channel solve). Returns [(line, qubits, outcome)]."""
    lines = [l.split("#")[0].strip() for l in text.strip().splitlines()]
    lines = [l for l in lines if l]
    noise_idx = [i for i, l in enumerate(lines) if _split(l)[0] in NOISE]
    base_lines = [l for i, l in enumerate(lines) if i not in noise_idx]
    base = "\n".join(base_lines) + "\n"
    recs, _ = parse_records(base)
    D = sum(1 for k, *_ in recs if k == "DETECTOR")
    O = 1 + max([int(a) for k, a, _ in recs if k == "OBSERVABLE_INCLUDE"] or [-1])
    meas0, exps0 = _run_exact(base, 400, seed)
    par0 = _parities(recs, meas0, D, O)
    assert (par0 == par0[0]).all(), "noiseless detector/observable parities not deterministic"
    ref_par = par0[0]
    ref_exp = exps0[0] if exps0.shape[1] else np.zeros(0)
    assert np.allclose(exps0, ref_exp, atol=1e-9), "noiseless expectations not shot-constant"
    dets = set(range(D))
    # a magnitude-0 column has no sign: its DEM bit is vacuous, marginalize it out
    zero = frozenset(D + O + r for r in range(len(ref_exp)) if abs(ref_exp[r]) < 1e-9)

    def marg(dist):
        out = {}
        for k, v in dist.items():
            out[k - zero] = out.get(k - zero, 0.0) + v
        return out
    tol1 = 5 * math.sqrt(0.25 / shots) + 1e-12
    report = []

    def with_line(pos, line):
        fl = list(base_lines)
        fl.insert(pos, line)
        return "\n".join(fl) + "\n"

    for li in noise_idx:
        name, arg, tg = _split(lines[li])
        pos = li - sum(1 for j in noise_idx if j < li)
        for qs, alts in channel_alternatives(name, arg, tg):
            oracle, ptot, special = {}, 0.0, False
            for p, d in alts:
                if p <= 0:
                    continue
                ptot += p
                fl = list(base_lines)
                fl[pos:pos] = [f"{P} {q}" for q, P in sorted(d.items())]
                meas, exps = _run_exact("\n".join(fl) + "\n", shots, seed + 1)
                sigs = _signatures(base, meas, exps, ref_par, ref_exp)
                one = _one_hot_channel(d, qs, 0.1)
                c1 = xtim.Circuit(with_line(pos, one))
                # a refusal is sound only if, on some shot where no detector fires, a column
                # differs from its noiseless value (magnitude, or sign of a random effect)
                random_eff = len(set(sigs)) > 1 or any(m for _, m in sigs)
                silent_bad = any((m or (s - dets)) and not (s & dets) for s, m in sigs)
                try:
                    r = c1.detector_error_model_with_reject(include_expectations=True)
                except XtimDemError as e:
                    assert "non-fault-tolerant" in str(e), (lines[li], d, str(e))
                    assert silent_bad and random_eff, (
                        "exporter refused an alternative the oracle finds harmless on the "
                        "accepted branch", lines[li], d, str(e))
                    special = True
                    report.append((lines[li], d, "refused"))
                    continue
                assert not any(m and not (s & dets) for s, m in sigs), (
                    "silent magnitude change neither refused nor flagged", lines[li], d)
                if r.postselect_faults:
                    # flagged: sound iff harmless whenever no detector fires
                    assert any(s & dets for s, _ in sigs), ("flagged but never detected", lines[li], d)
                    for s, m in sigs:
                        if not (s & dets):
                            assert not m and not (s - dets), (
                                "flagged fault is a silent logical error on the accepted branch",
                                lines[li], d, sorted(s))
                    special = True
                    report.append((lines[li], d, "flagged"))
                    continue
                assert not any(m for _, m in sigs), (
                    "exporter emitted a magnitude-changing alternative", lines[li], d)
                one_or = {}
                for s, _ in sigs:
                    one_or[s] = one_or.get(s, 0.0) + 0.1 / len(sigs)
                    oracle[s] = oracle.get(s, 0.0) + p / len(sigs)
                one_or[frozenset()] = one_or.get(frozenset(), 0.0) + 0.9
                got1 = marg(dem_distribution(r.dem, D))
                for k in set(one_or) | set(got1):
                    assert abs(one_or.get(k, 0.0) - got1.get(k, 0.0)) <= 0.1 * tol1, (
                        lines[li], d, sorted(k), one_or.get(k, 0.0), got1.get(k, 0.0), str(r.dem))
                report.append((lines[li], d, "ok"))
            if special:
                continue
            iso_text = with_line(pos, f"{name}({arg}) " + " ".join(str(q) for q in qs))
            oracle[frozenset()] = oracle.get(frozenset(), 0.0) + (1.0 - ptot)
            got = marg(dem_distribution(xtim.Circuit(iso_text).detector_error_model(), D))
            tol = ptot * tol1
            for k in set(oracle) | set(got):
                assert abs(oracle.get(k, 0.0) - got.get(k, 0.0)) <= tol, (
                    lines[li], qs, sorted(k), oracle.get(k, 0.0), got.get(k, 0.0),
                    str(xtim.Circuit(iso_text).detector_error_model()))
    return report


# ---------------------------------------------------------------------------------------
# 1. the reported reproducer and its readout twin
# ---------------------------------------------------------------------------------------
REPRO = "R 0 1\nH 0\nCX 0 1\nZ_ERROR(0.2) 0\nMX 0\nPAULI_EXPECTATION(0) X1 rec[-1]\n"


def test_reproducer_gate_fault_through_frame_is_emitted():
    assert canon(xtim.Circuit(REPRO).detector_error_model()) == {
        (("L", 0),): pytest.approx(0.2, rel=1e-15)}


def test_reproducer_matches_readout_twin_and_stim():
    twin = "R 0 1\nH 0\nCX 0 1\nMX(0.2) 0\nPAULI_EXPECTATION(0) X1 rec[-1]\n"
    assert_dem_equal(xtim.Circuit(REPRO).detector_error_model(),
                     xtim.Circuit(twin).detector_error_model())
    assert_dem_equal(xtim.Circuit(REPRO).detector_error_model(),
                     stim.Circuit(to_stim(REPRO)).detector_error_model())


def test_reproducer_single_fault_oracle():
    single_fault_check(REPRO)


# ---------------------------------------------------------------------------------------
# 2. Clifford family vs Stim: gate faults before M/MX/MY/MPP whose records enter frames
# ---------------------------------------------------------------------------------------
BELL = "R 0 1\nH 0\nCX 0 1\n"
GHZ3 = "R 0 1 2\nH 0\nCX 0 1\nCX 0 2\n"
GHZ4 = "R 0 1 2 3\nH 0\nCX 0 1\nCX 0 2\nCX 0 3\n"
_BASIS_P = {"M": "Z", "MX": "X", "MY": "Y"}
FAMILY = {}
for meas, P in _BASIS_P.items():
    for err in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        for q in (0, 1):
            FAMILY[f"bell_{err}{q}_{meas}"] = (
                BELL + f"{err}(0.13) {q}\n{meas} 0\nPAULI_EXPECTATION(0) {P}1 rec[-1]\n")
    FAMILY[f"bell_dep1_{meas}"] = (
        BELL + f"DEPOLARIZE1(0.09) 0 1\n{meas} 0\nPAULI_EXPECTATION(0) {P}1 rec[-1]\n")
    FAMILY[f"bell_dep2_{meas}"] = (
        BELL + f"DEPOLARIZE2(0.07) 0 1\n{meas} 0\nPAULI_EXPECTATION(0) {P}1 rec[-1]\n")
    FAMILY[f"bell_pc1_{meas}"] = (
        BELL + f"PAULI_CHANNEL_1(0.01,0.02,0.03) 0\n{meas} 0\nPAULI_EXPECTATION(0) {P}1 rec[-1]\n")
FAMILY.update({
    # MPP record in the frame
    "ghz3_mpp_xx": GHZ3 + "DEPOLARIZE1(0.06) 0 1 2\nMPP X0*X1\nPAULI_EXPECTATION(0) X2 rec[-1]\n",
    "ghz3_mpp_zz_pc2": GHZ3 + "PAULI_CHANNEL_2(" + ",".join(f"{0.002*(i+1):.3f}" for i in range(15))
    + ") 0 2\nMPP Z0*Z1\nPAULI_EXPECTATION(0) Z0*Z2 rec[-1]\n",
    # chained records: three records in one frame, and a frame that skips a record
    "ghz4_chain3": GHZ4 + "DEPOLARIZE1(0.05) 0 1 2 3\nMX 0 1 2\n"
    "PAULI_EXPECTATION(0) X3 rec[-1] rec[-2] rec[-3]\n",
    "ghz4_skip": GHZ4 + "R 4\nDEPOLARIZE2(0.04) 0 1 2 3\nX_ERROR(0.1) 4\nMX 0 1\nM 4\nMX 2\n"
    "PAULI_EXPECTATION(0) X3 rec[-1] rec[-3] rec[-4]\n",
    # two columns sharing a record, with a detector and an observable alongside
    "two_cols_mixed": GHZ3 + "R 3 4 5 6\nH 3 5\nCX 3 4 5 6\nMPP Z0*Z1\nDETECTOR rec[-1]\n"
    "DEPOLARIZE1(0.04) 0 1 2 3 4 5 6\nZ_ERROR(0.03) 0\nMX 0 1\nMX 3\nMX 5 6\n"
    "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\n"
    "PAULI_EXPECTATION(0) X2 rec[-4] rec[-5]\nPAULI_EXPECTATION(1) X2*X4 rec[-3] rec[-4] rec[-5]\n",
    # a fault between two frame measurements (acts on the second only)
    "fault_between_reads": GHZ3 + "MX 0\nZ_ERROR(0.11) 1\nX_ERROR(0.05) 1\nMX 1\n"
    "PAULI_EXPECTATION(0) X2 rec[-1] rec[-2]\n",
    # Y record frame and a Y-containing column
    "y_frame": BELL + "DEPOLARIZE1(0.08) 0\nMY 0\nPAULI_EXPECTATION(0) Y1 rec[-1]\n",
    # frame record plus a noisy readout on another frame record
    "readout_and_gate": GHZ3 + "Z_ERROR(0.1) 0\nMX(0.05) 0\nMX 1\n"
    "PAULI_EXPECTATION(0) X2 rec[-1] rec[-2]\n",
    # MR in the frame (reset afterwards; the column is on another qubit)
    "mr_frame": BELL + "X_ERROR(0.1) 0\nMR 0\nPAULI_EXPECTATION(0) Z1 rec[-1]\n",
    # expectation BEFORE further gates on the frame wire's partner
    "gate_after_frame_read": BELL + "Z_ERROR(0.1) 0\nMX 0\nH 1\n"
    "PAULI_EXPECTATION(0) Z1 rec[-1]\n",
})


@pytest.mark.parametrize("name", sorted(FAMILY))
def test_clifford_family_matches_stim(name):
    text = FAMILY[name]
    if "PAULI_CHANNEL_2" in text:
        # Stim only approximates PAULI_CHANNEL_2 (approximate_disjoint_errors); xtim solves the
        # channel exactly. The exact single-fault oracle covers it instead.
        single_fault_check(text)
        return
    assert_dem_equal(xtim.Circuit(text).detector_error_model(),
                     stim.Circuit(to_stim(text)).detector_error_model())


@pytest.mark.parametrize("name", sorted(FAMILY)[::4])
def test_clifford_family_single_fault_oracle(name):
    single_fault_check(FAMILY[name])


def _random_clifford_case(rng, n=5):
    lines = [f"R {' '.join(map(str, range(n)))}"]
    noise_ops = ["X_ERROR(0.03) {a}", "Y_ERROR(0.02) {a}", "Z_ERROR(0.04) {a}",
                 "DEPOLARIZE1(0.05) {a}", "DEPOLARIZE2(0.06) {a} {b}",
                 "PAULI_CHANNEL_1(0.01,0.02,0.03) {a}"]
    for _ in range(14):
        g = rng.choice(["H", "S", "CX", "CX", "CZ", "SQRT_X", "noise"])
        a, b = rng.sample(range(n), 2)
        if g == "noise":
            lines.append(rng.choice(noise_ops).format(a=a, b=b))
        elif g in ("CX", "CZ"):
            lines.append(f"{g} {a} {b}")
        else:
            lines.append(f"{g} {a}")
    k = rng.randint(1, 3)
    for q in range(k):
        lines.append(rng.choice(["M", "MX", "MY"]) + f" {q}")
        if rng.random() < 0.5:
            lines.append(rng.choice(noise_ops).format(a=rng.randrange(k, n), b=0))
    body = "\n".join(lines) + "\n"
    c = xtim.Circuit(body)
    cols = []
    for _ in range(30):
        P = "*".join(f"{rng.choice('XYZ')}{q}" for q in sorted(rng.sample(range(k, n), rng.randint(1, n - k))))
        try:
            fr = c.extract_frame(P)
        except Exception:
            continue
        cols.append((P, fr))
        if len(cols) == 2:
            break
    if not cols:
        return None
    for i, (P, fr) in enumerate(cols):
        body += f"PAULI_EXPECTATION({i}) {P}" + "".join(f" rec[-{f}]" for f in fr) + "\n"
    try:                                        # keep only columns that are +-1 (Stim-expressible)
        stim.Circuit(to_stim(body)).detector_error_model()
    except Exception:
        return None
    return body


def test_random_clifford_frames_match_stim():
    rng = random.Random(20261009)
    done = framed = 0
    while done < 150:
        text = _random_clifford_case(rng)
        if text is None:
            continue
        done += 1
        framed += "rec[" in text
        assert_dem_equal(xtim.Circuit(text).detector_error_model(),
                         stim.Circuit(to_stim(text)).detector_error_model())
    assert framed >= 40, framed


# ---------------------------------------------------------------------------------------
# 3. magic: the T variant must refuse loudly; a frame-carried Pauli flip must be emitted;
#    a detected magnitude change must be flagged
# ---------------------------------------------------------------------------------------
MAGIC = {
    # X fault on the measured qubit before T: S-type residual on the frame read; the
    # post-measurement q1 goes from X to Y (|<X1>| 1 -> 0) and no detector exists -> refuse
    "t_bell_x0_refuse": BELL + "X_ERROR(0.2) 0\nT 0\nT_DAG 1\nMX 0\nPAULI_EXPECTATION(0) X1 rec[-1]\n",
    # Z fault commutes with T: a clean flip of the frame record -> error(p) L0
    "t_bell_z0_emit": BELL + "Z_ERROR(0.2) 0\nT 0\nT_DAG 1\nMX 0\nPAULI_EXPECTATION(0) X1 rec[-1]\n",
    "t_bell_z0_after_t": BELL + "T 0\nT_DAG 1\nZ_ERROR(0.2) 0\nMX 0\nPAULI_EXPECTATION(0) X1 rec[-1]\n",
    # the X fault is caught by a Z0Z1 check -> flagged for post-selection, not refused
    "t_bell_x0_detected": BELL + "X_ERROR(0.2) 0\nMPP Z0*Z1\nDETECTOR rec[-1]\nT 0\nT_DAG 1\n"
    "MX 0\nPAULI_EXPECTATION(0) X1 rec[-1]\n",
    # magic magnitude column (|<P>| = 1/sqrt2) with a frame; Z faults on the frame wire
    "t_teleport_frame": BELL + "T 1\nZ_ERROR(0.1) 0\nDEPOLARIZE1(0.03) 1\nMX 0\n"
    "PAULI_EXPECTATION(0) X1 rec[-1]\n",
    "t_bell_x1_refuse": BELL + "X_ERROR(0.2) 1\nT 0\nT_DAG 1\nMX 0\nPAULI_EXPECTATION(0) X1 rec[-1]\n",
    # one correlated fault X0 X2 rotates the frame read (column magnitude 1 -> 0) AND randomizes
    # an unrelated detector (X2 X3 of a second Bell pair): no deterministic detector, and on the
    # accepted branch (D0 = 0) the column is still destroyed -> refuse (accepted-branch check)
    # anti-correlated: X on the magic wire before an XS check flips (D0, column) together or
    # not at all; Z on an independent |+> flips the column part X5 always. One alternative
    # X4 Z5: either D0 fires (column fine) or nothing fires and the column SIGN is flipped ->
    # a silent logical error on the accepted branch with an unchanged magnitude -> refuse
    "xs_check_accepted_branch_sign_refuse": "RX 4 5\nT 4\n"
    "PAULI_CHANNEL_2(0,0,0,0,0,0.1,0,0,0,0,0,0,0,0,0) 4 5\n"
    "RX 3\nT 4\nS_DAG 4\nCX 3 4\nT 4\nMX 3\nDETECTOR rec[-1]\nPAULI_EXPECTATION(0) X4*X5\n",
    # same check, X fault only: (D0, column) flip together -> flagged, harmless when accepted
    "xs_check_flagged": "RX 4 5\nT 4\nX_ERROR(0.1) 4\n"
    "RX 3\nT 4\nS_DAG 4\nCX 3 4\nT 4\nMX 3\nDETECTOR rec[-1]\nPAULI_EXPECTATION(0) X4*X5\n",
    # the same with the check's noiseless record = 1 (a Pauli Z before MX): the accepted branch
    # is "D0 reads its noiseless value", not "the raw parity is 0"
    "xs_check_flagged_neg_detector": "RX 4 5\nT 4\nX_ERROR(0.1) 4\n"
    "RX 3\nT 4\nS_DAG 4\nCX 3 4\nT 4\nZ 3\nMX 3\nDETECTOR rec[-1]\nPAULI_EXPECTATION(0) X4*X5\n",
    "t_two_bells_accepted_branch_refuse": "R 0 1 2 3\nH 0 2\nCX 0 1 2 3\n"
    "PAULI_CHANNEL_2(0,0,0,0,0.1,0,0,0,0,0,0,0,0,0,0) 0 2\nT 0 2\nT_DAG 1 3\nMX 0 2 3\n"
    "DETECTOR rec[-1] rec[-2]\nPAULI_EXPECTATION(0) X1 rec[-3]\n",
}


def test_magic_t_variant_refuses_loudly():
    c = xtim.Circuit(MAGIC["t_bell_x0_refuse"])
    with pytest.raises(XtimDemError, match="non-fault-tolerant"):
        c.detector_error_model()
    with pytest.raises(XtimDemError, match="non-fault-tolerant"):
        c.detector_error_model_with_reject()
    with pytest.raises(XtimDemError, match="non-fault-tolerant"):
        c.detector_error_model(include_expectations=False)


def test_accepted_branch_sign_flip_refused():
    c = xtim.Circuit(MAGIC["xs_check_accepted_branch_sign_refuse"])
    with pytest.raises(XtimDemError, match="sign on the accepted branch"):
        c.detector_error_model_with_reject(include_expectations=True)
    r = xtim.Circuit(MAGIC["xs_check_flagged"]).detector_error_model_with_reject()
    assert len(r.postselect_faults) == 1 and r.postselect_faults[0].detectors == [0]


def test_accepted_branch_refusal_message():
    c = xtim.Circuit(MAGIC["t_two_bells_accepted_branch_refuse"])
    with pytest.raises(XtimDemError, match="undetectable on the accepted branch"):
        c.detector_error_model_with_reject()


def test_magic_frame_flip_emitted():
    for name in ("t_bell_z0_emit", "t_bell_z0_after_t"):
        assert canon(xtim.Circuit(MAGIC[name]).detector_error_model()) == {
            (("L", 0),): pytest.approx(0.2, rel=1e-15)}, name


@pytest.mark.parametrize("name", sorted(MAGIC))
def test_magic_single_fault_oracle(name):
    single_fault_check(MAGIC[name])


def test_example_miniature_oracle_every_fault_exact():
    """The bundled MSP composition oracle (teleport byproduct = declared frame rec[-3], magic
    equator columns |<P>| = 1/sqrt2). 3.1.9 refused its DEM ("negative probability": the frame
    read was modelled as an independent coin and the frame-carried flips were missing on five
    channels). Every Pauli alternative of every channel is now checked against the exact engine;
    the magic-wire X/Y faults before the XS checks are flagged (they randomize D0/D1 together
    with the column sign) and are harmless on the accepted branch."""
    rep = single_fault_check(xtim.load_example("miniature_oracle").text, shots=800)
    outcomes = [o for *_, o in rep]
    assert "refused" not in outcomes
    assert outcomes.count("flagged") >= 2
    xtim.load_example("miniature_oracle").detector_error_model()   # exports (3.1.9 refused)


def test_corpus_hand_magic_frame_every_fault_exact():
    single_fault_check("R 0 1\nH 0\nCX 0 1\nT 1\nX_ERROR(0.01) 0 1\nMX 0\n"
                       "PAULI_EXPECTATION(0) Z1 rec[-1]\n")


# ---------------------------------------------------------------------------------------
# 4. sweep: the other record-defined columns already fold gate faults through records
# ---------------------------------------------------------------------------------------
def test_observable_include_with_records_matches_stim():
    text = GHZ3 + "DEPOLARIZE1(0.05) 0 1 2\nMX 0 1 2\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]\n"
    assert_dem_equal(xtim.Circuit(text).detector_error_model(),
                     stim.Circuit(text).detector_error_model())
    single_fault_check(text)


def test_decision_column_folds_gate_fault_through_record():
    # DECISION columns are emitted after expectations and OUTPUT_QUBITS frame columns:
    # here D=0, O=0, R=1, NOUT=0 -> decision 0 is L1. The Z fault flips the MX record,
    # hence both the expectation column (via its frame) and the decision.
    text = BELL + "Z_ERROR(0.2) 0\nMX 0\nDECISION(0) rec[-1]\nPAULI_EXPECTATION(0) X1 rec[-1]\n"
    r = xtim._xtim.export_dem_text(text, True, False, False, False)
    assert r["ok"], r
    assert canon(stim.DetectorErrorModel(r["dem"])) == {
        (("L", 0), ("L", 1)): pytest.approx(0.2, rel=1e-15)}


def test_feedback_is_refused_by_dem_export():
    text = BELL + "Z_ERROR(0.2) 0\nMX 0\nCZ rec[-1] 1\nPAULI_EXPECTATION(0) X1\n"
    with pytest.raises(Exception, match="feedback"):
        xtim.Circuit(text).detector_error_model()


# ---------------------------------------------------------------------------------------
# 5. Semantics of an expectation L-flip for ANY noiseless value (owner decision, 3.1.10):
#    an L-flip on column r means "the fault's propagated action multiplies P'_r = P_r . (frame
#    record Paulis) by -1", i.e. multiplies the reported <P> by -1 (records relabelled by the
#    fault's record flips). Columns whose noiseless value is 0, 1/sqrt2 or branch-dependent are
#    legitimate (logical Paulis of a magic state) and are NOT refused.
#    Oracle A (operator): stim Pauli-frame propagation of the fault to every record and to the
#    PAULI_EXPECTATION line (T/T_DAG pass a Z frame, a frame with X/Y on a T wire is non-Pauli).
#    Oracle B (exact engine, per shot): the faulted (records m, value v) satisfy
#    v == (-1)^s * v0(m XOR f) where v0 is the noiseless value table keyed by the full record
#    and (f, s) are oracle A's record flips and column flip.
# ---------------------------------------------------------------------------------------
_UNITARY_SKIP = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "DECISION", "QUBIT_COORDS"}


def _propagate(lines, start, fault, n):
    """Oracle A. Returns (record_flips list, {column index: flip}) or None if non-Pauli."""
    F = stim.PauliString(n)
    for q, P in fault.items():
        F[q] = P
    recs, cols, r = [], {}, 0
    for line in lines[start:]:
        name, arg, tg = _split(line)
        if name in NOISE or name in _UNITARY_SKIP:
            continue
        if name in ("T", "T_DAG"):
            if any(F[int(q)] in (1, 2) for q in tg):     # X or Y on a T wire: Clifford residual
                return None
            continue
        if name in ("M", "MX", "MY", "MR", "MRX", "MRY", "MZ"):
            b = {"M": "Z", "MZ": "Z", "MR": "Z", "MX": "X", "MRX": "X", "MY": "Y", "MRY": "Y"}[name]
            for q in tg:
                q = int(q)
                m = stim.PauliString(n); m[q] = b
                recs.append(0 if F.commutes(m) else 1)
                if name.startswith("MR"):
                    F[q] = 0
            continue
        if name == "MPP":
            for prod in tg:
                m = stim.PauliString(n)
                for term in prod.split("*"):
                    m[int(term[1:])] = term[0]
                recs.append(0 if F.commutes(m) else 1)
            continue
        if name in ("R", "RX", "RY"):
            for q in tg:
                F[int(q)] = 0
            continue
        if name == "PAULI_EXPECTATION":
            P = stim.PauliString(n)
            frame = []
            for t in tg:
                if t.startswith("rec["):
                    frame.append(len(recs) + int(t[4:-1]))
                else:
                    for term in t.split("*"):
                        P[int(term[1:])] = term[0]
            flip = 0 if F.commutes(P) else 1
            for a in frame:
                flip ^= recs[a]
            cols[r] = flip
            r += 1
            continue
        F = F.after(stim.Circuit(line))
    return recs, cols


def _value_table(text, shots=3000, seed=11):
    meas, exps = _run_exact(text, shots, seed)
    tab = {}
    for m, v in zip(meas, exps):
        k = tuple(int(b) for b in m)
        if k in tab:
            assert np.allclose(tab[k], v, atol=1e-12), "noiseless value not a function of the records"
        tab[k] = v
    return tab


def semantics_check(text, n):
    """Every Pauli alternative of every channel: oracle A's column flips == the exported
    single-alternative DEM's L bits (and its detector bits); oracle B on the exact engine.
    Non-Pauli alternatives are returned for the caller to classify."""
    lines = [l.split("#")[0].strip() for l in text.strip().splitlines()]
    lines = [l for l in lines if l]
    noise_idx = [i for i, l in enumerate(lines) if _split(l)[0] in NOISE]
    base_lines = [l for i, l in enumerate(lines) if i not in noise_idx]
    base = "\n".join(base_lines) + "\n"
    recs, _ = parse_records(base)
    dets = [rr for k, a, rr in recs if k == "DETECTOR"]
    D = len(dets)
    O = 1 + max([int(a) for k, a, _ in recs if k == "OBSERVABLE_INCLUDE"] or [-1])
    v0 = _value_table(base)
    nonpauli = []
    checked = 0
    for li in noise_idx:
        name, arg, tg = _split(lines[li])
        pos = li - sum(1 for j in noise_idx if j < li)
        for qs, alts in channel_alternatives(name, arg, tg):
            for p, d in alts:
                if p <= 0:
                    continue
                prop = _propagate(base_lines, pos, d, n)
                fl = list(base_lines)
                one = _one_hot_channel(d, qs, 0.1)
                if prop is None:
                    nonpauli.append((lines[li], d, "\n".join(fl[:pos] + [one] + fl[pos:]) + "\n"))
                    continue
                rflips, cflips = prop
                want = {dd for dd in range(D) if sum(rflips[a] for a in dets[dd]) & 1}
                want |= {D + O + c for c, f in cflips.items() if f}
                r = xtim.Circuit("\n".join(fl[:pos] + [one] + fl[pos:]) + "\n") \
                    .detector_error_model_with_reject(include_expectations=True)
                assert not r.postselect_faults, (lines[li], d)
                got = canon(r.dem)
                gotsig = {(v if t == "D" else D + v) for k in got for t, v in k}
                assert len(got) <= 1 and gotsig == want, (lines[li], d, sorted(want), got)
                # oracle B: per shot on the exact engine
                fl[pos:pos] = [f"{P} {q}" for q, P in sorted(d.items())]
                meas, exps = _run_exact("\n".join(fl) + "\n", 600, 5)
                for m, v in zip(meas, exps):
                    k = tuple(int(b) ^ rflips[i] for i, b in enumerate(m))
                    assert k in v0, ("faulted record pattern never seen noiselessly", lines[li], d)
                    sgn = np.array([-1.0 if cflips[c] else 1.0 for c in range(len(v))])
                    assert np.allclose(v, sgn * v0[k], atol=1e-9), (lines[li], d, m, v, v0[k])
                checked += 1
    return checked, nonpauli


# Bell pair with T on q0: <X0X1> = <Y0X1> = 1/sqrt2, <Z0Z1> = 1, <X0> = <Z0> = 0
TBELL = ("R 0 1\nH 0\nCX 0 1\nT 0\n{noise}"
         "PAULI_EXPECTATION(0) X0*X1\nPAULI_EXPECTATION(1) Y0*X1\nPAULI_EXPECTATION(2) Z0*Z1\n"
         "PAULI_EXPECTATION(3) X0\nPAULI_EXPECTATION(4) Z0\n")
# one-bit teleport of T|+> from q0 to q1: with frame <X1>,<Y1> = 1/sqrt2 (sign-constant);
# without frame they are branch-dependent (+-1/sqrt2 by the MX record, average 0); <Z1> = 0
TELE = ("RX 0\nT 0\n{pre}R 1\nCX 0 1\n{noise}MX 0\n"
        "PAULI_EXPECTATION(0) X1 rec[-1]\nPAULI_EXPECTATION(1) Y1 rec[-1]\n"
        "PAULI_EXPECTATION(2) X1\nPAULI_EXPECTATION(3) Y1\nPAULI_EXPECTATION(4) Z1\n")


def test_semantics_values_are_what_the_tests_claim():
    _, exps = _run_exact(TBELL.format(noise=""), 200, 1)
    assert np.allclose(exps[0], [2 ** -0.5, 2 ** -0.5, 1.0, 0.0, 0.0], atol=1e-12)
    _, exps = _run_exact(TELE.format(pre="", noise=""), 400, 1)
    assert np.allclose(exps[:, 0], 2 ** -0.5) and np.allclose(exps[:, 4], 0.0)
    assert set(np.round(exps[:, 2], 9)) == {round(2 ** -0.5, 9), -round(2 ** -0.5, 9)}
    c = xtim.Circuit(TELE.format(pre="", noise=""))
    assert c.num_expectations == 5 and c.detector_error_model_text().strip() != ""


@pytest.mark.parametrize("noise", [
    "DEPOLARIZE1(0.06) 0 1\n", "DEPOLARIZE2(0.06) 0 1\n",
    "PAULI_CHANNEL_2(" + ",".join(f"{0.002 * (i + 1):.3f}" for i in range(15)) + ") 1 0\n"])
def test_semantics_bell_t_columns_0_halfroot2_1(noise):
    checked, nonpauli = semantics_check(TBELL.format(noise=noise), 2)
    assert checked >= 3 and not nonpauli


@pytest.mark.parametrize("noise", [
    "DEPOLARIZE1(0.05) 0 1\n", "DEPOLARIZE2(0.05) 1 0\n", "Z_ERROR(0.1) 0\nX_ERROR(0.1) 1\n"])
def test_semantics_teleport_with_and_without_frame(noise):
    checked, nonpauli = semantics_check(TELE.format(pre="", noise=noise), 2)
    assert checked >= 2 and not nonpauli


def test_semantics_z_fault_before_t_is_pauli():
    # Z commutes with T: propagated as a Pauli through the magic gate
    checked, nonpauli = semantics_check(
        "RX 0\nZ_ERROR(0.1) 0\nT 0\nR 1\nCX 0 1\nMX 0\n"
        "PAULI_EXPECTATION(0) X1 rec[-1]\nPAULI_EXPECTATION(1) X1\nPAULI_EXPECTATION(2) Z1\n", 2)
    assert checked == 1 and not nonpauli


def test_clifford_residual_on_a_zero_column_is_refused_or_flagged():
    """X before T on the Bell pair leaves an S-type residual on q0: its action on X0 (value 0)
    is not +-1. 3.1.9 emitted it as 'no flip' on that column (the classifier read a 0 average);
    now it is refused with no detector and flagged when a Z0Z1 check catches it."""
    text = TBELL.format(noise="").replace("T 0\n", "X_ERROR(0.1) 0\nT 0\n")
    _, nonpauli = semantics_check(text, 2)
    assert len(nonpauli) == 1
    with pytest.raises(XtimDemError, match="non-fault-tolerant"):
        xtim.Circuit(text).detector_error_model()
    caught = text.replace("T 0\n", "MPP Z0*Z1\nDETECTOR rec[-1]\nT 0\n")
    r = xtim.Circuit(caught).detector_error_model_with_reject(include_expectations=True)
    assert [f.detectors for f in r.postselect_faults] == [[0]]
    assert canon(r.dem) == {}
