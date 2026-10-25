"""Stim's correlated-error chains: `E` / `CORRELATED_ERROR` + `ELSE_CORRELATED_ERROR`.

A chain is ONE noise site with a categorical Pauli law over arbitrary Pauli supports: term t
fires with probability q_t = p_t * prod_{j<t} (1 - p_j) (Stim's sequential-conditional
semantics; exactly one term or none fires). Oracles, in order of strength:

* real Stim (Clifford circuits): the DEM under `approximate_disjoint_errors` mechanism by
  mechanism (sorted targets, 1e-12 relative), its refusals (an ELSE term needs the option; a
  component above the threshold), and detector statistics at 1e6 shots;
* the exact categorical law written out by hand (pattern probabilities), against every xtim
  sampling path (records / exact / twirl / port) at >= 4e5 shots, and against forced faults;
* the independent dense oracles (`_dense_oracle`, `_forced_oracle`) on non-Clifford circuits:
  chains before and after T / CCZ, X-type chain outcomes crossing a CCZ, forced single
  outcomes (per-outcome states) and the sampled law;
* law equivalence with the native instructions: a 15-term chain encoding DEPOLARIZE2(p) must
  sample the same law as DEPOLARIZE2 / PAULI_CHANNEL_2 on every engine, on a magic-bearing
  example circuit too.

Mutation coverage (each run red by hand on 2026-10-13): categorical probabilities q_t := p_t
(the sequential-to-categorical conversion dropped) -> the hand law / stim DEM tests; the ELSE
exclusivity dropped (every term an independent Bernoulli) -> the pattern law tests; a
mis-ordered Pauli support (term factors on the wrong support qubit) -> the deterministic
component tests and the forced-fault oracle.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

import xtim
from _orig_examples import ORIG_EXAMPLES, load_orig  # noqa: E402,F401
from xtim import scale_noise
from xtim import twirl as _twirl
from xtim.port import Consume, compile as port_compile
from _dense_oracle import Oracle
from _forced_oracle import aggregate, branches as dense_branches, law_shifts

stim = pytest.importorskip("stim")

_XPART = {"I": 0, "X": 1, "Y": 1, "Z": 0}
_ZPART = {"I": 0, "X": 0, "Y": 1, "Z": 1}

# The 4-qubit probe of test_pauli_channel_order: qubits 0,1 are Z-prepared / Z-read (detectors
# D0, D1 see the X-part of a Pauli on them), qubits 2,3 X-prepared / X-read (D2, D3 see the
# Z-part). A chain term is a word over qubits (0, 1, 2, 3).
PROBE = ("R 0 1\nRX 2 3\n{}\nM 0 1\nMX 2 3\n"
         "DETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n")


def _term(word, qubits=(0, 1, 2, 3)):
    return " ".join(f"{c}{q}" for c, q in zip(word, qubits) if c != "I")


def _chain(seq_probs, words, qubits=(0, 1, 2, 3)):
    lines = []
    for t, (p, w) in enumerate(zip(seq_probs, words)):
        lines.append(("E" if t == 0 else "ELSE_CORRELATED_ERROR") + f"({p!r}) " + _term(w, qubits))
    return "\n".join(lines)


def _categorical(seq_probs):
    q, rem = [], 1.0
    for p in seq_probs:
        q.append(p * rem)
        rem *= 1 - p
    return q


def _sequential(cat_probs):
    """The sequential arguments whose categorical law is `cat_probs`."""
    seq, rem = [], 1.0
    for q in cat_probs:
        seq.append(q / rem if rem > 0 else 0.0)
        rem -= q
    return seq


def _pattern(word):
    """Detector pattern of a probe word: (X-part q0, X-part q1, Z-part q2, Z-part q3)."""
    return (_XPART[word[0]], _XPART[word[1]], _ZPART[word[2]], _ZPART[word[3]])


def _law(seq_probs, words):
    """The exact detector-pattern law of one chain on the probe."""
    law = {}
    q = _categorical(seq_probs)
    for p, w in zip(q, words):
        k = _pattern(w)
        law[k] = law.get(k, 0.0) + p
    law[(0, 0, 0, 0)] = law.get((0, 0, 0, 0), 0.0) + 1 - sum(q)
    return law


def _counts(dets):
    out = {}
    for row in map(tuple, np.asarray(dets, dtype=int)):
        out[row] = out.get(row, 0) + 1
    return out


def _assert_law(dets, law, who, zmax=4.5):
    n = len(dets)
    cnt = _counts(dets)
    for k in set(cnt) | set(law):
        p = law.get(k, 0.0)
        c = cnt.get(k, 0)
        if p == 0.0:
            assert c == 0, f"{who}: impossible pattern {k} seen {c} times"
            continue
        sd = math.sqrt(n * p * (1 - p)) if 0 < p < 1 else 0.0
        z = (c - n * p) / sd if sd > 0 else (0.0 if c == round(n * p) else math.inf)
        assert abs(z) < zmax, f"{who}: pattern {k}: {c}/{n} vs p={p:.6g} (z={z:.2f})"


def _unpack(packed, k):
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1, bitorder="little")[:, :k].astype(bool)


def _dets_all_paths(text, shots, seed):
    """{path name: bool[shots, D]} for every xtim sampling path (the probe's detectors are
    the records themselves: both preparations are deterministic +1 eigenstates)."""
    c = xtim.Circuit(text)
    D = c.num_detectors
    out = {"compile_sampler": np.asarray(c.compile_sampler(seed=seed).sample(shots), dtype=bool)}
    out["exact"] = np.asarray(c.compile_detector_sampler(seed=seed, engine="exact").sample(shots), dtype=bool)
    out["twirl_engine"] = np.asarray(c.compile_detector_sampler(seed=seed, engine="twirl").sample(shots), dtype=bool)
    tw = _twirl.compile_twirl_sampler(text, selfcheck=0)
    d, _ = tw.sample(shots, seed=seed)
    out["compile_twirl_sampler"] = _unpack(d, D)
    out["port_bare"] = np.asarray(port_compile(text, Consume(dets=True)).run(shots=shots, seed=seed).dets, dtype=bool)
    rec = port_compile(text, Consume(dets=True, groups=True)).run(shots=shots, seed=seed)
    out["port_record"] = np.asarray(rec.dets, dtype=bool)
    return out


def _canon(dem):
    """{sorted-target-tuple: probability} of the mechanisms of a flattened DEM; two lines with
    one signature (stim keeps them separate across instructions) XOR-compose, and a decomposed
    mechanism's components are XORed back into its undecomposed signature (the decomposition
    split itself is the decomposer's choice, not the channel's law)."""
    errors = {}
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        parity = {}
        for t in ins.targets_copy():
            if not t.is_separator():
                parity[str(t)] = parity.get(str(t), 0) ^ 1
        key = tuple(sorted(k for k, v in parity.items() if v))
        p = ins.args_copy()[0]
        q = errors.get(key, 0.0)
        errors[key] = q + p - 2 * q * p
    return errors


def _assert_dem_equal(got, want, rel=1e-12):
    g, w = _canon(got), _canon(want)
    assert set(g) == set(w), (sorted(set(g) ^ set(w)))
    for k in w:
        assert g[k] == pytest.approx(w[k], rel=rel, abs=1e-300), (k, g[k], w[k])


# ── parsing: everything stim accepts, stim's chain rule, named refusals ──────────────────────

def test_parse_accepts_stim_forms_and_round_trips():
    text = PROBE.format("E(0.1) !X0\nELSE_CORRELATED_ERROR(0.5) Z2 X1\nELSE_CORRELATED_ERROR(0.25) X0 Y3 X1\n"
                        "E(0.2)\nELSE_CORRELATED_ERROR(0.3) y1\nCORRELATED_ERROR[tag](0.4) Z3")
    stim.Circuit(text)                                   # stim-valid
    c = xtim.Circuit(text)
    sites = c.compile_exact_branches().noise_sites()
    assert [s["channel"] for s in sites] == ["CORRELATED_ERROR"] * 3
    # support = distinct qubits in first-appearance order; terms as words over it
    assert sites[0]["qubits"] == [0, 2, 1, 3]
    assert sites[0]["probs"] == [0.1, 0.5, 0.25]
    assert sites[0]["paulis"] == ["XIII", "IZXI", "XIXY"]
    assert sites[1]["qubits"] == [1] and sites[1]["paulis"] == ["I", "Y"]   # the empty E term
    assert sites[2]["qubits"] == [3] and sites[2]["paulis"] == ["Z"]
    # round trip through the resolved-text emitter: same sites, same DEM
    rt = xtim._xtim.resolve_branches_text(text, [])
    assert "E(0.1) X0\nELSE_CORRELATED_ERROR(0.5) Z2 X1\nELSE_CORRELATED_ERROR(0.25) X0 X1 Y3\n" in rt
    assert "E(0.2)\nELSE_CORRELATED_ERROR(0.3) Y1\n" in rt
    _assert_dem_equal(xtim.Circuit(rt).detector_error_model(approximate_disjoint_errors=True),
                      c.detector_error_model(approximate_disjoint_errors=True))
    # a chain inside a REPEAT body, every iteration its own site
    r = xtim.Circuit(PROBE.format("REPEAT 3 {\nE(0.1) X0\nELSE_CORRELATED_ERROR(0.2) Z2\n}"))
    assert len(r.compile_exact_branches().noise_sites()) == 3


@pytest.mark.parametrize("body, why", [
    ("ELSE_CORRELATED_ERROR(0.2) Z0", "must directly follow"),
    ("E(0.1) X0\nTICK\nELSE_CORRELATED_ERROR(0.2) Z0", "must directly follow"),
    ("E(0.1) X0\nX_ERROR(0.1) 1\nELSE_CORRELATED_ERROR(0.2) Z0", "must directly follow"),
    ("E(0.1) X0\nDETECTOR\nELSE_CORRELATED_ERROR(0.2) Z0", "must directly follow"),
    ("E(0.1) X0\nREPEAT 2 {\nELSE_CORRELATED_ERROR(0.2) Z0\n}", "must directly follow"),
    ("E(0.1) 0", "only takes Pauli targets"),
    ("E(0.1) X0 rec[-1]", "only takes Pauli targets"),
    ("E(1.5) X0", "probability out of"),
    ("E(0.1, 0.2) X0", "exactly one probability"),
    ("E(0.1) X0 Y0", "repeated within one Pauli product"),
])
def test_parse_refusals_named(body, why):
    with pytest.raises(xtim.XtimParseError, match=why):
        xtim.Circuit(PROBE.format(body))


# ── the DEM: stim's mechanisms, refusals and threshold, exactly ──────────────────────────────

def _random_chain_lines(rng, nq, max_terms=6):
    terms = rng.randint(1, max_terms)
    lines = []
    for t in range(terms):
        k = rng.randint(0, min(4, nq))
        qs = rng.sample(range(nq), k)
        p = rng.choice([0.0, 1.0, rng.random() * 0.3, rng.random() * 0.3, rng.random()])
        if t == 0 and rng.random() < 0.3:
            p = rng.random() * 0.1
        prod = " ".join(rng.choice("XYZ") + str(q) for q in qs)
        lines.append(("E" if t == 0 else "ELSE_CORRELATED_ERROR") + f"({p!r}) {prod}")
    return lines


def _memory_with_chains(seed, task="repetition_code:memory", distance=3, rounds=2, nchains=3):
    rng = random.Random(seed)
    base = stim.Circuit.generated(task, rounds=rounds, distance=distance,
                                  after_clifford_depolarization=0.001,
                                  before_measure_flip_probability=0.001)
    lines = str(base.flattened()).splitlines()
    nq = base.num_qubits
    # insert chains at random operation boundaries (never between an E and its ELSE)
    slots = [i for i, l in enumerate(lines) if not l.strip().startswith(("DETECTOR", "OBSERVABLE"))]
    for pos in sorted(rng.sample(slots, min(nchains, len(slots))), reverse=True):
        lines[pos:pos] = _random_chain_lines(rng, nq)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("seed", range(24))
def test_dem_matches_stim_on_random_clifford_chains(seed):
    task = ["repetition_code:memory", "surface_code:rotated_memory_z",
            "surface_code:rotated_memory_x", "color_code:memory_xyz"][seed % 4]
    text = _memory_with_chains(seed, task=task, nchains=2 + seed % 3)
    sc = stim.Circuit(text)
    xc = xtim.Circuit(text)
    _assert_dem_equal(xc.detector_error_model(approximate_disjoint_errors=True),
                      sc.detector_error_model(approximate_disjoint_errors=True))
    # with decomposition too (decompose_errors over the folded mechanisms)
    _assert_dem_equal(
        xc.detector_error_model(approximate_disjoint_errors=True, decompose_errors=True,
                                ignore_decomposition_failures=True),
        sc.detector_error_model(approximate_disjoint_errors=True, decompose_errors=True,
                                ignore_decomposition_failures=True))
    # stim's refusal without the option iff some chain has an ELSE term
    multi = "ELSE_CORRELATED_ERROR" in text
    refused_stim = refused_xtim = False
    try:
        sc.detector_error_model()
    except ValueError:
        refused_stim = True
    try:
        xc.detector_error_model_text()
    except xtim.XtimDemError as e:
        refused_xtim = True
        assert "ELSE_CORRELATED_ERROR" in str(e) and "approximate_disjoint_errors" in str(e)
    assert refused_stim == refused_xtim == multi


def test_dem_single_term_exact_without_option_and_bit_exact_values():
    text = PROBE.format("E(0.7) X0 Z2")
    _assert_dem_equal(xtim.Circuit(text).detector_error_model(),
                      stim.Circuit(text).detector_error_model(), rel=0.0)
    # the categorical values are stim's bit for bit (same evaluation order)
    text = PROBE.format("E(0.1) X0\nELSE_CORRELATED_ERROR(0.5) X1\nELSE_CORRELATED_ERROR(0.25) X0 X1")
    _assert_dem_equal(xtim.Circuit(text).detector_error_model(approximate_disjoint_errors=True),
                      stim.Circuit(text).detector_error_model(approximate_disjoint_errors=True), rel=0.0)


def test_dem_same_signature_terms_compose_like_stim():
    # X0 then Y0 both flip D0: stim folds them as independent mechanisms (0.1 (+) 0.09 = 0.172),
    # not as one merged 0.19 mechanism.
    text = "R 0\nE(0.1) X0\nELSE_CORRELATED_ERROR(0.1) Y0\nM 0\nDETECTOR rec[-1]\n"
    got = _canon(xtim.Circuit(text).detector_error_model(approximate_disjoint_errors=True))
    assert got[("D0",)] == pytest.approx(0.172, rel=1e-12)
    _assert_dem_equal(xtim.Circuit(text).detector_error_model(approximate_disjoint_errors=True),
                      stim.Circuit(text).detector_error_model(approximate_disjoint_errors=True))


def test_dem_threshold_is_the_categorical_component():
    text = PROBE.format("E(0.5) X0\nELSE_CORRELATED_ERROR(0.6) X1")   # components 0.5, 0.3
    with pytest.raises(xtim.XtimDemError, match="larger than the `approximate_disjoint_errors` threshold"):
        xtim.Circuit(text).detector_error_model_text(approximate_disjoint_errors=0.45)
    with pytest.raises(ValueError):
        stim.Circuit(text).detector_error_model(approximate_disjoint_errors=0.45)
    _assert_dem_equal(xtim.Circuit(text).detector_error_model(approximate_disjoint_errors=0.55),
                      stim.Circuit(text).detector_error_model(approximate_disjoint_errors=0.55))


def test_dem_identity_and_dead_terms():
    # an empty E term that fires blocks the ELSE; p=1 then p>0: the later term never fires
    for body in ("E(0.5)\nELSE_CORRELATED_ERROR(0.6) X1", "E(1) X0\nELSE_CORRELATED_ERROR(0.1) X1",
                 "E(0) X0\nELSE_CORRELATED_ERROR(0.1) X1", "E(0.1) Z0\nELSE_CORRELATED_ERROR(0.5) X1"):
        text = PROBE.format(body)
        _assert_dem_equal(xtim.Circuit(text).detector_error_model(approximate_disjoint_errors=True),
                          stim.Circuit(text).detector_error_model(approximate_disjoint_errors=True))


# ── sampling: deterministic components, the hand law, stim statistics, every path ───────────

@pytest.mark.parametrize("words, fire", [
    (["XIII", "IYZX", "ZZZZ"], 1),         # second term fires: support order and exclusivity
    (["XYZI"], 0),
    (["IIIZ", "XIIX", "YXZY"], 2),         # a 4-qubit word, first-appearance order != qubit order
    (["ZYXI", "IIZZ"], 0),
])
def test_deterministic_component_exact_on_every_path(words, fire):
    # term `fire` has probability 1, the earlier ones 0, the later ones 1 (never reached)
    seq = [0.0] * fire + [1.0] * (len(words) - fire)
    # write the firing term with its factors in a scrambled order (support order is the
    # chain's first-appearance order, the factors are bound by qubit, not by position)
    lines = []
    for t, (p, w) in enumerate(zip(seq, words)):
        facs = [f"{c}{q}" for q, c in enumerate(w) if c != "I"]
        if t == fire:
            facs = facs[::-1]
        lines.append(("E" if t == 0 else "ELSE_CORRELATED_ERROR") + f"({p!r}) " + " ".join(facs))
    text = PROBE.format("\n".join(lines))
    want = np.array(_pattern(words[fire]), dtype=bool)
    ref = stim.Circuit(text).compile_detector_sampler(seed=1).sample(64)
    assert (ref == want).all(), "oracle self-check: stim disagrees with the hand pattern"
    for path, dets in _dets_all_paths(text, 64, seed=3).items():
        assert dets.shape == (64, 4), path
        bad = ~(dets == want).all(axis=1)
        assert not bad.any(), f"{path}: chain fired {dets[bad][0].astype(int)} != {want.astype(int)}"


SEQ = [0.1, 0.5, 0.2, 0.3, 0.6]
WORDS = ["XIII", "IYZX", "XXII", "ZZYI", "IIXZ"]


def test_hand_law_on_every_path_and_stim_statistics():
    text = PROBE.format(_chain(SEQ, WORDS))
    law = _law(SEQ, WORDS)
    assert abs(sum(law.values()) - 1) < 1e-12
    shots = 400_000
    for path, dets in _dets_all_paths(text, shots, seed=5).items():
        _assert_law(dets, law, path)
    # stim itself, at 1e6 shots, against the same hand law (the oracle's own self-check)
    _assert_law(stim.Circuit(text).compile_detector_sampler(seed=7).sample(1_000_000), law, "stim")
    # and two chains + a chain sharing a qubit with a native channel: stim vs xtim twirl at 1e6
    text2 = PROBE.format(_chain([0.05, 0.3], ["XIII", "IIZI"]) + "\nDEPOLARIZE1(0.1) 1\n" +
                         _chain([0.2, 0.4, 0.5], ["IXII", "ZIII", "XZYX"]))
    D = 4
    s = stim.Circuit(text2).compile_detector_sampler(seed=9).sample(1_000_000)
    x = xtim.Circuit(text2).compile_detector_sampler(seed=9, engine="twirl").sample(1_000_000)
    cs, cx = _counts(s), _counts(x)
    n = 1_000_000
    for k in set(cs) | set(cx):
        ps, px = cs.get(k, 0) / n, cx.get(k, 0) / n
        sd = math.sqrt(2 * max(ps, px) * (1 - max(ps, px)) / n) or 1e-9
        assert abs(ps - px) < 5 * sd, (k, ps, px)
    for d in range(D):
        ms, mx = s[:, d].mean(), x[:, d].mean()
        assert abs(ms - mx) < 5 * math.sqrt(2 * max(ms, 1e-9) * (1 - ms) / n), (d, ms, mx)


# ── equivalence with the native categorical instructions ────────────────────────────────────

def _dep2_chain(p, a, b):
    words = [w for w in ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ",
                         "ZI", "ZX", "ZY", "ZZ")]
    seq = _sequential([p / 15] * 15)
    return _chain(seq, words, qubits=(a, b))


def test_depolarize2_chain_law_equals_native_on_every_path():
    p = 0.3
    pc2 = "PAULI_CHANNEL_2(" + ",".join([repr(p / 15)] * 15) + ") 0 2"
    variants = {"chain": PROBE.format(_dep2_chain(p, 0, 2)),
                "DEPOLARIZE2": PROBE.format(f"DEPOLARIZE2({p!r}) 0 2"),
                "PAULI_CHANNEL_2": PROBE.format(pc2)}
    # the hand law of a uniform 2-qubit depolarizing channel on (q0: X-part, q2: Z-part)
    law = {}
    for w in ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"):
        k = (_XPART[w[0]], 0, _ZPART[w[1]], 0)
        law[k] = law.get(k, 0.0) + p / 15
    law[(0, 0, 0, 0)] = law.get((0, 0, 0, 0), 0.0) + 1 - p
    for name, text in variants.items():
        for path, dets in _dets_all_paths(text, 300_000, seed=11).items():
            _assert_law(dets, law, f"{name}/{path}")
    # DEM: the chain is stim's chain DEM (disjoint mechanisms at p/15 each, stim's rule for an
    # ELSE chain); DEPOLARIZE2 is stim's exact independent conversion on both sides. The two
    # differ by stim's own exact-vs-disjoint conversion (0.0877 vs 0.08 per signature at p=0.3):
    # the same categorical LAW, two stim-faithful DEM conventions.
    _assert_dem_equal(xtim.Circuit(variants["chain"]).detector_error_model(approximate_disjoint_errors=True),
                      stim.Circuit(variants["chain"]).detector_error_model(approximate_disjoint_errors=True))
    _assert_dem_equal(xtim.Circuit(variants["DEPOLARIZE2"]).detector_error_model(),
                      stim.Circuit(variants["DEPOLARIZE2"]).detector_error_model())
    gc = _canon(xtim.Circuit(variants["chain"]).detector_error_model(approximate_disjoint_errors=True))
    gd = _canon(xtim.Circuit(variants["DEPOLARIZE2"]).detector_error_model())
    assert set(gc) == set(gd)
    for k in gc:
        x = 0.0
        for _ in range(4):                                                 # 4 words per signature,
            x = x + p / 15 - 2 * x * p / 15                                 # independent, XOR-composed
        assert gc[k] == pytest.approx(x, rel=1e-12)
        assert gd[k] == pytest.approx(0.08768943743823395809, rel=1e-12)      # stim's exact conversion
    disjoint_pc2 = PROBE.format("PAULI_CHANNEL_2(0,0,0,0.3,0,0,0,0,0,0,0,0,0,0,0) 0 2\n"
                                "PAULI_CHANNEL_2(0.3,0,0,0,0,0,0,0,0,0,0,0,0,0,0) 0 2")
    chain_eq = PROBE.format("E(0.3) X0\nE(0.3) X2")
    # (XI alone and IX alone: one-term chains == one-slot channels, exact on both sides)
    _assert_dem_equal(xtim.Circuit(chain_eq).detector_error_model(),
                      xtim.Circuit(disjoint_pc2).detector_error_model())


def _chains_for_depolarize2(text):
    """Rewrite every DEPOLARIZE2(p) a b c d ... line as per-pair 15-term chains."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("DEPOLARIZE2("):
            p = float(s[s.index("(") + 1:s.index(")")])
            qs = [int(t) for t in s[s.index(")") + 1:].split()]
            for i in range(0, len(qs), 2):
                out.append(_dep2_chain(p, qs[i], qs[i + 1]))
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def test_magic_example_chain_rewrite_matches_native_on_three_engines():
    """cultivation_d3_rate (T-type magic, post-selected) with every DEPOLARIZE2 rewritten as a
    15-term chain: the same detector / observable statistics as the native circuit on the
    exact and twirl engines (the chain is the same categorical law at every site), and the
    chain-circuit's DEM is stim-faithful: mechanism set equal to the native DEM, values equal up
    to the exact-vs-disjoint conversion stim itself applies to ELSE chains (O(p) relative)."""
    native = load_orig("cultivation_d3_rate").text
    chained = _chains_for_depolarize2(native)
    assert chained != native and "ELSE_CORRELATED_ERROR" in chained
    shots = 200_000
    stats = {}
    for name, text in (("native", native), ("chain", chained)):
        c = xtim.Circuit(text)
        for eng in ("exact", "twirl"):
            d, o = c.compile_detector_sampler(seed=21, engine=eng).sample(shots, separate_observables=True)
            stats[(name, eng)] = (np.asarray(d, dtype=float).mean(axis=0), np.asarray(o, dtype=float).mean())
    ref = stats[("native", "exact")]
    for key, (dm, om) in stats.items():
        for i, (a, b) in enumerate(zip(dm, ref[0])):
            sd = math.sqrt(2 * max(a, b, 1e-9) * (1 - min(a, b)) / shots)
            assert abs(a - b) < 5 * sd, (key, i, a, b)
        sd = math.sqrt(2 * max(om, ref[1], 1e-9) / shots)
        assert abs(om - ref[1]) < 5 * sd, (key, om, ref[1])
    gn = _canon(xtim.Circuit(native).detector_error_model(approximate_disjoint_errors=True))
    gc = _canon(xtim.Circuit(chained).detector_error_model(approximate_disjoint_errors=True))
    assert set(gn) == set(gc)
    for k in gn:
        assert gc[k] == pytest.approx(gn[k], rel=2e-2), (k, gc[k], gn[k])


# ── non-Clifford: dense oracles for the sampled law and for forced single outcomes ───────────

NONCLIFF = {
    "chain_before_ccz_x_type": (
        "RX 0 1 2 3\n"
        "E(0.3) X0 Y1\nELSE_CORRELATED_ERROR(0.5) X2 Z3\nELSE_CORRELATED_ERROR(0.2) X0 X1 X2\n"
        "CCZ 0 1 2\nT 3\nCX 0 3\nMX 0 1 2 3\n"
        "DETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"),
    "t_before_and_after": (
        "RX 0 1 2\nT 0\nT_DAG 1\n"
        "E(0.2) Z0 X1\nELSE_CORRELATED_ERROR(0.4) Y0\nELSE_CORRELATED_ERROR(0.5) X2 Z1\n"
        "T 0\nCX 2 1\nT 2\nMX 0\nM 1\nMX 2\n"
        "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"),
    "two_chains_ccz_between": (
        "RX 0 1 2\nR 3\n"
        "E(0.25) X0\nELSE_CORRELATED_ERROR(0.25) Z1 X2\n"
        "CCZ 0 1 2\n"
        "E(0.3) X3 Z0\nELSE_CORRELATED_ERROR(0.5) Y2\nELSE_CORRELATED_ERROR(0.5) X1 X3\n"
        "CX 0 3\nMX 0 1 2\nM 3\n"
        "DETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"),
}


@pytest.mark.parametrize("name", sorted(NONCLIFF))
def test_nonclifford_sampled_law_vs_dense_oracle(name):
    text = NONCLIFF[name]
    orc = Oracle(text)
    orc.run()
    dist = orc.record_dist()
    assert abs(sum(dist.values()) - 1) < 1e-9
    shots = 200_000
    c = xtim.Circuit(text)
    meas = np.asarray(c.compile_sampler(seed=13).sample(shots), dtype=int)
    cnt = _counts(meas)
    for k in set(cnt) | set(dist):
        p = dist.get(k, 0.0)
        cnum = cnt.get(k, 0)
        if p < 1e-12:
            assert cnum == 0, (k, cnum)
            continue
        z = (cnum - shots * p) / math.sqrt(shots * p * (1 - p))
        assert abs(z) < 4.5, (name, k, cnum / shots, p, z)


@pytest.mark.parametrize("name", sorted(NONCLIFF))
def test_nonclifford_forced_outcomes_vs_dense_oracle(name):
    """Every single chain outcome (and every other Pauli word on the support), forced: the exact
    factored law equals the dense statevector oracle aggregated by detectors."""
    text = NONCLIFF[name]
    n = xtim.Circuit(text).num_qubits
    sim = xtim.Circuit(text).compile_exact_branches()
    sites = sim.noise_sites()
    assert all(s["channel"] == "CORRELATED_ERROR" for s in sites)
    cases = [{}]
    rng = random.Random(1)
    for si, s in enumerate(sites):
        w = len(s["qubits"])
        for word in s["paulis"]:
            if set(word) != {"I"}:
                cases.append({(si, 0): word})
        for _ in range(3):
            word = "".join(rng.choice("IXYZ") for _ in range(w))
            if set(word) != {"I"}:
                cases.append({(si, 0): word})
    if len(sites) == 2:
        cases.append({(0, 0): sites[0]["paulis"][0], (1, 0): sites[1]["paulis"][-1]})
    cands = None
    for f in cases:
        law = sim.exact_branches([(s, t, p) for (s, t), p in f.items()])
        agg = aggregate(dense_branches(text, n, faults=f))
        cands = law_shifts(law, agg, cands)
        assert cands, f"{name}: forced {f}: law != dense oracle"


def test_forced_address_refusals():
    text = PROBE.format("E(0.1) X0\nELSE_CORRELATED_ERROR(0.5) Z2 X1")
    sim = xtim.Circuit(text).compile_exact_branches()
    assert sim.noise_sites()[0]["qubits"] == [0, 2, 1]
    for bad, why in ((("XX", ), "3 letters"), (("III",), "non-identity"), (("XIQ",), "non-identity")):
        with pytest.raises(ValueError, match=why):
            sim.exact_branches([(0, 0, bad[0])])
    with pytest.raises(ValueError, match="out of range"):
        sim.exact_branches([(0, 1, "XII")])
    law = sim.exact_branches([(0, 0, "IZX")])
    assert law.rows_detectors.astype(int).tolist() == [[0, 1, 1, 0]]


# ── the Python layer: scale_noise is exact on the categorical law; without_noise strips chains ──

def test_scale_noise_rescales_the_categorical_law():
    seq = [0.1, 0.5, 0.2]
    text = PROBE.format(_chain(seq, ["XIII", "IYZX", "XXII"]) + "\nDEPOLARIZE1(0.001) 1")
    for p in (1e-3, 2.5e-4, 5e-4):
        scaled = scale_noise(text, p, p0=1e-3)
        lines = [l for l in scaled.splitlines() if l.startswith(("E(", "ELSE_"))]
        got = [float(l[l.index("(") + 1:l.index(")")]) for l in lines]
        q = _categorical(got)
        want = [x * p / 1e-3 for x in _categorical(seq)]
        assert np.allclose(q, want, rtol=1e-12, atol=0), (q, want)
        xtim.Circuit(scaled)                               # parses
        assert "DEPOLARIZE1(%.12g)" % (0.001 * p / 1e-3) in scaled
    with pytest.raises(ValueError):
        scale_noise(text.replace("ELSE_CORRELATED_ERROR(0.5) Y1 Z2 X3", "ELSE_CORRELATED_ERROR(0.5"), 1e-3)


def test_without_noise_strips_chains_and_determinism_classifies():
    # (diagnose, which stripped the chains for its noiseless run, was removed at EXP-fix: the same
    # facts through the underlying APIs — the noiseless text, and the exact determinism partition.)
    text = PROBE.format(_chain(SEQ, WORDS))
    assert "E(" not in xtim.Circuit(text).without_noise().text
    from xtim._exact_scope import explicit_exact
    with explicit_exact():
        info = xtim._xtim.detector_determinism(text)
    assert info["ok"], info


# ── factory-shaped chains: 63-term uniform 3-qubit depolarizing channels at scale ───────────

def test_factory_shaped_63_term_chains_match_stim_dem():
    """Twenty uniform 63-outcome 3-qubit chains (the joint CCZ channel of a CCZ factory,
    sequential arguments p/(1-(v-1)p) as the factory writes them) on a d=5 surface memory: the
    DEM equals stim's mechanism by mechanism, categorical values bit-exact."""
    rng = random.Random(7)
    base = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=3, distance=5,
                                  after_clifford_depolarization=0.001,
                                  before_measure_flip_probability=0.001)
    lines = str(base.flattened()).splitlines()
    nq = base.num_qubits
    pc = 2e-3 / 63
    slots = [i for i, l in enumerate(lines) if not l.strip().startswith(("DETECTOR", "OBSERVABLE"))]
    for pos in sorted(rng.sample(slots, 20), reverse=True):
        t = rng.sample(range(nq), 3)
        chain = []
        for v in range(1, 64):
            ps = " ".join(f"{'IXYZ'[(v >> 2 * leg) & 3]}{q}" for leg, q in enumerate(t) if (v >> 2 * leg) & 3)
            chain.append(("E" if v == 1 else "ELSE_CORRELATED_ERROR") + f"({pc / (1 - (v - 1) * pc)!r}) {ps}")
        lines[pos:pos] = chain
    text = "\n".join(lines) + "\n"
    sites = xtim.Circuit(text).compile_exact_branches().noise_sites()
    assert sum(1 for s in sites if s["channel"] == "CORRELATED_ERROR") == 20
    gx = _canon(xtim.Circuit(text).detector_error_model(approximate_disjoint_errors=True))
    gs = _canon(stim.Circuit(text).detector_error_model(approximate_disjoint_errors=True))
    assert set(gx) == set(gs)
    for k in gs:
        assert gx[k] == pytest.approx(gs[k], rel=1e-12, abs=0), (k, gx[k], gs[k])
    with pytest.raises(xtim.XtimDemError, match="ELSE_CORRELATED_ERROR"):
        xtim.Circuit(text).detector_error_model_text()
