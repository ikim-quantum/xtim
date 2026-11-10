"""lean L1 (2026-11-04): idle memory — the twirl sampler's propagation table keeps every atom in the sparse a/v form
(sparsify_av) (lean D1-S3: the AoS per-alternative masks are deleted with their consumers). Byte-identical: the blob and every emitted byte are unchanged (the byte gate);
here the mechanisms:

* the sparse-a/v table equals the dense one atom for atom after rehydration (dense_c_prop), its lists are ascending and
  in-contract, and the dense readers of a sparse atom throw (F2);
* sparsify_av refuses (engine-bug logic_error) a dense vector its lists do not reproduce, before freeing it (F1);
* the footprint: `table` is a small multiple of the atom count (never 2 n bytes per atom) (lean D1-S3: the per-alternative
  masks are deleted);
* the construction digest and the samples of a selfcheck = 0 sampler equal a selfcheck > 0 sampler's (the digest
  recomputes the masks; the window is stream-neutral: test_selfcheck_once).
"""
import glob
import hashlib
import os

import numpy as np
import pytest

from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sorted(glob.glob(os.path.join(HERE, "data", "*.stim")) +
                glob.glob(os.path.join(HERE, "..", "examples", "*.stim")))
D5 = os.path.join(HERE, "..", "examples", "cultivation_d5.stim")
D3R = os.path.join(HERE, "data", "cultivation_d3_rate.stim")
WIDE = os.path.join(HERE, "..", "examples", "color5_cultivation.stim")   # n = 524 deferred wires


@pytest.mark.parametrize("path", CORPUS, ids=[os.path.basename(p) for p in CORPUS])
def test_sparse_av_table_equals_dense(path):
    res = TA._sparse_av_mismatches(open(path).read())
    if res["atoms"] == 0:
        pytest.skip("noise-free circuit: no atoms")
    assert res["mismatches"] == 0, res


# lean L6 review F1: the same check on seeded random circuits, so the Clifford-suffix sweep (every atom after the last
# non-Clifford gate) is held to the dense walk on every gate case, not only on what the corpus happens to contain.
# The full Clifford alphabet, feedback, MPP, M / MR / MX / MY / M(p), noise around every layer, and T / CS / CH / CCZ
# placed by mode: 0 none, 1 sprinkled, 2 early only, 3 the last gate, 4 many. Seeds >= 44: wide (multi-word rows).
_G1 = ["H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "H_YZ", "H_XY", "C_XYZ",
       "C_ZYX", "I"]
_G2 = ["CX", "CY", "CZ", "SWAP", "ISWAP", "ISWAP_DAG", "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG", "SQRT_ZZ",
       "SQRT_ZZ_DAG", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "CXSWAP", "CZSWAP"]
_RANDOM_SEEDS = range(50)


def _random_circuit(seed):
    """(text, the Clifford gate names after the last non-Clifford line)."""
    import random
    r = random.Random(seed)
    wide = seed >= 44
    n = r.randint(66, 140) if wide else r.randint(3, 9)
    depth = r.randint(20, 40) if wide else r.randint(12, 45)
    mode = seed % 5
    tpos = {0: set(), 1: {r.randrange(depth) for _ in range(r.randint(1, 4))}, 2: {r.randrange(depth // 3 + 1)},
            3: {depth - 1}, 4: {r.randrange(depth) for _ in range(3 if wide else depth // 2)}}[mode]
    qs = lambda k: " ".join(map(str, r.sample(range(n), k)))

    def noise():
        k = r.randrange(7)
        m = r.randint(1, n)
        if k == 0: return f"DEPOLARIZE1(0.01) {qs(m)}"
        if k == 1: return f"X_ERROR(0.02) {qs(m)}"
        if k == 2: return f"Z_ERROR(0.02) {qs(m)}"
        if k == 3: return f"Y_ERROR(0.02) {qs(m)}"
        if k == 4: return f"PAULI_CHANNEL_1(0.01,0.02,0.005) {qs(m)}"
        m2 = 2 * r.randint(1, n // 2)
        if k == 5: return f"DEPOLARIZE2(0.01) {qs(m2)}"
        return f"PAULI_CHANNEL_2(0.001,0,0,0.002,0,0,0,0.003,0,0,0,0,0,0,0.001) {qs(m2)}"

    L = [r.choice(["R", "RX"]) + " " + " ".join(map(str, range(n)))]
    nm = 0
    suffix = set()
    for d in range(depth):
        L.append(noise())                                    # a noise location before every layer
        k = r.random()
        if d in tpos:
            u = r.random()
            if u < 0.5: L.append(f"{r.choice(['T', 'T_DAG'])} {r.randrange(n)}")
            elif u < 0.7: L.append(f"CS {qs(2)}")
            elif u < 0.85: L.append(f"CH {qs(2)}")
            else: L.append(f"CCZ {qs(3)}")
            suffix = set()
        elif k < 0.35:
            g = r.choice(_G1); suffix.add(g)
            L.append(f"{g} {qs(r.randint(1, n))}")
        elif k < 0.75:
            g = r.choice(_G2); suffix.add(g)
            L.append(f"{g} {qs(2 * r.randint(1, n // 2))}")
        elif k < 0.85:
            L.append(f"{r.choice(['M', 'MX', 'MY', 'MR', 'MRX', 'M(0.01)'])} {r.randrange(n)}"); nm += 1
            if r.random() < 0.3: L.append("DETECTOR rec[-1]")
        elif k < 0.9 and nm > 0:
            L.append(f"{r.choice(['CX', 'CZ', 'CY'])} rec[-{r.randint(1, nm)}] {r.randrange(n)}")
        elif k < 0.95:
            a, b = r.sample(range(n), 2)
            L.append(f"MPP X{a}*Z{b}"); nm += 1
        else:
            L.append(f"{r.choice(['R', 'RX'])} {r.randrange(n)}")
        if r.random() < 0.5: L.append(noise())
    L.append(noise())
    L.append("M " + " ".join(map(str, range(n))))
    return "\n".join(L) + "\n", suffix


def test_random_circuits_cover_the_clifford_alphabet_in_the_suffix():
    """The generator's own coverage: every Clifford gate name lands after the last non-Clifford gate (so the sweep
    conjugates through it) of some circuit, behind a noise location; every non-Clifford placement mode occurs."""
    seen = set()
    for s in _RANDOM_SEEDS:
        seen |= _random_circuit(s)[1]
    assert seen == set(_G1) | set(_G2), sorted(set(_G1) | set(_G2) - seen)
    assert {s % 5 for s in _RANDOM_SEEDS} == set(range(5))


@pytest.mark.parametrize("seed", _RANDOM_SEEDS)
def test_sparse_av_table_equals_dense_random(seed):
    res = TA._sparse_av_mismatches(_random_circuit(seed)[0])
    assert res["atoms"] > 0, res
    assert res["mismatches"] == 0, res


def test_sparsify_av_check_has_teeth():
    """F1: an a byte outside Z4 or a v byte outside GF(2) makes sparsify_av throw; clean atoms sparsify and rehydrate."""
    total = 0
    for path in (D5, D3R):
        res = TA._sparsify_av_teeth(open(path).read(), 30)
        assert res["atoms"] > 0, res
        assert res["clean_ok"] == res["atoms"], res
        assert res["cases"] == 3 * res["atoms"] and res["thrown"] == res["cases"], res
        total += res["atoms"]
    assert total >= 40


def _fp(s):
    return {k: int(v) for k, v in s._footprint().items()}


def test_footprint_table_is_sparse_and_alt_masks_on_demand():
    text = open(WIDE).read()
    s = TA.compile_family_sampler(text, selfcheck=0)
    s.sample(500, 1)
    fp = _fp(s)
    n = int(s.record_layout()["n"])
    atoms = int(TA._propagation_walk_oracle(text, normalized=True)["atoms"])   # AnyPauli count >= the sampler's
    assert n >= 256, n                                   # a wire count where dense (2 n B / atom) and sparse differ
    # a dense atom retains 2 n + 8 ceil(n/64) bytes of vectors on top of its struct; the sparse form is the struct
    # (its vector slot may hold up to 2x capacity) + short lists — below the dense vectors alone on this circuit
    dense_vectors = 2 * n + 8 * ((n + 63) // 64)
    assert fp["table"] < atoms * dense_vectors, (fp["table"], atoms, n, dense_vectors)
    assert fp["table"] < atoms * 1024, (fp["table"], atoms)
    # (lean D1-S3: the AoS per-alternative masks and the alt blob are deleted with the streaming / accumulator paths:
    # no footprint group of theirs remains)
    assert not {"alt_masks", "alt_blob", "alt_atoms"} & set(fp), fp


def _h(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("path", [D3R, D5], ids=["d3_rate", "d5"])
def test_digest_and_samples_independent_of_the_window(path):
    text = open(path).read()
    a = TA.compile_family_sampler(text, selfcheck=0)
    b = TA.compile_family_sampler(text, selfcheck=50)
    assert a._ctor_digest() == b._ctor_digest()
    ha = _h(a.sample(3000, 7))
    hb = _h(b.sample(3000, 7))
    assert ha == hb
    assert b._ctor_digest() == a._ctor_digest()          # the digest does not depend on the masks being resident
