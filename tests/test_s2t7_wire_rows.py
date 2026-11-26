"""S2-T7: the sampler's per-wire record rows and readout-flip columns against the pre-S2-T7 dense builders.

`_xtim_testapi._wire_rows_oracle(text)` compiles `text` (compile_family_sampler, selfcheck 0) and regenerates, from the
rows the sampler stores, the dense words of the pre-S2-T7 layouts: qcol_x / qcol_z (n' x ceil(m/64) words), pxrow /
pzrow (n' x CW), xrow_x / xrow_z (n' x XR) and the readout-flip columns rf_col (nrf x CW). Each must equal the old
builder's words exactly, written out in the test API from the public accessors (the record operators, the certified
generators and the channel masks; the slot sets for rf_col). The digest (`_ctor_digest`, tests/test_lean_l*) hashes the
regenerated words, so its values do not move.

Inputs: the corpus (tests/data, examples, the private corpus by opaque index; magic circuits included), chain-300,
surface and colour codes, GHZ chains, word-boundary sizes 63/64/65/127/128/129 (channels, extras and readout-flip
slots), seeded random Clifford + T circuits. A scaling tripwire holds the stored bytes to the rows' total weight.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import stim

import xtim
from xtim import _xtim_testapi as TA

HERE = Path(__file__).resolve().parent
DIRS = [HERE / "data", HERE.parent / "examples", HERE / "private"]
BOUNDARY = (63, 64, 65, 127, 128, 129)


def _s2t6():
    spec = importlib.util.spec_from_file_location("_s2t6_rc", HERE / "test_s2t6_record_columns.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def corpus():
    out = []
    for d in DIRS:
        if d.is_dir():
            out += sorted(str(p) for p in d.rglob("*.stim"))
    return out


def check(text, tag):
    r = TA._wire_rows_oracle(text)
    if r is None:
        return None
    assert r["mismatches"] == [], (tag, r["mismatches"])
    assert r["entries"] == r["weight"], (tag, r["entries"], r["weight"])   # one entry per operator support bit
    return r


def test_corpus():
    paths = corpus()
    done = magic = rf = 0
    for i, p in enumerate(paths):
        text = Path(p).read_text()
        r = check(text, f"corpus#{i}")          # opaque index: the private corpus is never named
        if r is None:
            continue
        done += 1
        rf += r["nrf"] > 0
        if any(g in text for g in ("T ", "T_DAG", "CCZ", "CS", "CH")):
            magic += 1
    assert done >= 40 and magic >= 20 and rf >= 10, (done, magic, rf)


def surface(kind, d):
    return str(stim.Circuit.generated(kind, distance=d, rounds=d, after_clifford_depolarization=0.001,
                                      before_measure_flip_probability=0.001, after_reset_flip_probability=0.001))


def test_chain_surface_colour():
    chain = "".join(f"R {q}\nX_ERROR(0.01) {q}\nM(0.01) {q}\nDETECTOR rec[-1]\n" for q in range(300))
    r = check(chain, "chain-300")
    assert r["nchan"] == 300 and r["nrf"] == 300
    for d in (3, 5, 7):
        for kind in ("surface_code:rotated_memory_z", "surface_code:rotated_memory_x", "color_code:memory_xyz"):
            text = surface(kind, d)
            r = check(text, f"{kind} d{d}")
            assert r is not None, (kind, d)
            # readout flips (Stim M(p) / MR(p)) on every measurement
            text = text.replace("\nMR ", "\nMR(0.002) ").replace("\nM ", "\nM(0.002) ").replace("\nMX ", "\nMX(0.002) ")
            r = check(text, f"{kind} d{d} M(p)")
            assert r is not None and (r["nrf"] > 0 or not kind.startswith("surface")), (kind, d)


def ghz(n, flip=0.0):
    allq = " ".join(map(str, range(n)))
    m = f"M({flip})" if flip else "M"
    L = [f"R {allq}", "H 0"] + [f"CX {i} {i + 1}" for i in range(n - 1)] + [f"X_ERROR(0.01) {allq}", f"{m} {allq}"]
    L += [f"DETECTOR rec[-{n - i}] rec[-{n - i - 1}]" for i in range(n - 1)]
    return "\n".join(L) + "\n"


@pytest.mark.parametrize("n", BOUNDARY + (3, 200))
def test_ghz(n):
    r = check(ghz(n, 0.01), f"ghz n={n}")
    assert r is not None and r["nchan"] == n - 1


def boundary_circuit(k, gauge):
    """k deterministic detectors with readout flips, one Born observable, and (gauge) k random-outcome detectors."""
    L = []
    for q in range(k):
        L += [f"R {q}", f"X_ERROR(0.01) {q}", f"M(0.01) {q}", "DETECTOR rec[-1]"]
    a = k
    L += [f"R {a}", f"H {a}", f"M(0.02) {a}", "OBSERVABLE_INCLUDE(0) rec[-1]"]
    if gauge:
        for j in range(k):
            q = a + 1 + j
            L += [f"R {q}", f"H {q}", f"M(0.01) {q}", "DETECTOR rec[-1]"]
    return "\n".join(L) + "\n"


@pytest.mark.parametrize("k", BOUNDARY)
def test_word_boundaries(k):
    r = check(boundary_circuit(k, False), f"boundary k={k}")
    assert r is not None and r["nchan"] == k and r["nextra"] == 1 and r["nrf"] == k + 1, r
    r = check(boundary_circuit(k, True), f"boundary+gauge k={k}")
    assert r is not None and r["nextra"] >= 1, r


def test_random():
    rc = _s2t6().random_circuit
    done = 0
    for seed in range(300):
        if check(rc(seed), f"random seed {seed}") is not None:
            done += 1
    assert done >= 100, done


def test_magic_surface():
    c = surface("surface_code:rotated_memory_z", 5)
    nq = stim.Circuit(c).num_qubits
    a, b = nq, nq + 1
    r = check(f"R {a} {b}\nH {a}\nT {a}\nCX {a} {b}\n" + c + f"\nMX {a}\nM {b}\n", "surface d5 + T")
    assert r is not None


# The density switch (WireRows::choose_layout): a small circuit keeps the full rows' words as well (the recorded-shot
# reader XORs them), a large one the lists only; XTIM_PLANES_LAYOUT=sparse / dense forces either side (read once per
# process: subprocesses). Both sides give the same record bytes and fidelity values.
_SIDE = r'''
import hashlib, json, sys, warnings
warnings.simplefilter("ignore")
import numpy as np
import xtim
from xtim import _xtim_testapi as TA
out = {}
for i, text in enumerate(json.loads(sys.stdin.read())):
    r = TA._wire_rows_oracle(text)
    h = hashlib.sha256()
    ts = xtim.compile_twirl_sampler(text)
    buf = ts.sample_barrier(3000, seed=7)
    h.update(np.ascontiguousarray(buf.dets()).tobytes())
    try:
        rf = xtim.RecordFidelity(ts)
        h.update(repr(rf.evaluate(buf)).encode())
    except Exception as e:
        h.update(type(e).__name__.encode())
    out[i] = dict(dense=r["dense_words"], rf_dense=r["rf_dense_slots"], nrf=r["nrf"], bad=r["mismatches"],
                  h=h.hexdigest())
print(json.dumps(out))
'''


def hub(k, magic):
    """The review's hub (S2-T7 F1): the M(p) record of qubit 0 read by each of k detectors (one slot, k channels)."""
    L = ["R 0", "X_ERROR(0.3) 0", "M(0.2) 0"]
    if magic:
        L = ["R 0 1", "H 1", "T 1", "CX 1 0", "CX 1 0", "H 1", "X_ERROR(0.3) 0", "M(0.2) 0"]
    for i in range(k):
        q = 2 + i
        L += [f"R {q}", f"X_ERROR(0.01) {q}", f"M {q}", f"DETECTOR rec[-1] rec[-{i + 2}]"]
    if magic:
        L += ["MX 1"]
    return "\n".join(L) + "\n"


def _side(texts, layout):
    env = dict(os.environ)
    env.pop("XTIM_PLANES_LAYOUT", None)
    if layout:
        env["XTIM_PLANES_LAYOUT"] = layout
    p = subprocess.run([sys.executable, "-c", _SIDE], input=json.dumps(texts), capture_output=True,
                       text=True, env=env, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_density_switch_both_sides():
    texts = [xtim.load_example(n).text for n in ("cube_ccz", "cultivation_d3_faithful", "cultivation_d5",
                                                 "distillation_15_1_3")]
    s5 = surface("surface_code:rotated_memory_z", 5)
    texts.append(s5.replace("\nMR ", "\nMR(0.002) ").replace("\nM ", "\nM(0.002) "))
    texts.append(hub(300, False))                            # fix round 1: one M(p) record in 300 detectors
    texts.append(hub(300, True))                             # ... with a T gate
    auto, sparse, dense = _side(texts, None), _side(texts, "sparse"), _side(texts, "dense")
    for k in auto:
        for side in (auto, sparse, dense):
            assert side[k]["bad"] == [], (k, side[k]["bad"])
        assert not sparse[k]["dense"] and dense[k]["dense"], k
        assert sparse[k]["rf_dense"] == 0 and dense[k]["rf_dense"] == dense[k]["nrf"], (k, sparse[k], dense[k])
        assert auto[k]["h"] == sparse[k]["h"] == dense[k]["h"], k
    assert auto["0"]["dense"]                                # cube_ccz: the words are kept
    for k in (str(len(texts) - 2), str(len(texts) - 1)):     # the hub slot keeps its words, the others their lists
        assert auto[k]["rf_dense"] == 1 and auto[k]["nrf"] == 1, auto[k]
    # S2-CLEAN: the readout-flip switch is 8·e > CW (was 4·e > CW), so the lists-only case is d = 13 (CW >= 16) now; at
    # d = 9 (CW = 12) the slots feeding 2 channels keep their words (12 words against a 2-entry list: linear)
    s13 = surface("surface_code:rotated_memory_z", 13)
    big = check(s13.replace("\nMR ", "\nMR(0.002) ").replace("\nM ", "\nM(0.002) "), "surface d13 M(p)")
    assert not big["dense_words"], big                      # surface d13: the lists only
    assert big["nrf"] > 0 and big["rf_dense_slots"] == 0, big   # its slots feed <= 2 channels of CW >= 16 words


# The scaling tripwire: the stored per-wire rows cost O(total row weight + n'), not n' x D / 64 words. GHZ / chain at
# n = 4000: the pre-S2-T7 dense rows are ~2 * n' * D / 32 bytes (hundreds of MB at this size); the stored form is one
# int32 per entry plus two int32 offsets per wire and row kind.
@pytest.mark.parametrize("kind", ["ghz", "chain"])
def test_storage_scales_with_weight(kind):
    n = 4000
    if kind == "ghz":
        text = ghz(n, 0.01)
    else:
        text = "".join(f"R {q}\nX_ERROR(0.01) {q}\nM(0.01) {q}\nDETECTOR rec[-1]\n" for q in range(n))
    r = check(text, f"{kind} n={n}")
    wires = r["n"]
    bound = 2 * (4 * r["entries"] + 8 * (wires + 1)) + 4096
    assert r["bytes"] <= bound, (r["bytes"], bound, r["dense_bytes"])
    rf_bound = 2 * (4 * r["rf_entries"] + 4 * (r["nrf"] + 1)) * 2 + r["nrf"] + 4096
    assert r["rf_bytes"] <= rf_bound, (r["rf_bytes"], rf_bound, r["rf_dense_bytes"])
    assert r["dense_bytes"] > 50 * r["bytes"]            # the measured saving at this size


def fan(N, e, m=5):
    """m noisy M(p) slots, each read by e of the N detectors (CW = ceil(N/64) channel words); detector i also reads its
    own qubit's record (S2-T7-rev2's threshold family, small)."""
    L = ["R " + " ".join(map(str, range(m))), "X_ERROR(0.2) " + " ".join(map(str, range(m))),
         "M(0.15) " + " ".join(map(str, range(m)))]
    nrec = m
    for i in range(N):
        q = m + i
        L += [f"R {q}", f"X_ERROR(0.01) {q}", f"M {q}"]
        nrec += 1
        refs = ["rec[-1]"] + [f"rec[-{nrec - j}]" for j in range(m) if (i - j * e) % N < e]
        L.append("DETECTOR " + " ".join(refs))
    return "\n".join(L) + "\n"


def test_readout_flip_switch_threshold():
    # S2-CLEAN (S2-T7 review N1): a slot keeps its column's CW words iff 8·e > CW (was 4·e > CW). N = 512 detectors:
    # CW = 8, so a slot read by e = 2 detectors keeps its words, by e = 1 its list only.
    for e, dense in ((1, False), (2, True), (3, True)):
        r = check(fan(512, e), f"fan e={e}")
        assert r["nrf"] == 5, r
        assert r["rf_dense_slots"] == (5 if dense else 0), (e, r["rf_dense_slots"])
