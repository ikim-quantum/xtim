"""LM-3: the xdem writer's backward sweep (build_fault_columns / build_tau_columns) on sparse per-wire column lists with
a density switch to the dense planes, against the dense sweep it replaced (cpp/testapi/xdem_sweep_dense_oracle.hpp,
verbatim 2e89d05).

`_xtim_testapi.xdem_sweep_layouts(text)` runs the production sweep under each storage layout (auto, forced sparse,
forced dense) and the oracle on one writer model: the noise slots' column lists, the x ids, x_start (the dense words,
or the sparse per-column wire lists against the oracle's bits), the tau columns and the lean determinism verdicts on
those columns must all equal the oracle's. Inputs: the corpus (tests/data, examples, xtim/_examples, benchmarks, the
private corpus by opaque index; magic circuits included), the lean fuzz families, seeded random Clifford + magic
circuits at word-boundary sizes, the lean-magic families (small, and large enough that auto stays sparse), and a
wide-support family whose lists fill up mid-sweep (auto moves to the planes part-way). The public export's bytes
(format="xdem", and the stim format) are compared across XTIM_SWEEP_LAYOUT=sparse / dense / unset (read once per
process: subprocesses)."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from xtim import _xtim_testapi as TA
from xtim.errors import XtimDemError
import _xdem_lean_fuzz as lf
from test_lm2_lean_nondet import (BOUNDARY, MAGIC, chain_noisy_t, ghz_noisy_t, one_t_per_wire, random_magic,
                                  rep_two_noisy_t, surf_noisy_t, wide)

HERE = Path(__file__).resolve().parent
DIRS = [HERE / "data", HERE.parent / "examples", HERE.parent / "xtim" / "_examples", HERE.parent / "benchmarks",
        HERE / "private"]
LAYOUTS = ("auto", "sparse", "dense")


def check(text, tag):
    """Every layout's sweep outputs == the oracle's on `text` (None when the writer model is not built)."""
    try:
        r = TA.xdem_sweep_layouts(text)
    except XtimDemError:
        return None
    for L in LAYOUTS:
        for part in ("at", "xid", "x", "tau"):
            assert r[f"{L}_{part}"], (tag, L, part)
        assert r[f"{L}_nondet"] in (True, None), (tag, L, "nondet")
        assert (r[f"{L}_nondet"] is None) == (not r["admitted"]), (tag, L)
    assert r["sparse_final"] == "sparse" and r["sparse_tau_final"] == "sparse", (tag, r)
    assert r["dense_final"] == "dense" and r["dense_dense_from"] == 0, (tag, r)
    assert r["dense_tau_final"] == "dense" and r["dense_tau_dense_from"] == 0, (tag, r)
    assert r["sparse_dense_from"] == -1 and r["sparse_tau_dense_from"] == -1, (tag, r)
    return r


def corpus_paths():
    paths = []
    for d in DIRS:
        if d.is_dir():
            paths += sorted(str(p) for p in d.rglob("*.stim"))
    return paths


def test_corpus():
    done = magic = adm = 0
    for i, p in enumerate(corpus_paths()):
        text = Path(p).read_text()
        r = check(text, f"corpus#{i}")          # opaque index: the private corpus is never named
        if r is None:
            continue
        done += 1
        adm += r["admitted"]
        magic += any(g in text for g in MAGIC)
    assert done >= 60 and magic >= 30 and adm >= 40, (done, magic, adm)


def sweep(items):
    done = adm = 0
    for tag, text in items:
        r = check(text, tag)
        if r is None:
            continue
        done += 1
        adm += r["admitted"]
    return done, adm


def test_lean_fuzz_families():
    fams = {"fast": lf.generate(1, 450), "beta": lf.generate_beta(12345, 200),
            "boundary": lf.generate_boundary(77, 300), "mixed": lf.generate_mixed(5, 200)}
    for name, items in fams.items():
        done, adm = sweep(items)
        assert done >= 100 and adm >= 100, (name, done, adm)


@pytest.mark.slow
def test_lean_fuzz_slow():
    done, adm = sweep(lf.generate(2, 3000))
    assert done >= 1000, done


@pytest.mark.parametrize("size", BOUNDARY)
def test_word_boundary_sizes(size):
    rng = random.Random(1000 + size)
    done = 0
    for i in range(6):
        n, m = (size, rng.choice(BOUNDARY)) if i % 2 == 0 else (rng.choice((3, 5, 17)), size)
        r = check(random_magic(rng, n, m), f"random n={n} m={m} #{i}")
        if r is not None:
            assert r["T"] == m
            done += 1
    assert done >= 3, done


def test_random_small():
    rng = random.Random(20261124)
    done = 0
    for i in range(300):
        done += check(random_magic(rng, rng.randint(3, 9), rng.randint(1, 12)), f"random small #{i}") is not None
    assert done >= 150, done


def P(*rows):
    """Parameter rows (tag, text, ...) with the tag as the test id."""
    return [pytest.param(*r, id=r[0]) for r in rows]


@pytest.mark.parametrize("tag,text", P(
    *[(f"surface+T d{d}", surf_noisy_t(d)) for d in (3, 5, 9)],
    *[(f"chain+T {n}", chain_noisy_t(n)) for n in (63, 64, 65, 300, 3000)],
    *[(f"ghz+T {n}", ghz_noisy_t(n)) for n in (63, 64, 65, 200, 3000)],
    *[(f"one T per wire {n}", one_t_per_wire(n)) for n in (63, 64, 65, 200, 2000)],
    *[(f"rep+2T d{d}", rep_two_noisy_t(d)) for d in (5, 33, 1000)],
))
def test_lean_magic_families(tag, text):
    r = check(text, tag)
    assert r is not None and r["admitted"], tag


# The density switch. Few columns: the planes are at most kSweepDenseRatio (8) times the rows' fixed bytes alone (28 per
# row: CW <= 28), so auto is dense from the start. Many columns on sparse supports (the lean-magic families): auto stays
# sparse throughout, and x_start is the per-column wire lists (no wires x columns plane).
@pytest.mark.parametrize("tag,text", P(("chain+T 300", chain_noisy_t(300)), ("ghz+T 200", ghz_noisy_t(200)),
                                       ("surface+T d3", surf_noisy_t(3))))
def test_auto_dense_when_few_columns(tag, text):
    r = check(text, tag)
    assert r["C"] <= 64 * 28, r
    assert r["auto_final"] == "dense" and r["auto_dense_from"] == 0, r


@pytest.mark.parametrize("tag,text", P(("chain+T 3000", chain_noisy_t(3000)), ("ghz+T 3000", ghz_noisy_t(3000)),
                                       ("rep+2T d1000", rep_two_noisy_t(1000)), ("surface+T d15", surf_noisy_t(15))))
def test_auto_sparse_on_sparse_families(tag, text):
    r = check(text, tag)
    assert r["C"] > 64 * 28, r
    assert r["auto_final"] == "sparse" and r["auto_dense_from"] == -1, r
    assert r["auto_peak_nnz"] <= 16 * r["C"], r      # the lists hold O(C) entries on these families


def mirror_wide(n, seed):
    """An admitted, deterministic circuit whose columns spread: a noisy T pair, then a random CX network U over |0>^n,
    noise, U^-1, every wire read in Z with a detector. Swept back through U^-1 each detector's Z support grows to ~n/2
    wires (the lists fill up), then shrinks back to one wire through U."""
    rng = random.Random(seed)
    allq = " ".join(map(str, range(n)))
    net = []
    for _ in range(3 * n):
        a, b = rng.sample(range(n), 2)
        net.append(f"CX {a} {b}")
    L = ["R " + allq, f"R {n} {n + 1}", f"H {n}", f"X_ERROR(0.001) {n}", f"T {n}", f"CX {n} {n + 1}"]
    L += net + ["X_ERROR(0.01) " + allq, "DEPOLARIZE1(0.001) 0 1 2"] + net[::-1]
    L += ["M " + allq] + [f"DETECTOR rec[-{k}]" for k in range(1, n + 1)]
    L += [f"MX {n}", f"M {n + 1}"]
    return "\n".join(L) + "\n"


@pytest.mark.parametrize("tag,text,admitted", P(("wide 1600", wide(1600, 1600), True),
                                                ("mirror wide 2000", mirror_wide(2000, 7), True),
                                                ("mirror wide 2400", mirror_wide(2400, 8), True)))
def test_auto_switches_mid_sweep(tag, text, admitted):
    """Supports that fill up as the sweep walks back through a CX network: auto starts sparse and moves to the planes
    part-way (wide: refused as nondeterministic, but its model is admitted and swept; mirror wide: exported)."""
    r = check(text, tag)
    assert r is not None and r["admitted"] == admitted
    assert r["auto_final"] == "dense" and r["auto_dense_from"] > 0, r


# Fix round 1 (LM-3 review B1 / I1 / I2): the review's adversarial families (stage2/LM-3-review/scripts/adv.py).
def _tpair(a, b):
    return f"R {a} {b}\nH {a}\nX_ERROR(0.001) {a}\nT {a}\nCX {a} {b}\n", f"MX {a}\nM {b}\n"


def star(n):
    """Star GHZ (CX 0 i), detectors Z0 Zi: wire 0's row holds ~n columns and every CX strips one (I1: a list costs
    O(n) per gate, a bit row O(n / 64))."""
    pre, post = _tpair(n, n + 1)
    L = [pre + "R " + " ".join(map(str, range(n))), "H 0"] + [f"CX 0 {i}" for i in range(1, n)]
    L += ["X_ERROR(0.01) " + " ".join(map(str, range(n))), "M " + " ".join(map(str, range(n)))]
    L += [f"DETECTOR rec[-{n}] rec[-{n - i}]" for i in range(1, n)]
    return "\n".join(L) + "\n" + post


def bubble(n):
    """The swap path (B1): n detector wires fanned into p0, then p0's n-column row walked along n path wires by 3-CX
    swaps. The entries stay ~2n, but a store that kept every list's capacity would hold n columns per path wire."""
    d, p = list(range(n)), list(range(n, 2 * n))
    pre, post = _tpair(2 * n, 2 * n + 1)
    L = [pre + "R " + " ".join(map(str, range(2 * n)))]
    for j in range(n - 2, -1, -1):
        a, b = p[j], p[j + 1]
        L += [f"CX {a} {b}", f"CX {b} {a}", f"CX {a} {b}"]
    L += [f"CX {p[0]} {q}" for q in d]
    L += ["M " + " ".join(map(str, d))] + [f"DETECTOR rec[-{k}]" for k in range(1, n + 1)]
    return "\n".join(L) + "\n" + post


def schurn(n):
    """Wire 0 holds n X-columns and takes 4 (n // 4) S gates (each a row XOR of n columns)."""
    pre, post = _tpair(n, n + 1)
    m = 4 * (n // 4)
    L = [pre + "R " + " ".join(map(str, range(n))), "H 0"]
    L += ["S 0" for _ in range(m)] + ["H 0"] + [f"CX 0 {i}" for i in range(1, n)]
    L += ["M " + " ".join(map(str, range(1, n)))]
    L += [f"DETECTOR rec[-{k}]" for k in range(1, n)]
    return "\n".join(L) + "\n" + post


@pytest.mark.parametrize("tag,text", P(*[(f"star {n}", star(n)) for n in (63, 64, 65, 1600, 4000)],
                                       *[(f"bubble {n}", bubble(n)) for n in (63, 64, 65, 1600, 3000)],
                                       *[(f"schurn {n}", schurn(n)) for n in (64, 1600, 4000)],
                                       ("chain+T 3000", chain_noisy_t(3000))))
def test_adversarial_bounded(tag, text):
    """Every layout equals the oracle; the sparse store's actual bytes (list capacities, bit rows, headers) stay within
    the sizes' bound and under the planes / kSweepDenseRatio while auto is sparse; on these families its work is at most
    2x the planes'."""
    r = check(text, tag)
    assert r is not None and r["admitted"], tag
    planes = 16 * r["N"] * ((r["C"] + 63) // 64)
    cw = (r["C"] + 63) // 64
    for L in ("auto", "sparse"):
        if r[f"{L}_final"] == "sparse":
            if L == "auto":
                assert r["auto_peak_bytes"] * 8 < planes, r
            bound = 8 * r[f"{L}_peak_nnz"] + 156 * r["N"] + 16 * cw * (r[f"{L}_peak_bit_rows"] + 4) + 4096
            assert r[f"{L}_peak_bytes"] <= bound, (L, r[f"{L}_peak_bytes"], bound)
        # work <= 2x the planes' on THESE families (star, swap path, S-churn, chain); not a general bound: a row that
        # flips between list and bit row on every CX can exceed it (LM-3 fix-1 review N1, parked)
        assert r[f"{L}_work"] <= 2 * r["dense_work"] + 4096, (tag, L, r[f"{L}_work"], r["dense_work"])


@pytest.mark.parametrize("n", (3000, 4000))
def test_heavy_rows_are_bit_rows(n):
    """Star and the swap path keep the memory gain: auto stays sparse, the long rows become bit rows (a handful)."""
    for tag, text in ((f"star {n}", star(n)), (f"bubble {n}", bubble(n))):
        r = check(text, tag)
        assert r["auto_final"] == "sparse", (tag, r)
        assert r["auto_bit_rows_made"] >= 1 and r["auto_peak_bit_rows"] <= 8, (tag, r)


# Public bytes under each forced side (XTIM_SWEEP_LAYOUT is read once per process: one subprocess per side).
_SIDE = r'''
import hashlib, json, sys, warnings
warnings.simplefilter("ignore")
import xtim
from xtim import _xtim_testapi as TA
inp = json.loads(sys.stdin.read())
out = []
for p in inp["probes"]:       # the lever's effect: an auto request ends on the forced side (the same source, test API)
    r = TA.xdem_sweep_layouts(p)
    out.append([r["auto_final"], r["auto_tau_final"]])
for text in inp["texts"]:
    try:   # the lean route's stim-format results (no rotating line) come back through this call too
        s = xtim.Circuit(text).detector_error_model_text(format="xdem")
        out.append(hashlib.sha256(s.encode()).hexdigest())
    except Exception as e:   # the refusal text is part of the identity
        out.append("ERR:" + type(e).__name__ + ":" + hashlib.sha256(str(e).encode()).hexdigest())
print(json.dumps(out))
'''


PROBES = [chain_noisy_t(3000), chain_noisy_t(300)]       # auto: sparse / dense


def _side(texts, layout):
    env = dict(os.environ)
    env.pop("XTIM_SWEEP_LAYOUT", None)
    if layout:
        env["XTIM_SWEEP_LAYOUT"] = layout
    p = subprocess.run([sys.executable, "-c", _SIDE], input=json.dumps(dict(probes=PROBES, texts=texts)),
                       capture_output=True, text=True, env=env, timeout=1100)
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    return out[:len(PROBES)], out[len(PROBES):]


def test_public_bytes_both_sides():
    texts = [Path(p).read_text() for p in corpus_paths()]
    texts += [text for _, text in lf.generate(3, 150)]
    texts += [chain_noisy_t(3000), ghz_noisy_t(3000), surf_noisy_t(9), rep_two_noisy_t(1000), one_t_per_wire(2000),
              star(4000), bubble(3000), schurn(4000), wide(1600, 1600), mirror_wide(1600, 7), mirror_wide(2000, 8)]
    (pa, auto), (ps, sparse), (pd, dense) = _side(texts, None), _side(texts, "sparse"), _side(texts, "dense")
    assert pa == [["sparse", "dense"], ["dense", "dense"]], pa   # the tau sweep has one column (one axis): dense
    assert ps == [["sparse", "sparse"]] * 2 and pd == [["dense", "dense"]] * 2, (ps, pd)
    diff = [i for i in range(len(texts)) if not (auto[i] == sparse[i] == dense[i])]
    assert not diff, diff[:20]
    served = sum(not a.startswith("ERR:") for a in auto)
    assert served >= 150, served
    assert not any(a.startswith("ERR:") for a in auto[-2:]), auto[-2:]   # the mirror circuits export
