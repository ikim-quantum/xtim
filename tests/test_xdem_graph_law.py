"""XdemSampler's per-shot law in graph-state form (GSL): the gauge basis re-expressed once at load in image coordinates
(so the law matrix is the symmetric adjacency G = sum_{a in O} t_a t_a^T of the gauge graph), the W phase from the
mod-4 quadratic form on ker G, and a pivot-indexed elimination in today's pivot order. Same law, same kernel basis,
same coin words: every output is byte-identical to the previous law code (d002345) and to the force_general oracle.

* pinned hashes: sample (threads 1 / 3 / 4), sample_ps, status_of, law_of, score_lines none / r2b (pass, action, c,
  counters), both correction exports, on every bundled xdem file and synthetic files at the word boundaries (one-word
  r, rho <= 64 incl. rho < r and rank kappa < r, zero gauge, zero rotating lines, observables > 64, lossy; multi-word
  up to 9 words incl. rank kappa < r), infidelity(lines=True) at threads 1 / 3,
  pinned at d002345 (tests/data/xdem/graph_law_sha256.json; regenerate with `python tests/test_xdem_graph_law.py`
  ONLY on a build whose law code is known good);
* oracle: on one-word files (whose coin stream differs from the oracle's) the law of every sampled fired set equals
  force_general's as a coset (same coin span, offset difference in it), status C equal; the scorer equals the oracle.
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim import _xtim
from xtim import _xtim_testapi as ta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _xdem_synth import random_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "data" / "xdem" / "graph_law_sha256.json"


def _hi_p(text, seed):
    rng = random.Random(seed)
    return re.sub(r"^error\(([^)]*)\)", lambda m: f"error({rng.uniform(0.15, 0.45):.17g})", text, flags=re.M)


def _strip_rot(text):
    return re.sub(r" (rot|late)( [+-]Q\d+)+", "", text)


# synthetic files: (name, D, O, r, m, A, L, late, hi-p, no rotation)
SYNTH = [
    ("w1_r40_rho20", 15, 5, 40, 2, 50, 120, True, True, False),     # one-word, rho < r
    ("w1_r63_rho30", 25, 5, 63, 2, 70, 140, True, True, False),     # one-word, rho < r = 63
    ("w1_r63", 58, 5, 63, 2, 75, 150, True, False, False),
    ("w1_r64", 59, 5, 64, 2, 80, 160, False, True, False),
    ("w1_r20_dense", 40, 4, 20, 2, 25, 100, True, True, False),     # rho = r = 20 < n
    ("w1_r0_hidden", 30, 5, 0, 4, 25, 100, True, True, False),      # zero gauge
    ("w1_r50_obs70", 30, 70, 50, 2, 60, 150, True, True, False),    # observables > 64, r + O > 64
    ("w1_r40_norot", 35, 5, 40, 2, 50, 120, True, True, True),      # zero rotating lines
    ("w2_r65_rho64", 59, 5, 65, 2, 80, 160, True, True, False),
    ("w2_r65", 60, 5, 65, 2, 80, 160, True, False, False),
    ("w2_r100_rho50", 45, 5, 100, 2, 120, 200, True, True, False),  # multi-word, rho < r
    ("w2_r100_obs80", 40, 80, 100, 2, 120, 220, True, True, False),
    ("w2_r127", 122, 5, 127, 2, 140, 260, False, True, False),
    ("w2_r128", 123, 5, 128, 2, 140, 260, True, True, False),
    ("w3_r129", 124, 5, 129, 2, 140, 260, True, True, False),
    ("w3_r129_rho66", 61, 5, 129, 2, 140, 260, True, True, False),
    ("w3_r130_norot", 120, 5, 130, 2, 140, 240, True, True, True),
    ("w4_r200_rho190", 185, 5, 200, 2, 215, 380, True, True, False),
    ("w9_r530", 295, 5, 530, 2, 560, 300, True, False, False),      # 9 words: the run-time instance
]


def _synth_text(case):
    name, D, O, r, m, A, L, late, hp, norot = case
    t = random_text(random.Random(77000 + 13 * r + D + O), D=D, O=O, r=r, m=m, A=A, L=L, tc=L // 2, late_ok=late)
    if hp:
        t = _hi_p(t, r + D)
    if norot:
        t = _strip_rot(t)
    return t


# rank kappa < r: a model on r0 gauge qubits written into a file of r gauge qubits, its gauge bits shifted by `off`
# (gauge bits outside [off, off + r0) unused): (name, D, O, r0, m, A, L, r, off)
RANK_DEF = [
    ("w1_r64_rk40", 15, 5, 40, 2, 50, 120, 64, 0),
    ("w3_r130_rk64_hi", 55, 5, 64, 2, 75, 160, 130, 66),
]


def _rank_def_text(case):
    name, D, O, r0, m, A, L, r, off = case
    t = random_text(random.Random(88000 + r + r0), D=D, O=O, r=r0, m=m, A=A, L=L, tc=L // 2)
    t = _hi_p(t, r)
    t = t.replace(f" gauge {r0} hidden ", f" gauge {r} hidden ", 1)
    return re.sub(r"\bg(\d+)\b", lambda mm: f"g{int(mm.group(1)) + off}", t)


def _files():
    out = [(p.name, p.read_text()) for p in sorted((ROOT / "examples" / "xdem").glob("*.xdem"))]
    out += [("data/" + str(p.relative_to(ROOT / "tests" / "data" / "xdem")), p.read_text())
            for p in sorted((ROOT / "tests" / "data" / "xdem").rglob("*.xdem"))]
    out += [(c[0], _synth_text(c)) for c in SYNTH]
    out += [(c[0], _rank_def_text(c)) for c in RANK_DEF]
    return out


def _H(*xs):
    h = hashlib.sha256()
    for x in xs:
        h.update(np.ascontiguousarray(x).tobytes() if isinstance(x, np.ndarray) else repr(x).encode())
    return h.hexdigest()


def _hyp(off, ids, L, seed):
    """a non-trivial hypothesis: each fired line kept with probability 1/2, plus two random lines"""
    rng = np.random.default_rng(seed)
    out, o = [], [0]
    for s in range(len(off) - 1):
        f = ids[off[s]:off[s + 1]]
        out += list(f[rng.random(len(f)) < 0.5]) + list(rng.integers(0, L, 2))
        o.append(len(out))
    return np.array(o, np.int64), np.array(out, np.int32)


def _digest(text):
    """every law-reading output of the production lever, hashed per operation"""
    s = _xtim._XdemSampler(text, 65536)
    big = s.num_gauge > 64
    n = 600 if big else 3000
    res = {}
    runs = {th: ta.xdem_sample(text, [(n, 3), (n // 2, 4)], th, False) for th in (1, 3, 4)}
    for th in (3, 4):   # thread-count independence (the stream contract)
        assert _H(*[x for r in runs[th][0] for x in r], runs[th][1]) == _H(*[x for r in runs[1][0] for x in r],
                                                                          runs[1][1])
    res["sample"] = _H(*[x for r in runs[1][0] for x in r], runs[1][1])
    D = s.num_detectors
    rng = np.random.default_rng(5)
    mask = list((rng.random(D) < 0.1).astype(np.uint8))
    a, b, _, dps, dpl = ta.xdem_sample_ps(text, n, 9, 3, mask, False)
    res["sample_ps"] = _H(*[np.asarray(x) for x in list(a) + list(b)], (dps, dpl))
    d, o, cs, off, ids = s.sample(n // 3, 31, 1, 0, True)
    L = s.num_lines
    ho, hi = _hyp(off, ids, L, 7)
    sets = [list(ids[off[k]:off[k + 1]]) for k in range(0, len(off) - 1, max(1, (len(off) - 1) // 25))]
    sets += [list(hi[ho[k]:ho[k + 1]]) for k in range(0, len(ho) - 1, max(1, (len(ho) - 1) // 25))]
    st = [ta.xdem_status_of(text, x, False) for x in sets]
    res["status_of"] = _H([(z["cspace"], z["dim"], z["offset"], z["coins"]) for z in st])
    lo = [ta.xdem_law_of(text, x, False) for x in sets]
    res["law_of"] = _H([(z["dim"], z["offset"], z["coins"]) for z in lo])
    corrs = ["none", "optimal"] + (["r2b"] if s.r2b_available[0] else [])
    for c in corrs:
        p, act, cb, cn = ta.xdem_lines_fixed(text, off, ids, d, ho, hi, c, 0, False)
        cn = {k: v for k, v in cn.items() if k not in ("fast_path", "kfast_laws")}
        if c == "optimal":   # (the optimal leg fills neither action nor c)
            act = cb = None
        res[f"lines_{c}"] = _H(p, act, cb, sorted(cn.items()))
        if c != "none":
            res[f"correction_{c}"] = _H(ta.xdem_correction(text, d, ho, hi, c, False))
    S = xtim.XdemSampler(text)
    for th in (1, 3):   # infidelity(lines=True) with a fixed-pattern decoder (three lines per shot)
        r = S.infidelity(n // 3, lambda dd: [np.array(sorted({(7 * k + 1) % L, (13 * k + 5) % L, (k * k + 3) % L}),
                                                      np.int32) for k in range(dd.shape[0])],
                         seed=21, threads=th, lines=True, correction=corrs[-1])
        res[f"infidelity_t{th}"] = repr((r.mean, r.stderr, r.accepted))
    return res


FILES = _files()


def _golden():
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name,text", FILES, ids=[f[0] for f in FILES])
def test_outputs_equal_pinned_d002345(name, text):
    g = _golden()
    assert name in g, f"no pinned hashes for {name}"
    got = _digest(text)
    assert got == g[name]


def _rank(vs):
    return len(_echelon(vs))


def _shape(text):
    """(r, rho = dim span(img), words per coordinate vector, A, rotating lines) of a file"""
    P = ta.xdem_parse(text)
    rho = _rank([sum(1 << j for j in img) for _, img, _, _ in P["axes"]])
    r = P["r"]
    cw = (max(r, rho) + 63) // 64 if max(r, rho) > 64 else 1
    cw += cw in (5, 7)
    rot = sum(1 for ln in P["lines"] if ln[3] or ln[4])
    return r, rho, cw, P["A"], rot


def test_files_cover_the_boundaries():
    """the file set reaches the one-word path with rho < r, with r = 0 and with no rotating line, the multi-word
    path with rho < r, every multi-word instance (2, 3, 4 words; 9 at run time) and observables > 64"""
    words, seen, rank_def = set(), set(), set()
    for name, text in FILES:
        assert _xtim._XdemSampler(text, 65536).fast_path, name
        r, rho, cw, A, rot = _shape(text)
        if not A:
            continue
        words.add(cw)
        seen.add(("rho<r", cw == 1) if rho < r else None)
        seen.add("r0" if r == 0 else None)
        seen.add(("norot", cw == 1) if rot == 0 else None)
        seen.add("obs>64" if ta.xdem_parse(text)["O"] > 64 else None)
        P = ta.xdem_parse(text)
        if _rank([sum(1 << j for j in k) for _, _, k, _ in P["axes"]]) < r:
            rank_def.add(cw)
    assert {1, 3} <= rank_def   # rank kappa < r, one word and multi-word
    assert {1, 2, 3, 4, 9} <= words
    assert {("rho<r", True), ("rho<r", False), "r0", ("norot", True), ("norot", False), "obs>64"} <= seen


def _span_reduce(basis, x):
    """x reduced by an echelon (dict pivot -> row, pivot = highest bit)"""
    while x:
        p = x.bit_length() - 1
        if p not in basis:
            return x
        x ^= basis[p]
    return 0


def _echelon(rows):
    basis = {}
    for v in rows:
        v = _span_reduce(basis, v)
        if v:
            basis[v.bit_length() - 1] = v
    return basis


def _same_coset(a, b):
    ea, eb = _echelon(a["coins"]), _echelon(b["coins"])
    if len(ea) != len(eb) or any(_span_reduce(ea, v) for v in b["coins"]):
        return False
    return _span_reduce(ea, a["offset"] ^ b["offset"]) == 0


ONE_WORD = [f for f in FILES if _shape(f[1])[2] == 1 and _shape(f[1])[3] > 0]


@pytest.mark.parametrize("name,text", ONE_WORD, ids=[f[0] for f in ONE_WORD])
def test_one_word_law_equals_general(name, text):
    """one-word files (own coin stream): the fast law of every sampled fired set and hypothesis is the oracle's law
    (coset), with the same status C and dim C"""
    s = _xtim._XdemSampler(text, 65536)
    d, o, cs, off, ids = s.sample(600, 41, 1, 0, True)
    ho, hi = _hyp(off, ids, s.num_lines, 3)
    sets = [list(ids[off[k]:off[k + 1]]) for k in range(len(off) - 1)]
    sets += [list(hi[ho[k]:ho[k + 1]]) for k in range(0, len(ho) - 1, 3)]
    coins = 0
    for x in sets:
        a, b = ta.xdem_law_of(text, x, False), ta.xdem_law_of(text, x, True)
        assert a["dim"] == b["dim"] and _same_coset(a, b), (name, x)
        sa, sb = ta.xdem_status_of(text, x, False), ta.xdem_status_of(text, x, True)
        assert sa["cspace"] == sb["cspace"] and sa["dim"] == sb["dim"], (name, x)
        coins += len(a["coins"]) > 0
    if name.startswith("w1_") and "norot" not in name and "_r0_" not in name:
        assert coins > 0   # the laws with coins (the elimination and the W phase) are exercised


@pytest.mark.parametrize("name,text", ONE_WORD, ids=[f[0] for f in ONE_WORD])
def test_one_word_scoring_equals_general(name, text):
    """score_lines none / r2b / optimal and both exports on one-word files equal the force_general oracle"""
    s = _xtim._XdemSampler(text, 65536)
    d, o, cs, off, ids = s.sample(800, 33, 1, 0, True)
    ho, hi = _hyp(off, ids, s.num_lines, 11)
    for corr in ["none", "optimal"] + (["r2b"] if s.r2b_available[0] else []):
        a = ta.xdem_lines_fixed(text, off, ids, d, ho, hi, corr, 0, False)
        b = ta.xdem_lines_fixed(text, off, ids, d, ho, hi, corr, 0, True)
        assert np.array_equal(a[0], b[0])
        if corr != "optimal":
            assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
        skip = ("fast_path", "kfast_laws")
        assert {k: v for k, v in a[3].items() if k not in skip} == {k: v for k, v in b[3].items() if k not in skip}
        if corr != "none":
            assert ta.xdem_correction(text, d, ho, hi, corr, False) == ta.xdem_correction(text, d, ho, hi, corr, True)


if __name__ == "__main__":   # regenerate the pinned hashes (on a known-good build only)
    GOLDEN.write_text(json.dumps({name: _digest(text) for name, text in FILES}, indent=1, sort_keys=True) + "\n")
    print("wrote", GOLDEN)
