"""XdemSampler's status C bookkeeping (SCF): the per-shot C (the observable values the gauge qubits leave undetermined
given the detectors) and the append-only C table, on synthetic files where nearly every shot names a new, large C
(many observables in span(img_a), high p: the Bravyi-Haah regime, ~k/2-dimensional C over k outputs) and on four
bundled examples.

* oracle: every sampled shot's C table entry equals the force_general oracle's status of that shot's fired lines (the
  general path: law_into's coins, the detector part eliminated, the observable parts' fully reduced RREF), on one-word
  and multi-word files (C depends on the fired set only, not on the coin draw);
* the table contract: entry 0 = {0}; every entry the fully reduced RREF (pivot = highest observable, descending);
  entries distinct; ids in first-appearance order in shot order; the same table and ids at threads 1 / 3 / 4 and over
  successive calls (thread tables reused across calls); thousands of entries (index growth);
* pinned: sample (dets, obs, ids, the C table at threads 1 / 3 / 4), sample_ps and status_of at d34d19d
  (tests/data/xdem_status_c/sha256.json; regenerate with `python tests/test_xdem_status_c.py` ONLY on a build whose
  status code is known good).
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from xtim import _xtim
from xtim import _xtim_testapi as ta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _xdem_synth import random_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "xdem_status_c"
GOLDEN = DATA / "sha256.json"


def _hi_p(text, seed, lo=0.05, hi=0.3):
    rng = random.Random(seed)
    return re.sub(r"^error\(([^)]*)\)", lambda m: f"error({rng.uniform(lo, hi):.17g})", text, flags=re.M)


# synthetic files with many observables in span(img_a) (rho_L large): (name, D, O, r, A, L)
SYNTH = [
    ("w1_obs40", 6, 40, 60, 70, 160),       # one word: rho = 46, rho_L = 40
    ("w1_obs60", 3, 60, 64, 80, 200),       # one word at the boundary: rho = 63
    ("w2_obs90", 8, 90, 110, 120, 240),     # two words: rho_L = 90
    ("w3_obs150", 10, 150, 170, 180, 300),  # three words, observables over three words
]


def _synth(case):
    name, D, O, r, A, L = case
    t = random_text(random.Random(31000 + D + O + r), D=D, O=O, r=r, m=2, A=A, L=L, tc=L // 2)
    return _hi_p(t, r + O)


def _files():
    out = [(c[0], _synth(c)) for c in SYNTH]
    out += [(p.name, p.read_text()) for p in sorted((ROOT / "examples" / "xdem").glob("*.xdem"))
            if p.name in ("color5_cultivation.xdem", "cultivation_d5.xdem", "rm15rep.xdem", "toy2obs.xdem")]
    return out


FILES = _files()


def _H(*xs):
    h = hashlib.sha256()
    for x in xs:
        h.update(np.ascontiguousarray(x).tobytes() if isinstance(x, np.ndarray) else repr(x).encode())
    return h.hexdigest()


def _is_rref_desc(rows):
    """fully reduced echelon, pivot = highest bit, rows by descending pivot"""
    piv = [r.bit_length() - 1 for r in rows]
    if any(r == 0 for r in rows) or any(a <= b for a, b in zip(piv, piv[1:])):
        return False
    return all(not (rows[i] >> p & 1) for i in range(len(rows)) for j, p in enumerate(piv) if j != i)


def _check_table(table, ids):
    assert table[0] == []
    assert all(_is_rref_desc(t) for t in table)
    assert len({tuple(t) for t in table}) == len(table)    # distinct contents
    seen = 1   # id 0 = {0} exists before any shot
    for i in ids.tolist():
        assert i <= seen, "ids are not in first-appearance order"
        if i == seen:
            seen += 1
    return seen


@pytest.mark.parametrize("name,text", FILES, ids=[f[0] for f in FILES])
def test_shot_c_equals_general_oracle(name, text):
    """each shot's C table entry is the force_general oracle's status C of its fired lines"""
    s = _xtim._XdemSampler(text, 65536)
    n = 400
    d, o, cs, off, ids = s.sample(n, 17, 1, 0, True)
    table = s.cspaces(0, s.num_cspaces())
    dims = 0
    for k in range(0, n, 2):
        fired = [int(x) for x in ids[off[k]:off[k + 1]]]
        ref = ta.xdem_status_of(text, fired, True)
        got = np.asarray(table[int(cs[k])])   # uint8[dim, O]
        rows = [sum(int(b) << j for j, b in enumerate(r)) for r in got]
        assert rows == ref["cspace"], (name, k)
        dims += ref["dim"]
    if name.startswith("w"):
        assert dims > 4 * (n // 2), dims   # large C on most shots (mean dim C > 4)


@pytest.mark.parametrize("name,text", FILES, ids=[f[0] for f in FILES])
def test_table_contract_threads_and_calls(name, text):
    calls = [(3000, 5), (2000, 6), (3000, 5)]   # the third call repeats the first: every C already in the table
    runs = {th: ta.xdem_sample(text, calls, th, False) for th in (1, 3, 4)}
    (r1, t1, k1) = runs[1]
    for th in (3, 4):
        rr, tt, _ = runs[th]
        assert tt == t1
        for a, b in zip(rr, r1):
            for x, y in zip(a, b):
                assert np.array_equal(x, y)
    ids = np.concatenate([r[2] for r in r1])
    assert _check_table(t1, ids) == len(t1)
    assert np.array_equal(r1[2][2], r1[0][2])     # the repeated call names the same ids
    if name.startswith("w"):
        assert len(t1) > 2048, len(t1)            # the indexes grow several times


def _digest(text):
    s = _xtim._XdemSampler(text, 65536)
    res = {}
    for th in (1, 3, 4):
        runs, tab, _ = ta.xdem_sample(text, [(2500, 3), (1200, 4)], th, False)
        res[f"sample_t{th}"] = _H(*[x for r in runs for x in r], tab)
    D = s.num_detectors
    rng = np.random.default_rng(5)
    mask = list((rng.random(D) < 0.1).astype(np.uint8))
    a, b, _, dps, dpl = ta.xdem_sample_ps(text, 2500, 9, 3, mask, False)
    res["sample_ps"] = _H(*[np.asarray(x) for x in list(a) + list(b)], (dps, dpl))
    d, o, cs, off, ids = s.sample(60, 31, 1, 0, True)
    st = [ta.xdem_status_of(text, [int(x) for x in ids[off[k]:off[k + 1]]], False) for k in range(60)]
    res["status_of"] = _H([(z["cspace"], z["dim"], z["offset"], z["coins"]) for z in st])
    return res


@pytest.mark.parametrize("name,text", FILES, ids=[f[0] for f in FILES])
def test_outputs_equal_pinned_d34d19d(name, text):
    g = json.loads(GOLDEN.read_text())
    assert name in g, f"no pinned hashes for {name}"
    assert _digest(text) == g[name]


if __name__ == "__main__":   # regenerate the pinned hashes (on a known-good build only)
    GOLDEN.write_text(json.dumps({name: _digest(text) for name, text in FILES}, indent=1, sort_keys=True) + "\n")
    print("wrote", GOLDEN)
