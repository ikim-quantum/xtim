"""FCS (byte-identical fixed-cost fixes on the sampling hot path): the pins that hold each fix to the 4.1.0 bytes.

Fix 1 (the rotation loop's postselection column read per fired flagged channel) is held by test_lean_l7's ER_PINS (the
early counts) and the PS legs below; fix 5 (draw_rf's first draw without log) by test_lean_l5 / test_lean_l7's pure-Python
reference of the RF stream and cpp/tests/test_keyed_rng.cpp's threshold scan. This file pins the BORN stream (fix 2: a
fully grown trie path is served by walk_ro, no evaluation): the bytes of seeded + continued sample() calls and the Born
trie counters (walks / steps / nodes / node evaluations / root memo hits) on circuits whose shots take Born steps, the
route counters / used / plan misses (fix 8: a pre-pass rejection must touch nothing but its index and `early`), and a
noiseless circuit through the shot block (fix 2: a table with no mechanism skips the refill). Values: release 4.1.0
(4d1fa70), captured with XTIM_FCS_CAPTURE=1.
Test-of-test (lean/FCS/REPORT.md): walk_ro's counters dropped / a stale shot-block row / the column read's untouched
channel not rejected -> red.
"""
import hashlib
import os

import numpy as np
import pytest

import xtim

pytestmark = pytest.mark.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

M3 = ("RX 0\nT 0\nRX 1\nT_DAG 1\nRX 2\nT 2\nRX 3\nT 3\nRX 4\nT_DAG 4\nCS_DAG 3 2\n"
      "Z_ERROR(0.1) 2\nCS_DAG 0 3\nCS 4 2\nPAULI_CHANNEL_1(0.5,0,0) 2\nCS_DAG 4 3\n"
      "PAULI_CHANNEL_1(0,0.02,0) 3\nCCZ 4 3 2\n")
M3_CLEAN = "".join(l + "\n" for l in M3.splitlines() if "ERROR" not in l and "CHANNEL" not in l)
DET = "R 5\nX_ERROR(0.1) 5\nM 5\nDETECTOR rec[-1]\n"
CIRCS = {
    "m3_obs": M3 + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET,
    "m3_obs_rf": M3 + "MX(0.05) 3\nOBSERVABLE_INCLUDE(0) rec[-1]\nR 5\nX_ERROR(0.1) 5\nM(0.02) 5\nDETECTOR rec[-1]\n",
    "m3_det2": M3 + "MX 3\nMX 4\n" + DET + "DETECTOR rec[-2]\n",
    "m3_clean_obs": M3_CLEAN + "MX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "m3_clean_det": M3_CLEAN + "MX 3\nMX 4\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
    "ccz_obs": "RX 0 1 2\nX_ERROR(0.3) 0\nCCZ 0 1 2\nRX 3\nT 3\nCX 3 1\nMX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n" + DET,
    "cultivation_d5": open(os.path.join(DATA, "cultivation_d5.stim")).read(),
    "distillation_15_1_3": open(os.path.join(DATA, "distillation_15_1_3.stim")).read(),
}
TRIE = ("walks", "steps", "nodes", "node_evals", "root_memo_hits")


def _h(arrs):
    h = hashlib.sha256()
    for a in arrs:
        if not isinstance(a, dict):
            h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()[:16]


def _run(name, ps):
    text = CIRCS[name]
    s = xtim.compile_twirl_sampler(text, selfcheck=0)
    kw = {"postselect": list(range(s.num_detectors))} if ps and s.num_detectors else {}
    a = s.sample(5000, seed=3, **kw)
    b = s.sample(3000, **kw)                   # continued (the shot block's next rows)
    r0 = s.record_report()                     # (set_seed resets the counters: read before the seeded call)
    t0 = r0["born_trie"]
    routes = (tuple(sorted(r0["route_counts"].items())), r0["used"], r0["counters"]["plan_misses"])
    c = s.sample(700, seed=9, **kw)            # a partial block
    t = s.record_report()["born_trie"]
    cnt = [r[-1] for r in (a, b, c) if isinstance(r[-1], dict)]
    cnt = tuple((d["attempts"], d["passed"], d["early"]) for d in cnt)
    return _h(a) + _h(b) + _h(c), tuple(int(x[k]) for x in (t0, t) for k in TRIE), cnt, routes


PINS = {
    ('ccz_obs', False): ('2e19d8f5f8e358f413c50e46675f072f9d30df38ec852e8e', (8000, 8000, 2, 2, 0, 700, 700, 0, 0, 0), (), ((('clean', 5082), ('fast', 0), ('fast_accumulator', 0), ('record_kappa_pos', 2918), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 8000, 2)),
    ('ccz_obs', True): ('0d6704c138ca7c287927106ab8dfbf494587e237fbce29bc', (7238, 7238, 2, 2, 0, 624, 624, 0, 0, 0), ((5000, 4538, 462), (3000, 2700, 300), (700, 624, 76)), ((('clean', 5082), ('fast', 0), ('fast_accumulator', 0), ('record_kappa_pos', 2156), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 7238, 2)),
    ('cultivation_d5', False): ('fad53bda79cc614f755c06a24549eaba26fcccfb3beb3c92', (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (), ((('clean', 1197), ('fast', 6803), ('fast_accumulator', 0), ('record_kappa_pos', 0), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 8000, 2468)),
    ('cultivation_d5', True): ('7ec5aef647be740293a8587d138dd9f5fc0b2fb82dfe0f3c', (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), ((5000, 708, 4292), (3000, 407, 2591), (700, 92, 608)), ((('clean', 1097), ('fast', 20), ('fast_accumulator', 0), ('record_kappa_pos', 0), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 1117, 12)),
    ('distillation_15_1_3', False): ('f30394e3b65c98d2a96ae7d65e6ce805a294cf460d39325a', (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (), ((('clean', 7901), ('fast', 99), ('fast_accumulator', 0), ('record_kappa_pos', 0), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 8000, 1)),
    ('distillation_15_1_3', True): ('89a8b9b628fa36c01eadc37906b1b0c8046e251c8d58574c', (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), ((5000, 4936, 64), (3000, 2965, 35), (700, 689, 11)), ((('clean', 7901), ('fast', 0), ('fast_accumulator', 0), ('record_kappa_pos', 0), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 7901, 1)),
    ('m3_clean_det', False): ('617d64efbca517c8f53441a971cd67e3369e9d34a7c88328', (8000, 8000, 1, 1, 0, 700, 700, 0, 0, 0), (), ((('clean', 8000), ('fast', 0), ('fast_accumulator', 0), ('record_kappa_pos', 0), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 8000, 1)),
    ('m3_clean_obs', False): ('275fd79606d95c916999f8dcee1ffc02fcdc8438081db74e', (8000, 8000, 1, 1, 0, 700, 700, 0, 0, 0), (), ((('clean', 8000), ('fast', 0), ('fast_accumulator', 0), ('record_kappa_pos', 0), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 8000, 1)),
    ('m3_det2', False): ('27f6883c6ac9e2aabcfc78f4cd1d151a1a013a44abf0e94d', (7831, 7831, 2, 2, 0, 685, 685, 0, 0, 0), (), ((('clean', 3441), ('fast', 169), ('fast_accumulator', 0), ('record_kappa_pos', 4390), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 8000, 4)),
    ('m3_det2', True): ('339cb60e8c44fb60ce3454bc3fa74c362f1411c6a9eed5ac', (7009, 7009, 2, 2, 0, 616, 616, 0, 0, 0), ((5000, 3004, 539), (3000, 1878, 299), (700, 423, 71)), ((('clean', 3441), ('fast', 153), ('fast_accumulator', 0), ('record_kappa_pos', 3568), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 7162, 4)),
    ('m3_obs', False): ('52f8c1402dd21e60fe945ffb6c8f23b147f6e12d590deda3', (3929, 3929, 1, 1, 0, 347, 347, 0, 0, 0), (), ((('clean', 3441), ('fast', 4071), ('fast_accumulator', 0), ('record_kappa_pos', 488), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 8000, 4)),
    ('m3_obs', True): ('48bb070600a38396e53ee19406cde6ecbc2b1f88f7fcf876', (3511, 3511, 1, 1, 0, 313, 313, 0, 0, 0), ((5000, 4461, 539), (3000, 2701, 299), (700, 629, 71)), ((('clean', 3441), ('fast', 3651), ('fast_accumulator', 0), ('record_kappa_pos', 70), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 7162, 4)),
    ('m3_obs_rf', False): ('53804dedea12042a0e573600a7b05c66ce0ad1921ca3eea9', (3929, 3929, 1, 1, 0, 347, 347, 0, 0, 0), (), ((('clean', 3441), ('fast', 4071), ('fast_accumulator', 0), ('record_kappa_pos', 488), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 8000, 4)),
    ('m3_obs_rf', True): ('09335434fb20bb1bfd35390d50a5c210f320bb112b149c04', (3459, 3459, 1, 1, 0, 308, 308, 0, 0, 0), ((5000, 4384, 616), (3000, 2652, 348), (700, 616, 84)), ((('clean', 3386), ('fast', 3577), ('fast_accumulator', 0), ('record_kappa_pos', 73), ('retain_kappa0', 0), ('retain_kappa_pos', 0), ('slow_kappa0', 0)), 7036, 4)),
}


@pytest.mark.parametrize("key", sorted(PINS))
def test_born_stream_and_trie_counters_pinned(key):
    got = _run(*key)
    if os.environ.get("XTIM_FCS_CAPTURE"):
        print(f"    {key!r}: {got!r},")
        return
    assert got == PINS[key]


def _short_first(name):
    """A noiseless circuit whose first call fills fewer shot-block rows than a later call (fix 2's no-refill rule must
    still initialise the rows a longer fill reads: a stale / uninitialised key row costs extra plan builds)."""
    s = xtim.compile_twirl_sampler(CIRCS[name], selfcheck=0)
    a = s.sample(700, seed=5)
    b = s.sample(5000)
    rr = s.record_report()
    return _h(a) + _h(b), rr["counters"]["plan_misses"], rr["plans"], rr["used"]


SHORT_PINS = {   # release 4.1.0 (4d1fa70)
    'm3_clean_obs': ('c4d21af179e000fe417a534f6d3e781f', 1, 1, 5700),
    'm3_clean_det': ('a864821c1b92a9433ac02edfdd255672', 1, 1, 5700),
}


@pytest.mark.parametrize("name", sorted(SHORT_PINS))
def test_noiseless_short_first_block_pinned(name):
    assert _short_first(name) == SHORT_PINS[name]
