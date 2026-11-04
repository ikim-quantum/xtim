"""S2b mechanism gates (plan §3): L-2 relation basis, S-2 sign fixture, M-1 add_extra op counts,
M-2 BORN word j = canonical direction j.

J-b2 J1b / J2: these gates run on PRODUCTION (the public sampler, or the family sampler in the production
configuration where a test-API oracle accessor is needed); the legacy legs they had at J-b1 retired with the
legacy sampler (mapping: xtim-handoff/jb2/J1b-mapping.md).

* L-2: every DETERMINISTIC GF(2) relation of the emitted record columns (a functional constant on
  the support of the exact law (X)) holds on EVERY engine shot — an exact 0-violation counter, on
  sample() and sample_barrier() (mutation M-L2, a flipped beta of one dependent extra, turns it red).
* S-2: the A4 fixture (Z0Z1, X0X1, Y0Y1 reads of a two-coordinate register): Y0Y1's record is
  determined by the other two WITH the ordered-product sign — the relation y_YY = y_ZZ ^ y_XX ^ 1
  (raw) holds on every shot (mutation M-SIGN, the beta fold dropped, turns it red).
* M-1: each extra costs exactly ONE plane and ONE pivot reduction, and at most one G.reduce and one
  free-register echelon (exactly one each for a non-pivot extra), per plan build.
* M-2: on a plan with two record directions introduced by extras (the Born observable X0, the Born
  decision X1, each on T|+>), z_j is +1 (bit 0) iff the shot's BORN word j is below p(+1).
"""
import itertools
import os
import sys

import numpy as np
import pytest

import xtim
import xtim._xtim as _x

sys.path.insert(0, os.path.dirname(__file__))
from _record_law_x import emitted_law_x, oracle  # noqa: E402
from test_keyed_streams import BORN, _bits, draw  # noqa: E402
from test_s2b_record_law import _emitted, _family  # noqa: E402
from test_s2b_record_table import FIXTURES  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


def deterministic_relations(law):
    """Every nonzero f in GF(2)^m with f.y constant on the support: (f, constant)."""
    keys = [np.array(k, np.uint8) for k, p in law.items() if p > 1e-12]
    m = len(keys[0])
    base = keys[0]
    diffs = [k ^ base for k in keys[1:]]
    out = []
    for f in itertools.product((0, 1), repeat=m):
        f = np.array(f, np.uint8)
        if not f.any():
            continue
        if all(int(f @ d) % 2 == 0 for d in diffs):
            out.append((f, int(f @ base) % 2))
    return out


# (fixture, W ^ d0 ^ d1 on the faulted plan, W a decision): the A4 relation and the conditionals
_A4_LAW = [("a4_comb", 1, False), ("a4_minus_z", 0, False), ("a4_comb_rev", 1, False), ("a4d_rev", 1, True)]


# ── J-b2 J1b: the PRODUCTION legs (plan r2.1 J1b): the same relations, signs and conditionals on the PUBLIC
# sampler's emitted records, against the same oracles; table fields from the production builder (record-space B,
# tests/_public.recb_table). M-1 (test_add_extra_op_counts) counts the sigma dual builder's add_extra operations: it
# has no production counterpart (retired with the builder, J7b).
from _public import public_sampler, recb_table  # noqa: E402
from test_s2b_record_law import _in_family_public  # noqa: E402


def _violations_public(text, mode, seed):
    law = emitted_law_x(oracle(text))
    rel = deterministic_relations(law)
    s = public_sampler(text)
    import test_s2b_record_law as L
    old = L.N
    try:
        L.N = 6000
        M = _emitted(s, mode, seed)
    finally:
        L.N = old
    bad = 0
    for f, c in rel:
        bad += int(((M.astype(np.int64) @ f.astype(np.int64)) % 2 != c).sum())
    return bad, len(rel)


@pytest.mark.parametrize("mode", ["sample", "barrier"])
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_deterministic_relations_hold_on_every_shot_public(name, mode):
    text = FIXTURES[name]
    if not _in_family_public(text) or len(oracle(text).columns()) > 12:
        pytest.skip("outside the family / too many columns for the relation enumeration")
    bad, nrel = _violations_public(text, mode, 3)
    assert bad == 0, (name, mode, bad, nrel)


def test_relations_family_nonvacuous_public():
    seen = 0
    for name, text in _family()[:200]:
        if not _in_family_public(text) or len(oracle(text).columns()) > 10:
            continue
        bad, nrel = _violations_public(text, "sample", int(name[1:]) + 1)
        assert bad == 0, (name, bad)
        seen += nrel > 0
    assert seen > 20, seen


def _dec(s, r, k):
    return _bits(r[2] if not hasattr(r, "decisions") else r.decisions, k)


def test_dph_sign_relation_public():
    text = FIXTURES["dph_sign"]
    s = public_sampler(text)
    dec = _dec(s, s.sample(6000, seed=5), 2)
    assert (dec[:, 0] ^ dec[:, 1] == 1).all()
    assert 0.05 < dec[:, 0].mean() < 0.95
    assert list(np.asarray(recb_table(text)["rec_beta"])) == [0, 0, 1]


@pytest.mark.parametrize("name,rel,w_dec", _A4_LAW)
def test_a4_combined_direction_law_public(name, rel, w_dec):
    text = FIXTURES[name]
    s = public_sampler(text)
    N = 20000
    b = s.sample_barrier(N, seed=4)
    if w_dec:
        D3 = _bits(b.decisions(), 3)
        W, D = D3[:, 0], D3[:, 1:]
    else:
        W, D = _bits(b.obs(), 1)[:, 0], _bits(b.decisions(), 2)
    pi = np.asarray(b.frames()["plan_index"])
    ident = b.plan_structure(int(np.flatnonzero(pi == pi[0])[0]))["plan_content"]
    contents = {}
    faulted = np.zeros(N, bool)
    for i in range(N):
        p = int(pi[i])
        if p not in contents:
            contents[p] = b.plan_structure(i)
        faulted[i] = bytes(contents[p]["plan_content"]) != bytes(ident) or bool(np.asarray(contents[p]["a"]).any()) \
            or len(np.asarray(contents[p]["cz_pairs"])) > 0
    assert faulted.sum() > N // 3
    m = faulted
    assert ((W ^ D[:, 0] ^ D[:, 1])[m] == rel).all()
    assert D[m & (W == 1), 0].sum() == 0
    p = D[m & (W == 0), 0].mean()
    nz = int((m & (W == 0)).sum())
    assert abs(p - 1 / 3) < 5 * np.sqrt((1 / 3) * (2 / 3) / nz), p
    ps = contents[int(pi[int(np.flatnonzero(m)[0])])]
    bx, bz = (np.asarray(a) for a in ps["born_xz"][:2])
    lam = np.concatenate([bx, bz], 1)
    assert int(np.asarray(ps["kappa_rec"])[0]) == 2 and lam[1].sum() >= 2 and (lam[1] != lam[0]).any()


def test_a4_ordered_product_sign_relation_public():
    text = FIXTURES["a4_sign"]
    s = public_sampler(text)
    dec = _dec(s, s.sample(8000, seed=5), 3)
    rel = dec[:, 0] ^ dec[:, 1] ^ dec[:, 2]
    assert len(set(rel.tolist())) == 1
    law = emitted_law_x(oracle(text))
    nd, no = s.num_detectors, s.num_observables
    const = {k[nd + no] ^ k[nd + no + 1] ^ k[nd + no + 2] for k, p in law.items() if p > 1e-12}
    assert const == {int(rel[0])}


def test_born_word_j_is_canonical_direction_j_public():
    txt = ("RX 0\nT 0\nRX 1\nT 1\nMX 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nMX 1\nDECISION(0) rec[-1]\n"
           "R 2\nM 2\nDETECTOR rec[-1]\n")
    s = public_sampler(txt)
    t = recb_table(txt)
    assert t["kappa_rec"] == 2 and t["rho"] == 0
    seed, N = 12, 3000
    r = s.sample(N, seed=seed)
    o = _bits(r[1], 1)[:, 0]
    d = _bits(r[2], 1)[:, 0]
    p_plus = (1 + 2 ** -0.5) / 2
    z = [np.array([0 if ((draw(seed, i, BORN, j) >> 11) * 2.0 ** -53) < p_plus else 1 for i in range(N)],
                  np.uint8) for j in range(2)]
    assert len(set((o ^ z[0]).tolist())) == 1
    assert len(set((d ^ z[1]).tolist())) == 1
    assert 0.1 < z[1].mean() < 0.2
