"""Self-tests of the xdem model ORACLE (tests/_xdem_model.py) against the spec's own checks (SPEC §1.2, §1.3, §2
folding, §3.7, §4.2/§4.3) — before any production code exists."""
import random
from pathlib import Path

import pytest

stim = pytest.importorskip("stim")

from _xdem_model import XdemModel, affine, cancel, pc, rref, _tab  # noqa: E402
from _xdem_synth import boundary_text, random_text  # noqa: E402

DATA = Path(__file__).parent / "data" / "xdem"


def _value(off, coins, J):
    """the parity J on the affine space off + span(coins): its value, or None when it is not deterministic"""
    if any(pc(c & J) & 1 for c in coins.values()):
        return None
    return pc(off & J) & 1


def test_reader_and_composite_on_the_worked_file():
    M = XdemModel.from_text((DATA / "ex422_obs.xdem").read_text())
    assert (M.D, M.O, M.r, M.m, M.tc, M.A, len(M.lines)) == (3, 2, 3, 2, 3, 8, 3)
    f, tau, eta = M.compose([0])
    assert tau == 0b1100 and eta == {2: 3, 3: 1}            # x Q2 Q3 rot -Q2 +Q3
    f01, _, eta01 = M.compose([0, 1])
    assert eta01 == {} and f01 == M.lines[2][1]               # X2 then Z2 = the Y2 line (SPEC §3.7)
    assert cancel([1, 0, 1, 2, 0]) == [2]


def test_worked_file_law_of_the_x2_line():
    """SPEC §3.7: coins = span{D0, D1 D2 L0 L1}; D1+D2 = 1, D1+L0 = 1, D1+L1 = 0 (bits D0 D1 D2 L0 L1 = 0..4)"""
    M = XdemModel.from_text((DATA / "ex422_obs.xdem").read_text())
    off, coins = affine(M.law(M.circuit([0])), M.n)
    assert sorted(coins.values()) == sorted(rref([0b00001, 0b11110]).values())
    assert [_value(off, coins, J) for J in (0b00110, 0b01010, 0b10010)] == [1, 1, 0]


@pytest.mark.parametrize("seed", range(40))
def test_closed_form_law_equals_the_tableau(seed):
    """SPEC §1.3 (one stabilizer law) on random valid models, several subsets"""
    rng = random.Random(seed)
    text = random_text(rng, D=rng.randint(1, 4), O=rng.randint(1, 2), r=rng.randint(1, 3), m=rng.randint(0, 2),
                       A=rng.randint(1, 4), L=4)
    M = XdemModel.from_text(text)
    for ids in ([], [0], [1, 3], [0, 1, 2, 3], [2, 0, 2, 3]):
        assert affine(M.law(M.circuit(ids)), M.n) == affine(M.law13(ids), M.n), (text, ids)


@pytest.mark.parametrize("seed", range(40))
def test_cocycle_equals_the_ordered_product(seed):
    """SPEC §1.2: M_ik ... M_i1 = the composite line (f, tau, eta), as tableaux"""
    rng = random.Random(100 + seed)
    M = XdemModel.from_text(random_text(rng, D=2, O=1, r=2, m=2, A=4, L=5))
    ids = sorted(rng.sample(range(5), rng.randint(1, 5)))
    f, tau, eta = M.compose(ids)
    assert _tab(M.circuit(ids, prep=False), M.N) == _tab(M.line_circuit(f, tau, eta), M.N)


@pytest.mark.parametrize("seed", range(40))
def test_even_exponents_fold_into_the_flips(seed):
    """SPEC §2 'even exponents folded': a line with E_a = 2 in a field equals the line with img_a XORed into f"""
    rng = random.Random(200 + seed)
    M = XdemModel.from_text(random_text(rng, D=3, O=1, r=2, m=1, A=3, L=4))
    raw, folded = [], []
    for (p, f, tau, rot, late) in M.lines:
        a = rng.randrange(M.A)
        rot2 = dict(rot)
        rot2[a] = 2
        raw.append((p, f, tau, rot2, late))
        folded.append((p, f ^ M.img(a), tau, {b: e for b, e in rot.items() if b != a}, late))
    Mr = XdemModel(M.D, M.O, M.r, M.m, M.tc, M.axes, raw)
    Mf = XdemModel(M.D, M.O, M.r, M.m, M.tc, M.axes, folded)
    ids = sorted(rng.sample(range(4), rng.randint(1, 4)))
    assert affine(Mr.law(Mr.circuit(ids)), M.n) == affine(Mf.law(Mf.circuit(ids)), M.n)


@pytest.mark.parametrize("seed", range(60))
def test_optimal_k_rule(seed):
    """SPEC §4.2/§4.3: the closed-form eps equals the tableau sign; with E^ = E (non-lossy) pass = 1"""
    rng = random.Random(300 + seed)
    M = XdemModel.from_text(random_text(rng, D=rng.randint(1, 3), O=rng.randint(1, 2), r=rng.randint(1, 3),
                                        m=rng.randint(0, 2), A=rng.randint(1, 4), L=4))
    E = sorted(rng.sample(range(4), rng.randint(0, 4)))
    f, tau, eta = M.compose(E)
    for s in range(1 << M.D):
        if M.pass_of(M.law(M.circuit(E)), s, 0) is None:
            continue
        K = M.k_optimal(E, s)
        for k in range(M.O):
            if K["lossy"][k]:
                continue
            G = (1 << (M.D + k)) | K["P"][k]
            S = K["S"][k]
            AG = [a for a in range(M.A) if pc(M.img(a) & G) & 1]
            n2 = sum(1 for a in AG if eta.get(a, 0) == 2)
            q = (sum(eta[a] for a in S) + K["y"][k]) % 4
            assert q % 2 == 0
            sgn = (pc(f & G) + n2 + sum(M.axes[a][0] for a in S) + sum(tau >> a & 1 for a in S) + q // 2) & 1
            assert K["eps"][k] == (-1) ** sgn
        if not any(K["lossy"]):
            assert M.pass_optimal(E, E, s) == 1.0


def test_boundary_text_is_valid():
    for r in (64, 65):
        M = XdemModel.from_text(boundary_text(r, 60, 4))
        assert M.r == r and M.A == 2 * r
        for (_, _, tau, _, _) in M.lines:
            M.chi(tau)
