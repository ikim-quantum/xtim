"""xdem writer model (SPEC §2 canonical form): gauge/hidden-syndrome coordinates, bhat = beta on K, m = dim K - dim K_I,
strict tau, the E split at t_c and the xdem sign convention (T1(c), T1(d))."""
import pytest
import stim

from xtim import _xtim_testapi as ta

from _xdem_circuits import NAMES_EXTERNAL, NAMES_INREPO, circuit, present
from _xdem_model import rref

M_EXPECTED = {"rep3T_pre": 2, "rm15_x": 10, "msc_d3_canon": 4, "cultivation_d3_faithful": 4, "bt27_p1e-4": 48,
              "ex422_obs": 2}                                    # SPEC §6 T1(c)
R_EXPECTED = {"rep3T_pre": 1, "rm15_x": 9, "msc_d3_canon": 9, "cultivation_d3_faithful": 5, "ex422_obs": 3}   # §2 sizes


def _beta_by_stim(h, c):
    """beta(c) from the skeleton's final stabilizer generators and the axis Paulis, by stim (independent of C++)"""
    gens = [stim.PauliString(s) for s in h["skeleton_stabilizers"]]
    sim = stim.TableauSimulator()
    sim.set_state_from_stabilizers(gens)
    P = stim.PauliString(len(gens[0]))
    for a in c:
        P *= stim.PauliString(h["axis_paulis"][a])
    ex = sim.peek_observable_expectation(P)
    assert ex in (1, -1), "a K element is not +- a stabilizer"
    return 1 if ex == -1 else 0


def _pauli_bits(s):
    P = stim.PauliString(s)
    xs, zs = P.to_numpy()
    return sum(1 << i for i in range(len(xs)) if xs[i]) | (sum(1 << i for i in range(len(zs)) if zs[i]) << len(xs))


def _independent_m(h):
    """dim K - dim K_I from the axis Paulis alone (independent of the writer's coordinates)"""
    Q = [_pauli_bits(s) for s in h["axis_paulis"]]
    kap = [int(k, 16) for k in h["kappa_raw"]]
    rows, ker = {}, []                                   # kernel of kappa over the axes
    for a, v in enumerate(kap):
        c = 1 << a
        while v:
            p = v.bit_length() - 1
            if p not in rows:
                rows[p] = (v, c)
                break
            v ^= rows[p][0]
            c ^= rows[p][1]
        if not v:
            ker.append(c)
    imgs = []
    for c in ker:
        P = 0
        for a in range(len(Q)):
            if c >> a & 1:
                P ^= Q[a]
        imgs.append(P)
    return len(rref(imgs))                               # rank of c -> prod Q on K = dim K - dim K_I


@pytest.mark.parametrize("name", NAMES_INREPO + NAMES_EXTERNAL)
def test_canonical_coordinates(name):
    if not present(name):
        pytest.skip("external circuit absent")
    h = ta.xdem_header_debug(circuit(name))
    A = len(h["axis_paulis"])
    piv = [a for a in range(A) if h["pivot"][a]]
    assert h["r"] == len(piv) == len(rref([int(k, 16) for k in h["kappa_raw"]]))     # r = rank kappa
    for t, a in enumerate(piv):
        assert h["gauge"][a] == [t] and h["hidden"][a] == [] and h["bhat"][a] == 0
    assert h["m"] == _independent_m(h)                                                   # m = dim K - dim K_I
    if name in M_EXPECTED:
        assert h["m"] == M_EXPECTED[name]
    if name in R_EXPECTED:
        assert h["r"] == R_EXPECTED[name]
    assert sorted({t for a in range(A) for t in h["hidden"][a]}) == list(range(h["m"]))
    for c, beta in zip(h["K_basis"], h["beta_K"]):                                       # bhat = beta on K (T1(d))
        assert sum(h["bhat"][a] for a in c) % 2 == beta == _beta_by_stim(h, c)


def test_the_worked_example_header():
    h = ta.xdem_header_debug(circuit("ex422_obs"))
    assert (h["r"], h["m"]) == (3, 2)
    assert h["gauge"] == [[0], [1], [2], [0, 1, 2], [2], [0, 1, 2], [0], [1]]
    assert h["hidden"] == [[], [], [], [], [0], [0], [1], [1]] and h["bhat"] == [0, 0, 0, 0, 1, 1, 1, 1]


@pytest.mark.parametrize("name", NAMES_INREPO)
def test_KI_basis(name):
    """each K_I element is an axis product equal to +-I (by stim); |K| - |K_I| = m; the K_I elements are independent"""
    h = ta.xdem_header_debug(circuit(name))
    assert len(h["K_basis"]) - len(h["KI_basis"]) == h["m"]
    for c in h["KI_basis"]:
        P = stim.PauliString(len(h["axis_paulis"][0]))
        for a in c:
            P *= stim.PauliString(h["axis_paulis"][a])
        assert P.weight == 0
    assert len(rref([sum(1 << a for a in c) for c in h["KI_basis"]])) == len(h["KI_basis"])
