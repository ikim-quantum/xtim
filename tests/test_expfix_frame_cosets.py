"""EXP-fix #5 (E2): plan_frame_cosets enumerates the 2^rank cosets of K = ker(col) directly and sums each in
closed form (an exact Z4 Gauss sum) — never the 2^|supp a| masks. Plans with |supp a| = 32-48 and rank <= 12
(refused before: |supp a| > 30) are served and checked against a test-only per-coset brute force; the named
refusals (|supp a| > 64; the state budget) fire before any allocation. Mutations MF-SIGN / MF-CONV / MF-MINREP
turn the brute-force test red, MF-64 the m = 65 test, MF-BUDGET the budget test."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import xtim
from xtim import frames

from test_branch_frames import _bare_copy, _text


def _ref(n_plus, n_zero, seed, k_ghz=4):
    """A chi = 2 reference (the test_branch_frames output-port fixture form): a k_ghz-qubit GHZ with a T on its
    logical, an n_plus-qubit graph-state block (H + random CZ) and n_zero |0> qubits, all output wires. The |0>
    columns vanish (all in K); the graph block carries the rank."""
    rng = np.random.default_rng(seed)
    L = ["H 0"] + [f"CX 0 {q}" for q in range(1, k_ghz)] + ["T 0"]
    L.append("H " + " ".join(map(str, range(k_ghz, k_ghz + n_plus))))
    for _ in range(2 * n_plus):
        a, b = rng.choice(np.arange(k_ghz, k_ghz + n_plus), 2, replace=False)
        L.append(f"CZ {a} {b}")
    return _bare_copy(_text(L, k_ghz + n_plus + n_zero)).branch_frames()


def _gf2_kernel(M):
    """Basis of ker(M) over GF(2), M (rows x m) uint8."""
    M = M.copy() % 2
    rows, m = M.shape
    piv, r = [], 0
    for c in range(m):
        hit = np.nonzero(M[r:, c])[0]
        if hit.size == 0:
            continue
        i = r + hit[0]
        M[[r, i]] = M[[i, r]]
        for a in range(rows):
            if a != r and M[a, c]:
                M[a] ^= M[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(m) if c not in piv]
    K = []
    for f in free:
        v = np.zeros(m, np.uint8)
        v[f] = 1
        for i, c in enumerate(piv):
            if M[i, f]:
                v[c] = 1
        K.append(v)
    return np.array(K, np.uint8).reshape(len(K), m)


def _brute_A(fr, supp, rep, K):
    """A[i] = c_i sum_{u in K} (-i)^{|r+u|} lambda(u) (-1)^{(r+u).bx_i}, lambda(u) = <r|Z^u|r> (frames.lambda_functional)."""
    d = K.shape[0]
    free_cols, lb = frames.lambda_functional(fr, supp)
    lbv = np.zeros(len(supp), np.uint8)
    for t, l in zip(free_cols, lb):
        lbv[t] = l
    X = ((np.arange(1 << d)[:, None] >> np.arange(d)) & 1).astype(np.uint8)
    U = (X.astype(np.int64) @ K.astype(np.int64)) % 2                    # all of K
    V = U ^ rep[None, :]
    lam = np.where((U.astype(np.int64) @ lbv) % 2, -1.0, 1.0)
    amp = np.array([1, -1j, -1, 1j])[V.sum(axis=1) % 4] * lam
    bx = np.asarray(fr["branch_x"], np.uint8)[:, supp]
    coeff = np.asarray(fr["coeff"], complex)
    return np.array([coeff[i] * (amp * np.where((V.astype(np.int64) @ bx[i]) % 2, -1.0, 1.0)).sum()
                     for i in range(len(coeff))])


@pytest.mark.parametrize("n_plus,n_zero,seed", [(12, 16, 1), (14, 14, 2)])
def test_large_support_served_and_matches_brute_force(n_plus, n_zero, seed):
    fr = _ref(n_plus, n_zero, seed)
    n = int(fr["n"])
    m = 4 + n_plus + n_zero                                   # |supp a| = 32 (refused before: > 30)
    a = np.zeros(n, np.uint8)
    a[:m] = 1
    supp = np.nonzero(a)[0]
    tab = frames.plan_frame_cosets_fast(fr, a)
    K = _gf2_kernel(np.asarray(fr["stab_x"], np.uint8)[:, supp])
    r = m - K.shape[0]
    assert r <= 16 and K.shape[0] <= 20 and len(tab) == 2 ** r
    assert abs(sum(s["weight"] for s in tab) - 1.0) < 1e-12
    # sampled sectors against the brute force: |A|^2 ratios (the normalisation cancels)
    pick = [0, 1, len(tab) // 3, len(tab) // 2, len(tab) - 1]
    W = np.array([float((np.abs(_brute_A(fr, supp, np.asarray(tab[j]["vz"], np.uint8)[supp], K)) ** 2).sum())
                  for j in pick])
    got = np.array([tab[j]["weight"] for j in pick])
    np.testing.assert_allclose(got / got[0], W / W[0], rtol=1e-10, atol=0)
    # the sector order is the first-encounter (= integer-minimal representative) order
    reps = [int("".join(map(str, np.asarray(s["vz"], np.uint8)[supp][::-1])), 2) for s in tab]
    assert reps == sorted(reps)


def test_48_support_equals_the_numpy_oracle_on_its_rank_block():
    """|supp a| = 48 (d = 36, beyond a brute force): the |0> columns lie in K, so the table equals the
    numpy oracle's on the GHZ + graph block alone (same reps, weights to 1e-12)."""
    fr = _ref(10, 34, 3)
    n = int(fr["n"])
    a = np.zeros(n, np.uint8)
    a[:48] = 1
    big = frames.plan_frame_cosets_fast(fr, a)
    a14 = np.zeros(n, np.uint8)
    a14[:14] = 1
    small = frames.plan_frame_cosets_numpy(fr, a14)
    assert len(big) == len(small) > 1
    assert [s["vz"].tobytes() for s in big] == [np.asarray(s["vz"], np.uint8).tobytes() for s in small]
    for b, s in zip(big, small):
        assert abs(b["weight"] - s["weight"]) <= 1e-12 * s["weight"]
        assert b["frame_copy"] == s["frame_copy"] and b["logical_flip"] == s["logical_flip"]


def test_m65_refused_by_name():
    fr = _ref(4, 62, 4)
    a = np.zeros(int(fr["n"]), np.uint8)
    a[:65] = 1
    with pytest.raises(ValueError, match=r"\|supp a\| = 65 > 64 is not supported by the 64-bit coset masks"):
        frames.plan_frame_cosets_fast(fr, a)


def test_python_oracles_refuse_by_name_past_24():
    fr = _ref(4, 22, 5)
    a = np.zeros(int(fr["n"]), np.uint8)
    a[:25] = 1
    for fn in (frames.plan_frame_cosets_numpy, frames.plan_frame_cosets):
        with pytest.raises(ValueError, match=r"\|supp a\| = 25 enumerates 2\^25 masks"):
            fn(fr, a)


_BUDGET_PROBE = r"""
import sys, pickle, numpy as np
sys.path.insert(0, sys.argv[2])
from xtim import frames
fr = pickle.load(open(sys.argv[1], "rb"))
a = np.zeros(int(fr["n"]), np.uint8); a[:28] = 1
try:
    frames.plan_frame_cosets_fast(fr, a)
    print("SERVED")
except MemoryError as e:
    print("REFUSED", str(e))
"""


def test_budget_refusal_names_the_plan(tmp_path):
    import pickle
    fr = _ref(16, 8, 6)                                       # rank 16 -> 2^16 sectors x chi = 2
    p = tmp_path / "fr.pkl"
    with open(p, "wb") as fh:
        pickle.dump({k: (np.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in fr.items()}, fh)
    root = os.path.dirname(os.path.dirname(xtim.__file__))
    env = dict(os.environ, XTIM_MAX_STATE_BYTES="65536", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", _BUDGET_PROBE, str(p), root], env=env, capture_output=True,
                         text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.startswith("REFUSED"), out.stdout
    import re
    assert re.search(r"plan_frame_cosets \(2\^(\d+) sectors: rank r = \1, \|supp a\| = 28, chi = 2\)", out.stdout), out.stdout


def test_large_kernel_gauss_path_on_the_u2_plans():
    """The per-coset Gauss-sum path (kernel dimension d > 12; smaller kernels use the dense exact FWHT table): each
    multi-sector u2 plan, extended by 14 wires whose stab_x column is zero (|0>-type: in K, never in a
    representative), must reproduce the numpy oracle's table of the ORIGINAL plan — structure exactly (sector
    reps, frame fields, logical flips, frame copies), weight to 1e-12. MF-SIGN on the Gauss path turns this red."""
    import os as _os
    data = _os.path.join(_os.path.dirname(__file__), "data")
    masks = np.load(_os.path.join(data, "u2_qrm_magic_amasks.npy"))
    fr = _bare_copy(open(_os.path.join(data, "adaptq_qrm_magic_producer.stim")).read()).branch_frames()
    zero = np.nonzero(~np.asarray(fr["stab_x"], np.uint8).any(axis=0))[0]
    checked = 0
    for a in masks:
        small = frames.plan_frame_cosets_numpy(fr, a)
        if len(small) < 2:
            continue
        extra = [q for q in zero if not a[q]][:14]
        assert len(extra) == 14
        a2 = a.copy()
        a2[extra] = 1
        big = frames.plan_frame_cosets_fast(fr, a2)
        assert [s["vz"].tobytes() for s in big] == [np.asarray(s["vz"], np.uint8).tobytes() for s in small]
        for b, s in zip(big, small):
            assert b["fx"].tobytes() == np.asarray(s["fx"], np.uint8).tobytes()
            assert b["fz"].tobytes() == np.asarray(s["fz"], np.uint8).tobytes()
            assert b["logical_flip"] == s["logical_flip"] and b["frame_copy"] == s["frame_copy"]
            assert abs(b["weight"] - s["weight"]) <= 1e-12 * s["weight"]
        checked += 1
        if checked == 60:
            break
    assert checked == 60


@pytest.mark.parametrize("ref", ["magic10", "stab10"])
def test_large_kernel_gauss_path_on_the_synthetic_masks(ref):
    """As above on the test_branch_frames synthetic references (GHZ_10, with and without a T on the logical), padded
    with 14 idle |0> wires: random 1..9-qubit masks over the GHZ wires (whose coset sums need the quadratic term),
    extended by the 14 idle wires (d > 12: the Gauss path), equal the numpy oracle on the unextended mask.
    MF-SIGN / MF-CONV on the Gauss path turn this red."""
    from test_branch_frames import _SYNTH_REFS
    lines, n_data = _SYNTH_REFS[ref]
    fr = _bare_copy(_text(lines, n_data + 14)).branch_frames()
    n = int(fr["n"])
    rng = np.random.default_rng(20261022)
    checked = 0
    for size in range(1, 10):
        for _ in range(4):
            a = np.zeros(n, np.uint8)
            a[rng.choice(n_data, size=size, replace=False)] = 1
            small = frames.plan_frame_cosets_numpy(fr, a)
            a2 = a.copy()
            a2[n_data:n_data + 14] = 1
            big = frames.plan_frame_cosets_fast(fr, a2)
            assert [s["vz"].tobytes() for s in big] == [np.asarray(s["vz"], np.uint8).tobytes() for s in small]
            for b, s in zip(big, small):
                assert b["fx"].tobytes() == np.asarray(s["fx"], np.uint8).tobytes()
                assert b["fz"].tobytes() == np.asarray(s["fz"], np.uint8).tobytes()
                assert b["logical_flip"] == s["logical_flip"] and b["frame_copy"] == s["frame_copy"]
                assert abs(b["weight"] - s["weight"]) <= 1e-12 * s["weight"]
            checked += 1
    assert checked == 36
