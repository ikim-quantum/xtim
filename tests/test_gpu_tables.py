"""The xdem sampler's compiled tables as flat arrays (GPU sampler, Task 1): `_XdemSampler.device_tables()`,
`span_tables(flags)` and `cspace_of_keys(keys)`.

Each export is checked against an independent reading of the file (the error lines' p and targets parsed here, the
axis images from tests/_xdem_model.py) or against the sampler's own outputs (sample()'s rows, its C table). Internal
row layout: detector d = bit d of the row, observable o = bit 64 * WD + o.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import xtim

from _xdem_model import XdemModel, reduce, rref
from _gpu_ref import MULTIWORD, multiword_text

ROOT = Path(__file__).parent.parent
FILES = sorted(p.name for p in (ROOT / "examples" / "xdem").glob("*.xdem"))
assert FILES

_ERR = re.compile(r"error(?:\[[^\]]*\])?\((.*)\)")


@pytest.fixture(params=FILES)
def text(request):
    return (ROOT / "examples" / "xdem" / request.param).read_text()


# boundary_r65 (two words), cw8_r400 (the 8-word instance upstream pins in tests/test_xdem_multiword.py), wide_c
# (rhoL = 68: dim C past 64); tests/_gpu_ref.py builds them
MULTI = {name: (lambda name=name: multiword_text(name)) for name in MULTIWORD}


@pytest.fixture(params=sorted(MULTI))
def multi_text(request):
    return MULTI[request.param]()


def _error_lines(text):
    """[(p, detectors, observables)] of the file's error lines in file order; targets that repeat cancel"""
    out = []
    for raw in text.splitlines():
        tok = raw.split()
        if not tok or not tok[0].startswith("error"):
            continue
        dets, obs = set(), set()
        for t in tok[1:]:
            if t in ("x", "rot", "late"):
                break
            if t[0] in "DL":
                (dets if t[0] == "D" else obs).symmetric_difference_update({int(t[1:])})
        out.append((float(_ERR.fullmatch(tok[0]).group(1)), dets, obs))
    return out


def probs(text):
    return np.array([p for (p, _, _) in _error_lines(text)], dtype=np.float64)


def _internal(file_bits, D, WD):
    """a File-layout bit vector (detector d = bit d, observable o = bit D + o) in the internal row layout"""
    return (file_bits & ((1 << D) - 1)) | ((file_bits >> D) << (64 * WD))


def _row(words, k=None):
    """words (uint64 array) as a Python int, word w = bits [64 w, 64 w + 64)"""
    return sum(int(x) << (64 * w) for w, x in enumerate(words[:k]))


def _images(text, s, WD):
    """the axis images in internal layout (none for a conventional DEM)"""
    if s.num_axes == 0:
        return []
    M = XdemModel.from_text(text)
    return [_internal(M.img(a), M.D, WD) for a in range(M.A)]


def _bits_int(bits):
    """a 0/1 array as a Python int, entry i = bit i"""
    return sum(1 << i for i, b in enumerate(bits) if b)


def _beta(t):
    WF = t["WF"]
    return [_row(t["beta"][j * WF:(j + 1) * WF]) for j in range(t["rho"])]


def test_tables_shapes_and_dims(text):
    s = xtim.XdemSampler(text)
    t = s._s.device_tables()
    assert t["D"] == s.num_detectors and t["O"] == s.num_observables and t["A"] == s.num_axes
    assert t["WD"] == (t["D"] + 63) // 64 and t["WO"] == (t["O"] + 63) // 64 and t["WF"] == t["WD"] + t["WO"]
    assert t["W"] == t["WF"] and t["WA"] == (t["A"] + 63) // 64 and t["fast"] == s._s.fast_path
    assert t["chain_off"][-1] == len(t["mem_beg"]) == len(t["mem_line"]) == s.num_lines - int((probs(text) == 0).sum())
    assert len(t["chain_off"]) == len(t["chain_p"]) + 1 and len(t["mem_cnt"]) == len(t["mem_xo"]) == len(t["mem_beg"])
    assert len(t["ent_mask"]) == len(t["ent_word"])
    assert t["mean_fired"] == pytest.approx(probs(text).sum())
    for k, dt in (("chain_p", np.float64), ("chain_off", np.uint64), ("mem_beg", np.uint64), ("mem_cnt", np.uint32),
                  ("mem_xo", np.uint32), ("mem_line", np.int32), ("ent_mask", np.uint64), ("ent_word", np.uint64),
                  ("xa", np.uint64), ("Tc", np.uint64), ("Kc", np.uint64), ("bb", np.uint8), ("beta", np.uint64),
                  ("betaO", np.uint64), ("bLo", np.uint64), ("pL", np.int32)):
        assert t[k].dtype == dt and t[k].ndim == 1, k
    if t["fast"] and t["A"]:
        assert t["beta"].shape == (t["rho"] * t["WF"],) and t["Tc"].shape == (t["A"],)
        assert t["Kc"].shape == t["bb"].shape == (t["A"],)
        assert t["betaO"].shape == (t["rho"] * t["WO"],) and t["bLo"].shape == (t["rhoL"] * t["WO"],)
        assert t["pL"].shape == (t["rhoL"],) and 0 <= t["rhoL"] <= t["rho"] <= 64
        assert t["rmask"] == (1 << t["r"]) - 1


def test_chains_hold_the_lines_by_p(text):
    t = xtim.XdemSampler(text)._s.device_tables()
    p = probs(text)
    off = t["chain_off"]
    assert sorted(t["mem_line"].tolist()) == [i for i in range(len(p)) if p[i] != 0]
    for c in range(len(t["chain_p"])):
        ids = t["mem_line"][off[c]:off[c + 1]]
        assert len(ids) and (p[ids] == t["chain_p"][c]).all() and (np.diff(ids) > 0).all()
    assert len(set(t["chain_p"].tolist())) == len(t["chain_p"])


def test_flip_entries_reproduce_file_targets(text):
    t = xtim.XdemSampler(text)._s.device_tables()
    WD = t["WD"]
    lines = _error_lines(text)
    for m, i in enumerate(t["mem_line"]):
        _, dets, obs = lines[i]
        want = sum(1 << d for d in dets) | sum(1 << (64 * WD + o) for o in obs)
        got = 0
        for e in range(int(t["mem_beg"][m]), int(t["mem_beg"][m]) + int(t["mem_cnt"][m])):
            got ^= int(t["ent_mask"][e]) << (64 * int(t["ent_word"][e]))
        assert got == want, (i, got, want)
        # the extended-record offset: 0 iff the line has no exponent / tau entry; else a record header nE | nT << 32
        xo = int(t["mem_xo"][m])
        assert xo < len(t["xa"])
        if xo:
            assert int(t["xa"][xo]) != 0


def test_beta_spans_axis_images(text):
    s = xtim.XdemSampler(text)
    t = s._s.device_tables()
    if not (t["fast"] and t["A"]):
        pytest.skip("no fast-path tables (no axes, or the general path)")
    WD, WO = t["WD"], t["WO"]
    beta = _beta(t)
    assert len(rref(beta)) == t["rho"]   # independent
    for a, img in enumerate(_images(text, s, WD)):
        acc = 0
        for j in range(t["rho"]):
            if int(t["Tc"][a]) >> j & 1:
                acc ^= beta[j]
        assert acc == img, a
    # betaO = the observable words of beta; beta_L (the last rhoL vectors) has no detector bit; bLo / pL its
    # observable part and pivot (highest observable bit)
    rD = t["rho"] - t["rhoL"]
    for j, b in enumerate(beta):
        assert _row(t["betaO"][j * WO:(j + 1) * WO]) == b >> (64 * WD)
        if j >= rD:
            assert b & ((1 << (64 * WD)) - 1) == 0
            lo = _row(t["bLo"][(j - rD) * WO:(j - rD + 1) * WO])
            assert lo == b >> (64 * WD) and lo.bit_length() - 1 == t["pL"][j - rD]
        else:
            assert b & ((1 << (64 * WD)) - 1) != 0


def test_span_tables_agree_with_the_span_of_the_axis_images(text):
    """the exported span test (XdemSampler.span_tables) decides membership in the span of the masked axis images as
    an rref of those images does, on sampled and random rows (sample_ps's use of it: tests/test_gpu_postselect.py)"""
    _check_span_tables(text)


def _check_span_tables(text):
    s = xtim.XdemSampler(text)
    D = s.num_detectors
    WD = (D + 63) // 64
    rng = np.random.default_rng(7)
    flags = (rng.random(D) < 0.5).astype(np.uint8)
    st = s._s.span_tables(flags)
    mbits = sum(1 << d for d in range(D) if flags[d])
    imgs = [x & ((1 << (64 * WD)) - 1) for x in _images(text, s, WD)]
    assert st["rank"] == len(rref([x & mbits for x in imgs]))
    nw = len(st["widx"])
    assert st["wmask"].shape == (nw,) and st["basis"].shape == (st["rank"] * nw,)
    assert st["piv_word"].shape == st["piv_bit"].shape == (st["rank"],)
    assert sum(int(m) << (64 * int(w)) for w, m in zip(st["widx"], st["wmask"])) == mbits

    def in_span(d):   # XdemSpanTest::in_span, from the exported tables
        v = [(d >> (64 * int(w)) & ((1 << 64) - 1)) & int(m) for w, m in zip(st["widx"], st["wmask"])]
        dv = list(v)
        for b in range(st["rank"]):
            if dv[st["piv_word"][b]] & int(st["piv_bit"][b]):
                for k in range(nw):
                    v[k] ^= int(st["basis"][b * nw + k])
        return not any(v)

    dets, _, _ = s._s.sample(500, 3)
    rows = [int.from_bytes(r.tobytes(), "little") for r in dets]
    rows += [_bits_int(r) for r in rng.integers(0, 2, size=(200, D))]
    outcomes = set()
    for d in rows:
        want = reduce(d & mbits, rref([x & mbits for x in imgs])) == 0
        assert in_span(d) == want, d
        outcomes.add(want)
    if st["rank"] and 0 < st["rank"] < bin(mbits).count("1"):
        assert outcomes == {True, False}


def _beta_l_key(t, c_rows):
    """the key of C (rows: O-bit ints, File observable layout) in beta_L coordinates: each vector's coordinates over
    beta_L's observable parts (independent), fully reduced RREF, pivot = highest bit, descending"""
    rD, WO = t["rho"] - t["rhoL"], t["WO"]
    bl = {}
    for i in range(t["rhoL"]):
        bl[rD + i] = _row(t["bLo"][i * WO:(i + 1) * WO])
    # solve each row over the beta_L observable parts (elimination tracking combinations)
    basis = []   # (vector, combination)
    for j, v in bl.items():
        c = 1 << j
        for bv, bc in basis:
            if v >> (bv.bit_length() - 1) & 1:
                v, c = v ^ bv, c ^ bc
        assert v
        basis = [(bv ^ v, bc ^ c) if bv >> (v.bit_length() - 1) & 1 else (bv, bc) for bv, bc in basis]
        basis.append((v, c))
    coords = []
    for x in c_rows:
        c = 0
        for bv, bc in basis:
            if x >> (bv.bit_length() - 1) & 1:
                x, c = x ^ bv, c ^ bc
        assert x == 0
        coords.append(c)
    return sorted(rref(coords).values(), reverse=True)


def test_cspace_of_keys_matches_status(text):
    s = xtim.XdemSampler(text)
    t = s._s.device_tables()
    assert (s._s.cspace_of_keys(np.zeros((3, 2), np.uint64)) == 0).all()
    if not t["fast"]:
        pytest.skip("general path: no beta_L coordinates")
    _, _, cs = s._s.sample(20000, 11)
    n = s._s.num_cspaces()
    table = s._s.cspaces(0, n)
    keys = [_beta_l_key(t, [_bits_int(r) for r in c]) for c in table[1:]]
    k = max([len(x) for x in keys] + [1])
    arr = np.zeros((len(keys) + 2, k), np.uint64)
    for i, key in enumerate(keys):
        arr[i, :len(key)] = key
    arr[-1] = arr[0] if keys else 0   # a repeated row (deduplicated inside)
    ids = s._s.cspace_of_keys(arr)
    assert ids.dtype == np.int32 and ids.shape == (len(keys) + 2,)
    assert ids[:len(keys)].tolist() == list(range(1, n)) and ids[-2] == 0 and ids[-1] == ids[0]
    assert s._s.num_cspaces() == n and set(cs.tolist()) <= set(range(n))
    assert (s._s.cspace_of_keys(arr) == ids).all()   # stable across calls
    # the key of all of beta_L: its id is the existing entry of that C, or the next id appended; the entry is that C
    if t["rhoL"]:
        rD = t["rho"] - t["rhoL"]
        allkey = np.array([[1 << (rD + i) for i in reversed(range(t["rhoL"]))]], np.uint64)
        new = int(s._s.cspace_of_keys(allkey)[0])
        assert new < n or (new == n and s._s.num_cspaces() == n + 1)
        assert s._s.cspace_of_keys(allkey)[0] == new
        got = s._s.cspaces(new, 1)[0]
        assert len(got) == t["rhoL"]
        assert _beta_l_key(t, [_bits_int(r) for r in got]) == allkey[0].tolist()
        if t["rhoL"] > 1:   # not fully reduced: refused, the table unchanged
            n2 = s._s.num_cspaces()
            with pytest.raises(ValueError, match="not fully reduced"):
                s._s.cspace_of_keys(np.array([[int(allkey[0, 0]) | int(allkey[0, 1]), int(allkey[0, 1])]], np.uint64))
            assert s._s.num_cspaces() == n2
    if t["rho"] < 64:   # a bit outside the beta_L coordinates: refused
        with pytest.raises(ValueError):
            s._s.cspace_of_keys(np.array([[1 << 63]], np.uint64))


# ---- multi-word files (upstream XFM: the fast path in CW_ >= 2 words per coordinate vector). The export holds the
# multi-word law's tables (build_multi's: KcW, GcW, gamma, GbW, bb, betaO, bLo, pL) and leaves the one-word ones (Tc,
# Kc, beta) empty; fast stays False (it names the one-word tables). The chain / member / entry tables are exported as
# for any file.


def _words(x, cw):
    return [(x >> (64 * u)) & ((1 << 64) - 1) for u in range(cw)]


def _vec(arr, i, cw):
    """vector i of a flat array of cw-word vectors, as an int"""
    return _row(arr[i * cw:(i + 1) * cw])


def test_multiword_file_exports_its_tables(multi_text, tmp_path):
    # a crash (the pre-fix export read the one-word byte tables, empty here) must fail this test, not the session
    src = tmp_path / "f.xdem"
    src.write_text(multi_text)
    code = "import sys, xtim; xtim.XdemSampler(open(sys.argv[1]).read())._s.device_tables()"
    res = subprocess.run([sys.executable, "-c", code, str(src)], capture_output=True, text=True)
    assert res.returncode == 0, (res.returncode, res.stderr[-2000:])
    s = xtim.XdemSampler(multi_text)
    t = s._s.device_tables()
    assert s._s.fast_path   # the CPU samples it on the (multi-word) fast path
    cw, A, WD, WO, WF, rho, rhoL = (t[k] for k in ("coord_words", "A", "WD", "WO", "WF", "rho", "rhoL"))
    assert t["fast"] is False and cw >= 2 and t["r"] > 64 and t["rmask"] == 0
    assert max(t["r"], rho) <= 64 * cw and 0 < rho and 0 <= rhoL <= rho
    for k in ("Tc", "Kc", "beta"):
        assert len(t[k]) == 0, k
    for k, n in (("KcW", A * cw), ("GcW", A * cw), ("gamma", rho * WF), ("GbW", rho * cw), ("bb", A),
                 ("betaO", rho * WO), ("bLo", rhoL * WO), ("pL", rhoL)):
        assert t[k].shape == (n,), k
    assert len(t["mem_beg"]) > 0 and t["chain_off"][-1] == len(t["mem_beg"])
    assert t["A"] == s.num_axes and t["D"] == s.num_detectors and t["O"] == s.num_observables
    assert len(t["xa"]) > 0
    # gamma: independent, ascending pivots (highest bit), fully reduced; img_a = XOR_{l in GcW[a]} gamma_l
    gamma = [_vec(t["gamma"], l, WF) for l in range(rho)]
    piv = [g.bit_length() - 1 for g in gamma]
    assert piv == sorted(set(piv)) and len(piv) == rho
    assert all(not (gamma[i] >> piv[j] & 1) for i in range(rho) for j in range(rho) if i != j)

    def comb(x):
        acc = 0
        for l in range(rho):
            if x >> l & 1:
                acc ^= gamma[l]
        return acc

    for a, img in enumerate(_images(multi_text, s, WD)):
        g = _vec(t["GcW"], a, cw)
        assert g >> rho == 0 and comb(g) == img, a
        assert _vec(t["KcW"], a, cw) >> t["r"] == 0
    assert any(_vec(t["KcW"], a, cw) >> 64 for a in range(A))   # kappa past one word
    # beta_j = XOR_{l in GbW[j]} gamma_l: independent; betaO its observable words; beta_L (j >= rho - rhoL) has no
    # detector bit, bLo / pL its observable part and pivot
    beta = [comb(_vec(t["GbW"], j, cw)) for j in range(rho)]
    assert len(rref(beta)) == rho
    rD = rho - rhoL
    for j, b in enumerate(beta):
        assert _row(t["betaO"][j * WO:(j + 1) * WO]) == b >> (64 * WD)
        if j >= rD:
            assert b & ((1 << (64 * WD)) - 1) == 0
            lo = _row(t["bLo"][(j - rD) * WO:(j - rD + 1) * WO])
            assert lo == b >> (64 * WD) and lo.bit_length() - 1 == t["pL"][j - rD]
        else:
            assert b & ((1 << (64 * WD)) - 1) != 0


def test_multiword_cspace_of_keys_matches_status(multi_text):
    """cspace_of_keys takes keys of cw-word vectors: each C of the sampler's table, keyed in beta_L coordinates (bits
    rho - rhoL + i, spread over the words), maps to its own id; malformed keys are refused, the table unchanged"""
    s = xtim.XdemSampler(multi_text)
    t = s._s.device_tables()
    cw = t["coord_words"]
    assert (s._s.cspace_of_keys(np.zeros((3, 2 * cw), np.uint64)) == 0).all()
    _, _, cs = s._s.sample(2000, 11)   # (wide_c: almost every shot a new C)
    n = s._s.num_cspaces()
    table = s._s.cspaces(0, n)
    keys = [_beta_l_key(t, [_bits_int(r) for r in c]) for c in table[1:]]
    k = max([len(x) for x in keys] + [1])
    arr = np.zeros((len(keys) + 2, k * cw), np.uint64)
    for i, key in enumerate(keys):
        arr[i, :len(key) * cw] = sum((_words(v, cw) for v in key), [])
    arr[-1] = arr[0] if keys else 0
    ids = s._s.cspace_of_keys(arr)
    assert ids[:len(keys)].tolist() == list(range(1, n)) and ids[-2] == 0 and ids[-1] == ids[0]
    assert s._s.num_cspaces() == n and set(cs.tolist()) <= set(range(n))
    rD = t["rho"] - t["rhoL"]
    if t["rhoL"]:
        allkey = np.array([sum((_words(1 << (rD + i), cw) for i in reversed(range(t["rhoL"]))), [])], np.uint64)
        new = int(s._s.cspace_of_keys(allkey)[0])
        assert new < n or (new == n and s._s.num_cspaces() == n + 1)
        got = s._s.cspaces(new, 1)[0]
        assert len(got) == t["rhoL"]
    m = s._s.num_cspaces()
    bad = [_words(1 << max(rD - 1, 0), cw) if rD else _words(1 << (64 * cw - 1), cw),   # outside beta_L
           [1] * (cw - 1) + [0] if cw > 1 else [1]]                                        # a word count mismatch
    with pytest.raises(ValueError):
        s._s.cspace_of_keys(np.array([bad[0]], np.uint64))
    with pytest.raises(ValueError, match="coord_words"):
        s._s.cspace_of_keys(np.array([[1] * (cw + 1)], np.uint64))
    if t["rhoL"] > 1:
        two = _words((1 << (rD + 1)) | (1 << rD), cw) + _words(1 << rD, cw)
        with pytest.raises(ValueError, match="not fully reduced"):
            s._s.cspace_of_keys(np.array([two], np.uint64))
    assert s._s.num_cspaces() == m


def test_multiword_span_tables_work(multi_text):
    _check_span_tables(multi_text)   # the span test does not depend on the law's coordinates


def test_one_word_files_report_coord_words(text):
    s = xtim.XdemSampler(text)
    t = s._s.device_tables()
    # every bundled file takes the one-word fast path; a file with no axes (a conventional DEM) is on it too
    # (XdemSampler::coord_words() = 1)
    assert t["fast"] is True and t["coord_words"] == 1
    if t["A"]:
        assert t["rho"] <= 64 and t["r"] <= 64 and len(t["Tc"]) == t["A"]


def test_pack_takes_multiword_tables(multi_text):
    from xtim.gpu import tables as gt

    t = xtim.XdemSampler(multi_text)._s.device_tables()
    caps = gt.capacities(t, 64, 64)
    assert caps["XK_CAP_CW"] == t["coord_words"] >= 2 and caps["XK_CAP_RHO"] == 1
    assert caps["XK_CAP_DIMC"] == max(64, t["rhoL"])
    buf = gt.pack(t, block=64, pair_cap=64, shots=64, seed=1)
    d = buf["dims"]
    assert d[gt.XK_DIM_CW] == t["coord_words"] and d[gt.XK_DIM_RHO] == t["rho"]
    for name in ("KcW", "GcW", "gamma", "GbW"):
        off = int(d[getattr(gt, "XK_DIM_OFF_" + name.upper())])
        assert buf["tab64"][off:off + len(t[name])].tolist() == t[name].tolist(), name
    assert buf["mws"].shape == (64 * gt.mw_words(t),)
    assert buf["ckey"].shape == (64 * max(t["rhoL"], 1) * t["coord_words"],)
    assert buf["coin_in"].shape == (64 * t["coord_words"],)
