"""xdem Task 5 / SPEC §6 T2: per-fault-configuration exactness of `format="xdem"` against xtim's EXACT engine.

Port of tests/test_dc_oracle_configs.py (dc-DEM Task 5) to the xdem file. For every circuit of the registry
(tests/_xdem_circuits.py), the writer's own atom map (test API `xdem_atom_lines`: each channel alternative / readout
flip as the exact engine's forced-fault address, with the index of the xdem line carrying it) drives a fixed
configuration set:

  singles   every atom alone (an atom on no line: its configuration is the empty set);
  pairs     <= 2000 pairs of distinct lines (one representative atom each; sampled, never materialised);
  rand      random sets of 3..12 atoms (300 in the fast run, 10^4 per circuit in the slow run);
  sameloc   2..6 atoms whose lines share an axis in tau ∪ rot ∪ late (preflight S3);
  merge     two atoms on the SAME line (the file fires that line twice: it cancels), half of them with 1..3 more.

Per configuration the exact engine (`_exact_engine.compile_exact_branches(text).exact_branches(addrs)`, test-only
oracle) gives the law of (detectors, observables) relative to the noiseless reference; it must be an affine coset
(offset + uniform span) -- a non-uniform law (a MIXTURE) is counted and fails the gate. The file's law is the C++
reader's `ta.xdem_laws(text, lines)` (SPEC §1.2 composite, §1.3 law). Agreement is EXACT over GF(2): equal fully
reduced coin bases, and the two offsets equal modulo that span. Two atoms of one channel instance (or one record
twice) cannot be forced together: such configurations are skipped.

A circuit whose export is the conventional DEM (nothing rotates, SPEC §3.4) or is refused is skipped (preflight S3):
its law is the default export's, covered by that route's own suites.

Mutation legs (SPEC §6 T2/T3 mutations, on the xdem text): negate one line's E; drop one tau bit on an axis a later
line rotates about; swap two non-commuting neighbours on one side of t_c. Each must disagree with the exact engine on
configurations exercising the edit (or, for a dropped tau bit, the reader refuses the line as not an operator).

Fast set by default; the rest of the registry is `slow` (`--runslow`). Each run prints its per-mode counts.
"""
import itertools
import random
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pytest

from xtim import _xtim_testapi as ta
from xtim._exact_scope import explicit_exact
from xtim.errors import XtimDemError, XtimXdemError

from _exact_law import instances, readout_records
from _exact_engine import compile_exact_branches
from _xdem_circuits import NAMES_EXTERNAL, NAMES_INREPO, circuit, present, xdem_export
from _xdem_model import XdemModel, reduce, rref

FAST = ["cultivation_d3_faithful", "distillation_15_1_3:gatenoise", "miniature_oracle", "ex422_obs", "rep3T_pre",
        "ccz_toy_obs"]
SLOW = [n for n in NAMES_INREPO + NAMES_EXTERNAL if n not in FAST]
# configurations per mode (pairs cap 2000 = SPEC / brief); the slow run draws 10^4 random sets per circuit
NCFG_FAST = dict(pairs=2000, rand=300, sameloc=300, merge=300)
NCFG_SLOW = dict(pairs=2000, rand=10_000, sameloc=300, merge=300)
SEED = 20261116
# the registry circuits this gate skips, with the reason they must have (preflight S3: a conventional export is the
# default export's law; the code_switching circuits have a random observable, which format="xdem" refuses). Any other
# conventional export or refusal FAILS.
EXPECTED_SKIP = {
    "distillation_15_1_3": ("conventional", ""),
    "cube_ccz": ("conventional", ""),
    "code_switching_faithful": ("refused", "non-deterministic observable"),
    "code_switching_rate": ("refused", "non-deterministic observable"),
}


def toint(row):
    v = 0
    for i in np.flatnonzero(np.asarray(row, bool)):
        v |= 1 << int(i)
    return v


def line_axes(line):
    """the axes a model line names: tau ∪ rot ∪ late (a set of axis ids)"""
    _, _, tau, rot, late = line
    return {a for a in range(tau.bit_length()) if tau >> a & 1} | set(rot) | set(late)


class Gate:
    """One circuit: its xdem file, the writer's atom map, the exact engine and the oracle law cache."""

    def __init__(self, name, text=None):
        self.name = name
        self.text = circuit(name) if text is None else text
        try:
            self.dem = xdem_export(self.text)
            got = None if self.dem.startswith("xdem 1") else ("conventional", "")
        except XtimDemError as e:
            got = ("refused", str(e))
        want = EXPECTED_SKIP.get(name)
        if got is not None or want is not None:     # only the pinned circuits skip, and only for their pinned reason
            assert got is not None and want is not None and got[0] == want[0] and want[1] in got[1], (name, got, want)
            pytest.skip(f"{name}: {want[0]} {want[1]}".rstrip())
        with explicit_exact():                                   # the export is explicit (J-b1)
            self.raw_atoms = ta.xdem_atom_lines(self.text)
        self.F = XdemModel.from_text(self.dem)
        self.nlines = len(self.F.lines)
        self.sim = compile_exact_branches(self.text)
        L0 = self.sim.exact_branches([], expectations=[])
        assert L0.rows_detectors.shape[0] == 1 and L0.uniform_detectors.shape[0] == 0
        assert L0.uniform_observables.shape[0] == 0
        self.ref = toint(np.concatenate([L0.rows_detectors[0], L0.rows_observables[0]]))
        self.D, self.O = L0.rows_detectors.shape[1], L0.rows_observables.shape[1]
        assert (self.D, self.O) == (self.F.D, self.F.O)
        self.refused = set(self.sim.refused_records)
        self.atoms = []                                          # (address, slot, line)
        for at in self.raw_atoms:
            site, addr, line = at["site"], at["addr"], at["line"]
            assert line < self.nlines
            if site == -1:
                j = addr[1]
                if j in self.refused:
                    continue
                self.atoms.append(({"record": j}, ("r", j), line))
            else:
                self.atoms.append(((site, addr[0], addr[1]), (site, addr[0]), line))
        self.line_atoms = defaultdict(list)
        for i, (_, _, line) in enumerate(self.atoms):
            if line >= 0:
                self.line_atoms[line].append(i)
        self.cache = {}
        self.t_oracle = 0.0
        self.n_oracle = 0

    def oracle(self, idx):
        """(offset, sorted fully-reduced coin basis) of xtim's exact law under atoms idx, or None (a mixture)."""
        key = tuple(sorted(idx))
        if key in self.cache:
            return self.cache[key]
        t = time.time()
        L = self.sim.exact_branches([self.atoms[i][0] for i in key], expectations=[])
        rows = np.concatenate([L.rows_detectors, L.rows_observables], 1)
        U = np.concatenate([L.uniform_detectors, L.uniform_observables], 1)
        p = np.asarray(L.rows_probabilities)
        gens = [toint(u) for u in U]
        r0 = toint(rows[0]) ^ self.ref
        out = None
        if rows.shape[0] > 1:
            d = [toint(r) ^ toint(rows[0]) for r in rows[1:]]
            uniform = np.ptp(p) <= 1e-12 and not (rows.shape[0] & (rows.shape[0] - 1))
            if uniform and (1 << (len(rref(gens + d)) - len(rref(gens)))) == rows.shape[0]:
                gens = gens + d
            else:
                gens = None
        if gens is not None:
            b = rref(gens)
            out = (reduce(r0, b), tuple(sorted(b.values())))
        self.t_oracle += time.time() - t
        self.n_oracle += 1
        self.cache[key] = out
        return out

    def lines_of(self, idx):
        return [self.atoms[i][2] for i in idx if self.atoms[i][2] >= 0]

    def ok_slots(self, idx):
        return len({self.atoms[i][1] for i in idx}) == len(idx)


def file_laws(text, cfgs):
    out = []
    for o, co in ta.xdem_laws(text, cfgs):
        b = rref([int(c, 16) for c in co])
        out.append((reduce(int(o, 16), b), tuple(sorted(b.values()))))
    return out


@lru_cache(maxsize=None)
def gate(name):
    return Gate(name)


def configs(G, mode, n, rng):
    """atom-index configurations of one mode (slot conflicts skipped)."""
    A = len(G.atoms)
    out = []
    if mode == "singles":
        out = [[i] for i in range(A)]
    elif mode == "pairs":
        lines = sorted(G.line_atoms)
        reps = {k: G.line_atoms[k][0] for k in lines}
        npairs = len(lines) * (len(lines) - 1) // 2
        if npairs <= n:
            pairs = list(itertools.combinations(lines, 2))
        else:
            seen = set()
            while len(seen) < n:
                a, b = rng.sample(lines, 2)
                seen.add((min(a, b), max(a, b)))
            pairs = sorted(seen)
        for a, b in pairs:
            ia, ib = reps[a], reps[b]
            if G.atoms[ia][1] == G.atoms[ib][1]:
                alt = [j for j in G.line_atoms[b] if G.atoms[j][1] != G.atoms[ia][1]]
                if not alt:
                    continue
                ib = alt[0]
            out.append([ia, ib])
    elif mode == "rand":
        tries = 0
        while len(out) < n and tries < 20 * n:
            tries += 1
            c = rng.sample(range(A), min(A, rng.randint(3, 12)))
            if G.ok_slots(c):
                out.append(c)
    elif mode == "sameloc":                                      # preflight S3: grouped by the axes a line names
        by = defaultdict(list)
        for i, (_, _, line) in enumerate(G.atoms):
            if line >= 0:
                for a in line_axes(G.F.lines[line]):
                    by[a].append(i)
        gl = sorted(a for a in by if len(by[a]) >= 2)
        tries = 0
        while gl and len(out) < n and tries < 20 * n:
            tries += 1
            a = rng.choice(gl)
            c = rng.sample(by[a], min(len(by[a]), rng.randint(2, 6)))
            if G.ok_slots(c):
                out.append(c)
    elif mode == "merge":
        multi = sorted(k for k, v in G.line_atoms.items() if len(v) >= 2)
        tries = 0
        while multi and len(out) < n and tries < 20 * n:
            tries += 1
            a, b = rng.sample(G.line_atoms[rng.choice(multi)], 2)
            c = [a, b] if rng.random() < 0.5 else [a, b] + rng.sample(range(A), rng.randint(1, 3))
            if len(set(c)) == len(c) and G.ok_slots(c):
                out.append(c)
    return out


def compare(G, cfgs, text=None, perm=None):
    """(agree, disagree, mixtures, examples) of the file `text` (default: the writer's) vs the exact engine; perm maps
    the writer's line ids to the ids of the same lines in `text` (a mutant that reorders lines)."""
    perm = perm or {}
    laws = file_laws(G.dem if text is None else text, [[perm.get(l, l) for l in G.lines_of(c)] for c in cfgs])
    agree = dis = mix = 0
    ex = []
    for c, fl in zip(cfgs, laws):
        o = G.oracle(c)
        if o is None:
            mix += 1
            continue
        if o == fl:
            agree += 1
        else:
            dis += 1
            if len(ex) < 3:
                ex.append(dict(atoms=[G.atoms[i][0] for i in c], lines=G.lines_of(c), oracle=o, file=fl))
    return agree, dis, mix, ex


# ── the atom map itself ───────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", FAST + [pytest.param(n, marks=pytest.mark.slow) for n in SLOW])
def test_atom_map_is_the_export_and_the_engine_sites(name):
    if not present(name):
        pytest.skip("external circuit absent")
    G = gate(name)
    # every alternative with p > 0 of every noise site, and every readout flip, exactly once (as a set: an E chain
    # may repeat a word) -- enumerated independently from the channel definitions (_exact_law.instances)
    want = {(s, t, w) for s, site in enumerate(G.sim.noise_sites()) for t, al in instances(site) for w, _ in al}
    got = {(a["site"], a["addr"][0], a["addr"][1]) for a in G.raw_atoms if a["site"] >= 0}
    assert got == want
    assert {a["addr"][1] for a in G.raw_atoms if a["site"] == -1} == {j for j, _ in readout_records(G.text)}
    # every file line carries at least one atom
    assert sorted(G.line_atoms) == list(range(G.nlines))


def run_gate(name, ncfg):
    G = gate(name)
    rng = random.Random(SEED)
    t0 = time.time()
    table = {}
    total_mix = 0
    for mode in ("singles", "pairs", "rand", "sameloc", "merge"):
        cfgs = configs(G, mode, ncfg.get(mode, 0), rng)
        a, d, m, ex = compare(G, cfgs)
        table[mode] = (len(cfgs), a, d, m)
        total_mix += m
        assert d == 0, (name, mode, ex)
    print(f"\n[xdem per-config] {name}: atoms={len(G.atoms)} lines={G.nlines} "
          + " ".join(f"{k}={v[0]}(agree {v[1]}, dis {v[2]}, mix {v[3]})" for k, v in table.items())
          + f" oracle_calls={G.n_oracle} oracle_s={G.t_oracle:.1f} total_s={time.time() - t0:.1f}")
    assert total_mix == 0
    assert table["singles"][0] == len(G.atoms) and table["pairs"][0] > 0
    return table


@pytest.mark.parametrize("name", FAST)
def test_per_config_exact_fast(name):
    run_gate(name, NCFG_FAST)


@pytest.mark.slow
@pytest.mark.parametrize("name", SLOW)
def test_per_config_exact_slow(name):
    if not present(name):
        pytest.skip("external circuit absent")
    run_gate(name, NCFG_SLOW)


def test_gate_covers_observable_and_late_circuits():
    # coverage obligations: an O >= 1 circuit and a circuit with a late rotation (E^>) in the default per-config gate.
    # No FAST circuit has lines at or after t_c (index >= tc). The slow set has them: toy2obs (2 lines), rm15rep (1)
    # and bt27_p1e-4 / bt27_p1e-3 (9 each, the observable readout flips); the two small ones are asserted here. The
    # fast law fuzz asserts such lines too (test_xdem_fuzz.py).
    assert gate("cultivation_d3_faithful").O >= 1 and gate("ex422_obs").O == 2
    assert any(gate(n).F.late_axes() for n in FAST)
    assert all(gate(n).F.tc < gate(n).nlines for n in ("toy2obs", "rm15rep")) and "toy2obs" in SLOW


# ── mutation legs ─────────────────────────────────────────────────────────────────────────────────────────────────
MUTANTS = ["negate_E", "drop_tau_bit", "swap_noncommuting"]


def fields(line):
    """an error line as (head 'error(p)', {"t": targets, "x": [...], "rot": [...], "late": [...]})"""
    tok = line.split()
    out, cur = {"t": [], "x": [], "rot": [], "late": []}, "t"
    for t in tok[1:]:
        if t in ("x", "rot", "late"):
            cur = t
        else:
            out[cur].append(t)
    return tok[0], out


def unfields(head, f):
    s = head + "".join(" " + t for t in f["t"])
    for k in ("x", "rot", "late"):
        if f[k]:
            s += f" {k} " + " ".join(f[k])
    return s


def mutate(text, kind, rng):
    """one seeded edit of the xdem text -> (text, perm): perm maps old line ids to new ones (the swap), so a
    configuration still fires the same lines and only the edit differs; the text is unchanged when the file offers
    no candidate"""
    lines = text.splitlines()
    perm = {}
    tc = int(lines[0].split()[-1])
    err = [i for i, l in enumerate(lines) if l.startswith("error")]
    F = {i: fields(lines[i]) for i in err}
    rotated = lambda i: {t[1:] for t in F[i][1]["rot"] + F[i][1]["late"]}  # noqa: E731
    if kind == "negate_E":                                 # one line's E negated
        cands = [i for i in err if rotated(i)]
        if not cands:
            return text, perm
        i = rng.choice(cands)
        h, f = F[i]
        for k in ("rot", "late"):
            f[k] = [("-" if t[0] == "+" else "+") + t[1:] for t in f[k]]
        lines[i] = unfields(h, f)
    elif kind == "drop_tau_bit":                           # a tau bit on an axis a later line rotates about
        cands = [(i, q) for i in err for q in F[i][1]["x"] if any(q in rotated(j) for j in err if j > i)]
        if not cands:
            return text, perm
        i, q = rng.choice(cands)
        h, f = F[i]
        f["x"] = [t for t in f["x"] if t != q]
        lines[i] = unfields(h, f)
    elif kind == "swap_noncommuting":                      # neighbours on one side of t_c; the earlier's x meets
        pos = {i: n for n, i in enumerate(err)}            # the later's rotation
        cands = [(a, b) for a, b in zip(err, err[1:]) if (pos[a] < tc) == (pos[b] < tc)
                 and set(F[a][1]["x"]) & rotated(b)]
        if not cands:
            return text, perm
        a, b = rng.choice(cands)
        lines[a], lines[b] = lines[b], lines[a]
        perm = {pos[a]: pos[b], pos[b]: pos[a]}
    return "\n".join(lines) + "\n", perm


SUBJECTS = ["ex422_obs", "ccz_toy_obs", "ccz_toy_obs_p5", "rep3T_pre", "toy2obs", "rm15rep"]


@pytest.mark.parametrize("kind", MUTANTS)
def test_mutant_disagrees(kind):
    """each mutation kind is caught on at least one subject circuit: a configuration exercising the edit disagrees
    with the exact engine (or, for drop_tau_bit, the reader refuses the line as not an operator, SPEC §3.5(c)).
    The law is invariant under negating every exponent at once, so a single edit can be law-invisible on a given
    circuit; each subject gets 8 seeded edits and the table is printed."""
    caught, table = False, []
    for name in SUBJECTS:
        G = gate(name)
        rng = random.Random(SEED + MUTANTS.index(kind) + 1)
        for attempt in range(8):
            text, perm = mutate(G.dem, kind, rng)
            if text == G.dem:
                table.append((name, "no candidate"))
                break
            try:
                ta.xdem_parse(text)
            except XtimXdemError as e:
                assert kind == "drop_tau_bit" and "not an operator" in str(e), str(e)
                table.append((name, "refused: not an operator"))
                caught = True
                break
            cf = configs(G, "singles", 0, rng) + configs(G, "pairs", 500, rng) + configs(G, "rand", 300, rng)
            a, d, m, _ = compare(G, cf, text, perm)
            if d:
                table.append((name, f"disagrees on {d} of {len(cf)}"))
                caught = True
                break
        else:
            table.append((name, "law-invisible in 8 edits"))
    print(f"\n[xdem mutant] {kind}: {table}")
    assert caught, table


def test_identity_mutant_agrees():
    # the mutation harness itself: a perm-only relabelling (no edit) agrees everywhere
    G = gate("ex422_obs")
    rng = random.Random(SEED)
    cf = configs(G, "singles", 0, rng) + configs(G, "pairs", 500, rng) + configs(G, "rand", 300, rng)
    a, d, m, _ = compare(G, cf, G.dem, {})
    assert d == 0 and m == 0 and a == len(cf)


def test_wrong_atom_map_is_detected():
    # test-of-test for the atom map (xdem_atom_lines' frame-provenance inversion): relabelling every atom's letters
    # X <-> Z (what a wrong frame inversion through an H would do) must make single-atom configurations disagree
    G = gate("cultivation_d3_faithful")
    swap = str.maketrans("XZ", "ZX")
    W = Gate.__new__(Gate)
    W.__dict__.update(G.__dict__)
    W.atoms = [((a[0], a[1], a[2].translate(swap)) if isinstance(a, tuple) else a, s, l) for a, s, l in G.atoms]
    W.cache = {}
    a, d, m, _ = compare(W, [[i] for i in range(len(W.atoms))])
    print(f"\n[xdem atom-map mutant] cultivation_d3_faithful X<->Z: agree {a}, disagree {d}, mix {m}")
    assert d >= 1


# a frame commuted through noise (H; noise; H -> relabelled noise in the deferred stream) and a readout flip read by
# coherentized feedback (its fold): the two non-trivial branches of the atom map's address translation
FRAMED = """RX 0 1
R 2
H 0
PAULI_CHANNEL_1(0.01, 0.02, 0.03) 0
DEPOLARIZE2(0.02) 0 1
H 0
T 0
H 1
E(0.02) X1 Z2
H 1
T 1
M(0.05) 2
CX rec[-1] 0
T_DAG 0
T_DAG 1
MX 0 1
DETECTOR rec[-1]
DETECTOR rec[-2]
DETECTOR rec[-3]
"""


def test_framed_noise_and_fold_atoms():
    G = Gate("framed", FRAMED)
    want = {(s, t, w) for s, site in enumerate(G.sim.noise_sites()) for t, al in instances(site) for w, _ in al}
    assert {(a["site"], a["addr"][0], a["addr"][1]) for a in G.raw_atoms if a["site"] >= 0} == want
    assert {a["addr"][1] for a in G.raw_atoms if a["site"] == -1} == {0}
    rng = random.Random(SEED)
    cf = [c for mode in ("singles", "pairs", "rand", "sameloc", "merge") for c in configs(G, mode, 300, rng)]
    a, d, m, ex = compare(G, cf)
    print(f"\n[xdem per-config] framed hand circuit: atoms={len(G.atoms)} configs={len(cf)} agree {a} dis {d} mix {m}")
    assert d == 0 and m == 0, ex



# fuzz circuit 63 of seed 5201 (tests/_xdem_fuzz_gen.py): its file carries rotations of both signs on axes reached
# through gates of both signs, so a writer that drops the axis sign of a residue (the C++ mutant
# `writer_drop_axis_sign`, task-5 report) is wrong on it (TV 3.7e-2); before this circuit only the slow fuzz caught it
SIGN_TOY = """R 0 1 2
S 1
SWAP 0 1
CS_DAG 0 2
CCZ 0 2 1
DEPOLARIZE2(0.2) 0 1
CS_DAG 0 2
X 0
MXX(0.15) !1 2
CS 2 0
CCZ 0 1 2
MY 2
MPP(0.15) X1*Z2*Y0
MPP(0.2) Y0*Z2
DETECTOR rec[-4] rec[-3] rec[-2] rec[-1]
MRX !1
MRY 1
DETECTOR rec[-6] rec[-5] rec[-2]
DETECTOR rec[-6] rec[-5] rec[-4] rec[-3]
T_DAG 0
SQRT_XX 0 1
MY 0
MX !1
M 2
DETECTOR rec[-7] rec[-6] rec[-5]
DETECTOR rec[-9] rec[-8] rec[-5]
OBSERVABLE_INCLUDE(0) rec[-7] rec[-6] rec[-5]
OBSERVABLE_INCLUDE(1) rec[-9] rec[-8] rec[-7] rec[-6]
OBSERVABLE_INCLUDE(2) rec[-9] rec[-8] rec[-5]
"""


def every_config(G):
    """every fault configuration: per channel instance (slot) none or one of its atoms"""
    slots = defaultdict(list)
    for i, (_, slot, _) in enumerate(G.atoms):
        slots[slot].append(i)
    out = [[]]
    for v in slots.values():
        out = [c + ([i] if i is not None else []) for c in out for i in [None] + v]
    return [c for c in out if c]


def _hand(name):
    from test_xdem_export import TDAG_CYCLE
    return {"tdag_cycle": TDAG_CYCLE, "sign_toy": SIGN_TOY}[name]


@pytest.mark.parametrize("name,nlines", [("tdag_cycle", 4), ("sign_toy", 5)])
def test_hand_circuit_every_config(name, nlines):
    """every fault configuration of a small hand circuit against the exact engine:
    tdag_cycle -- the merge erratum's counterexample (ERRATA E1): the pre-erratum merge (one line per key across a
    visible swap) fails here; sign_toy -- a writer that drops a residue's axis sign fails here (see SIGN_TOY)."""
    G = Gate(name, _hand(name))
    cf = every_config(G)
    a, d, m, ex = compare(G, cf)
    print(f"\n[xdem per-config] {name}: atoms={len(G.atoms)} lines={G.nlines} configs={len(cf)} agree {a} dis {d} "
          f"mix {m}")
    assert d == 0 and m == 0 and a == len(cf), ex
    assert G.nlines == nlines


def test_expected_skips_are_pinned():
    # the pinned names are registry names, and an unpinned conventional export fails instead of skipping
    assert set(EXPECTED_SKIP) <= set(NAMES_INREPO)
    with pytest.raises(AssertionError):
        Gate("not_pinned", "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n")
