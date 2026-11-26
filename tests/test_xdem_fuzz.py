"""SPEC §6 T3: random circuits. Law part: the xdem file's exact outcome law vs xtim's exact law, TV <= 1e-12. Loop
part: per (configuration, d), score_lines "optimal" == the engine's sign-free projector form and "r2b" == the engine's
physical Pauli at t_c (tests/_xdem_engine.py), exactly; the T3 mutations must fail."""
import random
from collections import Counter

import numpy as np
import pytest
import xtim
from xtim import _xtim_testapi as ta
from xtim.errors import XtimDemError, XtimRejectError, XtimXdemError

from _exact_law import xtim_exact_law
from _xdem_fuzz_gen import generate, generate_ch
from _xdem_model import XdemModel, exact_file_law


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def negate_one_E(text, rng):
    """mutation (test-of-test): one rotating line's exponents negated, or None when no line rotates"""
    lines = text.splitlines()
    rot = [i for i, l in enumerate(lines) if l.startswith("error") and (" rot " in l or " late " in l)]
    if not rot:
        return None
    i = rng.choice(rot)
    tok = lines[i].split()
    cur = None
    for j, t in enumerate(tok):
        if t in ("x", "rot", "late"):
            cur = t
        elif cur in ("rot", "late"):
            tok[j] = ("-" if t[0] == "+" else "+") + t[1:]
    lines[i] = " ".join(tok)
    return "\n".join(lines) + "\n"


def run(seed, count, min_checked):
    stats = Counter()
    rng = random.Random(seed)
    max_tv = 0.0
    gen = {}
    for text in generate(seed, count, stats=gen):
        try:
            x = xtim.Circuit(text).detector_error_model_text(format="xdem")
        except XtimDemError as e:
            stats["refused:" + str(e).split(":")[0][:40]] += 1
            continue
        if not x.startswith("xdem 1"):
            stats["conventional"] += 1
            continue
        M = XdemModel.from_text(x)
        if len(M.lines) > 14:
            stats["too_many_lines"] += 1
            continue
        law, tail = xtim_exact_law(text)
        assert tail == 0.0
        d = tv(exact_file_law(x), law)
        stats["checked"] += 1
        max_tv = max(max_tv, d)
        stats["late"] += " late " in x
        stats["feedback"] += " rec[" in text and any(l.startswith("CX rec") for l in text.splitlines())
        stats["obs>1"] += M.O > 1
        stats["lines at or after t_c"] += len(M.lines) - M.tc
        assert d <= 1e-12, (text, x, d)
        if stats["mutant: files"] < 40:                   # test-of-test: a negated E must be seen somewhere
            m = negate_one_E(x, rng)
            if m is not None:
                stats["mutant: files"] += 1
                stats["mutant: disagreeing"] += tv(exact_file_law(m), law) > 1e-12
    print(dict(stats), f"max_tv={max_tv:.3e}")
    print("generator:", gen)
    assert stats["checked"] >= min_checked
    assert stats["lines at or after t_c"] > 0
    assert stats["mutant: disagreeing"] >= 1


def test_law_fuzz_fast():
    run(5101, 200, min_checked=50)


@pytest.mark.slow
def test_law_fuzz_slow():
    # SPEC §6 T3 ">= 2 000 random circuits" = at least 2000 COMPARED (extended) circuits (controller ruling, Task 5
    # fix round 1); about 39 % of the generated circuits export an extended file
    run(5201, 5400, min_checked=2000)


# ── Hadamard-test CH circuits (CH in the xdem export: two pi/8 Y-rotations around a CZ) ─────────────────────────────
def run_ch(seed, count, min_checked):
    """Every aligned circuit (but one with a multi-term channel stim's approximate_disjoint_errors would be needed
    for, counted) exports an extended file whose law is xtim's exact law of the CH circuit: the whole law
    (TV <= 1e-12) when the file has <= 14 lines, else per fault subset (singles and pairs, exact over GF(2)). An
    unaligned one is rejected (the class reject), refused by the diag check or the determinism check, or (no CH meets
    its T-state) checked alike."""
    from test_xdem_oracle_configs import Gate, compare
    stats = Counter()
    max_tv = 0.0
    for k, (text, unaligned) in enumerate(generate_ch(seed, count, flags=True)):
        try:
            x = xtim.Circuit(text).detector_error_model_text(format="xdem")
        except XtimDemError as e:
            msg = str(e)
            if "approximate_disjoint_errors" in msg:             # a multi-term channel with no exact conversion
                stats["refused: approximate_disjoint_errors"] += 1
                continue
            assert unaligned, (text, msg)
            assert "does not stay diagonal" in msg or "non-deterministic" in msg or "nondeterministic" in msg, msg
            stats["unaligned refused"] += 1
            continue
        except XtimRejectError as e:                             # non-commuting magic: the class reject (as before)
            assert unaligned and "outside the simulable class" in str(e), (text, str(e))
            stats["unaligned rejected"] += 1
            continue
        if not x.startswith("xdem 1"):                            # no fault rotates: the default export, bytes
            assert x == xtim.Circuit(text).detector_error_model_text(), text
            stats["conventional (no fault rotates)"] += 1
            continue
        M = XdemModel.from_text(x)
        if len(M.lines) <= 14:
            law, tail = xtim_exact_law(text)
            assert tail == 0.0
            d = tv(exact_file_law(x), law)
            max_tv = max(max_tv, d)
            assert d <= 1e-12, (text, x, d)
            stats["checked: whole law"] += 1
        else:
            G = Gate(f"ch_fuzz_{seed}_{k}", text)
            n = len(G.atoms)
            cfgs = [c for c in [[i] for i in range(n)] + [[i, j] for i in range(n) for j in range(i + 1, n)]
                    if G.ok_slots(c)]
            agree, dis, mix, ex = compare(G, cfgs)
            assert dis == 0 and mix == 0, (text, ex)
            stats["checked: per subset"] += 1
            stats["subsets"] += len(cfgs)
        stats["checked"] += 1
        stats["late"] += " late " in x
        stats["unaligned exported"] += unaligned
        stats["multi-pair CH line"] += any(l.startswith("CH ") and len(l.split()) > 3 for l in text.splitlines())
        stats["control index < target"] += any(l.startswith("CH ") and int(l.split()[1]) < int(l.split()[2])
                                                for l in text.splitlines())
    print(dict(stats), f"max_tv={max_tv:.3e}")
    assert stats["checked"] >= min_checked and stats["late"] > 0
    assert stats["unaligned refused"] + stats["unaligned rejected"] > 0         # the refusal boundary is exercised
    assert stats["multi-pair CH line"] > 0 and stats["control index < target"] > 0


def test_law_fuzz_ch_fast():
    run_ch(7301, 100, min_checked=60)


@pytest.mark.slow
def test_law_fuzz_ch_slow():
    run_ch(7302, 1500, min_checked=900)


# ── T3, the loop part: per (configuration, d), score_lines == the engine oracles (tests/_xdem_engine.py) ──────────────
def _errs(x):
    """(text lines, the indices of the error lines in file order)"""
    t = x.splitlines()
    return t, [i for i, l in enumerate(t) if l.startswith("error")]


def _sect(tok):
    """an error line's tokens split into (head, targets, x, rot, late)"""
    out, cur = {"t": [], "x": [], "rot": [], "late": []}, "t"
    for t in tok[1:]:
        if t in ("x", "rot", "late"):
            cur = t
        else:
            out[cur].append(t)
    return tok[0], out


def _join(head, s):
    tok = [head] + s["t"]
    for k in ("x", "rot", "late"):
        if s[k]:
            tok += [k] + s[k]
    return " ".join(tok)


def _odd(M, i):
    _, _, _, rot, late = M.lines[i]
    return {a for a, e in M.total(rot, late).items() if e % 2}


def mut_drop_tau(x, M, rng):
    """one tau bit dropped on an axis a later line rotates about (ids unchanged)"""
    t, el = _errs(x)
    later = [set(M.lines[j][3]) | set(M.lines[j][4]) for j in range(len(M.lines))]
    cand = [(i, a) for i in range(len(M.lines)) for a in range(M.A)
            if M.lines[i][2] >> a & 1 and any(a in later[j] for j in range(i + 1, len(M.lines)))]
    if not cand:
        return None
    i, a = rng.choice([c for c in cand if M.img(c[1])] or cand)
    head, s = _sect(t[el[i]].split())
    s["x"] = [q for q in s["x"] if q != f"Q{a}"]
    t[el[i]] = _join(head, s)
    return "\n".join(t) + "\n", list(range(len(M.lines))), [i]


def _swap(x, M, i, j):
    t, el = _errs(x)
    t[el[i]], t[el[j]] = t[el[j]], t[el[i]]
    ids = list(range(len(M.lines)))
    ids[i], ids[j] = j, i
    return "\n".join(t) + "\n", ids, [i, j]


def mut_swap_noncommuting(x, M, rng):
    """two adjacent lines on one side of t_c whose operators do not commute, swapped"""
    cand = [i for i in range(len(M.lines) - 1) if (i < M.tc) == (i + 1 < M.tc) and
            (any(M.lines[i][2] >> a & 1 for a in _odd(M, i + 1)) or
             any(M.lines[i + 1][2] >> a & 1 for a in _odd(M, i)))]
    if not cand:
        return None
    vis = [i for i in cand if any(M.img(a) for a in _odd(M, i) | _odd(M, i + 1))]
    i = rng.choice(vis or cand)
    return _swap(x, M, i, i + 1)


def mut_move_tc(x, M, rng):
    """the line at tc - 1 and the one at tc swapped, tc kept (one line moved across t_c; r2b only)"""
    if not 0 < M.tc < len(M.lines):
        return None
    return _swap(x, M, M.tc - 1, M.tc)


def merge_visible(M, i, j):
    """a sufficient condition for an (f, x) merge of lines i, j to change some optimal score: a line l and an
    observable k whose S = {odd axes of l with img on L_k} is a product with no gauge part (a stabilizer: K's string
    there is deterministic) and meets tau_i ^ tau_j an odd number of times, so the string's sign tau_E(S) (SPEC §4.3
    step 6) reads the tau the merge erases. On 4000 fuzz circuits (213 candidate pairs) 5 pairs satisfy it, all
    visible; 11 pairs are visible."""
    gmask = ((1 << M.r) - 1) << M.n
    dl = M.lines[i][2] ^ M.lines[j][2]
    for l in range(len(M.lines)):
        od = _odd(M, l)
        for k in range(M.O):
            S = [a for a in od if M.img(a) >> (M.D + k) & 1]
            phi = 0
            for a in S:
                phi ^= M.axes[a][1]
            if S and not phi & gmask and bin(sum(1 << a for a in S) & dl).count("1") % 2:
                return True
    return False


def _merge_candidates(M):
    L = len(M.lines)
    return [(i, j) for i in range(L) for j in range(i + 1, L) if (i < M.tc) == (j < M.tc) and
            M.lines[i][1] == M.lines[j][1] and M.lines[i][3] == M.lines[j][3] and M.lines[i][4] == M.lines[j][4] and
            M.lines[i][2] != M.lines[j][2]]


def mut_fx_merge(x, M, rng):
    """(f, x) merging: two lines with equal targets and rotations but different x (tau), on one side of t_c, merged
    into the first (its x kept), probabilities combined (optimal only)"""
    L = len(M.lines)
    cand = _merge_candidates(M)
    if not cand:
        return None
    vis = [c for c in cand if merge_visible(M, *c)]
    rng.shuffle(vis)
    rest = [c for c in cand if c not in vis]
    rng.shuffle(rest)
    out = []
    for i, j in (vis + rest)[:4]:      # every candidate up to 4: few (f, x) merges are visible (about 5 %, T3 notes)
        t, el = _errs(x)
        pi, pj = M.lines[i][0], M.lines[j][0]
        head, s = _sect(t[el[i]].split())
        t[el[i]] = _join(f"error({pi * (1 - pj) + pj * (1 - pi):.17g})", s)
        del t[el[j]]
        if j < M.tc:
            t = [l.replace(f" tc {M.tc}", f" tc {M.tc - 1}") if l.startswith("xdem 1") else l for l in t]
        ids = [i if k == j else (k - 1 if k > j else k) for k in range(L)]
        out.append(("\n".join(t) + "\n", ids, [i, j]))
    return out


def mut_negate_E(x, M, rng):
    """one rotating line's exponents negated (ids unchanged)"""
    rot = [i for i in range(len(M.lines)) if M.lines[i][3] or M.lines[i][4]]
    if not rot:
        return None
    # prefer a line sharing an axis with a readout image with another rotating line: negating one line's whole E is
    # invisible in its own law (the deterministic parities' (sum eta)/2 changes by an even number)
    seen = [i for i in rot if any(M.img(a) and any(a in set(M.lines[l][3]) | set(M.lines[l][4]) for l in rot if l != i)
                                  for a in set(M.lines[i][3]) | set(M.lines[i][4]))]
    i = rng.choice(seen or rot)
    t, el = _errs(x)
    head, s = _sect(t[el[i]].split())
    for k in ("rot", "late"):
        s[k] = [("-" if q[0] == "+" else "+") + q[1:] for q in s[k]]
    t[el[i]] = _join(head, s)
    return "\n".join(t) + "\n", list(range(len(M.lines))), [i]


LOOP_MUTANTS = {"negate E": (mut_negate_E, ("optimal", "r2b")),
                "drop tau": (mut_drop_tau, ("optimal", "r2b")),
                "swap non-commuting": (mut_swap_noncommuting, ("optimal", "r2b")),
                "move across t_c": (mut_move_tc, ("r2b",)),
                "(f, x) merge": (mut_fx_merge, ("optimal",))}


def loop_rows(text, M, rng):
    """[(E, Ê, d)]: <= 20 configurations E of lines (singles, pairs, random <= 4; one representative atom each),
    Ê in {E, a random <= 2-line set}, every d with Pr(d | E) > 0 by the engine"""
    from _xdem_engine import engine_detector_law, representative_atoms
    rep = representative_atoms(text)
    L = len(M.lines)
    have = sorted(rep)
    cfgs = [[i] for i in rng.sample(have, min(8, len(have)))]
    for _ in range(6):
        if len(have) >= 2:
            cfgs.append(sorted(rng.sample(have, 2)))
    while len(cfgs) < 20:
        cfgs.append(sorted(rng.sample(have, rng.randint(1, min(4, len(have))))))
    out, seen = [], set()
    for E in cfgs:
        if tuple(E) in seen:
            continue
        seen.add(tuple(E))
        law = engine_detector_law(text, [rep[i] for i in E])
        wrong = sorted(rng.sample(range(L), rng.randint(0, min(2, L))))
        for Eh in (E, wrong):
            for d, q in sorted(law.items()):
                if q > 0:
                    out.append((E, Eh, d))
    return out, rep


def targeted_rows(text, M, rep, touched):
    """[(E, Ê, d)] around the mutated lines m: E in {{m}, {m, l}, {l}}, Ê in {[], {m}, {l}, {m, l}} for every other
    line l, every d with Pr(d | E) > 0 (a self-consistent row E = Ê scores 1 under any model, so the mutants need
    wrong hypotheses next to the edit)"""
    from _xdem_engine import engine_detector_law
    out, seen = [], set()
    for m in touched:
        for l in range(len(M.lines)):
            if l == m:
                continue
            for E in ([m], sorted([m, l]), [l]):
                law = engine_detector_law(text, [rep[i] for i in E])
                for Eh in ([], [m], [l], sorted([m, l])):
                    if (tuple(E), tuple(Eh)) in seen:
                        continue
                    seen.add((tuple(E), tuple(Eh)))
                    out += [(E, Eh, d) for d, q in sorted(law.items()) if q > 0]
    return out


def engine_rows(text, M, rep, rows, corr, stats=None):
    """the engine oracle per row (optimal rows with a CZ edge to a lossy observable are counted, and checked)"""
    from _xdem_engine import engine_r2b, engine_vdag_proj, vdag_args
    out = []
    for E, Eh, d in rows:
        act, hyp = [rep[i] for i in E], [rep[i] for i in Eh]
        if corr == "optimal":
            piv, lossy, edge = vdag_args(M, Eh, d)
            if stats is not None:
                stats["rows optimal lossy"] += bool(lossy)
                stats["rows optimal lossy with a CZ edge to a lossy observable"] += edge
            v = engine_vdag_proj(text, act, hyp, d, piv, lossy)
            assert v is not None, (text, E, d)
        else:
            v = engine_r2b(text, act, hyp, d)
            assert v is not None, (text, E, d)
        out.append(v)
    return out


def score_rows(x, rows, D, corr, ids=None):
    from test_xdem_lines import csr
    mp = (lambda l: [ids[i] for i in l]) if ids is not None else (lambda l: l)
    fo, fi = csr([mp(E) for E, _, _ in rows])
    ho, hi = csr([mp(Eh) for _, Eh, _ in rows])
    dets = np.zeros((len(rows), (D + 7) // 8), np.uint8)
    for k, (_, _, d) in enumerate(rows):
        dets[k] = np.frombuffer(int(d).to_bytes((D + 7) // 8, "little"), np.uint8)
    return ta.xdem_lines_fixed(x, fo, fi, dets, ho, hi, corr)[0]


def run_loop(seed, target, mutant_files=40):
    stats = Counter()
    rng = random.Random(seed)
    chunk = 0
    while stats["compared"] < target:
        for text in generate(seed * 1000 + chunk, 100):
            if stats["compared"] >= target:
                break
            try:
                x = xtim.Circuit(text).detector_error_model_text(format="xdem")
            except XtimDemError:
                stats["refused"] += 1
                continue
            if not x.startswith("xdem 1"):
                stats["conventional"] += 1
                continue
            M = XdemModel.from_text(x)
            if len(M.lines) > 14:
                stats["too_many_lines"] += 1
                continue
            stats["compared"] += 1
            rows, rep = loop_rows(text, M, rng)
            r2b = ta.xdem_facts(x)["r2b_available"]
            stats["r2b available"] += bool(r2b)
            corrs = ("optimal",) + (("r2b",) if r2b else ())
            for corr in corrs:
                got = score_rows(x, rows, M.D, corr)
                e = engine_rows(text, M, rep, rows, corr, stats)
                bad = [k for k in range(len(rows)) if e[k] is not None and got[k] != e[k]]
                assert not bad, (corr, text, x, [(rows[k], got[k], e[k]) for k in bad[:5]])
                stats[f"rows {corr}"] += len(rows)
                stats[f"rows {corr} pass<1"] += sum(v is not None and v < 1 for v in e)
                if corr == "optimal":                       # ERRATA E4: no Ê = E row below "none" (a theorem)
                    gn = score_rows(x, rows, M.D, "none")
                    stats["rows optimal with Ê = E below none"] += int(sum(
                        rows[k][0] == rows[k][1] and got[k] < gn[k] for k in range(len(rows))))
            if stats["mutant files"] >= mutant_files:
                continue
            stats["mutant files"] += 1                      # T3 mutations: each must fail somewhere
            for kind, (fn, kcorrs) in LOOP_MUTANTS.items():
                m = fn(x, M, rng)
                for mx, ids, touched in ([] if m is None else m if isinstance(m, list) else [m]):
                    trows = targeted_rows(text, M, rep, touched)
                    for corr in (c for c in kcorrs if c in corrs):
                        k = f"mutant {kind} / {corr}: "
                        stats[k + "tried"] += 1
                        try:
                            got = score_rows(mx, trows, M.D, corr, ids)
                        except XtimXdemError:
                            stats[k + "refused"] += 1
                            continue
                        e = engine_rows(text, M, rep, trows, corr)
                        g0 = score_rows(x, trows, M.D, corr)
                        assert all(e[r] is None or g0[r] == e[r] for r in range(len(trows))), (kind, corr, text)
                        stats[f"rows {corr} (targeted)"] += len(trows)
                        stats[k + "disagreeing"] += any(e[r] is not None and got[r] != e[r] for r in range(len(trows)))
        chunk += 1
    # the (f, x) merge mutant is visible on few pairs (merge_visible): its own pool of fuzz circuits that have one
    from _xdem_engine import representative_atoms
    pool = 0
    while stats["mutant (f, x) merge / optimal: disagreeing"] == 0 and pool < 3000:
        for text in generate(seed * 1000 + 999_000 + pool // 100, 100):
            pool += 1
            try:
                x = xtim.Circuit(text).detector_error_model_text(format="xdem")
            except XtimDemError:
                continue
            if not x.startswith("xdem 1"):
                continue
            M = XdemModel.from_text(x)
            if len(M.lines) > 14 or not any(merge_visible(M, *c) for c in _merge_candidates(M)):
                continue
            stats["merge pool files"] += 1
            rep = representative_atoms(text)
            mx, ids, touched = mut_fx_merge(x, M, rng)[0]
            trows = targeted_rows(text, M, rep, touched)
            e = engine_rows(text, M, rep, trows, "optimal")
            g0 = score_rows(x, trows, M.D, "optimal")
            assert all(e[r] is None or g0[r] == e[r] for r in range(len(trows)))
            got = score_rows(mx, trows, M.D, "optimal", ids)
            stats["mutant (f, x) merge / optimal: tried"] += 1
            stats["mutant (f, x) merge / optimal: disagreeing"] += any(
                e[r] is not None and got[r] != e[r] for r in range(len(trows)))
            if stats["mutant (f, x) merge / optimal: disagreeing"]:
                break
    stats["merge pool scanned"] = pool
    print(dict(sorted(stats.items())))
    assert stats["compared"] >= target and stats["r2b available"] > 0
    assert stats["rows optimal pass<1"] > 0 and stats["rows r2b pass<1"] > 0
    # under Ê = E a lossy parity the hypothesis fixes commutes with every non-lossy rotated readout, so K's CZ edges add
    # no coin and, with c in the support, "optimal" = 2^-k' >= "none" (ERRATA E4 clarification, E4 review item 2)
    assert stats["rows optimal with Ê = E below none"] == 0, dict(stats)
    for kind, (_, corrs) in LOOP_MUTANTS.items():
        for corr in corrs:
            k = f"mutant {kind} / {corr}: "
            assert stats[k + "tried"] > 0, (kind, corr)
            if kind == "move across t_c":                  # a rotating line past t_c: R2b refuses the file by name
                assert stats[k + "disagreeing"] + stats[k + "refused"] >= 1, (kind, corr, dict(stats))
            else:
                assert stats[k + "disagreeing"] >= 1, (kind, corr, dict(stats))


def test_loop_fuzz_fast():
    run_loop(53, 100, mutant_files=100)


@pytest.mark.slow
def test_loop_fuzz_slow():
    run_loop(54, 2000, mutant_files=400)
