"""EXP-fix DEM comparator (owner ruling 2026-10-21: DEM probabilities move within 1e-6 relative).

compare(old, new) -> (issues, flips) for two DEM texts:
  * every non-`error` line identical, in order (detector / coordinate / observable lines);
  * the mechanism SETS equal, a mechanism being the XOR of its `^` components, with |q| < 1e-15 counted as absent on
    both sides;
  * every probability within `rel` (1e-6) relative;
  * decompositions equal, OR both splits made of real graphlike model edges whose observable-mass scores (the
    decompose_acc criterion, recomputed from the NEW model's edge probabilities) agree within 1e-9 relative: a
    rounding-broken tie, reported in `flips` (review M7b).
"""
import re

_ERR = re.compile(r"^error\(([^)]*)\)\s*(.*)$")


def _parse(text):
    other, mech = [], {}
    for line in text.splitlines():
        s = line.strip()
        m = _ERR.match(s)
        if not m:
            other.append(s)
            continue
        p = float(m.group(1))
        comps = [tuple(sorted(c.split(), key=lambda t: (t[0] != "D", int(t[1:])))) for c in m.group(2).split("^")]
        acc = set()
        for c in comps:
            acc ^= set(c)
        key = tuple(sorted(acc, key=lambda t: (t[0] != "D", int(t[1:]))))
        if key in mech:
            raise ValueError(f"duplicate mechanism {key}")
        mech[key] = (p, tuple(comps))
    return other, mech


def _ndet(c):
    return sum(1 for t in c if t[0] == "D")


def _score(comps, edges):
    s = 0.0
    for c in comps:
        if _ndet(c) > 2 or c not in edges:
            return None                       # not a split into real model edges
        if any(t[0] == "L" for t in c):
            s += edges[c]
    return s


def compare(old, new, rel=1e-6, tiny=1e-15):
    issues, flips = [], []
    oo, mo = _parse(old)
    on, mn = _parse(new)
    if oo != on:
        issues.append(("non-error lines differ", len(oo), len(on)))
    ko = {k for k, (p, _) in mo.items() if abs(p) >= tiny}
    kn = {k for k, (p, _) in mn.items() if abs(p) >= tiny}
    for k in sorted(ko - kn):
        issues.append(("mechanism missing in new", k, mo[k][0]))
    for k in sorted(kn - ko):
        issues.append(("mechanism new", k, mn[k][0]))
    edges = {k: p for k, (p, _) in mn.items() if 1 <= _ndet(k) <= 2}
    worst = 0.0
    for k in sorted(ko & kn):
        po, co = mo[k]
        pn, cn = mn[k]
        r = abs(pn - po) / abs(po)
        worst = max(worst, r)
        if r > rel:
            issues.append(("probability", k, po, pn, r))
        if co != cn:
            so, sn = _score(co, edges), _score(cn, edges)
            if so is not None and sn is not None and abs(so - sn) <= 1e-9 * max(abs(so), abs(sn), 1e-300):
                flips.append((k, co, cn))
            elif so is not None and sn is not None and so == sn == 0.0:
                flips.append((k, co, cn))
            else:
                issues.append(("decomposition", k, co, cn, so, sn))
    return issues, flips, worst
