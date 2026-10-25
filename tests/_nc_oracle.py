"""J-b1 P5: an independent oracle of the owner's non-Clifford rule, the STATIC fixed-axis merge (plan r3.1 + r3.1-fix;
the P1 census oracle, committed).

Shares no code with TermResidual or the bare builder: it reads the deferred stream a FAM sampler's TermResidual reads
(gates / noise sites / reads, with probabilities times p_factor; xtim._xtim_testapi._deferred_stream),
decomposes the non-Clifford gates into Pauli rotation terms exp(-i k pi/8 A) (k = +-1), pushes every term axis and
every fault Pauli (components with probability > 0 only, r3.1-fix R3-4) to the END with a stim tableau, and applies:

  merge terms i < j with the same axis A (up to sign) iff (a) no unmerged term between them anticommutes with A and
  (b) no fault Pauli of a noise site between them (in time) anticommutes with A; partner = the LATEST earlier
  same-axis term (greedy; anything blocking it blocks every earlier one); the angles add:
    k odd     -> one surviving term at i with angle k;
    k even    -> both removed; the Clifford G = exp(-i k pi/8 A) (k = 2, 6: S-type; 4: Pauli; 0: identity) stays at
                 slot i, so every item EARLIER than i is end-pushed through it: X -> G X G^dag (convention (A),
                 r3.1-fix R3-1: op insertion at the slot + end-push; the earlier quantities change, nothing later);
  passes in time order (j ascending), repeated to a fixpoint; ACCEPT iff the surviving axes commute pairwise.

Screen (exact, not a guard): if every end axis is Z-type, or the unmerged axes already commute pairwise, no merge
can change the verdict (merges only remove terms; a fold conjugates by exp(-i k pi/8 A) with A commuting with every
other axis, which changes nothing) -> ACCEPT without the fault pass.
"""
import itertools

import stim

GATE = {0: "X", 1: "Y", 2: "Z", 3: "S", 4: "S_DAG", 5: "CZ", 6: "CX", 7: "H"}
T_, CS_, CCZ_, CH_ = 8, 9, 10, 11
K_GATE, K_NOISE, K_MEAS, K_OBS = 0, 1, 2, 4
PCH = "IXYZ"


def _z(n, qs):
    p = stim.PauliString(n)
    for q in qs:
        p[q] = "Z"
    return p


def ops_of(n, stream):
    """Time-ordered ops: ('C', name, targets) | ('T', PauliString, k) | ('F', [PauliString], label)."""
    ops = []
    for d in stream:
        k = d["kind"]
        if k == K_GATE:
            g, t = d["gate"], d["targets"]
            if g in GATE:
                ops.append(("C", GATE[g], t))
            elif g == T_:
                ops.append(("T", _z(n, [t[0]]), +1))
            elif g == CS_:
                ops += [("T", _z(n, [t[0]]), +1), ("T", _z(n, [t[1]]), +1), ("T", _z(n, t[:2]), -1)]
            elif g == CCZ_:
                a, b, c = t[:3]
                ops += [("T", _z(n, [a]), +1), ("T", _z(n, [b]), +1), ("T", _z(n, [c]), +1),
                        ("T", _z(n, [a, b]), -1), ("T", _z(n, [a, c]), -1), ("T", _z(n, [b, c]), -1),
                        ("T", _z(n, [a, b, c]), +1)]
            elif g == CH_:
                c, q = t[0], t[1]           # CH = (I (x) V) CZ (I (x) V^dag), V = S H T H S^dag (time order below)
                ops += [("C", "S_DAG", [q]), ("C", "H", [q]), ("T", _z(n, [q]), -1), ("C", "H", [q]),
                        ("C", "S", [q]), ("C", "CZ", [c, q]),
                        ("C", "S_DAG", [q]), ("C", "H", [q]), ("T", _z(n, [q]), +1), ("C", "H", [q]),
                        ("C", "S", [q])]
            else:
                raise ValueError("unknown gate kind %d" % g)
        elif k == K_NOISE:
            sup = supports(n, d)
            if sup:
                ops.append(("F", sup, d["channel"]))
        elif k in (K_MEAS, K_OBS):
            continue
        else:
            raise ValueError("deferred instruction kind %d outside gate / noise / measure / observable" % k)
    return ops


def _p(n, terms):
    p = stim.PauliString(n)
    for q, c in terms:
        if c:
            p[q] = PCH[c]
    return p


def supports(n, d):
    """The fault Paulis of one noise instruction with probability > 0 (every qubit / pair of it)."""
    ch, qs, pr = d["channel"], d["qubits"], d["probs"]
    out = []
    if ch in (0, 1, 2):                                   # X / Y / Z_ERROR
        if pr and pr[0] > 0:
            out = [_p(n, [(q, ch + 1)]) for q in qs]
    elif ch == 3:                                         # DEPOLARIZE1
        if pr and pr[0] > 0:
            out = [_p(n, [(q, c)]) for q in qs for c in (1, 2, 3)]
    elif ch == 4:                                         # DEPOLARIZE2
        if pr and pr[0] > 0:
            out = [_p(n, [(a, c1), (b, c2)]) for a, b in zip(qs[0::2], qs[1::2])
                   for c1 in range(4) for c2 in range(4) if c1 or c2]
    elif ch == 5:                                         # PAULI_CHANNEL_1(px, py, pz)
        out = [_p(n, [(q, i + 1)]) for q in qs for i in range(3) if pr[i] > 0]
    elif ch == 6:                                         # PAULI_CHANNEL_2: slot t fires digits of t+1 (base 4)
        out = [_p(n, [(a, (t + 1) >> 2), (b, (t + 1) & 3)]) for a, b in zip(qs[0::2], qs[1::2])
               for t in range(15) if pr[t] > 0]
    elif ch == 7:                                         # CORRELATED chain: row t = term t over `qubits`
        m = len(qs)
        out = [_p(n, [(qs[i], d["corr"][t * m + i]) for i in range(m)]) for t in range(len(pr)) if pr[t] > 0]
    else:
        raise ValueError("unknown noise channel %d" % ch)
    return out


def end_push(n, ops, faults=True):
    """Items in time order: ['T', axis(+sign), k] / ['F', [Paulis]], every Pauli pushed through the LATER Cliffords
    (faults=False: the fault items keep no Paulis — the screen needs the term axes only)."""
    T = stim.Tableau(n)
    items = []
    for op in reversed(ops):
        if op[0] == "C":
            T.prepend(stim.Tableau.from_named_gate(op[1]), op[2])
        elif op[0] == "T":
            B = T(op[1])
            k = op[2]
            if B.sign == -1:
                B, k = -B, -k
            items.append(["T", B, k])
        else:
            items.append(["F", [T(P) for P in op[1]] if faults else []])
    items.reverse()
    return items


def _conj(X, A, k):
    """G X G^dag for G = exp(-i k pi/8 A), k even: X if [X, A] = 0, else exp(-i k pi/4 A) X."""
    if X.commutes(A):
        return X
    k %= 8
    if k == 0:
        return X
    if k == 4:
        return -X
    c = -1j if k == 2 else 1j                            # exp(-i pi/2 A) = -iA, exp(-i 3pi/2 A) = iA
    return c * (A * X)


def _count_anti(ps):
    """Number of anticommuting pairs (symplectic form over GF(2), numpy)."""
    import numpy as np
    if len(ps) < 2:
        return 0
    X = np.array([p.to_numpy()[0] for p in ps], dtype=np.int64)
    Z = np.array([p.to_numpy()[1] for p in ps], dtype=np.int64)
    S = (X @ Z.T + Z @ X.T) % 2
    return int(np.triu(S, 1).sum())


def _anti(a, b):
    return not a.commutes(b)


def verdict(n, stream, return_detail=False):
    ops = ops_of(n, stream)
    terms0 = [op for op in ops if op[0] == "T"]
    det = {"terms": len(terms0), "x_axes": 0, "anti_pairs": 0, "merges": 0, "folds": 0, "passes": 0}
    items = end_push(n, ops, faults=False) if terms0 else []
    T = [x for x in items if x[0] == "T"]
    det["x_axes"] = sum(1 for x in T if any(x[1].to_numpy()[0]))
    det["axes"] = [x[1] for x in T]                       # the surviving signed axes (stim.PauliString), time order
    if det["x_axes"] == 0:
        return ("ACCEPT", det) if return_detail else "ACCEPT"
    det["anti_pairs"] = _count_anti([x[1] for x in T])
    if not det["anti_pairs"]:
        return ("ACCEPT", det) if return_detail else "ACCEPT"
    items = end_push(n, ops, faults=True)
    while True:
        det["passes"] += 1
        merged = False
        j = 0
        while j < len(items):
            it = items[j]
            if it[0] != "T":
                j += 1
                continue
            A = it[1]
            partner = None
            for i in range(j - 1, -1, -1):
                x = items[i]
                if x[0] != "T":
                    continue
                if x[1] == A:
                    partner = i
                    break
                if _anti(x[1], A):
                    break
            if partner is not None and any(_anti(P, A) for x in items[partner + 1:j] if x[0] == "F" for P in x[1]):
                partner = None
            if partner is None:
                j += 1
                continue
            i = partner
            k = (items[i][2] + items[j][2]) % 8
            det["merges"] += 1
            merged = True
            if k % 2:
                items[i][2] = k
                del items[j]
                continue                                  # the item now at j is the next one
            del items[j]
            del items[i]
            if k:
                det["folds"] += 1
                for x in items[:i]:                       # convention (A): the earlier items through G
                    if x[0] == "T":
                        B = _conj(x[1], A, k)
                        if B.sign == -1:
                            B, x[2] = -B, -x[2]
                        x[1] = B
                    else:
                        x[1] = [_conj(P, A, k) for P in x[1]]
            j = i                                         # resume at the slot (items before i did not move in time)
        if not merged:
            break
    T = [x for x in items if x[0] == "T"]
    bad = [(a, b) for a, b in itertools.combinations(range(len(T)), 2) if _anti(T[a][1], T[b][1])]
    det["surviving"] = len(T)
    det["axes"] = [x[1] for x in T]
    if bad:
        det["pair"] = (str(T[bad[0][0]][1]), str(T[bad[0][1]][1]))
    v = "REFUSE" if bad else "ACCEPT"
    return (v, det) if return_detail else v


def verdict_of_text(text, p_factor=1.0, return_detail=False):
    """The static-rule verdict of a circuit text (its family-normalized deferred stream)."""
    from xtim import _xtim_testapi as TA
    n, st = TA._deferred_stream(text, p_factor)
    return verdict(n, st, return_detail)
