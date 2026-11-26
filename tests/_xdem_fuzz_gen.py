"""xdem Task 5 fuzz generator (TEST ONLY; SPEC §6 T3): random small circuits for the xdem law fuzz.

Started from tests/_dc_fuzz_gen.py (families A / B / C / N, the instruction mix, the probabilities, the cost cap,
`nullspace` and the deterministic-parity `finish`, the seeds 5101 / 5201 of the test); extended to:
  * up to `max_qubits` (8) qubits;
  * non-Clifford gates drawn from T, T_DAG, CS, CS_DAG, CCZ, at most `max_nonclifford` (6) per circuit (a draw over
    the cap becomes a Clifford; family B's compute / uncompute layer counts both halves);
  * `observables` = (1, 3): every circuit has 1 to 3 observables, each a nonzero deterministic parity;
  * deferred feedback `CX rec[-k] q` (k <= 3) right before a non-Clifford gate in a quarter of the circuits (a
    measurement is inserted first when no record precedes the gate);
  * detectors over 2-3 rounds: 1-2 mid-circuit measurement layers are cut into the body, and each round's
    detectors are deterministic parities of the records measured so far, declared right after the round;
  * `generate` returns only circuits whose noiseless detectors and observables are all deterministic per
    `_xtim.detector_determinism` (the dc generator's deliberate non-deterministic "_nd" detector is dropped, and a
    circuit the exact-engine probe refuses -- e.g. R_X / U3, which xtim does not parse -- is redrawn).
Family N (a non-diagonal ingredient) and heralded noise stay in the mix: the xdem writer refuses them by name, and
the fuzz counts those refusals.

generate(seed, count, ...) is a pure function of its arguments (one random.Random(seed) stream)."""
import random

import numpy as np

from xtim import _xtim

from _exact_engine import compile_exact_branches

P1 = ["H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "H_YZ"]
P2G = ["CX", "CZ", "CY", "SWAP", "ISWAP", "SQRT_XX", "XCZ", "CXSWAP"]
INV = {"S": "S_DAG", "S_DAG": "S", "H": "H", "X": "X", "Y": "Y", "Z": "Z", "SQRT_X": "SQRT_X_DAG", "H_YZ": "H_YZ",
       "CX": "CX", "CZ": "CZ", "CY": "CY", "SWAP": "SWAP", "T": "T_DAG", "T_DAG": "T", "CS": "CS_DAG",
       "CS_DAG": "CS", "CCZ": "CCZ", "ISWAP": "ISWAP_DAG", "SQRT_XX": "SQRT_XX_DAG", "XCZ": "XCZ",
       "CXSWAP": "SWAPCX"}
HERALD_RATE = 0.03          # family A: fraction of noise ops that are heralded
TERMINAL_MP_RATE = 0.3      # finish(): fraction of the plain terminal `M q` reads written `M(p) q`


def pr(rng):
    return round(rng.choice([0.03, 0.05, 0.08, 0.1, 0.15, 0.2]), 3)


NC_ARITY = {"T": 1, "T_DAG": 1, "CS": 2, "CS_DAG": 2, "CCZ": 3}
NC = list(NC_ARITY)
FEEDBACK_RATE = 0.25        # fraction of circuits with `CX rec[-k] q` right before a non-Clifford gate
_REC1 = {"M", "MX", "MY", "MR", "MRX", "MRY", "MPP", "MPAD", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"}
_REC2 = {"MXX", "MYY", "MZZ"}


def records_of(line):
    """the number of measurement records an instruction line writes"""
    tok = line.split()
    if not tok or tok[0].startswith("#"):
        return 0
    name = tok[0].split("(")[0]
    if name in _REC1:
        return len(tok) - 1
    if name in _REC2:
        return (len(tok) - 1) // 2
    return 0


class G:
    """the dc generator's circuit builder (copied from tests/_dc_fuzz_gen.py at cdb14de) with a non-Clifford cap"""

    def __init__(self, rng, n, heralds=False, max_nc=6):
        self.rng, self.n = rng, n
        self.lines = []
        self.nrec = 0
        self.cost = 1          # product of (1 + atoms) for the xtim enumeration
        self.heralds = heralds
        self.nnc, self.max_nc = 0, max_nc

    def q(self, k=1):
        return self.rng.sample(range(self.n), k)

    def noise(self):
        r = self.rng
        if self.heralds and r.random() < HERALD_RATE:
            qs = self.q(r.choice([1, 2]))
            if r.random() < 0.5:
                self.lines.append(f"HERALDED_ERASE({pr(r)}) " + " ".join(map(str, qs)))
            else:
                ps = [r.choice([0.0, 0.01, 0.03, 0.05]) for _ in range(4)]
                self.lines.append(f"HERALDED_PAULI_CHANNEL_1({ps[0]}, {ps[1]}, {ps[2]}, {ps[3]}) "
                                  + " ".join(map(str, qs)))
            self.nrec += len(qs)
            return
        kind = r.random()
        if kind < 0.3:
            c = r.choice(["X_ERROR", "Y_ERROR", "Z_ERROR"])
            qs = self.q(r.choice([1, 1, 2]))
            self.lines.append(f"{c}({pr(r)}) " + " ".join(map(str, qs)))
            self.cost *= 2 ** len(qs)
        elif kind < 0.6:
            qs = self.q(r.choice([1, 1, 2]))
            self.lines.append(f"DEPOLARIZE1({pr(r)}) " + " ".join(map(str, qs)))
            self.cost *= 4 ** len(qs)
        elif kind < 0.7 and self.n >= 2:
            qs = self.q(2)
            self.lines.append(f"DEPOLARIZE2({pr(r)}) {qs[0]} {qs[1]}")
            self.cost *= 16
        elif kind < 0.85:
            # PAULI_CHANNEL_1: single-term, or multi-term (some solvable, some not)
            ps = [0.0, 0.0, 0.0]
            for i in r.sample(range(3), r.choice([1, 2, 3])):
                ps[i] = r.choice([0.02, 0.05, 0.1, 0.2])
            self.lines.append(f"PAULI_CHANNEL_1({ps[0]}, {ps[1]}, {ps[2]}) {self.q()[0]}")
            self.cost *= 4
        else:
            k = r.choice([1, 2])
            n = 0
            for j in range(k):
                qs = self.q(r.choice([1, 2]))
                w = " ".join(f"{r.choice('XYZ')}{q}" for q in qs)
                self.lines.append(("E" if j == 0 else "ELSE_CORRELATED_ERROR") + f"({pr(r)}) {w}")
                n += 1
            self.cost *= (1 + n)

    def meas(self, inv_p=0.4, p_p=0.3):
        r = self.rng
        kind = r.random()
        bang = lambda: "!" if r.random() < inv_p else ""   # noqa: E731
        arg = f"({pr(r)})" if r.random() < p_p else ""
        if kind < 0.45:
            g = r.choice(["M", "MX", "MY", "MR", "MRX", "MRY"])
            qs = self.q(r.choice([1, 1, 2]))
            self.lines.append(f"{g}{arg} " + " ".join(bang() + str(q) for q in qs))
            self.nrec += len(qs)
            if arg:
                self.cost *= 2 ** len(qs)
        elif kind < 0.75 or self.n < 2:
            k = r.choice([1, 2, 2, 3]) if self.n >= 3 else min(self.n, r.choice([1, 2]))
            qs = self.q(k)
            prod = "*".join(bang() + r.choice("XYZ") + str(q) for q in qs)
            self.lines.append(f"MPP{arg} {prod}")
            self.nrec += 1
            if arg:
                self.cost *= 2
        else:
            g = r.choice(["MXX", "MYY", "MZZ"])
            qs = self.q(2)
            self.lines.append(f"{g}{arg} {bang()}{qs[0]} {bang()}{qs[1]}")
            self.nrec += 1
            if arg:
                self.cost *= 2

    def feedback(self):
        if not self.nrec:
            return
        r = self.rng
        k = r.randint(1, min(self.nrec, 3))
        g = r.choice(["CX", "CZ", "CY"])
        self.lines.append(f"{g} rec[-{k}] {self.q()[0]}")

    def cliff(self):
        r = self.rng
        if r.random() < 0.6 or self.n < 2:
            g, qs = r.choice(P1), self.q()
        else:
            g, qs = r.choice(P2G), self.q(2)
        self.lines.append(f"{g} " + " ".join(map(str, qs)))
        return g, qs

    def _nondiag_dc(self):
        """family N: one non-diagonal ingredient (the dc generator's)."""
        r = self.rng
        x = r.random()
        if x < 0.45 and self.n >= 2:
            qs = self.q(2)
            self.lines.append(f"CH {qs[0]} {qs[1]}")
        elif x < 0.8:
            q = self.q()[0]
            self.lines += [f"T {q}", f"{r.choice(['H', 'SQRT_X', 'H_YZ'])} {q}", f"T {q}"]
        else:
            q = self.q()[0]
            self.lines.append(r.choice([f"R_X(0.3) {q}", f"R_Z(0.3) {q}", f"U3(0.1, 0.2, 0.3) {q}"]))

    def nc(self, gate=None):
        r = self.rng
        g = gate or r.choice(NC)
        ar = NC_ARITY[g]
        if ar > self.n:
            g, ar = "T", 1
        qs = self.q(ar)
        k = 1
        if ar == 1 and r.random() < 0.05:
            qs, k = qs + qs, 2
        if self.nnc + k > self.max_nc:
            return self.cliff()
        self.nnc += k
        self.lines.append(f"{g} " + " ".join(map(str, qs)))
        return g, qs

    def nondiag(self):
        if self.nnc + 2 > self.max_nc:                    # the T sandwich adds two T's
            q = self.q()[0]
            self.lines.append(f"R_X(0.3) {q}")
            return
        n0 = len(self.lines)
        self._nondiag_dc()
        self.nnc += sum(1 for l in self.lines[n0:] if l.startswith("T "))


def fam_A(rng, n, max_nc, heralds=True):
    g = G(rng, n, heralds=heralds, max_nc=max_nc)
    g.lines.append("R " + " ".join(map(str, range(n))))
    for _ in range(rng.randint(6, 22)):
        x = rng.random()
        if x < 0.35:
            g.cliff()
        elif x < 0.55:
            g.nc()
        elif x < 0.72 and g.cost < 3000:
            g.noise()
        elif x < 0.87:
            g.meas()
        else:
            g.feedback()
    return g


def fam_B(rng, n, max_nc):
    g = G(rng, n, max_nc=max_nc)
    g.lines.append("R " + " ".join(map(str, range(n))))
    C = []
    for _ in range(rng.randint(2, 8)):
        C.append(g.cliff())
    if rng.random() < 0.5 and g.cost < 3000:
        g.noise()
    layer = []
    g.max_nc = max_nc // 2                                # the uncompute layer doubles the count
    for _ in range(rng.randint(1, 4)):
        layer.append(g.nc())
        if rng.random() < 0.3 and g.cost < 3000:
            g.noise()
    g.max_nc = max_nc
    if rng.random() < 0.4:
        g.meas(inv_p=0.5, p_p=0.4)
    for _ in range(rng.randint(1, 3)):
        if g.cost < 3000:
            g.noise()
    if rng.random() < 0.3:
        g.feedback()
    for gt, qs in reversed(layer):
        g.lines.append(f"{INV[gt]} " + " ".join(map(str, qs)))
        g.nnc += len(qs) // NC_ARITY[gt] if gt in NC_ARITY else 0
        if rng.random() < 0.25 and g.cost < 3000:
            g.noise()
    for gt, qs in reversed(C):
        g.lines.append(f"{INV[gt]} " + " ".join(map(str, qs)))
    if rng.random() < 0.3 and g.cost < 3000:
        g.noise()
    return g


def fam_C(rng, n, max_nc):
    g = G(rng, n, max_nc=max_nc)
    g.lines.append("R " + " ".join(map(str, range(n))))
    for _ in range(rng.randint(6, 18)):
        x = rng.random()
        if x < 0.3:
            gt = rng.choice(["X", "CX", "SWAP", "S", "CZ"]) if n >= 2 else "X"
            qs = g.q(2 if gt in ("CX", "SWAP", "CZ") else 1)
            g.lines.append(f"{gt} " + " ".join(map(str, qs)))
        elif x < 0.55:
            g.nc()
        elif x < 0.7 and g.cost < 3000:
            g.noise()
        elif x < 0.85:
            g.meas()
        else:
            g.feedback()
    return g


def fam_N(rng, n, max_nc):
    g = fam_A(rng, n, max_nc, heralds=False)
    at = rng.randint(1, len(g.lines))
    while at < len(g.lines) and g.lines[at].startswith("ELSE_CORRELATED_ERROR"):   # keep E / ELSE chains whole
        at += 1
    tail, g.lines = g.lines[at:], g.lines[:at]
    g.nondiag()
    g.lines += tail
    return g


def nullspace(M, R):
    M = M.copy() % 2
    piv = []
    row = 0
    M = M.astype(np.uint8)
    for col in range(R):
        if row >= M.shape[0]:
            break
        nz = np.flatnonzero(M[row:, col])
        if not nz.size:
            continue
        p = row + nz[0]
        M[[row, p]] = M[[p, row]]
        for i in range(M.shape[0]):
            if i != row and M[i, col]:
                M[i] ^= M[row]
        piv.append(col)
        row += 1
    free = [c for c in range(R) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(R, np.uint8)
        v[f] = 1
        for i, pc in enumerate(piv):
            if M[i, f]:
                v[pc] = 1
        basis.append(v)
    return basis


def _cut_ok(lines, at):
    return at >= len(lines) or not lines[at].startswith("ELSE_CORRELATED_ERROR")


def add_feedback(g, rng):
    """`CX rec[-k] q` (k <= 3) right before one non-Clifford gate; a measurement first when no record precedes it"""
    ncs = [i for i, l in enumerate(g.lines) if l.split()[0] in NC_ARITY]
    if not ncs:
        return False
    i = rng.choice(ncs)
    before = sum(records_of(l) for l in g.lines[:i])
    ins = []
    if before == 0:
        ins.append(f"{rng.choice(['M', 'MR', 'MX'])} {g.q()[0]}")
        before = 1
    ins.append(f"CX rec[-{rng.randint(1, min(3, before))}] {g.q()[0]}")
    g.lines[i:i] = ins
    return True


def add_rounds(g, rng, nrounds):
    """nrounds - 1 mid-circuit measurement layers, each followed by a round marker (replaced by detectors)"""
    cuts = sorted(rng.sample(range(1, len(g.lines) + 1), min(nrounds - 1, len(g.lines))), reverse=True)
    for k, at in enumerate(cuts):
        while not _cut_ok(g.lines, at):
            at += 1
        saved, g.lines = g.lines, []
        for _ in range(rng.randint(1, 3)):
            g.meas(inv_p=0.2, p_p=0.3)
        layer = g.lines
        g.lines = saved
        g.lines[at:at] = layer + ["#ROUND"]


def _combo(basis, rng, R):
    for _ in range(20):
        c = np.zeros(R, np.uint8)
        for b in basis:
            if rng.random() < 0.5:
                c ^= b
        if c.any():
            return c
    return None


def finish(g, n, rng, observables):
    """terminal reads, then per round deterministic detector parities of the records so far and 1..3 observables, via
    xtim's exact noiseless law (raises if the exact engine refuses the probe; None if no observable exists)."""
    r = rng
    for q in range(n):
        if r.random() < 0.5:
            g.lines.append(f"{r.choice(['M', 'MX', 'MY'])} {'!' if r.random() < 0.3 else ''}{q}")
        elif r.random() < TERMINAL_MP_RATE:
            g.lines.append(f"M({pr(r)}) {q}")
            g.cost *= 2
        else:
            g.lines.append(f"M {q}")
    g.lines.append("#ROUND")
    body = [l for l in g.lines if l != "#ROUND"]
    R = sum(records_of(l) for l in body)
    probe = "\n".join(("MPAD " + " ".join("0" for _ in l.split()[1:])) if l.startswith("HERALDED") else l
                      for l in body) + "\n" + "\n".join(f"DETECTOR rec[-{R - j}]" for j in range(R))
    sim = compile_exact_branches(probe)
    L0 = sim.exact_branches([], expectations=[])
    rows = np.asarray(L0.rows_detectors, np.uint8)
    U = np.asarray(L0.uniform_detectors, np.uint8).reshape(-1, R)
    M = np.concatenate([U, (rows[1:] ^ rows[0])], 0) if rows.shape[0] > 1 else U
    basis = nullspace(M, R)
    if not basis:
        return None
    obs = []
    for _ in range(r.randint(*observables)):
        c = _combo(basis, r, R)
        if c is not None:
            obs.append(c)
    if not obs:
        return None
    out, seen = [], 0
    for l in g.lines:
        if l != "#ROUND":
            out.append(l)
            seen += records_of(l)
            continue
        pb = nullspace(M[:, :seen], seen) if seen else []
        made = set()
        for _ in range(r.randint(1, 3) if pb else 0):
            c = _combo(pb, r, seen)
            if c is not None and tuple(c) not in made:     # distinct detectors within a round
                made.add(tuple(c))
                out.append("DETECTOR " + " ".join(f"rec[-{seen - j}]" for j in np.flatnonzero(c)))
    for i, c in enumerate(obs):
        out.append(f"OBSERVABLE_INCLUDE({i}) " + " ".join(f"rec[-{R - j}]" for j in np.flatnonzero(c)))
    return "\n".join(out) + "\n"


def all_deterministic(text):
    d = _xtim.detector_determinism(text)
    return (d["ok"] and not list(d["gauge_detectors"]) and not list(d["gauge_observables"])
            and len(list(d["deterministic_detectors"])) == d["num_detectors"]
            and len(list(d["deterministic_observables"])) == d["num_observables"])


def generate(seed, count, max_qubits=8, max_nonclifford=6, observables=(1, 3), max_cost=4000, stats=None):
    """`count` circuit texts from one random.Random(seed) stream. Draws over the enumeration cost cap, refused by the
    exact-engine probe or by `detector_determinism`, without an observable, or not all-deterministic are redrawn
    (counted in `stats` when a dict is given)."""
    rng = random.Random(seed)
    st = stats if stats is not None else {}

    def bump(k):
        st[k] = st.get(k, 0) + 1
    out = []
    while len(out) < count:
        bump("drawn")
        assert st["drawn"] <= 100 * count + 100, ("generator: too many redraws", st)
        fam = rng.choice("AABBCN")
        n = rng.randint(2, max_qubits)
        g = {"A": fam_A, "B": fam_B, "C": fam_C, "N": fam_N}[fam](rng, n, max_nonclifford)
        if rng.random() < FEEDBACK_RATE and add_feedback(g, rng):
            bump("feedback before a non-Clifford gate")
        add_rounds(g, rng, rng.choice([2, 3]))
        if g.cost > max_cost:
            bump("redrawn: over the cost cap")
            continue
        try:
            txt = finish(g, n, rng, observables)
        except Exception:   # noqa: BLE001  (the exact-engine probe refused the body)
            bump("redrawn: the exact-engine probe refused")
            continue
        if txt is None or g.cost > max_cost:
            bump("redrawn: no observable" if txt is None else "redrawn: over the cost cap")
            continue
        try:
            det = all_deterministic(txt)
        except Exception:   # noqa: BLE001
            det = False
        if not det:
            bump("redrawn: not all deterministic")
            continue
        bump("family " + fam)
        out.append(txt)
    return out


# ── Hadamard-test CH circuits (CH in the xdem export) ────────────────────────────────────────────────────────────────
H_PREP = ["H", "T", "H", "S", "H"]          # (H S H) T H |0> = |H>, the +1 eigenstate of H (examples/ch_cultivation)
H_UNPREP = ["H", "S_DAG", "H", "T_DAG", "H"]


def _ch_noise(r, qs):
    q = r.choice(qs)
    c = r.random()
    if c < 0.4:
        return f"{r.choice(['X_ERROR', 'Y_ERROR', 'Z_ERROR'])}({pr(r)}) {q}"
    if c < 0.75:
        return f"DEPOLARIZE1({pr(r)}) {q}"
    if c < 0.9 and len(qs) >= 2:
        a, b = r.sample(qs, 2)
        return f"DEPOLARIZE2({pr(r)}) {a} {b}"
    ps = [r.choice([0.0, 0.02, 0.05, 0.1]) for _ in range(3)]
    if not any(ps):
        ps[0] = 0.05
    return f"PAULI_CHANNEL_1({ps[0]}, {ps[1]}, {ps[2]}) {q}"


def generate_ch(seed, count, max_noise=4, unaligned_rate=0.1, flags=False):
    """`count` Hadamard-test CH circuit texts from one random.Random(seed) stream: 1-3 data qubits prepared in |H>
    (a Y makes it the -1 eigenstate; with probability `unaligned_rate` the first is left as the unaligned T-state
    H T |0>, which the writer must refuse), 1-4 rounds each measuring H on 1-2 data qubits by a fresh |+> ancilla (CH
    onto each, MX, DETECTOR; the ancilla `MX(p)` sometimes) or, sometimes, two ancillas onto two data qubits in ONE
    multi-pair line `CH a1 q1 a2 q2`, at most `max_noise` single noise ops placed among the rounds, then each data qubit
    either uncomputed (the inverse preparation, M, DETECTOR, a late T group) or left as output; one observable over a
    random nonempty set of the final data reads when there are some. The qubit labels are a random permutation, so a
    control may have a lower or a higher index than its target. Every circuit is noiselessly deterministic when aligned
    (CH|+>|H> = |+>|H>). flags=True: (text, unaligned) pairs."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        r = rng
        nd = r.randint(1, 3)
        data = list(range(nd))
        L = []                                    # (format string, symbolic qubits) relabelled at the end
        unaligned = r.random() < unaligned_rate
        for i, q in enumerate(data):
            L += [(g + " {}", [q]) for g in (H_PREP if not (unaligned and i == 0) else ["H", "T"])]
            if r.random() < 0.3:
                L.append(("Y {}", [q]))
        anc = nd
        nnoise = 0

        def noise(qs):
            head, tail = _ch_noise(r, qs).split(") ")       # "CHANNEL(args) q [q]"
            tq = [int(x) for x in tail.split()]
            return (head + ")" + " {}" * len(tq), tq)
        for _ in range(r.randint(1, 4)):
            if nd >= 2 and r.random() < 0.3:      # two Hadamard tests in one multi-pair CH line
                a1, a2 = anc, anc + 1
                anc += 2
                q1, q2 = r.sample(data, 2)
                L.append(("RX {} {}", [a1, a2]))
                if nnoise < max_noise and r.random() < 0.6:
                    L.append(noise(data + [a1, a2]))
                    nnoise += 1
                L.append(("CH {} {} {} {}", [a1, q1, a2, q2]))
                L.append(("MX {} {}", [a1, a2]))
                L.append(("DETECTOR rec[-2]", []))
                L.append(("DETECTOR rec[-1]", []))
                continue
            a = anc
            anc += 1
            L.append(("RX {}" if r.random() < 0.5 else "H {}", [a]))
            if nnoise < max_noise and r.random() < 0.6:
                L.append(noise(data + [a]))
                nnoise += 1
            for q in r.sample(data, r.randint(1, min(2, nd))):
                L.append(("CH {} {}", [a, q]))
                if nnoise < max_noise and r.random() < 0.3:
                    L.append(noise(data + [a]))
                    nnoise += 1
            L.append((f"MX({r.choice([0.01, 0.03])}) {{}}" if r.random() < 0.2 else "MX {}", [a]))
            L.append(("DETECTOR rec[-1]", []))
        reads = []
        for q in data:
            if r.random() < 0.6:
                if nnoise < max_noise and r.random() < 0.3:
                    L.append(noise([q]))
                    nnoise += 1
                L += [(g + " {}", [q]) for g in (H_UNPREP if not (unaligned and q == 0) else ["T_DAG", "H"])]
                L.append(("M {}", [q]))
                reads.append(q)
        if reads and r.random() < 0.5:
            k = len(reads)
            pick = r.sample(range(k), r.randint(1, k))
            L.append(("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[-{k - j}]" for j in sorted(pick)), []))
            reads = []
        for j in range(len(reads)):
            L.append((f"DETECTOR rec[-{len(reads) - j}]", []))
        if nnoise == 0:
            continue
        lab = list(range(anc))
        r.shuffle(lab)
        text = "\n".join(fmt.format(*(lab[q] for q in qs)) for fmt, qs in L) + "\n"
        out.append((text, unaligned) if flags else text)
    return out
