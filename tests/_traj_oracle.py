"""J-b1 P9b test oracle: a vectorized statevector TRAJECTORY sampler for small stim circuits (any T-depth; n <= ~10).

Independent of every xtim engine path: all shots evolve together as a (shots, 2^n) array; Pauli noise is sampled
per shot and applied by masks; measurements sample per shot and project. Supported: R RX H S S_DAG T T_DAG X Y Z
CX CZ CCZ, M MX (with readout inversion `!`), X_ERROR Y_ERROR Z_ERROR DEPOLARIZE1(p),
DETECTOR / OBSERVABLE_INCLUDE(k) rec[-i]; OUTPUT_QUBITS / INPUT_QUBITS lines are ignored. Qubit q is bit q of the
basis index (little endian)."""
import re

import numpy as np

_W = np.exp(1j * np.pi / 4)


def _parse(text):
    ops = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith(("OUTPUT_QUBITS", "INPUT_QUBITS")):
            continue
        m = re.match(r"([A-Z_0-9]+)(?:\(([^)]*)\))?\s*(.*)", line)
        name, arg, rest = m.group(1), m.group(2), m.group(3).split()
        ops.append((name, float(arg) if arg else None, rest))
    return ops


def sample(text, shots, seed=1, relative=True):
    """(detectors uint8[shots, D], observables uint8[shots, O]). relative=True (xtim's convention, A1 §2): every
    column XORed with its value on the noiseless REFERENCE run (each measurement takes outcome 0 whenever its
    probability is > 1e-12, as the engines' reference read does)."""
    D, O = _run(text, shots, seed, False)
    if relative:
        rd, ro = _run(text, 1, seed, True)
        D, O = D ^ rd, O ^ ro
    return D, O


def _run(text, shots, seed, ref):
    ops = _parse(text)
    n = 0
    for name, _, rest in ops:
        for t in rest:
            if not t.startswith("rec["):
                n = max(n, int(t.lstrip("!")) + 1)
    rng = np.random.default_rng(seed)
    dim = 1 << n
    psi = np.zeros((shots, dim), complex)
    psi[:, 0] = 1.0
    idx = np.arange(dim)
    recs, dets, obs = [], [], {}

    def bit(q):
        return (idx >> q) & 1

    def perm_x(q, mask=None):
        nonlocal psi
        new = psi[:, idx ^ (1 << q)]
        psi = new if mask is None else np.where(mask[:, None], new, psi)

    def phase(q, ph, mask=None):
        nonlocal psi
        f = np.where(bit(q) == 1, ph, 1.0)
        psi = psi * f[None, :] if mask is None else np.where(mask[:, None], psi * f[None, :], psi)

    def h(q):
        nonlocal psi
        a = psi.copy()
        b = psi[:, idx ^ (1 << q)]
        s = np.where(bit(q) == 1, -1.0, 1.0)
        psi = (b + s[None, :] * a) / np.sqrt(2)

    def measure_z(q):
        nonlocal psi
        p1 = (np.abs(psi[:, bit(q) == 1]) ** 2).sum(1)
        out = ((p1 > 1 - 1e-12) if ref else (rng.random(shots) < p1)).astype(np.uint8)
        keep = (bit(q)[None, :] == out[:, None])
        psi = np.where(keep, psi, 0)
        psi = psi / np.linalg.norm(psi, axis=1)[:, None]
        return out

    def reset(q):
        out = measure_z(q)
        perm_x(q, out.astype(bool))

    for name, arg, rest in ops:
        qs = [t for t in rest if not t.startswith("rec[")]
        if name in ("R", "RZ"):
            for t in qs: reset(int(t))
        elif name == "RX":
            for t in qs: reset(int(t)); h(int(t))
        elif name == "H":
            for t in qs: h(int(t))
        elif name == "S":
            for t in qs: phase(int(t), 1j)
        elif name == "S_DAG":
            for t in qs: phase(int(t), -1j)
        elif name == "T":
            for t in qs: phase(int(t), _W)
        elif name == "T_DAG":
            for t in qs: phase(int(t), np.conj(_W))
        elif name == "X":
            for t in qs: perm_x(int(t))
        elif name == "Z":
            for t in qs: phase(int(t), -1.0)
        elif name == "Y":
            for t in qs: perm_x(int(t)); phase(int(t), -1.0); psi *= 1j
        elif name in ("CX", "CNOT", "CZ"):
            for a, b in zip(qs[0::2], qs[1::2]):
                a, b = int(a), int(b)
                if name == "CZ":
                    f = np.where((bit(a) & bit(b)) == 1, -1.0, 1.0)
                    psi = psi * f[None, :]
                else:
                    src = np.where(bit(a) == 1, idx ^ (1 << b), idx)
                    psi = psi[:, src]
        elif name == "CCZ":
            for a, b, c in zip(qs[0::3], qs[1::3], qs[2::3]):
                f = np.where((bit(int(a)) & bit(int(b)) & bit(int(c))) == 1, -1.0, 1.0)
                psi = psi * f[None, :]
        elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"):
            for t in ([] if ref else qs):
                q = int(t)
                if name == "DEPOLARIZE1":
                    r = rng.random(shots)
                    k = np.where(r < arg, rng.integers(1, 4, shots), 0)
                else:
                    k = np.where(rng.random(shots) < arg, {"X_ERROR": 1, "Y_ERROR": 3, "Z_ERROR": 2}[name], 0)
                xm, zm = (k & 1).astype(bool), (k >= 2)
                ym = k == 3
                zm = (k == 2) | ym
                perm_x(q, xm | ym)
                phase(q, -1.0, zm)
        elif name in ("M", "MZ", "MX"):
            for t in qs:
                inv = t.startswith("!")
                q = int(t.lstrip("!"))
                if name == "MX": h(q)
                out = measure_z(q)
                if name == "MX": h(q)
                recs.append(out ^ (1 if inv else 0))
        elif name == "DETECTOR":
            ks = [int(re.match(r"rec\[-(\d+)\]", t).group(1)) for t in rest]
            v = np.zeros(shots, np.uint8)
            for k in ks: v ^= recs[-k]
            dets.append(v)
        elif name == "OBSERVABLE_INCLUDE":
            ks = [int(re.match(r"rec\[-(\d+)\]", t).group(1)) for t in rest]
            v = obs.setdefault(int(arg), np.zeros(shots, np.uint8))
            for k in ks: v ^= recs[-k]
        else:
            raise ValueError("trajectory oracle: unsupported instruction " + name)
    D = np.stack(dets, 1) if dets else np.zeros((shots, 0), np.uint8)
    O = np.stack([obs[k] for k in sorted(obs)], 1) if obs else np.zeros((shots, 0), np.uint8)
    return D, O
