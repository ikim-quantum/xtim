"""Noise rescaling of circuit text: ``xtim.scale_noise`` (and the noise-line shapes it recognizes).

(exact-out T5: moved here from the removed ``xtim.collect`` module; ``collect`` / ``Task`` are gone.)
"""
from __future__ import annotations

import re


# Baked-in noise strength of the committed benchmarks (scripts/msp_e2e.py's P0).
DEFAULT_BAKED_P = 1e-3

# Noise-line shapes of the benchmark dialect (ported from scripts/msp_e2e.py /
# scripts/msp_accounting.py — keep in sync). _NOISE_LINE recognizes a noise
# instruction; _NOISE_ARGS additionally captures its argument list for rewriting.
# A measurement's (...) argument is its flip probability (Stim M(p)): a noise argument too, rescaled like the rest
# (FCS: the 4.1.0 rewrite left every M(p) / MR(p) / MX(p) / MPP(p) / MXX(p) ... at the baked-in strength).
_MEAS = r"MRX|MRY|MRZ|MPP|MXX|MYY|MZZ|MR|MX|MY|MZ|M"
# An instruction may carry a Stim tag between its name and its arguments (M[tag](p), X_ERROR[tag](p)): _TAG matches it
# (kept verbatim on rewrite).
_TAG = r"(\[[^\]]*\])?"
_NOISE_LINE = re.compile(
    r"^\s*(DEPOLARIZE1|DEPOLARIZE2|X_ERROR|Y_ERROR|Z_ERROR|PAULI_CHANNEL"
    r"|E" + _TAG + r"\(|CORRELATED_ERROR|ELSE_CORRELATED_ERROR|(?:" + _MEAS + r")" + _TAG + r"\()")
_NOISE_ARGS = re.compile(
    r"^(\s*)(DEPOLARIZE1|DEPOLARIZE2|X_ERROR|Y_ERROR|Z_ERROR|PAULI_CHANNEL\w*|" + _MEAS + r")" + _TAG +
    r"\(([^)]+)\)")
# One term of an E/ELSE_CORRELATED_ERROR chain. Its argument is a SEQUENTIAL probability
# (fires iff no earlier term fired), so the chain is rescaled through its categorical law
# (see scale_noise), never argument by argument.
_CHAIN_ARGS = re.compile(
    r"^(\s*)(E|CORRELATED_ERROR|ELSE_CORRELATED_ERROR)" + _TAG + r"\(([^)]+)\)")


def scale_noise(text: str, p: float, p0: float = DEFAULT_BAKED_P) -> str:
    """Scale every noise-instruction argument of `text` by p/p0 — the measurement flip probabilities
    (M(p), MR(p), MX(p), MY(p), MRX(p), MPP(p), MXX(p), ...) included.

    The ported scripts/msp_e2e.py helper: the rewrite must be surjective onto the
    noise lines (every recognized noise line gets rescaled) and bijective on line
    count — any mismatch raises ValueError, never silently mis-scales.

    This is a pure TEXT rewrite: it does not validate that the scaled probabilities
    stay in [0, 1] (e.g. p=1.0 on a p0=1e-3 arg yields a >1 probability). Those bounds
    are checked when the resulting circuit is parsed, not here.

    An ``E``/``CORRELATED_ERROR`` + ``ELSE_CORRELATED_ERROR`` chain is ONE categorical
    channel whose arguments are sequential probabilities (term k fires with p_k iff no
    earlier term fired), so it is rescaled through its categorical law: every outcome
    probability q_k = p_k * prod_{j<k} (1 - p_j) is multiplied by p/p0 and the chain is
    written back in sequential form (p'_k = q'_k / (1 - sum_{j<k} q'_j)) — exactly what
    multiplying every PAULI_CHANNEL argument by the factor does. Scaling the sequential
    arguments one by one would NOT be that (it changes the outcome law at second order)."""
    if not isinstance(text, str):
        extra = " — pass its .text" if type(text).__name__ == "Circuit" else ""
        raise TypeError(
            f"scale_noise expects circuit text (str), got {type(text).__name__}{extra}")
    if p0 == 0:
        raise ValueError("p0 (the baked-in noise strength) must be > 0 to rescale by p/p0")
    factor = p / p0
    lines = text.splitlines()
    out = []
    n_noise = sum(1 for l in lines if _NOISE_LINE.match(l))
    n_scaled = 0
    rem_old = rem_new = 1.0        # chain state: P(no earlier term fired), original / scaled
    for line in lines:
        m = _NOISE_ARGS.match(line)
        c = None if m else _CHAIN_ARGS.match(line)
        if m:
            args = [float(a) for a in m.group(4).split(",")]
            scaled = ", ".join("%.12g" % (a * factor) for a in args)
            line = f"{m.group(1)}{m.group(2)}{m.group(3) or ''}({scaled})" + line[m.end():]
            n_scaled += 1
        elif c:
            if c.group(2) != "ELSE_CORRELATED_ERROR":
                rem_old = rem_new = 1.0            # E / CORRELATED_ERROR opens a chain
            p_old = float(c.group(4))
            q_new = p_old * rem_old * factor       # the outcome's categorical probability, scaled
            p_new = q_new / rem_new if rem_new > 0.0 else 0.0
            rem_old *= 1.0 - p_old
            rem_new -= q_new
            line = f"{c.group(1)}{c.group(2)}{c.group(3) or ''}({'%.17g' % p_new})" + line[c.end():]
            n_scaled += 1
        else:
            rem_old = rem_new = 1.0                # any other line ends a chain (parser rule)
        out.append(line)
    if n_scaled != n_noise:
        raise ValueError(f"noise rewrite mismatch: {n_scaled} lines scaled but "
                         f"{n_noise} noise lines present")
    if len(out) != len(lines):
        raise ValueError("noise rewrite changed the line count")
    return "\n".join(out) + "\n"
