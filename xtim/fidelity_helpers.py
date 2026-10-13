"""xtim.fidelity_helpers — byproduct-frame extraction helpers for logical operators.

Primary entry point::

    offsets = xtim.extract_frame(circuit, "X15*X16*X17*X18*X19*X20*X21")
    # offsets is a list[int] of positive k values such that rec[-k] are the records
    # whose parity pins the sign of the given logical Pauli expectation value.

Mechanism
---------
The function reuses the existing ``_xtim.expectation_frames`` binding, which
computes the canonical minimal sign-controlling record set (``required``) for each
declared ``PAULI_EXPECTATION`` column.

To probe an arbitrary operator without conflicts with existing declarations, the
function:

1. Strips all existing ``PAULI_EXPECTATION`` lines from the circuit text.
2. Appends a frameless ``PAULI_EXPECTATION(0) <operator>`` — column 0.
3. Calls ``_xtim.expectation_frames``.
4. If ``solvable[0]`` is falsy, raises ``XtimError`` (sign not pinnable).
5. Converts ``required[0]`` (absolute 0-indexed record indices) to rec[-k] offsets:
   ``k = circuit.num_measurements - abs_index``.

This keeps the column index predictably at 0 and avoids any label-clash with
existing declarations, which must have unique integer labels.
"""
from __future__ import annotations

import itertools
import re
import warnings

from . import _xtim
from .collect import DEFAULT_BAKED_P
from .errors import XtimError

# These two names appear in this module's public signatures.  Until 2026-08-29
# they were spelled as attributes of a package name this module never imports,
# so nothing could resolve them: harmless at runtime under postponed
# annotations, but typing.get_type_hints(), IDEs and doc tooling all failed on
# the public API of a package meant to be read by students.
#
# Imported for REAL, not under TYPE_CHECKING: the guarded form fixes static
# checkers but leaves get_type_hints() still raising, because the names do not
# exist at runtime.  Verified no cycle -- circuit.py imports THIS module only
# from inside functions -- and with these, all 13 callables here resolve.
from .circuit import Circuit          # noqa: F401  (used in annotations)
from .ler import PostselectedLER      # noqa: F401  (used in annotations)

# Matches any top-level PAULI_EXPECTATION line (single-line instructions only;
# REPEAT blocks with interior indentation are not matched, which is correct —
# PAULI_EXPECTATION cannot legally appear inside a REPEAT body).
_PE_LINE_RE = re.compile(r'^PAULI_EXPECTATION\b.*$', re.MULTILINE)


def _strip_pauli_expectations(text: str) -> str:
    """Remove all PAULI_EXPECTATION instruction lines from circuit text."""
    return _PE_LINE_RE.sub('', text)


def extract_frame(
    circuit: "Circuit",
    operator: str,
    *,
    minimal: bool = True,
) -> list[int]:
    """Return the byproduct-frame rec[-k] offsets for *operator*.

    Parameters
    ----------
    circuit : Circuit
        The circuit containing the protocol.
    operator : str
        A Pauli product in Stim/xtim ``PAULI_EXPECTATION`` notation,
        e.g. ``"X15*X16*X17*X18*X19*X20*X21"``.
    minimal : bool, default True
        If True (default), return the canonical minimal sign-controlling set —
        the smallest set of records whose XOR pins the sign of ``<operator>`` in
        the circuit's bare state.  This is the ``required`` field from
        ``_xtim.expectation_frames``.

        If False, attempt to return a larger "full readout" representative —
        the declared frame from an existing ``PAULI_EXPECTATION`` declaration that
        matches this operator's ``required`` set and is a superset of it, if one
        exists in the circuit.  Falls back to the minimal set when no such
        declaration is found (best-effort with no warning).

    Returns
    -------
    list[int]
        Positive integers *k* such that ``rec[-k]`` are the sign-controlling
        records, ordered by ascending absolute record index.  Convert back to
        0-indexed absolute positions via ``abs_index = circuit.num_measurements - k``.

        An empty list ``[]`` is a **valid** result meaning the operator's sign is
        already determined by the bare state without any measurement conditioning.
        This happens when the operator is in the stabilizer group (ideal expectation
        ±1 regardless of any measurement outcomes) or when its ideal expectation is
        exactly 0 (a "magic direction" such as ⟨Z̄⟩=0 for a T-state).  The resulting
        ``PAULI_EXPECTATION`` line simply carries no ``rec[-k]`` records.  To see
        the numeric expectation value, call ``xtim.diagnose``.

    Raises
    ------
    xtim.XtimError
        If the sign of *operator* is not pinnable by any measurement-record frame
        (e.g. the operator touches a free or logical qubit whose sign is not fixed
        by the bare state's stabilizer frame).
    xtim.XtimError
        If *operator* is syntactically invalid in the circuit context.
    """
    num_records = circuit.num_measurements

    # Build a probe text: strip existing PAULI_EXPECTATION declarations to avoid
    # duplicate-label errors, then append a frameless declaration for the operator.
    stripped = _strip_pauli_expectations(circuit.text)
    probe_text = stripped.rstrip() + f'\nPAULI_EXPECTATION(0) {operator}\n'

    # Surface syntax errors early with a clean message.
    info = _xtim.parse_info(probe_text)
    if not info['ok']:
        msgs = '; '.join(msg for _, msg in info.get('errors', []))
        raise XtimError(
            f"extract_frame: operator {operator!r} is invalid in this circuit: {msgs}"
        )

    result = _xtim.expectation_frames(probe_text)

    if not result.get('ok'):
        errs = result.get('errors', [])
        msg = '; '.join(m for _, m in errs) if errs else (result.get('error') or 'unknown')
        raise XtimError(
            f"extract_frame: could not analyze operator {operator!r} "
            "(the circuit or operator may be malformed — check qubit indices): "
            f"{msg}"
        )

    if result['num_columns'] == 0:
        raise XtimError(
            f"extract_frame: operator {operator!r} produced no expectation column "
            "(internal error — circuit may not have parsed correctly)"
        )

    # Column 0 is our probed operator.
    col = 0
    if not result['solvable'][col]:
        raise XtimError(
            f"extract_frame: operator {operator!r}'s sign is not pinnable by a "
            "measurement-record frame (the operator has a free or logical component "
            "whose sign is not determined by the bare-state stabilizer frame)"
        )

    abs_indices: list[int] = result['required'][col]
    minimal_offsets = [num_records - idx for idx in abs_indices]

    if minimal:
        return minimal_offsets

    # minimal=False: try to find a spanning declared frame from the original circuit.
    declared_offsets = _try_declared_frame(
        circuit, abs_indices, num_records
    )
    return declared_offsets if declared_offsets is not None else minimal_offsets


def _try_declared_frame(
    circuit: "Circuit",
    required_abs: list[int],
    num_records: int,
) -> list[int] | None:
    """For ``minimal=False``: look for an existing PAULI_EXPECTATION declaration in
    *circuit* whose ``required`` set equals *required_abs* and whose ``declared`` set
    is a superset.  Returns the declared frame as rec[-k] offsets, or None to signal
    fallback to the minimal set.
    """
    result = _xtim.expectation_frames(circuit.text)
    if not result.get('ok') or result['num_columns'] == 0:
        return None

    required_set = frozenset(required_abs)
    for col_req, col_dec in zip(result['required'], result['declared']):
        if frozenset(col_req) == required_set and required_set.issubset(frozenset(col_dec)):
            return [num_records - idx for idx in col_dec]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# fidelity_from_logicals — full-support logical fidelity from k generator pairs
# ─────────────────────────────────────────────────────────────────────────────

# One (letter, qubit) term of a Pauli-product string, e.g. "X15".
_PAULI_TERM_RE = re.compile(r'([XYZ])(\d+)')

# Symplectic (x, z) bits per single-qubit Pauli letter and the inverse map.
_LETTER_TO_XZ = {'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
_XZ_TO_LETTER = {(1, 0): 'X', (1, 1): 'Y', (0, 1): 'Z'}

# Warn once 4**k grows large (k>=5 => >=1023 declared expectation columns).
_LARGE_K = 5


def _parse_pauli(operator: str) -> dict[int, tuple[int, int]]:
    """Parse a Pauli-product string into ``{qubit: (x, z)}`` symplectic bits.

    Repeated qubits are combined via GF(2) XOR (a standard Pauli product, phase
    discarded).  Identity qubits (``(0, 0)``) are dropped.  Raises ``XtimError`` on
    a string that contains no recognizable Pauli terms."""
    acc: dict[int, tuple[int, int]] = {}
    matched = False
    for letter, q in _PAULI_TERM_RE.findall(operator):
        matched = True
        qi = int(q)
        x, z = _LETTER_TO_XZ[letter]
        px, pz = acc.get(qi, (0, 0))
        acc[qi] = (px ^ x, pz ^ z)
    if not matched:
        raise XtimError(
            f"fidelity_from_logicals: {operator!r} is not a valid Pauli-product "
            "string (expected terms like 'X15*Z16*Y17')")
    return {qi: xz for qi, xz in acc.items() if xz != (0, 0)}


def _pauli_mul(a: dict[int, tuple[int, int]],
               b: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    """GF(2) product (per-qubit XOR) of two symplectic Pauli reps; phase discarded."""
    out = dict(a)
    for qi, (x, z) in b.items():
        px, pz = out.get(qi, (0, 0))
        r = (px ^ x, pz ^ z)
        if r == (0, 0):
            out.pop(qi, None)
        else:
            out[qi] = r
    return out


def _commute(a: dict[int, tuple[int, int]], b: dict[int, tuple[int, int]]) -> bool:
    """True iff two Paulis commute (even symplectic inner product over GF(2))."""
    acc = 0
    for qi in set(a) | set(b):
        ax, az = a.get(qi, (0, 0))
        bx, bz = b.get(qi, (0, 0))
        acc ^= (ax & bz) ^ (az & bx)
    return acc == 0


def _pauli_to_str(d: dict[int, tuple[int, int]]) -> str:
    """Render a symplectic Pauli rep as a ``PAULI_EXPECTATION`` product string."""
    return "*".join(f"{_XZ_TO_LETTER[d[qi]]}{qi}" for qi in sorted(d))


def _validate_logicals(logicals) -> "list[tuple[dict, dict]]":
    """Validate the logical-Pauli algebra and return ``[(X̄_i, Z̄_i), ...]``.

    ``logicals`` maps each logical qubit to ``{"X": <str>, "Z": <str>}``.  Checks the
    canonical relations via symplectic inner products — ``X̄_i`` and ``Z̄_i``
    anticommute; every cross-qubit (i!=j) pair commutes — and raises ``XtimError``
    naming the offending pair on any violation."""
    if not logicals:
        raise XtimError("fidelity_from_logicals: `logicals` is empty; supply at least "
                        "one logical qubit's {'X': ..., 'Z': ...} generators")

    gens: list[tuple[dict, dict]] = []
    keys = sorted(logicals)
    for key in keys:
        spec = logicals[key]
        if not (hasattr(spec, "get") and "X" in spec and "Z" in spec):
            raise XtimError(
                f"fidelity_from_logicals: logical qubit {key!r} must map to both an "
                f"'X' and a 'Z' generator string, got {spec!r}"
                " — did you forget the outer qubit-index key? "
                "Expected {0: {'X': ..., 'Z': ...}}"
            )
        xbar = _parse_pauli(spec["X"])
        zbar = _parse_pauli(spec["Z"])
        gens.append((xbar, zbar))

    for i, (xi, zi) in enumerate(gens):
        if _commute(xi, zi):
            raise XtimError(
                f"fidelity_from_logicals: X̄_{keys[i]} and Z̄_{keys[i]} must "
                "ANTICOMMUTE (they are the conjugate logical operators of one qubit) "
                "but they COMMUTE — check the generators for this logical qubit")
    for a in range(len(gens)):
        for b in range(a + 1, len(gens)):
            for na, pa in (("X", gens[a][0]), ("Z", gens[a][1])):
                for nb, pb in (("X", gens[b][0]), ("Z", gens[b][1])):
                    if not _commute(pa, pb):
                        raise XtimError(
                            f"fidelity_from_logicals: {na}̄_{keys[a]} and "
                            f"{nb}̄_{keys[b]} must COMMUTE (distinct logical "
                            "qubits) but they ANTICOMMUTE — the generators do not form "
                            "a valid logical Pauli algebra")
    return gens


def _logical_products(gens: "list[tuple[dict, dict]]") -> list[str]:
    """The ``4**k - 1`` non-identity logical Pauli products as product strings.

    Each element chooses one of ``{I, X, Y, Z}`` per logical qubit; ``X``->``X̄_i``,
    ``Z``->``Z̄_i``, ``Y``->``X̄_i·Z̄_i`` (combined per physical qubit — the grammar's
    per-qubit ``Y`` carries the Hermitian ``i`` phase).  The all-identity element is
    skipped."""
    k = len(gens)
    products: list[str] = []
    for choice in itertools.product("IXYZ", repeat=k):
        if all(c == "I" for c in choice):
            continue
        acc: dict[int, tuple[int, int]] = {}
        for i, c in enumerate(choice):
            xbar, zbar = gens[i]
            if c == "X":
                acc = _pauli_mul(acc, xbar)
            elif c == "Z":
                acc = _pauli_mul(acc, zbar)
            elif c == "Y":
                acc = _pauli_mul(_pauli_mul(acc, xbar), zbar)
        if not acc:
            raise XtimError(
                "fidelity_from_logicals: logical product %r collapses to identity — "
                "the generators are linearly dependent and do not form an independent "
                "logical Pauli algebra" % (choice,))
        products.append(_pauli_to_str(acc))
    return products


def _augmented_circuit(circuit: "Circuit", products: list[str]) -> "Circuit":
    """A copy of *circuit* with its ``PAULI_EXPECTATION`` lines replaced by one column
    per logical Pauli *product*, each carrying its ``extract_frame`` byproduct frame."""
    from .circuit import Circuit

    lines = []
    for j, prod in enumerate(products):
        offsets = extract_frame(circuit, prod)  # raises XtimError if unpinnable
        frame = " ".join(f"rec[-{k}]" for k in offsets)
        lines.append(f"PAULI_EXPECTATION({j}) {prod}" + (f" {frame}" if frame else ""))
    body = _strip_pauli_expectations(circuit.text).rstrip()
    text = body + "\n" + "\n".join(lines) + "\n"
    return Circuit(text, _source=getattr(circuit, "_source", None))


def _noiseless_F_warning(betas, k: int) -> "str | None":
    """Advisory when the beta-only noiseless fidelity ``F = (1 + Σβ²)/2**k`` is < 1.

    For a genuine magic state with a correct full-support pinnable algebra Parseval
    forces ``Σβ² = 2**k - 1`` exactly, so ``F == 1``.  A shortfall means the declared
    logicals are wrong for this code, or the prep is genuinely imperfect (a mixed
    reference).  Returns the warning string, or None when ``F`` is 1 within fp."""
    noiseless_F = (1.0 + sum(b * b for b in betas if b == b)) / float(2 ** k)
    if noiseless_F < 1.0 - 1e-9:
        return (
            "noiseless 1-F = %.6g: the declared logicals may be wrong for this code, "
            "or the prep is genuinely imperfect (a correct full-support magic-state "
            "algebra reaches F=1 noiselessly)." % (1.0 - noiseless_F))
    return None


def _full_support_fidelity(circuit: "Circuit",
                           gens: "list[tuple[dict, dict]]",
                           k: int,
                           *,
                           p: "float | None",
                           shots: int,
                           seed: int,
                           p0: float = DEFAULT_BAKED_P) -> "PostselectedLER":
    """Shared full-support-fidelity core: score the ``4**k`` logical Pauli products.

    Emits one ``PAULI_EXPECTATION`` column per non-identity product (with its
    byproduct frame), auto-post-selects on the deterministic detectors, and returns
    ``1 - F`` with ``F = (1/2**k) Σ_P β_P⟨P⟩`` as a fidelity-mode ``PostselectedLER``.
    ``gens`` is the validated ``[(X̄_i, Z̄_i), ...]`` list from ``_validate_logicals``."""
    from .collect import Task, collect
    from .diagnose import diagnose
    from .ler import _build_fidelity_result, _keep_from_deterministic

    msgs: list[str] = []
    if k >= _LARGE_K:
        msgs.append(
            "k=%d logical qubits => %d full-support Pauli expectation columns: this is "
            "expensive to build and score (4**k). Consider a smaller support or fewer "
            "shots." % (k, 4 ** k - 1))

    products = _logical_products(gens)
    aug = _augmented_circuit(circuit, products)

    # Noiseless-F validation: the beta-only ceiling F = (1 + Σβ²)/2**k from one
    # noiseless run. Correct generators for a genuine magic state give exactly 1
    # (Parseval: I + Σ_P β_P² = 2**k).
    betas = [e.signed_value for e in diagnose(aug).expectations]
    warn = _noiseless_F_warning(betas, k)
    if warn is not None:
        msgs.append(warn)

    keep = _keep_from_deterministic(diagnose(circuit).deterministic_detectors)
    task = Task(circuit=aug, p=p, shots=shots, seed=seed, keep=keep, target_k=k, p0=p0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (row,) = collect([task])
    for w in caught:
        if issubclass(w.category, UserWarning):
            msgs.append(str(w.message).split("\n", 1)[0].strip())

    result = _build_fidelity_result(row, shots=shots, target_k=k, p=p, msgs=msgs)
    for m in result.warnings:
        warnings.warn(m, UserWarning, stacklevel=3)
    return result


def fidelity_from_logicals(circuit: "Circuit",
                           logicals: dict,
                           *,
                           p: "float | None" = None,
                           shots: int = 10 ** 6,
                           seed: int = 0,
                           p0: float = DEFAULT_BAKED_P) -> "PostselectedLER":
    """Post-selected full-support logical **infidelity** ``1 - F`` from k generator pairs.

    Unlike :func:`postselected_logical_error_rate` — which scores only the
    ``PAULI_EXPECTATION`` columns a circuit happens to declare — this computes the
    TRUE full-support fidelity of the prepared logical state against its ideal,

        ``F = (1 / 2**k) Σ_P β_P ⟨P⟩``,

    where ``P`` runs over all ``4**k`` logical Pauli products built from the supplied
    generators and ``β_P`` is each product's noiseless target.  It auto-declares every
    product with its byproduct frame, auto-post-selects on the deterministic detectors,
    and returns ``1 - F`` as a fidelity-mode :class:`PostselectedLER`.

    Parameters
    ----------
    circuit : Circuit
        The magic-state-prep protocol.
    logicals : dict
        Maps each logical qubit to its generator pair, e.g.
        ``{0: {"X": "X15*X16*X17*X18*X19*X20*X21", "Z": "Z15*..."}}``.  The generators
        must form a valid logical Pauli algebra (``X̄_i``/``Z̄_i`` anticommute, distinct
        qubits commute); a violation raises :class:`xtim.XtimError` naming the pair.
        ``k`` (the fidelity denominator ``2**k``) is inferred as the number of pairs.
    p : float | None, default None
        Physical error rate to scale the noise to.  ``None`` runs the circuit's
        baked-in noise as-is — unlike the sibling :func:`postselected_logical_error_rate`
        whose default is ``p=1e-3``.
    shots : int, default 1_000_000
        Number of shots (matches the default of :func:`postselected_logical_error_rate`).
    seed : int, default 0
        Engine seed.
    p0 : float, default 1e-3
        The circuit's baked-in noise strength, used when rescaling from ``p0`` to ``p``.
        Pass the value that was used when the circuit was built if it differs from 1e-3.

    Returns
    -------
    PostselectedLER
        Fidelity-mode result; ``.value`` is ``1 - F``.  ``.target_k`` is the inferred
        number of logical qubits ``k`` (the fidelity denominator is ``2**k``).
        ``.warnings`` flags a noiseless ``F < 1`` (wrong logicals or an imperfect prep),
        a large ``k``, and the usual zero-error / low-acceptance advisories.

    Raises
    ------
    xtim.XtimError
        If the generators do not form a valid logical Pauli algebra, if a generator
        string is malformed, or if a product's sign is not pinnable by the circuit's
        measurement-record frame.
    """
    from .circuit import Circuit

    if not isinstance(circuit, Circuit):
        raise TypeError(
            "fidelity_from_logicals expects an xtim.Circuit, got "
            f"{type(circuit).__module__}.{type(circuit).__name__}")
    if p is not None:
        if (isinstance(p, bool) or not isinstance(p, (int, float))
                or p != p or not (0.0 <= p <= 1.0)):
            raise ValueError(f"p must be a float in [0,1] or None, got {p!r}")
        p = float(p)
    shots = int(shots)
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")

    gens = _validate_logicals(logicals)
    return _full_support_fidelity(circuit, gens, len(gens),
                                  p=p, shots=shots, seed=seed, p0=p0)
