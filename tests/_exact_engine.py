"""The EXACT engine's Python surface — TEST API only (exact-out T5, owner ruling 2026-10-30).

The exact engine (TreeBuilder / forced law / reference battery) is the test oracle; its public entry points were
removed from ``xtim`` (``Circuit.compile_detector_sampler(engine="exact")``, ``Circuit.compile_sampler``,
``Circuit.compile_exact_branches``, ``Circuit.compile_reference`` / ``xtim.Reference`` / ``xtim.cache_dir``; the
reference is computed in memory since bloat-cut C2).
The engine bindings live in ``xtim._xtim_testapi`` (``CompiledProgram``, ``ExactBranchProgram``,
``ExactLawExpander``, ``ref_compile_text`` / ``ref_verify_cheap`` / ``ref_info`` / ``deferred_signature``); this
module is the former Python layer over them, verbatim in behaviour:

    exact_detector_sampler(circuit, seed=s)   # was circuit.compile_detector_sampler(seed=s, engine="exact")
    exact_sampler(circuit, seed=s)            # was circuit.compile_sampler(seed=s)
    compile_exact_branches(circuit)           # was circuit.compile_exact_branches()
    resolve(circuit)                          # was circuit._resolve_reference()

``circuit`` is an ``xtim.Circuit`` or circuit text. Every construction runs inside ``explicit_exact()`` (the
session counter of tests/conftest.py stays at zero)."""
from __future__ import annotations

import operator

import numpy as np

import xtim
from xtim import _xtim_testapi as _ta
from xtim._exact_scope import explicit_exact
from xtim.circuit import _refuse_unresolved_if, _unpack_bits
from xtim.errors import XtimParseError, XtimRejectError

import _exact_reference as _refmod
from _exact_forced import ExactBranches, ExactBranchSimulator, ExpandedLawTooLarge  # noqa: F401
from _exact_reference import Reference, compile_reference  # noqa: F401


def _circ(circuit):
    return circuit if isinstance(circuit, xtim.Circuit) else xtim.Circuit(circuit)


def resolve(circuit):
    """The invisible reference of ``circuit`` (memoised per Circuit object; computed in memory), or None."""
    c = _circ(circuit)
    memo = c.__dict__.setdefault("_exact_ref_memo", {})
    if c.text not in memo:
        with explicit_exact():             # the reference battery is the exact engine, explicitly
            memo[c.text] = _refmod.resolve_reference(c.text, source=c._source)
    return memo[c.text]


def exact_sampler(circuit, *, seed: int = 0) -> "CompiledSampler":
    """A raw measurement-record sampler on the exact engine (the former ``Circuit.compile_sampler``)."""
    c = _circ(circuit)
    _refuse_unresolved_if(c.text, "compile_sampler")
    with explicit_exact():
        return CompiledSampler(c, seed, resolve(c))


def exact_detector_sampler(circuit, *, seed: int = 0) -> "CompiledDetectorSampler":
    """A detector sampler on the exact engine, with raw PAULI_EXPECTATION values and raw measurement records
    (the former ``Circuit.compile_detector_sampler(engine="exact")``)."""
    c = _circ(circuit)
    _refuse_unresolved_if(c.text, "compile_detector_sampler(engine='exact')")
    with explicit_exact():
        return CompiledDetectorSampler(c, seed, resolve(c))


def compile_exact_branches(circuit) -> ExactBranchSimulator:
    """Forced faults, exact branches (the former ``Circuit.compile_exact_branches``; tests/_exact_forced.py)."""
    c = _circ(circuit)
    _refuse_unresolved_if(c.text, "compile_exact_branches")
    with explicit_exact():
        return ExactBranchSimulator(c)


class _CompiledBase:
    def __init__(self, circuit, seed: int, ref):
        self._circuit = circuit
        # The engine seed is an unsigned 64-bit int; validate here so a bad seed
        # raises a clear error instead of leaking a raw pybind11 TypeError from sample().
        try:
            seed_i = operator.index(seed)
        except TypeError:
            raise TypeError(f"seed must be an int, got {type(seed).__name__}") from None
        if not 0 <= seed_i < 2**64:
            raise ValueError(f"seed must be in [0, 2**64), got {seed_i}")
        self._seed = seed_i
        self._ref = ref
        # Compile ONCE here (deferral / propagation / bare-state / cascade) and hold the stateful
        # program, so every _run(shots) just samples it — no per-call recompile (the ~870ms d5 /
        # ~66ms d3 compile is paid a single time). Byte-identical to the old run_packed path.
        self._program = _ta.CompiledProgram(
            self._circuit.text,
            self._ref.text if self._ref is not None else "")
        if self._program.ref_error:
            raise XtimRejectError(
                -1, kind="ref_mismatch",
                message="rejected: supplied reference failed to load: "
                f"{self._program.ref_error}")
        if not self._program.ok:  # parse errors (unreachable: __init__ parses first)
            raise XtimParseError(self._program.errors)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(circuit={self._circuit!r}, "
                f"seed={self._seed})")

    @property
    def seed(self) -> int:
        return self._seed

    def _run(self, shots: int, *, want_meas: bool = True, want_det: bool = True,
             want_obs: bool = True, want_exp: bool = True) -> dict:
        # Validate the type up front so a str/None/float gives a clear error rather
        # than a leaked comparison TypeError or a silent float->int truncation.
        try:
            shots = operator.index(shots)
        except TypeError:
            raise TypeError(
                f"shots must be an integer, got {type(shots).__name__}") from None
        if shots < 0:
            raise ValueError("shots must be >= 0")
        # The program was compiled (and parse/ref-error checked) ONCE in __init__; just sample it.
        # The want_* mask tells the binding to materialize only the record-buffer channels this
        # sampler will actually read (the others' numpy arrays are skipped, trimming the per-shot
        # marshalling). The scalar counts/chi/reject metadata is always returned. Defaults are
        # all-True so any direct caller is unaffected.
        r = self._program.sample(int(shots), self._seed, want_meas, want_det,
                                 want_obs, want_exp)
        if r["rejected"]:
            raise self._reject_error(r["reject_gate_index"])
        return r

    def _reject_error(self, gate_index: int) -> XtimRejectError:
        """Resolve a run-path reject into a typed XtimRejectError with a DISTINCT
        kind. A gate_index >= 0 is an out-of-class gate (the propagation-class check,
        the SOLE rejection criterion). A -1 is either a supplied-reference n-mismatch
        or a non-gate-localized propagation-class rejection.
        """
        if gate_index >= 0:
            return XtimRejectError(gate_index, kind="class")
        if self._ref is not None:
            return XtimRejectError(
                -1, kind="ref_mismatch", ref_n=self._ref.n,
                circuit_n=self._circuit.num_qubits,
            )
        # no reference supplied: non-gate-localized propagation-class rejection.
        return XtimRejectError(-1, kind="class")


class CompiledSampler(_CompiledBase):
    """Raw measurement-record sampler (Stim's compile_sampler analog)."""

    def sample(self, shots: int, *, bit_packed: bool = False) -> np.ndarray:
        """One engine run -> the raw measurement records ``bool[shots, M]`` (mirrors
        ``stim.CompiledMeasurementSampler.sample``).

        With ``bit_packed=True`` the records come Stim-b8-packed (``uint8[shots,
        ceil(M/8)]``, little-endian bit order). All records of one call come from the
        SAME single engine run (one RNG stream)."""
        # Raw measurement sampler: only the measurements channel is ever read.
        r = self._run(shots, want_meas=True, want_det=False, want_obs=False,
                      want_exp=False)
        if bit_packed:
            return r["measurements"]
        return _unpack_bits(r["measurements"], r["num_measurements"])


class CompiledDetectorSampler(_CompiledBase):
    """Detector-event sampler (Stim's compile_detector_sampler analog) with the one
    record Stim cannot have: raw PAULI_EXPECTATION values."""

    def engine_report(self) -> dict:
        """Which engine backs this sampler (``"exact"``) and how its plans came out."""
        return {"engine": "exact",
                # zero_error_plan / expectation_channel: "built", "failed (live block)", ...;
                # "skipped: zero-error plan failed (live block)" is the channel build NOT being
                # attempted after the zero-error extraction of the same block failed.
                **self._program.plan_report()}

    def sample(self, shots: int, *, separate_observables: bool = False,
               append_observables: bool = False,
               return_expectations: bool = False, return_measurements: bool = False,
               bit_packed: bool = False) -> "np.ndarray | tuple[np.ndarray, ...]":
        """One engine run -> (dets[, obs][, exps][, meas]).

        dets bool[shots, D] detection events; obs bool[shots, O] observable flips;
        exps float64[shots, R] RAW expectation values; meas bool[shots, M] raw
        measurement bits. With bit_packed=True the bit records come Stim-b8-packed
        (uint8[shots, ceil(K/8)], little-endian bit order); exps stay float64.
        All requested records come from the SAME single engine run (one RNG stream).
        Returns the dets array alone unless extra records are requested.

        append_observables=True appends the observable-flip columns onto the detector
        array (Stim's combined `(shots, D+O)` layout) instead of returning them
        separately; it is mutually exclusive with separate_observables."""
        if append_observables and separate_observables:
            raise ValueError(
                "pass at most one of append_observables / separate_observables")
        # Request only the channels this call will read (mirrors the r[...] accesses below):
        #   detectors  — always (the primary return, both layouts).
        #   observables — only when appended onto / returned alongside the detectors.
        #   expectations — only with return_expectations.
        #   measurements — only with return_measurements.
        r = self._run(
            shots,
            want_det=True,
            want_obs=append_observables or separate_observables,
            want_exp=return_expectations,
            want_meas=return_measurements,
        )

        def bits(buf, k):
            return buf if bit_packed else _unpack_bits(buf, k)

        if append_observables:
            # Stim's default detector-sampler layout: observable flips appended as
            # extra columns after the detectors. Combine in the unpacked domain, then
            # re-pack if requested.
            dets = _unpack_bits(r["detectors"], r["num_detectors"])
            obs = _unpack_bits(r["observables"], r["num_observables"])
            combined = np.concatenate([dets, obs], axis=1) if obs.shape[1] else dets
            first = (np.packbits(combined, axis=1, bitorder="little")
                     if bit_packed else combined)
        else:
            first = bits(r["detectors"], r["num_detectors"])

        out = [first]
        if separate_observables:
            out.append(bits(r["observables"], r["num_observables"]))
        if return_expectations:
            out.append(r["expectations"])
        if return_measurements:
            out.append(bits(r["measurements"], r["num_measurements"]))
        return out[0] if len(out) == 1 else tuple(out)


def engine_detector_sampler(circuit, *, seed: int = 0, engine=None):
    """The detector sampler of a parametrized engine name: ``"exact"`` -> :func:`exact_detector_sampler` (test API);
    ``"twirl"`` / None -> the public ``Circuit.compile_detector_sampler`` (the only public engine)."""
    if engine == "exact":
        return exact_detector_sampler(circuit, seed=seed)
    if engine not in (None, "twirl"):
        raise ValueError(f"engine must be 'exact', 'twirl' or None, got {engine!r}")
    return _circ(circuit).compile_detector_sampler(seed=seed)


# The exact engine runs on the test API's OWN engine copy (its own state budget counters, scratch and owners): the
# process totals the exact-engine tests read are the sums over both extensions.
def live_state_bytes() -> int:
    return int(xtim.live_state_bytes()) + int(_ta.live_state_bytes())


def thread_scratch_bytes() -> int:
    return int(xtim._xtim._thread_scratch_bytes()) + int(_ta._thread_scratch_bytes())


def scratch_owner_count() -> int:
    return int(xtim._xtim._scratch_owner_count()) + int(_ta._scratch_owner_count())


def release_thread_state_scratch() -> None:
    xtim._xtim._release_thread_state_scratch()
    _ta._release_thread_state_scratch()
