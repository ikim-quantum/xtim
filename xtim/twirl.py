"""Twirl record sampler — the exact-channel twirl product as a Python sampler.

``compile_twirl_sampler`` compiles a circuit into a :class:`_xtim.TwirlSampler`
(engine class ``qeccore::TwirlRecordSampler``): per shot, the sparse noise draw is
composed exactly, reduced to a content-keyed plan, and DETECTOR/OBSERVABLE record
channels are emitted directly in channel space.

Semantics contract (full text: docs/twirl_record_sampler.md):

* **Deterministic detector/observable channels are EXACT** — their laws equal the
  production sampler's (5σ-gated on all benchmarks, including the CH-class ones).
* **Gauge-record marginals/correlations are TWIRLED**: gauge detector columns are
  DECLARED independent fair coins in this record product.
* **Anti detectors** refuse at compile time (``ValueError``).
* **Refused observables** (operator not in the certified group — the usual logical
  case) make ``sample()`` raise ``RuntimeError`` naming them, unless the sampler was
  compiled with ``skip_refused_observables=True`` (then those columns are all-zero
  and ``channel_report()["refused_observables"]`` marks them).
* **No shot is ever dropped** (V3-T3): shots that trip a twirl guard are computed
  per-shot by the exact engine and delivered like any other shot;
  ``channel_report()["exact_shots"]`` counts them (diagnostic only).
* The plan-cache ``disk_cache`` path is noise-blind: one cache file serves p-sweeps.

Task 5 addition — IF-branch partition:
When the circuit has IF blocks, ``compile_twirl_sampler`` returns a
:class:`PartitionedTwirlSampler` that groups shots by their decision-bit pattern,
compiles a concrete (branch-free) circuit per pattern, and scatters results back in
original shot order.  Circuits without IF blocks return the existing
``_xtim.TwirlSampler`` unchanged (byte-identical hot path).

Task 6 addition — superset records + live-mask:
:func:`PartitionedTwirlSampler.sample` accepts ``return_mask=True`` to return a
:class:`TwirlBatch` namedtuple ``(dets, obs, live_mask, amps)`` where ``dets`` and
``live_mask`` are Stim b8-packed arrays of **superset width** (one column per
detector declared anywhere in the original circuit).  For a given shot, columns whose
enclosing IF branch was DROPPED carry ``dets=0, live_mask=0``; columns whose branch
was TAKEN (or that are outside all branches) carry the sampled bit with
``live_mask=1``.  ``return_mask=False`` (the default) is byte-identical to the
Task-5 behaviour.
"""
from __future__ import annotations

import os
from collections import namedtuple

import numpy as np

from . import _xtim

# Task 6: namedtuple returned by sample(return_mask=True).
# dets, obs:      Stim b8-packed uint8 arrays of shape (shots, ceil(K/8)).
# live_mask:      b8-packed uint8 of the same shape as dets; 1 = detector active for this shot.
# amps:           reserved for Task 7 (amplitude retention); always None in Task 6.
# decisions:      Task 2b.1 — Stim b8-packed uint8 (shots, ceil(nD/8)) of declared Born DECISION
#                 output bits; None for circuits without DECISIONs.  Defaulted so existing
#                 4-arg TwirlBatch(...) constructions (the IF-partition path) keep working.
TwirlBatch = namedtuple("TwirlBatch", ["dets", "obs", "live_mask", "amps", "decisions"],
                        defaults=(None,))


def twirl_cache_dir() -> str:
    """The automatic plan-cache directory: ``$XTIM_TWIRL_CACHE_DIR`` if set, else
    ``~/.cache/xtim/twirl``. Used by ``compile_twirl_sampler(disk_cache=True)``."""
    return os.environ.get("XTIM_TWIRL_CACHE_DIR",
                          os.path.join(os.path.expanduser("~"), ".cache", "xtim", "twirl"))


def _make_inner_sampler(circuit_text, p_factor, disk_cache, selfcheck,
                        skip_refused_observables):
    """Create a bare ``_xtim.TwirlSampler`` from resolved text (no disk_cache forwarding
    to inner per-pattern samplers — reserved for a future extension)."""
    return _xtim.TwirlSampler(
        str(circuit_text),
        float(p_factor),
        "",           # disk_cache not forwarded to per-pattern inner samplers
        int(selfcheck),
        bool(skip_refused_observables),
        "",           # disk_cache_auto_dir likewise
    )


class _DecisionTwirlSampler:
    """Task 2b.1 wrapper for a branch-free ``_xtim.TwirlSampler`` that declares ≥1 ``DECISION``.

    The engine's ``sample()`` returns a 3-tuple ``(dets, obs, decisions)`` when decisions are
    declared; this wrapper repackages it into a :class:`TwirlBatch` (so callers read
    ``batch.decisions``).  Every other attribute — ``sample_barrier`` (whose ``BarrierBuffer``
    already carries ``.decisions()``), ``channel_report``, ``num_detectors``, ``num_decisions``,
    … — delegates to the inner engine sampler unchanged.

    Decision-free circuits never reach this wrapper: ``compile_twirl_sampler`` returns the raw
    ``_xtim.TwirlSampler`` for them, byte-identical to the pre-2b.1 ``(dets, obs)`` API.
    """

    def __init__(self, inner: "_xtim.TwirlSampler") -> None:
        self._inner = inner

    def sample(self, shots, seed=None, input_pauli=None):
        out = self._inner.sample(shots, seed, input_pauli)
        # Engine returns (dets, obs, decisions) when DECISIONs are declared.
        dets, obs, decisions = out
        return TwirlBatch(dets, obs, None, None, decisions)

    def __getattr__(self, name):
        # Delegate everything else (sample_barrier, channel_report, num_*, …) to the engine.
        return getattr(self._inner, name)


class PartitionedTwirlSampler:
    """Twirl sampler for circuits with IF-branch blocks (Task 5 / decoder-feedback).

    Groups shots by their decision-bit pattern, resolves each pattern to a concrete
    branch-free circuit, compiles per-pattern ``_xtim.TwirlSampler`` instances
    (cached), samples each group, and scatters results back in original shot order.

    Not intended for direct construction; use :func:`compile_twirl_sampler`.
    """

    # Maximum number of IF-driving bits before we raise to avoid a combinatorial explosion.
    _MAX_DRIVING_BITS = 20

    def __init__(self, circuit_text: str, p_factor: float,
                 disk_cache, selfcheck: int, skip_refused_observables: bool) -> None:
        self._text = str(circuit_text)
        self._p_factor = float(p_factor)
        self._disk_cache = disk_cache
        self._selfcheck = int(selfcheck)
        self._skip_refused = bool(skip_refused_observables)

        # IF-driving bits info (from C++ helpers).
        self._driving_bits: list = _xtim.if_driving_bits(self._text)
        self._driving_global_offsets: list = _xtim.if_driving_global_offsets(self._text)
        self._total_width: int = _xtim.bits_total_width(self._text)

        if len(self._driving_global_offsets) > self._MAX_DRIVING_BITS:
            raise ValueError(
                f"circuit has {len(self._driving_global_offsets)} IF-driving bits; "
                f"cap is {self._MAX_DRIVING_BITS} to avoid combinatorial explosion"
            )

        # Task 6: superset detector layout — one entry per detector declared anywhere in
        # the original circuit.  Each entry is a list of (global_offset, cond_value) guard
        # tuples; an empty list means the detector is outside all IF blocks (always active).
        self._det_spans: list = _xtim.detector_branch_spans(self._text)

        # Per-pattern sampler cache: bytes(driving_pattern) -> _xtim.TwirlSampler
        self._sampler_cache: dict = {}

        # Reference sampler (all-zeros pattern) for num_detectors / num_observables /
        # channel_report.  Constructed eagerly to surface compile errors at
        # compile_twirl_sampler() call time rather than at sample() time.
        self._ref_sampler = self._get_sampler(bytes(len(self._driving_global_offsets)))

    # ── private helpers ───────────────────────────────────────────────────────

    def _full_pattern_from_driving(self, driving_key: bytes) -> list:
        """Build a full bit_values list (length = total_width) from the driving-bits
        key (length = # driving bits).  Non-driving bits are set to 0 (irrelevant for
        IF conditions)."""
        full = [0] * self._total_width
        for idx, global_off in enumerate(self._driving_global_offsets):
            if idx < len(driving_key):
                full[global_off] = driving_key[idx]
        return full

    def _get_sampler(self, driving_key: bytes) -> "_xtim.TwirlSampler":
        """Return (possibly cached) per-pattern inner sampler for this driving key."""
        if driving_key not in self._sampler_cache:
            full_pattern = self._full_pattern_from_driving(driving_key)
            resolved_text = _xtim.resolve_branches_text(self._text, full_pattern)
            sampler = _make_inner_sampler(
                resolved_text, self._p_factor, self._disk_cache,
                self._selfcheck, self._skip_refused)
            self._sampler_cache[driving_key] = sampler
        return self._sampler_cache[driving_key]

    def _active_det_mask(self, full_pattern: list) -> list:
        """Return list[bool] of length len(self._det_spans) indicating which superset
        detector columns are active for the given full bit_values pattern.

        A detector is active if all its IF-guard conditions are satisfied by
        ``full_pattern``.  Detectors outside all IF blocks have an empty guard list
        and are always active.
        """
        return [
            all(full_pattern[off] == val for off, val in guards)
            for guards in self._det_spans
        ]

    # ── public interface (mirrors _xtim.TwirlSampler) ────────────────────────

    @property
    def num_detectors(self) -> int:
        return self._ref_sampler.num_detectors

    @property
    def num_observables(self) -> int:
        return self._ref_sampler.num_observables

    @property
    def num_decisions(self) -> int:
        # Bug-hunt wave 5: this forwarder was missing, so any IF-bearing circuit that
        # also declares DECISIONs raised AttributeError on a documented attribute
        # (the class docstring lists num_decisions among the forwarded ones).
        return self._ref_sampler.num_decisions

    def __getattr__(self, name):
        # Delegate anything not defined here (sample_barrier, output_wires, ...) to the
        # reference sampler.  Without this, run_protocol's stage-0 barrier call crashed
        # with AttributeError for any protocol whose FIRST stage contains IF blocks
        # (bug-hunt wave 6; pre-existing).  The delegated sample_barrier samples the
        # all-zeros driving pattern, which is exactly the stage-0 case (its IF bits, if
        # any, come from its own records, not from an upstream stage).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._ref_sampler, name)

    def channel_report(self) -> dict:
        """Channel report from the all-zeros-pattern (reference) inner sampler."""
        return self._ref_sampler.channel_report()

    def sample(
        self,
        shots: int,
        seed=None,
        *,
        decision_bits=None,
        input_pauli=None,
        return_mask: bool = False,
    ):
        """Sample ``shots`` records using decision bits to select per-shot circuits.

        Parameters
        ----------
        shots:
            Number of shots to sample.
        seed:
            RNG seed (forwarded to each inner sampler with per-group derivation).
        decision_bits:
            Required when the circuit has IF blocks.  A ``(shots, total_bits_width)``
            uint8 array.  Each row gives the full decision-bit vector for one shot;
            shots with the same driving-bit pattern are grouped and sampled together.
        input_pauli:
            Optional ``(shots, 2*support)`` uint8 per-shot Pauli frame (Task 4).
            The group slice is forwarded to each inner sampler's ``input_pauli=``.
        return_mask:
            If ``True``, return a :class:`TwirlBatch` namedtuple
            ``(dets, obs, live_mask, amps=None)`` where ``dets`` and ``live_mask``
            use the **superset** detector layout (one column per detector declared
            anywhere in the original circuit, in stream order).  Columns whose
            enclosing IF branch was DROPPED for a given shot carry ``dets=0,
            live_mask=0``; active columns carry the sampled bit with ``live_mask=1``.
            If ``False`` (the default), return a plain ``(dets, obs)`` tuple —
            byte-identical to the Task-5 behaviour.

        Returns
        -------
        When ``return_mask=False``: ``(dets, obs)`` tuple of uint8 numpy arrays,
            Stim b8-packed, shape ``(shots, …)``.
        When ``return_mask=True``: :class:`TwirlBatch` namedtuple with superset-width
            ``dets``, ``obs``, and ``live_mask`` (b8 LE-packed).
        """
        shots = int(shots)
        if decision_bits is None:
            raise ValueError(
                f"circuit has IF branches ({len(self._driving_bits)} driving bit(s)); "
                "decision_bits must be provided as a (shots, total_bits_width) uint8 array"
            )

        db = np.asarray(decision_bits, dtype=np.uint8)
        if db.ndim != 2 or db.shape[0] != shots or db.shape[1] != self._total_width:
            raise ValueError(
                f"decision_bits: expected shape ({shots}, {self._total_width}), "
                f"got {tuple(db.shape)}"
            )

        # Finding 1 guard: the non-mask scatter path packs inner detectors into a
        # fixed ref-width buffer starting at bit 0.  When branch-dependent detectors
        # are present, different decision patterns activate DIFFERENT detector SETS,
        # so a given physical detector lands at a different column for different groups
        # -> silently wrong values.  Loud-refuse here; use return_mask=True instead.
        if not return_mask and any(self._det_spans):
            raise ValueError(
                "this circuit has detectors inside IF branches; call "
                "sample(..., return_mask=True) for the superset detector layout + live-mask")

        # Pre-check input_pauli shape if provided.
        ip_arr = None
        if input_pauli is not None:
            ip_arr = np.asarray(input_pauli, dtype=np.uint8)

        # Extract driving-pattern columns (subset of decision_bits).
        if self._driving_global_offsets:
            driving_cols = db[:, self._driving_global_offsets]  # (shots, n_driving)
        else:
            driving_cols = np.empty((shots, 0), dtype=np.uint8)

        # Group shot indices by driving pattern.
        groups: dict[bytes, list[int]] = {}
        for i in range(shots):
            key = bytes(driving_cols[i].tolist())
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        # Allocate output arrays.
        OBB = (self.num_observables + 7) // 8
        # Width 0 when the circuit declares no observables: padding to 1 made the
        # per-group scatter `obs_out[indices] = obs_g` fail with a shape mismatch for
        # (n_group, 0) group output (bug-hunt wave 6).
        obs_out = np.zeros((shots, OBB), dtype=np.uint8)
        decs_out = None          # lazily sized from the first group's decisions

        # Task 6: superset detector layout dimensions.
        n_superset = len(self._det_spans)
        SDBB = (n_superset + 7) // 8   # bytes for superset-width dets / live_mask

        if return_mask:
            # Superset-width output buffers.
            dets_out = np.zeros((shots, max(SDBB, 1)), dtype=np.uint8)
            live_mask_out = np.zeros((shots, max(SDBB, 1)), dtype=np.uint8)
        else:
            # Task-5 byte-identical path: use ref-sampler detector count.
            DBB = (self.num_detectors + 7) // 8
            dets_out = np.zeros((shots, max(DBB, 1)), dtype=np.uint8)

        for p_idx, (pattern_key, indices) in enumerate(groups.items()):
            if not indices:
                continue

            # Per-group seed (Finding M-1):
            # - If seed is None: pass None to each sampler (each continues its current stream),
            #   giving entropy across calls and reproducibility within per-shot order.
            # - If seed is S: derive a reproducible per-group seed using the offset counter.
            if seed is None:
                group_seed = None
            else:
                group_seed = (int(seed) + p_idx * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF

            # Get or create inner sampler for this driving pattern.
            sampler = self._get_sampler(pattern_key)

            # Build input_pauli slice for this group (or None).
            ip_slice = None
            if ip_arr is not None:
                ip_slice = ip_arr[indices]

            # Sample the group.  The engine returns a 3-tuple when the circuit
            # declares DECISIONs (bug-hunt wave 5: unpacking 2 crashed with
            # "too many values to unpack" for any circuit that has BOTH IF blocks
            # — which routes here — AND a DECISION declaration).
            _out_g = sampler.sample(len(indices), seed=group_seed,
                                    input_pauli=ip_slice)
            if isinstance(_out_g, tuple) and len(_out_g) == 3:
                dets_g, obs_g, decs_g = _out_g
            else:
                dets_g, obs_g = _out_g
                decs_g = None
            if decs_g is not None:
                if decs_out is None:
                    decs_out = np.zeros((shots, np.asarray(decs_g).shape[1]),
                                        dtype=np.uint8)
                decs_out[indices] = np.asarray(decs_g, dtype=np.uint8)

            if return_mask:
                # ── Task 6 superset scatter ────────────────────────────────────────
                # active_mask[i] = True  iff superset detector i is live for this pattern.
                # active_cols   = superset column indices of live detectors, in order.
                # The inner sampler's k-th detector corresponds to active_cols[k].
                full_pattern = self._full_pattern_from_driving(pattern_key)
                active_mask = self._active_det_mask(full_pattern)
                active_cols = [i for i, a in enumerate(active_mask) if a]
                n_active = len(active_cols)
                n_g = len(indices)

                # Unpack inner detector bits (LE Stim b8 → per-bit columns).
                unpacked_g = np.unpackbits(dets_g, axis=1, bitorder="little")
                # unpacked_g shape: (n_g, dets_g.shape[1] * 8); the first
                # n_active columns are the valid ones.

                # Scatter det bits to superset positions.
                superset_bits = np.zeros((n_g, max(SDBB * 8, 8)), dtype=np.uint8)
                if active_cols:
                    superset_bits[:, active_cols] = unpacked_g[:, :n_active]
                dets_out[indices] = np.packbits(
                    superset_bits[:, :SDBB * 8], axis=1, bitorder="little")

                # Build live_mask: 1 at each active superset column, 0 elsewhere.
                mask_bits = np.zeros((n_g, max(SDBB * 8, 8)), dtype=np.uint8)
                if active_cols:
                    mask_bits[:, active_cols] = 1
                live_mask_out[indices] = np.packbits(
                    mask_bits[:, :SDBB * 8], axis=1, bitorder="little")

                obs_out[indices] = obs_g
            else:
                # Task-5 byte-identical scatter: direct copy.
                dets_out[indices] = dets_g
                obs_out[indices] = obs_g

        # Return correctly-shaped arrays.
        if return_mask:
            return TwirlBatch(
                dets_out[:, :SDBB],
                obs_out[:, :OBB],
                live_mask_out[:, :SDBB],
                None,
                decs_out,
            )
        if decs_out is not None:
            # Mirror the single-sampler contract: DECISION-declaring circuits get a
            # TwirlBatch carrying .decisions (see _DecisionTwirlSampler.sample).
            return TwirlBatch(dets_out[:, :DBB], obs_out[:, :OBB], None, None, decs_out)
        return dets_out[:, :DBB], obs_out[:, :OBB]


def compile_twirl_sampler(
    circuit_text: str,
    p_factor: float = 1.0,
    disk_cache: "str | bool | None" = None,
    selfcheck: int = 2000,
    skip_refused_observables: bool = False,
):
    """Compile ``circuit_text`` into a twirl record sampler.

    Usage::

        s = xtim.compile_twirl_sampler(text, p_factor=1.0)
        dets, obs = s.sample(shots, seed=1)   # Stim-b8-packed uint8 (shots, ceil(K/8))
        rep = s.channel_report()

    Parameters
    ----------
    circuit_text:
        Stim-superset circuit text (the same normalization pipeline as the engine's
        bench driver: coherentize feedback -> strip -> defer -> bare state).
    p_factor:
        Noise scale relative to the circuit's declared probabilities (p / p_circ).
    disk_cache:
        Plan-cache persistence (noise-blind; serves p-sweeps and reruns).
        * ``None``/``False`` — off.
        * a ``str`` — explicit cache-file path, used verbatim.
        * ``True`` — automatic path (V3-T3 R2): ``twirl_cache_dir()`` (i.e.
          ``$XTIM_TWIRL_CACHE_DIR``, default ``~/.cache/xtim/twirl/``) with the file
          keyed ``<deferred-signature-fnv64>-<group-token>.twpl``. The signature is
          structural (noise values excluded), so one file serves whole p-sweeps; the
          group token guards cross-circuit reuse and is re-verified inside the TWPL
          framing — stale/corrupt files load nothing and rebuild silently. The
          resolved path is reported in ``channel_report()["disk_path"]``.
    selfcheck:
        Oracle window: the first ``selfcheck`` fired shots also run the full sigma
        path and must match EXACTLY (a trip raises ``RuntimeError`` from ``sample``).
        The window runs ONCE per sampler lifetime — on the first ``sample()`` call
        after compilation — and never re-arms on later (seeded or unseeded) calls.
        It is a stream-neutral verification overlay: sampled bytes are identical
        with the window on or off. A one-line stderr diagnostic announces the
        window on that first call (silence with ``XTIM_QUIET=1``).
    skip_refused_observables:
        Emit all-zero columns for observables that have no exact record channel
        instead of raising at ``sample()`` time.

    Notes
    -----
    **IF-branch circuits (Task 5):** when the circuit text contains ``IF`` blocks,
    the returned sampler is a :class:`PartitionedTwirlSampler`.  Its ``sample()``
    method requires a ``decision_bits`` keyword argument (a ``(shots, total_bits_width)``
    uint8 array); it groups shots by driving-bit pattern, compiles and caches one
    concrete branch-free circuit per pattern, samples each group with the existing C++
    sampler, and scatters results back in original shot order.

    **Branch-free circuits:** the existing ``_xtim.TwirlSampler`` is returned
    unchanged — byte-identical to the pre-Task-5 behavior.

    The returned sampler's ``sample()`` method also accepts an optional ``input_pauli``
    keyword argument: a ``(shots, 2 * support)`` uint8 array encoding a per-shot
    Pauli frame (X-part then Z-part). When provided, each shot's detector/observable
    bits are sign-relabeled by the frame before being written to the output buffers
    (Task 4 / decoder-feedback). ``input_pauli=None`` (the default) is byte-identical
    to the frame-free path.

    Raises
    ------
    ValueError
        On parse failure, bare-state rejection, no deterministic detector channels,
        an out-of-class propagation table / unsupported noise channel, or an ANTI
        detector.
    """
    circuit_text = str(circuit_text)

    # Task 5: detect IF-driving bits.  If any exist, return the partition wrapper.
    # The _xtim.if_driving_bits call parses the circuit; for IF-free circuits (the
    # overwhelming majority) it is a cheap text scan.  We rely on the parse being
    # idempotent (same result as the TwirlSampler constructor's internal parse).
    try:
        driving_bits = _xtim.if_driving_bits(circuit_text)
    except (ValueError, RuntimeError):
        # Parse failed — let the TwirlSampler constructor surface the error.
        driving_bits = []

    if driving_bits:
        # IF-containing circuit: return the partition-aware wrapper.
        return PartitionedTwirlSampler(
            circuit_text, p_factor, disk_cache, selfcheck, skip_refused_observables)

    # No IF blocks — exact same path as before Task 5.
    auto_dir = ""
    if disk_cache is True:
        auto_dir = twirl_cache_dir()
        os.makedirs(auto_dir, exist_ok=True)
    sampler = _xtim.TwirlSampler(
        circuit_text,
        float(p_factor),
        "" if (disk_cache is None or disk_cache is False or disk_cache is True)
        else str(disk_cache),
        int(selfcheck),
        bool(skip_refused_observables),
        auto_dir,
    )
    # Task 2b.1: a circuit with declared DECISIONs returns a TwirlBatch (with .decisions) from
    # sample(); wrap it.  Decision-free circuits return the raw engine sampler (byte-identical
    # (dets, obs) 2-tuple, all existing consumers unchanged).
    if sampler.num_decisions > 0:
        return _DecisionTwirlSampler(sampler)
    return sampler


# ── S2.1: compile_from_state — build a twirl sampler from a provided collapsed state ──────

def _bare_state_of(circuit_text: str) -> "_xtim.FramedSuperposition":
    """Build and return the :class:`_xtim.FramedSuperposition` bare state of ``circuit_text``.

    Runs the same normalization + ``build_bare_state`` path as
    :func:`compile_twirl_sampler`.  The returned handle is opaque; pass it to
    :func:`compile_twirl_sampler_from_state` to build a sampler whose bare state is
    substituted (rather than rebuilt from the circuit).

    Raises
    ------
    ValueError
        On parse failure or bare-state rejection.
    """
    return _xtim._bare_state_of(str(circuit_text))


#: Public alias for :func:`_bare_state_of` (same implementation, no validation added).
#: The underscore name is kept as a deprecated alias for one release.
bare_state_of = _bare_state_of


class _InputState:
    """A carried quantum state + the wire indices at which its carried qubits live.

    Produced by :func:`_input_state_of`.  ``state`` is the frame-explicit
    :class:`_xtim.FramedSuperposition` (the carried qubits are physically present in
    the frame — the prep's Clifford is baked because there is no trailing measurement
    on them); ``carried_wires`` are the deferred-space wire indices (``final_wire``
    coordinates) of the declared ``OUTPUT_QUBITS``, in declaration order.  The
    ⊗-composition (:func:`compile_twirl_sampler_from_state`) reads both to splice the
    carried state onto a consumer stage's ``INPUT_QUBITS`` port.
    """

    __slots__ = ("state", "carried_wires")

    def __init__(self, state, carried_wires):
        self.state = state
        self.carried_wires = list(carried_wires)


def _input_state_of(circuit_text: str) -> "_InputState":
    """Build the physical carried state of ``circuit_text`` + its carried-wire map.

    Same normalization + ``build_bare_state`` path as :func:`_bare_state_of`, but the
    returned :class:`_InputState` also records the ``OUTPUT_QUBITS`` wire positions.
    Declare the carried qubits with ``OUTPUT_QUBITS`` and give them NO trailing
    measurement so their prep Clifford stays frame-baked (e.g. ``"R 0\\nH 0\\n
    OUTPUT_QUBITS out 0\\n"`` carries a physical ``|+>``).

    Also validates the port contract: the declared ``OUTPUT_QUBITS`` wires must form a
    separable, stabilizer-backed partition (``rank(S ∩ P_B) == |B|``).  If not, raises
    ``ValueError`` with the witness and a note about the post-collapse escape when the
    port is only separable after measurement collapse.

    Raises
    ------
    ValueError
        On parse failure, bare-state rejection, or a failed port-contract check.
    """
    # Port-contract validation is performed in the C++ binding (py_input_state_of):
    # it raises ValueError with witness + post-collapse escape note when product=False.
    # Both the public and underscore C++ names share the same validated implementation.
    st, wires = _xtim._input_state_of(str(circuit_text))
    return _InputState(st, wires)


#: Public alias for :func:`_input_state_of` (same implementation, same validation).
#: The underscore name is kept as a deprecated alias for one release.
input_state_of = _input_state_of


# Single-qubit stabilizer state -> a Clifford prep gate list from |0> (Task 2b.2).
#   axis/sign          state    gates (applied left-to-right to |0>)
#   Z=+1               |0>      []
#   Z=-1               |1>      X
#   X=+1               |+>      H
#   X=-1               |->      X, H
#   Y=+1               |+i>     H, S
#   Y=-1               |-i>     H, Sdg
def _single_qubit_prep_gates(state, wire, tol=1e-6):
    x = state.pauli_expectation_x(wire)
    y = state.pauli_expectation_y(wire)
    z = state.pauli_expectation_z(wire)
    axes = (("Z", z), ("X", x), ("Y", y))
    dom = max(axes, key=lambda a: abs(a[1]))
    # A carried qubit is prep-splice-able iff it is a single-qubit STABILIZER state:
    # |<P>| == 1 on one axis and 0 on the others. Failing that is TWO distinct
    # conditions, and the message must not conflate them (it used to call a magic
    # state "not pure", which is false -- T|+> is pure, just not axis-aligned):
    #
    #   r^2 = <X>^2+<Y>^2+<Z>^2 < 1  =>  the marginal is MIXED, i.e. the port is
    #                                    ENTANGLED with the rest. Unsupported.
    #   r^2 == 1, no axis at |1|      =>  PURE but non-stabilizer, i.e. MAGIC. Legal,
    #                                    but belongs on the state-splice path.
    #
    # Both still raise ValueError (feedback.py routes 1-wide magic on that refusal),
    # so this only sharpens the diagnosis -- it does not change control flow.
    r2 = x * x + y * y + z * z
    if abs(abs(dom[1]) - 1.0) > tol:
        if r2 < 1.0 - tol:
            raise ValueError(
                f"input carried qubit at wire {wire} is not a pure single-qubit "
                f"stabilizer state: its marginal is MIXED (|Bloch|^2={r2:.4f} < 1, "
                f"<X>={x:.3f}, <Y>={y:.3f}, <Z>={z:.3f}), i.e. the port is entangled "
                "with the rest of the register. If the port is only separable AFTER "
                "measurement collapse, pass the post-collapse state "
                "(sample_barrier(...).materialize(i)); otherwise the protocol is out "
                "of scope. Diagnose with xtim._xtim.port_contract_of_state(state, wires).")

        raise ValueError(
            f"input carried qubit at wire {wire} is not a pure single-qubit stabilizer "
            f"state: it is PURE (|Bloch|^2={r2:.4f}) but not axis-aligned "
            f"(<X>={x:.3f}, <Y>={y:.3f}, <Z>={z:.3f}) -- i.e. a MAGIC state. Magic "
            "carried qubits cannot be re-synthesised as a Clifford prep; they belong "
            "on the state-splice path (compile_twirl_sampler_from_state).")
    for name, val in axes:
        if name != dom[0] and abs(val) > tol:
            raise ValueError(
                f"input carried qubit at wire {wire} has ambiguous stabilizer axis "
                f"(<X>={x:.3f}, <Y>={y:.3f}, <Z>={z:.3f})")
    axis, val = dom
    pos = val > 0
    if axis == "Z":
        return [] if pos else ["X"]
    if axis == "X":
        return ["H"] if pos else ["X", "H"]
    return ["H", "S"] if pos else ["H", "S_DAG"]   # axis == "Y"


def _is_single_qubit_stabilizer(state, wire, tol=1e-6) -> bool:
    """True iff the carried state's reduced 1-qubit density matrix at ``wire`` is a
    PURE single-qubit stabilizer state (|<P>|==1 on exactly one Pauli axis).

    This is the classifier that decides F2 routing: a 1-wide STABILIZER port takes the
    fast single-qubit prep-splice; a 1-wide MAGIC port (e.g. T|+>, |<X>|=|<Y>|=1/sqrt2)
    is NOT a stabilizer state and must route to state-level injection instead (the splice
    path's ``_single_qubit_prep_gates`` loud-refuses it — that refusal is correct for the
    splice, but the port still has a native carry route via injection).
    """
    x = state.pauli_expectation_x(wire)
    y = state.pauli_expectation_y(wire)
    z = state.pauli_expectation_z(wire)
    axes = ((abs(z)), (abs(x)), (abs(y)))
    dom = max(axes)
    if abs(dom - 1.0) > tol:
        return False                       # mixed/entangled/magic — no dominant ±1 axis
    # exactly one axis at |1|, the others ~0
    return sum(1 for a in axes if a > tol) == 1


def _compose_input_stage_text(inp: "_InputState", stage_text: str, q_in) -> str:
    """Splice the carried state (as a Clifford prep) onto the stage's INPUT_QUBITS port.

    The composed circuit = ``R Q_in`` + per-qubit prep gates (preparing the carried
    single-qubit stabilizer states) + the stage body with its ``INPUT_QUBITS`` line(s)
    removed.  Compiling this normally realizes exactly
    ``(provided state on Q_in) ⊗ (compressed fresh block)``: non-Clifford front-pushed
    onto a ``Q_in`` qubit now cancels against the spliced prep (deferred cancellation,
    spec §6.1), fresh qubits compile as today, and the whole thing runs through the
    single unmodified ``build_bare_state`` path.
    """
    carried = inp.carried_wires
    if len(carried) != len(q_in):
        raise ValueError(
            f"input state carries {len(carried)} qubit(s) but the stage's INPUT_QUBITS "
            f"port declares {len(q_in)} (ports {q_in}); a Level-2 interface mismatch")
    # Critical-1 fix (final review): the carried single-qubit stabilizer state is prepped from
    # the port qubit's DEFAULT |0> by the prep gates alone — NO explicit `R` reset.  A reset
    # would wipe the per-shot `input_pauli` frame, which the engine applies at t=0 (the input
    # wire); the reset severs the port qubit from that frame and the decoder's correction would
    # silently vanish (verified: `R q` before the port ⇒ frame is a no-op).  Dropping the reset
    # leaves the port qubit a genuine frame-carrying input in |0>, on which the prep gates build
    # the carried state, so the sign-relabel (Sub-plan-1 Task-4 mechanism, spec §5) FIRES.
    prep_lines = []
    for j, q in enumerate(q_in):
        for g in _single_qubit_prep_gates(inp.state, carried[j]):
            prep_lines.append(f"{g} {q}")
    # RETAIN the INPUT_QUBITS declaration(s), moved to sit AFTER the spliced prep, so the
    # compiled sampler's `input_port_qubits_` is populated.  The declaration is pure metadata
    # (it injects NO gates into the stream), so the bare-state compilation is unchanged.  The
    # frame varies per shot WITHIN a group, so it cannot be folded into the prep — the relabel
    # is the only compile-once-compatible mechanism.  For a non-Pauli prep (carried X/Y-axis
    # state) the standard-basis correction must be conjugated through the prep before it reaches
    # this port; run_protocol does that (``_conjugate_frame_through_prep``).
    input_qubits_lines, body_lines = [], []
    for ln in stage_text.splitlines():
        if ln.strip().split()[:1] == ["INPUT_QUBITS"]:
            input_qubits_lines.append(ln)
        else:
            body_lines.append(ln)
    return "\n".join(prep_lines + input_qubits_lines + body_lines) + "\n"


def compile_twirl_sampler_from_state(
    state,
    circuit_text: str,
    p_factor: float = 1.0,
    disk_cache: "str | bool | None" = None,
    selfcheck: int = 2000,
    skip_refused_observables: bool = False,
    inject: "bool | None" = None,
):
    """Build a twirl sampler from a PROVIDED ``FramedSuperposition`` bare state.

    Identical to :func:`compile_twirl_sampler` except the bare state is taken from
    ``state`` verbatim instead of being rebuilt from ``circuit_text``.  This is the
    seam that allows stage-chaining: collapse the state after stage-1 decoding, then
    pass the post-collapse bare into stage-2's sampler without re-running the full
    build pipeline.

    **Oracle:** ``compile_twirl_sampler_from_state(_bare_state_of(C), C)`` is
    byte-identical to ``compile_twirl_sampler(C)`` at the same seed.

    Parameters
    ----------
    state:
        A :class:`_xtim.FramedSuperposition` handle, typically from
        :func:`_bare_state_of`.
    circuit_text:
        The deferred-noise circuit (same text as you would pass to
        ``compile_twirl_sampler``).
    p_factor, disk_cache, selfcheck, skip_refused_observables:
        Same semantics as :func:`compile_twirl_sampler`.

    Returns
    -------
    _xtim.TwirlSampler
        A compiled twirl sampler whose bare state is ``state``.

    Raises
    ------
    ValueError
        On parse failure, bare-state rejection, no deterministic detector channels,
        an out-of-class propagation table / unsupported noise channel, or an ANTI
        detector.
    """
    circuit_text = str(circuit_text)

    # ── Task 2b.2: ⊗-composition when the stage declares an INPUT_QUBITS port ──────────────
    # module bare = (provided carried state on Q_in) ⊗ (compressed fresh block).  We realize
    # it by splicing the carried state (as a Clifford prep) in place of the port and compiling
    # the combined circuit through the ONE unmodified build_bare_state path — so scoped/deferred
    # cancellation (non-Clifford front-pushed onto a Q_in qubit) happens against the provided
    # state automatically (spec §6.1).
    q_in = _xtim.input_qubits(circuit_text)
    if q_in:
        if not isinstance(state, _InputState):
            raise TypeError(
                "compile_twirl_sampler_from_state: the stage declares an INPUT_QUBITS port; "
                "pass the carried state as an _InputState from xtim._input_state_of(...) so "
                "the composition knows where the carried qubits live (their final_wire coords)")
        # Routing (spec §1): port size 1 -> the single-qubit prep-splice fast path (byte-stable,
        # keeps scoped/deferred cancellation); port size >1 -> STATE-LEVEL INJECTION (carry the
        # whole entangled / magic patch verbatim). `inject=True` forces injection for a size-1
        # port too (needed for single-qubit MAGIC, which the splice fast path loud-refuses).
        use_inject = (len(q_in) > 1) if inject is None else bool(inject)
        if use_inject:
            auto_dir = ""
            if disk_cache is True:
                auto_dir = twirl_cache_dir()
                os.makedirs(auto_dir, exist_ok=True)
            sampler = _xtim.TwirlSampler(
                state.state, list(state.carried_wires), circuit_text,
                float(p_factor),
                "" if (disk_cache is None or disk_cache is False or disk_cache is True)
                else str(disk_cache),
                int(selfcheck), bool(skip_refused_observables), auto_dir)
            if sampler.num_decisions > 0:
                return _DecisionTwirlSampler(sampler)
            return sampler
        combined = _compose_input_stage_text(state, circuit_text, q_in)
        return compile_twirl_sampler(
            combined, p_factor=p_factor, disk_cache=disk_cache,
            selfcheck=selfcheck, skip_refused_observables=skip_refused_observables)

    # ── No INPUT_QUBITS port: S2.1 verbatim substitution (byte-identical oracle) ───────────
    if isinstance(state, _InputState):
        state = state.state
    auto_dir = ""
    if disk_cache is True:
        auto_dir = twirl_cache_dir()
        os.makedirs(auto_dir, exist_ok=True)
    return _xtim.TwirlSampler(
        state,
        circuit_text,
        float(p_factor),
        "" if (disk_cache is None or disk_cache is False or disk_cache is True)
        else str(disk_cache),
        int(selfcheck),
        bool(skip_refused_observables),
        auto_dir,
    )


# ── S2.2: _project_bare_onto_syndrome — independent oracle for post-barrier states ─────────

def _project_bare_onto_syndrome(circuit_text: str, sigmas) -> list:
    """Independent post-barrier oracle for :class:`_xtim.FramedSuperposition` objects.

    For each shot's recorded syndrome ``σ`` (a list of 0/1 bits over the bare state's
    certified generators, as returned by :meth:`_xtim.BarrierBuffer.sigma`), reconstruct
    the collapsed post-barrier state by applying the frame's DESTABILISERS
    ``{ d_i : σ_i = 1 }`` to a FRESH ``build_bare_state`` bare — the unique ``k=0`` state
    with syndrome ``σ``.

    This is genuinely independent of :func:`twirl_collapse`: it uses the general
    ``FramedSuperposition`` Clifford-apply engine on the frame's destabilisers, NOT the
    ``DiagNormalForm`` residual the sampler applies to build ``amps``. Raises
    ``ValueError`` when the bare state has ``k>0`` logical DOF (σ then does not determine
    the state).

    Use this to verify :meth:`_xtim.TwirlSampler.sample_barrier` correctness::

        buf = s.sample_barrier(N, seed=k)
        ref = xtim._project_bare_onto_syndrome(text, [buf.sigma(i) for i in range(N)])
        for i in range(N):
            assert buf.state(i).approx_equal(ref[i], tol=1e-12)

    Parameters
    ----------
    circuit_text : str
        The Stim-superset circuit text.
    sigmas : list[list[int]]
        Per-shot syndrome bit lists (each ``buf.sigma(i)``).

    Returns
    -------
    list[:class:`_xtim.FramedSuperposition`]
    """
    return _xtim._project_bare_onto_syndrome(
        str(circuit_text), [list(int(b) for b in s) for s in sigmas])
