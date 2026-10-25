"""Twirl record sampler — xtim's default sampling engine, as a Python sampler.

``compile_twirl_sampler`` compiles a circuit into a :class:`_xtim.TwirlSampler`
(engine class ``qeccore::TwirlRecordSampler``): per shot, the sparse noise draw is
composed exactly into one Clifford residual, reduced to a content-keyed plan, and every
DETECTOR / OBSERVABLE / DECISION record is emitted as a row of the plan's record table.
``Circuit.compile_detector_sampler()`` returns this engine by default (``engine="twirl"``).

Semantics contract (xtim-engine-13; full text: docs/twirl_record_sampler.md):

* **One joint record law.** Every record — deterministic, gauge or ANTI — is sampled from
  the exact joint law after the shot's residual (5σ-gated against the exact engine on all
  benchmarks, including the CH-class ones). Gauge detectors are exact Born records; ANTI
  detectors / observables are served and reported as nondeterministic in
  ``record_report()``; no observable is refused.
* **No fallback engine.** Every shot is served by the law; there is no per-shot
  exact-engine route. What the engine cannot serve is refused BY NAME at compile with
  :class:`xtim.XtimTwirlRefusal` (``twirl_class``: OUTSIDE_T_DEPTH_1, NONCOMMUTING_MAGIC,
  LAW_LIMITATION_PROPAGATION); the non-Clifford scope rule is the static fixed-axis merge
  (docs/twirl_record_sampler.md, point 4). The exact engine runs only on explicit request
  (``compile_detector_sampler(engine="exact")``).
* **Reference-relative bits**, including INPUT_QUBITS / ``ideal_state`` port columns.
* The plan-cache ``disk_cache`` path is noise-blind: one cache file serves p-sweeps
  (TWPL v6; older files load nothing and rebuild silently).

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
import re
from collections import namedtuple

import numpy as np

from . import _xtim
from .errors import XtimFidelityRefusal

_OUTPUT_MEASURED = re.compile(
    r"^\s*line \d+: (OUTPUT_QUBITS '[^']*': qubit \d+ is measured in the circuit) \(output qubits must be unmeasured\)$")


def _construct(make):
    """Run an engine TwirlSampler constructor; a parse failure whose every error is a measured OUTPUT_QUBITS
    qubit is the register-fidelity refusal ``RF_OUTPUT_MEASURED`` (an :class:`XtimFidelityRefusal`)."""
    try:
        return make()
    except ValueError as e:
        lines = [l for l in str(e).splitlines()[1:] if l.strip()]
        hits = [_OUTPUT_MEASURED.match(l) for l in lines]
        if not lines or not all(hits):
            raise
        raise XtimFidelityRefusal(
            "RF_OUTPUT_MEASURED: " + "; ".join(h.group(1) for h in hits) + ". Output qubits must stay "
            "unmeasured: move the output state onto a fresh wire (noiseless R + SWAP) and measure the old one.",
            "RF_OUTPUT_MEASURED") from None

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


def _with_budget(sampler, plan_payload_budget):
    """S0(c) / D6: apply the caller's plan-PAYLOAD byte budget (``None`` = the engine default,
    512 MiB; ``math.inf`` = unbounded; ``0`` is refused by name by the engine). Output bytes never
    depend on it."""
    if plan_payload_budget is not None:
        sampler.set_plan_payload_budget(plan_payload_budget)
    return sampler


def _make_inner_sampler(circuit_text, p_factor, disk_cache, selfcheck,
                        decisions=True, plan_payload_budget=None):
    """Create a bare ``_xtim.TwirlSampler`` from resolved text (no disk_cache forwarding
    to inner per-pattern samplers — reserved for a future extension)."""
    return _with_budget(_xtim.TwirlSampler(
        str(circuit_text),
        float(p_factor),
        "",           # disk_cache not forwarded to per-pattern inner samplers
        int(selfcheck),
        "",           # disk_cache_auto_dir likewise
        bool(decisions),
    ), plan_payload_budget)


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

    def sample(self, shots, seed=None, input_pauli=None, input_branch=None):
        out = self._inner.sample(shots, seed, input_pauli, input_branch)
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

    Only ``sample(decision_bits=...)`` samples here. A call that samples WITHOUT bit
    values (``sample_barrier``) is refused by name: no driving pattern is correct on its
    own, because an ``IF`` condition always reads a bit of an ``INPUT_BITS`` port the
    caller supplies (never one of the circuit's own records). ``sampler_for_pattern(bits)``
    returns the inner sampler of one full bit vector for per-pattern ``sample_barrier``.
    Metadata (``num_*``, ``output_wires``, ``channel_report``, ...) is read from the
    all-zeros pattern's inner sampler.

    Not intended for direct construction; use :func:`compile_twirl_sampler`.
    """

    # Maximum number of IF-driving bits before we raise to avoid a combinatorial explosion.
    _MAX_DRIVING_BITS = 20

    def __init__(self, circuit_text: str, p_factor: float,
                 disk_cache, selfcheck: int,
                 decisions: bool = True, plan_payload_budget: "int | float | None" = None) -> None:
        self._text = str(circuit_text)
        self._p_factor = float(p_factor)
        self._disk_cache = disk_cache
        self._selfcheck = int(selfcheck)
        self._decisions = bool(decisions)
        self._plan_payload_budget = plan_payload_budget

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
                self._selfcheck, self._decisions, self._plan_payload_budget)
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

    # Inner-sampler calls that SAMPLE (consume the RNG and return per-shot records).  They are
    # refused by name here: the reference sampler is the all-zeros driving pattern, and no
    # driving pattern can be chosen without the caller's bits.  The IR gives no pattern that
    # is correct by itself: an ``IF`` condition reads ``decision[cond_port][cond_bit]`` of an
    # ``INPUT_BITS`` port (cpp/include/qeccore/circuit_ir.hpp, Instr::cond_port; the parser
    # refuses an IF on any other name), never one of the circuit's own records -- so every
    # IF body is a per-shot choice of the CALLER, and this wrapper only exists when at least
    # one IF drives a bit.  Serving pattern 0 silently treated every body as "not taken"
    # (dec-review (d): adaptq's choi.harness_port_ref on a Conditional fix-up).
    # Allow-list of inner-sampler attributes that are pure metadata of the compiled (all-zeros
    # pattern) circuit; every other name is denied by name (a future sampling method on the binding
    # can never delegate silently).
    _METADATA_DELEGATES = frozenset((
        "num_detectors", "num_observables", "num_decisions", "output_wires", "output_ports",
        "record_layout", "record_operators", "plan_structure", "born_dec", "channel_one_counts"))

    def _if_conditions(self) -> list:
        from .circuit import _unresolved_if_conditions
        return _unresolved_if_conditions(self._text)

    def _refuse_sampling_delegate(self, name: str):
        conds = ", ".join("IF " + c for c in self._if_conditions())
        raise ValueError(
            f"PartitionedTwirlSampler.{name}: the circuit has IF blocks on INPUT_BITS "
            f"({conds}; {len(self._driving_global_offsets)} driving bit(s)) and {name} takes "
            "no bit values -- it would sample the all-zeros driving pattern (every IF body "
            "not taken), which is not the circuit's law for any shot whose bits differ. "
            "Resolve per pattern instead: sampler_for_pattern(bits) gives the inner sampler "
            "of one full decision-bit vector (its sample_barrier is exact for shots with "
            "those bits), or call sample(shots, decision_bits=...) for grouped records")

    def sampler_for_pattern(self, decision_bits) -> "_xtim.TwirlSampler":
        """The (cached) inner ``_xtim.TwirlSampler`` of ONE full decision-bit vector
        (``total_bits_width`` entries, the same row layout as ``sample(decision_bits=...)``).
        Its ``sample``/``sample_barrier`` are exact for shots whose bits equal this vector;
        non-driving bits are ignored. Read pattern-dependent metadata (``output_wires()``,
        ``channel_report()``, ...) from the RETURNED sampler: a body with a Reset or measurement
        changes the deferred wire placement per pattern, and this wrapper's own metadata is the
        all-zeros pattern's."""
        row = np.asarray(decision_bits, dtype=np.uint8).reshape(-1)
        if row.shape[0] != self._total_width:
            raise ValueError(
                f"sampler_for_pattern: expected {self._total_width} bit(s) "
                f"(total_bits_width), got {row.shape[0]}")
        key = bytes(int(row[off]) for off in self._driving_global_offsets)
        return self._get_sampler(key)

    def sample_barrier(self, *args, **kwargs):
        self._refuse_sampling_delegate("sample_barrier")

    def __getattr__(self, name):
        # Delegate metadata of the compiled circuit (output_wires, channel_one_counts,
        # plan_structure, born_dec, sig_bits, ...) to the reference sampler; a sampling
        # call is refused by name above (never silently served from pattern 0).
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._METADATA_DELEGATES:
            return getattr(self._ref_sampler, name)
        if hasattr(self._ref_sampler, name):
            self._refuse_sampling_delegate(name)
        raise AttributeError(name)

    def record_report(self) -> dict:
        """J-b1 (plan §1.7, ruled): the reference (all-zeros pattern) sampler's static schema (``classes``,
        ``nondeterministic_*``, ``record_layout``), plus ``route_counts`` and ``counters`` SUMMED over the inner
        samplers built so far, plus ``patterns`` (their count). Per-pattern counters:
        ``sampler_for_pattern(bits).record_report()``."""
        ref = self._ref_sampler.record_report()
        out = {k: ref[k] for k in ("classes", "nondeterministic_detectors", "nondeterministic_observables")}
        out["record_layout"] = self._ref_sampler.record_layout()
        rc, ct = {}, {}
        for s in self._sampler_cache.values():
            r = s.record_report()
            for k, v in r["route_counts"].items():
                rc[k] = rc.get(k, 0) + int(v)
            for k, v in r["counters"].items():
                ct[k] = ct.get(k, 0) + int(v)
        out["route_counts"] = rc
        out["counters"] = ct
        out["patterns"] = len(self._sampler_cache)
        return out

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
            # Superset-width output buffers (exactly SDBB bytes: 0 for a detector-free circuit,
            # where the per-group packbits yields (n, 0) rows — same Minor-9 shape as below).
            dets_out = np.zeros((shots, SDBB), dtype=np.uint8)
            live_mask_out = np.zeros((shots, SDBB), dtype=np.uint8)
        else:
            # Task-5 byte-identical path: use ref-sampler detector count.  Width exactly
            # DBB (0 for a detector-free circuit): the engine returns (n, 0) packed rows
            # there, and the padded (shots, 1) buffer made the per-group scatter fail with
            # a shape mismatch (dec-review Minor-9); the returned slice was [:, :DBB] anyway.
            DBB = (self.num_detectors + 7) // 8
            dets_out = np.zeros((shots, DBB), dtype=np.uint8)

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
    decisions: bool = True,
    plan_payload_budget: "int | float | None" = None,
):
    """Compile ``circuit_text`` into a twirl record sampler.

    Usage::

        s = xtim.compile_twirl_sampler(text, p_factor=1.0)
        dets, obs = s.sample(shots, seed=1)   # Stim-b8-packed uint8 (shots, ceil(K/8))
        rep = s.record_report()

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
          framing — stale/corrupt files load nothing and rebuild silently.
    selfcheck:
        Oracle window: the first ``selfcheck`` fired shots are re-checked by the
        engine's in-process oracles (a trip raises ``RuntimeError`` from ``sample``).
        The window runs ONCE per sampler lifetime — on the first ``sample()`` call
        after compilation — and never re-arms on later (seeded or unseeded) calls.
        It is a stream-neutral verification overlay: sampled bytes are identical
        with the window on or off. One ``xtim.XtimVerificationWarning`` per process
        announces the first active window, raised after that call returns (filter it
        with ``warnings``). The per-compile channel-classification diagnostic
        (``[twirl_records] circuit channels: ...``) is a ``logging`` INFO record on the
        ``xtim.twirl`` logger (silent unless the application configures logging).
    decisions:
        ``True`` (default): the circuit's ``DECISION`` declarations are compiled as
        output channels — ``sample()`` returns a :class:`TwirlBatch` with ``.decisions``
        and a Born-class decision is measured per shot. ``False``: the engine is compiled
        WITHOUT them — exactly the engine of the same circuit with its ``DECISION``
        lines removed (``num_decisions == 0``, the ``(dets, obs)`` 2-tuple, byte-identical
        det/obs streams, no per-shot Born-decision path and none of its refusals). A
        ``DECISION`` is never consumed inside the engine (``IF`` conditions read
        ``INPUT_BITS`` port bits the caller supplies), so this is lossless for every
        consumer that reads detectors/observables only; ``compile_detector_sampler``
        uses it.
    plan_payload_budget:
        ``None`` (default): the engine default, 512 MiB; ``math.inf``: unbounded (the same
        meanings as ``set_plan_payload_budget`` and ``RecordFidelity(cache_budget=)``,
        and ``record_report()["plan_memory"]["budget"]`` reports them). A positive byte count bounds the
        resident plan PAYLOADS (law vectors, channel rows, observable data), the readers'
        tables and the term route's dispatch entries: they are evicted only between shots
        (dispatch entries at a new key) and rebuilt on demand, so every output byte is
        identical for every budget. Plan headers (the key, stored once, and the
        content id) are permanent. A shot's own payloads are exempt — any transient
        overshoot is reported in bytes (``record_report()["plan_memory"]``), never
        refused. ``0`` is refused by name.

    Notes
    -----
    **Thread safety:** one sampler object must not be used from two threads at once (its run
    counters, plan caches and pending diagnostics are unsynchronised; sampling releases the
    GIL). Separate samplers are independent.

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
            circuit_text, p_factor, disk_cache, selfcheck, decisions, plan_payload_budget)

    # No IF blocks — exact same path as before Task 5.
    auto_dir = ""
    if disk_cache is True:
        auto_dir = twirl_cache_dir()
        os.makedirs(auto_dir, exist_ok=True)
    sampler = _construct(lambda: _xtim.TwirlSampler(
        circuit_text,
        float(p_factor),
        "" if (disk_cache is None or disk_cache is False or disk_cache is True)
        else str(disk_cache),
        int(selfcheck),
        auto_dir,
        bool(decisions),
    ))
    # Task 2b.1: a circuit with declared DECISIONs returns a TwirlBatch (with .decisions) from
    # sample(); wrap it.  Decision-free circuits return the raw engine sampler (byte-identical
    # (dets, obs) 2-tuple, all existing consumers unchanged).
    _with_budget(sampler, plan_payload_budget)
    if sampler.num_decisions > 0:
        return _DecisionTwirlSampler(sampler)
    return sampler


# ── bare / input states (the carried-state providers) ──────────────────────────────────────

def _bare_state_of(circuit_text: str) -> "_xtim.FramedSuperposition":
    """Build and return the :class:`_xtim.FramedSuperposition` bare state of ``circuit_text``.

    Runs the same normalization + ``build_bare_state`` path as
    :func:`compile_twirl_sampler`.  The returned handle is opaque; pass it to
    :func:`compile_twirl_sampler_from_ideal_state` (via :func:`input_state_of`) to build a sampler whose bare state is
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
    ⊗-composition (:func:`compile_twirl_sampler_from_ideal_state`) reads both to splice the
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


def _ideal_state_arg(ideal_state):
    """An ``_xtim.IdealState`` from the accepted forms: an IdealState (a producer's ``ideal_state()``, with its
    provenance) or an :class:`_InputState` (``input_state_of(text)``: a rows-only ideal state, branch 0 only)."""
    if isinstance(ideal_state, _InputState):
        return _xtim.ideal_state_of_state(ideal_state.state, list(ideal_state.carried_wires))
    if type(ideal_state).__name__ == "IdealState":
        return ideal_state
    raise TypeError(
        "compile_twirl_sampler_from_ideal_state: ideal_state must be an xtim._xtim.IdealState (a producer "
        "sampler's ideal_state()) or an input_state_of(text) result; a bare FramedSuperposition carries no wire "
        "map (J-b1: the verbatim bare-state substitution of compile_twirl_sampler_from_state was removed)")


def compile_twirl_sampler_from_ideal_state(
    circuit_text: str,
    *,
    ideal_state,
    p_factor: float = 1.0,
    disk_cache: "str | bool | None" = None,
    selfcheck: int = 2000,
    decisions: bool = True,
    plan_payload_budget: "int | float | None" = None,
):
    """Build a twirl sampler for a consumer stage whose ``INPUT_QUBITS`` carry ``ideal_state``.

    J-b1 (replaces ``compile_twirl_sampler_from_state``): every carried state — entangled, magic or a single-qubit
    stabilizer state — is injected at the state level as BRANCH 0 of the consumer's branch table (the former
    single-qubit prep-splice and the ``inject=`` switch are gone). DETECTOR / OBSERVABLE columns are
    REFERENCE-RELATIVE: relative to the noiseless run from the ideal state (DECISION columns raw). A producer's other
    record outcomes reach the consumer as branches (``add_branch`` with the producer's ``branch_handles``) or as
    per-shot ``input_pauli`` frames (the producer's ``branches()``).

    Parameters
    ----------
    circuit_text:
        The consumer stage text; it declares the ``INPUT_QUBITS`` port the ideal state fills.
    ideal_state:
        An ``_xtim.IdealState`` (``producer.ideal_state()``, keeps the producer's provenance so its branches can be
        added) or an :func:`input_state_of` result (a rows-only ideal state: branch 0 only).
    p_factor, disk_cache, selfcheck, decisions, plan_payload_budget:
        Same semantics as :func:`compile_twirl_sampler`.

    Raises
    ------
    xtim.errors.XtimTwirlRefusal
        A carried law limitation (``LAW_LIMITATION_CARRIED_AXIS``) or a twirl-class refusal of the consumer.
    ValueError
        On parse failure, a port-size mismatch or an injection refusal.
    """
    circuit_text = str(circuit_text)
    from .circuit import _refuse_unresolved_if
    _refuse_unresolved_if(circuit_text, "compile_twirl_sampler_from_ideal_state")
    ideal = _ideal_state_arg(ideal_state)
    auto_dir = ""
    if disk_cache is True:
        auto_dir = twirl_cache_dir()
        os.makedirs(auto_dir, exist_ok=True)
    sampler = _construct(lambda: _xtim.TwirlSampler.from_ideal_state(
        ideal_state=ideal, circuit_text=circuit_text, p_factor=float(p_factor),
        disk_cache="" if (disk_cache is None or disk_cache is False or disk_cache is True) else str(disk_cache),
        selfcheck=int(selfcheck), disk_cache_auto_dir=auto_dir, decisions=bool(decisions)))
    _with_budget(sampler, plan_payload_budget)
    if sampler.num_decisions > 0:
        return _DecisionTwirlSampler(sampler)
    return sampler
