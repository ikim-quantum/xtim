"""xtim.port — the declared-consumption segment port.

A consumer DECLARES what it will read; the port compiles once, routes to the
cheapest engine surface that serves the declaration, and computes nothing
undeclared.  Section references below ("spec §1/§2") are to the port's design
note, which lives with the consumer, not in this package.

## Two nouns, two verbs

    consume = port.Consume(dets=True, decisions=True, groups=True, residual_law=True)
    seg     = port.compile(circuit_text, consume)   # cached per (text, consume, plan_cache)
    result  = seg.run(shots=10_000, seed=42)

    # per-shot (declared) outputs
    result.dets              # bool[shots, n_det]        (consume.dets)
    result.decisions         # bool[shots, n_dec]        (consume.decisions)

    # group-indexed record surface (consume.groups); xtim-engine-13 (J-b1 F): layout-3 records, sigma retired
    result.group_id          # int32[shots]              per-shot group index
    result.n_groups          # int
    result.group_keys        # tuple[bytes] × n_groups   layout-3 record keys (BarrierBuffer.record_groups()[0])
    result.group_first_shot  # int32[n_groups]           representative (first-occurrence) shot
    result.group_dets        # bool[n_groups, n_det]     representative dets row   (needs dets)
    result.group_decisions   # bool[n_groups, n_dec]     representative decisions  (needs decisions)
    result.group_outcomes    # tuple[bytes] × n_groups   outcome record u ‖ z per group (lazy, once)
    result.group_plan_content # per group (kind, content, 128-bit digest), kind 0 and kind 2 alike
                             #   (renamed from group_plans; group_exact / group_plans_or_none removed)
    result.group_plan(g)     # dict {kind, content, digest} (+ {a, cz} for kind 0)
    result.group_prefix      # uint64[n_groups, 2*PNW]   each group's P_rec (X words, then Z words)
    result.group_signs(ps)   # int8[n_groups, len(ps)]   signs of each group's first shot
    result.signs(ps)         # int8[shots, len(ps)]      BarrierBuffer.signs (+1/-1 definite, 0 indefinite)
    # removed at xtim-engine-13 (AttributeError): group_sigmas, sig_bits, group_plans, group_exact,
    #   group_plans_or_none, frame_dressing

    # plan-structure read model (consume.residual_law) — Segment-level, pure functions of the
    # compiled circuit (computed once at compile):
    seg.born_dec             # dict: Born-measured DECISION operators {nborn, n, dec_index,
                             #   xz, phase, inv} (declaration order)
    seg.plan_structure(pk)   # dict per plan content (memoized): the record's plan (no sigma fields;
                             #   prefix_words, kind, rho, kappa_rec, record_layout, ...)
    seg.record_layout        # the layout-3 record layout
    seg.output_wires         # tuple[int] — deferred-space OUTPUT_QUBITS wires
    # removed at xtim-engine-13 (AttributeError): seg.sig_bits, seg.gw, seg.sigma_law,
    #   seg.frame_dressing_report

This is the ONE read model: a consumer never touches raw BarrierBuffer methods
(they remain on the BarrierBuffer surface).

## Routing (spec §2 — cheapest surface serving the declaration)

    dets/decisions-only          →  bare TwirlSampler.sample(shots, seed) path
    groups and/or residual_law   →  record surface (TwirlSampler.sample_barrier, layout-3
                                    record store — a port internal; group keys = the buffer's
                                    record_groups() keys)
    nothing declared             →  ValueError (loud, unconditional)

NOTE (byte streams): the bare and record surfaces are different byte streams of the
same distribution at a fixed seed (fast-path coin reservoir vs per-coin draws — an
engine invariant, see sk2 profile §0). Each surface is separately deterministic;
declaring groups/residual_law therefore selects the record-surface stream — exactly
the stream the old sample_records surface produces.

## Refusals (spec §2)

Every refusal names BOTH the circuit property AND the colliding declaration.
CH/PPR (non-diagonal) circuits refuse groups/residual_law — state retention is
diagonal-only; physically there is no Pauli residual to read — but serve
dets/decisions.  IF-driven circuits refuse every declaration (see below).

## Compact-record internal (spec §2 sanction)

The record route samples with ``sample_barrier(..., compact=True)``: the engine
stores per shot the COMPACT group identity (interned dense plan id + residual-prefix
support words + σ words + coins) instead of serializing ~600 B of plan bytes per
shot, and materializes the legacy padded key bytes ONCE PER GROUP at read time.
CONTRACT (oracle-gated in tests/test_port_groups.py): the group partition, the
first-occurrence order, and every legacy key byte are IDENTICAL to the historical
``record_groups()`` output. Old callers (compact=False, the default) are
byte-for-byte and cost-for-cost untouched.

## Plan disk cache — ``compile(..., plan_cache=…)`` (spec §1: a compile option)

    plan_cache=None (default)  — off (no persistence; in-memory plan cache only).
    plan_cache=True            — automatic path: ``$XTIM_TWIRL_CACHE_DIR`` or
                                 ``~/.cache/xtim/twirl/``, one ``.twpl`` file per
                                 (structural circuit signature, certified-group token).
    plan_cache="<path>"        — explicit cache-file path, used verbatim.

Growth policy (documented per plan Task 2):
  * LOCATION — as above; the resolved path is a property of the compiled circuit
    (noise values excluded), so ONE file serves whole p-sweeps and re-runs.
  * SIZE — the file stores the engine's interned residual plans; it grows with the
    circuit's plan space (loaded at compile, re-saved after each run that added
    plans). cultivation-d5 class: ~10⁵ plans after ~10⁶ shots, still growing a few
    percent per 200k-shot call (the plan space does not saturate — sk2 profile).
  * EVICTION — none. The cache is append-only per file; stale/corrupt/skew files
    are REJECTED as a load and silently rebuilt/overwritten, never served. The
    TWPL loader rejects on seven checks, each covered by tests/test_twpl_cache_guards.py
    (mutation-verified — removing any one turns its test red): bad magic, bad
    payload checksum, version skew, another ENGINE_VERSION (the header carries it,
    S0(b)), wrong certified-group token, a short read inside a plan record, and
    trailing bytes after the last record.
    PRECISION (corrected 2026-08-30 — this bullet previously said such files
    "load NOTHING", which is not true of one branch): the first five checks all
    run BEFORE any plan is parsed, so a stale, skewed, wrong-group or corrupt
    file genuinely contributes nothing. The short-read branch instead stops
    mid-loop without rolling back, so plans parsed before it stay in the
    in-memory cache — but those have already cleared the checksum AND the group
    token, i.e. they are genuine plans of the current version and group, never
    stale or foreign ones. Reclaiming space = deleting files; the auto directory
    is safe to wipe at any time (pure cache).
  * BYTES — on the RECORD route (groups/residual_law) a pre-warmed plan cache does
    NOT move sampled bytes (retention forces the per-shot slow path regardless of
    warmth). On the BARE route, pre-warmed plans can flip shots from the slow to the
    fast collapse-RNG discipline across PROCESSES at a fixed seed (documented engine
    caveat) — same distribution, different bytes. Within one process, repeated
    same-seed runs of one Segment are byte-stable on both routes.

## In-memory compile-cache growth policy (T1 review Minor, documented)

``compile()`` memoizes Segments in an UNBOUNDED ``functools.lru_cache`` keyed on
(circuit_text, consume, plan_cache). Growth is bounded by the number of DISTINCT
(circuit, declaration, cache-option) triples compiled in the process — for the
port's consumer class (a fixed set of gate-class circuits compiled once, spec §1
compile-once contract) this is small and intentional; an eviction policy would
reintroduce the per-call rebuild class of regressions the port structurally
excludes. Escape hatch (tests/long-lived exotic processes):
``port._cached_build.cache_clear()``.

## THE SEAM CONTRACT — only classical data crosses a seam (ratified)

The port contract, ratified by the frame-forwarded arc in its strongest form:
**only classical data ever crosses a segment seam — the engine holds no state
between calls.**  Each ``Segment.run`` is a stateless draw; everything that
connects one consumer segment to the next is per-shot CLASSICAL data held and
applied BY THE CONSUMER as record post-processing, never an engine injection:

  * the per-shot **Pauli frame** on the seam's boundary data qubits (consumer
    records are frame-corrected by anticommutation — a bit relabel — not
    resampled);
  * the **boundary partial-syndrome channel** for non-CSS seams;
  * the **charge-matched physical advance** of the frame across each gate and
    the **terminal syndrome XOR** closing the accumulated syndrome for a
    decoded terminal;
  * Born-branch / driven bits, consumed as classical relabels.

Consequently the port carries NO frame-injection or frame-extraction surface:
``run(shots, seed)`` is the whole entry point and ``Result`` carries only the
declared record outputs.  (The dead ``frame_in`` parameter — a loud
NotImplementedError — and the always-``None`` ``Result.frame_out`` were
REMOVED at 3.0; a live ``frame_in`` would invite exactly the engine-held-state
coupling the contract forbids.)  A caller that genuinely needs an engine-side
input Pauli uses the raw sampler's ``input_pauli`` surface on ``xtim.twirl``,
outside the port.

## IF-driven circuits are refused (3.0)

A circuit whose text carries ``IF``-branch blocks (``INPUT_BITS`` + ``IF x[k]``)
is refused at ``compile`` time: the per-shot driven-bits row-gather surface
(``run(..., input_bits=…)``) was removed at 3.0 for want of consumers.  Resolve
the branches yourself (``xtim._xtim.resolve_branches_text``) and compile each
resolved text into its own Segment, or use the raw
``xtim.twirl.PartitionedTwirlSampler`` surface (``sample(..., decision_bits=…)``),
which is unchanged.

## Immutability

``Consume`` is frozen; ``Segment`` and ``Result`` are ``__slots__`` objects whose
``__setattr__``/``__delattr__`` raise. Lazily-built read-model members are cached
via ``object.__setattr__`` (identity-stable; observable state never mutates).
"""
from __future__ import annotations

import dataclasses
import functools
import numpy as np
from typing import Optional

from . import _xtim
from . import twirl as _twirl
from ._record_groups import RecordGroups

# Record-key byte layout this module parses (engine: _xtim.RECORD_KEY_LAYOUT). Layout 3 (J-b1, design §4):
# [kind u8 (0 diagonal plan, 2 term-route plan)] ‖ [len u32-LE] ‖ content ‖ P_rec words ‖ y words — sigma retired.
_RECORD_KEY_LAYOUT = 3


# ── Consume ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Consume:
    """Declares exactly what the consumer will read from a Segment.

    Nothing undeclared is computed, stored, serialized, or surfaced
    (spec §1). All fields default to False so only requested outputs
    are produced.

    Fields
    ------
    dets        : include detector-event arrays (bool[shots, n_det]) in Result.
    decisions   : include DECISION-bit arrays (bool[shots, n_dec]) in Result.
    groups      : per-shot ``group_id`` + per-GROUP record arrays (layout-3 keys, plan contents,
                  outcomes, P_rec rows, representative rows) in Result — the record surface.
    residual_law: the plan-structure read model on the Segment (``born_dec``, ``plan_structure``,
                  ``output_wires``, ``record_layout``) — the record surface.

    J-b1 (xtim-engine-13): ``frame_dressing`` is removed (TypeError) — a supplied input frame is folded
    into the record's P_rec by the frames class (``BarrierBuffer.frame_folded``); there is no static-relabel
    dressing to export.
    """
    dets: bool = False
    decisions: bool = False
    groups: bool = False
    residual_law: bool = False

    def __post_init__(self) -> None:
        # Type-coerce booleans to make the hash stable even if the caller
        # passes truthy ints (dataclass frozen fields can't be reassigned,
        # so use object.__setattr__ for the coercion).
        object.__setattr__(self, "dets", bool(self.dets))
        object.__setattr__(self, "decisions", bool(self.decisions))
        object.__setattr__(self, "groups", bool(self.groups))
        object.__setattr__(self, "residual_law", bool(self.residual_law))


# ── Result ────────────────────────────────────────────────────────────────────

class Result:
    """Immutable container for one Segment.run() call.

    Only the fields declared in the corresponding Consume are present.
    Accessing an undeclared field raises AttributeError.

    Attributes (present only when declared in Consume)
    ----------
    dets        : bool ndarray (shots, n_det)                       [dets]
    decisions   : bool ndarray (shots, n_dec)                       [decisions]
    group_id    : int32 ndarray (shots,)                            [groups]
    n_groups    : int                                               [groups]
    group_keys  : tuple[bytes] — the layout-3 record keys            [groups]
    group_first_shot : int64 ndarray (n_groups,)                    [groups]
    group_dets  : bool ndarray (n_groups, n_det)                    [groups+dets]
    group_decisions : bool ndarray (n_groups, n_dec)                [groups+decisions]
    group_outcomes : tuple[uint8 ndarray] — u ‖ z per group          [groups]
    group_plan_content : tuple[(kind, content, digest)] per group    [groups]
    group_plan(g) : dict {kind, content, digest[, a, cz]}            [groups]
    group_prefix : uint64 ndarray (n_groups, 2*PNW) — P_rec rows      [groups]
    group_signs(paulis), signs(paulis) : int8 sign tables            [groups]

    Removed at J-b1 (sigma retired; no exact class): group_sigmas, sig_bits, group_plans,
    group_plans_or_none, group_exact, frame_dressing (AttributeError).
    """

    __slots__ = (
        "_dets", "_decisions", "_has_dets", "_has_decisions", "_has_groups",
        "_buf", "_groups", "_group_dets", "_group_decisions",
    )

    def __init__(
        self,
        *,
        dets: Optional[np.ndarray],
        decisions: Optional[np.ndarray],
        has_dets: bool,
        has_decisions: bool,
        has_groups: bool = False,
        buffer=None,
    ) -> None:
        object.__setattr__(self, "_dets", dets)
        object.__setattr__(self, "_decisions", decisions)
        object.__setattr__(self, "_has_dets", has_dets)
        object.__setattr__(self, "_has_decisions", has_decisions)
        object.__setattr__(self, "_has_groups", has_groups)
        object.__setattr__(self, "_buf", buffer)
        groups = RecordGroups(buffer) if has_groups else None
        object.__setattr__(self, "_groups", groups)
        first = groups.group_first_shot if groups is not None else None
        object.__setattr__(self, "_group_dets", dets[first] if (groups is not None and dets is not None) else None)
        object.__setattr__(self, "_group_decisions",
                           decisions[first] if (groups is not None and decisions is not None) else None)

    # ── per-shot outputs ────────────────────────────────────────────────────

    @property
    def dets(self) -> np.ndarray:
        if not self._has_dets:
            raise AttributeError(
                "Result has no 'dets' — declare Consume(dets=True) to request it")
        return self._dets

    @property
    def decisions(self) -> np.ndarray:
        if not self._has_decisions:
            raise AttributeError(
                "Result has no 'decisions' — declare Consume(decisions=True) to request it")
        return self._decisions

    # ── group-indexed outputs (Consume.groups) ──────────────────────────────

    def _need_groups(self, name: str):
        if not self._has_groups:
            raise AttributeError(
                f"Result has no {name!r} — declare Consume(groups=True) to request "
                f"the group-indexed record surface")

    @property
    def group_id(self) -> np.ndarray:
        self._need_groups("group_id")
        return self._groups.group_id

    @property
    def n_groups(self) -> int:
        self._need_groups("n_groups")
        return int(self._groups.n_groups)

    @property
    def group_keys(self) -> tuple:
        self._need_groups("group_keys")
        return tuple(self._groups.group_keys)

    @property
    def group_first_shot(self) -> np.ndarray:
        self._need_groups("group_first_shot")
        return self._groups.group_first_shot

    @property
    def group_dets(self) -> np.ndarray:
        self._need_groups("group_dets")
        if not self._has_dets:
            raise AttributeError(
                "Result has no 'group_dets' — declare Consume(dets=True) as well")
        return self._group_dets

    @property
    def group_decisions(self) -> np.ndarray:
        self._need_groups("group_decisions")
        if not self._has_decisions:
            raise AttributeError(
                "Result has no 'group_decisions' — declare Consume(decisions=True) as well")
        return self._group_decisions

    @property
    def group_outcomes(self) -> tuple:
        """Each group's outcomes u ‖ z (the first shot's; identical across the group by construction)."""
        self._need_groups("group_outcomes")
        return tuple(self._groups.group_outcomes())

    @property
    def group_plan_content(self) -> tuple:
        """Per group (kind, content bytes, 128-bit digest): kind 0 (diagonal S^a·CZ plans) and kind 2 (term-route
        plans) alike (J-b1: renamed from group_plans)."""
        self._need_groups("group_plan_content")
        return tuple(self._groups.group_plan_content())

    def group_plan(self, g: int) -> dict:
        """Group ``g``'s plan {kind, content, digest} and, for kind 0, its S^a·CZ view {a, cz}."""
        self._need_groups("group_plan")
        return self._groups.group_plan(int(g))

    @property
    def group_prefix(self) -> np.ndarray:
        """uint64[n_groups, 2*PNW]: each group's P_rec (X words then Z words) from its layout-3 key."""
        self._need_groups("group_prefix")
        return self._groups.group_prefix()

    def group_signs(self, paulis) -> np.ndarray:
        """int8[n_groups, len(paulis)]: signs() of each group's first shot (+1 / -1 definite, 0 indefinite or not
        certified)."""
        self._need_groups("group_signs")
        return self._groups.group_signs(paulis)

    def signs(self, paulis) -> np.ndarray:
        """int8[shots, len(paulis)]: BarrierBuffer.signs (+1 / -1 definite, 0 indefinite or not certified)."""
        self._need_groups("signs")
        return np.asarray(self._buf.signs(list(paulis)))

    def __setattr__(self, name, value):
        raise AttributeError(
            f"Result is immutable — cannot set attribute {name!r}")

    def __delattr__(self, name):
        raise AttributeError(
            f"Result is immutable — cannot delete attribute {name!r}")

    def __repr__(self) -> str:
        parts = []
        if self._has_dets and self._dets is not None:
            parts.append(f"dets={self._dets.shape}")
        if self._has_decisions and self._decisions is not None:
            parts.append(f"decisions={self._decisions.shape}")
        if self._has_groups:
            parts.append(f"n_groups={int(self._groups.n_groups)}")
        return f"xtim.port.Result({', '.join(parts)})"


# ── Segment ───────────────────────────────────────────────────────────────────

class Segment:
    """A compiled, immutable segment: one circuit + one Consume declaration.

    Built once via ``port.compile()``; cached per (text, consume, plan_cache).
    Immutable except the seed stream (which is always reset in-place
    at each ``run()`` call — no mutable visible state on the Segment itself).

    ``run(shots, seed)`` is the sole entry point.

    When ``consume.residual_law`` is declared the Segment additionally carries the
    plan-structure read model (``born_dec``, ``plan_structure``, ``output_wires``,
    ``record_layout``) — pure functions of the compiled circuit. Nothing undeclared is
    computed (spec §1). J-b1: ``sigma_law``, ``sig_bits``, ``gw`` and ``frame_dressing_report``
    are removed (sigma retired; AttributeError).
    """

    __slots__ = (
        "_text",
        "_consume",
        "_sampler",     # the bare _xtim.TwirlSampler (or wrapper)
        "_n_det",
        "_n_dec",
        "_route",       # "bare" | "record"
        "_born_dec",    # dict | None (residual_law only)
        "_out_wires",   # tuple[int] | None (residual_law only)
        "_ps_memo",     # dict: plan-content bytes -> plan_structure dict (residual_law only)
    )

    def __init__(
        self,
        text: str,
        consume: Consume,
        sampler,
        n_det: int,
        n_dec: int,
        route: str,
        born_dec=None,
        out_wires=None,
    ) -> None:
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_consume", consume)
        object.__setattr__(self, "_sampler", sampler)
        object.__setattr__(self, "_n_det", n_det)
        object.__setattr__(self, "_n_dec", n_dec)
        object.__setattr__(self, "_route", route)
        object.__setattr__(self, "_born_dec", born_dec)
        object.__setattr__(self, "_out_wires", out_wires)
        object.__setattr__(self, "_ps_memo", {} if consume.residual_law else None)

    def __setattr__(self, name, value):
        raise AttributeError(
            f"Segment is immutable — cannot set attribute {name!r}")

    def __delattr__(self, name):
        raise AttributeError(
            f"Segment is immutable — cannot delete attribute {name!r}")

    def __repr__(self) -> str:
        return (f"xtim.port.Segment(n_det={self._n_det}, n_dec={self._n_dec}, "
                f"route={self._route!r}, consume={self._consume!r})")

    # ── plan-structure read model (Consume.residual_law) ────────────────────

    def _need_residual_law(self, name: str):
        if not self._consume.residual_law:
            raise AttributeError(
                f"Segment has no {name!r} — declare Consume(residual_law=True) to "
                f"request the plan-structure read model")

    @property
    def born_dec(self) -> dict:
        """The Born-measured DECISION operators (declaration order); a decision's raw outcome is a record of the
        shot's plan table."""
        self._need_residual_law("born_dec")
        return self._born_dec

    @property
    def output_wires(self) -> tuple:
        self._need_residual_law("output_wires")
        return self._out_wires

    def plan_structure(self, plan_content: bytes) -> dict:
        """plan_structure() (design §4 fields) of one layout-3 plan content ([kind][len] content, e.g.
        ``BarrierBuffer.plan_content(i)[0]``); memoized per distinct content."""
        self._need_residual_law("plan_structure")
        pk = bytes(plan_content)
        hit = self._ps_memo.get(pk)
        if hit is None:
            hit = self._sampler.plan_structure(pk)
            self._ps_memo[pk] = hit
        return hit

    def record_layout(self) -> dict:
        """The layout-3 record key and outcome order (``TwirlSampler.record_layout``)."""
        return self._sampler.record_layout()

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self, shots: int, seed: int) -> Result:
        """Sample ``shots`` records, seeded in-place (no reconstruction).

        Parameters
        ----------
        shots : int
            Number of shots to sample (>= 0).
        seed : int
            Engine seed in [0, 2**64). The sampler is reseeded IN PLACE — no
            rebuild. Same seed on the same Segment → byte-identical Result.

        Returns
        -------
        Result
            Contains only the declared outputs.

        Notes
        -----
        A run is a STATELESS draw: nothing crosses from one call to the next.
        See "THE SEAM CONTRACT" in the module docstring for why there is no
        frame-injection parameter.
        """
        shots = int(shots)
        if shots < 0:
            raise ValueError(f"shots must be >= 0, got {shots}")

        if self._route == "record":
            return self._run_record(shots, seed)
        return self._run_bare(shots, seed)

    # ── bare route ───────────────────────────────────────────────────────────

    def _run_bare(self, shots: int, seed: int) -> Result:
        consume = self._consume
        sampler = self._sampler

        # Route: bare TwirlSampler.sample(shots, seed, input_pauli=None).
        # The seed argument triggers in-place reseeding (set_seed path), no
        # reconstruction. This is the only seeding path on this surface.
        raw = sampler.sample(shots, seed, None)

        # Unpack: 2-tuple (dets_b8, obs_b8) when num_decisions == 0;
        #         3-tuple (dets_b8, obs_b8, dec_b8) when num_decisions > 0.
        if len(raw) == 3:
            dets_b8, _obs_b8, dec_b8 = raw
        else:
            dets_b8, _obs_b8 = raw
            dec_b8 = None

        # Only unpack what is declared.
        dets_arr = None
        if consume.dets:
            dets_arr = _unpack_b8(dets_b8, self._n_det, shots)

        dec_arr = None
        if consume.decisions:
            if dec_b8 is not None:
                dec_arr = _unpack_b8(dec_b8, self._n_dec, shots)
            else:
                # Circuit has no DECISION declarations → empty array
                dec_arr = np.zeros((shots, 0), dtype=bool)

        return Result(
            dets=dets_arr,
            decisions=dec_arr,
            has_dets=consume.dets,
            has_decisions=consume.decisions,
        )

    # ── record route ─────────────────────────────────────────────────────────

    def _run_record(self, shots: int, seed: int) -> Result:
        consume = self._consume
        sampler = self._sampler

        # ONE batched engine call; in-place reseed; the layout-3 record of every shot.
        buf = sampler.sample_barrier(shots, seed)
        if getattr(_xtim, "RECORD_KEY_LAYOUT", 1) != _RECORD_KEY_LAYOUT:
            raise RuntimeError(
                "xtim.port: engine record-key layout %r != the layout this reader parses (%d)"
                % (getattr(_xtim, "RECORD_KEY_LAYOUT", 1), _RECORD_KEY_LAYOUT))

        dets_arr = None
        if consume.dets:
            dets_arr = _unpack_b8(np.asarray(buf.dets(), np.uint8),
                                  self._n_det, shots)
        dec_arr = None
        if consume.decisions:
            dec_arr = _unpack_b8(np.asarray(buf.decisions(), np.uint8),
                                 self._n_dec, shots)

        return Result(
            dets=dets_arr,
            decisions=dec_arr,
            has_dets=consume.dets,
            has_decisions=consume.decisions,
            has_groups=consume.groups,
            buffer=buf if consume.groups else None,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _unpack_b8(b8: np.ndarray, ncols: int, shots: int) -> np.ndarray:
    """Stim LE b8-packed uint8 (shots, ceil(K/8)) → bool (shots, ncols)."""
    if ncols == 0:
        return np.zeros((shots, 0), dtype=bool)
    return np.unpackbits(b8, axis=1, bitorder="little")[:, :ncols].astype(bool)


def _fetch_read_model(sampler):
    """The Segment-level plan-structure read model (residual_law declarations only): (born_dec dict,
    OUTPUT_QUBITS wires). Module-level seam (monkeypatchable) so the undeclared-not-computed contract is
    spy-testable: a groups-only compile must NEVER call this."""
    wires = tuple(int(w) for w in sampler.output_wires())
    return sampler.born_dec(), wires


def _build_segment(text: str, consume: Consume, plan_cache) -> "Segment":
    """Build a Segment for (text, consume, plan_cache). Called once per distinct triple."""
    # Validate consume: nothing declared is an error (spec §1)
    if not (consume.dets or consume.decisions or consume.groups
            or consume.residual_law):
        raise ValueError(
            "Consume declares nothing — at least one of dets, decisions, "
            "groups, residual_law must be True")

    # selfcheck=0 per adaptq's hot-path contract (engine.py:382-404).
    # plan_cache forwards to the engine's TWPL disk layer (compile option, spec §1).
    tw_sampler = _twirl.compile_twirl_sampler(text, selfcheck=0, disk_cache=plan_cache)

    # ── IF-driven circuits: refused at 3.0 ──────────────────────────────────
    # compile_twirl_sampler returns a PartitionedTwirlSampler for IF-bearing
    # texts.  The port's per-shot driven-bits row-gather surface was removed
    # at 3.0 (no consumers); rather than silently serve the all-zeros resolved
    # branch, refuse loudly and name both sides (spec §2 refusal semantics).
    if isinstance(tw_sampler, _twirl.PartitionedTwirlSampler):
        raise RuntimeError(
            "circuit property: IF-branch blocks driven by external INPUT_BITS "
            "(a per-shot branch choice, not a single circuit); colliding "
            "declaration: any — xtim.port serves ONE resolved circuit per "
            "Segment (the driven input_bits row-gather surface was removed at "
            "3.0). Resolve the branches (xtim._xtim.resolve_branches_text) and "
            "compile each resolved text into its own Segment, or use "
            "xtim.twirl.PartitionedTwirlSampler.sample(decision_bits=...) "
            "directly.")

    sampler = tw_sampler._inner if hasattr(tw_sampler, "_inner") else tw_sampler

    n_det = int(sampler.num_detectors)
    n_dec = int(getattr(sampler, "num_decisions", 0))

    # Routing (spec §2): groups/residual_law → the record surface; otherwise bare. J-b1: CH/PPR circuits are
    # served on the record surface (kind-2 records); the former CH/PPR refusal is gone.
    route = "record" if (consume.groups or consume.residual_law) else "bare"

    born_dec = None
    out_wires = None
    if consume.residual_law:
        born_dec, out_wires = _fetch_read_model(sampler)

    return Segment(text, consume, sampler, n_det, n_dec, route, born_dec=born_dec, out_wires=out_wires)


# ── compile (cached) ──────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _cached_build(text: str, consume: Consume, plan_cache) -> "Segment":
    """Inner cached builder. Consume is frozen+hashable; plan_cache is
    None/True/str — all hashable cache-key components."""
    return _build_segment(text, consume, plan_cache)


def compile(circuit_text: str, consume: Consume, plan_cache=None) -> "Segment":
    """Compile a circuit for the declared consumption; cached per
    (text, consume, plan_cache).

    Parameters
    ----------
    circuit_text : str
        The xtim/Stim circuit text to compile.
    consume : Consume
        Declares what the caller will read. Nothing undeclared is computed.
    plan_cache : None | True | str
        Plan disk-cache compile option (spec §1). ``None`` (default) = off;
        ``True`` = automatic path under ``$XTIM_TWIRL_CACHE_DIR`` /
        ``~/.cache/xtim/twirl``; a ``str`` = explicit ``.twpl`` file path.
        Growth policy: see the module docstring ("Plan disk cache").

    Returns
    -------
    Segment
        An immutable compiled segment. Two calls with the same arguments return
        the SAME (identity-equal) Segment object (cache hit).

    Raises
    ------
    ValueError
        If consume declares nothing.
    xtim.errors.XtimTwirlRefusal
        If the circuit is outside the twirl class (by name, at compile).
    RuntimeError
        If the circuit property is incompatible with the declared consumption
        (IF-branch blocks; spec §2). The message names BOTH the circuit property and
        the colliding declaration.
    """
    if not isinstance(consume, Consume):
        raise TypeError(
            f"consume must be a port.Consume instance, got {type(consume).__name__}")
    if plan_cache is None or plan_cache is False:
        plan_cache = None
    elif plan_cache is not True:
        plan_cache = str(plan_cache)
    return _cached_build(str(circuit_text), consume, plan_cache)
