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

    # group-indexed record surface (consume.groups)
    result.group_id          # int32[shots]              per-shot group index
    result.n_groups          # int
    result.group_keys        # uint8[n_groups, W]        legacy padded record-key rows,
                             #   byte-identical to BarrierBuffer.record_groups()[2]
    result.group_first_shot  # int32[n_groups]           representative (first-occurrence) shot
    result.group_sigmas      # uint64[n_groups, gw]      σ words per group (record data)
    result.sig_bits          # int                       σ bit width (= #certified generators)
    result.group_dets        # bool[n_groups, n_det]     representative dets row   (needs dets)
    result.group_decisions   # bool[n_groups, n_dec]     representative decisions  (needs decisions)
    result.group_coins       # tuple[bytes] × n_groups   coin record per group (lazy, once)
    result.group_plans       # tuple[bytes] × n_groups   canonical residual plan bytes per group
                             #   (lazy, once; b"" = clean/fallback group)

    # σ-law / plan-structure read model (consume.residual_law) — Segment-level,
    # pure functions of the compiled circuit (computed once at compile):
    seg.sig_bits             # int   (also on Result when groups declared)
    seg.gw                   # ceil(sig_bits / 64) — σ word count
    seg.sigma_law            # dict: port_sigma_law of the bare state on the OUTPUT_QUBITS
                             #   wires {W, ngens, gw, ns, k_port, stab, comb, ref_signs,
                             #   logicals, ok} — sign_j(shot) = ref_signs[j] ^ parity(comb[j] & σ)
    seg.born_dec             # dict: Born-measured DECISION operators {nborn, n, dec_index,
                             #   xz, phase, inv} (declaration order)
    seg.plan_structure(pk)   # dict per plan-key bytes (memoized): residual normal form +
                             #   ShotLaw data {n, ngens, gw, r, kappa, fallback, prefix_xz, a,
                             #   cz, det_signs, coin_masks, kernel_masks, kernel_base,
                             #   kernel_foldable, kernel_xz, kernel_phase}
    seg.output_wires         # tuple[int] — deferred-space OUTPUT_QUBITS wires (the σ-law port)

This is the ONE read model: a consumer never touches raw BarrierBuffer methods
(they remain on the old surface, untouched).

## Routing (spec §2 — cheapest surface serving the declaration)

    dets/decisions-only          →  bare TwirlSampler.sample(shots, seed) path
    groups and/or residual_law   →  record surface (TwirlSampler.sample_barrier, COMPACT
                                    record store — a port internal; group keys byte-identical
                                    to the legacy record_groups() output)
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
    (wrong version, wrong group token, bad checksum) load NOTHING and are silently
    rebuilt/overwritten. Reclaiming space = deleting files; the auto directory is
    safe to wipe at any time (pure cache).
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
    groups      : per-shot ``group_id`` + per-GROUP record arrays (keys, σ,
                  coins, plans, representative rows) in Result — the record
                  surface (Task 2).
    residual_law: the σ-law / plan-structure read model on the Segment
                  (``sigma_law``, ``born_dec``, ``plan_structure``) — the
                  record surface (Task 2).
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
    group_keys  : uint8 ndarray (n_groups, W) — legacy padded keys  [groups]
    group_first_shot : int32 ndarray (n_groups,)                    [groups]
    group_sigmas: uint64 ndarray (n_groups, gw)                     [groups]
    sig_bits    : int                                               [groups]
    group_dets  : bool ndarray (n_groups, n_det)                    [groups+dets]
    group_decisions : bool ndarray (n_groups, n_dec)                [groups+decisions]
    group_coins : tuple[bytes] (lazy, built once per group)         [groups]
    group_plans : tuple[bytes] (lazy, built once per group)         [groups]
    """

    __slots__ = (
        "_dets", "_decisions", "_has_dets", "_has_decisions",
        "_has_groups", "_group_id", "_group_keys", "_group_first_shot",
        "_group_sigmas", "_group_dets", "_group_decisions", "_sig_bits",
        "_coins_cache", "_plans_cache",
    )

    def __init__(
        self,
        *,
        dets: Optional[np.ndarray],
        decisions: Optional[np.ndarray],
        has_dets: bool,
        has_decisions: bool,
        has_groups: bool = False,
        group_id: Optional[np.ndarray] = None,
        group_keys: Optional[np.ndarray] = None,
        group_first_shot: Optional[np.ndarray] = None,
        group_sigmas: Optional[np.ndarray] = None,
        group_dets: Optional[np.ndarray] = None,
        group_decisions: Optional[np.ndarray] = None,
        sig_bits: int = 0,
    ) -> None:
        object.__setattr__(self, "_dets", dets)
        object.__setattr__(self, "_decisions", decisions)
        object.__setattr__(self, "_has_dets", has_dets)
        object.__setattr__(self, "_has_decisions", has_decisions)
        object.__setattr__(self, "_has_groups", has_groups)
        object.__setattr__(self, "_group_id", group_id)
        object.__setattr__(self, "_group_keys", group_keys)
        object.__setattr__(self, "_group_first_shot", group_first_shot)
        object.__setattr__(self, "_group_sigmas", group_sigmas)
        object.__setattr__(self, "_group_dets", group_dets)
        object.__setattr__(self, "_group_decisions", group_decisions)
        object.__setattr__(self, "_sig_bits", int(sig_bits))
        object.__setattr__(self, "_coins_cache", None)
        object.__setattr__(self, "_plans_cache", None)

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
        return self._group_id

    @property
    def n_groups(self) -> int:
        self._need_groups("n_groups")
        return int(self._group_first_shot.shape[0])

    @property
    def group_keys(self) -> np.ndarray:
        self._need_groups("group_keys")
        return self._group_keys

    @property
    def group_first_shot(self) -> np.ndarray:
        self._need_groups("group_first_shot")
        return self._group_first_shot

    @property
    def group_sigmas(self) -> np.ndarray:
        self._need_groups("group_sigmas")
        return self._group_sigmas

    @property
    def sig_bits(self) -> int:
        self._need_groups("sig_bits")
        return self._sig_bits

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

    def _parse_key_fields(self):
        """Parse (coins, plans) per group from the legacy padded key rows.

        Key row layout (the engine's record_groups format): σ bytes (sig_bits)
        ‖ [len_coins u16-LE] ‖ coins padded ‖ [len_plan u16-LE] ‖ plan padded.
        The pad widths are the per-batch maxima, recoverable from the stored
        length fields (every shot's coins/plan equals its group's, so the max
        over groups equals the max over shots). Built ONCE per Result (lazy).
        """
        keys = self._group_keys
        ng = keys.shape[0]
        sbw = self._sig_bits if ng else 0
        if ng == 0:
            object.__setattr__(self, "_coins_cache", ())
            object.__setattr__(self, "_plans_cache", ())
            return
        lc = keys[:, sbw].astype(np.int64) | (keys[:, sbw + 1].astype(np.int64) << 8)
        maxc = int(lc.max())
        po = sbw + 2 + maxc
        lp = keys[:, po].astype(np.int64) | (keys[:, po + 1].astype(np.int64) << 8)
        coins = tuple(keys[g, sbw + 2: sbw + 2 + int(lc[g])].tobytes()
                      for g in range(ng))
        plans = tuple(keys[g, po + 2: po + 2 + int(lp[g])].tobytes()
                      for g in range(ng))
        object.__setattr__(self, "_coins_cache", coins)
        object.__setattr__(self, "_plans_cache", plans)

    @property
    def group_coins(self) -> tuple:
        self._need_groups("group_coins")
        if self._coins_cache is None:
            self._parse_key_fields()
        return self._coins_cache

    @property
    def group_plans(self) -> tuple:
        self._need_groups("group_plans")
        if self._plans_cache is None:
            self._parse_key_fields()
        return self._plans_cache

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
            parts.append(f"n_groups={int(self._group_first_shot.shape[0])}")
        return f"xtim.port.Result({', '.join(parts)})"


# ── Segment ───────────────────────────────────────────────────────────────────

class Segment:
    """A compiled, immutable segment: one circuit + one Consume declaration.

    Built once via ``port.compile()``; cached per (text, consume, plan_cache).
    Immutable except the seed stream (which is always reset in-place
    at each ``run()`` call — no mutable visible state on the Segment itself).

    ``run(shots, seed)`` is the sole entry point.

    When ``consume.residual_law`` is declared the Segment additionally carries
    the σ-law/plan-structure read model (``sigma_law``, ``born_dec``,
    ``plan_structure``, ``sig_bits``, ``gw``, ``output_wires``) — computed once
    at compile time (pure functions of the circuit). Nothing undeclared is
    computed (spec §1): a groups-only Segment performs NO σ-law fetch.
    """

    __slots__ = (
        "_text",
        "_consume",
        "_sampler",     # the bare _xtim.TwirlSampler (or wrapper)
        "_n_det",
        "_n_dec",
        "_is_ch_ppr",   # True iff sample_barrier refuses (CH/PPR circuit)
        "_route",       # "bare" | "record"
        "_sig_bits",    # σ bit width (record route; -1 = not fetched)
        "_sigma_law",   # dict | None (residual_law only)
        "_born_dec",    # dict | None (residual_law only)
        "_out_wires",   # tuple[int] | None (residual_law only)
        "_ps_memo",     # dict: plan-key bytes -> plan_structure dict (residual_law only)
    )

    def __init__(
        self,
        text: str,
        consume: Consume,
        sampler,
        n_det: int,
        n_dec: int,
        is_ch_ppr: bool,
        route: str,
        sig_bits: int = -1,
        sigma_law=None,
        born_dec=None,
        out_wires=None,
    ) -> None:
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_consume", consume)
        object.__setattr__(self, "_sampler", sampler)
        object.__setattr__(self, "_n_det", n_det)
        object.__setattr__(self, "_n_dec", n_dec)
        object.__setattr__(self, "_is_ch_ppr", is_ch_ppr)
        object.__setattr__(self, "_route", route)
        object.__setattr__(self, "_sig_bits", int(sig_bits))
        object.__setattr__(self, "_sigma_law", sigma_law)
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

    # ── σ-law / plan-structure read model (Consume.residual_law) ────────────

    def _need_residual_law(self, name: str):
        if not self._consume.residual_law:
            raise AttributeError(
                f"Segment has no {name!r} — declare Consume(residual_law=True) to "
                f"request the σ-law/plan-structure read model")

    @property
    def sigma_law(self) -> dict:
        """The bare state's combination-tracked port decomposition on the
        OUTPUT_QUBITS wires: sign_j(shot) = ref_signs[j] ^ parity(comb[j] & σ),
        in the σ WORD layout of ``Result.group_sigmas``."""
        self._need_residual_law("sigma_law")
        return self._sigma_law

    @property
    def born_dec(self) -> dict:
        """The Born-measured DECISION operators (declaration order); raw
        outcome bit of decision j on a group = coins[r + kappa + j]."""
        self._need_residual_law("born_dec")
        return self._born_dec

    @property
    def output_wires(self) -> tuple:
        self._need_residual_law("output_wires")
        return self._out_wires

    @property
    def gw(self) -> int:
        """σ word count = ceil(sig_bits / 64) — the group_sigmas column count."""
        if self._sig_bits < 0:
            raise AttributeError(
                "Segment has no 'gw' — declare Consume(groups=True) or "
                "Consume(residual_law=True)")
        return (self._sig_bits + 63) // 64

    @property
    def sig_bits(self) -> int:
        if self._sig_bits < 0:
            raise AttributeError(
                "Segment has no 'sig_bits' — declare Consume(groups=True) or "
                "Consume(residual_law=True)")
        return self._sig_bits

    def plan_structure(self, plan_key: bytes) -> dict:
        """Residual normal form + ShotLaw data for one plan-key (memoized per
        distinct plan bytes; ``b""`` = the clean/identity plan). The keys are
        exactly ``Result.group_plans`` entries."""
        self._need_residual_law("plan_structure")
        pk = bytes(plan_key)
        hit = self._ps_memo.get(pk)
        if hit is None:
            hit = self._sampler.plan_structure(pk)
            self._ps_memo[pk] = hit
        return hit

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

        # ONE batched engine call; in-place reseed; COMPACT record store (the
        # port internal — legacy group keys reconstructed per GROUP at read).
        buf = sampler.sample_barrier(shots, seed, None, True)

        dets_arr = None
        if consume.dets:
            dets_arr = _unpack_b8(np.asarray(buf.dets(), np.uint8),
                                  self._n_det, shots)
        dec_arr = None
        if consume.decisions:
            dec_arr = _unpack_b8(np.asarray(buf.decisions(), np.uint8),
                                 self._n_dec, shots)

        if not consume.groups:
            # residual_law-only declaration: record surface streams, no group
            # arrays computed (nothing undeclared is computed — spec §1).
            return Result(
                dets=dets_arr,
                decisions=dec_arr,
                has_dets=consume.dets,
                has_decisions=consume.decisions,
            )

        # Coherence guard: the buffer's σ width must equal the Segment's
        # (both derive from the same compiled bare state; a mismatch means
        # engine wiring skew — loud, never silent).
        if int(buf.sig_bits) != self._sig_bits:
            raise RuntimeError(
                "record route: BarrierBuffer.sig_bits=%d != Segment.sig_bits=%d "
                "— engine σ-width skew" % (int(buf.sig_bits), self._sig_bits))

        gids, first_occ, keys = buf.record_groups()
        gids = np.asarray(gids, np.int32)
        first_occ = np.asarray(first_occ, np.int32)
        keys = np.asarray(keys, np.uint8)   # kept as a MATRIX (no .tolist — L-C)
        sig = np.asarray(buf.sigmas(), np.uint64)
        group_sigmas = np.ascontiguousarray(sig[first_occ])

        return Result(
            dets=dets_arr,
            decisions=dec_arr,
            has_dets=consume.dets,
            has_decisions=consume.decisions,
            has_groups=True,
            group_id=gids,
            group_keys=keys,
            group_first_shot=first_occ,
            group_sigmas=group_sigmas,
            group_dets=(dets_arr[first_occ] if dets_arr is not None else None),
            group_decisions=(dec_arr[first_occ] if dec_arr is not None else None),
            sig_bits=self._sig_bits,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _unpack_b8(b8: np.ndarray, ncols: int, shots: int) -> np.ndarray:
    """Stim LE b8-packed uint8 (shots, ceil(K/8)) → bool (shots, ncols)."""
    if ncols == 0:
        return np.zeros((shots, 0), dtype=bool)
    return np.unpackbits(b8, axis=1, bitorder="little")[:, :ncols].astype(bool)


def _probe_ch_ppr(sampler) -> bool:
    """Return True iff the circuit is CH/PPR-class (state retention unsupported).

    Detection: attempt sample_barrier(0, 0) — a zero-shot retained call.
    Diagonal circuits succeed; CH/PPR circuits raise RuntimeError naming
    'diagonal-class'. This is the reliable, engine-native detection (the
    channel_report() does not expose the ppr_tables flag at the Python level).
    """
    try:
        sampler.sample_barrier(0, 0)
        return False
    except RuntimeError as e:
        msg = str(e)
        if "diagonal-class" in msg or "PPR" in msg or "CH" in msg or "retention" in msg:
            return True
        # Unexpected RuntimeError — re-raise rather than swallowing
        raise


def _fetch_sigma_law(text: str, sampler):
    """Fetch the Segment-level σ-law read model (residual_law declarations only).

    Module-level seam (monkeypatchable) so the undeclared-not-computed contract
    is spy-testable: a groups-only compile must NEVER call this.

    Returns (sigma_law_dict, born_dec_dict, out_wires_tuple).
    Raises ValueError loudly if the bare state's certified-generator count
    disagrees with the sampler's σ width (the pinned engine contract).
    """
    st = _xtim.bare_state_of(str(text))
    wires = tuple(int(w) for w in sampler.output_wires())
    law = st.port_sigma_law(list(wires))
    if int(law["ngens"]) != int(sampler.sig_bits):
        raise ValueError(
            "residual_law: bare_state_of ngens=%d != sampler sig_bits=%d — "
            "the σ-law bit order would not match sampled σ words (engine "
            "contract violation)" % (int(law["ngens"]), int(sampler.sig_bits)))
    born = sampler.born_dec()
    return law, born, wires


def _build_segment(text: str, consume: Consume, plan_cache) -> "Segment":
    """Build a Segment for (text, consume, plan_cache). Called once per distinct triple."""
    # Validate consume: nothing declared is an error (spec §1)
    if not (consume.dets or consume.decisions or consume.groups
            or consume.residual_law):
        raise ValueError(
            "Consume declares nothing — at least one of dets, decisions, "
            "groups, residual_law must be True")

    # Compile the bare TwirlSampler first so we can detect the circuit class.
    # selfcheck=0 per adaptq's hot-path contract (engine.py:382-404).
    # plan_cache forwards to the engine's TWPL disk layer (compile option, spec §1).
    try:
        tw_sampler = _twirl.compile_twirl_sampler(text, selfcheck=0,
                                                  disk_cache=plan_cache)
    except ValueError as e:
        if "no deterministic detector channels" in str(e):
            # Port-level guidance (port-v3 T3 fold-in): the engine's record
            # sampler refuses a circuit with ZERO record channels of any kind
            # (no DETECTOR, no DECISION — e.g. the raw cube_ccz.stim, whose
            # only records are MPPs consumed internally by feedback).  The
            # natural gate form declares those feedback MPPs as DECISION(k)
            # (spec §2), which makes the whole surface compile.
            raise ValueError(
                str(e) + " — circuit has no DETECTOR/DECISION channels — "
                "declare feedback MPPs as DECISION(k) (the natural gate form, "
                "spec §2); a record sampler needs at least one declared "
                "record channel") from e
        raise

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

    # Probe for CH/PPR-class at compile time (spec §2 — refusals at declaration time).
    is_ch_ppr = _probe_ch_ppr(sampler)

    # Enforce spec §2 refusals: CH/PPR circuits refuse groups/residual_law.
    # The error must name BOTH the circuit property AND the colliding declaration.
    if is_ch_ppr and (consume.groups or consume.residual_law):
        raise RuntimeError(_ch_ppr_groups_refusal_message(consume))

    # Routing (spec §2): groups/residual_law → the record surface; otherwise bare.
    route = "record" if (consume.groups or consume.residual_law) else "bare"

    sig_bits = -1
    sigma_law = None
    born_dec = None
    out_wires = None
    if route == "record":
        sig_bits = int(sampler.sig_bits)
        if consume.residual_law:
            # σ-law read model: fetched ONCE at compile (pure circuit function).
            # A groups-only declaration never reaches this call (spy-gated).
            sigma_law, born_dec, out_wires = _fetch_sigma_law(text, sampler)

    return Segment(text, consume, sampler, n_det, n_dec, is_ch_ppr, route,
                   sig_bits=sig_bits, sigma_law=sigma_law, born_dec=born_dec,
                   out_wires=out_wires)


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
        If consume declares nothing, or if the circuit has no deterministic
        detector/decision channels (the TwirlSampler requires at least one).
    RuntimeError
        If the circuit property is incompatible with the declared consumption
        and no engine surface can serve the pair (spec §2). The message names
        BOTH the circuit property and the colliding declaration.
    """
    if not isinstance(consume, Consume):
        raise TypeError(
            f"consume must be a port.Consume instance, got {type(consume).__name__}")
    if plan_cache is None or plan_cache is False:
        plan_cache = None
    elif plan_cache is not True:
        plan_cache = str(plan_cache)
    return _cached_build(str(circuit_text), consume, plan_cache)


# ── CH/PPR refusal message (spec §2) ─────────────────────────────────────────

def _ch_ppr_groups_refusal_message(consume: Consume) -> str:
    """Build the spec §2 refusal message for a CH/PPR circuit + groups/residual_law.

    Named for clarity in test assertions.
    """
    colliding = []
    if consume.groups:
        colliding.append("groups")
    if consume.residual_law:
        colliding.append("residual_law")
    decl = " and ".join(colliding)
    return (
        f"circuit property: CH/PPR (non-diagonal Clifford) — "
        f"state retention requires a diagonal-class circuit; "
        f"colliding declaration: {decl}. "
        f"CH/PPR circuits serve dets/decisions only (spec §2)."
    )
