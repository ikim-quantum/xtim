"""Oracle tests for xtim.port — bare-surface route (dets/decisions-only).

Coverage
--------
PORT-01  port outputs byte-equal to direct bare-surface calls (small workload)
PORT-02  ch_cultivation compiles and runs under Consume(dets=True, decisions=True)
PORT-03  ch_cultivation Consume(groups=True) is SERVED on the layout-3 record route (J-b1 F; was a refusal)
PORT-04  same-seed reproducible (byte-equal across re-runs on the same Segment)
PORT-05  used Segment re-run at a new seed == fresh Segment at that seed (byte-equal)
PORT-06  compile-once cache: two compile() calls return the SAME Segment (identity)
PORT-07  zero-shot seeded call < 5 ms on a warm Segment
PORT-08  undeclared-not-computed: Consume(decisions=True) Result lacks .dets
PORT-09  Consume frozen: attribute mutation raises FrozenInstanceError
PORT-10  Segment rejects attribute mutation
PORT-11  Nothing-declared Consume raises ValueError
PORT-12  the removed 3.0 surfaces stay removed: run() takes only (shots, seed)
         — no frame_in/input_bits — and Result carries no frame_out; IF-driven
         circuits refuse at compile naming both sides
PORT-13  port outputs byte-equal to direct bare-surface calls (ch_cultivation.stim)
PORT-14  groups/residual_law on diagonal circuit compiles to the record route
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pytest

import xtim
from xtim.port import Consume, Segment, Result, compile as port_compile
from xtim import twirl as _twirl


# ── fixtures / helpers ────────────────────────────────────────────────────────

_EXAMPLES = Path(__file__).resolve().parent / "data"
_DATA = Path(__file__).resolve().parent / "data"

_CH_PATH = _EXAMPLES / "ch_cultivation.stim"
_CH_TEXT = _CH_PATH.read_text()

# Small workload: qrm_magic_producer has n_det=15, n_dec=1 — exercises both fields.
_SMALL_TEXT = (_DATA / "adaptq_qrm_magic_producer.stim").read_text()

# Diagonal circuit with DECISIONs (for decisions-only test).
# M 0 twice creates a deterministic DETECTOR; the DECISION comes from M 1 after
# an X — its value is deterministically 1, so the decisions stream is provably
# NONZERO (PORT-08 non-vacuity: an all-zero stream could mask an unpack bug).
_DECISION_TEXT = """M 0 0
DETECTOR rec[-1] rec[-2]
DEPOLARIZE1(0.001) 0
M 0
M 0
DETECTOR rec[-1] rec[-2]
X 1
M 1
DECISION(0) rec[-1]
"""


def _direct_sample(text: str, shots: int, seed: int, n_det: int, n_dec: int):
    """Reference: compile_twirl_sampler, sample directly, unpack to bool arrays."""
    s = _twirl.compile_twirl_sampler(text, selfcheck=0)
    inner = s._inner if hasattr(s, "_inner") else s
    raw = inner.sample(shots, seed, None)
    if len(raw) == 3:
        dets_b8, _obs, dec_b8 = raw
    else:
        dets_b8, _obs = raw
        dec_b8 = None
    dets = _unpack(dets_b8, n_det, shots)
    decs = _unpack(dec_b8, n_dec, shots) if dec_b8 is not None else np.zeros((shots, 0), dtype=bool)
    return dets, decs


def _unpack(b8, ncols, shots):
    if b8 is None or ncols == 0:
        return np.zeros((shots, 0), dtype=bool)
    return np.unpackbits(np.asarray(b8, dtype=np.uint8), axis=1, bitorder="little")[:, :ncols].astype(bool)


# ── PORT-01: byte-equality vs direct bare-surface (small workload) ────────────

def test_port_byte_equal_small_workload():
    """PORT-01: port dets == direct TwirlSampler.sample dets (small circuit)."""
    text = _SMALL_TEXT
    consume = Consume(dets=True)
    seg = port_compile(text, consume)

    # Direct reference
    s = _twirl.compile_twirl_sampler(text, selfcheck=0)
    inner = s._inner if hasattr(s, "_inner") else s
    n_det = int(inner.num_detectors)

    seed = 77
    shots = 400

    ref_dets, _ = _direct_sample(text, shots, seed, n_det, 0)
    result = seg.run(shots=shots, seed=seed)

    assert result._has_dets
    assert np.array_equal(result.dets, ref_dets), (
        f"port dets differ from direct TwirlSampler: "
        f"{int((result.dets != ref_dets).sum())} differing bits"
    )


# ── PORT-02: ch_cultivation compiles and runs ──────────────────────────────────

def test_ch_cultivation_compiles_and_runs():
    """PORT-02: ch_cultivation.stim compiles and runs under Consume(dets=True, decisions=True)."""
    consume = Consume(dets=True, decisions=True)
    seg = port_compile(_CH_TEXT, consume)

    assert isinstance(seg, Segment)
    result = seg.run(shots=500, seed=99)

    # ch_cultivation has 3 detectors, 0 decisions
    assert result._has_dets
    assert result._has_decisions
    assert result.dets.shape == (500, 3)   # 3 DETECTOR lines
    assert result.decisions.shape == (500, 0)  # no DECISION declarations
    assert result.dets.dtype == bool


# ── PORT-03: CH/PPR + groups → SERVED (refusal -> exactness at J-b1 F) ─────────

def test_ch_cultivation_groups_is_served():
    """PORT-03, RE-PINNED 2026-10-20 (J-b1, xtim-engine-13; M-DECPPR), refusal -> exactness: ch_cultivation with
    Consume(groups=True) compiles to the layout-3 record route (CH/PPR shots are kind-2 records) and runs; the port
    groups equal the raw buffer's (byte oracle: tests/test_port_groups.py::test_pg08)."""
    seg = port_compile(_CH_TEXT, Consume(dets=True, groups=True))
    res = seg.run(shots=500, seed=1)
    assert res.group_id.shape == (500,)
    assert res.n_groups >= 1
    assert len(res.group_keys) == res.n_groups


# ── PORT-04: same-seed reproducible ──────────────────────────────────────────

def test_same_seed_reproducible():
    """PORT-04: same seed on same Segment → byte-identical Result."""
    consume = Consume(dets=True, decisions=True)
    seg = port_compile(_CH_TEXT, consume)

    r1 = seg.run(shots=300, seed=42)
    r2 = seg.run(shots=300, seed=42)

    assert np.array_equal(r1.dets, r2.dets), "same-seed dets not reproducible"
    assert np.array_equal(r1.decisions, r2.decisions), "same-seed decisions not reproducible"


# ── PORT-05: used Segment re-run at new seed == fresh Segment ─────────────────

def test_used_segment_same_as_fresh_at_new_seed():
    """PORT-05: Segment reused at a new seed == fresh Segment at that seed (byte-equal)."""
    consume = Consume(dets=True)
    text = _SMALL_TEXT

    fresh = port_compile(text, consume)
    # Fresh segment, different seed is a new compile — but compile is cached,
    # so re-run at new seed on the SAME segment must equal a fresh compile at that seed.
    # Because the cache returns the same Segment, just run at seed=7 twice:
    # once on a "used" segment (previously called at seed=42) and check equality.
    seg = port_compile(text, consume)
    _ = seg.run(shots=200, seed=42)    # dirty the stream
    r_warm = seg.run(shots=200, seed=7)

    # Fresh segment (new compile, but cache returns same object — test both)
    r_fresh = fresh.run(shots=200, seed=7)

    assert np.array_equal(r_warm.dets, r_fresh.dets), (
        "used Segment at new seed does not match fresh Segment at same seed — "
        "in-place reseeding (set_seed) is broken")


# ── PORT-06: compile-once cache ───────────────────────────────────────────────

def test_compile_once_cache_identity():
    """PORT-06: two compile() calls with identical (text, consume) return the same Segment."""
    consume = Consume(dets=True, decisions=True)
    text = _CH_TEXT

    seg1 = port_compile(text, consume)
    seg2 = port_compile(text, consume)

    assert seg1 is seg2, (
        "compile() returned two different Segment objects for the same (text, consume) — "
        "the cache is not working or the cache key is wrong")


# ── PORT-07: zero-shot call < 5 ms ───────────────────────────────────────────

def test_zero_shot_seeded_call_fast():
    """PORT-07: zero-shot seeded call < 5 ms on a warm Segment (no reconstruction)."""
    consume = Consume(dets=True, decisions=True)
    seg = port_compile(_CH_TEXT, consume)

    # Warm the segment (ensure the sampler is warm)
    seg.run(shots=1000, seed=1)

    # Measure zero-shot call latency (min of 5 trials)
    times = []
    for k in range(5):
        t0 = time.perf_counter()
        seg.run(shots=0, seed=100 + k)
        times.append(time.perf_counter() - t0)

    best = min(times)
    assert best < 5e-3, (
        f"zero-shot seeded call took {best * 1e3:.1f} ms — expected < 5 ms; "
        f"in-place reseeding (set_seed) contract may be violated")


# ── PORT-08: undeclared not computed (non-vacuous) ────────────────────────────

def test_undeclared_not_computed():
    """PORT-08: Consume(decisions=True) Result lacks .dets; accessing it raises.

    Strengthened (port-v3 T3 fold-in): the declared decisions stream must be
    NONZERO — the circuit's DECISION bit is deterministically 1 (X before M),
    so an unpack/column bug that silently zeroed the stream fails loudly."""
    consume = Consume(decisions=True)
    # Use a circuit that actually has decisions
    seg = port_compile(_DECISION_TEXT, consume)
    result = seg.run(shots=50, seed=13)

    # decisions declared → present, and NON-VACUOUSLY nonzero
    assert result._has_decisions
    assert result.decisions.shape == (50, 1)
    assert result.decisions.all(), (
        "DECISION(0) is deterministically 1 (X 1; M 1) but the port returned "
        "zeros — the decisions stream is not being read")

    # dets NOT declared → not present
    assert not result._has_dets
    with pytest.raises(AttributeError, match="dets"):
        _ = result.dets


# ── PORT-08b: dets-only result has no decisions ───────────────────────────────

def test_dets_only_result_has_no_decisions():
    """PORT-08b: Consume(dets=True) Result lacks .decisions."""
    consume = Consume(dets=True)
    seg = port_compile(_SMALL_TEXT, consume)
    result = seg.run(shots=50, seed=13)

    assert result._has_dets
    assert not result._has_decisions
    with pytest.raises(AttributeError, match="decisions"):
        _ = result.decisions


# ── PORT-09: Consume frozen ───────────────────────────────────────────────────

def test_consume_is_frozen():
    """PORT-09: Consume is a frozen dataclass — attribute mutation raises FrozenInstanceError."""
    c = Consume(dets=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.dets = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.decisions = True  # type: ignore[misc]


# ── PORT-10: Segment rejects attribute mutation ───────────────────────────────

def test_segment_rejects_mutation():
    """PORT-10: Segment.__setattr__ raises AttributeError on any mutation attempt."""
    seg = port_compile(_SMALL_TEXT, Consume(dets=True))
    with pytest.raises(AttributeError):
        seg._text = "SOMETHING ELSE"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        seg.new_attr = 42  # type: ignore[attr-defined]


# ── PORT-11: nothing-declared ValueError ──────────────────────────────────────

def test_nothing_declared_raises_valueerror():
    """PORT-11: Consume() with nothing True raises ValueError."""
    with pytest.raises(ValueError, match="declares nothing"):
        port_compile(_SMALL_TEXT, Consume())


# ── PORT-12: the 3.0-removed surfaces stay removed ───────────────────────────

# A one-driving-bit IF circuit (external INPUT_BITS) — the class the removed
# ``input_bits`` row-gather path used to serve.
_IF_TEXT = """
INPUT_BITS drv 1
R 0
H 0
M 0
DECISION(0) rec[-1]
R 1
IF drv[0] {
  X 1
}
M 1
DECISION(1) rec[-1]
"""


def test_run_signature_has_no_frame_or_driven_parameters():
    """PORT-12a: ``run`` takes only (shots, seed) — the dead ``frame_in``
    (seam contract: only classical data crosses a seam) and ``input_bits``
    (row-gather, no consumers) parameters were REMOVED at 3.0."""
    import inspect
    params = list(inspect.signature(Segment.run).parameters)
    assert params == ["self", "shots", "seed"], params

    seg = port_compile(_SMALL_TEXT, Consume(dets=True))
    with pytest.raises(TypeError):
        seg.run(shots=5, seed=0, frame_in=np.zeros((5, 2), dtype=np.uint8))
    with pytest.raises(TypeError):
        seg.run(shots=5, seed=0, input_bits=np.zeros((5, 1), dtype=np.uint8))


def test_result_has_no_frame_out():
    """PORT-12b: the always-None ``Result.frame_out`` attribute is gone."""
    seg = port_compile(_SMALL_TEXT, Consume(dets=True))
    result = seg.run(shots=5, seed=0)
    assert not hasattr(result, "frame_out")
    assert "frame_out" not in Result.__slots__
    assert "frame_out" not in repr(result)


def test_if_driven_circuit_refuses_at_compile():
    """PORT-12c: an IF-driven circuit now refuses at compile (never silently
    serves the all-zeros resolved branch), naming both sides + the escape."""
    with pytest.raises(RuntimeError, match="IF-branch blocks") as exc:
        port_compile(_IF_TEXT, Consume(dets=True, decisions=True))
    msg = str(exc.value)
    assert "circuit property:" in msg and "colliding declaration:" in msg
    assert "resolve_branches_text" in msg


# ── PORT-13: byte-equality vs direct bare-surface (ch_cultivation) ────────────

def test_port_byte_equal_ch_cultivation():
    """PORT-13: port dets == direct TwirlSampler.sample dets for ch_cultivation.stim."""
    text = _CH_TEXT
    consume = Consume(dets=True, decisions=True)
    seg = port_compile(text, consume)

    # Direct reference compile (independent — separate sampler instance)
    s = _twirl.compile_twirl_sampler(text, selfcheck=0)
    inner = s._inner if hasattr(s, "_inner") else s
    n_det = int(inner.num_detectors)
    n_dec = int(getattr(inner, "num_decisions", 0))

    seed = 55
    shots = 600
    ref_dets, ref_decs = _direct_sample(text, shots, seed, n_det, n_dec)

    result = seg.run(shots=shots, seed=seed)

    assert np.array_equal(result.dets, ref_dets), (
        f"ch_cultivation port dets differ from direct: "
        f"{int((result.dets != ref_dets).sum())} differing bits")
    assert np.array_equal(result.decisions, ref_decs), (
        f"ch_cultivation port decisions differ from direct: "
        f"{int((result.decisions != ref_decs).sum())} differing bits")


# ── PORT-14: groups/residual_law on diagonal circuit → NotImplementedError ────

def test_groups_on_diagonal_circuit_compiles_record_route():
    """PORT-14 (updated at T2): Consume(groups=True) on a diagonal circuit now compiles
    to the record route and runs (full oracle coverage in tests/test_port_groups.py)."""
    seg = port_compile(_SMALL_TEXT, Consume(dets=True, groups=True))
    res = seg.run(shots=64, seed=5)
    assert res.group_id.shape == (64,)
    assert res.n_groups >= 1


def test_residual_law_on_diagonal_circuit_compiles_record_route():
    """PORT-14b (updated at T2): Consume(residual_law=True) on a diagonal circuit now
    compiles; the Segment carries the read model (layout 3 at J-b1 F: born_dec / output_wires; sigma retired)."""
    seg = port_compile(_SMALL_TEXT, Consume(dets=True, residual_law=True))
    assert isinstance(seg.born_dec, dict)
    assert isinstance(seg.output_wires, tuple)


# ── PORT-15: Result immutability ──────────────────────────────────────────────

def test_result_rejects_mutation():
    """PORT-15: Result.__setattr__ raises AttributeError on any mutation attempt."""
    seg = port_compile(_SMALL_TEXT, Consume(dets=True))
    result = seg.run(shots=10, seed=0)
    with pytest.raises(AttributeError):
        result._dets = np.zeros(3)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        result.new_field = 42  # type: ignore[attr-defined]


# ── PORT-16: zero-channel circuit served empty; the DECISION form runs (J-b1 F; was a refusal) ──

def test_zero_channel_circuit_is_served_empty_and_decision_form_runs():
    """PORT-16, RE-PINNED 2026-10-20 (J-b1, xtim-engine-13), refusal -> exactness: the raw cube_ccz.stim has ZERO
    record channels (no DETECTOR, no DECISION — its only records are MPPs consumed by feedback). The legacy sampler
    refused it ("no deterministic detector channels"); the FAM sampler relaxes the empty gate (J-a; the
    gate itself was deleted at J-b2 J3), so it compiles and serves valid EMPTY streams. The DECISION-form variant of
    the same circuit then compiles and runs with real Born decisions."""
    cube_text = (_EXAMPLES / "cube_ccz.stim").read_text()
    empty = port_compile(cube_text, Consume(dets=True, decisions=True)).run(shots=20, seed=1)
    assert empty.dets.shape == (20, 0) and empty.decisions.shape == (20, 0)

    # The natural gate form: strip the PAULI_EXPECTATION payload, declare the
    # 4 feedback MPPs as DECISION(0..3) — compiles and runs on the bare route.
    lines = [ln for ln in cube_text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")
             and not ln.startswith("PAULI_EXPECTATION")]
    dec_form = "\n".join(lines) + "\n" + "".join(
        f"DECISION({k}) rec[{k - 4}]\n" for k in range(4))
    seg = port_compile(dec_form, Consume(dets=True, decisions=True))
    res = seg.run(shots=200, seed=11)
    assert res.decisions.shape == (200, 4)
    # The 4 MPP syndrome bits are fair Born coins — the stream must be nonzero
    # AND not all-ones (a stuck stream would be a silent-read bug).
    assert res.decisions.any() and not res.decisions.all()
