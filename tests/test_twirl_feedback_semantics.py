"""Classically-controlled Pauli feedback (`CX/CY/CZ rec[-k] q`): both engines vs Stim.

CONTRACT (docs/xtim_dialect.md §1b, table at line 120):

    CX rec[-k] q    apply X to qubit q iff record rec[-k] = 1

where the record is the RECORDED bit (`!` invert and `M(p)` flip included).
Real Stim runs `rec[-k]`-controlled Paulis natively and is the independent
oracle for detector VALUES; the exact engine's RAW measurements are the oracle
for the twirl engine's absolute readouts.

THREE DEFECTS THIS FILE PINS (all present through 3.1.0, fixed in 3.1.1):

* The TWIRL engine dropped every feedback whose byproduct did not cross a
  non-Clifford gate: the pybind ``TwirlSampler`` family normalized with
  ``Feedback::Strip`` and built no ``FeedbackPlan`` (unlike
  ``cpp/src/sampler.cpp``, which relabels), so the ``ControlledPauli`` was
  deleted from both the retained state and the record algebra.  Fixed by
  ``NormalizePolicy::coherentize_all`` behind ``normalize_for_twirl``: every
  feedback is rewritten into coherent gates (copy ancillas for reused control
  wires, ``M(p)`` folded into a pre-measurement error).
* The EXACT engine's detector/observable columns were wrong for non-crossing
  feedback whenever the circuit had a Hadamard: its noiseless reference ran the
  feedback relabel BEFORE the deterministic-invert XOR, while every shot runs
  them the other way round (``cpp/src/sampler.cpp``).
* The coherentizer's noisy-readout fold flipped the control's post-measurement
  STATE (Stim's ``M(p)`` flips only the reported bit); visible whenever the
  control is observed again before a ``R``.  Fixed by holding the flip instance
  on an ancilla and applying it before and after the measurement.
* A feedback onto its OWN idle control wire (``M q; CX rec[-1] q``, the
  measure-and-conditionally-reset idiom) was coherentized "in place" into a
  degenerate ``CX q q`` — not a unitary on the wire at all — and answered a
  fair coin on the twirl path.  Fixed by routing a self-targeting feedback
  through the copy-ancilla gadget (the relabel route was always right).

Every test here was red on its contract assertion against the pre-fix build,
with its in-test oracle-sanity assertions passing (the one exception is named
where it stands: the noisy-source self-target case, which already took the copy
gadget and is pinned as a regression guard).
"""
import numpy as np
import pytest

from xtim.circuit import Circuit
from xtim.twirl import compile_twirl_sampler


def _inner(sampler):
    return sampler._inner if hasattr(sampler, "_inner") else sampler


# `M 2` on a |1> ancilla is a DETERMINISTIC record = 1; on |0> it is 0.  The
# DETECTOR is what the twirl sampler needs to accept the circuit at all.
def _head(rec: int) -> str:
    return "R 0 1 2\n" + ("X 2\n" if rec else "") + "M 2\nDETECTOR rec[-1]\n"


# ── the oracle: the EXACT engine, which agrees with real Stim ────────────────

def test_exact_engine_feedback_matches_documented_semantics():
    """`CX rec[-1] 0` flips the later `M 0` iff the control record is 1."""
    for rec in (0, 1):
        c = Circuit(_head(rec) + "CX rec[-1] 0\nM 0\n")
        meas = c.compile_detector_sampler(seed=1, engine="exact").sample(
            4, return_measurements=True)[-1]
        got = np.asarray(meas)[:, 1]
        assert (got == bool(rec)).all(), (
            f"rec={rec}: exact-engine M 0 = {got}, expected all {bool(rec)}")


# ── C1: the EXACT engine's detector/observable columns vs real Stim ──────────
#
# The exact engine's detector/observable channels are reference-relative: the
# emitted bit is parity(shot) XOR parity(noiseless reference), and with feedback
# the reference's records are relabeled by the same triangular pass as every
# shot's.  That pass must read the RECORDED control bit — `!`-invert included —
# in the same order the shot path does.  `eliminate_hadamards` only rebuilds a
# circuit that has an H to eliminate, and when it does it also absorbs every
# single-qubit Clifford frame into the terminal reads (`X c; M c` -> `M c` with a
# stamped invert); so any Hadamard anywhere used to change the ORDER in which
# the reference saw invert vs feedback, and inverted every detector the
# feedback fed (200/200 vs Stim 0/200), while the raw measurements stayed right.

_C1_MIDS = {
    "H_on_idle_wire": "H 1\n",         # a Hadamard on a wire that is never measured
    "HH_on_target": "H 0\nH 0\n",      # a net identity on the feedback target
    "reused_control": "R 2\nH 2\n",    # the control wire reused after its measurement
}


@pytest.mark.parametrize("mid", sorted(_C1_MIDS))
@pytest.mark.parametrize("p", [0.0, 0.3])
def test_exact_engine_detector_columns_match_stim_with_hadamard(mid, p):
    """Non-crossing feedback + any Hadamard: detector/observable columns vs Stim.

    `X 2; M 2` is a deterministic 1, so the feedback always fires and `M 0`
    reads 1 on every noiseless shot: the parity detector is deterministically 0
    and the observable never flips.  With `X_ERROR(p) 0` before `M 0` the
    detector must fire exactly on the shots where the raw parity is odd (the
    reference parity is 0), i.e. `det == parity(raw records)` shot for shot.
    """
    noise = f"X_ERROR({p}) 0\n" if p > 0 else ""
    text = ("R 0 1 2\nX 2\nM 2\nCX rec[-1] 0\n" + _C1_MIDS[mid] + noise +
            "M 0\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    shots = 200

    s = Circuit(text).compile_detector_sampler(seed=1, engine="exact")
    dets, obs, meas = s.sample(shots, separate_observables=True, return_measurements=True)
    dets = np.asarray(dets, np.uint8).reshape(shots, -1)
    obs = np.asarray(obs, np.uint8).reshape(shots, -1)
    meas = np.asarray(meas, np.uint8).reshape(shots, -1)

    assert (meas[:, 0] == 1).all(), "oracle sanity: the control record is a deterministic 1"
    raw_parity = meas[:, 0] ^ meas[:, 1]
    if p == 0:
        assert _stim_dets(text, shots).sum() == 0, "oracle sanity: stim"
        assert (meas[:, 1] == 1).all(), f"{mid}: the feedback must reach M 0 on every shot"
        assert dets.sum() == 0, (
            f"{mid}: exact-engine detector fired on {int(dets.sum())}/{shots} noiseless "
            f"shots; Stim says 0")
        assert obs.sum() == 0, f"{mid}: observable flipped on {int(obs.sum())} noiseless shots"
    else:
        assert 0 < raw_parity.sum() < shots, "oracle sanity: the X_ERROR must fire sometimes"
        assert (dets[:, 0] == raw_parity).all(), (
            f"{mid}: detector must equal the raw record parity shot for shot "
            f"(reference parity 0); mismatches: {int((dets[:, 0] != raw_parity).sum())}")
        assert (obs[:, 0] == (meas[:, 1] ^ 1)).all(), (
            f"{mid}: observable flip must track M 0 against its noiseless value 1")


# ── the defect ───────────────────────────────────────────────────────────────

def test_twirl_retained_state_applies_rec_driven_pauli():
    """On the RETAINED state, `CX rec[-1] 0` must act as X on wire 0 iff rec=1.

    Read as <Z> on the declared output qubit: +1 unflipped, -1 flipped.
    `X 0` at the same position is the control and acts correctly today.
    """
    def z_of(body: str, rec: int) -> float:
        s = _inner(compile_twirl_sampler(
            _head(rec) + (body + "\n" if body else "") + "OUTPUT_QUBITS out 0\n",
            p_factor=1.0, selfcheck=0))
        st = s.sample_barrier(1, 0).materialize(0)
        return float(st.pauli_expectation_z(list(s.output_wires())[0]))

    # Control: an unconditional X *does* reach the retained state.
    assert z_of("X 0", 0) == pytest.approx(-1.0)
    assert z_of("", 0) == pytest.approx(+1.0)

    # The contract: the rec-driven X fires iff the record is 1.
    assert z_of("CX rec[-1] 0", 0) == pytest.approx(+1.0)
    assert z_of("CX rec[-1] 0", 1) == pytest.approx(-1.0)


def test_twirl_records_see_rec_driven_pauli():
    """The teleport parity `rec(M 0) XOR rec(M 1)` is deterministically 0.

    `H 1; M 1` makes the control record a fair coin; `CX rec[-1] 0` copies it
    onto qubit 0, so the two records agree on every shot and their parity is a
    well-formed deterministic detector.  With the feedback dropped, `M 0` is
    identically 0 and the parity is a fair coin -- the twirl compile then
    refuses the circuit outright ("no deterministic detector channels").
    """
    text = "R 0 1\nH 1\nM 1\nCX rec[-1] 0\nM 0\nDETECTOR rec[-1] rec[-2]\n"
    s = _inner(compile_twirl_sampler(text, p_factor=1.0, selfcheck=0))
    dets = np.asarray(s.sample_barrier(64, 7).dets(), np.uint8)
    assert dets.sum() == 0, "teleport parity detector must be deterministically 0"


# ── the two harder coherentize paths, each against the EXACT engine ──────────
#
# `coherentize_classically_controlled_feedback` has two non-trivial branches
# beyond the plain idle-Z-control rewrite, and the twirl engine must be correct
# on BOTH once every feedback is routed through it:
#
#   * the COPY-ANCILLA gadget (coherentize_feedback.cpp:244-258) — taken when
#     the control WIRE is reused between its measurement and the feedback, so
#     the eigenvalue cannot be read in place and is copied onto a fresh ancilla
#     before the measurement;
#   * the NOISY-READOUT FOLD (coherentize_feedback.cpp:222-224) — taken when the
#     source measurement has a readout-flip probability, which is folded into an
#     anticommuting error placed BEFORE the measurement so the recorded bit and
#     the driving value flip TOGETHER.
#
# The oracle in both is the EXACT engine's raw measurement records on the SAME
# circuit text (it honours the documented `rec[-k]` semantics and agrees with
# real Stim — pinned by the first test above); real Stim is the oracle for
# detector VALUES wherever a circuit is expressible in Stim.


def _exact_measurements(text: str, shots: int, seed: int = 5) -> np.ndarray:
    """Raw per-shot measurement records from the EXACT engine (the oracle)."""
    s = Circuit(text).compile_detector_sampler(seed=seed, engine="exact")
    return np.asarray(s.sample(shots, return_measurements=True)[-1])


def test_twirl_feedback_with_reused_control_wire_matches_exact():
    """Control measured, then its WIRE reused, THEN the feedback fires.

    `R 2; H 2` between `M 2` and `CX rec[-1] 0` makes the control wire non-idle,
    so the feedback must be driven from a fresh copy ancilla rather than in
    place.  The record value is still deterministic, so the exact engine's `M 0`
    is an exact per-shot oracle for the twirl engine's retained-state <Z> on the
    output wire (+1 = unflipped, -1 = flipped).
    """
    def body(rec: int) -> str:
        return ("R 0 1 2\n" + ("X 2\n" if rec else "") + "M 2\n"
                "DETECTOR rec[-1]\n"
                "R 2\nH 2\n"                 # control WIRE reused before the feedback
                "CX rec[-1] 0\n")

    for rec in (0, 1):
        # Oracle: the exact engine, on the same circuit, measuring qubit 0.
        meas = _exact_measurements(body(rec) + "M 0\n", 4)
        expect_flipped = meas[:, 1]
        assert (expect_flipped == bool(rec)).all(), (
            f"oracle sanity: exact-engine M 0 = {expect_flipped}, want {bool(rec)}")

        # The twirl engine's retained state must agree, shot for shot.
        s = _inner(compile_twirl_sampler(
            body(rec) + "OUTPUT_QUBITS out 0\n", p_factor=1.0, selfcheck=0))
        st = s.sample_barrier(1, 0).materialize(0)
        z = float(st.pauli_expectation_z(list(s.output_wires())[0]))
        assert z == pytest.approx(-1.0 if rec else +1.0), (
            f"rec={rec}: twirl retained <Z> = {z}, exact engine says "
            f"M 0 = {bool(rec)} (i.e. <Z> = {-1.0 if rec else 1.0})")


def _stim_dets(text: str, shots: int, seed: int = 3) -> np.ndarray:
    """Real Stim's detector columns (the independent oracle for detector VALUES)."""
    stim = pytest.importorskip("stim")
    return np.asarray(
        stim.Circuit(text).compile_detector_sampler(seed=seed).sample(shots), np.uint8)


def _twirl_dets(text: str, shots: int, seed: int = 9) -> np.ndarray:
    """Twirl-engine detector columns, unpacked to one uint8 column per DETECTOR."""
    s = _inner(compile_twirl_sampler(text, p_factor=1.0, selfcheck=0))
    packed = np.asarray(s.sample_barrier(shots, seed).dets(), np.uint8)
    return np.unpackbits(packed, axis=1, bitorder="little")


# ── I3: the reused-control copy gadget driven by a RANDOM record ─────────────
#
# `test_twirl_feedback_with_reused_control_wire_matches_exact` above drives the
# copy-ancilla gadget only from a DETERMINISTIC record (`X 2; M 2` — a
# computational basis state).  A gadget that MIS-ENTANGLED the copy ancilla with
# the control would still reproduce a deterministic record shot for shot, so that
# test alone does not establish the gadget is sound.  These do: the record is a
# fair coin in each of the three measurement bases, so the copy must carry the
# per-shot eigenvalue (not a fixed value, and not a value entangled with the
# control's later life) for the teleport parity to stay deterministic.

_REUSE_CASES = {
    # (name, source measurement producing a RANDOM record in that basis)
    "Z": "H 1\nM 1\n",     # Z-basis read of |+>  -> fair coin
    "X": "MX 1\n",         # X-basis read of |0>  -> fair coin
    "Y": "MY 1\n",         # Y-basis read of |0>  -> fair coin
}


@pytest.mark.parametrize("basis", sorted(_REUSE_CASES))
def test_twirl_feedback_reused_control_random_record_matches_exact(basis):
    """Copy-ancilla gadget, RANDOM record, all three source bases.

    `R 1; H 1` between the source measurement and the feedback reuses the control
    WIRE, forcing the copy gadget (coherentize_feedback.cpp).  The record is a fair
    coin, so `M 0` must reproduce it shot for shot and the parity detector must be
    deterministically 0.
    """
    text = ("R 0 1\n" + _REUSE_CASES[basis] +
            "R 1\nH 1\n"                 # control WIRE reused before the feedback
            "CX rec[-1] 0\n"
            "M 0\nDETECTOR rec[-1] rec[-2]\n")

    # Oracle sanity A: the exact engine's RAW records agree shot for shot ...
    meas = _exact_measurements(text, 400)
    assert (meas[:, 0] == meas[:, 1]).all(), (
        f"{basis}: oracle sanity: exact-engine control record and fed-back "
        f"measurement must agree shot for shot")
    # ... and the record is genuinely RANDOM (this is what the deterministic-record
    # test could not establish).
    assert 0 < meas[:, 0].sum() < len(meas), (
        f"{basis}: oracle sanity: source record is not random; test is vacuous")

    # Oracle sanity B: real Stim says the detector is deterministically 0.
    assert _stim_dets(text, 400).sum() == 0, f"{basis}: oracle sanity: stim"

    dets = _twirl_dets(text, 400)
    assert dets.sum() == 0, (
        f"{basis}: reused-control parity detector must stay deterministically 0; "
        f"got {int(dets.sum())} set bits")


def test_twirl_feedback_two_feedbacks_share_one_reused_control():
    """Two feedbacks driven by ONE reused-control record share one copy ancilla.

    Keyed on the source measurement's stream index, so both must read the SAME
    per-shot eigenvalue: both parity detectors are deterministically 0.
    """
    text = ("R 0 1 2\nH 1\nM 1\n"
            "R 1\nH 1\n"                     # control WIRE reused
            "CX rec[-1] 0\nCX rec[-1] 2\n"   # two feedbacks, one shared source
            "M 0\nM 2\n"
            "DETECTOR rec[-2] rec[-3]\n"     # M 0 XOR the source record
            "DETECTOR rec[-1] rec[-3]\n")    # M 2 XOR the source record

    meas = _exact_measurements(text, 400)
    assert (meas[:, 0] == meas[:, 1]).all() and (meas[:, 0] == meas[:, 2]).all(), (
        "oracle sanity: exact-engine records must all agree shot for shot")
    assert 0 < meas[:, 0].sum() < len(meas), "oracle sanity: record not random"
    assert _stim_dets(text, 400).sum() == 0, "oracle sanity: stim"

    dets = _twirl_dets(text, 400)
    assert dets.sum() == 0, (
        f"both shared-source parity detectors must stay deterministically 0; "
        f"got {int(dets.sum())} set bits")


# ── I2: the noisy-readout fold must not flip the control's STATE ─────────────


@pytest.mark.parametrize("p", [1.0, 0.5])
def test_twirl_noisy_readout_does_not_disturb_the_control_state(p):
    """`M(p) c` flips the RECORD, never the control's post-measurement STATE.

    Stim's `M(p)` is a *readout* error: the measurement still projects to the true
    outcome and only the reported bit is flipped.  The coherentizer folds the flip
    into an anticommuting error placed BEFORE the measurement, which reproduces the
    record correctly but ALSO flips the projected state — visible whenever the
    control wire is touched again without an intervening `R`.

    Here wire 1 is re-measured right after.  `M 1` must read 0 (the state was never
    disturbed) while the source record reads the flipped value, and `M 0` must
    follow the RECORD.  Both detectors are therefore deterministically 0.
    """
    text = (f"R 0 1\n"
            f"M({p}) 1\n"          # record flips w.p. p; the STATE stays |0>
            f"CX rec[-1] 0\n"      # feedback follows the RECORD
            f"M 1\n"               # the control wire again: must still read 0
            f"M 0\n"
            f"DETECTOR rec[-1] rec[-3]\n"   # M 0 XOR the (flipped) source record
            f"DETECTOR rec[-2]\n")          # M 1 -- the undisturbed control state
    shots = 400 if p >= 1.0 else 2000

    # Oracle sanity: exact engine's RAW records, and real Stim's detectors.
    meas = _exact_measurements(text, shots)
    assert (meas[:, 0] == meas[:, 2]).all(), (
        "oracle sanity: exact-engine source record and fed-back M 0 must agree")
    assert meas[:, 1].sum() == 0, (
        "oracle sanity: the control's post-measurement state must be undisturbed")
    if p >= 1.0:
        assert meas[:, 0].sum() == len(meas), "oracle sanity: p=1 must always flip"
    else:
        assert 0 < meas[:, 0].sum() < len(meas), "oracle sanity: flip never fired"
    assert _stim_dets(text, shots).sum() == 0, "oracle sanity: stim"

    dets = _twirl_dets(text, shots)
    assert dets[:, 1].sum() == 0, (
        f"p={p}: the noisy readout must not flip the control's post-measurement "
        f"STATE; `M 1` came back 1 on {int(dets[:, 1].sum())}/{shots} shots")
    assert dets[:, 0].sum() == 0, (
        f"p={p}: the feedback must follow the (flipped) record; got "
        f"{int(dets[:, 0].sum())} set bits")


@pytest.mark.parametrize("p", [0.01, 0.5])
def test_twirl_feedback_with_noisy_readout_control_matches_exact(p):
    """`M(p) 1` before the feedback: the flip must reach BOTH record and state.

    A readout flip on the control makes the recorded bit random, but the
    feedback follows the RECORD — so `M 0` reproduces the (possibly flipped)
    control bit on every shot and the parity `rec(M 0) XOR rec(M 1)` stays
    deterministically 0.  That correlation is the whole point of the fold, and
    it is what the exact engine delivers.
    """
    text = (f"R 0 1 2\nX 2\nM({p}) 2\nCX rec[-1] 0\nM 0\n"
            "DETECTOR rec[-1] rec[-2]\n")

    # Oracle: exact engine — the two records agree on EVERY shot ...
    meas = _exact_measurements(text, 400)
    assert (meas[:, 0] == meas[:, 1]).all(), (
        "oracle sanity: exact-engine control record and fed-back measurement "
        "must agree shot for shot")
    if p >= 0.5:  # ... and the flip genuinely fires (the test is not vacuous)
        assert 0 < meas[:, 0].sum() < len(meas), (
            f"oracle sanity: readout flip at p={p} never fired; test is vacuous")

    # The twirl engine must see the same deterministic parity.
    s = _inner(compile_twirl_sampler(text, p_factor=1.0, selfcheck=0))
    dets = np.asarray(s.sample_barrier(400, 9).dets(), np.uint8)
    assert dets.sum() == 0, (
        f"p={p}: noisy-readout teleport parity must stay deterministically 0; "
        f"got {dets.sum()} set bits")


# ── X1: feedback onto its OWN control wire (measure-and-conditionally-reset) ──
#
# `M q; CX rec[-1] q` is the measure-and-conditionally-reset idiom: after it the
# wire is deterministically in the +1 eigenstate of the measured basis.  The
# coherentizer drove an IDLE control "in place" from its own wire, which for a
# self-targeting feedback is `CX q q` — a degenerate entangler that is not a
# unitary on the wire (it maps both |0> and |1> to |0>); the parser rejects it
# in user text, but the coherentizer bypasses the parser.  The twirl path
# answered a fair coin (or refused as anti-commuting) on every such circuit.  A
# self-targeting feedback must take the copy-ancilla gadget like a reused
# control does: the eigenvalue is copied before the measurement and the
# entangler runs from the copy.

_SELF_CASES = {
    # name: (source measurement + feedback, the re-read of q in the same basis,
    #        value of that re-read in Stim: the +1 eigenstate reads 0; a `!` on
    #        the source inverts the fired branch, so the wire ends in the -1
    #        eigenstate and reads 1)
    "Z":     ("H 0\nM 0\nCX rec[-1] 0\n",   "M 0\n",  0),
    "Z_CY":  ("H 0\nM 0\nCY rec[-1] 0\n",   "M 0\n",  0),
    "Z_inv": ("H 0\nM !0\nCX rec[-1] 0\n",  "M 0\n",  1),
    "X":     ("MX 0\nCZ rec[-1] 0\n",       "MX 0\n", 0),
    "X_inv": ("MX !0\nCZ rec[-1] 0\n",      "MX 0\n", 1),
    "Y":     ("MY 0\nCX rec[-1] 0\n",       "MY 0\n", 0),
    "Y_inv": ("MY !0\nCZ rec[-1] 0\n",      "MY 0\n", 1),
}


@pytest.mark.parametrize("case", sorted(_SELF_CASES))
def test_twirl_feedback_onto_own_control_wire_matches_stim(case):
    """`M q; CX rec[-1] q` leaves q in a deterministic eigenstate: Z/X/Y, with `!`.

    The source record is a fair coin; the re-read of q after the self-targeting
    feedback is deterministic (0, or 1 for an inverted source), so
    `DETECTOR rec[-1]` is a deterministic channel on every engine.  The idle
    wire 1 is measured too so the circuit has a second record to compare against.
    """
    head, reread, value = _SELF_CASES[case]
    text = "R 0 1\n" + head + reread + "M 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
    shots = 400

    meas = _exact_measurements(text, shots)
    assert 0 < meas[:, 0].sum() < shots, f"{case}: oracle sanity: source record not random"
    assert (meas[:, 1] == value).all(), (
        f"{case}: oracle sanity: exact-engine re-read of q must be {value} on every shot")
    assert _stim_dets(text, shots).sum() == 0, f"{case}: oracle sanity: stim"

    dets = _twirl_dets(text, shots)
    assert dets[:, 0].sum() == 0, (
        f"{case}: after `M q; CX rec[-1] q` the re-read of q must be deterministic; the "
        f"twirl engine flipped it on {int(dets[:, 0].sum())}/{shots} shots")
    assert dets[:, 1].sum() == 0, f"{case}: the idle wire's detector must not move"


def test_twirl_retained_state_after_feedback_onto_own_control():
    """RETAINED state: after `M 0; CX rec[-1] 0` the wire is |0> — copied out, <Z> = +1.

    The control record is a fair coin (`H 0; M 0`), so both branches occur:
    rec = 0 leaves |0>, rec = 1 resets |1> -> |0>.  An output wire must be
    unmeasured, so the reset wire's Z-value is copied onto wire 2 with a plain
    `CX 0 2`; the exact engine's `M 2` is the per-shot oracle, and the twirl
    engine's materialized state must read <Z> = +1 on EVERY shot.

    Pre-fix this read +1 while the RECORD channel on the same wire (`M 2`,
    `test_twirl_feedback_onto_own_control_wire_matches_stim`) was a fair coin —
    the degenerate `CX q q` corrupted the record algebra, not the materialized
    ray.  Pinned so the copy-gadget rewrite is checked on the retained-state
    path too.
    """
    body = "R 0 1 2\nH 0\nM 0\nCX rec[-1] 0\nCX 0 2\nM 1\nDETECTOR rec[-1]\n"
    shots = 16
    meas = _exact_measurements(body + "M 2\n", 64)
    assert 0 < meas[:, 0].sum() < 64, "oracle sanity: control record not random"
    assert (meas[:, 2] == 0).all(), "oracle sanity: exact engine must read the reset wire as 0"

    s = _inner(compile_twirl_sampler(body + "OUTPUT_QUBITS out 2\n", p_factor=1.0, selfcheck=0))
    barrier = s.sample_barrier(shots, 0)
    wire = list(s.output_wires())[0]
    zs = [float(barrier.materialize(i).pauli_expectation_z(wire)) for i in range(shots)]
    assert all(z == pytest.approx(+1.0) for z in zs), (
        f"retained <Z> after measure-and-conditionally-reset must be +1 on every shot; got {zs}")


_SELF_REUSE_TAILS = {
    # the wire is REUSED after the conditional reset; every parity below is
    # deterministic in Stim (value 0 as the reset leaves |0>)
    "bell_pair":      "H 0\nCX 0 1\nM 0\nM 1\nDETECTOR rec[-1] rec[-2]\n",
    "second_feedback": "CX rec[-1] 1\nM 0\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] rec[-3]\n",
    "reset_again":    "H 0\nM 0\nCX rec[-1] 0\nM 0\nM 1\nDETECTOR rec[-2]\n",
}


@pytest.mark.parametrize("tail", sorted(_SELF_REUSE_TAILS))
def test_twirl_feedback_onto_own_control_then_wire_reused(tail):
    """Measure-and-conditionally-reset followed by REUSE of q.

    The reset wire feeds a Bell pair, a second feedback off the same record
    (which must still read the ORIGINAL record, not the reset wire), or a second
    conditional reset; each detector is deterministically 0 in Stim.
    """
    text = "R 0 1\nH 0\nM 0\nCX rec[-1] 0\n" + _SELF_REUSE_TAILS[tail]
    shots = 400
    meas = _exact_measurements(text, shots)
    assert 0 < meas[:, 0].sum() < shots, f"{tail}: oracle sanity: source record not random"
    assert _stim_dets(text, shots).sum() == 0, f"{tail}: oracle sanity: stim"

    dets = _twirl_dets(text, shots)
    assert dets.sum() == 0, (
        f"{tail}: every detector after the conditional reset is deterministic; the twirl "
        f"engine set {int(dets.sum())} bits over {shots} shots")


@pytest.mark.parametrize("p", [0.3, 1.0])
def test_twirl_feedback_onto_own_control_noisy_source(p):
    """`M(p) q; CX rec[-1] q` on |0>: the wire ends in |flip>, tracking the RECORD.

    The record is the readout flip f, the X fires iff f, so the re-read equals
    the record shot for shot (parity detector deterministically 0) while the
    control's projected state was never disturbed by the fold.  This case took
    the copy gadget already (the noisy fold is undone on an observed control) and
    was green before the self-target fix; it is pinned so the two rules cannot
    drift apart.
    """
    text = (f"R 0 1\nM({p}) 0\nCX rec[-1] 0\nM 0\nM 1\n"
            "DETECTOR rec[-2] rec[-3]\nDETECTOR rec[-1]\n")
    shots = 400 if p >= 1.0 else 2000
    meas = _exact_measurements(text, shots)
    assert (meas[:, 0] == meas[:, 1]).all(), "oracle sanity: re-read must equal the record"
    if p >= 1.0:
        assert meas[:, 0].sum() == shots, "oracle sanity: p=1 always flips"
    else:
        assert 0 < meas[:, 0].sum() < shots, "oracle sanity: flip never fired"
    assert _stim_dets(text, shots).sum() == 0, "oracle sanity: stim"

    dets = _twirl_dets(text, shots)
    assert dets.sum() == 0, (
        f"p={p}: noisy self-target feedback must track the record; got "
        f"{int(dets.sum())} set bits over {shots} shots")
