"""Five small xtim-side items (fix/xtim-small-items, 2026-10-12).

1. PartitionedTwirlSampler never samples the all-zeros driving pattern on a caller's behalf:
   sample_barrier (and any other sampling delegate) is refused by name; sampler_for_pattern(bits)
   is the per-pattern escape. The IR has no record-sourced IF (an IF reads an INPUT_BITS port
   bit, cpp/include/qeccore/circuit_ir.hpp), so no pattern is correct without the caller's bits.
2. sample(decision_bits=...) on a detector-free IF circuit returns the engine's (n, 0) shape
   (dec-review Minor-9: a padded (n, 1) buffer made the scatter fail).
3. compile_exact_branches / compile_reference / `python -m xtim state verify` refuse an
   unresolved IF by name (they ran every IF body unconditionally); bits-supplied use unchanged.
4. A zero-probability noise instruction is the identity channel: in the twirl record samplers a
   pending Clifford frame passes through it (`H 1; Z_ERROR(0) 1; H 1` cancels), so the circuit
   keeps the diagonal residual class and sample_barrier accepts it; every byte equals the twin
   with the p=0 line deleted. The exact / forced path keeps the flushed form (a forced Pauli at
   the p=0 site stays at the user's position).
5. The non-commuting-magic bare-state refusal renders its terms on user qubits.
"""
from __future__ import annotations

import numpy as np
import pytest

import xtim
from xtim import _xtim
from xtim.twirl import PartitionedTwirlSampler

ANCHOR = "R 9\nM 9\nDECISION(0) rec[-1]\nDETECTOR rec[-1]\n"

# Two driving bits, two non-Pauli fix-ups on a retained wire: the all-zeros pattern is NOT the law
# of a shot with drv = (1, *) or (*, 0).
IF_PORT = ("INPUT_BITS drv 2\nRX 0\nX_ERROR(0.05) 0\nIF drv[0] {\nZ 0\n}\nIF !drv[1] {\nS 0\n}\n"
           "RX 1\nCX 0 1\nM 1\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0\n")
# Detector-free (a DECISION is the channel that lets the twirl engine compile it): the Minor-9 shape.
IF_NODET = ("INPUT_BITS b 1\nR 0\nX_ERROR(0.1) 0\nIF b[0] {\nX 0\n}\nM 0\nDECISION(0) rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1]\n")


def _inner(text):
    return _xtim.TwirlSampler(text, 1.0, "", 0, "")


# ── 1. sampling delegates refuse by name ─────────────────────────────────────────────────────

def test_partitioned_sample_barrier_refuses_by_name():
    ps = xtim.compile_twirl_sampler(IF_PORT)
    assert isinstance(ps, PartitionedTwirlSampler)
    assert hasattr(ps, "sample_barrier")           # adaptq's `s.sample_barrier(shots, seed)` shape
    with pytest.raises(ValueError) as ei:
        ps.sample_barrier(16, 1)
    msg = str(ei.value)
    for needle in ("PartitionedTwirlSampler.sample_barrier", "IF drv[0]", "IF drv[1]",
                   "2 driving bit(s)", "all-zeros driving pattern", "sampler_for_pattern",
                   "decision_bits"):
        assert needle in msg, (needle, msg)
    with pytest.raises(ValueError, match="sample_barrier"):
        getattr(ps, "sample_barrier")(16, seed=1)
    # Pure metadata still delegates.
    assert ps.num_detectors == 0 and ps.num_observables == 0 and ps.num_decisions == 1
    assert list(ps.output_wires()) and isinstance(ps.record_layout(), dict)   # J-b1 F: channel_report removed


def test_sampler_for_pattern_is_the_resolved_compile_and_patterns_differ():
    ps = xtim.compile_twirl_sampler(IF_PORT)
    means = {}
    for bits in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        inner = ps.sampler_for_pattern(bits)
        direct = _inner(_xtim.resolve_branches_text(IF_PORT, list(bits)))
        bi, bd = inner.sample_barrier(400, 3), direct.sample_barrier(400, 3)
        assert np.asarray(bi.decisions()).tobytes() == np.asarray(bd.decisions()).tobytes(), bits
        w = list(inner.output_wires())
        reads = [([w[0]], [], 0), ([], [w[0]], 0)]
        assert (np.asarray(bi.pauli_expectations_all(reads)).tobytes()
                == np.asarray(bd.pauli_expectations_all(reads)).tobytes()), bits
        means[bits] = float(np.asarray(bi.pauli_expectations_all(reads)).real[:, 0].mean())
    # The Z fix-up (drv[0]) flips <X0> of the retained |+>: (1,1) is not (0,1)'s law.
    assert means[(0, 1)] > 0.5 > -0.5 > means[(1, 1)]
    assert ps.sampler_for_pattern((0, 0)) is ps.sampler_for_pattern(np.zeros(2, np.uint8))
    with pytest.raises(ValueError, match="sampler_for_pattern"):
        ps.sampler_for_pattern((0,))


# ── 2. detector-free IF circuit: engine shapes ───────────────────────────────────────────────

def test_partitioned_sample_zero_detectors_matches_engine_shapes_and_bytes():
    ps = xtim.compile_twirl_sampler(IF_NODET)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(50, 1)).astype(np.uint8)
    r = ps.sample(50, seed=1, decision_bits=bits)      # fd4f31c: shape mismatch (25,0)->(25,1)
    d, o, dec = r.dets, r.obs, r.decisions
    eng = _inner(_xtim.resolve_branches_text(IF_NODET, [0])).sample(5, 1)
    assert d.shape == (50, 0) and d.dtype == np.uint8 and o.shape == (50, 1) and dec.shape == (50, 1)
    assert [np.asarray(a).shape for a in eng] == [(5, 0), (5, 1), (5, 1)]
    # Group-wise bytes == the per-pattern engine with the wrapper's documented seed derivation.
    groups: dict = {}
    for i in range(50):
        groups.setdefault(bytes(bits[i].tolist()), []).append(i)
    for p_idx, (key, idx) in enumerate(groups.items()):
        gs = (1 + p_idx * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        _dg, og, dg = _inner(_xtim.resolve_branches_text(IF_NODET, list(key))).sample(len(idx), gs)
        assert o[idx].tobytes() == np.asarray(og).tobytes() and dec[idx].tobytes() == np.asarray(dg).tobytes()
    b = ps.sample(50, seed=1, decision_bits=bits, return_mask=True)
    assert b.dets.shape == (50, 0) and b.live_mask.shape == (50, 0) and b.obs.tobytes() == o.tobytes()


# ── 3. compile_exact_branches / compile_reference / CLI state verify ─────────────────────────

def test_compile_exact_branches_refuses_unresolved_if_bits_supplied_unchanged():
    with pytest.raises(ValueError, match="compile_exact_branches: the circuit has IF blocks on INPUT_BITS"):
        xtim.Circuit(IF_NODET).compile_exact_branches()
    for bit in (0, 1):
        law = xtim.Circuit(_xtim.resolve_branches_text(IF_NODET, [bit])).compile_exact_branches().exact_branches([])
        assert law.num_rows >= 1


def test_compile_reference_and_cli_state_verbs_refuse_unresolved_if(tmp_path, capsys):
    from xtim.__main__ import main
    from xtim.reference import compile_reference
    with pytest.raises(ValueError, match="compile_reference: the circuit has IF blocks on INPUT_BITS"):
        compile_reference(IF_NODET)
    circ = tmp_path / "if.stim"
    circ.write_text(IF_NODET)
    plain = tmp_path / "plain.stim"
    plain.write_text(_xtim.resolve_branches_text(IF_NODET, [0]))
    ref = tmp_path / "plain.ref"
    assert main(["state", "compile", str(plain), str(ref)]) == 0
    assert main(["state", "verify", str(plain), str(ref)]) == 0
    assert main(["state", "compile", str(circ), str(tmp_path / "if.ref")]) == 1
    assert "compile_reference: the circuit has IF blocks on INPUT_BITS (IF b[0])" in capsys.readouterr().err
    assert main(["state", "verify", str(circ), str(ref)]) == 1
    assert "state verify: the circuit has IF blocks on INPUT_BITS (IF b[0])" in capsys.readouterr().err


# ── 4. p=0 noise under a pending frame (twirl family) ────────────────────────────────────────

ZERO_CHANNELS = ["Z_ERROR(0) 1", "X_ERROR(0) 1", "Y_ERROR(0) 1", "DEPOLARIZE1(0) 1",
                 "PAULI_CHANNEL_1(0,0,0) 1", "PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) 0 1",
                 "DEPOLARIZE2(0) 0 1"]


def _barrier_bytes(text, shots=2000, seed=5):
    T = _inner(text)
    b = T.sample_barrier(shots, seed)
    ow = list(T.output_wires())
    reads = [([w], [], 0) for w in ow] + [([], [w], 0) for w in ow] + [([w], [w], 1) for w in ow]
    r = T.sample(shots, seed + 1)
    return (np.asarray(b.decisions()).tobytes(), np.asarray(b.pauli_expectations_all(reads)).tobytes(),
            np.asarray(r[0]).tobytes(), np.asarray(r[2]).tobytes(),
            T.record_report()["term_route"]["plans_kind2"])          # J-b1 F: was channel_report()["ppr_plans"]


@pytest.mark.parametrize("ch", ZERO_CHANNELS)
def test_zero_noise_under_frame_equals_deleted_line(ch):
    """`X_ERROR(0.3) 1; T_DAG 1; H 1; <p=0 channel>; H 1` with OUTPUT_QUBITS: refused at
    sample_barrier on fd4f31c (the flushed H pair turned the X atom's residual off-diagonal ->
    CH/PPR routing, no state retention). The p=0 channel is the identity, so every byte must equal
    the twin without that line (which fd4f31c accepts and samples with diagonal plans)."""
    head = ANCHOR + "RX 0 1\nX_ERROR(0.3) 1\nT_DAG 1\nH 1\n"
    tail = "H 1\nCX 1 0\nOUTPUT_QUBITS out 0 1\n"
    with_line, without = head + ch + "\n" + tail, head + tail
    a, b = _barrier_bytes(with_line), _barrier_bytes(without)
    assert a == b
    assert a[-1] == 0                       # diagonal plans, not PPR


def test_zero_noise_under_frame_detector_stream_equals_deleted_line():
    """Same identity on a detector circuit (both accepted on fd4f31c, one PPR-routed)."""
    head = ANCHOR + "RX 1\nX_ERROR(0.3) 1\nT_DAG 1\nH 1\n"
    tail = "H 1\nT 1\nMX 1\nDETECTOR rec[-1]\n"
    s1, s2 = xtim.compile_twirl_sampler(head + "Z_ERROR(0) 1\n" + tail), xtim.compile_twirl_sampler(head + tail)
    r1, r2 = s1.sample(3000, seed=9), s2.sample(3000, seed=9)
    assert np.asarray(r1[0]).tobytes() == np.asarray(r2[0]).tobytes()
    assert np.asarray(r1.decisions).tobytes() == np.asarray(r2.decisions).tobytes()
    assert (s1.record_report()["term_route"]["plans_kind2"] == 0 ==
            s2.record_report()["term_route"]["plans_kind2"])          # J-b1 F: was ppr_plans


def test_nonzero_noise_under_frame_commutes_relabelled():
    """A p>0 channel between the H pair: the pending H commutes THROUGH it with the channel
    conjugated (H Z H = X), so the pair cancels and the circuit is the twin with X_ERROR(0.1) in
    place of `H; Z_ERROR(0.1); H` — byte for byte, diagonal plans. (Until fix/noncommuting-magic-
    readout the frame was flushed here and sample_barrier refused: 'state retention requires a
    diagonal-class circuit'.)"""
    head = ANCHOR + "RX 1\nX_ERROR(0.3) 1\nT_DAG 1\n"
    tail = "OUTPUT_QUBITS out 1\n"
    a = _barrier_bytes(head + "H 1\nZ_ERROR(0.1) 1\nH 1\n" + tail)
    b = _barrier_bytes(head + "X_ERROR(0.1) 1\n" + tail)
    assert a == b
    assert a[-1] == 0                       # diagonal plans, not PPR


def test_forced_fault_at_zero_site_under_frame_is_at_the_user_position():
    """A Pauli forced at the p=0 site between the H pair acts BETWEEN the H's (Z there is X after
    the pair), exactly as the dense forced oracle says: the forced path commutes the frame through
    the site with its letters relabelled (Instr::origin_letter), never as a transparent frame
    (which would fire the user letter on the wrong side of the pair)."""
    pytest.importorskip("stim")
    from _forced_oracle import aggregate, branches, law_shifts
    text = "RX 0\nX_ERROR(0.3) 0\nT 0\nH 0\nZ_ERROR(0) 0\nH 0\nT 0\nMX 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    sim = xtim.Circuit(text).compile_exact_branches()
    assert [s["probs"] for s in sim.noise_sites()][1] == [0.0]
    cands = None
    for p in "XYZ":
        law = sim.exact_branches([(1, 0, p)])
        cands = law_shifts(law, aggregate(branches(text, 1, {(1, 0): p}, set())), cands)
        assert cands, p


# ── 5. non-commuting magic refusal in user terms ─────────────────────────────────────────────

ZREAD_HY = (ANCHOR + "RX 0 1 2 3\nX_ERROR(1) 0\nT 0\nCZ 0 1\nM 2\nDECISION(1) rec[-1]\nMX 0\n"
            "DECISION(2) rec[-1]\nT 3\nCZ rec[-2] 3\nH 3\nT 3\nCY rec[-1] 2\nMY 2\nDECISION(3) rec[-1]\n"
            "OUTPUT_QUBITS out 1 3\n")


def test_non_commuting_magic_refusal_names_user_qubits():
    with pytest.raises(ValueError) as ei:
        xtim.compile_twirl_sampler(ZREAD_HY)
    msg = str(ei.value)
    assert "bare state rejected: non-commuting magic: X15 and X14Z15 anticommute" in msg
    assert "; in user qubits: " in msg
    assert "(deferred wire 15)" in msg and "(deferred wire 14)" in msg
    # Wires 14/15 are q2/q3's initial RX incarnations (the coherentizer's copy ancilla is wire 10).
    assert "X on q3 (deferred wire 15) and X on q2 (deferred wire 14) * Z on q3 (deferred wire 15)" in msg
    assert "[after its reset]" not in msg
    assert "T 3 (#15), T 3 (#18)" in msg and "T 0 (#" not in msg     # the gates on the named qubits only
    # A single-qubit case: `RX 0` is the qubit's first incarnation (fresh wire 1, named plainly);
    # RX desugars to R;H so the flat indices of the three T's are 2, 4, 6.
    with pytest.raises(ValueError) as ei2:
        xtim.compile_twirl_sampler("RX 0\nT 0\nH 0\nT 0\nH 0\nT 0\nM 0\nDETECTOR rec[-1]\n")
    m2 = str(ei2.value)
    assert "non-commuting magic: X1 and Z1 anticommute; in user qubits: " in m2
    assert "X on q0 (deferred wire 1) and Z on q0 (deferred wire 1)" in m2
    assert "T 0 (#2), T 0 (#4), T 0 (#6)" in m2 and "[after its reset]" not in m2


# ── fix round 1 (review small-review.md) ─────────────────────────────────────────────────────

def test_compile_twirl_sampler_from_ideal_state_refuses_unresolved_if():
    """Important-1: the from-state constructor compiled IF bodies unconditionally. J-b1 F: its successor
    compile_twirl_sampler_from_ideal_state refuses them by name too."""
    from xtim import twirl as _twirl
    ideal = _twirl.input_state_of("R 0\nOUTPUT_QUBITS out 0\n")
    with pytest.raises(ValueError, match="compile_twirl_sampler_from_ideal_state: the circuit has IF blocks on INPUT_BITS"):
        _twirl.compile_twirl_sampler_from_ideal_state(IF_NODET, ideal_state=ideal)


def test_partitioned_delegation_is_an_allow_list(monkeypatch):
    """Minor-7: metadata names delegate; every other inner attribute is denied BY NAME, including a
    sampling method that does not exist yet."""
    ps = xtim.compile_twirl_sampler(IF_PORT)
    assert isinstance(ps.record_layout(), dict)                  # J-b1 F: sig_bits retired
    assert isinstance(ps.born_dec(), dict) and isinstance(ps.channel_one_counts(), list)
    with pytest.raises(AttributeError):
        ps.no_such_attribute
    monkeypatch.setattr(_xtim.TwirlSampler, "sample_future", lambda self, shots: shots, raising=False)
    assert _xtim.TwirlSampler.sample_future is not None
    with pytest.raises(ValueError, match="PartitionedTwirlSampler.sample_future"):
        ps.sample_future(3)


def test_dem_zero_noise_under_frame_equals_deleted_line():
    """Important-3: the DEM export (a ChannelSupport consumer) shares the transparent frame, so the
    circuit the review found over-refused ("outside the simulable class at deferred-stream gate
    index 4") exports the SAME DEM as its twin without the p=0 line."""
    head = ANCHOR + "RX 0 1\nX_ERROR(0.3) 1\nT_DAG 1\nH 1\n"
    tail = "H 1\nOUTPUT_QUBITS out 0 1\n"
    a = xtim.Circuit(head + "Z_ERROR(0) 1\n" + tail).detector_error_model_text()
    b = xtim.Circuit(head + tail).detector_error_model_text()
    assert a == b and "error(" in a


def test_fold_across_zero_noise_born_law_equals_deleted_line():
    """Declared byte movement (barfuzz/197 class): a frame that folds across a p=0 line into a read
    changes the deferred read form (`H; N0; MY` -> folded read), so the Born-decision BYTES of a
    base-accepted circuit may move; the LAW is the deleted-line twin's (both exact)."""
    head = ANCHOR + "RX 0\n"
    tail = "T 0\nX_ERROR(0.2) 0\nMY 0\nDECISION(1) rec[-1]\nRX 1\nOUTPUT_QUBITS out 1\n"
    means = []
    for t in (head + "DEPOLARIZE1(0) 0\n" + tail, head + tail):
        s = _inner(t)
        b = s.sample_barrier(20000, 11)
        d = np.unpackbits(np.asarray(b.decisions(), np.uint8).reshape(20000, -1), axis=1, bitorder="little")[:, 1]
        means.append(d.mean())
    assert abs(means[0] - means[1]) < 5 * np.sqrt(0.25 / 20000) * np.sqrt(2) + 1e-9
    assert 0.1 < means[0] < 0.9        # a genuinely Born decision, not a deterministic one


def test_non_commuting_magic_refusal_flip_ancilla_and_desugar_labels():
    """Minor-5/6: a held readout-flip ancilla is not a copy ancilla; MPP desugar wires are not user qubits."""
    flip = ("RX 0 1\nT 0\nM(0.1) 0\nCZ rec[-1] 1\nH 0\nT 0\nT 1\nH 1\nT 1\nM 0\nDETECTOR rec[-1]\n"
            "M 1\nDETECTOR rec[-1]\n")
    with pytest.raises(ValueError) as ei:
        xtim.compile_twirl_sampler(flip)
    m = str(ei.value)
    assert "readout-flip ancilla of q0's measurement #0 (coherentized M(p))" in m
    assert "copy of q0 (coherentized-feedback ancilla)" in m
    desugar = "RX 0 1\nMPP X0*X1\nT 0\nH 0\nT 0\nM 0\nDETECTOR rec[-1]\n"
    with pytest.raises(ValueError) as ei2:
        xtim.compile_twirl_sampler(desugar)
    m2 = str(ei2.value)
    assert "X on desugar ancilla wire 2 (MPP/SPP gadget) (deferred wire 2) * X on q0 (deferred wire 3)" in m2
    assert "T 0 (#8), T 0 (#10)" in m2
