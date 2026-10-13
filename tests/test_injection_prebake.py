"""State-level injection pre-bake rule (3.1.5 fix): a consumer stage whose
fresh ancillas interact with the port and are then acted on again — two
syndrome rounds in one injected stage — compiles and reproduces the
monolithic circuit's deterministic records.  Before the fix every fresh-only
gate was pre-baked into the fresh block regardless of position, and the
composition's exactness guard refused such stages."""
import numpy as np

import xtim._xtim as _x

# a 2-qubit Bell pair as the carried port
PREFIX = "R 0 1\nH 0\nCX 0 1\n"


def _round(anc, k):
    """One X-type parity check of the pair on fresh ancilla `anc`: H, CX to both
    port qubits, H, M — the second H is a fresh-only Clifford AFTER a port gate."""
    return (f"R {anc}\nH {anc}\nCX {anc} 0\nCX {anc} 1\nH {anc}\nM {anc}\nDECISION({k}) rec[-1]\n")


STAGE_TWO_ROUNDS = ("INPUT_QUBITS in 0 1\n" + _round(2, 0) + _round(3, 1) + "OUTPUT_QUBITS out 0 1\n")


def test_two_syndrome_rounds_in_one_injected_stage_compile_and_agree():
    bare = _x.bare_state_of(PREFIX + "R 9\nM 9\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1\n")
    src = list(_x._output_wires_of(PREFIX + "R 9\nM 9\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1\n"))
    # before the 3.1.5 fix: ValueError "... failed its exactness guard ..."
    inj = _x.TwirlSampler(bare, src, STAGE_TWO_ROUNDS, 0.0, "", 0, False, "")
    mono = _x.TwirlSampler(PREFIX + STAGE_TWO_ROUNDS.replace("INPUT_QUBITS in 0 1\n", ""),
                           0.0, "", 0, False, "")
    bi = inj.sample_barrier(8, 1)
    bm = mono.sample_barrier(8, 2)
    di = np.unpackbits(np.asarray(bi.decisions(), np.uint8), axis=1, bitorder="little")[:, :2]
    dm = np.unpackbits(np.asarray(bm.decisions(), np.uint8), axis=1, bitorder="little")[:, :2]
    # both parity checks of a Bell pair are deterministic (+1): records all zero, both ways
    assert not di.any() and not dm.any()
    # the carried port survives both rounds unchanged: <XX> = <ZZ> = +1 on every shot
    for i in range(8):
        st = bi.materialize(i)
        ow = list(inj.output_wires())
        assert abs(st.pauli_expectation_xz(ow, []) - 1.0) < 1e-12
        assert abs(st.pauli_expectation_xz([], ow) - 1.0) < 1e-12


def test_one_round_still_classifies_identically():
    """A stage whose fresh-only gates all precede their port contact (the
    previously accepted shape) is unchanged by the rule."""
    bare = _x.bare_state_of(PREFIX + "R 9\nM 9\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1\n")
    src = list(_x._output_wires_of(PREFIX + "R 9\nM 9\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 0 1\n"))
    stage = "INPUT_QUBITS in 0 1\n" + _round(2, 0) + "OUTPUT_QUBITS out 0 1\n"
    inj = _x.TwirlSampler(bare, src, stage, 0.0, "", 0, False, "")
    b = inj.sample_barrier(4, 3)
    d = np.unpackbits(np.asarray(b.decisions(), np.uint8), axis=1, bitorder="little")[:, :1]
    assert not d.any()
