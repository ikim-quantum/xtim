"""Named refusals: what xtim refuses, by name, and how to fix each.

A refusal means the request is outside xtim's model; nothing is silently approximated. This script triggers the
common ones on tiny circuits and prints the name: RF_NO_OUTPUT, RF_OUTPUT_MEASURED and RF_OUTPUT_NOT_STRICT
(`xtim.XtimFidelityRefusal`, `.code`), OUTSIDE_T_DEPTH_1 and EXPECTATIONS_NOT_SERVED (`xtim.XtimTwirlRefusal`,
`.twirl_class`). The full table is in docs/usage.md.

Run:  python examples/07_refusals.py
"""
import xtim

TELEPORT = """RX 0
T 0
R 1 2
H 1
CX 1 2
CX 0 1
H 0
M 0 1
"""


def show(label, fn):
    try:
        fn()
    except xtim.XtimFidelityRefusal as e:
        print(f"{label:28s} XtimFidelityRefusal {e.code}")
    except xtim.XtimTwirlRefusal as e:
        print(f"{label:28s} XtimTwirlRefusal {e.twirl_class.name}")
    else:
        raise AssertionError(label + " was not refused")


def fidelity(text):
    return xtim.RecordFidelity(xtim.compile_twirl_sampler(text, selfcheck=0))


if __name__ == "__main__":
    # 1. no OUTPUT_QUBITS: the register fidelity has nothing to score
    show("no OUTPUT_QUBITS", lambda: fidelity(TELEPORT))
    # 2. an output qubit that is measured (move it onto a fresh wire with a noiseless R + SWAP instead)
    show("measured output qubit", lambda: fidelity(TELEPORT + "OUTPUT_QUBITS out 0\n"))
    # 3. an open frame: the teleported state depends on the outcomes until Pauli feedback closes it
    show("open frame (teleport)", lambda: fidelity(TELEPORT + "OUTPUT_QUBITS out 2\n"))
    try:
        fidelity(TELEPORT + "OUTPUT_QUBITS out 2\n")
    except xtim.XtimFidelityRefusal as e:
        print("   ", str(e).split(". (Diagnostic")[0].split(": ", 2)[2])
    closed = TELEPORT + "CX rec[-1] 2\nCZ rec[-2] 2\nOUTPUT_QUBITS out 2\n"
    s = xtim.compile_twirl_sampler(closed, p_factor=0.0, selfcheck=0)
    F = xtim.RecordFidelity(s).evaluate(s.sample_barrier(100, seed=1))
    assert abs(F.min() - 1.0) < 1e-12
    print("    closed with CX rec[-1] 2 / CZ rec[-2] 2: F = 1 on every noiseless shot")
    # 4. non-Clifford gates that do not fold into one commuting pi/8 layer
    show("outside T-depth 1", lambda: xtim.compile_twirl_sampler("RX 0\nT 0\nH 0\nT 0\n", selfcheck=0))
    # 5. per-shot expectation values are not a twirl-engine output (score the end state with RecordFidelity)
    show("expectations requested", lambda: xtim.Circuit("RX 0\nT 0\nMX 0\nDETECTOR rec[-1]\n")
         .compile_detector_sampler(seed=1).sample(10, return_expectations=True))
