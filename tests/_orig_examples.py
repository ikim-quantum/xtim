"""The ORIGINAL bundled example circuits, kept byte-identical in tests/data as fixtures (pins unchanged).

The bundled examples (xtim.load_example, examples/) are the fixed public versions (OUTPUT_QUBITS declared, frames
closed, no PAULI_EXPECTATION); tests that pin numbers on the originals read them from here.
"""
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
ORIG_EXAMPLES = ("ch_cultivation", "code_switching_faithful", "code_switching_rate", "cube_ccz",
                 "cultivation_d3_faithful", "cultivation_d3_rate", "cultivation_d5", "distillation_15_1_3",
                 "miniature_oracle")


def orig_path(name: str) -> Path:
    name = name[:-5] if name.endswith(".stim") else name
    assert name in ORIG_EXAMPLES, name
    return DATA / (name + ".stim")


def load_orig(name: str):
    """The original circuit as an xtim.Circuit (what xtim.load_example returned before the README rehaul)."""
    import xtim
    return xtim.Circuit.from_file(str(orig_path(name)))
