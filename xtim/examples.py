"""Bundled example circuits — loadable without a repo clone.

The demo `.stim` protocols ship *inside* the installed package (``xtim/_examples/``),
so a plain ``pip install``-ed wheel can load them directly — no clone, no path
juggling::

    c     = xtim.load_example("cube_ccz")          # an xtim.Circuit, ready to sample
    path  = xtim.example_path("cube_ccz")          # the underlying .stim file path
    names = xtim.list_examples()                   # what's available

Names are accepted with or without the ``.stim`` suffix. See ``examples/README.md``
for what each protocol is.
"""
from __future__ import annotations

import os
from importlib import resources

# The demo protocols bundled with the package (mirrors examples/README.md).
EXAMPLES = (
    "cube_ccz",                 # [[8,3,2]] transversal CCZ — multi-magic (chi>2)
    "cultivation_d3_faithful",  # d=3 magic-state cultivation (decoder DEM + reject)
    "cultivation_d3_rate",      # ...its Clifford twin (plain DEM)
    "cultivation_d5",           # d=5 magic-state cultivation (ref chi=2; syndrome fixture, no PAULI_EXPECTATION)
    "distillation_15_1_3",      # [[15,1,3]] qRM transversal-T (15-to-1 distillation prep)
    "code_switching_faithful",  # qRM <-> Steane code switching
    "code_switching_rate",      # ...count-matched twin
    "miniature_oracle",         # teleported T|+> in a small rep code
    "ch_cultivation",           # H-eigenstate cultivation via controlled-H checks (chi=2)
)


def list_examples() -> "list[str]":
    """The names of the bundled example circuits (any is valid for load_example)."""
    return list(EXAMPLES)


def _normalize(name: str) -> str:
    s = str(name)
    # A bundled example is referred to by NAME, not a path — reject separators
    # rather than silently stripping them (e.g. "../cube_ccz" -> "cube_ccz").
    if "/" in s or "\\" in s or os.sep in s:
        raise ValueError(
            "%r is not an example name — pass a bare name like 'cube_ccz' "
            "(see xtim.list_examples()), or use xtim.Circuit.from_file(path) "
            "for an arbitrary file" % name)
    return s[:-5] if s.endswith(".stim") else s


def example_path(name: str) -> str:
    """Filesystem path to a bundled example `.stim` (accepts ``"cube_ccz"`` or
    ``"cube_ccz.stim"``). Works from a pip-installed wheel — no repo clone needed.

    Raises ``ValueError`` naming the available examples if `name` is unknown."""
    key = _normalize(name)
    if key not in EXAMPLES:
        raise ValueError(
            "unknown example %r; available: %s" % (name, ", ".join(EXAMPLES)))
    fname = key + ".stim"
    # 1) the bundled package-data copy — the installed-wheel / sdist path.
    try:
        res = resources.files("xtim").joinpath("_examples").joinpath(fname)
        if res.is_file():
            return str(res)
    except (ModuleNotFoundError, AttributeError, OSError):
        pass
    # 2) dev-tree fallback: the repo's benchmarks/ (xtim/_examples/ is generated at
    #    export, so it doesn't exist when running straight from the source tree).
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    for cand in (os.path.join(repo, "benchmarks", fname),
                 os.path.join(repo, "examples", fname)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "example %r is registered but its .stim was not found — broken install?" % key)


def load_example(name: str) -> "Circuit":  # noqa: F821 (Circuit imported lazily)
    """Load a bundled example as an ``xtim.Circuit`` (no repo clone needed).

    >>> import xtim
    >>> c = xtim.load_example("cube_ccz")     # the [[8,3,2]] transversal-CCZ demo
    >>> c.num_qubits
    12
    """
    from .circuit import Circuit
    return Circuit.from_file(example_path(name))
