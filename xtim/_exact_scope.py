"""J-b1 P13 (plan r3.2): the explicit-exact scope. Every explicit exact entry point (``engine="exact"``,
``compile_sampler``, ``compile_exact_branches``, the DEM export, ``detector_determinism``) constructs its exact
program inside ``explicit_exact()``; an exact construction OUTSIDE it bumps ``_xtim._exact_constructions()`` (a silent
exact route). The counter never changes an output.

J-b1 final review (MAJOR-4): the test-API extension ``xtim._xtim_testapi`` links its own copy of the engine, with its
own counter and scope. When that module is loaded (tests only; production never imports it), ``explicit_exact()``
enters both scopes and ``exact_constructions()`` sums both counters."""
import sys
from contextlib import contextmanager

from . import _xtim


def _modules():
    ta = sys.modules.get("xtim._xtim_testapi")
    return (_xtim, ta) if ta is not None and hasattr(ta, "_exact_constructions") else (_xtim,)


@contextmanager
def explicit_exact():
    mods = _modules()
    for m in mods:
        m._explicit_exact_enter()
    try:
        yield
    finally:
        for m in reversed(mods):
            m._explicit_exact_exit()


def exact_constructions() -> int:
    return sum(int(m._exact_constructions()) for m in _modules())


def reset_exact_constructions() -> None:
    for m in _modules():
        m._reset_exact_constructions()
