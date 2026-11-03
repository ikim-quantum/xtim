"""J-b1 F (F item 4): the oracle tests' RAW exact-engine calls, inside the explicit-exact scope. These tests build exact
programs ON PURPOSE (the DEM export / the exact sampler as the oracle); the session counter (tests/conftest.py) counts
only constructions outside a scope, i.e. a silent exact route in the library."""
from xtim import _xtim
from xtim._exact_scope import explicit_exact


def export_dem_text(*a, **k):
    with explicit_exact():
        return _xtim.export_dem_text(*a, **k)
