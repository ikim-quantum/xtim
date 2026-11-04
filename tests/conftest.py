"""J-b1 F (F item 4, plan r3.2): the exact-construction counter stays at its accounted value over the WHOLE suite.

Every exact-engine construction outside an explicit-exact scope (the DEM export; the test-API exact sampler / forced
branches / reference battery of tests/_exact_engine.py, exact-out T5; not ``detector_determinism``, a bare-state Born
check since twirl-routes T3) bumps
``_xtim._exact_constructions()``: a silent exact route. After F nothing in the library may take one, so the counter
must not move — except where a test builds an exact program through a RAW binding on purpose and accounts for it
(``account_exact_constructions``; tests/test_jb1_exact_counter.py). MUT-AUTO-FALLBACK (the thunk, a class-14 -> exact
branch, each reinstated in turn) turns this red.

J-b1 final review (MAJOR-4): ``exact_constructions()`` sums the production module's counter AND the test API's (which
links its own engine copy), so a silent exact route in the shared binding code is caught on either extension
(tests/test_jb1_engine_selection.py::test_carried_refusals_typed_without_exact; MUT-INJ-FALLBACK red).
"""
import importlib.util
import sys

import pytest

from xtim._exact_scope import exact_constructions

# The test API (xtim._xtim_testapi) is built only by in-place builds (`python setup.py build_ext --inplace`,
# `pip install -e .`); an installed wheel has none. Then every import of it -- a test module's top-level import or a
# helper's lazy one -- SKIPS that module / test with this reason instead of erroring. The one place this is decided.
TESTAPI_MISSING = ("xtim._xtim_testapi not built (wheel install): test-API tests need the in-place dev build "
                   "`python setup.py build_ext --inplace`")


class _SkipMissingTestAPI:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name == "xtim._xtim_testapi":
            pytest.skip(TESTAPI_MISSING, allow_module_level=True)
        return None


if importlib.util.find_spec("xtim._xtim_testapi") is None:
    sys.meta_path.insert(0, _SkipMissingTestAPI())

_ACCOUNTED = [0]
_PER_TEST = {}


def account_exact_constructions(n: int) -> None:
    """A test that builds exact programs through raw bindings ON PURPOSE declares how many it built."""
    _ACCOUNTED[0] += int(n)


@pytest.fixture(autouse=True)
def _exact_construction_attribution(request):
    a, acc = exact_constructions(), _ACCOUNTED[0]
    yield
    d = exact_constructions() - a - (_ACCOUNTED[0] - acc)
    if d:
        _PER_TEST[request.node.nodeid] = d


@pytest.fixture(scope="session", autouse=True)
def _exact_construction_counter_is_accounted():
    start = exact_constructions()
    yield
    moved = exact_constructions() - start - _ACCOUNTED[0]
    worst = sorted(_PER_TEST.items(), key=lambda kv: -kv[1])[:8]
    assert moved == 0, (
        f"{moved} exact-engine construction(s) outside an explicit-exact scope over this test session: a silent "
        "exact route (J-b1: every exact construction is explicit; see xtim/_exact_scope.py); by test: %s" % worst)
