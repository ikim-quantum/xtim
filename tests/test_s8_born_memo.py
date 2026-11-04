"""S8-4 (plan r2.7): the Born ROOT memo. A root node is p_+ of its direction on the freshly reset bare free register,
a pure function of (a, b, phase, par) on that trie's bare; the memo serves repeated root directions with the same
double. Checks: it is used (hits), every stored node equals its from-scratch definition bit for bit, sample bytes are
the same with the memo cleared on every shot boundary (a tiny payload budget) as without, and the selfcheck window
(every hit recomputed) passes."""
import os
import warnings

import numpy as np
import pytest

from xtim import _xtim_testapi as TA
from _private import private_fixture

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
TEXTS = {"color5": os.path.join(DATA, "magna_color5_fid.stim"),      # (kappa = 0 plans: memo unused)
         "factory": "private:ccz_factory_a.stim"}


def _fam(name, selfcheck=0, budget=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = TEXTS[name]
        p = private_fixture(p[8:]) if p.startswith("private:") else p
        s = TA.compile_family_sampler(open(p).read(), selfcheck=selfcheck)
    if budget is not None:
        s.set_plan_payload_budget(budget)
    return s


def _bytes(out):
    return b"".join(np.asarray(a).tobytes() for a in out)


def test_root_memo_hits_and_nodes_exact():
    s = _fam("factory")
    s.sample(6000, seed=7)
    bt = s.record_report()["born_trie"]
    assert bt["root_memo_hits"] > 0 and bt["node_evals"] + bt["root_memo_hits"] <= bt["nodes"]
    checked, bad = s.born_trie_check()                 # every stored node == its from-scratch definition
    assert checked > 0 and bad == 0


@pytest.mark.parametrize("name", ["color5", "factory"])
def test_bytes_unchanged_when_the_memo_is_cleared(name):
    """A tiny payload budget clears the root memos on every shot boundary (before payload eviction): the sample
    bytes equal the unbounded run's."""
    a = _bytes(_fam(name).sample(3000, seed=11))
    t = _fam(name, budget=1)
    b = _bytes(t.sample(3000, seed=11))
    assert a == b
    assert t.record_report()["plan_memory"]["evictions"] > 0


def test_selfcheck_window_recomputes_hits():
    s = _fam("factory", selfcheck=100000)               # every hit inside the window is recomputed and compared
    s.sample(4000, seed=7)
    assert s.record_report()["born_trie"]["root_memo_hits"] > 0
