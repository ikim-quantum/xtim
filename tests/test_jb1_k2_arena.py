"""J-b1 P11a: the cache-owned kind-2 / route-1 header arena on the FAM sampler (plan §2.1 P11a, r2.4, r3.6).

The (G) plan headers of a family sampler — kind 2, and kind 0 outside the diagonal cache's pair universe (route 1) —
are OWNED by the plan cache (TwirlPlanCache's arena), no longer by the sampler with adopt_payload (deleted). Their
payloads join the cache's FIFO, so the cache owns the header and every FIFO pointer to it.

Gates:
  * M-K2OWN: the arena's headers are counted in the cache's header bytes; every kind-2 / route-1 plan is one arena
    header;
  * destroy-then-evict: the cache kept alive past its sampler evicts every resident payload, kind-2 included
    (MUT-K2-OWN, the old sampler-owned header, is a use-after-free here: red under ASan).
(bloat-cut C: the TWPL v6 disk layer and its tests are deleted.)
"""
import gc
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_ja_ppr_records import CH_NOISY  # noqa: E402

import xtim  # noqa: E402,F401
from xtim import _xtim_testapi as TA  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


# RE-PINNED lean D1-S3 (owner-signed): the CH / PPR circuits' shots are served on the rotation loop; the term route — whose
# (G) headers the arena owns — serves the framed / branch / all-term recording shots, so the arena is driven here by the
# all-term recording option (record_route="term": every shot term-routed, kind 2 on the raw (P, S)). Route 1 (kind 0
# outside the pair universe) is no longer produced: its case is retired.
def fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0, record_route="term")


def run(s, n=6000, seed=3):
    return [np.asarray(a) for a in s.sample_records(n, seed=seed)[:3]]


@pytest.mark.parametrize("name,text", [("ch_noisy", CH_NOISY)])
def test_arena_owns_the_g_headers(name, text):
    s = fam(text)
    run(s)
    rs, c = s.route_stats(), s.record_counters()
    assert c["arena_headers"] == rs["plans_kind2"] + rs["plans_kind0_general"] > 0, (rs, c)
    h = s._cache_handle()
    assert h.arena_size() == c["arena_headers"]
    # lean D1-S2: record_counters' header_bytes also counts the rotation store's plans (a rotation-loop circuit)
    st = TA.rot_store_stats(s)
    assert c["header_bytes"] == h.header_bytes() + (st["plan_bytes"] if st["on"] else 0)


def test_destroy_then_evict():
    s = fam(CH_NOISY)
    run(s)
    rs = s.route_stats()
    assert rs["builds_kind2"] > 0
    h = s._cache_handle()
    assert h.resident_bytes() > 0
    del s
    gc.collect()
    assert h.evict_all() > 0                           # touches every FIFO entry, kind-2 headers included
    assert h.resident_bytes() == 0

