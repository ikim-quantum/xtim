"""J-b1 P11a: the cache-owned kind-2 / route-1 header arena and TWPL v6 on the FAM sampler (plan §2.1 P11a, r2.4, r3.6).

The (G) plan headers of a family sampler — kind 2, and kind 0 outside the diagonal cache's pair universe (route 1) —
are OWNED by the plan cache (TwirlPlanCache's arena), no longer by the sampler with adopt_payload (deleted). Their
payloads join the cache's FIFO, so the cache owns the header and every FIFO pointer to it.

Gates:
  * M-K2OWN: the arena's headers are counted in the cache's header bytes; every kind-2 / route-1 plan is one arena
    header;
  * destroy-then-evict: the cache kept alive past its sampler evicts every resident payload, kind-2 included
    (MUT-K2-OWN, the old sampler-owned header, is a use-after-free here: red under ASan);
  * TWPL v6 (a FAM cache; production keeps v5 until F): one class tag per plan, keys only, kind 2 included; a
    warm (loaded) sampler is byte-identical to a cold one; a v5 (production) file loads nothing; a term-set-token
    mismatch loads nothing (MUT-TWPL6-NOTOKEN -> red). Loads are read off the cache's header counts before any
    shot.
"""
import gc
import os
import struct
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_ja_ppr_records import CH_NOISY, K0G_OPEN, biased_fuzz_records  # noqa: E402

import xtim  # noqa: E402
from xtim import _xtim  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
ROUTE1 = biased_fuzz_records(K0G_OPEN[0], True)       # reaches the kind-0-outside-the-universe route (route 1)
_ENGINE = _xtim.ENGINE_VERSION.encode() if hasattr(_xtim, "ENGINE_VERSION") else None


def fam(text, disk_path=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0, disk_path=disk_path)


def run(s, n=6000, seed=3):
    return [np.asarray(a) for a in s.sample(n, seed=seed)]


@pytest.mark.parametrize("name,text", [("ch_noisy", CH_NOISY), ("route1", ROUTE1)])
def test_arena_owns_the_g_headers(name, text):
    s = fam(text)
    run(s)
    rs, c = s.route_stats(), s.record_counters()
    assert c["arena_headers"] == rs["plans_kind2"] + rs["plans_kind0_general"] > 0, (rs, c)
    if name == "route1":
        assert rs["plans_kind0_general"] > 0, rs
    h = s._cache_handle()
    assert h.arena_size() == c["arena_headers"]
    assert c["header_bytes"] == h.header_bytes()


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


def _events(s):
    return [tuple(e) for e in s.take_disk_events()]


@pytest.mark.parametrize("name,text", [("ch_noisy", CH_NOISY), ("route1", ROUTE1)])
def test_twpl6_round_trip_warm_equals_cold(name, text, tmp_path):
    p = str(tmp_path / "plans.twpl")
    a = fam(text, p)
    cold = run(a)
    blob = open(p, "rb").read()
    assert blob[:4] == b"TWPL" and struct.unpack("<I", blob[4:8])[0] == 6
    ca = a.record_counters()
    b = fam(text, p)
    cb = b.record_counters()
    assert cb["arena_headers"] == ca["arena_headers"] > 0       # kind-2 / route-1 keys persisted
    assert cb["diagonal_headers"] == ca["diagonal_headers"]
    warm = run(b)
    for x, y in zip(cold, warm):
        assert np.array_equal(x, y)                    # warm == cold bytes
    assert b.record_counters()["arena_headers"] == ca["arena_headers"]


def _fnv1a64(data):
    h = 1469598103934665603
    for x in data:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def test_twpl6_term_token_mismatch_loads_nothing(tmp_path):
    p = str(tmp_path / "plans.twpl")
    a = fam(CH_NOISY, p)
    run(a)
    blob = bytearray(open(p, "rb").read())
    (elen,) = struct.unpack("<Q", blob[8:16])
    off = 16 + elen + 8                                # magic | version | len | engine | group token | TERM TOKEN
    (tt,) = struct.unpack("<Q", blob[off:off + 8])
    assert tt != 0                                     # an armed term view (ppr_records on a CH circuit)
    blob[off:off + 8] = struct.pack("<Q", tt ^ 1)
    blob[-8:] = struct.pack("<Q", _fnv1a64(bytes(blob[8:-8])))   # the FNV covers the payload: a well-formed file
    open(p, "wb").write(bytes(blob))
    b = fam(CH_NOISY, p)
    c = b.record_counters()
    assert c["diagonal_headers"] == 0 and c["arena_headers"] == 0
    # the untouched file loads (the hand-edited framing is otherwise valid)
    blob[off:off + 8] = struct.pack("<Q", tt)
    blob[-8:] = struct.pack("<Q", _fnv1a64(bytes(blob[8:-8])))
    open(p, "wb").write(bytes(blob))
    c2 = fam(CH_NOISY, p).record_counters()
    assert c2["arena_headers"] > 0


def test_twpl_version5_header_loads_nothing(tmp_path):
    """J-b2 J1b (plan r2.1 J1b): a well-formed file whose header says TWPL version 5 (a v6 file of the same circuit
    with only the version field rewritten and the FNV recomputed) loads nothing at a production sampler's path
    (version skew), then is replaced by v6. The legacy v5 writer retires at J2."""
    p = str(tmp_path / "plans.twpl")
    a = fam(CH_NOISY, p)
    run(a)
    blob = bytearray(open(p, "rb").read())
    assert struct.unpack("<I", blob[4:8])[0] == 6
    assert fam(CH_NOISY, p).record_counters()["arena_headers"] > 0          # the unmodified file loads
    blob[4:8] = struct.pack("<I", 5)
    blob[-8:] = struct.pack("<Q", _fnv1a64(bytes(blob[8:-8])))
    open(p, "wb").write(bytes(blob))
    s = fam(CH_NOISY, p)
    c = s.record_counters()
    assert c["diagonal_headers"] == 0 and c["arena_headers"] == 0
    run(s)
    assert struct.unpack("<I", open(p, "rb").read()[4:8])[0] == 6
