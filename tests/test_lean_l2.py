"""lean L2 (2026-11-04): sparse per-fault data. lean D1-S3 (owner-signed deletions): the L2 per-alternative records (the
streaming fast path's alt blob) are deleted with the streaming path — the rotation loop and the Clifford path read the
rotation records (twirl_rotation.hpp); test_alt_blob_is_sparse retired with them. Kept: the window / non-window agreement.
"""
import hashlib
import os

import numpy as np
import pytest

from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
WIDE = os.path.join(HERE, "..", "examples", "color5_cultivation.stim")   # n = 524 deferred wires
D5 = os.path.join(HERE, "..", "examples", "cultivation_d5.stim")
D3R = os.path.join(HERE, "data", "cultivation_d3_rate.stim")


def _h(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("path", [D3R, D5, WIDE], ids=["d3_rate", "d5", "color5"])
def test_streaming_route_equals_window_route(path):
    """A selfcheck = 0 sampler and a selfcheck > 0 sampler (the first shots through the window with its oracles) give the
    same bytes, and the same construction digest."""
    text = open(path).read()
    a = TA.compile_family_sampler(text, selfcheck=0)
    b = TA.compile_family_sampler(text, selfcheck=200)
    assert a._ctor_digest() == b._ctor_digest()
    assert _h(a.sample(2500, 9)) == _h(b.sample(2500, 9))
    assert _h(a.sample(2500, None)) == _h(b.sample(2500, None))
