"""S8-0b: the construction-product digest (G-PREP, plan r2.3) and the materialize read-only contract (plan r2.1 N3).

_ctor_digest (test API) is the sha256 of every sampler construction product: the bare state, the certified
generators, the propagation table, the per-alternative fired lists / AltAtom contents / AltMasks / flat blob, the
pair universe, the channel rows and masks. G-PREP compares it to the parent's on the corpus at every S8 commit (a
mismatch is a STOP); here: it is deterministic and it separates circuits.
"""
import os
import warnings

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA
from _private import private_fixture

pytestmark = pytest.mark.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def test_ctor_digest_deterministic_and_separating():
    c5 = open(os.path.join(DATA, "magna_color5_fid.stim")).read()
    rp = open(os.path.join(DATA, "magna_rp2_fid.stim")).read()
    d = _fam(c5)._ctor_digest()
    assert len(d) == 64 and d == _fam(c5)._ctor_digest()
    assert _fam(rp)._ctor_digest() != d
    # one more noisy location (after the first line): new atoms, new alternatives
    head, rest = c5.split("\n", 1)
    assert _fam(head + "\nX_ERROR(0.001) 0\n" + rest)._ctor_digest() != d


def test_materialize_leaves_sampling_bytes_unchanged():
    """N3: materialize is read-only on the sampler (copy-then-project): the sampler's later samples are the same bytes
    with and without materialize calls in between."""
    text = open(os.path.join(DATA, "magna_color5_fid.stim")).read()
    out = []
    for touch in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = xtim.compile_twirl_sampler(text, selfcheck=0)
        buf = s.sample_barrier(64, seed=5)
        if touch:
            for i in range(0, 64, 3):
                buf.materialize(i)
        out.append(b"".join(np.asarray(a).tobytes() for a in s.sample(2000, seed=9)))
    assert out[0] == out[1]


@pytest.mark.parametrize("name", ["magna_color5_fid.stim", "magna_rp2_fid.stim", "private:ccz_factory_a.stim",
                                  "private:ccz_factory_a_out.stim"])
def test_alt_contents_equal_compose_fired(name):
    """S8-1' (5-a): every diagonal alternative's sparse content (S/Z values, CZ pairs, X support) composed from its
    atoms' sparse contents equals compose_fired + finalize (the dense chain), alternative by alternative."""
    path = private_fixture(name[8:]) if name.startswith("private:") else os.path.join(DATA, name)
    checked, bad = _fam(open(path).read())._alt_atoms_oracle()
    assert checked > 100 and bad == 0, (checked, bad)
