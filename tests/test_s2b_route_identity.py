"""S2b gate m2: sample() and sample_barrier() emit the SAME record bytes at the same seed.

J-b2 J1b: these gates run on the PUBLIC sampler (xtim.compile_twirl_sampler; the legacy sampler they ran on at J-b1
retires at J2). Production refuses no observable (M-ANTI), so the refusal-identity gate compares served bytes.

The barrier's sigma-4 draws (the generators' fair coins on COIN bits rho_rec.., the sigma-aux walk on
BORN words kappa_rec..) lie past every record index, so they never move a record byte (mutation MS-4:
an aux draw on BORN word kappa_rec - 1 turns this red). Also gated: the OBS stream is never drawn on a
diagonal record route (Q2: record_report()["rec"]["obs_draws"]["record"] == 0), and every
record-route barrier shot satisfies the sigma channel relation exactly: the barrier runs with
selfcheck >= shots, where a channel bit != mask.sigma is a NAMED throw (r6.5; the production
channel_violations counter is gone).
"""
import os
import sys

import numpy as np
import pytest

import xtim

sys.path.insert(0, os.path.dirname(__file__))
from test_s2b_record_table import FIXTURES  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "s2b"))
from test_sigma_joint_law import FIXTURES as SIGMA_FIXTURES  # noqa: E402
from _public import compile_public_twirl_sampler as compile_twirl_sampler  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

ALL = dict(SIGMA_FIXTURES)
ALL.update(FIXTURES)


def _sample(s, n, seed):
    r = s.sample(n, seed=seed)
    if not hasattr(r, "decisions"):
        return np.asarray(r[0]), np.asarray(r[1]), (np.asarray(r[2]) if len(r) > 2 else None)
    return np.asarray(r.dets), np.asarray(r.obs), np.asarray(r.decisions)


@pytest.mark.parametrize("name", sorted(ALL))
def test_sample_equals_barrier_records(name):
    text = ALL[name]
    s = compile_twirl_sampler(text, selfcheck=0, skip_refused_observables=True)
    d, o, dec = _sample(s, 4000, 9)
    b = compile_twirl_sampler(text, selfcheck=0, skip_refused_observables=True).sample_barrier(4000, seed=9)
    assert np.array_equal(d, np.asarray(b.dets()))
    assert np.array_equal(o, np.asarray(b.obs()))
    if dec is not None:
        assert np.array_equal(dec, np.asarray(b.decisions()))


def _outcome(fn):
    try:
        return ("ok", np.asarray(fn()).tobytes())
    except Exception as e:                            # refusal: compare type and message
        return (type(e).__name__, str(e))


@pytest.mark.parametrize("name", sorted(ALL))
def test_sample_and_barrier_obs_refuse_identically(name):
    """E1 + m2: under the default (no skip_refused_observables) sample() and sample_barrier().obs()
    refuse with the same type and message, or both return the same bytes."""
    text = ALL[name]
    a = _outcome(lambda: compile_twirl_sampler(text, selfcheck=0).sample(2000, seed=9)[1])
    b = _outcome(lambda: compile_twirl_sampler(text, selfcheck=0).sample_barrier(2000, seed=9).obs())
    assert a == b, (name, a[0], b[0])


def test_formerly_refused_observable_is_served():
    """ndhaz_F1x's second LOGICAL observable (refused by the legacy engine) is served by production (M-ANTI):
    sample() and the barrier return the same bytes for it (the identity gate above is not vacuous on it)."""
    a = _outcome(lambda: compile_twirl_sampler(ALL["ndhaz_F1x"], selfcheck=0).sample(100, seed=1)[1])
    b = _outcome(lambda: compile_twirl_sampler(ALL["ndhaz_F1x"], selfcheck=0).sample_barrier(100, seed=1).obs())
    assert a[0] == "ok" and a == b


@pytest.mark.parametrize("name", sorted(ALL))
def test_no_obs_draw_and_no_channel_violation_on_record_routes(name):
    text = ALL[name]
    s = compile_twirl_sampler(text, selfcheck=0, skip_refused_observables=True)
    s.sample(3000, seed=4)
    sb = compile_twirl_sampler(text, selfcheck=3000, skip_refused_observables=True)
    sb.sample_barrier(3000, seed=5)                  # the σ-4 channel oracle is live on every shot
    rep = s.record_report()
    assert rep["rec"]["obs_draws"]["record"] == 0, rep["rec"]
    assert sb.record_report()["rec"]["obs_draws"]["record"] == 0
    assert "sigma_aux" not in rep                       # no sigma on production
    for k in ("exact_born_obs_kappa_pos", "exact_born_dec_kappa_pos", "exact_obs_guard"):
        assert k not in rep["route_counts"], (k, rep["route_counts"])      # deleted routes (B2)
