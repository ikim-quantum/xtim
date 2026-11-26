"""QW-C (Stage 2, peak memory): the alternatives per TermResidual::for_alts pass of the rotation records and the P supports
are sized from the wire count (twirl_sampler_rot.cpp rot_chunk_: 16384 on narrow circuits, fewer on wide ones so a pass's
buffers stay within ~32 MiB). The outputs must not depend on it: every alternative is its own bit lane of the propagation
and the callbacks append in alternative order. Oracle: the same circuit compiled with tiny forced chunks
(TA._set_rot_chunk: 1, 7, 65 — partial words, a chunk boundary inside every word) against the default — the rotation
records (words and offsets), their decode check, and the sampled bytes of a selfcheck run (the window reads the P
supports) at a fixed seed. Corpus: tests/data, examples, examples/xdem, the private corpus (by opaque index) — all narrow
(the default there is 16384) — plus a wide surface-code circuit whose sized default is below 16384 (the formula path).
"""
import glob
import os
import warnings

import numpy as np
import pytest

from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

CORPUS = {}
for pat in ("tests/data/*.stim", "examples/*.stim", "examples/xdem/*.stim"):
    for p in sorted(glob.glob(os.path.join(ROOT, pat))):
        CORPUS[os.path.relpath(p, ROOT)] = p
_priv = os.environ.get("XTIM_PRIVATE_FIXTURES") or os.path.join(HERE, "private")
for i, p in enumerate(sorted(glob.glob(os.path.join(_priv, "*.stim")))):
    CORPUS[f"private#{i}"] = p          # the private corpus: opaque index only

SHOTS, SEED, SELFCHECK = 200, 11, 50


def _run(text, chunk):
    TA._set_rot_chunk(chunk)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = TA.compile_family_sampler(text, selfcheck=SELFCHECK)
        R = TA.rot_records(s)
        chk = TA.rot_check(s)
        d, o, c = s.sample(SHOTS, seed=SEED)
        return dict(alts=R["alts"], rec=np.asarray(R["rec"]).copy(), off=np.asarray(R["off"]).copy(), chk=chk,
                    d=np.asarray(d).tobytes(), o=np.asarray(o).tobytes(), c=np.asarray(c).tobytes())
    finally:
        TA._set_rot_chunk(0)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_tiny_chunks_match_default(name):
    text = open(CORPUS[name]).read()
    try:
        ref = _run(text, 0)
    except Exception as e:   # the corpus's compile refusals are not this test's subject
        pytest.skip(f"compile refused: {type(e).__name__}")
    assert ref["chk"] == (ref["alts"], 0)
    chunks = (1, 7, 65) if ref["alts"] <= 20000 else (7, 65)
    for chunk in chunks:
        got = _run(text, chunk)
        assert got["alts"] == ref["alts"], chunk
        assert np.array_equal(got["off"], ref["off"]), chunk
        assert np.array_equal(got["rec"], ref["rec"]), chunk
        assert got["chk"] == ref["chk"], chunk
        assert (got["d"], got["o"], got["c"]) == (ref["d"], ref["o"], ref["c"]), chunk


def test_wide_circuit_sized_default():
    """The formula path itself: a circuit wide enough that the sized default drops below 16384 (surface d=15 r=6: n' a few
    thousand wires, ~83k alternatives, several passes and a partial last one), against forced chunks (1, 64, 129, 16384)."""
    stim = pytest.importorskip("stim")
    text = str(stim.Circuit.generated("surface_code:rotated_memory_z", distance=15, rounds=6,
                                      after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                      after_reset_flip_probability=0.001))
    TA._set_rot_chunk(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = TA.compile_family_sampler(text, selfcheck=0)
    chunk = TA.rot_chunk(s)
    alts = TA.rot_records(s)["alts"]
    assert 64 <= chunk < 16384 and chunk % 64 == 0, chunk
    assert alts > 2 * chunk and alts % chunk != 0, (alts, chunk)
    ref = _run(text, 0)
    assert ref["chk"] == (ref["alts"], 0) and ref["alts"] == alts
    for forced in (1, 64, 129, 16384):
        got = _run(text, forced)
        assert got["alts"] == ref["alts"], forced
        assert np.array_equal(got["off"], ref["off"]), forced
        assert np.array_equal(got["rec"], ref["rec"]), forced
        assert got["chk"] == ref["chk"], forced
        assert (got["d"], got["o"], got["c"]) == (ref["d"], ref["o"], ref["c"]), forced


def test_override_resets():
    TA._set_rot_chunk(3)
    TA._set_rot_chunk(0)
    with pytest.raises(ValueError):
        TA._set_rot_chunk(-1)
