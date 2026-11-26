"""The carried-compose route's frame layout is pinned, byte for byte (Stage 1 T1 fix round 1).

`compose_carried_bare` (cpp/src/carried_compose.cpp) diagonalizes the consumer's merged axes restricted to the
carried state's constant rows. When nothing non-identity is left there (every axis is Z on carried free rows, or the
magic cancels after merging: T T_DAG on a carried wire), the general Lagrangian completion puts H on every constant
row and rotates it back, and the consumer's frame layout (row order, branch order, hence the ideal-state token that
a chained stage digests) depends on that. T1's trivial completion of an empty layer is therefore applied only on the
build_parity_form route (normal_form.cpp); this test pins the carried route against goldens recorded on the
pre-T1 build (884e374): the public ideal-state token and a hash of the family sampler's bare frames.
"""
import hashlib
import warnings

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")

# name: (producer, consumer, ideal-state token (hex), sha256 of the family sampler's bare_frames)
# The tokens hash ENGINE_VERSION: RE-PINNED 2026-11-25 for xtim-engine-18 (the bare_frames hashes are unchanged).
CASES = {
    "restI_fresh1": (
        'RX 0\nT 0\nOUTPUT_QUBITS out 0\n',
        'INPUT_QUBITS in 0\nR 1\nT 0\nCX 0 1\nM 0 1\n',
        "fff4a169b15d64dd1b950c484666a144",
        "fd5df18756ddca7c4beb60ce494ce481a9a059f79d76b90302581079ce152b93",
    ),
    "restI_fresh3": (
        'RX 0\nT 0\nOUTPUT_QUBITS out 0\n',
        'INPUT_QUBITS in 0\nR 1 2 3\nH 2\nT 0\nCX 0 1\nCX 2 3\nX_ERROR(0.1) 1\nM 0 1 2 3\nDETECTOR rec[-1] rec[-2]\n',
        "1be7f08a001bc8fab839c359d67c923b",
        "a2454c2d1d7abebcc69f488b82cf94881d413d8760f6bd64927621704ae1ff1e",
    ),
    "restI_two_carried": (
        'RX 0 1\nT 0 1\nOUTPUT_QUBITS out 0 1\n',
        'INPUT_QUBITS in 0 1\nR 2\nT 0\nT 1\nCX 0 2\nCX 1 2\nM 2\nMX 0 1\n',
        "430540448738def7f38ad598c897d081",
        "ac78a1752850a546aafa6b973dd2224134ce10b766fd63a9c794dda374aa1126",
    ),
    "m0_tdag": (
        'RX 0\nT 0\nOUTPUT_QUBITS out 0\n',
        'INPUT_QUBITS in 0\nR 1\nT 0\nT_DAG 0\nCX 0 1\nM 0 1\n',
        "39e3858ecce318df98c81fa4ccbe865f",
        "f363f52342bcad5651815bf5340c1ddec47e57cd488e11dde950e3fa2ff80679",
    ),
    "m0_tdag_ent": (
        'RX 0 1\nT 0\nCX 0 1\nOUTPUT_QUBITS out 0 1\n',
        'INPUT_QUBITS in 0 1\nR 2 3\nH 3\nT 1\nT_DAG 1\nCX 1 2\nCZ 3 0\nM 2\nMX 0 1 3\n',
        "379086dda189f0d23dc0588f5676b4c7",
        "8555223cefe8719af4b240e2af39f0f034d3b427e75f63c90767d169bb788425",
    ),
    "m0_tt_s": (
        'RX 0\nT 0\nOUTPUT_QUBITS out 0\n',
        'INPUT_QUBITS in 0\nR 1 2\nT 0\nT 0\nCX 0 1\nM 0 1 2\n',
        "c0421f24844cd4c5131096a270293b03",
        "751d74f4ebe9a55f8bf8dcd0e83b5b47b572781c95a9f0c9d8b59462689c84b2",
    ),
    "rnd4_36": (
        'RX 0 1\nT 0\nT 1\nCX 1 0\nOUTPUT_QUBITS out 0 1\n',
        'INPUT_QUBITS in 0 1\nR 2 3 4\nT_DAG 1\nT_DAG 1\nCZ 1 4\nX_ERROR(0.05) 0 1 2 3 4\nM 0 1 2 3 4\n',
        "3d1c5b93beda1fd96c44d58867212ccf",
        "df25cb2368d5216bb0cfa814d19f250efe550e3be5ec3f8e0356e1e7a214401a",
    ),
}


def frames_sha(c):
    fr = TA.bare_frames(c)
    canon = {k: repr(np.asarray(v).tolist()) if not isinstance(v, (int, str)) else repr(v) for k, v in fr.items()}
    return hashlib.sha256(json_dumps(canon).encode()).hexdigest()


def json_dumps(d):
    import json
    return json.dumps(d, sort_keys=True)


@pytest.mark.parametrize("name", sorted(CASES))
def test_carried_layout_pinned(name):
    pt, ct, token, fsha = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P = xtim.compile_twirl_sampler(pt, selfcheck=0)
        C = xtim.compile_twirl_sampler_from_ideal_state(ct, ideal_state=P.ideal_state(), selfcheck=0)
        assert C.ideal_state().token.hex() == token
        p = TA.compile_family_sampler(pt, selfcheck=0)
        c = TA.compile_family_sampler_from_ideal_state(p.ideal_state(), ct)
        assert frames_sha(c) == fsha
