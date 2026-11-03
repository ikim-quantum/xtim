"""twirl-routes (owner mandate 2026-10-30): ONE stabilizer-twirl routine (cpp fault_record_law + born_law) serves every
DEM law route; no route is exponential in anything but chi (2^kappa Born leaves), the dense output register, or the
requested output (2^v lines per record at the exact cutoff).

T1 gates here: the rp2 circuit (refused at dem-erasure by the deleted wdim > 20 guard: 57 faults with a Fourier support
of dimension 21-36, every one a uniform coset) now exports — pinned lines / records / md5 at the exact cutoff and at
t = 2; the mechanism counter (Born-law node evaluations <= sum over the census components of 2^kappa - 1: no 2^wdim
work); and the exported rp2 DEM against the twirl sampler (per-detector rates at p = 1e-4)."""
import hashlib
import math
import re

import numpy as np
import pytest

import xtim
from xtim import _xtim_testapi as TA
from xtim._exact_scope import explicit_exact

DATA = __import__("os").path.join(__import__("os").path.dirname(__file__), "data")


def _rp2(p=None):
    text = open(f"{DATA}/magna_rp2_fid.stim").read()
    if p is not None:
        text = re.sub(r"\(0\.001\)", f"({p!r})", text)
    return text


def census(text, **kw):
    with explicit_exact():
        return TA.dem_law_census(text, **kw)


def test_rp2_exports_at_both_cutoffs():
    # twirl-routes T1 (new pin, owner pre-signed 2026-10-30): was REFUSED ("a Fourier support of dimension 25 (> 20)").
    # T3 re-pin (owner pre-signed 2026-10-30): md5s moved with the one Pauli solver (<= 4.1e-13 relative, same lines).
    c = xtim.Circuit(_rp2())
    d = c.detector_error_model_text(clifford_cutoff=math.inf)     # dem-t2: exact requested explicitly
    lines = d.splitlines()
    assert lines[0] == "# clifford faults=6256 cutoff=inf dropped_mass=0"
    assert len(lines) == 74466 and sum(1 for l in lines if l.startswith("error")) == 68077
    assert hashlib.md5(d.encode()).hexdigest() == "6d33cb3cbe1a21bc8dba7221600df14e"
    d2 = c.detector_error_model_text(clifford_cutoff=2)
    assert len(d2.splitlines()) == 8513
    assert hashlib.md5(d2.encode()).hexdigest() == "c4598a070519ce0e012cd7f50748d323"
    assert c.detector_error_model_text() == d2                    # dem-t2: the default is t = 2


def test_born_work_is_bounded_by_kappa():
    # G3: the export evaluates at most 2^(kappa+1) - 1 Born-law nodes + leaves per distinct law (summed here over every census
    # component, an upper bound); rp2's laws are all kappa = 0: no Born sum at all (dem-erasure's walk paid 2^wdim).
    r = census(_rp2(), clifford_cutoff=-1)
    assert r["ok"], r["error"]
    comps = [c for i in r["instances"] for c in i["comps"]]
    assert all(c["kappa"] == 0 for c in comps) and sum(1 for c in comps if c["kind"] == "C" and c["v"] > 0) >= 6256
    assert r["born_sums"] == 0
    tz0 = "RX 0\nT 0\nRX 1\nCS 1 0\nX_ERROR(0.01) 1\nCS_DAG 1 0\nMX 1\nDETECTOR rec[-1]\n"
    r = census(tz0)
    comps = [c for i in r["instances"] for c in i["comps"]]
    assert r["ok"] and [c["kappa"] for c in comps] == [1]
    assert 0 < r["born_sums"] <= sum(2 ** (c["kappa"] + 1) - 1 for c in comps)   # nodes + leaves


def test_rp2_dem_matches_sampler():
    # G2: the exported DEM's per-detector flip rates vs the default (twirl) sampler at p = 1e-4. The stage
    # gate: 1e6 shots, |z| < 4.5 on every detector.
    stim = pytest.importorskip("stim")
    text = _rp2(1e-4)
    c = xtim.Circuit(text)
    dem = stim.DetectorErrorModel(c.detector_error_model_text(clifford_cutoff=math.inf))   # dem-t2: the exact split
    shots = 1_000_000
    a = dem.compile_sampler(seed=20261030).sample(shots)[0].astype(np.float64).mean(axis=0)
    b = c.compile_detector_sampler(seed=20261030).sample(shots).astype(np.float64).mean(axis=0)
    pbar = (a + b) / 2
    se = np.sqrt(np.maximum(pbar * (1 - pbar), 1.0 / shots) * 2.0 / shots)
    z = np.abs(a - b) / se
    assert z.max() < 4.5, (float(z.max()), int(z.argmax()))


def test_list_law_never_branches_on_coins():
    # T2 mechanism gate: a PAULI_EXPECTATION column whose declared frame is a randomized read (the list law, n fair
    # coins): the joint law is the routine's table — its coins are GF(2) rows, so NO Born / branch evaluation happens
    # (dem-erasure's joint walk branched on all 2^n coin outcomes and refused past its 4096-leaf cap).
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_expfix_dem_law import fanout
    for n in (4, 12):
        r = census(fanout(n, pe_frame=True))
        assert r["ok"], r["error"]
        (comp,) = [c for i in r["instances"] for c in i["comps"]]
        assert comp["kind"] == "L" and comp["rho"] == n and comp["v"] == n
        assert r["born_sums"] == 0
