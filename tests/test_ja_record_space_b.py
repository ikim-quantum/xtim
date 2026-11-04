"""J-a N6 (J-a-b): record-space B (cpp/src/record_space_b.cpp, qeccore/record_space_b.hpp) — THE record-table builder
since J-b2 J7b (the sigma dual basis it replaced is deleted, and with it G-B1's corpus half, which compared the two).

This module keeps the fixtures (FX-S2b + the sigma-4 set, the speed set) and plan-key helpers other gates import, and
the B-banking replay check: recb_replay's two modes ("plan_structure", the ledger's t_build call, and "recb", the bare
builder) build identical tables (digest of the canonical fields). The law itself is gated against independent
oracles elsewhere (test_s2b_record_table's dense / (X) comparisons; XTIM_MUTATE_B_ROW_SKIP -> red there); G-B2 (B's
operation counts under inactive padding) is the ctest record_space_b.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
from test_s2b_record_table import FIXTURES  # noqa: E402

from xtim import _xtim_testapi as TA  # noqa: E402
from _private import private_fixture, private_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = 2000
FIELDS = ("rho", "kappa_rec", "pivot_rec", "born_rec", "rec_A", "rec_B", "rec_beta", "rec_lam", "rec_lam_phase")


def _read(rel):
    return open(os.path.join(ROOT, rel)).read()


def _sigma_set():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "s2b"))
    from test_sigma_joint_law import FIXTURES as SF
    return {k: v for k, v in SF.items() if k not in FIXTURES}


FX_S2B = dict(FIXTURES)
FX_S2B.update({"sigma." + k: v for k, v in _sigma_set().items()})
FX_S2B.update({"F1." + k: v for k, v in F.FX_F1.items()})

SPEED = {
    "d5": _read("tests/data/cultivation_d5.stim"),
    "csr": _read("tests/data/code_switching_rate.stim"),
    "m3_obs": FIXTURES["m3_obs"],
    "m3_obs_2dec": FIXTURES["m3_obs_2dec"],
    "q3_dec": FIXTURES["q3_dec"],
}
for _k, _n in (("fac", "ccz_factory_a.stim"), ("fac_b", "ccz_factory_b.stim")):   # private (skipped when absent)
    if private_text(_n) is not None:
        SPEED[_k] = private_text(_n)
_CD = os.environ.get("LEDGER_CIRC_DIR")
BT81 = os.path.join(_CD, "stateroute", "circ", "bt81_ts.stim") if _CD else None
if BT81 and os.path.exists(BT81):
    SPEED["bt81"] = open(BT81).read()


def fam(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TA.compile_family_sampler(text, selfcheck=0)


def fam_any(text):
    """The FAM sampler, else None (refused by name)."""
    try:
        return fam(text)
    except ValueError:
        return None


def n_wires(s):
    return np.asarray(s.term_axes()[0]).shape[1] // 2


def diag_key(n, nf):
    """plan_structure's key bytes (identity prefix) of a diagonal normal form {a, cz}."""
    nw = (n + 63) // 64
    a = set(nf["a"])
    out = b"\0" * (16 * nw) + bytes(1 if q in a else 0 for q in range(n)) + b"\xff"
    for f, s2 in nf["cz"]:
        out += int(f).to_bytes(2, "little") + int(s2).to_bytes(2, "little")
    return out


def plan_keys(s, shots=SHOTS, seed=11):
    """Every diagonal plan key reached by `shots` shots (+ the identity plan), first-appearance order."""
    n = n_wires(s)
    keys, seen = [b""], {b""}
    for e in s.term_events(shots, seed):
        alts = [int(x) for x in e]
        if not alts:
            continue
        d = s.diag_shot(alts)
        if d is None:
            t = s.term_shot(alts)
            if t["kind"] != 0:
                continue
            d = t["nf"]
        k = diag_key(n, d)
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def test_recb_replay_digests_agree():
    """The B-banking replay builds the same tables by both builders (digest of the canonical fields)."""
    s = fam(private_fixture("ccz_factory_a.stim").read_text())
    keys = plan_keys(s, shots=400)
    h_ps, n_ps = TA.recb_replay(s, keys, 1, "plan_structure")
    h_b, n_b = TA.recb_replay(s, keys, 1, "recb")
    assert n_ps == n_b == len(keys)
    assert h_ps == h_b
