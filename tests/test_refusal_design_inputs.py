"""Minimized repros of two refusals met on a CCZ-factory port (dev/next, 2026-10-14).

Design input for the owner-approved twirl-sampler redesign (one Clifford-residual type, one law
builder, one per-shot retention record). Nothing here changes engine behaviour.

1. "twirl_records: state retention / Born decisions unsupported for CH/PPR circuits"
   (twirl_sampler.cpp run(), the `(need_state || has_born_dec_) && ppr_tables` refusal) and its
   compile-time twin "Born-class DECISION on a CH/PPR circuit" (setup_error 6). A fault before a
   CCZ leaves a CZ arm; a Clifford after it (here one H) makes the residual non-diagonal, so the
   circuit carries PPR tables and EVERY retention / Born-decision run is refused — even though the
   records themselves sample fine (sample() works, ppr_plans == 1). xfail(strict): these must flip
   when the redesign retains PPR-route shots.

2. "ANTI detector(s) ...: record operator anticommutes with the certified group" (xtim_py.cpp, the
   anti_detectors() check after twirl_sampler.cpp's classify). The factory's teleport parities
   declared as DETECTORs are ANTI. ANTI means the record operator W anticommutes with a certified
   +1 stabilizer g of the bare state, so <W> = <W g> = -<g W> = -<W> = 0: the noiseless parity is a
   FAIR COIN, never a deterministic +-1. That is exactly stim's "non-deterministic detector"
   (detector_error_model refuses; the sampler reports a random bit relative to its reference). A
   deterministic -1 parity is IN_GROUP with reference bit 1 and is accepted. So the refusal is the
   owner rule ("random parities remain refused by name"), pinned here as passing tests.
"""
import numpy as np
import pytest

import xtim._xtim as _x

stim = pytest.importorskip("stim")

# n = 3: the X fault on wire 0 (heralded by its own detector) passes the CCZ as a CZ(1,2) arm; the
# H on wire 1 conjugates it out of the diagonal class -> one PPR plan.
PPR_RETENTION = ("R 0\nRX 1 2\nX_ERROR(0.2) 0\nCCZ 0 1 2\nH 1\nM 0\nDETECTOR rec[-1]\n"
                 "OUTPUT_QUBITS out 1 2\n")
# n = 4: the same, plus a Born-class decision on an independent T-state wire.
PPR_BORN_DECISION = ("R 0\nRX 1 2 3\nT 3\nX_ERROR(0.2) 0\nCCZ 0 1 2\nH 1\nM 0\nDETECTOR rec[-1]\n"
                     "MX 3\nDECISION(0) rec[-1]\nOUTPUT_QUBITS out 1 2\n")


def _sampler(text):
    return _x.TwirlSampler(text, 1.0, "", 0)


def test_ppr_repro_records_sample_and_route_ppr():
    """The repro is in scope: records sample, and the shots really take the PPR (kind-2) route. RE-PINNED 2026-10-20
    (J-b1, xtim-engine-13): M-PPR / M-API — the PPR plan is a kind-2 term-route plan; no exact key exists."""
    s = _sampler(PPR_RETENTION)
    s.sample(2000, seed=1)
    rep = s.record_report()
    assert rep["term_route"]["plans_kind2"] == 1
    assert not [k for k in rep["route_counts"] if k.startswith("exact")]


def test_state_retention_served_for_ch_ppr_circuits():
    """J-b1 F (M-DECPPR): the strict xfail flipped — retention on a CH/PPR circuit is served (kind-2 records)."""
    buf = _sampler(PPR_RETENTION).sample_barrier(200, seed=1)
    det = np.unpackbits(np.asarray(buf.dets()), axis=1, bitorder="little")[:, 0]
    assert det.any()
    for i in np.flatnonzero(det)[:5]:
        buf.materialize(int(i))


def test_born_decisions_served_for_ch_ppr_circuits():
    """J-b1 F (M-DECPPR): the strict xfail flipped — a Born-class DECISION on a CH/PPR circuit is served."""
    _sampler(PPR_BORN_DECISION).sample(200, seed=1)


# ── 2. ANTI detectors == stim's non-deterministic detectors ────────────────────────────────────
# n = 4 teleportation: wire 0 -> wire 2; the Bell-measurement X-frame bit (rec of M 0) declared as a
# DETECTOR is a fair coin. The detector on wire 3 is an ordinary deterministic one.
TELEPORT = ("RX 0\n{magic}R 1 2\nH 1\nCX 1 2\nCX 0 1\nH 0\nX_ERROR(0.01) 2\nM 0 1\nR 3\nM 3\n"
            "DETECTOR rec[-1]\nDETECTOR rec[-3]\n{tail}")


@pytest.mark.parametrize("magic,tail", [("", ""), ("T 0\n", "OUTPUT_QUBITS out 2\n")])
def test_teleport_parity_detector_is_anti_and_served(magic, tail):
    """RE-PINNED 2026-10-20 (J-b1, xtim-engine-13): M-ANTI — the ANTI detector is served (it was refused by name)
    as a nondeterministic record: a fair coin, like stim's non-deterministic detector."""
    s = _sampler(TELEPORT.format(magic=magic, tail=tail))
    rep = s.record_report()
    assert rep["nondeterministic_detectors"] == [1]
    d, _ = s.sample(40000, seed=3)
    m = np.unpackbits(np.asarray(d), axis=1, bitorder="little")[:, :2].mean(0)
    assert m[0] == 0.0 and abs(m[1] - 0.5) < 0.02


def test_anti_is_exactly_stims_non_deterministic_detector():
    c = stim.Circuit(TELEPORT.format(magic="", tail=""))
    with pytest.raises(ValueError, match="non-deterministic detectors"):
        c.detector_error_model()
    m = c.compile_detector_sampler(seed=3).sample(40000).mean(0)
    assert m[0] == 0.0 and abs(m[1] - 0.5) < 0.02          # the ANTI detector is a fair coin


def test_deterministic_minus_one_detector_is_accepted():
    s = _sampler("RX 0\nT 0\nR 1\nX 1\nM 1\nDETECTOR rec[-1]\nOUTPUT_QUBITS out 0\n")
    assert s.record_report()["classes"]["detectors"] == {0: "deterministic"}
    d, _ = s.sample(100, seed=1)
    assert not np.unpackbits(np.asarray(d), axis=1, bitorder="little")[:, 0].any()  # ref-relative
