"""LM-1: a lean xdem result with no rotating line is answered by the ordinary (stim-format) export WITHOUT an eager bare
state: the lean core's exact determinism certificate replaces the bare-state gate, the bare state is built only on a
first real read (lazy, counted). Bytes: every result field equals the oracle route's (the pre-plan exporter, which
builds the bare state eagerly). Capacity (owner ruling 2026-11-23): a no-rotation circuit whose bare state is over
capacity exports instead of raising; a bare read that is really needed still raises.

Counters: `_xtim_testapi.dem_lazy_bare_counts()` = (certified exports, lazy builds), monotone. They count the test-API
module's own engine copy only (public `xtim` calls never move them), and they are process-wide relaxed atomics: the
deltas taken here around one call are valid only because these tests run exports one at a time.

Capacity ruling scope (controller, 2026-11-23): only lean-certified no-rotation circuits export over capacity; circuits
the lean route delegates to the oracle (PAULI_EXPECTATION, an H meeting a magic location) still raise."""
import random
from pathlib import Path

import pytest
import stim

from xtim import _xtim_testapi as TA

HERE = Path(__file__).resolve().parent
DIRS = [HERE / "data", HERE.parent / "examples", HERE.parent / "xtim" / "_examples", HERE.parent / "benchmarks",
        HERE / "private"]
FIELDS = ("ok", "dem", "error", "rejected", "reject_gate_index", "parse_errors", "parse_error_lines",
          "dropped_observables", "reject_detectors", "postselect_faults", "dropped_expectation_columns",
          "clifford_faults", "clifford_dropped_mass", "nondeterministic_targets", "exception",
          "exception_message")


def counts():
    return TA.dem_lazy_bare_counts()


def same(text, tag, **kw):
    """lean == oracle on every field; returns (lean dict, certified delta, lazy delta)."""
    o = TA.xdem_export_route(text, "oracle", **kw)
    c0 = counts()
    lean = TA.xdem_export_route(text, "lean", **kw)
    c1 = counts()
    assert lean["lean_exception"] == "", (tag, lean["lean_exception"])
    bad = [f for f in FIELDS if o[f] != lean[f]]
    assert not bad, (tag, bad, o["error"][:200], lean["error"][:200])
    return lean, c1[0] - c0[0], c1[1] - c0[1]


def test_corpus_identical_and_no_lazy_build():
    paths = []
    for d in DIRS:
        if d.is_dir():
            paths += sorted(str(p) for p in d.rglob("*.stim"))
    cert = 0
    for i, p in enumerate(paths):
        lean, dc, dl = same(Path(p).read_text(), f"corpus#{i}")      # opaque index: private never named
        assert dl == 0, f"corpus#{i}: lazy bare build"
        cert += dc
        if dc:
            assert lean["ok"] and lean["lean_used"] and not lean["dem"].startswith("xdem 1"), f"corpus#{i}"
    assert cert >= 3, cert


# The lean-magic probe's no-rotation families (noise-free T: no fault reaches a T location) at small sizes.
def _surf(d):
    return str(stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                      after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                      after_reset_flip_probability=0.001))


def _rep(d):
    return str(stim.Circuit.generated("repetition_code:memory", distance=d, rounds=2,
                                      after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                      after_reset_flip_probability=0.001))


def _tpair(a, b, noisy=False):
    e = f"X_ERROR(0.001) {a}\n" if noisy else ""
    return f"R {a} {b}\nH {a}\n{e}T {a}\nCX {a} {b}\n", f"MX {a}\nM {b}\n"


def surf_t(d, k=1, noisy=False):
    c = _surf(d)
    nq = stim.Circuit(c).num_qubits
    pre = post = ""
    for i in range(k):
        p, q = _tpair(nq + 2 * i, nq + 2 * i + 1, noisy)
        pre += p
        post += q
    return pre + c + "\n" + post


def chain_t(n, noisy=False):
    pre, post = _tpair(n, n + 1, noisy)
    return pre + "".join(f"R {q}\nX_ERROR(0.01) {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(n)) + post


def ghz_t(n, hth=True, noisy=False):
    L = ["R " + " ".join(map(str, range(n))), "H 0"] + [f"CX {i} {i + 1}" for i in range(n - 1)]
    L += [("X_ERROR(0.001) 0\n" if noisy else "") + ("H 0\nT 0\nH 0" if hth else "T 0"),
          "X_ERROR(0.01) " + " ".join(map(str, range(n))), "M " + " ".join(map(str, range(n)))]
    L += [f"DETECTOR rec[-{n - i}] rec[-{n - i - 1}]" for i in range(1, n - 1)]
    return "\n".join(L) + "\n"


def rep_t(d, noisy=False):
    c = _rep(d)
    nq = stim.Circuit(c).num_qubits
    p1, q1 = _tpair(nq, nq + 1, noisy)
    p2, q2 = _tpair(nq + 2, nq + 3, noisy)
    return p1 + p2 + c + "\n" + q1 + q2


NO_ROTATION = [
    *[(f"surface+T d{d}", surf_t(d)) for d in (3, 5)],
    ("surface+2T d3", surf_t(3, 2)),
    *[(f"chain+T {n}", chain_t(n)) for n in (63, 300)],
    *[(f"ghz+HTH {n}", ghz_t(n)) for n in (5, 63, 200)],
    *[(f"ghz+HTH noisy {n}", ghz_t(n, noisy=True)) for n in (5, 63, 200)],
    *[(f"ghz+T {n}", ghz_t(n, hth=False)) for n in (5, 65)],
    *[(f"rep+2T d{d}", rep_t(d)) for d in (5, 33)],
]


@pytest.mark.parametrize("tag,text", NO_ROTATION, ids=[t for t, _ in NO_ROTATION])
def test_no_rotation_families(tag, text):
    lean, dc, dl = same(text, tag)
    assert lean["ok"] and lean["lean_used"] and not lean["dem"].startswith("xdem 1"), tag
    assert dc == 1 and dl == 0, (tag, dc, dl)


@pytest.mark.parametrize("tag,text", [("surface+noisy T d3", surf_t(3, noisy=True)),
                                      ("chain+noisy T 64", chain_t(64, noisy=True)),
                                      ("rep+2 noisy T d5", rep_t(5, noisy=True))],
                         ids=["surface+noisy T d3", "chain+noisy T 64", "rep+2 noisy T d5"])
def test_rotating_families_untouched(tag, text):
    lean, dc, dl = same(text, tag)
    assert lean["ok"] and lean["dem"].startswith("xdem 1") and dc == 0 and dl == 0, tag


def test_forced_lazy_build_gives_identical_bytes():
    """The gate-off lever makes the certified export read the bare state (a lazy build): the bytes do not change."""
    built = 0
    for tag, text in NO_ROTATION[:6]:
        TA.set_dem_cert_gate_off(True)
        try:
            lean, dc, dl = same(text, tag)
        finally:
            TA.set_dem_cert_gate_off(False)
        assert dc == 1 and dl == 1, (tag, dc, dl)
        built += dl
    assert built == 6


def _wall_norot(k):
    """k |+> qubits with a noise-free T each, never read (bare rank k), and one noisy deterministic Z detector."""
    q = " ".join(map(str, range(k)))
    return f"RX {q}\nT {q}\nR {k}\nX_ERROR(0.01) {k}\nM {k}\nDETECTOR rec[-1]\n"


def test_capacity_no_rotation_now_exports():
    small = TA.xdem_export_route(_wall_norot(12), "oracle")
    assert small["ok"], small
    o = TA.xdem_export_route(_wall_norot(31), "oracle")
    assert o["exception"] == "BareStateCapacityError", o            # the oracle route (pre-ruling) still raises
    c0 = counts()
    lean = TA.xdem_export_route(_wall_norot(31), "lean")
    c1 = counts()
    assert lean["ok"] and lean["lean_used"] and lean["exception"] == "", lean
    assert lean["dem"] == small["dem"], (lean["dem"], small["dem"])   # pinned: the T wires never reach the DEM
    assert c1[0] - c0[0] == 1 and c1[1] - c0[1] == 0


def test_capacity_needed_read_still_raises():
    """A certified export whose bare state is really read (forced) still raises over capacity."""
    TA.set_dem_cert_gate_off(True)
    try:
        lean = TA.xdem_export_route(_wall_norot(31), "lean")
    finally:
        TA.set_dem_cert_gate_off(False)
    assert lean["exception"] == "BareStateCapacityError", lean


def test_public_capacity_no_rotation_exports():
    import xtim
    s = xtim.Circuit(_wall_norot(31)).detector_error_model_text(format="xdem")
    assert s == xtim.Circuit(_wall_norot(12)).detector_error_model_text(format="xdem")


@pytest.mark.parametrize("extra,why", [("PAULI_EXPECTATION(0) Z31\n", "PAULI_EXPECTATION"),
                                       ("H " + " ".join(map(str, range(31))) + "\n", "H meets a T location")],
                         ids=["pauli_expectation", "h_meets_location"])
def test_capacity_delegated_no_rotation_still_raises(extra, why):
    """The ruling covers lean-certified circuits only: a no-rotation rank-31 T wall that the lean route DELEGATES to the
    oracle (which builds the bare state eagerly) still raises XtimCapacityError on the public format="xdem" call."""
    import xtim
    from xtim.errors import XtimCapacityError
    k = 31
    q = " ".join(map(str, range(k)))
    text = f"RX {q}\nT {q}\n{extra if why != 'PAULI_EXPECTATION' else ''}R {k}\nX_ERROR(0.01) {k}\nM {k}\n" \
           f"DETECTOR rec[-1]\n{extra if why == 'PAULI_EXPECTATION' else ''}"
    lean = TA.xdem_export_route(text, "lean")
    assert not lean["lean_used"] and lean["exception"] == "BareStateCapacityError", (why, lean)
    with pytest.raises(XtimCapacityError):
        xtim.Circuit(text).detector_error_model_text(format="xdem")
    # the served control (no delegation) exports
    xtim.Circuit(_wall_norot(k)).detector_error_model_text(format="xdem")


def norot_fuzz(seed):
    """Mirror circuits C C^dag with noise (deterministic detectors) plus magic no fault reaches: (a) a noise-free
    T / CS / CCZ block on fresh qubits, read but in no detector; (b) magic right after the resets, before any noise
    (|0> or |+> start); (c) sometimes a noisy T (rotating: the extended export, compared too)."""
    rng = random.Random(seed)
    n = rng.randint(2, 5)
    allq = " ".join(map(str, range(n)))
    xb = rng.random() < 0.5
    L = [("RX " if xb else "R ") + allq]
    if rng.random() < 0.5:                                   # (b) magic before any noise
        q = rng.sample(range(n), min(n, 3))
        L.append(rng.choice([f"T {q[0]}", f"T_DAG {q[0]}", f"CS {q[0]} {q[1]}",
                             f"CCZ {q[0]} {q[1]} {q[-1]}" if n >= 3 else f"T {q[0]}"]))
    body = []
    for _ in range(rng.randint(2, 8)):
        q = rng.sample(range(n), 2)
        body.append(rng.choice([f"H {q[0]}", f"S {q[0]}", f"S_DAG {q[0]}", f"CX {q[0]} {q[1]}", f"CZ {q[0]} {q[1]}"]))
    inv = {"S": "S_DAG", "S_DAG": "S"}
    for g in body:
        L.append(g)
        if rng.random() < 0.6:
            L.append(f"{rng.choice(['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1'])}(0.01) {rng.randrange(n)}")
    for g in reversed(body):
        nm, *t = g.split()
        L.append(" ".join([inv.get(nm, nm)] + t))
    a = n
    if rng.random() < 0.8:                                   # (a) a noise-free magic block on fresh wires
        L += [f"R {a} {a + 1} {a + 2}", f"H {a}", f"H {a + 1}",
              rng.choice([f"T {a}", f"CS {a} {a + 1}", f"CCZ {a} {a + 1} {a + 2}"]), f"CX {a} {a + 2}"]
    if rng.random() < 0.15:                                  # (c) a rotating fault
        L += [f"RX {a + 3}", f"X_ERROR(0.01) {a + 3}", f"T {a + 3}"]
    L.append(("MX " if xb else "M ") + allq)
    L += [f"DETECTOR rec[-{k}]" for k in range(1, n + 1)]
    L.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return "\n".join(L) + "\n"


def test_no_rotation_fuzz():
    cert = ext = 0
    for s in range(400):
        lean, dc, dl = same(norot_fuzz(s), f"norot fuzz {s}")
        assert dl == 0, s
        cert += dc
        ext += lean["dem"].startswith("xdem 1")
    assert cert >= 200 and ext >= 10, (cert, ext)


@pytest.mark.slow
def test_no_rotation_fuzz_approx():
    for s in range(400, 700):
        lean, dc, dl = same(norot_fuzz(s), f"norot fuzz approx {s}", approximate_disjoint_errors=0.1)
        assert dl == 0, s
