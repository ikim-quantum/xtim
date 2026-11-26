"""J-b2 J1b: the PUBLIC twirl sampler for the production legs of the gates that ran on the legacy sampler at J-b1
(tests/_legacy.py, deleted at J2). `public_sampler` is xtim.compile_twirl_sampler's engine object (the decision
wrapper unwrapped, so barrier / record accessors are reachable on every circuit); `rec_order` is the record-column
order of the same compile (a pure function of the circuit), read through the test-API family sampler, which builds
the production configuration (FAMP == NEW)."""
import os
import warnings

import xtim


def public_sampler(text, selfcheck=0, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(str(text), selfcheck=selfcheck, **kw)
    return getattr(s, "_inner", s)


def full_table(text, selfcheck=0, **kw):
    """lean N1 (OWNER-SIGNED 2026-11-07, option b): sample() draws the PRUNED noise mechanism table, a recording run
    (sample_barrier / sample_records) the FULL channel table (exact P_rec), so for the same seed their bytes differ. A
    route-identity test (a sample run vs a recording run on the SAME events) compiles `text` as the test-API
    FamilySampler (the class the public TwirlSampler binds, plus the test accessors: compile_family_sampler) and puts it
    on the full table (TA.noise_force_full); the public sampler has no such lever (review M1)."""
    from xtim import _xtim_testapi as TA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = TA.compile_family_sampler(str(text), selfcheck=int(selfcheck), **kw)
    TA.noise_force_full(f, True)
    assert TA.noise_full(f)
    return f


def rec_order(text):
    """[(kind, index)] of the record columns (kind 0 detector, 1 observable, 2 decision), record order."""
    from xtim import _xtim_testapi as TA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = TA.compile_family_sampler(str(text), selfcheck=0)
    return [(o >> 24, o & ((1 << 24) - 1)) for o in f.plan_structure(b"", False)["rec_order"]]


def identity_kappa_rec(text):
    """kappa_rec of the identity plan (the clean shots' record table) of the production compile."""
    from xtim import _xtim_testapi as TA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = TA.compile_family_sampler(str(text), selfcheck=0)
    return int(f.plan_structure(b"", False)["kappa_rec"])


def public_engine(text, p_factor=1.0, selfcheck=0, skip_refused_observables=True, decisions=True):
    """Positional twin of the retired tests/_legacy.legacy_sampler on the PUBLIC sampler (the engine object).
    skip_refused_observables has no production meaning (nothing is refused) and is accepted for call-site
    compatibility only."""
    return public_sampler(text, selfcheck=int(selfcheck), p_factor=float(p_factor), decisions=bool(decisions))


def compile_public_twirl_sampler(text, p_factor=1.0, selfcheck=2000,
                                 skip_refused_observables=False, decisions=True, plan_payload_budget=None):
    """Keyword twin of the retired tests/_legacy.compile_legacy_twirl_sampler: xtim.compile_twirl_sampler itself."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xtim.compile_twirl_sampler(str(text), p_factor=p_factor, selfcheck=selfcheck,
                                          decisions=decisions, plan_payload_budget=plan_payload_budget)


def to_prod(st):
    """A FramedSuperposition of either extension, or a branch_frames() export, as an xtim._xtim FramedSuperposition
    (the two extensions' types are module_local, plan r3.3): rebuilt from its branch_frames() export through
    framed_from_rows — the same rows, phases, eps, free register and amplitudes. (J-b2 J1b: moved here from the
    retired tests/_legacy.py.)"""
    import numpy as np
    from xtim import _xtim
    d = st if isinstance(st, dict) else st.branch_frames()   # a branch_frames() export (layout-3 materialize) too
    n = int(d["n"])
    eps = [int(e) & 1 for e in np.asarray(d["eps"])]
    zph = [(int(p) + (2 if e else 0)) & 3 for p, e in zip(np.asarray(d["stab_phase"]), eps)]
    rows = lambda a: [[int(v) & 1 for v in r] for r in np.asarray(a).reshape(n, n)]
    sig = np.asarray(d["sigma"]); co = np.asarray(d["coeff"])
    ent = [([int(v) & 1 for v in sig[i]], complex(co[i])) for i in range(co.shape[0])]
    return _xtim.framed_from_rows(n, rows(d["stab_x"]), rows(d["stab_z"]), zph, rows(d["destab_x"]),
                                  rows(d["destab_z"]), [int(p) & 3 for p in np.asarray(d["destab_phase"])], eps,
                                  [int(f) for f in np.asarray(d["free"])], ent)


def recb_table(text, plan_key=b""):
    """The record-space-B table (the production builder) of a plan of the production compile, as fields (rho,
    kappa_rec, pivot_rec, born_rec, rec_A, rec_B, rec_beta, rec_lam, rec_lam_phase); plan_key in the
    plan_structure key format (b"" = the identity plan)."""
    from xtim import _xtim_testapi as TA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = TA.compile_family_sampler(str(text), selfcheck=0)
    r = TA.recb_plan_structure(f, plan_key)
    return r.get("b")
