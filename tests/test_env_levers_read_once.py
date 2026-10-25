"""S2b (LB2 side finding): env levers of the twirl engine are read ONCE per process.

The engine coding playbook's rule ("env levers are read ONCE via function-local statics") is a
mechanism here: every getenv() call site in the twirl engine's sources is either inside a
function-local static initializer / the read-once helper, or in a constructor-only function on the
allow-list below. A per-call getenv() inside run() (or a per-shot / per-plan path) made the ledger's
measured windows depend on the environment's size (+24 Ir per extra variable on t_fast/d5s).
"""
from pathlib import Path

CPP = Path(__file__).resolve().parent.parent / "cpp" / "src"
# J-a r2.1 (J1): the sampler's Impl, its levers and the shot loop live in two internal headers shared by
# twirl_sampler.cpp and twirl_sampler_fam.cpp.
TWIRL = ("twirl_sampler.cpp", "twirl_sampler_impl.inc.hpp", "twirl_sampler_runw.inc.hpp", "twirl_sampler_fam.cpp",
         "twirl_law.cpp", "twirl_kernel_sampler.cpp")
# (file, the call's line text) — read-once by construction (static initializer or the helper).
READ_ONCE = {
    ("twirl_law.cpp", 'bool on = std::getenv("QEC_TW_PROF") != nullptr;'),                # global object
}


def test_no_per_call_getenv_in_the_twirl_engine():
    found, bad = set(), []
    for f in TWIRL:
        for line in (CPP / f).read_text().splitlines():
            if "getenv(" not in line or line.strip().startswith("//"):
                continue
            key = (f, line.strip())
            found.add(key)
            if key not in READ_ONCE:
                bad.append(key)
    assert not bad, bad
    assert found == READ_ONCE, sorted(READ_ONCE - found)     # non-vacuity: the allow-list is live
