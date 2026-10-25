"""S2b R-m4 (null option): a twirl sampler's plan caches are its own — structural gates.

The C++ sharing API (SharedPlanCaches, the sharing constructor, the binding's token guard) is
deleted; cpp/tests/test_plan_cache_model.cpp pins the compile-time absence and the runtime
independence. Here: the source never re-introduces the sharing names, and the binding builds its
engine sampler only from its constructors (rebuild() is not a reseed fallback any more).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPP = ROOT / "cpp"


def _sources():
    for sub in ("src", "include", "bindings", "tests", "apps", "testapi"):
        d = CPP / sub
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.suffix in (".cpp", ".hpp", ".h", ".inc"):
                    yield p


def test_no_sharing_names_in_the_engine():
    bad = []
    for p in _sources():
        t = p.read_text(errors="replace")
        for name in ("SharedPlanCaches", "shared_caches_", "cache_token_", "cache_initialized_"):
            if name in t:
                bad.append(f"{p.relative_to(ROOT)}: {name}")
    assert not bad, bad


def test_family_sampler_builds_its_engine_once_and_reseeds_in_place():
    """J-b1 F: the production binding (cpp/bindings/family_sampler.hpp) builds its engine sampler in ONE place (the
    constructor) and a seeded call reseeds it in place (set_seed; a refusal is an error, never a rebuild)."""
    text = (CPP / "bindings" / "family_sampler.hpp").read_text()
    assert text.count("TwirlRecordSampler::make(") == 1
    run = text[text.index("    void run_core_("):]
    run = run[:run.index("\n    }\n")]
    assert "set_seed(" in run and "TwirlRecordSampler::make(" not in run and "set_seed refused" in run


