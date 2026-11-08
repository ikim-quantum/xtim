"""plancache-design (owner-signed rev 4, 2026-11-01): the readers' ONE record-table choke point.

lean D1-S3 (owner-signed deletions): the K2' kernel over the CZ pair universe — Π's B-part, its elimination, cap, O3 audit and
the null-option lever (`_plan_class_identity`) — is deleted with the pair universe (the rotation loop serves every circuit
with rotations; a diagonal-cache key is the a-mask of a pair-free content). The gates of this module that pinned the Π classes
on circuits with rotations retired with them: the null option's byte identity, thread-identical class counters, O1 on the
corpus, readers reading the class table, class eviction, the factory classes, second-chance class eviction, class ids at
every thread count and the lane arena bound under class builds (the cache's classes now hold the Clifford path's identity
plan; its eviction / byte neutrality is test_plan_payload_budget.py's and test_s5a_fast_retention.py's).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_record_table_builder_callers_are_the_choke_points():
    """plan §1 (review F2, low): build_record_table_recB's production callers are grep-gated to the readers' ONE choke
    point (rec_table_of_: the class representative's table, or an unclassed route-1 content's own), the class build
    (build_rec_table_fam_, the parked-core miss path), O1's per-key oracle (class_rep_check_) and the DEM path
    (fault_record_law: not a plan reader). Every other reader goes through rec_table_memo_ / rec_table_held_ /
    content_table_ / reader_table_fresh_."""
    import re
    root = HERE.parent / "cpp"
    allowed = {
        "src/record_space_b.cpp",            # the definition
        "src/twirl_sampler_impl.inc.hpp",    # rec_table_of_ (the choke point's builder)
        "src/twirl_sampler_fam.cpp",         # build_rec_table_fam_ (no parked core: expected 0, counted)
        "src/twirl_sampler.cpp",             # class_rep_check_ (O1)
        "src/fault_record_law.cpp",          # the DEM's per-law table (not a plan reader)
        "src/twirl_sampler_rotw.cpp",        # lean D1-S2: the gauge-map ORACLE's (D) chain (window / test API only)
    }
    found = set()
    for sub in ("src", "bindings"):
        for p in sorted((root / sub).rglob("*")):
            if p.suffix not in (".cpp", ".hpp"):
                continue
            t = p.read_text(errors="replace")
            if re.search(r"\bbuild_record_table_recB\s*\(", t):
                found.add(str(p.relative_to(root)))
    assert found == allowed, (found - allowed, allowed - found)
    impl = (root / "src" / "twirl_sampler_impl.inc.hpp").read_text()
    assert impl.count("build_record_table_recB(") == 1          # rec_table_of_ only
    ts = (root / "src" / "twirl_sampler.cpp").read_text()
    assert ts.count("build_record_table_recB(") == 1            # class_rep_check_ only (plan_structure reads the class)
