"""J-b1 P6c (fixplan-coldstart (B), ruling: twirl_sampler.cpp is the single out-of-line owner): the members
TwirlRecordSampler::Impl shares between the production TU (twirl_sampler.cpp) and the family TU
(twirl_sampler_fam.cpp) are DEFINED in the production TU only; the family TU references them (nm U), so the FAM hot
path links exactly one body per member — today's (the linker already chose the production TU's copy) — and it
cannot move when J-b2 removes the legacy run_w. Checked by symbol on the in-place build objects (skipped without
them); tools/codegen_eq.py --comdat (vague-linkage symbols only) reports no shared Impl member.

The objects are looked up in setuptools' build dir for the running interpreter (build/temp.<platform>-<cache tag>);
the check reads ELF symbol classes (T / W / t, comdat), so it runs on ELF objects only and skips elsewhere."""
import glob
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _obj_dir():
    """cpp/src of the newest in-place build dir for this interpreter that holds both TUs' objects, or None."""
    dirs = [os.path.join(d, "cpp", "src")
            for d in glob.glob(os.path.join(ROOT, "build", f"temp.*-{sys.implementation.cache_tag}"))]
    dirs = [d for d in dirs if all(os.path.exists(os.path.join(d, o)) for o in ("twirl_sampler.o", "twirl_sampler_fam.o"))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def _is_elf(path):
    with open(path, "rb") as f:
        return f.read(4) == b"\x7fELF"


# threads T2: record_rows_ gained its LANE parameter (the lanes' variants have the same single owner)
MEMBERS = ["record_rows_<false, false, false>", "record_rows_<false, true, false>", "record_rows_<true, false, false>",
           "record_rows_<true, true, false>", "record_rows_<false, false, true>", "record_rows_<false, true, true>",
           "record_rows_<true, false, true>", "record_rows_<true, true, true>",
           "build_rec_cold_", "payload_boundary_", "identity_plan_", "parse_plan_key",
           "record_values_words_", "emit_decisions", "semantic_oracle_"]   # J-b2 J6b: emit_shot, dress_*, apply_input_frame deleted


def syms(obj):
    out = subprocess.run(["nm", "-C", obj], capture_output=True, text=True, check=True).stdout
    d = {}
    for line in out.splitlines():
        f = line.split(None, 2) if not line.startswith(" ") else ["", *line.split(None, 1)]
        if len(f) == 3 and "TwirlRecordSampler::Impl::" in f[2]:
            name = f[2].split("TwirlRecordSampler::Impl::", 1)[1].split("(", 1)[0]
            d.setdefault(name, set()).add(f[1])
    return d


@pytest.mark.parametrize("member", MEMBERS)
def test_single_owner(member):
    obj = _obj_dir()
    if obj is None:
        pytest.skip("no in-place build objects for this interpreter")
    ts, fam = os.path.join(obj, "twirl_sampler.o"), os.path.join(obj, "twirl_sampler_fam.o")
    if not (_is_elf(ts) and _is_elf(fam)):
        pytest.skip("ELF-only check (nm symbol classes T/W/t and comdat groups); these build objects are not ELF")
    s_ts, s_fam = syms(ts), syms(fam)
    assert s_ts.get(member, set()) & {"T", "W", "t"}, (member, s_ts.get(member))
    assert not (s_fam.get(member, set()) & {"T", "W", "t"}), (member, s_fam.get(member))
