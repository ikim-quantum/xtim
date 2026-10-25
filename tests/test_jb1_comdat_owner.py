"""J-b1 P6c (fixplan-coldstart (B), ruling: twirl_sampler.cpp is the single out-of-line owner): the members
TwirlRecordSampler::Impl shares between the production TU (twirl_sampler.cpp) and the family TU
(twirl_sampler_fam.cpp) are DEFINED in the production TU only; the family TU references them (nm U), so the FAM hot
path links exactly one body per member — today's (the linker already chose the production TU's copy) — and it
cannot move when J-b2 removes the legacy run_w. Checked by symbol on the in-place build objects (skipped without
them); tools/codegen_eq.py --comdat (vague-linkage symbols only) reports no shared Impl member."""
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ = os.path.join(ROOT, "build", "temp.linux-x86_64-cpython-312", "cpp", "src")
MEMBERS = ["record_rows_<false, false>", "record_rows_<false, true>", "record_rows_<true, false>",
           "record_rows_<true, true>", "build_rec_cold_", "payload_boundary_", "identity_plan_", "parse_plan_key",
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
    ts, fam = os.path.join(OBJ, "twirl_sampler.o"), os.path.join(OBJ, "twirl_sampler_fam.o")
    if not (os.path.exists(ts) and os.path.exists(fam)):
        pytest.skip("no in-place build objects")
    s_ts, s_fam = syms(ts), syms(fam)
    assert s_ts.get(member, set()) & {"T", "W", "t"}, (member, s_ts.get(member))
    assert not (s_fam.get(member, set()) & {"T", "W", "t"}), (member, s_fam.get(member))
