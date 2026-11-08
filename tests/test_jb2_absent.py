"""J-b2 G-ABSENT (plan r1 §5.1; J9: the former J-a G-SWITCH, test_ja_switch.py, inverted at the J-b1 flip).

  (absence) every deleted legacy twirl name in tools/jb2_absent.txt is absent from both extensions' symbols / rodata
      and from the sources (tools/fx_leaves.py absent; "# src-ok" lines are binary-only);
  (a) every public sampler is the family sampler with every record class: no public module, binding or keyword
      reaches a legacy sampler, and (J-b2 J2) neither does the test API (its legacy surface is deleted);
  (b) each extension holds exactly one run_w<FW> instance per FW (J-b2 J9: the always-true FAM parameter is
      dropped; no run_w<FW, true / false> name remains), and no binding class of the legacy surface exists;
  (c) production behaviour is the family law: the gauge detector of GATE1 is an exact Born record, ANTI detectors are
      served, and the public sampler is byte-identical to the test API's compile_family_sampler
      (F-B1a in pytest form).
Q7: oracle_all_shots (the semantic oracle on every shot) is stream-neutral — bytes identical with and without it.
"""
import glob
import os
import re
import subprocess

import numpy as np
import pytest

import xtim
import _ja_fixtures as F

pytestmark = pytest.mark.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_TOKENS = ("compile_legacy_sampler", "legacy_twirl.hpp", "LegacyTwirlSampler", "LegacyBarrierBuffer",
                 "LegacyRecordFidelity")


def _public_sources():
    return sorted(glob.glob(os.path.join(ROOT, "xtim", "*.py"))) + [
        os.path.join(ROOT, "cpp", "bindings", "xtim_py.cpp"), os.path.join(ROOT, "cpp", "bindings", "family_sampler.hpp")]


def test_a_no_public_module_reaches_the_legacy_sampler():
    hits = []
    for path in _public_sources():
        text = open(path).read()
        for tok in LEGACY_TOKENS:
            if tok in text:
                hits.append((os.path.relpath(path, ROOT), tok))
        # the legacy instantiation is chosen by family = false: no public binding sets it
        if re.search(r"\.\s*family\s*=\s*false", text):
            hits.append((os.path.relpath(path, ROOT), "family = false"))
    assert hits == []
    for name in LEGACY_TOKENS[2:] + ("compile_legacy_sampler",):
        assert not hasattr(xtim._xtim, name), name
    from xtim import _xtim_testapi as TA
    for name in LEGACY_TOKENS[2:] + ("compile_legacy_sampler", "legacy_bare_state_of", "LegacyRecordFidelity"):
        assert not hasattr(TA, name), name                    # J-b2 J2: the test API's legacy surface is deleted
    s = xtim.compile_twirl_sampler(F.GATE1, selfcheck=0)
    rep = s.record_report()
    assert rep["classes"]["detectors"][0] == "born"           # the family law's record classes
    assert "route_counts" in rep and not [k for k in rep["route_counts"] if k.startswith("exact")]


def _nm(path):
    return subprocess.run(["nm", "-C", path], check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize("ext", ["_xtim", "_xtim_testapi"])
def test_b_one_instance_per_fw_and_bindings_never_reach_the_legacy_surface(ext):
    so = glob.glob(os.path.join(ROOT, "xtim", ext + ".cpython-*.so"))[0]
    syms = _nm(so)
    for fw in (8, 0):
        inst = re.findall(r"TwirlRecordSampler::Impl::run_w<%d>\(long\)" % fw, syms)
        assert len(inst) == 1, (ext, fw, inst)
    assert not re.findall(r"run_w<\d+, (?:true|false)>", syms)   # J-b2 J9: the FAM parameter is gone
    assert "PyTwirlSampler" not in syms and "PyBarrierBuffer" not in syms and "PyRecordFidelity" not in syms
    objs = glob.glob(os.path.join(ROOT, "build", "temp.*", "cpp", "bindings", "xtim_py.o"))
    if ext == "_xtim" and objs:
        undef = subprocess.run(["nm", "-C", "-u", objs[0]], check=True, capture_output=True, text=True).stdout
        assert "TwirlSamplerTestAccess" not in undef


def test_absence_list_is_absent():
    """tools/jb2_absent.txt: every pattern absent from both extensions and (unless "# src-ok") from the sources."""
    import sys
    sos = sorted(glob.glob(os.path.join(ROOT, "xtim", "_xtim*.cpython-*.so")))
    assert len(sos) == 2, sos
    cmd = [sys.executable, "-I", os.path.join(ROOT, "tools", "fx_leaves.py"), "absent",
           os.path.join(ROOT, "tools", "jb2_absent.txt")]
    for so in sos:
        cmd += ["--so", so]
    for d in ("cpp", "xtim", "tests", "tools"):
        cmd += ["--src", os.path.join(ROOT, d)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0 and re.search(r"present=0\b", p.stdout), p.stdout + p.stderr
    n = int(re.search(r"patterns=(\d+)", p.stdout).group(1))
    assert n >= 160, p.stdout                                  # non-vacuity: the cumulative list is read


def _public(text):
    return xtim.compile_twirl_sampler(text, selfcheck=0)


def test_c_public_gauge_detector_is_a_born_record():
    d, _ = _public(F.GATE1).sample(40_000, seed=3)
    p = F.unpack(d, 2)[:, 0].mean()
    assert abs(p - 0.5) > 0.2, p                              # the exact Born record (the law's 0.146), not a coin
    s = _public(F.DESIGN["bell_anti"])                        # ANTI detectors are served (were refused)
    s.sample(100, seed=1)


def _rep_memory(d, rounds):
    import stim
    return str(stim.Circuit.generated("repetition_code:memory", distance=d, rounds=rounds,
                                      after_clifford_depolarization=0.01, before_measure_flip_probability=0.01))


# The word-path boundaries (J-a final review M1) plus the gauge / ANTI / decision fixtures: the public sampler and the
# test API's FAM(all) sampler are one code path (cpp/bindings/family_sampler.hpp) — byte-identical (F-B1a).
IDENTITY = {
    "gate1": F.GATE1, "p4_gauge": F.DESIGN["p4_gauge"], "convention_invert": F.CONVENTION["invert"],
    "m3_obs_2dec": F.FIXTURES["m3_obs_2dec"], "sig_comb_x": F.FIXTURES["sig_comb_x"],
    "q3_dec": F.FIXTURES["q3_dec"], "a4_comb": F.FIXTURES["a4_comb"],
    "gauge_last": F.GATE1_LAST, "rep70": _rep_memory(36, 1), "rep80": _rep_memory(17, 4),
    "rep128": _rep_memory(65, 1), "bell_anti": F.DESIGN["bell_anti"],
}


@pytest.mark.parametrize("name", sorted(IDENTITY))
def test_c_public_is_byte_identical_to_fam_all(name):
    text = IDENTITY[name]
    pub = _public(text)
    fam = F.family(text)                                      # classes = ALL
    r = pub.sample(5000, seed=9)
    d, o, dc = fam.sample(5000, seed=9)
    exp = (np.asarray(r.dets), np.asarray(r.obs), np.asarray(r.decisions)) if hasattr(r, "decisions") else \
        (np.asarray(r[0]), np.asarray(r[1]))
    assert np.array_equal(d, exp[0]) and np.array_equal(o, exp[1])
    if len(exp) == 3:
        assert np.array_equal(dc, exp[2])
    bp, bf = pub.sample_barrier(800, seed=5), fam.sample_barrier(800, seed=5)
    assert bp.record_keys() == bf.record_keys()


def test_c_fam_refusals_by_name():
    """Retention of a family sampler is the layout-3 record route: sample(retain=True) is refused by name (J-b2 J6:
    the class-subset refusals went with the class bits)."""
    s = F.family(F.GATE1)
    with pytest.raises(RuntimeError, match="sigma export is refused on a family sampler"):
        s.sample(10, retain=True)


# ── Q7: oracle_all_shots is stream-neutral ──────────────────────────────────────────────────────
Q7 = dict(F.DESIGN)
Q7.update({k: v for k, v in sorted(F.twins().items())[:12]})


@pytest.mark.parametrize("name", sorted(Q7))
def test_q7_oracle_all_shots_is_stream_neutral(name):
    text = Q7[name]
    a = F.family(text).sample(3000, seed=4)
    s = F.family(text, oracle_all_shots=True)
    b = s.sample(3000, seed=4)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert s.oracle_counts()["shots"] >= 3000


# ── the selfcheck notice is once per process per extension (the FAM TU's own static) ────────────
_NOTICE = (
    "import json, sys, warnings, xtim\n"
    "from xtim import _xtim_testapi as TA\n"
    "text = sys.stdin.read()\n"
    "out = []\n"
    "s = xtim.compile_twirl_sampler(text, selfcheck=2000)\n"
    "with warnings.catch_warnings(record=True) as rec:\n"
    "    warnings.simplefilter('always')\n"
    "    s.sample(300, seed=2); s.sample(300, seed=3)\n"
    "    xtim.compile_twirl_sampler(text, selfcheck=2000).sample(300, seed=4)\n"
    "out.append([str(w.message) for w in rec if issubclass(w.category, xtim.XtimVerificationWarning)])\n"
    "f = TA.compile_family_sampler(text, selfcheck=2000)\n"
    "with warnings.catch_warnings(record=True) as rec:\n"
    "    warnings.simplefilter('always')\n"
    "    f.sample(300, seed=2)\n"
    "out.append([str(w.message) for w in rec if issubclass(w.category, xtim.XtimVerificationWarning)])\n"
    "print(json.dumps(out))\n")


def test_selfcheck_notice_is_once_per_process_per_extension():
    """Production announces its oracle window exactly once per process (the FAM TU's static), however many samplers
    run; the test API is a separate engine copy with its own once. simplefilter('always') (no per-location dedup): neither count is vacuous."""
    import json
    import sys
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    cult = open(os.path.join(ROOT, "tests", "data", "cultivation_d5.stim")).read()
    p = subprocess.run([sys.executable, "-c", _NOTICE], input=cult, capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    prod, tapi = json.loads(p.stdout.splitlines()[-1])
    assert len(prod) == 1 and "selfcheck=2000" in prod[0], prod
    assert len(tapi) == 1, tapi
