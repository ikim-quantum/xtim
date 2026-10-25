"""J-b2 J0: the gate-tool modes added for the deletion stage (plan r1 §4 J0 / T1-T2, r2.1), on synthetic inputs:
fx_leaves.py `absent` (G-ABSENT) and `dead` (the reachability report), codegen_eq.py `--disp` / `--map` and
codegen_so.py `--disp` (the DISPLACEMENT-ONLY object class)."""
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")

needs_gxx = pytest.mark.skipif(shutil.which("g++") is None or shutil.which("objdump") is None,
                               reason="needs g++ and objdump")


def _run(tool, *args):
    return subprocess.run([sys.executable, "-I", os.path.join(TOOLS, tool), *args], capture_output=True, text=True)


def _cc(tmp_path, name, src, shared=False):
    c = tmp_path / (name + ".cpp")
    c.write_text(src)
    out = tmp_path / (name + (".so" if shared else ".o"))
    subprocess.run(["g++", "-O2", "-fPIC", "-std=c++17"] + (["-shared"] if shared else ["-c"]) + [str(c), "-o", str(out)],
                   check=True)
    return str(out)


TOY = """
namespace qeccore {
__attribute__((noinline)) int kept_fn(int a) { return a * 3 + 1; }
__attribute__((noinline)) int legacy_only_fn(int a) { return a ^ 0x5a5a; }
const char* legacy_text() { return "QEC_TW_LEGACY_LEVER_TOY"; }
}
extern "C" int PyInit_toy(int a) { return qeccore::kept_fn(a); }
"""


@needs_gxx
def test_absent_binary_strings_and_sources(tmp_path):
    so = _cc(tmp_path, "toy", TOY, shared=True)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n# legacy_only_fn mentioned here\n")
    lst = tmp_path / "absent.txt"
    lst.write_text("# comment\nno_such_symbol_anywhere\n")
    r = _run("fx_leaves.py", "absent", str(lst), "--so", so, "--src", str(src))
    assert r.returncode == 0 and "present=0" in r.stdout, r.stdout + r.stderr
    lst.write_text("legacy_only_fn\n")                      # symbol in nm AND in a source file
    r = _run("fx_leaves.py", "absent", str(lst), "--so", so, "--src", str(src))
    assert r.returncode == 1 and "PRESENT nm" in r.stdout and "PRESENT src" in r.stdout, r.stdout
    lst.write_text("QEC_TW_LEGACY_LEVER_TOY  # src-ok\n")   # a rodata string; sources not checked
    r = _run("fx_leaves.py", "absent", str(lst), "--so", so, "--src", str(src))
    assert r.returncode == 1 and "PRESENT strings" in r.stdout and "PRESENT src" not in r.stdout, r.stdout


@needs_gxx
def test_dead_report(tmp_path):
    so = _cc(tmp_path, "toy", TOY, shared=True)
    out = str(tmp_path / "dead.json")
    r = _run("fx_leaves.py", "dead", out, "--so", so)
    assert r.returncode == 0, r.stdout + r.stderr
    dead = json.load(open(out))["dead"]
    # legacy_only_fn is exported (a dynamic symbol) but nothing references it: only a root makes it live
    assert not any("kept_fn" in d for d in dead) and any("legacy_only_fn" in d for d in dead), dead
    r = _run("fx_leaves.py", "dead", out, "--so", so, "--cut", r"kept_fn")
    assert any("kept_fn" in d for d in json.load(open(out))["dead"])


DISP = """
namespace qeccore { struct Impl { %s long a = 1; long b = 2; template <int FW, bool FAM> bool run_w(long); };
template <int FW, bool FAM> bool Impl::run_w(long n) { long s = 0; for (long i = 0; i < n; ++i) s += (i & 1) ? a : b;
  return s > %s; }
template bool Impl::run_w<8, true>(long);
}
"""


@needs_gxx
def test_codegen_disp_and_map(tmp_path):
    base = _cc(tmp_path, "base", DISP % ("long dead_member[40] = {};", "7"))
    tip = _cc(tmp_path, "tip", DISP % ("", "7"))                # a deleted data member: displacements move
    other = _cc(tmp_path, "other", DISP % ("", "9"))            # a real instruction change
    roots = r"run_wILi\d+ELb1EEEbl$"
    r = _run("codegen_eq.py", base, tip, "--roots", roots)
    assert r.returncode == 1 and "DIFF" in r.stdout, r.stdout
    r = _run("codegen_eq.py", base, tip, "--roots", roots, "--disp")
    assert r.returncode == 0 and "SAME-DISP" in r.stdout, r.stdout
    r = _run("codegen_eq.py", base, other, "--roots", roots, "--disp")
    assert r.returncode == 1, r.stdout                          # --disp never hides an immediate / opcode change
    r = _run("codegen_eq.py", base, base, "--roots", roots, "--map", "ELb1EEEbl=ELb1EEEbl")
    assert r.returncode == 0, r.stdout
    bso = _cc(tmp_path, "bso", DISP % ("long dead_member[40] = {};", "7"), shared=True)
    tso = _cc(tmp_path, "tso", DISP % ("", "7"), shared=True)
    r = _run("codegen_so.py", bso, tso, tip, tip)
    assert r.returncode == 1, r.stdout
    r = _run("codegen_so.py", bso, tso, tip, tip, "--disp")
    assert r.returncode == 0 and "DIFFS=0" in r.stdout, r.stdout
