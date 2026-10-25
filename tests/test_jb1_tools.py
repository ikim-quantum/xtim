"""J-b1 P0: the committed gate tools (tools/*.py) on synthetic inputs, plus the codegen gate against a base build
when one is supplied (skipped otherwise: XTIM_CODEGEN_BASE=<a built base tree>)."""
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
import repin_check  # noqa: E402

needs_gxx = pytest.mark.skipif(shutil.which("g++") is None or shutil.which("objdump") is None,
                               reason="needs g++ and objdump")


def _run(tool, *args):
    return subprocess.run([sys.executable, "-I", os.path.join(TOOLS, tool), *args], capture_output=True, text=True)


def _cc(tmp_path, name, src, shared=False, flags=("-O2",)):
    c = tmp_path / (name + ".cpp")
    c.write_text(src)
    out = tmp_path / (name + (".so" if shared else ".o"))
    cmd = ["g++", *flags, "-fPIC", "-std=c++17"] + (["-shared"] if shared else ["-c"]) + [str(c), "-o", str(out)]
    subprocess.run(cmd, check=True)
    return str(out)


SRC = """
namespace qeccore { struct Impl { template <int FW, bool FAM> bool run_w(long); int helper(int); int x = 1; };
__attribute__((noinline)) int Impl::helper(int a) { return a * %s + x; }
template <int FW, bool FAM> bool Impl::run_w(long n) { long s = 0; for (long i = 0; i < n; ++i) s += helper((int)i); return s > 7; }
template bool Impl::run_w<8, false>(long);
template bool Impl::run_w<8, true>(long);
}
"""


@needs_gxx
def test_codegen_eq_same_and_diff(tmp_path):
    a = _cc(tmp_path, "a", SRC % "3")
    b = _cc(tmp_path, "b", SRC % "3")
    c = _cc(tmp_path, "c", SRC % "5")
    r = _run("codegen_eq.py", a, b)
    assert r.returncode == 0 and "DIFFS=0" in r.stdout, r.stdout + r.stderr
    r = _run("codegen_eq.py", a, c)            # helper (a callee of run_w<8, false>) differs
    assert r.returncode == 1 and "DIFF" in r.stdout and "helper" in r.stdout, r.stdout
    r = _run("codegen_eq.py", a, c, "--allow", "qeccore::Impl::helper")
    assert r.returncode == 0, r.stdout
    r = _run("codegen_eq.py", a, c, "--roots", r"run_wILi\d+ELb1EEEbl$")   # the FAM instance pins its callee too
    assert r.returncode == 1 and "helper" in r.stdout, r.stdout
    r = _run("codegen_eq.py", a, b, "--roots", "no_such_root")
    assert r.returncode == 1                    # no root: never a vacuous pass


@needs_gxx
def test_codegen_so_same_and_diff(tmp_path):
    a = _cc(tmp_path, "a", SRC % "3", shared=True)
    b = _cc(tmp_path, "b", SRC % "3", shared=True)
    c = _cc(tmp_path, "c", SRC % "5", shared=True)
    o = _cc(tmp_path, "o", SRC % "5")
    r = _run("codegen_so.py", a, b, o, o)
    assert r.returncode == 0 and "DIFFS=0" in r.stdout, r.stdout + r.stderr
    assert "NEW" not in r.stdout and "new=0" in r.stdout, r.stdout   # a post-J-a base's run_w<N, false> is COMPARED
    fam = _cc(tmp_path, "fam", "int fam_only_fn(int a) { return a * 7; }\n")
    ts = _cc(tmp_path, "ts", SRC % "5")
    r = _run("codegen_so.py", a, c, ts, fam)          # helper is defined in the production TU: a production diff
    assert r.returncode == 1 and "DIFF  " in r.stdout, r.stdout
    r = _run("codegen_so.py", a, c, o, o)
    assert r.returncode == 1 and "DIFF" in r.stdout, r.stdout


FX = """
__attribute__((noinline)) int leaf_exact(int a) { return a * 3 + 1; }
__attribute__((noinline)) int shared_fn(int a) { return a ^ 5; }
__attribute__((noinline)) int exact_route(int a) { return leaf_exact(a) + shared_fn(a); }
__attribute__((noinline)) int law_root(int a) { return shared_fn(a) * 2; }
__attribute__((noinline)) int bad_root(int a) { return leaf_exact(a) - 1; }
int (*volatile fp)(int) = shared_fn;
__attribute__((noinline)) int indirect_root(int a) { return fp(a); }
"""


@needs_gxx
def test_fx_leaves_and_check(tmp_path):
    o = _cc(tmp_path, "fx", FX, flags=("-O2", "-fno-inline"))
    so = _cc(tmp_path, "fx", FX, shared=True)
    out = str(tmp_path / "leaves.json")
    r = _run("fx_leaves.py", "leaves", out, "--so", so, "--obj", o, "--exact-root", r"^exact_route\(",
             "--exact-root", r"^exact_route\(", "--law-root", r"^law_root\(")
    # the tool demands the four exact roots of the twirl sampler; a one-root toy is refused by name
    assert r.returncode != 0 and "roots not found" in (r.stdout + r.stderr)
    import fx_leaves
    E, I = fx_leaves.merged_graph([o, so])
    dm = fx_leaves.demangle(set(E))
    ex = fx_leaves.match(dm, [r"^exact_route\("])
    law = fx_leaves.match(dm, [r"^law_root\("])
    ce, _ = fx_leaves.closure(E, sorted(ex))
    cl, _ = fx_leaves.closure(E, sorted(law), barriers=ex)
    leaves = {dm[n] for n in (ce - cl) | ex}
    overlap = {dm[n] for n in ce & cl - ex}
    assert leaves == {"exact_route(int)", "leaf_exact(int)"} and overlap == {"shared_fn(int)"}
    json.dump({"leaves": sorted((ce - cl) | ex)}, open(out, "w"))
    assert _run("fx_leaves.py", "check", out, "--so", so, "--root", r"^law_root\(").returncode == 0
    r = _run("fx_leaves.py", "check", out, "--so", so, "--root", r"^bad_root\(")
    assert r.returncode == 1 and "REACHABLE LEAF" in r.stdout, r.stdout
    r = _run("fx_leaves.py", "check", out, "--so", so, "--root", r"^indirect_root\(")
    assert r.returncode == 1 and "UNRESOLVED INDIRECT" in r.stdout, r.stdout
    r = _run("fx_leaves.py", "check", out, "--so", so, "--root", r"^indirect_root\(", "--root", r"^shared_fn\(",
             "--indirect-ok", r"^indirect_root\(")
    assert r.returncode == 0, r.stdout


DIFF = """diff --git a/tests/test_a.py b/tests/test_a.py
--- a/tests/test_a.py
+++ b/tests/test_a.py
@@ -10 +10 @@ def test_pin():
-    assert h == "abc123"
+    assert h == "def456"
@@ -20,0 +21 @@ def test_other():
+    assert x == 7
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -3 +3 @@
-engine xtim-engine-12
+engine xtim-engine-13
"""


def test_repin_check():
    rows = [{"file": "tests/test_a.py", "old": "abc123", "new": "def456", "mechanism": "M-KEY"},
            {"file": "README.md", "new": "xtim-engine-13", "mechanism": "ENGINE_VERSION"}]
    H, un, stale = repin_check.check(DIFF, rows)
    assert len(H) == 3 and len(un) == 1 and un[0][1].startswith("@@ -20,0") and stale == []
    rows.append({"file": "tests/test_a.py", "hunk": "test_other", "mechanism": "new test row"})
    rows.append({"file": "tests/test_b.py", "old": "zzz", "mechanism": "stale"})
    H, un, stale = repin_check.check(DIFF, rows)
    assert un == [] and [r["mechanism"] for r in stale] == ["stale"]
    assert not repin_check.maps({"file": "tests/test_a.py", "mechanism": "no field"}, H[0])


BASE = os.environ.get("XTIM_CODEGEN_BASE")


@pytest.mark.skipif(not BASE, reason="no base build (XTIM_CODEGEN_BASE=<built base tree>)")
def test_codegen_vs_base_build():
    T = "build/temp.linux-x86_64-cpython-312/cpp/src"
    r = _run("codegen_eq.py", os.path.join(BASE, T, "twirl_sampler.o"), os.path.join(ROOT, T, "twirl_sampler.o"),
             "--allow", "qeccore::TwirlRecordSampler::run(long)")
    assert r.returncode == 0, r.stdout[-2000:]
