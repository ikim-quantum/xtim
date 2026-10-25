"""Zero allocations PER SHOT on the warm fast paths (D14 ledger, stage LB2; owner decision O1 "A + D").

A warm sampler call may allocate a per-CALL constant (output arrays, binding conversions), but nothing
that scales with the shot count: for a fixed circuit and seed, a fully warm call of N2 shots performs
EXACTLY as many allocation calls as a fully warm call of N1 < N2 shots. Covered warm paths:
  * t_fast/d5s  (cultivation_d5, sample mode): the warm hot loop + the noise draw (event source);
  * t_fast/fac  (private CCZ factory a):        hot loop + noise draw + the Born-trie warm walk (kappa > 0);
  * t_fast/fac_b (private CCZ factory b):       the same on the second factory (both skip when absent);
  * *_ps: the same calls with a postselection mask on every detector (early-reject: the masked shot loop, its
    early test and the late filter).
"Warm" = every plan the call reaches is resident: each measured call repeats an earlier identical
(shots, seed) call, and the larger call is measured first so any retained per-call buffer is already
at its final capacity.

Mechanism: a counting allocator interposed with LD_PRELOAD in a child process (malloc / calloc /
realloc / free and the aligned entry points, forwarded to glibc's __libc_* entry points; operator
new/delete reach it through malloc/free). Counting is switched on around the measured call only.
Linux/glibc + a C compiler required (skipped otherwise).

Scope: the C heap (malloc family, aligned entry points, operator new/delete). Python-heap allocations
(PyObject_Malloc / PyMem_*) cannot be interposed here (the interpreter is statically linked, so its own definitions
win symbol lookup); in-window interpreter work is gated by the D14 ledger instead (python-class Ir in a measured
window must be 0; benchmarks/ledger, LB2).
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import xtim
from _private import private_fixture, private_path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"

WORKLOADS = [  # name, circuit, (N1, N2) shots
    ("d5s", ROOT / "tests" / "data" / "cultivation_d5.stim", (2000, 4000)),
    ("fac", private_path("ccz_factory_a.stim"), (150, 300)),
    ("fac_b", private_path("ccz_factory_b.stim"), (150, 300)),
    ("d5s_ps", ROOT / "tests" / "data" / "cultivation_d5.stim", (2000, 4000)),
    # (the accepted rows of both calls stay above numpy's small-buffer cache, 1 KiB: its hits make no malloc call)
    ("fac_ps", private_path("ccz_factory_a.stim"), (300, 600)),
]
SEED = 5

SHIM = r"""
#include <stddef.h>
#include <stdatomic.h>
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);
extern void __libc_free(void *);
static atomic_int on;
static atomic_long n_alloc, n_free;
void za_start(void) { atomic_store(&n_alloc, 0); atomic_store(&n_free, 0); atomic_store(&on, 1); }
void za_stop(long *out) { atomic_store(&on, 0); out[0] = atomic_load(&n_alloc); out[1] = atomic_load(&n_free); }
#define A() do { if (atomic_load_explicit(&on, memory_order_relaxed)) atomic_fetch_add(&n_alloc, 1); } while (0)
void *malloc(size_t n) { A(); return __libc_malloc(n); }
void *calloc(size_t a, size_t b) { A(); return __libc_calloc(a, b); }
void *realloc(void *p, size_t n) { A(); return __libc_realloc(p, n); }
void *memalign(size_t al, size_t n) { A(); return __libc_memalign(al, n); }
void *aligned_alloc(size_t al, size_t n) { A(); return __libc_memalign(al, n); }
int posix_memalign(void **pp, size_t al, size_t n) {
    A();
    if (al % sizeof(void *) || (al & (al - 1))) return 22;
    void *p = __libc_memalign(al, n);
    if (!p) return 12;
    *pp = p;
    return 0;
}
void free(void *p) {
    if (p && atomic_load_explicit(&on, memory_order_relaxed)) atomic_fetch_add(&n_free, 1);
    __libc_free(p);
}
"""

CHILD = textwrap.dedent(r"""
    import ctypes, json, os, sys
    sys.path.insert(0, sys.argv[1])
    import xtim._xtim as _x
    shim = ctypes.CDLL(sys.argv[2])
    out = (ctypes.c_long * 2)()
    def counted(f):
        shim.za_start(); f(); shim.za_stop(out)
        return [out[0], out[1]]
    res = {}
    for name, circ, n1, n2 in json.loads(sys.argv[3]):
        s = _x.TwirlSampler(open(circ).read(), 1.0, "", 0)
        kw = {"postselect": list(range(s.num_detectors))} if name.endswith("_ps") else {}
        s.sample(n2, seed=%d, **kw); s.sample(n1, seed=%d, **kw)          # cold, then warm-up of both sizes
        a2 = counted(lambda: s.sample(n2, seed=%d, **kw))
        a1 = counted(lambda: s.sample(n1, seed=%d, **kw))
        a2b = counted(lambda: s.sample(n2, seed=%d, **kw))
        res[name] = {"n1": n1, "n2": n2, "a1": a1, "a2": a2, "a2b": a2b}
    print(json.dumps(res))
""" % ((SEED,) * 5))


def _cc():
    for c in ("cc", "gcc", "clang"):
        if shutil.which(c):
            return c
    return None


@pytest.fixture(scope="module")
def counts(tmp_path_factory):
    if platform.system() != "Linux" or platform.libc_ver()[0] != "glibc" or _cc() is None:
        pytest.skip("needs Linux/glibc and a C compiler (LD_PRELOAD counting allocator)")
    d = tmp_path_factory.mktemp("zero_alloc")
    (d / "za.c").write_text(SHIM)
    so = d / "libza.so"
    subprocess.run([_cc(), "-O2", "-fPIC", "-shared", str(d / "za.c"), "-o", str(so)], check=True)
    pkg_parent = str(Path(xtim.__file__).resolve().parents[1])
    wl = [[n, str(c), a, b] for n, c, (a, b) in WORKLOADS if Path(c).is_file()]
    env = dict(os.environ, LD_PRELOAD=str(so))
    r = subprocess.run([sys.executable, "-I", "-c", CHILD, pkg_parent, str(so), json.dumps(wl)], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_counter_is_live(counts):
    # non-vacuity: the counter sees the per-call allocations of every warm call
    for name, c in counts.items():
        assert c["a1"][0] > 0 and c["a2"][0] > 0, (name, c)


@pytest.mark.parametrize("name", [w[0] for w in WORKLOADS])
def test_zero_allocations_per_shot_warm(counts, name):
    if name not in counts:
        private_fixture(os.path.basename(str(dict((w[0], w[1]) for w in WORKLOADS)[name])))
    c = counts[name]
    # non-vacuity per test: the counter must have seen this workload's per-call allocations
    assert c["a1"][0] > 0 and c["a2"][0] > 0, (name, c)
    per_shot = (c["a2"][0] - c["a1"][0]) / (c["n2"] - c["n1"])
    assert c["a2"][0] == c["a1"][0], (
        f"{name}: warm {c['n2']}-shot call made {c['a2'][0]} allocations, warm {c['n1']}-shot call "
        f"{c['a1'][0]}: {per_shot:+.4f} allocations per shot (must be 0)")
    assert c["a2b"] == c["a2"], (name, c)          # warm calls are allocation-deterministic
