#!/usr/bin/env python3
"""Validate the GPU xdem sampler on this machine's GPU in one command, and report (human summary + JSON; "format": 3).

    python tools/gpu_validate.py --out report.json            # the device xtim.gpu.available() finds
    python tools/gpu_validate.py --device cuda --out r.json   # insist on CUDA (fails if it cannot be opened)
    python tools/gpu_validate.py --quick --out r.json         # small shot counts (a minute or so)

Sections (each "passed", "failed" or "skipped"; the exit code is 0 only when every section that applies to the
device passed, and 1 otherwise):
  tests         ``pytest -m <device> tests``: the device's marked tests (bit-equality with the kernels' C++
                reference, the launch loop's golden bytes, the public API on the device). Fails if none ran.
  kernels       every bundled example the GPU takes (examples/xdem), with and without a postselection mask, with
                the normal and a tiny pair capacity (overflowing blocks; generated files only without a mask, since an overflowed
                block's per-shot outputs are unspecified and the walk's span screen makes them order-dependent), plus the modes of the sampler: the
                queued law (XK_MODE_DEFER: "resolve_law") and sample without the status (XK_MODE_NOKEY |
                XK_MODE_NOSTATUS); and generated files (the tests' generators: Bravyi-Haah-like files, whose law
                is the structured law, one-word and multi-word, and multi-word files of the generic law). All
                eight kernels of xtim/gpu/_stubs.py (walk, resolve, resolve_law, reduce, reduce_groups, reduce2,
                scan_blocks, emit) on the device and on the C++ reference (tests/_gpu_ref.py); every buffer
                byte-equal; the section fails if some kernel ran in no case. A failure names the first differing
                buffer, index and values.
  entry_points  GpuBackend's sample / sample_ps / sample(status=False) / sample_ps(status=False) / estimate on the
                device vs on the reference, byte-equal (default batches; in full mode also one-block batches), on
                the bundled files, the surface-code DEM and the generated files (each with the law path it takes);
                the status=False rows also equal the default call's rows on the device.
  stress        one device sampler shared by several threads (every result equal to its single-threaded one, with
                sample(status=False) among the calls); a
                file firing about 510 extended lines per shot (pair capacity near 2^17 at the first sample, then
                past 2^17 after forced relaunches: byte-equal to the reference); a DEM firing 4000 lines per shot
                (every submission's wall time, which the batch sizing keeps to a few seconds at most).
  benchmark     shots/s of XdemSampler.sample, sample(status=False), sample(postselect=all), infidelity(None) and
                infidelity(None, postselect=all) on cultivation_d3_faithful, cultivation_d5, color5_cultivation and
                two generated Bravyi-Haah-like files (bh_like36: one-word structured law; bh_like70: multi-word
                structured law; tests/_xdem_synth.py, not the external merlin_bh files): the device, and the
                CPU at 1 thread and at every thread. For the two infidelity calls the CPU is measured two ways:
                infidelity(None) itself, and sample(postselect=...) + score(status, zeros) in chunks of 2^20 shots
                (the stronger CPU baseline); ``speedup_vs_best_cpu`` is the device against the fastest CPU number of
                the row. The device's compile time (construction plus a first small call: kernel compile, table
                upload) is recorded apart from its run time.
With no device (``--device none``, or none found) the GPU sections are skipped and only the CPU benchmark runs; the
summary then says that nothing was validated on a GPU.

It needs the development install (``pip install -e ".[dev]"`` from a checkout: the reference kernels live in the test
extension xtim._xtim_testapi) and runs from any directory. It writes only the --out file.

Runbook: validating the CUDA path on a rented NVIDIA machine (also tools/GPU_VALIDATE.md)
-----------------------------------------------------------------------------------------
Any Linux machine with an NVIDIA GPU, its driver (``nvidia-smi`` works), Python >= 3.11, git and a C++17 compiler
(``sudo apt install -y build-essential python3-venv python3-dev`` on Ubuntu). The branch is perf/gpu-xdem-speed until
it is merged (then the default branch). What the CUDA build needs of the device: the walk kernel uses block shared
memory (XK_CAP_SHM 32-bit words per block), ``__syncthreads``, ``atomicAdd`` on 32-bit shared and device words (the
walk's lists, the law queue, emit's list) and ``atomicXor`` on 32-bit shared words and on 64-bit device words (the
rows), so compute capability 3.5 or later (64-bit ``atomicXor`` on unsigned long long). THE CUDA PATH HAS RUN ON ONE NVIDIA
GPU: a Quadro RTX 6000 (compute capability 7.5, driver 580.196.07 / CUDA 13.0, CuPy 14.2.0, Ubuntu 22.04, Python 3.13.18,
a Lambda on-demand cloud instance, 2026-11-25), where this tool passed (docs/assets/gpu/cuda-rtx6000.json). The report's
commit is 92c4627 with a dirty tree: the one difference was the test fix committed afterwards
(tests/test_gpu_cuda.py: CAPS1 lacked XK_CAP_SHM, which NVRTC rejects as fatal), plus untracked log files. Other cards,
drivers and CuPy versions are unchecked: a run on one may fail to compile or disagree with the reference; that is what
the run finds out.


    git clone <the repository URL> xtim && cd xtim && git checkout perf/gpu-xdem-speed
    python3 -m venv .venv && . .venv/bin/activate && python -m pip install -U pip
    pip install -e ".[dev]"                 # builds the C++ extension and its test API (a few minutes)
    nvidia-smi                              # "CUDA Version: 12.x" (top right) -> cupy-cuda12x; 11.x -> cupy-cuda11x;
                                            # 13.x -> cupy-cuda13x (worked: CuPy 14.2.0 on a 13.0 driver)
    pip install cupy-cuda12x                # if NVRTC is missing: pip install "cupy-cuda12x[ctk]" (unverified)
    python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"     # >= 1
    python -m pytest -m cuda tests -q       # the CUDA-marked tests
    python tools/gpu_validate.py --device cuda --out report.json

On the validated run (Lambda "GPU Base 22.04" image) the image's CUDA toolkit included NVRTC, so only cupy-cuda13x was
installed; the image ships Python 3.10 and 3.11 or newer is needed, so the environment was made with
``uv venv --python 3.13``.

A pass: pytest ends "N passed" with no failure and no skip, and the tool prints ``RESULT: PASS`` with every section
PASS (tests, kernels, entry_points, stress, benchmark), exit code 0. If anything fails, send back: report.json, the tool's
printed output, the pytest output (``python -m pytest -m cuda tests -q -rA 2>&1 | tail -200``), ``nvidia-smi``'s
output and ``pip list | grep -i -E "cupy|nvidia|numpy"``. A kernel failure names its first differing buffer, index
and values; a compile failure carries NVRTC's log. After a kernel fault (an illegal address, a launch failure) the
CUDA context is unusable for the rest of the process: only the first failing case is meaningful.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))   # tests/_gpu_ref.py, tests/_gpu_compare.py

import numpy as np  # noqa: E402

import xtim  # noqa: E402

BENCH_EXAMPLES = ("cultivation_d3_faithful", "cultivation_d5", "color5_cultivation")   # examples/xdem/<name>.xdem
BENCH_GENERATED = ("bh_like36", "bh_like70")   # generated by tests/_xdem_synth.py (see _generated)
BENCH_FILES = BENCH_EXAMPLES + BENCH_GENERATED
BENCH_ENTRIES = ("sample", "sample_nostatus", "sample_ps_all", "infidelity", "infidelity_ps_all")
REPORT_FORMAT = 3   # 2: sections tests..benchmark; 3: + the status=False entries, generated files, law paths
# (quick, full) shot counts
KERNEL_SHOTS = (16 * 64 - 37, 256 * 64 - 37)
ENTRY_SHOTS = (2 * 1024 + 37, 16 * 1024 + 37)
BENCH_SHOTS_DEVICE = (1_000_000, 50_000_000)
BENCH_SHOTS_CPU = (200_000, 10_000_000)
BENCH_OUT_BYTES = 512 << 20   # sample's returned arrays are kept under this (fewer shots for wide files)
SCORE_CHUNK = 1 << 20   # shots per sample + score call of the CPU's second infidelity baseline
STRESS_THREADS = 6
HEAVY_DEM_SHOTS = (200_000, 1_000_000)
SUBMISSION_LIMIT_S = 5.0   # the stress section's bound on one submission's wall time ("a few seconds")


def _launcher_class(device: str):
    """the launcher class of a device (tests replace this)"""
    if device == "metal":
        from xtim.gpu.metal import MetalLauncher
        return MetalLauncher
    from xtim.gpu.cuda import CudaLauncher
    return CudaLauncher


def _generated(name: str) -> tuple[str, dict]:
    """(xdem text, GpuBackend options) of a generated file, built by the tests' generators (tests/_xdem_synth.py,
    tests/_gpu_ref.py): Bravyi-Haah-like files (bh_like<k>[_signs]: k observables, 300 lines; the structured law, the
    default of the backend above 32 span dimensions; bh_like12_signs is forced onto it, bh_like36_generic onto the
    generic law) and multi-word files of the generic law (boundary_r65, wide_c)"""
    from _gpu_ref import multiword_text
    from _xdem_synth import bh_like_text
    if name in ("boundary_r65", "wide_c"):
        return multiword_text(name), {}
    k = int(name[len("bh_like"):].split("_")[0])
    signs = name.endswith("_signs")
    return bh_like_text(k, 300, seed=k + 7 * signs, signs=signs), (
        {"structured": True} if signs or k < 33 else {"structured": False} if name.endswith("_generic") else {})


def _bench_text(name: str) -> str:
    return (ROOT / "examples" / "xdem" / f"{name}.xdem").read_text() if name in BENCH_EXAMPLES else _generated(name)[0]


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()


# ------------------------------------------------------------------------------------------------ scrubbing


def _scrub_table() -> list:
    """(local string, placeholder) pairs, longest first: what a report must not carry (it gets committed)."""
    import getpass
    import socket
    pairs = [(str(ROOT), "<repo>"), (os.path.realpath(ROOT), "<repo>"), (os.path.realpath(sys.prefix), "<venv>"),
             (sys.prefix, "<venv>"), (os.getcwd(), "<cwd>"), (tempfile.gettempdir(), "<tmp>"),
             (os.path.realpath(tempfile.gettempdir()), "<tmp>"), (str(Path.home()), "~")]
    for name in (socket.gethostname(),):
        if len(name) >= 4:
            pairs.append((name, "<host>"))
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    if len(user) >= 4:
        pairs.append((user, "<user>"))
    return sorted({(a, b) for a, b in pairs if a and a != "/"}, key=lambda ab: -len(ab[0]))


_HOME_PATH = re.compile(r"(/Users|/home|/private/var|/var/folders|/private/tmp|/tmp|/root)/[^\s\"'/:)\]]*")


def _scrub(obj, table):
    """``obj`` with every local path, user name and host name in any string replaced (recursively): the table's
    names first, then any home / temp path that is left (a path of some other prefix than the table's)."""
    if isinstance(obj, str):
        for a, b in table:
            obj = obj.replace(a, b)
        return _HOME_PATH.sub(lambda m: "<path>", obj)
    if isinstance(obj, dict):
        return {k: _scrub(v, table) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v, table) for v in obj]
    return obj


def _argv_record(argv) -> list:
    """The options as given, with the value of ``--out`` reduced to its file name."""
    out, prev = [], None
    for x in argv:
        if prev == "--out":
            x = os.path.basename(x)
        elif x.startswith("--out="):
            x = "--out=" + os.path.basename(x[6:])
        out.append(x)
        prev = x
    return out


# ------------------------------------------------------------------------------------------------ environment


def _environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor(), "cpu_count": os.cpu_count(), "numpy": np.__version__,
           "xtim": getattr(xtim, "__version__", "?")}
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                           check=True).stdout.strip()
        env["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                               capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        env["git_commit"] = None
    return env


def _detect(want: str) -> tuple[str | None, dict, dict]:
    """(device or None, its description, why each device did not open)"""
    errors: dict[str, str] = {}
    names = ("metal", "cuda") if want == "auto" else (() if want == "none" else (want,))
    for name in names:
        try:
            L = _launcher_class(name)()
        except Exception as e:
            errors[name] = _error(e)
            continue
        try:
            info = L.describe()
        except Exception as e:
            info = {"describe_error": _error(e)}
        try:   # after a compile a launcher may also describe its kernels (CUDA: local memory and registers)
            from xtim.gpu.backend import _SOURCE
            t = xtim.XdemSampler((ROOT / "examples" / "xdem" / "cultivation_d5.xdem").read_text())._s.device_tables()
            from xtim.gpu import tables as gt
            L.compile(_SOURCE.read_text(), gt.capacities(t, 64, max(64, int(np.ceil(4 * 64 * t["mean_ext"])))))
            kernels = L.describe().get("kernels")
            if kernels:
                info["kernels"] = kernels
        except Exception as e:
            info["kernels_error"] = _error(e)
        return name, info, errors
    return None, {}, errors


# ------------------------------------------------------------------------------------------------ sections


def _section_tests(device: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, "-m", "pytest", "-m", device, "tests", "-q", "-rfEs",
                            "-p", "no:cacheprovider", f"--junitxml={junit}"], cwd=ROOT, capture_output=True,
                           text=True)
        out = {"seconds": round(time.perf_counter() - t0, 2), "returncode": p.returncode,
               "command": f"python -m pytest -m {device} tests"}
        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        failed = []
        if junit.exists():
            root = ET.parse(junit).getroot()
            suites = [root] if root.tag == "testsuite" else list(root)
            for s in suites:
                for k in counts:
                    counts[k] += int(s.get(k, 0))
                for case in s.iter("testcase"):
                    for bad in ("failure", "error"):
                        el = case.find(bad)
                        if el is not None:
                            failed.append({"test": f"{case.get('classname')}::{case.get('name')}",
                                           "message": (el.get("message") or "")[:2000]})
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    out.update(counts, passed=passed, failed_tests=failed, tail=p.stdout[-4000:] + p.stderr[-2000:])
    ok = p.returncode == 0 and passed > 0 and not failed
    out["status"] = "passed" if ok else "failed"
    out["summary"] = (f"{passed} passed, {counts['failures'] + counts['errors']} failed, {counts['skipped']} "
                      f"skipped ({out['seconds']} s)")
    if passed == 0:
        out["summary"] += ": no test ran on the device"
    return out


def _files(only: set | None):
    import _gpu_compare as C
    return [p for p in C.supported_files() if only is None or p.stem in only]


# generated files of the kernels and entry-point sections: (name, how the kernels run it)
GENERATED_FILES = ("bh_like12_signs", "bh_like36", "bh_like70", "bh_like36_generic", "boundary_r65", "wide_c")


def _kernel_cases(only: set | None):
    """(label, text, {backend options}) of every file the kernels section runs: the bundled examples, then the
    generated ones"""
    out = [(p.name, p.read_text(), {}) for p in _files(only)]
    for name in GENERATED_FILES:
        if only is None or name in only:
            text, kw = _generated(name)
            out.append((name, text, kw))
    return out


def _section_kernels(device: str, quick: bool, only: set | None) -> dict:
    import _gpu_compare as C
    from _gpu_ref import RefLauncher
    from xtim.gpu import _stubs
    from xtim.gpu import tables as gt
    from xtim.gpu.backend import GpuBackend
    new = _launcher_class(device)
    shots = KERNEL_SHOTS[0 if quick else 1]
    cases = []
    for label, text, kw in _kernel_cases(only):
        s = xtim.XdemSampler(text)
        t0_ = s._s.device_tables()
        # the law the backend would run on this file (its own choice, under the file's options), and whether it can
        # queue its laws (K2b: not the generic multi-word law)
        b = GpuBackend(s._s, RefLauncher(), **kw)
        structured, multi = b._plan is not None, bool(gt.coord_words(t0_))
        defer = bool(t0_["A"]) and (structured or not multi)
        t_law = b.tables   # renumbered under the structured law
        # (mask, defer, status) variants: no queue (the plain law inside "resolve"), the queued law, and sample
        # without the status; the tiny pair capacity (overflowing blocks) with the status
        variants = [(None, False, True), ("half", False, True)] if not (structured and multi) else []
        if defer:
            variants += [("half", True, True)]
        variants += [(None, defer, False), ("half", defer, False)]
        for kind, dfr, status in variants:
            # the tiny capacity: every block overflows (the batch is rerun by the host, which reads nothing else), so
            # the per-shot outputs of such blocks are unspecified; the walk's span screen (a file firing 3 lines
            # per shot or more, so the generated ones) makes `flag` there depend on the order of its atomics:
            # under a mask only the bundled files are compared at the tiny capacity
            tiny = status and not dfr and (kind is None or label not in GENERATED_FILES)
            for E in (C.pair_cap(t_law), 2) if tiny else (C.pair_cap(t_law),):
                case = {"file": label, "mask": kind, "pair_cap": E, "defer": dfr, "status": status,
                        "law_path": b.law_path, "shots": shots}
                try:
                    t, buf, plan = C.kernel_variant(s, kind, shots, 17, E, structured=structured, defer=dfr,
                                                    status=status, first_block=9)
                    ref = C.run_kernels(RefLauncher(), t, buf, E, shots, plan)
                    t0 = time.perf_counter()
                    got = C.run_kernels(new(), t, buf, E, shots, plan)
                    case.update(compile_s=round(got["_compile_s"], 4),
                                run_s=round(time.perf_counter() - t0 - got["_compile_s"], 4),
                                kernels=got["_kernels"])
                    case["diff"] = C.first_difference(ref, got)
                    case["ok"] = case["diff"] is None
                except Exception as e:
                    case.update(ok=False, error=_error(e), traceback=traceback.format_exc()[-3000:])
                cases.append(case)
    ok = sum(c["ok"] for c in cases)
    ran = {k for c in cases for k in c.get("kernels", [])}
    missing = sorted(set(_stubs.KERNELS) - ran)
    summary = (f"{ok}/{len(cases)} cases byte-equal to the reference ({len(ran)} of {len(_stubs.KERNELS)} kernels "
               f"run: {', '.join(sorted(ran))}; every buffer)")
    if missing and only is None:
        summary += f"; NOT RUN: {', '.join(missing)}"
    return {"status": "passed" if cases and ok == len(cases) and not (missing and only is None) else "failed",
            "cases": cases, "kernels": sorted(_stubs.KERNELS), "kernels_not_run": missing, "summary": summary}


def _section_entry_points(device: str, quick: bool, only: set | None) -> dict:
    import _gpu_compare as C
    from _gpu_ref import RefLauncher
    from xtim.gpu.backend import GpuBackend
    new = _launcher_class(device)
    shots = ENTRY_SHOTS[0 if quick else 1]
    todo = [(p.name, p.read_text(), {}) for p in _files(only)]
    surface = ROOT / "tests" / "data" / "xdem" / "synth" / "surface_d5r5.dem"
    if only is None or surface.stem in only:
        todo.append((surface.name, surface.read_text(), {}))
    for name in GENERATED_FILES:
        if only is None or name in only:
            text, kw = _generated(name)
            todo.append((name, text, kw))
    cases = []
    for label, text, opts in todo:
        try:
            law = GpuBackend(xtim.XdemSampler(text)._s, RefLauncher(), **opts).law_path
        except Exception as e:
            law = f"? ({_error(e)})"
        for kind in (None, "half"):
            want = None
            for kw in ({},) if quick else ({}, {"max_batch_bytes": 1}):
                case = {"file": label, "mask": kind, "shots": shots, "batches": "one-block" if kw else "default",
                        "law_path": law}
                try:
                    if want is None:
                        want = C.entry_outputs(text, kind, RefLauncher, shots, 2026, first_block=3, nostatus=True,
                                               **opts)
                    t0 = time.perf_counter()
                    got = C.entry_outputs(text, kind, new, shots, 2026, first_block=3, nostatus=True, **opts, **kw)
                    case["seconds"] = round(time.perf_counter() - t0, 4)
                    case["diff"] = C.first_difference(want, got)
                    if case["diff"] is None:   # status=False gives the default call's rows (on the device too)
                        stem = "sample" if kind is None else "sample_ps"
                        same = {f"{stem}_nostatus.{x}": f"{stem}.{x}" for x in ("dets", "obs")}
                        for a, b in same.items():
                            if got[a].tobytes() != got[b].tobytes():
                                case["diff"] = {"buffer": a, "index": -1, "ref": f"{b} (default call)", "got": "differs"}
                                break
                    case["ok"] = case["diff"] is None
                except Exception as e:
                    case.update(ok=False, error=_error(e), traceback=traceback.format_exc()[-3000:])
                cases.append(case)
    ok = sum(c["ok"] for c in cases)
    return {"status": "passed" if cases and ok == len(cases) else "failed", "cases": cases,
            "summary": f"{ok}/{len(cases)} cases byte-equal to the reference (sample / sample_ps / "
                       "sample(status=False) / sample_ps(status=False) / estimate)"}


def _section_stress(device: str, quick: bool) -> dict:
    import threading

    import _gpu_compare as C
    from _gpu_ref import RefLauncher
    from xtim.gpu.backend import GpuBackend
    new = _launcher_class(device)
    cases = []

    def case(name, file, fn):
        c = {"case": name, "file": file, "mask": None}
        t0 = time.perf_counter()
        try:
            c.update(fn())
        except Exception as e:
            c.update(ok=False, error=_error(e), traceback=traceback.format_exc()[-3000:])
        c["seconds"] = round(time.perf_counter() - t0, 3)
        cases.append(c)

    def threads():
        """STRESS_THREADS threads on one device sampler, each running every (call, seed) twice in its own order"""
        text = (ROOT / "examples" / "xdem" / "cultivation_d5.xdem").read_text()
        s = xtim.XdemSampler(text, device=device)
        mask = C.mask(s.num_detectors, "half").astype(bool)
        O = s.num_observables

        def cs(st):
            return [s._s.cspaces(int(i), 1)[0].tobytes() for i in st.cspace]

        jobs = {"sample_nostatus": lambda sd: (lambda d, o: (d.tobytes(), o.tobytes()))(
                    *s.sample(3000, seed=sd, status=False)),
                "sample": lambda sd: (lambda d, st: (d.tobytes(), st.obs.tobytes(), cs(st)))(*s.sample(3000, seed=sd)),
                "sample_ps": lambda sd: (lambda d, st, k: (d.tobytes(), cs(st), sorted(k.items())))(
                    *s.sample(3000, seed=sd, postselect=mask)),
                "infidelity_none": lambda sd: s.infidelity(20_000, None, seed=sd, postselect=mask),
                "infidelity_chunked": lambda sd: s.infidelity(
                    20_000, lambda d: np.zeros((len(d), O), bool) | d[:, :1], seed=sd, batch=1024)}
        want = {(j, sd): f(sd) for j, f in jobs.items() for sd in (1, 2, 3)}
        got, errs = [], []
        barrier = threading.Barrier(STRESS_THREADS)

        def work(k):
            try:
                barrier.wait()
                order = list(want)
                np.random.default_rng(k).shuffle(order)
                got.extend((key, jobs[key[0]](key[1])) for key in order * 2)
            except Exception as e:
                errs.append(_error(e))
        ts = [threading.Thread(target=work, args=(k,)) for k in range(STRESS_THREADS)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        bad = sorted({f"{k[0]} seed {k[1]}" for k, v in got if v != want[k]})
        return {"threads": STRESS_THREADS, "calls": len(got), "errors": errs[:3], "mismatches": bad[:10],
                "ok": not errs and not bad and len(got) == STRESS_THREADS * 2 * len(want)}

    def heavy_pairs():
        """about 510 extended lines per shot: the default pair capacity (near 2^17) at the first sample, then from
        2^16 relaunched past 2^17 (to 2^20) on blocks of 2^10 shots (about 522k records each); byte-equal to the
        reference (run at 2^20 from the start)"""
        text = C.heavy_ext(1700)
        s = xtim.XdemSampler(text, device=device)
        cap0 = s._gpu._pair_cap
        d, st = s.sample(3000, seed=2)
        want = GpuBackend(xtim.XdemSampler(text)._s, RefLauncher()).sample(3000, 2)
        same = d.tobytes() == want[0].tobytes() and st.obs.tobytes() == want[1].tobytes()
        cpu = xtim.XdemSampler(text)._s
        b = GpuBackend(cpu, new(), block=1 << 10)
        b._pair_cap = 1 << 16
        got = b.sample(3 * (1 << 10) + 5, 4)
        r = GpuBackend(xtim.XdemSampler(text)._s, RefLauncher(), block=1 << 10)
        r._pair_cap = 1 << 20   # the reference build's XK_CAP_E is 2^22: below the default 4 * 1024 * 510
        ref = r.sample(3 * (1 << 10) + 5, 4)
        same2 = all(x.tobytes() == y.tobytes() for x, y in zip(got[:2], ref[:2]))
        return {"pair_cap_first": cap0, "pair_cap_relaunched": b._pair_cap, "relaunches": b.relaunches,
                "ok": same and same2 and cap0 > 100_000 and b._pair_cap > 1 << 17}

    def heavy_dem():
        """a DEM firing 4000 lines per shot: estimate's submissions timed one by one"""
        cls = new
        times = []

        class Timed(cls):
            def run_batch(self, steps):
                t0 = time.perf_counter()
                super().run_batch(steps)
                if steps and steps[0][0] == "walk":
                    times.append((int(steps[1][1]), round(time.perf_counter() - t0, 3)))
        text = C.heavy_dem(lines=1_000_000, p=0.004)
        b = GpuBackend(xtim.XdemSampler(text)._s, Timed())
        b.prepare()
        n = HEAVY_DEM_SHOTS[0 if quick else 1]
        att, passed, hist = b.estimate(n, 1, None)
        worst = max(t for _, t in times)
        return {"shots": n, "submissions": times, "max_submission_s": worst, "limit_s": SUBMISSION_LIMIT_S,
                "ok": att == passed == n and worst <= SUBMISSION_LIMIT_S}

    case("threads", "cultivation_d5.xdem", threads)
    case("pair_capacity", "heavy_ext(1700)", heavy_pairs)
    case("submission_time", "heavy_dem(10^6 lines, 4000 fired per shot)", heavy_dem)
    ok = sum(bool(c.get("ok")) for c in cases)
    return {"status": "passed" if ok == len(cases) else "failed", "cases": cases,
            "summary": f"{ok}/{len(cases)} cases (threads, pair capacity past 2^17, submission time)"}


def _call(s, entry: str, shots: int, seed: int, threads: int):
    ps = np.ones(s.num_detectors, bool) if entry.endswith("_ps_all") else None
    if entry == "sample_nostatus":
        return s.sample(shots, seed=seed, threads=threads, status=False)
    if entry.startswith("sample"):
        return s.sample(shots, seed=seed, threads=threads, postselect=ps)
    return s.infidelity(shots, None, seed=seed, threads=threads, postselect=ps)


def _sample_score(s, entry: str, shots: int, seed: int, threads: int) -> None:
    """the second CPU baseline of an infidelity call: sample(postselect=...) + score(status, zeros), in chunks"""
    ps = np.ones(s.num_detectors, bool) if entry.endswith("_ps_all") else None
    zeros = np.zeros((min(shots, SCORE_CHUNK), s.num_observables), bool)
    for i, start in enumerate(range(0, shots, SCORE_CHUNK)):
        n = min(SCORE_CHUNK, shots - start)
        st = s.sample(n, seed=seed + i, threads=threads, postselect=ps)[1]
        s.score(st, zeros[:len(st)])


def _rate(fn, shots: int) -> float:
    t0 = time.perf_counter()
    fn()
    return shots / max(time.perf_counter() - t0, 1e-9)


def _section_benchmark(device: str | None, quick: bool, only: set | None, shots: int | None = None) -> dict:
    nd0, nc0 = (shots, shots) if shots else (BENCH_SHOTS_DEVICE[0 if quick else 1], BENCH_SHOTS_CPU[0 if quick else 1])
    threads = os.cpu_count() or 1
    rows = []
    for name in BENCH_FILES:
        if only is not None and name not in only:
            continue
        text = _bench_text(name)
        cpu = xtim.XdemSampler(text)
        per_shot = (cpu.num_detectors + 7) // 8 + (cpu.num_observables + 7) // 8 + 4
        for entry in BENCH_ENTRIES:
            cap = max(1, BENCH_OUT_BYTES // per_shot) if entry.startswith("sample") else nd0
            nd, nc = min(nd0, cap), min(nc0, cap)
            row = {"file": name, "source": f"examples/xdem/{name}.xdem" if name in BENCH_EXAMPLES else
                   f"generated: tests/_xdem_synth.py bh_like_text (k = {name[len('bh_like'):]}, 300 lines)",
                   "entry": entry, "device": None, "device_shots": nd, "device_compile_s": None,
                   "cpu_shots": nc, "threads_all": threads}
            try:
                _call(cpu, entry, 64, 0, 1)   # warm-up (law caches, mask)
                row["cpu_1"] = round(_rate(lambda: _call(cpu, entry, nc, 1, 1), nc), 1)
                row["cpu_all"] = round(_rate(lambda: _call(cpu, entry, nc, 1, threads), nc), 1)
                cpus = [row["cpu_1"], row["cpu_all"]]
                if entry.startswith("infidelity"):   # the stronger CPU baseline: sample + score(status, zeros)
                    row["cpu_score_1"] = round(_rate(lambda: _sample_score(cpu, entry, nc, 1, 1), nc), 1)
                    row["cpu_score_all"] = round(_rate(lambda: _sample_score(cpu, entry, nc, 1, threads), nc), 1)
                    cpus += [row["cpu_score_1"], row["cpu_score_all"]]
                row["cpu_best"] = max(cpus)
                if device is not None:
                    gpu = None   # free the previous row's sampler (its C table can hold millions of keys) untimed
                    gc.collect()
                    t0 = time.perf_counter()
                    gpu = xtim.XdemSampler(text, device=device)
                    _call(gpu, entry, 64, 0, 1)   # kernel compile, table upload, first buffers
                    row["device_compile_s"] = round(time.perf_counter() - t0, 4)
                    row["law_path"] = getattr(gpu._gpu, "law_path", None)
                    row["device"] = round(_rate(lambda: _call(gpu, entry, nd, 1, 1), nd), 1)
                    row["speedup_vs_cpu_all"] = round(row["device"] / row["cpu_all"], 2)
                    row["speedup_vs_best_cpu"] = round(row["device"] / row["cpu_best"], 2)
                row["ok"] = True
            except Exception as e:
                row.update(ok=False, error=_error(e), traceback=traceback.format_exc()[-3000:])
            rows.append(row)
    ok = all(r["ok"] for r in rows)
    return {"status": "passed" if rows and ok else "failed", "rows": rows,
            "summary": f"{sum(r['ok'] for r in rows)}/{len(rows)} measurements ran"}


# ------------------------------------------------------------------------------------------------ report


def _print_summary(r: dict, out_path: Path) -> None:
    dev = r["device"]
    mode = "quick" if r["quick"] else "full"
    info = r.get("device_info") or {}
    print(f"xtim GPU validation ({mode}), xtim {r['environment']['xtim']}, commit "
          f"{(r['environment'].get('git_commit') or '?')[:12]}")
    if dev is None:
        print("device: none -- nothing was validated on a GPU (only the CPU benchmark ran)")
        for k, v in r["device_errors"].items():
            print(f"  {k}: {v}")
    else:
        print(f"device: {dev} ({', '.join(f'{k}={v}' for k, v in info.items())})")
    for name, sec in r["sections"].items():
        st = {"passed": "PASS", "failed": "FAIL", "skipped": "skip"}[sec["status"]]
        print(f"  {name:<13} {st:<5} {sec.get('summary') or sec.get('reason', '')}")
        if sec["status"] == "failed":
            for t in sec.get("failed_tests", [])[:10]:
                print(f"      {t['test']}: {t['message'][:300]}")
            if sec.get("passed", 1) == 0 and "tail" in sec:
                print("      " + sec["tail"][-1500:].replace("\n", "\n      "))
            for c in [c for c in sec.get("cases", []) if not c["ok"]][:10]:
                where = (f"{c['file']} mask={c['mask']}" + (f" pair_cap={c['pair_cap']}" if "pair_cap" in c else "")
                         + (" defer" if c.get("defer") else "") + (" status=False" if c.get("status") is False else "")
                         + (f" case={c['case']}" if "case" in c else ""))
                if "batches" in c:
                    where += f" batches={c['batches']}"
                if c.get("diff"):
                    d = c["diff"]
                    print(f"      {where}: first difference {d['buffer']}[{d['index']}]: device {d['got']} != "
                          f"reference {d['ref']}")
                else:
                    print(f"      {where}: {c.get('error') or {k: v for k, v in c.items() if k != 'traceback'}}")
            for row in [x for x in sec.get("rows", []) if not x["ok"]][:10]:
                print(f"      {row['file']} {row['entry']}: {row['error']}")
    bench = r["sections"].get("benchmark", {})
    if bench.get("rows"):
        print("  shots/s (shots per row in the report; compile s = the device sampler's construction + first call;")
        print("  'call' = the entry itself on the CPU, 'smp+score' = sample + score(status, zeros), infidelity rows only;")
        print("  'x best' = device / the fastest CPU number of the row)")
        print(f"    {'file':<24}{'entry':<19}{'device':>12}{'call 1 thr':>12}{'call all':>12}{'smp+score 1':>12}"
              f"{'smp+score all':>14}{'x cpu all':>10}{'x best':>8}{'compile s':>10}")
        for row in bench["rows"]:
            def f(v, w, spec=",.0f"):
                return f"{'-' if v is None else format(v, spec):>{w}}"
            print(f"    {row['file']:<24}{row['entry']:<19}{f(row.get('device'), 12)}{f(row.get('cpu_1'), 12)}"
                  f"{f(row.get('cpu_all'), 12)}{f(row.get('cpu_score_1'), 12)}{f(row.get('cpu_score_all'), 14)}"
                  f"{f(row.get('speedup_vs_cpu_all'), 10, '.1f')}{f(row.get('speedup_vs_best_cpu'), 8, '.1f')}"
                  f"{f(row.get('device_compile_s'), 10, '.2f')}")
    print(f"RESULT: {'PASS' if r['passed'] else 'FAIL'}" + (" (no GPU)" if dev is None else ""))
    print(f"report: {out_path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the JSON report to write")
    ap.add_argument("--quick", action="store_true", help="small shot counts")
    ap.add_argument("--device", default="auto", choices=("auto", "metal", "cuda", "none"),
                    help="auto: the first device that opens (metal, then cuda); a name: that device or fail")
    ap.add_argument("--skip-tests", action="store_true", help="skip the pytest section")
    ap.add_argument("--skip-bench", action="store_true", help="skip the benchmark section")
    ap.add_argument("--bench-shots", type=int, default=None, help="shots per benchmark measurement (device and CPU)")
    ap.add_argument("--files", default=None, help="comma-separated file stems to restrict every section to")
    a = ap.parse_args(argv)
    only = None if a.files is None else {x.strip() for x in a.files.split(",") if x.strip()}
    out_path = Path(a.out).resolve()

    device, info, errors = _detect(a.device)
    r = {"tool": "tools/gpu_validate.py", "format": REPORT_FORMAT, "time": datetime.now(timezone.utc).isoformat(),
         "quick": a.quick, "argv": _argv_record(sys.argv[1:] if argv is None else argv), "device": device,
         "device_info": info, "device_errors": errors, "environment": _environment(), "sections": {}}
    sec = r["sections"]
    if device is None and a.device in ("metal", "cuda"):
        sec["device"] = {"status": "failed", "summary": f"--device {a.device} could not be opened: "
                                                         f"{errors.get(a.device)}"}
    gpu_skip = {"status": "skipped", "reason": "no GPU device"}

    def run(name, fn):
        t0 = time.perf_counter()
        try:
            sec[name] = fn()
        except Exception as e:
            sec[name] = {"status": "failed", "summary": _error(e), "traceback": traceback.format_exc()[-4000:]}
        sec[name].setdefault("seconds", round(time.perf_counter() - t0, 2))

    if device is None:
        sec["tests"] = dict(gpu_skip)
    elif a.skip_tests:
        sec["tests"] = {"status": "skipped", "reason": "--skip-tests"}
    else:
        run("tests", lambda: _section_tests(device))
    for name, fn in (("kernels", _section_kernels), ("entry_points", _section_entry_points)):
        if device is None:
            sec[name] = dict(gpu_skip)
        else:
            run(name, lambda fn=fn: fn(device, a.quick, only))
    if device is None:
        sec["stress"] = dict(gpu_skip)
    elif only is not None:
        sec["stress"] = {"status": "skipped", "reason": "--files given"}
    else:
        run("stress", lambda: _section_stress(device, a.quick))
    if a.skip_bench:
        sec["benchmark"] = {"status": "skipped", "reason": "--skip-bench"}
    else:
        run("benchmark", lambda: _section_benchmark(device, a.quick, only, a.bench_shots))
    r["passed"] = all(s["status"] != "failed" for s in sec.values())
    out_path.write_text(json.dumps(_scrub(r, _scrub_table()), indent=1, default=str) + "\n")
    _print_summary(r, out_path)
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
