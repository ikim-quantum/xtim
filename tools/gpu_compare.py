"""Time xtim's GPU xdem sampler against tsim's GPU sampler on the same circuits, same machine.

    python tools/gpu_compare.py --tsim-python /path/to/env/bin/python --out compare.jsonl CIRCUIT.stim ...

For every circuit and for postselection off / on every detector, one JSON line per measurement and a table at the
end, in ns per attempted shot (median of --reps calls of --shots shots, after one warm-up call of the same size):

    xtim-gpu sample       XdemSampler(device="gpu").sample: every shot's detectors, observables and status returned
    xtim-gpu nostatus     XdemSampler(device="gpu").sample(status=False): detectors and observables only ("sample,
                          no status": no C key on the device, no C id on the host)
    xtim-gpu estimate     XdemSampler(device="gpu").infidelity(shots, None): the zero decoder, counted on the device
    xtim-cpu sample       XdemSampler().sample at every thread
    tsim-gpu sample       tsim compile_detector_sampler(device="auto").sample, bit-packed, results read back
                          (run in the --tsim-python interpreter; skipped without it)

The xtim GPU sampler's one-word law path (GpuBackend.law_path: structured, generic or multi-word, with the reason
when not structured) is printed per circuit and recorded as "law" on its sample rows.
A circuit the xtim GPU sampler does not support is reported with xtim's reason and still timed on tsim and the xtim
CPU. tsim does not parse xtim's OUTPUT_QUBITS line: it is removed from tsim's input (noted in the row).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import warnings

_TSIM_CHILD = r'''
import sys, time, json, statistics, numpy as np, tsim
path, ps, n, reps, limit = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
lines = [l for l in open(path).read().splitlines() if not l.strip().startswith("OUTPUT_QUBITS")]
text = "\n".join(lines) + "\n"
t0 = time.perf_counter()
c = tsim.Circuit(text)
D = int(c.num_detectors)
s = c.compile_detector_sampler(seed=1, device="auto")
compile_s = time.perf_counter() - t0
mask = np.ones(D, dtype=bool) if ps else None
def call(k):
    r = s.sample(k, separate_observables=True, bit_packed=True, use_detector_reference_sample=True,
                 use_observable_reference_sample=True, postselection_mask=mask)
    a, b = np.asarray(r[0]), np.asarray(r[1])
    return len(a), int(a[:1].sum()) + int(b[:1].sum())
t0 = time.perf_counter(); call(n); first = time.perf_counter() - t0
ts = []; kept = 0
for i in range(reps):
    if i and (time.perf_counter() - t0) > limit: break
    t = time.perf_counter(); kept, _ = call(n); ts.append(time.perf_counter() - t)
stats = getattr(s, "stats", None)
stats = stats() if callable(stats) else stats
print(json.dumps(dict(ver=tsim.__version__, D=D, compile_s=compile_s, first_call_s=first, ns=statistics.median(ts) / n * 1e9,
                      ns_min=min(ts) / n * 1e9, accept=kept / n, device=str(getattr(stats, "device", "?")))))
'''


def _median_ns(call, shots: int, reps: int, limit: float):
    t0 = time.perf_counter()
    r = call(1)
    ts = []
    for i in range(reps):
        if i and time.perf_counter() - t0 > limit:
            break
        t = time.perf_counter()
        r = call(10 + i)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) / shots * 1e9, min(ts) / shots * 1e9, r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuits", nargs="+")
    ap.add_argument("--tsim-python", default=None, help="interpreter of an environment with tsim installed")
    ap.add_argument("--shots", type=int, default=1 << 24)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--limit", type=float, default=120.0, help="seconds per measurement before the reps stop")
    ap.add_argument("--out", default=None, help="append one JSON line per measurement here")
    ap.add_argument("--skip", default="", help="comma list of: xtim-gpu, xtim-cpu, tsim-gpu")
    a = ap.parse_args(argv)
    skip = set(filter(None, a.skip.split(",")))

    warnings.simplefilter("ignore")
    import numpy as np

    import xtim
    import xtim.gpu

    rows = []

    def emit(**r):
        rows.append(r)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "case"}, flush=True)

    threads = os.cpu_count() or 1
    for path in a.circuits:
        case = os.path.basename(path)[:-5]
        print(case, flush=True)
        text = open(path).read()
        t0 = time.perf_counter()
        x = xtim.Circuit(text).detector_error_model_text(format="xdem")
        export_s = time.perf_counter() - t0
        ok, why = xtim.gpu.supports(x)
        cpu = xtim.XdemSampler(x)
        D = int(cpu.num_detectors)
        gpu = None
        if ok and "xtim-gpu" not in skip:
            t0 = time.perf_counter()
            gpu = xtim.XdemSampler(x, device="gpu")
            gpu_s = time.perf_counter() - t0
            b = gpu._gpu   # the GpuBackend: which one-word law it runs (structured / generic / multi-word) and why
            print("   law path:", b.law_path, f"({b.law_path_reason})" if b.law_path_reason else "", flush=True)
        elif not ok:
            emit(case=case, tool="xtim-gpu", status="unsupported", why=why)
        for ps in (False, True):
            mask = np.ones(D, dtype=bool) if ps else None
            common = dict(case=case, ps=ps, shots=a.shots, D=D)
            if gpu is not None:
                ns, lo, r = _median_ns(lambda sd: gpu.sample(a.shots, seed=sd, postselect=mask), a.shots, a.reps, a.limit)
                emit(tool="xtim-gpu", entry="sample", ns=ns, ns_min=lo, accept=len(r[0]) / a.shots, build_s=export_s + gpu_s,
                     device=gpu.device, law=gpu._gpu.law_path, **common)
                ns, lo, r = _median_ns(lambda sd: gpu.sample(a.shots, seed=sd, postselect=mask, status=False), a.shots,
                                       a.reps, a.limit)
                emit(tool="xtim-gpu", entry="nostatus", ns=ns, ns_min=lo, accept=len(r[0]) / a.shots, **common)
                ns, lo, r = _median_ns(lambda sd: gpu.infidelity(a.shots, None, seed=sd, postselect=mask), a.shots,
                                       a.reps, a.limit)
                emit(tool="xtim-gpu", entry="estimate", ns=ns, ns_min=lo, **common)
            if "xtim-cpu" not in skip:
                n = min(a.shots, 1 << 22)
                ns, lo, r = _median_ns(lambda sd: cpu.sample(n, seed=sd, threads=threads, postselect=mask), n,
                                       min(a.reps, 3), a.limit)
                emit(tool="xtim-cpu", entry="sample", ns=ns, ns_min=lo, threads=threads, **dict(common, shots=n))
            if a.tsim_python and "tsim-gpu" not in skip:
                try:
                    q = subprocess.run([a.tsim_python, "-c", _TSIM_CHILD, path, "1" if ps else "0", str(a.shots),
                                        str(a.reps), str(a.limit)], capture_output=True, text=True, timeout=a.limit * 3)
                    if q.returncode == 0:
                        emit(tool="tsim-gpu", entry="sample", **{**json.loads(q.stdout.strip().splitlines()[-1]), **common})
                    else:
                        emit(tool="tsim-gpu", entry="sample", status="error",
                             why=" ".join(q.stderr.strip().splitlines()[-1:])[:300], **common)
                except subprocess.TimeoutExpired:
                    emit(tool="tsim-gpu", entry="sample", status="timeout", limit_s=a.limit * 3, **common)

    print("\nns per attempted shot (median)")
    cols = [("xtim-gpu", "sample"), ("xtim-gpu", "nostatus"), ("xtim-gpu", "estimate"), ("xtim-cpu", "sample"), ("tsim-gpu", "sample")]
    print(f"{'circuit':34s} {'ps':3s} " + " ".join(f"{t + ' ' + e:>18s}" for t, e in cols))
    for case in dict.fromkeys(r["case"] for r in rows):
        for ps in (False, True):
            cells = []
            for t, e in cols:
                m = [r for r in rows if r["case"] == case and r.get("ps") == ps and r["tool"] == t and r.get("entry") == e]
                u = [r for r in rows if r["case"] == case and r["tool"] == t and r.get("status") == "unsupported"]
                cells.append(f"{m[0]['ns']:18.2f}" if m and "ns" in m[0] else
                             f"{(m[0].get('status') if m else ('unsupported' if u else '-')):>18s}")
            print(f"{case:34s} {'yes' if ps else 'no':3s} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
