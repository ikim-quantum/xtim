"""Stage-2 scaling acceptance table: compile, compile + sample(1000) and DEM export time and peak memory on the large
Clifford circuits, one fresh process per cell. Prints a markdown table (and, with --json, writes the raw cells).

Cells (default): chain N = 100 000 (compile, DEM), REPEAT N = 100 000 (compile, DEM), surface d = 21, 25 and 29 with
d rounds (compile, compile + sample(1000), DEM). --cells picks a subset, e.g. --cells surface:25:compile,dem.
  chain   : N x (R q / X_ERROR(0.01) q / M q / DETECTOR rec[-1]) on N qubits;
  repeat  : REPEAT N { X_ERROR(0.01) 0 / M 0 / DETECTOR rec[-1] / R 0 } on one qubit;
  surface : stim's surface_code:rotated_memory_z, distance d, d rounds, p = 0.001 (needs stim).
Each cell reports the operation's wall time and CPU time (circuit parsing excluded), the process's peak RSS
(resource.ru_maxrss) and its RSS before the operation (about 40-200 MB of interpreter, extension and parsed circuit,
growing with the circuit), so a FAIL can be split into the circuit's share and the operation's. The criteria
column is the Stage-2 plan's: chain 100k compile <= 5 s / <= 600 MB and DEM <= 5 s; surface d = 21 compile <= 3 s and DEM <= 3 s.
--repeat k runs every cell in k fresh processes and reports the medians (the verdict is on the median wall time) and
the wall-time spread: use it for a cell near its bound (chain 100k compile).

Heavy (on the final Stage-2 code each cell peaks under 600 MB and the table takes about 3 minutes with --repeat 3;
older builds needed tens of GB; a cell that runs out of the address space is reported, not fatal): run it on a
quiet machine, pinned, e.g.
  taskset -c 5 python benchmarks/compile_dem_acceptance.py --label <commit> --repeat 3
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

DEFAULT_CELLS = ("chain:100000:compile,dem", "repeat:100000:compile,dem", "surface:21:compile,sample,dem",
                 "surface:25:compile,sample,dem", "surface:29:compile,sample,dem")
OP_NAMES = {"compile": "compile", "sample": "compile + sample(1000)", "dem": "DEM export"}
# (family, size, op) -> (max wall seconds, max peak RSS MB or None): the plan's acceptance criteria
CRITERIA = {("chain", 100000, "compile"): (5.0, 600.0), ("chain", 100000, "dem"): (5.0, None),
            ("surface", 21, "compile"): (3.0, None), ("surface", 21, "dem"): (3.0, None)}

_WORKER = r"""
import json, os, resource, sys, time
import xtim
family, size, op, want = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
got = os.path.dirname(os.path.realpath(xtim.__file__))
if got != want:
    sys.exit(f"XTIM-PATH worker imported xtim from {got}, not from {want}")
if family == "chain":
    text = "".join(f"R {q}\nX_ERROR(0.01) {q}\nM {q}\nDETECTOR rec[-1]\n" for q in range(size))
elif family == "repeat":
    text = f"REPEAT {size} {{\nX_ERROR(0.01) 0\nM 0\nDETECTOR rec[-1]\nR 0\n}}\n"
else:
    import stim
    text = str(stim.Circuit.generated("surface_code:rotated_memory_z", distance=size, rounds=size,
               after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
               after_reset_flip_probability=0.001))
c = xtim.Circuit(text)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
w0, t0 = time.perf_counter(), time.process_time()
if op == "compile":
    c.compile_detector_sampler(seed=1)
elif op == "sample":
    c.compile_detector_sampler(seed=1).sample(1000)
else:
    c.detector_error_model_text()
w1, t1 = time.perf_counter(), time.process_time()
print("CELL " + json.dumps({"M": c.num_measurements, "wall": w1 - w0, "cpu": t1 - t0, "base_mb": base,
      "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}))
"""


def run_cell(family, size, op, timeout=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # one BLAS thread: numpy's OpenBLAS pool otherwise busy-waits after a BLAS call and inflates the CPU column
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        # cwd=root: `python -c` puts the working directory first on sys.path (an xtim/ there would shadow ours)
        want = os.path.join(os.path.realpath(root), "xtim")
        p = subprocess.run([sys.executable, "-c", _WORKER, family, str(size), op, want], env=env, cwd=root,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"over {timeout:g} s"}
    if "XTIM-PATH" in p.stderr:
        raise SystemExit(p.stderr.strip())        # a setup error, not a measurement: stop
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("CELL ")]
    if p.returncode != 0 or not lines:
        err = (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]
        return {"error": "out of memory" if "MemoryError" in err or "bad_alloc" in err else err[:80]}
    return json.loads(lines[-1][len("CELL "):])


def run_repeated(family, size, op, repeat, timeout=None):
    """The median over `repeat` fresh processes of each of wall, CPU, peak RSS and RSS before the op, plus the spread
    (min, max) of the wall times; the first failing run is returned as is."""
    rs = []
    for _ in range(repeat):
        r = run_cell(family, size, op, timeout)
        if "error" in r:
            return r
        rs.append(r)
    out = dict(rs[0], **{k: statistics.median(x[k] for x in rs) for k in ("wall", "cpu", "rss_mb", "base_mb")})
    out["walls"] = [x["wall"] for x in rs]
    return out


def circuit_name(family, size):
    return {"chain": f"chain N = {size}", "repeat": f"REPEAT N = {size}",
            "surface": f"surface d = {size}, {size} rounds"}[family]


def verdict(family, size, op, r):
    crit = CRITERIA.get((family, size, op))
    if crit is None:
        return "", ""
    t_max, mb_max = crit
    text = f"<= {t_max:g} s" + (f", <= {mb_max:g} MB" if mb_max else "")
    if "error" in r:
        return text, "FAIL"
    ok = r["wall"] <= t_max and (mb_max is None or r["rss_mb"] <= mb_max)
    return text, "pass" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", nargs="*", default=list(DEFAULT_CELLS),
                    help="family:size:op[,op...] (default: the table)")
    ap.add_argument("--label", default="", help="commit or build label printed above the table")
    ap.add_argument("--timeout", type=float, default=None, help="per-cell limit in seconds (reported, not fatal)")
    ap.add_argument("--json", default=None, help="also write the raw cells to this file")
    ap.add_argument("--repeat", type=int, default=1,
                    help="fresh processes per cell (>= 1); the table gives their medians and the wall-time spread")
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error(f"--repeat must be at least 1, got {args.repeat}")
    rows = []
    for spec in args.cells:
        family, size, ops = spec.split(":")
        for op in ops.split(","):
            r = run_repeated(family, int(size), op, args.repeat, args.timeout)
            rows.append(dict(family=family, size=int(size), op=op, **r))
            print(f"# {family} {size} {op}: {r}", file=sys.stderr, flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"label": args.label, "cells": rows}, f, indent=1)
    if args.label:
        print(f"Measured on {args.label}.\n")
    if args.repeat > 1:
        print(f"Median of {args.repeat} fresh processes per cell; spread = min - max wall.\n")
    spread = args.repeat > 1
    print("| circuit | M | operation | wall | " + ("spread | " if spread else "") +
          "CPU | peak RSS | RSS before op | criterion | verdict |")
    print("|---|---|---|---|---|---|---|---|---|" + ("---|" if spread else ""))
    for r in rows:
        crit, ok = verdict(r["family"], r["size"], r["op"], r)
        if "error" in r:
            print(f"| {circuit_name(r['family'], r['size'])} | | {OP_NAMES[r['op']]} | {r['error']} | "
                  + ("| " if spread else "") + f"| | | {crit} | {ok} |")
        else:
            sp = f"{min(r['walls']):.2f} - {max(r['walls']):.2f} s | " if spread else ""
            print(f"| {circuit_name(r['family'], r['size'])} | {r['M']} | {OP_NAMES[r['op']]} | {r['wall']:.2f} s | "
                  f"{sp}{r['cpu']:.2f} s | {r['rss_mb']:.0f} MB | {r['base_mb']:.0f} MB | {crit} | {ok} |")


if __name__ == "__main__":
    main()
