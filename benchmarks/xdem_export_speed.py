"""LEAN-EXPORT-PLAN speed harness: warm medians (ms) of the C++ xdem export per route, through the test API
xdem_export_route (the profile's "C++ export_dem total" plus < 2 ms of dict marshalling). Run under the heavy lock
with XTIM_BARE_MEMO=0. Writes one JSON file.

A circuit name is looked up as a file in the --circuit-dir directories (FILES maps the named benchmarks to their file
names; any other name is tried as <name>.stim), and otherwise as an in-repo example (xtim.load_example)."""
import argparse
import json
import os
import statistics
import time

import xtim
from xtim import _xtim_testapi as ta

FILES = {"bt27_p1e-3": "bt27_scored_p1e-3.stim", "msc_d5": "msc_d5_inject_cultivate_p1e-3.canon.stim"}
PROFILE_MS = {"bt27_p1e-3": 1042.0, "msc_d5": 162.0, "cultivation_d5": 151.0}   # export_profile.md, warm


def text_of(name, dirs):
    for d in dirs:
        p = os.path.join(d, FILES.get(name, name + ".stim"))
        if os.path.isfile(p):
            return open(p).read()
    try:
        return xtim.load_example(name).text
    except Exception as e:
        raise SystemExit(f"{name}: not found in --circuit-dir {dirs} and not an in-repo example ({e})")


def loadavg():
    """The 1/5/15-minute load average, or None where /proc/loadavg does not exist (non-Linux)."""
    try:
        with open("/proc/loadavg") as f:
            return f.read().split()[:3]
    except OSError:
        return None


def once(text, route):
    t0 = time.perf_counter()
    r = ta.xdem_export_route(text, route)
    ms = (time.perf_counter() - t0) * 1e3
    if not r["ok"]:
        raise SystemExit(f"{route}: export failed: {r['error'] or r['exception_message']}")
    return ms, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--routes", default="oracle,lean")
    ap.add_argument("--circuits", default="cultivation_d5",
                    help="comma-separated names, e.g. bt27_p1e-3,msc_d5,cultivation_d5 (the first two need --circuit-dir)")
    ap.add_argument("--circuit-dir", action="append", default=[],
                    help="directory holding circuit files (repeatable); in-repo examples need none")
    a = ap.parse_args()
    routes = a.routes.split(",")
    res = {"load_before": loadavg(), "reps": a.reps, "circuits": {}}
    for name in a.circuits.split(","):
        text = text_of(name, a.circuit_dir)
        outs = {r: once(text, r)[1] for r in routes}            # cold calls, discarded
        if len(routes) == 2:
            assert outs[routes[0]]["dem"] == outs[routes[1]]["dem"], f"{name}: routes differ"
        t = {r: [] for r in routes}
        for rep in range(a.reps):
            for r in (routes if rep % 2 == 0 else routes[::-1]):
                t[r].append(once(text, r)[0])
        py = []
        for _ in range(3):
            t0 = time.perf_counter()
            xtim.Circuit(text).detector_error_model_text(format="xdem")
            py.append((time.perf_counter() - t0) * 1e3)
        res["circuits"][name] = {
            **{r: {"median_ms": statistics.median(t[r]), "all_ms": t[r]} for r in routes},
            "lean_used": outs.get("lean", {}).get("lean_used"),
            "python_call_median_ms": statistics.median(py), "profile_ms": PROFILE_MS.get(name)}
    res["load_after"] = loadavg()
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: {r: v[r]["median_ms"] for r in routes} for k, v in res["circuits"].items()}))


if __name__ == "__main__":
    main()
