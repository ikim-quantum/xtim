"""D14 cost component "t_nullspace": the restricted-nullspace sub-stage of a kappa>0 plan build
(S2-pre, twirl_law.cpp::active_nullspace), measured by plan-build replay.

Workload (deterministic): the first NPLANS distinct kappa>0 plans of circuit X at seed s — the
plan identity is the prefix-stripped plan key (the (a, cz) residual the sampler builds with an
identity prefix), in first-appearance order over one compact sample_barrier run (shot count
doubled from 20k until NPLANS are found). Each replay process rebuilds them through
plan_structure(key) (= parse_plan_key + build_shot_law, uncached), single-threaded and pinned,
with QEC_TW_PROF=1; the engine's per-build nullspace timer (LawProf "nullspace=... us/build") is
the reading. One warm-up pass precedes REPS timed passes inside the process (the PROF mean covers
all passes; the keys are identical across passes).

Given two roots it interleaves A/B processes (ABBA by round) and reports the B/A ratio, its 95%
interval and the minimum detectable effect (80% power, alpha 0.05) from the per-round log ratios.
Only engines that carry the PROF nullspace timer (S2-pre and later) can be measured.

Baseline (S2-pre rebased on dev/next 266503c; seed 1, 2000 kappa>0 plans, 10 rounds x (1+5)
passes, core 45, host load 1.2-1.8; 2026-10-14):
  tests/private/ccz_factory_a.stim : 4.877 us/build (MDE 0.75%; build 424 us)
  ccz_factory_b.stim               : 5.641 us/build (MDE 0.66%; build 436 us)

usage:
  python benchmarks/replay_nullspace.py --root A [--root B] --circuit X.stim [--seed 1]
         [--plans 2000] [--rounds 10] [--reps 5] [--core 45] [--keys cache.npz]
"""
import argparse
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import tempfile


def collect(root, circuit, seed, nplans, out):
    sys.path.insert(0, root)
    import numpy as np
    import xtim._xtim as _x
    text = open(circuit).read()
    N = 20000
    while True:
        s = _x.TwirlSampler(text, 1.0, "", 0, True)
        b = s.sample_barrier(N, seed=seed, compact=True)
        keys, seen, NW = [], set(), None
        for i in range(N):
            kb = bytes(bytearray(b.plan_key(i)))
            if not kb:
                continue
            if NW is None:
                NW = (int(s.plan_structure(kb)["n"]) + 63) // 64
            kz = bytes(16 * NW) + kb[16 * NW:]
            if kz in seen:
                continue
            seen.add(kz)
            if int(s.plan_structure(kz)["kappa"]) > 0:
                keys.append(kz)
                if len(keys) >= nplans:
                    break
        if len(keys) >= nplans or N >= 1280000:
            break
        N *= 2
        del b, s
    np.savez(out, keys=np.array(keys, dtype=object), shots=N)
    return len(keys), N


def worker(root, circuit, keys, reps):
    sys.path.insert(0, root)
    import numpy as np
    import xtim._xtim as _x
    assert _x.__file__.startswith(root), _x.__file__
    ks = list(np.load(keys, allow_pickle=True)["keys"])
    s = _x.TwirlSampler(open(circuit).read(), 1.0, "", 0, True)
    for _ in range(1 + reps):
        for k in ks:
            s.plan_structure(k)


def run_one(root, a, keys):
    env = {**os.environ, "QEC_TW_PROF": "1", "OMP_NUM_THREADS": "1"}
    cmd = ["taskset", "-c", a.core, sys.executable, "-I", os.path.abspath(__file__), "--worker",
           "--root", root, "--circuit", a.circuit, "--keys", keys, "--reps", str(a.reps)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(r.stderr)
    m = re.search(r"kappa>0 builds=(\d+) total=([\d.]+) us/build nullspace=([\d.]+) us/build", r.stderr)
    if not m:
        raise SystemExit("engine has no PROF nullspace timer:\n" + r.stderr)
    return {"builds": int(m.group(1)), "build_us": float(m.group(2)), "nullspace_us": float(m.group(3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plans", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--core", default="45")
    ap.add_argument("--keys")
    ap.add_argument("--worker", action="store_true")
    a = ap.parse_args()
    if a.worker:
        worker(a.root[0], a.circuit, a.keys, a.reps)
        return
    keys = a.keys or os.path.join(tempfile.mkdtemp(), "keys.npz")
    if not os.path.exists(keys):
        n, N = collect(a.root[0], a.circuit, a.seed, a.plans, keys)
        print(f"collected {n} kappa>0 plans from {N} shots (seed {a.seed})", flush=True)
    res = {r: [] for r in a.root}
    for rd in range(a.rounds):
        order = a.root if rd % 2 == 0 else a.root[::-1]
        for r in order:
            res[r].append(run_one(r, a, keys))
    out = {"circuit": os.path.basename(a.circuit), "seed": a.seed, "plans": a.plans, "roots": {}}
    for r, v in res.items():
        ns = [x["nullspace_us"] for x in v]
        out["roots"][r] = {"nullspace_us_median": st.median(ns), "nullspace_us": ns,
                           "build_us_median": st.median([x["build_us"] for x in v]),
                           "builds_per_process": v[0]["builds"]}
    if len(a.root) == 2:
        A, B = a.root
        lr = [math.log(y["nullspace_us"] / x["nullspace_us"]) for x, y in zip(res[A], res[B])]
        m = st.mean(lr)
        se = st.stdev(lr) / math.sqrt(len(lr))
        out["B_over_A"] = math.exp(m)
        out["ci95"] = [math.exp(m - 1.96 * se), math.exp(m + 1.96 * se)]
        out["mde_pct"] = 100 * (math.exp(2.8 * se) - 1)
    else:
        ns = [x["nullspace_us"] for x in res[a.root[0]]]
        lr = [math.log(x) for x in ns]
        out["mde_pct"] = 100 * (math.exp(2.8 * st.stdev(lr) / math.sqrt(len(lr))) - 1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
