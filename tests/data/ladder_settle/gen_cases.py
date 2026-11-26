"""Generator note for tests/data/ladder_settle/cases.json (fix/ladder-settle, 2026-11-18). Run once, against the
PRE-FIX build (dev/next b0a06fc), where the plain bare build (`_xtim.bare_state_of`) refused every selected text with
"rotation-form terms anticommute (even-Clifford ladder gap)":

  usage: PYTHONPATH=<wt>:<wt>/tests python gen_cases.py PROBE_DIR SEEDS OUT.json

* "undo" cases: the 19 ladder-gap refusals of the undo-localization probe (PROBE_DIR/g1.jsonl + g2.jsonl, records with
  verdict "ladder-gap"; an internal probe): text = base_text(circuit) + rot_text(rotations, undo)
  and the clifford_paulis that circuit must read deterministically (the probe's family-build check: 19 of 19 signs).
* "base" cases: the probe's g1 generator (`gen` below, a verbatim copy of the probe's fuzz.gen) over seeds 0..SEEDS-1;
  a seed is kept when xtim.output_state accepts a random candidate subset WITH rotations while bare_state_of refuses
  the noiseless circuit itself by the ladder gap (the probe's "base-refused:ladder-gap", 405 of 60000 seeds); the 24
  shortest kept texts are written, with the qubit subset (the output_state result is recomputed by the tests).
Files are .json (not .stim) so the xdem in-repo hash sweep's input set is unchanged."""
import json
import os
import sys

import numpy as np

import xtim
from xtim import _xtim
import _output_state_oracle as O

LADDER = "rotation-form terms anticommute (even-Clifford ladder gap)"
ONE = ["H", "S", "S_DAG", "H_YZ", "SQRT_X"]
MAG1 = ["T", "T_DAG"]
MAG2 = ["CS", "CS_DAG"]


def gen(seed):
    rng = np.random.default_rng(seed)
    kind = seed % 3
    n = int(rng.integers(3, 7))
    L = []
    pm = [0.25, 0.35, 0.45][kind]
    for _ in range(int(rng.integers(8, 26))):
        r = rng.random()
        if r < pm:
            if rng.random() < 0.6:
                L.append(f"{rng.choice(MAG1)} {int(rng.integers(n))}")
            elif rng.random() < 0.8 or n < 3:
                c, t = (int(x) for x in rng.choice(n, 2, replace=False)); L.append(f"{rng.choice(MAG2)} {c} {t}")
            else:
                L.append("CCZ " + " ".join(str(int(x)) for x in rng.choice(n, 3, replace=False)))
        elif r < pm + (1 - pm) / 2:
            L.append(f"{rng.choice(ONE)} {int(rng.integers(n))}")
        else:
            c, t = (int(x) for x in rng.choice(n, 2, replace=False)); L.append(f"{rng.choice(['CX', 'CZ'])} {c} {t}")
        if kind == 2 and rng.random() < 0.06:
            a = n
            q, q2 = (int(x) for x in rng.choice(n, 2, replace=False))
            L.append(f"R {a}\nH {a}\nCZ {a} {q}\nH {a}\nM {a}\n{rng.choice(['CX', 'CZ'])} rec[-1] {q2}")
    return "\n".join(L) + "\n", n, rng


def ladder_refused(text):
    try:
        _xtim.bare_state_of(text)
        return False
    except ValueError as e:
        return LADDER in str(e)


def main():
    probe, seeds, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cases = []
    for fn in ("g1.jsonl", "g2.jsonl"):
        for line in open(os.path.join(probe, fn)):
            r = json.loads(line)
            if r["verdict"] != "ladder-gap":
                continue
            und = O.base_text(r["text"]) + O.rot_text([tuple(x) for x in r["rotations"]], undo=True)
            assert ladder_refused(und), r["seed"]
            cases.append(dict(name=f"undo_{fn[:2]}_{r['seed']}", kind="undo", text=und,
                              clifford_paulis=r["cp"]))
    base = []
    for seed in range(seeds):
        text, n, rng = gen(seed)
        try:
            cands = xtim.output_candidates(text)
            if not cands:
                continue
            k = int(rng.integers(1, len(cands) + 1))
            qs = sorted(int(x) for x in rng.choice(cands, k, replace=False))
            st = xtim.output_state(text, qs)
        except Exception:
            continue
        if st.rotations and ladder_refused(O.base_text(text)):
            base.append(dict(name=f"base_g1_{seed}", kind="base", text=text, qubits=qs))
    base.sort(key=lambda c: (len(c["text"]), c["name"]))
    cases += base[:24]
    with open(out, "w") as f:
        json.dump(cases, f, indent=1)
    print(f"{sum(c['kind'] == 'undo' for c in cases)} undo + {len(base[:24])} base (of {len(base)} found)")


if __name__ == "__main__":
    main()
