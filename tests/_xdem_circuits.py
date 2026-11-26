"""The circuit registry of the xdem tests (SPEC §6 T1 list). In-repo circuits are always present; external ones
(not in this repository) are looked up by file name in the directory $XTIM_EXTERNAL_CIRCUITS (no default; see
external_path) and skipped when absent or when the variable is unset. Never add a private factory circuit."""
import os
import re
from pathlib import Path

import xtim

TESTS = Path(__file__).parent


def external_path(name: str, var: str = "XTIM_EXTERNAL_CIRCUITS"):
    """The external circuit file `$var/name` (`name` a file name, no directories; `$var` a flat directory of the
    files or of symlinks to them). None when the variable is unset or the file is absent (the caller skips)."""
    root = os.environ.get(var)
    if not root:
        return None
    path = Path(root) / name
    return path if path.exists() else None


TOYS = {n: TESTS / "data/xdem/circuits" / f"{n}.stim"
        for n in ("ex422_obs", "toy2obs", "rm15rep", "ccz_toy", "ccz_toy_obs", "ccz_toy_obs_p5")}
EXTERNAL_FILES = {
    "msc_d3_canon": "msc_d3_inject_cultivate_p1e-3.canon.stim",
    "msc_d5_canon": "msc_d5_inject_cultivate_p1e-3.canon.stim",
    "bt27_p1e-4": "bt27_scored_p1e-4.stim",
    "bt27_p1e-3": "bt27_scored_p1e-3.stim",
    "rm15_x": "rm15_xpre.stim",
}
EXTERNAL = {n: external_path(f) for n, f in EXTERNAL_FILES.items()}
EXAMPLES = ("cultivation_d3_faithful", "cultivation_d5", "distillation_15_1_3", "color5_cultivation",
            "miniature_oracle", "code_switching_faithful", "code_switching_rate", "cube_ccz")
VARIANTS = ("distillation_15_1_3:gatenoise", "code_switching_faithful:noobs", "code_switching_rate:noobs")
NAMES_INREPO = list(EXAMPLES) + list(VARIANTS) + ["rep3T_pre", "rep3T_pre_p1e-2"] + list(TOYS)
NAMES_EXTERNAL = list(EXTERNAL)


def rep3t_pre(p: float) -> str:
    """tests/test_dc_lines.py::rep3t_pre at 19f2f78, unchanged from the earlier prototype test: rep-3 code, a noisy
    Z-check round before a transversal T, a noisy round after it, a noiseless stabilizer round, then the readout
    T_DAG + MPP X0X1X2."""
    return "\n".join([
        "RX 0", "CX 0 1 0 2", f"X_ERROR({p}) 0 1 2", "R 5 6", "CX 0 5 1 5 1 6 2 6", f"X_ERROR({p}) 5 6", "M 5 6",
        "DETECTOR rec[-2]", "DETECTOR rec[-1]", "T 0 1 2", f"X_ERROR({p}) 0 1 2", "R 3 4", "CX 0 3 1 3 1 4 2 4",
        f"X_ERROR({p}) 3 4", "M 3 4", "DETECTOR rec[-2] rec[-4]", "DETECTOR rec[-1] rec[-3]", "MPP Z0*Z1 Z1*Z2",
        "DETECTOR rec[-2] rec[-4]", "DETECTOR rec[-1] rec[-3]", "T_DAG 0 1 2", "MPP X0*X1*X2",
        "OBSERVABLE_INCLUDE(0) rec[-1]"]) + "\n"


_MEAS = {"M", "MX", "MY", "MR", "MRX", "MRY", "MPP", "MXX", "MYY", "MZZ"}
_ONE = {"H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "H_YZ", "H_XY"}
_TWO = {"CX", "CZ", "CY", "SWAP", "CNOT"}


def gatenoise(text, p=1e-3):
    """tests/test_dc_export.py::gatenoise at 19f2f78, unchanged (the prototype registry's `:gatenoise` variant)."""
    out = []
    for l in text.splitlines():
        s = l.split("#")[0].strip()
        toks = s.split()
        g = toks[0] if toks else ""
        if g in _MEAS or g.split("(")[0] in _MEAS:
            qs = sorted({int(re.sub(r"[^0-9]", "", u)) for t in toks[1:] for u in t.split("*")
                         if re.sub(r"[^0-9]", "", u) and not u.startswith("rec")})
            out.append(f"DEPOLARIZE1({p}) " + " ".join(map(str, qs)))
            out.append(l)
        elif g in _ONE:
            out += [l, f"DEPOLARIZE1({p}) " + " ".join(toks[1:])]
        elif g in _TWO and "rec[" not in s:
            out += [l, f"DEPOLARIZE2({p}) " + " ".join(toks[1:])]
        else:
            out.append(l)
    return "\n".join(out)


def variant(name: str) -> str:
    """tests/test_dc_export.py::load at 19f2f78, unchanged: name[:gatenoise][:noobs] from xtim/_examples."""
    base, *var = name.split(":")
    text = xtim.load_example(base).text
    for v in var:
        if v == "gatenoise":
            text = gatenoise(text)
        elif v == "noobs":
            text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("OBSERVABLE_INCLUDE"))
        else:
            raise ValueError(v)
    return text


def circuit(name: str) -> str:
    if name in VARIANTS:
        return variant(name)
    if name in EXAMPLES:
        return xtim.load_example(name).text
    if name == "rep3T_pre":
        return rep3t_pre(1e-3)
    if name == "rep3T_pre_p1e-2":
        return rep3t_pre(1e-2)
    if name in TOYS:
        return TOYS[name].read_text()
    return EXTERNAL[name].read_text()


def present(name: str) -> bool:
    return name not in EXTERNAL or (EXTERNAL[name] is not None and EXTERNAL[name].exists())


def xdem_export(text: str, **kw) -> str:
    return xtim.Circuit(text).detector_error_model_text(format="xdem", **kw)
