"""python -m xtim — Stim-style CLI verbs, a thin façade over the xtim package.

    python -m xtim sample         --in C.stim --shots N [--seed S]
                                  [--out F] [--out_format 01|b8|dets|r8]
    python -m xtim detect         --in C.stim --shots N [--seed S]
                                  [--out F] [--out_format 01|b8|dets|r8]
                                  [--obs_out F] [--exp_out F] [--meas_out F]
    python -m xtim analyze_errors --in C.stim [--out F]
                                  [--no-expectations]
    python -m xtim diagnose       <circuit.stim> [--shots N] [--seed S]
    python -m xtim state compile  <circuit.stim> <out.ref>
    python -m xtim state verify   <circuit.stim> <file.ref>

`sample` writes the raw measurement records; `detect` writes detection events to
--out and, on request, observable flips (--obs_out), raw PAULI_EXPECTATION values
(--exp_out) and the measurement records (--meas_out) — all four channels from ONE
engine run (one RNG stream, exactly `run_stim_main --b8`'s records).

Output formats (--out_format, default 01):
  01  Stim's text convention: one line per shot of '0'/'1' digits (bit channels).
      The expectation channel has no Stim analog; in 01 mode it is one line per
      shot of space-separated %.17g decimal floats (round-trips float64).
  b8  Stim's packed binary: bit k of a row at byte k>>3, bit k&7 — byte-identical
      to `run_stim_main --b8`'s per-channel files; the expectation channel is raw
      little-endian float64 (the .exp.f64 layout).
  dets  Stim's detector-event text: one `shot`-prefixed line per shot naming the set
      bits as `<L><index>` ascending, where the label L is the channel's Stim type
      hint — D detectors (--out of `detect`), L observables (--obs_out), M
      measurements (`sample`'s --out and `detect`'s --meas_out). A no-fire shot is the
      bare `shot`. xtim keeps detectors/observables/measurements in SEPARATE channels,
      so each is homogeneous — byte-identical to `stim … --out_format dets` with
      observables split off via --obs_out (Stim's own per-channel convention).
  r8  Stim's run-length bytes: per shot, the gap (count of 0s) before each set bit as
      a byte, then a final byte for the trailing 0-run (0xFF continuation for runs
      >= 255). Byte-identical to `stim … --out_format r8`, per channel.
  (dets/r8 are bit-record formats; the --exp_out float64 channel has no bit meaning
   and falls back to the 01 %.17g text — see --exp_out below.)

`analyze_errors` prints the Stim-format DEM (run_stim_main --dem semantics);
--no-expectations is the documented opt-out (detector/observable-only columns,
for circuits whose PAULI_EXPECTATION columns are not DEM-expressible).

`diagnose` does ONE noiseless run and prints the onboarding report (reference chi
+ cache status, deterministic-vs-gauge detectors, per-column |beta| / byproduct-
frame hint / DEM-expressibility, an advisory post-selection recipe) — the by-hand
facts `Circuit.diagnose()` packages. It reports; it applies nothing.

`state compile` runs the full ref_compile gate battery and writes the .ref;
`state verify` runs the cheap verify (load + n-match + noiseless invariants)
of a .ref against a circuit, printing PASS/FAIL.

Exit codes mirror the established CLI contract (cpp/apps/run_stim_main.cpp):
  0 success · 2 parse errors ("line N: message", 1-based) · 3 pipeline reject
  · 4 not DEM-expressible · 1 anything else (unreadable files, write failures,
  usage errors, state compile/verify failure — ref_compile's convention).
`run_stim_main` itself is untouched; these verbs wrap the package only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .circuit import Circuit
from .errors import (
    XtimDemError,
    XtimError,
    XtimParseError,
    XtimRejectError,
    XtimReferenceError,
)
from .reference import Reference, compile_reference

_USAGE_EXIT = 1  # run_stim_main: usage/IO problems exit 1


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse default exits 2; 2 is the parse-error code
        self.print_usage(sys.stderr)
        self.exit(_USAGE_EXIT, f"{self.prog}: error: {message}\n")


def _read_circuit(path: str) -> Circuit:
    p = Path(path)
    try:
        text = p.read_text()
    except OSError:
        print(f"cannot open {path}", file=sys.stderr)
        raise SystemExit(1)
    except UnicodeDecodeError:
        print(f"{path} is not a UTF-8 text circuit (binary/non-UTF-8 bytes)",
              file=sys.stderr)
        raise SystemExit(1)
    return Circuit(text, _source=p.name)


def _write_bytes(path: str | None, data: bytes) -> None:
    if path is None or path == "-":
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
    else:
        Path(path).write_bytes(data)


def _bits_01(rows: np.ndarray) -> bytes:
    out = []
    for row in rows.astype(np.uint8):
        out.append("".join("1" if b else "0" for b in row))
    return ("\n".join(out) + "\n" if out else "").encode()


def _exps_01(rows: np.ndarray) -> bytes:
    out = [" ".join(format(v, ".17g") for v in row) for row in rows]
    return ("\n".join(out) + "\n" if out else "").encode()


def _bits_dets(rows: np.ndarray, prefix: str) -> bytes:
    """Stim 'dets' text: one `shot`-prefixed line per shot listing the indices of
    the set bits as `<prefix><index>` (ascending). The label letter is the channel's
    Stim type hint: 'D' detectors, 'L' observables, 'M' measurements. A row with no
    set bits is the bare `shot`. Byte-exact to `stim ... --out_format dets` on a
    single-type channel (xtim keeps detectors/observables/measurements in separate
    channels, so each channel is homogeneous like Stim's --obs_out/--out split)."""
    out = []
    for row in rows.astype(bool):
        idx = np.flatnonzero(row)
        if idx.size:
            out.append("shot " + " ".join(f"{prefix}{i}" for i in idx))
        else:
            out.append("shot")
    return ("\n".join(out) + "\n" if out else "").encode()


def _bits_r8(rows: np.ndarray) -> bytes:
    """Stim 'r8' run-length bytes: per shot, for each set bit emit the gap (count of
    0s since the previous set bit) as a byte, then a FINAL byte for the trailing run
    of 0s after the last set bit (the whole length when no bit is set). A gap >= 255
    emits 0xFF for each full 255 then the remainder byte (always emitted, even 0x00,
    to terminate the run). Byte-exact to `stim ... --out_format r8`."""
    out = bytearray()
    for row in rows.astype(bool):
        gap = 0
        for bit in row:
            if bit:
                while gap >= 255:
                    out.append(0xFF)
                    gap -= 255
                out.append(gap)
                gap = 0
            else:
                gap += 1
        while gap >= 255:
            out.append(0xFF)
            gap -= 255
        out.append(gap)
    return bytes(out)


def _emit_bits(path: str | None, arr: np.ndarray, fmt: str,
               dets_prefix: str = "D") -> None:
    if fmt == "b8":
        data = arr.tobytes()
    elif fmt == "dets":
        data = _bits_dets(arr, dets_prefix)
    elif fmt == "r8":
        data = _bits_r8(arr)
    else:  # "01"
        data = _bits_01(arr)
    _write_bytes(path, data)


def _emit_exps(path: str | None, arr: np.ndarray, fmt: str) -> None:
    # The expectation channel is float64, with no Stim analog and no bit-record
    # meaning: 'b8' is its raw little-endian float64 layout; every text format
    # ('01', and the new bit-only 'dets'/'r8') falls back to the %.17g decimal line.
    _write_bytes(path, arr.tobytes() if fmt == "b8" else _exps_01(arr))


# ── verbs ──────────────────────────────────────────────────────────────────────────


def _cmd_sample(args) -> int:
    c = _read_circuit(args.in_file)
    meas = c.compile_sampler(seed=args.seed).sample(
        args.shots, bit_packed=(args.out_format == "b8"))
    _emit_bits(args.out, meas, args.out_format, dets_prefix="M")
    return 0


def _cmd_detect(args) -> int:
    c = _read_circuit(args.in_file)
    s = c.compile_detector_sampler(seed=args.seed)
    dets, obs, exps, meas = s.sample(
        args.shots, separate_observables=True, return_expectations=True,
        return_measurements=True, bit_packed=(args.out_format == "b8"))
    _emit_bits(args.out, dets, args.out_format, dets_prefix="D")
    if args.obs_out is not None:
        _emit_bits(args.obs_out, obs, args.out_format, dets_prefix="L")
    if args.exp_out is not None:
        _emit_exps(args.exp_out, exps, args.out_format)
    if args.meas_out is not None:
        _emit_bits(args.meas_out, meas, args.out_format, dets_prefix="M")
    return 0


def _cmd_analyze_errors(args) -> int:
    c = _read_circuit(args.in_file)
    text = c.detector_error_model_text(
        include_expectations=not args.no_expectations)
    _write_bytes(args.out, text.encode())
    return 0


def _cmd_diagnose(args) -> int:
    c = _read_circuit(args.circuit)
    print(c.diagnose(shots=args.shots, seed=args.seed))
    return 0


def _cmd_state_compile(args) -> int:
    c = _read_circuit(args.circuit)
    try:
        ref = compile_reference(c.text, source=Path(args.circuit).name)
    except XtimReferenceError as e:
        print(f"state compile FAIL: {e}", file=sys.stderr)
        return 1
    ref.save(args.out_ref)
    # internal_n is the DESUGARED qubit count (MPP/SPP cat-check ancillas etc.),
    # not circuit.num_qubits — label it so it doesn't look like a mismatch.
    print(f"compiled reference (chi={ref.chi}, internal_n={ref.n}) -> {args.out_ref}",
          file=sys.stderr)
    return 0


def _cmd_state_verify(args) -> int:
    c = _read_circuit(args.circuit)
    try:
        ref_text = Path(args.ref).read_text()
    except OSError:
        print(f"cannot open {args.ref}", file=sys.stderr)
        return 1
    except UnicodeDecodeError:
        print(f"{args.ref} is not a UTF-8 text .ref (binary/corrupt)", file=sys.stderr)
        return 1
    from . import _xtim
    v = _xtim.ref_verify_cheap(c.text, ref_text)
    if not v["ok"]:
        print(f"VERIFY FAIL: {v['error']}", file=sys.stderr)
        return 1
    ref = Reference(ref_text)
    print(f"VERIFY PASS (chi={ref.chi}, internal_n={ref.n}) -> ok", file=sys.stderr)
    return 0


# ── argument surface ───────────────────────────────────────────────────────────────


def _add_run_flags(p: _Parser, *, shots_required: bool) -> None:
    p.add_argument("--in", dest="in_file", required=True, metavar="FILE",
                   help="circuit file (extended Stim dialect)")
    if shots_required:
        p.add_argument("--shots", type=int, required=True,
                       help="number of shots to sample")
        p.add_argument("--seed", type=int, default=0,
                       help="engine RNG seed (default 0)")


def build_parser() -> _Parser:
    ap = _Parser(prog="python -m xtim", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="verb", required=True)

    sp = sub.add_parser("sample", help="sample raw measurement records")
    _add_run_flags(sp, shots_required=True)
    sp.add_argument("--out", default=None, metavar="FILE", help="default: stdout")
    sp.add_argument("--out_format", choices=("01", "b8", "dets", "r8"),
                    default="01",
                    help="01|b8|dets|r8 (Stim byte conventions; dets/r8 are bit-record "
                         "formats; measurement channel labelled M#)")
    sp.set_defaults(fn=_cmd_sample)

    dp = sub.add_parser("detect", help="sample detection events (+obs/exp/meas)")
    _add_run_flags(dp, shots_required=True)
    dp.add_argument("--out", default=None, metavar="FILE",
                    help="detection events (default: stdout)")
    dp.add_argument("--out_format", choices=("01", "b8", "dets", "r8"),
                    default="01",
                    help="01|b8|dets|r8 (Stim byte conventions; dets/r8 bit-record "
                         "formats label channels D#/L#/M#; exp_out stays %%.17g/f64)")
    dp.add_argument("--obs_out", default=None, metavar="FILE",
                    help="observable flips")
    dp.add_argument("--exp_out", default=None, metavar="FILE",
                    help="raw PAULI_EXPECTATION values")
    dp.add_argument("--meas_out", default=None, metavar="FILE",
                    help="measurement records (same single run)")
    dp.set_defaults(fn=_cmd_detect)

    ep = sub.add_parser("analyze_errors", help="export the detector error model")
    _add_run_flags(ep, shots_required=False)
    ep.add_argument("--out", default=None, metavar="FILE", help="default: stdout")
    ep.add_argument("--no-expectations", action="store_true",
                    help="omit the PAULI_EXPECTATION L-columns "
                         "(detector/observable-only DEM)")
    ep.set_defaults(fn=_cmd_analyze_errors)

    dg = sub.add_parser("diagnose", help="onboarding report (one noiseless run)")
    dg.add_argument("circuit", metavar="FILE",
                    help="circuit file (extended Stim dialect)")
    dg.add_argument("--shots", type=int, default=1024,
                    help="noiseless-run shots for the report (default 1024)")
    dg.add_argument("--seed", type=int, default=0,
                    help="engine RNG seed (default 0)")
    dg.set_defaults(fn=_cmd_diagnose)

    st = sub.add_parser("state", help="reference-state surface (.ref artifacts)")
    ssub = st.add_subparsers(dest="state_verb", required=True)
    sc = ssub.add_parser("compile", help="full ref_compile gate battery -> .ref")
    sc.add_argument("circuit", metavar="FILE",
                    help="circuit file (extended Stim dialect)")
    sc.add_argument("out_ref", metavar="OUT.ref",
                    help="destination .ref path to write")
    sc.set_defaults(fn=_cmd_state_compile)
    sv = ssub.add_parser("verify", help="cheap verify of a .ref against a circuit")
    sv.add_argument("circuit", metavar="FILE",
                    help="circuit file (extended Stim dialect)")
    sv.add_argument("ref", metavar="FILE.ref",
                    help=".ref file to verify against the circuit")
    sv.set_defaults(fn=_cmd_state_verify)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except XtimParseError as e:
        print(str(e), file=sys.stderr)          # "line N: message" per line
        return 2
    except XtimRejectError as e:
        print(str(e), file=sys.stderr)          # "rejected (gate index N): ..."
        return 3
    except XtimDemError as e:
        print(str(e), file=sys.stderr)          # "not DEM-expressible: ..."
        return 4
    except XtimError as e:
        print(str(e), file=sys.stderr)
        return 1
    except (ValueError, TypeError) as e:
        # Validation errors (e.g. --shots -5) — a clean one-liner
        # instead of a raw traceback, matching the rest of the CLI's error surface.
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
