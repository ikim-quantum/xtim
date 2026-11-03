"""python -m xtim — Stim-style CLI verbs, a thin façade over the xtim package.

    python -m xtim detect         --in C.stim --shots N [--seed S]
                                  [--out F] [--out_format 01|b8|dets|r8] [--obs_out F]
    python -m xtim analyze_errors --in C.stim [--out F]
                                  [--no-expectations] [--clifford-cutoff T|inf]

`detect` writes detection events to --out and, on request, observable flips (--obs_out),
both from ONE run of the twirl record engine (``Circuit.compile_detector_sampler``; one
stream per seed). Raw measurement records and PAULI_EXPECTATION values are not served
(score the end state with OUTPUT_QUBITS + sample_barrier + RecordFidelity).

Output formats (--out_format, default 01):
  01  Stim's text convention: one line per shot of '0'/'1' digits.
  b8  Stim's packed binary: bit k of a row at byte k>>3, bit k&7.
  dets  Stim's detector-event text: one `shot`-prefixed line per shot naming the set
      bits as `<L><index>` ascending, where the label L is the channel's Stim type
      hint — D detectors (--out), L observables (--obs_out). A no-fire shot is the
      bare `shot`. xtim keeps detectors/observables in SEPARATE channels, so each is
      homogeneous — byte-identical to `stim … --out_format dets` with observables split
      off via --obs_out (Stim's own per-channel convention).
  r8  Stim's run-length bytes: per shot, the gap (count of 0s) before each set bit as
      a byte, then a final byte for the trailing 0-run (0xFF continuation for runs
      >= 255). Byte-identical to `stim … --out_format r8`, per channel.

`analyze_errors` prints the Stim-format DEM (Circuit.detector_error_model_text);
--no-expectations is the documented opt-out (detector/observable-only columns,
for circuits whose PAULI_EXPECTATION columns are not DEM-expressible).
--clifford-cutoff T keeps each twirled Clifford fault's patterns flipping <= T detectors
(default 2: matching-friendly, but dropped patterns can lower the suppression order and the
dropped Clifford mass is reported; `inf` = exact, every pattern); see Circuit.detector_error_model_text.

Exit codes (the established CLI contract):
  0 success · 2 parse errors ("line N: message", 1-based) · 3 pipeline reject
  · 4 not DEM-expressible · 1 anything else (unreadable files, write failures,
  usage errors, a twirl refusal).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from .circuit import Circuit
from .errors import (
    XtimDemError,
    XtimError,
    XtimParseError,
    XtimRejectError,
)

_USAGE_EXIT = 1  # usage/IO problems exit 1


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


# ── verbs ──────────────────────────────────────────────────────────────────────────


def _cmd_detect(args) -> int:
    c = _read_circuit(args.in_file)
    s = c.compile_detector_sampler(seed=args.seed)
    dets, obs = s.sample(args.shots, separate_observables=True, bit_packed=(args.out_format == "b8"))
    _emit_bits(args.out, dets, args.out_format, dets_prefix="D")
    if args.obs_out is not None:
        _emit_bits(args.obs_out, obs, args.out_format, dets_prefix="L")
    return 0


def _cutoff_arg(s: str):
    """argparse type for --clifford-cutoff: 'inf' or an int 0 <= T < 2^31 (else a usage error)."""
    if s.strip().lower() == "inf":
        return math.inf
    try:
        t = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'inf' or an integer >= 0, got {s!r}") from None
    if not 0 <= t < 2 ** 31:
        raise argparse.ArgumentTypeError(f"expected 'inf' or an integer 0 <= T < 2^31, got {s!r}")
    return t


def _cmd_analyze_errors(args) -> int:
    c = _read_circuit(args.in_file)
    text = c.detector_error_model_text(
        include_expectations=not args.no_expectations, clifford_cutoff=args.clifford_cutoff)
    _write_bytes(args.out, text.encode())
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

    dp = sub.add_parser("detect", help="sample detection events (+ observable flips)")
    _add_run_flags(dp, shots_required=True)
    dp.add_argument("--out", default=None, metavar="FILE",
                    help="detection events (default: stdout)")
    dp.add_argument("--out_format", choices=("01", "b8", "dets", "r8"),
                    default="01",
                    help="01|b8|dets|r8 (Stim byte conventions; dets/r8 bit-record "
                         "formats label channels D#/L#)")
    dp.add_argument("--obs_out", default=None, metavar="FILE",
                    help="observable flips")
    dp.set_defaults(fn=_cmd_detect)

    ep = sub.add_parser("analyze_errors", help="export the detector error model")
    _add_run_flags(ep, shots_required=False)
    ep.add_argument("--out", default=None, metavar="FILE", help="default: stdout")
    ep.add_argument("--no-expectations", action="store_true",
                    help="omit the PAULI_EXPECTATION L-columns "
                         "(detector/observable-only DEM)")
    ep.add_argument("--clifford-cutoff", default=2, metavar="T", type=_cutoff_arg,
                    help="keep only each Clifford fault's patterns flipping <= T detectors "
                         "(default 2 = graphlike/matching-friendly, but dropped patterns can lower the "
                         "suppression order and faults whose fixed part flips > 2 detectors vanish; the "
                         "dropped Clifford mass is reported; 'inf' = exact, every pattern)")
    ep.set_defaults(fn=_cmd_analyze_errors)

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
