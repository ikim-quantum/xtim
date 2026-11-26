"""S7 grep gates (cleanup stage): removed levers stay removed.

* XTIM_QUIET (F5) is read nowhere: no C++ source, binding, app, build file, Python module, tool or benchmark mentions
  it outside comments (tests may set it only to prove it has no effect);
* the debug levers (F6) are gone: FSYND_XCHECK, OVERLAP_FORCE_DENSE (env), CB_BATCH_XCHECK, IP_W1_XCHECK (build flags);
* the fold-in levers are gone: XTIM_FRAME_COSETS_BACKEND (env; the backend is a test-only argument) and the
  bare-state memo's trace mode; the internal-invariant abort()s of normal_form / carried_compose /
  residue_projection are exceptions;
* the lever notices (F1) are gone from the exact engine, and the twirl ctor's circuit-channels line is an event, not a
  stderr write.
Code lines only: C/C++ comments (// and /* */) and Python / CMake / shell comments are dropped; every preprocessor
line (#if, #define, #undef, ...) is code. setup.py and the CMake files are scanned too.
Mutation: re-adding any of these tokens on a code line (a `#define XTIM_QUIET`, a setup.py macro) turns the matching
test red.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("cpp/src", "cpp/include", "cpp/bindings", "cpp/apps", "cpp/testapi", "xtim", "benchmarks", "tools")
CODE_FILES = ("setup.py", "cpp/CMakeLists.txt")
C_SUFFIXES = (".cpp", ".hpp", ".h", ".inc")
HASH_SUFFIXES = (".py", ".sh", ".txt", ".cmake")


def _files():
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and p.suffix in C_SUFFIXES + HASH_SUFFIXES and "build" not in p.parts:
                yield p
    for f in CODE_FILES:
        yield ROOT / f


def _code_lines(path):
    text = path.read_text(errors="replace")
    if path.suffix in C_SUFFIXES:
        text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    for i, line in enumerate(text.splitlines(), 1):
        code = line.split("//")[0] if path.suffix in C_SUFFIXES else line.split("#")[0]
        if code.strip():
            yield i, code


def _hits(tokens):
    out = []
    for p in _files():
        for i, code in _code_lines(p):
            for t in tokens:
                if t in code:
                    out.append(f"{p.relative_to(ROOT)}:{i}: {t}")
    return out


def test_scanner_sees_preprocessor_and_build_files(tmp_path):
    h = tmp_path / "x.hpp"
    h.write_text("// XTIM_QUIET in a comment\n/* XTIM_QUIET\n   in a block */\n#define XTIM_QUIET 1\n*p = XTIM_QUIET;\n")
    assert [i for i, _ in _code_lines(h)] == [4, 5]
    s = tmp_path / "setup.py"
    s.write_text("# XTIM_QUIET\nmacros = [('XTIM_QUIET', '1')]\n")
    assert [i for i, _ in _code_lines(s)] == [2]
    assert all((ROOT / f).is_file() for f in CODE_FILES)


def test_xtim_quiet_is_read_nowhere():
    assert _hits(("XTIM_QUIET", "xtim_quiet", "XTIM_TWIRL_VERBOSE", "_quiet()")) == []


def test_debug_levers_are_gone():
    assert _hits(("FSYND_XCHECK", "OVERLAP_FORCE_DENSE", "CB_BATCH_XCHECK", "IP_W1_XCHECK")) == []


def test_fold_in_levers_are_gone():
    assert _hits(("XTIM_FRAME_COSETS_BACKEND", "FRAME_COSETS_BACKEND", "cfg.trace")) == []
    for f in ("cpp/src/normal_form.cpp", "cpp/src/carried_compose.cpp", "cpp/include/qeccore/residue_projection.hpp",
              "cpp/src/bare_state_memo.cpp"):
        assert [c for _, c in _code_lines(ROOT / f) if "abort(" in c or "fprintf(stderr" in c] == [], f


def test_lever_notices_and_channels_line_are_not_stderr():
    fs = (ROOT / "cpp/src/framed_sampler.cpp").read_text()
    assert "fprintf" not in fs
    ts = (ROOT / "cpp/src/twirl_sampler.cpp").read_text()
    assert 'fprintf(stderr, "[twirl_records] circuit channels' not in ts
    assert 'fprintf(stderr, "twirl_records: table not all-in-class' not in ts
    assert "BLOB VERIFY FAIL" not in "".join(c for _, c in _code_lines(ROOT / "cpp/src/twirl_sampler.cpp")
                                             if "fprintf" in c)
    fam = (ROOT / "cpp/src/twirl_sampler_fam.cpp").read_text()
    assert 'std::fprintf(stderr, "twirl_records: %s\\n"' not in fam
