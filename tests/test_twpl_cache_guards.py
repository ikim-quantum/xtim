"""TWPL plan-cache disk-guard tests (audit slice D closure, 2026-08-30).

The claim under test: xtim/port.py:97-98 (plan disk-cache growth policy) --
"stale/corrupt/skew files (wrong version, wrong group token, bad checksum)
load NOTHING and are silently rebuilt/overwritten." Backing comment,
cpp/src/twirl_sampler.cpp:2129-2130: "the TWPL framing re-verifies the token
on load, so stale/corrupt/skew files rebuild silently."

The audit (2026-08-30, slice D) found this UNENFORCED: it removed the
version-skew guard in TwirlPlanCache::load_file, rebuilt the extension, and
the full suite (160/4 at the time) passed unchanged -- no test anywhere wrote
a corrupt/skewed .twpl file and checked it got rejected. This file closes
that gap.

Mechanism (read from source, not assumed): the on-disk blob is
    [4B magic "TWPL"][u32 version][u64 group_token][u64 count][per-plan
    payload...][u64 FNV-1a64 hash of everything from group_token to the end
    of the last plan] -- see plan_cache::begin_blob/finish_blob/open_blob
    (cpp/src/plan_cache.cpp) and TwirlPlanCache::save_file/load_file
    (cpp/src/twirl_kernel_sampler.cpp:291-373). The hash deliberately does
    NOT cover the version field (plan_cache.hpp:67) -- version skew is
    caught by a SEPARATE explicit equality check -- which is exactly why a
    checksum-only reading of "corrupt" would miss it.

load_file rejects (returns false, nothing added to the in-memory store) on
SIX distinct branches; the claim names three explicitly (version, group
token, checksum) plus generic "corrupt", which this file reads as covering
bad-magic and truncation too:

    G1  bad magic            plan_cache.cpp open_blob(), memcmp(magic) != 0
    G2  bad checksum         plan_cache.cpp open_blob(), fnv1a64 mismatch
    G3  version skew         twirl_kernel_sampler.cpp:342, ver != kTwplVersion
    G4  wrong group token    twirl_kernel_sampler.cpp:345, tok != group_token
    G5  short/truncated read twirl_kernel_sampler.cpp:366, mid-record !r.ok
    G6  trailing garbage     twirl_kernel_sampler.cpp:372, !r.at_end()
        (bonus: not literally named by the port.py claim's parenthetical,
        but a real, separate guard -- included for completeness.)

Each test below hand-builds a blob that is byte-perfect EXCEPT for the one
property under test (see _build_twpl), so that removing any ONE guard in the
C++ source is expected to flip exactly the corresponding test from pass to
fail and no other. This is verified by mutation (see the branch history /
task report for the transcript); it is not re-verified by this file itself.

CLAIM CORRECTION shipped with this file: port.py's bullet used to say such
files "load NOTHING". True of G1-G4 (all of which run before any plan is
parsed) but NOT of G5, which stops mid-loop with no rollback, leaving the
already-parsed prefix in the cache -- as the C++'s own "loaded prefix stands"
comment states. Since that prefix has already cleared the checksum and the
group token, it is genuine current-group data, not the stale/skewed content
the claim is about; port.py now says this precisely, and
test_short_read_retains_already_validated_prefix pins it.

All assertions read the "[twirl_records] disk '<path>': LOADED|absent/rebuild
(...)" banner that TwirlRecordSampler::Impl's constructor prints at
load_file() time (twirl_sampler.cpp:2149) -- i.e. as soon as a TwirlSampler
object is constructed with disk_cache=<path>, BEFORE any .sample() call.
Tests therefore never need to exercise a plan that a bypassed guard let
through (see test_truncated_short_read_rejected for why that specifically
matters). XTIM_QUIET gates that banner and is a function-local static read
ONCE per process (documented in twirl_sampler.cpp: "tests toggling them need
subprocesses") -- so every test here runs its TwirlSampler construction in a
subprocess with XTIM_QUIET explicitly unset, mirroring the existing pattern
in test_born_fastpath.py.
"""
import os
import pathlib
import struct
import subprocess
import sys

import pytest

_DATA = pathlib.Path(__file__).parent / "data"
_TEXT_PATH = _DATA / "adaptq_steane_h_producer.stim"

_TWPL_MAGIC = b"TWPL"
_TWPL_VERSION = 2   # must match kTwplVersion, cpp/src/twirl_kernel_sampler.cpp:278

# Literal copy of plan_cache::fnv1a64's constants (cpp/src/plan_cache.cpp) --
# cross-validated against a REAL engine-written blob in
# test_fnv1a64_matches_engine below, so a transcription slip here would be
# caught rather than silently making every "valid except for X" blob below
# ALSO checksum-invalid (which would make every rejection test pass for the
# wrong reason and blind the mutation check on G3/G4/G5/G6).
_FNV_OFFSET = 1469598103934665603
_FNV_PRIME = 1099511628211
_MASK64 = (1 << 64) - 1


def _fnv1a64(data: bytes) -> int:
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & _MASK64
    return h


def _build_twpl(version: int, token: bytes, count: int, extra_payload: bytes = b"",
                magic: bytes = _TWPL_MAGIC, corrupt_hash: bool = False) -> bytes:
    """Hand-build a TWPL blob byte-for-byte:
        magic(4) | version(u32 LE) | token(u64 LE) | count(u64 LE) | extra_payload
        | fnv1a64-hash(u64 LE)
    matching plan_cache::begin_blob/finish_blob + TwirlPlanCache::save_file's
    header layout exactly (verified in test_wellformed_synthetic_cache_is_loaded).
    `count` is a bare declared integer -- for count=0 it's simply correct; for
    count>0 with extra_payload NOT containing that many well-formed plan
    records, it is a deliberate lie used to build a truncated-but-checksum-
    honest blob (see test_truncated_short_read_rejected). The hash always
    covers the REAL bytes present (token+count+extra_payload), matching
    finish_blob's "hash the payload, not the promise" behavior, so a plain
    truncation-of-a-real-file is NOT how G5 is exercised (that would fail the
    checksum, i.e. G2, instead -- see the module docstring).
    """
    assert len(token) == 8
    header = magic + struct.pack("<I", version)
    payload = token + struct.pack("<Q", count) + extra_payload
    h = _fnv1a64(payload)
    if corrupt_hash:
        h ^= 1
    return header + payload + struct.pack("<Q", h)


def _run(prog: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("XTIM_QUIET", None)   # guarantee informational banners print (fresh
                                   # process -- required, see module docstring)
    return subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=env)


def _disk_banner(cache_path: pathlib.Path, blob: bytes) -> subprocess.CompletedProcess:
    """Write `blob` to cache_path, construct a TwirlSampler against it (which
    alone triggers load_file + the disk banner) in a subprocess, and return the
    completed process (stdout/stderr/returncode)."""
    cache_path.write_bytes(blob)
    prog = (
        "import xtim._xtim as _x\n"
        f"text = open({str(_TEXT_PATH)!r}).read()\n"
        f"_x.TwirlSampler(text, 1.0, {str(cache_path)!r}, 0, False, '')\n"
    )
    return _run(prog)


def _assert_rejected(cache_path: pathlib.Path, blob: bytes, label: str) -> None:
    r = _disk_banner(cache_path, blob)
    assert r.returncode == 0, f"{label}: subprocess crashed:\n{r.stderr}"
    lines = [l for l in r.stderr.splitlines() if "twirl_records] disk" in l]
    assert lines, f"{label}: no disk banner in stderr:\n{r.stderr}"
    assert "absent/rebuild" in lines[0], f"{label}: cache was NOT rejected: {lines[0]}"


@pytest.fixture(scope="module")
def real_token(tmp_path_factory) -> bytes:
    """The REAL, live TwirlPlanCache group token for the steane-class circuit,
    captured once per session. No Python binding exposes
    TwirlRecordSampler::group_token() directly, so the only way to learn it is
    to let the engine write a genuine .twpl file and read back its token field
    (bytes[8:16] -- verified byte-for-byte in test_wellformed_synthetic_cache_is_loaded).
    Pure function of the circuit text (not seed/shots-dependent), so sharing
    this across tests in the module is safe."""
    cache = tmp_path_factory.mktemp("token_capture") / "capture.twpl"
    prog = (
        "import xtim._xtim as _x\n"
        f"text = open({str(_TEXT_PATH)!r}).read()\n"
        f"s = _x.TwirlSampler(text, 1.0, {str(cache)!r}, 0, False, '')\n"
        "s.sample(16, 1, None)\n"
    )
    r = _run(prog)
    assert r.returncode == 0, f"token-capture run failed:\n{r.stderr}"
    blob = cache.read_bytes()
    assert blob[:4] == _TWPL_MAGIC and len(blob) >= 24, "capture did not produce a TWPL blob"
    return blob[8:16]


def test_fnv1a64_matches_engine(real_token, tmp_path):
    """Self-check: the Python fnv1a64 reimplementation used to build synthetic
    .twpl blobs below EXACTLY matches plan_cache::fnv1a64, verified against a
    REAL blob the engine itself wrote (the real_token fixture's capture file
    is exactly such a blob). If this fails, every test below is unreliable."""
    cache = tmp_path / "capture_reuse.twpl"
    prog = (
        "import xtim._xtim as _x\n"
        f"text = open({str(_TEXT_PATH)!r}).read()\n"
        f"s = _x.TwirlSampler(text, 1.0, {str(cache)!r}, 0, False, '')\n"
        "s.sample(16, 1, None)\n"
    )
    r = _run(prog)
    assert r.returncode == 0, r.stderr
    blob = cache.read_bytes()
    stored_hash = struct.unpack("<Q", blob[-8:])[0]
    payload = blob[8:-8]
    assert _fnv1a64(payload) == stored_hash, "Python fnv1a64 does not match plan_cache::fnv1a64"


def test_wellformed_synthetic_cache_is_loaded(real_token, tmp_path):
    """Positive control: a byte-perfect, hand-built, EMPTY (count=0) TWPL blob
    carrying the real live group token IS accepted ('LOADED'). Without this
    control the rejection tests below can't distinguish "my malformed blob is
    correctly rejected" from "ANY hand-built file is rejected, even a valid
    one" -- i.e. it proves _build_twpl's byte layout is right and the
    rejections that follow are actually about the corrupted field."""
    cache = tmp_path / "wellformed.twpl"
    blob = _build_twpl(_TWPL_VERSION, real_token, 0)
    r = _disk_banner(cache, blob)
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stderr.splitlines() if "twirl_records] disk" in l]
    assert lines, f"no disk banner in stderr:\n{r.stderr}"
    assert "LOADED" in lines[0], f"well-formed synthetic cache was NOT loaded: {lines[0]}"


def test_bad_magic_rejected(real_token, tmp_path):
    """G1: wrong 4-byte magic; checksum/version/token are otherwise all VALID
    (the hash covers only bytes past the magic+version, so a wrong magic does
    not by itself invalidate it) -- isolates open_blob's memcmp guard."""
    blob = _build_twpl(_TWPL_VERSION, real_token, 0, magic=b"XXXX")
    _assert_rejected(tmp_path / "bad_magic.twpl", blob, "bad-magic")


def test_bad_checksum_rejected(real_token, tmp_path):
    """G2: correct magic/version/token/count, but the trailing FNV-1a64 hash is
    deliberately wrong (real hash XOR 1) -- isolates open_blob's hash-compare
    guard."""
    blob = _build_twpl(_TWPL_VERSION, real_token, 0, corrupt_hash=True)
    _assert_rejected(tmp_path / "bad_checksum.twpl", blob, "bad-checksum")


def test_version_skew_rejected(real_token, tmp_path):
    """G3: correct magic/checksum/token, WRONG version. This is exactly the
    scenario the audit's own mutation constructed by hand ("a file with a
    wrong-but-otherwise-well-formed version field") to show the guard was
    unenforced -- the checksum deliberately excludes the version field
    (plan_cache.hpp:67), so ONLY the explicit `ver != kTwplVersion` check
    (twirl_kernel_sampler.cpp:342) can catch this."""
    blob = _build_twpl(_TWPL_VERSION + 37, real_token, 0)
    _assert_rejected(tmp_path / "version_skew.twpl", blob, "version-skew")


def test_wrong_group_token_rejected(tmp_path):
    """G4: correct magic/version/checksum, WRONG group token -- isolates
    load_file's `tok != group_token` guard (twirl_kernel_sampler.cpp:345). A
    real live token is not needed here (any token virtually certainly differs
    from the true one), so this test does not depend on the real_token
    fixture."""
    wrong_token = struct.pack("<Q", 0xDEADBEEFCAFEBABE)
    blob = _build_twpl(_TWPL_VERSION, wrong_token, 0)
    _assert_rejected(tmp_path / "wrong_token.twpl", blob, "wrong-token")


def test_truncated_short_read_rejected(real_token, tmp_path):
    """G5: correct magic/version/checksum/token, but count=1 while ZERO bytes
    of that one promised plan record are actually present -- the checksum is
    computed honestly over the truncated payload (token+count only), so it
    passes, and the rejection can ONLY come from load_file's mid-loop
    `if (!r.ok) return false` (twirl_kernel_sampler.cpp:366, "any short read:
    stop"). Deliberately does NOT call .sample() after construction: with the
    guard intact this is moot (nothing was loaded, store_ stays untouched),
    and per the module docstring this test's Python code must stay identical
    (and therefore safe) even when mutation-testing the guard's removal, where
    a bypassed guard would splice one all-default/empty CachedPlan into the
    live cache -- inspecting only the load-time banner avoids ever exercising
    that plan."""
    blob = _build_twpl(_TWPL_VERSION, real_token, count=1, extra_payload=b"")
    _assert_rejected(tmp_path / "truncated.twpl", blob, "truncated")


def test_trailing_garbage_rejected(real_token, tmp_path):
    """G6 (bonus -- not literally named by the port.py claim's parenthetical,
    but a real, separate guard worth closing too): correct magic/version/
    checksum/token/count=0, plus 4 extra junk bytes appended INSIDE the hashed
    region (so the checksum is honestly valid for the padded payload). Every
    declared record parses fine (there are zero), but the reader does not
    land exactly on the end of the frame -- isolates the final
    `return r.at_end()` (twirl_kernel_sampler.cpp:372)."""
    blob = _build_twpl(_TWPL_VERSION, real_token, count=0, extra_payload=b"\xff\xff\xff\xff")
    _assert_rejected(tmp_path / "trailing_garbage.twpl", blob, "trailing-garbage")


def test_short_read_retains_already_validated_prefix(tmp_path):
    """FINDING (claim correction, 2026-08-30): the G5 short-read branch does NOT
    satisfy port.py's literal "load NOTHING". load_file pushes each parsed plan
    into store_ as it goes and simply `return false`s mid-loop on a short read
    -- with no rollback -- so the plans parsed BEFORE the short read stay in the
    live cache. The C++ comment at that line says as much ("loaded prefix
    stands", twirl_kernel_sampler.cpp:366); it was port.py's summary that
    over-claimed. port.py's growth-policy text is corrected in the same commit
    as this test, and this test is the mechanism holding the corrected wording
    honest.

    Not a correctness exposure, and the distinction is why: the retained prefix
    is only reachable AFTER the magic, the whole-payload checksum and the group
    token have all already passed, so those plans are genuine, current-version,
    right-group plans -- never the stale/skewed/wrong-group data the claim is
    actually about. It is also barely reachable in practice: a genuinely
    truncated file fails the checksum (G2) long before the loop starts, because
    truncation reinterprets real plan bytes as the trailing hash. Reaching G5 at
    all takes a checksum-honest blob whose declared count exceeds its records --
    which is what this test constructs, by taking a REAL engine-written file and
    bumping its count field by one, re-hashing so the blob stays checksum-valid.

    Asserted here: the file is still REJECTED as a load ('absent/rebuild', so
    nothing downstream treats it as a warm cache) while the banner's plan count
    shows the validated prefix was kept."""
    real = tmp_path / "genuine.twpl"
    prog = (
        "import xtim._xtim as _x\n"
        f"text = open({str(_TEXT_PATH)!r}).read()\n"
        f"s = _x.TwirlSampler(text, 1.0, {str(real)!r}, 0, False, '')\n"
        "s.sample(512, 99, None)\n"
    )
    r = _run(prog)
    assert r.returncode == 0, r.stderr
    blob = real.read_bytes()
    n = struct.unpack("<Q", blob[16:24])[0]
    assert n >= 1, "engine wrote no plans; probe cannot construct a non-empty prefix"

    # Declare one more plan than the file actually carries, keeping the checksum
    # honest for the bytes that ARE present (so G2 passes and G5 is reached).
    payload = bytearray(blob[8:-8])
    payload[8:16] = struct.pack("<Q", n + 1)
    doctored = blob[:8] + bytes(payload) + struct.pack("<Q", _fnv1a64(bytes(payload)))

    cache = tmp_path / "count_overrun.twpl"
    res = _disk_banner(cache, doctored)
    assert res.returncode == 0, res.stderr
    lines = [l for l in res.stderr.splitlines() if "twirl_records] disk" in l]
    assert lines, f"no disk banner in stderr:\n{res.stderr}"
    assert "absent/rebuild" in lines[0], f"count-overrun file was NOT rejected: {lines[0]}"
    # The documented-and-now-corrected behavior: the validated prefix stands.
    assert f"({n} plans" in lines[0], (
        "expected the already-validated prefix to be retained (the C++ 'loaded "
        f"prefix stands' contract); banner was: {lines[0]}")


def test_corrupt_cache_transparently_rebuilds_end_to_end(real_token, tmp_path):
    """Integration check for the port.py claim's actual wording ('load NOTHING
    ... silently rebuilt/overwritten'): a version-skewed cache file is
    rejected at load, the run still completes correctly (never crashes or
    surfaces the corruption to the caller), and the file on disk afterward is
    a fully valid, CORRECT-schema blob (right magic/version/live token, a
    self-consistent checksum) with n_plans>0 -- i.e. the corrupt file was
    genuinely overwritten with fresh plans, not merely ignored in memory."""
    cache = tmp_path / "will_rebuild.twpl"
    cache.write_bytes(_build_twpl(_TWPL_VERSION + 1, real_token, 0))
    prog = (
        "import xtim._xtim as _x\n"
        f"text = open({str(_TEXT_PATH)!r}).read()\n"
        f"s = _x.TwirlSampler(text, 1.0, {str(cache)!r}, 0, False, '')\n"
        "out = s.sample(64, 5, None)\n"
        "print('SHOTS_OK', out[0].shape[0])\n"
    )
    r = _run(prog)
    assert r.returncode == 0, r.stderr
    assert "SHOTS_OK 64" in r.stdout, f"run did not complete cleanly:\n{r.stdout}\n{r.stderr}"
    lines = [l for l in r.stderr.splitlines() if "twirl_records] disk" in l]
    assert any("absent/rebuild" in l for l in lines), f"expected a rejection banner:\n{r.stderr}"
    assert any("SAVED" in l for l in lines), f"expected a save banner:\n{r.stderr}"

    blob = cache.read_bytes()
    assert blob[:4] == _TWPL_MAGIC, "rebuilt file has the wrong magic"
    assert struct.unpack("<I", blob[4:8])[0] == _TWPL_VERSION, "rebuilt file has the wrong version"
    assert blob[8:16] == real_token, "rebuilt file does not carry the live group token"
    count = struct.unpack("<Q", blob[16:24])[0]
    assert count > 0, "expected at least one rebuilt plan"
    stored_hash = struct.unpack("<Q", blob[-8:])[0]
    assert _fnv1a64(blob[8:-8]) == stored_hash, "rebuilt file's checksum is self-inconsistent"
