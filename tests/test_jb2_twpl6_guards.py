"""J-b2 J1b: the TWPL disk-guard claims of tests/test_twpl_cache_guards.py on PRODUCTION's format (TWPL v6, the
family sampler's keys-only file; plan r2.1 J1b). The v5 guards there run on the legacy sampler's cache, which retires
at J2 (and the v5 format with it at J7b).

Claim (xtim/port.py's plan disk-cache policy): stale / corrupt / skewed files (wrong magic, version, engine, group
token, bad checksum, truncated, trailing garbage) are refused at load ("absent/rebuild") and silently rebuilt.
Each blob below is a REAL production-written v6 file with exactly ONE property changed (the FNV-1a64 recomputed
whenever the edited bytes are inside the hashed region, so only the guard under test can reject it). The verdict is
the "[twirl_records] disk '<path>': LOADED|absent/rebuild" INFO record (logger xtim.twirl), read in a fresh
subprocess constructing xtim.compile_twirl_sampler(text, disk_cache=path).

v6 layout (cpp/src/twirl_sampler_fam.cpp save_file_v6_ / load_file_v6_; plan_cache begin_blob / finish_blob):
    magic "TWPL" | u32 version 6 | u64 len + ENGINE_VERSION | u64 group token | u64 term token
    | u64 count | per-plan records (u8 class tag, then the key) | u64 FNV-1a64 of everything after the version field
"""
import os
import pathlib
import struct
import subprocess
import sys
import warnings

import pytest

import xtim
import xtim._xtim as _xe

DATA = pathlib.Path(__file__).parent / "data"
TEXT_PATH = DATA / "adaptq_steane_h_producer.stim"
ENGINE = _xe.ENGINE_VERSION.encode()
TOK = 16 + len(ENGINE)                       # magic | version | len | engine | GROUP TOKEN

_LOG_PREAMBLE = (
    "import logging, sys\n"
    "_h = logging.StreamHandler(sys.stdout); _h.setFormatter(logging.Formatter('%(message)s'))\n"
    "logging.getLogger('xtim.twirl').addHandler(_h)\n"
    "logging.getLogger('xtim.twirl').setLevel(logging.INFO)\n"
)


def _fnv1a64(data):
    h = 1469598103934665603
    for x in data:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _rehash(blob):
    blob = bytearray(blob)
    blob[-8:] = struct.pack("<Q", _fnv1a64(bytes(blob[8:-8])))
    return bytes(blob)


def _banner(path):
    root = str(pathlib.Path(xtim.__file__).resolve().parent.parent)
    prog = (_LOG_PREAMBLE + f"import sys; sys.path.insert(0, {root!r})\nimport warnings; warnings.simplefilter('ignore')\n"
            "import xtim\n"
            f"xtim.compile_twirl_sampler(open({str(TEXT_PATH)!r}).read(), disk_cache={str(path)!r}, selfcheck=0)\n")
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=env, cwd="/")
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if "twirl_records] disk" in ln]
    assert lines, (r.stdout, r.stderr)
    return lines[0]


@pytest.fixture(scope="module")
def real_blob(tmp_path_factory):
    p = tmp_path_factory.mktemp("twpl6") / "plans.twpl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = xtim.compile_twirl_sampler(TEXT_PATH.read_text(), disk_cache=str(p), selfcheck=0)
        s.sample(4000, seed=1)
        del s
    blob = p.read_bytes()
    assert struct.unpack("<I", blob[4:8])[0] == 6 and blob[16:16 + len(ENGINE)] == ENGINE
    (count,) = struct.unpack("<Q", blob[TOK + 16:TOK + 24])
    assert count > 0                                   # non-vacuous: the file carries plans
    return blob


def _check(tmp_path, blob, expect):
    p = tmp_path / "plans.twpl"
    p.write_bytes(blob)
    line = _banner(p)
    assert expect in line, line


def test_wellformed_file_is_loaded(tmp_path, real_blob):
    _check(tmp_path, real_blob, "LOADED")
    _check(tmp_path, _rehash(real_blob), "LOADED")    # the FNV transcription is the engine's


def test_bad_magic_rejected(tmp_path, real_blob):
    _check(tmp_path, b"XXXX" + real_blob[4:], "absent/rebuild")


def test_bad_checksum_rejected(tmp_path, real_blob):
    b = bytearray(real_blob)
    b[-1] ^= 1
    _check(tmp_path, bytes(b), "absent/rebuild")


def test_version_skew_rejected(tmp_path, real_blob):
    for v in (5, 7):
        _check(tmp_path, real_blob[:4] + struct.pack("<I", v) + real_blob[8:], "absent/rebuild")


def test_engine_version_skew_rejected(tmp_path, real_blob):
    b = bytearray(real_blob)
    b[16] ^= 0x20                                      # same length, another engine string
    _check(tmp_path, _rehash(b), "absent/rebuild")


def test_wrong_group_token_rejected(tmp_path, real_blob):
    b = bytearray(real_blob)
    b[TOK] ^= 1
    _check(tmp_path, _rehash(b), "absent/rebuild")


def test_truncated_record_rejected(tmp_path, real_blob):
    body = real_blob[:-8]
    _check(tmp_path, _rehash(body[:-3] + b"\0" * 8), "absent/rebuild")   # the last record cut short, honest FNV


def test_trailing_garbage_rejected(tmp_path, real_blob):
    body = real_blob[:-8]
    _check(tmp_path, _rehash(body + b"\x07\x07\x07" + b"\0" * 8), "absent/rebuild")


def test_corrupt_file_rebuilds_end_to_end(tmp_path, real_blob):
    """A rejected file is rebuilt silently: sampling succeeds, and the bytes equal a cache-free run."""
    import numpy as np
    p = tmp_path / "plans.twpl"
    p.write_bytes(b"garbage")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = xtim.compile_twirl_sampler(TEXT_PATH.read_text(), disk_cache=str(p), selfcheck=0).sample(2000, seed=3)
        b = xtim.compile_twirl_sampler(TEXT_PATH.read_text(), selfcheck=0).sample(2000, seed=3)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
