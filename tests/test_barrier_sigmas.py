"""Exact TDD test for BarrierBuffer.sigmas() and sig_bits (E1, exact-residual arc).

Asserts for EVERY shot i:
  - unpacking sigmas()[i] bit-by-bit equals the existing per-shot sigma(i) exactly
  - shape is (nshots, SGW) where SGW = ceil(sig_bits / 64)
  - dtype is uint64
  - tail bits beyond sig_bits are zero (as promised by the docstring contract)

Also checks the sig_bits property equals the length of sigma(i).
"""
import math

import numpy as np
import pytest

import xtim
import xtim._xtim as _xtim


# ---------------------------------------------------------------------------
# Shared fixture: a noisy 3-qubit circuit that produces non-trivial σ values.
# ---------------------------------------------------------------------------
_CIRCUIT_TEXT = """\
R 0 1 2
X_ERROR(0.1) 0 1 2
M 0 1 2
DETECTOR rec[-1]
DETECTOR rec[-2]
DETECTOR rec[-3]
"""

_N_SHOTS = 200
_SEED = 1234


@pytest.fixture(scope="module")
def buf():
    """Sample a BarrierBuffer once; all tests in this module share it."""
    s = _xtim.TwirlSampler(_CIRCUIT_TEXT)
    return s.sample_barrier(_N_SHOTS, _SEED)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sigmas_shape(buf):
    """sigmas() must have shape (nshots, SGW) with SGW = ceil(sig_bits / 64)."""
    arr = buf.sigmas()
    sig_bits = buf.sig_bits
    expected_sgw = max(1, math.ceil(sig_bits / 64)) if sig_bits > 0 else 0
    assert arr.shape == (_N_SHOTS, expected_sgw), (
        f"expected shape ({_N_SHOTS}, {expected_sgw}), got {arr.shape}"
    )


def test_sigmas_dtype(buf):
    """sigmas() must return a uint64 array."""
    arr = buf.sigmas()
    assert arr.dtype == np.dtype("uint64"), (
        f"expected dtype uint64, got {arr.dtype}"
    )


def test_sig_bits_equals_sigma_len(buf):
    """sig_bits must equal len(sigma(i)) for every shot."""
    sig_bits = buf.sig_bits
    for i in range(buf.num_shots):
        s = buf.sigma(i)
        assert len(s) == sig_bits, (
            f"shot {i}: sig_bits={sig_bits} but len(sigma(i))={len(s)}"
        )


def test_sigmas_bit_unpacking_equals_per_shot_sigma(buf):
    """CORE oracle: unpack sigmas() bit-by-bit; must equal sigma(i) for every shot.

    Unpacking rule (little-endian 64-bit words, same as sigma() expansion):
        bit b of shot i = (row[b >> 6] >> (b & 63)) & 1
    """
    arr = buf.sigmas()
    sig_bits = buf.sig_bits
    for i in range(buf.num_shots):
        ref = list(buf.sigma(i))  # ground truth (existing per-shot accessor)
        row = arr[i]
        unpacked = [
            int((row[b >> 6] >> np.uint64(b & 63)) & np.uint64(1))
            for b in range(sig_bits)
        ]
        assert unpacked == ref, (
            f"shot {i}: sigmas() unpack differs from sigma(i)\n"
            f"  sigmas unpack: {unpacked}\n"
            f"  sigma(i)     : {ref}"
        )


def test_sigmas_tail_bits_are_zero(buf):
    """Tail bits in the last word (positions sig_bits .. SGW*64-1) must be zero.

    Contract: tail bits are zero by construction — source sigma vectors are
    initialized to GW zero words with only bits 0..ngens-1 ever written, and the
    entire store is pre-zeroed before sampling (the sink does not zero per-row).
    """
    arr = buf.sigmas()
    sig_bits = buf.sig_bits
    sgw = arr.shape[1] if arr.ndim == 2 else 0
    if sgw == 0 or sig_bits == 0:
        return  # vacuous: no tail bits
    tail_start = sig_bits % 64  # first unused bit in last word (0 means none)
    if tail_start == 0:
        return  # last word is fully used; no tail to check
    mask = np.uint64(~((np.uint64(1) << np.uint64(tail_start)) - np.uint64(1)))
    last_words = arr[:, sgw - 1]
    assert np.all((last_words & mask) == np.uint64(0)), (
        f"tail bits in last word are non-zero for sig_bits={sig_bits}: "
        f"{last_words[last_words & mask != 0]}"
    )


def test_sigmas_is_copy_not_alias(buf):
    """sigmas() must return a COPY (or at least an independent buffer).

    Mutating the returned array must not corrupt a second call.
    """
    a1 = buf.sigmas()
    a1[:] = np.uint64(0xFFFFFFFFFFFFFFFF)
    a2 = buf.sigmas()
    # a2 must still round-trip correctly (test via bit-unpacking first shot)
    sig_bits = buf.sig_bits
    ref = list(buf.sigma(0))
    row = a2[0]
    unpacked = [
        int((row[b >> 6] >> np.uint64(b & 63)) & np.uint64(1))
        for b in range(sig_bits)
    ]
    assert unpacked == ref, (
        "sigmas() is aliasing the internal buffer — mutation corrupted the next call"
    )


def test_sigmas_non_trivial(buf):
    """Sanity: at least one σ bit must be 1 in the 200-shot sample (p=0.1 error).

    With 200 shots × 3 bits × p=0.1 the probability of all-zero is (0.9)^600 ≈ 0,
    so this is a deterministic check (not a statistical gate).
    """
    arr = buf.sigmas()
    assert arr.any(), (
        "All sigmas are zero across 200 shots with 10% per-qubit error — "
        "something is wrong with σ assembly or the circuit."
    )


# ---------------------------------------------------------------------------
# Additional workload: cultivation_d5 (larger ngens, SGW might be > 1)
# ---------------------------------------------------------------------------

def test_sigmas_cultivation_d5():
    """Run the same oracle on cultivation_d5 (larger ngens, SGW = ceil(297/64) = 5).

    Exercises the multi-word case (SGW > 1) with the full twirl machinery.
    Uses compile_twirl_sampler to obtain an _xtim.TwirlSampler directly (the
    compile_detector_sampler wrapper does not expose sample_barrier).
    """
    c = xtim.load_example("cultivation_d5")
    s = xtim.compile_twirl_sampler(str(c))
    buf_c = s.sample_barrier(50, 7)

    arr = buf_c.sigmas()
    sig_bits = buf_c.sig_bits

    # Shape check
    expected_sgw = max(1, math.ceil(sig_bits / 64)) if sig_bits > 0 else 0
    assert arr.shape == (50, expected_sgw), (
        f"cultivation_d5: expected shape (50, {expected_sgw}), got {arr.shape}"
    )
    assert arr.dtype == np.dtype("uint64")

    # Per-shot oracle for all 50 shots
    for i in range(50):
        ref = list(buf_c.sigma(i))
        row = arr[i]
        unpacked = [
            int((row[b >> 6] >> np.uint64(b & 63)) & np.uint64(1))
            for b in range(sig_bits)
        ]
        assert unpacked == ref, (
            f"cultivation_d5 shot {i}: sigmas() unpack differs from sigma(i)\n"
            f"  sigmas unpack: {unpacked}\n"
            f"  sigma(i)     : {ref}"
        )
