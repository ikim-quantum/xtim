"""Private test fixtures: circuits that are not distributed with xtim.

They live in ``tests/private/`` (git-ignored), or in the directory named by the environment variable
``XTIM_PRIVATE_FIXTURES``. A test that needs one calls ``private_fixture(name)``, which returns the path or
skips the test with a clear reason when the file is absent. Module-level corpora use ``private_text(name)``,
which returns None when absent (the test then calls ``private_fixture`` to skip).

The fixtures are CCZ-factory circuits and their pinned JSON data; the tests that use them name the file they need.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

PRIVATE_DIR = Path(os.environ.get("XTIM_PRIVATE_FIXTURES") or Path(__file__).resolve().parent / "private")


def private_path(name: str) -> Path:
    return PRIVATE_DIR / name


def private_fixture(name: str) -> Path:
    """The fixture's path; skips the calling test when the file is absent."""
    p = private_path(name)
    if not p.is_file():
        pytest.skip(f"private fixture {name} not present in {PRIVATE_DIR} (not distributed)")
    return p


def private_text(name: str) -> "str | None":
    """The fixture's text, or None when absent (for module-level corpora)."""
    p = private_path(name)
    return p.read_text() if p.is_file() else None


# The private circuits that join the tests/data corpora (oracle sweeps) when present.
PRIVATE_CORPUS = [str(private_path(n)) for n in ("ccz_factory_a.stim", "ccz_factory_a_out.stim", "ccz_factory_b.stim")
                  if private_path(n).is_file()]
