"""The three version strings must agree.

A release bumps the version in THREE places by hand (cut checklist step 1):
``pyproject.toml``'s ``version``, ``xtim/__init__.py``'s dev fallback, and
``CITATION.cff``'s ``version``. The claims audit (2026-08-30, slice D) found
nothing checking they agree: mutating the ``__init__`` fallback out of sync and
running the whole suite in an isolated env left it green, 156 passed / 8
skipped, unchanged.

A skewed fallback is not cosmetic. ``xtim.__version__`` prefers the INSTALLED
distribution metadata and only falls back to the literal when the package is
not installed — i.e. exactly in a source checkout, which is where contributors
and students read it. It would report the previous release indefinitely.

Scope: these compare SOURCE files, so they are a source-tree check by nature.
Run against an installed wheel (which ships none of these files) they skip, and
say so plainly — an expected structural skip, not a silent hole.
"""
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"
_INIT = _ROOT / "xtim" / "__init__.py"
_CITATION = _ROOT / "CITATION.cff"


def _require_source_tree():
    missing = [p.name for p in (_PYPROJECT, _INIT, _CITATION) if not p.is_file()]
    if missing:
        pytest.skip(f"source-tree check: {', '.join(missing)} not present "
                    f"(running against an installed package, which ships none "
                    f"of them)")


def _pyproject_version() -> str:
    import tomllib
    return tomllib.loads(_PYPROJECT.read_text())["project"]["version"]


def _fallback_version() -> str:
    m = re.search(r'__version__ = "([^"]+)\+dev"', _INIT.read_text())
    assert m, "could not find the __version__ dev fallback in xtim/__init__.py"
    return m.group(1)


def _citation_version() -> str:
    m = re.search(r'^version:\s*"([^"]+)"', _CITATION.read_text(), re.M)
    assert m, "could not find version in CITATION.cff"
    return m.group(1)


def test_pyproject_and_dev_fallback_agree():
    _require_source_tree()
    assert _fallback_version() == _pyproject_version(), (
        f"xtim/__init__.py's dev fallback ({_fallback_version()}+dev) disagrees "
        f"with pyproject.toml ({_pyproject_version()}) — a source checkout would "
        f"report the wrong version, which is precisely where the fallback is the "
        f"value anyone sees")


def test_citation_matches_pyproject():
    _require_source_tree()
    assert _citation_version() == _pyproject_version(), (
        f"CITATION.cff ({_citation_version()}) disagrees with pyproject.toml "
        f"({_pyproject_version()}) — citations would name a version that was "
        f"never released")
