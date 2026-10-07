"""Execute every python block in docs/xtim_tour.md, verbatim.

The tour is the canonical worked example; running it here means it cannot silently
rot as the API evolves. Mirrors the upstream gate (tests/test_xtim_lean.py) so the
student repo carries the same guarantee.
"""
from pathlib import Path

import pytest


def _find_tour() -> Path:
    """Locate docs/xtim_tour.md by walking up from this file (works in both the
    standalone repo layout and the monorepo, where tests live deeper)."""
    for base in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        cand = base / "docs" / "xtim_tour.md"
        if cand.is_file():
            return cand
    raise FileNotFoundError("docs/xtim_tour.md not found above this test")


def _python_blocks(md: str) -> list[str]:
    blocks, cur, in_block = [], [], False
    for line in md.splitlines():
        if not in_block and line.strip() == "```python":
            in_block, cur = True, []
        elif in_block and line.strip() == "```":
            blocks.append("\n".join(cur))
            in_block = False
        elif in_block:
            cur.append(line)
    return blocks


def test_tour_document_executes(monkeypatch):
    # The tour uses the optional decode stack and a committed data file.
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")

    tour = _find_tour()
    repo_root = tour.parent.parent  # <root>/docs/xtim_tour.md -> <root>
    if not (repo_root / "benchmarks" / "cultivation_d3_rate.stim").is_file():
        pytest.skip("tour data file (benchmarks/cultivation_d3_rate.stim) not present")
    monkeypatch.chdir(repo_root)  # the tour reads a relative benchmark path

    blocks = _python_blocks(tour.read_text())
    assert len(blocks) >= 3, "tour lost its python blocks"
    ns: dict = {}
    for i, block in enumerate(blocks):
        exec(compile(block, f"{tour.name}#block{i}", "exec"), ns)  # noqa: S102
