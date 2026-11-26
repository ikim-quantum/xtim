"""Every python block of README.md runs STANDALONE (its own fresh namespace: a reader pastes one block), and every
``# <text>`` comment line that directly follows a ``print(...)`` line at the end of a block equals the line that block
prints (the quickstart pins its numbers this way)."""
import contextlib
import io
import os
import re

import pytest

_README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
_FENCE = re.compile(r"```python\n(.*?)```", re.S)


def _blocks():
    return [m.group(1) for m in _FENCE.finditer(open(_README).read())]


def _expected(code):
    """The comment lines right after the block's last print(...) line (its expected output)."""
    lines = code.rstrip().splitlines()
    want = []
    for ln in reversed(lines):
        if ln.startswith("# "):
            want.append(ln[2:].rstrip())
        else:
            return list(reversed(want)) if ln.lstrip().startswith("print(") else []
    return []


@pytest.mark.skipif(not os.path.exists(_README), reason="README.md not shipped (installed wheel)")
def test_readme_python_blocks_run_and_print_their_comments(tmp_path, monkeypatch):
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    monkeypatch.chdir(tmp_path)
    blocks = _blocks()
    assert len(blocks) >= 1, len(blocks)
    pinned = 0
    for i, code in enumerate(blocks):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, "README.md:block%d" % i, "exec"), {"__name__": "__readme__"})
        want = _expected(code)
        if want:
            assert out.getvalue().splitlines()[-len(want):] == want, (i, out.getvalue(), want)
            pinned += 1
    assert pinned >= 1, pinned
