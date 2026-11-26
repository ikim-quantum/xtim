"""The README quick start runs without a warning and prints the documented rate.

The README's ```python block is run VERBATIM in a fresh process (the selfcheck notice is once per process, so an
in-process run could be vacuous). Its import lines run first; ``warnings.simplefilter("error")`` is then switched on
for the rest of the block (so an unrelated third-party import warning cannot fail the test). The block must raise
nothing, print README's own documented rate, and the oracle-window notice must arrive as an INFO record on
``xtim.twirl`` (never a warning).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_PRELUDE = '''
import json, logging, warnings
logs = []
class _H(logging.Handler):
    def emit(self, r): logs.append([r.name, r.levelname, r.getMessage()])
_lg = logging.getLogger("xtim.twirl"); _lg.addHandler(_H()); _lg.setLevel(logging.INFO)
'''

_EPILOGUE = '''
print("@@LOGS@@" + json.dumps(logs))
'''


def _readme_block():
    text = (_ROOT / "README.md").read_text()
    block = re.search(r"## Quick start.*?```python\n(.*?)```", text, re.S).group(1)
    rate = re.search(r"^# logical error rate (\S+)$", block, re.M).group(1)
    lines = block.splitlines()
    n_imp = 0                                   # the block's leading import (or blank) lines
    while n_imp < len(lines) and (not lines[n_imp].strip() or re.match(r"(import |from \S+ import )", lines[n_imp])):
        n_imp += 1
    assert n_imp >= 1 and "compile_twirl_sampler" in block, block
    return "\n".join(lines[:n_imp]), "\n".join(lines[n_imp:]), rate


def test_readme_quickstart_is_warning_free_and_reproduces_the_rate():
    imports, body, rate = _readme_block()
    script = _PRELUDE + imports + '\nwarnings.simplefilter("error")\n' + body + "\n" + _EPILOGUE
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stderr
    assert "Warning" not in p.stderr, p.stderr
    out = p.stdout.splitlines()
    logs = json.loads(out[-1][len("@@LOGS@@"):])
    assert out[:-1] == [f"logical error rate {rate}"], out          # the README's own documented output
    assert rate == "6.78e-03", rate
    notice = [m for name, lvl, m in logs if name == "xtim.twirl" and lvl == "INFO" and "selfcheck=" in m]
    assert notice == ["[xtim] selfcheck=2000: oracle window active on the fast path (first run of this sampler "
                      "only; slower; diagnostic)"], logs
