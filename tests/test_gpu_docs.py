"""The GPU documentation: examples/10_gpu_estimate.py runs (on the CPU when no GPU opens), the python blocks of the
"GPU sampling" section of docs/xdem.md run standalone, and every number of the GPU table in docs/performance.md is
in the committed validation report it cites (docs/assets/gpu/metal-m3max.json)."""
import json
import os
import pathlib
import re
import subprocess
import sys

import xtim

ROOT = pathlib.Path(__file__).resolve().parents[1]
_FENCE = re.compile(r"```python\n(.*?)```", re.S)


def _section(md, heading):
    """The text of the `## ...` section whose heading starts with `heading`, up to the next `## `."""
    m = re.search(r"^## " + re.escape(heading) + r".*?(?=^## |\Z)", md, re.S | re.M)
    assert m, f"no section {heading!r}"
    return m.group(0)


def test_example_runs_on_cpu_fallback(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(xtim.__file__).resolve().parents[1]))
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "10_gpu_estimate.py")], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    from xtim import gpu
    want = gpu.available() or "cpu"
    assert f"device: {want}" in r.stdout, r.stdout
    m = re.search(r"infidelity = ([0-9.e-]+) \+- ([0-9.e-]+)", r.stdout)
    assert m and float(m[1]) > 0 and float(m[2]) > 0, r.stdout


def test_docs_code_blocks_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sec = _section((ROOT / "docs" / "xdem.md").read_text(), "11. GPU sampling")
    blocks = _FENCE.findall(sec)
    assert len(blocks) >= 2, len(blocks)
    ns = {"__name__": "__docs__"}    # the section's blocks read in order: a later one continues an earlier one
    for i, code in enumerate(blocks):
        exec(compile(code, "docs/xdem.md:gpu-block%d" % i, "exec"), ns)


def test_performance_table_numbers_are_in_the_report():
    report = json.loads((ROOT / "docs" / "assets" / "gpu" / "metal-m3max.json").read_text())
    sec = _section((ROOT / "docs" / "performance.md").read_text(), "How fast is the GPU sampler")
    sec = sec.split("\n### ")[0]    # the M3 Max part: the subsections below it are other machines
    assert "metal-m3max.json" in sec and report["device_info"]["name"] in sec
    rows = report["sections"]["benchmark"]["rows"]
    keys = ("device", "cpu_all", "cpu_1", "cpu_score_all", "cpu_score_1", "speedup_vs_cpu_all", "speedup_vs_best_cpu")
    want = {(r["file"], r["entry"]): r for r in rows}
    table = [ln for ln in sec.splitlines() if ln.startswith("| ") and "`" in ln.split("|")[1]]
    assert len(table) == len(rows), len(table)
    for ln in table:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        r = want[(cells[0].strip("`"), cells[1].strip("`"))]
        assert len(cells) == 2 + len(keys), ln
        for cell, k in zip(cells[2:], keys):
            if r.get(k) is None:   # the sample+score baseline is measured for the infidelity rows only
                assert cell == "–", (ln, k)
                continue
            # rates in millions to one decimal, ratios to two
            w, tol = (r[k], 0.0051) if k.startswith("speedup") else (r[k] / 1e6, 0.051)
            assert abs(float(cell) - w) <= tol, (ln, k, r[k])
        best = max(r[k] for k in keys[1:5] if r.get(k) is not None)
        assert r["cpu_best"] == best and abs(r["speedup_vs_best_cpu"] - r["device"] / best) <= 0.006
    # the headline ratios quoted in the text are the stronger baseline's
    for x in (r for r in rows if r["entry"].startswith("infidelity")):
        assert f"{x['speedup_vs_best_cpu']:.2f}" in sec, x["file"]
    assert report["environment"]["git_commit"][:7] in sec and report["passed"]


def test_cuda_section_matches_its_report():
    """The RTX 6000 subsection quotes the committed CUDA report's device, counts and benchmark table."""
    report = json.loads((ROOT / "docs" / "assets" / "gpu" / "cuda-rtx6000.json").read_text())
    md = (ROOT / "docs" / "performance.md").read_text()
    sec = _section(md, "How fast is the GPU sampler")
    sub = re.search(r"^### On an NVIDIA GPU.*?(?=^### |\Z)", sec, re.S | re.M)
    assert sub, "no RTX 6000 subsection"
    sub = sub[0]
    s = report["sections"]
    assert report["passed"] and "cuda-rtx6000.json" in sub and report["device_info"]["name"] in sub
    assert report["environment"]["git_commit"][:7] in sub and report["device_info"]["cupy"] in sub
    assert f"{s['tests']['passed']} CUDA-marked tests" in sub and s["tests"]["failures"] == 0 == s["tests"]["skipped"]
    for sec_name, label in (("kernels", "kernel cases"), ("entry_points", "entry-point cases")):
        n = len(s[sec_name]["cases"])
        assert all(c["ok"] for c in s[sec_name]["cases"])
        assert f"{n} of {n} {label}" in sub or f"{n} of {n} kernel cases (all" in sub, label
    rows = s["benchmark"]["rows"]
    table = [ln for ln in sub.splitlines() if ln.startswith("| `")]
    assert len(table) == len(rows) == 25
    for ln, r in zip(table, rows):
        cells = [c.strip() for c in ln.strip("|").split("|")]
        assert cells[0].strip("`") == r["file"] and cells[1].strip("`") == r["entry"], ln
        assert abs(float(cells[2]) - r["device"] / 1e6) <= 0.051, ln
        assert abs(float(cells[-1]) - r["speedup_vs_best_cpu"]) <= 0.0051, ln
    assert report["environment"]["git_dirty"] and "committed afterwards" in sub   # the report's one difference
    # the same name and driver in the xdem status paragraph
    assert report["device_info"]["name"] in (ROOT / "docs" / "xdem.md").read_text()


def test_validation_report_carries_no_local_paths(tmp_path):
    """A report is committed to a public repo: no home directory, user name, repo or temp path in it (--out reduced to
    its file name); the committed Metal report is held to the same rule."""
    import getpass
    import socket
    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_validate.py"), "--device", "none", "--quick",
                        "--bench-shots", "2000", "--files", "cultivation_d3_faithful", "--out", str(out)],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert out.exists(), r.stderr[-2000:]
    texts = {"fresh": out.read_text(), "committed": (ROOT / "docs" / "assets" / "gpu" / "metal-m3max.json").read_text(),
             "cuda": (ROOT / "docs" / "assets" / "gpu" / "cuda-rtx6000.json").read_text()}
    assert json.loads(texts["fresh"])["argv"][-2:] == ["--out", "report.json"]
    bad = [str(ROOT), str(pathlib.Path.home()), str(tmp_path), os.path.realpath(tmp_path), sys.prefix,
           "/Users/", "/home/", "/private/tmp"]
    user = getpass.getuser()
    if len(user) >= 4:
        bad.append(user)
    if len(socket.gethostname()) >= 4:
        bad.append(socket.gethostname())
    for name, text in texts.items():
        for b in bad:
            assert b not in text, (name, b)
