"""examples/xdem: every <name>.xdem is the current default xdem export of its <name>.stim, byte for byte, loads and
samples in XdemSampler, and equals its tests/data/xdem fixture where one exists; every <name>.stim is its source
(examples/<name>.stim, a tests/_xdem_circuits.py variant, a tests/data/xdem/circuits toy, or TDAG_CYCLE of
tests/test_xdem_export.py) with at most some leading comment lines added; the README table lists exactly the pairs
present and its numeric columns (axes, rotating, late, tc / lines, lines, bytes) match the files; the lossy column needs
sampling and is not checked; the README's refused examples are still refused with the listed message prefix, and every
bundled example is a pair, a listed refusal or cube_ccz (whose export is empty). Skipped when examples/ is absent."""
import re
from pathlib import Path

import pytest

import xtim
from _xdem_circuits import TOYS, VARIANTS, variant

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
DIR = ROOT / "examples" / "xdem"
FIXTURES = TESTS / "data" / "xdem"
STIMS = sorted(DIR.glob("*.stim"))


def _readme():
    f = DIR / "README.md"
    if not f.exists():
        pytest.skip("examples/xdem/README.md absent (not a source checkout)")
    return f.read_text()


def _rows():
    """name -> the table's cells after the name."""
    rows = {}
    for line in _readme().splitlines():
        m = re.match(r"^\| `([^`]+)` \|(.*)\|$", line)
        if m:
            assert m.group(1) not in rows, m.group(1)
            rows[m.group(1)] = [c.strip() for c in m.group(2).split("|")]
    return rows


def _refused():
    section = _readme().split("## Refused", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^- `([^`]+)`: `([^`]+)`$", section, flags=re.M)


def _source(name):
    if (ROOT / "examples" / f"{name}.stim").exists():
        return (ROOT / "examples" / f"{name}.stim").read_text()
    for v in VARIANTS:
        if v.replace(":", "_") == name:
            t = variant(v)
            return t if t.endswith("\n") else t + "\n"
    if name in TOYS:
        return TOYS[name].read_text()
    if name == "tdag_cycle":
        m = re.search(r'^TDAG_CYCLE = """(.*?)"""', (TESTS / "test_xdem_export.py").read_text(), flags=re.S | re.M)
        return m.group(1)
    raise AssertionError(f"{name}: no known source")


def test_every_circuit_has_its_pair():
    if not DIR.exists():
        pytest.skip("examples/xdem absent")
    assert STIMS
    assert sorted(p.stem for p in DIR.glob("*.xdem")) == sorted(p.stem for p in STIMS)


@pytest.mark.parametrize("stim", STIMS, ids=lambda p: p.stem)
def test_committed_xdem_is_the_export(stim):
    text = xtim.Circuit(stim.read_text()).detector_error_model_text(format="xdem")
    assert text == stim.with_suffix(".xdem").read_text(), "run examples/xdem/regenerate.py"


@pytest.mark.parametrize("stim", STIMS, ids=lambda p: p.stem)
def test_copy_is_its_source(stim):
    copy, src = stim.read_text(), _source(stim.stem)
    assert copy.endswith(src), stim.stem
    prefix = copy[:len(copy) - len(src)]
    assert prefix == "" or prefix.endswith("\n"), (stim.stem, prefix[-40:])   # no '#' glued onto the first line
    added = prefix.splitlines()
    assert all(l.startswith("#") for l in added), (stim.stem, added)


@pytest.mark.parametrize("stim", STIMS, ids=lambda p: p.stem)
def test_xdem_equals_its_fixture(stim):
    fx = [FIXTURES / (stim.stem + s) for s in (".xdem", ".dem") if (FIXTURES / (stim.stem + s)).exists()]
    if not fx:
        pytest.skip("no tests/data/xdem fixture")
    assert stim.with_suffix(".xdem").read_text() == fx[0].read_text()


@pytest.mark.parametrize("stim", STIMS, ids=lambda p: p.stem)
def test_xdem_loads_and_samples(stim):
    text = stim.with_suffix(".xdem").read_text()
    s = xtim.XdemSampler(text)
    assert s.extended == text.startswith("xdem 1")
    dets, status = s.sample(64, seed=3)
    assert len(dets) == 64


def test_readme_table_lists_exactly_the_pairs():
    assert sorted(_rows()) == sorted(p.stem for p in STIMS)


@pytest.mark.parametrize("stim", STIMS, ids=lambda p: p.stem)
def test_readme_numbers_match_the_file(stim):
    row = _rows()[stim.stem]
    kind, axes, rot, late, tc, _lossy, nlines, nbytes = row[1:]
    text = stim.with_suffix(".xdem").read_text()
    lines = text.splitlines()
    assert (int(nlines), int(nbytes)) == (len(lines), len(text.encode())), stim.stem
    if not text.startswith("xdem 1"):
        assert kind.startswith("conventional") and [axes, rot, late, tc, _lossy] == ["-"] * 5, stim.stem
        assert text == xtim.Circuit(stim.read_text()).detector_error_model_text()
        return
    assert kind == "extended", stim.stem
    err = [l for l in lines if l.startswith("error")]
    got = (sum(l.startswith("axis") for l in lines), sum(" rot " in l or " late " in l for l in err),
           sum(" late " in l for l in err), f"{int(lines[0].split(' tc ')[1])} / {len(err)}")
    assert got == (int(axes), int(rot), int(late), tc), (stim.stem, got)


def test_readme_lists_refusals():
    assert len(_refused()) == 2


def test_refused_examples_still_refused():
    for name, prefix in _refused():
        with pytest.raises(xtim.XtimDemError) as e:
            xtim.Circuit(xtim.load_example(name).text).detector_error_model_text(format="xdem")
        assert str(e.value).startswith(prefix), (name, str(e.value))


def test_every_bundled_example_is_accounted_for():
    if not DIR.exists():
        pytest.skip("examples/xdem absent")
    known = {p.stem for p in STIMS} | {n for n, _ in _refused()} | {"cube_ccz"}
    assert set(xtim.list_examples()) <= known, set(xtim.list_examples()) - known
    assert xtim.Circuit(xtim.load_example("cube_ccz").text).detector_error_model_text(format="xdem") == ""
