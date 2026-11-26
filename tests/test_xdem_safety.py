"""SPEC §3.6 and §5 after the migration: the public reader refuses dc-DEM v1 files by name; a hand-stripped xdem is
refused by the xdem reader as a file with x/rot/late fields and no header."""
from pathlib import Path

import pytest
import xtim

from _xdem_circuits import circuit, xdem_export

DATA = Path(__file__).parent / "data" / "xdem"


@pytest.mark.parametrize("name", ["cultivation_d3_faithful.dem", "distillation_15_1_3.dem"])
def test_v1_files_are_refused_with_the_reexport_message(name):
    with pytest.raises(xtim.XtimXdemError, match='re-export the circuit with format="xdem"'):
        xtim.XdemSampler((DATA / "v1" / name).read_text())


def test_hand_stripped_xdem_is_refused():
    x = xdem_export(circuit("ex422_obs"))
    no_hdr_axes = "\n".join(l for l in x.splitlines() if l.startswith("error")) + "\n"
    with pytest.raises(xtim.XtimXdemError, match="no xdem header"):
        xtim.XdemSampler(no_hdr_axes)
