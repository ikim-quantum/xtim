"""J-b1 P2 / S7: the FAM translation unit reads no environment, and the Python drain no longer reads XTIM_QUIET
(removed outright at S7): the two informational events that TU always queues (selfcheck_window, disk_save) are
emitted whatever the variable says; silencing is by warnings filters / logging levels."""
import logging
import warnings

import pytest

from xtim import _diagnostics as D
from xtim.errors import XtimVerificationWarning


def test_no_quiet_helper():
    assert not hasattr(D, "_quiet")


@pytest.mark.parametrize("quiet", [None, "1", "yes"])
def test_selfcheck_window_and_disk_save_ignore_removed_quiet(monkeypatch, caplog, quiet):
    if quiet is None:
        monkeypatch.delenv("XTIM_QUIET", raising=False)
    else:
        monkeypatch.setenv("XTIM_QUIET", quiet)
    with warnings.catch_warnings(record=True) as rec, caplog.at_level(logging.INFO, logger="xtim.twirl"):
        warnings.simplefilter("always")
        D._emit_one("selfcheck_window", "", 2000, 0, "", 0.0)
        D._emit_one("disk_save", "SAVED", 7, 0, "/tmp/x.twpl", 0.0)
    got_w = [w for w in rec if issubclass(w.category, XtimVerificationWarning)]
    got_l = [r for r in caplog.records if "SAVED (7 plans)" in r.getMessage()]
    assert (len(got_w), len(got_l)) == (1, 1), (quiet, got_w, got_l)
