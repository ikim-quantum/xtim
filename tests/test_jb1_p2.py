"""J-b1 P2 / S7: the FAM translation unit reads no environment, and the Python drain no longer reads XTIM_QUIET
(removed outright at S7): the informational event that TU always queues (selfcheck_window) is emitted whatever the
variable says; silencing is by warnings filters / logging levels. (bloat-cut C: the disk_save / disk_load events are
deleted with the TWPL disk layer; an event of either kind is an engine/binding skew and raises.)"""
import logging
import warnings

import pytest

from xtim import _diagnostics as D
from xtim.errors import XtimVerificationWarning


def test_no_quiet_helper():
    assert not hasattr(D, "_quiet")


@pytest.mark.parametrize("quiet", [None, "1", "yes"])
def test_selfcheck_window_ignores_removed_quiet(monkeypatch, quiet):
    if quiet is None:
        monkeypatch.delenv("XTIM_QUIET", raising=False)
    else:
        monkeypatch.setenv("XTIM_QUIET", quiet)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        D._emit_one("selfcheck_window", "", 2000, 0, "", 0.0)
    got_w = [w for w in rec if issubclass(w.category, XtimVerificationWarning)]
    assert len(got_w) == 1, (quiet, got_w)


@pytest.mark.parametrize("kind", ["disk_load", "disk_save"])
def test_disk_events_are_unknown_kinds(kind, caplog):
    with caplog.at_level(logging.INFO, logger="xtim.twirl"), pytest.raises(RuntimeError, match="unknown twirl"):
        D._emit_one(kind, "SAVED", 7, 0, "/tmp/x.twpl", 0.0)
