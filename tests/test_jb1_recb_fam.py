"""J-b1 P6: record-space B inside (D) on the FAM path (plan §2.1 P6).

Every FAM (D) record table is built by build_record_table_recB (J-b2 J7b: the σ dual basis and its G-B1 comparator
are deleted; the law is gated against independent oracles in test_s2b_record_table). Gates, on every plan reached by
the shots, recording runs included, on the S2b fixtures, the speed set, examples/ and tests/data (plus every .stim
under $JB1_RECB_CORPUS, a ':'-separated list of directories: the scratch core / big / fuzz corpora):
  OP     tables_recb > 0.
  ONE LAW (fixplan-coldstart (A)) every table is built from the payload's parked core law: recb_core_rebuilds = 0,
         tables_recb_parked = tables_recb (MUT-1A-DOUBLE -> red).
"""
import glob
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _ja_fixtures as F  # noqa: E402
from test_ja_record_space_b import FX_S2B, SPEED  # noqa: E402

from _private import PRIVATE_CORPUS  # noqa: E402
from xtim import _xtim_testapi as TA  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORPUS = dict(FX_S2B)
CORPUS.update({"speed." + k: v for k, v in SPEED.items()})
for f in sorted(glob.glob(os.path.join(ROOT, "examples", "*.stim")) + glob.glob(os.path.join(ROOT, "tests", "data", "*.stim")) + PRIVATE_CORPUS):
    CORPUS[os.path.relpath(f, ROOT)] = open(f).read()
for d in filter(None, os.environ.get("JB1_RECB_CORPUS", "").split(":")):
    for f in sorted(glob.glob(os.path.join(d, "**", "*.stim"), recursive=True)):
        CORPUS["ext:" + os.path.relpath(f, d)] = open(f).read()


def fam_any(text):
    from xtim.errors import XtimCapacityError
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TA.compile_family_sampler(text, selfcheck=0)
    except ValueError:
        return None
    except XtimCapacityError:                       # a corpus circuit above the χ cap: refused by name
        return None


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_recb_every_fam_d_build(name):
    s = fam_any(CORPUS[name])
    if s is None:
        pytest.skip("refused by name")
    s.sample(2000, seed=11)
    try:
        s.sample_records(500, seed=13)              # the recording run's (D) builds too
    except (ValueError, RuntimeError):
        pass                                        # a configuration that refuses a recording run by name
    st = s.route_stats()
    assert st["tables_recb"] > 0, st
    # fixplan-coldstart (A): ONE law per FAM plan — every record-space-B table is built from the payload's parked
    # core (MUT-1A-DOUBLE -> red here), and the payload builds no σ law (laws_payload counts the core builds)
    assert st["recb_core_rebuilds"] == 0, st
    assert st["tables_recb_parked"] == st["tables_recb"], st
    assert st["laws_payload"] >= st["tables_recb"], st
