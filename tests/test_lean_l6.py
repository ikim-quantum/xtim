"""lean L6 (2026-11-05): compile time. The bare-state frame build, the propagation table and the sampler constructor
are rebuilt on sparse / transposed / event-driven data structures without changing a byte of what they build:
`_ctor_digest` (sha256 over the frame rows U.Zrow / U.Xrow with phases, eps, free, the branch entries, the certified
generators, every propagation-table atom in its dense form with its reject data, the alternatives and their atoms, the
records) is pinned here on public corpus circuits to the values captured on the L5 tip (f112429, lean/L6/
ctor_digests_tip.txt) — a moved frame row, table atom or record moves the pin. The mechanisms' own oracles are the C++
test cpp/tests/test_tableau_lean.cpp (left_batch == the row-wise left_* sequence, reset_dual == 2n dual_images,
from_generators == the dense synthesis, the lazy anchor == the eager one), tests/test_compile_speed.py (the table's walk
== propagate_atom atom for atom) and tests/test_lean_l1.py (the sparse table — the Clifford-suffix sweep, the
position-bitset walk and its support-local CZ layer — == the dense walk's table, atom for atom).
"""
import os

import pytest

from xtim import _xtim_testapi as TA

pytestmark = pytest.mark.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# path (relative to the repository root) -> sha256 of _ctor_digest at p_factor 1, selfcheck 0, on the L5 tip f112429;
# RE-PINNED 2026-11-06 (lean D1-S2, owner-signed `_ctor_digest` move: the old products + the rotation model / records +
# alt_xrow / xrow_x_ / xrow_z_, alt_off as size_t — every circuit moves, Clifford ones included: a digest, not a byte)
# RE-PINNED 2026-11-06 (lean D1-S3 step 2, owner-signed deletions: the digest no longer hashes the deleted products —
# the diagonal substrate, its masks, the L2 records, the extras' per-alt rows and the pair universe; every circuit moves)
CTOR_DIGEST_PINS = {
    "examples/cultivation_d3_rate.stim": "078e231214da570ac0916ec8036db71156a287781b24e48bf02a3ec40a31355c",
    "tests/data/adaptq_born5_dec_enum_producer.stim": "eaca4065e4860e98258bbb062f4fb10ca6f5604dc8ca9ac0d3346d9177f2c2f1",
    "tests/data/adaptq_qrm_magic_producer.stim": "f81d63fbe8c23452302fd3acc656573d8798f33a163d37826fe4e8c97c3b7bd9",
    "tests/data/ch_cultivation.stim": "9c6e61a03870c3090cf21f5659f0575d75a63bd4246a877b28ca1184db7d4a20",
    "tests/data/code_switching_rate.stim": "d5f16b027fbcd3e08467376bb06a1d4551b3a2a9ed334c15d150329c82a0691f",
    "tests/data/cube_ccz.stim": "b48424533a3957ed44fc23fb9cd3aaefa6dfe420fe591b584bc026fa9c0494f8",
    "tests/data/cultivation_d3_faithful.stim": "208ac6c51f70b6e927097d1c4f7800eb9d5ca69dcbcbcfaea126b16cf7456710",
    "tests/data/cultivation_d5.stim": "52b3c29d8e97ac0fcb36358386ef1cf72e14c9cfb67c75c6d87b7aef79892937",
    "tests/data/distillation_15_1_3.stim": "2193125efe98413fbdf1462c0b9d2d54ea6bc08cef179f042ff9a0ce2ce776fc",
    "tests/data/ja_seed23.stim": "26cd7ca014271aafe701f7f87c759d2b50b6c8a05991e82d82cb59a8f5e47d6f",
    "tests/data/magna_color5_fid.stim": "ff9534c7af44a3560945738033366be73ad3e6d303f9de1a162ff0ddd1aa8677",
    "tests/data/magna_rp2_fid.stim": "a53743b039bdc621a320c35a1ee1d6e6f563183c8f2cfdfc06057f2b09000c54",
    "tests/data/miniature_oracle.stim": "04c213246b9a0728d2a020c122ec4a591cbaf029aa9641c82224d48be310f7c1",
    "tests/data/sign_live_bytes_c40022.stim": "a5fe2a9e5b402a19229fb122b927443a34d810970cdcf7be140cd575e07e3595",
}


@pytest.mark.parametrize("path", sorted(CTOR_DIGEST_PINS), ids=lambda p: os.path.basename(p))
def test_ctor_digest_pinned_to_l5(path):
    s = TA.compile_family_sampler(open(os.path.join(ROOT, path)).read(), p_factor=1.0, selfcheck=0)
    assert s._ctor_digest() == CTOR_DIGEST_PINS[path]
