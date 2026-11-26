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
# RE-PINNED 2026-11-19 (fp-contract: -ffp-contract=off; the old pins were a -march=native build's FMA-fused values,
# which a portable build never matched; previous: git history)
CTOR_DIGEST_PINS = {
    "examples/cultivation_d3_rate.stim": "078e231214da570ac0916ec8036db71156a287781b24e48bf02a3ec40a31355c",
    "tests/data/adaptq_born5_dec_enum_producer.stim": "e60cb6085399e13a18ac163ebfb3f4add841b4d043fa5b8ae4201b580593663e",
    "tests/data/adaptq_qrm_magic_producer.stim": "0cd260ab4691ef431cfb7e5bed2fd7be6d6c9e2e62e54659a0872112c560fe57",
    "tests/data/ch_cultivation.stim": "4f27b85a5e50e1b03dfd39f857fcfa98bd89f3a92bd4e9aedebca02939ad8adb",
    "tests/data/code_switching_rate.stim": "7931708c231daf92f2aa381ad8cb3c507ad0c135b774f6e7f5822793f35e7f6f",
    "tests/data/cube_ccz.stim": "cf093c002ae2420a119f616da3cb9932e65a13fc154c03663fbd925d3c14ef18",
    "tests/data/cultivation_d3_faithful.stim": "09c8ac72824a64e2747dcc79dc352f0ae897824a018bc46c730bb052655e00b9",
    "tests/data/cultivation_d5.stim": "04bf6c4cf3e8ba388fb8489da0d9cab7627fdc00fafe56891998741fcc175efd",
    "tests/data/distillation_15_1_3.stim": "cad26d73572841b1ed6b5ba792db3e22a67b8be59f3953d0180ace07be0801f0",
    "tests/data/ja_seed23.stim": "bb0bed2ae1f71c5ce04d60e610b0f4aa766620105f87afc8490483334a1b7ae3",
    "tests/data/magna_color5_fid.stim": "789819018f917a53c4ca942e18ad45c471909c86456601c8ac1fe0fb5f0a6f56",
    "tests/data/magna_rp2_fid.stim": "031aed2883692e740d7e44e73fec689e76e525e02cbb2ea09f5f3d8de042d2c0",
    "tests/data/miniature_oracle.stim": "917f7469b28cef421b3ac68c54978425af225bb8048168b33a2e5da244817105",
    "tests/data/sign_live_bytes_c40022.stim": "8ae472e4aaec76cca9c0a6276c566d58ebe8689bf6cabf2b3638ac03c5389622",
}


@pytest.mark.parametrize("path", sorted(CTOR_DIGEST_PINS), ids=lambda p: os.path.basename(p))
def test_ctor_digest_pinned_to_l5(path):
    s = TA.compile_family_sampler(open(os.path.join(ROOT, path)).read(), p_factor=1.0, selfcheck=0)
    assert s._ctor_digest() == CTOR_DIGEST_PINS[path]
