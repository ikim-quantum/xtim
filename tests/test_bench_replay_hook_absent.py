"""The fast-path replay micro-gate's hook (design §5.0) is BENCH-ONLY: it is compiled only with
-DXTIM_BENCH_REPLAY (``python setup.py build_ext -D XTIM_BENCH_REPLAY``) and must never reach the shipped
extension. Pinned here: the shipped TwirlSampler exposes none of the ``_bench_*`` methods. (The zero-cost
proof itself — macro-off .so byte-identical to the pre-hook tree — is in the bench harness report.)"""
import xtim._xtim as _x


def test_shipped_extension_has_no_bench_replay_hook():
    names = [n for n in dir(_x.TwirlSampler) if n.startswith("_bench")]
    assert names == [], names
