"""Host-side tables of the GPU sampler: integer thresholds for the geometric skips (design spec section 5.4) and
the kernels' buffers (``pack``, ``capacities``; xtim/gpu/xdem_kernels.h).

GPU kernels use no floating point, so the skip to the next firing slot is sampled from 64-bit
integer draws. For a geometric variable ``skip`` with success probability ``p`` (``skip = k``
means k non-firing slots precede the next firing) and ``q = 1 - p``, the binary digits of
``skip`` are independent: bit ``j`` is 1 with probability ``q^(2^j) / (1 + q^(2^j))``. The
thresholds are computed exactly on the host, in integer arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from math import floor
from typing import NamedTuple

import numpy as np

_FRAC_BITS = 256  # fixed-point precision of the repeated squaring
_ONE = 1 << _FRAC_BITS
_U64_MAX = (1 << 64) - 1


def geo_thresholds(p: float, slots: int) -> tuple[int, np.ndarray]:
    """Thresholds for sampling ``skip ~ Geometric(p)`` truncated at ``2^J`` slots.

    Returns ``(J, thr)`` with ``J = max(1, ceil(log2(slots)))`` and ``thr`` a ``uint64[J + 1]``.
    A draw ``x < thr[0]`` means ``skip >= 2^J`` (the chain ends; ``thr[0] = floor(q^(2^J) * 2^64)``);
    a draw ``x < thr[1 + j]`` sets bit ``j`` of the skip
    (``thr[1 + j] = floor(q^(2^j) / (1 + q^(2^j)) * 2^64)``). ``q = 1 - p`` is taken as the exact
    rational value of the double ``p``; powers are formed by repeated squaring rounded down to
    256 fractional bits. A threshold that would be 2^64 clamps to 2^64 - 1.

    Raises ``ValueError`` unless ``0 < p < 1``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must satisfy 0 < p < 1, got {p!r}")
    if slots < 1:
        raise ValueError(f"slots must be >= 1, got {slots!r}")
    J = max(1, (slots - 1).bit_length())   # ceil(log2(slots)) in integers (a float log2 rounds near 2^49 + 1)
    power = floor((1 - Fraction(p)) * _ONE)  # q^(2^0) in fixed point
    thr = [0] * (J + 1)
    for j in range(J):
        thr[1 + j] = min((power << 64) // (_ONE + power), _U64_MAX)
        power = (power * power) >> _FRAC_BITS  # q^(2^(j+1))
    thr[0] = min(power >> (_FRAC_BITS - 64), _U64_MAX)
    return J, np.array(thr, dtype=np.uint64)


_LOW56 = (1 << 56) - 1


def geo_skip_py(thr: Sequence[int], words: Sequence[int], tie: Callable[[int, int], int]) -> int | None:
    """Reference of the kernel's skip (xk_skip): ``None`` when the segment ends, else the skip ``k``.

    A skip is ``J + 1 = len(thr)`` tests: test 0 the end test (true: the segment ends), test ``1 + j`` bit ``j`` of the
    skip; test ``i`` is true with probability exactly ``thr[i] / 2^64``. ``words``: the pass's ``ceil(len(thr) / 8)``
    unsigned 64-bit draws (all of them drawn, whatever the end test says); tests ``8 q .. 8 q + 7`` share word ``q``,
    test ``8 q + i`` comparing byte ``i`` of it (bits ``8 i .. 8 i + 7``) with the top byte of its threshold (below:
    true; above: false). The ``j``-th tie on that byte (ascending ``i``, ``j`` from 0) is decided by ``tie(q, j)``:
    true iff its top 56 bits are below the threshold's low 56 bits.
    """
    nt = len(thr)
    lt_all = 0
    for q in range((nt + 7) // 8):
        x = int(words[q])
        j = 0
        for i in range(min(8, nt - 8 * q)):
            xb, tb = (x >> (8 * i)) & 0xFF, int(thr[8 * q + i]) >> 56
            if xb < tb or (xb == tb and (int(tie(q, j)) >> 8) < (int(thr[8 * q + i]) & _LOW56)):
                lt_all |= 1 << (8 * q + i)
            j += xb == tb
    if lt_all & 1:
        return None
    return lt_all >> 1


def pass_words(nt: int) -> int:
    """the words a pass of nt tests draws (ceil(nt / 8))"""
    return (nt + 7) // 8


class WalkPlan(NamedTuple):
    """The walk's work for blocks of ``block`` shots (xdem_kernels.h, K1; ``walk_plan``): every block has the same
    segments and items. ``thr``: the threshold rows (``geo_thresholds`` of a chain's p for a segment length: J + 1
    tests each); ``seg`` uint32[nseg, SEG_WORDS]: per segment its row, its tests nt (0: p = 1, every slot fires), the
    chain's first member m0, its first slot t0 and its slots L (slot t of a chain: member m0 + t // block, shot t %
    block); ``item_beg`` uint32[nitems + 1]: item i walks segments item_beg[i] .. item_beg[i + 1] - 1 in order."""
    block: int
    thr: list
    seg: np.ndarray
    item_beg: np.ndarray

    @property
    def J(self) -> int:
        """the largest J of a threshold row (0: none)"""
        return max((len(x) - 1 for x in self.thr), default=0)

    @property
    def nitems(self) -> int:
        return len(self.item_beg) - 1


SEG_WORDS = 5   # XK_SEG_WORDS
# the walk's segments: about WALK_FIRES expected fires each (a chain with fewer per block is one segment); segments
# expected to cost less than WALK_ITEM_COST words (draw words, an end pass included) share an item with their
# neighbours until it reaches that cost
WALK_FIRES = 8.0
WALK_ITEM_COST = 1.0


def walk_plan(tables: dict, block: int, *, fires: float | None = None, item_cost: float | None = None) -> WalkPlan:
    """The walk's segments and items for blocks of ``block`` shots (exact integer thresholds: costly for many chains,
    so a caller packing many batches computes it once and passes it to ``capacities`` and ``static_tables``).

    Chain c (M members, probability p) has M * block slots per block, cut into ``max(1, round(M block p / fires))``
    segments of ceil(slots / pieces) slots (the last one shorter), each with the thresholds of its own length (a row
    per distinct (chain, J)). A segment's expected cost in draw words is (fires + 1) * ceil(nt / 8) (its passes); the
    segments of at least ``item_cost`` are items of their own, the others are dealt into items in chain order until an
    item reaches ``item_cost``. Items are ordered by cost, largest first, so the threads of a SIMD group get similar
    work. Raises ``ValueError`` for a chain of 2^32 slots or more."""
    fires = WALK_FIRES if fires is None else float(fires)
    item_cost = WALK_ITEM_COST if item_cost is None else float(item_cost)
    if block < 1:
        raise ValueError("walk_plan: block must be >= 1")
    off = np.asarray(tables["chain_off"], np.int64)
    cp = tables["chain_p"]
    thr, rowix = [], {}
    segs = []   # (cost, row, nt, m0, t0, L)
    for c in range(len(cp)):
        p = float(cp[c])
        m0, M = int(off[c]), int(off[c + 1] - off[c])
        N = M * int(block)
        if N == 0 or p == 0.0:
            continue
        if N >= 1 << 32:
            raise ValueError(f"walk_plan: chain {c} has {N} slots per block (at most 2^32 - 1)")
        f = N * p
        pieces = max(1, min(N, round(f / fires)))
        L = -(-N // pieces)
        for t0 in range(0, N, L):
            n = min(L, N - t0)
            if p == 1.0:
                segs.append((n * 0.25, 0, 0, m0, t0, n))
                continue
            J = max(1, (n - 1).bit_length())
            if (c, J) not in rowix:
                rowix[(c, J)] = len(thr)
                thr.append(geo_thresholds(p, n)[1])
            nt = J + 1
            segs.append(((n * p + 1) * -(-nt // 8), rowix[(c, J)], nt, m0, t0, n))
    items = []   # lists of segments
    small = []
    acc = 0.0
    for sg in segs:
        if sg[0] >= item_cost:
            items.append([sg])
            continue
        small.append(sg)
        acc += sg[0]
        if acc >= item_cost:
            items.append(small)
            small, acc = [], 0.0
    if small:
        items.append(small)
    items.sort(key=lambda it: -sum(x[0] for x in it))   # stable: equal costs keep chain order
    seg = np.array([x[1:] for it in items for x in it], np.uint32).reshape(-1, SEG_WORDS)
    beg = np.cumsum([0] + [len(it) for it in items]).astype(np.uint32)
    return WalkPlan(int(block), thr, seg, beg)


def item_pos(seed: int, block: int, item: int) -> int:
    """the stream position of item ``item`` of global block ``block`` (xk_item_pos): its draws are mix(pos + k gamma),
    word n of the item at k = 2 n + 1, the j-th tie of word n at k = 2 (8 n + j + 1)"""
    bkey = mix(stream_key(seed, K_TAG_GPU_NOISE) ^ mix(block))
    return mix((bkey + (item + 1) * K_ITEM_GAMMA) & _U64_MAX)


def walk_block_py(tables: dict, plan: WalkPlan, block: int, *, seed: int, first_block: int = 0,
                  draws: list | None = None) -> list[tuple[int, int]]:
    """Reference of the walk's draws (xdem_kernels.h, K1 phase 1) for one block (local index ``block`` of a batch
    starting at global block ``first_block``): the fired ``(member, shot)`` slots, item by item, each in walk order
    (shot = index within the block; shots past the batch's end included). ``plan``: ``walk_plan(tables, B)``.
    ``draws``: when given, (item, words drawn, tie draws) is appended per item."""
    B = plan.block
    out = []
    for item in range(plan.nitems):
        pos = item_pos(seed, first_block + block, item)
        k = 0
        nties = 0

        def draw(i):
            return mix((pos + i * K_GAMMA) & _U64_MAX)

        def tie(q, j, base):
            nonlocal nties
            nties += 1
            return draw(2 * (8 * (base + q) + j + 1))

        for row, nt, m0, t0, L in plan.seg[int(plan.item_beg[item]):int(plan.item_beg[item + 1])].tolist():
            t = 0
            while t < L:
                if nt:
                    w = pass_words(nt)
                    words = [draw(2 * (k + q) + 1) for q in range(w)]
                    g = geo_skip_py(plan.thr[row], words, lambda q, j, base=k: tie(q, j, base))
                    k += w
                    if g is None:
                        break
                    t += g
                    if t >= L:
                        break
                out.append((m0 + (t0 + t) // B, (t0 + t) % B))
                t += 1
        if draws is not None:
            draws.append((item, k, nties))
    return out


# ------------------------------------------------------------------------------------------------------------------
# Kernel buffers (xtim/gpu/xdem_kernels.h). The dims[] indices and mode bits below are the header's #defines; the
# header is their single source of truth (tests/test_gpu_kernels.py checks that the two agree).

XK_DIM_W = 0
XK_DIM_WD = 1
XK_DIM_WO = 2
XK_DIM_WF = 3
XK_DIM_WA = 4
XK_DIM_RHO = 5
XK_DIM_RHOL = 6
XK_DIM_R = 7
XK_DIM_B = 8
XK_DIM_E = 9
XK_DIM_NCHAINS = 10
XK_DIM_NSHOTS = 11
XK_DIM_NBLOCKS = 12
XK_DIM_MODE = 13
XK_DIM_SPAN_RANK = 14
XK_DIM_SPAN_WORDS = 15
XK_DIM_OFF_CHAIN_THR = 16
XK_DIM_OFF_ENT_MASK = 17
XK_DIM_OFF_XA = 18
XK_DIM_OFF_TC = 19
XK_DIM_OFF_KC = 20
XK_DIM_OFF_BETA = 21
XK_DIM_OFF_BETAO = 22
XK_DIM_OFF_BLO = 23
XK_DIM_OFF_CHAIN_OFF = 24
XK_DIM_OFF_ITEM = 25  # the walk's items (walk_plan's item_beg) and segments (its seg), tab32
XK_DIM_OFF_SEG = 26
XK_DIM_OFF_MEM_BEG = 27
XK_DIM_OFF_MEM_CNT = 28
XK_DIM_OFF_MEM_XO = 29
XK_DIM_OFF_ENT_WORD = 30
XK_DIM_OFF_BB = 31
XK_DIM_OFF_PL = 32
XK_DIM_THR_STRIDE = 33
XK_DIM_OFF_SPAN_WMASK = 34
XK_DIM_OFF_SPAN_BASIS = 35
XK_DIM_OFF_SPAN_PIV_BIT = 36
XK_DIM_OFF_SPAN_WIDX = 37
XK_DIM_OFF_SPAN_PIV_WORD = 38
XK_DIM_OFF_CHAIN_THR8 = 39
XK_DIM_THR8_STRIDE = 40
XK_DIM_CW = 41  # coordinate words of the multi-word law (0: the one-word law)
XK_DIM_OFF_KCW = 42
XK_DIM_OFF_GCW = 43
XK_DIM_OFF_GAMMA = 44
XK_DIM_OFF_GBW = 45
XK_DIM_DIMC = 46  # the largest dim C the counters hold, max(64, rhoL): the hist layout (hist_layout)
XK_DIM_OFF_XLAW = 47  # the single-line law table (law_table): xlaw in tab32, its entries lawt in tab64
XK_DIM_OFF_LAWT = 48
XK_DIM_LAWT = 49  # 1: the table is in the arenas
XK_DIM_NBD = 50  # K4's output bytes per shot: detectors ceil(D / 8), observables ceil(O / 8)
XK_DIM_NBO = 51
XK_DIM_OFF_STAB = 52  # the zero decoder's observable table (obs_table, one-word law): stab and rtab (tab64)
XK_DIM_OFF_RTAB = 53
XK_DIM_RTAB = 54  # 1: rtab is in the arena (some observable has a rest outside beta_L's observable parts)
XK_DIM_ST = 55  # 1: the structured one-word law (xtim.gpu.structure), its sizes and its table sct (tab64)
XK_DIM_ST_NG = 56
XK_DIM_ST_NT = 57
XK_DIM_ST_NN = 58
XK_DIM_OFF_ST = 59
XK_DIM_RECSH = 60  # the walk's packed records: the shot's shift (rec_shift; 0: unpacked)
XK_DIM_ST_W = 61  # the structured law on a multi-word file (xk_law_struct_mw): its slot words (0: one-word)
XK_DIM_NITEMS = 62  # the walk's items per block (walk_plan)
XK_DIM_WALK = 63  # the walk's options (XK_WALK_* bits)
XK_DIM_COUNT = 64

XK_MODE_POSTSELECT = 1  # K2 writes flag[] (needs pack's span tables)
XK_MODE_COIN_IN = 2  # coin words from the coin_in buffer instead of the coin stream
XK_MODE_GIVEN = 4  # no walk: rows and the per-shot extended lists are given
XK_MODE_DEFER = 8  # K2 queues the shots whose law it must compute (lawq); K2b ("resolve_law") computes them
XK_MODE_EMIT = 16  # K1 empties cmiss for K4 ("emit")
XK_MODE_NOKEY = 32  # the estimate: K2 writes no C key (one-word law), dim C and the zero decoder's verdict only
# sample without the status (with XK_MODE_NOKEY and XK_MODE_EMIT): no law writes a C key or needs the zero decoder's
# verdict, K4 writes rows but no C id (cmiss: the error codes only)
XK_MODE_NOSTATUS = 64
XK_LAW_SPREAD = 4  # K2b runs law_threads(nshots) = ceil(nshots / 4) threads; thread t takes entries t, t + G, ...


def law_threads(nshots: int) -> int:
    """K2b's ("resolve_law") thread count for a batch of nshots shots (xk_law_threads)"""
    return -(-int(nshots) // XK_LAW_SPREAD)

XK_OVERFLOW_PAIRS = 1  # overflow[b]: more than E extended lines in block b (relaunch with a larger E)
XK_OVERFLOW_CAPACITY = 2  # overflow[b]: a size exceeds a capacity macro (nothing computed)
XK_CDIM_CAPACITY = 254  # cdim[s]: a size exceeds a capacity macro (the row left as it was)
XK_CDIM_INTERNAL = 255  # cdim[s]: post_fast's internal check failed (the CPU raises)

XK_FLAG_SURVIVE = 1  # flag[s]: the shot survives (every shot without postselection; or its cdim is an error code)
XK_FLAG_EARLY = 2  # flag[s]: rejected by the span test before the law (never with XK_FLAG_SURVIVE)
XK_FLAG_PENDING = 8  # flag[s]: kept by the walk's screen (XK_WALK_SCREEN), for K2 to finish
XK_FLAG_ZERO_PASS = 4  # flag[s]: a survivor without an error code that the zero decoder passes

# The hist layout at dims[XK_DIM_DIMC] = 64 (every one-word file; ``hist_layout(dimc)`` in general: the error slot,
# the stride and hist_total's slots move up by dimc - 64)
XK_HIST_SURVIVE = 0  # hist[b * XK_HIST_STRIDE + slot] (K3): survivors without an error code
XK_HIST_PASS = 1  # + k: those with XK_FLAG_ZERO_PASS and dim C = k, k = 0..64 (0..dimc)
XK_HIST_ERROR = 66  # survivors whose cdim is an error code (counted in no other slot)
XK_HIST_STRIDE = 67  # u32 counters per block
XK_HIST_TOTAL_PAIRS = 67  # hist_total[] (K5, u64): slots < XK_HIST_STRIDE the batch sums, then the blocks with
XK_HIST_TOTAL_CAPACITY = 68  # XK_OVERFLOW_PAIRS, then those with XK_OVERFLOW_CAPACITY
XK_HIST_TOTAL_COUNT = 69  # u64 slots of hist_total
XK_HIST_GROUPS = 256  # groups of blocks K5 sums into hist_group (u64[XK_HIST_GROUPS * XK_HIST_TOTAL_COUNT])
XK_MAX_WA = 64  # the largest XK_CAP_WA (axis words: at most 4096 axes); it sizes thread-local arrays
XK_MAX_RHO = 64  # the largest XK_CAP_RHO (the one-word law's rho <= 64)
XK_MAX_DIMC = 253  # the largest XK_CAP_DIMC: dim C is a u8 below the error codes (254, 255); it sizes K3's counters
XK_MAX_ST_NC = 8  # the structured law's capacities' bounds (xtim.gpu.structure.ST_MAX_*): they size thread-local arrays
XK_MAX_ST_NG = 8
XK_MAX_ST_T = 32
XK_MAX_ST_N = 16
XK_MAX_ST_W = 4
XK_MAX_BLOCK = 1024  # the walk's largest block of shots (its per-shot counters live in group memory)
XK_MAX_SHM = 8192  # the walk's largest group memory, u32 words (32 KiB: Metal's threadgroup memory)
XK_WALK_TGROWS = 1  # dims[XK_DIM_WALK]: the block's rows in group memory (else XORed in device memory)
XK_EMIT_RUN = 64  # K7's offsets: one per run of this many shots (boff)
XK_WALK_SCREEN = 2  # dims[XK_DIM_WALK]: the postselection screen in the walk (else in K2)
# the postselection screen moves into the walk (XK_WALK_SCREEN) for a file firing at least this many lines per shot
# on average (M3 Max, K1 + K2 GPU ns per post-selected estimated shot, screen in the walk vs in K2: merlin_bh70 2.48 vs
# 2.86, merlin_bh50 1.59 vs 1.80, merlin_bh36 1.36 vs 1.32; below it cultivation_d5 0.90 vs 0.87, magna_rp2-d5 1.40 vs
# 1.34, merlin_bh12 0.64 vs 0.61)
WALK_SCREEN_FIRED = 3.0


def shm_words(block: int, W: int, walk: int) -> int:
    """the walk's group memory in u32 words (xdem_kernels.h, xk_shm_need)"""
    return 6 + 3 * block + block // 8 + (2 * block * W if walk & XK_WALK_TGROWS else 0)


def walk_flags(tables: dict, block: int) -> int:
    """the walk's default options (dims[XK_DIM_WALK]): rows XORed in device memory (group memory was slower on every
    file measured: it holds fewer groups per core), the postselection screen in the walk for a file firing at least
    WALK_SCREEN_FIRED lines per shot"""
    return XK_WALK_SCREEN if float(tables.get("mean_fired", 0.0)) >= WALK_SCREEN_FIRED else 0


class HistLayout(NamedTuple):
    """The counters' layout for dims[XK_DIM_DIMC] = dimc (xdem_kernels.h, hist): per block ``stride`` u32 (SURVIVE,
    PASS + k for k = 0..dimc, ``error``); hist_total ``count`` u64 (the ``stride`` sums, ``pairs``, ``capacity``)."""
    dimc: int
    error: int
    stride: int
    pairs: int
    capacity: int
    count: int


def hist_layout(dimc: int = 64) -> HistLayout:
    """the hist layout of dims[XK_DIM_DIMC] = dimc (64: the XK_HIST_* constants)"""
    return HistLayout(dimc, dimc + 2, dimc + 3, dimc + 3, dimc + 4, dimc + 5)


def dimc_of(tables: dict) -> int:
    """dims[XK_DIM_DIMC] of these tables: max(64, rhoL)"""
    return max(64, int(tables["rhoL"]))


def coord_words(tables: dict) -> int:
    """dims[XK_DIM_CW]: the multi-word law's words per coordinate vector (coord_words >= 2 with axes), else 0"""
    cw = int(tables["coord_words"])
    return cw if cw >= 2 and int(tables["A"]) else 0


def key_words(tables: dict) -> int:
    """u64 words of a shot's status key (ckey): max(rhoL, 1) vectors of max(CW, 1) words"""
    return max(int(tables["rhoL"]), 1) * max(coord_words(tables), 1)


def mw_words(tables: dict) -> int:
    """the multi-word law's scratch words per shot (xk_mw_words: 8 cw + 4 rho cw + 4 rho), 0 for a one-word file"""
    cw, rho = coord_words(tables), int(tables["rho"])
    return 8 * cw + 4 * rho * cw + 4 * rho if cw else 0

# Stream tags of the GPU sampler (distinct from the CPU sampler's, so no GPU stream coincides with a CPU stream).
K_TAG_GPU_NOISE = 0x6770_4E6F_6973_6501
K_TAG_GPU_COIN = 0x6770_436F_696E_0002
K_GAMMA = 0x9E37_79B9_7F4A_7C15
K_ITEM_GAMMA = 0xD1B5_4A32_D192_ED03  # XK_ITEM_GAMMA

# (table, dims index of its offset) in arena order
_TAB64 = (("ent_mask", XK_DIM_OFF_ENT_MASK), ("xa", XK_DIM_OFF_XA), ("Tc", XK_DIM_OFF_TC),
          ("Kc", XK_DIM_OFF_KC), ("beta", XK_DIM_OFF_BETA), ("betaO", XK_DIM_OFF_BETAO), ("bLo", XK_DIM_OFF_BLO))
_TAB32 = (("chain_off", XK_DIM_OFF_CHAIN_OFF), ("mem_beg", XK_DIM_OFF_MEM_BEG), ("mem_cnt", XK_DIM_OFF_MEM_CNT),
          ("mem_xo", XK_DIM_OFF_MEM_XO), ("ent_word", XK_DIM_OFF_ENT_WORD), ("bb", XK_DIM_OFF_BB),
          ("pL", XK_DIM_OFF_PL))
# the multi-word law's tables (empty for a one-word file)
_TABMW = (("KcW", XK_DIM_OFF_KCW), ("GcW", XK_DIM_OFF_GCW), ("gamma", XK_DIM_OFF_GAMMA), ("GbW", XK_DIM_OFF_GBW))
# the span test (XdemSampler.span_tables' keys), appended to the arenas
_SPAN64 = (("wmask", XK_DIM_OFF_SPAN_WMASK), ("basis", XK_DIM_OFF_SPAN_BASIS), ("piv_bit", XK_DIM_OFF_SPAN_PIV_BIT))
_SPAN32 = (("widx", XK_DIM_OFF_SPAN_WIDX), ("piv_word", XK_DIM_OFF_SPAN_PIV_WORD))


def mix(z: int) -> int:
    """krng::mix (SplitMix64's output finalizer) on a 64-bit integer."""
    z &= _U64_MAX
    z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & _U64_MAX
    z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & _U64_MAX
    return z ^ (z >> 31)


def stream_key(seed: int, tag: int) -> int:
    """krng::stream_key: ``mix(seed ^ mix(tag))``."""
    return mix((seed & _U64_MAX) ^ mix(tag))


def _check_fast(tables: dict) -> None:
    if tables["A"] and not tables["fast"] and int(tables["coord_words"]) < 2:
        raise ValueError("the GPU kernels support the CPU sampler's fast path only (one-word or multi-word "
                         f"coordinates, coord_words >= 1); got coord_words = {tables.get('coord_words')} (the general "
                         "path)")
    if int(tables["WA"]) > XK_MAX_WA:
        raise ValueError(f"the GPU kernels support at most {64 * XK_MAX_WA} axes ({XK_MAX_WA} words of thread-local "
                         f"memory per plane); got {tables['A']} axes ({tables['WA']} words)")
    if int(tables["rhoL"]) > XK_MAX_DIMC:
        raise ValueError(f"the GPU kernels count dim C up to {XK_MAX_DIMC} (a byte per shot); this file's status C "
                         f"can reach rhoL = {tables['rhoL']}")


# ------------------------------------------------------------------------------------------------------------------
# The single-line law table (xdem_kernels.h, xk_law_entry / xk_law_table)

LAW_TABLE_MAX_WORDS = 1 << 22  # a file whose table would hold more u64 words gets none (32 MiB)
# GpuBackend's default: the table and K2's queue only for a file with at most this many extended lines per shot on
# average (mean_ext). Past it nearly every shot's law is computed anyway (about (1 + m) e^-m of the shots fire at
# most one extended line), and the queue's second composition and the table's lookups cost more than they save
# (M3 Max, estimate: merlin bh36, mean_ext 3.4, 80.6 vs 82.6 ns per shot with / without both; bh50, mean_ext 5.2,
# 186 vs 164)
LAW_MAX_MEAN_EXT = 4.0


def _bits(x: int):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b


def line_law(tables: dict, odd: list[int]) -> tuple | None:
    """post_fast's objects for the odd axis set ``odd`` of a one-word file, as xk_law_fast computes them: ``(omega,
    pivots [(p, col_p, S[p])] ascending, key)`` (key: C's fully reduced echelon, pivot = highest bit, descending), or
    None where post_fast's internal checks fail (odd phase on W / W phase not linear)."""
    rho, rhoL = int(tables["rho"]), int(tables["rhoL"])
    Tc, Kc, bb = tables["Tc"], tables["Kc"], tables["bb"]
    col = [0] * rho
    for a in odd:
        k = int(Kc[a])
        for j in _bits(int(Tc[a])):
            col[j] ^= k
    S: dict[int, int] = {}   # the coin span's fully reduced echelon, pivot = lowest bit
    for i in _bits(_or(col)):
        g = sum(((col[j] >> i) & 1) << j for j in range(rho))
        for p, v in S.items():
            if g >> p & 1:
                g ^= v
        if g:
            tb = (g & -g).bit_length() - 1
            for p in S:
                if S[p] >> tb & 1:
                    S[p] ^= g
            S[tb] = g
    pm = sum(1 << p for p in S)
    fm = ((1 << rho) - 1) & ~pm
    omega = 0
    if fm:
        plo = phi = lin = 0
        Bq = [0] * rho
        for a in odd:
            tt = int(Tc[a])
            cq = tt
            for p in _bits(tt & pm):
                cq ^= S[p]
            cq &= fm
            phi ^= plo & cq
            plo ^= cq
            if bb[a]:
                lin ^= cq
            for q in _bits(cq):
                Bq[q] ^= cq
        if plo or any(Bq):
            return None
        omega = phi ^ lin
    H: dict[int, int] = {}   # C: the S[p] past rho - rhoL, fully reduced with pivot = highest bit
    for p in sorted(S):
        if p < rho - rhoL:
            continue
        v = S[p]
        for b, h in H.items():
            if v >> b & 1:
                v ^= h
        tb = v.bit_length() - 1
        for b in H:
            if H[b] >> tb & 1:
                H[b] ^= v
        H[tb] = v
    return omega, [(p, col[p], S[p]) for p in sorted(S)], [H[b] for b in sorted(H, reverse=True)]


def _or(xs):
    out = 0
    for x in xs:
        out |= x
    return out


def _records(tables: dict) -> list[int]:
    """the extended lines' record offsets into xa (distinct, ascending)"""
    return sorted({int(x) for x in np.asarray(tables["mem_xo"]) if int(x)})


def rec_shift(tables: dict, block: int) -> int:
    """the walk's packed records (xdem_kernels.h, XK_DIM_RECSH): K1 appends xo | s << shift (s < block the shot, xo
    the record offset into xa) when both fit in 32 bits, which lets sort_ext scatter in one pass; 0 otherwise (and for
    block 1). The order of the sorted lists is the same either way."""
    bits = (int(block) - 1).bit_length()
    sh = 32 - bits
    return sh if bits and len(np.asarray(tables["xa"])) <= (1 << sh) else 0


def law_pays(tables: dict) -> bool:
    """whether GpuBackend uses the single-line law table and K2's queue by default (``LAW_MAX_MEAN_EXT``)"""
    return float(tables["mean_ext"]) <= LAW_MAX_MEAN_EXT


def law_table(tables: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(xlaw uint32[len(xa)], lawt uint64[])``: the single-line law table of a one-word file with axes
    (xdem_kernels.h, xk_law_table: one entry per distinct nonzero line odd mask lo_l whose law passes post_fast's
    checks; xlaw[xo] the entry of the line at record offset xo, 0 for none). Empty arrays for a multi-word file, a
    file without axes, or a table past ``LAW_TABLE_MAX_WORDS`` words."""
    none = (np.zeros(0, np.uint32), np.zeros(0, np.uint64))
    if coord_words(tables) or not int(tables["A"]) or not tables["fast"]:
        return none
    xa = np.asarray(tables["xa"], np.uint64)
    WA = int(tables["WA"])
    xlaw = np.zeros(len(xa), np.uint32)
    words = [0]   # lawt[0]: a pad word, so that offset 0 means no entry
    at: dict[tuple, int] = {}
    for xo in _records(tables):
        h = int(xa[xo])
        lo = [0] * WA
        for k in range(h & 0xFFFF_FFFF):
            lo[int(xa[xo + 1 + 3 * k])] ^= int(xa[xo + 2 + 3 * k])
        lo = tuple(lo)
        if not any(lo):
            continue
        if lo not in at:
            law = line_law(tables, [64 * w + b for w in range(WA) for b in _bits(lo[w])])
            if law is None:   # the computed law reports post_fast's internal error
                at[lo] = 0
                continue
            omega, piv, key = law
            at[lo] = len(words)
            words += list(lo) + [omega, len(piv) | len(key) << 8]
            for _, cp, sp in piv:
                words += [cp, sp]
            words += key
            if len(words) > LAW_TABLE_MAX_WORDS:
                return none
        xlaw[xo] = at[lo]
    if len(words) > 0xFFFF_FFFF:
        return none
    return xlaw, np.array(words, np.uint64)


def obs_table(tables: dict, sw: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(stab uint64[64 WO], rtab uint64[64 WO * WO])``: the zero decoder's map from the observable row v to its
    beta_L coordinates s and its rest (xdem_kernels.h, xk_zero_passes: v reduced by beta_L's observable parts bLo,
    pivots pL, in build order), which is linear in v: stab[o] = s(e_o), rtab[o * WO ...] = rest(e_o) for every bit o
    of the WO observable words. A row's observable bits past O are always 0 (no flip mask or beta vector has one), so
    their entries are 0 and rtab is empty when the rest of every observable o < O is 0. Both empty for a multi-word
    file or rhoL = 0. ``sw`` >= 1 (a multi-word file's structured law, xk_law_struct_mw): stab[o] is sw words
    (stab uint64[64 WO * sw]; the multi-word tables have the same bLo and pL)."""
    none = (np.zeros(0, np.uint64), np.zeros(0, np.uint64))
    rhoL, WO = int(tables["rhoL"]), int(tables["WO"])
    if (coord_words(tables) and not sw) or not rhoL or not WO:
        return none
    sw = max(int(sw), 1)
    rD = int(tables["rho"]) - rhoL
    pL = [int(x) for x in np.asarray(tables["pL"])]
    bLo = np.asarray(tables["bLo"], np.uint64).reshape(rhoL, WO)
    bL = [sum(int(bLo[i, w]) << (64 * w) for w in range(WO)) for i in range(rhoL)]
    stab = np.zeros((64 * WO, sw), np.uint64)
    rest = np.zeros((64 * WO, WO), np.uint64)
    for o in range(int(tables["O"])):
        x, s = 1 << o, 0
        for i in range(rhoL):   # the kernel's order: bit pL[i] of v XOR the bLo_j applied so far
            if x >> pL[i] & 1:
                s |= 1 << (rD + i)
                x ^= bL[i]
        stab[o] = [(s >> (64 * w)) & _U64_MAX for w in range(sw)]
        rest[o] = [(x >> (64 * w)) & _U64_MAX for w in range(WO)]
    return stab.reshape(-1), (rest.reshape(-1) if rest.any() else np.zeros(0, np.uint64))


def capacities(tables: dict, block: int, pair_cap: int, *, plan: WalkPlan | None = None, struct=None,
               walk: int | None = None) -> dict[str, int]:
    """The kernel's capacity macros for these tables: the actual sizes (at least 1 each). ``plan``:
    ``walk_plan(tables, block)`` if already computed; ``walk``: the walk's options (None: ``walk_flags``), which set
    XK_CAP_SHM. A block above XK_MAX_BLOCK shots, or group memory above XK_MAX_SHM words, raises ``ValueError``.
    ``struct``: the structured law's ``structure.Plan`` (its sizes XK_CAP_ST_NC, _NG, _T, _N compile it in, and on a
    multi-word file XK_CAP_ST_W its slot words; None: not compiled, no such macro)."""
    _check_fast(tables)
    if not 1 <= int(block) <= XK_MAX_BLOCK:
        raise ValueError(f"the GPU walk takes blocks of 1 to {XK_MAX_BLOCK} shots, got {block}")
    plan = plan if plan is not None else walk_plan(tables, block)
    walk = walk_flags(tables, block) if walk is None else int(walk)
    shm = shm_words(int(block), int(tables["W"]), walk)
    if shm > XK_MAX_SHM:
        raise ValueError(f"the walk's group memory ({shm} words) exceeds {XK_MAX_SHM}: rows in device memory "
                         "(walk without XK_WALK_TGROWS) or a smaller block")
    cw = coord_words(tables)
    st = {} if struct is None else {
        "XK_CAP_ST_NC": struct.rD + 1, "XK_CAP_ST_NG": max(struct.nG, 1), "XK_CAP_ST_T": max(len(struct.records), 1),
        "XK_CAP_ST_N": max(len(struct.nonleaf), 1), **({"XK_CAP_ST_W": struct.sw} if struct.sw else {})}
    return {**st,
        "XK_CAP_W": max(int(tables["W"]), 1),
        "XK_CAP_WA": max(int(tables["WA"]), 1),
        "XK_CAP_RHO": 1 if cw else max(int(tables["rho"]), 1),   # the one-word law's arrays (unused when cw >= 2)
        "XK_CAP_E": max(int(pair_cap), 1),
        "XK_CAP_J": max(plan.J, 1),
        "XK_CAP_SHM": shm,
        "XK_CAP_CW": max(cw, 1),   # > 1 compiles the multi-word law in (it sizes nothing)
        "XK_CAP_DIMC": dimc_of(tables),
    }


def _u32(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.size and (int(a.min()) < 0 or int(a.max()) > 0xFFFF_FFFF):
        raise ValueError(f"table {name} does not fit in 32 bits")
    return a.astype(np.uint32)


def _span_parts(tables: dict, span: dict | None) -> tuple[int, int, list, list]:
    """(rank, words, tab64 parts, tab32 parts) of the span test ``span`` (``XdemSampler.span_tables``), checked
    against the tables; ``None``: empty"""
    if span is None:
        return (0, 0, [(np.zeros(0, np.uint64), i) for _, i in _SPAN64],
                [(np.zeros(0, np.uint32), i) for _, i in _SPAN32])
    a = {k: np.asarray(span[k]) for k, _ in _SPAN64 + _SPAN32}
    rank, nw = int(span["rank"]), len(a["widx"])
    if (rank < 0 or a["wmask"].shape != (nw,) or a["basis"].shape != (rank * nw,) or a["piv_word"].shape != (rank,)
            or a["piv_bit"].shape != (rank,)):
        raise ValueError("pack: span tables of inconsistent sizes")
    if (nw and int(a["widx"].max()) >= int(tables["WD"])) or (rank and int(a["piv_word"].max()) >= nw):
        raise ValueError("pack: span tables index outside the detector words")
    return (rank, nw, [(a[k].astype(np.uint64), i) for k, i in _SPAN64],
            [(_u32(a[k], "span_" + k), i) for k, i in _SPAN32])


class StaticTables(NamedTuple):
    """The read-only part of the kernels' buffers (``static_tables``): uploaded once per (tables, mask)."""
    tab64: np.ndarray  # the u64 arena
    tab32: np.ndarray  # the u32 arena
    dims: np.ndarray  # u32[XK_DIM_COUNT]: sizes, offsets, span rank / words; the run fields (E, nshots, nblocks,
    #                   mode) 0
    rmask: int
    span: bool  # whether the span tables of a mask are in the arenas


def static_tables(tables: dict, *, block: int, span: dict | None = None, plan: WalkPlan | None = None,
                  law=True, struct=None, walk: int | None = None) -> StaticTables:
    """The arenas ``tab64`` / ``tab32`` (every read-only table at its offset) and the ``dims`` fields that do not
    change from batch to batch. ``span``: the postselection span test (``XdemSampler.span_tables(flags)``), appended
    to the arenas (an empty mask: rank 0, no word). ``plan``: ``walk_plan(tables, block)`` if already computed;
    ``walk``: the walk's options (dims[XK_DIM_WALK]; None: ``walk_flags``). ``law``: the single-line law table,
    ``law_table(tables)`` (True: computed here; False: none, every law computed; the outputs are the same either
    way). ``struct``: the structured law's ``structure.Plan`` (the tables
    must be ``structure.renumber``'s): its table ``structure.st_table`` joins the arenas and dims[XK_DIM_ST] = 1 (a
    multi-word plan: dims[XK_DIM_ST_W] its slot words, the observable table of that width and the beta vectors
    ``renumber`` added)."""
    _check_fast(tables)
    if not 1 <= block <= XK_MAX_BLOCK:
        raise ValueError(f"pack: block must be in [1, {XK_MAX_BLOCK}]")
    t = tables
    plan = plan if plan is not None else walk_plan(t, block)
    if plan.block != block:
        raise ValueError("pack: the walk plan is for another block size")
    walk = walk_flags(t, block) if walk is None else int(walk)
    # the threshold rows at a common stride (the largest nt), and their top bytes
    nch = len(plan.thr)
    stride = plan.J + 1 if nch else 1
    chain_thr = np.zeros(nch * stride, np.uint64)
    for c, x in enumerate(plan.thr):
        chain_thr[c * stride:c * stride + len(x)] = x
    dims = np.zeros(XK_DIM_COUNT, np.uint32)
    rank, nw, s64, s32 = _span_parts(t, span)
    xlaw, lawt = law_table(t) if law is True else (law if law is not False else (np.zeros(0, np.uint32),
                                                                                 np.zeros(0, np.uint64)))
    # the thresholds' top bytes, eight tests to a word (byte i of word q: test 8 q + i)
    stride8 = (stride + 7) // 8
    top = np.zeros((nch, stride8 * 8), np.uint64)
    top[:, :stride] = chain_thr.reshape(nch, stride) >> np.uint64(56)
    chain_thr8 = (top.reshape(nch, stride8, 8) << (np.arange(8, dtype=np.uint64) * np.uint64(8))).sum(
        axis=2, dtype=np.uint64).reshape(-1)
    stab, rtab = obs_table(t, struct.sw if struct is not None else 0)
    sct = np.zeros(0, np.uint64)
    if struct is not None:
        from xtim.gpu.structure import st_table
        sct = st_table(struct)
    # the span tables last, so that the plain tables sit at the same offsets with and without a mask
    p64 = ([(chain_thr, XK_DIM_OFF_CHAIN_THR)] + [(np.asarray(t[k], np.uint64), i) for k, i in _TAB64]
           + [(chain_thr8, XK_DIM_OFF_CHAIN_THR8)]
           + [(np.asarray(t.get(k, ()), np.uint64), i) for k, i in _TABMW]
           + [(np.asarray(lawt, np.uint64), XK_DIM_OFF_LAWT), (stab, XK_DIM_OFF_STAB), (rtab, XK_DIM_OFF_RTAB),
              (sct, XK_DIM_OFF_ST)]
           + s64)
    p32 = ([(_u32(t[k], k), i) for k, i in _TAB32[:1]]
           + [(plan.item_beg, XK_DIM_OFF_ITEM), (plan.seg.reshape(-1), XK_DIM_OFF_SEG)]
           + [(_u32(t[k], k), i) for k, i in _TAB32[1:]]
           + [(np.asarray(xlaw, np.uint32), XK_DIM_OFF_XLAW)] + s32)
    arenas = []
    for parts in (p64, p32):
        at = 0
        for a, i in parts:
            dims[i] = at
            at += len(a)
        if at > 0xFFFF_FFFF:
            raise ValueError("pack: tables too large for 32-bit offsets")
        arenas.append(np.concatenate([a for a, _ in parts] + [np.zeros(1, parts[0][0].dtype)]))
    for k in ("W", "WD", "WO", "WF", "WA", "rho", "rhoL", "r"):
        dims[globals()["XK_DIM_" + k.upper()]] = int(t[k])
    dims[XK_DIM_B] = block
    dims[XK_DIM_NCHAINS] = len(t["chain_p"])
    dims[XK_DIM_NITEMS] = plan.nitems
    dims[XK_DIM_WALK] = walk
    dims[XK_DIM_THR_STRIDE] = stride
    dims[XK_DIM_THR8_STRIDE] = stride8
    dims[XK_DIM_SPAN_RANK] = rank
    dims[XK_DIM_SPAN_WORDS] = nw
    dims[XK_DIM_CW] = coord_words(t)
    dims[XK_DIM_DIMC] = dimc_of(t)
    dims[XK_DIM_LAWT] = 1 if len(lawt) else 0
    dims[XK_DIM_RTAB] = 1 if len(rtab) else 0
    if struct is not None:
        dims[XK_DIM_ST] = 1
        dims[XK_DIM_ST_NG] = struct.nG
        dims[XK_DIM_ST_NT] = len(struct.records)
        dims[XK_DIM_ST_NN] = len(struct.nonleaf)
        dims[XK_DIM_ST_W] = struct.sw
    dims[XK_DIM_RECSH] = rec_shift(t, block)
    dims[XK_DIM_NBD] = (int(t["D"]) + 7) // 8
    dims[XK_DIM_NBO] = (int(t["O"]) + 7) // 8
    return StaticTables(arenas[0], arenas[1], dims, int(t["rmask"]), span is not None)


def pack_batch(st: StaticTables, *, pair_cap: int, shots: int, seed: int, first_block: int = 0,
               mode: int = 0) -> dict[str, np.ndarray]:
    """The per-batch buffers ``dims`` (``st.dims`` with the run fields) and ``seed`` (the stream keys of ``seed``,
    ``first_block``: the global index of the batch's first block, ``rmask``)."""
    if mode & XK_MODE_POSTSELECT and not st.span:
        raise ValueError("pack: postselection (mode bit XK_MODE_POSTSELECT) needs the span tables (span=)")
    if mode & XK_MODE_NOSTATUS and not (mode & XK_MODE_NOKEY and mode & XK_MODE_EMIT):
        raise ValueError("pack: XK_MODE_NOSTATUS needs XK_MODE_NOKEY and XK_MODE_EMIT")
    if mode & XK_MODE_NOKEY and mode & XK_MODE_EMIT and not mode & XK_MODE_NOSTATUS:
        raise ValueError("pack: XK_MODE_NOKEY writes no C key, which XK_MODE_EMIT reads (without XK_MODE_NOSTATUS)")
    block = int(st.dims[XK_DIM_B])
    if pair_cap < 1 or shots < 1:
        raise ValueError("pack: block, pair_cap and shots must be >= 1")
    if not 0 <= seed <= _U64_MAX or not 0 <= first_block <= _U64_MAX:
        raise ValueError("pack: seed and first_block must be unsigned 64-bit integers")
    nblocks = -(-shots // block)
    if first_block + nblocks - 1 > _U64_MAX:
        raise ValueError("pack: the last block's index (first_block + blocks - 1) must be a 64-bit integer")
    if nblocks * (block + 1) > 0xFFFF_FFFF or nblocks * pair_cap > 0xFFFF_FFFF:
        raise ValueError("pack: batch too large for 32-bit indices")
    dims = st.dims.copy()
    dims[XK_DIM_E] = pair_cap
    dims[XK_DIM_NSHOTS] = shots
    dims[XK_DIM_NBLOCKS] = nblocks
    dims[XK_DIM_MODE] = mode
    return {
        "dims": dims,
        "seed": np.array([stream_key(seed, K_TAG_GPU_NOISE), stream_key(seed, K_TAG_GPU_COIN), first_block,
                          st.rmask], np.uint64),
    }


def pack(tables: dict, *, block: int, pair_cap: int, shots: int, seed: int, first_block: int = 0,
         mode: int = 0, span: dict | None = None, struct=None) -> dict[str, np.ndarray]:
    """The kernels' buffers for one batch, by launcher buffer name (xtim/gpu/xdem_kernels.h): ``static_tables`` and
    ``pack_batch`` together, plus the working buffers.

    ``tables`` is ``XdemSampler._s.device_tables()``. Read-only tables go into the arenas ``tab64`` / ``tab32`` at
    the offsets recorded in ``dims``; ``seed`` holds the stream keys of ``seed``, ``first_block`` (the global index
    of the batch's first block of ``block`` shots) and ``rmask``. The working buffers (``rows``, ``epos``, ``eids``,
    ``psh``, ``overflow``, ``ckey``, ``cdim``, ``coin_in``, ``flag``, ``hist``, ``hist_group``, ``hist_total``,
    ``mws``, ``lawq``) are zeroed, sized for ``shots`` shots, ``pair_cap`` extended records per block,
    ``key_words`` key words and ``max(coord_words, 1)`` coin words per shot, ``hist_layout(dimc_of(tables)).stride``
    counters per block, ``mw_words`` scratch words per shot (multi-word files; one word otherwise) and ``shots + 1``
    queue words (K2's queue under ``XK_MODE_DEFER``). The arenas hold the single-line law table (``law_table``).
    Emit's buffers (``boff``, ``out_det``, ``out_obs``, ``out_cs``, ``cmiss``) are zeroed and sized for every shot
    surviving; ``ctab`` is the empty C-id table (``cspace_table``).

    ``span``: the postselection span test (``XdemSampler.span_tables(flags)``), required with ``mode`` bit
    ``XK_MODE_POSTSELECT``; its tables are appended to the arenas (an empty mask: rank 0, no word). ``struct``: the
    structured law's plan (``static_tables``; ``tables`` then ``structure.renumber``'s).
    """
    _check_fast(tables)
    if mode & XK_MODE_POSTSELECT and span is None:
        raise ValueError("pack: postselection (mode bit XK_MODE_POSTSELECT) needs the span tables (span=)")
    if block < 1 or pair_cap < 1 or shots < 1:
        raise ValueError("pack: block, pair_cap and shots must be >= 1")
    st = static_tables(tables, block=block, span=span, struct=struct)
    out = {**pack_batch(st, pair_cap=pair_cap, shots=shots, seed=seed, first_block=first_block, mode=mode),
           "tab64": st.tab64, "tab32": st.tab32}
    nblocks = -(-shots // block)
    W = int(tables["W"])
    hl = hist_layout(dimc_of(tables))
    out.update({
        "rows": np.zeros(shots * W, np.uint64),
        "epos": np.zeros(nblocks * (block + 1), np.uint32),
        "eids": np.zeros(nblocks * pair_cap, np.uint32),
        "psh": np.zeros(nblocks * 2 * pair_cap, np.uint32),
        "overflow": np.zeros(nblocks, np.uint32),
        "ckey": np.zeros(shots * key_words(tables), np.uint64),
        "cdim": np.zeros(shots, np.uint8),
        "coin_in": np.zeros(shots * max(coord_words(tables), 1), np.uint64),
        "flag": np.zeros(shots, np.uint8),
        "hist": np.zeros(nblocks * hl.stride, np.uint32),
        "hist_group": np.zeros(XK_HIST_GROUPS * hl.count, np.uint64),
        "hist_total": np.zeros(hl.count, np.uint64),
        "mws": np.zeros(max(shots * mw_words(tables), 1), np.uint64),
        "lawq": np.zeros(shots + 1, np.uint32),
        "boff": np.zeros(-(-shots // XK_EMIT_RUN), np.uint32),
        "ctab": cspace_table(np.zeros((0, key_words(tables)), np.uint64), np.zeros(0, np.int32)),
        "out_det": np.zeros(shots * int(st.dims[XK_DIM_NBD]), np.uint8),
        "out_obs": np.zeros(shots * int(st.dims[XK_DIM_NBO]), np.uint8),
        "out_cs": np.zeros(shots, np.uint32),
        "cmiss": np.zeros(2 * shots + 1, np.uint32),
    })
    return out


# ------------------------------------------------------------------------------------------------ K4's C-id table

CSPACE_TABLE_MAX_KEYS = 1 << 12   # GpuBackend's device table of known C keys holds at most this many keys
CSPACE_TABLE_MAX_WORDS = 1 << 18   # ... and at most this many key words


def key_hash(keys: np.ndarray) -> np.ndarray:
    """xk_key_hash of each row of keys (u64[n, kw]): h = gamma, then h = mix(h ^ word) + gamma per word"""
    keys = np.asarray(keys, np.uint64)
    g = np.uint64(0x9E3779B97F4A7C15)
    h = np.full(len(keys), g, np.uint64)
    with np.errstate(over="ignore"):
        for i in range(keys.shape[1]):
            z = h ^ keys[:, i]
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            h = (z ^ (z >> np.uint64(31))) + g
    return h


def cspace_table(keys: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """K4's read-only C-id table (xdem_kernels.h, xk_cspace_lookup) of distinct nonzero keys (u64[m, kw]) and their
    ids (int32[m], >= 0): [n] (n = the slots, a power of two >= 2 m; 0 when m = 0), the n slots ((id + 1) << 32 | key
    index, 0 = empty; linear probing from key_hash & (n - 1)), then the keys"""
    keys = np.ascontiguousarray(keys, np.uint64)
    m = len(keys)
    if not m:
        return np.zeros(1, np.uint64)
    n = 1 << max(1, (2 * m - 1).bit_length())
    slots = np.zeros(n, np.uint64)
    for i, h in enumerate(key_hash(keys).tolist()):
        s = h & (n - 1)
        while slots[s]:
            s = (s + 1) & (n - 1)
        slots[s] = (int(ids[i]) + 1) << 32 | i
    return np.concatenate([np.array([n], np.uint64), slots, keys.reshape(-1)])
