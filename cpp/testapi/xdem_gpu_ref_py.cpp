// GPU xdem sampler, Task 4 test API (not a public interface): the kernel source xtim/gpu/xdem_kernels.h built as
// plain C++, as the reference the GPU launchers are checked against (tests/_gpu_ref.py, tests/test_gpu_kernels.py).
// The source is compiled three times, in namespaces with different capacity macros (variant 0: the reference
// capacities, XK_CAP_E large enough for pair capacities past 2^18: since the walk keeps its list in the psh buffer,
// no capacity sizes a large local array; variant 1: tight ones; variant 2: variant 0 with XK_CAP_RHO = 32, which
// selects the other implementation of the one-word law; variants 0 and 2 also compile the structured laws in, one-word
// and multi-word, at their largest capacities XK_CAP_ST_*), so a test can check that no output depends on the
// capacities and that a size above a capacity is reported by the kernels. Registered in
// record_law_oracle_py.cpp (PYBIND11_MODULE _xtim_testapi).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace xk_ref {   // the multi-word law compiled in (XK_CAP_CW: any CW up to 64, i.e. r and rho up to 4096)
#define XK_CAP_W 64
#define XK_CAP_WA 64
#define XK_CAP_RHO 64
#define XK_CAP_E 4194304
#define XK_CAP_J 40
#define XK_CAP_CW 64
#define XK_CAP_DIMC 253
#define XK_CAP_ST_NC 8
#define XK_CAP_ST_NG 8
#define XK_CAP_ST_T 32
#define XK_CAP_ST_N 16
#define XK_CAP_ST_W 4
#define XK_CAP_SHM 140000
#include "../../xtim/gpu/xdem_kernels.h"
constexpr unsigned kCaps[13] = {XK_CAP_W,     XK_CAP_WA,    XK_CAP_RHO,   XK_CAP_E,   XK_CAP_J, XK_CAP_CW,
                                XK_CAP_DIMC,  XK_CAP_ST_NC, XK_CAP_ST_NG, XK_CAP_ST_T, XK_CAP_ST_N, XK_CAP_ST_W,
                                XK_CAP_SHM};
#undef XK_CAP_W
#undef XK_CAP_WA
#undef XK_CAP_RHO
#undef XK_CAP_E
#undef XK_CAP_J
#undef XK_CAP_CW
#undef XK_CAP_DIMC
#undef XK_CAP_ST_NC
#undef XK_CAP_ST_NG
#undef XK_CAP_ST_T
#undef XK_CAP_ST_N
#undef XK_CAP_ST_W
#undef XK_CAP_SHM
}  // namespace xk_ref

namespace xk_tight {   // the largest bundled example's sizes (cultivation_d5 at B = 64), E = 256; one-word only
#define XK_CAP_W 3
#define XK_CAP_WA 2
#define XK_CAP_RHO 15
#define XK_CAP_E 256
#define XK_CAP_J 17
#define XK_CAP_CW 1
#define XK_CAP_DIMC 64
#define XK_CAP_SHM 590
#include "../../xtim/gpu/xdem_kernels.h"
constexpr unsigned kCaps[13] = {XK_CAP_W, XK_CAP_WA, XK_CAP_RHO, XK_CAP_E, XK_CAP_J, XK_CAP_CW, XK_CAP_DIMC,
                                0,        0,         0,          0,         0,         XK_CAP_SHM};   // no structured law
#undef XK_CAP_W
#undef XK_CAP_WA
#undef XK_CAP_RHO
#undef XK_CAP_E
#undef XK_CAP_J
#undef XK_CAP_CW
#undef XK_CAP_DIMC
#undef XK_CAP_SHM
}  // namespace xk_tight

namespace xk_wide {   // variant 0 with XK_CAP_RHO = 32: the one-word law's register form (xk_law_fast's first
                      // implementation) on every one-word file with rho <= 32; variant 0 runs the second
#define XK_CAP_W 64
#define XK_CAP_WA 64
#define XK_CAP_RHO 32
#define XK_CAP_E 4194304
#define XK_CAP_J 40
#define XK_CAP_CW 64
#define XK_CAP_DIMC 253
#define XK_CAP_ST_NC 8
#define XK_CAP_ST_NG 8
#define XK_CAP_ST_T 32
#define XK_CAP_ST_N 16
#define XK_CAP_ST_W 4
#define XK_CAP_SHM 140000
#include "../../xtim/gpu/xdem_kernels.h"
constexpr unsigned kCaps[13] = {XK_CAP_W,     XK_CAP_WA,    XK_CAP_RHO,   XK_CAP_E,   XK_CAP_J, XK_CAP_CW,
                                XK_CAP_DIMC,  XK_CAP_ST_NC, XK_CAP_ST_NG, XK_CAP_ST_T, XK_CAP_ST_N, XK_CAP_ST_W,
                                XK_CAP_SHM};
#undef XK_CAP_W
#undef XK_CAP_WA
#undef XK_CAP_RHO
#undef XK_CAP_E
#undef XK_CAP_J
#undef XK_CAP_CW
#undef XK_CAP_DIMC
#undef XK_CAP_ST_NC
#undef XK_CAP_ST_NG
#undef XK_CAP_ST_T
#undef XK_CAP_ST_N
#undef XK_CAP_ST_W
#undef XK_CAP_SHM
}  // namespace xk_wide

namespace {

// the capacity macros; the structured law's five (0 in a variant: not compiled in, no such macro)
const char* const kCapNames[13] = {"XK_CAP_W",     "XK_CAP_WA",    "XK_CAP_RHO",  "XK_CAP_E",
                                   "XK_CAP_J",     "XK_CAP_CW",    "XK_CAP_DIMC", "XK_CAP_ST_NC",
                                   "XK_CAP_ST_NG", "XK_CAP_ST_T",  "XK_CAP_ST_N", "XK_CAP_ST_W",
                                   "XK_CAP_SHM"};
const unsigned* const kVariantCaps[3] = {xk_ref::kCaps, xk_tight::kCaps, xk_wide::kCaps};

[[noreturn]] void refuse(const std::string& what) { throw std::invalid_argument("xdem_gpu_ref: " + what); }

// buffers[name] as a C-contiguous writeable array of T with at least `need` elements
template <class T>
T* buffer(const py::dict& buffers, const char* name, uint64_t need) {
    if (!buffers.contains(name)) refuse(std::string("missing buffer '") + name + "'");
    if (!py::isinstance<py::array>(buffers[name])) refuse(std::string("buffer '") + name + "' is not an array");
    py::array a = py::reinterpret_borrow<py::array>(buffers[name]);
    if (!a.dtype().is(py::dtype::of<T>())) refuse(std::string("buffer '") + name + "' has the wrong dtype");
    if (!(a.flags() & py::array::c_style) || !a.writeable())
        refuse(std::string("buffer '") + name + "' must be C-contiguous and writeable");
    if ((uint64_t)a.size() < need)
        refuse(std::string("buffer '") + name + "' has " + std::to_string(a.size()) + " elements, needs " +
               std::to_string(need));
    return static_cast<T*>(a.mutable_data());
}

struct Bufs {
    uint32_t* dims = nullptr;
    uint64_t* seed = nullptr;
    uint64_t* tab64 = nullptr;
    uint32_t* tab32 = nullptr;
    uint64_t* rows = nullptr;
    uint32_t* epos = nullptr;
    uint32_t* eids = nullptr;
    uint32_t* psh = nullptr;
    uint32_t* overflow = nullptr;
    uint64_t* ckey = nullptr;
    uint8_t* cdim = nullptr;
    uint64_t* coin_in = nullptr;
    uint8_t* flag = nullptr;
    uint32_t* hist = nullptr;
    uint64_t* hist_group = nullptr;
    uint64_t* hist_total = nullptr;
    uint64_t* mws = nullptr;
    uint32_t* lawq = nullptr;
    uint32_t* boff = nullptr;
    uint64_t* ctab = nullptr;
    uint8_t* out_det = nullptr;
    uint8_t* out_obs = nullptr;
    uint32_t* out_cs = nullptr;
    uint32_t* cmiss = nullptr;
};

enum Kernel { kWalk, kResolve, kEmit, kReduce, kReduceGroups, kReduce2, kResolveLaw, kScan };

// the buffers each kernel reads (xdem_kernels.h's header comment; xtim/gpu/metal.py's _KERNELS): only these must be
// uploaded, the others are passed as null pointers (the Metal launcher binds a placeholder)
bool needs(Kernel k, const std::string& name) {
    static const char* const common[] = {"dims", "seed", "tab64", "tab32", "rows"};
    for (const char* c : common)
        if (name == c) return true;
    static const char* const walk[] = {"epos", "eids", "psh", "overflow", "lawq", "cmiss", "flag"};
    static const char* const resolve[] = {"epos", "eids", "ckey", "cdim", "coin_in", "flag", "mws", "lawq"};
    static const char* const emit[] = {"ckey", "cdim", "flag", "boff", "ctab", "out_det", "out_obs", "out_cs",
                                       "cmiss"};
    static const char* const scan[] = {"flag", "hist", "hist_group", "boff"};
    static const char* const reduce[] = {"cdim", "flag", "hist"};
    static const char* const groups[] = {"overflow", "hist", "hist_group"};
    static const char* const reduce2[] = {"hist_group", "hist_total"};
    auto in = [&](const char* const* b, const char* const* e) {
        for (; b != e; ++b)
            if (name == *b) return true;
        return false;
    };
    switch (k) {
        case kWalk: return in(std::begin(walk), std::end(walk));
        case kResolve:
        case kResolveLaw: return in(std::begin(resolve), std::end(resolve));
        case kReduce: return in(std::begin(reduce), std::end(reduce));
        case kReduceGroups: return in(std::begin(groups), std::end(groups));
        case kReduce2: return in(std::begin(reduce2), std::end(reduce2));
        case kScan: return in(std::begin(scan), std::end(scan));
        default: return in(std::begin(emit), std::end(emit));
    }
}

template <class Ctx, class Init, class Walk, class Resolve, class ResolveLaw, class Emit, class Reduce, class Groups,
          class Reduce2, class Scan>
void run_variant(Init init, Walk walk, Resolve resolve, ResolveLaw resolve_law, Emit emit, Reduce reduce,
                 Groups groups, Reduce2 reduce2, Scan scan, Kernel k, uint32_t n, uint32_t lanes, uint32_t shm_words,
                 const Bufs& b) {
    Ctx c;
    init(&c, b.dims, b.seed, b.tab64, b.tab32, b.rows, b.epos, b.eids, b.psh, b.overflow, b.ckey, b.cdim, b.coin_in,
         b.flag, b.hist, b.hist_group, b.hist_total, b.mws, b.lawq, b.boff, b.ctab, b.out_det, b.out_obs, b.out_cs,
         b.cmiss);
    // the walk's group memory: XK_CAP_SHM words, filled with garbage so that a read of a word no phase wrote shows
    std::vector<uint32_t> shm(k == kWalk ? shm_words : 0, 0xA5A5A5A5u);
    for (uint32_t i = 0; i < n; ++i) {
        if (k == kWalk) walk(&c, i, 0u, lanes, shm.data());
        else if (k == kResolve) resolve(&c, i);
        else if (k == kResolveLaw) resolve_law(&c, i);
        else if (k == kReduce) reduce(&c, i);
        else if (k == kReduceGroups) groups(&c, i);
        else if (k == kReduce2) reduce2(&c, i);
        else if (k == kScan) scan(&c, i);
        else emit(&c, i);
    }
}

void xdem_gpu_ref(const std::string& kernel, int64_t n, const py::dict& buffers, int64_t lanes) {
    Kernel k = kWalk;
    if (kernel == "walk") k = kWalk;
    else if (kernel == "resolve") k = kResolve;
    else if (kernel == "emit") k = kEmit;
    else if (kernel == "scan_blocks") k = kScan;
    else if (kernel == "reduce") k = kReduce;
    else if (kernel == "reduce_groups") k = kReduceGroups;
    else if (kernel == "reduce2") k = kReduce2;
    else if (kernel == "resolve_law") k = kResolveLaw;
    else refuse("kernel must be 'walk', 'resolve', 'resolve_law', 'reduce', 'reduce_groups', 'reduce2', "
                "'scan_blocks' or 'emit'");
    // buffers[name] when kernel k reads it (refused when missing or too short), else nullptr
    auto get = [&](auto tag, const char* name, uint64_t need) -> decltype(tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        return needs(k, name) ? buffer<T>(buffers, name, need) : nullptr;
    };
    uint32_t variant = 0;
    if (buffers.contains("_caps")) variant = *buffer<uint32_t>(buffers, "_caps", 1);
    if (variant > 2) refuse("_caps selects variant 0, 1 or 2");
    Bufs b;
    b.dims = buffer<uint32_t>(buffers, "dims", XK_DIM_COUNT);
    const uint32_t* d = b.dims;
    const uint64_t W = d[XK_DIM_W], rho = d[XK_DIM_RHO], rhoL = d[XK_DIM_RHOL], B = d[XK_DIM_B],
                   E = d[XK_DIM_E], nch = d[XK_DIM_NCHAINS], nshots = d[XK_DIM_NSHOTS], nblocks = d[XK_DIM_NBLOCKS],
                   stride = d[XK_DIM_THR_STRIDE], rank = d[XK_DIM_SPAN_RANK], nw = d[XK_DIM_SPAN_WORDS];
    const uint64_t cw = d[XK_DIM_CW], kcw = cw ? cw : 1, kw = (rhoL ? rhoL : 1) * kcw;
    const uint64_t hs = (uint64_t)d[XK_DIM_DIMC] + 3, hc = hs + 2;
    if ((uint64_t)d[XK_DIM_WD] + d[XK_DIM_WO] != d[XK_DIM_WF] || W != d[XK_DIM_WF])
        refuse("dims: W, WD, WO, WF disagree");
    if (B < 1 || nblocks != (nshots + B - 1) / B) refuse("dims: nblocks must be ceil(nshots / B)");
    if (cw == 1) refuse("dims: CW is 0 (one-word) or >= 2");
    if (rhoL > rho || (!cw && (rho > 64 || d[XK_DIM_R] > 64)) || (cw && (rho > 64 * cw || d[XK_DIM_R] > 64 * cw)))
        refuse("dims: rho, rhoL or r out of range (the fast path, CW words)");
    if (d[XK_DIM_DIMC] < 64 || d[XK_DIM_DIMC] > 253) refuse("dims: DIMC out of range");
    // no capacity check here: a size above its capacity is the kernels' to report (xk_caps_ok), and the thread
    // count may exceed nblocks / nshots (the kernels ignore the extra ids), as on a rounded-up GPU grid
    if (n < 0 || n > (int64_t)UINT32_MAX) refuse("n out of range");
    if (lanes < 1 || lanes > 1024) refuse("lanes out of range (1 to 1024)");
    b.seed = buffer<uint64_t>(buffers, "seed", 4);
    b.tab64 = buffer<uint64_t>(buffers, "tab64", 1);
    b.tab32 = buffer<uint32_t>(buffers, "tab32", 1);
    const uint64_t n64 = (uint64_t)py::len(buffers["tab64"]), n32 = (uint64_t)py::len(buffers["tab32"]);
    for (int i = XK_DIM_OFF_CHAIN_THR; i <= XK_DIM_OFF_BLO; ++i)
        if (d[i] > n64) refuse("dims: a tab64 offset is past the arena");
    if (cw && (d[XK_DIM_OFF_KCW] > n64 || d[XK_DIM_OFF_GCW] > n64 || (uint64_t)d[XK_DIM_OFF_GAMMA] + rho * W > n64 ||
               (uint64_t)d[XK_DIM_OFF_GBW] + rho * cw > n64))
        refuse("dims: a multi-word table is past the arena");
    for (int i = XK_DIM_OFF_CHAIN_OFF; i <= XK_DIM_OFF_PL; ++i)
        if (d[i] > n32) refuse("dims: a tab32 offset is past the arena");
    // the walk's items and segments inside the arenas, each segment's threshold row inside chain_thr
    const uint64_t nit = d[XK_DIM_NITEMS];
    if (d[XK_DIM_OFF_CHAIN_OFF] + nch + 1 > n32 || (uint64_t)d[XK_DIM_OFF_ITEM] + nit + 1 > n32)
        refuse("dims: the chain tables do not fit in the arenas");
    if (k == kWalk && !(d[XK_DIM_MODE] & XK_MODE_GIVEN)) {
        const uint32_t* ib = b.tab32 + d[XK_DIM_OFF_ITEM];
        const uint64_t nseg = ib[nit];
        if ((uint64_t)d[XK_DIM_OFF_SEG] + 5 * nseg > n32) refuse("dims: the walk's segments are past the arena");
        for (uint64_t i = 0; i < nit; ++i)
            if (ib[i] > ib[i + 1]) refuse("item_beg is not monotone");
        for (uint64_t g = 0; g < nseg; ++g) {
            const uint32_t* r = b.tab32 + d[XK_DIM_OFF_SEG] + 5 * g;
            // a segment with tests (r[1] > 0) reads its rows of chain_thr and chain_thr8; one without reads neither
            if (r[1] > stride || (r[1] && ((uint64_t)d[XK_DIM_OFF_CHAIN_THR] + (uint64_t)(r[0] + 1) * stride > n64 ||
                (uint64_t)d[XK_DIM_OFF_CHAIN_THR8] + (uint64_t)(r[0] + 1) * d[XK_DIM_THR8_STRIDE] > n64)))
                refuse("a walk segment's threshold row is past the arena");
        }
    }
    // the span test: its tables inside the arenas, its words inside the detector words, its pivots inside its words
    if ((uint64_t)d[XK_DIM_OFF_SPAN_WMASK] + nw > n64 || (uint64_t)d[XK_DIM_OFF_SPAN_BASIS] + rank * nw > n64 ||
        (uint64_t)d[XK_DIM_OFF_SPAN_PIV_BIT] + rank > n64 || (uint64_t)d[XK_DIM_OFF_SPAN_WIDX] + nw > n32 ||
        (uint64_t)d[XK_DIM_OFF_SPAN_PIV_WORD] + rank > n32)
        refuse("dims: the span tables do not fit in the arenas");
    for (uint64_t i = 0; i < nw; ++i)
        if (b.tab32[d[XK_DIM_OFF_SPAN_WIDX] + i] >= d[XK_DIM_WD]) refuse("span_widx outside the detector words");
    for (uint64_t i = 0; i < rank; ++i)
        if (b.tab32[d[XK_DIM_OFF_SPAN_PIV_WORD] + i] >= nw) refuse("span_piv_word outside the span words");
    b.rows = buffer<uint64_t>(buffers, "rows", nshots * W);
    b.epos = get((uint32_t*)nullptr, "epos", nblocks * (B + 1));
    b.eids = get((uint32_t*)nullptr, "eids", nblocks * E);
    b.psh = get((uint32_t*)nullptr, "psh", nblocks * 2 * E);
    b.overflow = get((uint32_t*)nullptr, "overflow", nblocks);
    b.ckey = get((uint64_t*)nullptr, "ckey", nshots * kw);
    b.cdim = get((uint8_t*)nullptr, "cdim", nshots);
    // coin_in is read only under XK_MODE_COIN_IN (a backend binds a one-element buffer otherwise)
    b.coin_in = get((uint64_t*)nullptr, "coin_in", (d[XK_DIM_MODE] & XK_MODE_COIN_IN) ? nshots * kcw : 0);
    b.flag = get((uint8_t*)nullptr, "flag", nshots);
    b.hist = get((uint32_t*)nullptr, "hist", nblocks * hs);
    b.hist_group = get((uint64_t*)nullptr, "hist_group", (uint64_t)XK_HIST_GROUPS * hc);
    b.hist_total = get((uint64_t*)nullptr, "hist_total", hc);
    // the multi-word law's scratch (xk_mw_words per shot); a one-word file or the structured law binds any buffer
    b.mws = get((uint64_t*)nullptr, "mws", cw && !d[XK_DIM_ST] ? nshots * (8 * cw + 4 * rho * cw + 4 * rho) : 0);
    // K2's queue (XK_MODE_DEFER): a count and up to nshots entries
    b.lawq = get((uint32_t*)nullptr, "lawq", (d[XK_DIM_MODE] & XK_MODE_DEFER) ? nshots + 1 : 0);
    // K4's outputs (a row per shot: at most nshots survivors), its list of C ids left to the host (XK_MODE_EMIT) and
    // K7's block offsets
    const uint64_t nbd = d[XK_DIM_NBD], nbo = d[XK_DIM_NBO];
    if (nbd > 8 * (uint64_t)d[XK_DIM_WD] || nbo > 8 * (uint64_t)d[XK_DIM_WO]) refuse("dims: NBD or NBO past the row");
    b.boff = get((uint32_t*)nullptr, "boff", (nshots + XK_EMIT_RUN - 1) / XK_EMIT_RUN);
    b.out_det = get((uint8_t*)nullptr, "out_det", nshots * nbd);
    b.out_obs = get((uint8_t*)nullptr, "out_obs", nshots * nbo);
    b.out_cs = get((uint32_t*)nullptr, "out_cs", nshots);
    b.cmiss = get((uint32_t*)nullptr, "cmiss", (d[XK_DIM_MODE] & XK_MODE_EMIT) ? 2 * nshots + 1 : 0);
    if (needs(k, "ctab")) {   // the C-id table: its slots and keys inside the buffer, kw words per key
        b.ctab = buffer<uint64_t>(buffers, "ctab", 1);
        const uint64_t len = (uint64_t)py::len(buffers["ctab"]), ns = b.ctab[0];
        if (ns & (ns - 1)) refuse("ctab: the slot count must be 0 or a power of two");
        if (ns + 1 > len) refuse("ctab: the slots are past the buffer");
        for (uint64_t i = 0; i < ns; ++i) {
            const uint64_t e = b.ctab[1 + i];
            if (e && (1 + ns + ((e & 0xFFFFFFFFu) + 1) * kw > len || (e >> 32) == 0))
                refuse("ctab: a slot's key is past the buffer");
        }
    }
    // the single-line law table inside the arenas (its entries are the host's to get right)
    if (d[XK_DIM_LAWT] && ((uint64_t)d[XK_DIM_OFF_XLAW] > n32 || (uint64_t)d[XK_DIM_OFF_LAWT] > n64))
        refuse("dims: the law table is past the arenas");
    // the zero decoder's observable table (one-word law; the structured multi-word law: ST_W words per entry): 64 WO
    // entries, and 64 WO * WO rest words when present
    const uint64_t WO = d[XK_DIM_WO], stw = d[XK_DIM_ST_W];
    const bool st_mw = cw && d[XK_DIM_ST];
    if ((!cw || st_mw) && rhoL &&
        ((uint64_t)d[XK_DIM_OFF_STAB] + 64 * WO * (st_mw ? stw : 1) > n64 ||
         (d[XK_DIM_RTAB] && (uint64_t)d[XK_DIM_OFF_RTAB] + 64 * WO * WO > n64)))
        refuse("dims: the observable table is past the arenas");
    // the structured law's table: its header, two words per type record and per non-leaf axis (multi-word: Lall and u
    // of ST_W words, and ST_W + 2 words per non-leaf axis), and on a multi-word file its beta vectors (rho of WF words)
    const uint64_t st_end = (uint64_t)d[XK_DIM_OFF_ST] + 16 +
                            (st_mw ? 2 * stw + 2 * (uint64_t)d[XK_DIM_ST_NT] + (stw + 2) * (uint64_t)d[XK_DIM_ST_NN]
                                   : 2 * ((uint64_t)d[XK_DIM_ST_NT] + d[XK_DIM_ST_NN]));
    const uint64_t beta_end = (uint64_t)d[XK_DIM_OFF_BETA] + (uint64_t)d[XK_DIM_RHO] * d[XK_DIM_WF];
    if (d[XK_DIM_ST] && (st_end > n64 || (st_mw && (stw == 0 || beta_end > n64))))
        refuse("dims: the structured law's table is past the arenas");
    if (k == kResolveLaw && (d[XK_DIM_MODE] & XK_MODE_DEFER) && b.lawq[0] > nshots)
        refuse("lawq holds more than nshots entries");
    if (k == kResolve || k == kResolveLaw)   // the lists K2 reads: each block's CSR within its E records
        for (uint64_t blk = 0; blk < nblocks; ++blk) {
            const uint32_t* fp = b.epos + blk * (B + 1);
            for (uint64_t s = 0; s < B; ++s)
                if (fp[s] > fp[s + 1]) refuse("epos is not monotone");
            if (fp[B] > E) refuse("epos lists more than E records");
        }
    if (k == kEmit && (d[XK_DIM_MODE] & XK_MODE_POSTSELECT))   // K7's offsets: survivors before each run
        for (uint64_t j = 0; j < (nshots + XK_EMIT_RUN - 1) / XK_EMIT_RUN; ++j)
            if (b.boff[j] > nshots) refuse("boff past nshots");
    const unsigned shm_words = (variant == 0 ? xk_ref::kCaps : variant == 1 ? xk_tight::kCaps : xk_wide::kCaps)[12];
    py::gil_scoped_release rel;
    if (variant == 0)
        run_variant<xk_ref::XkCtx>(xk_ref::xk_ctx_init, xk_ref::xk_walk_group, xk_ref::xk_resolve_shot,
                                   xk_ref::xk_resolve_law, xk_ref::xk_emit, xk_ref::xk_reduce_block,
                                   xk_ref::xk_reduce_groups, xk_ref::xk_reduce_total, xk_ref::xk_scan_blocks,
                                   k, (uint32_t)n, (uint32_t)lanes, shm_words, b);
    else if (variant == 1)
        run_variant<xk_tight::XkCtx>(xk_tight::xk_ctx_init, xk_tight::xk_walk_group, xk_tight::xk_resolve_shot,
                                     xk_tight::xk_resolve_law, xk_tight::xk_emit, xk_tight::xk_reduce_block,
                                     xk_tight::xk_reduce_groups, xk_tight::xk_reduce_total, xk_tight::xk_scan_blocks,
                                     k, (uint32_t)n, (uint32_t)lanes, shm_words, b);
    else
        run_variant<xk_wide::XkCtx>(xk_wide::xk_ctx_init, xk_wide::xk_walk_group, xk_wide::xk_resolve_shot,
                                    xk_wide::xk_resolve_law, xk_wide::xk_emit, xk_wide::xk_reduce_block,
                                    xk_wide::xk_reduce_groups, xk_wide::xk_reduce_total, xk_wide::xk_scan_blocks,
                                    k, (uint32_t)n, (uint32_t)lanes, shm_words, b);
}

}  // namespace

void register_xdem_gpu_ref(py::module_& m) {
    m.def("xdem_gpu_ref", &xdem_gpu_ref, py::arg("kernel"), py::arg("n"), py::arg("buffers"), py::arg("lanes") = 256,
          "TEST API: run kernel 'walk' (K1, n groups = blocks of `lanes` threads, each phase run for every thread in "
          "turn), 'resolve' (K2, thread ids = shots), 'resolve_law' "
          "(K2b, thread ids = K2's queue entries), 'reduce' (K3, "
          "thread ids = blocks), 'emit' (K4, thread ids = shots), 'reduce_groups' (K5, thread ids = group * "
          "XK_HIST_TOTAL_COUNT + slot), 'reduce2' (K6, thread ids = hist_total slots) or 'scan_blocks' (K7, thread "
          "ids = blocks) of xtim/gpu/xdem_kernels.h "
          "for thread ids 0..n-1 on the CPU, in place on buffers (dict of numpy arrays by launcher buffer name: dims, "
          "seed, tab64, tab32, rows, epos, eids, psh, overflow, ckey, cdim, coin_in, flag, hist, hist_group, "
          "hist_total, mws, lawq, boff, ctab, out_det, out_obs, out_cs, cmiss; only "
          "the kernel's own buffers are needed, the others are not read; '_caps' uint32[1] picks the build variant, "
          "default 0).");
    m.def("xdem_gpu_ref_caps", []() {
        py::list out;
        for (const unsigned* caps : kVariantCaps) {
            py::dict v;
            for (int i = 0; i < 13; ++i)
                if (caps[i]) v[kCapNames[i]] = caps[i];
            out.append(v);
        }
        return out;
    }, "TEST API: the capacity macros of xdem_gpu_ref's build variants (index = the '_caps' value).");
}
