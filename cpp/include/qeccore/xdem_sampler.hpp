// xdem_sampler.hpp — file-only sampler, per-shot status and scorer for an xdem file or a conventional DEM
// (SPEC §1.2, §1.3, §4.7). The successor of the dc-DEM v1 sampler (removed): the same noise walk, stream, chains,
// postselection, fast path, law cache and C table; the X-pattern words of its shot row are replaced by ordered
// composition.
//
// Per shot: every line fires independently with its p. The shot row holds the D + O readout bits only: flip = XOR of
// the fired lines' targets. The fired EXTENDED lines (a line is extended iff its total exponent E = E^< + E^> mod 4 or
// its tau restricted to the rotated axes is nonempty; rotated = the axes some line rotates about) are gathered,
// sorted into file order and composed by the cocycle of SPEC §1.2 with a sign register over the axes:
//   eta_a = sum_j (-1)^{tau_{<j}(a)} E_{i_j, a}  (mod 4),
// each later line's rotation about a negated once per earlier fired line whose tau holds a. tau on an axis no line
// rotates about is never read (it only negates rotations), so the sampler keeps tau restricted to the rotated axes.
// The outcome (D detector bits d, O observable bits v) is drawn uniformly from off + span(coins), the law of §1.3 of
// the composite (qeccore::xdem::law_into), and the shot records C = {e_L : e in span(coins), e_D = 0} as an index into
// an append-only per-sampler table of distinct C. pass = 2^-dim C if (v XOR c) in C, else 0 (score).
//
// A conventional DEM (no xdem header) has no axes and no extended line: it samples byte-identically to the dc-DEM v1
// sampler's plain-DEM mode at BASE 19f2f78 (same constants kBlockShots, kTagNoise, kTagCoin, same walk, same row
// layout; pinned by test_xdem_sampler_golden).
//
// Stream (the ONE definition): shots are cut into blocks of kBlockShots = 1024; global block b (shot indices
// [1024 b, 1024 b + 1024) of the seed's stream) draws only from
//   noise stream  N_b(c) = mix(key_b(seed, kTagNoise) + (c + 1) γ),   coin stream  C_b(c) = mix(key_b(seed, kTagCoin)
//   + (c + 1) γ),   key_b(seed, tag) = mix(stream_key(seed, tag) ^ mix(b))     (qeccore/keyed_rng.hpp mix / γ),
// consumed sequentially inside the block. So a block's shots depend only on (seed, b): the output is identical at
// every thread count and scheduling, sample(n) is a prefix of sample(m) for n <= m, and sample(n, first_block=b0)
// returns shots [1024 b0, 1024 b0 + n) of the stream.
//   Noise: lines are grouped into chains of equal p (first appearance in file order; p = 0 lines never fire, p = 1
//   lines always fire, no draw). A chain of M lines owns M*1024 slots per block, slot = m*1024 + shot (line-major,
//   lines ascending); geometric skipping over the slots: first fired slot floor(ln u / ln(1-p)), then
//   t += 1 + floor(ln u / ln(1-p)), u = 1 - u01(draw) in (0, 1]. Chains are walked in order; slots of shots beyond
//   a truncated last block are drawn and ignored (prefix property). The walk is chain-major, so a shot's fired lines
//   arrive out of file order when chains interleave: the extended ones are sorted per shot before composition.
//   Coins: a shot whose law has coins reads, in shot order, either one word of C_b (the one-word fast path, taken
//   by a file with r <= 64 and rho <= 64 unless force_general: t = r coin bits, the coin combination sum_j t_j
//   coin_j) or ceil(k / 64) words (the multi-word fast path, r > 64 or rho > 64, and the general path under
//   force_general for every file: k = rank, combination over the canonical basis, law_into's fully reduced RREF in
//   descending pivot order; word i's bit b selects coin 64 i + b). So a file with r > 64 or rho > 64 samples
//   byte-identically on both levers; a file with r, rho <= 64 draws differently under force_general (as before).
// Law: eta_a in {2, 3} XORs img_a into the outcome (eta = 2: the law's even part; eta = 3 equals eta = 1 plus img_a
// modulo the coins), so the rest of the law depends on the composite only through the ODD-AXIS SET
// O = {a : eta_a odd}: outcome = flip ^ sum_{eta_a in {2,3}} img_a ^ q(O) ^ (uniform combination of coins(O)).
// Two paths compute (q(O), coins(O), C):
//  * fast path (every file): per shot in coordinates over a basis beta of span(img_a) (beta_D: independent detector
//    parts first, then beta_L: zero detector part), the kernel of N_O with private pivots and the W phase on the
//    kernel basis, no cache. Graph-state form (GSL): the gauge basis is re-expressed once at load in image coordinates
//    (kappa'_a = (t_a | kappa_a on a completing set of gauge coordinates), injective on span(kappa), so N_O keeps its
//    kernel and its rho image rows are the symmetric G = sum_{a in O} t_a t_a^T); the W phase is the mod-4 quadratic
//    form sum_{i in sigma} d_i + 2 sum_{i < j in sigma} G_ij (d_i = |{a in O : t_a bit i}|) on the kernel basis, and
//    the elimination is pivot-indexed in the same column order (the same kernel basis, coins and coin words);
//    dim C = rank(coins) - rank(coins on the beta_D coordinates), C built only when > 0, from the kernel basis. One
//    word per coordinate vector when rho = dim span(img_a) <= 64 and r <= 64 (post_fast); otherwise
//    W = ceil(max(r, rho) / 64) words (post_multi, instances W = 2, 3, 4, 6, 8 with 5, 7 padded to the next one; any
//    larger W at run time), in the coordinates of the canonical basis gamma of span(img_a) (fully
//    reduced echelon, pivot = highest bit) with the columns eliminated in descending order: the kernel basis is then
//    W's lowest-pivot RREF, whose dual is the coins' canonical RREF and the W-phase offset the offset reduced modulo
//    it, law_into's coordinate by coordinate, drawn with the general path's coin stream (status C via beta_L as
//    above). A file with no axes (a conventional DEM) is on the fast path too: nothing to compute, no cache table.
//    The scorer (score_lines, correction, law_of) reads the same laws: one-word (post_fast<LAW>, kind 1) or multi-word
//    (post_multi<LAW>, kind 3: the kernel basis in beta coordinates, CW_ words per vector), and the optimal
//    correction's law of [T, K] in beta' coordinates in one word or cw words (KFast; in the same graph-state form, with
//    kappa' = [kappa | delta] re-expressed once in image coordinates); no size sends any file to the general path;
//  * general path (the test lever force_general only): (q, coins) = qeccore::xdem::law_into at flip = 0,
//    eta = 1 on O on a cache miss. Law cache (per thread slot): content-keyed by O (a bitmask over the axes, never an
//    address) -> (q, coin basis, C). Bounded: when a slot's cache holds `cache_cap` entries it is cleared (counted).
//    The cache never changes an output byte.
// The W-phase conditions (linear, even on W) hold for every file the reader accepts (§3.5(b), AUDIT Cor. 1.2(ii)): on
// the fast path (the sampler's laws and the law of [T, K]) they are identities of the graph-state form
// (sigma^T G sigma' = 0 on ker G, G symmetric), whose one premise, t a linear function of kappa, is the reader's header
// check §3.5(b) (k_build's step 4 for K's strings); the general path keeps them as per-law internal checks.
// C table determinism: each thread slot names C contents by local ids; after a call, blocks are merged IN BLOCK ORDER
// and every new C content gets the next global id (first appearance in shot order). Id 0 = {0} (dim 0, also every
// shot with no extended line). The table is append-only for the sampler's lifetime, so ids stay valid for score().
// Postselection (sample_ps): mask M over the detectors; the call returns the shots whose detectors in M are all 0.
// The noise walk is sample()'s (same stream, same per-shot rows d0 after the line XOR). Per shot, in order:
//  * no extended line: the row is final; pass iff d0_M = 0. No coin word.
//  * an extended line, d0_M not in span(img_a)_M ("early"): rejected with no further work and no coin word. The final
//    detector row is d0 + v with v in span(img_a)_D on both paths (the even images, the law's offset and coins all
//    lie in span(img_a)), so no correction can make it pass. XdemSpanTest decides it exactly.
//  * otherwise: the correction runs as in sample(), reading the next coin words of the block's coin stream; pass iff
//    its final row_M = 0; a survivor then gets its status C (a rejected shot never computes C).
// Which shots read coin words, and how many, is a function of the block's noise alone, so the survivors have the
// joint law of sample() followed by the filter; they depend only on (seed, shots), never on threads. C ids: first
// appearance among the survivors in shot order.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>   // _aligned_malloc / _aligned_free (MSVC's CRT has no std::aligned_alloc)
#elif defined(__APPLE__)
#include <stdlib.h>   // posix_memalign
#endif

#include "qeccore/xdem_file.hpp"

namespace qeccore::xdem {

constexpr int kBlockShots = 1024;
constexpr size_t kDefaultCacheCap = 65536;
constexpr uint64_t kTagNoise = 0x6463'4e6f'6973'6501ull;   // the v1 sampler's stream tags, kept: a conventional DEM
constexpr uint64_t kTagCoin = 0x6463'436f'696e'0002ull;    // samples byte-identically to today (SPEC §6 T4)

// Postselection test on a detector row (internal layout: detector d = bit d of words [0, ceil(D / 64))) for a mask M:
// zero(d) = (d_M == 0); in_span(d) = (d_M in V_M), V_M = the span of the given vectors restricted to M. Built once
// per mask (a fully reduced echelon basis of V_M over M's words only, pivot = lowest bit); in_span is one masked XOR
// per basis vector (the pivot bits of d select the combination) and an OR. Exact: no heuristic.
class XdemSpanTest {
  public:
    XdemSpanTest() = default;
    // flags: D bytes (nonzero = detector in M); vecs: nvec vectors, vector i at vecs + i * stride (>= ceil(D / 64)
    // words, bits >= D ignored).
    XdemSpanTest(int D, const uint8_t* flags, const uint64_t* vecs, size_t nvec, size_t stride);
    bool zero(const uint64_t* d) const {
        uint64_t acc = 0;
        for (size_t k = 0; k < widx_.size(); ++k) acc |= d[widx_[k]] & wmask_[k];
        return acc == 0;
    }
    // scratch: >= words() words
    bool in_span(const uint64_t* d, uint64_t* scratch) const;
    size_t words() const { return widx_.size(); }   // words of the row that hold a detector of M
    const uint32_t* word_index() const { return widx_.data(); }   // row word of compressed word k
    const uint64_t* word_mask() const { return wmask_.data(); }   // M's bits in it
    int rank() const { return rank_; }
    bool selects() const { return !widx_.empty(); }   // false: M is empty (every row passes)
    // the echelon basis (rank() * words() compressed words, basis i at i * words()) and each basis vector's pivot
    // (compressed word, bit mask), in build order (the export of XdemSampler::span_tables)
    const std::vector<uint64_t>& basis() const { return basis_; }
    const std::vector<uint32_t>& pivot_word() const { return pk_; }
    const std::vector<uint64_t>& pivot_bit() const { return pb_; }

  private:
    std::vector<uint32_t> widx_;
    std::vector<uint64_t> wmask_;
    int rank_ = 0;
    std::vector<uint64_t> basis_;   // rank_ * words(), compressed words
    std::vector<uint32_t> pk_;      // pivot of basis i: compressed word, bit mask, bit index
    std::vector<uint64_t> pb_;
    std::vector<uint8_t> ps_;
    // rank <= 64 and <= 4 words: pivot masks per word and ceil(rank / 8) byte tables of basis combinations
    std::vector<uint64_t> pm_, tab_;
    int ntab_ = 0;
};

// A postselected call's counts (sample_ps): attempted shots, survivors, and the shots rejected by the span test
// before any correction work.
struct XdemPsCounts {
    uint64_t attempts = 0, passed = 0, early = 0;
};

// Portable aligned storage: a block of at least `bytes` bytes on an `align` boundary (align a power of two), or
// nullptr. The size is rounded up to a multiple of align, as std::aligned_alloc requires. MSVC: _aligned_malloc;
// macOS: posix_memalign (std::aligned_alloc needs macOS 10.15+; posix_memalign exists on every macOS and its block is
// released by free() too); elsewhere: std::aligned_alloc. A block is released only by xdem_aligned_free (MSVC requires
// _aligned_free).
inline void* xdem_aligned_malloc(size_t align, size_t bytes) noexcept {
    if (bytes > SIZE_MAX - (align - 1)) return nullptr;
    const size_t r = std::max(align, (bytes + align - 1) / align * align);
#if defined(_MSC_VER)
    return _aligned_malloc(r, align);
#elif defined(__APPLE__)
    void* p = nullptr;
    return posix_memalign(&p, std::max(align, sizeof(void*)), r) == 0 ? p : nullptr;
#else
    return std::aligned_alloc(align, r);
#endif
}
inline void xdem_aligned_free(void* p) noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// Large-block storage of XdemUninitAlloc (xdem_sampler.cpp): a block of >= kHugeBytes is 2 MiB aligned and advised
// for transparent huge pages, as numpy does for its own arrays, so its first touch costs a fault per 2 MiB, not per
// 4 KiB page (measured: the fired output's page faults were the fired / sample gap on msc_d3 in a process whose heap
// the export had grown). huge_free releases a block of huge_alloc.
constexpr size_t kHugeBytes = size_t(1) << 22;
void* huge_alloc(size_t bytes);
void huge_free(void* p) noexcept;

// An allocator whose resize() leaves new elements uninitialized (the sampler writes every element); blocks of
// >= kHugeBytes come from huge_alloc.
template <class T>
struct XdemUninitAlloc : std::allocator<T> {
    template <class U>
    struct rebind {
        using other = XdemUninitAlloc<U>;
    };
    XdemUninitAlloc() = default;
    template <class U>
    XdemUninitAlloc(const XdemUninitAlloc<U>&) noexcept {}
    T* allocate(size_t n) {
        return n * sizeof(T) < kHugeBytes ? std::allocator<T>::allocate(n) : static_cast<T*>(huge_alloc(n * sizeof(T)));
    }
    void deallocate(T* p, size_t n) noexcept {
        if (n * sizeof(T) < kHugeBytes) std::allocator<T>::deallocate(p, n);
        else huge_free(p);
    }
    template <class U>
    void construct(U* p) noexcept {
        ::new (static_cast<void*>(p)) U;
    }
    template <class U, class... A>
    void construct(U* p, A&&... a) {
        ::new (static_cast<void*>(p)) U(std::forward<A>(a)...);
    }
};

// Per-thread scratch of the sampler: every block starts on its own 64-byte cache line and is padded to whole lines, so
// the small per-shot buffers of two sampler threads never share a line (Task 12 fix round 1: with default malloc
// placement the 16-byte eta planes / sign registers of different thread slots could land in one line, and the
// resulting false sharing cost +33-40 % at 8-16 threads on cultivation_d3_faithful, depending on the process's heap).
template <class T>
struct XdemLineAlloc {
    using value_type = T;
    XdemLineAlloc() = default;
    template <class U>
    XdemLineAlloc(const XdemLineAlloc<U>&) noexcept {}
    T* allocate(size_t n) {
        if (n > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
        void* p = xdem_aligned_malloc(64, n * sizeof(T));   // rounded up to whole lines (>= one line)
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) noexcept { xdem_aligned_free(p); }
    template <class U>
    bool operator==(const XdemLineAlloc<U>&) const noexcept { return true; }
    template <class U>
    bool operator!=(const XdemLineAlloc<U>&) const noexcept { return false; }
};
using XdemLineWords = std::vector<uint64_t, XdemLineAlloc<uint64_t>>;

// The fired lines of a sample call (sample / sample_ps with fired != nullptr): the CSR of each returned shot's fired
// line ids, ascending (= file order). Shot s fired ids[off[s] .. off[s + 1]); off has rows + 1 entries. A line fires
// at most once per shot, so no id repeats within a shot. Under postselection only the survivors' lists.
struct XdemFired {
    std::vector<int64_t, XdemUninitAlloc<int64_t>> off;
    std::vector<int32_t, XdemUninitAlloc<int32_t>> ids;
};

struct XdemCounters {
    uint64_t shots = 0;         // shots sampled (all calls)
    uint64_t ext_shots = 0;     // shots with >= 1 fired extended line
    uint64_t coin_shots = 0;    // shots whose law has coins
    uint64_t cache_hits = 0, cache_misses = 0, cache_clears = 0;   // general path only (fast path: no cache)
    uint64_t cache_entries = 0; // current, summed over thread slots
    uint64_t cache_cap = 0;     // per thread slot
    uint64_t num_cspaces = 0;
    uint64_t thread_slots = 0;
    // scoring with lines (Task 7/8; 0 until then): shots scored, R2b applied / withheld, coin after the correction,
    // lossy shots, shots outside the hypothesis' own law, the optimal correction's memo
    uint64_t lines_shots = 0, lines_applied = 0, lines_withheld = 0, lines_coin_after = 0;
    uint64_t lines_lossy = 0, lines_outside = 0, kmemo_entries = 0, kmemo_clears = 0;
    uint64_t kmemo_law_clears = 0;   // the extended-law cache cleared alone (full); kmemo_clears: memo + law cache
    uint64_t kfast_laws = 0;   // laws of [T, K] built in beta' coordinates (the fast law; 0: every one was generic)
};

// The compiled tables of a sampler as flat arrays (XdemSampler::device_tables; the GPU sampler uploads them). Copies:
// nothing is recomputed. Dimensions and the sampler's members of the same name (D = D_, ...). Chains (first appearance
// of p in file order; p = 0 lines are in none): chain c has p = chain_p[c] and members [chain_off[c], chain_off[c + 1]);
// member m: its flip entries [mem_beg[m], mem_beg[m] + mem_cnt[m]) of (ent_mask, ent_word) (internal layout: detector
// d = bit d, observable o = bit 64 WD + o; the arena's kEntPad zero entries included), its extended record's offset in
// xa (mem_xo; 0 = not extended) and its line id (mem_line). coord_words = XdemSampler::coord_words(). fast: the file is
// sampled by the ONE-WORD fast path (post_fast) or has no axes, i.e. coord_words == 1 (the GPU kernels' path); a
// multi-word file (coord_words >= 2) and the general path (force_general, coord_words = 0) have fast = false. One-word
// tables (fast and A > 0; else empty, and rho = rhoL = rmask = 0; r is always the gauge count): per axis Tc, Kc, bb;
// the basis beta (rho vectors of WF words, beta_D first, then rhoL beta_L vectors), betaO (their observable words, WO
// each), bLo / pL (beta_L's observable parts, WO words each, and their pivots: highest observable bit).
// Multi-word tables (coord_words = CW >= 2 and A > 0, build_multi's; else empty): rho, rhoL, bb, betaO, bLo and pL as
// above, and per axis a, CW words each: KcW (kappa_a, r bits) and GcW (img_a's gamma coordinates, rho bits); gamma (the
// canonical basis of span(img), rho vectors of WF words in ascending pivot order: img_a = XOR_{l in GcW[a]} gamma_l);
// GbW (beta_j's gamma coordinates, CW words per j < rho). fast stays false there (it names the one-word tables).
struct XdemDeviceTables {
    int D = 0, O = 0, W = 0, WD = 0, WO = 0, WF = 0, A = 0, WA = 0, rho = 0, rhoL = 0, r = 0;
    int coord_words = 0;
    bool fast = false;
    uint64_t rmask = 0;
    double mean_fired = 0, mean_ext = 0;
    std::vector<double> chain_p;
    std::vector<uint64_t> chain_off;
    std::vector<uint64_t> mem_beg;
    std::vector<uint32_t> mem_cnt, mem_xo;
    std::vector<int32_t> mem_line;
    std::vector<uint64_t> ent_mask, ent_word;
    std::vector<uint64_t> xa;
    std::vector<uint64_t> Tc, Kc;
    std::vector<uint8_t> bb;
    std::vector<uint64_t> beta, betaO, bLo;
    std::vector<int32_t> pL;
    std::vector<uint64_t> KcW, GcW, gamma, GbW;
};

// The postselection span test of a mask as flat arrays (XdemSampler::span_tables; XdemSpanTest's members): the row
// words holding a masked detector (widx) and the mask in them (wmask); rank basis vectors of widx.size() compressed
// words each (basis), each with its pivot (piv_word: compressed word, piv_bit: bit mask).
struct XdemSpanTables {
    std::vector<uint32_t> widx;
    std::vector<uint64_t> wmask, basis;
    std::vector<uint32_t> piv_word;
    std::vector<uint64_t> piv_bit;
    int rank = 0;
};

// score_lines (SPEC §4.3, §4.4, §4.5): the correction applied for the hypothesis.
enum class Correction : int { None = 0, R2b = 1, Optimal = 2 };

// The optimal correction's per-observable result (SPEC §4.3): lossy (no solution P_k: a fair coin for every K,
// P = S = empty, frame = ERRATA E4's c_k, the predicted value); P_k (pivot detector ids, ascending), S_k (axis ids,
// ascending), y_k = [L_k in Phi_{S_k}], frame = f_K,k = s.P_k xor [eps_k != (-1)^{y_k}].
struct KObs {
    bool lossy = false;
    std::vector<int32_t> P, S;
    uint8_t y = 0, frame = 0;
};

// correction()'s export (SPEC §4.6), flat. "optimal": per (shot, observable), row = shot * O + k: S_k =
// axes[axes_off[row] .. axes_off[row + 1]), P_k = piv[piv_off[row] .. piv_off[row + 1]), y, frame, lossy. "r2b": per
// shot: eta^> (late axis, exponent 1..3, ascending axis) = late_*[late_off[s] .. late_off[s + 1]), tau_K =
// tau_axis[tau_off[s] .. tau_off[s + 1]) (ascending), applied (0/1), c (shots * O bytes 0/1). The other correction's
// members stay empty.
struct XdemCorrectionOut {
    std::vector<int64_t> axes_off;
    std::vector<int32_t> axes;
    std::vector<int64_t> piv_off;
    std::vector<int32_t> piv;
    std::vector<uint8_t> y, frame, lossy;
    std::vector<int64_t> late_off;
    std::vector<int32_t> late_axis;
    std::vector<int8_t> late_eta;
    std::vector<int64_t> tau_off;
    std::vector<int32_t> tau_axis;
    std::vector<uint8_t> applied, c;
};
// score_lines options. mut: a test-of-test lever (0 = none; reachable from cpp/testapi only), one branch each in the
// scorer, never in sampling: K from E^< and E^> (not E^> alone); no guard; c = f_h|_O always; no cocycle sign in K's
// exponents; t_c - 1 in place of t_c.
struct XdemLinesOptions {
    Correction corr = Correction::Optimal;
    int mut = 0;
};
enum : int { kXdemMutEarlyLate = 1, kXdemMutNoGuard = 2, kXdemMutXorFlips = 3, kXdemMutNoCocycle = 4,
             kXdemMutTcMinus1 = 5 };
// score_lines' per-shot action (trace output): no K (correction "none", or eta^> = 0), K applied, K withheld by the
// guard
enum : int8_t { kLinesNone = 0, kLinesApplied = 1, kLinesWithheld = 2 };

class XdemSampler {
  public:
    // force_general (test-only oracle lever): use the cached exact-law path even when the fast path applies.
    // Takes the File by value and keeps only its header: the lines move into the sampler's own sparse arenas.
    explicit XdemSampler(File F, size_t cache_cap = kDefaultCacheCap, bool force_general = false);
    ~XdemSampler();
    XdemSampler(const XdemSampler&) = delete;
    XdemSampler& operator=(const XdemSampler&) = delete;

    bool extended() const { return F_.extended; }
    int num_detectors() const { return D_; }
    int num_observables() const { return O_; }
    int num_gauge() const { return F_.r; }
    int num_hidden() const { return F_.m; }
    int num_axes() const { return A_; }
    size_t num_lines() const { return L_; }
    size_t num_rotating() const { return n_rot_; }   // lines with a nonempty rot or late field
    int64_t tc() const { return F_.tc; }
    size_t det_bytes() const { return nbD_; }
    size_t obs_bytes() const { return nbO_; }

    // dets: shots * det_bytes() (stim bit-packed), obs: shots * obs_bytes(), cspace: shots. threads >= 1.
    // Shots [1024 first_block, 1024 first_block + shots) of the seed's stream.
    // fired != nullptr: also the shots' fired lines (dets / obs / cspace are the same bytes as without it). It runs a
    // separate instantiation of the block loop.
    void sample(size_t shots, uint64_t seed, int threads, uint8_t* dets, uint8_t* obs, int32_t* cspace,
                uint64_t first_block = 0, XdemFired* fired = nullptr);
    // Postselected: `shots` attempted shots. dets / obs / cspace have room for `shots` rows; their first
    // counts.passed rows receive the survivors (detectors with flags[d] != 0 all 0; D bytes), in shot order. The span
    // test is rebuilt only when the mask changes.
    void sample_ps(size_t shots, uint64_t seed, int threads, const std::vector<uint8_t>& flags, uint8_t* dets,
                   uint8_t* obs, int32_t* cspace, XdemPsCounts& counts, uint64_t first_block = 0,
                   XdemFired* fired = nullptr);

    // Global C table: entry i = dim(i) rows of obs_words() words (fully reduced RREF over the O observable bits,
    // pivot = highest bit, descending). Entry 0 is empty (dim 0). cspaces() is the live table WITHOUT the lock (for
    // callers that never sample concurrently); cspaces_snapshot() copies entries [first, first + n) under the lock.
    const std::vector<std::vector<uint64_t>>& cspaces() const { return cspaces_; }
    std::vector<std::vector<uint64_t>> cspaces_snapshot(size_t first = 0, size_t n = SIZE_MAX) const;
    size_t num_cspaces() const;
    size_t obs_words() const { return (size_t)WO_; }
    int cspace_dim(int id) const;

    // out[s] = 2^-dim C[cspace[s]] if (v_s XOR flips_s) in C, else 0. flips: shots * O bytes, each 0 or 1.
    void score(size_t shots, const uint8_t* obs, const int32_t* cspace, const uint8_t* flips, double* out) const;

    // Scoring with lines (SPEC §4.1, §4.4, §4.5). Per shot s: T = fired ids [foff[s], foff[s+1]) (sample(..., fired)),
    // d = dets row s (stim bit packing, det_bytes() per row), H = hypothesis ids [hoff[s], hoff[s+1]) (any order;
    // repeats cancel, SPEC §1.2). h = H sorted with pairs cancelled; f_h = its flips; a list's law is the §1.3 law of
    // its composite (§1.2), K joining it as a virtual line (tau = 0, E = eta^>) between lines tc - 1 and tc.
    //   "none": c = law(h)'s observable value at d when it is definite (dim C = 0, d in its detector support), else
    //           f_h|_O; out[s] = Pr(o = c | d, T) = 2^-dim C if (d, c) in law(T), else 0.
    //   "r2b":  eta^> = h's lines < tc composed by the cocycle on their late exponents alone (r2b_k); L0 = law(h),
    //           L1 = law(h with K); k0, k1 their dim C. Applied iff eta^> != 0 and k1 <= k0; c = the chosen law's
    //           definite value at d, else f_h|_O; out[s] = Pr(o = c | d, T with K if applied).
    //   "optimal": K = Z[f_K] K_diag of SPEC §4.3 (k_optimal) appended after T's lines, c = 0; out[s] = Pr(o = 0 | d,
    //           [T, K]) = 2^-dim C if (d, 0) in the law of the extended composite (K's strings join it as extra axes,
    //           their cocycle sign tau_T(S_k)), else 0. K_diag depends on h only through its composite (eta, tau):
    //           memoised per that content (cache_cap entries; cleared when full); d enters only through the frame bits.
    //           A lossy observable (ERRATA E4) gets no string and the frame bit c_j: the point of the support of
    //           law([h, K]) restricted to the lossy observables at d whose difference from f_h there is
    //           lexicographically least (observable 0 first): c = f_h on them when f_h is in that support; 0 when d
    //           is outside law(h)'s detector support (SPEC §7 Q1).
    // Refuses: corr = R2b on a file with r2b_available() false (XdemFileError naming the reason); a malformed CSR
    // (invalid_argument) or an id outside [0, num_lines) (out_of_range).
    // action (optional, shots): kLinesNone / Applied / Withheld; cbits (optional, shots * O bytes 0/1): the c used.
    void score_lines(size_t shots, const uint8_t* dets, const int64_t* foff, const int32_t* fids, size_t nfids,
                     const int64_t* hoff, const int32_t* hids, size_t nhids, double* out,
                     const XdemLinesOptions& opt = XdemLinesOptions(), int8_t* action = nullptr,
                     uint8_t* cbits = nullptr);
    // SPEC §4.3 steps 1-5 for one shot, from scratch (no memo): h = ids [0, nh) (any order; repeats cancel; checked
    // against [0, num_lines)), s = the detector row (ceil(D / 64) words, detector d = bit d). out: O entries; a lossy
    // observable's frame is ERRATA E4's c (from the generic law of [h, K], whatever path the sampler is on). Throws
    // std::logic_error on an internal check (a non-lossy rotated readout with a Z on a detector, LOOP §4(ii); an odd q;
    // an asymmetric coupling).
    void k_optimal(const int32_t* h, size_t nh, const uint64_t* s, std::vector<KObs>& out) const;
    // The correction for a physical implementation (SPEC §4.6), per shot of the hypothesis CSR and the detector rows
    // (the same arguments as score_lines' h and d). "optimal": k_optimal per shot (through score_lines' memo);
    // "r2b": SPEC §4.4 steps 1-4 (the same K, guard and c as score_lines); "none": invalid_argument ("nothing to
    // export"). Counts no lines_* counter (an export is not a scoring call); the memo counters do count.
    void correction(size_t shots, const uint8_t* dets, const int64_t* hoff, const int32_t* hids, size_t nhids,
                    Correction corr, XdemCorrectionOut& out);
    // The file facts of SPEC §4.4 (computed from the File at construction, before its lines are released).
    const std::vector<int>& late_axes() const { return late_axes_; }
    const std::pair<bool, std::string>& r2b_available() const { return r2b_; }
    int invisible_residues() const { return inv_; }   // rank img[A1] - rank img[A1]|_D (xdem_file.hpp)

    XdemCounters counters() const;
    // true: the coordinate path (every file); false: the cached exact-law path (the test lever force_general).
    bool fast_path() const { return fast_; }
    // the fast path's words per coordinate vector: 1 (rho <= 64 and r <= 64, or no axes), else ceil(max(r, rho) / 64)
    // (5 and 7 padded to 6 and 8); 0 under force_general
    int coord_words() const { return !fast_ ? 0 : CW_ ? CW_ : 1; }

    // ---- export for the GPU sampler (reads only; no output byte or stream changes)
    // The compiled tables as flat copies (XdemDeviceTables).
    XdemDeviceTables device_tables() const;
    // The span test sample_ps builds for the mask flags (D bytes, nonzero = detector in M), built afresh: the cached
    // one is not touched.
    XdemSpanTables span_tables(const std::vector<uint8_t>& flags) const;
    // The global C id of a fast-path status key: key = nk vectors of coord_words() words each (one word on the
    // one-word path, CW_ on a multi-word file), the fully reduced basis of C in beta_L coordinates (bit j of a vector
    // = beta vector j, j in [rho - rhoL, rho); pivot = highest bit, distinct, descending), the GPU kernels' key. It is
    // converted here to the thread slots' canonical form (the y-RREF over beta_L, fast_cspace's key, with its hash), so
    // the id is the one sample() gives the same C. A C not yet in the table is appended (under mu_). nk = 0: 0. Throws
    // invalid_argument off the fast path (general path, or no axes) or on a key that is not such a basis.
    int cspace_of_beta_key(const uint64_t* key, int nk);

    // ---- test hooks (cpp/testapi only; same code paths as sample)
    // Each thread slot's per-shot scratch blocks written by its worker (eta planes, dirty bitmap, sign register,
    // odd axes, C key, the slot object itself) as (address, bytes), slot-major, bytes = the block padded to whole
    // 64-byte lines; no allocation. A test checks that no two slots share a line (false sharing, Task 12 fr1).
    std::vector<std::vector<std::pair<uintptr_t, size_t>>> slot_scratch_spans() const;
    // The general path's gauge re-basing (build_law_axes): (sum of the file's kappa weights, sum of the re-based
    // kappa weights); (0, 0) on the fast path.
    std::pair<long long, long long> law_kappa_weights() const;
    // Per-line fire counts over `shots` shots of the stream (the production noise walk), and for each (i, j) in
    // pairs the number of shots where both lines fired.
    void fire_counts(size_t shots, uint64_t seed, uint64_t first_block, std::vector<uint64_t>& counts,
                     const std::vector<std::pair<int, int>>& pairs, std::vector<uint64_t>& pair_counts);
    // Every shot fires exactly `line_ids` (any order, repeats cancel: SPEC §1.2); post-processing (composition, law,
    // coins, status) is the production path. Output layout as sample().
    void sample_fixed(const std::vector<int>& line_ids, size_t shots, uint64_t seed, uint8_t* dets, uint8_t* obs,
                      int32_t* cspace);
    // The production status of a forced fired multiset: the global C id, plus the law used (offset at the given
    // multiset's flip, coin basis; not reduced) in File bit layout (detector d = bit d, observable o = bit D+o).
    int status_of(const std::vector<int>& line_ids, Bits& offset, std::vector<Bits>& coins);
    // The law of the same multiset with no coin draw (post_fast's LAW instantiation: offset + the kernel basis of N_O,
    // the coins rebuilt from it; general path: the cache entry): offset, a basis of the coin span; returns dim C.
    int law_of(const std::vector<int>& line_ids, Bits& offset, std::vector<Bits>& coins);
    // The span test's premise on the stream's shots (1 thread slot): for every shot with an extended line, the
    // correction's change of the detector row lies in span(img_a)_D. Returns the shots where it fails;
    // *ext_shots = shots with >= 1 extended line.
    uint64_t span_check(size_t shots, uint64_t seed, uint64_t* ext_shots);

  private:
    // per member line of a chain: its flip-entry range (begin, count) and its extended record's offset in xa_ (0: not
    // extended), so a fired slot reaches everything the walk needs in one 16-byte load; lines = the members' ids
    // (the fired lists and fire_counts)
    struct Member {
        uint64_t beg;
        uint32_t cnt;
        uint32_t xo;
    };
    struct Chain {
        double p = 0, lp = 0;
        bool always = false;
        std::vector<Member> mem;
        std::vector<int> lines;
    };
    struct ThreadState;

    template <class Fire>
    void walk_chains(uint64_t seed, uint64_t block, Fire&& fire) const;
    void run_blocks(size_t shots, uint64_t seed, int threads, uint64_t first_block, uint8_t* dets, uint8_t* obs,
                    int32_t* cspace, const std::vector<int>* fixed, const XdemSpanTest* ps = nullptr,
                    XdemPsCounts* pc = nullptr, XdemFired* fired = nullptr);
    // PS: survivors to consecutive rows (their number to kept); KEEP: also the fired lines (T.fcnt / T.fids)
    template <bool PS, bool KEEP = false>
    void run_one_block(ThreadState& T, uint64_t seed, uint64_t block, size_t n, uint8_t* dets, uint8_t* obs,
                       int32_t* cspace, std::vector<int>& first, const std::vector<int>* fixed,
                       const XdemSpanTest* ps, size_t& kept);
    // The block's extended (shot << 32 | record offset) pairs T.epair[0, np) as a per-shot CSR: T.exo, T.epos (shot s:
    // [epos[s], epos[s + 1])), a counting sort by shot (stable: walk order). A shot's list is sorted into file order
    // (record offsets increase with the line id) where it is composed, so a shot never composed is never sorted.
    void sort_ext(ThreadState& T, size_t np, size_t n) const;
    // The shot's fired extended lines (record offsets ascending = file order) composed by the cocycle into T's eta
    // planes (bit-sliced eta mod 4, dirty words marked); the sign register is cleared after. n >= 1.
    void compose_ext(ThreadState& T, const uint32_t* xo, uint32_t n) const;
    // A forced multiset (status_of / law_of / sample_fixed): sorted, pairs cancelled; flips XOR-ed into row (if
    // non-null), the extended lines' record offsets to ext. Throws out_of_range on an id outside [0, num_lines).
    void canonical_ids(const std::vector<int>& line_ids, const char* who, uint64_t* row,
                       std::vector<uint32_t>& ext) const;
    // From T's eta planes (compose_ext; drained here): XOR img_a into the row for eta_a in {2, 3}, and return the law
    // entry of the odd-axis set (-1 when no axis is odd: no coins, C = {0}).
    int resolve(ThreadState& T, uint64_t* row);
    // fast path: post-process a shot row in place from T's eta planes (drained; coin word from *cpos when cpos !=
    // nullptr, else coins = 0); returns the shot's local C id; coins_out (status_of) receives the r coin vectors in
    // beta coordinates. PS: -1 when the final row fails zt's mask. LAW (law_of; cpos = coins_out = nullptr): row <- the
    // law's offset, the kernel basis of N_O to T's law scratch; returns 1 when the law has coins, else 0.
    template <bool PS = false, bool LAW = false>
    int post_fast(ThreadState& T, uint64_t* row, uint64_t* cpos, std::vector<uint64_t>* coins_out,
                  const XdemSpanTest* zt = nullptr);
    // the multi-word fast path (CW_ >= 2 words per coordinate vector; NW = CW_, or 0: CW_ at run time): post_fast's
    // contract, with the general path's coin stream (ceil(k / 64) words of *cpos over the canonical coins); coins_out
    // (status_of) receives the k canonical coins in gamma coordinates (CW_ words each, descending pivot). post_mw
    // dispatches on CW_ (instances 2, 3, 4, 6, 8; else run time). LAW (law_row / law_of; cpos = nullptr): row <- the
    // law's offset, the kernel basis in beta coordinates to T's multi-word law scratch, dim C; returns 3 when the law
    // has coins, else 0.
    template <int NW, bool PS, bool LAW>
    int post_multi(ThreadState& T, uint64_t* row, uint64_t* cpos, std::vector<uint64_t>* coins_out,
                   const XdemSpanTest* zt);
    template <bool PS = false, bool LAW = false>
    int post_mw(ThreadState& T, uint64_t* row, uint64_t* cpos, std::vector<uint64_t>* coins_out,
                const XdemSpanTest* zt = nullptr);
    void build_multi();               // build_beta's multi-word coordinates (fast_ = true, CW_ >= 2)
    // the gauge basis in image coordinates (GSL; at load): out (ow words per axis) <- kappa'_a = (t_a | kappa_a on a
    // completing set of gauge coordinates), t = the axes' image coordinates (tw words per axis, rho_ bits); tb = their
    // beta coordinates (bw words per axis, beta_D then beta_L), creator[j] = the axis that created beta_j
    void image_kappa(const uint64_t* tb, size_t bw, const std::vector<int>& creator, const uint64_t* t, size_t tw,
                     uint64_t* out, size_t ow) const;
    void xor_gamma(uint64_t* row, const uint64_t* z) const;   // row ^= sum_j z_j gamma_j (z: CW_ words)
    size_t mw_words() const;          // a thread slot's multi-word scratch (words)
    // a thread slot's C table (SCF): the id of the canonical form key[0, n) (words) with hash h (-1: absent);
    // local_insert appends an absent form (key copied, its content moved in, remap -1) and returns its id
    int local_cspace(ThreadState& T, const uint64_t* key, size_t n, uint64_t h) const;
    int local_insert(ThreadState& T, const uint64_t* key, size_t n, uint64_t h, std::vector<uint64_t>&& content);
    // the fast paths' status C from its y-RREF (nr rows in T.ckey): the slot's id (a new C: its content)
    int fast_cspace(ThreadState& T, int nr);
    void build_beta(bool force_general);
    void build_ycoords();   // status C's canonical coordinates over beta_L (yW_, yA_, yE_), at load
    void build_law_axes();
    void construct(bool force_general);   // the constructor's body (a std::bad_alloc there -> XdemCapacityError)
    void xor_beta(uint64_t* row, uint64_t z) const;
    int lookup(ThreadState& T, const uint64_t* key);
    // the general path's correction of a composed row (run_one_block and span_check): the cache entry idx's offset
    // and its coin combination, reading coin words at cpos; returns the entry's local C id
    int apply_entry(ThreadState& T, uint64_t* row, int idx, uint64_t& cpos) const;
    // the global id of a thread slot's C content c with its slot hash h: a new C moves c into the table, a known one
    // frees c. Invariant: every thread slot of one sampler keys C by the same kind of canonical form (the y-RREF on the
    // fast path, the content itself under force_general), so h is a function of C. A sampler mixing the two kinds
    // would give one C two hashes; the probe stops at an empty slot, so the merge would append a duplicate C entry
    // (an output change).
    int global_cspace(std::vector<uint64_t>& c, uint64_t h);
    ThreadState& slot(int t);
    // law_of's law of a composed row (T's eta planes set), no coin draw: row <- the law's offset; returns the kind
    // (0: a point law; 1: fast path, the kernel basis in T's law scratch; 2: general path, *entry_out its entry;
    // 3: multi-word fast path, the kernel basis in beta coordinates in T's multi-word law scratch)
    int law_row(ThreadState& T, uint64_t* row, int* entry_out);
    int law_dim(ThreadState& T, int kind, int entry);
    // the law of T's composite (eta planes set; drained) with an explicit coin basis: row <- the offset (flips on
    // entry), rows <- coin rows (WF_ words each, internal layout); returns the kind (law_row's, 3: multi-word); law_of
    // and k_support
    int law_coins(ThreadState& T, uint64_t* row, std::vector<uint64_t>& rows, int* entry_out);
    // score_lines' law readers (after the law_row and law_dim they refer to; ported from the v1 sampler):
    // o (WO_ words) <- an observable value of {o : (d, o) in law} (the definite prediction when dim C = 0); false
    // when d (WD_ words) is outside the law's detector support
    bool law_given(ThreadState& T, int kind, const uint64_t* off, const uint64_t* d, uint64_t* o);
    // (d, c) in the law?
    bool law_has(ThreadState& T, int kind, const uint64_t* off, const uint64_t* d, const uint64_t* c);
    // x (WO_ words; destroyed) in C?
    bool law_in_c(ThreadState& T, int kind, uint64_t* x) const;
    // fast path: the beta_D coordinates of d ^ off_D; ok = it lies in span(beta_D) on the detectors
    uint64_t beta_d_coords(ThreadState& T, const uint64_t* off, const uint64_t* d, bool& ok) const;
    // the same in CW_ words (the multi-word law, kind 3): sD <- the coordinates; true iff ok
    bool beta_d_coords_w(ThreadState& T, const uint64_t* off, const uint64_t* d, uint64_t* sD) const;
    // SPEC §4.4 step 1: eta^> (nonzero (axis, exponent mod 4), ascending axis) of h's lines below tc (h[0, nh), sorted,
    // cancelled; the caller stops at tc) by the cocycle restricted to the late parts, and tau_K = their tau on the late
    // axes (WA_ words; nullptr: not computed). tau_K is for the export: under §4.4's availability no line >= tc
    // rotates, so no later line reads it. mut: kXdemMutEarlyLate (E^< too), kXdemMutNoCocycle (no sign).
    void r2b_k(const int32_t* h, size_t nh, std::vector<std::pair<int, int>>& eta_late, std::vector<uint64_t>* tau_k,
               int mut = 0) const;
    // The law of a canonical id list ids[0, n) (sorted, cancelled) with K (eta^>, as exponent groups kE: (w, odd bits,
    // 2-bits) per axis word) inserted before ids[at] (kE empty: no K): the composite into T's eta planes, then law_row
    // (fast path: through a content-keyed memo). row: the list's flips on entry, the law's offset on return. Returns
    // the law kind (0: a point law, no rotation; *entry_out: law_row's).
    int law_list(ThreadState& T, const int32_t* ids, size_t n, size_t at, const std::vector<uint64_t>& kE,
                 uint64_t* row, int* entry_out);
    // score_lines' body (under mu_); count = false: no lines_* counter changes (correction()'s r2b export)
    void score_lines_locked(size_t shots, const uint8_t* dets, const int64_t* foff, const int32_t* fids, size_t nfids,
                            const int64_t* hoff, const int32_t* hids, size_t nhids, double* out,
                            const XdemLinesOptions& opt, int8_t* action, uint8_t* cbits, bool count);
    // "optimal" (SPEC §4.3), under mu_ (CSRs checked by the caller): per shot the memo entry of h's composite
    // (KMemo, xdem_sampler.cpp) and its frame bits at d; out != nullptr: the law of [T, K] at (d, 0) to out (T = the
    // fired CSR) and the lines_* counters count; cx != nullptr: each shot's O rows appended to cx's "optimal" members
    struct KEntry;
    struct KMemo;
    // the memo key of a canonical id list (sorted, cancelled): its composite's eta planes (odd, 2-bit) and tau on the
    // rotated axes, 3 * ceil(A / 64) words (key); K depends on h only through it
    void k_key(const int32_t* ids, size_t n, uint64_t* key) const;
    struct KScratch;
    struct KFast;
    void k_fast_setup(KFast& F) const;   // beta' (Task 8 fix round 1); F.ok = false only under force_general
    void k_fast_setup1(KFast& F) const;  // its one-word instance (F.ok = false: rho' > 64)
    // GSL: kappa' = [kappa | delta] in image coordinates for the law of [T, K] (tb: the file axes' beta' coordinates,
    // eo: e_{L_o}'s; w words per vector): F.S and the file axes' kappa'' (F.KcB, or F.Kc1 for one word)
    void k_fast_rebase(KFast& F, const uint64_t* tb, const uint64_t* eo, size_t w) const;
    // row (WF_ words) ^= beta'.z for beta' coordinates z (F.cw words)
    void k_xor_beta_x(const KFast& F, const uint64_t* z, uint64_t* row) const;
    void k_build(const uint64_t* key, KEntry& E, KScratch& X) const;   // h's key -> the entry (X: buffers)
    void k_support(KEntry& E);                            // law(h)'s detector support (lines_outside), on first use
    // fobs (WO_ words) <- the frame bits f_K at d (WD_ words) for h's flips fh (WF_ words)
    void k_frames(const KEntry& E, const uint64_t* fh, const uint64_t* d, uint64_t* fobs) const;
    struct XLaw;   // the law of [T, K] for one odd set (xdem_sampler.cpp)
    void k_hi_images(const uint64_t* key, uint64_t* row) const;                 // row ^= img_a, eta_a in {2, 3}
    void k_strings(const KEntry& E, const uint8_t* sg, uint64_t* row) const;    // K's strings' {2, 3} images
    void k_xax(KEntry& E) const;                                                // K's strings as File-layout axes
    void k_law_generic(KEntry& E, const uint64_t* odd, XLaw& X, LawScratch& LS) const;   // law_into over [T, K]
    // ERRATA E4: the lossy observables' frame. k_lossy_setup (once per entry, from X = law([h, K]); KF serves a
    // fast law) and k_lossy_frames (per shot: fobs's lossy bits <- c at d, the others kept; u: WF + WO scratch words)
    void k_lossy_setup(KEntry& E, const XLaw& X, const KFast& KF) const;
    void k_lossy_frames(const KEntry& E, const uint64_t* fh, const uint64_t* d, uint64_t* fobs, uint64_t* u) const;
    void score_optimal(size_t shots, const uint8_t* dets, const int64_t* foff, const int32_t* fids,
                       const int64_t* hoff, const int32_t* hids, double* out, XdemCorrectionOut* cx);

    File F_;   // the header only (D, O, r, m, tc, axes; law_into reads it): its lines are released
    size_t L_ = 0, n_rot_ = 0;
    int D_, O_, WD_, WO_, WF_, W_;
    size_t nbD_, nbO_;
    size_t cap_;
    int A_ = 0, WA_ = 0;
    std::vector<uint64_t> img_;     // per axis WF_ words (internal layout)
    // The extended lines' records, one contiguous run of words per line in file order (xo_of_[i] = line i's offset,
    // 0 = not extended; xa_[0] is a pad so 0 names no record): a header nE | nT << 32, then nE exponent groups (w, the
    // bits a of axis word w with E_a odd, those with E_a & 2), E = its total exponent per axis (E^< + E^>) mod 4, then
    // nT sign register entries (w, mask): its tau restricted to the rotated axes; words ascending, masks nonzero.
    // Extended = nE or nT > 0. One record per fired line keeps the composition's reads in one place (the walk carries
    // the offset), and composition costs one step per axis word, not per axis (compose_ext).
    std::vector<uint64_t> xa_;
    std::vector<uint32_t> xo_of_;
    // fast path (build_beta)
    bool fast_ = false;
    int r_ = 0, rho_ = 0, rhoL_ = 0, ntab_ = 0;
    uint64_t rmask_ = 0;
    std::vector<uint64_t> Tc_, Kc_;   // per axis: img coordinates over beta, kappa word
    std::vector<uint8_t> bb_;
    std::vector<uint64_t> btab_;      // ntab_ * 256 * WF_: byte tables of sum_j z_j beta_j
    std::vector<uint64_t> betaO_;     // rho_ * WO_: observable part of beta_j
    // Per line, sparse: the shot-row bits it flips (internal layout: detector d = bit d, observable o = bit
    // 64 WD_ + o), as (word, mask) entries lent_[loff_[i] .. loff_[i + 1]): one entry per row word holding any of the
    // line's bits (words ascending, masks nonzero). The arena ends with kEntPad zero entries (mask 0, word 0), so the
    // hot loop reads a line's first kEntPad entries unconditionally (an entry past the line's own is a no-op).
    struct LineEnt {
        uint64_t mask;
        uint64_t word;
    };
    static constexpr int kEntPad = 4;
    std::vector<uint64_t> loff_;
    std::vector<LineEnt> lent_;
    std::vector<Chain> chains_;
    std::vector<std::unique_ptr<ThreadState>> ts_;
    std::vector<std::vector<uint64_t>> cspaces_;
    std::vector<int> cdim_;
    std::vector<std::vector<int>> cpiv_;
    struct VecHash {
        size_t operator()(const std::vector<uint64_t>& v) const noexcept;
    };
    // the C table's index (SCF): open addressing over the ids (-1 = empty; load <= 1/2, cslot_mask_ = size - 1), each
    // id's slot hash in chash_ (compared before the content)
    std::vector<int32_t> cslot_;
    std::vector<uint64_t> chash_;
    size_t cslot_mask_ = 0;
    uint64_t shots_total_ = 0;
    std::vector<uint8_t> ps_flags_;   // the last postselection mask (D bytes, 0/1) and its span test
    XdemSpanTest ps_;
    bool ps_built_ = false;
    mutable std::mutex mu_;
    double mean_fired_ = 0, mean_ext_ = 0;   // sums of the lines' p (all lines; extended lines): expected per shot
    // score_lines (Task 7; after the sampling members, so the block loop's members keep their offsets). Per line, from
    // the File: E^< (rot) entries, the E^> (late) offsets (its exponents are kl_'s word groups), and its full tau (all
    // axes; the sampler's records keep tau on the rotated axes only; Task 8's hidden-syndrome signs need it whole). The
    // file facts of SPEC §4.4. Fast path: beta_D's detector parts (WD_ words each) with their pivots (lowest detector
    // bit), beta_L's observable parts (WO_ words each) with their pivots (highest observable bit), in build order.
    // Counters (under mu_).
    std::vector<uint64_t> early_off_, late_off_, tau_full_off_;
    std::vector<Rot> early_;
    std::vector<int32_t> tau_full_;
    std::vector<int> late_axes_;
    std::pair<bool, std::string> r2b_;
    int inv_ = 0;
    std::vector<uint64_t> bDd_, bLo_;
    std::vector<int> pD_, pL_;
    // r2b_k: per line its E^> as word groups (kl_of_[i] = offset in kl_, 0 = none: a count, then (w, bits with e odd,
    // bits with e & 2) per word, ascending); its scratch (score_lines runs under mu_): eta^> as two bit planes over the
    // axes, the sign register, the planes' dirty-word bitmap; zero between calls
    std::vector<uint64_t> kl_;
    std::vector<uint32_t> kl_of_;
    mutable std::vector<uint64_t> k_lo_, k_hi_, k_sg_, k_dm_;
    uint64_t lines_shots_ = 0, lines_applied_ = 0, lines_withheld_ = 0, lines_coin_after_ = 0, lines_outside_ = 0;
    uint64_t lines_lossy_ = 0;
    // "optimal" (Task 8): the memo of K per h's extended lines and the extended-law cache (xdem_sampler.cpp)
    std::unique_ptr<KMemo> km_;
    // general path (Task 12 fix round 1): per axis, kappa in a lower-weight gauge basis (lookup()'s law_into reads it
    // through AxisView::kbase in place of F_.axes[a].k; img, hidden and b stay F_.axes'). Empty on the fast path.
    std::vector<std::vector<uint64_t>> law_k_;
    // (XFM; last, so the members above keep their offsets) one-word coordinates in use (post_fast and its LAW instance,
    // Tc_ / Kc_ / btab_ built); false when CW_ >= 2 or under force_general
    bool fast1_ = false;
    // multi-word fast path (build_multi; CW_ = 0 otherwise): CW_ words per vector. Per axis: kappa in image
    // coordinates (KgW_, image_kappa over GcW_), img coordinates over gamma (GcW_); per beta_j its gamma coordinates
    // (GbW_); gtab_: gntab_ * 256 * WF_ byte tables of sum_l z_l gamma_l
    int CW_ = 0, gntab_ = 0;
    std::vector<uint64_t> KgW_, GcW_, GbW_, gtab_;
    // (GSL) one-word fast path: per axis kappa in image coordinates (image_kappa over Tc_); the law matrix reads it,
    // the coin draw and coins_out read the file's kappa (Kc_)
    std::vector<uint64_t> Kg_;
    // (GSL) the axis that created each beta vector (beta_D, then beta_L): beta_j's creator's highest beta coordinate
    // is j
    std::vector<int> creator_;
    // (SCF) status C's canonical coordinates over beta_L (build_ycoords): E = the fully reduced echelon of span(beta_L's
    // observable parts) (pivot = highest bit), rows by ascending pivot P_k; a beta_L combination x has y coordinates
    // y_k = bit P_k of its observable part (yA_: beta_L_j's y, yW_ words each). E's pivot bits lie only in their own
    // rows, so the y-RREF (pivot = highest bit) of a subspace maps row by row (y -> sum_k y_k E_k, yE_: rhoL_ x WO_
    // words) onto its observable parts' fully reduced RREF in descending pivot order: the C table's content
    int yW_ = 0;
    std::vector<uint64_t> yA_, yE_;
};

// out[s] = 1 if v_s == flips_s (all O observables) else 0. obs: shots * ceil(O/8) bit-packed, flips: shots * O.
void score_plain(size_t shots, int O, const uint8_t* obs, const uint8_t* flips, double* out);

}  // namespace qeccore::xdem
