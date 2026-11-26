// xdem_kernels.h: the GPU xdem sampler's kernels (design spec 2026-11-23-gpu-xdem-sampler, sections 5.3 to 5.6).
//
// One source for every target: Metal (the launcher compiles this text at run time), CUDA (CuPy RawModule) and plain
// C++ (the test reference, cpp/testapi/xdem_gpu_ref_py.cpp). It follows the CPU sampler, cpp/src/xdem_sampler.cpp:
//   K1 xk_walk_group    walk_chains + sort_ext       (one group of threads per block of B shots, its threads taking
//                       the block's segments of chains in turn; under XK_WALK_SCREEN also K2's postselection span
//                       test)
//   K2 xk_resolve_shot  sort_ids + compose_ext + post_fast (post_multi when coord_words >= 2), xor_beta from the
//                       beta vectors (xor_gamma from the gamma basis); under
//                       XK_MODE_POSTSELECT run_one_block's postselected loop: span_reduce, then the decision
//                                                    (one thread per shot); every surviving shot also gets
//                       XK_FLAG_ZERO_PASS when the zero decoder passes it (score's rule, law_in_c's reduction);
//                       the one-word law from the single-line table where it applies, and under XK_MODE_DEFER a shot
//                       whose one-word law must be computed is queued (lawq) instead; under XK_MODE_NOKEY (the
//                       estimate) the one-word law gives dim C and the zero decoder's verdict without C's key;
//                       under XK_MODE_NOSTATUS (sample without the status, with XK_MODE_NOKEY and XK_MODE_EMIT) no
//                       law builds C's key (the multi-word law neither) and none needs the zero decoder's verdict;
//                       a file with a structured law (dims[XK_DIM_ST] = 1, xtim.gpu.structure) takes xk_law_struct
//                       instead of the generic one-word law (xk_law_fast); a multi-word file with one takes
//                       xk_law_struct_mw instead of xk_law_multi, in K2b only (XK_MODE_DEFER required: K2 queues every
//                       shot with a fired line)
//   K2b xk_resolve_law  K2's queued shots: their law and outputs (ceil(nshots / 4) threads, thread t taking the
//                       queue entries t, t + G, t + 2 G, ...)
//   K3 xk_reduce_block  a block's survivors and zero-decoder passes by dim C into hist (one thread per block)
//   K4 xk_emit          a survivor's detector bytes, observable bytes and C id (from the read-only table ctab) to
//                       its output row (one thread per shot), in sample's layouts; under XK_MODE_NOSTATUS no C id
//   K5 xk_reduce_groups K3's counters and the walk's overflow codes summed over each of XK_HIST_GROUPS groups of
//                       blocks into hist_group      (one thread per (group, hist_total slot))
//   K6 xk_reduce_total  the groups' sums into hist_total, the batch's totals (one thread per hist_total slot)
//   K7 xk_scan_blocks   under postselection, each run of XK_EMIT_RUN shots' first output row (one thread per
//                       block)
// Both CPU fast paths are supported: one coordinate word (rho <= 64, r <= 64, or no axes: post_fast) and several
// (coord_words = cw >= 2: post_multi, any number of gauge qubits). The general path (coord_words = 0) is not.
//
// Language rules (the common subset of MSL, CUDA C and C++17): integer arithmetic only, no floating point; no
// recursion, no dynamic allocation, every loop bounded; atomics only through the helpers spelled per target
// (xk_atomic_inc: K2's queue and K4's list; xk_tg_inc, xk_tg_xor on group memory and xk_dev_xor on a device word: K1);
// every device-buffer pointer is declared XK_BUF, every pointer to thread-local storage XK_LOC, every pointer to group
// memory XK_TG (Metal needs the three address spaces spelled out), every function XK_FN. No #include: the defaults
// below pick the target by its predefined macro, and a launcher may define XK_BUF, XK_LOC, XK_FN before the text
// instead.
// Group kernels (K1 only): the entry stub declares the group's memory, u32[XK_CAP_SHM] (Metal: threadgroup, CUDA:
// __shared__), and calls xk_walk_group(ctx, group, thread, threads, memory) from every thread of the group. The
// kernel runs as phases separated by XK_BARRIER() (threadgroup_barrier / __syncthreads, which every thread of the
// group reaches: no thread returns before the last phase), each phase a function of (group, thread, phase). On a
// target without XK_PARALLEL_GROUP (the C++ reference) xk_walk_group runs one group: each phase as a loop over the
// threads in turn, which is a legal schedule of the parallel one. So no phase may wait on another thread of the same
// phase, and no output may depend on the number of threads or on the order of the atomics (K1's lists are sorted per
// shot, its rows are XORs; which thread walks a segment does not change its draws).
//
// Capacities: the launcher #defines XK_CAP_W, XK_CAP_WA, XK_CAP_RHO, XK_CAP_E, XK_CAP_J, XK_CAP_SHM, XK_CAP_CW,
// XK_CAP_DIMC before this text; the actual sizes are read at run time from dims[]. A launcher's compile sets every
// capacity >= the matching size (W, WA, rho of a one-word file, E, the largest segment J, the walk's group memory
// xk_shm_need(B, W, walk) in u32 words (<= XK_MAX_SHM = 8192: 32 KiB), coord_words, dims[XK_DIM_DIMC]); a multi-word
// file's rho has no capacity (XK_CAP_RHO is 1 there: the one-word law is not run). The structured law's capacities
// XK_CAP_ST_NC, XK_CAP_ST_NG, XK_CAP_ST_T, XK_CAP_ST_N are optional: defined (all four) only for a file that runs it,
// which compiles it in; a multi-word file's also defines XK_CAP_ST_W (its slot words, <= XK_MAX_ST_W = 4). Output
// never depends on the capacities: where a size exceeds its capacity anyway (xk_caps_ok), the kernels compute
// nothing and say so instead of answering differently: K1 sets overflow[block] = XK_OVERFLOW_CAPACITY with empty
// lists, K2 sets cdim[shot] = XK_CDIM_CAPACITY and leaves the row as it was.
//
// Thread-local memory: only XK_CAP_WA, XK_CAP_RHO, XK_CAP_DIMC and the structured law's XK_CAP_ST_NC, _NG, _T, _W size
// local arrays, and all are bounded at compile time (XK_CAP_WA <= XK_MAX_WA = 64, XK_CAP_RHO <= 64: the one-word law;
// XK_CAP_DIMC <= XK_MAX_DIMC = 253; XK_CAP_ST_NC, _NG, _T, _W <= 8, 8, 32, 4, an #error otherwise;
// xtim.gpu.supports refuses a file with WA > 64, i.e. more than 4096 axes, or rhoL > 253); XK_CAP_J sizes K1's
// u64[(XK_CAP_J + 8) / 8] arrays of a pass's draws and threshold top bytes (XK_CAP_J < 64: at most 64 bytes each).
// XK_CAP_W, XK_CAP_E and XK_CAP_CW size nothing (the walk's per-record shot list is the device buffer psh, the row is
// corrected
// in place, the zero decoder's test reduces without a copy, the multi-word law works in the device buffer mws); they
// are only checked. Worst case per thread, at XK_CAP_WA = XK_CAP_RHO = 64 (8-byte words): K2 (resolve) 7936 bytes of
// arrays (H: 512; lo, hi and compose's sign planes: 3 * 512; law_fast's eleven u64[rho] arrays (its second
// implementation, XK_CAP_RHO > 32): 5632, and depj u32[rho]: 256); at XK_CAP_RHO <= 32 law_fast's first
// implementation holds col u64[rho], S and Bq u32[rho] (at most 512 bytes, in registers once its loops are
// unrolled); xk_law_struct (instead of law_fast, at most 187 u64 and 16 u32: 1560 bytes at the maxima, 696 at the
// merlin_bh files' 4, 3, 8: Wt u64[T]; C, R, Cp, G, Q, X, BI, BL, GR, GP, Lam and xk_st_pivots' two u64[NC] or [NG];
// Ev, Ep, wr u64[NG + 1 + NC]; n, GC u32[NC]); K2b: xk_law_struct_mw (the same arrays with slot vectors of
// XK_CAP_ST_W words and lifted vectors of XK_CAP_ST_W + 1: at most 3764 bytes at the maxima W, NC, NG, T = 4, 8, 8, 32,
// about 1100 at merlin_bh70 / bh100's 2, 4, 3, 12; no mws); K3 (reduce) h u32[XK_CAP_DIMC + 3]: 268 bytes at 64,
// 1024 at 253; K1 (walk): 2 u64[(XK_CAP_J + 8) / 8] (at most 128 bytes); K4, K5, K6, K7: none.
// Plus a few hundred bytes of XkCtx and scalars. Every bundled example has WA <= 2 and rho <= 15: under 1 KiB.
// Why the multi-word law uses device scratch rather than thread-local arrays (measured on an Apple M3 Max, 65536
// threads each touching an array of n u64 words): Metal refuses a 512 KiB thread-local array at pipeline creation
// ("exceeds available stack space"; 128 KiB compiles), and from n = 1024 the thread-local array is slower than the
// interleaved device buffer (1.30 vs 0.56 ns per access at n = 1024, 5.19 vs 1.44 at n = 16384; equal at n = 256).
// So mws (below) sets no limit on rho or cw.
//
// Thread ids: K1 (its groups), K3 and K7 ignore block >= nblocks, K2 and K4 shot >= nshots, so a launcher may round
// its grid up. K1 takes any number of threads per group (1 to XK_MAX_BLOCK = 1024).
//
// Random numbers (keyed counter generator of cpp/include/qeccore/keyed_rng.hpp, xk_mix = krng::mix): item i of block
// b (K1's unit of work, a run of segments; xtim.gpu.tables.walk_plan) draws mix(pos + k gamma), k = 1, 2, ..., pos =
// mix(bkey + (i + 1) XK_ITEM_GAMMA), bkey = mix(noise_key ^ mix(b)) (K1, below); the coin word of global shot g is
// mix(coin_key + (g + 1) gamma) & rmask. noise_key = stream_key(seed, kTagGpuNoise), coin_key = stream_key(seed,
// kTagGpuCoin) (xtim/gpu/tables.py computes both). Block b of a seed depends only on (seed, b, B).
//
// Buffers (little-endian; the names are the launcher's):
//   dims     u32[XK_DIM_COUNT]  sizes, mode and the arena offsets below
//   seed     u64[4]             noise key, coin key, first block (global index of local block 0), rmask
//   tab64    u64[]              read-only tables, each at its dims[XK_DIM_OFF_*] offset (chain_thr, ent_mask, xa, Tc,
//   tab32    u32[]              Kc, beta, betaO, bLo; chain_off, item_beg, seg, mem_beg, mem_cnt, mem_xo,
//                               ent_word, bb, pL): XdemSampler::device_tables() under the same names, plus the
//                               walk's plan (xtim.gpu.tables.walk_plan): item_beg u32[nitems + 1], seg
//                               u32[nseg * XK_SEG_WORDS], and per threshold row r the J + 1 skip thresholds
//                               chain_thr[r * THR_STRIDE ...] (xtim.gpu.tables.geo_thresholds) and their top bytes
//                               chain_thr8[r * THR8_STRIDE ...] (byte i of word q: test 8 q + i);
//                               the postselection span test (XdemSampler::span_tables of the mask): span_wmask,
//                               span_basis, span_piv_bit (tab64), span_widx, span_piv_word (tab32), with
//                               dims[XK_DIM_SPAN_RANK] basis vectors of dims[XK_DIM_SPAN_WORDS] words (both 0, the
//                               tables empty, without postselection or with an empty mask); the single-line law
//                               table (dims[XK_DIM_LAWT] = 1): xlaw (tab32, one u32 per xa word) and lawt (tab64);
//                               the zero decoder's observable table (one-word law): stab and rtab (tab64); the
//                               structured law's table (dims[XK_DIM_ST] = 1): sct (tab64,
//                               xtim.gpu.structure.st_table)
//   rows     u64[nshots * W]    shot rows (internal layout: detector d = bit d, observable o = bit 64 WD + o)
//   epos     u32[nblocks * (B + 1)]  per block, shot s's extended records are eids[epos[s], epos[s + 1])
//   eids     u32[nblocks * E]   per block, the fired extended lines' record offsets into xa, grouped by shot
//   psh      u32[nblocks * 2 E] K1's working list per block: [0, E) the shot of each appended record (in no fixed
//                               order) and [E, 2 E) its offset; with packed records (XK_DIM_RECSH) [0, E) the record
//                               itself, offset | shot << RECSH (read by nothing else)
//   overflow u32[nblocks]       0; XK_OVERFLOW_PAIRS: the block fired more than E extended lines (its lists are
//                               empty; relaunch with a larger E); XK_OVERFLOW_CAPACITY: a size exceeds a capacity
//   ckey     u64[nshots * max(rhoL, 1) * max(cw, 1)]  status C key: post_fast's / post_multi's T.ckey (descending
//                               pivots; cw words per vector under the multi-word law), zero-padded; not written by
//                               the one-word law or the structured laws under XK_MODE_NOKEY (the estimate, which
//                               never reads it)
//   cdim     u8[nshots]         dim C; XK_CDIM_INTERNAL: post_fast's internal check failed (the CPU throws
//                               std::logic_error); XK_CDIM_CAPACITY: a size exceeds a capacity
//   coin_in  u64[nshots * max(cw, 1)]  coin words, read instead of the coin stream under XK_MODE_COIN_IN
//   flag     u8[nshots]         K2 writes it (and K1 first under XK_WALK_SCREEN with postselection: XK_FLAG_EARLY for
//                               a shot its span test rejects, which K2 then skips, XK_FLAG_PENDING for the others,
//                               which K2 overwrites). Without XK_MODE_POSTSELECT every shot is XK_FLAG_SURVIVE; with it
//                               XK_FLAG_SURVIVE, XK_FLAG_EARLY (rejected by the span test before the law: no coin
//                               read, ckey / cdim / row not written) or 0. A shot whose cdim is an error code
//                               (XK_CDIM_*) is marked XK_FLAG_SURVIVE, so the code reaches K4 (cmiss) / hist and the
//                               host reports it whatever the mask. A survivor without an error code also has
//                               XK_FLAG_ZERO_PASS when its observable row lies in its C (score(status, zeros) > 0)
//   hist     u32[nblocks * stride]  K3's counters of block b at hist[b * stride ...] (stride = dimc + 3, dimc =
//                               dims[XK_DIM_DIMC]; 67 = XK_HIST_STRIDE for every one-word file): [XK_HIST_SURVIVE]
//                               survivors without an error code, [XK_HIST_PASS + k] those of them with
//                               XK_FLAG_ZERO_PASS and dim C = k (k = 0..dimc), [dimc + 2] survivors whose cdim is an
//                               error code (counted nowhere else; the host raises on any)
//   hist_group u64[XK_HIST_GROUPS * (stride + 2)]  K5: group g's sums at hist_group[g * (stride + 2) ...], laid out
//                               as hist_total; group g = blocks [g * ceil(nblocks / XK_HIST_GROUPS), (g + 1) *
//                               ceil(nblocks / XK_HIST_GROUPS)) (empty past nblocks: zeros)
//   hist_total u64[stride + 2]  K6: slot i < stride = the sum over the batch's blocks of hist slot i; [stride] /
//                               [stride + 1] = the blocks whose overflow is XK_OVERFLOW_PAIRS / XK_OVERFLOW_CAPACITY
//                               (a device-only estimate downloads only this)
//   mws      u64[nshots * xk_mw_words(rho, cw)]  K2's working words under the multi-word law (one-word files and the
//                               structured multi-word law: unused, a placeholder), interleaved: word i of shot s
//                               at mws[i * nshots + s]
//   lawq     u32[nshots + 1]    under XK_MODE_DEFER: [0] the number of queued shots (K1's block 0 sets it to 0, K2
//                               counts up atomically), [1 + k] queued shot k (in no particular order)
//   boff     u32[ceil(nshots / XK_EMIT_RUN)]  K7: run j's first output row (the survivors among the shots before
//                               shot j XK_EMIT_RUN)
//   ctab     u64[]              read-only, the host's (uploaded when it grows): known C keys and their ids (K4's
//                               xk_cspace_lookup); u64[1] = {0} is the empty table
//   out_det  u8[nshots * nbd], out_obs u8[nshots * nbo], out_cs u32[nshots]: K4's output rows [0, survivors) (nbd =
//                               ceil(D / 8) = dims[XK_DIM_NBD], nbo = ceil(O / 8) = dims[XK_DIM_NBO])
//   cmiss    u32[2 nshots + 1]  under XK_MODE_EMIT: [0] the pairs (K1's block 0 sets it to 0, K4 counts up
//                               atomically), then pairs (output row, shot) whose C id K4 left to the host (under
//                               XK_MODE_NOSTATUS: only the shots whose cdim is an error code; out_cs is not written)
//
// Each kernel reads only its own buffers (launchers bind a placeholder for the others): K1 dims, seed, tab64,
// tab32, rows, epos, eids, psh, overflow, lawq, cmiss, flag (XK_WALK_SCREEN with postselection); K2 and K2b those
// but overflow and cmiss, plus ckey, cdim, coin_in, mws; K3 dims, seed, tab64, tab32, rows, cdim, flag, hist; K4 dims,
// seed, tab64, tab32, rows, ckey, cdim, flag, boff, ctab, out_det, out_obs, out_cs, cmiss; K5 dims, seed, tab64,
// tab32, rows, overflow, hist, hist_group; K6 dims, seed, tab64, tab32, rows, hist_group, hist_total; K7 dims, seed,
// tab64, tab32, rows, flag, hist, hist_group, boff.
//
// Buffer reuse (a launcher keeps its buffers across batches; one batch = dims[XK_DIM_NSHOTS] shots, nblocks blocks,
// a buffer may be larger than the batch and hold an earlier batch's values): nothing reads an element the current
// batch did not write. The audit (K1 as the backend runs it: XK_MODE_GIVEN unset; "batch" = every element of the
// batch's range):
//   buffer      written by (per batch)                         read by                          fully written?
//   dims, seed  host, every batch                              every kernel                     yes
//   tab64/32    host, once per (tables, mask)                  every kernel                     yes (static)
//   rows        K1: zeroes then XORs [0, nshots W) (with       K1 its block (screen); K2 its    yes (K1) but for
//               rows in group memory, the shots its screen     shot; K4 survivors               shots K1 rejects
//               keeps); K2: the row of a shot not rejected                                      (never read); not
//               early                                                                           on XK_OVERFLOW_
//                                                                                               CAPACITY (raises)
//   epos        K1: [0, B] of every block (all 0 on overflow)  K2 its block                     yes
//   eids        K1: a block's first epos[B] records            K2 [epos[s], epos[s + 1])        no: read only
//               (overflow: <= E records, epos all 0)           (within K1's prefix)             where written
//   psh         K1: a block's first np (<= E) entries (and     K1 only, [0, np) of its block    no: read only
//               [E, E + np) unpacked)                          after a barrier                  where written
//   overflow    K1: every block                                K5 (every block); host (tests)   yes
//   ckey        K2: shots not rejected early, without a        K4, host: survivors with dim C   no: early shots,
//               capacity error (one-word and structured        in [1, dimc] (XK_MODE_EMIT,      capacity errors,
//               laws: not under XK_MODE_NOKEY)                 never with XK_MODE_NOKEY)        XK_MODE_NOKEY
//   cdim        K2: shots not rejected early (under            K3, K4 survivors; host: cmiss's  no: early shots;
//               XK_WALK_SCREEN: not those K1's walk rejects)   shots, estimate's error path
//                                                              (survivors only)                 read only under
//                                                                                               XK_FLAG_SURVIVE
//   coin_in     host (tests: XK_MODE_COIN_IN)                  K2 under XK_MODE_COIN_IN only    -
//   mws         K2: a shot's words, before it reads them       K2, the same shot only           no: read only
//                                                                                               where written
//   lawq        K1 block 0: [0] (XK_MODE_DEFER); K2: [0] and   K2b: [0] and [1, 1 + [0])        no: read only
//               [1, 1 + [0]) (atomic slots)                                                     where written
//   flag        K2: every shot (K1 first under XK_WALK_SCREEN  K2 (after K1's screen), K3, K4,  yes
//               with postselection; K2 skips the shots K1      K7; host (sample_ps)
//               rejected: their XK_FLAG_EARLY is final)
//   hist        K3: every block, every slot                    K5, K7 (every block); host       yes
//                                                              (tests)
//   boff        K7: every run of XK_EMIT_RUN shots             K4 (its shot's run)              yes
//   ctab        host, when its table grows                     K4                               yes (static)
//   out_*       K4: [0, survivors)                             host [0, survivors)              yes (bijection)
//   cmiss       K1 block 0: [0] (XK_MODE_EMIT); K4: [0] and    host: [0] and [1, 1 + 2 [0])     no: read only
//               [1, 1 + 2 [0]) (atomic slots)                                                   where written
//   hist_group  K5: every group, every slot                    K6                               yes
//   hist_total  K6: every slot                                 host                             yes
// K2b finishes exactly the shots K2 queued, writing what K2 would have (ckey, cdim, flag, the row), so the rows K2 and
// K2b write together are those of the table above. So K2 on a block that K1 overflowed (XK_OVERFLOW_PAIRS: epos all
// 0) resolves its shots with no extended record (no read of eids), K3, K5 and K6 count it, and the host, seeing
// hist_total[XK_HIST_TOTAL_PAIRS] > 0, reruns the batch before reading anything else (K7 and K4, which may run on
// such a batch, read only what K2 and K3 wrote, and their outputs are rewritten by the rerun). A launcher may therefore
// submit K1, K2, K2b, K3, K5, K6, K7, K4 at once.

// ------------------------------------------------------------------------------------------------ target prelude
#if defined(__METAL_VERSION__)
typedef ulong u64;
typedef uint u32;
typedef uchar u8;
#ifndef XK_BUF
#define XK_BUF device
#endif
#ifndef XK_LOC
#define XK_LOC thread
#endif
#ifndef XK_FN
#define XK_FN inline
#endif
#ifndef XK_U64
#define XK_U64(x) x##ul
#endif
#ifndef XK_TG
#define XK_TG threadgroup
#endif
#define XK_PARALLEL_GROUP 1
#define XK_BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#elif defined(__CUDACC__)
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
#ifndef XK_FN
#define XK_FN __device__ inline
#endif
#ifndef XK_U64
#define XK_U64(x) x##ull
#endif
#define XK_PARALLEL_GROUP 1
#define XK_BARRIER() __syncthreads()
#elif defined(__UINT64_TYPE__)
typedef __UINT64_TYPE__ u64;   // = uint64_t, so the C++ reference passes numpy buffers without casts
typedef __UINT32_TYPE__ u32;
typedef __UINT8_TYPE__ u8;
#else
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
#endif
#ifndef XK_BUF
#define XK_BUF
#endif
#ifndef XK_LOC
#define XK_LOC
#endif
#ifndef XK_FN
#define XK_FN inline
#endif
#ifndef XK_U64
#define XK_U64(x) x##ull
#endif
#ifndef XK_TG
#define XK_TG
#endif

#if !defined(XK_CAP_W) || !defined(XK_CAP_WA) || !defined(XK_CAP_RHO) || !defined(XK_CAP_E) || !defined(XK_CAP_J) || \
    !defined(XK_CAP_CW) || !defined(XK_CAP_DIMC)
#error "xdem_kernels.h: define XK_CAP_W, XK_CAP_WA, XK_CAP_RHO, XK_CAP_E, XK_CAP_J, XK_CAP_CW and XK_CAP_DIMC first"
#endif

// the bounds of the capacities that size thread-local arrays (header comment, thread-local memory)
#define XK_MAX_WA 64
#define XK_MAX_RHO 64
#define XK_MAX_DIMC 253
#if XK_CAP_WA > XK_MAX_WA || XK_CAP_RHO > XK_MAX_RHO
#error "xdem_kernels.h: XK_CAP_WA and XK_CAP_RHO must be <= 64 (they size thread-local arrays)"
#endif
#if XK_CAP_DIMC < 64 || XK_CAP_DIMC > XK_MAX_DIMC
#error "xdem_kernels.h: XK_CAP_DIMC must be in [64, 253] (it sizes K3's counters; dim C is a u8 below the error codes)"
#endif
// the walk's group memory (u32 words; the entry stub declares the array) and its largest block of shots
#define XK_MAX_BLOCK 1024
#define XK_MAX_SHM 8192
#if !defined(XK_CAP_SHM)
#error "xdem_kernels.h: define XK_CAP_SHM (the walk's group memory in u32 words) first"
#endif
#if defined(XK_PARALLEL_GROUP) && XK_CAP_SHM > XK_MAX_SHM
#error "xdem_kernels.h: XK_CAP_SHM must be at most 8192 words (32 KiB of threadgroup / shared memory)"
#endif
// the structured law (optional: compiled in when XK_CAP_ST_T is defined, with XK_CAP_ST_NC, XK_CAP_ST_NG, XK_CAP_ST_N)
#define XK_MAX_ST_NC 8
#define XK_MAX_ST_NG 8
#define XK_MAX_ST_T 32
#define XK_MAX_ST_N 16
#if defined(XK_CAP_ST_T)
#if !defined(XK_CAP_ST_NC) || !defined(XK_CAP_ST_NG) || !defined(XK_CAP_ST_N)
#error "xdem_kernels.h: the structured law needs XK_CAP_ST_NC, XK_CAP_ST_NG, XK_CAP_ST_T and XK_CAP_ST_N"
#endif
#if XK_CAP_ST_NC > XK_MAX_ST_NC || XK_CAP_ST_NG > XK_MAX_ST_NG || XK_CAP_ST_T > XK_MAX_ST_T || XK_CAP_ST_N > XK_MAX_ST_N
#error "xdem_kernels.h: XK_CAP_ST_NC, _NG, _T, _N must be at most 8, 8, 32, 16 (they size thread-local arrays)"
#endif
#endif
// the structured law on a multi-word file (xk_law_struct_mw; optional: XK_CAP_ST_W, its slot words, with the four
// above)
#define XK_MAX_ST_W 4
#if defined(XK_CAP_ST_W)
#if !defined(XK_CAP_ST_T) || XK_CAP_ST_W < 1 || XK_CAP_ST_W > XK_MAX_ST_W || XK_CAP_ST_NC < 1
#error "xdem_kernels.h: XK_CAP_ST_W must be in [1, 4], with XK_CAP_ST_NC, _NG, _T, _N (it sizes thread-local arrays)"
#endif
#endif

// ------------------------------------------------------------------------------------------------ dims[] indices
// The single source of truth; xtim/gpu/tables.py holds the same values (tests/test_gpu_kernels.py compares them).
#define XK_DIM_W 0
#define XK_DIM_WD 1
#define XK_DIM_WO 2
#define XK_DIM_WF 3
#define XK_DIM_WA 4
#define XK_DIM_RHO 5
#define XK_DIM_RHOL 6
#define XK_DIM_R 7
#define XK_DIM_B 8
#define XK_DIM_E 9
#define XK_DIM_NCHAINS 10
#define XK_DIM_NSHOTS 11
#define XK_DIM_NBLOCKS 12
#define XK_DIM_MODE 13
#define XK_DIM_SPAN_RANK 14
#define XK_DIM_SPAN_WORDS 15
// arena offsets: tab64
#define XK_DIM_OFF_CHAIN_THR 16
#define XK_DIM_OFF_ENT_MASK 17
#define XK_DIM_OFF_XA 18
#define XK_DIM_OFF_TC 19
#define XK_DIM_OFF_KC 20
#define XK_DIM_OFF_BETA 21
#define XK_DIM_OFF_BETAO 22
#define XK_DIM_OFF_BLO 23
// arena offsets: tab32
#define XK_DIM_OFF_CHAIN_OFF 24
// the walk's work (xtim.gpu.tables.walk_plan): item i's segments item_beg[i] .. item_beg[i + 1] - 1, and the segments
// (XK_SEG_WORDS u32 each)
#define XK_DIM_OFF_ITEM 25
#define XK_DIM_OFF_SEG 26
#define XK_DIM_OFF_MEM_BEG 27
#define XK_DIM_OFF_MEM_CNT 28
#define XK_DIM_OFF_MEM_XO 29
#define XK_DIM_OFF_ENT_WORD 30
#define XK_DIM_OFF_BB 31
#define XK_DIM_OFF_PL 32
// u64 words per threshold row in chain_thr (>= the largest test count nt)
#define XK_DIM_THR_STRIDE 33
// arena offsets of the span test: tab64, then tab32
#define XK_DIM_OFF_SPAN_WMASK 34
#define XK_DIM_OFF_SPAN_BASIS 35
#define XK_DIM_OFF_SPAN_PIV_BIT 36
#define XK_DIM_OFF_SPAN_WIDX 37
#define XK_DIM_OFF_SPAN_PIV_WORD 38
// the skip thresholds' top bytes, eight tests to a word (tab64), and its u64 words per chain
#define XK_DIM_OFF_CHAIN_THR8 39
#define XK_DIM_THR8_STRIDE 40
// the multi-word law: coordinate words CW (0: the one-word law), and its tables' offsets (tab64)
#define XK_DIM_CW 41
#define XK_DIM_OFF_KCW 42
#define XK_DIM_OFF_GCW 43
#define XK_DIM_OFF_GAMMA 44
#define XK_DIM_OFF_GBW 45
// the largest dim C the counters hold (max(64, rhoL)): it sets the hist layout (XK_HIST_* below)
#define XK_DIM_DIMC 46
// the single-line law table (one-word law): xlaw (tab32, indexed by an extended line's record offset) and its entries
// lawt (tab64); XK_DIM_LAWT = 1 when the table is in the arenas (0: no table, every law computed)
#define XK_DIM_OFF_XLAW 47
#define XK_DIM_OFF_LAWT 48
#define XK_DIM_LAWT 49
// emit's byte layout (K7): bytes per shot of the detector and observable rows (ceil(D / 8), ceil(O / 8))
#define XK_DIM_NBD 50
#define XK_DIM_NBO 51
// the zero decoder's observable table (one-word law; xtim.gpu.tables.obs_table): stab (tab64, u64[64 WO]: the beta_L
// coordinates s of observable bit o) and rtab (tab64, u64[64 WO * WO]: its rest), XK_DIM_RTAB = 1 when rtab is there
// (0: every rest is 0)
#define XK_DIM_OFF_STAB 52
#define XK_DIM_OFF_RTAB 53
#define XK_DIM_RTAB 54
// the structured one-word law (xk_law_struct; xtim.gpu.structure): XK_DIM_ST = 1 to use it, its core gauge bits, type
// records and non-leaf axes, and the offset of its table sct (tab64)
#define XK_DIM_ST 55
#define XK_DIM_ST_NG 56
#define XK_DIM_ST_NT 57
#define XK_DIM_ST_NN 58
#define XK_DIM_OFF_ST 59
// the walk's packed records (xtim.gpu.tables.rec_shift): K1 appends xo | s << XK_DIM_RECSH to psh (shot s < B, every
// record offset xo < 2^RECSH) and sort_ext scatters them into eids in one stable pass; 0: xo to eids and s to psh,
// sorted by a cycle permutation in place (the same order)
#define XK_DIM_RECSH 60
// the structured law on a multi-word file (xk_law_struct_mw, with XK_DIM_ST = 1 and XK_DIM_CW >= 2): its slot words
#define XK_DIM_ST_W 61
// the walk: its items per block (threads of work), and its options (XK_WALK_* bits)
#define XK_DIM_NITEMS 62
#define XK_DIM_WALK 63
#define XK_DIM_COUNT 64

// dims[XK_DIM_MODE] bits
#define XK_MODE_POSTSELECT 1
#define XK_MODE_COIN_IN 2
#define XK_MODE_GIVEN 4
#define XK_MODE_DEFER 8
#define XK_MODE_EMIT 16
#define XK_MODE_NOKEY 32
#define XK_MODE_NOSTATUS 64

// overflow[] and cdim[] codes
#define XK_OVERFLOW_PAIRS 1
#define XK_OVERFLOW_CAPACITY 2
#define XK_CDIM_CAPACITY 254
#define XK_CDIM_INTERNAL 255

// flag[] bits
#define XK_FLAG_SURVIVE 1
#define XK_FLAG_EARLY 2
#define XK_FLAG_ZERO_PASS 4
#define XK_FLAG_PENDING 8

// dims[XK_DIM_WALK] bits: the block's rows in group memory (else XORed in device memory), the postselection screen in
// the walk (else in K2)
#define XK_WALK_TGROWS 1
#define XK_WALK_SCREEN 2

// hist[] slots of a block (K3). The layout depends on dimc = dims[XK_DIM_DIMC] (max(64, rhoL)): SURVIVE, then
// PASS + k for k = 0..dimc, then the error slot dimc + 2; stride dimc + 3, then hist_total's two overflow counts
// (XkCtx's hist_stride, hist_count). The values below are those at dimc = 64 (every one-word file).
#define XK_HIST_SURVIVE 0
#define XK_HIST_PASS 1
#define XK_HIST_ERROR 66
#define XK_HIST_STRIDE 67
// hist_total[] slots (K5): XK_HIST_STRIDE batch sums, then the overflow counts
#define XK_HIST_TOTAL_PAIRS 67
#define XK_HIST_TOTAL_CAPACITY 68
#define XK_HIST_TOTAL_COUNT 69
// groups of blocks K5 sums on their own (K6 then sums the groups)
#define XK_HIST_GROUPS 256

#define XK_GAMMA XK_U64(0x9E3779B97F4A7C15)

// ------------------------------------------------------------------------------------------------ the context
// Every buffer pointer and the run-time sizes, filled once per thread by xk_ctx_init (the one place pointers into
// the arenas are derived) and passed to every helper.
struct XkCtx {
    // sizes (dims)
    u32 W, WD, WO, WF, WA, rho, rhoL, B, E, nchains, nshots, nblocks, mode, thr_stride, span_rank, span_words;
    u32 thr8_stride;
    u32 gauge, cw, dimc, hist_stride, hist_count;   // gauge = r; hist_stride = dimc + 3; hist_count = hist_stride + 2
    // seed
    u64 noise_key, coin_key, first_block, rmask;
    // walk tables
    XK_BUF const u64* chain_thr;
    XK_BUF const u64* chain_thr8;
    XK_BUF const u32* chain_off;
    XK_BUF const u32* item_beg;
    XK_BUF const u32* seg;
    u32 nitems, walk;
    XK_BUF const u32* mem_beg;
    XK_BUF const u32* mem_cnt;
    XK_BUF const u32* mem_xo;
    XK_BUF const u64* ent_mask;
    XK_BUF const u32* ent_word;
    // law tables
    XK_BUF const u64* xa;
    XK_BUF const u64* Tc;
    XK_BUF const u64* Kc;
    XK_BUF const u32* bb;
    XK_BUF const u64* beta;
    XK_BUF const u64* betaO;
    XK_BUF const u64* bLo;
    XK_BUF const u32* pL;
    // multi-word law tables (cw >= 2)
    XK_BUF const u64* KcW;
    XK_BUF const u64* GcW;
    XK_BUF const u64* gamma;
    XK_BUF const u64* GbW;
    // the single-line law table (lawt_on = dims[XK_DIM_LAWT])
    u32 lawt_on;
    XK_BUF const u32* xlaw;
    XK_BUF const u64* lawt;
    // the zero decoder's observable table (rtab_on = dims[XK_DIM_RTAB])
    u32 rtab_on;
    XK_BUF const u64* stab;
    XK_BUF const u64* rtab;
    // the structured law (st = dims[XK_DIM_ST]): core gauge bits, type records, non-leaf axes, its table
    u32 st, st_ng, st_nt, st_nn, st_w;   // st_w: slot words (multi-word files)
    XK_BUF const u64* sct;
    u32 recsh;   // XK_DIM_RECSH
    // span test tables
    XK_BUF const u32* span_widx;
    XK_BUF const u64* span_wmask;
    XK_BUF const u64* span_basis;
    XK_BUF const u32* span_piv_word;
    XK_BUF const u64* span_piv_bit;
    // working and output buffers
    XK_BUF u64* rows;
    XK_BUF u32* epos;
    XK_BUF u32* eids;
    XK_BUF u32* psh;
    XK_BUF u32* overflow;
    XK_BUF u64* ckey;
    XK_BUF u8* cdim;
    XK_BUF const u64* coin_in;
    XK_BUF u8* flag;
    XK_BUF u32* hist;
    XK_BUF u64* hist_group;
    XK_BUF u64* hist_total;
    XK_BUF u64* mws;
    XK_BUF u32* lawq;
    XK_BUF u32* boff;
    XK_BUF const u64* ctab;
    XK_BUF u8* out_det;
    XK_BUF u8* out_obs;
    XK_BUF u32* out_cs;
    XK_BUF u32* cmiss;
    u32 nbd, nbo;
};

XK_FN void xk_ctx_init(XK_LOC XkCtx* c, XK_BUF const u32* dims, XK_BUF const u64* seed, XK_BUF const u64* tab64,
                       XK_BUF const u32* tab32, XK_BUF u64* rows, XK_BUF u32* epos, XK_BUF u32* eids,
                       XK_BUF u32* psh, XK_BUF u32* overflow, XK_BUF u64* ckey, XK_BUF u8* cdim, XK_BUF const u64* coin_in,
                       XK_BUF u8* flag, XK_BUF u32* hist, XK_BUF u64* hist_group, XK_BUF u64* hist_total,
                       XK_BUF u64* mws, XK_BUF u32* lawq, XK_BUF u32* boff, XK_BUF const u64* ctab,
                       XK_BUF u8* out_det, XK_BUF u8* out_obs, XK_BUF u32* out_cs, XK_BUF u32* cmiss) {
    c->W = dims[XK_DIM_W];
    c->WD = dims[XK_DIM_WD];
    c->WO = dims[XK_DIM_WO];
    c->WF = dims[XK_DIM_WF];
    c->WA = dims[XK_DIM_WA];
    c->rho = dims[XK_DIM_RHO];
    c->rhoL = dims[XK_DIM_RHOL];
    c->B = dims[XK_DIM_B];
    c->E = dims[XK_DIM_E];
    c->nchains = dims[XK_DIM_NCHAINS];
    c->nshots = dims[XK_DIM_NSHOTS];
    c->nblocks = dims[XK_DIM_NBLOCKS];
    c->mode = dims[XK_DIM_MODE];
    c->thr_stride = dims[XK_DIM_THR_STRIDE];
    c->thr8_stride = dims[XK_DIM_THR8_STRIDE];
    c->gauge = dims[XK_DIM_R];
    c->cw = dims[XK_DIM_CW];
    c->dimc = dims[XK_DIM_DIMC];
    c->hist_stride = c->dimc + 3;
    c->hist_count = c->hist_stride + 2;
    c->span_rank = dims[XK_DIM_SPAN_RANK];
    c->span_words = dims[XK_DIM_SPAN_WORDS];
    c->noise_key = seed[0];
    c->coin_key = seed[1];
    c->first_block = seed[2];
    c->rmask = seed[3];
    c->chain_thr = tab64 + dims[XK_DIM_OFF_CHAIN_THR];
    c->chain_thr8 = tab64 + dims[XK_DIM_OFF_CHAIN_THR8];
    c->ent_mask = tab64 + dims[XK_DIM_OFF_ENT_MASK];
    c->xa = tab64 + dims[XK_DIM_OFF_XA];
    c->Tc = tab64 + dims[XK_DIM_OFF_TC];
    c->Kc = tab64 + dims[XK_DIM_OFF_KC];
    c->beta = tab64 + dims[XK_DIM_OFF_BETA];
    c->betaO = tab64 + dims[XK_DIM_OFF_BETAO];
    c->bLo = tab64 + dims[XK_DIM_OFF_BLO];
    c->chain_off = tab32 + dims[XK_DIM_OFF_CHAIN_OFF];
    c->item_beg = tab32 + dims[XK_DIM_OFF_ITEM];
    c->seg = tab32 + dims[XK_DIM_OFF_SEG];
    c->nitems = dims[XK_DIM_NITEMS];
    c->walk = dims[XK_DIM_WALK];
    c->mem_beg = tab32 + dims[XK_DIM_OFF_MEM_BEG];
    c->mem_cnt = tab32 + dims[XK_DIM_OFF_MEM_CNT];
    c->mem_xo = tab32 + dims[XK_DIM_OFF_MEM_XO];
    c->ent_word = tab32 + dims[XK_DIM_OFF_ENT_WORD];
    c->bb = tab32 + dims[XK_DIM_OFF_BB];
    c->pL = tab32 + dims[XK_DIM_OFF_PL];
    c->KcW = tab64 + dims[XK_DIM_OFF_KCW];
    c->GcW = tab64 + dims[XK_DIM_OFF_GCW];
    c->gamma = tab64 + dims[XK_DIM_OFF_GAMMA];
    c->GbW = tab64 + dims[XK_DIM_OFF_GBW];
    c->lawt_on = dims[XK_DIM_LAWT];
    c->xlaw = tab32 + dims[XK_DIM_OFF_XLAW];
    c->lawt = tab64 + dims[XK_DIM_OFF_LAWT];
    c->rtab_on = dims[XK_DIM_RTAB];
    c->stab = tab64 + dims[XK_DIM_OFF_STAB];
    c->rtab = tab64 + dims[XK_DIM_OFF_RTAB];
    c->st = dims[XK_DIM_ST];
    c->st_ng = dims[XK_DIM_ST_NG];
    c->st_nt = dims[XK_DIM_ST_NT];
    c->st_nn = dims[XK_DIM_ST_NN];
    c->st_w = dims[XK_DIM_ST_W];
    c->sct = tab64 + dims[XK_DIM_OFF_ST];
    c->recsh = dims[XK_DIM_RECSH];
    c->span_wmask = tab64 + dims[XK_DIM_OFF_SPAN_WMASK];
    c->span_basis = tab64 + dims[XK_DIM_OFF_SPAN_BASIS];
    c->span_piv_bit = tab64 + dims[XK_DIM_OFF_SPAN_PIV_BIT];
    c->span_widx = tab32 + dims[XK_DIM_OFF_SPAN_WIDX];
    c->span_piv_word = tab32 + dims[XK_DIM_OFF_SPAN_PIV_WORD];
    c->rows = rows;
    c->epos = epos;
    c->eids = eids;
    c->psh = psh;
    c->overflow = overflow;
    c->ckey = ckey;
    c->cdim = cdim;
    c->coin_in = coin_in;
    c->flag = flag;
    c->hist = hist;
    c->hist_group = hist_group;
    c->hist_total = hist_total;
    c->mws = mws;
    c->lawq = lawq;
    c->boff = boff;
    c->ctab = ctab;
    c->out_det = out_det;
    c->out_obs = out_obs;
    c->out_cs = out_cs;
    c->cmiss = cmiss;
    c->nbd = dims[XK_DIM_NBD];
    c->nbo = dims[XK_DIM_NBO];
}

// ------------------------------------------------------------------------------------------------ bit helpers
// (arguments are nonzero where the result is used)
XK_FN u32 xk_ctz(u64 x) {
#if defined(__METAL_VERSION__)
    return (u32)ctz(x);
#elif defined(__CUDACC__)
    return (u32)(__ffsll((long long)x) - 1);
#else
    return (u32)__builtin_ctzll(x);
#endif
}

XK_FN u32 xk_highest(u64 x) {   // index of the highest set bit
#if defined(__METAL_VERSION__)
    return 63u - (u32)clz(x);
#elif defined(__CUDACC__)
    return 63u - (u32)__clzll((long long)x);
#else
    return 63u - (u32)__builtin_clzll(x);
#endif
}

XK_FN u64 xk_parity(u64 x) {
#if defined(__METAL_VERSION__)
    return (u64)(popcount(x) & 1);
#elif defined(__CUDACC__)
    return (u64)(__popcll(x) & 1);
#else
    return (u64)(__builtin_popcountll(x) & 1);
#endif
}

XK_FN u32 xk_popcount(u64 x) {
#if defined(__METAL_VERSION__)
    return (u32)popcount(x);
#elif defined(__CUDACC__)
    return (u32)__popcll(x);
#else
    return (u32)__builtin_popcountll(x);
#endif
}

XK_FN u64 xk_bit(u32 j) { return (u64)1 << j; }

// *p += 1, atomically among the grid's threads; returns the old value (the C++ reference runs one thread at a time)
XK_FN u32 xk_atomic_inc(XK_BUF u32* p) {
#if defined(__METAL_VERSION__)
    return atomic_fetch_add_explicit((XK_BUF atomic_uint*)p, 1u, memory_order_relaxed);
#elif defined(__CUDACC__)
    return atomicAdd(p, 1u);
#else
    return (*p)++;
#endif
}

// the walk's atomics among a group's threads (K1): *p += 1 in group memory (returns the old value), *p ^= v in group
// memory, and a row word ^= v in device memory (Metal has no 64-bit XOR atomic: two 32-bit halves, little-endian)
XK_FN u32 xk_tg_inc(XK_TG u32* p) {
#if defined(__METAL_VERSION__)
    return atomic_fetch_add_explicit((XK_TG atomic_uint*)p, 1u, memory_order_relaxed);
#elif defined(__CUDACC__)
    return atomicAdd(p, 1u);
#else
    return (*p)++;
#endif
}

XK_FN void xk_tg_xor(XK_TG u32* p, u32 v) {
#if defined(__METAL_VERSION__)
    atomic_fetch_xor_explicit((XK_TG atomic_uint*)p, v, memory_order_relaxed);
#elif defined(__CUDACC__)
    atomicXor(p, v);
#else
    *p ^= v;
#endif
}

XK_FN void xk_dev_xor(XK_BUF u64* p, u64 v) {
#if defined(__METAL_VERSION__)
    XK_BUF atomic_uint* h = (XK_BUF atomic_uint*)p;
    if ((u32)v) atomic_fetch_xor_explicit(h, (u32)v, memory_order_relaxed);
    if ((u32)(v >> 32)) atomic_fetch_xor_explicit(h + 1, (u32)(v >> 32), memory_order_relaxed);
#elif defined(__CUDACC__)
    atomicXor(p, v);
#else
    *p ^= v;
#endif
}

// the walk's group memory (u32 words; K1): [0] the block's appended records, [1] the items taken, then cnt, start,
// fill (B + 1 each: per shot its records, its first record, its records placed), part (B / 8 + 1: per 8 shots their
// records), and with
// XK_WALK_TGROWS the block's rows (2 W words per shot: the u64 row words as little-endian halves)
#define XK_SHM_NP 0
#define XK_SHM_NEXT 1
XK_FN u32 xk_shm_cnt(u32 B) { return 2; }
XK_FN u32 xk_shm_start(u32 B) { return 3 + B; }
XK_FN u32 xk_shm_fill(u32 B) { return 4 + 2 * B; }
XK_FN u32 xk_shm_part(u32 B) { return 5 + 3 * B; }
XK_FN u32 xk_shm_rows(u32 B) { return 6 + 3 * B + B / 8; }
XK_FN u64 xk_shm_need(u32 B, u32 W, u32 walk) {
    return (u64)xk_shm_rows(B) + ((walk & XK_WALK_TGROWS) ? (u64)2 * B * W : 0);
}

// every run-time size within its capacity (the largest chain J = thr_stride - 1; rho bounds the one-word law's arrays,
// cw >= 2 needs the multi-word law compiled in: XK_CAP_CW >= cw)
XK_FN bool xk_caps_ok(XK_LOC const XkCtx* c) {
#if defined(XK_CAP_ST_T)   // the structured law: compiled in, its sizes within its capacities (one-word: rho + 1 <= 64;
                           // multi-word: XK_CAP_ST_W compiled in, its slot words cover rho and are coin words,
                           // XK_MODE_DEFER set)
    const bool st_sz = c->rho - c->rhoL + 1 <= XK_CAP_ST_NC && c->st_ng <= XK_CAP_ST_NG && c->st_nt <= XK_CAP_ST_T &&
                       c->st_nn <= XK_CAP_ST_N;
#if defined(XK_CAP_ST_W)
    const bool st_mw = c->st_w >= 1 && c->st_w <= XK_CAP_ST_W && c->st_w <= c->cw && 64 * c->st_w >= c->rho &&
                       (c->mode & XK_MODE_DEFER) != 0;   // it runs from K2b's queue only (xk_finish_shot)
#else
    const bool st_mw = false;
#endif
    const bool st_ok = !c->st || (st_sz && (c->cw ? st_mw : (c->rho < 64 && !c->st_w)));
#else
    const bool st_ok = !c->st;
#endif
    return c->W <= XK_CAP_W && c->WA <= XK_CAP_WA && (c->cw ? c->cw <= XK_CAP_CW : c->rho <= XK_CAP_RHO) &&
           c->E <= XK_CAP_E && c->thr_stride <= (u32)XK_CAP_J + 1 && c->dimc >= 64 && c->dimc <= XK_CAP_DIMC && st_ok &&
           c->B <= XK_MAX_BLOCK && xk_shm_need(c->B, c->W, c->walk) <= (u64)XK_CAP_SHM;
}

// krng::mix (SplitMix64's output finalizer)
XK_FN u64 xk_mix(u64 z) {
    z = (z ^ (z >> 30)) * XK_U64(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * XK_U64(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

// ------------------------------------------------------------------------------------------------ postselection
// span_reduce (the plain reduction, not the byte-table variant): d_M in V_M. The basis is fully reduced, so d_M is
// in V_M iff d_M = the XOR of the basis vectors whose pivot bit d has; every selector reads d itself, so each
// compressed word k is reduced on its own (no local array).
XK_FN bool xk_span_in(XK_LOC const XkCtx* c, XK_BUF const u64* d) {
    const u32 nw = c->span_words, rank = c->span_rank;
    u64 acc = 0;
    for (u32 k = 0; k < nw; ++k) {
        u64 v = d[c->span_widx[k]] & c->span_wmask[k];
        for (u32 b = 0; b < rank; ++b) {
            const u64 msk = (u64)0 - (u64)((d[c->span_widx[c->span_piv_word[b]]] & c->span_piv_bit[b]) != 0);
            v ^= c->span_basis[(u64)b * nw + k] & msk;
        }
        acc |= v;
    }
    return acc == 0;
}

// run_one_block's zero(): every masked detector of the row is 0 (an empty mask: true)
XK_FN bool xk_masked_zero(XK_LOC const XkCtx* c, XK_BUF const u64* row) {
    u64 acc = 0;
    for (u32 k = 0; k < c->span_words; ++k) acc |= row[c->span_widx[k]] & c->span_wmask[k];
    return acc == 0;
}

// ------------------------------------------------------------------------------------------------ K1: walk
// walk_chains with the geometric skip drawn from integer thresholds (design 5.4; xtim.gpu.tables.walk_plan and
// walk_block_py), one group of threads per block of B shots. The host cuts chain c's M * B slots (slot t = member
// m0 + t / B, shot t % B) into segments of about the same expected fires, each a range [t0, t0 + L) of the chain's
// slots with its own skip thresholds for L slots, and deals the segments out to the block's items (one item: one or
// several whole segments, walked in order); every block has the same items. Item i of global block g draws from its
// own stream (xk_item_pos), so the draws depend only on (seed, g, i, B and the plan), never on the batch, the group's
// threads or their timing.
//
// A segment's skip (one pass) is nt independent tests, test 0 the end test (true: skip >= 2^(nt - 1) >= L, the
// segment ends) and test 1 + j bit j of the skip, test i true with probability thr[i] / 2^64. Eight tests share one
// word: byte i of the word against the threshold's top byte (thr8); only a tie (1 in 256) takes a draw of its own,
// whose top 56 bits are compared with the threshold's low 56, so each test's probability is exactly thr[i] / 2^64
// (the exhaustive count in tests/test_gpu_geo.py). An item's words are its stream's odd counters, its tie draws the
// even ones, indexed by the word and the tie's rank in it: a pass draws all its ceil(nt / 8) words together, whatever
// the end test says, and no draw is ever used twice, so the tests of every pass are independent of each other and
// of every other pass. A skip past the segment's end ends it; the next pass starts at the slot after a fire, so a
// segment's slots fire independently, each with the chain's probability (the geometric law of thr, as in the CPU
// walk). A segment with nt = 0 (p = 1) fires at every slot.
//
// Phases (barriers between them; the C++ reference runs each phase for every thread of the group in turn):
//   0 the block's rows (group memory with XK_WALK_TGROWS, else device memory) and counters zeroed
//   1 each thread walks items: item lane first, then the next one not taken (the group's counter), so a thread done
//     early takes more (items are ordered by expected cost, largest first); a fired member XORs its flip entries
//     into the shot's row (atomic XOR: 32-bit halves in group memory, a 64-bit word or two 32-bit halves in device
//     memory), an extended one appends (shot, record offset) to psh through the group's counter and counts it for
//     its shot
//   2 per shot: under XK_WALK_SCREEN (postselection) the span test of K2 on the walk's row: a shot rejected (an
//     extended line and the row outside the span: XK_FLAG_EARLY; none and a masked detector set: 0) gets its flag
//     and keeps no record, the others XK_FLAG_PENDING (K2 finishes them); without XK_WALK_SCREEN nothing
//   3 the rows of the shots kept written to device memory (XK_WALK_TGROWS); per 8 shots their records
//   4 per shot its first record (a prefix over those sums) into epos; overflow[block]
//   5 each record placed in eids at its shot's list (an atomic slot within the list)
//   6 each shot's list sorted by record offset (= file order), so the lists do not depend on the atomics' order
// More than E records: XK_OVERFLOW_PAIRS, every list empty (the rows are still written). Shots at or past nshots (a
// partial last block) draw the same numbers and are not written.

// the span test and run_one_block's zero() on a row in group memory (xk_span_in, xk_masked_zero)
XK_FN u64 xk_tg_word(XK_TG const u32* r, u32 w) { return (u64)r[2 * w] | (u64)r[2 * w + 1] << 32; }

XK_FN bool xk_span_in_tg(XK_LOC const XkCtx* c, XK_TG const u32* d) {
    const u32 nw = c->span_words, rank = c->span_rank;
    u64 acc = 0;
    for (u32 k = 0; k < nw; ++k) {
        u64 v = xk_tg_word(d, c->span_widx[k]) & c->span_wmask[k];
        for (u32 b = 0; b < rank; ++b) {
            const u64 bit = xk_tg_word(d, c->span_widx[c->span_piv_word[b]]) & c->span_piv_bit[b];
            const u64 msk = (u64)0 - (u64)(bit != 0);
            v ^= c->span_basis[(u64)b * nw + k] & msk;
        }
        acc |= v;
    }
    return acc == 0;
}

XK_FN bool xk_masked_zero_tg(XK_LOC const XkCtx* c, XK_TG const u32* row) {
    u64 acc = 0;
    for (u32 k = 0; k < c->span_words; ++k) acc |= xk_tg_word(row, c->span_widx[k]) & c->span_wmask[k];
    return acc == 0;
}

// the stream of item i of local block b: mix(pos + k gamma), k = 1, 2, ... (pos = xk_item_pos); a pass's words take
// the odd k (word n of the item: k = 2 n + 1), its tie draws the even k (the j-th tie of word n: k = 2 (8 n + j + 1))
#define XK_ITEM_GAMMA XK_U64(0xD1B54A32D192ED03)
XK_FN u64 xk_block_key(XK_LOC const XkCtx* c, u32 block) {
    return xk_mix(c->noise_key ^ xk_mix(c->first_block + block));
}
XK_FN u64 xk_item_pos(u64 bkey, u32 item) { return xk_mix(bkey + (u64)(item + 1) * XK_ITEM_GAMMA); }

// a segment record (tab32, XK_SEG_WORDS u32): its threshold row (chain_thr / chain_thr8 at row * stride), its tests nt
// (0: p = 1), the chain's first member m0, its first slot t0 and its slots L
#define XK_SEG_WORDS 5
// a pass's words at most (nt <= XK_CAP_J + 1 tests, eight to a word)
#define XK_PASS_WORDS ((XK_CAP_J + 8) / 8)

// eight byte tests of a draw against the top bytes of eight thresholds (SWAR on 32-bit halves, no carry crossing a
// byte): bit i of *lt: byte i of x below byte i of y; of *eq: equal
XK_FN u32 xk_gather4(u32 b) { return ((b >> 7) * 0x00204081u) >> 21 & 0xFu; }   // bits 7, 15, 23, 31 to 0..3
XK_FN void xk_bytes_lt(u64 x, u64 y, XK_LOC u32* lt, XK_LOC u32* eq) {
    const u32 H = 0x80808080u;
    u32 l = 0, e = 0;
    for (u32 h = 0; h < 2; ++h) {
        const u32 a = (u32)(x >> (32 * h)), t = (u32)(y >> (32 * h));
        const u32 d = (a | H) - (t & ~H);
        const u32 z = a ^ t;
        l |= xk_gather4(((~a & t) | (~z & ~d)) & H) << (4 * h);
        e |= xk_gather4(~(((z & ~H) + ~H) | z) & H) << (4 * h);
    }
    *lt = l;
    *eq = e;
}

// the skip of one pass (header above): its value, or *end when the end test is true. The pass reads the item's words
// n .. n + w - 1 (w = ceil(nt / 8)), all of them whatever the end test says, so they are drawn together.
XK_FN u32 xk_skip(u64 pos, u32 n, XK_BUF const u64* thr, XK_LOC const u64* thr8, u32 nt, XK_LOC bool* end) {
    const u32 w = (nt + 7) / 8;
    u64 x[XK_PASS_WORDS];
    for (u32 q = 0; q < XK_PASS_WORDS; ++q)
        if (q < w) x[q] = xk_mix(pos + (u64)(2 * (n + q) + 1) * XK_GAMMA);
    u64 g = 0;
    bool e = false;
    for (u32 q = 0; q < XK_PASS_WORDS; ++q) {
        if (q >= w) break;
        u32 lt, eq;
        xk_bytes_lt(x[q], thr8[q], &lt, &eq);
        const u32 valid = nt - q * 8 < 8 ? (1u << (nt - q * 8)) - 1u : 255u;
        lt &= valid;
        eq &= valid;
        u64 tk = 2 * (8 * (u64)(n + q));
        for (; eq; eq &= eq - 1) {   // a tie on the top byte: 56 fresh bits against the low 56
            const u32 i = xk_ctz((u64)eq);
            tk += 2;
            lt |= (u32)((xk_mix(pos + tk * XK_GAMMA) >> 8) < (thr[q * 8 + i] & XK_U64(0x00FFFFFFFFFFFFFF))) << i;
        }
        if (q == 0) {
            e = (lt & 1u) != 0;
            g = lt >> 1;
        } else {
            g |= (u64)lt << (8 * q - 1);
        }
    }
    *end = e;
    return g > 0xFFFFFFFFu ? 0xFFFFFFFFu : (u32)g;   // a skip past 2^32 - 1 ends the segment anyway (L < 2^32)
}

// the shots of local block b (n <= B; a partial last block has fewer)
XK_FN u32 xk_block_shots(XK_LOC const XkCtx* c, u32 block) {
    const u64 shot0 = (u64)block * c->B;
    return shot0 + c->B <= c->nshots ? c->B : (u32)(c->nshots - shot0);
}

// phase 1 for one thread: item `lane`, then items taken from the group's counter (XK_SHM_NEXT) until none is left, so
// a thread done early takes more (which thread walks an item changes nothing: each item has its own stream). One
// flat loop over the passes (a pass: one skip, then a fired slot or the segment's end), bounded: every pass fires a
// slot or ends a segment, every segment ends.
XK_FN void xk_walk_items(XK_LOC XkCtx* c, u32 block, u32 lane, u32 nl, XK_TG u32* shm) {
    const u32 B = c->B, W = c->W, E = c->E, sh = c->recsh, n = xk_block_shots(c, block), nitems = c->nitems;
    const bool tg = (c->walk & XK_WALK_TGROWS) != 0;
    XK_TG u32* cnt = shm + xk_shm_cnt(B);
    XK_TG u32* trows = shm + xk_shm_rows(B);
    XK_BUF u64* drows = c->rows + (u64)block * B * W;
    XK_BUF u32* psh = c->psh + (u64)block * 2 * E;   // the records: packed, or shots then (at + E) offsets
    XK_BUF const u32* seg = c->seg;
    XK_BUF const u32* ib = c->item_beg;
    XK_BUF const u64* thr0 = c->chain_thr;
    XK_BUF const u64* thr80 = c->chain_thr8;
    const u32 ts = c->thr_stride, ts8 = c->thr8_stride;
    const u64 bkey = xk_block_key(c, block);
    const bool pow2 = (B & (B - 1)) == 0;   // a shift instead of a division
    const u32 bshift = pow2 ? xk_ctz((u64)B) : 0;
    u32 item = lane;
    if (item >= nitems) return;
    u32 sg = ib[item], se = ib[item + 1], k = 0;
    u64 pos = xk_item_pos(bkey, item);
    u32 nt = 0, m0 = 0, t0 = 0, L = 0, t = 0;
    XK_BUF const u64* thr = thr0;
    u64 thr8[XK_PASS_WORDS];   // the segment's threshold top bytes
    for (u32 q = 0; q < XK_PASS_WORDS; ++q) thr8[q] = 0;
    bool load = true;
    for (;;) {
        if (sg >= se) {   // the item is done: the next one
            item = nl + xk_tg_inc(shm + XK_SHM_NEXT);
            if (item >= nitems) break;
            sg = ib[item];
            se = ib[item + 1];
            pos = xk_item_pos(bkey, item);
            k = 0;
            load = true;
            continue;
        }
        if (load) {
            XK_BUF const u32* r = seg + (u64)sg * XK_SEG_WORDS;
            thr = thr0 + (u64)r[0] * ts;
            nt = r[1];
            for (u32 q = 0; q < XK_PASS_WORDS; ++q)
                if (q * 8 < nt) thr8[q] = thr80[(u64)r[0] * ts8 + q];
            m0 = r[2];
            t0 = r[3];
            L = r[4];
            t = 0;
            load = false;
        }
        bool end = t >= L;   // an empty segment
        if (!end && nt) {
            const u32 g = xk_skip(pos, k, thr, thr8, nt, &end);
            k += (nt + 7) / 8;
            if (g >= L - t) end = true;
            else t += g;
        }
        if (end) {
            ++sg;
            load = true;
            continue;
        }
        const u32 slot = t0 + t;
        ++t;
        const u32 m = m0 + (pow2 ? slot >> bshift : slot / B);
        const u32 s = pow2 ? slot & (B - 1) : slot % B;
        if (s >= n) continue;
        const u32 e0 = c->mem_beg[m], ne = c->mem_cnt[m];
        if (tg) {
            XK_TG u32* row = trows + s * 2 * W;
            for (u32 e = e0; e < e0 + ne; ++e) {
                const u64 mk = c->ent_mask[e];
                const u32 w = c->ent_word[e];
                if ((u32)mk) xk_tg_xor(row + 2 * w, (u32)mk);
                if ((u32)(mk >> 32)) xk_tg_xor(row + 2 * w + 1, (u32)(mk >> 32));
            }
        } else {
            XK_BUF u64* row = drows + (u64)s * W;
            for (u32 e = e0; e < e0 + ne; ++e) xk_dev_xor(row + c->ent_word[e], c->ent_mask[e]);
        }
        const u32 xo = c->mem_xo[m];
        if (xo) {
            const u32 at = xk_tg_inc(shm + XK_SHM_NP);
            if (at < E) {
                if (sh) {
                    psh[at] = xo | (s << sh);
                } else {
                    psh[at] = s;
                    psh[at + E] = xo;
                }
                xk_tg_inc(cnt + s);
            }
        }
    }
}

// a record's shot and offset (phase 5)
XK_FN u32 xk_rec_shot(XK_LOC const XkCtx* c, XK_BUF const u32* psh, u32 i) {
    return c->recsh ? psh[i] >> c->recsh : psh[i];
}
XK_FN u32 xk_rec_off(XK_LOC const XkCtx* c, XK_BUF const u32* psh, u32 i) {
    return c->recsh ? psh[i] & ((1u << c->recsh) - 1u) : psh[i + c->E];
}

#define XK_REJECT 0x80000000u   // cnt[s]: the shot was rejected by the screen (it keeps no record)

XK_FN void xk_walk_phase(XK_LOC XkCtx* c, u32 block, u32 lane, u32 nl, XK_TG u32* shm, u32 ph) {
    const u32 B = c->B, W = c->W, E = c->E, n = xk_block_shots(c, block);
    const u64 shot0 = (u64)block * B;
    const bool tg = (c->walk & XK_WALK_TGROWS) != 0;
    XK_TG u32* cnt = shm + xk_shm_cnt(B);
    XK_TG u32* start = shm + xk_shm_start(B);
    XK_TG u32* fill = shm + xk_shm_fill(B);
    XK_TG u32* part = shm + xk_shm_part(B);
    XK_TG u32* trows = shm + xk_shm_rows(B);
    XK_BUF u64* R = c->rows + shot0 * W;
    XK_BUF u32* fp = c->epos + (u64)block * (B + 1);
    XK_BUF u32* kid = c->eids + (u64)block * E;
    XK_BUF const u32* psh = c->psh + (u64)block * 2 * E;
    if (ph == 0) {
        if (lane == 0) shm[XK_SHM_NP] = shm[XK_SHM_NEXT] = 0;
        for (u32 s = lane; s <= B; s += nl) cnt[s] = 0;
        if (tg) {
            for (u32 i = lane; i < 2 * B * W; i += nl) trows[i] = 0;
        } else {
            for (u32 i = lane; i < n * W; i += nl) R[i] = 0;
        }
    } else if (ph == 1) {
        xk_walk_items(c, block, lane, nl, shm);
    } else if (ph == 2) {
        if (!(c->walk & XK_WALK_SCREEN) || !(c->mode & XK_MODE_POSTSELECT)) return;
        for (u32 s = lane; s < n; s += nl) {
            const u32 ne = cnt[s];
            u32 f = XK_FLAG_PENDING;
            if (tg) {
                XK_TG const u32* row = trows + (u64)s * 2 * W;
                if (ne ? !xk_span_in_tg(c, row) : !xk_masked_zero_tg(c, row)) f = ne ? XK_FLAG_EARLY : 0;
            } else {
                XK_BUF const u64* row = R + (u64)s * W;
                if (ne ? !xk_span_in(c, row) : !xk_masked_zero(c, row)) f = ne ? XK_FLAG_EARLY : 0;
            }
            c->flag[shot0 + s] = (u8)f;
            if (f != XK_FLAG_PENDING) cnt[s] = XK_REJECT;
        }
    } else if (ph == 3) {
        if (tg)
            for (u32 i = lane; i < n * W; i += nl) {
                const u32 s = i / W;
                if (cnt[s] != XK_REJECT) R[i] = xk_tg_word(trows + (u64)s * 2 * W, i - s * W);
            }
        for (u32 k = lane; k * 8 < B; k += nl) {
            u32 a = 0;
            for (u32 s = 8 * k; s < 8 * k + 8 && s < B; ++s) a += cnt[s] == XK_REJECT ? 0 : cnt[s];
            part[k] = a;
        }
    } else if (ph == 4) {
        const bool over = shm[XK_SHM_NP] > E;
        if (lane == 0) c->overflow[block] = over ? XK_OVERFLOW_PAIRS : 0;
        for (u32 s = lane; s <= B; s += nl) {
            u32 a = 0;
            if (!over) {
                for (u32 k = 0; k < s / 8; ++k) a += part[k];
                for (u32 j = s & ~7u; j < s; ++j) a += cnt[j] == XK_REJECT ? 0 : cnt[j];
            }
            start[s] = a;
            fill[s] = 0;
            fp[s] = a;
        }
    } else if (ph == 5) {
        if (shm[XK_SHM_NP] > E) return;
        const u32 np = shm[XK_SHM_NP];
        for (u32 i = lane; i < np; i += nl) {
            const u32 s = xk_rec_shot(c, psh, i);
            if (cnt[s] == XK_REJECT) continue;
            kid[start[s] + xk_tg_inc(fill + s)] = xk_rec_off(c, psh, i);
        }
    } else if (ph == 6) {
        if (shm[XK_SHM_NP] > E) return;
        for (u32 s = lane; s < B; s += nl) {
            XK_BUF u32* L = kid + start[s];
            const u32 m = start[s + 1] - start[s];
            for (u32 i = 1; i < m; ++i) {   // insertion sort (a shot's few records)
                const u32 v = L[i];
                u32 j = i;
                for (; j > 0 && L[j - 1] > v; --j) L[j] = L[j - 1];
                L[j] = v;
            }
        }
    }
}

#define XK_WALK_PHASES 7

// K1 for one group (block `block`, thread `lane` of nl; on the C++ reference every thread in turn per phase): every
// thread of the group returns together (the conditions below are the group's), so each reaches every barrier
XK_FN void xk_walk_group(XK_LOC XkCtx* c, u32 block, u32 lane, u32 nl, XK_TG u32* shm) {
    if (block >= c->nblocks) return;   // a rounded-up grid
#if defined(XK_PARALLEL_GROUP)
    const u32 l0 = lane, l1 = lane + 1;
#else
    const u32 l0 = 0, l1 = nl;
#endif
    if (block == 0 && l0 == 0 && (c->mode & XK_MODE_DEFER)) c->lawq[0] = 0;   // K2's queue starts empty
    if (block == 0 && l0 == 0 && (c->mode & XK_MODE_EMIT)) c->cmiss[0] = 0;   // and K4's list of C ids not in ctab
    if (c->mode & XK_MODE_GIVEN) return;   // rows and lists are given: no walk
    if (!xk_caps_ok(c)) {
        XK_BUF u32* fp = c->epos + (u64)block * (c->B + 1);
        for (u32 l = l0; l < l1; ++l) {
            for (u32 s = l; s <= c->B; s += nl) fp[s] = 0;
            if (l == 0) c->overflow[block] = XK_OVERFLOW_CAPACITY;
        }
        return;
    }
    for (u32 ph = 0; ph < XK_WALK_PHASES; ++ph) {
        for (u32 l = l0; l < l1; ++l) xk_walk_phase(c, block, l, nl, shm, ph);
#if defined(XK_PARALLEL_GROUP)
        XK_BARRIER();
#endif
    }
}

// ------------------------------------------------------------------------------------------------ K2: resolve
// sort_ids: a shot's records into file order (record offsets increase with the line id), in place
XK_FN void xk_sort_ids(XK_BUF u32* L, u32 n) {
    for (u32 i = 1; i < n; ++i) {
        const u32 v = L[i];
        u32 j = i;
        for (; j > 0 && L[j - 1] > v; --j) L[j] = L[j - 1];
        L[j] = v;
    }
}

// compose_ext: eta mod 4 over the axes as two planes (eta_a = 2 hi[a] + lo[a]), lines in file order, each line's
// exponents negated where the sign register (the earlier lines' tau parity) is set. Record at xa + xo: header nE |
// nT << 32, then nE groups (word, odd bits, 2-bits), then nT tau entries (word, mask).
XK_FN void xk_compose(XK_LOC XkCtx* c, XK_BUF const u32* L, u32 n, XK_LOC u64* lo, XK_LOC u64* hi) {
    u64 sg[XK_CAP_WA];
    for (u32 w = 0; w < c->WA; ++w) lo[w] = hi[w] = sg[w] = 0;
    u32 prev = 0;   // offset 0 is no record
    for (u32 j = 0; j < n; ++j) {
        const u32 xo = L[j];
        if (prev) {   // line j - 1's tau, applied just before line j
            XK_BUF const u64* p = c->xa + prev;
            const u64 h = p[0];
            XK_BUF const u64* t = p + 1 + 3 * (h & 0xffffffffu);
            for (u64 k = 0; k < (h >> 32); ++k, t += 2) sg[t[0]] ^= t[1];
        }
        XK_BUF const u64* r = c->xa + xo;
        const u64 h = r[0];
        XK_BUF const u64* g = r + 1;
        for (u64 k = 0; k < (h & 0xffffffffu); ++k, g += 3) {
            const u64 w = g[0], e0 = g[1], e1 = g[2] ^ (g[1] & sg[w]);
            const u64 cy = lo[w] & e0;
            lo[w] ^= e0;
            hi[w] ^= e1 ^ cy;
        }
        prev = xo;
    }
}

// xor_beta: row ^= sum_{j : z_j = 1} beta_j (beta: rho vectors of WF words)
XK_FN void xk_xor_beta(XK_LOC XkCtx* c, XK_BUF u64* r, u64 z) {
    for (; z; z &= z - 1) {
        XK_BUF const u64* bv = c->beta + (u64)xk_ctz(z) * c->WF;
        for (u32 w = 0; w < c->WF; ++w) r[w] ^= bv[w];
    }
}

// ------------------------------------------------------------------------------------------------ zero decoder
// score(status, flips = 0)'s test, pass iff the observable row v lies in C, on the fast path's coordinates (law_in_c,
// kind 1): v is reduced by beta_L's observable parts bLo (pivots pL, in build order), which gives its beta_L
// coordinates s (bits rho - rhoL + i) and a rest; v is in C iff the rest is 0 and s lies in C. That reduction is
// linear in v, so s and the rest are read from the host's table (xtim.gpu.tables.obs_table): s = XOR_{o in v} stab[o],
// rest = XOR_{o in v} rtab[o] (every rest 0 when rtab_on = 0; a row has no observable bit past O). xk_obs_coords
// returns false when the rest is not 0, else s in *s.
XK_FN bool xk_obs_coords(XK_LOC const XkCtx* c, XK_BUF const u64* v, XK_LOC u64* s) {
    const u32 WO = c->WO;
    u64 x = 0;
    for (u32 w = 0; w < WO; ++w)
        for (u64 y = v[w]; y; y &= y - 1) x ^= c->stab[w * 64 + xk_ctz(y)];
    *s = x;
    if (!c->rtab_on) return true;
    u64 nz = 0;
    for (u32 k = 0; k < WO; ++k) {   // rest word k
        u64 r = 0;
        for (u32 w = 0; w < WO; ++w)
            for (u64 y = v[w]; y; y &= y - 1) r ^= c->rtab[(u64)(w * 64 + xk_ctz(y)) * WO + k];
        nz |= r;
    }
    return nz == 0;
}

// the test with C's key by pivot (H[b]: the key vector whose highest bit is b, hm bit b set; fully reduced); dim C = 0
// (hm = 0): v = 0
XK_FN bool xk_zero_passes(XK_LOC const XkCtx* c, XK_BUF const u64* row, XK_LOC const u64* H, u64 hm) {
    const u32 WO = c->WO;
    XK_BUF const u64* v = row + c->WD;
    if (hm == 0) {
        u64 nz = 0;
        for (u32 w = 0; w < WO; ++w) nz |= v[w];
        return nz == 0;
    }
    u64 s = 0;
    if (!xk_obs_coords(c, v, &s)) return false;
    for (u32 b = 0; b < XK_CAP_RHO; ++b) s ^= H[b] & ((u64)0 - ((s >> b) & (hm >> b) & 1));
    return s == 0;
}

// post_fast (PS = false, LAW = false) on the composed planes: XORs the outcome into the row r (in place: r is read
// by nothing here, and post_fast's row is written whatever the result) and returns dim C; C's key (post_fast's T.ckey:
// fully reduced over the beta_L coordinates, pivot = highest bit) goes to ko[0, dim C) in descending order (ko holds
// zeros before the call) and to H / hm by pivot: H[b] the key vector whose highest bit is b (hm bit b set), 0
// elsewhere (H is all 0 on entry). XK_CDIM_INTERNAL where post_fast throws std::logic_error (the row then left as it
// was). The odd axes are the set bits of lo.
//
// Two implementations with the same outcome, chosen at compile time by XK_CAP_RHO. Up to 32 (every bundled example):
// every thread-local array has XK_CAP_RHO entries and is indexed only by loop counters bounded by XK_CAP_RHO, so a
// compiler that unrolls those loops keeps the arrays in registers (an array indexed by data, such as col[ctz(T_a)],
// lives in stack memory: on an Apple M3 Max that alone cost cultivation_d5 2.9 ns per shot against 0.85 for the masked
// form). The objects computed are post_fast's, each unique, so the outcome is post_fast's for the same coin word:
//   col_j = XOR_{a odd, T_a bit j} k_a (r bits); the coin span (W^perp, W = ker N_O) is spanned by the vectors
//   g_i = (bit i of col_j)_j (i < r); its fully reduced echelon (pivot = lowest bit) is S[p] = e_p ^ Cp[p] for p in
//   P, kept by pivot (pm: the pivots). P is post_fast's pivot columns (a column is a pivot iff it is independent of
//   the earlier ones), npv = |P|; for a free column j (F: not in P) col_j = XOR_{p in P, Cp[p] bit j} col_p, so
//   sigma_j = e_j ^ XOR_{p in P, Cp[p] bit j} e_p (j in F) is post_fast's private-pivot kernel basis, and omega (bit
//   j = f(sigma_j), j in F) and its two internal checks are post_fast's; C = the coin span's vectors without beta_D
//   part = the span of S[p] for p in P at or past rho - rhoL, dim C = npv - npvD. Vectors over the beta coordinates
//   are 32-bit words (a 64-bit integer operation costs several 32-bit ones on a GPU).
// Past 32 those arrays would no longer fit in registers (merlin_bh50, rho = 53: 3.4 times slower than below), so
// post_fast's own construction is used, its loops running over the pivots and odd axes only.
#if XK_CAP_RHO <= 32
XK_FN u32 xk_bit32(u32 j) { return (u32)1 << j; }

XK_FN u32 xk_law_fast(XK_LOC XkCtx* c, XK_BUF u64* r, XK_LOC const u64* lo, XK_LOC const u64* hi, u32 shot,
                      XK_BUF u64* ko, XK_LOC u64* H, XK_LOC u64* hm, XK_LOC u32* zp) {
    const u32 rho = c->rho, rhoL = c->rhoL;
    *hm = 0;
    u64 zhi = 0;
    u32 m = 0;
    for (u32 w = 0; w < c->WA; ++w) {
        for (u64 y = hi[w]; y; y &= y - 1) zhi ^= c->Tc[w * 64 + xk_ctz(y)];
        for (u64 y = lo[w]; y; y &= y - 1) ++m;
    }
    if (m == 0) {
        xk_xor_beta(c, r, zhi);
        return 0;
    }
    u64 col[XK_CAP_RHO];
    for (u32 j = 0; j < XK_CAP_RHO; ++j) col[j] = 0;
    for (u32 w = 0; w < c->WA; ++w)
        for (u64 y = lo[w]; y; y &= y - 1) {
            const u32 a = w * 64 + xk_ctz(y);
            const u64 k = c->Kc[a], tt = c->Tc[a];   // T_a has no bit at or past rho
            for (u32 j = 0; j < XK_CAP_RHO; ++j) col[j] ^= k & ((u64)0 - ((tt >> j) & 1));
        }
    // the coin span's echelon from the g_i: each reduced by S in one pass (S is fully reduced, so XOR-ing S[p] changes
    // no other pivot bit), a new pivot clears its bit from the others
    u32 S[XK_CAP_RHO];
    u32 pm = 0;
    for (u32 p = 0; p < XK_CAP_RHO; ++p) S[p] = 0;
    u64 gi = 0;
    for (u32 j = 0; j < XK_CAP_RHO; ++j) gi |= col[j];
    for (; gi; gi &= gi - 1) {
        const u32 i = xk_ctz(gi);
        u32 v = 0;
        for (u32 j = 0; j < XK_CAP_RHO; ++j) v |= (u32)((col[j] >> i) & 1) << j;
        for (u32 p = 0; p < XK_CAP_RHO; ++p) v ^= S[p] & ((u32)0 - ((v >> p) & (pm >> p) & 1));
        if (!v) continue;
        const u32 tb = xk_ctz((u64)v);
        for (u32 p = 0; p < XK_CAP_RHO; ++p) S[p] = p == tb ? v : S[p] ^ (v & ((u32)0 - ((S[p] >> tb) & 1)));
        pm |= xk_bit32(tb);
    }
    const u32 fm = (rho >= 32 ? ~(u32)0 : xk_bit32(rho) - 1) & ~pm;   // the free columns
    u32 omega = 0;
    if (fm) {   // the W phase on sigma_j, bit-sliced over j: x_a(sigma_j) = T_a bit j ^ XOR_{p in P ∩ T_a} S[p] bit j
        u32 plo = 0, phi = 0, lin = 0, bad = 0;
        u32 Bq[XK_CAP_RHO];
        for (u32 q = 0; q < XK_CAP_RHO; ++q) Bq[q] = 0;
        for (u32 w = 0; w < c->WA; ++w)
            for (u64 y = lo[w]; y; y &= y - 1) {
                const u32 a = w * 64 + xk_ctz(y);
                const u32 tt = (u32)c->Tc[a];
                u32 cq = tt;
                for (u32 p = 0; p < XK_CAP_RHO; ++p) cq ^= S[p] & ((u32)0 - ((tt >> p) & 1));   // S = 0 off P
                cq &= fm;
                phi ^= plo & cq;
                plo ^= cq;
                if (c->bb[a]) lin ^= cq;
                for (u32 q = 0; q < XK_CAP_RHO; ++q) Bq[q] ^= cq & ((u32)0 - ((cq >> q) & 1));   // B(j, j')
            }
        for (u32 q = 0; q < XK_CAP_RHO; ++q) bad |= Bq[q];
        if (plo || bad) return XK_CDIM_INTERNAL;   // odd phase on W / W phase not linear
        omega = phi ^ lin;
    }
    // coins in beta coordinates: coin word t picks tau = XOR_{a odd, parity(t & Kc_a)} Tc_a
    u64 tau = 0;
    if (pm) {
        const u32 b = shot / c->B, s = shot % c->B;
        const u64 g = (c->first_block + b) * c->B + s;   // global shot index
        u64 t = (c->mode & XK_MODE_COIN_IN) ? c->coin_in[shot] : xk_mix(c->coin_key + (g + 1) * XK_GAMMA);
        t &= c->rmask;
        for (u32 w = 0; w < c->WA; ++w)
            for (u64 y = lo[w]; y; y &= y - 1) {
                const u32 a = w * 64 + xk_ctz(y);
                tau ^= c->Tc[a] & ((u64)0 - xk_parity(t & c->Kc[a]));
            }
    }
    xk_xor_beta(c, r, zhi ^ (u64)omega ^ tau);
    if (!rhoL) return 0;
    const u32 rD = rho - rhoL;
    if (!((u64)pm >> rD)) return 0;
    if (c->mode & XK_MODE_NOKEY) {   // dim C = |P past rD|; v in C iff s reduces to 0 by the S[p] (no key built)
        u64 s = 0;
        *zp = 0;
        if (xk_obs_coords(c, r + c->WD, &s)) {
            for (u32 p = 0; p < XK_CAP_RHO; ++p) s ^= (u64)S[p] & ((u64)0 - ((s >> p) & (pm >> p) & 1));
            *zp = s == 0;
        }
        return xk_popcount((u64)pm >> rD);
    }
    // C's key: the fully reduced echelon (pivot = highest bit) of S[p], p in P past rD, built in H by pivot
    u64 hmask = 0;   // H is all 0 on entry
    for (u32 p = 0; p < XK_CAP_RHO; ++p) {
        if (p < rD || !((pm >> p) & 1)) continue;
        u64 v = (u64)S[p];
        for (u32 b = 0; b < XK_CAP_RHO; ++b) v ^= H[b] & ((u64)0 - ((v >> b) & (hmask >> b) & 1));
        if (!v) continue;   // (the S[p] are independent: unreachable)
        const u32 tb = xk_highest(v);
        for (u32 b = 0; b < XK_CAP_RHO; ++b) H[b] = b == tb ? v : H[b] ^ (v & ((u64)0 - ((H[b] >> tb) & 1)));
        hmask |= xk_bit(tb);
    }
    u32 n = 0;
    for (u32 b = XK_CAP_RHO; b-- > 0;)
        if ((hmask >> b) & 1) ko[n++] = H[b];
    *hm = hmask;
    return n;
}
#else
XK_FN u32 xk_law_fast(XK_LOC XkCtx* c, XK_BUF u64* r, XK_LOC const u64* lo, XK_LOC const u64* hi, u32 shot,
                      XK_BUF u64* ko, XK_LOC u64* H, XK_LOC u64* hm, XK_LOC u32* zp) {
    const u32 rho = c->rho, rhoL = c->rhoL;
    *hm = 0;
    u64 zhi = 0;
    u32 m = 0;
    for (u32 w = 0; w < c->WA; ++w) {
        for (u64 y = hi[w]; y; y &= y - 1) zhi ^= c->Tc[w * 64 + xk_ctz(y)];
        for (u64 y = lo[w]; y; y &= y - 1) ++m;
    }
    if (m == 0) {
        xk_xor_beta(c, r, zhi);
        return 0;
    }
    u64 col[XK_CAP_RHO];
    for (u32 j = 0; j < rho; ++j) col[j] = 0;
    for (u32 w = 0; w < c->WA; ++w)
        for (u64 y = lo[w]; y; y &= y - 1) {
            const u32 a = w * 64 + xk_ctz(y);
            const u64 k = c->Kc[a];
            for (u64 tt = c->Tc[a]; tt; tt &= tt - 1) col[xk_ctz(tt)] ^= k;
        }
    u64 pv_vec[XK_CAP_RHO], pv_cmb[XK_CAP_RHO], pv_low[XK_CAP_RHO], Kt[XK_CAP_RHO];
    u32 depj[XK_CAP_RHO];
    u32 npv = 0, nker = 0, npvD = 0;
    for (u32 j = 0; j < rho; ++j) Kt[j] = 0;
    for (u32 j = 0; j < rho; ++j) {
        if (j == rho - rhoL) npvD = npv;   // the beta_D coordinates come first
        u64 v = col[j], cc = xk_bit(j);
        if (!v) {   // zero column: kernel vector e_j
            Kt[j] |= xk_bit(nker);
            depj[nker++] = j;
            continue;
        }
        for (u32 q = 0; q < npv; ++q) {   // pivot q's lowest bit set in v
            const u64 msk = (u64)0 - (u64)((v & pv_low[q]) != 0);
            v ^= pv_vec[q] & msk;
            cc ^= pv_cmb[q] & msk;
        }
        if (v) {
            pv_vec[npv] = v;
            pv_cmb[npv] = cc;
            pv_low[npv] = v & ((u64)0 - v);
            ++npv;
        } else {
            for (u64 x = cc; x; x &= x - 1) Kt[xk_ctz(x)] |= xk_bit(nker);
            depj[nker++] = j;
        }
    }
    u64 omega = 0;
    if (nker) {
        u64 plo = 0, phi = 0, lin = 0, bad = 0;
        u64 Bq[XK_CAP_RHO];
        for (u32 q = 0; q < nker; ++q) Bq[q] = 0;
        for (u32 w = 0; w < c->WA; ++w)
            for (u64 y = lo[w]; y; y &= y - 1) {
                const u32 a = w * 64 + xk_ctz(y);
                u64 cq = 0;
                for (u64 tt = c->Tc[a]; tt; tt &= tt - 1) cq ^= Kt[xk_ctz(tt)];
                phi ^= plo & cq;
                plo ^= cq;
                if (c->bb[a]) lin ^= cq;
                for (u64 x = cq; x; x &= x - 1) Bq[xk_ctz(x)] ^= cq;   // B(q, q') = sum_a c_a(q) c_a(q')
            }
        for (u32 q = 0; q < nker; ++q) bad |= Bq[q];
        if (plo || bad) return XK_CDIM_INTERNAL;   // odd phase on W / W phase not linear
        for (u64 rhs = phi ^ lin; rhs; rhs &= rhs - 1) omega |= xk_bit(depj[xk_ctz(rhs)]);
    }
    // coins in beta coordinates: coin word t picks tau = XOR_{a odd, parity(t & Kc_a)} Tc_a
    u64 tau = 0;
    if (npv) {
        const u32 b = shot / c->B, s = shot % c->B;
        const u64 g = (c->first_block + b) * c->B + s;   // global shot index
        u64 t = (c->mode & XK_MODE_COIN_IN) ? c->coin_in[shot] : xk_mix(c->coin_key + (g + 1) * XK_GAMMA);
        t &= c->rmask;
        for (u32 w = 0; w < c->WA; ++w)
            for (u64 y = lo[w]; y; y &= y - 1) {
                const u32 a = w * 64 + xk_ctz(y);
                tau ^= c->Tc[a] & ((u64)0 - xk_parity(t & c->Kc[a]));
            }
    }
    xk_xor_beta(c, r, zhi ^ omega ^ tau);
    // dim C = npv - npvD; C = the linear dependencies among the Kt[j], j in [rho - rhoL, rho)
    if (!rhoL || npv - npvD == 0) return 0;
    if (c->mode & XK_MODE_NOKEY) {   // v in C iff s is orthogonal to the kernel W (Kt[j] bit q = sigma_q[j]): no key
        u64 s = 0, acc = 0;
        *zp = 0;
        if (xk_obs_coords(c, r + c->WD, &s)) {
            for (; s; s &= s - 1) acc ^= Kt[xk_ctz(s)];
            *zp = acc == 0;
        }
        return npv - npvD;
    }
    u64 kern[XK_CAP_RHO], lv[XK_CAP_RHO], lc[XK_CAP_RHO], llow[XK_CAP_RHO];
    u32 nk = 0, nl = 0;
    for (u32 j = rho - rhoL; j < rho; ++j) {
        u64 v = Kt[j], cc = xk_bit(j);
        for (u32 q = 0; q < nl; ++q) {
            const u64 msk = (u64)0 - (u64)((v & llow[q]) != 0);
            v ^= lv[q] & msk;
            cc ^= lc[q] & msk;
        }
        if (v) {
            lv[nl] = v;
            lc[nl] = cc;
            llow[nl] = v & ((u64)0 - v);
            ++nl;
        } else {
            kern[nk++] = cc;
        }
    }
    if (nk != npv - npvD) return XK_CDIM_INTERNAL;   // status C dimension mismatch
    // fully reduced RREF over the beta_L coordinates (pivot = highest bit), then descending
    u64 key[XK_CAP_RHO];
    u32 n = 0;
    for (u32 i = 0; i < nk; ++i) {
        u64 v = kern[i];
        for (u32 k = 0; k < n; ++k)
            if (v & xk_bit(xk_highest(key[k]))) v ^= key[k];
        if (!v) continue;
        const u64 pb = xk_bit(xk_highest(v));
        for (u32 k = 0; k < n; ++k)
            if (key[k] & pb) key[k] ^= v;
        key[n++] = v;
    }
    for (u32 i = 1; i < n; ++i) {
        const u64 v = key[i];
        u32 j = i;
        for (; j > 0 && key[j - 1] < v; --j) key[j] = key[j - 1];
        key[j] = v;
    }
    for (u32 i = 0; i < n; ++i) {
        const u32 tb = xk_highest(key[i]);
        ko[i] = key[i];
        H[tb] = key[i];
        *hm |= xk_bit(tb);
    }
    return n;
}
#endif

// The single-line law table (xtim.gpu.tables.law_table): for an extended line whose odd mask lo_l is not 0 and whose
// law passes post_fast's internal checks, the objects xk_law_fast computes from lo = lo_l alone, so a shot whose
// composed lo equals a fired line's lo_l skips the elimination. xlaw[xo] (xo: the line's record offset into xa) is
// its entry's offset into lawt, 0 for none (lawt[0] is a pad word); entry e:
//   lawt[e + w], w < WA: lo_l;  lawt[e + WA]: omega;  lawt[e + WA + 1]: npv | dim C << 8;
//   then npv pairs (col_p, S[p]) for the pivots p in ascending order;  then dim C key vectors, descending.
// The outcome is xk_law_fast's for the same coin word: tau = XOR_{a odd, parity(t & k_a)} T_a has bit j = parity(t
// & col_j) and lies in the coin span, whose echelon is the S[p], so tau = XOR_p parity(t & col_p) S[p].
XK_FN u32 xk_law_entry(XK_LOC const XkCtx* c, XK_LOC const u64* lo, XK_BUF const u32* L, u32 ne) {
    for (u32 j = 0; j < ne; ++j) {
        const u32 e = c->xlaw[L[j]];
        if (!e) continue;
        u64 d = 0;
        for (u32 w = 0; w < c->WA; ++w) d |= lo[w] ^ c->lawt[e + w];
        if (!d) return e;
    }
    return 0;
}

// xk_law_fast's outcome from table entry e (above): the row, ko, H and hm as there; returns dim C
XK_FN u32 xk_law_table(XK_LOC XkCtx* c, XK_BUF u64* r, XK_LOC const u64* hi, u32 e, u32 shot, XK_BUF u64* ko,
                       XK_LOC u64* H, XK_LOC u64* hm) {
    XK_BUF const u64* t = c->lawt + e + c->WA;
    const u32 npv = (u32)(t[1] & 0xff), dim = (u32)(t[1] >> 8) & 0xff;
    u64 z = t[0];   // omega
    for (u32 w = 0; w < c->WA; ++w)
        for (u64 y = hi[w]; y; y &= y - 1) z ^= c->Tc[w * 64 + xk_ctz(y)];
    XK_BUF const u64* pv = t + 2;
    if (npv) {
        const u32 b = shot / c->B, s = shot % c->B;
        const u64 g = (c->first_block + b) * c->B + s;   // global shot index
        u64 cw = (c->mode & XK_MODE_COIN_IN) ? c->coin_in[shot] : xk_mix(c->coin_key + (g + 1) * XK_GAMMA);
        cw &= c->rmask;
        for (u32 k = 0; k < npv; ++k) z ^= pv[2 * k + 1] & ((u64)0 - xk_parity(cw & pv[2 * k]));
    }
    xk_xor_beta(c, r, z);
    XK_BUF const u64* key = pv + 2 * npv;
    const bool nokey = (c->mode & XK_MODE_NOKEY) != 0;
    u64 hmask = 0;
    for (u32 k = 0; k < dim; ++k) {
        const u64 v = key[k];
        const u32 tb = xk_highest(v);
        if (!nokey) ko[k] = v;
        for (u32 b = 0; b < XK_CAP_RHO; ++b) H[b] = b == tb ? v : H[b];
        hmask |= xk_bit(tb);
    }
    *hm = hmask;
    return dim;
}

// ------------------------------------------------------------------------------------------------ structured law
// xk_law_struct: the one-word law of a file whose law factors (xtim.gpu.structure: detected, certified, axes
// renumbered), from one word per axis type instead of an elimination over the rho coordinates. Notation (the design
// note on the structured law): a core of nG gauge bits and the rD beta_D coordinates, leaves i (gauge bit = coordinate
// = slot f_i), non-leaf axes (core-only and dense: kappa = k + eps u, T = t + eta v, u the leaves' gauge bits). A leaf
// axis of type tau on leaf i has kappa = k_tau + x_tau g_i, T = t_tau + y_tau f_i; its record's odd axes are the word
// W_tau (bit f_i: leaf i's axis is odd). Every vector has rho + 1 bits: the auxiliary coordinate tau = bit rho lifts
// the dense rows (W' = span{R_gamma, leaf rows c_i + d_i f_i + u_i tau, T_dense + tau}; the coin span S = {y : (y, 0)
// in W'}, rank M = dim W' - 1, C = {y in W' : y_D = 0, y_tau = 0}). The lifted core coordinates are k < rD (beta_D
// bit k) and k = rD (tau); C[k] holds the leaves' core parts bit-sliced (slot f_i: leaf i's row has core bit k).
// The rows' echelon: the leaves with d_i = 1 (S1) are private pivots f_i + c_i; the rest (the R_gamma, T_dense + tau
// and a basis of the S0 leaves' core parts, reduced by S1) is fully reduced into E (pivot: highest leaf bit, else
// highest bit), at most nG + 1 + rD + 1 vectors: zD (core pivots) and the w-atoms (leaf pivots). Cp: C reduced modulo
// zD. Then dim C = |S1| + #w-atoms - rank{images}, tau(t) = M^T t from the rows (the generic law's vector), the
// phase's offset omega from n_j mod 4 over the free columns (per type: popcounts; the free leaf columns: a 2-bit
// counter over the slots), equal to xk_law_fast's modulo S (the same law: the coin word t -> t ^ t0 with M^T t0 = the
// difference is a bijection); C's key exactly (the dependent atoms in descending pivot order, each completed by the
// unique combination of the independent atoms with the same image); the zero decoder's test as membership in W'.
// The file is certified (T_a linear in kappa_a), so post_fast's internal checks cannot fail and are not made.
// Thread-local arrays here are written at indices fixed by unrolled loops (the core echelon's slot k: coordinate k)
// or element by element through a select (E's and the key's slot n), never by a store guarded by a run-time slot
// test: Metal (Apple M3 Max) compiled the core echelon's earlier form, `if (k == n) { G[k] = v; ... }`, so that E's
// pivots read back as 0 at the merlin_bh capacities (wrong rows, keys and XK_CDIM_INTERNAL), while the C++ reference
// and other capacities were right.
// Returns dim C (the key to ko[0, dim C) in descending order unless XK_MODE_NOKEY; *zp = the zero decoder's verdict,
// 1 / 0, set where the shot has odd axes) after XORing the correction into the row r.
#if defined(XK_CAP_ST_T)
#define XK_ST_E (XK_CAP_ST_NG + 1 + XK_CAP_ST_NC)   // E's entries
#define XK_ST_HDR 16                                // sct's header words (xtim.gpu.structure.st_table)

// lifted core coordinate k as a one-bit word (0 past tau = bit rho at k = rD)
XK_FN u64 xk_st_cb(u32 k, u32 rD, u64 taub) { return k < rD ? xk_bit(k) : (k == rD ? taub : 0); }

// a type record's odd axes as a word over the slots: plane bits [off, off + n) moved to [s0, s0 + n) (n < 64)
XK_FN u64 xk_st_word(XK_LOC const u64* plane, u32 WA, u64 rec) {
    const u32 off = (u32)(rec & 0xffff), n = (u32)(rec >> 16) & 0xff, s0 = (u32)(rec >> 24) & 0xff;
    const u32 w = off >> 6, b = off & 63;
    u64 x = plane[w] >> b;
    if (b && w + 1 < WA) x |= plane[w + 1] << (64 - b);
    return (x & (xk_bit(n) - 1)) << s0;
}

// L(T): the axis vector T's coefficients on the kernel basis of W' (T reduced by the lifted echelon: S1's rows through
// Cp, then E), on the free columns fm
XK_FN u64 xk_st_L(u64 T, u64 S1, XK_LOC const u64* Cp, XK_LOC const u64* Ev, XK_LOC const u64* Ep, u64 fm, u32 rD,
                  u64 taub) {
    const u64 x = T & S1;
    u64 q = T ^ x;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) q ^= xk_st_cb(k, rD, taub) & ((u64)0 - xk_parity(Cp[k] & x));
    for (u32 e = 0; e < XK_ST_E; ++e) q ^= Ev[e] & ((u64)0 - (u64)((T & Ep[e]) != 0));
    return q & fm;
}

// w reduced by S1's rows f_i + c_i: (w & ~S1) ^ sum_k parity(C[k] & w & S1) e_k
XK_FN u64 xk_st_red(u64 w, u64 S1, XK_LOC const u64* C, u32 rD, u64 taub) {
    const u64 x = w & S1;
    w &= ~S1;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) w ^= xk_st_cb(k, rD, taub) & ((u64)0 - xk_parity(C[k] & x));
    return w;
}

// the pivot columns (lowest bits) of an echelon of the rows C[k] & S: the columns there are a basis of the columns'
// span
XK_FN u64 xk_st_pivots(XK_LOC const u64* C, u64 S) {
    u64 X[XK_CAP_ST_NC], P[XK_CAP_ST_NC], pv = 0;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) X[k] = P[k] = 0;   // (the later entries are 0: no effect below)
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
        u64 v = C[k] & S;
        for (u32 q = 0; q < XK_CAP_ST_NC; ++q) v ^= X[q] & ((u64)0 - (u64)((v & P[q]) != 0));
        X[k] = v;
        P[k] = v & ((u64)0 - v);
        pv |= P[k];
    }
    return pv;
}

// slot p's column of the bit-sliced core parts C (p a one-bit word) as a core vector
XK_FN u64 xk_st_col(XK_LOC const u64* C, u64 p, u32 rD, u64 taub) {
    u64 v = 0;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) v |= xk_st_cb(k, rD, taub) & ((u64)0 - (u64)((C[k] & p) != 0));
    return v;
}

// v into the core-space echelon G, slot k holding the entry whose highest bit is the lifted coordinate k (k descending
// = bits descending; every slot is written at its own index, no slot chosen at run time): 1 if v is independent
XK_FN u32 xk_st_core_insert(u64 v, XK_LOC u64* G, u32 rD, u64 taub) {
    u32 ins = 0;
    for (u32 i = 0; i < XK_CAP_ST_NC; ++i) {
        const u32 k = XK_CAP_ST_NC - 1 - i;
        if (!(v & xk_st_cb(k, rD, taub))) continue;
        if (G[k]) {
            v ^= G[k];
        } else {
            G[k] = v;
            ins = 1;
            v = 0;
        }
    }
    return ins;
}

// w reduced into E (fully reduced, pivot = highest leaf bit, else highest bit); *nE grows when w is independent
XK_FN void xk_st_insert(u64 w, u64 Lall, XK_LOC u64* Ev, XK_LOC u64* Ep, XK_LOC u32* nE) {
    for (u32 e = 0; e < XK_ST_E; ++e) w ^= Ev[e] & ((u64)0 - (u64)((w & Ep[e]) != 0));
    if (!w) return;
    const u64 lp = w & Lall;
    const u64 pb = xk_bit(xk_highest(lp ? lp : w));
    const u32 n = *nE;
    for (u32 e = 0; e < XK_ST_E; ++e) {   // (selects, not a guarded store: the note above)
        const u64 v = Ev[e] ^ (w & ((u64)0 - (u64)((Ev[e] & pb) != 0)));
        Ev[e] = e == n ? w : v;
        Ep[e] = e == n ? pb : Ep[e];
    }
    *nE = n + 1;
}

XK_FN u32 xk_law_struct(XK_LOC XkCtx* c, XK_BUF u64* r, XK_LOC const u64* lo, XK_LOC const u64* hi, u32 shot,
                        XK_BUF u64* ko, XK_LOC u32* zp) {
    XK_BUF const u64* st = c->sct;
    XK_BUF const u64* rec = st + XK_ST_HDR;            // the type records, two words each
    XK_BUF const u64* nla = rec + 2 * (u64)c->st_nt;   // the non-leaf axes, two words each
    const u32 WA = c->WA, nt = c->st_nt, nn = c->st_nn, ng = c->st_ng, rD = c->rho - c->rhoL;
    const u64 taub = xk_bit(c->rho), Lall = st[0], uslot = st[1], nlm = st[2];
    const u64 cm = (xk_bit(rD) - 1) | taub;            // the lifted core coordinates
    // 0. zhi = XOR of T_a over the 2-bits of eta (hi)
    u64 zhi = 0, odd = 0;
    for (u32 w = 0; w < WA; ++w) odd |= lo[w];
    for (u32 j = 0; j < XK_CAP_ST_T; ++j) {
        if (j >= nt) break;
        const u64 rc = rec[2 * j], tD = rec[2 * j + 1];
        const u64 x = xk_st_word(hi, WA, rc);
        zhi ^= (tD & ((u64)0 - xk_parity(x))) ^ (x & ((u64)0 - ((rc >> 33) & 1)));
    }
    {
        const u64 h = hi[0] & nlm;
        for (u32 j = 0; j < XK_CAP_ST_N; ++j)
            if (j < nn && ((h >> j) & 1)) zhi ^= nla[2 * j];
    }
    if (!odd) {
        xk_xor_beta(c, r, zhi);
        return 0;
    }
    // 1. the type words; the leaf rows bit-sliced (d: diagonal, C[k]: core parts), the core rows R, T_dense
    u64 Wt[XK_CAP_ST_T], C[XK_CAP_ST_NC], R[XK_CAP_ST_NG];
    u64 d = 0, Td = 0, Tbb = 0;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) C[k] = 0;
    for (u32 i = 0; i < XK_CAP_ST_NG; ++i) R[i] = 0;
    for (u32 j = 0; j < XK_CAP_ST_T; ++j) {
        Wt[j] = 0;
        if (j >= nt) continue;
        const u64 rc = rec[2 * j], tD = rec[2 * j + 1];
        const u64 x = xk_st_word(lo, WA, rc);
        Wt[j] = x;
        const u64 mx = (u64)0 - ((rc >> 32) & 1), my = (u64)0 - ((rc >> 33) & 1);
        d ^= x & mx & my;
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) C[k] ^= x & mx & ((u64)0 - ((tD >> k) & 1));   // tD: bits < rD
        const u64 T = (tD & ((u64)0 - xk_parity(x))) ^ (x & my);   // XOR of the record's odd axes' T
        const u32 kg = (u32)(rc >> 40) & 0xff;
        for (u32 i = 0; i < XK_CAP_ST_NG; ++i) R[i] ^= T & ((u64)0 - (u64)((kg >> i) & 1));
        Tbb ^= T & ((u64)0 - ((rc >> 34) & 1));
    }
    const u64 nlo = lo[0] & nlm;
    for (u32 j = 0; j < XK_CAP_ST_N; ++j) {
        if (j >= nn || !((nlo >> j) & 1)) continue;
        const u64 T = nla[2 * j], f = nla[2 * j + 1];
        for (u32 i = 0; i < XK_CAP_ST_NG; ++i) R[i] ^= T & ((u64)0 - ((f >> i) & 1));
        Td ^= T & ((u64)0 - ((f >> 8) & 1));
        Tbb ^= T & ((u64)0 - ((f >> 9) & 1));
    }
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) C[k] = k == rD ? uslot : C[k];   // the lifted rows' tau slice: u
    const u64 S1 = d;
    u64 Z = d;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) Z |= C[k];
    const u64 S0 = Z & ~d;
    // 2. E: R_gamma, T_dense + tau and a basis of the S0 leaves' core parts (their columns at the pivots of the rows
    // C[k] & S0: pure core, untouched by S1), each reduced by S1's rows, fully reduced into E
    u64 Ev[XK_ST_E], Ep[XK_ST_E];
    u32 nE = 0;
    for (u32 e = 0; e < XK_ST_E; ++e) Ev[e] = Ep[e] = 0;
    for (u32 i = 0; i < XK_CAP_ST_NG; ++i)
        if (i < ng) xk_st_insert(xk_st_red(R[i], S1, C, rD, taub), Lall, Ev, Ep, &nE);
    xk_st_insert(xk_st_red(Td | taub, S1, C, rD, taub), Lall, Ev, Ep, &nE);
    for (u64 pv = xk_st_pivots(C, S0); pv; pv &= pv - 1)
        xk_st_insert(xk_st_col(C, pv & ((u64)0 - pv), rD, taub), Lall, Ev, Ep, &nE);
    const u32 rank = xk_popcount(S1) + nE - 1;   // dim W' - 1
    // 3. Cp = C reduced modulo zD (E's core-pivot vectors, pure core: a pivot coordinate's slice is cleared, the
    // others take it where the vector has their bit); the images' rank (the S1 leaves': Cp's columns, the w-atoms':
    // their core parts) in core space; dim C = |S1| + #w-atoms - that rank
    u64 Cp[XK_CAP_ST_NC];
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) Cp[k] = C[k];
    u64 wpiv = 0;   // the w-atoms' pivots
    for (u32 e = 0; e < XK_ST_E; ++e) {
        const u64 p = Ep[e];
        wpiv |= p & Lall;
        if (!p || (p & Lall)) continue;
        u64 cpv = 0;   // C's slice at the pivot coordinate
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) cpv ^= C[k] & ((u64)0 - (u64)(xk_st_cb(k, rD, taub) == p));
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k)
            Cp[k] ^= cpv & ((u64)0 - (u64)((Ev[e] & xk_st_cb(k, rD, taub)) != 0));
    }
    u32 irank = 0;
    u64 G[XK_CAP_ST_NC];   // a core-space echelon
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) G[k] = 0;
    for (u64 pv = xk_st_pivots(Cp, S1); pv; pv &= pv - 1)
        irank += xk_st_core_insert(xk_st_col(Cp, pv & ((u64)0 - pv), rD, taub), G, rD, taub);
    for (u32 e = 0; e < XK_ST_E; ++e)
        if (Ep[e] & Lall) irank += xk_st_core_insert(Ev[e] & cm, G, rD, taub);
    const u32 nw = xk_popcount(wpiv);
    const u32 dim = xk_popcount(S1) + nw - irank;
    // 4. the coins: tau = M^T t = XOR_{gamma in t} R_gamma ^ (d & tL) ^ sum_k parity(C[k] & tL) e_k ^ <t, u> T_dense
    u64 tau = 0;
    if (rank) {
        const u32 b = shot / c->B, s = shot % c->B;
        const u64 g = (c->first_block + b) * c->B + s;   // global shot index
        u64 t = (c->mode & XK_MODE_COIN_IN) ? c->coin_in[shot] : xk_mix(c->coin_key + (g + 1) * XK_GAMMA);
        t &= c->rmask;
        const u64 tL = t & Lall;
        for (u32 i = 0; i < XK_CAP_ST_NG; ++i)
            if (i < ng) tau ^= R[i] & ((u64)0 - (u64)((t & st[8 + i]) != 0));
        tau ^= d & tL;
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k)
            if (k < rD) tau ^= xk_bit(k) & ((u64)0 - xk_parity(C[k] & tL));
        tau ^= Td & ((u64)0 - xk_parity(t & uslot));
    }
    // 5. omega: phi_j = bit 1 of n_j = #{odd a : L(T_a)_j = 1} on the free columns, plus the linear term L(T_bb)
    u64 piv = S1;
    for (u32 e = 0; e < XK_ST_E; ++e) piv |= Ep[e];
    const u64 fm = (taub | (taub - 1)) & ~piv, JL = fm & Lall;
    u64 Q[XK_CAP_ST_NC];   // free core column k: the leaves whose L(f_i) has bit k (S1: Cp; w-pivots: E)
    u32 n[XK_CAP_ST_NC];
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
        u64 q = Cp[k] & S1;
        const u64 cb = xk_st_cb(k, rD, taub);
        for (u32 e = 0; e < XK_ST_E; ++e) q |= Ep[e] & Lall & ((u64)0 - (u64)((Ev[e] & cb) != 0));
        Q[k] = q;
        n[k] = 0;
    }
    u64 c0 = 0, c1 = 0;   // the odd y-axes per leaf, mod 4, bit-sliced over the slots
    for (u32 j = 0; j < XK_CAP_ST_T; ++j) {
        if (j >= nt) break;
        const u64 rc = rec[2 * j], x = Wt[j];
        const u64 lt = xk_st_L(rec[2 * j + 1], S1, Cp, Ev, Ep, fm, rD, taub);
        const u32 pw = xk_popcount(x);
        const bool y = ((rc >> 33) & 1) != 0;
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
            const u64 cb = xk_st_cb(k, rD, taub);
            if (!(fm & cb)) continue;
            const bool l = (lt & cb) != 0;
            if (y) {
                const u32 q = xk_popcount(x & Q[k]);
                n[k] += l ? pw - q : q;
            } else if (l) {
                n[k] += pw;
            }
        }
        if (y) {
            c1 ^= c0 & x;
            c0 ^= x;
        }
    }
    for (u32 e = 0; e < XK_ST_E; ++e) {   // a w-pivot leaf p's m_p odd y-axes each carry E[p] on the free leaf columns
        const u64 p = Ep[e] & Lall;
        if (!p) continue;
        const u64 x = Ev[e] & JL;
        if (c0 & p) {
            c1 ^= c0 & x;
            c0 ^= x;
        }
        if (c1 & p) c1 ^= x;
    }
    u64 lc0 = c0 & JL, lc1 = c1 & JL;
    for (u32 j = 0; j < XK_CAP_ST_N; ++j) {   // the odd non-leaf axes, one by one
        if (j >= nn || !((nlo >> j) & 1)) continue;
        const u64 q = xk_st_L(nla[2 * j], S1, Cp, Ev, Ep, fm, rD, taub);
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) n[k] += (u32)((q & fm & xk_st_cb(k, rD, taub)) != 0);
        const u64 x = q & JL;
        lc1 ^= lc0 & x;
        lc0 ^= x;
    }
    u64 wh = lc1;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) wh |= xk_st_cb(k, rD, taub) & fm & ((u64)0 - (u64)((n[k] >> 1) & 1));
    if (st[3]) wh ^= xk_st_L(Tbb, S1, Cp, Ev, Ep, fm, rD, taub);   // the linear term (some axis has a bb bit)
    const u64 omega = (wh & ~taub) ^ (Td & ((u64)0 - (u64)((wh & taub) != 0)));
    xk_xor_beta(c, r, zhi ^ omega ^ tau);
    // 6. the zero decoder: v in C iff its coordinates s (rest 0) reduce to 0 in W' with no core part left
    if (dim == 0) {
        u64 nz = 0;
        for (u32 w = 0; w < c->WO; ++w) nz |= r[c->WD + w];
        *zp = nz == 0;
    } else {
        u64 s = 0;
        *zp = 0;
        if (xk_obs_coords(c, r + c->WD, &s)) {
            const u64 x = s & S1;
            u64 img = 0, r2 = s & ~S1;
            for (u32 k = 0; k < XK_CAP_ST_NC; ++k) img ^= xk_st_cb(k, rD, taub) & ((u64)0 - xk_parity(Cp[k] & x));
            for (u32 e = 0; e < XK_ST_E; ++e) {
                const u64 m = (u64)0 - (u64)((r2 & Ep[e] & Lall) != 0);
                r2 ^= Ev[e] & Lall & m;
                img ^= Ev[e] & cm & m;
            }
            *zp = (r2 | img) == 0;
        }
    }
    if (dim == 0 || (c->mode & XK_MODE_NOKEY)) return dim;
    // 7. C's key. The atoms (S1 leaves: image Cp's column, leaf part e_p; w-atoms: image E & cm, leaf part E & Lall)
    // in ascending pivot order; the independent ones (image outside the span of the earlier ones' images) found
    // greedily, the S1 leaves' images reduced bit-sliced (X), the w-atoms' one by one (wr)
    u64 X[XK_CAP_ST_NC], wr[XK_ST_E], BI[XK_CAP_ST_NC], BL[XK_CAP_ST_NC];
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
        X[k] = Cp[k] & S1;
        BI[k] = BL[k] = 0;
    }
    for (u32 e = 0; e < XK_ST_E; ++e) wr[e] = Ev[e] & cm & ((u64)0 - (u64)((Ep[e] & Lall) != 0));
    u64 IS = 0, IW = 0;   // the independent S1 leaves, the independent w-atoms' pivots
    u32 na = 0;
    for (u32 round = 0; round <= XK_CAP_ST_NC; ++round) {
        u64 notin = 0;
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) notin |= X[k];
        const u64 lp = notin & ((u64)0 - notin);
        u64 wp = 0, row = 0, img = 0, leaf = 0;
        for (u32 e = 0; e < XK_ST_E; ++e) {   // the lowest-pivot w-atom outside the span
            const u64 p = Ep[e];
            if (wr[e] && (!wp || p < wp)) {
                wp = p;
                row = wr[e];
                img = Ev[e] & cm;
                leaf = Ev[e] & Lall;
            }
        }
        if (!lp && !wp) break;
        if (lp && (!wp || lp < wp)) {
            row = 0;
            img = 0;
            for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
                row |= xk_st_cb(k, rD, taub) & ((u64)0 - (u64)((X[k] & lp) != 0));
                img |= xk_st_cb(k, rD, taub) & ((u64)0 - (u64)((Cp[k] & lp) != 0));
            }
            leaf = lp;
            IS |= lp;
        } else {
            IW |= wp;
        }
        for (u32 a = 0; a < XK_CAP_ST_NC; ++a) {
            BI[a] = a == na ? img : BI[a];
            BL[a] = a == na ? leaf : BL[a];
        }
        ++na;
        // reduce the remaining images by row (pivot: its highest bit, which no earlier row has)
        const u64 pb = xk_bit(xk_highest(row));
        u64 M = 0;
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) M |= X[k] & ((u64)0 - (u64)(xk_st_cb(k, rD, taub) == pb));
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) X[k] ^= M & ((u64)0 - (u64)((row & xk_st_cb(k, rD, taub)) != 0));
        for (u32 e = 0; e < XK_ST_E; ++e) wr[e] ^= row & ((u64)0 - (u64)((wr[e] & pb) != 0));
    }
    // the independent images' fully reduced echelon with combinations: an image y in their span is XOR_{a in lam} BI[a]
    // with lam = XOR_{c : y has GP[c]} GC[c]
    u64 GR[XK_CAP_ST_NC], GP[XK_CAP_ST_NC];
    u32 GC[XK_CAP_ST_NC];
    for (u32 a = 0; a < XK_CAP_ST_NC; ++a) {   // (the entries past a are 0 while a is inserted: no effect)
        GR[a] = GP[a] = 0;
        GC[a] = 0;
    }
    for (u32 a = 0; a < XK_CAP_ST_NC; ++a) {
        u64 v = BI[a];
        u32 cc = a < na ? 1u << a : 0u;
        for (u32 q = 0; q < XK_CAP_ST_NC; ++q) {
            const u64 m = (u64)0 - (u64)((v & GP[q]) != 0);
            v ^= GR[q] & m;
            cc ^= GC[q] & (u32)m;
        }
        const u64 pb = v ? xk_bit(xk_highest(v)) : 0;
        for (u32 q = 0; q < XK_CAP_ST_NC; ++q) {
            const u64 m = (u64)0 - (u64)((GR[q] & pb) != 0);
            GR[q] ^= v & m;
            GC[q] ^= cc & (u32)m;
        }
        GR[a] = v;
        GP[a] = pb;
        GC[a] = cc;
    }
    // per independent atom a, the S1 leaves whose image's combination holds a (bit-sliced)
    u64 Lam[XK_CAP_ST_NC];
    for (u32 a = 0; a < XK_CAP_ST_NC; ++a) {
        u64 x = 0;
        for (u32 q = 0; q < XK_CAP_ST_NC; ++q) {
            if (!((GC[q] >> a) & 1)) continue;
            for (u32 k = 0; k < XK_CAP_ST_NC; ++k) x ^= Cp[k] & ((u64)0 - (u64)(xk_st_cb(k, rD, taub) == GP[q]));
        }
        Lam[a] = x;
    }
    // the dependent atoms, descending: key vector = leaf part ^ the independent atoms' leaf parts of its combination
    u64 dep = (S1 & ~IS) | (wpiv & ~IW);
    u32 nk = 0;
    for (u32 i = 0; i < 64 && dep; ++i) {
        const u32 pi = xk_highest(dep);
        const u64 p = xk_bit(pi);
        dep ^= p;
        u64 v = 0;
        if (S1 & p) {
            v = p;
            for (u32 a = 0; a < XK_CAP_ST_NC; ++a) v ^= BL[a] & ((u64)0 - (u64)((Lam[a] & p) != 0));
        } else {
            u64 img = 0;
            for (u32 e = 0; e < XK_ST_E; ++e)
                if (Ep[e] == p) {
                    img = Ev[e] & cm;
                    v = Ev[e] & Lall;
                }
            u32 lam = 0;
            for (u32 q = 0; q < XK_CAP_ST_NC; ++q) lam ^= GC[q] & (0u - (u32)((img & GP[q]) != 0));
            for (u32 a = 0; a < XK_CAP_ST_NC; ++a) v ^= BL[a] & ((u64)0 - (u64)((lam >> a) & 1));
        }
        if (nk < dim) ko[nk] = v;
        ++nk;
    }
    return nk == dim ? dim : XK_CDIM_INTERNAL;   // (the dependent atoms number dim C: a failure is a bug)
}
#endif

// ------------------------------------------------------------------------------------------------ multi-word law
// The CPU's multi-word fast path (post_multi, PS = false, LAW = false; build_multi's coordinates): every vector is cw
// words (coord_words), coordinates are over the canonical basis gamma of span(img) (GcW[a]: img_a's coordinates;
// gamma: the basis vectors, rho of WF words; GbW[j]: beta_j's coordinates), kappa_a is KcW[a] (r bits). Compiled only
// when XK_CAP_CW > 1, so a one-word file's kernels hold none of it.
//
// Scratch: no thread-local array (its size grows with rho * cw, unbounded): shot s's working vectors live in the device
// buffer mws, interleaved over the batch's shots (word i of shot s at mws[i * nshots + s], so the shots of a SIMD group
// touch neighbouring words). xk_mw_words(rho, cw) = 8 cw + 4 rho cw + 4 rho words per shot (xtim.gpu.tables.mw_words):
//   vectors (cw words): ZG (the correction's gamma coordinates), TV, TC (a vector being reduced and its combination),
//   PLO, PHI, LIN (the W phase's accumulators), CQ, TT (the coin word);
//   rho x cw rows: COL (col_l, r bits, reduced in place: COL[pcol[q]] is pivot q's vector), PVC (pivot q's
//   combination), KT (Kt[l] bit q = sigma_q[l]: the kernel basis), BQ (the W phase's bilinear form);
//   rho words each: PW, PVM (pivot q's lowest word and bit), DEPJ (kernel vector q's dependent column), PCOL (pivot q's
//   column).
// The status pass reuses COL (its reduced vectors), BQ (their combinations), PVC (the kernel rows, then C's key),
// PW / PVM (its pivots), PCOL (the key's pivot bits), DEPJ (the key's order).
// Every word is written before it is read, by the same thread, within one shot.
#if XK_CAP_CW > 1
XK_FN u64 xk_mw_words(u32 rho, u32 cw) { return (u64)8 * cw + (u64)4 * rho * cw + (u64)4 * rho; }

// m[off + i] = 0 for i < n (scratch words, stride S)
XK_FN void xk_mz(XK_BUF u64* m, u64 S, u64 off, u64 n) {
    for (u64 i = 0; i < n; ++i) m[(off + i) * S] = 0;
}

// scratch vector dst ^= scratch vector src (cw words)
XK_FN void xk_mx(XK_BUF u64* m, u64 S, u64 dst, u64 src, u32 cw) {
    for (u32 q = 0; q < cw; ++q) m[(dst + q) * S] ^= m[(src + q) * S];
}

// scratch vector dst = scratch vector src (cw words)
XK_FN void xk_mc(XK_BUF u64* m, u64 S, u64 dst, u64 src, u32 cw) {
    for (u32 q = 0; q < cw; ++q) m[(dst + q) * S] = m[(src + q) * S];
}

// scratch vector dst ^= table vector t (cw words)
XK_FN void xk_mxt(XK_BUF u64* m, u64 S, u64 dst, XK_BUF const u64* t, u32 cw) {
    for (u32 q = 0; q < cw; ++q) m[(dst + q) * S] ^= t[q];
}

// bit p of scratch vector v
XK_FN bool xk_mbit(XK_BUF const u64* m, u64 S, u64 v, u32 p) { return ((m[(v + (p >> 6)) * S] >> (p & 63)) & 1) != 0; }

// the index of scratch vector v's lowest nonzero word, cw if it is 0
XK_FN u32 xk_mlow(XK_BUF const u64* m, u64 S, u64 v, u32 cw) {
    u32 q = 0;
    while (q < cw && !m[(v + q) * S]) ++q;
    return q;
}

// the highest set bit of scratch vector v, or -1 (as 0xFFFFFFFF) if it is 0
XK_FN u32 xk_mhigh(XK_BUF const u64* m, u64 S, u64 v, u32 cw) {
    for (u32 q = cw; q-- > 0;) {
        const u64 x = m[(v + q) * S];
        if (x) return q * 64 + xk_highest(x);
    }
    return 0xFFFFFFFFu;
}

// row r ^= sum_l z_l gamma_l, z = scratch vector v (xor_gamma, one basis vector per set bit)
XK_FN void xk_xor_gamma(XK_LOC const XkCtx* c, XK_BUF u64* r, XK_BUF const u64* m, u64 S, u64 v) {
    for (u32 u = 0; u < c->cw; ++u)
        for (u64 z = m[(v + u) * S]; z; z &= z - 1) {
            XK_BUF const u64* g = c->gamma + (u64)(u * 64 + xk_ctz(z)) * c->WF;
            for (u32 w = 0; w < c->WF; ++w) r[w] ^= g[w];
        }
}

// coin word k (< cw) of the shot: word 0 is the one-word law's (keyed per shot by coin_key), word k >= 1 the same
// counter under the key mix(coin_key ^ mix(k)); bits at or past r cleared, so the cw words are a uniform t in GF(2)^r
XK_FN u64 xk_coin_word(XK_LOC const XkCtx* c, u32 shot, u32 k, u32 r) {
    u64 t;
    if (c->mode & XK_MODE_COIN_IN) {
        t = c->coin_in[(u64)shot * c->cw + k];
    } else {
        const u32 b = shot / c->B, s = shot % c->B;
        const u64 g = (c->first_block + b) * c->B + s;   // global shot index
        const u64 key = k ? xk_mix(c->coin_key ^ xk_mix((u64)k)) : c->coin_key;
        t = xk_mix(key + (g + 1) * XK_GAMMA);
    }
    if (r <= 64 * k) return 0;
    return r - 64 * k >= 64 ? t : t & ((xk_bit(r - 64 * k)) - 1);
}

// post_multi on the composed planes (lo: the odd axes, hi: eta's 2-bits): XORs the outcome into the row r and returns
// dim C, writing C's key (in beta coordinates, bits [rho - rhoL, rho) of cw-word vectors: the fully reduced echelon,
// pivot = highest bit, descending pivots, post_multi's T.ckey) to key[0, dim C * cw); XK_CDIM_INTERNAL where post_multi
// throws std::logic_error. The coins: t uniform in GF(2)^r gives XOR_{a odd, <t, kappa_a> = 1} GcW[a], a uniform
// element of the coin span (the image of t -> (<t, col_l>)_l, whose annihilator is the kernel W of the columns col_l):
// post_multi's law, drawn with other coins.
XK_FN u32 xk_law_multi(XK_LOC XkCtx* c, XK_BUF u64* r, XK_LOC const u64* lo, XK_LOC const u64* hi, u32 shot,
                       XK_BUF u64* key, u32 slot) {
    const u32 CW = c->cw, R = c->rho, rD = c->rho - c->rhoL;
    const u64 S = c->nshots;
    XK_BUF u64* m = c->mws + slot;
    const u64 ZG = 0, TV = CW, TC = 2 * CW, PLO = 3 * CW, PHI = 4 * CW, LIN = 5 * CW, CQ = 6 * CW, TT = 7 * CW;
    const u64 COL = 8 * (u64)CW, PVC = COL + (u64)R * CW, KT = PVC + (u64)R * CW, BQ = KT + (u64)R * CW;
    const u64 PW = BQ + (u64)R * CW, PVM = PW + R, DEPJ = PVM + R, PCOL = DEPJ + R;
    xk_mz(m, S, ZG, CW);
    u32 nodd = 0;
    for (u32 w = 0; w < c->WA; ++w) {
        for (u64 y = hi[w]; y; y &= y - 1) xk_mxt(m, S, ZG, c->GcW + (u64)(w * 64 + xk_ctz(y)) * CW, CW);
        for (u64 y = lo[w]; y; y &= y - 1) ++nodd;
    }
    if (nodd == 0) {
        xk_xor_gamma(c, r, m, S, ZG);
        return 0;
    }
    // col_l = XOR_{a odd, GcW[a] bit l} kappa_a
    xk_mz(m, S, COL, (u64)R * CW);
    for (u32 w = 0; w < c->WA; ++w)
        for (u64 y = lo[w]; y; y &= y - 1) {
            const u32 a = w * 64 + xk_ctz(y);
            XK_BUF const u64* k = c->KcW + (u64)a * CW;
            XK_BUF const u64* t = c->GcW + (u64)a * CW;
            for (u32 u = 0; u < CW; ++u)
                for (u64 tt = t[u]; tt; tt &= tt - 1) xk_mxt(m, S, COL + (u64)(u * 64 + xk_ctz(tt)) * CW, k, CW);
        }
    // the kernel of the columns, l = R - 1 .. 0 (reduction: pivot = lowest bit): Kt, depj
    xk_mz(m, S, KT, (u64)R * CW);
    u32 npv = 0, nker = 0;
    for (u32 j = R; j-- > 0;) {
        const u64 V = COL + (u64)j * CW;
        if (xk_mlow(m, S, V, CW) == CW) {   // zero column: kernel vector e_j
            m[(KT + (u64)j * CW + (nker >> 6)) * S] |= xk_bit(nker & 63);
            m[(DEPJ + nker) * S] = j;
            ++nker;
            continue;
        }
        xk_mz(m, S, TC, CW);
        m[(TC + (j >> 6)) * S] = xk_bit(j & 63);
        for (u32 q = 0; q < npv; ++q)
            if (m[(V + m[(PW + q) * S]) * S] & m[(PVM + q) * S]) {
                xk_mx(m, S, V, COL + m[(PCOL + q) * S] * CW, CW);
                xk_mx(m, S, TC, PVC + (u64)q * CW, CW);
            }
        const u32 lw = xk_mlow(m, S, V, CW);
        if (lw < CW) {
            xk_mc(m, S, PVC + (u64)npv * CW, TC, CW);
            const u64 x = m[(V + lw) * S];
            m[(PW + npv) * S] = lw;
            m[(PVM + npv) * S] = x & ((u64)0 - x);
            m[(PCOL + npv) * S] = j;
            ++npv;
        } else {
            for (u32 u = 0; u < CW; ++u)
                for (u64 x = m[(TC + u) * S]; x; x &= x - 1)
                    m[(KT + (u64)(u * 64 + xk_ctz(x)) * CW + (nker >> 6)) * S] |= xk_bit(nker & 63);
            m[(DEPJ + nker) * S] = j;
            ++nker;
        }
    }
    // the W phase on the kernel basis -> omega, already reduced modulo the coins
    if (nker) {
        xk_mz(m, S, PLO, 3 * (u64)CW);   // PLO, PHI, LIN
        xk_mz(m, S, BQ, (u64)nker * CW);
        for (u32 w = 0; w < c->WA; ++w)
            for (u64 y = lo[w]; y; y &= y - 1) {
                const u32 a = w * 64 + xk_ctz(y);
                xk_mz(m, S, CQ, CW);
                XK_BUF const u64* t = c->GcW + (u64)a * CW;
                for (u32 u = 0; u < CW; ++u)
                    for (u64 tt = t[u]; tt; tt &= tt - 1) xk_mx(m, S, CQ, KT + (u64)(u * 64 + xk_ctz(tt)) * CW, CW);
                const u64 bm = c->bb[a] ? ~(u64)0 : 0;
                for (u32 q = 0; q < CW; ++q) {
                    const u64 cq = m[(CQ + q) * S];
                    m[(PHI + q) * S] ^= m[(PLO + q) * S] & cq;
                    m[(PLO + q) * S] ^= cq;
                    m[(LIN + q) * S] ^= bm & cq;
                }
                for (u32 u = 0; u < CW; ++u)   // B(q, q') = sum_a c_a(q) c_a(q')
                    for (u64 x = m[(CQ + u) * S]; x; x &= x - 1)
                        xk_mx(m, S, BQ + (u64)(u * 64 + xk_ctz(x)) * CW, CQ, CW);
            }
        u64 bad = 0;
        for (u64 i = 0; i < (u64)nker * CW; ++i) bad |= m[(BQ + i) * S];
        for (u32 q = 0; q < CW; ++q) bad |= m[(PLO + q) * S];
        if (bad) return XK_CDIM_INTERNAL;   // odd phase on W / W phase not linear
        for (u32 u = 0; u < CW; ++u)
            for (u64 x = m[(PHI + u) * S] ^ m[(LIN + u) * S]; x; x &= x - 1) {
                const u32 j = (u32)m[(DEPJ + u * 64 + xk_ctz(x)) * S];
                m[(ZG + (j >> 6)) * S] ^= xk_bit(j & 63);
            }
    }
    // coins: t uniform in GF(2)^r picks XOR_{a odd, parity(t & kappa_a)} GcW[a]
    if (npv) {
        for (u32 k = 0; k < CW; ++k) m[(TT + k) * S] = xk_coin_word(c, shot, k, c->gauge);
        for (u32 w = 0; w < c->WA; ++w)
            for (u64 y = lo[w]; y; y &= y - 1) {
                const u32 a = w * 64 + xk_ctz(y);
                XK_BUF const u64* k = c->KcW + (u64)a * CW;
                u64 x = 0;
                for (u32 q = 0; q < CW; ++q) x ^= m[(TT + q) * S] & k[q];
                if (xk_parity(x)) xk_mxt(m, S, ZG, c->GcW + (u64)a * CW, CW);
            }
    }
    xk_xor_gamma(c, r, m, S, ZG);
    // status C (post_multi's rule in beta coordinates): the dependencies among KtL[j] = XOR_{l in GbW[j]} Kt[l],
    // j in [rho_D, rho)
    if (!c->rhoL || !npv) return 0;
    const u64 LV = COL, LC = BQ, KERN = PVC, LPW = PW, LPM = PVM, KP = PCOL, ORD = DEPJ;
    u32 nl = 0, nk = 0;
    for (u32 j = rD; j < R; ++j) {
        xk_mz(m, S, TV, CW);
        XK_BUF const u64* gb = c->GbW + (u64)j * CW;
        for (u32 u = 0; u < CW; ++u)
            for (u64 x = gb[u]; x; x &= x - 1) xk_mx(m, S, TV, KT + (u64)(u * 64 + xk_ctz(x)) * CW, CW);
        xk_mz(m, S, TC, CW);
        m[(TC + (j >> 6)) * S] = xk_bit(j & 63);
        for (u32 q = 0; q < nl; ++q)
            if (m[(TV + m[(LPW + q) * S]) * S] & m[(LPM + q) * S]) {
                xk_mx(m, S, TV, LV + (u64)q * CW, CW);
                xk_mx(m, S, TC, LC + (u64)q * CW, CW);
            }
        const u32 lw = xk_mlow(m, S, TV, CW);
        if (lw < CW) {
            xk_mc(m, S, LV + (u64)nl * CW, TV, CW);
            xk_mc(m, S, LC + (u64)nl * CW, TC, CW);
            const u64 x = m[(TV + lw) * S];
            m[(LPW + nl) * S] = lw;
            m[(LPM + nl) * S] = x & ((u64)0 - x);
            ++nl;
        } else {
            xk_mc(m, S, KERN + (u64)nk * CW, TC, CW);
            ++nk;
        }
    }
    // XK_MODE_NOSTATUS (sample without the status): dim C = nk (the kernel rows are independent: row i holds bit
    // j_i, which no earlier row holds), no key
    if (c->mode & XK_MODE_NOSTATUS) return nk > c->dimc ? XK_CDIM_CAPACITY : nk;
    // the key: the fully reduced echelon of the kernel rows (pivot = highest bit), in place, then descending pivots
    u32 n = 0;
    for (u32 i = 0; i < nk; ++i) {
        xk_mc(m, S, TV, KERN + (u64)i * CW, CW);
        for (u32 k = 0; k < n; ++k)
            if (xk_mbit(m, S, TV, (u32)m[(KP + k) * S])) xk_mx(m, S, TV, KERN + (u64)k * CW, CW);
        const u32 p = xk_mhigh(m, S, TV, CW);
        if (p == 0xFFFFFFFFu) continue;
        for (u32 k = 0; k < n; ++k)
            if (xk_mbit(m, S, KERN + (u64)k * CW, p)) xk_mx(m, S, KERN + (u64)k * CW, TV, CW);
        xk_mc(m, S, KERN + (u64)n * CW, TV, CW);
        m[(KP + n) * S] = p;
        ++n;
    }
    if (n > c->dimc) return XK_CDIM_CAPACITY;   // (dimc >= rhoL >= n: unreachable)
    for (u32 i = 0; i < n; ++i) {   // insertion sort of the row order by pivot, descending
        const u64 v = i;
        u32 j = i;
        for (; j > 0 && m[(KP + m[(ORD + j - 1) * S]) * S] < m[(KP + v) * S]; --j)
            m[(ORD + j) * S] = m[(ORD + j - 1) * S];
        m[(ORD + j) * S] = v;
    }
    for (u32 i = 0; i < n; ++i) {
        const u64 row = KERN + m[(ORD + i) * S] * CW;
        for (u32 q = 0; q < CW; ++q) key[(u64)i * CW + q] = m[(row + q) * S];
    }
    return n;
}

// xk_zero_passes in cw words (law_in_c, kind 3): the observable row's beta_L coordinates s (bits rho - rhoL + i, in
// the scratch vector TV of the shot), the rest 0, and s in the span of C's key (key: dim vectors of cw words, fully
// reduced, pivot = highest bit)
XK_FN bool xk_zero_passes_multi(XK_LOC const XkCtx* c, XK_BUF const u64* row, XK_BUF const u64* key, u32 dim,
                                u32 slot) {
    const u32 WO = c->WO, CW = c->cw;
    XK_BUF const u64* v = row + c->WD;
    u64 nz = 0;
    if (dim == 0) {
        for (u32 w = 0; w < WO; ++w) nz |= v[w];
        return nz == 0;
    }
    const u64 S = c->nshots, TV = CW;
    XK_BUF u64* m = c->mws + slot;
    const u32 rD = c->rho - c->rhoL, rhoL = c->rhoL;
    xk_mz(m, S, TV, CW);
    for (u32 i = 0; i < rhoL; ++i) {
        const u32 p = c->pL[i], pw = p >> 6, pb = p & 63;
        u64 bit = (v[pw] >> pb) & 1;
        for (u32 j = 0; j < i; ++j)
            if (xk_mbit(m, S, TV, rD + j)) bit ^= (c->bLo[(u64)j * WO + pw] >> pb) & 1;
        if (bit) m[(TV + ((rD + i) >> 6)) * S] |= xk_bit((rD + i) & 63);
    }
    for (u32 w = 0; w < WO; ++w) {
        u64 x = v[w];
        for (u32 i = 0; i < rhoL; ++i)
            if (xk_mbit(m, S, TV, rD + i)) x ^= c->bLo[(u64)i * WO + w];
        nz |= x;
    }
    if (nz) return false;
    for (u32 k = 0; k < dim; ++k) {
        XK_BUF const u64* kv = key + (u64)k * CW;
        u32 p = 0;
        for (u32 q = CW; q-- > 0;)
            if (kv[q]) {
                p = q * 64 + xk_highest(kv[q]);
                break;
            }
        if (xk_mbit(m, S, TV, p)) xk_mxt(m, S, TV, kv, CW);
    }
    return xk_mlow(m, S, TV, CW) == CW;
}

// ---------------------------------------------------------------------------------------- structured, multi-word
// xk_law_struct_mw: xk_law_struct (above: the same derivation, notation, steps and outputs) on a multi-word file
// (xtim.gpu.structure: detected, certified, axes renumbered; dims[XK_DIM_ST_W] = st_w slot words, ceil(rho / 64)).
// Its vectors are over the beta coordinates (beta_D, then beta_L), as the one-word law's; xk_law_multi's are over the
// basis gamma. A slot vector (XkSv) is XK_CAP_ST_W words, bit f = coordinate f (a leaf's slot; the bits below rD stay
// 0); a lifted vector (XkLv) is a core word c (bit k < rD: beta_D coordinate k; bit XK_SMW_TAU = XK_CAP_ST_NC - 1: the
// auxiliary coordinate tau, so tau has an index of its own instead of bit rho) and a slot vector s. E's pivots are
// indices Ep[e] (a slot, or a core bit) with two bit masks over the entries saying which kind (kl: a slot, kc: a core
// bit; neither: no entry). It runs in K2b only (xk_finish_shot): inlined into K2, Metal's compiler gave wrong
// results (a tagged pivot index compared against its tag, scalars updated through pointers, struct-valued selects
// and a guarded store each read back wrong there; the forms below avoid all four, but K2 still disagreed on a few
// shots), while K2b and the C++ reference agree byte for byte. Every loop over a vector's words is bounded by
// XK_CAP_ST_W and indexed by its counter; a word picked at run time (a type record's, a pivot's) is read or written
// through a select over the words (the note above: no store guarded by a run-time slot test). A type record covers
// one slot word (structure.plan cuts the runs at multiples of 64). The coin words are xk_coin_word's first st_w (the
// gauge bits are the core gauge bits and the leaves' slots, all below 64 st_w), so tau(t) is xk_law_multi's coin
// vector for the same coins, in beta coordinates; the correction z is XORed from the beta vectors (beta, at
// XK_DIM_OFF_BETA: structure.beta_rows). C's key: cw words per vector, as xk_law_multi's (beta coordinates, the same
// canonical echelon). The zero decoder: the observable row's coordinates s from stab (st_w words per observable bit),
// then membership in W' as xk_law_struct's. No scratch in mws: all of it in thread-local arrays (sizes in the header).
#if defined(XK_CAP_ST_W)
#define XK_SW XK_CAP_ST_W
#define XK_SMW_TAU (XK_CAP_ST_NC - 1)

struct XkSv {
    u64 w[XK_SW];
};
struct XkLv {
    u64 c;
    XkSv s;
};

XK_FN u64 xk_m(u64 b) { return (u64)0 - b; }   // b in {0, 1}: all ones or 0

XK_FN XkSv xk_sv0() {
    XkSv v;
    for (u32 i = 0; i < XK_SW; ++i) v.w[i] = 0;
    return v;
}
XK_FN XkSv xk_sv_and(XkSv a, XkSv b) {
    for (u32 i = 0; i < XK_SW; ++i) a.w[i] &= b.w[i];
    return a;
}
XK_FN XkSv xk_sv_andn(XkSv a, XkSv b) {   // a & ~b
    for (u32 i = 0; i < XK_SW; ++i) a.w[i] &= ~b.w[i];
    return a;
}
XK_FN XkSv xk_sv_or(XkSv a, XkSv b) {
    for (u32 i = 0; i < XK_SW; ++i) a.w[i] |= b.w[i];
    return a;
}
XK_FN XkSv xk_sv_xor(XkSv a, XkSv b) {
    for (u32 i = 0; i < XK_SW; ++i) a.w[i] ^= b.w[i];
    return a;
}
XK_FN XkSv xk_sv_msk(XkSv a, u64 m) {   // m: all ones or 0
    for (u32 i = 0; i < XK_SW; ++i) a.w[i] &= m;
    return a;
}
XK_FN bool xk_sv_any(XkSv a) {
    u64 x = 0;
    for (u32 i = 0; i < XK_SW; ++i) x |= a.w[i];
    return x != 0;
}
XK_FN u64 xk_sv_par(XkSv a) {
    u64 x = 0;
    for (u32 i = 0; i < XK_SW; ++i) x ^= a.w[i];
    return xk_parity(x);
}
XK_FN u32 xk_sv_pop(XkSv a) {
    u32 n = 0;
    for (u32 i = 0; i < XK_SW; ++i) n += xk_popcount(a.w[i]);
    return n;
}
XK_FN XkSv xk_sv_at(u32 q, u64 x) {   // x as word q (q past the words: 0)
    XkSv v;
    for (u32 i = 0; i < XK_SW; ++i) v.w[i] = i == q ? x : 0;
    return v;
}
XK_FN u64 xk_sv_get(XkSv a, u32 q) {   // word q (past the words: 0)
    u64 x = 0;
    for (u32 i = 0; i < XK_SW; ++i) x |= i == q ? a.w[i] : 0;
    return x;
}
// a pivot's bit: 0 unless a slot
XK_FN u64 xk_sv_bit(XkSv a, u32 f) { return (xk_sv_get(a, f >> 6) >> (f & 63)) & 1; }
XK_FN XkSv xk_sv_one(u32 f) { return xk_sv_at(f >> 6, xk_bit(f & 63)); }   // e_f (a core pivot or none: 0)
XK_FN XkSv xk_sv_low(XkSv a) {   // the lowest set bit
    u64 found = 0;
    for (u32 i = 0; i < XK_SW; ++i) {
        const u64 x = a.w[i];
        a.w[i] = found ? 0 : x & ((u64)0 - x);
        found |= x;
    }
    return a;
}
XK_FN u32 xk_sv_high(XkSv a) {   // the index of the highest set bit (a != 0)
    u32 h = 0;
    for (u32 i = 0; i < XK_SW; ++i) h = a.w[i] ? 64 * i + xk_highest(a.w[i] | 1) : h;
    return h;
}
XK_FN XkSv xk_sv_sel(bool b, XkSv a, XkSv o) {   // b ? a : o, word by word through masks (no select of a struct)
    const u64 m = xk_m((u64)b);
    for (u32 i = 0; i < XK_SW; ++i) a.w[i] = (a.w[i] & m) | (o.w[i] & ~m);
    return a;
}
XK_FN XkSv xk_sv_load(XK_BUF const u64* p, u32 sw) {   // sw table words, then zeros
    XkSv v;
    for (u32 i = 0; i < XK_SW; ++i) v.w[i] = i < sw ? p[i] : 0;
    return v;
}

XK_FN XkLv xk_lv0() {
    XkLv v;
    v.c = 0;
    v.s = xk_sv0();
    return v;
}
XK_FN XkLv xk_lv_xor(XkLv a, XkLv b) {
    a.c ^= b.c;
    a.s = xk_sv_xor(a.s, b.s);
    return a;
}
XK_FN XkLv xk_lv_msk(XkLv a, u64 m) {
    a.c &= m;
    a.s = xk_sv_msk(a.s, m);
    return a;
}
XK_FN bool xk_lv_any(XkLv a) { return a.c != 0 || xk_sv_any(a.s); }
XK_FN XkLv xk_lv_sel(bool b, XkLv a, XkLv o) {
    a.c = b ? a.c : o.c;
    a.s = xk_sv_sel(b, a.s, o.s);
    return a;
}
// a's bit at E's entry e's pivot P (kl, kc: the kinds' masks; no entry: 0)
XK_FN u64 xk_smw_pbit(XkLv a, u32 P, u32 e, u32 kl, u32 kc) {
    return (xk_sv_bit(a.s, P) & ((kl >> e) & 1)) | ((a.c >> (P & 63)) & ((kc >> e) & 1));
}

// a type record's slot word ((rec >> 24 & 0xff) >> 6) and its odd axes there: plane bits [off, off + n) moved to slots
// [s0, s0 + n) (n <= 64, within one slot word)
XK_FN u32 xk_smw_q(u64 rec) { return ((u32)(rec >> 24) & 0xff) >> 6; }
XK_FN u64 xk_smw_word(XK_LOC const u64* plane, u32 WA, u64 rec) {
    const u32 off = (u32)(rec & 0xffff), n = (u32)(rec >> 16) & 0xff, s0 = (u32)(rec >> 24) & 0xff;
    const u32 w = off >> 6, b = off & 63;
    u64 x = plane[w] >> b;
    if (b && w + 1 < WA) x |= plane[w + 1] << (64 - b);
    return (n >= 64 ? x : x & (xk_bit(n) - 1)) << (s0 & 63);
}

// a non-leaf axis's T (table words: its beta_D part, then its slot part)
XK_FN XkLv xk_smw_nl(XK_BUF const u64* p, u32 sw) {
    XkLv v;
    v.c = p[0];
    v.s = xk_sv_load(p + 1, sw);
    return v;
}

// w reduced by S1's rows f_i + c_i (C: the core parts, bit-sliced)
XK_FN XkLv xk_smw_red(XkLv w, XkSv S1, XK_LOC const XkSv* C) {
    const XkSv x = xk_sv_and(w.s, S1);
    w.s = xk_sv_andn(w.s, S1);
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) w.c ^= xk_bit(k) & xk_m(xk_sv_par(xk_sv_and(C[k], x)));
    return w;
}

// L(T) (xk_st_L): T reduced by the lifted echelon (S1's rows through Cp, then E), on the free columns fm
XK_FN XkLv xk_smw_L(XkLv T, XkSv S1, XK_LOC const XkSv* Cp, XK_LOC const XkLv* Ev, XK_LOC const u32* Ep, u32 kl,
                    u32 kc, XkLv fm) {
    XkLv q = xk_smw_red(T, S1, Cp);
    for (u32 e = 0; e < XK_ST_E; ++e) q = xk_lv_xor(q, xk_lv_msk(Ev[e], xk_m(xk_smw_pbit(T, Ep[e], e, kl, kc))));
    q.c &= fm.c;
    q.s = xk_sv_and(q.s, fm.s);
    return q;
}

// the pivot columns (lowest bits) of an echelon of the rows C[k] & S (xk_st_pivots)
XK_FN XkSv xk_smw_pivots(XK_LOC const XkSv* C, XkSv S) {
    XkSv X[XK_CAP_ST_NC], P[XK_CAP_ST_NC];
    XkSv pv = xk_sv0();
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) X[k] = P[k] = xk_sv0();
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
        XkSv v = xk_sv_and(C[k], S);
        for (u32 q = 0; q < XK_CAP_ST_NC; ++q) v = xk_sv_xor(v, xk_sv_msk(X[q], xk_m(xk_sv_any(xk_sv_and(v, P[q])))));
        X[k] = v;
        P[k] = xk_sv_low(v);
        pv = xk_sv_or(pv, P[k]);
    }
    return pv;
}

// the column of the bit-sliced core parts C at the slots p as a core word
XK_FN u64 xk_smw_col(XK_LOC const XkSv* C, XkSv p) {
    u64 v = 0;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) v |= xk_bit(k) & xk_m(xk_sv_any(xk_sv_and(C[k], p)));
    return v;
}

// v (a core word: XK_CAP_ST_NC <= 8 bits) into the core echelon G, packed in one word (byte k: the entry whose highest
// bit is k, 0 for none), returned (G changes iff v is independent). No thread-local array: on Metal (Apple M3 Max) the
// echelon as an array G[k] written by selects read back wrong in K2 (the w-atoms' images inserted after the S1
// leaves'), while K2b and the C++ reference were right.
XK_FN u64 xk_smw_core_insert(u64 v, u64 G) {
    for (u32 i = 0; i < XK_CAP_ST_NC; ++i) {
        const u32 k = XK_CAP_ST_NC - 1 - i;
        const u64 g = (G >> (8 * k)) & 0xff;
        const bool hit = ((v >> k) & 1) != 0, fresh = hit && g == 0;
        G |= fresh ? v << (8 * k) : 0;
        v = fresh ? 0 : (hit ? v ^ g : v);
    }
    return G;
}

// w reduced into E (fully reduced, pivot = highest slot bit, else highest core bit). E's scalars travel as one word,
// returned: XK_SMW_N (the entries nE), XK_SMW_KL (kl), XK_SMW_KC (kc); nE grows when w is independent. Branch-free (a
// dependent w reduces to 0: no entry chosen, no row changed). (Not through pointers to the caller's scalars: on Metal
// kl written so read back wrong in K2.)
#define XK_SMW_N(x) ((u32)(x) & 0xff)
#define XK_SMW_KL(x) ((u32)((x) >> 8) & 0xffffff)
#define XK_SMW_KC(x) ((u32)((x) >> 32) & 0xffffff)
XK_FN u64 xk_smw_insert(XkLv w, XK_LOC XkLv* Ev, XK_LOC u32* Ep, u64 es) {
    const u32 l0 = XK_SMW_KL(es), c0 = XK_SMW_KC(es);
    for (u32 e = 0; e < XK_ST_E; ++e) w = xk_lv_xor(w, xk_lv_msk(Ev[e], xk_m(xk_smw_pbit(w, Ep[e], e, l0, c0))));
    const bool nz = xk_lv_any(w), sl = xk_sv_any(w.s);
    const u32 pb = sl ? xk_sv_high(w.s) : xk_highest(w.c | 1);
    const u32 n = XK_SMW_N(es);
    for (u32 e = 0; e < XK_ST_E; ++e) {   // (selects, not a guarded store)
        const u64 h = sl ? xk_sv_bit(Ev[e].s, pb) : ((Ev[e].c >> pb) & 1);
        const XkLv v = xk_lv_xor(Ev[e], xk_lv_msk(w, xk_m(h)));
        Ev[e] = xk_lv_sel(nz && e == n, w, v);
        Ep[e] = nz && e == n ? pb : Ep[e];
    }
    const u64 kl = l0 | (nz && sl ? 1u << n : 0u), kc = c0 | (nz && !sl ? 1u << n : 0u);
    return (u64)(n + (nz ? 1u : 0u)) | kl << 8 | kc << 32;
}

// row r ^= sum_j z_j beta_j (z in beta coordinates: its core word's beta_D bits and its slot bits)
XK_FN void xk_smw_xor_beta(XK_LOC const XkCtx* c, XK_BUF u64* r, XkLv z) {
    const u32 WF = c->WF, rD = c->rho - c->rhoL;
    for (u64 y = z.c & (xk_bit(rD) - 1); y; y &= y - 1) {
        XK_BUF const u64* bv = c->beta + (u64)xk_ctz(y) * WF;
        for (u32 w = 0; w < WF; ++w) r[w] ^= bv[w];
    }
    for (u32 q = 0; q < XK_SW; ++q)
        for (u64 y = z.s.w[q]; y; y &= y - 1) {
            XK_BUF const u64* bv = c->beta + (u64)(64 * q + xk_ctz(y)) * WF;
            for (u32 w = 0; w < WF; ++w) r[w] ^= bv[w];
        }
}

// xk_obs_coords with st_w-word coordinates: s, and whether the rest is 0
XK_FN XkSv xk_smw_obs_s(XK_LOC const XkCtx* c, XK_BUF const u64* v) {
    const u32 WO = c->WO, sw = c->st_w;
    XkSv x = xk_sv0();
    for (u32 w = 0; w < WO; ++w)
        for (u64 y = v[w]; y; y &= y - 1) x = xk_sv_xor(x, xk_sv_load(c->stab + (u64)(w * 64 + xk_ctz(y)) * sw, sw));
    return x;
}
XK_FN bool xk_smw_obs_rest0(XK_LOC const XkCtx* c, XK_BUF const u64* v) {
    const u32 WO = c->WO;
    if (!c->rtab_on) return true;
    u64 nz = 0;
    for (u32 k = 0; k < WO; ++k) {
        u64 r = 0;
        for (u32 w = 0; w < WO; ++w)
            for (u64 y = v[w]; y; y &= y - 1) r ^= c->rtab[(u64)(w * 64 + xk_ctz(y)) * WO + k];
        nz |= r;
    }
    return nz == 0;
}

XK_FN u32 xk_law_struct_mw(XK_LOC XkCtx* c, XK_BUF u64* r, XK_LOC const u64* lo, XK_LOC const u64* hi, u32 shot,
                           XK_BUF u64* ko, XK_LOC u32* zp) {
    XK_BUF const u64* st = c->sct;
    const u32 sw = c->st_w, WA = c->WA, nt = c->st_nt, nn = c->st_nn, rD = c->rho - c->rhoL;
    XK_BUF const u64* rec = st + XK_ST_HDR + 2 * (u64)sw;   // the type records (after Lall and u), two words each
    XK_BUF const u64* nla = rec + 2 * (u64)nt;              // the non-leaf axes, ns words each
    const u32 ns = sw + 2;
    const u64 taub = xk_bit(XK_SMW_TAU), cmv = (xk_bit(rD) - 1) | taub, nlm = xk_bit(nn) - 1;
    const XkSv Lall = xk_sv_load(st + XK_ST_HDR, sw), uslot = xk_sv_load(st + XK_ST_HDR + sw, sw);
    // 0. zhi = XOR of T_a over the 2-bits of eta (hi). (Records and non-leaf axes past nt, nn: a zero record (no
    // axis), a masked axis. An axis past nn reads the table's header instead of nla[j]: with nn = 0 (XK_CAP_ST_N is
    // 1) nla is the table's end, and nothing there is ours to read.)
    XkLv zhi = xk_lv0();
    u64 odd = 0;
    for (u32 w = 0; w < WA; ++w) odd |= lo[w];
    const u64 hnl = hi[0] & nlm, nlo = lo[0] & nlm;
    for (u32 j = 0; j < XK_CAP_ST_T; ++j) {
        const u64 rc = j < nt ? rec[2 * j] : 0, tD = j < nt ? rec[2 * j + 1] : 0;
        const u64 x = xk_smw_word(hi, WA, rc);
        zhi.c ^= tD & xk_m(xk_parity(x));
        zhi.s = xk_sv_xor(zhi.s, xk_sv_at(xk_smw_q(rc), x & xk_m((rc >> 33) & 1)));
    }
    for (u32 j = 0; j < XK_CAP_ST_N; ++j)
        zhi = xk_lv_xor(zhi, xk_lv_msk(xk_smw_nl((j < nn ? nla + (u64)j * ns : st), sw), xk_m((hnl >> j) & 1)));
    if (!odd) {
        xk_smw_xor_beta(c, r, zhi);
        return 0;
    }
    // 1. the type words; the leaf rows bit-sliced (d, C[k]), the core rows R, T_dense, T_bb
    u64 Wt[XK_CAP_ST_T];
    XkSv C[XK_CAP_ST_NC];
    XkLv R[XK_CAP_ST_NG];
    XkSv d = xk_sv0();
    XkLv Td = xk_lv0(), Tbb = xk_lv0();
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) C[k] = xk_sv0();
    for (u32 i = 0; i < XK_CAP_ST_NG; ++i) R[i] = xk_lv0();
    for (u32 j = 0; j < XK_CAP_ST_T; ++j) {
        const u64 rc = j < nt ? rec[2 * j] : 0, tD = j < nt ? rec[2 * j + 1] : 0;
        const u64 x = xk_smw_word(lo, WA, rc);
        const u32 q = xk_smw_q(rc);
        Wt[j] = x;
        const u64 mx = xk_m((rc >> 32) & 1), my = xk_m((rc >> 33) & 1);
        d = xk_sv_xor(d, xk_sv_at(q, x & mx & my));
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) C[k] = xk_sv_xor(C[k], xk_sv_at(q, x & mx & xk_m((tD >> k) & 1)));
        XkLv T;   // the XOR of the record's odd axes' T
        T.c = tD & xk_m(xk_parity(x));
        T.s = xk_sv_at(q, x & my);
        const u32 kg = (u32)(rc >> 40) & 0xff;
        for (u32 i = 0; i < XK_CAP_ST_NG; ++i) R[i] = xk_lv_xor(R[i], xk_lv_msk(T, xk_m((kg >> i) & 1)));
        Tbb = xk_lv_xor(Tbb, xk_lv_msk(T, xk_m((rc >> 34) & 1)));
    }
    for (u32 j = 0; j < XK_CAP_ST_N; ++j) {
        XK_BUF const u64* p = (j < nn ? nla + (u64)j * ns : st);
        const u64 f = ((nlo >> j) & 1) ? p[sw + 1] : 0;   // (an even axis: no flag, no T)
        const XkLv T = xk_lv_msk(xk_smw_nl(p, sw), xk_m((nlo >> j) & 1));
        for (u32 i = 0; i < XK_CAP_ST_NG; ++i) R[i] = xk_lv_xor(R[i], xk_lv_msk(T, xk_m((f >> i) & 1)));
        Td = xk_lv_xor(Td, xk_lv_msk(T, xk_m((f >> 8) & 1)));
        Tbb = xk_lv_xor(Tbb, xk_lv_msk(T, xk_m((f >> 9) & 1)));
    }
    C[XK_SMW_TAU] = uslot;   // the lifted rows' tau slice: u (bits < rD of tD: C[XK_SMW_TAU] was 0)
    const XkSv S1 = d;
    XkSv Z = d;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) Z = xk_sv_or(Z, C[k]);
    const XkSv S0 = xk_sv_andn(Z, d);
    // 2. E: R_gamma, T_dense + tau and a basis of the S0 leaves' core parts, each reduced by S1's rows (a zero
    // vector inserts nothing: R[i] past ng, the columns past the pivots)
    XkLv Ev[XK_ST_E];
    u32 Ep[XK_ST_E];
    u64 es = 0;   // nE, kl, kc (xk_smw_insert)
    for (u32 e = 0; e < XK_ST_E; ++e) {
        Ev[e] = xk_lv0();
        Ep[e] = 0;
    }
    for (u32 i = 0; i < XK_CAP_ST_NG; ++i) es = xk_smw_insert(xk_smw_red(R[i], S1, C), Ev, Ep, es);
    {
        XkLv w = Td;
        w.c |= taub;
        es = xk_smw_insert(xk_smw_red(w, S1, C), Ev, Ep, es);
    }
    {
        XkSv pv = xk_smw_pivots(C, S0);
        for (u32 i = 0; i < XK_CAP_ST_NC; ++i) {
            const XkSv p = xk_sv_low(pv);
            pv = xk_sv_xor(pv, p);
            XkLv w = xk_lv0();
            w.c = xk_smw_col(C, p);
            es = xk_smw_insert(w, Ev, Ep, es);
        }
    }
    const u32 nE = XK_SMW_N(es), kl = XK_SMW_KL(es), kc = XK_SMW_KC(es);
    const u32 rank = xk_sv_pop(S1) + nE - 1;   // dim W' - 1

    // 3. Cp = C reduced modulo zD (E's core-pivot vectors); the images' rank in core space; dim C
    XkSv Cp[XK_CAP_ST_NC];
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) Cp[k] = C[k];
    XkSv wpiv = xk_sv0();   // the w-atoms' pivots
    for (u32 e = 0; e < XK_ST_E; ++e) {
        const u32 P = Ep[e];
        wpiv = xk_sv_or(wpiv, xk_sv_msk(xk_sv_one(P), xk_m((kl >> e) & 1)));
        const u64 core = xk_m((kc >> e) & 1);
        XkSv cpv = xk_sv0();   // C's slice at the pivot coordinate
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) cpv = xk_sv_xor(cpv, xk_sv_msk(C[k], core & xk_m((u64)(k == (P & 63)))));
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) Cp[k] = xk_sv_xor(Cp[k], xk_sv_msk(cpv, xk_m((Ev[e].c >> k) & 1)));
    }

    u32 irank = 0;
    u64 G = 0;   // a core-space echelon (xk_smw_core_insert)
    {
        XkSv pv = xk_smw_pivots(Cp, S1);
        for (u32 i = 0; i < XK_CAP_ST_NC; ++i) {
            const XkSv p = xk_sv_low(pv);
            pv = xk_sv_xor(pv, p);
            const u64 G2 = xk_smw_core_insert(xk_smw_col(Cp, p), G);
            irank += G2 != G ? 1u : 0u;
            G = G2;
        }
    }

    for (u32 e = 0; e < XK_ST_E; ++e) {
        const u64 G2 = xk_smw_core_insert(Ev[e].c & xk_m((kl >> e) & 1), G);
        irank += G2 != G ? 1u : 0u;
        G = G2;
    }
    const u32 dim = xk_sv_pop(S1) + xk_sv_pop(wpiv) - irank;

    // 4. the coins: tau = M^T t = XOR_{gamma in t} R_gamma ^ (d & tL) ^ sum_k parity(C[k] & tL) e_k ^ <t, u> T_dense
    // (rank 0: tau = 0, the coins not read)
    XkSv t;
    for (u32 q = 0; q < XK_SW; ++q) t.w[q] = q < sw && rank ? xk_coin_word(c, shot, q, c->gauge) : 0;
    const XkSv tL = xk_sv_and(t, Lall);
    XkLv tau = xk_lv0();
    for (u32 i = 0; i < XK_CAP_ST_NG; ++i)   // (R[i] past ng is 0; sct's header words 8 + i are there)
        tau = xk_lv_xor(tau, xk_lv_msk(R[i], xk_m(xk_sv_bit(t, (u32)st[8 + i]))));
    tau.s = xk_sv_xor(tau.s, xk_sv_and(d, tL));
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) tau.c ^= xk_bit(k) & xk_m((u64)(k < rD) & xk_sv_par(xk_sv_and(C[k], tL)));
    tau = xk_lv_xor(tau, xk_lv_msk(Td, xk_m(xk_sv_par(xk_sv_and(t, uslot)))));
    // 5. omega: phi_j = bit 1 of n_j = #{odd a : L(T_a)_j = 1} on the free columns, plus the linear term L(T_bb)
    XkLv fm;
    {
        u64 pivc = 0;
        XkSv pivs = S1;
        for (u32 e = 0; e < XK_ST_E; ++e) {
            pivs = xk_sv_or(pivs, xk_sv_msk(xk_sv_one(Ep[e]), xk_m((kl >> e) & 1)));
            pivc |= xk_bit(Ep[e] & 63) & xk_m((kc >> e) & 1);
        }
        fm.c = cmv & ~pivc;
        fm.s = xk_sv_andn(Lall, pivs);
    }
    const XkSv JL = fm.s;
    XkSv Q[XK_CAP_ST_NC];   // free core column k: the leaves whose L(f_i) has bit k (S1: Cp; w-pivots: E)
    u32 n[XK_CAP_ST_NC];
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
        XkSv q = xk_sv_and(Cp[k], S1);
        for (u32 e = 0; e < XK_ST_E; ++e)
            q = xk_sv_or(q, xk_sv_msk(xk_sv_one(Ep[e]), xk_m((kl >> e) & (Ev[e].c >> k) & 1)));
        Q[k] = q;
        n[k] = 0;
    }
    XkSv c0 = xk_sv0(), c1 = xk_sv0();   // the odd y-axes per leaf, mod 4, bit-sliced over the slots
    for (u32 j = 0; j < XK_CAP_ST_T; ++j) {   // (past nt: Wt[j] = 0, nothing counted)
        const u64 rc = j < nt ? rec[2 * j] : 0, t0 = j < nt ? rec[2 * j + 1] : 0, x = Wt[j];
        const u32 q = xk_smw_q(rc);
        u64 lt = t0;   // L(t_tau): pure core (E's core-pivot vectors have no slot bit), on the free columns
        for (u32 e = 0; e < XK_ST_E; ++e)
            lt ^= Ev[e].c & xk_m((kc >> e) & (t0 >> (Ep[e] & 63)) & 1);
        lt &= fm.c;
        const u32 pw = xk_popcount(x);
        const bool y = ((rc >> 33) & 1) != 0;
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
            const bool l = ((lt >> k) & 1) != 0;
            const u32 qq = xk_popcount(x & xk_sv_get(Q[k], q));
            const u32 add = y ? (l ? pw - qq : qq) : (l ? pw : 0u);
            n[k] += ((fm.c >> k) & 1) ? add : 0u;
        }
        const u64 xy = x & xk_m((u64)y);
        for (u32 i = 0; i < XK_SW; ++i) {
            const u64 a = i == q ? xy : 0;
            c1.w[i] ^= c0.w[i] & a;
            c0.w[i] ^= a;
        }
    }
    for (u32 e = 0; e < XK_ST_E; ++e) {   // a w-pivot leaf p's m_p odd y-axes each carry E[p] on the free leaf columns
        const XkSv p = xk_sv_msk(xk_sv_one(Ep[e]), xk_m((kl >> e) & 1));
        const XkSv x = xk_sv_and(Ev[e].s, JL);
        const u64 a0 = xk_m((u64)xk_sv_any(xk_sv_and(c0, p)));
        c1 = xk_sv_xor(c1, xk_sv_msk(xk_sv_and(c0, x), a0));
        c0 = xk_sv_xor(c0, xk_sv_msk(x, a0));
        c1 = xk_sv_xor(c1, xk_sv_msk(x, xk_m((u64)xk_sv_any(xk_sv_and(c1, p)))));
    }
    XkSv lc0 = xk_sv_and(c0, JL), lc1 = xk_sv_and(c1, JL);
    for (u32 j = 0; j < XK_CAP_ST_N; ++j) {   // the odd non-leaf axes, one by one (an even one: L(0) = 0)
        const XkLv T = xk_lv_msk(xk_smw_nl((j < nn ? nla + (u64)j * ns : st), sw), xk_m((nlo >> j) & 1));
        const XkLv qv = xk_smw_L(T, S1, Cp, Ev, Ep, kl, kc, fm);
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) n[k] += (u32)((qv.c >> k) & 1);
        const XkSv x = xk_sv_and(qv.s, JL);
        lc1 = xk_sv_xor(lc1, xk_sv_and(lc0, x));
        lc0 = xk_sv_xor(lc0, x);
    }
    XkLv wh;
    wh.s = lc1;
    wh.c = 0;
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) wh.c |= xk_bit(k) & fm.c & xk_m((n[k] >> 1) & 1);
    // the linear term (some axis has a bb bit)
    wh = xk_lv_xor(wh, xk_lv_msk(xk_smw_L(Tbb, S1, Cp, Ev, Ep, kl, kc, fm), xk_m((u64)(st[3] != 0))));
    XkLv omega = wh;
    omega.c &= ~taub;
    omega = xk_lv_xor(omega, xk_lv_msk(Td, xk_m((wh.c >> XK_SMW_TAU) & 1)));
    xk_smw_xor_beta(c, r, xk_lv_xor(xk_lv_xor(zhi, omega), tau));
    // 6. the zero decoder: v = 0 at dim C = 0; else v in C iff its coordinates s (rest 0) reduce to 0 in W' with no
    // core part left
    {
        XK_BUF const u64* v = r + c->WD;
        u64 nz = 0;
        for (u32 w = 0; w < c->WO; ++w) nz |= v[w];
        const XkSv s = xk_smw_obs_s(c, v);
        const XkSv x = xk_sv_and(s, S1);
        XkSv r2 = xk_sv_andn(s, S1);
        u64 img = 0;
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) img ^= xk_bit(k) & xk_m(xk_sv_par(xk_sv_and(Cp[k], x)));
        for (u32 e = 0; e < XK_ST_E; ++e) {
            const u64 m = xk_m(xk_sv_bit(r2, Ep[e]) & ((kl >> e) & 1));
            r2 = xk_sv_xor(r2, xk_sv_msk(Ev[e].s, m));
            img ^= Ev[e].c & m;
        }
        *zp = dim == 0 ? (nz == 0) : (xk_smw_obs_rest0(c, v) && !xk_sv_any(r2) && img == 0);
    }
    if (dim == 0 || (c->mode & XK_MODE_NOKEY)) return dim;
    // 7. C's key (xk_law_struct's step 7): the atoms in ascending pivot order, the independent ones found greedily
    // (rounds past the last: act = false, nothing changed)
    XkSv X[XK_CAP_ST_NC], BL[XK_CAP_ST_NC];
    u64 wr[XK_ST_E], BI[XK_CAP_ST_NC];
    for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
        X[k] = xk_sv_and(Cp[k], S1);
        BI[k] = 0;
        BL[k] = xk_sv0();
    }
    for (u32 e = 0; e < XK_ST_E; ++e) wr[e] = Ev[e].c & xk_m((kl >> e) & 1);
    XkSv IS = xk_sv0(), IW = xk_sv0();   // the independent S1 leaves, the independent w-atoms' pivots
    u32 na = 0;
    for (u32 round = 0; round <= XK_CAP_ST_NC; ++round) {
        XkSv notin = xk_sv0();
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) notin = xk_sv_or(notin, X[k]);
        const XkSv lp = xk_sv_low(notin);
        u32 wp = 0;
        bool wf = false;   // a w-atom found (wp its pivot)
        u64 wrow = 0, wimg = 0;
        XkSv wleaf = xk_sv0();
        for (u32 e = 0; e < XK_ST_E; ++e) {   // the lowest-pivot w-atom outside the span (wr: 0 unless a w-atom)
            const bool b = wr[e] != 0 && (!wf || Ep[e] < wp);
            wf = wf || b;
            wp = b ? Ep[e] : wp;
            wrow = b ? wr[e] : wrow;
            wimg = b ? Ev[e].c : wimg;
            wleaf = xk_sv_sel(b, Ev[e].s, wleaf);
        }
        const bool hl = xk_sv_any(lp);
        const bool act = hl || wf;
        const bool isS = hl && (!wf || xk_sv_high(lp) < wp);   // the S1 leaf lp (else the w-atom wp, if act)
        u64 lrow = 0, limg = 0;
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) {
            lrow |= xk_bit(k) & xk_m((u64)xk_sv_any(xk_sv_and(X[k], lp)));
            limg |= xk_bit(k) & xk_m((u64)xk_sv_any(xk_sv_and(Cp[k], lp)));
        }
        const u64 row = isS ? lrow : wrow;   // (0 when not act)
        const u64 img = isS ? limg : wimg;
        const XkSv leaf = xk_sv_sel(isS, lp, wleaf);
        IS = xk_sv_or(IS, xk_sv_msk(lp, xk_m((u64)isS)));
        IW = xk_sv_or(IW, xk_sv_msk(xk_sv_one(wp), xk_m((u64)(act && !isS))));
        for (u32 a = 0; a < XK_CAP_ST_NC; ++a) {
            BI[a] = act && a == na ? img : BI[a];
            BL[a] = xk_sv_sel(act && a == na, leaf, BL[a]);
        }
        na += act ? 1u : 0u;
        // reduce the remaining images by row (pivot: its highest bit, which no earlier row has)
        const u64 pb = row ? xk_bit(xk_highest(row | 1)) : 0;
        XkSv M = xk_sv0();
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) M = xk_sv_or(M, xk_sv_msk(X[k], xk_m((pb >> k) & 1)));
        for (u32 k = 0; k < XK_CAP_ST_NC; ++k) X[k] = xk_sv_xor(X[k], xk_sv_msk(M, xk_m((row >> k) & 1)));
        for (u32 e = 0; e < XK_ST_E; ++e) wr[e] ^= row & xk_m((u64)((wr[e] & pb) != 0));
    }
    // the independent images' fully reduced echelon with combinations (xk_law_struct's GR, GP, GC)
    u64 GR[XK_CAP_ST_NC], GP[XK_CAP_ST_NC];
    u32 GC[XK_CAP_ST_NC];
    for (u32 a = 0; a < XK_CAP_ST_NC; ++a) {
        GR[a] = GP[a] = 0;
        GC[a] = 0;
    }
    for (u32 a = 0; a < XK_CAP_ST_NC; ++a) {
        u64 v = BI[a];
        u32 cc = a < na ? 1u << a : 0u;
        for (u32 q = 0; q < XK_CAP_ST_NC; ++q) {
            const u64 m = xk_m((u64)((v & GP[q]) != 0));
            v ^= GR[q] & m;
            cc ^= GC[q] & (u32)m;
        }
        const u64 pb = v ? xk_bit(xk_highest(v | 1)) : 0;
        for (u32 q = 0; q < XK_CAP_ST_NC; ++q) {
            const u64 m = xk_m((u64)((GR[q] & pb) != 0));
            GR[q] ^= v & m;
            GC[q] ^= cc & (u32)m;
        }
        GR[a] = v;
        GP[a] = pb;
        GC[a] = cc;
    }
    // per independent atom a, the S1 leaves whose image's combination holds a (bit-sliced)
    XkSv Lam[XK_CAP_ST_NC];
    for (u32 a = 0; a < XK_CAP_ST_NC; ++a) {
        XkSv x = xk_sv0();
        for (u32 q = 0; q < XK_CAP_ST_NC; ++q)
            for (u32 k = 0; k < XK_CAP_ST_NC; ++k)
                x = xk_sv_xor(x, xk_sv_msk(Cp[k], xk_m((u64)((GC[q] >> a) & 1) & ((GP[q] >> k) & 1))));
        Lam[a] = x;
    }
    // the dependent atoms, descending: key vector = leaf part ^ the independent atoms' leaf parts of its combination
    XkSv dep = xk_sv_or(xk_sv_andn(S1, IS), xk_sv_andn(wpiv, IW));
    const u32 cw = c->cw;
    u32 nk = 0;
    for (u32 i = 0; i < 64 * XK_SW; ++i) {
        if (!xk_sv_any(dep)) break;
        const u32 pi = xk_sv_high(dep);
        const XkSv p = xk_sv_one(pi);
        dep = xk_sv_xor(dep, p);
        // an S1 leaf: e_p and its image's combination (Lam); a w-atom: E's vector and its image's combination
        XkSv vs = p;
        for (u32 a = 0; a < XK_CAP_ST_NC; ++a)
            vs = xk_sv_xor(vs, xk_sv_msk(BL[a], xk_m((u64)xk_sv_any(xk_sv_and(Lam[a], p)))));
        u64 img = 0;
        XkSv vw = xk_sv0();
        for (u32 e = 0; e < XK_ST_E; ++e) {
            const bool b = ((kl >> e) & 1) != 0 && Ep[e] == pi;
            img = b ? Ev[e].c : img;
            vw = xk_sv_sel(b, Ev[e].s, vw);
        }
        u32 lam = 0;
        for (u32 q = 0; q < XK_CAP_ST_NC; ++q) lam ^= GC[q] & (0u - (u32)((img & GP[q]) != 0));
        for (u32 a = 0; a < XK_CAP_ST_NC; ++a) vw = xk_sv_xor(vw, xk_sv_msk(BL[a], xk_m((u64)((lam >> a) & 1))));
        const XkSv v = xk_sv_sel(xk_sv_bit(S1, pi) != 0, vs, vw);
        if (nk < dim)
            for (u32 q = 0; q < XK_SW; ++q)
                if (q < sw) ko[(u64)nk * cw + q] = v.w[q];
        ++nk;
    }
    return nk == dim ? dim : XK_CDIM_INTERNAL;   // (the dependent atoms number dim C: a failure is a bug)
}
#endif
#endif

// ------------------------------------------------------------------------------------------------ K2: one shot
// Its extended lines (if any) sorted into file order, composed, and the fast-path law applied to the row; writes the
// row, dim C and the C key (zero-padded to max(rhoL, 1) words). Under XK_MODE_POSTSELECT it follows run_one_block's
// postselected loop: a shot with no extended line is final after the line XOR; one with an extended line whose row
// fails the span test is rejected early (XK_FLAG_EARLY: no coin read, no composition, nothing else written);
// otherwise the law is applied and the corrected row decides (XK_FLAG_SURVIVE when its masked detectors are 0).
// Without XK_MODE_POSTSELECT every shot survives. A survivor without an error code gets XK_FLAG_ZERO_PASS when the
// zero decoder passes it (xk_zero_passes).
//
// The one-word law comes from the single-line table when the composed lo equals a fired line's lo (xk_law_entry).
// Under XK_MODE_DEFER a shot that needs the one-word law computed (odd axes, no table entry) is only composed and
// queued (the multi-word law is not: on merlin_bh70 its queued shots ran 1.25 times slower, their scratch columns no
// longer neighbours):
// lawq[1 + k] = shot, k = lawq[0]++ (atomic; K1 empties the queue), and K2b (xk_resolve_law) finishes it,
// so the threads of a SIMD group that run the elimination all need it. A shot's outputs do not depend on the queue's
// order (its coin word is keyed by the shot).

// the law (table, or computed) of a composed shot past the span test, and its outputs: row, ckey, cdim, flag
XK_FN void xk_finish_shot(XK_LOC XkCtx* c, u32 shot, XK_BUF u64* row, XK_BUF u32* L, u32 ne, XK_LOC const u64* lo,
                          XK_LOC const u64* hi, u32 e, u32 slot, bool queued) {
    const bool ps = (c->mode & XK_MODE_POSTSELECT) != 0;
    const u32 kw = (c->rhoL ? c->rhoL : 1) * (c->cw ? c->cw : 1);
    XK_BUF u64* ko = c->ckey + (u64)shot * kw;
    // under XK_MODE_NOKEY (the estimate) the one-word law and the structured laws write no key: ko is neither cleared
    // nor written
    if (!(c->mode & XK_MODE_NOSTATUS) && ((c->cw && !c->st) || !(c->mode & XK_MODE_NOKEY)))
        for (u32 i = 0; i < kw; ++i) ko[i] = 0;
#if XK_CAP_CW > 1
#if defined(XK_CAP_ST_W)
    if (c->cw && c->st) {   // the structured multi-word law (the zero decoder's verdict too, where there are odd axes)
        // K2 queues every shot with a fired line (XK_MODE_DEFER, required by xk_caps_ok), so only K2b (queued, a
        // compile-time constant at each call) holds the law's code: inlined into K2 as well, Metal's compiler
        // miscompiled it (wrong dim C and keys, while K2b and the C++ reference were right)
        u32 zp = 2;
        const u32 dim = queued && ne ? xk_law_struct_mw(c, row, lo, hi, shot, ko, &zp) : 0;
        c->cdim[shot] = (u8)dim;
        u32 f = XK_FLAG_SURVIVE;
        if (dim < XK_CDIM_CAPACITY) {
            if (ps && !xk_masked_zero(c, row)) f = 0;
            if (zp == 2) {   // dim C = 0: v = 0
                u64 nz = 0;
                for (u32 w = 0; w < c->WO; ++w) nz |= row[c->WD + w];
                zp = nz == 0;
            }
            if (f && zp) f |= XK_FLAG_ZERO_PASS;
        }
        c->flag[shot] = (u8)f;
        return;
    }
#endif
    if (c->cw) {   // the multi-word law: the key is built in place in ko, the scratch in mws
        const u32 dim = ne ? xk_law_multi(c, row, lo, hi, shot, ko, slot) : 0;
        c->cdim[shot] = (u8)dim;
        u32 f = XK_FLAG_SURVIVE;
        if (dim < XK_CDIM_CAPACITY) {
            if (ps && !xk_masked_zero(c, row)) f = 0;
            // (XK_MODE_NOSTATUS: no key, and no verdict needed)
            if (f && !(c->mode & XK_MODE_NOSTATUS) && xk_zero_passes_multi(c, row, ko, dim, slot))
                f |= XK_FLAG_ZERO_PASS;
        }
        c->flag[shot] = (u8)f;
        return;
    }
#endif
#if defined(XK_CAP_ST_T)
    if (c->st && !e) {   // the structured law (it gives the zero decoder's verdict itself where the shot has odd axes)
        u32 zp = 2;
        const u32 dim = ne ? xk_law_struct(c, row, lo, hi, shot, ko, &zp) : 0;
        c->cdim[shot] = (u8)dim;
        u32 f = XK_FLAG_SURVIVE;
        if (dim < XK_CDIM_CAPACITY) {
            if (ps && !xk_masked_zero(c, row)) f = 0;
            if (zp == 2) {   // dim C = 0: v = 0
                u64 nz = 0;
                for (u32 w = 0; w < c->WO; ++w) nz |= row[c->WD + w];
                zp = nz == 0;
            }
            if (f && zp) f |= XK_FLAG_ZERO_PASS;
        }
        c->flag[shot] = (u8)f;
        return;
    }
#endif
    u64 H[XK_CAP_RHO];   // C's key by pivot (xk_law_fast)
    u64 hm = 0;
    for (u32 i = 0; i < XK_CAP_RHO; ++i) H[i] = 0;
    u32 dim = 0, zp = 2;   // zp: the zero decoder's verdict where the law gave it (XK_MODE_NOKEY), 2: from H, hm
    if (e) dim = xk_law_table(c, row, hi, e, shot, ko, H, &hm);
    else if (ne) dim = xk_law_fast(c, row, lo, hi, shot, ko, H, &hm, &zp);
    c->cdim[shot] = (u8)dim;
    u32 f = XK_FLAG_SURVIVE;
    if (dim < XK_CDIM_CAPACITY) {
        if (ps && !xk_masked_zero(c, row)) f = 0;
        if (f && (zp == 2 ? xk_zero_passes(c, row, H, hm) : zp == 1)) f |= XK_FLAG_ZERO_PASS;
    }
    c->flag[shot] = (u8)f;
}

XK_FN void xk_resolve_shot(XK_LOC XkCtx* c, u32 shot) {
    if (shot >= c->nshots) return;   // a rounded-up grid
    const bool ps = (c->mode & XK_MODE_POSTSELECT) != 0;
    if (!xk_caps_ok(c)) {
        c->cdim[shot] = XK_CDIM_CAPACITY;
        c->flag[shot] = XK_FLAG_SURVIVE;   // the code reaches the host
        return;
    }
    const u32 W = c->W, B = c->B;
    const u32 b = shot / B, s = shot % B;
    XK_BUF u64* row = c->rows + (u64)shot * W;
    XK_BUF u32* fp = c->epos + (u64)b * (B + 1);
    const u32 e0 = fp[s], ne = fp[s + 1] - e0;
    if (ps && (c->walk & XK_WALK_SCREEN) && !(c->mode & XK_MODE_GIVEN)) {
        if (c->flag[shot] != XK_FLAG_PENDING) return;   // the walk rejected it (its flag written, nothing else)
    } else if (ps && ne && !xk_span_in(c, row)) {   // rejected for any correction
        c->flag[shot] = XK_FLAG_EARLY;
        return;
    }
    XK_BUF u32* L = c->eids + (u64)b * c->E + e0;
    u64 lo[XK_CAP_WA], hi[XK_CAP_WA];
    u32 e = 0;
    if (ne) {
        xk_sort_ids(L, ne);
        if (c->cw && c->st && (c->mode & XK_MODE_DEFER)) {   // the structured multi-word law: K2b's (xk_finish_shot)
            c->lawq[1 + xk_atomic_inc(c->lawq)] = shot;
            return;
        }
        xk_compose(c, L, ne, lo, hi);
        u64 odd = 0;
        for (u32 w = 0; w < c->WA; ++w) odd |= lo[w];
        if (odd && !c->cw && c->lawt_on) e = xk_law_entry(c, lo, L, ne);
        if (odd && !e && !c->cw && (c->mode & XK_MODE_DEFER)) {
            c->lawq[1 + xk_atomic_inc(c->lawq)] = shot;
            return;
        }
    }
    xk_finish_shot(c, shot, row, L, ne, lo, hi, e, shot, false);
}

// K2b: thread t of xk_law_threads(nshots) finishes K2's queue entries k = t, t + G, t + 2 G, ... (G = the thread
// count; XK_MODE_DEFER): the shot's sorted list composed again and its law computed. The multi-word law's scratch is
// mws column k (the queue entry), so neighbouring threads touch neighbouring words as in K2.
#define XK_LAW_SPREAD 4
XK_FN u32 xk_law_threads(u32 nshots) { return nshots / XK_LAW_SPREAD + (nshots % XK_LAW_SPREAD != 0); }

XK_FN void xk_resolve_law(XK_LOC XkCtx* c, u32 t) {
    const u32 G = xk_law_threads(c->nshots);
    if (!(c->mode & XK_MODE_DEFER) || t >= G) return;
    const u32 nq = c->lawq[0];
    for (u32 k = t; k < nq; k += G) {
        const u32 shot = c->lawq[1 + k];
        const u32 B = c->B, b = shot / B, s = shot % B;
        XK_BUF u32* fp = c->epos + (u64)b * (B + 1);
        const u32 e0 = fp[s], ne = fp[s + 1] - e0;
        XK_BUF u32* L = c->eids + (u64)b * c->E + e0;
        u64 lo[XK_CAP_WA], hi[XK_CAP_WA];
        xk_compose(c, L, ne, lo, hi);
        xk_finish_shot(c, shot, c->rows + (u64)shot * c->W, L, ne, lo, hi, 0, k, true);
    }
}

// ------------------------------------------------------------------------------------------------ K3: reduce
// infidelity()'s accumulation of score(status, zeros) for one block (its shots [block * B, min(nshots, (block + 1)
// * B)): hist[block * stride ...] gets the survivors, the zero-decoder passes by dim C and the error-code
// shots (header comment, hist). Per-block counts are at most B; the host sums the blocks in 64 bits.
XK_FN void xk_reduce_block(XK_LOC XkCtx* c, u32 block) {
    if (block >= c->nblocks) return;   // a rounded-up grid
    const u64 shot0 = (u64)block * c->B;
    const u32 n = shot0 + c->B <= c->nshots ? c->B : (u32)(c->nshots - shot0);
    const u32 hs = c->hist_stride, err = hs - 1;
    XK_BUF u32* out = c->hist + (u64)block * hs;
    if (c->dimc > XK_CAP_DIMC || c->dimc < 64) {   // K1 reports the capacity (XK_OVERFLOW_CAPACITY); count nothing
        for (u32 i = 0; i < hs; ++i) out[i] = 0;
        return;
    }
    u32 h[XK_CAP_DIMC + 3];
    for (u32 i = 0; i < hs; ++i) h[i] = 0;
    for (u32 s = 0; s < n; ++s) {
        const u32 f = c->flag[shot0 + s];
        if (!(f & XK_FLAG_SURVIVE)) continue;
        const u32 d = c->cdim[shot0 + s];
        if (d > c->dimc) {   // an error code (XK_CDIM_*; dim C <= rhoL <= dimc otherwise)
            ++h[err];
            continue;
        }
        ++h[XK_HIST_SURVIVE];
        if (f & XK_FLAG_ZERO_PASS) ++h[XK_HIST_PASS + d];
    }
    for (u32 i = 0; i < hs; ++i) out[i] = h[i];
}


// ------------------------------------------------------------------------------------------------ K5, K6: totals

// The batch's totals in two steps, so that no thread loops over every block: K5's thread (group g, slot) sums slot
// over group g's blocks (K3's hist slot, or for XK_HIST_TOTAL_PAIRS / XK_HIST_TOTAL_CAPACITY the blocks whose
// overflow is XK_OVERFLOW_PAIRS / XK_OVERFLOW_CAPACITY) into hist_group; K6's thread (slot) sums the groups into
// hist_total. 64-bit sums; integer addition, so the totals do not depend on the grouping.
XK_FN void xk_reduce_groups(XK_LOC XkCtx* c, u32 t) {
    const u32 hs = c->hist_stride, hc = c->hist_count;
    if (t >= XK_HIST_GROUPS * hc) return;   // a rounded-up grid
    const u32 slot = t % hc, g = t / hc;
    const u32 nb = c->nblocks;
    const u32 per = nb / XK_HIST_GROUPS + (nb % XK_HIST_GROUPS != 0);   // blocks per group
    const u32 b0 = g * per < nb ? g * per : nb;
    const u32 b1 = b0 + per < nb ? b0 + per : nb;
    u64 acc = 0;
    if (slot < hs) {
        for (u32 b = b0; b < b1; ++b) acc += c->hist[(u64)b * hs + slot];
    } else {
        const u32 code = slot == hs ? XK_OVERFLOW_PAIRS : XK_OVERFLOW_CAPACITY;
        for (u32 b = b0; b < b1; ++b) acc += (u64)(c->overflow[b] == code);
    }
    c->hist_group[t] = acc;
}

XK_FN void xk_reduce_total(XK_LOC XkCtx* c, u32 slot) {
    const u32 hc = c->hist_count;
    if (slot >= hc) return;   // a rounded-up grid
    u64 acc = 0;
    for (u32 g = 0; g < XK_HIST_GROUPS; ++g) acc += c->hist_group[(u64)g * hc + slot];
    c->hist_total[slot] = acc;
}

// ------------------------------------------------------------------------------------------------ K7, K4: outputs
// K7 xk_scan_blocks (thread = block, after K5; postselection only): boff[j] = the batch's survivors (flag
// XK_FLAG_SURVIVE: hist's SURVIVE and error slots, as K3 counts them) among the shots before shot j XK_EMIT_RUN, so
// K4 counts at most XK_EMIT_RUN - 1 flags whatever the block. Block b starts from the sums of the K5 groups before
// its own and of its group's blocks before it, then counts its shots' flags, writing boff[j] at each shot j
// XK_EMIT_RUN it holds (every run start lies in exactly one block).
#define XK_EMIT_RUN 64
XK_FN void xk_scan_blocks(XK_LOC XkCtx* c, u32 block) {
    const u32 nb = c->nblocks;
    if (block >= nb) return;   // a rounded-up grid
    const u32 hs = c->hist_stride, hc = c->hist_count, err = hs - 1;
    const u32 per = nb / XK_HIST_GROUPS + (nb % XK_HIST_GROUPS != 0);   // K5's blocks per group
    const u32 g = block / per;
    u64 off = 0;
    for (u32 h = 0; h < g; ++h) off += c->hist_group[(u64)h * hc + XK_HIST_SURVIVE] + c->hist_group[(u64)h * hc + err];
    for (u32 b = g * per; b < block; ++b)
        off += (u64)c->hist[(u64)b * hs + XK_HIST_SURVIVE] + c->hist[(u64)b * hs + err];
    const u64 s0 = (u64)block * c->B;
    const u64 s1 = s0 + c->B < c->nshots ? s0 + c->B : c->nshots;
    for (u64 s = s0; s < s1; ++s) {
        if (s % XK_EMIT_RUN == 0) c->boff[s / XK_EMIT_RUN] = (u32)off;
        off += c->flag[s] & XK_FLAG_SURVIVE;
    }
}

// the C-id table's hash of a key of kw words (xtim.gpu.tables.key_hash)
XK_FN u64 xk_key_hash(XK_BUF const u64* key, u32 kw) {
    u64 h = XK_GAMMA;
    for (u32 i = 0; i < kw; ++i) h = xk_mix(h ^ key[i]) + XK_GAMMA;
    return h;
}

// the C id of a key of kw words in the read-only table ctab (xtim.gpu.tables.cspace_table): ctab[0] = n slots (a
// power of two; 0: an empty table), slot s at ctab[1 + s] = (id + 1) << 32 | key index (0: an empty slot), key i at
// ctab[1 + n + i kw ...]; linear probing from xk_key_hash & (n - 1). 0xFFFFFFFF: the key is not in the table.
XK_FN u32 xk_cspace_lookup(XK_LOC const XkCtx* c, XK_BUF const u64* key, u32 kw) {
    const u64 n = c->ctab[0];
    const u64 h = xk_key_hash(key, kw);
    for (u64 p = 0; p < n; ++p) {
        const u64 e = c->ctab[1 + ((h + p) & (n - 1))];
        if (!e) break;
        XK_BUF const u64* k = c->ctab + 1 + n + (e & 0xFFFFFFFFu) * kw;
        u64 d = 0;
        for (u32 i = 0; i < kw; ++i) d |= k[i] ^ key[i];
        if (!d) return (u32)(e >> 32) - 1;
    }
    return 0xFFFFFFFFu;
}

// K4 xk_emit (thread = shot, after K2 / K2b and, under XK_MODE_POSTSELECT, K7; run under XK_MODE_EMIT): a survivor's
// output row k (the shot itself without postselection; with it boff[run] + the survivors before it in its run of
// XK_EMIT_RUN shots, so the survivors in shot order) in sample's layouts: its detector bytes out_det[k nbd ...]
// (bytes 0..nbd - 1 of the row), observable bytes out_obs[k nbo ...] (bytes 8 WD ..) and C id out_cs[k]: 0 at dim C = 0, else ctab's id of
// its key, else 0xFFFFFFFF (a key not in ctab, or an error code in cdim) and the pair (k, shot) appended to cmiss
// (cmiss[0] the pairs, K1's block 0 sets it to 0; cmiss[1 + 2 i], cmiss[2 + 2 i] pair i, in no particular order), for
// the host to finish. Other shots write nothing. Under XK_MODE_NOSTATUS (sample without the status: ckey was not
// written) out_cs is not written and only a shot whose cdim is an error code goes to cmiss.
XK_FN void xk_emit(XK_LOC XkCtx* c, u32 shot) {
    if (shot >= c->nshots) return;   // a rounded-up grid
    if (!(c->flag[shot] & XK_FLAG_SURVIVE)) return;
    u32 k = shot;
    if (c->mode & XK_MODE_POSTSELECT) {
        const u32 s0 = shot - shot % XK_EMIT_RUN;
        k = c->boff[shot / XK_EMIT_RUN];
        for (u32 i = s0; i < shot; ++i) k += c->flag[i] & XK_FLAG_SURVIVE;
    }
    if (k >= c->nshots) return;   // a guard only: K7's offsets count these same flags (K3), so k < nshots
    XK_BUF const u64* row = c->rows + (u64)shot * c->W;
    XK_BUF u8* od = c->out_det + (u64)k * c->nbd;
    for (u32 i = 0; i < c->nbd; ++i) od[i] = (u8)(row[i >> 3] >> (8 * (i & 7)));
    XK_BUF const u64* ob = row + c->WD;
    XK_BUF u8* oo = c->out_obs + (u64)k * c->nbo;
    for (u32 i = 0; i < c->nbo; ++i) oo[i] = (u8)(ob[i >> 3] >> (8 * (i & 7)));
    const u32 d = c->cdim[shot];
    if (c->mode & XK_MODE_NOSTATUS) {   // no C id (out_cs not written): only an error code goes to the host
        if (d > c->dimc) {
            const u32 m = xk_atomic_inc(c->cmiss);
            c->cmiss[1 + 2 * m] = k;
            c->cmiss[2 + 2 * m] = shot;
        }
        return;
    }
    u32 id = 0;
    if (d) {
        const u32 kw = (c->rhoL ? c->rhoL : 1) * (c->cw ? c->cw : 1);
        id = d > c->dimc ? 0xFFFFFFFFu : xk_cspace_lookup(c, c->ckey + (u64)shot * kw, kw);
        if (id == 0xFFFFFFFFu) {
            const u32 m = xk_atomic_inc(c->cmiss);
            c->cmiss[1 + 2 * m] = k;
            c->cmiss[2 + 2 * m] = shot;
        }
    }
    c->out_cs[k] = id;
}
