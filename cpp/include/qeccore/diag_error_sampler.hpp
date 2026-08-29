#pragma once
// Batch sampler for diagonal-Clifford error channels (the MSP sampling hot path).
//
// MODEL. A fixed set of K error locations; location k fires independently with probability p_k
// per shot. Each location is a DIAGONAL Clifford: S-powers a_q ∈ ℤ4 on a small qubit support and
// CZ indicators on a small pair support. Diagonal Cliffords compose ABELIANLY and component-wise
// (a adds mod 4, CZ XORs; the global phase of the combined error is unobservable and dropped), so
// "combine the fired errors of a shot" is a ℤ4 vector sum plus a GF(2) pair-set XOR.
//
// ALGORITHM. At physical error rates (p ~ 1e-4..1e-3) the dominant cost of naive sampling is the
// K Bernoulli draws per shot, almost all of which come up empty. The sampler therefore walks each
// error's shot axis with GEOMETRIC SKIPS — one log() per FIRED event plus one per (error, batch)
// — so total work scales with S·Σp_k (the expected number of fired events), not with S·K. This is
// the same rare-event regime treatment Stim applies below its dense-sampling threshold; measured
// against Stim's FrameSimulator on a matched workload (n=100, K=4000 noisy 2q channels at 1e-3)
// this sampler produces COMBINED diagonal-Clifford errors several times faster than Stim samples
// its (simpler) Pauli frames.
//
// Two output formats over a batch of S shots:
//   * sample_events  — fired (shot → error-index) lists, shot-bucketed (counting sort). The
//                      per-shot combined error is a trivial fold; fold_shot() is provided.
//   * sample_planes  — Stim-style bit-sliced lanes: bit s of plane word gives shot s's value.
//                      Per qubit the ℤ4 power is two S-bit planes (lo/hi); per union pair one
//                      S-bit plane. Events are applied as single-bit read-modify-writes (the
//                      planes are ~all-zero at these p; dense masked adds would waste the lanes).
//                      transpose_planes() unpacks lanes to per-shot words when a per-shot
//                      consumer needs them.
// Both formats consume the SAME event stream: with equal seeds, fold(sample_events) ==
// transpose(sample_planes) bit-for-bit (tested).
//
// DISTRIBUTION EXACTNESS. Skips use the integer-exact geometric formula
//   next = s + 1 + floor(log(U)/log1p(-p)),  U ∈ (0,1],
// with double-precision std::log: the per-event firing law is Bernoulli(p_k) i.i.d. across shots
// up to ~1-ulp rounding of the log ratio (bias ~1e-15, far below any statistical resolution).
#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>

namespace qeccore {

// One diagonal-Clifford error location (independent Bernoulli — the one-alternative channel).
struct DiagError {
    std::vector<std::pair<int, uint8_t>> a;   // (qubit, S-power 1..3)
    std::vector<std::pair<int, int>> cz;      // CZ pairs (q, r), q != r
    double p = 0.0;                           // firing probability, 0 <= p <= 1 (1 = every shot)
};

// A CATEGORICAL error channel — the general stim-noise shape (DEPOLARIZE1/2, PAULI_CHANNEL_1/2,
// E/ELSE chains, heralded channels): ONE location that fires with p_total = Σ alternative
// probabilities and, when fired, realizes EXACTLY ONE alternative (mutually exclusive). Skip
// sampling is exact via the identity "fire w.p. Σp_i, then choose i w.p. p_i/Σ" — the geometric
// machinery is unchanged and one conditional draw is paid per FIRED event only.
// Each alternative's action is the PROPAGATED end-of-circuit form of its Pauli: a diagonal
// Clifford (S-powers + CZ pairs) plus an outcome/herald flip mask (the X^v frame part's effect
// on up to 64 reads/record bits) — e.g. a circuit-level DEPOLARIZE2 enters as 15 alternatives at
// p/15 whose actions the propagation table supplies.
struct ErrorChannel {
    struct Alt {
        double p = 0.0;
        std::vector<std::pair<int, uint8_t>> a;
        std::vector<std::pair<int, int>> cz;
        uint64_t flips = 0;
    };
    std::vector<Alt> alts;                    // 1..255 alternatives; Σp <= 1 (= 1: forced)
};

class DiagErrorSampler {
  public:
    // n = qubit count; channels = the fixed set; seed -> deterministic stream.
    DiagErrorSampler(int n, const std::vector<ErrorChannel>& channels, uint64_t seed);
    // Convenience: a set of independent Bernoulli errors (each = a one-alternative channel).
    DiagErrorSampler(int n, const std::vector<DiagError>& errors, uint64_t seed);

    int n() const { return n_; }
    int K() const { return K_; }                                      // channel count
    int alt_total() const { return (int)flips_.size(); }              // flat alternative count
    int pairs() const { return P_; }                                  // union CZ pairs
    const std::vector<std::pair<int, int>>& pair_list() const { return pair_list_; }
    int32_t channel_of(int32_t altid) const { return alt_k_[altid]; } // alternative -> channel
    uint64_t flips(int32_t altid) const { return flips_[altid]; }     // alternative's flip mask
    void reseed(uint64_t seed);

    // ---- event-list output (shot-bucketed) ----
    // ev entries are flat ALTERNATIVE ids in [0, alt_total()); for an all-Bernoulli set the id
    // equals the error index. channel_of()/flips() map ids back.
    struct SparseBatch {
        int S = 0;
        std::vector<int32_t> shot_off;   // size S+1; shot s's fired events are ev[off[s]..off[s+1])
        std::vector<int32_t> ev;         // alternative ids, shot-major
    };
    void sample_events(int S, SparseBatch& out);

    // Fold one shot's fired errors into the combined (a, B). aout: n ℤ4 entries (caller-zeroed or
    // reused via the returned dirty lists); czbits: ceil(P/64) words. Definitional reference —
    // the abelian compose IS this sum.
    struct FoldScratch {
        std::vector<uint8_t> a;            // n entries, ℤ4
        std::vector<uint64_t> czbits;      // ceil(P/64) words
        std::vector<int32_t> dirty_q, dirty_w;
        std::vector<uint8_t> in_q, in_w;   // membership flags: the dirty lists are duplicate-
                                           // free (an accumulator passing through 0 must not
                                           // re-push — consumers apply per-occurrence effects;
                                           // same 2026-06-11 bug class as run_shots_packed)
        void init(int n, int P) { a.assign(n, 0); czbits.assign((size_t)(P + 63) / 64, 0);
                                  in_q.assign(n, 0); in_w.assign(czbits.size(), 0); }
        void clear();                      // clears only dirty entries
    };
    void fold_shot(const SparseBatch& b, int s, FoldScratch& acc) const;

    // ---- bit-sliced plane output ----
    // Block-interleaved layout: the planes of one 64-shot block are CONTIGUOUS (stride = 2n + P
    // words: alo[0..n) | ahi[0..n) | cz[0..P)), so applying a shot's events touches one ~stride*8B
    // chunk (L1-resident when events are visited in shot order) and the per-block transpose reads
    // sequentially. Lane bit j of block b = shot 64b + j.
    struct Planes {
        int S = 0, SW = 0, n = 0, P = 0, stride = 0;
        std::vector<uint64_t> w;           // SW blocks x stride words
        uint64_t alo(int q, int b) const { return w[(size_t)b * stride + q]; }
        uint64_t ahi(int q, int b) const { return w[(size_t)b * stride + n + q]; }
        uint64_t cz(int idx, int b) const { return w[(size_t)b * stride + 2 * n + idx]; }
    };
    void sample_planes(int S, Planes& out);                            // S must be a multiple of 64

    // Unpack shot s from planes into per-shot words: aw_lo/aw_hi = ceil(n/64) words (bit q = lo/hi
    // of qubit q's ℤ4 power), czw = ceil(P/64) words. O(planes/64) per 64-shot block when called
    // sequentially — uses an internal transposed cache rebuilt per block.
    void transpose_shot(const Planes& pl, int s,
                        std::vector<uint64_t>& aw_lo, std::vector<uint64_t>& aw_hi,
                        std::vector<uint64_t>& czw) const;

  private:
    int n_ = 0, K_ = 0, P_ = 0;
    std::vector<std::pair<int, int>> pair_list_;
    // Per-ALTERNATIVE action in ONE 16-byte record (the dominant gate-error shape: <=2 S-power
    // entries + <=1 CZ): one cache line per fired event instead of 4-5 scattered CSR loads.
    // Alternatives with larger supports set csr >= 0 and take the general CSR path.
    struct Rec {
        int32_t q1, q2;       // a-support qubits; q2 = -1 if single (q1 = -1 if none)
        int32_t cz;           // union pair index, -1 if none
        uint8_t c1, c2;       // Z4 powers
        int16_t csr;          // -1 = record complete; else index into the CSR fallback tables
    };
    std::vector<Rec> recs_;                // indexed by flat alternative id
    std::vector<uint64_t> flips_;          // per alternative: outcome/record flip mask
    std::vector<int32_t> alt_k_;           // per alternative: owning channel
    std::vector<int32_t> altbase_;         // per channel: first alternative id (size K+1)
    std::vector<double> cum_;              // per channel: cumulative alternative probs (flat)
    std::vector<int32_t> cumoff_;          // per channel: offset into cum_ (size K+1)
    std::vector<double> ptot_;             // per channel: Σ alternative probs
    // CSR fallback (only alternatives too big for a Rec): a-entries then cz-plane indices
    std::vector<int32_t> a_off_, a_q_;
    std::vector<uint8_t> a_c_;
    std::vector<int32_t> cz_off_, cz_idx_;
    std::vector<double> skipc_;            // 1 / log1p(-p_total_k)
    std::vector<int64_t> next_;            // per-channel next firing index, relative to batch start
    uint64_t rs_[4];                       // xoshiro256** state

    // event scratch: k-major packed (k<<32 | s) from generation, then shot-sorted
    std::vector<uint64_t> ev_ks_;
    std::vector<int32_t> counts_, sort_off_, sort_ev_, sort_s_;
    mutable std::vector<uint64_t> tcache_; // transpose_shot block cache
    mutable int tcache_block_ = -1;
    mutable const Planes* tcache_src_ = nullptr;

    inline uint64_t rnext();
    inline double runi_open();             // (0, 1]
    void gen_events(int S);                // fills ev_ks_ (k-major)
    void sort_events(int S);               // fills sort_off_/sort_ev_ (shot-major)
};

}  // namespace qeccore
