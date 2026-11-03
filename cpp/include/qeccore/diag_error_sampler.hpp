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
// Output over a batch of S shots: sample_events — fired (shot → alternative-id) lists, shot-bucketed
// (counting sort). (bloat-cut B5: the bit-sliced plane output, its transpose and the per-shot fold
// reference had no caller and are deleted.)
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
    std::vector<Alt> alts;                    // >= 1 alternatives (any count); Σp <= 1 (= 1: forced)
};

class DiagErrorSampler {
  public:
    // n = qubit count; channels = the fixed set; seed -> deterministic stream.
    // keyed = false: one sequential xoshiro256** stream shared by all channels (channel-major within
    //   a batch, so the BYTES depend on the batch partition; the law does not).
    // keyed = true (the twirl sampler, S0(b)): channel k draws from its OWN keyed counter stream,
    //   draw c = mix(krng::channel_base(seed, k) + (c+1)·γ) — every channel's skip sequence is then a
    //   pure function of (seed, k), independent of the batch partition and call split.
    DiagErrorSampler(int n, const std::vector<ErrorChannel>& channels, uint64_t seed,
                     bool keyed = false);

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
    // scaling S1: the same S shots' events drawn `chunk` shots at a time (the stream is partition-independent: the
    // same events as sample_events(S)), so the generation and the shot sort stay cache-sized for a large S.
    void sample_events_chunked(int S, int chunk, SparseBatch& out);
    // scaling T8: the same events generated in parallel over channel ranges (keyed streams only). A task owns the
    // channels [k0, k1); gen_range draws their events of the next S shots (it advances only those channels' streams,
    // so tasks on disjoint ranges may run concurrently) and buckets them by block (B shots), each block k-major.
    // merge_block lays block b (its c shots) out shot-major from the G tasks in range order — per shot the events in
    // ascending channel, exactly sample_events' order: ev (the block's events) and shot_end[i] = base + the end of
    // shot i's events (i < c: the batch's shot_off[b·B + 1 + i] when ev = batch.ev + base). Returns the event count.
    struct EventBuckets {
        int k0 = 0, k1 = 0;               // the channel range
        std::vector<uint64_t> tmp, ev;    // (alt << 32 | s): generation order, then bucketed by block
        std::vector<int32_t> off, pos;    // block b's events: ev[off[b], off[b + 1])
    };
    std::vector<int> channel_split(int G) const;   // G + 1 range bounds over [0, K), about equal expected events
    // scaling R1 / fix-2 F1: the stream's position (every channel's counter and next firing; the sequential state; the
    // draw is a pure function of it). A failed sampler call puts the stream back where the call found it.
    // restore_pos(p) requires p saved from THIS sampler (sizes equal: checked by save_pos's caller) and does not throw.
    struct StreamPos {
        std::vector<uint64_t> kpos;
        std::vector<int64_t> next;
        uint64_t rs[4] = {0, 0, 0, 0};
    };
    void save_pos(StreamPos& p) const;
    bool pos_matches(const StreamPos& p) const { return p.kpos.size() == kpos_.size() && p.next.size() == next_.size(); }
    void restore_pos(const StreamPos& p) noexcept;
    void gen_range(int S, int B, EventBuckets& out);
    static int merge_block(const EventBuckets* tasks, int G, int b, int B, int S, std::vector<int32_t>& cnt,
                           int32_t base, int32_t* shot_end, int32_t* ev);

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
    uint64_t rs_[4];                       // xoshiro256** state (keyed_ == false)
    bool keyed_ = false;
    std::vector<uint64_t> kpos_;           // keyed_: channel k's stream position (base + c·γ)
    // perf-glue §4.4: the state reseed(seed) derives (kpos_ / rs_ and next_), kept for a seed reseeded twice in a row:
    // reseeding to it again restores the copy instead of re-deriving it (O(K) mix + log draws). A new seed costs no
    // copy (the snapshot is taken on the second consecutive reseed to the same seed).
    bool snap_ok_ = false, last_ok_ = false;
    uint64_t snap_seed_ = 0, last_seed_ = 0;
    uint64_t snap_rs_[4] = {0, 0, 0, 0};
    std::vector<uint64_t> snap_kpos_;
    std::vector<int64_t> snap_next_;

    // event scratch: k-major packed (k<<32 | s) from generation, then shot-sorted
    std::vector<uint64_t> ev_ks_;
    std::vector<int32_t> counts_, sort_off_, sort_ev_, sort_s_;

    inline uint64_t rnext();
    inline double runi_open();             // (0, 1]
    static inline double kuni_open(uint64_t& pos);   // keyed_: next draw of one channel, (0, 1]
    void gen_events(int S);                // fills ev_ks_ (k-major)
    template <bool Keyed> void gen_events_t(int S);
    void gen_range_events_(int S, int k0, int k1, std::vector<uint64_t>& ev);   // scaling T8 (keyed)
    void sort_events(int S);               // fills sort_off_/sort_ev_ (shot-major)
};

}  // namespace qeccore
