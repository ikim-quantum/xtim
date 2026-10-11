// The noiseless REFERENCE sample — Stim's Circuit.reference_sample() convention — shared by every
// engine (audit finding K). Compile-time only: nothing here is on a per-shot path. Kept in its own
// translation unit so the per-shot code in framed_superposition.cpp / sampler.cpp /
// twirl_sampler.cpp is not re-inlined around it (an added helper in a hot TU moved the twirl
// shot loop by 5-17% through codegen alone).
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "qeccore/feedback.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli_kernels.hpp"   // pmul_into
#include "qeccore/sampler.hpp"

namespace qeccore {

// chi==1 sequential stabilizer pass with a per-read bias: batch_chi1_pre's algorithm, except a
// random read takes out = want(k) (o = want ^ sign) instead of a uniform coin.
static void batch_chi1_pre_toward(const FramedSuperposition& L, const std::vector<Pauli>& Qpre,
                                  const std::vector<int>& gidx,
                                  const std::function<int(int, const std::vector<int>&)>& want,
                                  std::vector<int>& gout) {
    const int N = L.n(), W = (N + 63) / 64;
    std::vector<uint64_t> eps_pk((size_t)W, 0);
    for (int a = 0; a < N; ++a) if (L.eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);
    std::vector<MeasCoin> coins;
    for (size_t k = 0; k < Qpre.size(); ++k) {
        Pauli acc = Qpre[k];
        int sign = 0;
        for (auto& c : coins)
            if ((acc.x[c.pivot >> 6] >> (c.pivot & 63)) & 1ull) { pmul_into(acc, c.Qc); sign ^= c.o; }
        int xnz = -1;
        for (int w = 0; w < W; ++w) if (acc.x[w]) { xnz = (w << 6) + __builtin_ctzll(acc.x[w]); break; }
        const int g = gidx[k];
        if (xnz < 0) {
            int zpar = 0; for (int w = 0; w < W; ++w) zpar += __builtin_popcountll(acc.z[w] & eps_pk[w]);
            gout[(size_t)g] = (((acc.phase & 3) == 2) ? 1 : 0) ^ (zpar & 1) ^ sign;
        } else {
            const int o = (want(g, gout) & 1) ^ sign;
            gout[(size_t)g] = o ^ sign;
            coins.push_back({std::move(acc), xnz, o});
        }
    }
}

void framed_reference_records(FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                              const std::function<int(int, const std::vector<int>&)>& want,
                              std::vector<int>& out) {
    out.assign(reads.size(), -1);
    size_t k = 0;
    for (; k < reads.size() && L.chi() > 1; ++k) {
        const FramedRefRead r = framed_reference_read_toward(
            L, reads[k].first, reads[k].second, want((int)k, out) & 1);
        out[k] = (r.out == +1) ? 0 : 1;
    }
    if (k == reads.size()) return;
    std::vector<std::pair<int, int>> rem(reads.begin() + (long)k, reads.end());
    std::vector<int> gidx;
    for (size_t j = k; j < reads.size(); ++j) gidx.push_back((int)j);
    L.U.ensure_dual();                                        // conjugate_single: O(n) per read
    std::vector<Pauli> Qpre;
    Qpre.reserve(rem.size());
    for (const auto& rd : rem) Qpre.push_back(L.U.conjugate_single(rd.first, rd.second));
    batch_chi1_pre_toward(L, Qpre, gidx, want, out);
}

std::vector<uint8_t> noiseless_reference_records(const FramedSuperposition& bare,
                                                 const Circuit& deferred,
                                                 const std::vector<std::pair<int, int>>& reads,
                                                 const std::vector<uint8_t>& rec_invert,
                                                 const FeedbackPlan* fb) {
    const int M = (int)reads.size();
    std::vector<uint8_t> R((size_t)M, 0);
    if (M == 0) return R;
    // elim_k: the sign eliminate_hadamards stamped on deferred read k (deferred Measure order ==
    // terminal-read order). raw user record_k = b_k ^ elim_k ^ flip_k, so the +1 bias is
    // want(b_k) = elim_k ^ flip_k (flip_k: the feedback relabel's flip of record k given the
    // reference's EARLIER records — triangular, so it is final when read k is collapsed).
    std::vector<uint8_t> elim;
    elim.reserve((size_t)M);
    for (const Instr& ins : deferred.stream)
        if (ins.kind == Instr::Kind::Measure) elim.push_back(ins.invert ? 1 : 0);
    if ((int)elim.size() != M)
        throw std::logic_error("noiseless_reference_records: deferred Measure count != reads");
    std::vector<uint8_t> flip((size_t)M, 0);
    std::vector<std::vector<int>> ops_at;                     // feedback ops by control record
    if (fb) {
        ops_at.assign((size_t)M, {});
        for (size_t i = 0; i < fb->ops.size(); ++i)
            if (fb->ops[i].control_record >= 0 && fb->ops[i].control_record < M)
                ops_at[(size_t)fb->ops[i].control_record].push_back((int)i);
    }
    int done = 0;                                             // records folded into R so far
    // R_k = b_k ^ rec_invert_k ^ flip_k: the RECORDED bit (the feedback relabel reads the recorded
    // control, invert included — pack_shot_records' order: invert first, feedback second).
    auto settle = [&](const std::vector<int>& b, int upto) {
        for (; done < upto; ++done) {
            const int j = done;
            int r = b[(size_t)j] ^ flip[(size_t)j];
            if (!rec_invert.empty() && rec_invert[(size_t)j]) r ^= 1;
            R[(size_t)j] = (uint8_t)(r & 1);
            if (fb && r)
                for (int oi : ops_at[(size_t)j]) {
                    const std::vector<uint64_t>& mf = fb->ops[(size_t)oi].mflips;
                    for (int k = j + 1; k < M; ++k)
                        if ((size_t)(k >> 6) < mf.size() && ((mf[(size_t)(k >> 6)] >> (k & 63)) & 1))
                            flip[(size_t)k] ^= 1;
                }
        }
    };
    FramedSuperposition work = bare;
    std::vector<int> ro;
    framed_reference_records(work, reads,
        [&](int k, const std::vector<int>& b) {
            settle(b, k);                                     // records < k are final: fold them
            return (int)elim[(size_t)k] ^ (int)flip[(size_t)k];
        }, ro);
    settle(ro, M);
    return R;
}

}  // namespace qeccore
