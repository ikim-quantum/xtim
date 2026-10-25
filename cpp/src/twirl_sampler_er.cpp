// twirl_sampler_er — early-reject (OD-2026-10-25-early-reject-stage): the opt-in postselection mask of the twirl
// sampler — its set-up (er_set_), the late filter's drop and the public surface. The masked instances of the shot loop,
// run_w<8 | kErRun> and run_w<0 | kErRun> (body: twirl_sampler_runw.inc.hpp), the term route's test and the record
// roll-back live in twirl_sampler_fam.cpp (the FAM route state is complete there); run() takes the masked instances
// only while a mask is set (Impl::run).
//
// The rule (owner design, OD-2026-10-25-early-reject): per shot the fired faults compose into a Pauli part P and a
// Clifford part C. For a flagged deterministic detector channel c whose operator M_c shares no qubit with C, C
// commutes with M_c, so the channel's emitted value is ⟨P, M_c⟩ ⊕ rf_c ⊕ K_c, K_c the clean shot's emitted value —
// fixed before any plan lookup, plan build or Born draw. If it is 1 the shot is rejected there. Sound and incomplete:
// every emitted shot of a masked run also passes the exact late filter on its packed detector row. Every shot still
// takes its cumulative index shot_ (assigned before any route), so the accepted shots draw exactly the counter-based
// streams they draw in an unmasked run: their bytes equal that run's, filtered afterwards.

#include "twirl_sampler_impl.inc.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qeccore {

TwirlRecordSampler::Impl::ErHold::~ErHold() { delete p; }

__attribute__((noinline)) void TwirlRecordSampler::Impl::er_late_drop_() {
    FamilySink& fs = *family_sink_;
    const long r = fs.row - 1;
    if (fs.DBB) std::memset(fs.dets + (size_t)r * (size_t)fs.DBB, 0, (size_t)fs.DBB);
    if (fs.OBB) std::memset(fs.obs + (size_t)r * (size_t)fs.OBB, 0, (size_t)fs.OBB);
    if (fs.DECB) std::memset(fs.dec + (size_t)r * (size_t)fs.DECB, 0, (size_t)fs.DECB);
    fs.row = r;
    fam_rec_rollback_();
    ++er_.p->late;
}

// Cold: the mask's precomputed data. `det_flags[d]` != 0 flags detector d (its length is the detector count).
void TwirlRecordSampler::Impl::er_set_(const std::vector<uint8_t>& det_flags) {
    delete er_.p;
    er_.p = nullptr;
    bool any = false;
    for (uint8_t f : det_flags) any |= f != 0;
    if (!any) return;
    auto R = std::make_unique<ErState>();
    const size_t cw = (size_t)std::max(CW, 1);
    R->F.assign(cw, 0);
    R->K.assign(cw, 0);
    R->tmp.assign(cw, 0);
    R->hit.assign(cw, 0);
    // The late filter: every flagged detector (deterministic channels, gauge / ANTI extras alike), packed b8.
    for (size_t d = 0; d < det_flags.size(); ++d) {
        if (!det_flags[d]) continue;
        const int b = (int)(d >> 3);
        auto it = std::find(R->dbyte.begin(), R->dbyte.end(), b);
        if (it == R->dbyte.end()) { R->dbyte.push_back(b); R->dmask.push_back(0); it = R->dbyte.end() - 1; }
        R->dmask[(size_t)(it - R->dbyte.begin())] |= (uint8_t)(1u << (d & 7));
    }
    // The early test reads the flagged DETERMINISTIC detector channels.
    bool anyF = false;
    for (size_t c = 0; c < chan_det.size(); ++c) {
        const int d = chan_det[c];
        if (d >= 0 && (size_t)d < det_flags.size() && det_flags[(size_t)d]) {
            R->F[c >> 6] |= 1ULL << (c & 63);
            anyF = true;
        }
    }
    er_.p = R.release();
    ErState& S = *er_.p;
    if (!anyF) return;
    // K: the clean shot's emitted value (the identity plan's base row, internal convention, ⊕ the reporting ref). A
    // channel the identity plan's coin / Born rows reach is not deterministic there: it leaves the early set (sound).
    const CachedPlan& p0 = identity_plan_();
    ensure_rec_(p0);
    const int nr = 1 + p0.emit_rho + p0.emit_kappa;
    const uint64_t* E = p0.emit.data();
    for (int r = 1; r < nr; ++r)
        for (int w = 0; w < CW; ++w) S.F[(size_t)w] &= ~E[(size_t)r * cw + (size_t)w];
    for (int w = 0; w < CW; ++w) S.K[(size_t)w] = E[w];
    for (size_t c = 0; c < chan_det.size(); ++c)
        if (chan_ref[c]) S.K[c >> 6] ^= 1ULL << (c & 63);
    for (int w = 0; w < CW; ++w) S.K[(size_t)w] &= S.F[(size_t)w];
    // full: every flagged detector is a channel the early test reads (no gauge / ANTI extra, none dropped above), so
    // a shot the early test fully predicts needs no late filter.
    {
        size_t inF = 0, flagged = 0;
        for (uint8_t f : det_flags) flagged += f != 0;
        for (size_t c = 0; c < chan_det.size(); ++c) inF += (S.F[c >> 6] >> (c & 63)) & 1ULL;
        S.full = inF == flagged;
    }
    S.clean_ok = true;
    for (int w = 0; w < CW; ++w) S.clean_ok &= S.K[(size_t)w] == 0;
    // The wire rows: flagged channels whose operator has support on wire q (the diagonal routes' C-support test).
    S.wire.assign((size_t)std::max(n, 1) * cw, 0);
    for (int c = 0; c < nchan; ++c) {
        if (!((S.F[(size_t)(c >> 6)] >> (c & 63)) & 1ULL)) continue;
        const Pauli& M = rec_cols_[(size_t)c].M;
        for (int q = 0; q < n && q < M.n; ++q)
            if (M.xbit(q) || M.zbit(q)) S.wire[(size_t)q * cw + (size_t)(c >> 6)] |= 1ULL << (c & 63);
    }
    fam_er_terms_();
}

// ── The public surface ─────────────────────────────────────────────────────────────────────────────────────────
void TwirlRecordSampler::set_postselect(const std::vector<uint8_t>& det_flags) { impl_->er_set_(det_flags); }
bool TwirlRecordSampler::postselect_active() const { return impl_->er_.p != nullptr; }
long TwirlRecordSampler::postselect_early() const { return impl_->er_.p ? impl_->er_.p->early : 0; }
long TwirlRecordSampler::postselect_late() const { return impl_->er_.p ? impl_->er_.p->late : 0; }

}  // namespace qeccore
