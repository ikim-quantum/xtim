// twirl_sampler_rot.cpp — lean D1-S1 (2026-11-05): the rotation model and the compact per-alternative rotation records,
// compiled once at the end of the sampler's constructor from the term view fam_setup_ built (lean/D1/DESIGN.md §1
// items 1–3; the model itself: qeccore/twirl_rotation.hpp / twirl_rotation.cpp).
//
// Per alternative a ONE u32 record (RotationRecords): [sign words S_f][token = plan_key_hash(S_f) ⊕ C][XR extras words
// ⟨P_f, M_e⟩][nz][(w, lo, hi) × nz: the nonzero channel words of t(P_f)]. t(P_f) is the end-pushed Clifford-only frame's
// relabel over the record columns: ⊕_{q ∈ x(P_f)} pxrow[q] ⊕ ⊕_{q ∈ z(P_f)} pzrow[q] on the channels (the per-wire prefix
// rows the ctor builds) and xrow_x_ / xrow_z_ on the extras — the same rows the streaming route reads today.
// The alternatives' (P_f, S_f) come from the term view's propagation in CHUNKS (TermResidual::for_alts: kRotChunk
// alternatives per bit-sliced pass — D1 review M2: no A × NW dense transient; the chunk's rows are a few tens of MB on the
// full d5 / s25 circuits); each chunk's records are appended, then VERIFIED (decoded, bounds-checked, == the chunk's rows
// they were encoded from, as the L2 blob is checked against its masks); a corrupted word is the named refusal
// "twirl_records: ROT RECORD VERIFY FAIL ..." (setup_error 5; the test seam g_twirl_rot_rec_test_hook corrupts the vector
// per chunk, cpp/tests/test_rotation_verify.cpp). The test API's family_rotation_check re-propagates and decodes again.
// S1: nothing sampled reads the model or the records; `family_ctor_bytes` does not hash them (the digest's move is
// signed with S2's stream change).
#include "twirl_sampler_impl.inc.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace qeccore {

namespace {
constexpr uint64_t kHashC = 0xcbf29ce484222325ull;   // plan_key_hash's constant: h(a ⊕ b) = h(a) ⊕ h(b) ⊕ C
constexpr int kRotChunk = 16384;                     // alternatives per propagation pass (256 words per wire row)
inline uint64_t val64(const uint32_t* p) { return (uint64_t)p[0] | ((uint64_t)p[1] << 32); }
}  // namespace

// t(P_f) over the channels (CW words) and the extras (XR words) from the frame's NW-word x / z rows.
void TwirlRecordSampler::Impl::rot_row_of_(const uint64_t* px, const uint64_t* pz, uint64_t* chan, uint64_t* xw) const {
    const int XR = XR_;
    for (int w = 0; w < CW; ++w) chan[w] = 0;
    for (int w = 0; w < XR; ++w) xw[w] = 0;
    for (int w = 0; w < NW; ++w) {
        for (uint64_t word = px[w]; word; word &= word - 1) {
            const int q = w * 64 + __builtin_ctzll(word);
            for (int u = 0; u < CW; ++u) chan[u] ^= pxrow[(size_t)q * CW + (size_t)u];
            for (int u = 0; u < XR; ++u) xw[u] ^= xrow_x_[(size_t)q * XR + (size_t)u];
        }
        for (uint64_t word = pz[w]; word; word &= word - 1) {
            const int q = w * 64 + __builtin_ctzll(word);
            for (int u = 0; u < CW; ++u) chan[u] ^= pzrow[(size_t)q * CW + (size_t)u];
            for (int u = 0; u < XR; ++u) xw[u] ^= xrow_z_[(size_t)q * XR + (size_t)u];
        }
    }
}

void TwirlRecordSampler::Impl::rot_record_of_(const uint64_t* s, const uint64_t* chan, const uint64_t* xw,
                                              std::vector<uint32_t>& out) const {
    const int TW = rot_rec.TW, XR = XR_;
    auto put64 = [&](uint64_t v) { out.push_back((uint32_t)v); out.push_back((uint32_t)(v >> 32)); };
    for (int w = 0; w < TW; ++w) put64(s[w]);
    put64(plan_key_hash(s, TW) ^ kHashC);
    for (int w = 0; w < XR; ++w) put64(xw[w]);
    const size_t nzp = out.size();
    out.push_back(0);
    uint32_t nz = 0;
    for (int w = 0; w < CW; ++w)
        if (chan[w]) { out.push_back((uint32_t)w); put64(chan[w]); ++nz; }
    out[nzp] = nz;
}

// Decode alternative a's record and compare it with its source rows (s: TW sign words; chan: CW channel words; xw: XR
// extras words): every count against the record's length, every channel-word index against its range (strictly
// ascending, nonzero value), the sign words, the token and the extras words. A corrupted word fails here, never a read
// past the record.
bool TwirlRecordSampler::Impl::rot_record_check_(int a, const uint64_t* s, const uint64_t* chan, const uint64_t* xw,
                                                 std::string& why) const {
    const RotationRecords& R = rot_rec;
    const int TW = R.TW, XR = R.XR;
    if (a < 0 || (size_t)a + 1 >= R.off.size()) { why = "no offset"; return false; }
    const size_t off = R.off[(size_t)a], end = R.off[(size_t)a + 1];
    const size_t hdr = (size_t)2 * TW + 2 + (size_t)2 * XR + 1;
    if (end < off || end > R.rec.size() || end - off < hdr) { why = "record shorter than its header"; return false; }
    const uint32_t* B = &R.rec[off];
    for (int w = 0; w < TW; ++w) if (val64(B + 2 * w) != s[w]) { why = "sign words"; return false; }
    if (val64(B + 2 * TW) != (plan_key_hash(s, TW) ^ kHashC)) { why = "token"; return false; }
    for (int w = 0; w < XR; ++w) if (val64(B + 2 * TW + 2 + 2 * w) != xw[w]) { why = "extras words"; return false; }
    const uint32_t nz = B[hdr - 1];
    if (hdr + (size_t)3 * nz != end - off) { why = "count does not match the record length"; return false; }
    const uint32_t* p = B + hdr;
    static thread_local std::vector<uint64_t> d;
    d.assign((size_t)std::max(CW, 1), 0);
    for (uint32_t i = 0; i < nz; ++i, p += 3) {
        if (p[0] >= (uint32_t)CW || (i && p[0] <= *(p - 3)) || val64(p + 1) == 0) { why = "channel entry out of range, not ascending or zero"; return false; }
        d[p[0]] = val64(p + 1);
    }
    for (int w = 0; w < CW; ++w) if (d[(size_t)w] != chan[(size_t)w]) { why = "channel words"; return false; }
    return true;
}

__attribute__((noinline, cold)) void TwirlRecordSampler::Impl::rot_setup_() {
    rot_rec = RotationRecords();
    TermResidual* tr = fam_term_view_();
    if (!tr) throw std::logic_error("twirl_records: rotation setup without a term view (engine bug)");
    const bool k2 = rot.kind2;                       // the view's X-content flag (fam_terms_setup_): kept as information
    build_rotation_model(*tr, G, bare, rec_cols_, nchan, qcol_z_, qcol_x_, QCW_, rot);
    rot.kind2 = k2;
    const int A = (int)sides.size(), TW = tr->term_words();
    RotationRecords& R = rot_rec;
    R.TW = TW; R.XR = XR_; R.CW = CW;
    R.off.assign((size_t)A + 1, 0);
    R.rec.reserve((size_t)A * ((size_t)2 * TW + 2 + (size_t)2 * R.XR + 4));
    // the chunk's rows (kRotChunk × (CW + XR) words: a few MB), encoded then checked against
    const int CWn = std::max(CW, 1), XRn = std::max(XR_, 1), NWw = NW, TWw = TW;
    std::vector<uint64_t> chan((size_t)kRotChunk * (size_t)CWn, 0), xw((size_t)kRotChunk * (size_t)XRn, 0);
    int chunk_index = 0;
    bool ok = true;
    tr->for_alts(sides, kRotChunk, [&](int a0, int cnt, const uint64_t* px, const uint64_t* pz, const uint64_t* sv) {
        for (int i = 0; i < cnt; ++i) {
            rot_row_of_(px + (size_t)i * NWw, pz + (size_t)i * NWw, &chan[(size_t)i * CWn], &xw[(size_t)i * XRn]);
            R.off[(size_t)(a0 + i)] = R.rec.size();
            rot_record_of_(sv + (size_t)i * TWw, &chan[(size_t)i * CWn], &xw[(size_t)i * XRn], R.rec);
        }
        R.off[(size_t)(a0 + cnt)] = R.rec.size();
        if (g_twirl_rot_rec_test_hook) g_twirl_rot_rec_test_hook(R.rec, chunk_index);   // TEST seam (ctest only)
        for (int i = 0; i < cnt; ++i) {
            std::string why;
            if (!rot_record_check_(a0 + i, sv + (size_t)i * TWw, &chan[(size_t)i * CWn], &xw[(size_t)i * XRn], why)) {
                family_refusal_ = "twirl_records: ROT RECORD VERIFY FAIL at alt " + std::to_string(a0 + i) + ": " + why +
                                  " — the rotation record differs from its source rows (engine bug)";
                setup_error = 5; ok = false; return false;   // the bindings raise family_refusal_
            }
        }
        ++chunk_index;
        return true;
    });
    if (!ok) return;
    R.off[(size_t)A] = R.rec.size();
    R.rec.shrink_to_fit();
    rot_setup_store_();                               // lean D1-S2: the loop's builder and store (rot_on_)
}

// ══════════════════════════ lean D1-S2: the rotation shot loop's glue (lean/D1/S2/plan.md) ══════════════════════════
// The dispatch (per circuit, at construction): rot_on_ = a model was compiled and the circuit has rotations (lean D1-S3:
// any commuting axes — CH / PPR circuits included, every shot). A t = 0 circuit keeps the clean / identity-plan path.
void TwirlRecordSampler::Impl::rot_setup_store_() {
    rot_on_ = rot.built && rot.t > 0;
    if (!rot_on_) return;
    const TermResidual* tr = fam_term_view_();
    std::vector<Pauli> axes;
    for (int j = 0; j < rot.t; ++j) axes.push_back(tr->axis(j));
    rot_builder_.reset(new RotBuilder(rot, G, bare, rec_cols_, axes));
    rstore_.reset(new RotPlanStore());
    rstore_->init(rot.TW);
    rot_s_.assign((size_t)std::max(rot.TW, 1), 0);
    const int MW = std::max(1, ((int)rec_cols_.size() + 63) / 64) + 1;   // A / B words ≤ ⌈rows/64⌉ + 1
    rot_u_.assign((size_t)MW, 0);
    rot_z_.assign((size_t)MW, 0);
    rot_id_o_ = rot_identity_();                      // the s = 0 plan, at construction: the store is T-invariant
    const uint64_t* P0 = rstore_->plan(rot_id_o_);
    rot_id_skip_ = rstore_->ne(P0) == 0 && rstore_->rho(P0) == 0 && rstore_->kappa(P0) == 0;
}
// a lane's own builder and overlay store (run_threads_); the home's store is read only during a super-block
void TwirlRecordSampler::Impl::rot_lane_setup_(Impl& lane) const {
    if (!rot_on_) return;
    lane.rot_builder_.reset(new RotBuilder(rot, G, bare, rec_cols_, rot_builder_->axes()));
    lane.rstore_.reset(new RotPlanStore());
    lane.rstore_->init(rot.TW);
    lane.rot_home_ = rstore_.get();
    lane.rot_cow_.clear();
    lane.rot_id_o_ = rot_id_o_;                       // the home's s = 0 plan (a lane reads it there: home_plan)
    lane.rot_id_skip_ = rot_id_skip_;
}
// the sparse P supports: per alternative the wires of P_f's x and z parts (CSR), from the term view's chunked propagation
// (no A × NW dense table). Built once on the home thread (a recording run, the window), before any lane starts.
__attribute__((noinline, cold)) void TwirlRecordSampler::Impl::ensure_rot_plists_() {
    RotPList& L = rot_plist;
    if (L.built) return;
    if (lane_) throw std::logic_error("rot_plists: built on a lane (engine bug)");
    TermResidual* tr = fam_term_view_();
    if (!tr) throw std::logic_error("rot_plists: no term view (engine bug)");
    const int A = (int)sides.size(), NWw = NW;
    L.xq.clear(); L.zq.clear();
    L.xoff.assign((size_t)A + 1, 0); L.zoff.assign((size_t)A + 1, 0);
    tr->for_alts(sides, kRotChunk, [&](int a0, int cnt, const uint64_t* px, const uint64_t* pz, const uint64_t*) {
        for (int i = 0; i < cnt; ++i) {
            L.xoff[(size_t)(a0 + i)] = L.xq.size();
            L.zoff[(size_t)(a0 + i)] = L.zq.size();
            for (int w = 0; w < NWw; ++w) {
                for (uint64_t word = px[(size_t)i * NWw + (size_t)w]; word; word &= word - 1) L.xq.push_back((uint32_t)(w * 64 + __builtin_ctzll(word)));
                for (uint64_t word = pz[(size_t)i * NWw + (size_t)w]; word; word &= word - 1) L.zq.push_back((uint32_t)(w * 64 + __builtin_ctzll(word)));
            }
        }
        return true;
    });
    L.xoff[(size_t)A] = L.xq.size();
    L.zoff[(size_t)A] = L.zq.size();
    L.built = true;
}
// P = ⊕ P_f (mod phase) over the fired alternatives, into a buffer of n wires (cleared here)
void TwirlRecordSampler::Impl::rot_P_of_events_(const int32_t* ev, int cnt, Pauli& P) const {
    const RotPList& L = rot_plist;
    if (!L.built) throw std::logic_error("rot_P_of_events_: the P supports are not built (engine bug)");
    if (P.n != n) P = Pauli(n);
    else {
        std::fill(P.x.begin(), P.x.end(), 0ULL);
        std::fill(P.z.begin(), P.z.end(), 0ULL);
        P.phase = 0;
    }
    for (int e = 0; e < cnt; ++e) {
        const size_t a = (size_t)ev[e];
        for (size_t i = L.xoff[a]; i < L.xoff[a + 1]; ++i) P.flipx((int)L.xq[i]);
        for (size_t i = L.zoff[a]; i < L.zoff[a + 1]; ++i) P.flipz((int)L.zq[i]);
    }
}
// a miss: the key's plan built and stored (a lane: into its overlay, its ordinals above the home's)
__attribute__((noinline)) size_t TwirlRecordSampler::Impl::rot_miss_(const uint64_t* s, uint64_t h, bool lane) {
    rot_builder_->build(s, rot_build_);
    const uint64_t ord = lane ? (uint64_t)rot_home_->plans() + rstore_->plans() : rstore_->plans();
    return rstore_->insert(s, h, rot_build_, ord, lane ? rot_home_ : nullptr);
}
size_t TwirlRecordSampler::Impl::rot_identity_() {
    std::vector<uint64_t> z((size_t)std::max(rot.TW, 1), 0);
    const uint64_t h = plan_key_hash(z.data(), rot.TW);
    size_t o = rstore_->find(z.data(), h);
    if (!o) o = rot_miss_(z.data(), h, false);
    return o;
}
// the tuple trie walk of a plan: the home (in place), a lane on its own tuple (in place), a lane on a HOME tuple
// (read only; copy on write at the first node it needs — merged at the barrier)
__attribute__((noinline)) int TwirlRecordSampler::Impl::rot_walk_(const uint64_t* P, const RotPlanStore* ps, bool home_plan, bool lane) {
    const uint64_t tref = ps->tref(P);
    if (tref == RotPlanStore::kNoTuple) throw std::logic_error("rot_walk_: a κ > 0 plan without a tuple (engine bug)");
    const bool htuple = home_plan || (tref & RotPlanStore::kHomeTuple) != 0;
    const uint64_t ix = tref & ~RotPlanStore::kHomeTuple;
    const krng::Stream& born = born_at_shot_();
    trie_.root_memo_verify = in_window_;
    const long e0 = trie_.node_evals;
    int bad;
    if (!lane) {
        RotTuple& T = rstore_->tuple_mut(ix);
        bad = trie_.walk(LamSpan(T.dirs), T.trie, 0, born, rec_uz_);
        rstore_->note_grown(ix);
    } else if (htuple) {
        const RotTuple& T = rot_home_->tuple(ix);
        auto it = rot_cow_.find(ix);
        if (it != rot_cow_.end()) bad = trie_.walk(LamSpan(T.dirs), it->second, 0, born, rec_uz_);
        else {
            bad = trie_.walk_ro(LamSpan(T.dirs), T.trie, born, rec_uz_);
            if (bad == -2) {
                std::vector<BornNode>& cp = rot_cow_.emplace(ix, T.trie).first->second;
                bad = trie_.walk(LamSpan(T.dirs), cp, 0, born, rec_uz_);
            }
        }
    } else {
        RotTuple& T = rstore_->tuple_mut(ix);
        bad = trie_.walk(LamSpan(T.dirs), T.trie, 0, born, rec_uz_);
    }
    if (XTIM_UNLIKELY(trie_.node_evals != e0 && !born_node_event_)) {   // decision 7: one BORN_NODE event per run()
        born_node_event_ = true;
        diag_.push_back({TwirlRecordSampler::DiagEvent::BORN_NODE, "", (long)bare.chi(),
                         (long)ps->kappa(P), "", trie_.timed_eval_seconds});
    }
    return bad;
}
// record_report()["classes"]: per record column the NOISELESS class from the s = 0 plan (0 deterministic / 1 coin / 2 born)
std::vector<int> TwirlRecordSampler::Impl::rot_record_classes_() const {
    Impl& I = const_cast<Impl&>(*this);
    const size_t o = I.rot_identity_();
    const uint64_t* P = rstore_->plan(o);
    const int ne = rstore_->ne(P), AW = rstore_->AW(P), BW = rstore_->BW(P), es = 1 + AW + BW;
    std::vector<int> cls(rec_cols_.size(), 0);
    const uint64_t* En = P + rstore_->entries_at(P);
    for (int i = 0; i < ne; ++i, En += es) {
        const int c = RotPlanStore::ent_col(*En);
        bool coin = false, born = false;
        for (int w = 0; w < AW; ++w) coin = coin || En[1 + w] != 0;
        for (int w = 0; w < BW; ++w) born = born || En[1 + AW + w] != 0;
        cls[(size_t)c] = born ? 2 : coin ? 1 : 0;
    }
    return cls;
}

// ── test API ──
const RotationModel& TwirlRecordSampler::family_rotation_model() const { return impl_->rot; }
const RotationRecords& TwirlRecordSampler::family_rotation_records() const { return impl_->rot_rec; }
// Every record decoded against a FRESH propagation of its alternative (for_alts again): {checked, mismatches}.
std::pair<long, long> TwirlRecordSampler::family_rotation_check() const {
    const Impl& I = *impl_;
    const TermResidual* tr = I.fam_term_view_();
    if (!tr || I.rot_rec.off.empty()) return {0, 0};
    long checked = 0, bad = 0;
    std::vector<uint64_t> chan((size_t)std::max(I.CW, 1), 0), xw((size_t)std::max(I.XR_, 1), 0);
    const int NWw = I.NW, TWw = I.rot_rec.TW;
    tr->for_alts(I.sides, kRotChunk, [&](int a0, int cnt, const uint64_t* px, const uint64_t* pz, const uint64_t* sv) {
        for (int i = 0; i < cnt; ++i) {
            I.rot_row_of_(px + (size_t)i * NWw, pz + (size_t)i * NWw, chan.data(), xw.data());
            std::string why;
            ++checked;
            if (!I.rot_record_check_(a0 + i, sv + (size_t)i * TWw, chan.data(), xw.data(), why)) ++bad;
        }
        return true;
    });
    return {checked, bad};
}

}  // namespace qeccore
