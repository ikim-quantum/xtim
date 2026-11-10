// twirl_sampler_rotw.cpp — lean D1-S2: the GAUGE-MAP ORACLE of the rotation shot loop (lean/D1/S2/plan.md §3; review H2).
// On every selfcheck-window shot (and under oracle_all_shots) the shot's rotation plan is held against
//   (a) (D) — today's exact chain compose_fired → finalize → residual_normal_form → canonicalize_mod_stabilizers → the
//       record-space-B table T_D — under the COMPUTED gauge: Δ = P ⊕ P_D (P = ⊕ P_f the Clifford-only frame, P_D the
//       canonical prefix), g_c = ⟨Δ, M_c⟩, δ_j = [Λ^R_j = −Λ^D_j]: equal pivot / introducer columns, equal A / B rows,
//       directions equal as (a, b), and for EVERY column full_R ⊕ full_D == A_c·g ⊕ B_c·δ with full_X = ⟨P_X, M_c⟩ ⊕ β^X_c;
//       for every walked node p0_R(z') == p0_D(z' ⊕ δ) (bit for bit when δ_j = 0; 1 − p0_D within 2^-50 when δ_j = 1);
//   (b) (G) — build_record_table_general on the RAW s (the table the barrier's readers use) == the plan field by field;
//   (c) the hash-token linearity (the XOR of the fired tokens == plan_key_hash(s)).
// A trip is an ORACLE_FAILURE event and run() returns false. The test API's rot_gauge_of runs the same code off-line.
#include "twirl_sampler_impl.inc.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "qeccore/record_general.hpp"
#include "qeccore/term_residual.hpp"
#include <algorithm>

namespace qeccore {

namespace {
inline bool bitw_(const uint64_t* w, int i) { return (w[i >> 6] >> (i & 63)) & 1ULL; }
}  // namespace

void TwirlRecordSampler::Impl::rot_gauge_check_(const int32_t* ev, int cnt, const uint64_t* P, const RotPlanStore* ps,
                                                const uint8_t* uz, uint64_t h, bool have_h, RotGaugeCheck& out) {
    out = RotGaugeCheck();
    auto fail = [&](const std::string& why) { out.ok = false; out.why = why; };
    const int TW = rot.TW, m = (int)rec_cols_.size(), QW = QCW_;
    const RotationRecords& R = rot_rec;
    // the key and the hash from the records
    out.s.assign((size_t)std::max(TW, 1), 0);
    uint64_t hh = 0xcbf29ce484222325ull;
    for (int e = 0; e < cnt; ++e) {
        const uint32_t* B = &R.rec[R.off[(size_t)ev[e]]];
        for (int w = 0; w < TW; ++w) out.s[(size_t)w] ^= (uint64_t)B[2 * w] | ((uint64_t)B[2 * w + 1] << 32);
        hh ^= (uint64_t)B[2 * TW] | ((uint64_t)B[2 * TW + 1] << 32);
    }
    if (hh != plan_key_hash(out.s.data(), TW)) return fail("the XOR of the fired tokens differs from plan_key_hash(s)");
    if (have_h && h != hh) return fail("the loop's hash differs from the records' token XOR");
    // the plan view: the stored plan, or a fresh build
    std::vector<int> piv, intro;
    std::vector<uint64_t> A, B;
    std::vector<uint8_t> flip;
    std::vector<FreeDir> dirs;
    int rho, kap, AW, BW;
    if (P) {
        rho = ps->rho(P); kap = ps->kappa(P);
        AW = ps->AW(P); BW = ps->BW(P);
        for (int w = 0; w < TW; ++w) if (P[w] != out.s[(size_t)w]) return fail("the stored plan's key differs from the events' s");
        const int pb = ps->pivots_at(P);
        for (int k = 0; k < rho; ++k) piv.push_back(RotPlanStore::col_at(P, pb, k));
        const int ib = ps->intros_at(P);
        for (int j = 0; j < kap; ++j) intro.push_back(RotPlanStore::col_at(P, ib, j));
        A.assign((size_t)m * (size_t)std::max(AW, 1), 0); B.assign((size_t)m * (size_t)std::max(BW, 1), 0); flip.assign((size_t)m, 0);
        const int ne = ps->ne(P), es = 1 + AW + BW;
        const uint64_t* En = P + ps->entries_at(P);
        for (int i = 0; i < ne; ++i, En += es) {
            const int c = RotPlanStore::ent_col(*En);
            if (c < 0 || c >= m) return fail("a plan entry's column is out of range");
            flip[(size_t)c] = (uint8_t)RotPlanStore::ent_flip(*En);
            for (int w = 0; w < AW; ++w) A[(size_t)c * (size_t)std::max(AW, 1) + (size_t)w] = En[1 + w];
            for (int w = 0; w < BW; ++w) B[(size_t)c * (size_t)std::max(BW, 1) + (size_t)w] = En[1 + AW + w];
        }
        const uint64_t tref = ps->tref(P);
        if (kap) {
            if (tref == RotPlanStore::kNoTuple) return fail("a κ > 0 plan without a tuple");
            const bool ht = (tref & RotPlanStore::kHomeTuple) != 0;
            const uint64_t ix = tref & ~RotPlanStore::kHomeTuple;
            const RotPlanStore* ts = ht ? rot_home_ : ps;
            if (!ts || ix >= ts->tuples()) return fail("a plan's tuple index is out of range");
            dirs = ts->tuple(ix).dirs;
        }
    } else {
        RotBuild& b = rot_build_;
        rot_builder_->build(out.s.data(), b);
        rho = b.ncoin; kap = b.ndir;
        AW = (rho + 63) / 64; BW = (kap + 63) / 64;
        piv = b.pivot_cols; intro = b.intro_cols; dirs = b.dirs;
        A.assign((size_t)m * (size_t)std::max(AW, 1), 0); B.assign((size_t)m * (size_t)std::max(BW, 1), 0); flip.assign((size_t)m, 0);
        for (size_t i = 0; i < b.cols.size(); ++i) {
            const int c = b.cols[i];
            flip[(size_t)c] = b.flip[i];
            for (int w = 0; w < AW; ++w) A[(size_t)c * (size_t)std::max(AW, 1) + (size_t)w] = b.A[i * (size_t)b.AW + (size_t)w];
            for (int w = 0; w < BW; ++w) B[(size_t)c * (size_t)std::max(BW, 1) + (size_t)w] = b.B[i * (size_t)b.BW + (size_t)w];
        }
    }
    const int AWn = std::max(AW, 1), BWn = std::max(BW, 1);
    out.rho = rho; out.kappa = kap; out.piv = piv; out.intro = intro; out.A = A; out.B = B; out.AW = AW; out.BW = BW;
    out.flip = flip; out.dirs = dirs;
    if ((int)dirs.size() != kap) return fail("the plan's directions do not number κ");
    // (a) the base route's table T_D and Pauli P_D: (D) for an all-diagonal shot; a shot with a PPR-retry alternative: the
    // term route's REDUCED (P, S) (compose_terms_ → reduce_key_ → kind_of_): kind 0 → the canonical normal form (prefix ⊕ P)
    // and its record-space-B table (a pair outside the pair universe: (G) of the reduced S, route 1), kind 2 → (G) of the
    // reduced S with P. Any of them is an exact table of the same operator, so the gauge rule below holds against it.
    RecTable TD;
    Pauli PD(n);
    if (shot_has_ppr_ev_(ev, cnt)) {
        TermResidual* tr = fam_term_view_();
        if (!tr || !tr->complete() || !tr->alts_ready()) return fail("(a): no armed term view for a PPR shot");
        TermShot sh;
        tr->init_shot(sh);
        tr->compose_terms_(ev, cnt, sh);
        tr->reduce_key_(sh);
        bool general = tr->kind_of_(sh.S) != 0;
        if (!general) {
            DiagNormalForm nf0 = tr->kind0_normal_form_(Pauli(n), sh.S, G);
            if (!nf0.cz.empty()) general = true;   // a CZ part: (G) of the reduced S (the base's route 1)
            if (!general) {
                if (!build_record_table_recB(G, bare, nf0, rec_cols_, TD)) return fail("(a): kind 0: record-space B reports a fallback");
                PD = nf0.prefix;
                for (size_t w = 0; w < PD.x.size(); ++w) { PD.x[w] ^= sh.P.x[w]; PD.z[w] ^= sh.P.z[w]; }
            }
        }
        if (general) {
            build_record_table_general(G, bare, *tr, sh.S, rec_cols_, TD);
            PD = sh.P;
        }
    } else {
        fired.clear();
        for (int e = 0; e < cnt; ++e)
            for (const FiredPauli& f : sides[(size_t)ev[e]]) fired.push_back(f);
        DiagPauliClifford D = compose_fired(table, fired);
        D.finalize();
        DiagNormalForm nfD = residual_normal_form(D);
        if (!nfD.diagonal_class) return fail("(D): the residual is outside the diagonal class");
        canonicalize_mod_stabilizers_inplace(nfD, G);
        if (!build_record_table_recB(G, bare, nfD, rec_cols_, TD)) return fail("(D): record-space B reports a fallback");
        PD = nfD.prefix;
    }
    if (TD.CWX != QW) return fail("(D): table width differs from the column rows");
    // P = ⊕ P_f; Δ = P ⊕ P_D; the relabels t(P), t(P_D) and g = t(Δ) over every column
    ensure_rot_plists_();
    Pauli Pr(n);
    rot_P_of_events_(ev, cnt, Pr);
    out.Px = Pr.x; out.Pz = Pr.z; out.PDx = PD.x; out.PDz = PD.z;
    std::vector<uint64_t> tP((size_t)QW, 0), tPD((size_t)QW, 0), gcol((size_t)QW, 0);
    auto relabel = [&](const Pauli& Q, std::vector<uint64_t>& tq) {
        for (int q = 0; q < n; ++q) {
            if (Q.xbit(q)) for (int w = 0; w < QW; ++w) tq[(size_t)w] ^= qcol_x_[(size_t)q * QW + (size_t)w];
            if (Q.zbit(q)) for (int w = 0; w < QW; ++w) tq[(size_t)w] ^= qcol_z_[(size_t)q * QW + (size_t)w];
        }
    };
    relabel(Pr, tP);
    relabel(PD, tPD);
    for (int w = 0; w < QW; ++w) gcol[(size_t)w] = tP[(size_t)w] ^ tPD[(size_t)w];
    if (TD.rho != rho || TD.kappa != kap) return fail("(D): ρ / κ differ (" + std::to_string(TD.rho) + "/" + std::to_string(TD.kappa) + " vs " + std::to_string(rho) + "/" + std::to_string(kap) + ")");
    for (int k = 0; k < rho; ++k) if (TD.pivot_rec[(size_t)k] != piv[(size_t)k]) return fail("(D): pivot columns differ at slot " + std::to_string(k));
    for (int j = 0; j < kap; ++j) if (TD.born_rec[(size_t)j] != intro[(size_t)j]) return fail("(D): introducer columns differ at direction " + std::to_string(j));
    out.g.assign((size_t)rho, 0);
    for (int k = 0; k < rho; ++k) out.g[(size_t)k] = (uint8_t)bitw_(gcol.data(), piv[(size_t)k]);
    out.delta.assign((size_t)kap, 0);
    for (int j = 0; j < kap; ++j) {
        const FreeDir& x = dirs[(size_t)j];
        const FreeDir& y = TD.lam[(size_t)j];
        if (x.a != y.a || x.b != y.b) return fail("(D): direction " + std::to_string(j) + " differs as (a, b)");
        const int dph = (fd_eff_phase(x) - fd_eff_phase(y)) & 3;
        if (dph & 1) return fail("(D): direction " + std::to_string(j) + " differs by an odd phase");
        out.delta[(size_t)j] = (uint8_t)(dph == 2 ? 1 : 0);
    }
    // every column: A / B equal; full_R ⊕ full_D == A·g ⊕ B·δ
    for (int c = 0; c < m; ++c) {
        const uint64_t* Ac = &A[(size_t)c * (size_t)AWn];
        const uint64_t* Bc = &B[(size_t)c * (size_t)BWn];
        for (int k = 0; k < rho; ++k)
            if (bitw_(Ac, k) != bitw_(TD.coin_row[(size_t)k].data(), c)) return fail("(D): coin row " + std::to_string(k) + " differs at column " + std::to_string(c));
        for (int j = 0; j < kap; ++j)
            if (bitw_(Bc, j) != bitw_(TD.born_row[(size_t)j].data(), c)) return fail("(D): born row " + std::to_string(j) + " differs at column " + std::to_string(c));
        const int ref = c < nchan ? (rec_cols_[(size_t)c].ref & 1) : 0;
        const int betaR = flip[(size_t)c] ^ ref;                  // the raw base of the rotation plan
        const int fullR = bitw_(tP.data(), c) ^ betaR;
        const int fullD = bitw_(tPD.data(), c) ^ (int)bitw_(TD.base.data(), c);
        int want = 0;
        for (int k = 0; k < rho; ++k) want ^= bitw_(Ac, k) & out.g[(size_t)k];
        for (int j = 0; j < kap; ++j) want ^= bitw_(Bc, j) & out.delta[(size_t)j];
        if ((fullR ^ fullD) != want) return fail("(D): column " + std::to_string(c) + ": the base differs from the gauge rule (full_R ⊕ full_D ≠ A·g ⊕ B·δ)");
        ++out.columns;
    }
    // the walked nodes: p0_R(z') vs p0_D(z' ⊕ δ)
    if (uz && kap > 0 && P) {
        const uint64_t tref = ps->tref(P);
        // a HOME tuple: the plan's tuple marked kHomeTuple, or a home plan read by a lane (its tuple index is the home's)
        const bool ht = (tref & RotPlanStore::kHomeTuple) != 0 || (lane_ && ps == rot_home_);
        const uint64_t ix = tref & ~RotPlanStore::kHomeTuple;
        const RotPlanStore* ts = ht && lane_ ? rot_home_ : ps;
        const std::vector<BornNode>* trie = &ts->tuple(ix).trie;
        if (lane_ && ht) { auto it = rot_cow_.find(ix); if (it != rot_cow_.end()) trie = &it->second; }
        std::vector<uint8_t> zD((size_t)kap, 0);
        for (int j = 0; j < kap; ++j) zD[(size_t)j] = (uint8_t)(uz[rho + j] ^ out.delta[(size_t)j]);
        uint32_t node = 0;
        for (int j = 0; j < kap; ++j) {
            if (node >= trie->size()) return fail("the walked trie path is missing a node");
            const double pR = (*trie)[node].p0;
            if (std::isnan(pR)) return fail("a walked node is unevaluated");
            const double pD = trie_.p0_from_scratch(LamSpan(TD.lam), zD.data(), j);
            if (out.delta[(size_t)j] == 0) {
                if (std::memcmp(&pR, &pD, sizeof pR) != 0) return fail("node " + std::to_string(j) + ": p0_R differs from p0_D bit for bit");
            } else if (std::fabs(pR - (1.0 - pD)) > 0x1.0p-50) {
                return fail("node " + std::to_string(j) + ": p0_R differs from 1 - p0_D beyond 2^-50");
            }
            ++out.nodes;
            node = (*trie)[node].kid[uz[rho + j] ? 1 : 0];
            if (node == 0 && j + 1 < kap) return fail("the walked trie path ends early");
        }
    }
    // (b) (G) on the raw s == the plan
    {
        TermResidual* tr = fam_term_view_();
        if (!tr) return fail("(G): no term view");
        RecTable TG;
        build_record_table_general(G, bare, *tr, out.s, rec_cols_, TG);
        if (TG.CWX != QW || TG.rho != rho || TG.kappa != kap) return fail("(G): ρ / κ / width differ");
        for (int k = 0; k < rho; ++k) if (TG.pivot_rec[(size_t)k] != piv[(size_t)k]) return fail("(G): pivot columns differ");
        for (int j = 0; j < kap; ++j) if (TG.born_rec[(size_t)j] != intro[(size_t)j]) return fail("(G): introducer columns differ");
        for (int j = 0; j < kap; ++j) {
            const FreeDir& x = dirs[(size_t)j];
            const FreeDir& y = TG.lam[(size_t)j];
            if (x.a != y.a || x.b != y.b || fd_eff_phase(x) != fd_eff_phase(y)) return fail("(G): direction " + std::to_string(j) + " differs");
        }
        for (int c = 0; c < m; ++c) {
            const int ref = c < nchan ? (rec_cols_[(size_t)c].ref & 1) : 0;
            if ((flip[(size_t)c] ^ ref) != (int)bitw_(TG.base.data(), c)) return fail("(G): the base differs at column " + std::to_string(c));
            for (int k = 0; k < rho; ++k)
                if (bitw_(&A[(size_t)c * (size_t)AWn], k) != bitw_(TG.coin_row[(size_t)k].data(), c)) return fail("(G): coin row differs at column " + std::to_string(c));
            for (int j = 0; j < kap; ++j)
                if (bitw_(&B[(size_t)c * (size_t)BWn], j) != bitw_(TG.born_row[(size_t)j].data(), c)) return fail("(G): born row differs at column " + std::to_string(c));
        }
        out.g_table_equal = true;
    }
    out.ok = true;
}

bool TwirlRecordSampler::Impl::rot_window_check_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1, long shot,
                                                 const uint64_t* P, const RotPlanStore* ps, uint64_t h) {
    RotGaugeCheck chk;
    rot_gauge_check_(b.ev.data() + e0, (int)(e1 - e0), P, ps, rec_uz_.data(), h, true, chk);
    ++rot_window_shots_;
    if (chk.ok) return true;
    diag_.push_back({TwirlRecordSampler::DiagEvent::ORACLE_FAILURE, "rotation_gauge", shot, 0,
                     "twirl_records ROTATION GAUGE ORACLE FAIL shot=" + std::to_string(shot) + ": " + chk.why + " (engine bug)"});
    return false;
}

// ── test API ──
bool TwirlRecordSampler::family_rot_on() const { return impl_->rot_on_; }
const RotPlanStore* TwirlRecordSampler::family_rot_store() const { return impl_->rot_on_ ? impl_->rstore_.get() : nullptr; }
long TwirlRecordSampler::family_rot_window_shots() const { return impl_->rot_window_shots_; }
void TwirlRecordSampler::family_set_shot_block(bool on) { impl_->sb_off_ = !on; }   // P3 S-b
long TwirlRecordSampler::family_shot_block_shots() const { return impl_->sb_shots_; }
const NoiseModel& TwirlRecordSampler::family_noise_model() const { return impl_->noise; }                 // lean N1
bool TwirlRecordSampler::family_noise_full() const { return &impl_->run_table_() == &impl_->noise.full(); }
void TwirlRecordSampler::family_set_noise_full(bool on) { impl_->noise_full_test_ = on; }
long TwirlRecordSampler::family_rot_anti_products() const { return impl_->rot_on_ ? impl_->rot_builder_->anti_products() : 0; }
void TwirlRecordSampler::family_rot_gauge(const std::vector<int32_t>& alts, bool stored, const std::vector<uint8_t>* uz,
                                          RotGaugeCheck& out) {
    Impl& I = *impl_;
    if (!I.rot_on_) throw std::invalid_argument("rot_gauge: the sampler is not on the rotation loop (t = 0)");
    for (int32_t a : alts)
        if (a < 0 || (size_t)a >= I.sides.size()) throw std::invalid_argument("rot_gauge: alternative id out of range");
    const uint64_t* P = nullptr;
    if (stored) {
        std::vector<uint64_t> s((size_t)std::max(I.rot.TW, 1), 0);
        const RotationRecords& R = I.rot_rec;
        for (int32_t a : alts) {
            const uint32_t* B = &R.rec[R.off[(size_t)a]];
            for (int w = 0; w < I.rot.TW; ++w) s[(size_t)w] ^= (uint64_t)B[2 * w] | ((uint64_t)B[2 * w + 1] << 32);
        }
        const uint64_t h = plan_key_hash(s.data(), I.rot.TW);
        size_t o = I.rstore_->find(s.data(), h);
        if (!o) o = I.rot_miss_(s.data(), h, false);
        P = I.rstore_->plan(o);
    }
    I.rot_gauge_check_(alts.data(), (int)alts.size(), P, I.rstore_.get(), uz ? uz->data() : nullptr, 0, false, out);
}

}  // namespace qeccore
