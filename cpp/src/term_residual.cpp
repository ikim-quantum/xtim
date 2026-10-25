// term_residual — J-a (N2): the term-indexed shot residual (design.final.md §2.5). See
// qeccore/term_residual.hpp for the mathematics. Switched OFF in production: only a FAM sampler
// (xtim._xtim_testapi.compile_family_sampler) constructs a TermResidual.
//
// This file extracts no prefix sign and no (a, cz) of its own: a kind-0 composite is handed, as an
// exact DiagPauliClifford, to residual_normal_form + canonicalize_mod_stabilizers_inplace — the
// diagonal route's normal form (twirl_normal_form.cpp). Checked by tests/test_ja_term_key.py
// (nm + source structure).
#include "qeccore/term_residual.hpp"

#include <algorithm>
#include <cstring>

#include "qeccore/clifford_tableau.hpp"

namespace qeccore {

namespace {

inline bool getb(const uint64_t* w, int i) { return (w[i >> 6] >> (i & 63)) & 1ULL; }
inline void flipb(uint64_t* w, int i) { w[i >> 6] ^= 1ULL << (i & 63); }

// A Z-string axis with sign (+1 / -1).
Pauli zstring(int n, std::initializer_list<int> qs, int sign) {
    Pauli A(n);
    for (int q : qs) A.setz(q);
    A.phase = sign > 0 ? 0 : 2;
    return A;
}

// Conjugate M by a Clifford gate (M <- g M g†), exact phase.
void conj_gate(Pauli& M, GateKind g, int a, int b) {
    switch (g) {
        case GateKind::X: CliffordTableau::conj_x(M, a); break;
        case GateKind::Y: CliffordTableau::conj_x(M, a); CliffordTableau::conj_z(M, a); break;
        case GateKind::Z: CliffordTableau::conj_z(M, a); break;
        case GateKind::S: CliffordTableau::conj_s(M, a); break;
        case GateKind::SDG: CliffordTableau::conj_sdg(M, a); break;
        case GateKind::H: CliffordTableau::conj_h(M, a); break;
        case GateKind::CX: CliffordTableau::conj_cx(M, a, b); break;
        case GateKind::CZ: CliffordTableau::conj_cz(M, a, b); break;
        default: throw std::logic_error("term_residual: a non-Clifford gate among the Clifford ops (engine bug)");
    }
}

std::string words_key(char tag, const std::vector<uint64_t>& w) {
    std::string s(1, tag);
    s.append(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(uint64_t));
    return s;
}

}  // namespace

// ── TermView ─────────────────────────────────────────────────────────────────────────────────
void TermResidual::build_ops_(const Circuit& deferred) {
    loc_op_.assign(deferred.stream.size(), -1);
    size_t cur_si = 0;
    auto term = [&](Pauli A) {
        Op o;
        o.kind = 1;
        o.term = (int)local_.size();
        ops_.push_back(o);
        local_.push_back(std::move(A));
        term_si_.push_back((int)cur_si);
    };
    auto cliff = [&](GateKind g, int a, int b = -1) {
        Op o;
        o.g = g;
        o.a = a;
        o.b = b;
        ops_.push_back(o);
    };
    for (size_t si = 0; si < deferred.stream.size(); ++si) {
        cur_si = si;
        const Instr& ins = deferred.stream[si];
        switch (ins.kind) {
            case Instr::Kind::Gate: {
                const std::vector<int>& t = ins.targets;
                switch (ins.gate) {
                    case GateKind::T:                // T ∝ exp(-iπ/8 Z)
                        term(zstring(n_, {t[0]}, +1));
                        break;
                    case GateKind::CS:               // exp(iπ/8 (1 - Za - Zb + ZaZb))
                        term(zstring(n_, {t[0]}, +1));
                        term(zstring(n_, {t[1]}, +1));
                        term(zstring(n_, {t[0], t[1]}, -1));
                        break;
                    case GateKind::CCZ:              // exp(iπ/8 (1 - ΣZ + ΣZZ - ZZZ))
                        term(zstring(n_, {t[0]}, +1));
                        term(zstring(n_, {t[1]}, +1));
                        term(zstring(n_, {t[2]}, +1));
                        term(zstring(n_, {t[0], t[1]}, -1));
                        term(zstring(n_, {t[0], t[2]}, -1));
                        term(zstring(n_, {t[1], t[2]}, -1));
                        term(zstring(n_, {t[0], t[1], t[2]}, +1));
                        break;
                    case GateKind::CH: {             // (I⊗V) CZ (I⊗V†), V = S H T H S†
                        const int c = t[0], q = t[1];
                        cliff(GateKind::SDG, q); cliff(GateKind::H, q);   // V† in time order
                        term(zstring(n_, {q}, -1));
                        cliff(GateKind::H, q); cliff(GateKind::S, q);
                        cliff(GateKind::CZ, c, q);
                        cliff(GateKind::SDG, q); cliff(GateKind::H, q);   // V in time order
                        term(zstring(n_, {q}, +1));
                        cliff(GateKind::H, q); cliff(GateKind::S, q);
                        break;
                    }
                    default:
                        cliff(ins.gate, t[0], t.size() > 1 ? t[1] : -1);
                        break;
                }
                break;
            }
            case Instr::Kind::Noise:
                loc_op_[si] = (int)ops_.size();      // a fault here acts before ops_[loc_op_[si]]
                break;
            case Instr::Kind::Measure:                 // terminal reads
            case Instr::Kind::Observable:              // non-collapsing, no record
                break;
            default:
                throw std::invalid_argument(
                    "term_residual: a deferred instruction outside gate / noise / measure / observable "
                    "(the term view reads the twirl sampler's deferred stream)");
        }
    }
    T_ = (int)local_.size();
    TW_ = (T_ + 63) / 64;
}

// B_j: the local axis pushed through every later Clifford op — the gates and (J-b1 P5) the folded S-type
// rotations G = exp(-i k π/8 A): G B G† = exp(-i k π/4 A)·B = (k = 2: −i, k = 6: +i)·A·B when {A, B} = 0.
void TermResidual::push_axes_() {
    B_.assign((size_t)T_, Pauli(n_));
    std::vector<int> pos((size_t)T_, -1);
    for (size_t i = 0; i < ops_.size(); ++i)
        if (ops_[i].kind == 1) pos[(size_t)ops_[i].term] = (int)i;
    for (int j = 0; j < T_; ++j) {
        if (pos[(size_t)j] < 0) continue;            // a merged (removed) term
        Pauli B = local_[(size_t)j];
        for (size_t i = (size_t)pos[(size_t)j] + 1; i < ops_.size(); ++i) {
            const Op& o = ops_[i];
            if (o.kind == 0) conj_gate(B, o.g, o.a, o.b);
#ifdef XTIM_MUTATE_NC_NOREPUSH
            else if (false) {                        // mutation build only (MUT-NC-NOREPUSH): folds not pushed through
#else
            else if (o.kind == 2) {
#endif
                const Fold& F = folds_[(size_t)o.term];
                if (Pauli::anticommute_bit(F.A, B)) {
                    B = Pauli::multiply(F.A, B);
                    B.phase = (B.phase + (F.k == 2 ? 3 : 1)) & 3;   // × (−i) or × (+i)
                }
            }
        }
        B_[(size_t)j] = std::move(B);
    }
}

namespace {
bool same_unsigned(const Pauli& a, const Pauli& b) { return a.x == b.x && a.z == b.z; }
}  // namespace

// J-b1 P5: the static fixed-axis merge (header). One pass scans j in time order; the partner is the LATEST earlier
// live term with the same pushed axis (greedy: anything blocking it blocks every earlier one); a fold ends the pass
// (convention (A): every axis and alternative is recomputed — one rebuild per pass, ruling R3-6); passes repeat
// until one makes no merge.
void TermResidual::merge_(const std::vector<std::vector<FiredPauli>>& sides, const std::vector<char>& live) {
    if (!live.empty() && live.size() != sides.size())
        throw std::logic_error("term_residual: the live mask and the alternatives differ in size (engine bug)");
    mstats_.terms_before = T_;
    std::vector<int> alt_pos(sides.size(), -1);
    std::vector<Inj> inj;
    for (size_t a = 0; a < sides.size(); ++a)
        for (const FiredPauli& f : sides[a]) {
            if (f.location_index < 0 || f.location_index >= (int)loc_op_.size() ||
                loc_op_[(size_t)f.location_index] < 0)
                throw std::logic_error("term_residual: an alternative's location is not a deferred noise site "
                                       "(engine bug)");
            alt_pos[a] = loc_op_[(size_t)f.location_index];
            inj.push_back({alt_pos[a], (int)a, f.qubit, f.pauli});
        }
    const int A = (int)sides.size();
    bool any = true;
    while (any) {
        any = false;
        ++mstats_.passes;
        std::vector<uint64_t> px, pz, sv;
        propagate_(inj, A, px, pz, sv);
        std::vector<int> pos((size_t)T_, -1);
        for (size_t i = 0; i < ops_.size(); ++i)
            if (ops_[i].kind == 1) pos[(size_t)ops_[i].term] = (int)i;
        bool folded = false;
        for (int j = 0; j < T_ && !folded; ++j) {
            if (pos[(size_t)j] < 0) continue;
            int partner = -1;
            for (int i = j - 1; i >= 0; --i) {
                if (pos[(size_t)i] < 0) continue;
                if (same_unsigned(B_[(size_t)i], B_[(size_t)j])) { partner = i; break; }
                if (Pauli::anticommute_bit(B_[(size_t)i], B_[(size_t)j])) break;   // (a) blocks
            }
            if (partner < 0) continue;
            const int i = partner, pi = pos[(size_t)i], pj = pos[(size_t)j];
            int blocker = -1;                        // (b): a fault between them that flips the later term only
            for (int a = 0; a < A && blocker < 0; ++a)
#if defined(XTIM_MUTATE_NC_MERGE)
                (void)a;                             // mutation build only (MUT-NC-MERGE): faults between ignored
#elif defined(XTIM_MUTATE_NC_ORDER)
                if (alt_pos[(size_t)a] > pi && alt_pos[(size_t)a] <= pj)   // mutation build only (MUT-NC-ORDER):
                    blocker = a;                     // ANY noise site between cuts the stretch
#else
                if (alt_pos[(size_t)a] > pi && alt_pos[(size_t)a] <= pj && (live.empty() || live[(size_t)a]) &&
                    getb(&sv[(size_t)a * TW_], j))
                    blocker = a;
#endif
            if (blocker >= 0) {
                if (blocked_.empty()) {
                    const FiredPauli& f = sides[(size_t)blocker].front();
                    blocked_gi_ = term_si_[(size_t)i];
                    blocked_gj_ = term_si_[(size_t)j];
                    blocked_site_ = f.location_index;
                    blocked_ = "terms " + std::to_string(i) + " (deferred gate " + std::to_string(term_si_[(size_t)i]) +
                               ") and " + std::to_string(j) + " (deferred gate " + std::to_string(term_si_[(size_t)j]) +
                               ") merge only when no fault occurs at deferred noise site " +
                               std::to_string(f.location_index);
                }
                continue;
            }
#ifdef XTIM_MUTATE_NC_NOFOLD
            if (B_[(size_t)i].phase == B_[(size_t)j].phase) continue;   // mutation build only: S remainders kept
#endif
            ++mstats_.merges;
            any = true;
            ops_[(size_t)pj].kind = 3;               // the later factor joins the earlier one
            const bool same_sign = B_[(size_t)i].phase == B_[(size_t)j].phase;
            if (same_sign) {                         // exp(-iπ/8 A)² = exp(-iπ/4 A): an S-type Clifford, FOLDED
                folds_.push_back({local_[(size_t)i], 2});
                ops_[(size_t)pi].kind = 2;
                ops_[(size_t)pi].term = (int)folds_.size() - 1;
                ++mstats_.folds;
                folded = true;                       // the earlier axes change: end the pass, recompute
            } else {                                 // opposite signs cancel
                ops_[(size_t)pi].kind = 3;
                ++mstats_.cancels;
            }
            pos[(size_t)i] = pos[(size_t)j] = -1;
        }
#ifndef XTIM_MUTATE_NC_NOREPUSH
        if (folded) push_axes_();
#endif                                               // mutation build only (MUT-NC-NOREPUSH): no recompute
#ifdef XTIM_MUTATE_NC_SINGLEPASS
        any = false;                                 // mutation build only (MUT-NC-SINGLEPASS): one pass
#endif
    }
    // compact the live terms (time order kept)
    std::vector<int> remap((size_t)T_, -1);
    std::vector<Pauli> L;
    std::vector<int> si;
    for (size_t o = 0; o < ops_.size(); ++o)
        if (ops_[o].kind == 1) {
            const int t = ops_[o].term;
            remap[(size_t)t] = (int)L.size();
            L.push_back(local_[(size_t)t]);
            si.push_back(term_si_[(size_t)t]);
            ops_[o].term = remap[(size_t)t];
        }
    local_ = std::move(L);
    term_si_ = std::move(si);
    T_ = (int)local_.size();
    TW_ = (T_ + 63) / 64;
    push_axes_();
}

void TermResidual::finish_(const std::vector<std::pair<int, int>>& reads) {
    check_commuting_();
    term_rows_.assign((size_t)T_, {});
    for (int j = 0; j < T_; ++j) {
        const Pauli& B = B_[(size_t)j];
        for (size_t r = 0; r < reads.size(); ++r) {
            const int basis = reads[r].first, w = reads[r].second;
            const bool x = B.xbit(w), z = B.zbit(w);
            const bool anti = basis == 0 ? z : basis == 2 ? x : (x != z);
            if (anti) term_rows_[(size_t)j].push_back((int)r);
        }
    }
    build_kernel_();
}

TermResidual::TermResidual(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads)
    : n_(deferred.n), NW_((deferred.n + 63) / 64) {
    build_ops_(deferred);
    push_axes_();
    finish_(reads);
}

TermResidual::TermResidual(const Circuit& deferred, const std::vector<std::pair<int, int>>& reads,
                           const std::vector<std::vector<FiredPauli>>& sides, const std::vector<char>& live,
                           bool z_screen)
    : n_(deferred.n), NW_((deferred.n + 63) / 64) {
    build_ops_(deferred);
    push_axes_();
    if (z_screen) {
        bool x = false;
        for (int j = 0; j < T_ && !x; ++j)
            for (uint64_t w : B_[(size_t)j].x) if (w) { x = true; break; }
        if (!x) { z_only_ = true; return; }
    }
    bool anti = false;                               // the merge is vacuous on a commuting set (exact screen)
    for (int j = 0; j < T_ && !anti; ++j)
        for (int k = j + 1; k < T_ && !anti; ++k)
            anti = Pauli::anticommute_bit(B_[(size_t)j], B_[(size_t)k]);
#ifdef XTIM_MUTATE_NC_NOMERGE
    anti = false;                                    // mutation build only (MUT-NC-NOMERGE): r1's unmerged check
#endif
    if (anti) merge_(sides, live);
    finish_(reads);
    set_alts(sides);
}

TermResidual::TermResidual(int n, const std::vector<Pauli>& axes)
    : n_(n), NW_((n + 63) / 64), B_(axes) {
    T_ = (int)B_.size();
    TW_ = (T_ + 63) / 64;
    for (const Pauli& B : B_)
        if (B.n != n) throw std::invalid_argument("term_residual: axis width differs from n");
    check_commuting_();
    term_rows_.assign((size_t)T_, {});
    build_kernel_();
}

// THE term-class check (one site, both constructors): the term-indexed residual covers commuting term
// families only (design §2.5); an anticommuting pair is a class boundary, not an engine bug.
void TermResidual::check_commuting_() const {
#ifdef XTIM_MUTATE_TC_ALWAYS
    throw TermClassRefusal("term_residual: TERM CLASS REFUSAL (mutation build: every term set refused)");
#endif
#ifndef XTIM_MUTATE_TC_SKIP
    for (int j = 0; j < T_; ++j)
        for (int k = j + 1; k < T_; ++k)
            if (Pauli::anticommute_bit(B_[(size_t)j], B_[(size_t)k])) {
                TermClassRefusal e(
                    "term_residual: TERM CLASS REFUSAL: term axes " + std::to_string(j) + " and " +
                    std::to_string(k) + " anticommute (the term-indexed residual, design §2.5, covers "
                    "commuting term families only)" + (blocked_.empty() ? std::string() : "; " + blocked_));
                e.axes = {B_[(size_t)j], B_[(size_t)k]};
                e.gate_i = blocked_gi_;
                e.gate_j = blocked_gj_;
                e.site = blocked_site_;
                throw e;
            }
#endif   // mutation build only (MUT-TC-SKIP): the check skipped
}

// ── per-alternative (P_f, S_f): one bit-sliced forward pass over all alternatives ────────────────
void TermResidual::set_alts(const std::vector<std::vector<FiredPauli>>& sides) {
    A_ = (int)sides.size();
    std::vector<Inj> inj;
    for (int a = 0; a < A_; ++a)
        for (const FiredPauli& f : sides[(size_t)a]) {
            if (f.location_index < 0 || f.location_index >= (int)loc_op_.size() ||
                loc_op_[(size_t)f.location_index] < 0)
                throw std::logic_error("term_residual: an alternative's location is not a deferred noise site "
                                       "(engine bug)");
            if (f.qubit < 0 || f.qubit >= n_) throw std::logic_error("term_residual: fault qubit out of range");
            inj.push_back({loc_op_[(size_t)f.location_index], a, f.qubit, f.pauli});
        }
    propagate_(inj, A_, apx_, apz_, as_);
    alts_ready_ = true;
}

// J-a (N4b): an input frame component (a Pauli on a port wire at t = 0, before the first op) as an atom
// (P_F, S_F), by THE propagation of the fault atoms. Clean components come out with P_F = the anchor image
// (S_F folding to the kernel); poisoned ones with the terms they cross.
void TermResidual::port_atoms(const std::vector<std::pair<int, PauliBasis>>& comps, std::vector<uint64_t>& px,
                              std::vector<uint64_t>& pz, std::vector<uint64_t>& s) const {
    std::vector<Inj> inj;
    for (size_t c = 0; c < comps.size(); ++c) {
        if (comps[c].first < 0 || comps[c].first >= n_) throw std::logic_error("term_residual: port qubit out of range");
        inj.push_back({0, (int)c, comps[c].first, comps[c].second});
    }
    propagate_(inj, (int)comps.size(), px, pz, s);
}

// Every injected Pauli (op position pos = before ops_[pos]; mod phase) pushed to the end of the stream: the
// Clifford gates conjugate it, each term factor it crosses with anticommutation joins its S. Bit-sliced over
// the A atoms; output (A × NW) X / Z words and (A × TW) term words.
void TermResidual::propagate_(std::vector<Inj> inj, int A, std::vector<uint64_t>& px, std::vector<uint64_t>& pz,
                              std::vector<uint64_t>& sv) const {
    const int AW = (A + 63) / 64;
    std::stable_sort(inj.begin(), inj.end(), [](const Inj& u, const Inj& v) { return u.pos < v.pos; });
    std::vector<uint64_t> X((size_t)n_ * AW, 0), Z((size_t)n_ * AW, 0), Sb((size_t)T_ * AW, 0);
    auto xr = [&](int q) { return X.data() + (size_t)q * AW; };
    auto zr = [&](int q) { return Z.data() + (size_t)q * AW; };   // data(): A = 0 leaves X / Z empty (UBSan)
    size_t ii = 0;
    for (size_t i = 0; i <= ops_.size(); ++i) {
        for (; ii < inj.size() && inj[ii].pos == (int)i; ++ii) {
            const Inj& e = inj[ii];
            if (e.b == PauliBasis::X || e.b == PauliBasis::Y) flipb(xr(e.q), e.alt);
            if (e.b == PauliBasis::Z || e.b == PauliBasis::Y) flipb(zr(e.q), e.alt);
        }
        if (i == ops_.size()) break;
        const Op& o = ops_[i];
        if (o.kind == 3) continue;               // J-b1 P5: a merged (removed) factor
#ifdef XTIM_MUTATE_NC_NOREPUSH
        if (o.kind == 2) continue;               // mutation build only (MUT-NC-NOREPUSH): faults skip the fold
#endif
        if (o.kind == 2) {                       // J-b1 P5: a folded S-type rotation about a Z-string A:
            const Pauli& Af = folds_[(size_t)o.term].A;   // P → P·A (mod phase) iff {A, P} = 0
            std::vector<uint64_t> anti((size_t)AW, 0);
            for (int q = 0; q < n_; ++q)
                if (Af.zbit(q)) { const uint64_t* x = xr(q); for (int w = 0; w < AW; ++w) anti[(size_t)w] ^= x[w]; }
            for (int q = 0; q < n_; ++q)
                if (Af.zbit(q)) { uint64_t* z = zr(q); for (int w = 0; w < AW; ++w) z[w] ^= anti[(size_t)w]; }
            continue;
        }
        if (o.kind == 1) {                       // anticommutation with the (Z-string) local axis
            const Pauli& Aj = local_[(size_t)o.term];
            uint64_t* srow = Sb.data() + (size_t)o.term * AW;
            for (int q = 0; q < n_; ++q) {
                if (Aj.xbit(q)) throw std::logic_error("term_residual: a local term axis with X support (engine bug)");
                if (Aj.zbit(q)) { const uint64_t* x = xr(q); for (int w = 0; w < AW; ++w) srow[w] ^= x[w]; }
            }
            continue;
        }
        switch (o.g) {
            case GateKind::X: case GateKind::Y: case GateKind::Z: break;   // mod phase: no change
            case GateKind::H: std::swap_ranges(xr(o.a), xr(o.a) + AW, zr(o.a)); break;
            case GateKind::S: case GateKind::SDG: {
                uint64_t* z = zr(o.a); const uint64_t* x = xr(o.a);
                for (int w = 0; w < AW; ++w) z[w] ^= x[w];
                break;
            }
            case GateKind::CX: {
                uint64_t* xt = xr(o.b); const uint64_t* xc = xr(o.a);
                uint64_t* zc = zr(o.a); const uint64_t* zt = zr(o.b);
                for (int w = 0; w < AW; ++w) { xt[w] ^= xc[w]; zc[w] ^= zt[w]; }
                break;
            }
            case GateKind::CZ: {
                uint64_t* za = zr(o.a); uint64_t* zb = zr(o.b);
                const uint64_t* xa = xr(o.a); const uint64_t* xb = xr(o.b);
                for (int w = 0; w < AW; ++w) { za[w] ^= xb[w]; zb[w] ^= xa[w]; }
                break;
            }
            default: throw std::logic_error("term_residual: a non-Clifford gate among the Clifford ops (engine bug)");
        }
    }
    px.assign((size_t)A * NW_, 0);
    pz.assign((size_t)A * NW_, 0);
    sv.assign((size_t)A * TW_, 0);
    for (int q = 0; q < n_; ++q)
        for (int w = 0; w < AW; ++w) {
            uint64_t bx = xr(q)[w], bz = zr(q)[w];
            while (bx) { const int a = w * 64 + __builtin_ctzll(bx); bx &= bx - 1; flipb(&px[(size_t)a * NW_], q); }
            while (bz) { const int a = w * 64 + __builtin_ctzll(bz); bz &= bz - 1; flipb(&pz[(size_t)a * NW_], q); }
        }
    for (int j = 0; j < T_; ++j)
        for (int w = 0; w < AW; ++w) {
            uint64_t bs = Sb[(size_t)j * AW + w];
            while (bs) { const int a = w * 64 + __builtin_ctzll(bs); bs &= bs - 1; flipb(&sv[(size_t)a * TW_], j); }
        }
}

// ── composition ──────────────────────────────────────────────────────────────────────────────
void TermResidual::init_shot(TermShot& sh) const {
    sh.P = Pauli(n_);
    sh.N = Pauli(n_);
    sh.S.assign((size_t)TW_, 0);
    sh.C.assign((size_t)TW_, 0);
}

void TermResidual::clear_shot(TermShot& sh) const {
    std::fill(sh.P.x.begin(), sh.P.x.end(), 0ULL);
    std::fill(sh.P.z.begin(), sh.P.z.end(), 0ULL);
    sh.P.phase = 0;
    std::fill(sh.S.begin(), sh.S.end(), 0ULL);
}

void TermResidual::compose_atom_(const uint64_t* px, const uint64_t* pz, const uint64_t* s, TermShot& sh) const {
    // Corrections, decided against the PRE-atom (P, S): rot(S_f) P rot(S_f)† = P·Π_{j∈S_f,{B_j,P}=0}(iB_j),
    // rot(S_f)·rot(S) = rot(S_f ⊕ S)·Π_{j∈S_f∩S}(iB_j).
    for (int w = 0; w < TW_; ++w) {
        uint64_t word = s[w], c = 0;
        while (word) {
            const int b = __builtin_ctzll(word);
            word &= word - 1;
            const int j = w * 64 + b;
            bool corr = getb(sh.S.data(), j);
#ifndef XTIM_MUTATE_TK_PCORR
            corr ^= (bool)Pauli::anticommute_bit(B_[(size_t)j], sh.P);
#endif
            if (corr) c |= 1ULL << b;
        }
        sh.C[(size_t)w] = c;
    }
    for (int w = 0; w < NW_; ++w) { sh.P.x[(size_t)w] ^= px[w]; sh.P.z[(size_t)w] ^= pz[w]; }
    for (int w = 0; w < TW_; ++w) {
        uint64_t c = sh.C[(size_t)w];
        while (c) {
            const int j = w * 64 + __builtin_ctzll(c);
            c &= c - 1;
            const Pauli& B = B_[(size_t)j];
            for (int v = 0; v < NW_; ++v) { sh.P.x[(size_t)v] ^= B.x[(size_t)v]; sh.P.z[(size_t)v] ^= B.z[(size_t)v]; }
        }
        sh.S[(size_t)w] ^= s[w];
    }
}

void TermResidual::compose_terms_(const int32_t* alts, int count, TermShot& sh) const {
    clear_shot(sh);
    for (int e = 0; e < count; ++e) {
        const int a = alts[e];
        compose_atom_(alt_px(a), alt_pz(a), alt_s(a), sh);
    }
}

// ── KerPhi ───────────────────────────────────────────────────────────────────────────────────
Pauli TermResidual::rot_image(const std::vector<uint64_t>& S, const Pauli& V) const {
    Pauli R = V;
    for (int w = 0; w < TW_ && w < (int)S.size(); ++w) {
        uint64_t word = S[(size_t)w];
        while (word) {
            const int j = w * 64 + __builtin_ctzll(word);
            word &= word - 1;
            if (Pauli::anticommute_bit(B_[(size_t)j], V)) {   // rot(B) V rot(B)† = iB·V  ({V,B} = 0)
                Pauli iB = B_[(size_t)j];
                iB.phase = (iB.phase + 1) & 3;
                R = Pauli::multiply(iB, R);
            }
        }
    }
    return R;
}

void TermResidual::build_kernel_() {
    const int VW = 2 * NW_;                       // b_j as (x | z) words
    auto vec_of = [&](const Pauli& B) {
        std::vector<uint64_t> v((size_t)VW, 0);
        for (int w = 0; w < NW_; ++w) { v[(size_t)w] = B.x[(size_t)w]; v[(size_t)(NW_ + w)] = B.z[(size_t)w]; }
        return v;
    };
    // A fully reduced basis of span{b_j}: coordinates of b_j = its bits at the pivot columns.
    std::vector<std::vector<uint64_t>> rows;
    std::vector<int> piv;
    for (int j = 0; j < T_; ++j) {
        std::vector<uint64_t> v = vec_of(B_[(size_t)j]);
        for (size_t r = 0; r < rows.size(); ++r)
            if (getb(v.data(), piv[r])) for (int w = 0; w < VW; ++w) v[(size_t)w] ^= rows[r][(size_t)w];
        int p = -1;
        for (int w = 0; w < VW && p < 0; ++w) if (v[(size_t)w]) p = w * 64 + __builtin_ctzll(v[(size_t)w]);
        if (p < 0) continue;
        for (size_t r = 0; r < rows.size(); ++r)
            if (getb(rows[r].data(), p)) for (int w = 0; w < VW; ++w) rows[r][(size_t)w] ^= v[(size_t)w];
        rows.push_back(std::move(v));
        piv.push_back(p);
    }
    const int r = (int)rows.size();
    // Φ(e_j) ↔ the symmetric r×r matrix c_j c_jᵀ (injective on span{b_j}): feature bits (i ≤ k).
    const int F = r * (r + 1) / 2, FW = (F + 63) / 64;
    auto tri = [&](int i, int k) { return i * r - i * (i - 1) / 2 + (k - i); };
    std::vector<std::vector<uint64_t>> frows, fcomb;
    std::vector<int> fpiv;
    std::vector<uint8_t> is_pivot((size_t)T_, 0);
    for (int j = 0; j < T_; ++j) {
        const std::vector<uint64_t> bj = vec_of(B_[(size_t)j]);
        std::vector<int> cj;
        for (int i = 0; i < r; ++i) if (getb(bj.data(), piv[(size_t)i])) cj.push_back(i);
        std::vector<uint64_t> f((size_t)FW, 0), comb((size_t)TW_, 0);
        for (size_t u = 0; u < cj.size(); ++u)
            for (size_t v = u; v < cj.size(); ++v) flipb(f.data(), tri(cj[u], cj[v]));
        flipb(comb.data(), j);
        for (size_t t = 0; t < frows.size(); ++t)
            if (getb(f.data(), fpiv[t])) {
                for (int w = 0; w < FW; ++w) f[(size_t)w] ^= frows[t][(size_t)w];
                for (int w = 0; w < TW_; ++w) comb[(size_t)w] ^= fcomb[t][(size_t)w];
            }
        int p = -1;
        for (int w = 0; w < FW && p < 0; ++w) if (f[(size_t)w]) p = w * 64 + __builtin_ctzll(f[(size_t)w]);
        if (p >= 0) {
            frows.push_back(std::move(f));
            fcomb.push_back(std::move(comb));
            fpiv.push_back(p);
            is_pivot[(size_t)j] = 1;
            pivot_terms_.push_back(j);
            continue;
        }
        // Φ(comb) = 0: comb = {j} ∪ (pivot terms). N_k = rot(k), read off its image signs.
        KernelVec kv;
        kv.j = j;
        kv.k = std::move(comb);
        kv.N = Pauli(n_);
        for (int q = 0; q < n_; ++q)
            for (int zx = 0; zx < 2; ++zx) {
                Pauli V(n_);
                if (zx) V.setz(q); else V.setx(q);
                const Pauli I = rot_image(kv.k, V);
                if (I.x != V.x || I.z != V.z || (I.phase & 1))
                    throw std::logic_error("term_residual: a kernel vector of Phi whose rotation is not a Pauli "
                                           "(engine bug)");
                if (I.phase == 2) { if (zx) kv.N.setx(q); else kv.N.setz(q); }
            }
        // termroute-a F5: N_k = rot(k) is a Pauli in the group the (pairwise commuting) term axes generate, so it
        // commutes with every B_j — fold_kernel_ relies on it (checked once here, not per shot).
        for (int t = 0; t < T_; ++t)
            if (Pauli::anticommute_bit(B_[(size_t)t], kv.N))
                throw std::logic_error("term_residual: a kernel factor N_k anticommutes with a term axis (engine bug)");
        kern_.push_back(std::move(kv));
    }
}

void TermResidual::fold_kernel_(int i, TermShot& sh) const {
    const KernelVec& kv = kern_[(size_t)i];
    // N′ = N_k · Π_{j∈S∩k} B_j  (S = the CURRENT set, adv-A kfold.py's sequential form)
#ifdef XTIM_MUTATE_TK_FOLD
    const std::vector<uint64_t>& Sf = sh.C;      // mutation build only: the ORIGINAL S (copied by the caller)
#else
    const std::vector<uint64_t>& Sf = sh.S;
#endif
    sh.N.x = kv.N.x;
    sh.N.z = kv.N.z;
    sh.N.phase = 0;
    for (int w = 0; w < TW_; ++w) {
        uint64_t word = Sf[(size_t)w] & kv.k[(size_t)w];
        while (word) {
            const int j = w * 64 + __builtin_ctzll(word);
            word &= word - 1;
            for (int v = 0; v < NW_; ++v) { sh.N.x[(size_t)v] ^= B_[(size_t)j].x[(size_t)v]; sh.N.z[(size_t)v] ^= B_[(size_t)j].z[(size_t)v]; }
        }
    }
    // rot(S) = rot(S △ k)·N′: move N′ left, rot(S′) N′ = N′·Π_{j∈S′,{B_j,N′}=0} B_j · rot(S′) (mod phase).
    for (int w = 0; w < TW_; ++w) sh.S[(size_t)w] ^= kv.k[(size_t)w];
    for (int v = 0; v < NW_; ++v) { sh.P.x[(size_t)v] ^= sh.N.x[(size_t)v]; sh.P.z[(size_t)v] ^= sh.N.z[(size_t)v]; }
    // termroute-a F5: rot(S′) N′ rot(S′)† = N′ — N′ = N_k · Π_{S∩k} B_j commutes with every term axis (the axes commute
    // pairwise, check_commuting_; N_k commutes with them, build_kernel_), so the old per-term dressing loop over S′
    // never fired and is gone.
}

void TermResidual::fold_sequence_(const std::vector<int>& idx, TermShot& sh) const {
    std::copy(sh.S.begin(), sh.S.end(), sh.C.begin());   // the original S (the TK_FOLD mutation reads it)
    for (int i : idx) {
        if (i < 0 || i >= (int)kern_.size()) throw std::invalid_argument("term_residual: kernel index out of range");
        fold_kernel_(i, sh);
    }
}

void TermResidual::reduce_key_(TermShot& sh) const {
#ifdef XTIM_MUTATE_TK_CANON
    (void)sh;                                     // mutation build only: no reduction mod ker Φ
    return;
#else
    std::copy(sh.S.begin(), sh.S.end(), sh.C.begin());   // the original S (the TK_FOLD mutation reads it)
    for (size_t i = 0; i < kern_.size(); ++i)
        if (getb(sh.S.data(), kern_[i].j)) fold_kernel_((int)i, sh);
#endif
}

std::vector<uint64_t> TermResidual::key_of(const std::vector<uint64_t>& Sred) const {
#ifdef XTIM_MUTATE_TK_CANON
    return Sred;                                  // mutation build only: the raw term set is the key
#else
    std::vector<uint64_t> key((size_t)((rank() + 63) / 64), 0);
    for (int t = 0; t < rank(); ++t)
        if (getb(Sred.data(), pivot_terms_[(size_t)t])) flipb(key.data(), t);
    return key;
#endif
}

// ── kind ─────────────────────────────────────────────────────────────────────────────────────
int TermResidual::kind_of_(const std::vector<uint64_t>& Sred) {
    const std::string key = words_key('S', Sred);
    auto it = kind_cache_.find(key);
    if (it != kind_cache_.end()) return it->second.kind;
    KindEntry e;
    // Diagonal ⇔ rot(S) fixes every Z_q up to sign.
    // termroute-a F4: the support of rot(S) Z_q rot(S)† is Z_q ⊕ Σ_{j∈S, x_q(B_j)=1} B_j (rot_image), so rot(S) is
    // diagonal ⟺ for every q, ⊕_{j∈S, x_q(B_j)=1} (B_j.x | B_j.z) = 0. One pass over S into a per-object scratch
    // row per touched q; the exact image loop below runs only for the diagonal keys (it builds D).
    if (kscr_.empty()) { kscr_.assign((size_t)n_ * 2 * (size_t)NW_, 0); kmark_.assign((size_t)n_, 0); }
    bool fast_diag = true;
    for (int w = 0; w < TW_; ++w)
        for (uint64_t word = Sred[(size_t)w]; word; word &= word - 1) {
            const Pauli& B = B_[(size_t)(w * 64 + __builtin_ctzll(word))];
            for (int v = 0; v < NW_; ++v)
                for (uint64_t xs = B.x[(size_t)v]; xs; xs &= xs - 1) {
                    const int q = v * 64 + __builtin_ctzll(xs);
                    uint64_t* r = &kscr_[(size_t)q * 2 * (size_t)NW_];
                    if (!kmark_[(size_t)q]) { kmark_[(size_t)q] = 1; ktouch_.push_back(q); }
                    for (int u = 0; u < NW_; ++u) r[u] ^= B.x[(size_t)u];
#ifndef XTIM_MUTATE_F4_XONLY
                    for (int u = 0; u < NW_; ++u) r[NW_ + u] ^= B.z[(size_t)u];
#endif   // mutation build only (MUT-F4-XONLY): the z movement ignored
                }
        }
    for (int q : ktouch_) {
        uint64_t* r = &kscr_[(size_t)q * 2 * (size_t)NW_];
        for (int u = 0; u < 2 * NW_; ++u) { if (r[u]) fast_diag = false; r[u] = 0; }
        kmark_[(size_t)q] = 0;
    }
    ktouch_.clear();
    if (!fast_diag) {
        const int k = e.kind;
        kind_cache_.emplace(key, std::move(e));
        return k;
    }
    std::vector<Pauli> Zim((size_t)n_);
    bool diag = true;
    for (int q = 0; q < n_ && diag; ++q) {
        Pauli V(n_);
        V.setz(q);
        Zim[(size_t)q] = rot_image(Sred, V);
        if (Zim[(size_t)q].x != V.x || Zim[(size_t)q].z != V.z) diag = false;
    }
    if (diag) {
        // rot(S) = D0 · Pd exactly (mod global phase): D0 = Π S_q^{M_qq} Π CZ_{ql}^{M_ql} with X_q ↦ ±X_q Z^{M_q}
        // (the symplectic part), Pd the Pauli whose anticommutation pattern is the image-sign
        // difference (rot(S) V rot(S)† = (−1)^{⟨Pd,V⟩} D0 V D0†).
        DiagPauliClifford D0 = DiagPauliClifford::identity(n_);
        DiagPauliClifford Pd = DiagPauliClifford::identity(n_);
        std::vector<Pauli> Xim((size_t)n_);
        for (int q = 0; q < n_; ++q) {
            Pauli V(n_);
            V.setx(q);
            Xim[(size_t)q] = rot_image(Sred, V);
            for (int w = 0; w < NW_; ++w)
                if (Xim[(size_t)q].x[(size_t)w] != V.x[(size_t)w])
                    throw std::logic_error("term_residual: a Z-fixing term rotation moves an X support (engine bug)");
        }
        for (int q = 0; q < n_; ++q) {
            if (Xim[(size_t)q].zbit(q)) D0.a[(size_t)q] = 1;
            for (int l = q + 1; l < n_; ++l) {
                if (Xim[(size_t)q].zbit(l) != Xim[(size_t)l].zbit(q))
                    throw std::logic_error("term_residual: an asymmetric diagonal action (engine bug)");
                if (Xim[(size_t)q].zbit(l)) D0.B.flip_sym(q, l);
            }
        }
        for (int q = 0; q < n_; ++q) {
            Pauli V(n_);
            V.setx(q);
            if (D0.a[(size_t)q]) CliffordTableau::conj_s(V, q);
            for (int l = 0; l < n_; ++l)
                if (l != q && Xim[(size_t)q].zbit(l)) CliffordTableau::conj_cz(V, q, l);
            if (V.x != Xim[(size_t)q].x || V.z != Xim[(size_t)q].z || ((Xim[(size_t)q].phase - V.phase) & 1))
                throw std::logic_error("term_residual: diagonal image mismatch (engine bug)");
            if (((Xim[(size_t)q].phase - V.phase) & 3) == 2) Pd.a[(size_t)q] = 2;   // Pd ∋ Z_q
            if (Zim[(size_t)q].phase == 2) Pd.v[(size_t)q] = 1;                     // Pd ∋ X_q
            else if (Zim[(size_t)q].phase != 0)
                throw std::logic_error("term_residual: a non-Hermitian Z image (engine bug)");
        }
        e.kind = 0;
        e.D = Pd.then(D0);                          // D0 ∘ Pd = D0 · Pd
    }
    const int k = e.kind;
    kind_cache_.emplace(key, std::move(e));
    return k;
}

size_t TermResidual::kind_bytes_(const std::string& sbytes) const {
    auto it = kind_cache_.find('S' + sbytes);
    if (it == kind_cache_.end()) return 0;
    const DiagPauliClifford& D = it->second.D;
    return it->first.capacity() + sizeof(KindEntry) + 64 + D.a.capacity() + D.v.capacity() + D.B.data_.capacity() * 8;
}
void TermResidual::kind_forget_(const std::string& sbytes) { kind_cache_.erase('S' + sbytes); }

DiagNormalForm TermResidual::kind0_normal_form_(const Pauli& P, const std::vector<uint64_t>& Sred,
                                                const CertifiedGroupPlanes& G, DiagNormalForm* raw) {
    if (kind_of_(Sred) != 0) throw std::logic_error("term_residual: kind0_normal_form_ on a kind-2 key (engine bug)");
    const KindEntry& e = kind_cache_.find(words_key('S', Sred))->second;
    DiagPauliClifford Pc = DiagPauliClifford::identity(n_);      // the operator X^x Z^z (mod phase)
    for (int q = 0; q < n_; ++q) {
        if (P.xbit(q)) Pc.v[(size_t)q] = 1;
        if (P.zbit(q)) Pc.a[(size_t)q] = 2;
    }
    const DiagPauliClifford R = e.D.then(Pc);                    // P ∘ rot(S): the shot's composite
#ifdef XTIM_MUTATE_TK_PSIGN
    // mutation build only (MUT-TK-PSIGN): the term route's OWN prefix extraction with the opposite
    // convention (S† = S³ kept, S = S†·Z folding a Z into the prefix)
    DiagNormalForm nf;
    nf.prefix = Pauli(n_);
    nf.a.assign((size_t)n_, 0);
    for (int q = 0; q < n_; ++q) {
        const int aq = R.a[(size_t)q] & 3;
        if (aq & 1) nf.a[(size_t)q] = 1;
        if (aq == 1 || aq == 2) nf.prefix.setz(q);
        if (R.v[(size_t)q] & 1) nf.prefix.setx(q);
        for (int l = q + 1; l < n_; ++l) if (R.B.get(q, l)) nf.cz.emplace_back(q, l);
    }
    nf.diagonal_class = true;
#else
    DiagNormalForm nf = residual_normal_form(R);
#endif
    if (raw) *raw = nf;
    canonicalize_mod_stabilizers_inplace(nf, G);
    return nf;
}

// ── plan identity: one intern table for every route (the caller builds the key bytes) ─────────────
long TermResidual::plan_id_(const std::string& key) {
    auto it = plan_ids_.find(key);
    if (it != plan_ids_.end()) return it->second;
    const long id = (long)plan_ids_.size();
    plan_ids_.emplace(key, id);
    return id;
}

}  // namespace qeccore
