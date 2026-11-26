#include "qeccore/frame.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <iterator>
#include <stdexcept>
#include <string>

#include "qeccore/pauli_kernels.hpp"   // single_pauli, sparse_canonical

namespace qeccore {

namespace {
std::atomic<long> g_frame_expansions{0};
}  // namespace

long frame_expansions() { return g_frame_expansions.load(std::memory_order_relaxed); }
void reset_frame_expansions() { g_frame_expansions.store(0, std::memory_order_relaxed); }

namespace {

// kind 0:h 1:s 2:sdg 3:x 4:y 5:z 6:cx 7:cz (CliffordTableau::LeftGate). The inverse gate: S <-> S†, the rest self-inverse.
inline uint8_t inverse_kind(uint8_t k) { return k == 1 ? 2 : k == 2 ? 1 : k; }

// M <- g M g† through the ONE source of the gate rules (CliffordTableau::conj_*; Y = X then Z as left_y does).
inline void conj_gate(Pauli& M, uint8_t kind, int a, int b) {
    switch (kind) {
        case 0: CliffordTableau::conj_h(M, a); break;
        case 1: CliffordTableau::conj_s(M, a); break;
        case 2: CliffordTableau::conj_sdg(M, a); break;
        case 3: CliffordTableau::conj_x(M, a); break;
        case 4: CliffordTableau::conj_x(M, a); CliffordTableau::conj_z(M, a); break;
        case 5: CliffordTableau::conj_z(M, a); break;
        case 6: CliffordTableau::conj_cx(M, a, b); break;
        default: CliffordTableau::conj_cz(M, a, b); break;
    }
}

// The scans' touched-wire map (one byte per wire, sized to the frame on demand, reset over the touched wires after
// each scan). Per thread, as the scans' scratch.
struct ScanMarks {
    std::vector<uint8_t> touched;
    std::vector<int> dirty;
    void begin(int n) { if ((int)touched.size() < n) touched.assign((size_t)n, 0); dirty.clear(); }
    void set(int q) { if (!touched[(size_t)q]) { touched[(size_t)q] = 1; dirty.push_back(q); } }
    void end() { for (int q : dirty) touched[(size_t)q] = 0; }
};
thread_local ScanMarks scan_marks;

}  // namespace

// ── construction ────────────────────────────────────────────────────────────────────────────────────────────────
Frame::Frame(int n_) : n(n_) {
    std::vector<int> pi((size_t)n_);
    for (int a = 0; a < n_; ++a) pi[(size_t)a] = a;
    set_support_({}, {}, std::move(pi));
}

void Frame::set_support_(std::vector<int> fidx, std::vector<int> wires, std::vector<int> pi) const {
    if (fidx.size() != wires.size() || (int)pi.size() != n)
        throw std::logic_error("Frame: |F_S| != |W_S| or a pi of the wrong length (engine bug)");
    if ((int)fidx.size() == n) {
        // F_S = all (ascending, duplicate-free ⇒ 0..n-1, and so W_S): the expanded shape, no index maps stored
        for (int a = 0; a < n; ++a)
            if (fidx[(size_t)a] != a || wires[(size_t)a] != a)
                throw std::logic_error("Frame: F_S / W_S must be ascending, duplicate-free frame indices / wires (engine bug)");
        full_ = true;
        for (std::vector<int>* v : {&fidx_, &wires_, &fpos_, &wpos_, &pi_, &pinv_}) std::vector<int>().swap(*v);
        return;
    }
    full_ = false;
    fidx_ = std::move(fidx);
    wires_ = std::move(wires);
    pi_ = std::move(pi);
    fpos_.assign((size_t)n, -1);
    wpos_.assign((size_t)n, -1);
    pinv_.assign((size_t)n, -1);
    for (size_t i = 0; i < fidx_.size(); ++i) {
        const int a = fidx_[i], q = wires_[i];
        if (a < 0 || a >= n || q < 0 || q >= n || fpos_[(size_t)a] >= 0 || wpos_[(size_t)q] >= 0 ||
            (i && (fidx_[i - 1] >= a || wires_[i - 1] >= q)))
            throw std::logic_error("Frame: F_S / W_S must be ascending, duplicate-free frame indices / wires (engine bug)");
        fpos_[(size_t)a] = (int)i;
        wpos_[(size_t)q] = (int)i;
    }
    for (int a = 0; a < n; ++a) {
        const int q = pi_[(size_t)a];
        if (fpos_[(size_t)a] >= 0) {
            if (q != -1) throw std::logic_error("Frame: pi must be -1 on F_S (engine bug)");
            continue;
        }
        if (q < 0 || q >= n || wpos_[(size_t)q] >= 0 || pinv_[(size_t)q] >= 0)
            throw std::logic_error("Frame: pi must map the frame indices off F_S onto the wires off W_S (engine bug)");
        pinv_[(size_t)q] = a;
    }
}

Frame Frame::compact(int n, std::vector<int> fidx, std::vector<int> wires, std::vector<int> pi,
                     CliffordTableau U0, std::vector<LeftGate> C) {
    if (U0.n != (int)fidx.size()) throw std::logic_error("Frame::compact: U0 must act on |S| qubits (engine bug)");
    Frame F;
    F.n = n;
    F.set_support_(std::move(fidx), std::move(wires), std::move(pi));
    F.U0_ = std::move(U0);
    F.C_ = std::move(C);
    return F;
}

Frame Frame::expanded(CliffordTableau U, std::vector<LeftGate> C) {
    Frame F;                                   // full_ (no index maps)
    F.n = U.n;
    F.U0_ = std::move(U);
    F.C_ = std::move(C);
    return F;
}

const std::vector<int>& Frame::all_indices_() const {
    if ((int)all_.size() != n) {
        all_.resize((size_t)n);
        for (int a = 0; a < n; ++a) all_[(size_t)a] = a;
    }
    return all_;
}

void Frame::append(const LeftGate& g) { C_.push_back(g); invalidate_caches(); }
void Frame::append(const std::vector<LeftGate>& gs) { C_.insert(C_.end(), gs.begin(), gs.end()); invalidate_caches(); }

void Frame::invalidate_caches() const {
    d_.rows_ready_ = false;
    std::vector<SparsePauli>().swap(d_.rows_z_);
    std::vector<SparsePauli>().swap(d_.rows_x_);
    d_.inv_ready_ = d_.inv_dense_ = false;
    d_.conj_queries_ = 0;
    std::vector<SparsePauli>().swap(d_.inv_x_);
    std::vector<SparsePauli>().swap(d_.inv_z_);
    d_.inv_tab_ = CliffordTableau(0);
}

// U† X_q U and U† Z_q U for every wire q (S2-T4b; see the header).
void Frame::build_inv_() const {
    if (full_ || !fidx_.empty()) {
        // a nonempty block (the magic circuits' small frames): the dense dual rows, built as today's dense frame builds
        // them — the forward rows of U (the base, then left_batch(C)) in a transient tableau, reset_dual, and the
        // forward rows dropped. The frame itself keeps its shape (no expansion); the cache folds as today's dual does.
        CliffordTableau D = full_ ? U0_ : base_tableau_();
        D.left_batch(C_);
        D.reset_dual();
        std::vector<Pauli>().swap(D.Xrow);
        std::vector<Pauli>().swap(D.Zrow);
        d_.inv_tab_ = std::move(D);
        d_.inv_dense_ = true;
        d_.inv_ready_ = true;
        return;
    }
    // F_S = ∅ (the Clifford circuits): C† X_q C / C† Z_q C by one backward sweep of the 2n wire Paulis, relabelled by
    // π⁻¹ (the base)
    std::vector<SparsePauli> seeds((size_t)2 * (size_t)n), out;
    for (int q = 0; q < n; ++q) {
        SparsePauli& x = seeds[(size_t)q];
        x.n = n; x.x.push_back(q);
        SparsePauli& z = seeds[(size_t)n + (size_t)q];
        z.n = n; z.z.push_back(q);
    }
    sweep_gates_(seeds, false, out);
    std::vector<SparsePauli>().swap(seeds);
    if (!base_is_identity_()) {
        SparsePauli t;
        for (SparsePauli& p : out) { relabel_(p, false, t); std::swap(p, t); }
    }
    d_.inv_x_.assign(std::make_move_iterator(out.begin()), std::make_move_iterator(out.begin() + n));
    d_.inv_z_.assign(std::make_move_iterator(out.begin() + n), std::make_move_iterator(out.end()));
    d_.inv_dense_ = false;
    d_.inv_ready_ = true;
}

// ── the scans ───────────────────────────────────────────────────────────────────────────────────────────────────
// M <- C M C† (forward: the gates in apply order) or C† M C (backward: the inverse gates in reverse order), applying a
// gate iff one of its targets is in the operator's touched support — the propagation_table.cpp Walk condition (a gate
// on untouched wires is the identity on M; a two-qubit gate with one touched target may touch the other). One byte
// test per gate: O(|C|) per query, which beats a cone-sized heap merge at every measured size (cones are 6–30 % of C).
void Frame::walk_(Pauli& M, bool forward) const {
    if (C_.empty()) return;
    ScanMarks& marks = scan_marks;
    marks.begin(n);
    for (size_t w = 0; w < M.x.size(); ++w)
        for (uint64_t bits = M.x[w] | M.z[w]; bits; bits &= bits - 1) marks.set((int)(64 * w + __builtin_ctzll(bits)));
    const uint8_t* touched = marks.touched.data();
    const int L = (int)C_.size();
    uint64_t work = 0;
    for (int k = 0; k < L; ++k) {
        const LeftGate& g = C_[(size_t)(forward ? k : L - 1 - k)];
        const bool two = g.kind >= 6;
        if (!touched[g.a] && !(two && touched[g.b])) continue;
        conj_gate(M, forward ? g.kind : inverse_kind(g.kind), g.a, g.b);
        ++work;
        marks.set(g.a);
        if (two) marks.set(g.b);
    }
    frame_walk_work += work;
    marks.end();
}

// ── the base U0 ⊕ π ─────────────────────────────────────────────────────────────────────────────────────────────
// (U0 ⊕ π) P (U0 ⊕ π)† for P on frame indices: P = i^ph · A · B with A the F_S part and B the rest (disjoint supports,
// both phase 0 in canonical order), so the image is i^{ph + phase(U0 A U0†)} times the union of the two images.
Pauli Frame::base_forward_(const Pauli& P) const {
    if (full_) return U0_.forward_image(P);
    const int s = (int)fidx_.size();
    Pauli M(n);
    M.phase = P.phase;
    Pauli A(s);
    bool any = false;
    for (size_t w = 0; w < P.x.size(); ++w) {
        for (uint64_t bits = P.x[w]; bits; bits &= bits - 1) {
            const int a = (int)(64 * w + __builtin_ctzll(bits));
            if (fpos_[(size_t)a] >= 0) { A.setx(fpos_[(size_t)a]); any = true; } else M.setx(pi_[(size_t)a]);
        }
        for (uint64_t bits = P.z[w]; bits; bits &= bits - 1) {
            const int a = (int)(64 * w + __builtin_ctzll(bits));
            if (fpos_[(size_t)a] >= 0) { A.setz(fpos_[(size_t)a]); any = true; } else M.setz(pi_[(size_t)a]);
        }
    }
    if (any) {
        const Pauli B = U0_.forward_image(A);
        M.phase = (M.phase + B.phase) & 3;
        for (size_t w = 0; w < B.x.size(); ++w) {
            for (uint64_t bits = B.x[w]; bits; bits &= bits - 1) M.setx(wires_[(size_t)(64 * w + __builtin_ctzll(bits))]);
            for (uint64_t bits = B.z[w]; bits; bits &= bits - 1) M.setz(wires_[(size_t)(64 * w + __builtin_ctzll(bits))]);
        }
    }
    return M;
}

// (U0 ⊕ π)† M (U0 ⊕ π) for M on wires: the W_S part through U0's inverse rows (U0.conjugate: ensure_dual on the block),
// the rest relabelled by π⁻¹.
Pauli Frame::base_backward_(const Pauli& M) const {
    if (full_) return U0_.conjugate(M);
    const int s = (int)fidx_.size();
    Pauli Q(n);
    Q.phase = M.phase;
    Pauli A(s);
    bool any = false;
    for (size_t w = 0; w < M.x.size(); ++w) {
        for (uint64_t bits = M.x[w]; bits; bits &= bits - 1) {
            const int q = (int)(64 * w + __builtin_ctzll(bits));
            if (wpos_[(size_t)q] >= 0) { A.setx(wpos_[(size_t)q]); any = true; } else Q.setx(pinv_[(size_t)q]);
        }
        for (uint64_t bits = M.z[w]; bits; bits &= bits - 1) {
            const int q = (int)(64 * w + __builtin_ctzll(bits));
            if (wpos_[(size_t)q] >= 0) { A.setz(wpos_[(size_t)q]); any = true; } else Q.setz(pinv_[(size_t)q]);
        }
    }
    if (any) {
        const Pauli B = U0_.conjugate(A);
        Q.phase = (Q.phase + B.phase) & 3;
        for (size_t w = 0; w < B.x.size(); ++w) {
            for (uint64_t bits = B.x[w]; bits; bits &= bits - 1) Q.setx(fidx_[(size_t)(64 * w + __builtin_ctzll(bits))]);
            for (uint64_t bits = B.z[w]; bits; bits &= bits - 1) Q.setz(fidx_[(size_t)(64 * w + __builtin_ctzll(bits))]);
        }
    }
    return Q;
}

// (U0 ⊕ π) Z_a / X_a (U0 ⊕ π)† on wires, sparse: Z_{π(a)} / X_{π(a)} off F_S, else U0's local row relabelled (W_S is
// ascending, so the local support order is the wire order).
void Frame::base_row_(int a, bool isz, SparsePauli& out) const {
    out.n = n;
    out.x.clear();
    out.z.clear();
    const int i = row_pos(a);
    if (i < 0) {
        out.phase = 0;
        (isz ? out.z : out.x).push_back(pi_[(size_t)a]);
        return;
    }
    const Pauli& R = isz ? U0_.Zrow[(size_t)i] : U0_.Xrow[(size_t)i];
    out.phase = R.phase;
    for (size_t w = 0; w < R.x.size(); ++w) {
        for (uint64_t bits = R.x[w]; bits; bits &= bits - 1) { const int j = (int)(64 * w + __builtin_ctzll(bits)); out.x.push_back(full_ ? j : wires_[(size_t)j]); }
        for (uint64_t bits = R.z[w]; bits; bits &= bits - 1) { const int j = (int)(64 * w + __builtin_ctzll(bits)); out.z.push_back(full_ ? j : wires_[(size_t)j]); }
    }
}

// F_S = ∅: the base is the relabelling π (forward: frame index → wire) or π⁻¹ (backward); phase unchanged, the lists
// re-sorted (π need not be monotone).
void Frame::relabel_(const SparsePauli& in, bool forward, SparsePauli& out) const {
    const std::vector<int>& m = forward ? pi_ : pinv_;
    out.n = n;
    out.phase = in.phase;
    out.x.resize(in.x.size());
    out.z.resize(in.z.size());
    for (size_t i = 0; i < in.x.size(); ++i) out.x[i] = m[(size_t)in.x[i]];
    for (size_t i = 0; i < in.z.size(); ++i) out.z[i] = m[(size_t)in.z[i]];
    std::sort(out.x.begin(), out.x.end());
    std::sort(out.z.begin(), out.z.end());
}

bool Frame::base_is_identity_() const {
    if (full_ || !fidx_.empty()) return false;
    for (int a = 0; a < n; ++a) if (pi_[(size_t)a] != a) return false;
    return true;
}

// ── queries ─────────────────────────────────────────────────────────────────────────────────────────────────────
Pauli Frame::conjugate(const Pauli& P) const {
    if (!C_.empty() && !d_.inv_ready_ && ++d_.conj_queries_ * 64 >= (uint64_t)n) build_inv_();   // ski rental (header)
    if (C_.empty() || !d_.inv_ready_) {
        Pauli M = P;
        walk_(M, false);
        return base_backward_(M);
    }
    if (d_.inv_dense_) return d_.inv_tab_.conjugate(P);   // the dual rows' fold (a valid dual: reads only)
    // U† P U as the fold of the inverse wire images, in CliffordTableau::fold_rows_' order (ascending wire, X before Z)
    Pauli M(n);
    M.phase = P.phase;
    const int W = (int)P.x.size();
    for (int w = 0; w < W; ++w) {
        uint64_t xb = P.x[(size_t)w], zb = P.z[(size_t)w];
        const int base = w << 6;
        while (xb | zb) {
            const int qx = xb ? base + __builtin_ctzll(xb) : n + 1;
            const int qz = zb ? base + __builtin_ctzll(zb) : n + 1;
            if (qx <= qz) { pmul_into(M, d_.inv_x_[(size_t)qx]); xb &= xb - 1; }
            else          { pmul_into(M, d_.inv_z_[(size_t)qz]); zb &= zb - 1; }
        }
    }
    return M;
}

Pauli Frame::forward_image(const Pauli& P) const {
    Pauli M = base_forward_(P);
    walk_(M, true);
    return M;
}

SparsePauli Frame::row_z(int a) const {
    if (d_.rows_ready_) return d_.rows_z_[(size_t)a];
    return from_dense(forward_image(single_pauli(2, a, n)));
}
SparsePauli Frame::row_x(int a) const {
    if (d_.rows_ready_) return d_.rows_x_[(size_t)a];
    return from_dense(forward_image(single_pauli(0, a, n)));
}

const std::vector<SparsePauli>& Frame::rows_z() const { if (!d_.rows_ready_) build_rows_(); return d_.rows_z_; }
const std::vector<SparsePauli>& Frame::rows_x() const { if (!d_.rows_ready_) build_rows_(); return d_.rows_x_; }

// The rows cache: one forward sweep of the 2n base rows through C. DROP-IN (Stage-1 T5): T5's per-wire sweep
// (propagation_table.cpp sweep_clifford_suffix: U <- U·g on the images of X_q / Z_q, reverse gate order) computes the
// images of the 2n WIRE Paulis under C; on a frame with F_S = ∅ those are these rows up to the relabelling π, so its
// sparse update can replace the sweep_gates_ call below for such frames (for F_S ≠ ∅ the F_S rows are then folds of
// the wire images, pmul_sparse over the base row's support). Either way the rows are the same unique operators.
void Frame::build_rows_() const {
    std::vector<SparsePauli> seeds((size_t)2 * (size_t)n), out;
    for (int a = 0; a < n; ++a) { base_row_(a, true, seeds[(size_t)a]); base_row_(a, false, seeds[(size_t)n + (size_t)a]); }
    sweep_gates_(seeds, true, out);
    d_.rows_z_.assign(out.begin(), out.begin() + n);
    d_.rows_x_.assign(out.begin() + n, out.end());
    d_.rows_ready_ = true;
}

std::vector<SparsePauli> Frame::certified_rows(const std::vector<int>& free, const std::vector<uint8_t>* eps) const {
    if (eps && (int)eps->size() < n) throw std::logic_error("Frame::certified_rows: eps must have n entries (engine bug)");
    std::vector<uint8_t> is_free((size_t)(n > 0 ? n : 1), 0);
    for (int f : free) if (f >= 0 && f < n) is_free[(size_t)f] = 1;
    std::vector<SparsePauli> out;
    out.reserve((size_t)n - std::min((size_t)n, free.size()));
    if (d_.rows_ready_) {
        for (int a = 0; a < n; ++a) if (!is_free[(size_t)a]) out.push_back(d_.rows_z_[(size_t)a]);
    } else {
        // S2-T4b: no rows cache is built (the compile reads these rows once): one forward sweep of the Z base rows of
        // the certified indices only
        std::vector<SparsePauli> seeds;
        seeds.reserve(out.capacity());
        for (int a = 0; a < n; ++a) {
            if (is_free[(size_t)a]) continue;
            seeds.emplace_back();
            base_row_(a, true, seeds.back());
        }
        sweep_gates_(seeds, true, out);
    }
    size_t i = 0;
    for (int a = 0; a < n; ++a) {
        if (is_free[(size_t)a]) continue;
        if (eps && (*eps)[(size_t)a]) out[i].phase = (out[i].phase + 2) & 3;   // ×(-1) = ×i²
        ++i;
    }
    return out;
}

void Frame::sweep(const std::vector<SparsePauli>& seeds, bool forward, std::vector<SparsePauli>& out) const {
    if (base_is_identity_()) { sweep_gates_(seeds, forward, out); return; }   // U = C (Clifford circuits): no base fold
    const bool relabel = !full_ && fidx_.empty();                                       // F_S = ∅, π ≠ id: a sparse relabelling
    if (forward) {
        std::vector<SparsePauli> based(seeds.size());
        for (size_t i = 0; i < seeds.size(); ++i) {
            assert(sparse_canonical(seeds[i]));
            if (relabel) relabel_(seeds[i], true, based[i]);
            else based[i] = from_dense(base_forward_(to_dense(seeds[i])));
        }
        sweep_gates_(based, true, out);
    } else {
        sweep_gates_(seeds, false, out);
        SparsePauli t;
        for (SparsePauli& s : out) {
            if (relabel) { relabel_(s, false, t); std::swap(s, t); }
            else s = from_dense(base_backward_(to_dense(s)));
        }
    }
}

void Frame::inverse_images(const std::vector<Pauli>& Ps, std::vector<Pauli>& out) const {
    out.clear();
    out.reserve(Ps.size());
    if (expanded() && C_.empty()) {
        for (const Pauli& P : Ps) out.push_back(inverse_image(P));
        return;
    }
    std::vector<SparsePauli> seeds(Ps.size());
    for (size_t i = 0; i < Ps.size(); ++i) seeds[i] = from_dense(Ps[i]);
    inverse_images(seeds, out);
}

void Frame::inverse_images(const std::vector<SparsePauli>& seeds, std::vector<Pauli>& out) const {
    out.clear();
    out.reserve(seeds.size());
    if ((expanded() && C_.empty()) || d_.inv_ready_) {   // the dense dual, or the inverse-image fold (conjugate)
        for (const SparsePauli& s : seeds) out.push_back(inverse_image(to_dense(s)));
        return;
    }
    std::vector<SparsePauli> res;
    sweep(seeds, false, res);
    for (SparsePauli& r : res) {
        out.push_back(to_dense(r));
        std::vector<int32_t>().swap(r.x);
        std::vector<int32_t>().swap(r.z);
    }
}

void Frame::prepare_queries() const {
    if (expanded()) {
        ensure_expanded();               // an expanded frame: flushes pending gates only (no expansion)
        U0_.ensure_dual();
    } else if (C_.empty()) {
        U0_.ensure_dual();               // conjugate is the base's fold (the block's dual)
    } else if (!d_.inv_ready_) {
        build_inv_();                    // conjugate / inverse_images fold the inverse images (no block dual read)
    }
}

// out[i] = C seeds[i] C† (forward) or C† seeds[i] C (backward), bit-sliced: ≤ kChunk seeds per pass, for every touched
// wire a row of x-plane and z-plane words (bit i = seed i) and the two phase bit-planes P0/P1; a gate is applied to its
// targets' rows iff one of them exists (an untouched wire's planes are zero, so the gate is the identity there) with
// the closed forms of CliffordTableau::left_batch. Memory O(touched wires × kChunk/64) per pass.
void Frame::sweep_gates_(const std::vector<SparsePauli>& seeds, bool forward, std::vector<SparsePauli>& out) const {
    const size_t N = seeds.size();
    out.assign(N, SparsePauli{});
    const int L = (int)C_.size();
    std::vector<int> row_of((size_t)n, -1), touched;
    std::vector<uint64_t> X, Z, P0, P1;
    for (size_t s0 = 0; s0 < N; s0 += (size_t)kChunk) {
        const int cnt = (int)std::min((size_t)kChunk, N - s0);
        const int CW = (cnt + 63) / 64;
        for (int q : touched) row_of[(size_t)q] = -1;
        touched.clear();
        X.clear(); Z.clear();
        P0.assign((size_t)CW, 0); P1.assign((size_t)CW, 0);
        auto ensure = [&](int q) {
            int r = row_of[(size_t)q];
            if (r < 0) {
                r = (int)touched.size();
                row_of[(size_t)q] = r;
                touched.push_back(q);
                X.resize(X.size() + (size_t)CW, 0);
                Z.resize(Z.size() + (size_t)CW, 0);
            }
            return (size_t)r * (size_t)CW;
        };
        for (int i = 0; i < cnt; ++i) {
            const SparsePauli& s = seeds[s0 + (size_t)i];
            assert(sparse_canonical(s) && s.n == n);
            const size_t w = (size_t)(i >> 6);
            const uint64_t bit = 1ull << (i & 63);
            for (int32_t q : s.x) X[ensure(q) + w] |= bit;
            for (int32_t q : s.z) Z[ensure(q) + w] |= bit;
            if (s.phase & 1) P0[w] |= bit;
            if (s.phase & 2) P1[w] |= bit;
        }
        for (int k = 0; k < L; ++k) {
            const LeftGate& g = C_[(size_t)(forward ? k : L - 1 - k)];
            const uint8_t kind = forward ? g.kind : inverse_kind(g.kind);
            if (kind < 6) {
                const int ra = row_of[(size_t)g.a];
                if (ra < 0) continue;
                uint64_t* Xa = X.data() + (size_t)ra * CW;
                uint64_t* Za = Z.data() + (size_t)ra * CW;
                switch (kind) {
                    case 0: for (int w = 0; w < CW; ++w) { P1[(size_t)w] ^= Xa[w] & Za[w]; std::swap(Xa[w], Za[w]); } break;             // H
                    case 1: for (int w = 0; w < CW; ++w) { Za[w] ^= Xa[w]; P1[(size_t)w] ^= P0[(size_t)w] & Xa[w]; P0[(size_t)w] ^= Xa[w]; } break;    // S
                    case 2: for (int w = 0; w < CW; ++w) { Za[w] ^= Xa[w]; P1[(size_t)w] ^= ~P0[(size_t)w] & Xa[w]; P0[(size_t)w] ^= Xa[w]; } break;   // S†
                    case 3: for (int w = 0; w < CW; ++w) P1[(size_t)w] ^= Za[w]; break;                                                   // X
                    case 4: for (int w = 0; w < CW; ++w) P1[(size_t)w] ^= Za[w] ^ Xa[w]; break;                                           // Y
                    default: for (int w = 0; w < CW; ++w) P1[(size_t)w] ^= Xa[w]; break;                                                  // Z
                }
            } else {
                if (row_of[(size_t)g.a] < 0 && row_of[(size_t)g.b] < 0) continue;
                const size_t oa = ensure(g.a), ob = ensure(g.b);
                uint64_t* Xa = X.data() + oa; uint64_t* Za = Z.data() + oa;
                uint64_t* Xt = X.data() + ob; uint64_t* Zt = Z.data() + ob;
                if (kind == 6) for (int w = 0; w < CW; ++w) { Xt[w] ^= Xa[w]; Za[w] ^= Zt[w]; }                                           // CX
                else for (int w = 0; w < CW; ++w) { P1[(size_t)w] ^= Xa[w] & Xt[w]; Zt[w] ^= Xa[w]; Za[w] ^= Xt[w]; }                   // CZ
            }
        }
        std::sort(touched.begin(), touched.end());    // ascending wires ⇒ ascending supports per seed
        for (int i = 0; i < cnt; ++i) {
            SparsePauli& o = out[s0 + (size_t)i];
            o.n = n;
            o.phase = (int)((P0[(size_t)(i >> 6)] >> (i & 63)) & 1ull) | (int)(((P1[(size_t)(i >> 6)] >> (i & 63)) & 1ull) << 1);
        }
        for (int q : touched) {
            const uint64_t* Xq = X.data() + (size_t)row_of[(size_t)q] * CW;
            const uint64_t* Zq = Z.data() + (size_t)row_of[(size_t)q] * CW;
            for (int w = 0; w < CW; ++w) {
                for (uint64_t bits = Xq[w]; bits; bits &= bits - 1) out[s0 + (size_t)((w << 6) + __builtin_ctzll(bits))].x.push_back(q);
                for (uint64_t bits = Zq[w]; bits; bits &= bits - 1) out[s0 + (size_t)((w << 6) + __builtin_ctzll(bits))].z.push_back(q);
            }
        }
    }
#ifndef NDEBUG
    for (const SparsePauli& o : out) assert(sparse_canonical(o));
#endif
}

// ── shape ───────────────────────────────────────────────────────────────────────────────────────────────────────
// The dense n-qubit tableau of the base U0 ⊕ π, forward rows only: no identity dual is allocated (CliffordTableau(n)
// would) and none is built — QW-B: the dense dual is built on first use (ensure_dual), as from_generators + left_batch
// leave today's dense frame.
CliffordTableau Frame::base_tableau_() const {
    CliffordTableau D(0);
    D.n = n;
    D.Xrow.resize((size_t)n);
    D.Zrow.resize((size_t)n);
    D.invalidate_dual();
    SparsePauli r;
    for (int a = 0; a < n; ++a) {
        base_row_(a, true, r);  D.Zrow[(size_t)a] = to_dense(r);
        base_row_(a, false, r); D.Xrow[(size_t)a] = to_dense(r);
    }
    return D;
}

void Frame::ensure_expanded() const {
    if (!expanded()) {
        assert(!frozen() && "Frame::ensure_expanded: a frozen (thread-shared) frame would be expanded (engine bug)");
        g_frame_expansions.fetch_add(1, std::memory_order_relaxed);
        U0_ = base_tableau_();
        full_ = true;
        for (std::vector<int>* v : {&fidx_, &wires_, &fpos_, &wpos_, &pi_, &pinv_}) std::vector<int>().swap(*v);
        U0_.left_batch(C_);
        C_.clear();
        invalidate_caches();
    } else if (!C_.empty()) {
        assert(!frozen() && "Frame::ensure_expanded: a frozen (thread-shared) frame's gates would be flushed (engine bug)");
        U0_.left_batch(C_);                  // today's flush_gates (the dual patched or rebuilt as left_batch does)
        C_.clear();
        invalidate_caches();
    }
}

CliffordTableau& Frame::dense_slow_() const { ensure_expanded(); return U0_; }

size_t Frame::parts_bytes() const {
    const size_t s = full_ ? (size_t)U0_.n : fidx_.size();
    size_t b = (full_ ? 0 : (4 * (size_t)n + 2 * s) * sizeof(int)) + C_.size() * sizeof(LeftGate);
    const size_t W0 = (s + 63) / 64;
    b += (U0_.dual_valid() ? 4 : 2) * s * (2 * W0 * sizeof(uint64_t) + sizeof(Pauli));
    return b;
}

size_t Frame::compact_bytes() const {
    size_t b = parts_bytes();
    for (const SparsePauli& r : d_.rows_z_) b += sizeof(SparsePauli) + (r.x.size() + r.z.size()) * sizeof(int32_t);
    for (const SparsePauli& r : d_.rows_x_) b += sizeof(SparsePauli) + (r.x.size() + r.z.size()) * sizeof(int32_t);
    // S2-T10a: and the inverse-image cache when built (S2-T4b; prepare_queries builds it for thread lanes): the sparse
    // images of the 2n wire Paulis, or the dense dual rows of the transient tableau.
    for (const auto* v : {&d_.inv_x_, &d_.inv_z_})
        for (const SparsePauli& r : *v) b += sizeof(SparsePauli) + (r.x.size() + r.z.size()) * sizeof(int32_t);
    if (d_.inv_dense_) {
        const size_t m = (size_t)d_.inv_tab_.n, Wm = (m + 63) / 64;
        b += 2 * m * (2 * Wm * sizeof(uint64_t) + sizeof(Pauli));
    }
    return b;
}

}  // namespace qeccore
