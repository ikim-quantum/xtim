#include "qeccore/clifford_conjugate.hpp"
#include <vector>
namespace qeccore {
bool conjugate_by_gate(DiagPauliClifford& D, GateKind U, int a, int b, int c) {
    // A finalized cache describes the operator at finalize() time; any mutation makes
    // it stale, and a copy sharing the shared_ptr would silently serve pre-mutation
    // pairs. No production path finalizes before conjugating today — this reset makes
    // that ordering structurally impossible rather than an unguarded convention. The
    // null-check branch is predicted-true throughout the propagation walk (atoms are
    // never finalized), so the hot path pays one predictable compare.
    if (D.cache) D.cache.reset();
    auto& A = D.a; auto& B = D.B; auto& v = D.v; auto& g = D.gamma;
    auto adda = [&](int q,int d){ A[q]=(uint8_t)((A[q]+d)&3); };
    switch (U) {
        case GateKind::Z:   if (v[a]) g.add_z8(4); return true;                  // Z X Z = -X
        case GateKind::S:   if (v[a]) { adda(a,2); g.add_z8(2);} return true;    // S X S† = iXZ
        case GateKind::SDG: if (v[a]) { adda(a,2); g.add_z8(6);} return true;    // S† X S = -iXZ
        case GateKind::CZ:
            if (v[a]) adda(b,2);
            if (v[b]) adda(a,2);
            if (v[a] && v[b]) g.add_z8(4);
            return true;
        case GateKind::X: {                       // q'(z) = q(z⊕e_a)
            g.add_z8(2 * (A[a] & 3));              // γ ·= i^{a_a}
            const uint64_t* ra = B.row(a);         // a_l += 2 over B-neighbors of a (bit-scan;
            for (int w = 0; w < B.words(); ++w)    // the per-bit get() loop was a table hot spot)
                for (uint64_t bits = ra[w]; bits; bits &= bits - 1) {
                    const int l = (w << 6) + __builtin_ctzll(bits);
                    if (l != a) adda(l, 2);
                }
            A[a] = (uint8_t)((4 - A[a]) & 3);      // a_a ← -a_a
            return true;
        }
        case GateKind::Y: {                        // Y D Y† = X (Z D Z) X
            if (v[a]) g.add_z8(4);                 // Z(a) rule
            g.add_z8(2 * (A[a] & 3));              // then X(a) rule:
            const uint64_t* ra = B.row(a);
            for (int w = 0; w < B.words(); ++w)
                for (uint64_t bits = ra[w]; bits; bits &= bits - 1) {
                    const int l = (w << 6) + __builtin_ctzll(bits);
                    if (l != a) adda(l, 2);
                }
            A[a] = (uint8_t)((4 - A[a]) & 3);
            return true;
        }
        case GateKind::CX: {                       // control a, target b
            const int at = A[b];                    // ORIGINAL a_target
            const int Btc = B.get(b, a);            // ORIGINAL B_{ab}
            // ORIGINAL B couplings on target b: stack snapshot + bit-scan (the per-step heap
            // vector + per-bit get() here dominated build_propagation_table).
            // W == 0 is the LAZY all-zero B: no couplings exist, so the snapshot (and its
            // a/b mask writes) must be skipped entirely — the masks index snap[a>>6],
            // which is only in bounds when B is materialized (W == ceil(n/64) > a>>6).
            // Sizing the buffer by W and masking unconditionally was an out-of-bounds
            // stack write for lazy B with targets >= qubit 4096 (caught by
            // test_packed_invariants::test_lazy_cx_wide).
            const int W = B.words();
            uint64_t sbuf[64];
            std::vector<uint64_t> hbuf;             // n > 4096 fallback
            uint64_t* snap = nullptr;
            if (W > 0) {
                snap = (W <= 64) ? sbuf : (hbuf.resize(W), hbuf.data());
                const uint64_t* rb = B.row(b);
                for (int w = 0; w < W; ++w) snap[w] = rb[w];
                snap[a >> 6] &= ~(1ull << (a & 63));   // exclude m == a and m == b
                snap[b >> 6] &= ~(1ull << (b & 63));
            }
            adda(a, at + 2 * Btc);                  // a_c += a_t + 2 B_{tc}
            if (at & 1) B.flip_sym(b, a);           // B_{tc} ^= (a_t mod 2)
            for (int w = 0; w < W; ++w)             // B_{cm} ^= B_{tm} (flip_sym keeps symmetry)
                for (uint64_t bits = snap[w]; bits; bits &= bits - 1)
                    B.flip_sym(a, (w << 6) + __builtin_ctzll(bits));
            v[b] = (uint8_t)(v[b] ^ v[a]);          // v_t ^= v_c
            return true;
        }
        case GateKind::T:
            if (v[a]) { adda(a, 3); g.add_z8(1); }     // T X T† = ζ8·X·S†
            return true;
        case GateKind::CS:
            if (v[a] && !v[b]) { adda(b, 1); B.flip_sym(a, b); }
            else if (!v[a] && v[b]) { adda(a, 1); B.flip_sym(a, b); }
            else if (v[a] && v[b]) { adda(a, 3); adda(b, 3); g.add_z8(2); }
            return true;
        case GateKind::CCZ: {
            int qs[3] = { a, b, c };
            int cnt = v[a] + v[b] + v[c];
            if (cnt == 1) {                            // X on one → CZ on the other two
                int p = -1, r = -1;
                for (int t : qs) if (!v[t]) { if (p < 0) p = t; else r = t; }
                B.flip_sym(p, r);
            } else if (cnt == 2) {                     // X on two (i,j) → a_k+=2, CZ_{ik}, CZ_{jk}
                int k = -1; for (int t : qs) if (!v[t]) k = t;
                int i = -1, j = -1; for (int t : qs) if (v[t]) { if (i < 0) i = t; else j = t; }
                adda(k, 2); B.flip_sym(i, k); B.flip_sym(j, k);
            } else if (cnt == 3) {                     // all three → Z each, CZ each pair, γ·=−1
                adda(a, 2); adda(b, 2); adda(c, 2);
                B.flip_sym(a, b); B.flip_sym(a, c); B.flip_sym(b, c);
                g.add_z8(4);
            }
            return true;
        }
        case GateKind::H: {
            // In-class iff D acts as a Pauli on qubit a: a_a ∈ {0,2} and no B coupling on a.
            if (A[a] & 1) return false;
            // Word-scan row a for any coupling (the diagonal is invariant-zero, so the
            // old loop's l != a exclusion excluded a bit that is always 0); a lazy
            // all-zero B has words()==0 and skips outright. The per-l get() loop this
            // replaces was O(n) per H visit — the dominant propagation-table cost on
            // memory circuits (5.8M H visits at d=11) for a B that is empty on every
            // Clifford-only path. Same accept/reject decision, bit for bit.
            for (int w = 0, bw = B.words(); w < bw; ++w)
                if (B.row(a)[w]) return false;
            // Now D on a is X^{v_a} Z^{a_a/2}; H swaps X_a ↔ Z_a.  H Z H = X, H X H = Z, H Y H = -Y.
            int za = (A[a] == 2) ? 1 : 0;          // Z present on a?
            int xa = v[a];                          // X present on a?
            A[a] = (uint8_t)(xa ? 2 : 0);           // new Z-part = old X-part
            v[a]  = (uint8_t)za;                     // new X-part = old Z-part
            if (xa && za) g.add_z8(4);              // H Y H = -Y
            return true;
        }
        case GateKind::CH: return false;            // fast-path trigger only: the sampler's
                                                    // ppr_retry re-propagates via PprResidual (CH is supported there);
                                                    // DEM/feedback callers keep the reject (no linear-DEM form).
    }
    // Every enumerator is handled above; omitting `default:` lets -Wswitch flag a future
    // gate that is added but not handled here. If U is an out-of-range value, reject —
    // for an error-propagation engine, a silent identity-propagation would be a corruption.
    return false;
}
}
