#include "qeccore/ppr_residual.hpp"
#include "qeccore/clifford_tableau.hpp"
#include <cstdlib>

namespace qeccore {

// ---------------------------------------------------------------------------
// Clifford conjugation: Q ← U·Q·U† for a single Clifford gate U, in place.
// Thin dispatcher onto the tableau engine's exact single-Pauli rules
// (CliffordTableau::conj_* — the single source of truth for Pauli-through-
// Clifford conjugation; O(1) per gate, no allocation). Y conjugates as X∘Z:
// Y = iXZ, and the scalar i cancels in conjugation.
// ---------------------------------------------------------------------------
static void conj_cliff(Pauli& Q, GateKind g, const std::vector<int>& t) {
    const int a = t[0];
    switch (g) {
        case GateKind::X:   CliffordTableau::conj_x(Q, a); return;
        case GateKind::Y:   CliffordTableau::conj_z(Q, a);
                            CliffordTableau::conj_x(Q, a); return;
        case GateKind::Z:   CliffordTableau::conj_z(Q, a); return;
        case GateKind::H:   CliffordTableau::conj_h(Q, a); return;
        case GateKind::S:   CliffordTableau::conj_s(Q, a); return;
        case GateKind::SDG: CliffordTableau::conj_sdg(Q, a); return;
        case GateKind::CX:  CliffordTableau::conj_cx(Q, a, t[1]); return;
        case GateKind::CZ:  CliffordTableau::conj_cz(Q, a, t[1]); return;
        default:
            std::abort();   // not a Clifford gate — propagate() never routes others here
    }
}

// ---------------------------------------------------------------------------
// Restore the canonical invariant for a stored rotation axis:
//   phase ∈ {0,1}; if phase ≥ 2, fold the -1 into the exponent.
// Identity: exp(iπ/4 · e · (-A)) = exp(iπ/4 · (8-e) · A)
// ---------------------------------------------------------------------------
static void canonicalize_axis(Pauli& A, int& e) {
    if (A.phase >= 2) {
        A.phase -= 2;          // 2→0 or 3→1
        e = (8 - e) % 8;
    }
}

// ---------------------------------------------------------------------------
// Merge a new rotation (axis Q, exponent delta in 1..7) into rots list.
// If the same axis is already present: add exponents mod 8, remove if 0.
// ---------------------------------------------------------------------------
// Exponents live in Z/8Z (period 8, since exp(i(π/4)·8·Q)=exp(i·2π·Q)=I).
// Range: 1..7 (0 = identity, removed).
//
// Representation invariant: stored axes always have phase ∈ {0,1}.
// A negative axis Q (phase ∈ {2,3}) equals −Q_can (phase−2), and
//   exp(i(π/4)e·Q) = exp(i(π/4)(8−e)·Q_can),
// so the sign is folded into the exponent (see canonicalize_axis).
// merge_rot enforces this on insert; propagate() re-canonicalizes after
// every Clifford conjugation.
static void merge_rot(std::vector<std::pair<Pauli,int>>& rots, const Pauli& Q, int delta) {
    // Canonicalize the incoming axis (phase ∈ {0,1}; sign folded into the exponent).
    Pauli Qcan = Q;
    int dcan = ((delta % 8) + 8) % 8;
    canonicalize_axis(Qcan, dcan);
    if (dcan == 0) return;

    // Stored axes are already canonical (propagate() re-canonicalizes after every
    // Clifford conjugation), so a plain field compare decides equality. For a fixed
    // x/z support the Hermitian phase parity is determined, so the phase compare is
    // redundant — kept as a cheap consistency guard.
    for (auto& pr : rots) {
        bool same = (pr.first.phase == Qcan.phase);
        for (int w = 0; w < (int)Qcan.x.size() && same; ++w)
            if (pr.first.x[w] != Qcan.x[w] || pr.first.z[w] != Qcan.z[w]) same = false;
        if (same) {
            int ne = (pr.second + dcan) % 8;
            if (ne == 0) {
                pr = rots.back();
                rots.pop_back();
            } else {
                pr.second = ne;
            }
            return;
        }
    }
    rots.push_back({Qcan, dcan});
}

// ---------------------------------------------------------------------------
// propagate: R ← U·R·U†  for one gate.
// ---------------------------------------------------------------------------
bool PprResidual::propagate(GateKind g, const std::vector<int>& targets) {
    switch (g) {
        // ---- Clifford gates: conjugate pauli and every axis ----
        case GateKind::X:
        case GateKind::Y:
        case GateKind::Z:
        case GateKind::S:
        case GateKind::SDG:
        case GateKind::H:
        case GateKind::CX:
        case GateKind::CZ: {
            conj_cliff(pauli, g, targets);
            for (auto& pr : rots) {
                conj_cliff(pr.first, g, targets);
                canonicalize_axis(pr.first, pr.second);
            }
            return true;
        }

        // ---- T: generator Z_t, rotation coefficient -1 ----
        // T = e^{iπ/8} exp(-iπ/8 Z_t). The generator sign: c=-1.
        // If P anticommutes with Z_t: add exponent = -c mod 4 = 1.
        case GateKind::T: {
            int t0 = targets[0];
            Pauli Zt(n); Zt.setz(t0);

            // Check axes.
            for (auto& pr : rots) {
                if (Pauli::anticommute_bit(pr.first, Zt)) {
                    rejected = true;
                    return false;
                }
            }
            // If Pauli anticommutes, append rotation.
            if (Pauli::anticommute_bit(pauli, Zt))
                merge_rot(rots, Zt, 1);
            return true;
        }

        // ---- CS: generators Z_a(c=-1), Z_b(c=-1), Z_ab(c=+1) ----
        case GateKind::CS: {
            int a = targets[0], b = targets[1];
            Pauli Za(n); Za.setz(a);
            Pauli Zb(n); Zb.setz(b);
            Pauli Zab(n); Zab.setz(a); Zab.setz(b);

            struct Gen { const Pauli& Q; int delta; };
            // delta = -ε mod 8: ε=-1 (c=-1) → delta=1, ε=+1 (c=+1) → delta=7.
            Gen gens[3] = { {Za, 1}, {Zb, 1}, {Zab, 7} };

            for (auto& gen : gens) {
                // Check existing axes (including any just added).
                for (auto& pr : rots) {
                    if (Pauli::anticommute_bit(pr.first, gen.Q)) {
                        rejected = true;
                        return false;
                    }
                }
                if (Pauli::anticommute_bit(pauli, gen.Q))
                    merge_rot(rots, gen.Q, gen.delta);
            }
            return true;
        }

        // ---- CCZ: 7 generators ----
        // Za:-1, Zb:-1, Zc:-1, Zab:+1, Zac:+1, Zbc:+1, Zabc:-1
        case GateKind::CCZ: {
            int a = targets[0], b = targets[1], c = targets[2];
            auto Zs = [&](std::initializer_list<int> qs) {
                Pauli p(n); for (int q : qs) p.setz(q); return p;
            };
            struct Gen { Pauli Q; int delta; };
            Gen gens[7] = {
                {Zs({a}),     1}, {Zs({b}),     1}, {Zs({c}),     1},
                {Zs({a,b}),   7}, {Zs({a,c}),   7}, {Zs({b,c}),   7},
                {Zs({a,b,c}), 1},
            };

            for (auto& gen : gens) {
                for (auto& pr : rots) {
                    if (Pauli::anticommute_bit(pr.first, gen.Q)) {
                        rejected = true;
                        return false;
                    }
                }
                if (Pauli::anticommute_bit(pauli, gen.Q))
                    merge_rot(rots, gen.Q, gen.delta);
            }
            return true;
        }

        // ---- CH: decompose as rot(+1,Y_t) · CZ(c,t) · rot(-1,Y_t) ----
        // Time order (first→last): rot(+1,Y_t), CZ(c,t), rot(-1,Y_t).
        // Propagate through each sub-op in that order.
        // KEEP IN LOCKSTEP with the compile-side decomposition (pauli_rotation_form.cpp,
        // case GateKind::CH): same three sub-ops, same sign convention; each side is
        // independently pinned by its dense-unitary oracle.
        case GateKind::CH: {
            int t = targets[1];   // control (targets[0]) enters only via the CZ sub-op
            // Y_t generator: i^1 · X_t Z_t (phase=1, xbit=1, zbit=1 on qubit t).
            Pauli Yt(n); Yt.setx(t); Yt.setz(t); Yt.phase = 1;

            // Sub-step 1: rot(+1, Y_t)  [c_coeff=+1, delta = 7]
            for (auto& pr : rots)
                if (Pauli::anticommute_bit(pr.first, Yt)) { rejected = true; return false; }
            if (Pauli::anticommute_bit(pauli, Yt))
                merge_rot(rots, Yt, 7);

            // Sub-step 2: CZ(c, t) — Clifford conjugation of all stored Paulis.
            conj_cliff(pauli, GateKind::CZ, targets);
            for (auto& pr : rots) {
                conj_cliff(pr.first, GateKind::CZ, targets);
                canonicalize_axis(pr.first, pr.second);
            }

            // Sub-step 3: rot(-1, Y_t)  [c_coeff=-1, delta = 1]
            // Generator is the original Y_t (from the CH decomposition),
            // not the CZ-conjugated version stored in rots.
            for (auto& pr : rots)
                if (Pauli::anticommute_bit(pr.first, Yt)) { rejected = true; return false; }
            if (Pauli::anticommute_bit(pauli, Yt))
                merge_rot(rots, Yt, 1);

            return true;
        }

        default:
            // Unknown gate: refuse loudly rather than silently treating it as identity.
            // Unreachable today (every GateKind is handled above); this guards gate-set growth.
            rejected = true;
            return false;
    }
}

// ---------------------------------------------------------------------------
// Seeds
// ---------------------------------------------------------------------------
PprResidual ppr_seed_x(int n, int q) {
    PprResidual R;
    R.n = n;
    R.pauli = Pauli(n);
    R.pauli.setx(q);
    return R;
}

PprResidual ppr_seed_z(int n, int q) {
    PprResidual R;
    R.n = n;
    R.pauli = Pauli(n);
    R.pauli.setz(q);
    return R;
}

// ---------------------------------------------------------------------------
// clifford_tableau(): recompile Π_j exp(i(π/4) e_j A_j) into a CliffordTableau.
//
// Each factor exp(i(π/4) e_j A_j) is realized as a basis-changed S† gate:
//   pick W with W A_j W† = Z_{q0}  (q0 = first support qubit of A_j)
//   then exp(i(π/4) A_j) = e^{iπ/4} W† S†_{q0} W  (up to global phase)
// We apply S† e_j times for the e_j-th power.
//
// Building W:
//   Phase 1: single-qubit rotations to diagonalize each qubit's Pauli component to Z.
//     X_q -> Z_q: apply H_q  (H X H = Z)
//     Y_q -> Z_q: apply S†_q then H_q
//       Reason: S†YS = X (verified: S†_q Y_q S_q = X_q), then H X H = Z.
//       So (H S†) Y (H S†)† = H S† Y S H = H X H = Z. ✓
//     Z_q -> Z_q: no gate needed.
//   Phase 2: CX ladder to fold all Z support into Z_{q0}.
//     For each q != q0 in support: apply CX(q, q0).
//     CX(ctrl=q, tgt=q0) conjugation: Z_{q0} -> Z_q Z_{q0}.
//     For product Z_{q0} Z_{q1}: apply CX(q1,q0) -> Z_{q1}Z_{q0}*Z_{q1} = Z_{q0}. ✓
//   W† (inverse): Phase 2 reverse (CX self-inverse), Phase 1 reverse (H^{-1}=H, (S†H)^{-1}=HS).
//
// Tableau accumulation (left_* means U <- G*U):
//   Apply W gates, then S†^{e_j} on q0, then W† gates.
//   Net: U <- (W† S†^{e_j} W) * U.
// ---------------------------------------------------------------------------
CliffordTableau PprResidual::clifford_tableau() const {
    CliffordTableau U(n);

    for (auto& pr : rots) {
        const Pauli& A = pr.first;
        int e = pr.second;  // exponent in 1..7

        // Find q0 and support.
        int q0 = -1;
        std::vector<int> support;
        for (int q = 0; q < n; ++q) {
            if (A.xbit(q) || A.zbit(q)) {
                support.push_back(q);
                if (q0 < 0) q0 = q;
            }
        }
        if (q0 < 0) continue;  // identity axis

        // Apply W: Phase 1 single-qubit diagonalization.
        for (int q : support) {
            bool xb = A.xbit(q), zb = A.zbit(q);
            if (xb && !zb) {
                U.left_h(q);
            } else if (xb && zb) {
                // Y_q -> Z_q via S†_q then H_q.
                U.left_sdg(q);
                U.left_h(q);
            }
            // Z only: no gate.
        }

        // Apply W: Phase 2 CX ladder.
        for (int q : support) {
            if (q == q0) continue;
            U.left_cx(q, q0);
        }

        // Apply S†^{e} on q0.
        // invariant: stored axes are canonical phase∈{0,1} (see propagate),
        // so W maps A → Z_{q0} and exp(i(π/4) e A) = W† S†^{e} W directly.
        for (int k = 0; k < e; ++k)
            U.left_sdg(q0);

        // Apply W†: Phase 2 inverse (reverse order, CX self-inverse).
        for (int i = (int)support.size() - 1; i >= 0; --i) {
            int q = support[i];
            if (q == q0) continue;
            U.left_cx(q, q0);
        }

        // Apply W†: Phase 1 inverse (reverse order, conjugate gates).
        // H^{-1} = H; (S†H)^{-1} = H^{-1}(S†)^{-1} = HS.
        for (int i = (int)support.size() - 1; i >= 0; --i) {
            int q = support[i];
            bool xb = A.xbit(q), zb = A.zbit(q);
            if (xb && !zb) {
                U.left_h(q);
            } else if (xb && zb) {
                U.left_h(q);
                U.left_s(q);
            }
        }
    }

    return U;
}

// ---------------------------------------------------------------------------
// to_error_tableau(): full operator E = pauli · (Π rots Clifford).
//
// Start from clifford_tableau() to get U (the rotation part), then prepend the
// Pauli component by applying left_x/left_z for each bit set in `pauli`.
// X bits are applied first, then Z bits; the order is irrelevant for different
// qubits (they commute), and for the same qubit produces iY instead of -iY —
// a global phase difference that is unobservable in conjugation.
// ---------------------------------------------------------------------------
CliffordTableau PprResidual::to_error_tableau() const {
    CliffordTableau U = clifford_tableau();
    for (int q = 0; q < n; ++q) {
        if (pauli.xbit(q)) U.left_x(q);
    }
    for (int q = 0; q < n; ++q) {
        if (pauli.zbit(q)) U.left_z(q);
    }
    return U;
}

}  // namespace qeccore
