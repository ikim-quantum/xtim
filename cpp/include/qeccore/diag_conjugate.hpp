#pragma once
// Single source of truth for conjugating a Pauli through a DIAGONAL Clifford factor
// Delta = diag(S^a) · ∏ CZ — i.e. the Delta†·P·Delta rule, P = i^ph X^x Z^z.
//
// Delta is diagonal so it fixes P's X-support; each S-power and CZ-arm contributes a leftover
// Z and/or an i-phase, accumulated into a Z-mask (the Z bits to XOR into P.z) and a dphase
// (the i-exponent delta, mod 4). Both the dense path (conjugate_through over a DiagPauliClifford
// in dem_export) and the per-shot SPARSE fold (sampler's cascade) walk the SAME two rules below —
// previously hand-copied across those sites, each carrying the same "CZ both-endpoint sign"
// comment after that sign was a latent bug (caught by the d3 high-weight campaign, 2026-06-12).
// Routing every site through these inlines makes that rule live in exactly one place.
#include <cstdint>

#include "qeccore/pauli.hpp"

namespace qeccore {

// Conjugate P through diag(S^a) on qubit q, the Delta†·P·Delta direction:
//   S^{-a} X S^a = i^{-a} X Z^a   ⇒   leftover Z^{a mod 2} on q, phase i^{-a}.
// No-op unless P has X on q. Mutates the Z-mask (P.z-width words) and the i-exponent in place.
inline void diag_spow_conjugate(const Pauli& P, int q, int a, uint64_t* zmask, int& dphase) {
    if (!P.xbit(q)) return;
    dphase = (dphase + ((4 - (a & 3)) & 3)) & 3;
    if (a & 1) zmask[q >> 6] ^= 1ull << (q & 63);
}

// Conjugate P through a single CZ on the pair (qa, qb):
//   X on an endpoint propagates a Z onto the partner; X on BOTH endpoints carries the
//   CZ(X⊗X)CZ = −(X⊗X)(Z⊗Z) sign — the +2 i-phase. Call ONCE per pair (qa < qb).
// THE recurring latent-sign-bug site — kept here, exactly once.
inline void diag_cz_conjugate(const Pauli& P, int qa, int qb, uint64_t* zmask, int& dphase) {
    const bool xa = P.xbit(qa), xb = P.xbit(qb);
    if (xa) zmask[qb >> 6] ^= 1ull << (qb & 63);
    if (xb) zmask[qa >> 6] ^= 1ull << (qa & 63);
    if (xa && xb) dphase = (dphase + 2) & 3;
}

}  // namespace qeccore
