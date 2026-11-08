#pragma once
// Loud bare-state refusals.  The bare state is Σ_{σ<χ} c_σ |φ_σ⟩ with χ = 2^r branches, r the
// branch rank of the reduced magic (after FastTODD).  Two failure modes must never degrade into a
// silently WRONG state (the 3.1.7 defect: `1 << r` wrapped mod 32 for r > 30, so a rank-33 circuit
// built a truncated χ = 2 state of norm² 2^-32 that the exact fallback then sampled):
//
//   * BareStateCapacityError — χ = 2^r cannot be represented (r > kMaxBareRank: branch indices,
//     CanonicalStabSum::chi() and every per-branch loop are `int`) or its allocation failed.
//     Translated to Python `xtim.XtimCapacityError` (a MemoryError), which engine="auto" does NOT
//     swallow: no other engine can represent the state either.
//   * BareStateInvariantError — the assembled state violates an invariant every valid bare state
//     satisfies (its squared norm Σ|c_σ|² ≠ 1: the branches are orthonormal, so this is exact up to
//     rounding).  Translated to Python `xtim.XtimInvariantError`.
#include <stdexcept>
#include <string>

namespace qeccore {

// Largest branch rank the bare-state builders accept: χ = 2^30 is the largest power of two an
// `int` branch count holds.  (Memory runs out well before this on most hosts; that case surfaces
// as a BareStateCapacityError from the allocation guard instead.)
constexpr int kMaxBareRank = 30;

struct BareStateCapacityError : std::overflow_error {
    using std::overflow_error::overflow_error;
};

struct BareStateInvariantError : std::logic_error {
    using std::logic_error::logic_error;
};

// The refusal message for a branch rank above kMaxBareRank (shared by every builder that forms χ).
inline std::string bare_rank_refusal(int r, const char* where) {
    return std::string(where) + ": the bare state needs chi = 2^" + std::to_string(r) +
           " stabilizer branches (magic branch rank r = " + std::to_string(r) +
           " after T-count reduction); the engine limit is chi <= 2^" +
           std::to_string(kMaxBareRank) + " (r <= " + std::to_string(kMaxBareRank) +
           "). xtim cannot represent this circuit's magic as a sum over stabilizer branches.";
}

}  // namespace qeccore
