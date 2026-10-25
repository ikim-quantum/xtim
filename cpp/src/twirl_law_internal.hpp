#pragma once
// Internal header of twirl_law.cpp (pattern: stab_affine_internal.hpp): the core entry (build_shot_law_core) for
// cpp/src/record_space_b.cpp and twirl_kernel_sampler.cpp. Hidden visibility: not exported from _xtim. (J-b2 J7b:
// the σ kernel section and its test seams are deleted.)
#include <cstdint>
#include <vector>

#include "qeccore/twirl_kernel.hpp"

namespace qeccore {
namespace twirl_law_detail {

#if defined(__GNUC__) || defined(__clang__)
#define QECCORE_TWIRL_LAW_INTERNAL __attribute__((visibility("hidden")))
#else
#define QECCORE_TWIRL_LAW_INTERNAL
#endif

// J-a N6 (record-space B, switched OFF in production; cpp/src/record_space_b.cpp is the only caller):
// build_shot_law's own body run WITHOUT the κ>0 σ-fold stage (qrows, W = C_det^⊥, the ng-wide preimage
// solves, kernel_dual_basis and its asserts) and without kernel_base. The returned law carries tier1, r,
// κ, det_signs, coin_masks and kernel_logicals exactly as build_shot_law's; kernel_masks / kernel_pre /
// kernel_base / kernel_foldable are EMPTY. `core` receives the record-space inputs of the same build:
struct ShotLawCore {
    int n = 0, ng = 0;
    std::vector<int> aidx;                         // active generator indices, ascending
    std::vector<std::vector<uint64_t>> v;          // per active generator: its dressing v_i (n-bit)
    std::vector<uint8_t> phi;                      // per active generator: the phase of h_i (h-free form)
    std::vector<std::vector<uint64_t>> D;          // a basis of D = V ∩ G_z (n-bit)
    std::vector<std::vector<uint64_t>> U;          // the kernel directions u_j (n-bit); L_j = law.kernel_logicals[j]
};
QECCORE_TWIRL_LAW_INTERNAL ShotLaw build_shot_law_core(const CertifiedGroupPlanes& G, const DiagNormalForm& nf,
                                                       ShotLawCore& core);

}  // namespace twirl_law_detail
}  // namespace qeccore
