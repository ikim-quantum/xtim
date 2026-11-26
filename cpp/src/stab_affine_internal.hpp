#pragma once
// Internal test seam for stab_affine.cpp. NOT a public header: only stab_affine.cpp and the
// ctest cpp/tests/test_restrict_to_support.cpp include it. The symbol has hidden visibility,
// so it is not exported from a shared object (_xtim).
#include <cstdint>
#include <vector>
#include "qeccore/stab_affine.hpp"

namespace qeccore {
namespace affine_detail {

#if defined(__GNUC__) || defined(__clang__)
#define QECCORE_AFFINE_INTERNAL __attribute__((visibility("hidden")))
#else
#define QECCORE_AFFINE_INTERNAL
#endif

// canonicalize()'s single-kernel-vector routine on an n×k R. Returns false iff R has full column
// rank; else fills v (length k). `force_augmented` = the augmented Gauss–Jordan oracle path (the
// production path uses back-substitution when the nullity is exactly 1).
QECCORE_AFFINE_INTERNAL bool kernel_vec_for_test(const PackedMat& R, int n, int k,
                                                 std::vector<uint8_t>& v, bool force_augmented);

}  // namespace affine_detail
}  // namespace qeccore
