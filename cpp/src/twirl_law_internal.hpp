#pragma once
// Internal test seam for twirl_law.cpp. NOT a public header: only twirl_law.cpp and the ctest
// cpp/tests/test_kernel_dual_basis.cpp include it (pattern: stab_affine_internal.hpp). The symbol
// has hidden visibility, so it is not exported from a shared object (_xtim).
#include <cstdint>
#include <vector>

namespace qeccore {
namespace twirl_law_detail {

#if defined(__GNUC__) || defined(__clang__)
#define QECCORE_TWIRL_LAW_INTERNAL __attribute__((visibility("hidden")))
#else
#define QECCORE_TWIRL_LAW_INTERNAL
#endif

// The canonical dual-basis solver of build_shot_law's kernel section on raw rows: W =
// nullspace(cdet_rows), coin RREF = rref(coin_rows); returns κ fold masks with
// preimages[j]·δ_k = δ_jk, each in W and reduced mod the coin RREF. Throws std::logic_error when
// the preimages are dependent on W (Lemma B says the production call site cannot be).
QECCORE_TWIRL_LAW_INTERNAL std::vector<std::vector<uint64_t>> kernel_dual_basis_for_test(
    const std::vector<std::vector<uint64_t>>& preimages,
    const std::vector<std::vector<uint64_t>>& cdet_rows,
    const std::vector<std::vector<uint64_t>>& coin_rows, int ng);

}  // namespace twirl_law_detail
}  // namespace qeccore
