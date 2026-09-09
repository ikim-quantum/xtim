#include "qeccore/residue_projection.hpp"
#include <cstddef>

namespace qeccore {

int popcount_row(const Row& r) {
    int c = 0;
    for (uint64_t w : r) c += __builtin_popcountll(w);
    return c;
}
bool row_zero(const Row& r) {
    for (uint64_t w : r) if (w) return false;
    return true;
}
Row row_xor(const Row& a, const Row& b) {
    Row r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] ^ b[i];
    return r;
}

void moebius(std::vector<uint8_t>& h, int r) {
    const size_t N = (size_t)1 << r;
    for (int b = 0; b < r; ++b) {
        const size_t bit = (size_t)1 << b;
        for (size_t x = 0; x < N; ++x)
            if (x & bit) h[x] = (uint8_t)((h[x] - h[x ^ bit]) & 7);
    }
}

}  // namespace qeccore
