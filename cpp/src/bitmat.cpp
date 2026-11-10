#include "qeccore/bitmat.hpp"
#include <algorithm>

namespace qeccore {

// Full RREF in place, column-ascending pivots (matches gf2.py). Returns rank.
static int rref_inplace(GF2Mat& a) {
    int r = 0;
    for (int c = 0; c < a.ncols && r < a.rows; ++c) {
        int piv = -1;
        for (int i = r; i < a.rows; ++i)
            if (a.get(i, c)) { piv = i; break; }
        if (piv < 0) continue;
        if (piv != r)
            for (int w = 0; w < a.words; ++w) std::swap(a.row(r)[w], a.row(piv)[w]);
        for (int i = 0; i < a.rows; ++i) {
            if (i != r && a.get(i, c)) {
                Word* x = a.row(i);
                const Word* y = a.row(r);
                for (int w = 0; w < a.words; ++w) x[w] ^= y[w];
            }
        }
        ++r;
    }
    return r;
}

int rank(const GF2Mat& m) {
    GF2Mat a = m;
    return rref_inplace(a);
}

GF2Mat row_reduce(const GF2Mat& m) {
    GF2Mat a = m;
    int r = rref_inplace(a);
    GF2Mat out(r, m.ncols);
    for (int i = 0; i < r; ++i)
        for (int w = 0; w < a.words; ++w) out.row(i)[w] = a.row(i)[w];
    return out;
}

}  // namespace qeccore
