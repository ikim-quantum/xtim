#include "qeccore/bitmat.hpp"
#include <algorithm>

namespace qeccore {

GF2Mat from_dense(const std::vector<std::vector<uint8_t>>& m) {
    int r = (int)m.size();
    int c = r ? (int)m[0].size() : 0;
    GF2Mat out(r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            if (m[i][j] & 1) out.set1(i, j);
    return out;
}

std::vector<std::vector<uint8_t>> to_dense(const GF2Mat& m) {
    std::vector<std::vector<uint8_t>> out(m.rows, std::vector<uint8_t>(m.ncols, 0));
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.ncols; ++j) out[i][j] = m.get(i, j) ? 1 : 0;
    return out;
}

int popcount_row(const Word* r, int words) {
    int s = 0;
    for (int w = 0; w < words; ++w) s += __builtin_popcountll(r[w]);
    return s;
}

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

// Port of gf2.py gf2_kernel: reduce aug=[H^T | I_n] over the first m columns,
// return the trailing rows' I_n block (the null-space basis).
GF2Mat kernel(const GF2Mat& H) {
    int m = H.rows, n = H.ncols;
    GF2Mat aug(n, m + n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            if (H.get(i, j)) aug.set1(j, i);     // H^T
    for (int j = 0; j < n; ++j) aug.set1(j, m + j);  // I_n
    int r = 0;
    for (int c = 0; c < m && r < n; ++c) {
        int piv = -1;
        for (int i = r; i < n; ++i)
            if (aug.get(i, c)) { piv = i; break; }
        if (piv < 0) continue;
        if (piv != r)
            for (int w = 0; w < aug.words; ++w) std::swap(aug.row(r)[w], aug.row(piv)[w]);
        for (int i = 0; i < n; ++i)
            if (i != r && aug.get(i, c)) {
                for (int w = 0; w < aug.words; ++w) aug.row(i)[w] ^= aug.row(r)[w];
            }
        ++r;
    }
    GF2Mat K(n - r, n);
    for (int i = r; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (aug.get(i, m + j)) K.set1(i - r, j);
    return K;
}

// Port of min_weight.py _systematic_disjoint_forms.
std::vector<GF2Mat> systematic_disjoint_forms(const GF2Mat& G) {
    int k = G.rows, N = G.ncols;
    std::vector<char> used(N, 0);
    std::vector<GF2Mat> forms;
    while (true) {
        GF2Mat A = G;
        std::vector<int> pivots;
        int r = 0;
        for (int c = 0; c < N; ++c) {
            if (used[c]) continue;
            int piv = -1;
            for (int i = r; i < k; ++i)
                if (A.get(i, c)) { piv = i; break; }
            if (piv < 0) continue;
            if (piv != r)
                for (int w = 0; w < A.words; ++w) std::swap(A.row(r)[w], A.row(piv)[w]);
            for (int i = 0; i < k; ++i)
                if (i != r && A.get(i, c)) {
                    for (int w = 0; w < A.words; ++w) A.row(i)[w] ^= A.row(r)[w];
                }
            pivots.push_back(c);
            ++r;
            if (r == k) break;
        }
        if (r < k) break;
        forms.push_back(std::move(A));
        for (int c : pivots) used[c] = 1;
    }
    return forms;
}

}  // namespace qeccore
