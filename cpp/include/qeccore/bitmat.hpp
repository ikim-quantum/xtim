#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace qeccore {

using Word = uint64_t;
inline int nwords(int ncols) { return (ncols + 63) / 64; }

// Row-major bit matrix: row i occupies words [i*words, (i+1)*words);
// bit j is word (j>>6), position (j&63), LSB-first. High bits past ncols are 0.
struct GF2Mat {
    int rows = 0, ncols = 0, words = 0;
    std::vector<Word> data;

    GF2Mat() = default;
    GF2Mat(int r, int c)
        : rows(r), ncols(c), words(nwords(c)),
          data(static_cast<size_t>(r) * nwords(c), 0) {}

    Word* row(int i) { return data.data() + static_cast<size_t>(i) * words; }
    const Word* row(int i) const {
        return data.data() + static_cast<size_t>(i) * words;
    }
    bool get(int i, int j) const {
        return (row(i)[j >> 6] >> (j & 63)) & 1ULL;
    }
    void set1(int i, int j) { row(i)[j >> 6] |= (1ULL << (j & 63)); }
};

int rank(const GF2Mat& m);
GF2Mat row_reduce(const GF2Mat& m);   // independent RREF rows (rank x ncols)

}  // namespace qeccore
