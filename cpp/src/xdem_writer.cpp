// xdem_writer.cpp — the xdem writer (SPEC §2): the writer model (skeleton, location groups, raw axes, kappa, b, the
// diagonality check), the fault and strict tau columns (one backward sweep), t_c and the canonical-form check, the
// canonical coordinates, bhat, the E split, folding and the strict line merge. See xdem_writer.hpp for the
// construction and its correspondence to the reference model (dccore.py / writer.py).
#include "qeccore/xdem_writer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "qeccore/clifford_tableau.hpp"

namespace qeccore::xdem {

namespace {

using Words = std::vector<uint64_t>;

bool getb(const Words& w, size_t i) { return (w[i >> 6] >> (i & 63)) & 1ull; }
void flipb(Words& w, size_t i) { w[i >> 6] ^= 1ull << (i & 63); }
void xorw(Words& a, const Words& b) { for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i]; }
bool zero(const Words& a) {
    for (uint64_t x : a) if (x) return false;
    return true;
}
int hibit(const Words& a) {
    for (size_t i = a.size(); i-- > 0;)
        if (a[i]) return (int)(i * 64 + 63 - (size_t)__builtin_clzll(a[i]));
    return -1;
}

// Hermitian sign of i^phase X^x Z^z: +1 (phase == #Y), -1 (phase == #Y + 2); 0 = not Hermitian.
int herm_sign(const Pauli& P) {
    const int d = (P.phase - P.xz_overlap()) & 3;
    return d == 0 ? 1 : d == 2 ? -1 : 0;
}

// The sign routine: an operator already pulled back to the start of the skeleton (M = U^dag P U) is +- a stabilizer
// of S's final state iff M has no X part and a real phase (i^phase Z^z |0> = i^phase |0>). 1 iff <P> = -1, 0 iff
// <P> = +1, -1 iff P is not +- a stabilizer.
int pulled_sign(const Pauli& M) {
    if (!zero(M.x) || (M.phase & 1)) return -1;
    return M.phase == 2 ? 1 : 0;
}

const char* gate_name(GateKind g, bool dag) {
    switch (g) {
        case GateKind::T: return dag ? "T_DAG" : "T";
        case GateKind::CS: return dag ? "CS_DAG" : "CS";
        case GateKind::CCZ: return "CCZ";
        case GateKind::CH: return "CH";
        default: return "?";
    }
}
bool is_nonclifford(GateKind g) {
    return g == GateKind::T || g == GateKind::CS || g == GateKind::CCZ || g == GateKind::CH;
}

std::vector<int> parity_reduced(std::vector<int> v) {
    parity_reduce_sorted(v);
    return v;
}

// ── residue tables (port of the reference residue_terms) ──────────────────────────────────────────────────────────
const char* const kGates[5] = {"T", "T_DAG", "CS", "CS_DAG", "CCZ"};
const int kArity[5] = {1, 1, 2, 2, 3};

int gate_index(const std::string& g) {
    for (int i = 0; i < 5; ++i) if (g == kGates[i]) return i;
    return -1;
}

// PHASE[gate](z) in units of pi/4 (reference PHASE dict).
int phase_of(int gi, const std::vector<int>& z) {
    switch (gi) {
        case 0: return z[0];
        case 1: return -z[0];
        case 2: return 2 * z[0] * z[1];
        case 3: return -2 * z[0] * z[1];
        default: return 4 * z[0] * z[1] * z[2];
    }
}

inline int mod(int a, int m) { return ((a % m) + m) % m; }

// Brute force over e in Z4^(2^m - 1), weight-stable lexicographic.
std::vector<std::pair<int, int>> compute_residue(int gi, uint32_t x) {
    const int m = kArity[gi];
    const int nz = 1 << m;
    // zs = itertools.product([0,1], repeat=m): z[0] most significant in enumeration order.
    std::vector<std::vector<int>> zs;
    for (int t = 0; t < nz; ++t) {
        std::vector<int> z((size_t)m);
        for (int i = 0; i < m; ++i) z[(size_t)i] = (t >> (m - 1 - i)) & 1;
        zs.push_back(z);
    }
    std::vector<int> g, h;
    for (const auto& z : zs) {
        std::vector<int> zx = z;
        for (int i = 0; i < m; ++i) zx[(size_t)i] ^= (int)((x >> i) & 1u);
        g.push_back(mod(phase_of(gi, zx) - phase_of(gi, z), 8));
    }
    for (int gv : g) {
        const int hv = mod(gv - g[0], 8);
        if (hv % 2) throw std::logic_error("xdem residue_terms: odd phase difference");
        h.push_back(hv / 2);
    }
    std::vector<int> zint;
    for (const auto& z : zs) {
        int v = 0;
        for (int i = 0; i < m; ++i) v |= z[(size_t)i] << i;
        zint.push_back(v);
    }
    const int L = nz - 1;   // vs = 1..2^m-1
    const long total = 1L << (2 * L);
    for (int wt = 0; wt <= L; ++wt) {
        for (long t = 0; t < total; ++t) {
            int es[7];
            int w = 0;
            for (int i = 0; i < L; ++i) {
                es[i] = (int)((t >> (2 * (L - 1 - i))) & 3);
                w += es[i] != 0;
            }
            if (w != wt) continue;
            bool ok = true;
            for (int zi = 0; zi < nz && ok; ++zi) {
                int s = 0;
                for (int i = 0; i < L; ++i) s += es[i] * (__builtin_popcount((unsigned)((i + 1) & zint[(size_t)zi])) & 1);
                ok = mod(s, 4) == h[(size_t)zi];
            }
            if (ok) {
                std::vector<std::pair<int, int>> out;
                for (int i = 0; i < L; ++i) if (es[i]) out.emplace_back(i + 1, es[i]);
                return out;
            }
        }
    }
    throw std::logic_error("xdem residue_terms: no solution");
}

const std::vector<std::pair<int, int>>& residue_by_index(int gi, uint32_t x) {
    static const std::array<std::array<std::vector<std::pair<int, int>>, 8>, 5> table = [] {
        std::array<std::array<std::vector<std::pair<int, int>>, 8>, 5> t;
        for (int g = 0; g < 5; ++g)
            for (uint32_t xx = 1; xx < (1u << kArity[g]); ++xx) t[(size_t)g][xx] = compute_residue(g, xx);
        return t;
    }();
    return table[(size_t)gi][x];
}

// ── the backward sweep (shared by the fault columns and the strict tau columns) ──────────────────────────────────
using SweepAt = std::vector<std::vector<std::array<std::vector<int32_t>, 2>>>;

// LM-3: the oracle lever XTIM_SWEEP_LAYOUT=sparse|dense (a TEST lever, header: SweepLayout): it forces an Auto request
// to one side; unset or any other value = the automatic switch. Pure performance (every layout gives identical
// columns: the byte oracle runs under each value). Read once per process (a function-local static): tests that toggle
// it need a subprocess.
SweepLayout env_sweep_layout() {
    static const SweepLayout L = [] {
        const char* e = std::getenv("XTIM_SWEEP_LAYOUT");
        if (e && !std::strcmp(e, "sparse")) return SweepLayout::Sparse;
        if (e && !std::strcmp(e, "dense")) return SweepLayout::Dense;
        return SweepLayout::Auto;
    }();
    return L;
}

// The sweep's column store (LM-3, header: SweepLayout): per wire, the columns with X / with Z there. Sparse: each row
// a sorted list, or its own bit row when the list is long (fix round 1); dense: the wires x columns planes. Every form
// holds the same sets after every operation, so the noise slots read off any of them are identical; the sparse store
// moves to the planes at most once (go_dense).
class ColumnStore {
  public:
    ColumnStore(int N, size_t CW, bool dense)
        : N_(N), CW_(CW), up_(std::max<size_t>((size_t)kSweepRowMinList, CW / (size_t)kSweepRowListDiv)),
          down_(up_ / 4) {
        if (dense) go_dense();
        else { ls_.resize(2 * (size_t)N); did_.assign(2 * (size_t)N, -1); }
    }
    bool dense() const { return dense_; }
    long long nnz() const { return nnz_; }
    long long bit_rows() const { return bit_rows_; }
    long long bit_rows_made() const { return made_; }
    // the gates' work so far (tests): list entries merged / moved plus bit-row and plane words touched
    long long work() const { return work_; }
    // the sparse store's bytes: list capacities, the scratch list, the bit-row pool, and per row a header + slot id
    double bytes() const {
        return 4.0 * (double)(cap_ + (long long)tmp_.capacity()) + 8.0 * (double)pool_.capacity() +
               (double)ls_.size() * (double)(sizeof(std::vector<int32_t>) + sizeof(int32_t));
    }
    double planes_bytes() const { return 16.0 * (double)N_ * (double)CW_; }
    // Auto's rule: the planes' bytes are at most kSweepDenseRatio times the sparse store's
    bool planes_pay() const { return planes_bytes() <= (double)kSweepDenseRatio * bytes(); }
    void go_dense() {
        X_.assign((size_t)N_ * CW_, 0);
        Z_.assign((size_t)N_ * CW_, 0);
        for (size_t r = 0; r < ls_.size(); ++r) {
            uint64_t* p = ((r & 1) ? Z_ : X_).data() + (r >> 1) * CW_;
            if (did_[r] >= 0) std::copy_n(bits(r), CW_, p);
            else for (int32_t c : ls_[r]) p[(size_t)c >> 6] |= 1ull << (c & 63);
        }
        std::vector<std::vector<int32_t>>().swap(ls_);
        std::vector<int32_t>().swap(did_);
        std::vector<uint64_t>().swap(pool_);
        std::vector<int32_t>().swap(free_);
        std::vector<int32_t>().swap(tmp_);
        nnz_ = cap_ = bit_rows_ = 0;
        dense_ = true;
    }
    // activation at the end: column c (ascending across calls) gets its X / Z part on wire w
    void set_end(bool zpart, int w, int c) {
        if (dense_) { (zpart ? Z_ : X_)[(size_t)w * CW_ + ((size_t)c >> 6)] |= 1ull << (c & 63); return; }
        const size_t r = 2 * (size_t)w + (zpart ? 1 : 0);
        if (did_[r] >= 0) { bits(r)[(size_t)c >> 6] |= 1ull << (c & 63); return; }
        const size_t cap0 = ls_[r].capacity(), n0 = ls_[r].size();
        ls_[r].push_back((int32_t)c);
        settle(r, cap0, n0);
    }
    void H(int a) {
        if (dense_) {
            uint64_t* x = xr(a); uint64_t* z = zr(a);
            for (size_t w = 0; w < CW_; ++w) std::swap(x[w], z[w]);
            work_ += 2 * (long long)CW_;
        } else {
            ls_[2 * (size_t)a].swap(ls_[2 * (size_t)a + 1]);
            std::swap(did_[2 * (size_t)a], did_[2 * (size_t)a + 1]);
        }
    }
    void S(int a) {                                  // X <-> Y: z ^= x
        if (dense_) {
            const uint64_t* x = xr(a); uint64_t* z = zr(a);
            for (size_t w = 0; w < CW_; ++w) z[w] ^= x[w];
            work_ += (long long)CW_;
        } else {
            xor_rows(2 * (size_t)a + 1, 2 * (size_t)a);
        }
    }
    void CX(int a, int b) {                          // X_a -> X_a X_b, Z_b -> Z_a Z_b
        if (dense_) {
            const uint64_t* xa = xr(a); uint64_t* xb = xr(b);
            uint64_t* za = zr(a); const uint64_t* zb = zr(b);
            for (size_t w = 0; w < CW_; ++w) { xb[w] ^= xa[w]; za[w] ^= zb[w]; }
            work_ += 2 * (long long)CW_;
        } else {
            xor_rows(2 * (size_t)b, 2 * (size_t)a);
            xor_rows(2 * (size_t)a + 1, 2 * (size_t)b + 1);
        }
    }
    void CZ(int a, int b) {                          // X_a -> X_a Z_b, X_b -> Z_a X_b
        if (dense_) {
            const uint64_t* xa = xr(a); const uint64_t* xb = xr(b);
            uint64_t* za = zr(a); uint64_t* zb = zr(b);
            for (size_t w = 0; w < CW_; ++w) { za[w] ^= xb[w]; zb[w] ^= xa[w]; }
            work_ += 2 * (long long)CW_;
        } else {
            xor_rows(2 * (size_t)a + 1, 2 * (size_t)b);
            xor_rows(2 * (size_t)b + 1, 2 * (size_t)a);
        }
    }
    void toggle(bool zpart, int w, int c) {          // one column on one wire
        if (dense_) {
            (zpart ? Z_ : X_)[(size_t)w * CW_ + ((size_t)c >> 6)] ^= 1ull << (c & 63);
            return;
        }
        const size_t r = 2 * (size_t)w + (zpart ? 1 : 0);
        if (did_[r] >= 0) { bits(r)[(size_t)c >> 6] ^= 1ull << (c & 63); return; }
        std::vector<int32_t>& v = ls_[r];            // at most up_ entries: the insert costs <= a bit row's words
        const size_t cap0 = v.capacity(), n0 = v.size();
        const auto it = std::lower_bound(v.begin(), v.end(), (int32_t)c);
        if (it != v.end() && *it == c) v.erase(it);
        else v.insert(it, (int32_t)c);
        work_ += (long long)n0;
        settle(r, cap0, n0);
    }
    void toggle_list(bool zpart, int w, const std::vector<int>& cs) {   // cs ascending and distinct
        if (dense_) {
            uint64_t* r = zpart ? zr(w) : xr(w);
            for (int c : cs) r[(size_t)c >> 6] ^= 1ull << (c & 63);
            return;
        }
        xor_list(2 * (size_t)w + (zpart ? 1 : 0), cs);
    }
    // the ascending columns with Z (zpart) / X at wire w
    std::vector<int32_t> row(bool zpart, int w) const {
        const uint64_t* r;
        if (dense_) {
            r = (zpart ? Z_ : X_).data() + (size_t)w * CW_;
        } else {
            const size_t k = 2 * (size_t)w + (zpart ? 1 : 0);
            if (did_[k] < 0) return ls_[k];
            r = bits(k);
        }
        std::vector<int32_t> v;
        for (size_t k = 0; k < CW_; ++k)
            for (uint64_t b = r[k]; b; b &= b - 1) v.push_back((int32_t)(k * 64 + (size_t)__builtin_ctzll(b)));
        return v;
    }
    // x_start: dense -> the X plane (moved out, CW words per wire); sparse -> per column c < keep, the wires holding
    // its X bit, ascending (CSR: keep + 1 offsets).
    void take_x(std::vector<uint64_t>& words, size_t& cw, std::vector<size_t>& off, std::vector<int32_t>& wire,
                int keep) {
        if (dense_) { words = std::move(X_); cw = CW_; return; }
        auto each = [&](size_t w, int lim, auto&& f) {   // wire w's X columns < lim, ascending
            const size_t r = 2 * w;
            if (did_[r] < 0) {
                for (int32_t c : ls_[r]) {
                    if (c >= lim) break;
                    f(c);
                }
                return;
            }
            const uint64_t* b = bits(r);
            for (size_t k = 0; k < CW_; ++k)
                for (uint64_t x = b[k]; x; x &= x - 1) {
                    const int32_t c = (int32_t)(k * 64 + (size_t)__builtin_ctzll(x));
                    if (c >= lim) return;
                    f(c);
                }
        };
        off.assign((size_t)keep + 1, 0);
        for (size_t w = 0; w < (size_t)N_; ++w) each(w, keep, [&](int32_t c) { ++off[(size_t)c + 1]; });
        for (size_t c = 1; c < off.size(); ++c) off[c] += off[c - 1];
        // dense supports (only possible through bit rows): the lists would outweigh the readout words, so x_start is
        // given in the dense form, the readout words (columns < kw * 64, as the planes' cut in build_fault_columns)
        const size_t kw = (size_t)std::max(1, (keep + 63) / 64);
        if (4 * off.back() > 8 * (size_t)N_ * kw) {
            std::vector<size_t>().swap(off);
            words.assign((size_t)N_ * kw, 0);
            cw = kw;
            for (size_t w = 0; w < (size_t)N_; ++w)
                each(w, (int)std::min(kw * 64, (size_t)INT32_MAX),
                     [&](int32_t c) { words[w * kw + ((size_t)c >> 6)] |= 1ull << (c & 63); });
            return;
        }
        wire.resize(off.back());
        std::vector<size_t> cur(off.begin(), off.end() - 1);
        for (size_t w = 0; w < (size_t)N_; ++w) each(w, keep, [&](int32_t c) { wire[cur[(size_t)c]++] = (int32_t)w; });
    }

  private:
    int N_;
    size_t CW_;
    size_t up_, down_;                               // a list above up_ entries becomes a bit row; back at <= down_
    bool dense_ = false;
    std::vector<uint64_t> X_, Z_;                    // dense: wire-major planes, CW_ words per wire
    // sparse: row r = 2 w + (Z part); ls_[r] its ascending column ids, or did_[r] >= 0 its bit row's slot in pool_
    std::vector<std::vector<int32_t>> ls_;
    std::vector<int32_t> did_;
    std::vector<uint64_t> pool_;                     // bit-row slots, CW_ words each
    std::vector<int32_t> free_;                      // released slots
    size_t slots_ = 0;
    std::vector<int32_t> tmp_;
    long long nnz_ = 0, cap_ = 0, bit_rows_ = 0, made_ = 0;   // list entries / capacities; bit rows live / made
    long long work_ = 0;
    uint64_t* xr(int w) { return X_.data() + (size_t)w * CW_; }
    uint64_t* zr(int w) { return Z_.data() + (size_t)w * CW_; }
    uint64_t* bits(size_t r) { return pool_.data() + (size_t)did_[r] * CW_; }
    const uint64_t* bits(size_t r) const { return pool_.data() + (size_t)did_[r] * CW_; }
    size_t popcount(size_t r) {
        const uint64_t* b = bits(r);
        work_ += (long long)CW_;
        size_t n = 0;
        for (size_t k = 0; k < CW_; ++k) n += (size_t)__builtin_popcountll(b[k]);
        return n;
    }
    // after a change to list r (capacity cap0 and size n0 before): keep its capacity within 2 x size + 16, account
    // the entries / capacities, and make it a bit row when it is long
    void settle(size_t r, size_t cap0, size_t n0) {
        std::vector<int32_t>& v = ls_[r];
        if (v.capacity() > 2 * v.size() + 16) std::vector<int32_t>(v.begin(), v.end()).swap(v);
        cap_ += (long long)v.capacity() - (long long)cap0;
        nnz_ += (long long)v.size() - (long long)n0;
        if (v.size() > up_) to_bits(r);
    }
    void to_bits(size_t r) {
        int32_t s;
        if (!free_.empty()) { s = free_.back(); free_.pop_back(); }
        else { s = (int32_t)slots_++; pool_.resize(slots_ * CW_); }
        did_[r] = s;
        uint64_t* b = bits(r);
        std::fill(b, b + CW_, 0ull);
        for (int32_t c : ls_[r]) b[(size_t)c >> 6] |= 1ull << (c & 63);
        work_ += (long long)(CW_ + ls_[r].size());
        cap_ -= (long long)ls_[r].capacity();
        nnz_ -= (long long)ls_[r].size();
        std::vector<int32_t>().swap(ls_[r]);
        ++bit_rows_;
        ++made_;
    }
    void to_list(size_t r) {
        std::vector<int32_t>& v = ls_[r];
        v.reserve(popcount(r));
        const uint64_t* b = bits(r);
        for (size_t k = 0; k < CW_; ++k)
            for (uint64_t x = b[k]; x; x &= x - 1) v.push_back((int32_t)(k * 64 + (size_t)__builtin_ctzll(x)));
        work_ += (long long)(CW_ + v.size());
        cap_ += (long long)v.capacity();
        nnz_ += (long long)v.size();
        free_.push_back(did_[r]);
        did_[r] = -1;
        --bit_rows_;
    }
    // row d ^= the ascending distinct ids src (src is not row d's own list)
    void xor_list(size_t d, const std::vector<int32_t>& src) {
        if (src.empty()) return;
        if (did_[d] >= 0) {
            uint64_t* b = bits(d);
            for (int32_t c : src) b[(size_t)c >> 6] ^= 1ull << (c & 63);
            work_ += (long long)src.size();
            return;
        }
        std::vector<int32_t>& dst = ls_[d];
        const size_t cap0 = dst.capacity(), n0 = dst.size();
        work_ += (long long)(n0 + src.size());
        if (dst.empty()) {
            dst = src;
        } else {
            tmp_.clear();
            std::set_symmetric_difference(dst.begin(), dst.end(), src.begin(), src.end(), std::back_inserter(tmp_));
            dst.swap(tmp_);                          // tmp_ keeps dst's old buffer (<= 2 x up_ + 16 entries)
        }
        settle(d, cap0, n0);
    }
    // row d ^= row s
    void xor_rows(size_t d, size_t s) {
        if (d == s) {                                // an aliased CX: the row cancels itself
            if (did_[d] >= 0) { std::fill(bits(d), bits(d) + CW_, 0ull); to_list(d); return; }
            const size_t cap0 = ls_[d].capacity(), n0 = ls_[d].size();
            ls_[d].clear();
            settle(d, cap0, n0);
            return;
        }
        if (did_[s] >= 0 && popcount(s) <= down_) to_list(s);
        if (did_[s] < 0) { xor_list(d, ls_[s]); return; }
        if (did_[d] < 0) to_bits(d);
        uint64_t* bd = bits(d);
        const uint64_t* bs = bits(s);
        for (size_t k = 0; k < CW_; ++k) bd[k] ^= bs[k];
        work_ += (long long)CW_;
        if (popcount(d) <= down_) to_list(d);
    }
};

// One backward sweep of the skeleton over C column Paulis (per wire, the X / Z parts of every column at the current
// backward time), from the end of the deferred stream to its start. A column is activated (XORed in)
//   * at the end: end[c] for c < end->size() (the column's end Pauli);
//   * at a read: read_cols[j] = the columns whose record set holds terminal read j, activated at its Measure;
//   * at a location (loc_base >= 0): Z on qubit i of group g is column loc_base + (offset of g) + i.
// Result: per deferred noise ordinal and qubit position, [0] the columns an X fault there anticommutes with (the
// columns with Z or Y at the qubit), [1] the columns a Z fault anticommutes with (X or Y). LM-3: the store's layout
// (ColumnStore; header: SweepLayout) changes no output. xo (optional): the X parts at the start (ColumnStore::take_x).
struct XStartOut {
    std::vector<uint64_t>* words; size_t* cw; std::vector<size_t>* off; std::vector<int32_t>* wire; int keep;
};
SweepAt backward_sweep(const Skeleton& sk, const Circuit& def, int C, const std::vector<Pauli>* end,
                       const std::vector<std::vector<int>>* read_cols, int loc_base, SweepLayout layout,
                       SweepStats* stats, const XStartOut* xo = nullptr) {
    if (sk.op_of_stream.size() != def.stream.size() + 1)
        throw WriterError("xdem writer: internal pipeline mismatch: the skeleton is not of this deferred stream");
    const size_t CW = (size_t)std::max(1, (C + 63) / 64);
    const int N = sk.N;
    if (layout == SweepLayout::Auto) layout = env_sweep_layout();
    ColumnStore st(N, CW, layout == SweepLayout::Dense);
    if (end)
        for (size_t c = 0; c < end->size(); ++c) {
            const Pauli& Q = (*end)[c];
            for (int zpart = 0; zpart < 2; ++zpart) {
                const std::vector<uint64_t>& qw = zpart ? Q.z : Q.x;
                for (size_t k = 0; k < qw.size(); ++k)
                    for (uint64_t b = qw[k]; b; b &= b - 1) {
                        const int w = (int)(k * 64 + (size_t)__builtin_ctzll(b));
                        if (w < N) st.set_end(zpart != 0, w, (int)c);
                    }
            }
        }
    long long peak = st.nnz(), dense_from = st.dense() ? 0 : -1, peak_bits = st.bit_rows();
    double peak_bytes = st.dense() ? st.planes_bytes() : st.bytes();
    const bool autoswitch = layout == SweepLayout::Auto;
    if (autoswitch && st.planes_pay()) { st.go_dense(); dense_from = 0; peak_bytes = st.planes_bytes(); }
    std::vector<int> lq_off;                          // group -> its first location column - loc_base
    if (loc_base >= 0) {
        lq_off.assign(sk.group_qubits.size() + 1, 0);
        for (size_t g = 0; g < sk.group_qubits.size(); ++g)
            lq_off[g + 1] = lq_off[g] + (int)sk.group_qubits[g].size();
    }
    // noise ordinals / read indices of the deferred stream
    std::vector<int> noise_ord(def.stream.size(), -1), read_ord(def.stream.size(), -1);
    int nn = 0, nr = 0;
    for (size_t i = 0; i < def.stream.size(); ++i) {
        if (def.stream[i].kind == Instr::Kind::Noise) noise_ord[i] = nn++;
        else if (def.stream[i].kind == Instr::Kind::Measure) read_ord[i] = nr++;
    }
    if (nr != (int)sk.reads.size())
        throw WriterError("xdem writer: internal pipeline mismatch: Measure count != terminal reads");
    SweepAt at((size_t)nn);
    auto bwd_op = [&](const Skeleton::Op& o) {
        switch (o.kind) {
            case Skeleton::H: st.H(o.a); break;
            case Skeleton::S:
            case Skeleton::SDG: st.S(o.a); break;
            case Skeleton::CX: st.CX(o.a, o.b); break;
            case Skeleton::CZ: st.CZ(o.a, o.b); break;
            case Skeleton::LOC: {                    // activate Z on each of the group's qubits (identity in S)
                if (loc_base < 0) break;
                const int g = o.a;
                const auto& qs = sk.group_qubits[(size_t)g];
                for (int i = 0; i < (int)qs.size(); ++i)
                    st.toggle(true, qs[(size_t)i], loc_base + lq_off[(size_t)g] + i);
                break;
            }
            default: break;                          // X, Y, Z: commutation unchanged
        }
    };
    for (size_t i = def.stream.size(); i-- > 0;) {
        for (int oi = sk.op_of_stream[i + 1]; oi-- > sk.op_of_stream[i];) bwd_op(sk.ops[(size_t)oi]);
        const Instr& ins = def.stream[i];
        if (read_ord[i] >= 0) {
            if (read_cols) {
                const auto& [b, w] = sk.reads[(size_t)read_ord[i]];
                const std::vector<int>& cs = (*read_cols)[(size_t)read_ord[i]];
                if (b == 0 || b == 1) st.toggle_list(false, w, cs);
                if (b == 2 || b == 1) st.toggle_list(true, w, cs);
            }
        } else if (noise_ord[i] >= 0) {
            auto& slot = at[(size_t)noise_ord[i]];
            slot.resize(ins.qubits.size());
            for (size_t qi = 0; qi < ins.qubits.size(); ++qi) {
                const int q = ins.qubits[qi];
                slot[qi][0] = st.row(true, q);       // an X fault anticommutes with the columns with Z (or Y) at q
                slot[qi][1] = st.row(false, q);      // a Z fault anticommutes with the columns with X (or Y) at q
            }
        }
        if (!st.dense()) {
            peak = std::max(peak, st.nnz());
            peak_bits = std::max(peak_bits, st.bit_rows());
            peak_bytes = std::max(peak_bytes, st.bytes());
            if (autoswitch && st.planes_pay()) {
                st.go_dense();
                dense_from = (long long)(def.stream.size() - i);
                peak_bytes = std::max(peak_bytes, st.planes_bytes());
            }
        }
    }
    const long long made = st.bit_rows_made();
    if (xo) st.take_x(*xo->words, *xo->cw, *xo->off, *xo->wire, xo->keep);
    if (stats)
        *stats = SweepStats{st.dense() ? 2 : 1, dense_from, peak, (long long)peak_bytes, peak_bits, made, st.work()};
    return at;
}

const char* noise_name(NoiseChannel c) {
    switch (c) {
        case NoiseChannel::X_ERROR: return "X_ERROR";
        case NoiseChannel::Y_ERROR: return "Y_ERROR";
        case NoiseChannel::Z_ERROR: return "Z_ERROR";
        case NoiseChannel::DEPOLARIZE1: return "DEPOLARIZE1";
        case NoiseChannel::DEPOLARIZE2: return "DEPOLARIZE2";
        case NoiseChannel::PAULI_CHANNEL_1: return "PAULI_CHANNEL_1";
        case NoiseChannel::PAULI_CHANNEL_2: return "PAULI_CHANNEL_2";
        case NoiseChannel::CORRELATED: return "E/ELSE_CORRELATED_ERROR";
    }
    return "?";
}

std::string fmt_g(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%g", v);
    return b;
}

Words pauli_xz_bits(const Pauli& P) {
    Words v = P.x;
    v.insert(v.end(), P.z.begin(), P.z.end());
    return v;
}

// A fully reduced GF(2) echelon with combination tracking (each row's pivot bit appears in no other row, so one pass
// in any order reduces a vector).
struct Echelon {
    std::vector<Words> rows, comb;
    std::vector<int> hb;
    bool reduce(Words& v, Words& c) const {          // v, c <- v + (rows), c + (their combinations); true iff v != 0
        for (size_t i = 0; i < rows.size(); ++i)
            if (getb(v, (size_t)hb[i])) { xorw(v, rows[i]); xorw(c, comb[i]); }
        return !zero(v);
    }
    void insert(Words v, Words c) {
        const int h = hibit(v);
        for (size_t i = 0; i < rows.size(); ++i)
            if (getb(rows[i], (size_t)h)) { xorw(rows[i], v); xorw(comb[i], c); }
        rows.push_back(std::move(v)); comb.push_back(std::move(c)); hb.push_back(h);
    }
};

}  // namespace

void parity_reduce_sorted(std::vector<int32_t>& v, size_t from) {
    std::sort(v.begin() + (std::ptrdiff_t)from, v.end());
    size_t o = from;
    for (size_t a = from; a < v.size();) {
        size_t b = a;
        while (b < v.size() && v[b] == v[a]) ++b;
        if ((b - a) & 1) v[o++] = v[a];
        a = b;
    }
    v.resize(o);
}

void conj_forward(Pauli& M, const Skeleton::Op& o) {
    switch (o.kind) {
        case Skeleton::H: CliffordTableau::conj_h(M, o.a); break;
        case Skeleton::S: CliffordTableau::conj_s(M, o.a); break;
        case Skeleton::SDG: CliffordTableau::conj_sdg(M, o.a); break;
        case Skeleton::X: CliffordTableau::conj_x(M, o.a); break;
        case Skeleton::Y: CliffordTableau::conj_x(M, o.a); CliffordTableau::conj_z(M, o.a); break;
        case Skeleton::Z: CliffordTableau::conj_z(M, o.a); break;
        case Skeleton::CX: CliffordTableau::conj_cx(M, o.a, o.b); break;
        case Skeleton::CZ: CliffordTableau::conj_cz(M, o.a, o.b); break;
        default: break;
    }
}

void conj_backward(Pauli& M, const Skeleton::Op& o) {
    switch (o.kind) {
        case Skeleton::S: CliffordTableau::conj_sdg(M, o.a); break;
        case Skeleton::SDG: CliffordTableau::conj_s(M, o.a); break;
        default: conj_forward(M, o); break;   // H, X, Y, Z, CX, CZ are self-inverse
    }
}

WriterModel build_writer_model(const DemPreamble& pre) {
    const Circuit& user = pre.p.circuit;
    const Circuit& def = pre.dn.deferred;
    WriterModel W;
    W.sk = std::make_unique<Skeleton>();
    Skeleton& sk = *W.sk;
    sk.N = def.n;
    const int N = def.n;

    // ── 1. user gate names: the parser writes T_DAG as `T; S_DAG`, CS_DAG as `CS; CZ` (same targets, the suffix
    //       tagged Origin::DagSuffix; a user-written `T q; S_DAG q` is untagged and stays a T group) ──
    //       Each name also records its targets as (user qubit, reset epoch): between two resets a user qubit stays on
    //       one deferred wire, so the deferred gate matched to it by position must sit on the same wires.
    struct UName { GateKind g; bool dag; std::vector<std::pair<int, int>> qe; };
    std::vector<UName> unames;
    std::vector<int> epoch((size_t)std::max(user.n, 0), 0);
    for (size_t i = 0; i < user.stream.size(); ++i) {
        const Instr& ins = user.stream[i];
        if (ins.kind == Instr::Kind::Reset)
            for (int q : ins.qubits)
                if (q >= 0 && (size_t)q < epoch.size()) ++epoch[(size_t)q];
        if (ins.kind != Instr::Kind::Gate || !is_nonclifford(ins.gate)) continue;
        bool dag = false;
        if (i + 1 < user.stream.size()) {
            const Instr& nx = user.stream[i + 1];
            if (nx.kind == Instr::Kind::Gate && nx.origin == Instr::Origin::DagSuffix && nx.targets == ins.targets &&
                ((ins.gate == GateKind::T && nx.gate == GateKind::SDG) ||
                 (ins.gate == GateKind::CS && nx.gate == GateKind::CZ)))
                dag = true;
        }
        std::vector<std::pair<int, int>> qe;
        for (int q : ins.targets) qe.emplace_back(q, q >= 0 && (size_t)q < epoch.size() ? epoch[(size_t)q] : 0);
        unames.push_back({ins.gate, dag, std::move(qe)});
    }

    // ── 2. the skeleton ops from the deferred stream ──
    std::vector<std::string> ggate;
    std::vector<std::string> glabel;   // a group's name in the diag-check messages (a CH's rotations name the CH)
    sk.op_of_stream.assign(def.stream.size() + 1, 0);
    // (user qubit, epoch) <-> deferred wire, learned from the matched gates; must stay a consistent injection
    std::map<std::pair<int, int>, int> wire_of;
    std::map<int, std::pair<int, int>> key_of;
    auto same_wires = [&](const UName& u, const std::vector<int>& t) {
        if (u.qe.size() != t.size()) return false;
        for (size_t k = 0; k < t.size(); ++k) {
            const int w = wire_of.emplace(u.qe[k], t[k]).first->second;
            const std::pair<int, int> key = key_of.emplace(t[k], u.qe[k]).first->second;
            if (w != t[k] || key != u.qe[k]) return false;
        }
        return true;
    };
    size_t nc = 0;
    for (size_t i = 0; i < def.stream.size(); ++i) {
        sk.op_of_stream[i] = (int)sk.ops.size();
        const Instr& ins = def.stream[i];
        switch (ins.kind) {
            case Instr::Kind::Gate: {
                const auto& t = ins.targets;
                switch (ins.gate) {
                    case GateKind::H: sk.ops.push_back({Skeleton::H, t[0], -1}); break;
                    case GateKind::S: sk.ops.push_back({Skeleton::S, t[0], -1}); break;
                    case GateKind::SDG: sk.ops.push_back({Skeleton::SDG, t[0], -1}); break;
                    case GateKind::X: sk.ops.push_back({Skeleton::X, t[0], -1}); break;
                    case GateKind::Y: sk.ops.push_back({Skeleton::Y, t[0], -1}); break;
                    case GateKind::Z: sk.ops.push_back({Skeleton::Z, t[0], -1}); break;
                    case GateKind::CX: sk.ops.push_back({Skeleton::CX, t[0], t[1]}); break;
                    case GateKind::CZ: sk.ops.push_back({Skeleton::CZ, t[0], t[1]}); break;
                    case GateKind::CH: {
                        // CH(c,t) = (W T W^dag) CZ(c,t) (W T_DAG W^dag) on t exactly, W = S H
                        // (W Z W^dag = Y): two pi/8 rotations about Y_t around a CZ, the same three sub-ops as the
                        // bare state's fold (pauli_rotation_form.cpp) and the PPR residual (ppr_residual.cpp). In time
                        // order: S_DAG H T_DAG H S; CZ; S_DAG H T H S on t. Two T-type groups on {t}, both at this
                        // stream index; whether their axes commute is the diag check below (step 3), as for any group.
                        if (nc >= unames.size() || unames[nc].g != ins.gate || !same_wires(unames[nc], t))
                            throw WriterError("xdem writer: internal pipeline mismatch: deferred non-Clifford "
                                              "gate #" + std::to_string(nc) + " does not match the user circuit's");
                        ++nc;
                        const int c = t[0], q = t[1];
                        for (int r = 0; r < 2; ++r) {
                            sk.ops.push_back({Skeleton::SDG, q, -1});
                            sk.ops.push_back({Skeleton::H, q, -1});
                            const int g = (int)ggate.size();
                            ggate.push_back(r == 0 ? "T_DAG" : "T");
                            glabel.push_back(std::string(r == 0 ? "T_DAG" : "T") + " (CH, rotation " +
                                             std::to_string(r + 1) + "/2)");
                            sk.group_op.push_back((int)sk.ops.size());
                            sk.group_stream.push_back((int)i);
                            sk.group_qubits.push_back({q});
                            sk.ops.push_back({Skeleton::LOC, g, -1});
                            sk.ops.push_back({Skeleton::H, q, -1});
                            sk.ops.push_back({Skeleton::S, q, -1});
                            if (r == 0) sk.ops.push_back({Skeleton::CZ, c, q});
                        }
                        break;
                    }
                    case GateKind::T:
                    case GateKind::CS:
                    case GateKind::CCZ: {
                        // matched by position: the gate kind AND its wires (as a consistent relabelling of the
                        // user's qubits) must agree, so a pass that reordered non-Clifford gates is a loud error,
                        // not a silently swapped T / T_DAG name (a wrong law)
                        if (nc >= unames.size() || unames[nc].g != ins.gate || !same_wires(unames[nc], t))
                            throw WriterError("xdem writer: internal pipeline mismatch: deferred non-Clifford "
                                              "gate #" + std::to_string(nc) + " does not match the user circuit's");
                        const bool dag = unames[nc].dag;
                        ++nc;
                        const int g = (int)ggate.size();
                        ggate.push_back(gate_name(ins.gate, dag));
                        glabel.push_back(ggate.back());
                        sk.group_op.push_back((int)sk.ops.size());
                        sk.group_stream.push_back((int)i);
                        sk.group_qubits.push_back(t);
                        sk.ops.push_back({Skeleton::LOC, g, -1});
                        // physical gate = (U_deferred U_g^dag) U_g: T = S T_DAG, CS = CZ CS_DAG
                        if (dag && ins.gate == GateKind::T) sk.ops.push_back({Skeleton::S, t[0], -1});
                        if (dag && ins.gate == GateKind::CS) sk.ops.push_back({Skeleton::CZ, t[0], t[1]});
                        break;
                    }
                }
                break;
            }
            case Instr::Kind::IfBegin:
            case Instr::Kind::IfEnd:
                throw WriterError("xdem writer: IF blocks (DECISION-conditioned branches) are outside the format "
                                  "(refusing)");
            case Instr::Kind::Reset:
            case Instr::Kind::ControlledPauli:
                throw WriterError("xdem writer: internal pipeline mismatch: a Reset / feedback instruction "
                                  "survived deferral");
            default:   // Noise, Measure (terminal reads: pre.reads), Observable (PAULI_EXPECTATION: refused in Task 4)
                break;
        }
    }
    sk.op_of_stream[def.stream.size()] = (int)sk.ops.size();
    if (nc != unames.size())
        throw WriterError("xdem writer: internal pipeline mismatch: the deferred circuit has " +
                          std::to_string(nc) + " non-Clifford gates, the user circuit " +
                          std::to_string(unames.size()));
    const int G = (int)ggate.size();
    sk.reads = pre.reads;

    // targets: detectors then observables (raw record sets, as export_dem reads them)
    const ParsedStim& p = pre.p;
    const int D = (int)p.detectors.size();
    int max_obs = -1;
    for (auto& kv : p.observables) max_obs = std::max(max_obs, kv.first);
    const int O = max_obs + 1;
    sk.targets.assign((size_t)(D + O), {});
    for (int d = 0; d < D; ++d) sk.targets[(size_t)d] = parity_reduced(p.detectors[(size_t)d]);
    for (auto& kv : p.observables) sk.targets[(size_t)(D + kv.first)] = parity_reduced(kv.second);
    for (const auto& t : sk.targets)
        for (int j : t)
            if (j < 0 || j >= (int)sk.reads.size())
                throw WriterError("xdem writer: a detector/observable references an out-of-range measurement");

    // ── 3. forward pass: Z on every location qubit carried to the end; diag check at every later location ──
    struct Lq { int g; int i; };                   // location qubit: group, position in its tuple
    std::vector<Lq> lq;
    std::vector<int> lq_off((size_t)G + 1, 0);
    for (int g = 0; g < G; ++g) {
        lq_off[(size_t)g] = (int)lq.size();
        for (int i = 0; i < (int)sk.group_qubits[(size_t)g].size(); ++i) lq.push_back({g, i});
    }
    lq_off[(size_t)G] = (int)lq.size();
    std::vector<Pauli> F(lq.size(), Pauli(N)), B(lq.size(), Pauli(N));
    {
        std::vector<int> active;
        for (const Skeleton::Op& o : sk.ops) {
            if (o.kind == Skeleton::LOC) {
                const int g = o.a;
                const auto& qs = sk.group_qubits[(size_t)g];
                for (int k : active)
                    for (int i = 0; i < (int)qs.size(); ++i)
                        if (F[(size_t)k].xbit(qs[(size_t)i])) {
                            const Lq& a = lq[(size_t)k];
                            const int qa = sk.group_qubits[(size_t)a.g][(size_t)a.i];
                            W.diag_violations.push_back(
                                "G" + std::to_string(a.g) + " (" + glabel[(size_t)a.g] + ", wire " +
                                std::to_string(qa) + "): its Z reaches the later location G" + std::to_string(g) +
                                " (" + glabel[(size_t)g] + ", wire " + std::to_string(qs[(size_t)i]) + ") with " +
                                (F[(size_t)k].zbit(qs[(size_t)i]) ? "Y" : "X"));
                        }
                for (int i = 0; i < (int)qs.size(); ++i) {
                    const int k = lq_off[(size_t)g] + i;
                    F[(size_t)k].setz(qs[(size_t)i]);
                    active.push_back(k);
                }
                continue;
            }
            // The xbit half cannot fire while the flag is false (Task 2 review M2): every F[k] stays Z-type until an H
            // meets its support, and that H sets the flag first. It is kept as a belt-and-braces check; the zbit
            // half is the load-bearing one (dropping it gives false accepts).
            if (o.kind == Skeleton::H && !W.h_meets_location)
                for (int k : active)
                    if (F[(size_t)k].xbit(o.a) || F[(size_t)k].zbit(o.a)) { W.h_meets_location = true; break; }
            for (int k : active) conj_forward(F[(size_t)k], o);
        }
    }
    // ── 4. backward pass: Z on every location qubit pulled back to the start (U_before^dag Z U_before) ──
    {
        std::vector<int> active;
        for (size_t oi = sk.ops.size(); oi-- > 0;) {
            const Skeleton::Op& o = sk.ops[oi];
            if (o.kind == Skeleton::LOC) {
                const int g = o.a;
                const auto& qs = sk.group_qubits[(size_t)g];
                for (int i = 0; i < (int)qs.size(); ++i) {
                    const int k = lq_off[(size_t)g] + i;
                    B[(size_t)k].setz(qs[(size_t)i]);
                    active.push_back(k);
                }
                continue;
            }
            for (int k : active) conj_backward(B[(size_t)k], o);
        }
    }

    // ── 5. per (g, v): signed Q, dedupe to axes ──
    std::map<std::pair<Words, Words>, int> axis_id;
    std::vector<Pauli> axis_p0;                      // unsigned U^dag Q_a U
    W.header.groups.resize((size_t)G);
    sk.gv_offset.assign((size_t)G, 0);
    for (int g = 0; g < G; ++g) {
        const int m = (int)sk.group_qubits[(size_t)g].size();
        Group& gr = W.header.groups[(size_t)g];
        gr.gate = ggate[(size_t)g];
        sk.gv_offset[(size_t)g] = (int)sk.gv_q.size();
        for (uint32_t v = 1; v < (1u << m); ++v) {
            Pauli Q(N), P0(N);
            for (int i = 0; i < m; ++i)
                if ((v >> i) & 1u) {
                    Q = Pauli::multiply(Q, F[(size_t)(lq_off[(size_t)g] + i)]);
                    P0 = Pauli::multiply(P0, B[(size_t)(lq_off[(size_t)g] + i)]);
                }
            const int s = herm_sign(Q);
            if (s == 0) throw WriterError("xdem writer: theory check failed: a propagated Z(v) is not Hermitian");
            sk.gv_q.push_back(Q);
            if (zero(Q.x) && zero(Q.z)) { gr.v_axis.push_back({-1, 0}); continue; }
            auto key = std::make_pair(Q.x, Q.z);
            auto it = axis_id.find(key);
            if (it == axis_id.end()) {
                const int a = (int)sk.axis_q.size();
                it = axis_id.emplace(std::move(key), a).first;
                Pauli Qu = Q;
                Qu.phase = Q.xz_overlap() & 3;
                sk.axis_q.push_back(std::move(Qu));
                if (s < 0) P0.phase = (P0.phase + 2) & 3;
                axis_p0.push_back(std::move(P0));
            }
            gr.v_axis.push_back({it->second, s});
        }
    }
    const int A = (int)sk.axis_q.size();
    sk.axis_p0 = axis_p0;

    // ── 6. images: reads anticommuting with the axis, folded into the targets ──
    W.header.D = D;
    W.header.O = O;
    W.header.axes.resize((size_t)A);
    const size_t TW = words_for(D + O);
    for (int a = 0; a < A; ++a) {
        const Pauli& Q = sk.axis_q[(size_t)a];
        std::vector<uint8_t> rf(sk.reads.size(), 0);
        for (size_t j = 0; j < sk.reads.size(); ++j) {
            const auto& [b, w] = sk.reads[j];
            const bool xb = Q.xbit(w), zb = Q.zbit(w);
            rf[j] = (uint8_t)(b == 0 ? zb : b == 2 ? xb : (xb != zb));
        }
        RawAxis& ax = W.header.axes[(size_t)a];
        ax.img.w.assign(TW, 0);
        ax.b = 0;
        for (int t = 0; t < D + O; ++t) {
            int par = 0;
            for (int j : sk.targets[(size_t)t]) par ^= rf[(size_t)j];
            if (par) ax.img.w[(size_t)t >> 6] |= 1ull << (t & 63);
        }
    }

    // ── 7. kappa: k_a = X-part of U^dag Q_a U (N bits); information set J = highest-bit RREF pivots ──
    std::vector<int> J;
    std::vector<Words> ker;                          // kernel combos (A bits)
    {
        const size_t AW = (size_t)((A + 63) / 64);
        std::map<int, std::pair<Words, Words>, std::greater<int>> basis;   // pivot -> (vec, combo)
        for (int a = 0; a < A; ++a) {
            Words v = axis_p0[(size_t)a].x, c(AW ? AW : 1, 0);
            flipb(c, (size_t)a);
            for (auto& [pv, bc] : basis)
                if (getb(v, (size_t)pv)) { xorw(v, bc.first); xorw(c, bc.second); }
            const int h = hibit(v);
            if (h < 0) ker.push_back(std::move(c));
            else basis.emplace(h, std::make_pair(std::move(v), std::move(c)));
        }
        for (auto& kv : basis) J.push_back(kv.first);
        std::sort(J.begin(), J.end());
    }
    const int r = (int)J.size();
    W.header.r = r;
    for (int a = 0; a < A; ++a) {
        RawAxis& ax = W.header.axes[(size_t)a];
        ax.k.assign(words_for(r), 0);
        for (int i = 0; i < r; ++i)
            if (axis_p0[(size_t)a].xbit(J[(size_t)i])) ax.k[(size_t)i >> 6] |= 1ull << (i & 63);
    }
    // (J is a column basis of the A x N kappa matrix: the restriction keeps every linear relation, i.e. ker kappa)

    // ── 8. b: sigma on ker kappa (sign of <prod Q_a> on S's final state), the unique RREF solution ──
    if (W.diag_violations.empty() && !ker.empty()) {
        const size_t GW = (size_t)((A + 1 + 63) / 64);
        std::vector<Words> graph;                    // bit 0 = sigma, bit a+1 = combo bit a
        for (const Words& c : ker) {
            Pauli P(N);
            for (int a = 0; a < A; ++a)
                if (getb(c, (size_t)a)) P = Pauli::multiply(P, axis_p0[(size_t)a]);
            const int s = pulled_sign(P);            // P = U^dag (prod Q_a) U
            if (s < 0)
                throw WriterError("xdem writer: theory check failed: a kernel product of axes is not +- a stabilizer "
                                  "of the skeleton's final state");
            Words gv(GW, 0);
            if (s) gv[0] |= 1ull;
            for (int a = 0; a < A; ++a)
                if (getb(c, (size_t)a)) flipb(gv, (size_t)a + 1);
            graph.push_back(std::move(gv));
        }
        std::map<int, Words, std::greater<int>> rr;   // fully reduced, pivot = highest bit
        for (Words v : graph) {
            for (auto& [pv, bv] : rr) if (getb(v, (size_t)pv)) xorw(v, bv);
            const int h = hibit(v);   // >= 1: the kernel combos are independent
            for (auto& [pv, bv] : rr) if (getb(bv, (size_t)h)) xorw(bv, v);
            rr.emplace(h, std::move(v));
        }
        for (auto& [pv, bv] : rr)
            if (bv[0] & 1ull) W.header.axes[(size_t)(pv - 1)].b = 1;
        for (size_t i = 0; i < ker.size(); ++i) {
            int par = 0;
            for (int a = 0; a < A; ++a)
                if (getb(ker[i], (size_t)a)) par ^= W.header.axes[(size_t)a].b;
            if (par != (int)(graph[i][0] & 1ull))
                throw WriterError("xdem writer: theory check failed: b does not reproduce sigma on ker kappa");
        }
    }
    return W;
}

// ── Fault columns: the backward column sweep (see xdem_writer.hpp) ────────────────────────────────────────────────
FaultColumns build_fault_columns(const Skeleton& sk, const Circuit& def,
                                 const std::vector<std::vector<int>>& col_recs, SweepLayout layout, SweepStats* stats) {
    FaultColumns fc;
    fc.ncols = (int)col_recs.size();
    int nloc = 0;
    for (int g = 0; g < (int)sk.group_qubits.size(); ++g)
        for (int i = 0; i < (int)sk.group_qubits[(size_t)g].size(); ++i, ++nloc) fc.xid.push_back({g, i});
    // read j -> the columns whose record set contains it (by parity)
    std::vector<std::vector<int>> read_cols(sk.reads.size());
    for (int c = 0; c < fc.ncols; ++c)
        for (int j : parity_reduced(col_recs[(size_t)c])) {
            if (j < 0 || j >= (int)sk.reads.size())
                throw WriterError("xdem writer: a column references an out-of-range measurement");
            read_cols[(size_t)j].push_back(c);
        }
    const XStartOut xo{&fc.x_start, &fc.cw, &fc.xs_off, &fc.xs_wire, fc.ncols};
    fc.at = backward_sweep(sk, def, fc.ncols + nloc, nullptr, &read_cols, fc.ncols, layout, stats, &xo);
    // final review M6: keep only the readout columns' words (nondeterministic_columns reads columns < D + O <= ncols);
    // LM-3: the sparse form holds only those columns already
    const size_t kw = (size_t)std::max(1, (fc.ncols + 63) / 64);
    if (!fc.x_sparse() && kw < fc.cw) {
        std::vector<uint64_t> xs((size_t)sk.N * kw);
        for (size_t w = 0; w < (size_t)sk.N; ++w)
            std::copy_n(fc.x_start.begin() + (std::ptrdiff_t)(w * fc.cw), kw, xs.begin() + (std::ptrdiff_t)(w * kw));
        fc.x_start = std::move(xs);
        fc.cw = kw;
    }
    return fc;
}

void FaultColumns::signature(int li, const std::vector<std::pair<int, PauliBasis>>& factors,
                             std::vector<int32_t>& sig) const {
    sig.clear();
    const auto& slot = at[(size_t)li];
    for (const auto& [qi, pb] : factors) {
        const auto& e = slot[(size_t)qi];
        if (pb == PauliBasis::X || pb == PauliBasis::Y) sig.insert(sig.end(), e[0].begin(), e[0].end());
        if (pb == PauliBasis::Z || pb == PauliBasis::Y) sig.insert(sig.end(), e[1].begin(), e[1].end());
    }
    parity_reduce_sorted(sig);
}

// ── residue_terms ─────────────────────────────────────────────────────────────────────────────────────────────
const std::vector<std::pair<int, int>>& residue_terms(const std::string& gate, uint32_t x) {
    const int gi = gate_index(gate);
    if (gi < 0 || x == 0 || x >= (1u << kArity[gi])) throw std::invalid_argument("xdem residue_terms: bad (gate, x)");
    return residue_by_index(gi, x);
}

int gate_rotation_count(const std::string& gate, uint32_t v) {
    const int gi = gate_index(gate);
    if (gi < 0 || v == 0 || v >= (1u << kArity[gi]))
        throw std::invalid_argument("xdem gate_rotation_count: bad (gate, v)");
    const int m = kArity[gi];
    int s = 0;
    for (uint32_t t = 0; t < (1u << m); ++t) {
        std::vector<int> z((size_t)m);
        for (int i = 0; i < m; ++i) z[(size_t)i] = (int)((t >> i) & 1u);
        s += (__builtin_popcount(v & t) & 1) ? -phase_of(gi, z) : phase_of(gi, z);
    }
    const int num = -2 * s;
    if (num % (1 << m)) throw std::logic_error("xdem gate_rotation_count: not a pi/8 multiple (engine bug)");
    return num / (1 << m);
}

// ── t_c and the canonical form ────────────────────────────────────────────────────────────────────────────────
TcPoint last_detector_point(const ParsedStim& p) {
    const std::vector<Instr>& st = p.circuit.stream;
    if (p.stream_step.size() != st.size() || p.stream_line.size() != st.size())
        throw std::logic_error("last_detector_point: the circuit was parsed without track_source");
    TcPoint t;
    for (size_t d = 0; d < p.detectors.size(); ++d)
        for (int r : parity_reduced(p.detectors[d]))
            if (r > t.record) { t.record = r; t.detector = (int)d; }
    if (t.record < 0) return t;
    int k = 0;                                                  // the record's Measure instruction
    for (size_t i = 0; i < st.size(); ++i)
        if (st[i].kind == Instr::Kind::Measure && k++ == t.record) {
            t.step = p.stream_step[i];
            return t;
        }
    throw std::logic_error("last_detector_point: a detector references an out-of-range measurement");
}

int late_group_count(const ParsedStim& p) {
    const std::vector<Instr>& st = p.circuit.stream;
    const int step_c = last_detector_point(p).step;
    int n = 0;
    for (size_t i = 0; i < st.size(); ++i)
        if (st[i].kind == Instr::Kind::Gate && is_nonclifford(st[i].gate) && p.stream_step[i] < step_c)
            n += st[i].gate == GateKind::CH ? 2 : 1;   // a CH is two groups (its two rotations, build_writer_model)
    return n;
}

std::string canonical_violation(const ParsedStim& p) {
    const std::vector<Instr>& st = p.circuit.stream;
    if (p.stream_step.size() != st.size() || p.stream_line.size() != st.size())
        throw std::logic_error("canonical_violation: the circuit was parsed without track_source");
    std::vector<size_t> meas;                                   // record -> its Measure instruction (stream index)
    for (size_t i = 0; i < st.size(); ++i)
        if (st[i].kind == Instr::Kind::Measure) meas.push_back(i);
    auto line_of = [&](int r) { return std::to_string(p.stream_line[meas[(size_t)r]]); };
    // t_c: right after the user instruction that measures the last detector record (step_c = its dispatch ordinal;
    // -1 = before everything when no detector reads a record).
    const TcPoint tcp = last_detector_point(p);
    const int last_rec = tcp.record, last_det = tcp.detector, step_c = tcp.step;
    const std::string tc = last_rec >= 0
        ? "the last detector record (record #" + std::to_string(last_rec) + " of D" + std::to_string(last_det) +
              ", line " + line_of(last_rec) + ")"
        : "t_c = the start of the circuit (no detector reads a record)";
    std::string first;
    long more = 0;
    auto note = [&](std::string why) {
        if (first.empty()) first = std::move(why); else ++more;
    };
    // (C1') every observable reads at least one record measured by a later user instruction; earlier records are a
    // classical frame (XOR). An observable with no record (none declared, or all cancelled) violates it.
    for (const auto& kv : p.observables) {
        const std::vector<int> recs = parity_reduced(kv.second);
        const std::string L = "observable L" + std::to_string(kv.first);
        if (recs.empty()) {
            note(L + " reads no record (none measured after " + tc + ")");
            continue;
        }
        const int r = recs.back();                              // the latest record
        const int s = p.stream_step[meas[(size_t)r]];
        if (s > step_c) continue;
        note(L + " reads no record measured after " + tc + ": its latest, record #" + std::to_string(r) + " (line " +
             line_of(r) + "), is measured " + (s == step_c ? "by the same instruction" : "before it"));
    }
    // (C2') no noise between t_c and the last non-Clifford gate (noise after it is a classical readout flip).
    size_t last_nc = 0;
    bool any_nc = false;
    for (size_t i = 0; i < st.size(); ++i)
        if (st[i].kind == Instr::Kind::Gate && is_nonclifford(st[i].gate) && p.stream_step[i] > step_c) {
            last_nc = i;
            any_nc = true;
        }
    const std::string nc = any_nc ? "the last non-Clifford gate (" +
                                        std::string(gate_name(st[last_nc].gate, last_nc + 1 < st.size() &&
                                                                           st[last_nc + 1].origin ==
                                                                               Instr::Origin::DagSuffix)) + ", line " +
                                        std::to_string(p.stream_line[last_nc]) + ")"
                                  : std::string();
    for (size_t i = 0; any_nc && i < last_nc; ++i) {
        if (p.stream_step[i] <= step_c) continue;
        const Instr& ins = st[i];
        if (ins.kind == Instr::Kind::Noise) {
            bool live = false;
            for (double q : ins.probs) live = live || q > 0.0;
            if (live)
                note("noise " + std::string(noise_name(ins.channel)) + " at line " +
                     std::to_string(p.stream_line[i]) + " acts between " + tc + " and " + nc);
        } else if (ins.kind == Instr::Kind::Measure && ins.readout_flip_p > 0.0) {
            const long r = std::lower_bound(meas.begin(), meas.end(), i) - meas.begin();
            note("noisy measurement at line " + std::to_string(p.stream_line[i]) + " (record #" + std::to_string(r) +
                 ", readout flip p=" + fmt_g(ins.readout_flip_p) + ") acts between " + tc + " and " + nc);
        }
    }
    if (more > 0) first += " (and " + std::to_string(more) + " more violation(s))";
    return first;
}

// ── the canonical coordinates (SPEC §2) ───────────────────────────────────────────────────────────────────────
int stabilizer_sign(const Skeleton& sk, const Pauli& P, const std::string& what) {
    Pauli M = P;
    for (size_t oi = sk.ops.size(); oi-- > 0;) conj_backward(M, sk.ops[oi]);   // U^dag P U
    const int s = pulled_sign(M);
    if (s < 0)
        throw WriterError("xdem writer: theory check failed: the product of axes " + what +
                          " is not +- a stabilizer of the skeleton's final state");
    return s;
}

Canon canonical_coordinates(const WriterModel& wm) {
    const auto& axes = wm.header.axes;                 // raw kappa in axes[a].k (information-set basis)
    const auto& q = wm.sk->axis_q;                     // unsigned end Pauli per axis
    const size_t A = axes.size(), AW = A / 64 + 1;
    Canon C;
    C.pivot.assign(A, 0); C.kappa.resize(A); C.hidden.resize(A); C.bhat.assign(A, 0);
    Echelon g;                                         // 1. gauge: walk the axes in file order
    std::vector<int> piv;                              // piv[t] = the axis owning gauge qubit g_t
    for (size_t a = 0; a < A; ++a) {
        Words v = axes[a].k, c(AW, 0);
        if (g.reduce(v, c)) {                          // kappa_a not in the span of the earlier ones: a pivot
            Words u(AW, 0);
            flipb(u, piv.size());
            C.kappa[a] = u;
            xorw(u, c);                                // the residual row = kappa_a + the rows used (B4)
            g.insert(std::move(v), std::move(u));
            C.pivot[a] = 1; piv.push_back((int)a);
        } else {
            C.kappa[a] = c;                            // the unit vectors of the pivots its kappa depends on
        }
    }
    C.r = (int)piv.size();
    for (auto& k : C.kappa) k.resize(words_for(C.r));
    Echelon hdn;                                       // 2. hidden-syndrome: generators of K mod K_I through c -> prod Q
    for (size_t a = 0; a < A; ++a) {
        if (C.pivot[a]) continue;
        Pauli P = q[a];                                // c_a = {a} + its pivots (kappa_{c_a} = 0)
        std::string what = "{" + std::to_string(a);
        for (size_t t = 0; t < (size_t)C.r; ++t)
            if (getb(C.kappa[a], t)) {
                P = Pauli::multiply(P, q[(size_t)piv[t]]);
                what += "," + std::to_string(piv[t]);
            }
        what += "}";
        Words v = pauli_xz_bits(P), c(AW, 0);          // phase dropped: K_I = the kernel of this map
        if (hdn.reduce(v, c)) {                        // a new generator: the next hidden-syndrome qubit
            Words u(AW, 0);
            flipb(u, (size_t)C.m);
            xorw(u, c);                                // the residual row = Q_{c_a} + the rows used (B4)
            hdn.insert(std::move(v), std::move(u));
            C.hidden[a] = {C.m++};
        } else {
            for (size_t t = 0; t < (size_t)C.m; ++t)   // its coordinates in the generators (mod K_I)
                if (getb(c, t)) C.hidden[a].push_back((int32_t)t);
        }
        C.bhat[a] = (uint8_t)stabilizer_sign(*wm.sk, P, what);   // beta(c_a)
    }
    return C;
}

// The fault columns' backward sweep with A columns, each activated at the END of the deferred stream with the axis's
// end Pauli (no read or location activations).
TauColumns build_tau_columns(const Skeleton& sk, const Circuit& def, SweepLayout layout, SweepStats* stats) {
    TauColumns tc;
    tc.at = backward_sweep(sk, def, (int)sk.axis_q.size(), &sk.axis_q, nullptr, -1, layout, stats);
    return tc;
}

void fault_rotations(const Header& H, const std::vector<std::pair<int, uint32_t>>& x, int Gl,
                     std::vector<uint8_t>& Elt, std::vector<uint8_t>& Egt) {
    for (const auto& [g, xg] : x) {                   // x_g = 0 for groups before the fault (not listed)
        const Group& gr = H.groups[(size_t)g];
        std::vector<uint8_t>& E = g < Gl ? Elt : Egt;
        for (const auto& [v, e] : residue_terms(gr.gate, xg)) {
            const auto& [a, sgn] = gr.v_axis[(size_t)(v - 1)];
            if (a < 0) continue;                      // an identity axis: a global phase
            E[(size_t)a] = (uint8_t)((E[(size_t)a] + 4 - ((sgn * e) & 3)) & 3);   // -(sign * e) mod 4
        }
    }
}

void fold_field(const std::vector<Axis>& axes, std::vector<uint8_t>& E, Bits& f) {
    for (size_t a = 0; a < E.size(); ++a)
        if (E[a] == 2) {
            for (size_t w = 0; w < f.w.size(); ++w) f.w[w] ^= axes[a].img.w[w];
            E[a] = 0;
        }
}

bool TimeKey::operator<(const TimeKey& o) const {
    return std::tie(step, ordinal, mech) < std::tie(o.step, o.ordinal, o.mech);
}

bool LineKey::operator<(const LineKey& o) const {
    return std::tie(side, f, tau, lt, gt) < std::tie(o.side, o.f, o.tau, o.lt, o.gt);
}

// P + p - 2 P p (see the header): out of line on purpose, the one copy both merges execute.
[[gnu::noinline]] double xor_compose(double P, double p) { return P + p - 2.0 * P * p; }

// The axes with an odd total exponent (E^< + E^>), ascending (declared in the header for the lean merge).
std::vector<int32_t> line_odd_axes(const LineKey& k) {
    std::vector<int32_t> o;
    size_t i = 0, j = 0;
    while (i < k.lt.size() || j < k.gt.size()) {
        const int32_t a = j >= k.gt.size() || (i < k.lt.size() && k.lt[i].first <= k.gt[j].first) ? k.lt[i].first
                                                                                                    : k.gt[j].first;
        int e = 0;
        if (i < k.lt.size() && k.lt[i].first == a) e += k.lt[i++].second;
        if (j < k.gt.size() && k.gt[j].first == a) e += k.gt[j++].second;
        if (e & 1) o.push_back(a);
    }
    return o;
}

namespace {
std::vector<int32_t> intersect(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {   // both ascending
    std::vector<int32_t> o;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(o));
    return o;
}
std::vector<int32_t> swap_set_odd(const LineKey& a, const std::vector<int32_t>& odd_a, const LineKey& b,
                                  const std::vector<int32_t>& odd_b) {
    const std::vector<int32_t> u = intersect(a.tau, odd_b), v = intersect(b.tau, odd_a);
    std::vector<int32_t> o;
    std::set_symmetric_difference(u.begin(), u.end(), v.begin(), v.end(), std::back_inserter(o));
    return o;
}
bool image_nonzero(const std::vector<int32_t>& S, const std::vector<Axis>& axes) {
    if (S.empty()) return false;
    if ((size_t)S.back() >= axes.size())
        throw std::logic_error("xdem LineAccumulator: an axis id beyond the axes given to set_axes (engine bug)");
    if (S.size() == 1) {
        for (uint64_t w : axes[(size_t)S[0]].img.w) if (w) return true;
        return false;
    }
    std::vector<uint64_t> acc(axes[(size_t)S[0]].img.w);
    for (size_t i = 1; i < S.size(); ++i) {
        const auto& w = axes[(size_t)S[i]].img.w;
        for (size_t j = 0; j < acc.size(); ++j) acc[j] ^= w[j];
    }
    for (uint64_t w : acc) if (w) return true;
    return false;
}
}  // namespace

std::vector<int32_t> swap_set(const LineKey& a, const LineKey& b) {
    return swap_set_odd(a, line_odd_axes(a), b, line_odd_axes(b));
}

bool swap_visible(const LineKey& a, const LineKey& b, const std::vector<Axis>& axes) {
    return image_nonzero(swap_set(a, b), axes);
}

void LineAccumulator::add(const LineKey& k, double p, const TimeKey& t, int flush) {
    if (!(p > 0.0) || k.empty()) return;
    mechs_.push_back({k, p, t, flush});
    merged_ = false;
}

void LineAccumulator::merge() const {
    if (merged_) return;
    order_.resize(mechs_.size());
    for (size_t i = 0; i < order_.size(); ++i) order_[i] = i;
    std::stable_sort(order_.begin(), order_.end(), [&](size_t a, size_t b) { return mechs_[a].t < mechs_[b].t; });
    lines_.clear();
    mech_line_.assign(mechs_.size(), -1);
    std::map<LineKey, int> open;                       // key -> its open line
    std::vector<char> is_open;
    // Visibility depends on a key only through u = (tau, odd(E)): open lines are grouped by u, and an event evaluates
    // each candidate class once (a visible swap closes the whole class). by_tau / by_odd: axis -> classes listed
    // while they hold open lines (pruned lazily, re-listed when a class gets a line again).
    // expire: the last sorted index at which a mechanism can join one of the class's open lines (their keys' last
    // members); past it no event's decision about the class can matter, so it is pruned from the indexes.
    struct Class {
        std::vector<int32_t> tau, odd;
        std::vector<int> lines;
        bool listed = false;
        int gen = 0;
        size_t expire = 0;
    };
    std::map<LineKey, size_t> last_of;                 // key -> its last sorted mechanism index
    for (size_t r = 0; r < order_.size(); ++r) last_of[mechs_[order_[r]].k] = r;
    std::vector<Class> cls;
    std::map<std::pair<std::vector<int32_t>, std::vector<int32_t>>, int> cls_of;
    std::vector<int> line_cls;
    std::map<int32_t, std::vector<std::pair<int, int>>> by_tau, by_odd;   // (class, its listing generation)
    std::vector<int> cand, stamp;
    int now = 0;
    const size_t W = axes_.empty() ? 0 : axes_[0].img.w.size();
    std::vector<uint64_t> acc(W, 0);
    auto visible = [&](const Class& c, const Class& k) {   // img((tau_c & odd_k) xor (tau_k & odd_c)) != 0
        bool any = false;
        auto add_meet = [&](const std::vector<int32_t>& x, const std::vector<int32_t>& y) {
            for (size_t i = 0, j = 0; i < x.size() && j < y.size();) {
                if (x[i] < y[j]) { ++i; continue; }
                if (y[j] < x[i]) { ++j; continue; }
                if ((size_t)x[i] >= axes_.size())
                    throw std::logic_error("xdem LineAccumulator: an axis id beyond set_axes (engine bug)");
                const auto& w = axes_[(size_t)x[i]].img.w;
                for (size_t q = 0; q < W; ++q) acc[q] ^= w[q];
                any = true;
                ++i; ++j;
            }
        };
        add_meet(c.tau, k.odd);
        add_meet(k.tau, c.odd);                        // an axis in both halves cancels: the xor of the sets
        if (!any) return false;
        bool nz = false;
        for (size_t q = 0; q < W; ++q) { nz |= acc[q] != 0; acc[q] = 0; }
        return nz;
    };
    auto list_class = [&](int c) {
        Class& C = cls[(size_t)c];
        if (C.listed) return;
        C.listed = true;
        ++C.gen;
        for (int32_t a : C.tau) by_tau[a].push_back({c, C.gen});
        for (int32_t a : C.odd) by_odd[a].push_back({c, C.gen});
    };
    for (size_t r = 0; r < order_.size(); ++r) {
        const Mech& m = mechs_[order_[r]];
        int id;
        auto it = open.find(m.k);
        if (it != open.end()) {
            id = it->second;
            Line& L = lines_[(size_t)id];
            L.p = xor_compose(L.p, m.p);                // the default export's XOR composition
        } else {
            id = (int)lines_.size();
            lines_.push_back({&m.k, m.p});
            is_open.push_back(1);
            open.emplace(m.k, id);
            auto key = std::make_pair(m.k.tau, line_odd_axes(m.k));
            auto ci = cls_of.find(key);
            int c;
            if (ci == cls_of.end()) {
                c = (int)cls.size();
                Class C;
                C.tau = key.first; C.odd = key.second;
                cls.push_back(std::move(C));
                stamp.push_back(-1);
                cls_of.emplace(std::move(key), c);
            } else {
                c = ci->second;
            }
            line_cls.push_back(c);
            Class& C = cls[(size_t)c];
            const size_t last = last_of.find(m.k)->second;
            if (C.lines.empty() || last > C.expire) C.expire = last;
            C.lines.push_back(id);
            list_class(c);
        }
        mech_line_[r] = id;
        // this event closes every open line whose swap with m.k is visible
        const int kc = line_cls[(size_t)id];
        ++now;
        cand.clear();
        auto collect = [&](std::map<int32_t, std::vector<std::pair<int, int>>>& idx,
                           const std::vector<int32_t>& axes) {
            for (int32_t a : axes) {
                auto f = idx.find(a);
                if (f == idx.end()) continue;
                auto& v = f->second;
                size_t keep = 0;
                for (const auto& [c, g] : v) {
                    Class& C = cls[(size_t)c];
                    if (!C.listed || C.gen != g) continue;   // emptied (or re-listed) since: pruned
                    if (C.expire <= r) {                // no later mechanism can join its lines: pruned for good
                        C.lines.clear();                // (they stay open; nothing reads them again)
                        C.listed = false;
                        continue;
                    }
                    v[keep++] = {c, g};
                    if (stamp[(size_t)c] != now) { stamp[(size_t)c] = now; cand.push_back(c); }
                }
                v.resize(keep);
            }
        };
        collect(by_tau, cls[(size_t)kc].odd);
        collect(by_odd, cls[(size_t)kc].tau);
        for (int c : cand) {
            Class& C = cls[(size_t)c];
            if (c == kc || !visible(C, cls[(size_t)kc])) continue;   // own class: an empty swap set
            for (int l : C.lines)
                if (is_open[(size_t)l]) { is_open[(size_t)l] = 0; open.erase(*lines_[(size_t)l].k); }
            C.lines.clear();
            C.listed = false;                          // pruned from the indexes on their next scan
        }
    }
    merged_ = true;
}

bool LineAccumulator::any_rotation() const {
    merge();
    for (const Line& L : lines_)
        if (!L.k->lt.empty() || !L.k->gt.empty()) return true;
    return false;
}

File file_begin(int D, int O, const Canon& C, const std::vector<Axis>& axes) {
    File F;
    F.extended = true;
    F.D = D; F.O = O; F.r = C.r; F.m = C.m;
    F.axes = axes;
    F.flip_off = {0}; F.tau_off = {0}; F.rot_off = {0}; F.late_off = {0};
    F.tc = 0;
    return F;
}

void file_add_line(File& F, const LineKey& k, double p) {
    if (k.side == 0) {
        if (F.tc != (int64_t)F.p.size())
            throw WriterError("xdem writer: internal error: a line before t_c follows a line after it");
        ++F.tc;
    }
    F.p.push_back(p);
    F.flip_cols.insert(F.flip_cols.end(), k.f.begin(), k.f.end());
    F.tau_ax.insert(F.tau_ax.end(), k.tau.begin(), k.tau.end());
    for (const auto& [a, e] : k.lt) F.rot.push_back({a, e});
    for (const auto& [a, e] : k.gt) F.late.push_back({a, e});
    F.flip_off.push_back(F.flip_cols.size());
    F.tau_off.push_back(F.tau_ax.size());
    F.rot_off.push_back(F.rot.size());
    F.late_off.push_back(F.late.size());
    F.src_line.push_back(-1);
}

File LineAccumulator::to_file(int D, int O, const Canon& C, const std::vector<Axis>& axes) const {
    merge();
    File F = file_begin(D, O, C, axes);
    for (const Line& L : lines_) file_add_line(F, *L.k, L.p);
    return F;
}

std::vector<int> LineAccumulator::atom_lines() const {
    merge();
    std::map<std::pair<int, LineKey>, int> at;         // (flush, key) -> the line its mechanism joined
    for (size_t r = 0; r < order_.size(); ++r) {
        const Mech& m = mechs_[order_[r]];
        if (m.flush < 0) continue;
        const auto [it, fresh] = at.emplace(std::make_pair(m.flush, m.k), mech_line_[r]);
        if (!fresh && it->second != mech_line_[r])
            throw std::logic_error("xdem LineAccumulator: one flush's equal keys on two lines (engine bug)");
    }
    std::vector<int> out;
    out.reserve(atoms_.size());
    for (const auto& [k, f] : atoms_) {
        const auto it = at.find(std::make_pair(f, k));
        out.push_back(it == at.end() ? -1 : it->second);
    }
    return out;
}

}  // namespace qeccore::xdem
