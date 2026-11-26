// xdem_file.hpp — the xdem reader (SPEC §3), the canonical emitter, the composite (§1.2) and the law (§1.3).
//
// Readout qubits: detector d = bit d, observable o = bit D+o (D+O columns, multiword). An axis is the signed Z-string
// (-1)^b Z[img | kappa | sigma] on the readout, gauge and hidden-syndrome qubits. A line's composite with others is
// the cocycle of §1.2 (each later rotation about a is negated once per earlier line whose x field holds a); its law
// is the stabilizer law of §1.3 (offset + coins). Every refusal of the reader is an XdemFileError whose message
// starts "xdem: line <n>: " (n = 1-based text line).
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/visibility.hpp"

namespace qeccore::xdem {

struct QECCORE_SHARED_TYPE XdemFileError : std::runtime_error {             // every reader refusal; the binding maps it to XtimXdemError
    explicit XdemFileError(const std::string& m) : std::runtime_error(m) {}
};
// An allocation failed while reading a file or building its sampler (std::bad_alloc from sizes the file declares or
// implies: a header's counts, a large target index, a repeat count). No size cap: the failure itself is reported,
// naming the sizes. The binding maps it to xtim.XtimCapacityError (a MemoryError).
struct QECCORE_SHARED_TYPE XdemCapacityError : std::runtime_error {
    explicit XdemCapacityError(const std::string& m) : std::runtime_error(m) {}
};

struct Bits { std::vector<uint64_t> w; };                // D+O columns: detector d = bit d, observable o = bit D+o
inline size_t words_for(long long ncols) { return (size_t)((ncols + 63) / 64); }
std::string to_hex(const Bits& b);                       // lower-case hex, "0" when zero (the test API's encoding)

struct Rot { int32_t axis; uint8_t e; };                 // a file holds e = 1 (+Q) or 3 (-Q)

struct Axis {
    Bits img;                                            // readout part of Phi(a)
    std::vector<uint64_t> k;                             // kappa: gauge-qubit bits (r bits)
    std::vector<int32_t> hidden;                         // sigma: hidden-syndrome qubit ids, ascending
    int b = 0;                                           // bhat (sign -)
};

// Read-only view of the axes a law runs over: the file's axes, then `extra` (K's strings, SPEC §4.3 step 6).
// kbase (optional, nbase entries): kappa of the base axes in another gauge basis, read in place of base[a].k (the
// sampler's general path, XdemSampler::build_law_axes); the extra axes always use their own k.
struct AxisView {
    const Axis* base = nullptr; size_t nbase = 0;
    const Axis* extra = nullptr; size_t nextra = 0;
    const std::vector<uint64_t>* kbase = nullptr;
    size_t size() const { return nbase + nextra; }
    const Axis& operator[](size_t a) const { return a < nbase ? base[a] : extra[a - nbase]; }
    const std::vector<uint64_t>& kappa(size_t a) const {
        return a < nbase ? (kbase ? kbase[a] : base[a].k) : extra[a - nbase].k;
    }
};

struct File {
    bool extended = false;                               // the xdem header was present
    int D = 0, O = 0, r = 0, m = 0;
    int64_t tc = 0;                                      // first late line; a conventional DEM: num_lines()
    std::vector<Axis> axes;
    std::vector<double> p;                               // per error line, file order (repeat unrolled)
    std::vector<uint64_t> flip_off; std::vector<int32_t> flip_cols;   // CSR; ascending, distinct (mod-2 reduced)
    std::vector<uint64_t> tau_off;  std::vector<int32_t> tau_ax;      // CSR; ascending
    std::vector<uint64_t> rot_off;  std::vector<Rot> rot;             // E^<; ascending axis
    std::vector<uint64_t> late_off; std::vector<Rot> late;            // E^>; ascending axis
    std::vector<int32_t> src_line;                       // 1-based text line of each error line (messages)
    size_t num_lines() const { return p.size(); }
};

inline constexpr long long kMaxColumns = (1LL << 31) - 1;   // D + O bound (int32 column index), refused by name

File parse(const std::string& text);                     // throws XdemFileError (SPEC §3.5)
std::string emit(const File& F);                         // canonical text of an extended File (writer; fixed point)

struct Composed {
    Bits flip;                                           // f
    std::vector<uint64_t> tau;                           // tau, (A + 63) / 64 words
    std::vector<std::pair<int, int>> eta;                // (axis, eta mod 4), nonzero, ascending axis
};
Composed compose(const File& F, const std::vector<int>& ids);   // any order; sorted; pairs cancel (§1.2)

struct Law { Bits offset; std::vector<Bits> coins; };   // offset reduced mod coins; coins RREF, descending pivot

// law_into's results in reusable flat buffers (the sampler's per-miss path; law() wraps it): off (D+O words, reduced
// modulo the coins), rank, coins (rank rows of words_for(D+O) words, descending pivot). The other members are
// scratch.
struct LawScratch {
    std::vector<uint64_t> off, coins;
    int rank = 0;
    std::vector<int> odd, oddE, piv, crow, prow, brow;
    std::vector<uint64_t> rows, cols, basis, lo, hi, bb, pm, bpm, u;
    std::vector<size_t> ord;
};
void law_into(int D, int O, int r, const AxisView& ax, const Composed& c, LawScratch& S);
void law_into(const File& F, const Composed& c, LawScratch& S);   // AxisView{F.axes}, no extra
Law law(const File& F, const Composed& c);

// The file-level facts the sampler and Python expose (SPEC §4.4, §4.6).
std::vector<int> late_axes(const File& F);               // axes with a nonzero late exponent on some line
std::pair<bool, std::string> r2b_available(const File& F);   // (ok, reason naming the first offending line / axis)
int invisible_residues(const File& F);                   // rank img[A1] - rank img[A1]|_D, A1 = axes in some rot field
}  // namespace qeccore::xdem
