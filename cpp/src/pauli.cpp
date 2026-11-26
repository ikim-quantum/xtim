#include "qeccore/pauli.hpp"
#include <cstddef>
#include "qeccore/stab_affine.hpp"
#include <stdexcept>
namespace qeccore {

static int popcnt_and(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    int c = 0; for (size_t i=0;i<a.size();++i) c += __builtin_popcountll(a[i]&b[i]); return c;
}

int Pauli::anticommute_bit(const Pauli& a, const Pauli& b) {
    return (popcnt_and(a.x, b.z) + popcnt_and(a.z, b.x)) & 1;
}
int Pauli::xz_overlap() const { return popcnt_and(x, z); }

Pauli Pauli::multiply(const Pauli& a, const Pauli& b) {
    Pauli c(a.n);
    for (size_t i=0;i<c.x.size();++i){ c.x[i]=a.x[i]^b.x[i]; c.z[i]=a.z[i]^b.z[i]; }
    int sign = popcnt_and(a.z, b.x) & 1;
    c.phase = ((a.phase + b.phase + 2*sign) % 4 + 4) % 4;
    return c;
}

static void support_of(const std::vector<uint64_t>& words, std::vector<int32_t>& out) {
    int cnt = 0; for (uint64_t w : words) cnt += __builtin_popcountll(w);
    out.clear(); out.reserve(cnt);
    for (size_t i=0;i<words.size();++i)
        for (uint64_t w = words[i]; w; w &= w - 1) out.push_back((int32_t)(64*i + __builtin_ctzll(w)));
}

SparsePauli from_dense(const Pauli& P) {
    SparsePauli s; s.n = P.n; s.phase = P.phase;
    support_of(P.x, s.x); support_of(P.z, s.z);
    return s;
}

Pauli to_dense(const SparsePauli& s) {
    Pauli P(s.n); P.phase = s.phase;
    for (int32_t q : s.x) P.setx(q);
    for (int32_t q : s.z) P.setz(q);
    return P;
}

void Pauli::apply_to(StabState& s) const {
    // Set bits only, ascending q, every Z before every X (the order of the per-qubit scans).
    for (size_t t=0;t<z.size();++t)
        for (uint64_t w=z[t]; w; w&=w-1) { const int q=(int)(t<<6)+__builtin_ctzll(w); if (q>=n) break; s.apply_z(q); }
    for (size_t t=0;t<x.size();++t)
        for (uint64_t w=x[t]; w; w&=w-1) { const int q=(int)(t<<6)+__builtin_ctzll(w); if (q>=n) break; s.apply_x(q); }
    if (phase % 4 != 0) {
        if (auto* a = dynamic_cast<AffineState*>(&s)) a->omega = a->omega.mul(ExactPhase::zeta8((2*phase)%8));
        else throw std::logic_error("Pauli::apply_to: unknown StabState representation");
    }
}

}  // namespace qeccore
