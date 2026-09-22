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

void Pauli::apply_to(StabState& s) const {
    for (int q=0;q<n;++q) if (zbit(q)) s.apply_z(q);
    for (int q=0;q<n;++q) if (xbit(q)) s.apply_x(q);
    if (phase % 4 != 0) {
        if (auto* a = dynamic_cast<AffineState*>(&s)) a->omega = a->omega.mul(ExactPhase::zeta8((2*phase)%8));
        else throw std::logic_error("Pauli::apply_to: unknown StabState representation");
    }
}

}  // namespace qeccore
