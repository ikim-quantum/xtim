// test_pauli_kernels.cpp — S2k: the closed-form conj_cx_inplace against the previous
// implementation (kept here VERBATIM as the oracle: clear the (c,t) bits, build the correction
// product on thread_local scratch with pmul_into, multiply). Exhaustive over the 16 (x_c,z_c,x_t,z_t)
// patterns × phases −5..8 (outside 0..3 too) × n ∈ {2, 65, 130} (W = 1, 2, 3) × (c,t) pairs in the
// same word / different words / c<t / c>t, with random bits on every other qubit; every field (x, z,
// phase, n) must be identical.
#include "check.hpp"

#include <random>
#include <vector>

#include "qeccore/pauli_kernels.hpp"

using namespace qeccore;

namespace {

void conj_cx_ref(Pauli& M, int c, int t) {   // ab2ff72's conj_cx_inplace, verbatim
    bool xc = M.xbit(c), zc = M.zbit(c), xt = M.xbit(t), zt = M.zbit(t);
    if (xc) M.flipx(c);
    if (zc) M.flipz(c);
    if (xt) M.flipx(t);
    if (zt) M.flipz(t);
    thread_local Pauli img, p;
    const int w = (M.n + 63) / 64;
    auto reset = [&](Pauli& q) {
        if ((int)q.x.size() != w) { q = Pauli(M.n); }
        else { std::fill(q.x.begin(), q.x.end(), 0); std::fill(q.z.begin(), q.z.end(), 0); q.phase = 0; q.n = M.n; }
    };
    reset(img);
    if (xc) { reset(p); p.setx(c); p.setx(t); pmul_into(img, p); }   // X_c X_t
    if (zc) { reset(p); p.setz(c);            pmul_into(img, p); }   // Z_c
    if (xt) { reset(p); p.setx(t);            pmul_into(img, p); }   // X_t
    if (zt) { reset(p); p.setz(c); p.setz(t); pmul_into(img, p); }   // Z_c Z_t
    pmul_into(M, img);
}

void test_exhaustive() {
    std::mt19937_64 rng(20261016);
    long cases = 0;
    for (int n : {2, 65, 130}) {
        std::vector<std::pair<int, int>> pairs = {{0, 1}, {1, 0}};
        if (n > 64) { pairs.push_back({3, 70}); pairs.push_back({70, 3}); pairs.push_back({64, 63}); }
        if (n > 128) { pairs.push_back({129, 5}); pairs.push_back({66, 128}); }
        for (auto [c, t] : pairs)
            for (int pat = 0; pat < 16; ++pat)
                for (int ph = -5; ph <= 8; ++ph)
                    for (int rep = 0; rep < 4; ++rep) {
                        Pauli M(n);
                        for (auto& w : M.x) w = rng();
                        for (auto& w : M.z) w = rng();
                        if (n % 64) { M.x.back() &= (1ull << (n % 64)) - 1; M.z.back() &= (1ull << (n % 64)) - 1; }
                        auto set = [&](bool xz, int q, bool v) {
                            const bool cur = xz ? M.zbit(q) : M.xbit(q);
                            if (cur != v) { if (xz) M.flipz(q); else M.flipx(q); }
                        };
                        set(false, c, pat & 1); set(true, c, pat & 2); set(false, t, pat & 4); set(true, t, pat & 8);
                        M.phase = ph;
                        Pauli A = M, B = M;
                        conj_cx_ref(A, c, t);
                        conj_cx_inplace(B, c, t);
                        CHECK(A.x == B.x && A.z == B.z && A.phase == B.phase && A.n == B.n);
                        ++cases;
                    }
    }
    std::fprintf(stderr, "[pauli_kernels] conj_cx_inplace closed form == previous chain on %ld cases\n", cases);
}

}  // namespace

int main() {
    RUN(test_exhaustive);
    REPORT();
}
