// test_sparse_pauli.cpp — S2-T1: SparsePauli (sorted support lists) and its kernels against the dense
// Pauli routines kept as the oracle. Random cases at the word-boundary sizes n ∈ {1, 63, 64, 65, 127,
// 128, 129} (plus 200, 1000) and weights empty / sparse / half / full, phases −5..8 (outside 0..3 too, as
// Pauli::multiply accepts them). Every field (n, phase, x, z) must be identical:
//   from_dense      == the per-qubit scan of the dense bits;   to_dense(from_dense(D)) == D
//   to_dense        == the per-qubit setx/setz build;          from_dense(to_dense(S)) == S
//   anticommute_bit(D, S)     == Pauli::anticommute_bit(D, to_dense(S))
//   pmul_into(D, S)           == Pauli::multiply(D, to_dense(S))   (and the dense pmul_into)
//   pmul_sparse(S1, S2)       == from_dense(Pauli::multiply(to_dense(S1), to_dense(S2)))
#include "check.hpp"

#include <random>
#include <vector>

#include "qeccore/pauli_kernels.hpp"

using namespace qeccore;

namespace {

bool same(const Pauli& a, const Pauli& b) {
    return a.n == b.n && a.phase == b.phase && a.x == b.x && a.z == b.z;
}
bool same(const SparsePauli& a, const SparsePauli& b) {
    return a.n == b.n && a.phase == b.phase && a.x == b.x && a.z == b.z;
}

// Independent oracles for the conversions: per-qubit scans, no word tricks.
SparsePauli scan_sparse(const Pauli& D) {
    SparsePauli s; s.n = D.n; s.phase = D.phase;
    for (int q = 0; q < D.n; ++q) { if (D.xbit(q)) s.x.push_back(q); if (D.zbit(q)) s.z.push_back(q); }
    return s;
}
Pauli build_dense(const SparsePauli& s) {
    Pauli D(s.n); D.phase = s.phase;
    for (int q : s.x) D.setx(q);
    for (int q : s.z) D.setz(q);
    return D;
}

// weight mode: 0 empty, 1 full, 2 sparse (~3/n per plane), 3 half, 4 mixed (x full, z random)
Pauli random_dense(std::mt19937_64& rng, int n, int mode) {
    Pauli D(n);
    std::bernoulli_distribution sp(std::min(1.0, 3.0 / n));
    for (int q = 0; q < n; ++q) {
        bool bx = false, bz = false;
        switch (mode) {
            case 0: break;
            case 1: bx = bz = true; break;
            case 2: bx = sp(rng); bz = sp(rng); break;
            case 3: bx = rng() & 1; bz = rng() & 1; break;
            default: bx = true; bz = rng() & 1; break;
        }
        if (bx) D.setx(q);
        if (bz) D.setz(q);
    }
    D.phase = (int)(rng() % 14) - 5;
    return D;
}

void test_random() {
    std::mt19937_64 rng(20261121);
    long cases = 0;
    const int sizes[] = {1, 63, 64, 65, 127, 128, 129, 200, 1000};
    const int reps[]  = {3000, 3000, 3000, 3000, 3000, 3000, 3000, 2000, 400};
    for (int si = 0; si < 9; ++si) {
        const int n = sizes[si];
        for (int r = 0; r < reps[si]; ++r)
            for (int ma = 0; ma < 5; ++ma) {
                const int mb = (int)(rng() % 5);
                const Pauli A = random_dense(rng, n, ma), B = random_dense(rng, n, mb);
                const SparsePauli SA = scan_sparse(A), SB = scan_sparse(B);

                // conversions
                CHECK(same(from_dense(A), SA));
                CHECK(same(to_dense(SA), A));
                CHECK(same(to_dense(SB), build_dense(SB)));
                CHECK(same(from_dense(to_dense(SB)), SB));

                // anticommute_bit (dense, sparse)
                CHECK_EQ(anticommute_bit(A, SB), Pauli::anticommute_bit(A, B));
                CHECK_EQ(anticommute_bit(B, SA), Pauli::anticommute_bit(B, A));

                // pmul_into (dense <- sparse)
                const Pauli ref = Pauli::multiply(A, B);
                Pauli acc = A;
                pmul_into(acc, SB);
                CHECK(same(acc, ref));
                Pauli acc2 = A;
                pmul_into(acc2, B);
                CHECK(same(acc, acc2));

                // pmul_sparse (sparse, sparse), both forms; the out form reuses a dirty buffer
                const SparsePauli prod = pmul_sparse(SA, SB);
                CHECK(same(prod, scan_sparse(ref)));
                SparsePauli out = SB;
                pmul_sparse(SA, SB, out);
                CHECK(same(out, prod));
                CHECK(same(pmul_sparse(SB, SA), scan_sparse(Pauli::multiply(B, A))));
                ++cases;
            }
    }
    std::fprintf(stderr, "[sparse_pauli] %ld random cases == dense oracle\n", cases);
    CHECK(cases >= 100000);
}

void test_edges() {
    // n = 0 (no words), empty, full weight, and the identity / self products at each boundary size.
    {
        SparsePauli e; e.n = 0;
        CHECK(same(to_dense(e), Pauli(0)));
        CHECK(same(from_dense(Pauli(0)), e));
        CHECK(same(pmul_sparse(e, e), e));
    }
    for (int n : {1, 63, 64, 65, 127, 128, 129}) {
        Pauli full(n);
        for (int q = 0; q < n; ++q) { full.setx(q); full.setz(q); }
        full.phase = 3;
        const SparsePauli sf = from_dense(full);
        CHECK_EQ((int)sf.x.size(), n);
        CHECK_EQ((int)sf.z.size(), n);
        CHECK_EQ(sf.x.back(), n - 1);
        const SparsePauli id = from_dense(Pauli(n));
        CHECK(id.x.empty() && id.z.empty() && id.phase == 0 && id.n == n);
        // full·full = i^(3+3+2·(n&1)) · I
        const SparsePauli sq = pmul_sparse(sf, sf);
        CHECK(sq.x.empty() && sq.z.empty());
        CHECK_EQ(sq.phase, ((6 + 2 * (n & 1)) % 4));
        CHECK(same(pmul_sparse(sf, id), sf));
        CHECK(same(pmul_sparse(id, sf), sf));
        CHECK_EQ(anticommute_bit(full, sf), 0);
        Pauli acc = full; pmul_into(acc, id); CHECK(same(acc, full));
        Pauli acc2 = Pauli(n); pmul_into(acc2, sf); CHECK(same(acc2, full));
    }
}

}  // namespace

int main() {
    RUN(test_edges);
    RUN(test_random);
    REPORT();
}
