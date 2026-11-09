// test_amp_index.cpp — S2k: DenseAmplitudes with packed keys + the O(χ) key index.
//
// ORACLE: `OldAmps` below is the pre-S2k DenseAmplitudes VERBATIM (byte keys, one heap vector per
// key, first-match linear scans in inner/axpy, erase-in-loop prune). Random operation sequences —
// clone/assign_from, apply_diagonal (±1, ±i and generic complex factors), shift, axpy, inner, scale,
// normalize, drop_index (on constant coordinates), add_index, at, make_dense — run on both, and after
// every operation the two lists must agree EXACTLY: same length, same order, same keys, coefficient
// bits identical (memcmp), inner products bit-identical. Duplicate keys (reachable only through
// hand-built lists, e.g. from_rows) are included to pin the FIRST-match semantics.
// Mechanism (op counts, deterministic): per inner/axpy, the slot reads (probes) are ≥ χ_this+χ_other
// (every entry is indexed / looked up — a restored scan that bypasses the table reads 0) and
// ≤ 8·(χ_this+χ_other) (no Θ(χ²) term); table slots ≤ 4·(χ_this+χ_other)+2 — at χ up to 2^14, in both
// layouts (direct: dense k=14; hash: sparse k=48).
// Ownership: a state retains no index (measure_pauli's is per call); copies charge exactly their
// entries (24 B each); the Born trie's owned index is charged once while it lives and released with
// it. Width: k > 64 refused by name.
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "qeccore/amplitudes.hpp"
#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/state_budget.hpp"

using namespace qeccore;

namespace {

// ── the pre-S2k backend, verbatim (oracle) ──────────────────────────────────────────────────
struct OldAmps {
    int k_;
    using Entry = std::pair<std::vector<uint8_t>, cd>;
    std::vector<Entry> b_;
    explicit OldAmps(int k) : k_(k) {}
    cd at(const std::vector<uint8_t>& x) const {
        for (auto& [key, val] : b_) if (key == x) return val;
        return cd(0, 0);
    }
    void shift(const std::vector<uint8_t>& y) {
        for (auto& [key, val] : b_) for (int i = 0; i < k_; ++i) key[i] ^= y[i];
    }
    void apply_diagonal(cd g0, const std::vector<cd>& g) {
        for (auto& [key, val] : b_) {
            cd factor = g0;
            for (int i = 0; i < k_; ++i) if (key[i]) factor *= g[i];
            val *= factor;
        }
        prune();
    }
    void axpy(cd c, const OldAmps& o) {
        for (auto& [okey, oval] : o.b_) {
            cd contribution = c * oval;
            bool found = false;
            for (auto& [key, val] : b_) if (key == okey) { val += contribution; found = true; break; }
            if (!found) b_.emplace_back(okey, contribution);
        }
        prune();
    }
    cd inner(const OldAmps& o) const {
        cd result(0, 0);
        for (auto& [okey, oval] : o.b_)
            for (auto& [key, val] : b_) if (key == okey) { result += std::conj(val) * oval; break; }
        return result;
    }
    double norm2() const { double s = 0; for (auto& [k, v] : b_) s += std::norm(v); return s; }
    void scale(cd c) {
        if (std::abs(c) < 1e-15) { b_.clear(); return; }
        for (auto& [k, v] : b_) v *= c;
        prune();
    }
    void normalize() {
        double n2 = norm2();
        if (n2 < 1e-30) return;
        double inv = 1.0 / std::sqrt(n2);
        for (auto& [k, v] : b_) v *= inv;
    }
    void drop_index(int i) {
        for (auto& [key, val] : b_) key.erase(key.begin() + i);
        k_--;
    }
    void add_index() { for (auto& [key, val] : b_) key.push_back(0); k_++; }
    void prune() {
        auto it = b_.begin();
        while (it != b_.end()) { if (std::abs(it->second) < 1e-15) it = b_.erase(it); else ++it; }
    }
};

bool bits_eq(cd a, cd b) { return std::memcmp(&a, &b, sizeof(cd)) == 0; }

bool agree(const OldAmps& o, const DenseAmplitudes& n) {
    if (o.k_ != n.k_ || o.b_.size() != n.b_.size()) return false;
    for (size_t i = 0; i < o.b_.size(); ++i) {
        if ((int)o.b_[i].first.size() != n.k_) return false;
        if (pack_key(o.b_[i].first, "test") != n.b_[i].first) return false;
        if (!bits_eq(o.b_[i].second, n.b_[i].second)) return false;
    }
    return true;
}

std::vector<uint8_t> bytes_of(uint64_t x, int k) {
    std::vector<uint8_t> v((size_t)k);
    for (int i = 0; i < k; ++i) v[(size_t)i] = (uint8_t)((x >> i) & 1);
    return v;
}

cd rand_c(std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    return cd(nd(rng), nd(rng));
}

// A random state on both backends: support size chi over k bits; optional duplicate keys.
void fill(std::mt19937_64& rng, int k, int chi, bool dups, OldAmps& o, DenseAmplitudes& n) {
    o.b_.clear(); n.b_.clear(); o.k_ = n.k_ = k;
    const uint64_t mask = k >= 64 ? ~0ull : ((1ull << k) - 1);
    std::vector<uint64_t> keys;
    for (int i = 0; i < chi; ++i) {
        uint64_t x = rng() & mask;
        if (dups && !keys.empty() && (rng() % 5 == 0)) x = keys[rng() % keys.size()];
        keys.push_back(x);
        const cd c = rand_c(rng);
        o.b_.emplace_back(bytes_of(x, k), c);
        n.b_.emplace_back(x, c);
    }
}

std::vector<cd> rand_g(std::mt19937_64& rng, int k) {
    static const cd unit[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    std::vector<cd> g((size_t)k);
    const int mode = (int)(rng() % 3);
    for (int i = 0; i < k; ++i) {
        if (mode == 0) g[(size_t)i] = (rng() & 1) ? cd(-1, 0) : cd(1, 0);
        else if (mode == 1) g[(size_t)i] = unit[rng() % 4];
        else g[(size_t)i] = (rng() % 7 == 0) ? cd(0, 0) : rand_c(rng);   // includes exact zeros (prune)
    }
    return g;
}

void test_random_sequences() {
    std::mt19937_64 rng(20261016);
    long ops = 0, inners = 0, dup_cases = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const int k = 1 + (int)(rng() % 12);
        const int chi = 1 + (int)(rng() % std::min(200, 1 << std::min(k, 10)));
        const bool dups = (trial % 7) == 0;
        dup_cases += dups;
        OldAmps oa(k), ob(k);
        DenseAmplitudes na(k), nb(k);
        fill(rng, k, chi, dups, oa, na);
        fill(rng, k, 1 + (int)(rng() % chi), dups, ob, nb);
        AmpIndex idx;
        for (int step = 0; step < 24; ++step) {
            const int op = (int)(rng() % 10);
            ++ops;
            if (op == 0) {                                  // the trie / born_p1 pass
                ob = oa; nb.assign_from(na);
                const auto g = rand_g(rng, oa.k_);
                ob.apply_diagonal(cd(1, 0), g); nb.apply_diagonal(cd(1, 0), g);
                const uint64_t y = rng() & (oa.k_ >= 64 ? ~0ull : ((1ull << oa.k_) - 1));
                ob.shift(bytes_of(y, oa.k_)); nb.shift_key(y);
                CHECK(agree(ob, nb));
                const cd ri = oa.inner(ob), rn = na.inner(nb, idx);
                CHECK(bits_eq(ri, rn)); ++inners;
                const cd c = (rng() & 1) ? cd(1, 0) : cd(0, -1);
                oa.axpy(c, ob); na.axpy(c, nb, idx);
                oa.normalize(); na.normalize();
            } else if (op == 1) {
                const cd c = rand_c(rng);
                oa.axpy(c, ob); na.axpy(c, nb, idx);
            } else if (op == 2) {
                CHECK(bits_eq(ob.inner(oa), nb.inner(na, idx))); ++inners;
                CHECK(bits_eq(oa.inner(oa), na.inner(na, idx))); ++inners;   // self-inner
            } else if (op == 3) {
                const auto g = rand_g(rng, oa.k_);
                const cd g0 = rand_c(rng);
                oa.apply_diagonal(g0, g); na.apply_diagonal(g0, g);
            } else if (op == 4) {
                const uint64_t y = rng();
                std::vector<uint8_t> yb = bytes_of(y, oa.k_);
                oa.shift(yb); na.shift(yb);                 // the byte-key public signature
            } else if (op == 5) {
                const cd c = (rng() % 9 == 0) ? cd(1e-16, 0) : rand_c(rng);
                oa.scale(c); na.scale(c);
            } else if (op == 6) {
                oa.normalize(); na.normalize();
            } else if (op == 7 && oa.k_ > 1) {
                // make coordinate d constant (project onto x_d = b), then drop it on both
                const int d = (int)(rng() % oa.k_);
                const uint8_t b = (uint8_t)(rng() & 1);
                std::vector<cd> g((size_t)oa.k_, cd(1, 0));
                g[(size_t)d] = cd(0, 0);
                if (b) { std::vector<uint8_t> yb((size_t)oa.k_, 0); yb[(size_t)d] = 1;
                         oa.shift(yb); na.shift(yb); }
                oa.apply_diagonal(cd(1, 0), g); na.apply_diagonal(cd(1, 0), g);   // keeps x_d == 0
                if (b) { std::vector<uint8_t> yb((size_t)oa.k_, 0); yb[(size_t)d] = 1;
                         oa.shift(yb); na.shift(yb); }
                oa.drop_index(d); na.drop_index(d);
                ob = oa; nb.assign_from(na);                // ob/nb: follow a's register
            } else if (op == 8 && oa.k_ < 14) {
                oa.add_index(); na.add_index(); ob.add_index(); nb.add_index();
            } else if (op == 9) {
                const uint64_t x = rng();
                CHECK(bits_eq(oa.at(bytes_of(x, oa.k_)), na.at(bytes_of(x, oa.k_))));
                CHECK(bits_eq(oa.at(bytes_of(x, oa.k_ + 1)), na.at(bytes_of(x, oa.k_ + 1))));
            }
            CHECK(agree(oa, na));
            CHECK(agree(ob, nb));
            CHECK(na.support() == (int)oa.b_.size());
            const double x1 = oa.norm2(), x2 = na.norm2();
            CHECK(std::memcmp(&x1, &x2, sizeof(double)) == 0);
            if (oa.b_.empty()) break;
        }
        // clone / assign_from carry the list verbatim
        auto cl = na.clone();
        CHECK(agree(oa, *static_cast<DenseAmplitudes*>(cl.get())));
    }
    // make_dense (byte-key factory) vs the old merge
    for (int trial = 0; trial < 300; ++trial) {
        const int k = 1 + (int)(rng() % 8);
        std::vector<std::pair<std::vector<uint8_t>, cd>> br;
        for (int i = 0; i < 1 + (int)(rng() % 40); ++i)
            br.emplace_back(bytes_of(rng() % (1ull << k), k), (rng() % 6 == 0) ? cd(0, 0) : rand_c(rng));
        if (!br.empty() && trial % 3 == 0) br.push_back({br[0].first, -br[0].second});   // cancels
        OldAmps o(k);
        for (auto& [key, val] : br) {
            if (std::abs(val) < 1e-15) continue;
            bool found = false;
            for (auto& [ek, ev] : o.b_) if (ek == key) { ev += val; found = true; break; }
            if (!found) o.b_.emplace_back(key, val);
        }
        o.prune();
        auto d = make_dense(k, br);
        CHECK(agree(o, *static_cast<DenseAmplitudes*>(d.get())));
    }
    std::fprintf(stderr, "[amp_index] ops=%ld inner checks=%ld (dup-key trials %ld): all bit-identical\n",
                 ops, inners, dup_cases);
}

void test_op_counts() {
    std::mt19937_64 rng(7);
    for (int layout = 0; layout < 2; ++layout) {
        const int k = layout == 0 ? 14 : 48;                  // dense register (direct) / sparse (hash)
        for (int lg = 4; lg <= 14; lg += 2) {
            const int chi = 1 << lg;
            OldAmps oa(k), ob(k);
            DenseAmplitudes a(k), b(k);
            fill(rng, k, chi, false, oa, a);
            b.assign_from(a);
            b.shift_key(rng() & ((1ull << std::min(k, 13)) - 1));   // partial overlap
            AmpIndex idx;
            const uint64_t p0 = idx.probes, s0 = idx.slots;
            (void)a.inner(b, idx);
            const uint64_t tot = (uint64_t)a.support() + (uint64_t)b.support();
            const uint64_t dp = idx.probes - p0, ds = idx.slots - s0;
            CHECK(dp >= tot);                        // every entry indexed / looked up through the table
            CHECK(dp <= 8 * tot);                    // O(χ_this + χ_other): no scan term
            CHECK(ds <= 4 * (uint64_t)a.support() + 2);
            const uint64_t p1 = idx.probes, s1 = idx.slots;
            const size_t before = a.b_.size();
            a.axpy(cd(0.5, 0), b, idx);
            const uint64_t tot2 = (uint64_t)before + (uint64_t)b.support();
            CHECK(idx.probes - p1 >= tot2);
            CHECK(idx.probes - p1 <= 8 * tot2);
            CHECK(idx.slots - s1 <= 4 * tot2 + 2);
            if (layout == 1) CHECK(idx.charged_bytes() > 0 || chi <= 16);   // heap slots are charged
            std::fprintf(stderr, "[amp_index] k=%d chi=%d: inner probes/(chi_a+chi_b)=%.3f, axpy %.3f\n", k,
                         chi, (double)dp / (double)tot, (double)(idx.probes - p1) / (double)tot2);
        }
    }
}

void test_scratch_ownership() {
    // A state carries no key-lookup scratch: measure_pauli's per-call index (heap beyond 64 slots)
    // is charged during the call only; copies / assignments / moves charge exactly their entries.
    // The Born trie OWNS its index: charged once (high-water) while the trie lives, reused warm,
    // released with the trie.
    const int n = 16, k = 12, chi = 1 << k;
    const int64_t base = live_state_bytes();
    {
        auto f = std::make_unique<FramedSuperposition>(n);
        for (int i = 0; i < k; ++i) f->free.push_back(i);
        f->sync_alpha_k();
        std::mt19937_64 rng(3);
        f->entries().reserve((size_t)chi);
        for (int x = 0; x < chi; ++x) f->entries().emplace_back((AmpKey)x, rand_c(rng));
        f->alpha->normalize();
        const int64_t live0 = live_state_bytes();
        CHECK_EQ(live0 - base, (int64_t)chi * 24);           // 24 B per entry, no per-key block
        const Pauli P = f->U.Xrow[0];   // X on free coordinate 0: case (b), full support kept
        (void)f->measure_pauli(P, 0.3);
        CHECK_EQ(f->chi(), chi);
        CHECK_EQ(live_state_bytes(), live0);                 // nothing retained by the state
        {
            FramedSuperposition g = *f;                      // copy: entries only
            CHECK_EQ(live_state_bytes() - live0, (int64_t)chi * 24);
            (void)g.measure_pauli(P, 0.7);
            CHECK_EQ(live_state_bytes() - live0, (int64_t)chi * 24);
            FramedSuperposition h(n);
            h = g;                                           // copy-assign: entries only
            CHECK_EQ(live_state_bytes() - live0, (int64_t)chi * 48);
            FramedSuperposition m = std::move(h);            // a move charges nothing new
            CHECK_EQ(live_state_bytes() - live0, (int64_t)chi * 48);
        }
        CHECK_EQ(live_state_bytes(), live0);
        {
            // the trie's owned index: direct layout, 2^k slots of 4 B, charged once while held
            FreeDir a; a.a.assign(k, 0); a.b.assign(k, 0); a.a[0] = 1;
            const std::vector<FreeDir> lam = {a, a};
            BornTrie t(*f);
            uint8_t z[1] = {0};
            (void)t.p0_from_scratch(lam, z, 1);
            const int64_t l1 = live_state_bytes();           // + psi/beta working copies + index
            CHECK_EQ(t.idx_.charged_bytes(), (int64_t)chi * 4);
            CHECK_EQ(l1 - live0, (int64_t)chi * 24 * 2 + (int64_t)chi * 4);
            (void)t.p0_from_scratch(lam, z, 1);              // warm reuse: no new charge
            CHECK_EQ(live_state_bytes(), l1);
        }
        CHECK_EQ(live_state_bytes(), live0);                 // the trie's index released with it
    }
    CHECK_EQ(live_state_bytes(), base);
}

void test_width_refusals() {
    bool threw = false;
    try { check_amp_key_width(65, "test"); } catch (const BareStateCapacityError& e) {
        threw = std::string(e.what()).find("one 64-bit word") != std::string::npos;
    }
    CHECK(threw);
    threw = false;
    DenseAmplitudes d(64);
    try { d.add_index(); } catch (const BareStateCapacityError&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { (void)pack_key(std::vector<uint8_t>{0, 2}, "test"); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    FramedSuperposition f(70);
    for (int i = 0; i < 65; ++i) f.free.push_back(i);
    threw = false;
    try { f.sync_alpha_k(); } catch (const BareStateCapacityError&) { threw = true; }
    CHECK(threw);
}

void test_shift_key_mask() {
    // shift_key relabels only the register's k bits: bits >= k stay 0 (the key invariant) even for
    // a wider y (a caller bug; debug builds assert on it, so the wide case runs in Release only).
    DenseAmplitudes d(5);
    for (AmpKey x = 0; x < 32; x += 3) d.b_.emplace_back(x, cd(1.0 + x, 0));
    d.shift_key(0b10110);
    for (auto& e : d.b_) CHECK((e.first >> 5) == 0);
#ifdef NDEBUG
    d.shift_key(~AmpKey(0));
    for (auto& e : d.b_) CHECK((e.first >> 5) == 0);
    DenseAmplitudes z(0);
    z.b_.emplace_back(AmpKey(0), cd(1, 0));
    z.shift_key(~AmpKey(0));
    CHECK(z.b_[0].first == 0);
#endif
    DenseAmplitudes w(64);
    w.b_.emplace_back(AmpKey(5), cd(1, 0));
    w.shift_key(~AmpKey(0));
    CHECK(w.b_[0].first == ~AmpKey(5));
}

void test_branch_sigma_charge_pinned() {
    // A CanonicalStabSum branch (byte sigma, not packed by S2k) is still charged sizeof(Branch) + 48
    // per element: the LITERAL 48 here (not kSigmaHeapBytes) pins the number, via a real
    // live_state_bytes delta.
    const int64_t l0 = live_state_bytes();
    {
        CanonicalStabSum::BranchVec v;
        v.reserve(1000);
        CHECK_EQ(live_state_bytes() - l0, (int64_t)1000 * ((int64_t)sizeof(CanonicalStabSum::Branch) + 48));
    }
    CHECK_EQ(live_state_bytes(), l0);
}

}  // namespace

int main() {
    RUN(test_random_sequences);
    RUN(test_op_counts);
    RUN(test_scratch_ownership);
    RUN(test_width_refusals);
    RUN(test_shift_key_mask);
    RUN(test_branch_sigma_charge_pinned);
    REPORT();
}
