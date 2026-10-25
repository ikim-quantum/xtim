// test_tlc_scratch.cpp — Stage TLC (thread-local clean-up) mechanisms, each against an oracle:
//   * the scratch-free rewrites (M1): CliffordTableau::dual_image (word-range scan + stack/heap running
//     words), CertifiedGroupPlanes::reduce (reduced into the result) / reduce_cheap (in place),
//     framed_detail::first_x_off_free (the is_free table replaced) — each vs the pre-TLC code copied
//     verbatim below (renamed *_ref; the thread_local scratch kept as it was);
//   * FramedScratch (M2): copies never carry scratch or charge; framed_active_block / active_set on ONE
//     scratch alternating between two factored states (the (&f, uid) axis cache and the g2a all −1
//     invariant) == a fresh scratch per call == the f-scanning path; the Debug re-entrancy guard.
#include "check.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <fstream>
#include <memory>
#include <sstream>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "qeccore/bare_state_errors.hpp"
#include "qeccore/factored_stab.hpp"
#include "qeccore/framed_sampler.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/sampler.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/state_budget.hpp"
#include "qeccore/twirl_planes.hpp"

using namespace qeccore;

namespace {

// ── verbatim pre-TLC code (6c44b52), renamed ─────────────────────────────────────────────────────
Pauli dual_image_ref(const CliffordTableau& U, const Pauli& T) {
    const int n = U.n;
    const std::vector<Pauli>& Xrow = U.Xrow;
    const std::vector<Pauli>& Zrow = U.Zrow;
    const size_t W = (size_t)((n + 63) / 64);
    static thread_local std::vector<uint64_t> ax, az;
    ax.assign(W, 0); az.assign(W, 0);
    Pauli img(n);              // output X^α Z^β, phase filled at the end
    int acc_phase = 0;
    auto fold = [&](const Pauli& F) {
        int s = 0;
        for (size_t w = 0; w < W; ++w) s += __builtin_popcountll(az[w] & F.x[w]);
        acc_phase += F.phase + 2 * (s & 1);
        for (size_t w = 0; w < W; ++w) { ax[w] ^= F.x[w]; az[w] ^= F.z[w]; }
    };
    const uint64_t* Tx = T.x.data();
    const uint64_t* Tz = T.z.data();
    static thread_local std::vector<int> supp;
    supp.clear();
    for (int w = 0; w < (int)W; ++w) if (Tx[w] | Tz[w]) supp.push_back(w);
    for (int b = 0; b < n; ++b) {
        const Pauli& R = Zrow[b];
        int ac = 0;
        for (int w : supp) ac += __builtin_popcountll(Tx[w] & R.z[w]) + __builtin_popcountll(Tz[w] & R.x[w]);
        if (ac & 1) { img.setx(b); fold(Xrow[b]); }
    }
    for (int b = 0; b < n; ++b) {
        const Pauli& R = Xrow[b];
        int ac = 0;
        for (int w : supp) ac += __builtin_popcountll(Tx[w] & R.z[w]) + __builtin_popcountll(Tz[w] & R.x[w]);
        if (ac & 1) { img.setz(b); fold(Zrow[b]); }
    }
    img.phase = ((T.phase - acc_phase) % 4 + 4) % 4;
    return img;
}

bool sbit_ref(const Pauli& P, int c, int n) { return c < n ? P.xbit(c) : P.zbit(c - n); }

Membership reduce_ref(const CertifiedGroupPlanes& G, const Pauli& p) {
    thread_local Pauli w;
    w = p;
    for (int r = 0; r < G.rank; ++r) {
        int c = G.pivcol[r];
        if (sbit_ref(w, c, G.n_qubits)) pmul_into(w, G.rref[r]);
    }
    bool zero = true;
    for (size_t i = 0; i < w.x.size(); ++i) if (w.x[i] || w.z[i]) { zero = false; break; }
    if (zero) {
        int sign = ((w.phase >> 1) & 1) ? -1 : +1;
        return Membership::in_group(sign, w);
    }
    for (int r = 0; r < G.rank; ++r)
        if (Pauli::anticommute_bit(w, G.rref[r])) return Membership::anti(w);
    return Membership::logical(w);
}

Membership::Verdict reduce_cheap_ref(const CertifiedGroupPlanes& G, const Pauli& p, int& phase_out) {
    thread_local Pauli w;
    w = p;
    const int n_qubits = G.n_qubits;
    const int NWq = (n_qubits + 63) / 64;
    for (int blk = 0; blk < 2; ++blk) {
        std::vector<uint64_t>& vec = blk ? w.z : w.x;
        for (int wq = 0; wq < NWq; ++wq) {
            uint64_t word = vec[wq];
            while (word) {
                const int b = __builtin_ctzll(word);
                const int q = wq * 64 + b;
                const int c = blk ? n_qubits + q : q;
                const int r = G.col_to_row[c];
                if (r >= 0) {
                    pmul_into(w, G.rref[r]);
                    word = (b == 63) ? 0 : (vec[wq] & (~0ULL << (b + 1)));
                } else {
                    word &= word - 1;
                }
            }
        }
    }
    bool zero = true;
    for (size_t i = 0; i < w.x.size(); ++i) if (w.x[i] || w.z[i]) { zero = false; break; }
    if (zero) {
        phase_out = w.phase;
        return Membership::IN_GROUP;
    }
    for (int r = 0; r < G.rank; ++r)
        if (Pauli::anticommute_bit(w, G.rref[r])) return Membership::ANTI;
    return Membership::LOGICAL;
}

// The pre-TLC pivot search of expectation / born_p1 / measure_pauli (the is_free table).
int first_x_off_free_ref(const Pauli& Q, int n, const std::vector<int>& free) {
    std::vector<uint8_t> is_free(n > 0 ? (size_t)n : 1, 0);
    for (int f : free) is_free[(size_t)f] = 1;
    for (int b = 0; b < n; ++b)
        if (Q.xbit(b) && !is_free[(size_t)b]) return b;
    return -1;
}

// ── helpers ────────────────────────────────────────────────────────────────────────────────────
bool same_pauli(const Pauli& a, const Pauli& b) {
    return a.n == b.n && a.x == b.x && a.z == b.z && (a.phase & 3) == (b.phase & 3);
}

CliffordTableau random_tableau(std::mt19937_64& rng, int n, int gates) {
    CliffordTableau U(n);
    std::uniform_int_distribution<int> qd(0, n - 1), kind(0, 7);
    for (int g = 0; g < gates; ++g) {
        const int kd = kind(rng), a = qd(rng), b = qd(rng);
        switch (kd) {
            case 0: U.left_h(a); break;
            case 1: U.left_s(a); break;
            case 2: U.left_sdg(a); break;
            case 3: U.left_x(a); break;
            case 4: U.left_y(a); break;
            case 5: U.left_z(a); break;
            case 6: if (a != b) U.left_cx(a, b); break;
            default: if (a != b) U.left_cz(a, b); break;
        }
    }
    return U;
}

// Random Pauli supported on a random subset of words (zero-word gaps inside the range).
Pauli random_gapped_pauli(std::mt19937_64& rng, int n) {
    Pauli P(n);
    const int W = (n + 63) / 64;
    for (int w = 0; w < W; ++w) {
        if (rng() % 3 == 0) continue;                       // a zero word
        for (int b = 0; b < 64 && w * 64 + b < n; ++b) {
            if (rng() % 7 == 0) P.setx(w * 64 + b);
            if (rng() % 7 == 0) P.setz(w * 64 + b);
        }
    }
    if (rng() % 4 == 0) { P = Pauli(n); P.setz((int)(rng() % n)); }   // single-qubit reads
    P.phase = (int)(rng() & 3);
    return P;
}

// ── M1(d): dual_image ───────────────────────────────────────────────────────────────────────────
void test_dual_image_oracle() {
    std::mt19937_64 rng(1017);
    long cmp = 0;
    for (int n : {1, 5, 63, 64, 65, 300, 1024, 1100}) {   // 1100: W = 18 > 16 takes the heap path
        const int reps = n >= 1024 ? 3 : 12;
        for (int rep = 0; rep < reps; ++rep) {
            CliffordTableau U = random_tableau(rng, n, 4 * n);
            for (int t = 0; t < (n >= 1024 ? 6 : 25); ++t) {
                const Pauli T = random_gapped_pauli(rng, n);
                CHECK(same_pauli(U.dual_image(T), dual_image_ref(U, T)));
                ++cmp;
            }
            for (int q = 0; q < n && q < 8; ++q)
                for (int p = 0; p < 3; ++p) {
                    U.invalidate_dual();   // conjugate_single takes the dual_image path
                    const Pauli T = single_pauli(p, q, n);
                    CHECK(same_pauli(U.conjugate_single(p, q), dual_image_ref(U, T)));
                    ++cmp;
                }
        }
    }
    std::fprintf(stderr, "[tlc] dual_image: %ld comparisons vs the pre-TLC kernel\n", cmp);
}

// ── M1(b,c): reduce / reduce_cheap ──────────────────────────────────────────────────────────────
void test_reduce_oracle() {
    std::mt19937_64 rng(2026);
    long cmp = 0, in_group = 0, logical = 0, anti = 0;
    for (int n : {3, 17, 64, 70, 130}) {
        for (int rep = 0; rep < 6; ++rep) {
            CliffordTableau U = random_tableau(rng, n, 4 * n);
            std::vector<Pauli> gens;   // a commuting set: a random subset of the stabiliser rows
            for (int a = 0; a < n; ++a)
                if (rng() % 4 != 0) {
                    Pauli g = U.Zrow[a];
                    if (rng() & 1) g.phase = (g.phase + 2) & 3;
                    gens.push_back(g);
                }
            if (gens.empty()) gens.push_back(U.Zrow[0]);
            const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(gens, n);
            for (int t = 0; t < 60; ++t) {
                Pauli p(n);
                if (t % 3 == 0) {   // an element of the group (IN_GROUP), times i^k
                    for (const Pauli& g : gens) if (rng() & 1) p = Pauli::multiply(p, g);
                } else if (t % 3 == 1) {   // a logical: a destabiliser/stabiliser row outside gens
                    p = U.Zrow[rng() % n];
                } else {
                    p = random_gapped_pauli(rng, n);
                }
                p.phase = (p.phase + (int)(rng() & 3)) & 3;
                const Membership a = G.reduce(p), b = reduce_ref(G, p);
                CHECK(a.verdict == b.verdict && a.sign == b.sign && same_pauli(a.rep, b.rep));
                Pauli w = p;
                int pa = -7, pb = -7;
                const Membership::Verdict va = G.reduce_cheap(w, pa), vb = reduce_cheap_ref(G, p, pb);
                CHECK(va == vb && pa == pb);
                CHECK(va == a.verdict);
                if (va == Membership::IN_GROUP) CHECK(pa == a.rep.phase);
                in_group += va == Membership::IN_GROUP;
                logical += va == Membership::LOGICAL;
                anti += va == Membership::ANTI;
                cmp += 2;
            }
        }
    }
    std::fprintf(stderr, "[tlc] reduce/reduce_cheap: %ld comparisons (%ld in-group, %ld logical, %ld anti)\n",
                 cmp, in_group, logical, anti);
    CHECK(in_group > 0 && logical > 0 && anti > 0);
}

// ── M1(a): first_x_off_free ─────────────────────────────────────────────────────────────────────
void test_first_x_off_free_oracle() {
    std::mt19937_64 rng(77);
    long cmp = 0, none = 0;
    for (int n : {0, 1, 63, 64, 65, 128, 130, 300}) {
        for (int rep = 0; rep < 400; ++rep) {
            Pauli Q(n);
            const int dens = 1 + (int)(rng() % 40);
            for (int q = 0; q < n; ++q) { if ((int)(rng() % 100) < dens) Q.setx(q); if (rng() & 1) Q.setz(q); }
            // stray X bits at or above n in the last word (outside the Pauli; the n-bit mask must ignore them)
            if (n > 0 && (n & 63) && (rng() & 1)) Q.x[(size_t)((n - 1) >> 6)] |= ~0ull << (n & 63);
            Q.phase = (int)(rng() & 3);
            std::vector<int> perm(n);
            for (int q = 0; q < n; ++q) perm[(size_t)q] = q;
            std::shuffle(perm.begin(), perm.end(), rng);
            const int k = n == 0 ? 0 : (int)(rng() % (std::min(n, 64) + 1));
            std::vector<int> free(perm.begin(), perm.begin() + k);
            AmpKey y = 0;
            for (int d = 0; d < k; ++d) if (Q.xbit(free[(size_t)d])) y |= AmpKey(1) << d;
            const Pauli before = Q;
            const int a = framed_detail::first_x_off_free(Q, n, free, y);
            CHECK(a == first_x_off_free_ref(before, n, free));
            CHECK(Q.x == before.x && Q.z == before.z && Q.phase == before.phase);   // restored
            none += a < 0;
            ++cmp;
        }
    }
    std::fprintf(stderr, "[tlc] first_x_off_free: %ld comparisons (%ld with no off-free X bit)\n", cmp, none);
    CHECK(none > 0);
}

// ── M2: FramedScratch copies never carry scratch ────────────────────────────────────────────────
void test_scratch_not_copied() {
    std::mt19937_64 rng(5);
    FramedSuperposition L(40);
    for (int g = 0; g < 200; ++g) L.apply_clifford((uint8_t)(rng() % 6), (int)(rng() % 40), 0);
    for (int d = 0; d < 6; ++d) L.free.push_back(d);
    L.sync_alpha_k();
    for (AmpKey x = 0; x < 64; ++x) L.entries().emplace_back(x, std::complex<double>(1.0 + (double)x, 0.5));
    L.alpha->normalize();
    FramedScratch warm;
    for (int r = 0; r < 4; ++r) (void)framed_expectation(L, random_gapped_pauli(rng, 40), warm);
    FramedSuperposition c = L;
    framed_canonicalise(c, warm);
    CHECK(warm.charged_bytes() > 0);
    const int64_t live = live_state_bytes();
    FramedScratch copy(warm);                    // a copy starts empty and uncharged
    CHECK_EQ(copy.charged_bytes(), (int64_t)0);
    CHECK(copy.p.born.synd_pk.empty() && copy.p.canon.col.empty());
    CHECK_EQ(live_state_bytes(), live);
    FramedScratch moved(std::move(copy));        // so does a move
    CHECK_EQ(moved.charged_bytes(), (int64_t)0);
    FramedScratch target;
    target = warm;                               // assignment keeps the target's own (empty) scratch
    CHECK_EQ(target.charged_bytes(), (int64_t)0);
    CHECK_EQ(live_state_bytes(), live);
    const int64_t held = warm.charged_bytes();
    warm = target;                               // ... and the warm one keeps its own
    CHECK_EQ(warm.charged_bytes(), held);
}

bool same_state(const FramedSuperposition& A, const FramedSuperposition& B);

// ── M2: the χ-scaled charges — size, and charge BEFORE the buffers grow ──────────────────────────
FramedSuperposition chi64_state(std::mt19937_64& rng) {
    FramedSuperposition L(40);
    for (int g = 0; g < 200; ++g) L.apply_clifford((uint8_t)(rng() % 6), (int)(rng() % 40), 0);
    for (int d = 0; d < 6; ++d) L.free.push_back(d);
    L.sync_alpha_k();
    for (AmpKey x = 0; x < 64; ++x) L.entries().emplace_back(x, std::complex<double>(1.0 + (double)x, 0.5));
    L.alpha->normalize();
    return L;
}

void test_scratch_charge() {
    std::mt19937_64 rng(9);
    const FramedSuperposition L = chi64_state(rng);
    // An anticommuting reference read on L: the measurement scratch is charged > 100 B per branch
    // (>= 3 packed rows + c0 + 3 Pauli images per branch), the former thread-scratch pin.
    int q_anti = -1, p_anti = -1;
    for (int q = 0; q < L.n() && q_anti < 0; ++q)
        for (int p = 0; p < 3 && q_anti < 0; ++p) {
            FramedSuperposition A = L;
            FramedScratch fs;
            const FramedRefRead r = framed_reference_read_toward(A, p, q, 0, fs);
            if (!r.commuting && r.chi_pre == L.chi()) {
                CHECK(fs.p.fmag.charge.bytes() > 100 * (int64_t)r.chi_pre);
                q_anti = q; p_anti = p;
            }
        }
    CHECK(q_anti >= 0);
    // Pad the budget to 8 bytes below full: every χ-scaled part refuses BEFORE growing — the
    // refused part holds no buffer and no charge, and the state is untouched.
    FramedSuperposition A = L;   // (copied before the pad: a state copy is state storage)
    const Pauli P = single_pauli(p_anti, q_anti, L.n());
    const Pauli Q = L.U.dual_valid() ? L.U.conjugate(P) : L.U.dual_image(P);
    std::vector<int> Avec;
    for (int a = 0; a < L.n(); ++a) if (Q.xbit(a)) Avec.push_back(a);
    CHECK(!Avec.empty());
    StateBudgetTicket pad(state_budget_bytes() - live_state_bytes() - 8, "test pad");
    {
        FramedScratch fs;
        bool threw = false;
        Pauli Qx(L.n()); Qx.setx(L.free[0]);
        try { (void)framed_expectation_frame(L, Qx, fs); }
        catch (const BareStateCapacityError& e) { threw = std::string(e.what()).find("Born-probability scratch") != std::string::npos; }
        CHECK(threw);
        CHECK(fs.p.born.synd_pk.capacity() == 0 && fs.p.born.charge.bytes() == 0);
    }
    {
        FramedScratch fs;
        bool threw = false;
        try { (void)framed_measure_anticommuting_general(A, P, Q, Avec, 0.0, fs); }
        catch (const BareStateCapacityError& e) {
            threw = std::string(e.what()).find("framed measurement scratch") != std::string::npos;
        }
        CHECK(threw);
        CHECK(same_state(A, L));
        CHECK(fs.p.fmag.synd_pk.capacity() == 0 && fs.p.fmag.charge.bytes() == 0);
        CHECK(fs.p.born.synd_pk.capacity() == 0 && fs.p.born.charge.bytes() == 0);
    }
}

// ── M2: active block on one scratch across two factored states ──────────────────────────────────
FramedSuperposition bare_of(const std::string& text) {
    ParsedStim ps = parse_stim_circuit(text);
    CHECK(ps.ok());
    NormalizePolicy np;
    np.feedback = NormalizePolicy::Feedback::KeepCoherent;
    np.defer = true;
    NormalizeResult nr = normalize(ps.circuit, np);
    FramedBareState b = build_bare_state_framed(nr.normalized);
    CHECK(!b.rejected);
    return std::move(b.state);
}

bool same_state(const FramedSuperposition& A, const FramedSuperposition& B) {
    if (A.n() != B.n() || A.eps != B.eps || A.free != B.free || A.chi() != B.chi()) return false;
    for (int a = 0; a < A.n(); ++a)
        if (!same_pauli(A.U.Xrow[a], B.U.Xrow[a]) || !same_pauli(A.U.Zrow[a], B.U.Zrow[a])) return false;
    for (int i = 0; i < A.chi(); ++i) {
        if (A.entries()[i].first != B.entries()[i].first) return false;
        if (std::memcmp(&A.entries()[i].second, &B.entries()[i].second, sizeof(std::complex<double>)) != 0)
            return false;
    }
    return true;
}

FactoredDiagError random_error(std::mt19937_64& rng, int n) {
    FactoredDiagError E;
    E.a.assign(n, 0);
    E.v.assign(n, 0);
    for (int q = 0; q < n; ++q) {
        if (rng() % 5 == 0) E.a[q] = (uint8_t)(rng() & 3);
        if (rng() % 6 == 0) E.v[q] = 1;
    }
    const int ncz = (int)(rng() % 5);
    for (int i = 0; i < ncz; ++i) {
        const int a = (int)(rng() % n), b = (int)(rng() % n);
        if (a != b) E.cz.push_back({a, b});
    }
    return E;
}

void test_active_block_shared_scratch() {
    const FramedSuperposition b1 = bare_of("RX 0\nT 0\nRX 1\nT 1\nH 2\nCX 0 2\nCX 1 3\nCX 2 4\nH 5\nM 0 1 2 3 4 5\n");
    const FramedSuperposition b2 = bare_of("RX 0 1 2\nT 0\nT_DAG 1\nCCZ 0 1 2\nH 3\nCX 3 4\nCX 2 5\nH 6\nM 0 1 2 3 4 5 6\n");
    const FactoredBareState f1 = factorize_framed(b1), f2 = factorize_framed(b2);
    CHECK(f1.n > 0 && f2.n > 0 && f1.uid != f2.uid);
    ActiveSetWork w1, w2;
    build_active_set_work(f1, w1);
    build_active_set_work(f2, w2);
    std::mt19937_64 rng(31);
    FramedScratch shared;
    long cmp = 0;
    for (int t = 0; t < 400; ++t) {
        const bool second = (rng() & 1);
        const FactoredBareState& f = second ? f2 : f1;
        ActiveSetWork& w = second ? w2 : w1;
        const FactoredDiagError E = random_error(rng, f.n);
        FramedSuperposition a(0), b(0), c(0);
        std::vector<int> ga, gb, gc;
        framed_active_block(f, E, a, &ga, (t & 2) ? &w : nullptr, shared);   // one scratch, both f
        FramedScratch fresh;
        framed_active_block(f, E, b, &gb, &w, fresh);                       // fresh scratch, fast path
        FramedScratch fresh2;
        framed_active_block(f, E, c, &gc, nullptr, fresh2);                 // fresh, f-scanning path
        CHECK(same_state(a, b) && same_state(a, c) && ga == gb && ga == gc);
        for (int g : shared.p.ablock.g2a) CHECK(g == -1);                  // the between-calls invariant
        ++cmp;
    }
    std::fprintf(stderr, "[tlc] framed_active_block: %ld shared-scratch calls across two factored states\n", cmp);
}

// ── M2: the exact engine's scratch lives for one compile / run (released at every program entry) ──
SamplerProgram* compile_text(const std::string& text) {
    ParsedStim ps = parse_stim_circuit(text);
    CHECK(ps.ok());
    return compile_sampler_program(ps.circuit);
}

void test_engine_scratch_released_per_run() {
    // cultivation d3: the compile's plan extraction and the noisy shots (active block, batch
    // measurement, anticommuting collapses) run the framed kernels on the engine's scratch.
    const std::string src = __FILE__;
    std::ifstream in(src.substr(0, src.rfind('/')) + "/../../tests/data/cultivation_d3_faithful.stim");
    std::stringstream ss;
    ss << in.rdbuf();
    CHECK(!ss.str().empty());
    std::unique_ptr<SamplerProgram, void (*)(SamplerProgram*)> prog(compile_text(ss.str()), &delete_sampler_program);
    FramedCircuitShotSampler* f = sampler_program_framed(*prog);
    CHECK(f != nullptr);
    if (!f) return;
    const FramedScratch& fs = f->shot_sampler().scratch();
    auto empty = [&] {
        return fs.charged_bytes() == 0 && fs.p.fmag.synd_pk.capacity() == 0 && fs.p.bm.rem.capacity() == 0 &&
               fs.p.ablock.g2a.capacity() == 0 && fs.p.conj.Tg.capacity() == 0 && fs.p.dscan.az.capacity() == 0;
    };
    CHECK(empty());                                   // released when the compile returned
    const PackedRecords r = sample_program(*prog, 400, 7);
    CHECK(!r.rejected);
    CHECK(empty());                                   // ... and when the run returned
}

// ── M2: Debug re-entrancy guard ─────────────────────────────────────────────────────────────────
void test_reentrancy_guard() {
#ifndef NDEBUG
    // Nested use of two DIFFERENT parts is legitimate (fmag holds its part while canonicalise holds its own).
    {
        FramedScratch s;
        ScratchUse a(s.p.fmag.busy);
        ScratchUse b(s.p.canon.busy);
        CHECK(s.p.fmag.busy && s.p.canon.busy);
    }
    // Re-entering ONE part aborts (child process).
    const pid_t pid = fork();
    if (pid == 0) {
        FramedScratch s;
        ScratchUse a(s.p.fmag.busy);
        ScratchUse b(s.p.fmag.busy);   // assert fires
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT);
#else
    std::fprintf(stderr, "[tlc] re-entrancy guard: Release build (NDEBUG) — checked by the Debug ctest\n");
#endif
}

}  // namespace

int main() {
    RUN(test_dual_image_oracle);
    RUN(test_reduce_oracle);
    RUN(test_first_x_off_free_oracle);
    RUN(test_scratch_not_copied);
    RUN(test_scratch_charge);
    RUN(test_active_block_shared_scratch);
    RUN(test_engine_scratch_released_per_run);
    RUN(test_reentrancy_guard);
    REPORT();
}
