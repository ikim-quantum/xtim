// test_frame.cpp — Stage 2 S2-T2: the compact Frame against today's dense CliffordTableau (the oracle of the arc).
// Every query of the Frame is a unique operator in canonical (x, z, phase) form, so the oracles are exact: the dense
// tableau's conjugate / forward_image / Zrow / Xrow, and for the corpus the tableau today's path builds.
//   1. Random frames: compact(F_S, W_S, π, U0, C) vs the dense tableau assembled by CliffordTableau's own API
//      (left_rowwise, reset_dual_oracle): ensure_expanded().dense() byte for byte (dual rows included); conjugate /
//      forward_image (lightcone walk) == dense folds; row_z / row_x / rows cache == Zrow / Xrow; sweep (bit-sliced batch,
//      several chunks) == the walks; certified_rows; gates appended after the caches were built; frame_split() recovers a
//      frame with the same queries; the expanded shape answers identically (shape equivalence).
//   2. The corpus (examples/, tests/data): the bare state's pre-flush tableau + pending gates, decomposed by frame_split();
//      (b) ensure_expanded().dense() == today's flushed tableau, (c) conjugate of every single-qubit X_q / Z_q and of
//      random Paulis, forward images, all rows, (d) certified_rows == FramedSuperposition::certified_stabilizers.
//      Prints per circuit: n', |C|, |S|, π ≠ id count, isolated Z_q rows without X_q destabilizer, the storage-rule verdict.
//   test_frame --bench FILE.stim: timings of the key queries on one circuit (surface d=21, cultivation).
#include "check.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "qeccore/frame.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;

namespace {

using LeftGate = CliffordTableau::LeftGate;

bool same_pauli(const Pauli& a, const Pauli& b) { return a.n == b.n && a.phase == b.phase && a.x == b.x && a.z == b.z; }
bool same_sparse(const SparsePauli& a, const SparsePauli& b) { return a.n == b.n && a.phase == b.phase && a.x == b.x && a.z == b.z; }
bool same_sparse_dense(const SparsePauli& a, const Pauli& b) { return same_sparse(a, from_dense(b)); }
bool same_frame(const CliffordTableau& a, const CliffordTableau& b, std::string* why = nullptr) {
    auto fail = [&](const std::string& w) { if (why) *why = w; return false; };
    if (a.n != b.n) return fail("n");
    for (int q = 0; q < a.n; ++q) {
        if (!same_pauli(a.Xrow[(size_t)q], b.Xrow[(size_t)q])) return fail("Xrow[" + std::to_string(q) + "]");
        if (!same_pauli(a.Zrow[(size_t)q], b.Zrow[(size_t)q])) return fail("Zrow[" + std::to_string(q) + "]");
    }
    if (a.dual_valid() != b.dual_valid()) return fail("dual validity");
    if (a.dual_valid())
        for (int q = 0; q < a.n; ++q) {
            if (!same_pauli(a.xinv_row(q), b.xinv_row(q))) return fail("Xinv[" + std::to_string(q) + "]");
            if (!same_pauli(a.zinv_row(q), b.zinv_row(q))) return fail("Zinv[" + std::to_string(q) + "]");
        }
    return true;
}

// The test / oracle seam of the arc (S2-CLEAN: moved here from the shipped Frame, which has no production caller).
// Decompose a dense frame by its identity pairs: every a with Zrow[a] = Z_q and Xrow[a] = X_q (phases 0) goes off the
// support with π(a) = q, the rest is U0 (its dual rebuilt when U's was valid) — the dense tableau today's path builds,
// before its pending gates are flushed, plus those gates.
Frame frame_split(const CliffordTableau& U, std::vector<LeftGate> C) {
    const int n = U.n;
    // an identity pair: Zrow[a] = Z_q and Xrow[a] = X_q, both phase 0
    auto single = [](const Pauli& P, bool isz, int* q) {
        int cnt = 0, at = -1;
        for (size_t w = 0; w < P.x.size(); ++w) {
            if (isz ? P.x[w] : P.z[w]) return false;
            const uint64_t bits = isz ? P.z[w] : P.x[w];
            if (bits) { cnt += __builtin_popcountll(bits); at = (int)(64 * w + __builtin_ctzll(bits)); }
        }
        if (cnt != 1 || P.phase != 0) return false;
        *q = at;
        return true;
    };
    std::vector<int> pi((size_t)n, -1), fidx, wires;
    std::vector<uint8_t> wire_off((size_t)n, 0);
    for (int a = 0; a < n; ++a) {
        int qz, qx;
        if (single(U.Zrow[(size_t)a], true, &qz) && single(U.Xrow[(size_t)a], false, &qx) && qz == qx && !wire_off[(size_t)qz]) {
            pi[(size_t)a] = qz;
            wire_off[(size_t)qz] = 1;
        } else {
            fidx.push_back(a);
        }
    }
    for (int q = 0; q < n; ++q) if (!wire_off[(size_t)q]) wires.push_back(q);
    if ((int)fidx.size() == n) return Frame::expanded(U, std::move(C));
    // the block in local coordinates: every F_S row acts inside W_S (a valid symplectic frame guarantees it: a bit on an
    // identity pair's wire would anticommute with that pair's X_q or Z_q)
    const int s = (int)fidx.size();
    std::vector<int> wpos((size_t)n, -1);
    for (int j = 0; j < s; ++j) wpos[(size_t)wires[(size_t)j]] = j;
    CliffordTableau U0(s);
    auto restrict = [&](const Pauli& P, Pauli& out) {
        out = Pauli(s);
        out.phase = P.phase;
        for (size_t w = 0; w < P.x.size(); ++w) {
            for (uint64_t bits = P.x[w]; bits; bits &= bits - 1) {
                const int j = wpos[(size_t)(64 * w + __builtin_ctzll(bits))];
                if (j < 0) throw std::logic_error("frame_split: a support row acts on an identity pair's wire (not a symplectic frame)");
                out.setx(j);
            }
            for (uint64_t bits = P.z[w]; bits; bits &= bits - 1) {
                const int j = wpos[(size_t)(64 * w + __builtin_ctzll(bits))];
                if (j < 0) throw std::logic_error("frame_split: a support row acts on an identity pair's wire (not a symplectic frame)");
                out.setz(j);
            }
        }
    };
    for (int i = 0; i < s; ++i) {
        restrict(U.Zrow[(size_t)fidx[(size_t)i]], U0.Zrow[(size_t)i]);
        restrict(U.Xrow[(size_t)fidx[(size_t)i]], U0.Xrow[(size_t)i]);
    }
    if (U.dual_valid()) U0.reset_dual(); else U0.invalidate_dual();   // the block's dual is U's restricted (unique rows)
    return Frame::compact(n, std::move(fidx), std::move(wires), std::move(pi), std::move(U0), std::move(C));
}

// The pre-approved contingency rule (PLAN-stage2 §1; not applied, so it lives with the tests — S2-CLEAN): the frame
// would be stored expanded when its dense planes are not larger than its sparse data, 2·n² ≤ 64·(2·|C| + Σ_a |row_a|).
// Reads the rows cache (builds it).
bool storage_rule_prefers_expanded(const Frame& F) {
    uint64_t sum = 0;
    for (const SparsePauli& r : F.rows_z()) sum += r.x.size() + r.z.size();
    for (const SparsePauli& r : F.rows_x()) sum += r.x.size() + r.z.size();
    return 2ull * (uint64_t)F.n * (uint64_t)F.n <= 64ull * (2ull * (uint64_t)F.gates().size() + sum);
}

std::vector<LeftGate> random_gates(std::mt19937_64& rng, int n, size_t count) {
    std::vector<LeftGate> g;
    if (n == 0) return g;
    for (size_t i = 0; i < count; ++i) {
        const uint8_t kind = (uint8_t)(rng() % 8);
        const int a = (int)(rng() % (uint64_t)n);
        int b = -1;
        if (kind >= 6) { if (n < 2) continue; do b = (int)(rng() % (uint64_t)n); while (b == a); }
        g.push_back({kind, a, b});
    }
    return g;
}
// Gates confined to a wire subset (a cone that stays local), to make lightcones of different sizes.
std::vector<LeftGate> random_gates_on(std::mt19937_64& rng, const std::vector<int>& ws, size_t count) {
    std::vector<LeftGate> g;
    if (ws.empty()) return g;
    for (size_t i = 0; i < count; ++i) {
        const uint8_t kind = (uint8_t)(rng() % 8);
        const int a = ws[rng() % ws.size()];
        int b = -1;
        if (kind >= 6) { if (ws.size() < 2) continue; do b = ws[rng() % ws.size()]; while (b == a); }
        g.push_back({kind, a, b});
    }
    return g;
}

CliffordTableau random_tableau(std::mt19937_64& rng, int n, size_t gates) {
    CliffordTableau U(n);
    U.left_rowwise(random_gates(rng, n, gates));   // the dual stays valid (patched per gate)
    return U;
}

// A random Pauli on n qubits: mode 0 empty, 1 single X/Y/Z, 2 sparse (~3), 3 half, 4 full; random phase.
Pauli random_pauli(std::mt19937_64& rng, int n, int mode) {
    Pauli P(n);
    if (n == 0) { P.phase = (int)(rng() & 3); return P; }
    auto set = [&](int q) { const int k = 1 + (int)(rng() % 3); if (k & 1) P.setx(q); if (k & 2) P.setz(q); };
    if (mode == 1) set((int)(rng() % (uint64_t)n));
    else if (mode == 2) { for (int t = 0; t < 3; ++t) set((int)(rng() % (uint64_t)n)); }
    else if (mode == 3) { for (int q = 0; q < n; ++q) if (rng() & 1) set(q); }
    else if (mode == 4) { for (int q = 0; q < n; ++q) set(q); }
    P.phase = (int)(rng() & 3);
    return P;
}

// A random support split: |S| local qubits on random wires W_S with random frame indices F_S and a random bijection
// π on the complements. Returns (fidx, wires, pi).
struct Split { std::vector<int> fidx, wires, pi; };
Split random_split(std::mt19937_64& rng, int n, int s) {
    Split sp;
    std::vector<int> perm_w((size_t)n), perm_f((size_t)n);
    for (int i = 0; i < n; ++i) perm_w[(size_t)i] = perm_f[(size_t)i] = i;
    std::shuffle(perm_w.begin(), perm_w.end(), rng);
    std::shuffle(perm_f.begin(), perm_f.end(), rng);
    sp.wires.assign(perm_w.begin(), perm_w.begin() + s);
    sp.fidx.assign(perm_f.begin(), perm_f.begin() + s);
    std::sort(sp.wires.begin(), sp.wires.end());
    std::sort(sp.fidx.begin(), sp.fidx.end());
    std::vector<int> off_f(perm_f.begin() + s, perm_f.end()), off_w(perm_w.begin() + s, perm_w.end());
    if (rng() & 1) { std::sort(off_f.begin(), off_f.end()); std::sort(off_w.begin(), off_w.end()); }   // π = order-preserving half the time
    sp.pi.assign((size_t)n, -1);
    for (size_t i = 0; i < off_f.size(); ++i) sp.pi[(size_t)off_f[i]] = off_w[i];
    return sp;
}

// The dense tableau of (F_S, W_S, π, U0) WITHOUT the gates, assembled by hand.
CliffordTableau dense_base(int n, const Split& sp, const CliffordTableau& U0) {
    CliffordTableau D(n);
    const int s = (int)sp.fidx.size();
    for (int i = 0; i < s; ++i) {
        const int a = sp.fidx[(size_t)i];
        Pauli& Z = D.Zrow[(size_t)a]; Pauli& X = D.Xrow[(size_t)a];
        std::fill(Z.x.begin(), Z.x.end(), 0ull); std::fill(Z.z.begin(), Z.z.end(), 0ull);
        std::fill(X.x.begin(), X.x.end(), 0ull); std::fill(X.z.begin(), X.z.end(), 0ull);
        Z.phase = U0.Zrow[(size_t)i].phase; X.phase = U0.Xrow[(size_t)i].phase;
        for (int j = 0; j < s; ++j) {
            const int w = sp.wires[(size_t)j];
            if (U0.Zrow[(size_t)i].xbit(j)) Z.setx(w);
            if (U0.Zrow[(size_t)i].zbit(j)) Z.setz(w);
            if (U0.Xrow[(size_t)i].xbit(j)) X.setx(w);
            if (U0.Xrow[(size_t)i].zbit(j)) X.setz(w);
        }
    }
    for (int a = 0; a < n; ++a) {
        if (sp.pi[(size_t)a] < 0) continue;
        Pauli& Z = D.Zrow[(size_t)a]; Pauli& X = D.Xrow[(size_t)a];
        std::fill(Z.z.begin(), Z.z.end(), 0ull); std::fill(X.x.begin(), X.x.end(), 0ull);
        Z.setz(sp.pi[(size_t)a]); X.setx(sp.pi[(size_t)a]);
    }
    if (U0.dual_valid()) D.reset_dual_oracle(); else D.invalidate_dual();
    return D;
}

// S2-T4b: the compile / DEM readers' forms against the dense tableau, on a copy with no rows cache (certified_rows'
// no-cache sweep), and the no-expansion property of every one of them (a compact frame stays compact).
void check_reader_forms(const Frame& F0, const CliffordTableau& D, std::mt19937_64& rng, int n_paulis, const char* tag) {
    const int n = D.n;
    Frame F = F0;
    F.invalidate_caches();
    const bool was_expanded = F.expanded();
    const long e0 = frame_expansions();
    std::vector<Pauli> Ps, Qs;                  // first the scans (below the inverse cache's ski-rental threshold)
    for (int t = 0; t < n_paulis; ++t) {
        Ps.push_back(random_pauli(rng, n, t % 5));
        const Pauli Q = F.inverse_image(Ps.back());
        if (!same_pauli(Q, D.conjugate(Ps.back()))) std::fprintf(stderr, "  [%s] inverse_image mismatch\n", tag);
        CHECK(same_pauli(Q, D.conjugate(Ps.back())));
    }
    // then the inverse-image cache (built here when C ≠ ∅; a compact frame's single conjugate folds it): the same
    // operators
    F.prepare_queries();
    CHECK(F.queries_prepared());
    for (const Pauli& P : Ps) CHECK(same_pauli(F.inverse_image(P), D.conjugate(P)));
    for (int q = 0; q < n; ++q)
        for (int p = 0; p < 3; ++p) CHECK(same_pauli(F.conjugate(single_pauli(p, q, n)), D.conjugate(single_pauli(p, q, n))));
    F.inverse_images(Ps, Qs);
    CHECK(Qs.size() == Ps.size());
    for (size_t i = 0; i < Ps.size() && i < Qs.size(); ++i) CHECK(same_pauli(Qs[i], D.conjugate(Ps[i])));
    for (int a = 0; a < n; ++a) {
        CHECK(same_pauli(F.zrow(a), D.Zrow[(size_t)a]));
        CHECK(same_pauli(F.xrow(a), D.Xrow[(size_t)a]));
    }
    std::vector<int> free;
    std::vector<uint8_t> eps((size_t)n, 0), is_free((size_t)n, 0);
    for (int a = 0; a < n; ++a) { eps[(size_t)a] = (uint8_t)(rng() & 1); if (rng() % 3 == 0) { free.push_back(a); is_free[(size_t)a] = 1; } }
    const std::vector<SparsePauli> cr = F.certified_rows(free, &eps);   // no cache: the certified sweep
    size_t k = 0;
    for (int a = 0; a < n; ++a) {
        if (is_free[(size_t)a]) continue;
        CHECK(k < cr.size());
        if (k >= cr.size()) break;
        Pauli g = D.Zrow[(size_t)a];
        if (eps[(size_t)a]) g.phase = (g.phase + 2) & 3;
        CHECK(same_sparse_dense(cr[k], g));
        ++k;
    }
    CHECK(k == cr.size());
    CHECK(F.expanded() == was_expanded);
    CHECK(frame_expansions() == e0);
    CHECK(F.queries_prepared());
}

void check_queries(const Frame& F, const CliffordTableau& D, std::mt19937_64& rng, int n_paulis, const char* tag) {
    const int n = D.n;
    check_reader_forms(F, D, rng, n_paulis, tag);
    // conjugate (wires -> frame indices) and the backward sweep of the same seeds
    std::vector<SparsePauli> seeds, out;
    for (int t = 0; t < n_paulis; ++t) {
        const Pauli P = random_pauli(rng, n, t % 5);
        const Pauli Q = D.conjugate(P);
        const Pauli R = F.conjugate(P);
        if (!same_pauli(Q, R)) std::fprintf(stderr, "  [%s] conjugate mismatch (n=%d, |C|=%zu, mode %d)\n", tag, n, F.gates().size(), t % 5);
        CHECK(same_pauli(Q, R));
        seeds.push_back(from_dense(P));
    }
    F.sweep(seeds, false, out);
    CHECK(out.size() == seeds.size());
    for (size_t i = 0; i < seeds.size(); ++i) CHECK(same_sparse_dense(out[i], D.conjugate(to_dense(seeds[i]))));
    // forward_image (frame indices -> wires) and the forward sweep
    seeds.clear();
    for (int t = 0; t < n_paulis; ++t) {
        const Pauli P = random_pauli(rng, n, t % 5);
        const Pauli Q = D.forward_image(P);
        const Pauli R = F.forward_image(P);
        if (!same_pauli(Q, R)) std::fprintf(stderr, "  [%s] forward_image mismatch (n=%d, |C|=%zu, mode %d)\n", tag, n, F.gates().size(), t % 5);
        CHECK(same_pauli(Q, R));
        seeds.push_back(from_dense(P));
    }
    F.sweep(seeds, true, out);
    for (size_t i = 0; i < seeds.size(); ++i) CHECK(same_sparse_dense(out[i], D.forward_image(to_dense(seeds[i]))));
    // rows: the walk (cache not yet built), then the cache, then the walk-or-cache entry point again
    for (int a = 0; a < n; ++a) {
        CHECK(same_sparse_dense(F.row_z(a), D.Zrow[(size_t)a]));
        CHECK(same_sparse_dense(F.row_x(a), D.Xrow[(size_t)a]));
    }
    const std::vector<SparsePauli>& rz = F.rows_z();
    const std::vector<SparsePauli>& rx = F.rows_x();
    CHECK((int)rz.size() == n && (int)rx.size() == n);
    for (int a = 0; a < n; ++a) {
        if (!same_sparse_dense(rz[(size_t)a], D.Zrow[(size_t)a])) std::fprintf(stderr, "  [%s] rows_z[%d] mismatch (n=%d)\n", tag, a, n);
        CHECK(same_sparse_dense(rz[(size_t)a], D.Zrow[(size_t)a]));
        CHECK(same_sparse_dense(rx[(size_t)a], D.Xrow[(size_t)a]));
        CHECK(same_sparse(F.row_z(a), rz[(size_t)a]));
        CHECK(same_sparse(F.row_x(a), rx[(size_t)a]));
    }
    // certified rows over a random free set with random eps
    std::vector<int> free;
    std::vector<uint8_t> eps((size_t)n, 0), is_free((size_t)n, 0);
    for (int a = 0; a < n; ++a) { eps[(size_t)a] = (uint8_t)(rng() & 1); if (rng() % 4 == 0) { free.push_back(a); is_free[(size_t)a] = 1; } }
    const std::vector<SparsePauli> cr = F.certified_rows(free, &eps);
    const std::vector<SparsePauli> cr0 = F.certified_rows(free, nullptr);
    size_t k = 0;
    for (int a = 0; a < n; ++a) {
        if (is_free[(size_t)a]) continue;
        CHECK(k < cr.size());
        if (k >= cr.size()) break;
        Pauli g = D.Zrow[(size_t)a];
        CHECK(same_sparse_dense(cr0[k], g));
        if (eps[(size_t)a]) g.phase = (g.phase + 2) & 3;
        CHECK(same_sparse_dense(cr[k], g));
        ++k;
    }
    CHECK(k == cr.size());
}

// ── 1. random frames ──────────────────────────────────────────────────────────────────────────────────────────────
void test_random_frames() {
    std::mt19937_64 rng(0x5e2f);
    const int sizes[] = {1, 2, 3, 5, 17, 63, 64, 65, 130, 300};
    const size_t gate_counts[] = {0, 1, 7, 31, 32, 33, 200};
    int frames = 0;
    for (int n : sizes) {
        for (size_t gc : gate_counts) {
            for (int trial = 0; trial < 3; ++trial) {
                // |S| ∈ {0, 1, random, all}
                const int s = trial == 0 ? 0 : trial == 1 ? (int)(rng() % (uint64_t)(n + 1)) : n;
                const Split sp = random_split(rng, n, s);
                CliffordTableau U0 = random_tableau(rng, s, (size_t)3 * (size_t)s + 1);
                const bool stale = (rng() % 5 == 0);
                if (stale) U0.invalidate_dual();            // a stale block dual stays stale through the expansion
                std::vector<LeftGate> C = random_gates(rng, n, gc == 200 ? gc + (size_t)n : gc);
                CliffordTableau D = dense_base(n, sp, U0);
                D.left_rowwise(C);                          // the oracle's replay: row by row, dual patched when valid
                Frame F = Frame::compact(n, sp.fidx, sp.wires, sp.pi, U0, C);
                CHECK(F.n == n);
                CHECK((int)F.support_rows().size() == s);
                CHECK(F.expanded() == (s == n));
                // (I2) the expanded shape, byte for byte
                {
                    Frame E = F;
                    E.ensure_expanded();
                    CHECK(E.expanded() && E.gates().empty());
                    // QW-B: the expansion builds no dual — it stays valid only when a short replay (left_batch's
                    // row-wise cut) patched an already expanded valid block; built on first use it is the oracle's
                    CHECK(E.dense().dual_valid() == (s == n && !stale && C.size() < 32));
                    E.dense().ensure_dual();
                    D.ensure_dual();
                    std::string why;
                    if (!same_frame(E.dense(), D, &why)) std::fprintf(stderr, "  expanded != dense oracle: %s (n=%d s=%d |C|=%zu stale=%d)\n", why.c_str(), n, s, C.size(), (int)stale);
                    CHECK(same_frame(E.dense(), D));
                    // shape equivalence: the expanded frame answers every query identically
                    check_queries(E, D, rng, 10, "expanded");
                    E.ensure_expanded();                    // idempotent
                    CHECK(same_frame(E.dense(), D));
                }
                check_queries(F, D, rng, 15, "compact");
                // gates appended after the caches were built
                {
                    const std::vector<LeftGate> G2 = random_gates(rng, n, 1 + rng() % 40);
                    F.append(G2);
                    D.left_rowwise(G2);
                    CHECK(F.gates().size() == C.size() + G2.size());
                    check_queries(F, D, rng, 8, "appended");
                    // the lazy duals (the oracle's and the block's) may have been warmed by the conjugate calls above; the
                    // expansion builds no dual (QW-B), so warm the result and compare the dual rows too
                    const bool block_dual = F.block().dual_valid();
                    Frame E = F;
                    E.ensure_expanded();
                    CHECK(!E.dense().dual_valid() || block_dual);   // QW-B: the expansion never builds a dual
                    E.dense().ensure_dual();
                    std::string why;
                    if (!same_frame(E.dense(), D, &why)) std::fprintf(stderr, "  appended: expanded != dense oracle: %s (n=%d s=%d |C|=%zu stale=%d)\n", why.c_str(), n, s, F.gates().size(), (int)stale);
                    CHECK(same_frame(E.dense(), D));
                    for (const LeftGate& g : G2) F.append(g);
                    for (const LeftGate& g : G2) D.left_rowwise({g});
                    check_queries(F, D, rng, 4, "appended-one");
                }
                // frame_split() of the dense base recovers a frame with the same queries and the same expansion
                {
                    CliffordTableau Dbase = dense_base(n, sp, U0);
                    Frame S2 = frame_split(Dbase, F.gates());
                    // every π pair is recovered (identity pairs inside U0 may split off as well, so |S| can shrink)
                    CHECK((int)S2.support_rows().size() <= s);
                    for (int a = 0; a < n; ++a) if (sp.pi[(size_t)a] >= 0) CHECK(S2.pi(a) == sp.pi[(size_t)a]);
                    check_queries(S2, D, rng, 6, "split");
                    const bool block_dual = S2.block().dual_valid();
                    Frame E = S2;
                    E.ensure_expanded();
                    CHECK(!E.dense().dual_valid() || block_dual);   // QW-B: the expansion never builds a dual
                    E.dense().ensure_dual();
                    std::string why;
                    if (!same_frame(E.dense(), D, &why)) std::fprintf(stderr, "  split: expanded != dense oracle: %s (n=%d s=%d |C|=%zu stale=%d)\n", why.c_str(), n, s, F.gates().size(), (int)stale);
                    CHECK(same_frame(E.dense(), D));
                }
                // the storage rule and the accounting run on every shape
                CHECK(F.compact_bytes() > 0);
                (void)storage_rule_prefers_expanded(F);
                ++frames;
            }
        }
    }
    std::fprintf(stderr, "  %d random frames\n", frames);
    // the identity frame and n = 0
    {
        Frame F0(0);
        std::vector<SparsePauli> out;
        F0.sweep({}, true, out);
        CHECK(out.empty());
        CHECK(F0.rows_z().empty());
        Frame E = F0; E.ensure_expanded(); CHECK(E.dense().n == 0);
        Frame I(7);
        CliffordTableau D(7);
        check_queries(I, D, rng, 10, "identity");
        CHECK(I.certified_rows({}).size() == 7);
    }
}

// ── 1b. local lightcones and many chunks: the walk vs the bit-sliced sweep on a globally mixing C and on a local one ──
void test_sweep_many_chunks() {
    std::mt19937_64 rng(0x6e2f);
    const int n = 150;
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<LeftGate> C;
        if (mode == 0) C = random_gates(rng, n, 3000);                        // globally mixing
        else {                                                                // 30 disjoint 5-wire blocks: small cones
            for (int b = 0; b < 30; ++b) {
                std::vector<int> ws; for (int i = 0; i < 5; ++i) ws.push_back(5 * b + i);
                const std::vector<LeftGate> g = random_gates_on(rng, ws, 100);
                C.insert(C.end(), g.begin(), g.end());
            }
            std::shuffle(C.begin(), C.end(), rng);
        }
        const Split sp = random_split(rng, n, 0);                            // F_S = ∅, a random π
        Frame F = Frame::compact(n, sp.fidx, sp.wires, sp.pi, CliffordTableau(0), C);
        CliffordTableau D = dense_base(n, sp, CliffordTableau(0));
        D.left_batch(C);                                                      // the transposed replay this time
        std::vector<SparsePauli> seeds, out;
        for (int t = 0; t < 2 * Frame::kChunk + 37; ++t) seeds.push_back(from_dense(random_pauli(rng, n, 1 + t % 4)));
        F.sweep(seeds, false, out);
        for (size_t i = 0; i < seeds.size(); ++i) {
            const Pauli P = to_dense(seeds[i]);
            CHECK(same_sparse_dense(out[i], D.conjugate(P)));
            if (i % 7 == 0) CHECK(same_pauli(F.conjugate(P), D.conjugate(P)));
        }
        F.sweep(seeds, true, out);
        for (size_t i = 0; i < seeds.size(); ++i) {
            const Pauli P = to_dense(seeds[i]);
            CHECK(same_sparse_dense(out[i], D.forward_image(P)));
            if (i % 7 == 0) CHECK(same_pauli(F.forward_image(P), D.forward_image(P)));
        }
        for (int a = 0; a < n; ++a) CHECK(same_sparse_dense(F.rows_z()[(size_t)a], D.Zrow[(size_t)a]));
    }
}

// ── 1c. const expansion, prepare_queries(), and (I2) direct sum of from_generators ──────────────────────────────
void test_const_expansion_and_prepare() {
    std::mt19937_64 rng(0x8e2f);
    for (int trial = 0; trial < 40; ++trial) {
        const int n = 1 + (int)(rng() % 70);
        const int s = (int)(rng() % (uint64_t)(n + 1));
        const Split sp = random_split(rng, n, s);
        CliffordTableau U0 = random_tableau(rng, s, (size_t)3 * (size_t)s + 1);
        if (trial % 3 == 0) U0.invalidate_dual();
        const std::vector<LeftGate> C = random_gates(rng, n, rng() % 80);
        CliffordTableau D = dense_base(n, sp, U0);
        D.left_rowwise(C);
        // a const& frame expands in place (the shared `bare` of the sampler is a const&)
        const Frame F = Frame::compact(n, sp.fidx, sp.wires, sp.pi, U0, C);
        const bool block_dual = F.block().dual_valid();
        const Frame& cf = F;
        CHECK(cf.expanded() == (s == n));
        cf.ensure_expanded();
        CHECK(cf.expanded() && cf.gates().empty());
        CHECK(!cf.dense().dual_valid() || block_dual);   // QW-B: the expansion never builds a dual
        cf.dense().ensure_dual();                   // the expansion leaves the dense dual stale: warm both
        D.ensure_dual();                            // sides, then compare the dual rows too
        CHECK(same_frame(cf.dense(), D));
        check_queries(cf, D, rng, 4, "const-expanded");
        // dense() on a compact const frame expands it
        const Frame G = Frame::compact(n, sp.fidx, sp.wires, sp.pi, U0, C);
        const CliffordTableau& T = G.dense();
        CHECK(G.expanded());
        T.ensure_dual();
        CHECK(same_frame(T, D));
        // prepare_queries() (the threading contract, S2-CLEAN: warm() is gone): after it, the queries the header lists
        // build nothing lazily (no cache allocated, the duals unchanged) and answer as the dense oracle; never expands
        const Frame W = Frame::compact(n, sp.fidx, sp.wires, sp.pi, U0, C);
        W.prepare_queries();
        CHECK(W.queries_prepared());
        CHECK(!W.expanded() || s == n);
        {
            const size_t bytes = W.compact_bytes();
            const bool bdual = W.block().dual_valid();
            std::vector<Pauli> Ps, outs;
            std::vector<SparsePauli> seeds, fwd;
            for (int t = 0; t < 2 * n + 4; ++t) {            // past the n/64 ski-rental point of an unprepared frame
                const Pauli P = random_pauli(rng, n, t % 5);
                CHECK(same_pauli(W.conjugate(P), D.conjugate(P)));
                CHECK(same_pauli(W.inverse_image(P), D.conjugate(P)));
                CHECK(same_pauli(W.forward_image(P), D.forward_image(P)));
                Ps.push_back(P);
                seeds.push_back(from_dense(P));
            }
            W.inverse_images(Ps, outs);
            for (size_t i = 0; i < Ps.size(); ++i) CHECK(same_pauli(outs[i], D.conjugate(Ps[i])));
            W.sweep(seeds, true, fwd);
            for (size_t i = 0; i < seeds.size(); ++i) CHECK(same_sparse_dense(fwd[i], D.forward_image(Ps[i])));
            for (int a = 0; a < n; ++a) {
                CHECK(same_pauli(W.zrow(a), D.Zrow[(size_t)a]));
                CHECK(same_pauli(W.xrow(a), D.Xrow[(size_t)a]));
                CHECK(same_sparse_dense(W.row_z(a), D.Zrow[(size_t)a]));
            }
            (void)W.certified_rows({0});
            CHECK(W.compact_bytes() == bytes);       // the queries allocated no cache after prepare_queries()
            CHECK(W.block().dual_valid() == bdual);  // and built no dual
            CHECK(W.queries_prepared());
        }
        check_queries(W, D, rng, 4, "prepared");
    }
}

// (I2) for S2-T3: gens = a random stabilizer group on s wires W_S at frame indices F_S (random interleave) plus the
// isolated Z_q on the other wires at the other indices (random π). from_generators(gens) must equal the compact frame
// built from from_generators of the block alone — forward rows, dual rows, byte for byte.
void test_direct_sum_from_generators() {
    std::mt19937_64 rng(77);
    for (int t = 0; t < 400; ++t) {
        const int n = 1 + (int)(rng() % 90), s = (int)(rng() % (uint64_t)(n + 1));
        const Split sp = random_split(rng, n, s);
        CliffordTableau V = random_tableau(rng, s, (size_t)4 * (size_t)s + 3);
        std::vector<Pauli> gens((size_t)n, Pauli(n));
        for (int i = 0; i < s; ++i) {
            Pauli P(n);
            P.phase = V.Zrow[(size_t)i].phase;
            for (int j = 0; j < s; ++j) {
                if (V.Zrow[(size_t)i].xbit(j)) P.setx(sp.wires[(size_t)j]);
                if (V.Zrow[(size_t)i].zbit(j)) P.setz(sp.wires[(size_t)j]);
            }
            gens[(size_t)sp.fidx[(size_t)i]] = P;
        }
        for (int a = 0; a < n; ++a) if (sp.pi[(size_t)a] >= 0) { Pauli P(n); P.setz(sp.pi[(size_t)a]); gens[(size_t)a] = P; }
        std::vector<Pauli> block(V.Zrow.begin(), V.Zrow.end());
        const CliffordTableau full = CliffordTableau::from_generators(gens);
        CliffordTableau blk = s ? CliffordTableau::from_generators(block) : CliffordTableau(0);
        Frame F = Frame::compact(n, sp.fidx, sp.wires, sp.pi, std::move(blk), {});
        F.ensure_expanded();
        std::string why;
        if (!same_frame(F.dense(), full, &why)) std::fprintf(stderr, "  direct sum broken: %s (n=%d s=%d)\n", why.c_str(), n, s);
        CHECK(same_frame(F.dense(), full));
        // and the isolated pairs are what frame_split() finds
        Frame S2 = frame_split(full, {});
        for (int a = 0; a < n; ++a) if (sp.pi[(size_t)a] >= 0) CHECK(S2.pi(a) == sp.pi[(size_t)a]);
    }
}

// ── 2. the corpus ─────────────────────────────────────────────────────────────────────────────────────────────────
std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Circuit deferred_of(const std::string& text, bool* ok) {
    ParsedStim ps = parse_stim_circuit(text);
    *ok = ps.ok();
    if (!*ok) return Circuit{};
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    return normalize(ps.circuit, pol).normalized;
}

// Today's frame of a circuit: the bare state's tableau before its pending gates are flushed, and those gates.
struct TodayFrame { bool ok = false; CliffordTableau U{0}; std::vector<LeftGate> C; CanonicalStabSum state{0}; };
TodayFrame today_frame(const std::string& text) {
    TodayFrame t;
    bool ok = false;
    Circuit d = deferred_of(text, &ok);
    if (!ok) return t;
    BareState b;
    try { b = build_bare_state(d); } catch (const std::exception&) { return t; }
    if (b.rejected) return t;
    // pre-flush (build_bare_state never reads the frame after Step 6): the frame's base U0 ⊕ π densely, and its C
    const Frame& G = b.state.U;
    std::vector<int> pi((size_t)G.n);
    for (int a = 0; a < G.n; ++a) pi[(size_t)a] = G.pi(a);
    t.U = Frame::compact(G.n, G.support_rows(), G.support_wires(), pi, G.block(), {}).dense();
    t.C = G.gates();
    t.state = std::move(b.state);
    t.ok = true;
    return t;
}

bool is_single(const Pauli& P, bool isz, int* q) {
    int cnt = 0, at = -1;
    for (size_t w = 0; w < P.x.size(); ++w) {
        const uint64_t bits = isz ? P.z[w] : P.x[w];
        const uint64_t other = isz ? P.x[w] : P.z[w];
        if (other) return false;
        if (bits) { cnt += __builtin_popcountll(bits); at = (int)(64 * w + __builtin_ctzll(bits)); }
    }
    if (cnt != 1 || P.phase != 0) return false;
    *q = at;
    return true;
}

void corpus_circuit(const std::string& name, const std::string& text, std::mt19937_64& rng) {
    TodayFrame t = today_frame(text);
    if (!t.ok) { std::fprintf(stderr, "  %-40s skipped (refused / unparsed)\n", name.c_str()); return; }
    const int n = t.U.n;
    CliffordTableau D = t.U;
    D.left_batch(t.C);                               // today's flush_gates
    Frame F = frame_split(t.U, t.C);
    // the structure: |S|, π ≠ id, isolated Z_q rows whose destabilizer is not X_q (would keep that row on S)
    // the amendment's S2-T2(b) check: an ISOLATED lone Z_q (no other generator touches wire q) has destabilizer X_q
    // (from_generators' direct sum, S2-T3's recipe); a lone Z_q whose wire another generator touches stays on S.
    int pi_moved = 0, single_z = 0, isolated = 0, isolated_without_x = 0, nonisolated = 0, nonisolated_with_x = 0;
    for (int a = 0; a < n; ++a) {
        int q;
        if (!is_single(t.U.Zrow[(size_t)a], true, &q)) continue;
        ++single_z;
        if (q != a) ++pi_moved;
        bool others = false;
        for (int c = 0; c < n && !others; ++c)
            if (c != a && (t.U.Zrow[(size_t)c].xbit(q) || t.U.Zrow[(size_t)c].zbit(q))) others = true;
        int qx;
        const bool xq = is_single(t.U.Xrow[(size_t)a], false, &qx) && qx == q;
        if (!others) { ++isolated; if (!xq) ++isolated_without_x; CHECK(xq); CHECK(F.pi(a) == q); }
        else { ++nonisolated; if (xq) ++nonisolated_with_x; }
    }
    CHECK(isolated_without_x == 0);
    // (b) the expanded shape is today's tableau byte for byte
    {
        Frame E = F;
        E.ensure_expanded();
        std::string why;
        if (!same_frame(E.dense(), D, &why)) std::fprintf(stderr, "  %s: expanded != today's tableau: %s\n", name.c_str(), why.c_str());
        CHECK(same_frame(E.dense(), D));
    }
    // (c) conjugate of every single-qubit X_q / Z_q (the record operators' form) and of random Paulis, forward images
    for (int q = 0; q < n; ++q) {
        for (int p = 0; p < 3; p += 2) {
            const Pauli P = single_pauli(p, q, n);
            CHECK(same_pauli(F.conjugate(P), D.conjugate(P)));
        }
    }
    check_queries(F, D, rng, 20, name.c_str());
    // (d) certified rows == FramedSuperposition::certified_stabilizers
    {
        FramedSuperposition L = FramedSuperposition::from_css(t.state);
        const std::vector<Pauli> cs = L.certified_stabilizers();
        const std::vector<SparsePauli> cr = F.certified_rows(L.free, &L.eps);
        CHECK(cs.size() == cr.size());
        for (size_t i = 0; i < cs.size() && i < cr.size(); ++i) CHECK(same_sparse_dense(cr[i], cs[i]));
    }
    size_t sum_rows = 0;
    for (const SparsePauli& r : F.rows_z()) sum_rows += r.x.size() + r.z.size();
    for (const SparsePauli& r : F.rows_x()) sum_rows += r.x.size() + r.z.size();
    const size_t dense_bytes = 64ull * (size_t)n * (size_t)((n + 63) / 64);   // 2n rows × (x, z) × W words × 8 B, forward + dual
    std::fprintf(stderr, "  %-40s n'=%5d |C|=%6zu |S|=%4zu single-Z=%4d pi!=id=%4d isolated=%d (no X_q: %d) nonisolated=%d (with X_q: %d) sum|row|=%zu compact=%zu B dense(fwd+dual)=%zu B rule->%s\n",
                 name.c_str(), n, t.C.size(), F.support_rows().size(), single_z, pi_moved, isolated, isolated_without_x, nonisolated,
                 nonisolated_with_x, sum_rows, F.compact_bytes(), dense_bytes, storage_rule_prefers_expanded(F) ? "expanded" : "compact");
}

std::vector<std::string> stim_files(const std::string& dir) {
    // no <filesystem> dependency: a fixed list of the bundled circuits in each directory
    static const char* kNames[] = {
        "cultivation_d3_faithful.stim", "cultivation_d3_rate.stim", "cultivation_d5.stim", "distillation_15_1_3.stim",
        "cube_ccz.stim", "code_switching_faithful.stim", "code_switching_rate.stim", "ch_cultivation.stim",
        "color5_cultivation.stim", "miniature_oracle.stim", "ja_seed23.stim", "magna_color5_fid.stim", "magna_rp2_fid.stim",
        "merlin_switch3_bt27_p0.stim", "n1_mechanisms.stim", "sign_live_bytes_c40022.stim",
        "adaptq_steane_h_producer.stim", "adaptq_qrm_magic_producer.stim", "adaptq_born5_dec_enum_producer.stim",
        "xdem/circuits/ex422_obs.stim", "xdem/ccz_toy.stim", "xdem/rm15rep.stim", "xdem/tdag_cycle.stim", "xdem/toy2obs.stim",
    };
    std::vector<std::string> out;
    for (const char* nm : kNames) {
        const std::string p = dir + "/" + nm;
        std::ifstream f(p);
        if (f.good()) out.push_back(p);
    }
    return out;
}

void test_corpus(const std::vector<std::string>& dirs) {
    std::mt19937_64 rng(0x7e2f);
    int done = 0;
    for (const std::string& dir : dirs)
        for (const std::string& p : stim_files(dir)) {
            const std::string text = slurp(p);
            if (text.empty()) continue;
            corpus_circuit(p.substr(p.find_last_of('/') + 1), text, rng);
            ++done;
        }
    CHECK(done >= 1);
}

// ── bench: the key queries on one circuit ─────────────────────────────────────────────────────────────────────────
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// The key queries on one frame; D (when given) is the dense oracle a sample of the queries is checked against.
void bench_queries(const Frame& F0, const CliffordTableau* D) {
    Frame F = F0;                                                 // fresh caches
    const int n = F.n;
    std::mt19937_64 rng(1);
    std::printf("  |S|=%zu |C|=%zu compact_bytes(no caches)=%zu\n", F.support_rows().size(), F.gates().size(), F.compact_bytes());
    {
        const int reps = std::min(4000, 2 * n);
        std::vector<Pauli> P;
        for (int i = 0; i < reps; ++i) P.push_back(single_pauli(i & 1 ? 2 : 0, (int)(rng() % (uint64_t)n), n));
        const uint64_t w0 = frame_walk_work;
        F.conjugate(P[0]);
        double tmax = 0, tsum = 0;
        for (const Pauli& p : P) { const double a = now_ms(); F.conjugate(p); const double d = now_ms() - a; tsum += d; tmax = std::max(tmax, d); }
        std::printf("  conjugate(single-qubit) x%d: mean %.1f us, max %.1f us, mean cone %.0f gates\n", reps, 1000 * tsum / reps, 1000 * tmax,
                    (double)(frame_walk_work - w0) / (reps + 1));
        if (D) for (int i = 0; i < reps; i += 97) CHECK(same_pauli(F.conjugate(P[(size_t)i]), D->conjugate(P[(size_t)i])));
    }
    {
        std::vector<Pauli> P;
        for (int i = 0; i < 200; ++i) P.push_back(random_pauli(rng, n, 2));
        const uint64_t w0 = frame_walk_work;
        const double a = now_ms();
        for (const Pauli& p : P) F.forward_image(p);
        std::printf("  forward_image(weight~3) x200: mean %.1f us, mean cone %.0f gates\n", 1000 * (now_ms() - a) / 200, (double)(frame_walk_work - w0) / 200);
        if (D) for (int i = 0; i < 200; i += 23) CHECK(same_pauli(F.forward_image(P[(size_t)i]), D->forward_image(P[(size_t)i])));
    }
    {
        const double a = now_ms();
        const std::vector<SparsePauli>& rz = F.rows_z();
        const double b = now_ms();
        size_t sum = 0, mx = 0;
        for (const SparsePauli& r : rz) { sum += r.x.size() + r.z.size(); mx = std::max(mx, r.x.size() + r.z.size()); }
        for (const SparsePauli& r : F.rows_x()) sum += r.x.size() + r.z.size();
        std::printf("  rows cache (2n' rows, one sweep): %.1f ms; sum|row| = %zu, max |row_z| = %zu; storage rule -> %s; compact_bytes(with caches)=%zu\n",
                    b - a, sum, mx, storage_rule_prefers_expanded(F) ? "expanded" : "compact", F.compact_bytes());
        if (D) for (int a2 = 0; a2 < n; a2 += 61) CHECK(same_sparse_dense(rz[(size_t)a2], D->Zrow[(size_t)a2]));
    }
    {
        std::vector<SparsePauli> seeds, out;
        for (int q = 0; q < n; ++q) { seeds.push_back(from_dense(single_pauli(2, q, n))); seeds.push_back(from_dense(single_pauli(0, q, n))); }
        const double a = now_ms();
        F.sweep(seeds, false, out);
        std::printf("  sweep backward (2n' single-qubit seeds = the record operators' form): %.1f ms\n", now_ms() - a);
        if (D) for (size_t i = 0; i < seeds.size(); i += 101) CHECK(same_sparse_dense(out[i], D->conjugate(to_dense(seeds[i]))));
    }
    {
        const double a = now_ms();
        Frame E = F;
        E.ensure_expanded();
        const double b = now_ms();
        const CliffordTableau& T = E.dense();
        size_t bytes = 0;
        for (const Pauli& p : T.Xrow) bytes += (p.x.size() + p.z.size()) * 8 + sizeof(Pauli);
        bytes *= 2;
        if (T.dual_valid()) bytes *= 2;
        std::printf("  ensure_expanded (dense n'=%d tableau, left_batch, reset_dual): %.0f ms; dense bytes ~%zu\n", n, b - a, bytes);
        if (D) { std::string why; if (!same_frame(T, *D, &why)) std::fprintf(stderr, "  expanded != dense: %s\n", why.c_str()); CHECK(same_frame(T, *D)); }
        std::vector<Pauli> P;
        for (int i = 0; i < 1000; ++i) P.push_back(single_pauli(i & 1 ? 2 : 0, (int)(rng() % (uint64_t)n), n));
        const double c = now_ms();
        for (const Pauli& p : P) T.conjugate(p);
        std::printf("  dense conjugate(single-qubit) x1000: mean %.2f us\n", 1000 * (now_ms() - c) / 1000);
    }
}

int bench(const std::string& path) {
    const std::string text = slurp(path);
    if (text.empty()) { std::fprintf(stderr, "bench: cannot read %s\n", path.c_str()); return 2; }
    double t0 = now_ms();
    TodayFrame t = today_frame(text);
    double t1 = now_ms();
    if (!t.ok) { std::fprintf(stderr, "bench: the bare build refused %s\n", path.c_str()); return 2; }
    const int n = t.U.n;
    std::printf("circuit %s: n'=%d |C|=%zu  (bare build incl. parse/defer %.0f ms)\n", path.c_str(), n, t.C.size(), t1 - t0);
    t0 = now_ms();
    Frame F = frame_split(t.U, t.C);
    t1 = now_ms();
    std::printf("[real split at this base] split: %.1f ms\n", t1 - t0);
    CliffordTableau D = t.U;
    D.left_batch(t.C);
    bench_queries(F, &D);
    // The shape Stage-1 T1 + S2-T3 will give a Clifford circuit (F_S = ∅, π = id, U = C): the same gate list on the
    // identity base — a PROXY for the compact frame's query costs on this gate list (its frame is C, not C·U_base).
    std::printf("[proxy: F_S = empty, pi = id, the same C] (the post-T1 shape of a Clifford circuit; frame = C)\n");
    std::vector<int> id((size_t)n);
    for (int a = 0; a < n; ++a) id[(size_t)a] = a;
    Frame P = Frame::compact(n, {}, {}, id, CliffordTableau(0), t.C);
    CliffordTableau DP(n);
    DP.left_batch(t.C);
    bench_queries(P, &DP);
    if (check::failures()) { std::fprintf(stderr, "%d FAILURE(S) in the bench's checks\n", check::failures()); return 1; }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> dirs;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) return bench(argv[i + 1]);
        dirs.push_back(argv[i]);
    }
    RUN(test_random_frames);
    RUN(test_sweep_many_chunks);
    RUN(test_const_expansion_and_prepare);
    RUN(test_direct_sum_from_generators);
    if (!dirs.empty()) {
        std::fprintf(stderr, "[run] test_corpus\n");
        test_corpus(dirs);
    }
    REPORT();
}
