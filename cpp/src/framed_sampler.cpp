#include "qeccore/framed_sampler.hpp"
#include <cstddef>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

#include "qeccore/diag_conjugate.hpp"  // diag_spow/cz_conjugate (E1 expectation conjugation)
#include "qeccore/pauli_kernels.hpp"   // single_pauli / pmul_into / dual_image_rows_scan
#include "qeccore/sampler.hpp"         // FiredPauli / compose_fired (per-alternative atom build)

namespace qeccore {

// ── zero-error plan extraction ─────────────────────────────────────────────────────────────────
// Symbolic mirror of batch_chi1_pre (framed_superposition.cpp): the same conjugate + coin-reduction loop, but
// RECORDING structure instead of drawing outcomes. Read k's outcome bit = base_k XOR parity(coin
// bits selected by cmask_k); a fresh-coin read carries its own new coin id in its mask (out = o ^
// sign there), a determined read carries base = phase-bit ^ eps-parity plus the consumed coin ids.
// Valid for any read set none of which couples S.free (exactly batch_measure's Pass-2 precondition).
static bool extract_chi1_plan(const FramedSuperposition& S, const std::vector<std::pair<int, int>>& reads,
                              FramedPauliShotSampler::BlockPlan& out) {
    const int N = S.n(), W = (N + 63) / 64;
    out.base.assign(reads.size(), 0);
    out.cmask.clear();
    out.basew.clear();
    out.tog.clear();
    out.coin_src.clear();
    out.n_coins = 0;
    out.cw = 0;
    out.rw = 0;
    if (reads.empty()) return true;
    std::vector<Pauli> tg;
    tg.reserve(reads.size());
    for (const auto& r : reads) tg.push_back(single_pauli(r.first, r.second, N));
    std::vector<const Pauli*> tp(reads.size());
    for (size_t j = 0; j < tg.size(); ++j) tp[j] = &tg[j];
    std::vector<Pauli> Qc(reads.size());
    dual_image_rows_scan(S.U.Xrow, S.U.Zrow, N, tp.data(), (int)reads.size(), Qc.data());
    std::vector<uint64_t> eps_pk(W, 0);
    for (int a = 0; a < N; ++a) if (S.eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);

    struct Coin { Pauli Q; int pivot; int id; };
    std::vector<Coin> coins;
    std::vector<std::vector<int>> ids(reads.size());
    for (size_t k = 0; k < reads.size(); ++k) {
        Pauli acc = Qc[k];
        for (const auto& c : coins)
            if ((acc.x[c.pivot >> 6] >> (c.pivot & 63)) & 1ull) { pmul_into(acc, c.Q); ids[k].push_back(c.id); }
        int xnz = -1;
        for (int w = 0; w < W; ++w) if (acc.x[w]) { xnz = (w << 6) + __builtin_ctzll(acc.x[w]); break; }
        if (xnz < 0) {
            int zpar = 0;
            for (int w = 0; w < W; ++w) zpar += __builtin_popcountll(acc.z[w] & eps_pk[w]);
            out.base[k] = (uint8_t)((((acc.phase & 3) == 2) ? 1 : 0) ^ (zpar & 1));
        } else {
            const int id = (int)coins.size();
            ids[k].push_back(id);
            coins.push_back({std::move(acc), xnz, id});
            out.coin_src.push_back((int32_t)k);      // read k created coin id (its fresh coin)
        }
    }
    out.n_coins = (int)coins.size();
    out.cw = (out.n_coins + 63) / 64;
    out.rw = ((int)reads.size() + 63) / 64;
    out.cmask.assign(reads.size() * (size_t)out.cw, 0);
    out.basew.assign((size_t)out.rw, 0);
    out.tog.assign((size_t)out.n_coins * out.rw, 0);
    for (size_t k = 0; k < reads.size(); ++k) {
        if (out.base[k]) out.basew[k >> 6] |= 1ull << (k & 63);
        for (int id : ids[k]) {
            out.cmask[k * (size_t)out.cw + (id >> 6)] |= 1ull << (id & 63);
            out.tog[(size_t)id * out.rw + (k >> 6)] |= 1ull << (k & 63);
        }
    }
    return true;
}

FramedPauliShotSampler::FramedPauliShotSampler(const FramedSuperposition& bare,
                                               std::vector<std::pair<int, int>> reads)
    : n_(bare.n()), reads_(std::move(reads)) {
    init_(bare);
}

void FramedPauliShotSampler::init_(const FramedSuperposition& L) {
    f_ = factorize_framed(L);                  // A (product qubits) (x) B (entangled block)
    build_active_set_work(f_, asw_);
    build_a_lookup(f_, ax1_, sgn_);

    E_.a.assign(n_, 0);
    E_.v.assign(n_, 0);                      // E_.cz stays empty: Pauli errors never entangle
    block_work_ = std::make_unique<FramedSuperposition>(f_.block.n());

    // Fixed read partition: for Pauli errors the active set is ALWAYS exactly B (no CZ pulls),
    // so each read is either a block read (its f_.cls column) or a product-A read, once and for all.
    for (size_t k = 0; k < reads_.size(); ++k) {
        const int q = reads_[k].second;
        const int col = f_.cls[q];
        if (col >= 0) { block_reads_.push_back({reads_[k].first, col}); block_pos_.push_back((int)k); }
        else a_pos_.push_back((int)k);
    }
    // Zero-error A-read kinds (E_ is all-zero here): deterministic value or fair coin.
    a_kind_.reserve(a_pos_.size());
    for (int k : a_pos_) {
        const ReadResult rr = reduced_read(E_, reads_[k].first, reads_[k].second, ax1_, sgn_);
        a_kind_.push_back(rr.deterministic ? (rr.value == +1 ? 0 : 1) : 2);
    }
    // Baked zero-error out vector (deterministic A-reads pre-written) + the coin A-reads, so the
    // zero-error replay is one vector copy + a bulk fair-bit draw (pool order = the a_pos_ order
    // the per-read loop would consume, since deterministic reads draw nothing).
    zero_out_.assign(reads_.size(), +1);
    for (size_t i = 0; i < a_pos_.size(); ++i) {
        if (a_kind_[i] == 1) zero_out_[a_pos_[i]] = -1;
        else if (a_kind_[i] == 2) a_coin_pos_.push_back(a_pos_[i]);
    }
    w2r_.assign(n_, -1);
    for (size_t k = 0; k < reads_.size(); ++k) w2r_[reads_[k].second] = (int)k;
    build_plan_();
}

// ── TreePlan extraction (Stage T1; correction-chain since 2026-07-03) ──────────────────────────
// Build the conditional-coin DAG of a lean block's zero-error measurement by FORCED COLLAPSE on
// block clones. At each level: pick the next free-coupling read in batch_measure's Pass-1 order
// (the shared framed_read_couples_free predicate — order matches the live path by construction);
// its coin probability is the block Born p_plus = pr+/(pr+ + pr-) via framed_expectation, the
// SAME call (same FP operations) the former chi=2 two-variant extractor used, so the chi<=2
// streams are bit-identical; force both outcomes (batch_measure with u=0 / u=1-1e-9 — the live
// collapse code, so the child states are the live path's post-states by construction).
// No coupling read left => leaf: the chi=1 GF(2)-affine remainder (extract_chi1_plan).
//
// CORRECTION CHAINS (the ONE builder, R=0 plans and R>0 channels alike): at each biased coin the
// two forced posts are generically Pauli-related, |post_-> ∝ C·|post_+> (always at chi<=2); C is
// solved off the shared frame's dual structure (solve_pauli_correction) and stored on the node
// with its per-read anticommutation row rflip, and only the +1 subtree is recursed into
// (child[1] = child[0]) — a -1 coin fires the correction at replay: downstream reads draw at the
// toggled law, leaf outcome bits XOR the accumulated rflip. The build is LINEAR in depth for the
// generic case (formerly: the both-branch recursion exploded exponentially BEFORE the bottom-up
// cons could merge — ~cap block collapses per pattern, ~30 ms, and 4096-cap refusals on the
// code_switching class). Genuinely non-Pauli-related posts (possible at skewed chi>2) keep both
// subtrees — exact path dependence, still bounded by the node cap.
//
// HASH-CONSING (bottom-up, structural; non-channel builds): a node's replay behaviour is FULLY
// determined by (read_pos, p_plus, child ids, has_corr [+ p_sel[1], rflip]) — leaves by
// (pos list, plan content bitwise) — so nodes with equal keys are merged (p enters the KEY
// quantized to 2^-44, see below; the stored p stays exact). Sound (equal keys => same replay to
// ~1e-13); with chains it only dedups the residual genuine branching.
//
// Node cap: EXPANSIONS (recursion visits, pre-merge) are capped — default 4096, override
// QEC_TREEPLAN_CAP (read per extraction; compile-time only) — bounding compile cost at ~cap block
// collapses. Over cap / forcing failure / vanishing norm
// => return false; the caller keeps the G3 live-block fallback: live-block base sampling
// (base_outcomes) or error_shot (patterns). (The chi<=2 byte-compat depth clamp was REMOVED
// 2026-07-03 with user approval — a declared stream change for circuits whose chi<=2 patterns
// build depth>1 trees, e.g. cultivation_d5; distributions unchanged, validated by the xengine
// legacy-vs-framed guard. Stage-0 baselines regenerated at the same commit.)
namespace {

int treeplan_cap() {
    const char* e = std::getenv("QEC_TREEPLAN_CAP");
    if (e) {
        int v = std::atoi(e);
        static const auto _tc_quiet = []{ const char* q = std::getenv("XTIM_QUIET"); return q && q[0] && q[0] != '0'; }();
        static bool _tc_noticed = false;
        if (!_tc_noticed && !_tc_quiet) { _tc_noticed = true; fprintf(stderr, "[xtim] QEC_TREEPLAN_CAP=%d: compile-tree cap overridden (diagnostic)\n", v); }
        return v;
    }
    return 4096;
}

// Anticommutation of an arbitrary Pauli with the single-qubit read (p, q) — sampler.cpp's rule.
inline int fr_anti_single(const Pauli& C, int p, int q) {
    const int cx = C.xbit(q) ? 1 : 0, cz = C.zbit(q) ? 1 : 0;
    return p == 0 ? cz : p == 1 ? (cx ^ cz) : cx;
}

// ── Pauli correction solve (channel-mode tree nodes) ───────────────────────────────────────────
// Given the two forced-collapse posts of one biased tree coin, find a Pauli C with
// |post_-> ∝ C|post_+> (global phase irrelevant — the channel consumes only C's anticommutation
// bits). Both posts come from the SAME pre-state through the same collapse code, whose row
// updates depend only on x/z bits — so when Pauli-related they share the frame BITWISE and
// differ only in eps and amplitudes. The shared frame's exact dual structure (Xrow[a]
// anticommutes with Zrow[a] ONLY, rows otherwise pairwise commute, rows Hermitian) turns that
// difference into a syndrome C reads off directly:
//   * eps-normalise both reps (eps[free[j]] = 1 <-> amplitude index shift e_j — two forms of the
//     same physical state; forced collapse may emit either),
//   * the amplitude keys must then match as one coset (index shift m) with coefficients equal up
//     to a single unit gamma and a GF(2)-LINEAR sign form v over the free coordinates,
//   * C = Π_{a: δeps} Xrow[a] · Π_{j∈m} Xrow[free[j]] · Π_{j∈v} Zrow[free[j]].
// Every model condition is CHECKED against the actual amplitude data (signs to 1e-6 against the
// ~1e-12 collapse drift), so any pair outside the model — genuinely non-Pauli-related posts,
// possible at chi>2 — fails cleanly and the builder keeps both subtrees (exact fallback). At
// chi<=2 a correction always exists: fresh coin -> an anticommuting non-free generator; branch
// selection -> the free destabiliser; span rotation -> its paired stabiliser.
static bool solve_pauli_correction(const FramedSuperposition& A, const FramedSuperposition& B,
                                   Pauli& C) {
    const int nb = A.n(), W = (nb + 63) / 64;
    if (B.n() != nb || A.free != B.free) return false;
    const auto& ea = A.entries();
    const auto& eb = B.entries();
    if (ea.size() != eb.size() || ea.empty()) return false;
    const int k = (int)A.free.size();
    if (k > 62 || ea.size() > 64) return false;      // 62: the augmented solve packs <=63 dy bits
    // Rows must match in x/z BITS; phases may differ by 2 (the collapse folds the outcome into
    // the new pivot row's sign) — a Zrow sign is an eps flip in disguise (fold into B's effective
    // eps), an Xrow sign on a FREE destabiliser is an amplitude sign convention (fold into B's
    // coefficients), an Xrow sign on a non-free destabiliser never enters the state.
    static thread_local std::vector<uint8_t> zsgn;   // per row: Zrow_B = (-1)^zsgn Zrow_A
    zsgn.assign(nb, 0);
    uint64_t xsgn = 0;                               // per free j: Xrow_B = (-1) Xrow_A
    for (int a = 0; a < nb; ++a) {
        const Pauli &ga = A.U.Zrow[a], &gb = B.U.Zrow[a];
        const Pauli &da = A.U.Xrow[a], &db = B.U.Xrow[a];
        if (((gb.phase - ga.phase) & 1) || ((db.phase - da.phase) & 1)) {
            return false;
        }
        zsgn[a] = (uint8_t)(((gb.phase - ga.phase) >> 1) & 1);
        for (int w = 0; w < W; ++w)
            if (ga.x[w] != gb.x[w] || ga.z[w] != gb.z[w] || da.x[w] != db.x[w] ||
                da.z[w] != db.z[w])
                return false;
    }
    for (int j = 0; j < k; ++j)
        if (((B.U.Xrow[A.free[j]].phase - A.U.Xrow[A.free[j]].phase) >> 1) & 1)
            xsgn |= 1ull << j;
    // eps-normalised packed entries: canonical index y = sigma XOR (eps over free), eps[free]=0.
    // `zs`/`xs` fold B's row-sign gauge into A's convention.
    auto pack = [k](const FramedSuperposition& S, const uint8_t* zs, uint64_t xs,
                    std::vector<std::pair<uint64_t, std::complex<double>>>& es) {
        uint64_t shift = 0;
        for (int j = 0; j < k; ++j)
            if (S.eps[S.free[j]] ^ (zs ? zs[S.free[j]] : 0)) shift |= 1ull << j;
        es.clear();
        for (const auto& e : S.entries()) {
            uint64_t x = 0;
            for (int j = 0; j < (int)e.first.size(); ++j)
                if (e.first[j]) x |= 1ull << j;
            std::complex<double> c = e.second;
            if (xs && (__builtin_popcountll(x & xs) & 1)) c = -c;
            es.push_back({x ^ shift, c});
        }
    };
    static thread_local std::vector<std::pair<uint64_t, std::complex<double>>> esA, esB;
    pack(A, nullptr, 0, esA);
    pack(B, zsgn.data(), xsgn, esB);
    const size_t ne = esA.size();
    for (size_t cand = 0; cand < ne; ++cand) {       // candidate shifts: esB[0] pairs with esA[cand]
        const uint64_t m = esB[0].first ^ esA[cand].first;
        std::complex<double> gamma(0.0, 0.0);
        bool ok = true;
        uint64_t rows[64];                           // GF(2) solve of <v, dy> = ds, augmented bit 0
        int nr = 0;
        for (size_t i = 0; i < ne && ok; ++i) {
            const uint64_t yb = esA[i].first ^ m;
            size_t j = 0;
            while (j < ne && esB[j].first != yb) ++j;
            if (j == ne || std::abs(esA[i].second) < 1e-12) { ok = false; break; }
            const std::complex<double> r = esB[j].second / esA[i].second;
            if (std::abs(std::abs(r) - 1.0) > 1e-6) { ok = false; break; }
            if (i == 0) { gamma = r; continue; }     // ds relative to entry 0 (constant folds in)
            const std::complex<double> s = r / gamma;
            const int sb = (s.real() < 0.0) ? 1 : 0;
            if (std::abs(s.real() - (sb ? -1.0 : 1.0)) > 1e-6 || std::abs(s.imag()) > 1e-6) {
                ok = false;
                break;
            }
            uint64_t row = ((esA[i].first ^ esA[0].first) << 1) | (uint64_t)sb;
            for (int t = 0; t < nr; ++t)
                if ((row >> 1) >> (63 - __builtin_clzll(rows[t] >> 1)) & 1) row ^= rows[t];
            if (row >> 1) rows[nr++] = row;
            else if (row & 1) ok = false;            // inconsistent: signs not linear in y
        }
        if (!ok) continue;
        uint64_t v = 0;                              // back-substitute (free bits -> 0)
        for (int t = nr - 1; t >= 0; --t) {
            const uint64_t high = rows[t] >> 1;
            const int pv = 63 - __builtin_clzll(high);
            uint64_t val = rows[t] & 1;
            uint64_t rest = high & ~(1ull << pv);
            while (rest) {
                const int b2 = __builtin_ctzll(rest);
                rest &= rest - 1;
                val ^= (v >> b2) & 1;
            }
            v |= val << pv;
        }
        // Assemble C from the shared frame's dual structure.
        C = Pauli(nb);
        static thread_local std::vector<int8_t> freeof;
        freeof.assign(nb, -1);
        for (int j = 0; j < k; ++j) freeof[A.free[j]] = (int8_t)j;
        for (int a = 0; a < nb; ++a) {
            const uint8_t eA = freeof[a] >= 0 ? (uint8_t)0 : A.eps[a];   // normalised eps
            const uint8_t eB = freeof[a] >= 0 ? (uint8_t)0 : (uint8_t)(B.eps[a] ^ zsgn[a]);
            if (eA != eB) pmul_into(C, A.U.Xrow[a]);
        }
        for (int j = 0; j < k; ++j) {
            if ((m >> j) & 1) pmul_into(C, A.U.Xrow[A.free[j]]);
            if ((v >> j) & 1) pmul_into(C, A.U.Zrow[A.free[j]]);
        }
        return true;
    }
    return false;
}

struct TreeBuilder {
    using TreePlan = FramedPauliShotSampler::TreePlan;
    const std::vector<std::pair<int, int>>& breads;
    TreePlan& plan;
    std::map<std::vector<uint64_t>, int32_t> cons;   // structural key -> node id
    int expanded = 0;
    const int cap;
    bool fail = false;
    // THE one builder (correction-chain construction, all modes): at each biased coin, solve for
    // the Pauli relating the two forced-collapse posts (solve_pauli_correction). When found,
    // store it on the node (corr + its per-read anticommutation row rflip) and build ONLY the +1
    // subtree (child[1] = child[0]) — the -1 outcome replays as the same subtree with the
    // downstream read law toggled by rflip (and, on R>0 channels, the observables' signs by the
    // correction's anticommutation with their block restrictions): the leaf-plan coin-correction
    // rule extended to the whole tree. The build is then LINEAR in depth wherever corrections
    // exist (always at chi<=2), where the former path-unique / both-branch builds exploded
    // exponentially; genuinely non-Pauli-related posts (possible at chi>2) keep both subtrees —
    // exact path dependence, bounded by the node cap.
    // `channel` (R>0 expectation mode) changes packaging only: no structural cons (nodes are few
    // and leaf ids must key channel data) and each leaf's block state is captured once, at
    // creation (the channel's leaf posts).
    bool channel = false;
    std::vector<std::pair<int32_t, FramedSuperposition>>* leaf_out = nullptr;

    TreeBuilder(const std::vector<std::pair<int, int>>& br, TreePlan& pl)
        : breads(br), plan(pl), cap(treeplan_cap()) {}

    int32_t intern(TreePlan::Node&& nd, std::vector<uint64_t>&& key) {
        if (!channel) {
            auto it = cons.find(key);
            if (it != cons.end()) return it->second;
        }
        plan.nodes.push_back(std::move(nd));
        const int32_t id = (int32_t)plan.nodes.size() - 1;
        if (!channel) cons.emplace(std::move(key), id);
        return id;
    }

    // Force outcome vi of read `pick` on a copy of B into v. False (-> fail) on a forcing slip.
    bool force_child(const FramedSuperposition& B, int pick, int vi, FramedSuperposition& v) {
        v = B;
        const double fu = (vi == 0) ? 0.0 : 1.0 - 1e-9;
        std::vector<int> tmp;
        const std::vector<std::pair<int, int>> one{breads[pick]};
        batch_measure(v, one, [fu] { return fu; }, tmp);
        return tmp.size() == 1 && tmp[0] == vi;
    }

    // Returns the node id of the sub-DAG for block state B with `resolved` reads already
    // consumed along the path, or -1 with fail=true.
    int32_t build(const FramedSuperposition& B, std::vector<uint8_t>& resolved, int depth) {
        if (fail || ++expanded > cap) { fail = true; return -1; }
        int pick = -1;
        for (size_t k = 0; k < breads.size() && pick < 0; ++k)
            if (!resolved[k] && framed_read_couples_free(B, breads[k].first, breads[k].second))
                pick = (int)k;

        if (pick < 0) {   // leaf: the branch-agreeing remainder, in read order
            TreePlan::Node nd;
            std::vector<std::pair<int, int>> rem;
            for (size_t k = 0; k < breads.size(); ++k)
                if (!resolved[k]) { rem.push_back(breads[k]); nd.pos.push_back((int)k); }
            extract_chi1_plan(B, rem, nd.plan);
            std::vector<uint64_t> key;
            if (!channel) {
                key.push_back(~0ull);                                // leaf tag
                key.push_back((uint64_t)nd.pos.size());
                for (int p : nd.pos) key.push_back((uint64_t)p);
                key.push_back((uint64_t)nd.plan.n_coins);   // fixes cw = the flat row stride too
                for (uint64_t b : nd.plan.base) key.push_back(b);
                for (uint64_t w : nd.plan.cmask) key.push_back(w);
            }
            const int32_t id = intern(std::move(nd), std::move(key));
            if (channel && leaf_out) leaf_out->push_back({id, B});   // pre-remainder state, once
            return id;
        }

        const auto pr = framed_expectation(B, single_pauli(breads[pick].first,
                                                           breads[pick].second, B.n()));
        const double tot = pr.first + pr.second;
        if (tot < 1e-12) { fail = true; return -1; }
        TreePlan::Node nd;
        nd.read_pos = pick;
        nd.p_plus = pr.first / tot;
        nd.p_sel[0] = nd.p_plus;
        // p_sel[1] is the -1-frame draw probability. pr.second/tot (NOT 1.0 - p_plus): it is the
        // bitwise value framed_expectation would report on the Pauli-related -1 post (the |c|^2
        // sets swap roles, and IEEE a+b == b+a makes tot identical), so a toggled chain draw
        // reproduces the former both-branch build's fresh p to the ulp — load-bearing for the
        // R=0 byte-identity of chain-built plans on chi<=2 depth>1 circuits (cultivation_d5).
        nd.p_sel[1] = pr.second / tot;
        resolved[pick] = 1;
        if (nd.p_plus >= 1e-12 && nd.p_sel[1] >= 1e-12) {
            // Both outcomes possible: force both posts, try the Pauli correction (ALL modes —
            // the one linear builder; R=0 plans and R>0 channels consume the same chain).
            FramedSuperposition v0(0), v1(0);
            if (!force_child(B, pick, 0, v0) || !force_child(B, pick, 1, v1)) fail = true;
            Pauli C{0};
            if (!fail && solve_pauli_correction(v0, v1, C)) {
                nd.has_corr = true;
                nd.corr = std::move(C);
                nd.rflip.assign((breads.size() + 63) / 64, 0);
                for (size_t j = 0; j < breads.size(); ++j)
                    if (fr_anti_single(nd.corr, breads[j].first, breads[j].second))
                        nd.rflip[j >> 6] |= 1ull << (j & 63);
                nd.rflip0 = nd.rflip[0];
                plan.any_corr = true;
                plan.n_corr++;
                nd.child[0] = build(v0, resolved, depth + 1);
                nd.child[1] = nd.child[0];
            } else if (!fail) {
                // Genuinely non-Pauli-related posts (possible at skewed chi>2): a real branch —
                // the only recursion that can still grow exponentially (node cap holds it).
                nd.child[0] = build(v0, resolved, depth + 1);
                if (!fail) nd.child[1] = build(v1, resolved, depth + 1);
            }
        } else {
            for (int vi = 0; vi < 2 && !fail; ++vi) {
                if (nd.p_sel[vi] < 1e-12) continue;   // impossible outcome: never drawn per shot
                FramedSuperposition v(0);
                if (!force_child(B, pick, vi, v)) { fail = true; break; }
                nd.child[vi] = build(v, resolved, depth + 1);
            }
        }
        resolved[pick] = 0;                        // backtrack (children restored their own picks)
        if (fail) return -1;
        if (channel) return intern(std::move(nd), {});
        // Cons key: p_plus QUANTIZED to 2^-44 (keys only — the stored node keeps its exact p).
        // Forced collapse down different paths of a product tensor rounds the coefficients
        // differently (measured: 1-2 ulp on the xread family), which defeated bitwise-p merging
        // and blew the flat DAG up to the full binary tree. 2^-44 is ~1e3 x the ulp noise and
        // ~1e11 below any statistical resolution; a straddled rounding boundary only costs a
        // missed merge (bigger DAG), never a wrong one. chi<=2 is untouched (depth<=1 => the
        // root has no merge partner; its exact p replays bit-identically).
        // Corrected nodes additionally key has_corr + the rflip row: a corrected node toggles
        // the downstream law on fire, so merging it with an uncorrected (or differently-flipped)
        // node of equal (pick, p, children) would be replay-unsound.
        const uint64_t pq = (uint64_t)(int64_t)std::llround(nd.p_plus * 17592186044416.0);
        std::vector<uint64_t> key{(uint64_t)(uint32_t)pick, pq,
                                  (uint64_t)(uint32_t)nd.child[0], (uint64_t)(uint32_t)nd.child[1],
                                  (uint64_t)nd.has_corr};
        if (nd.has_corr) {
            key.push_back((uint64_t)(int64_t)std::llround(nd.p_sel[1] * 17592186044416.0));
            for (uint64_t w : nd.rflip) key.push_back(w);
        }
        return intern(std::move(nd), std::move(key));
    }
};

bool extract_tree_plan(const FramedSuperposition& B, const std::vector<std::pair<int, int>>& breads,
                       FramedPauliShotSampler::TreePlan& plan, bool channel = false,
                       std::vector<std::pair<int32_t, FramedSuperposition>>* leaf_out = nullptr) {
    plan.nodes.clear();
    plan.root = -1;
    plan.any_corr = false;
    plan.n_corr = 0;
    if (leaf_out) leaf_out->clear();
    TreeBuilder tb(breads, plan);
    tb.channel = channel;
    tb.leaf_out = leaf_out;
    std::vector<uint8_t> resolved(breads.size(), 0);
    const int32_t root = tb.build(B, resolved, 0);
    if (tb.fail) { plan.nodes.clear(); plan.any_corr = false; if (leaf_out) leaf_out->clear(); return false; }
    plan.root = root;
    return true;
}

// The per-shot fair-bit pool: uniform_real_distribution on mt19937_64 yields k/2^53, so 53 exact
// fair bits per draw. Shared by the tree-DAG leaf replay and the A-read coins that follow it
// (single pool per shot — the consumption-order contract).
struct FairPool {
    const std::function<double()>& rng;
    uint64_t pool = 0;
    int left = 0;
    int coin() {
        if (!left) { pool = (uint64_t)(rng() * 9007199254740992.0); left = 53; }
        const int b = (int)(pool & 1);
        pool >>= 1;
        --left;
        return b;
    }
    // Bulk grab: deposit the next n fair bits into words (bit i of words = the i-th coin), in
    // EXACTLY the order/refill points n successive coin() calls would consume them (LSB-first
    // chunks of the pool, refill on empty) — the byte-stream contract, at O(n/53) cost.
    void take(int n, uint64_t* words) {
        int i = 0;
        while (i < n) {
            if (!left) { pool = (uint64_t)(rng() * 9007199254740992.0); left = 53; }
            const int t = (n - i < left) ? n - i : left;      // t <= 53 < 64
            const uint64_t chunk = pool & ((1ull << t) - 1);
            words[i >> 6] |= chunk << (i & 63);
            if ((i & 63) + t > 64) words[(i >> 6) + 1] |= chunk >> (64 - (i & 63));
            pool >>= t;
            left -= t;
            i += t;
        }
    }
};

// Coin-space image of a read-space toggle row under the leaf plan's fresh-coin gauge.
// extract_chi1_plan pins every fresh-coin read's base to 0 (the coin IS the outcome bit), so a
// Pauli conjugation C of the leaf state does NOT flip fresh reads directly — it flips the COIN
// semantics: coin i's flip A_i = anti(C, its REDUCED coin operator), solved sequentially off the
// owner's cmask row (the anticommutation homomorphism over the coin-reduction products):
//   A_i = tg[pos[owner(i)]] XOR parity(owner's cmask & A_{<i}).
// The leaf outcome flip is then UNIFORMLY  flip_j = tg[pos[j]] XOR parity(cmask_j & A)  — fresh
// reads cancel to 0 through their own coin bit; determined reads pick up exactly
// extract_chi1_plan's eps-parity delta. This is the exact transform of the leaf plan under
// conjugation (validated leaf-for-leaf against the retired both-branch build on the
// cultivation_d5 pattern corpus — the naive "flip every anticommuting read" rule is
// LAW-equivalent, per-coin relabelling, but not byte-faithful). The same A adjusts the drawn
// coin words: c_implied = c_drawn XOR A is the reference-frame coin vector the R>0 channel's
// leaf corrections (coinC / sig masks) are anchored to.
static void coin_toggle_vec(const FramedPauliShotSampler::BlockPlan& bp,
                            const std::vector<int>& pos, const uint64_t* tg,
                            std::vector<uint64_t>& avec) {
    avec.assign((size_t)bp.cw, 0);
    for (int i = 0; i < bp.n_coins; ++i) {
        const int p = pos[bp.coin_src[i]];
        int a = (int)((tg[p >> 6] >> (p & 63)) & 1);
        const uint64_t* row = bp.cmask.data() + (size_t)bp.coin_src[i] * bp.cw;
        for (int w = 0; w < bp.cw; ++w) a ^= (int)(__builtin_popcountll(row[w] & avec[w]) & 1);
        if (a) avec[i >> 6] |= 1ull << (i & 63);   // own bit not yet set: row & avec = earlier coins
    }
}

// Walk the DAG with rng coins (u < p_plus => +1 => child 0, the measure_pauli convention), then
// replay the leaf's GF(2) plan on `fp`. out[block_pos[...]] receives every block read's ±1.
// `leaf_out` (if non-null) receives the leaf node id (the E1 expectation channel's leaf key).
// `oz` is caller scratch for the packed outcome words (reused across shots).
// Corrected (chain) trees — the generic build shape at any R (T.any_corr): `togg`/`fired` must
// be non-null. A corrected node's stored law is the ORIGINAL-path (+1-side reference) one; the
// accumulated toggle word carries the fired corrections' read-anticommutation rows, so a read
// whose toggle bit is set draws at p_sel[1], and the leaf plan replays under the coin-gauge
// transform (coin_toggle_vec) — the exact conjugated law under Π C_fired, byte-for-byte the
// plan the retired both-branch build extracted on the -1-path states. The fired node ids are
// the R>0 channel's per-node sign coins (dead weight at R=0: nodes' osig words are 0 there).
// Correction-free trees (all coins forced or genuinely non-Pauli) take the plain walk,
// bit-for-bit the pre-chain replay.
template <bool CORRS>
void tree_replay_block_impl(const FramedPauliShotSampler::TreePlan& T,
                            const std::vector<int>& block_pos,
                            const std::function<double()>& rng, FairPool& fp,
                            std::vector<uint64_t>& coin_words, std::vector<uint64_t>& oz,
                            std::vector<int>& out, int32_t* leaf_out,
                            std::vector<uint64_t>* togg, std::vector<int32_t>* fired,
                            const int32_t** fired_out, int* n_fired, uint64_t* osign) {
    int32_t id = T.root;
    uint64_t tg0 = 0;                        // register toggle word (block reads <= 64: the norm)
    const bool tg_wide = CORRS && block_pos.size() > 64;
    if (!CORRS) {
        // Plain walk — the R=0 instantiation, byte-identical to the pre-channel replay.
        while (T.nodes[id].read_pos >= 0) {
            const auto& nd = T.nodes[id];
            int v = (rng() < nd.p_plus) ? 0 : 1;   // 0: outcome +1, 1: outcome -1
            // Sub-1e-12 tail of a near-forced coin (u >= p_plus = 1-eps): the unbuilt branch is
            // a numerical zero — take the possible one. (The former two-variant replay
            // dereferenced an EMPTY plan here; this is the safe closure of the same corner.)
            if (nd.child[v] < 0) v ^= 1;
            out[block_pos[nd.read_pos]] = v == 0 ? +1 : -1;
            id = nd.child[v];
        }
        // Baked path observables (expanded channel trees; zeros on plain R=0 trees).
        if (osign) {
            const auto& lfn = T.nodes[id];
            *osign = lfn.osig;
            *fired_out = lfn.fpath.data();
            *n_fired = (int)lfn.fpath.size();
        }
    } else if (!tg_wide) {
        // Corrected walk, narrow (<= 64 block reads, the norm). Only the toggle word tg0 is
        // maintained in the loop (it feeds the next draw); fired ids go to a PRESIZED buffer
        // with a branchless store-always/advance-on-fire index (the fire coin is data-random —
        // a branch on it mispredicts); osig folds in AFTER the walk from the ~1 fired id.
        size_t nf = 0;
        if ((int)fired->size() < T.n_corr) fired->resize((size_t)T.n_corr);
        int32_t* fbuf = fired->data();
        while (T.nodes[id].read_pos >= 0) {
            const auto& nd = T.nodes[id];
            const int f = (int)((tg0 >> nd.read_pos) & 1);
            int v = (rng() < nd.p_sel[f]) ? 0 : 1;   // physical outcome bit (f ? 1-p : p)
            v ^= f;                            // original-frame outcome bit (selects the subtree)
            if (nd.child[v] < 0) v ^= 1;
            out[block_pos[nd.read_pos]] = (v ^ f) == 0 ? +1 : -1;   // physical outcome recorded
            const uint64_t m = (uint64_t)0 - (uint64_t)(nd.has_corr & (v == 1));
            tg0 ^= nd.rflip0 & m;
            fbuf[nf] = id;
            nf += (size_t)(m & 1);
            id = nd.child[v];
        }
        uint64_t osg = 0;
        for (size_t j = 0; j < nf; ++j) osg ^= T.nodes[fbuf[j]].osig;
        *fired_out = fbuf;
        *n_fired = (int)nf;
        *osign = osg;
    } else {
        // Corrected walk, wide (> 64 block reads: rare) — vector toggle rows.
        fired->clear();
        *osign = 0;
        togg->assign((block_pos.size() + 63) / 64, 0);
        while (T.nodes[id].read_pos >= 0) {
            const auto& nd = T.nodes[id];
            const int rp = nd.read_pos;
            const int f = (int)(((*togg)[rp >> 6] >> (rp & 63)) & 1);
            int v = (rng() < nd.p_sel[f]) ? 0 : 1;   // physical outcome bit (f ? 1-p : p)
            v ^= f;                            // original-frame outcome bit (selects the subtree)
            if (nd.child[v] < 0) v ^= 1;
            out[block_pos[rp]] = (v ^ f) == 0 ? +1 : -1;                   // physical, recorded
            if (nd.has_corr && v == 1) {
                fired->push_back(id);
                *osign ^= nd.osig;
                for (size_t w = 0; w < nd.rflip.size(); ++w) (*togg)[w] ^= nd.rflip[w];
            }
            id = nd.child[v];
        }
        *fired_out = fired->data();
        *n_fired = (int)fired->size();
    }
    const auto& lf = T.nodes[id];
    if (leaf_out) *leaf_out = lf.base_leaf >= 0 ? lf.base_leaf : id;   // channel data lives on
    const auto& bp = lf.plan;                                          // the BASE leaf (expanded)
    coin_words.assign((size_t)bp.cw, 0);
    if (bp.n_coins) fp.take(bp.n_coins, coin_words.data());   // bulk fair bits, coin i = bit i
    // Coin-gauge adjustment (coin_toggle_vec): fired corrections (chain walk) or the baked path
    // toggle (expanded leaf variants, cxor) act on the leaf THROUGH the coin vector — XOR the
    // coin-space toggle into the drawn words, so the outcome rows (tog loop below) and the
    // exported coin words (the R>0 channel's implied coins) both live in the reference frame.
    bool tg_any = false;
    if (!CORRS) {
        if (!lf.cxor.empty())
            for (int w = 0; w < bp.cw; ++w) coin_words[w] ^= lf.cxor[w];
    } else {
        if (tg_wide) { for (uint64_t w2 : *togg) tg_any = tg_any || (w2 != 0); }
        else tg_any = tg0 != 0;
        if (tg_any) {
            static thread_local std::vector<uint64_t> avec;
            coin_toggle_vec(bp, lf.pos, tg_wide ? togg->data() : &tg0, avec);
            for (int w = 0; w < bp.cw; ++w) coin_words[w] ^= avec[w];
        }
    }
    const int rw = bp.rw;
    oz.assign(bp.basew.begin(), bp.basew.end());              // packed outcome bits over reads
    for (int w = 0; w < bp.cw; ++w) {                         // XOR the set coins' toggle rows
        uint64_t bits = coin_words[w];
        while (bits) {
            const int c = (w << 6) + __builtin_ctzll(bits);
            bits &= bits - 1;
            const uint64_t* tr = bp.tog.data() + (size_t)c * rw;
            for (int v = 0; v < rw; ++v) oz[v] ^= tr[v];
        }
    }
    if (!CORRS || !tg_any) {
        for (size_t j = 0; j < lf.pos.size(); ++j)
            out[block_pos[lf.pos[j]]] = ((oz[j >> 6] >> (j & 63)) & 1) ? -1 : +1;
    } else {
        for (size_t j = 0; j < lf.pos.size(); ++j) {
            int bit = (int)((oz[j >> 6] >> (j & 63)) & 1);
            bit ^= (int)(((tg_wide ? (*togg)[lf.pos[j] >> 6] : tg0) >> (lf.pos[j] & 63)) & 1);
            out[block_pos[lf.pos[j]]] = bit ? -1 : +1;
        }
    }
}

void tree_replay_block(const FramedPauliShotSampler::TreePlan& T, const std::vector<int>& block_pos,
                       const std::function<double()>& rng, FairPool& fp,
                       std::vector<uint64_t>& coin_words, std::vector<uint64_t>& oz,
                       std::vector<int>& out, int32_t* leaf_out = nullptr,
                       std::vector<uint64_t>* togg = nullptr,
                       std::vector<int32_t>* fired = nullptr,
                       const int32_t** fired_out = nullptr, int* n_fired = nullptr,
                       uint64_t* osign = nullptr) {
    if (T.any_corr)
        tree_replay_block_impl<true>(T, block_pos, rng, fp, coin_words, oz, out, leaf_out, togg,
                                     fired, fired_out, n_fired, osign);
    else
        tree_replay_block_impl<false>(T, block_pos, rng, fp, coin_words, oz, out, leaf_out, togg,
                                      fired, fired_out, n_fired, osign);
}

// ── R>0 expectation-channel construction ───────────────────────────────────────────────────────

// Condition a post-batch_measure work state on the recorded outcomes of its Pass-2 fresh-coin
// reads (E2b, the CASCADE expectation convention). batch_measure's chi==1 remainder draws a
// fresh-coin read symbolically (GF(2) coin bookkeeping) WITHOUT collapsing the state, so an
// observable letter on such a read wire would report the unconditional marginal (exact 0)
// instead of the recorded ±1 — deviating from the leaf-channel/cascade semantics and from the
// physical post-state (dense-oracle-arbitrated in test_framed_r). This projects the state onto
// each recorded outcome: reads whose frame conjugate still carries X-support are force-collapsed
// via framed_measure_anticommuting at a FORCED u (no RNG draw — record streams are untouched);
// already-collapsed / deterministic reads are eigenreads and need nothing. The read projectors
// commute (distinct single-qubit reads), so the order is immaterial; consistency of the later
// determined reads with the recorded coins is guaranteed by batch_chi1's sign bookkeeping.
static void condition_on_recorded(FramedSuperposition& L,
                                  const std::vector<std::pair<int, int>>& reads,
                                  const std::vector<int>& bits) {
    const int N = L.n();
    const int W = (N + 63) / 64;
    static thread_local std::vector<int> A;
    for (size_t k = 0; k < reads.size(); ++k) {
        Pauli Qf = L.U.conjugate_single(reads[k].first, reads[k].second);
        A.clear();
        for (int w = 0; w < W; ++w) {
            uint64_t bits64 = Qf.x[w];
            while (bits64) { A.push_back((w << 6) + __builtin_ctzll(bits64)); bits64 &= bits64 - 1; }
        }
        if (A.empty()) continue;                       // eigenread: nothing to condition
        Pauli P = single_pauli(reads[k].first, reads[k].second, N);
        const double u = bits[k] ? 1.0 - 1e-12 : 0.0;  // force the recorded outcome (bit 1 == -1)
        (void)framed_measure_anticommuting_general(L, P, Qf, A, u);
    }
}

// Static per-observable tensor split at one plan's read bases / block column map. Ez must be the
// all-zero FactoredDiagError (the shot's error enters per shot, conjugated into the OBSERVABLE).
void build_obs_split(FramedObsChannel& ch, const std::vector<Pauli>& obsP, int n,
                     const std::vector<int8_t>& rbasis, const std::vector<int>& w2r,
                     const std::vector<int>& g2a, int nb,
                     const std::vector<uint8_t>& ax1, const std::vector<uint8_t>& sgn,
                     const FactoredDiagError& Ez) {
    ch.g2a = g2a;
    ch.rbasis = rbasis;
    ch.nb = nb;
    ch.split.assign(obsP.size(), {});
    for (size_t r = 0; r < obsP.size(); ++r) {
        const Pauli& P = obsP[r];
        FramedObsChannel::Split& sp = ch.split[r];
        sp.Pb = Pauli(nb);
        const int pre = ((P.phase - P.xz_overlap()) % 4 + 4) % 4;   // i^{ph-#Y} = ±1 (Hermitian)
        sp.s_pre = (pre == 2) ? -1.0 : 1.0;
        for (int q = 0; q < n; ++q) {
            const bool xb = P.xbit(q), zb = P.zbit(q);
            if (!xb && !zb) continue;
            const int sig = xb ? (zb ? 1 : 0) : 2;                  // letter 0:X 1:Y 2:Z
            const int col = g2a[q];
            if (col >= 0) {
                pmul_into(sp.Pb, single_pauli(sig, col, nb));
                sp.block_any = true;
            } else if (w2r[q] >= 0) {                               // read A-qubit
                const int j = w2r[q];
                if (sig != (int)rbasis[j]) { sp.zero = true; break; }   // <o·b|σ≠b|o·b> = 0
                sp.a_reads.push_back(j);
            } else {                                                // unread product qubit
                const ReadResult rr = reduced_read(Ez, sig, q, ax1, sgn);
                if (!rr.deterministic) { sp.zero = true; break; }   // off-axis letter: <σ> = 0
                if (rr.value == -1) sp.a_const = -sp.a_const;
            }
        }
    }
    ch.split_ok = true;
}

// Per-leaf reference posts + Case-B coin corrections + per-observable base values / sign masks.
// The leaf replay mirrors build_cascade_plan's correction derivation (framed_reference_read +
// the algebraic non-free-generator search) on the leaf's captured block state; the derived coin
// sequence is CROSS-CHECKED against the leaf plan's fresh-coin structure (the i-th Case-B read
// must carry its own coin id i in its cmask) — any mismatch refuses the whole channel (the
// caller then routes tier0 shots through the live block, which is always exact).
// Returns false on refusal; on true every leaf is populated and ch.leaves_ok is set.
bool build_obs_leaves(FramedObsChannel& ch, FramedPauliShotSampler::TreePlan& tree,
                      std::vector<std::pair<int32_t, FramedSuperposition>>& leaf_states,
                      const std::vector<std::pair<int, int>>& breads) {
    ch.leaves.assign(tree.nodes.size(), {});
    const int nb = ch.nb;
    const int W = (nb + 63) / 64;
    for (auto& [id, st] : leaf_states) {
        const auto& nd = tree.nodes[id];
        FramedObsChannel::Leaf& lf = ch.leaves[id];
        lf.post = std::move(st);
        FramedSuperposition& work = lf.post;
        bool ok = true;
        for (size_t k2 = 0; k2 < nd.pos.size() && ok; ++k2) {
            const int pj = breads[nd.pos[k2]].first, qc = breads[nd.pos[k2]].second;
            // Classify BEFORE the collapse: commuting test (Q = U†PU has no X bit) + Born coin.
            Pauli Pr = single_pauli(pj, qc, nb);
            Pauli Qr = work.U.dual_valid() ? work.U.conjugate(Pr) : work.U.dual_image(Pr);
            bool anti = false;
            for (int w = 0; w < W; ++w) if (Qr.x[w]) { anti = true; break; }
            const auto pp = framed_expectation(work, Pr);
            const bool coinread = (pp.first > 1e-12 && pp.second > 1e-12);
            if (coinread != anti) { ok = false; break; }   // Case-A at a leaf / degenerate: refuse
            if (coinread) {
                // Case-B correction: a non-free frame generator anticommuting with the read (the
                // build_cascade_plan search; only C's SYMPLECTIC bits enter the sign rule, so the
                // eps phase normalisation and the canonical coset reduction are value-irrelevant
                // here — any valid C gives the same physical <P>).
                bool found = false;
                for (int a = 0; a < nb && !found; ++a) {
                    const Pauli& g = work.U.Zrow[a];
                    if (!fr_anti_single(g, pj, qc)) continue;
                    bool varying = false;
                    for (int f2 : work.free) if (f2 == a) { varying = true; break; }
                    if (varying) continue;
                    lf.coinC.push_back(g);
                    found = true;
                }
                if (!found) { ok = false; break; }
                // coin-id cross-check vs the leaf plan's fresh-coin structure.
                const int cid = (int)lf.coinC.size() - 1;
                if (cid >= nd.plan.n_coins || k2 >= nd.plan.base.size() ||
                    (cid >> 6) >= nd.plan.cw ||
                    !((nd.plan.cmask[k2 * (size_t)nd.plan.cw + (cid >> 6)] >> (cid & 63)) & 1)) {
                    ok = false;
                    break;
                }
            }
            framed_reference_read(work, pj, qc);
        }
        if (ok && (int)lf.coinC.size() != nd.plan.n_coins) ok = false;
        if (!ok) return false;                            // plan-level refusal (live route holds)
        // Renormalise Σ|c|² = 1 (the collapse chain's float drift; build_cascade_plan does this
        // so stabiliser observables read exact ±1).
        double nrm = 0.0;
        for (const auto& br : work.entries()) nrm += std::norm(br.second);
        if (nrm > 0.0) {
            const double s = 1.0 / std::sqrt(nrm);
            for (auto& br : work.entries()) br.second *= s;
        }
        const int cwn = (nd.plan.n_coins + 63) / 64;
        lf.val0.assign(ch.split.size(), 1.0);
        lf.sig.assign(ch.split.size(), {});
        for (size_t r = 0; r < ch.split.size(); ++r) {
            const FramedObsChannel::Split& sp = ch.split[r];
            if (sp.block_any && !sp.zero) {
                const auto pq = framed_expectation(work, sp.Pb);
                lf.val0[r] = pq.first - pq.second;
            }
            lf.sig[r].assign((size_t)cwn, 0);
            for (int c = 0; c < (int)lf.coinC.size(); ++c)
                if (Pauli::anticommute_bit(lf.coinC[c], sp.Pb))
                    lf.sig[r][c >> 6] |= 1ull << (c & 63);
        }
        lf.ok = true;
    }
    // Corrected-node sign masks: bit r = the node's correction Pauli anticommutes with
    // observable r's block restriction. r < 64 lives on the NODE (osig — replay accumulates it
    // into the shot's sign word, no per-shot loop); r >= 64 in the channel's per-node masks.
    const size_t rW = (ch.split.size() + 63) / 64;
    ch.nsig.assign(rW > 1 ? tree.nodes.size() : 0, {});
    for (size_t id2 = 0; id2 < tree.nodes.size(); ++id2) {
        auto& nd = tree.nodes[id2];
        if (!nd.has_corr) continue;
        nd.osig = 0;
        std::vector<uint64_t>* msk = nullptr;
        if (rW > 1) { ch.nsig[id2].assign(rW, 0); msk = &ch.nsig[id2]; }
        for (size_t r = 0; r < ch.split.size(); ++r)
            if (Pauli::anticommute_bit(nd.corr, ch.split[r].Pb)) {
                if (r < 64) nd.osig |= 1ull << r;
                if (msk) (*msk)[r >> 6] |= 1ull << (r & 63);
            }
    }
    ch.leaves_ok = true;
    return true;
}

// ── Corrected-tree partial evaluation (build-time expansion; all R) ────────────────────────────
// A corrected (chain) tree replays with per-shot toggle bookkeeping; for SMALL trees the whole
// bookkeeping can be evaluated at build time instead: expand every corrected node into its two
// explicit outcome subtrees, folding the accumulated toggle into the drawing probabilities
// (p_sel[f] — the chain walk's toggled draw, to the bit) and the leaf plans' outcome bases
// (per-read toggle + the coin-gauge cxor word), and baking each path's observable sign word +
// fired-node list into a thin leaf VARIANT that shares the base leaf's channel data. The
// expanded tree is plain (any_corr = false) and replays on the zero-overhead plain walk.
// BYTE-TRANSPARENT vs the chain replay (same draw values in the same order, same leaf bits,
// same exported coin words) — expansion is purely a replay-speed choice, so R=0 plans use it
// too (the corrected walk costs the k-sweep family ~10-20%/shot over the plain walk). Purely
// mechanical: no state collapses, O(expanded size); bounded by the existing node cap (deep
// chains — where the old path-unique build exploded — stay in chain form and pay the per-shot
// toggles instead). Original nodes are retained (appended-to vector): fired ids and base-leaf
// ids keep indexing the channel's corr Paulis / nsig masks / leaf data. Narrow trees only
// (<= 64 block reads). At R>0 it must run AFTER the channel build (it bakes the nodes' osig
// words); at R=0 right after extraction (osig words are 0).
void expand_corrected_tree(FramedPauliShotSampler::TreePlan& T, size_t nreads) {
    using TreePlan = FramedPauliShotSampler::TreePlan;
    if (!T.any_corr || nreads > 64) return;
    const int cap = treeplan_cap();
    // Exact expansion size (corrected nodes double their subtree; genuine branches add).
    static thread_local std::vector<int64_t> sz;
    sz.assign(T.nodes.size(), 0);
    // nodes are appended bottom-up by the builder: children have smaller ids than parents.
    for (size_t i = 0; i < T.nodes.size(); ++i) {
        const auto& nd = T.nodes[i];
        if (nd.read_pos < 0) { sz[i] = 1; continue; }
        int64_t c0 = nd.child[0] >= 0 ? sz[nd.child[0]] : 0;
        int64_t c1 = nd.child[1] >= 0 ? sz[nd.child[1]] : 0;
        sz[i] = 1 + (nd.has_corr ? 2 * c0 : c0 + c1);
        if (sz[i] > cap) return;                       // too big: keep the chain form
    }
    if (T.root < 0 || sz[T.root] > cap) return;
    struct Baker {
        TreePlan& T;
        std::vector<int32_t> fpath;
        uint64_t os = 0;
        int32_t bake(int32_t id, uint64_t tg) {
            TreePlan::Node src = T.nodes[id];          // copy: push_back below invalidates refs
            if (src.read_pos < 0) {
                TreePlan::Node lf;
                lf.plan = src.plan;
                for (size_t j = 0; j < src.pos.size(); ++j)   // per-read toggle -> the bases
                    if ((tg >> src.pos[j]) & 1) lf.plan.basew[j >> 6] ^= 1ull << (j & 63);
                // Coin-gauge part of the toggle (coin_toggle_vec): baked as cxor — replay XORs
                // it into the drawn coin words (outcome rows via the tog loop + the channel's
                // implied-coin export), the chain walk's leaf transform evaluated at this path.
                std::vector<uint64_t> avec;
                coin_toggle_vec(src.plan, src.pos, &tg, avec);
                bool any = false;
                for (uint64_t w : avec) any = any || (w != 0);
                if (any) lf.cxor = std::move(avec);
                lf.pos = src.pos;
                lf.base_leaf = src.base_leaf >= 0 ? src.base_leaf : id;
                lf.osig = os;                          // baked path observable sign word
                lf.fpath = fpath;
                T.nodes.push_back(std::move(lf));
                return (int32_t)T.nodes.size() - 1;
            }
            TreePlan::Node nd;
            nd.read_pos = src.read_pos;
            const int f = (int)((tg >> src.read_pos) & 1);
            nd.p_plus = src.p_sel[f];              // the chain walk's toggled draw, to the bit
            nd.p_sel[0] = nd.p_plus;
            nd.p_sel[1] = src.p_sel[f ^ 1];
            for (int d = 0; d < 2; ++d) {              // d = drawn (physical) outcome bit
                const int b = d ^ f;                   // original-frame outcome bit
                if (src.has_corr) {
                    if (b == 1) {
                        fpath.push_back(id);
                        os ^= src.osig;
                        nd.child[d] = bake(src.child[0], tg ^ src.rflip0);
                        os ^= src.osig;
                        fpath.pop_back();
                    } else {
                        nd.child[d] = bake(src.child[0], tg);
                    }
                } else {
                    nd.child[d] = src.child[b] < 0 ? -1 : bake(src.child[b], tg);
                }
            }
            T.nodes.push_back(std::move(nd));
            return (int32_t)T.nodes.size() - 1;
        }
    };
    Baker bk{T, {}, 0};
    T.root = bk.bake(T.root, 0);
    T.any_corr = false;                                // plain replay from here on
    T.expanded = true;
}

}  // namespace

void FramedPauliShotSampler::build_plan_() {
    plan_ok_ = extract_tree_plan(f_.block, block_reads_, tree_);
    if (plan_ok_) expand_corrected_tree(tree_, block_reads_.size());
}

void FramedPauliShotSampler::live_base_outcomes(const std::function<double()>& rng,
                                                std::vector<int>& out) {
    if (block_reads_.empty()) *block_work_ = f_.block;   // keep block_work() = the (un)collapsed post
    live_base_shot_(rng, out);
    // Cascade convention (E2b): block_work_ is the expectation channel's post-state — condition
    // it on the recorded fresh-coin outcomes (bits_ holds the block outcomes; no RNG consumed).
    if (!block_reads_.empty()) condition_on_recorded(*block_work_, block_reads_, bits_);
}

bool FramedPauliShotSampler::rebuild_plan_channel(
        std::vector<std::pair<int32_t, FramedSuperposition>>& leaf_states) {
    plan_ok_ = extract_tree_plan(f_.block, block_reads_, tree_, /*channel=*/true, &leaf_states);
    return plan_ok_;
}

void FramedPauliShotSampler::sample_shot(const std::vector<std::pair<int, int>>& fired,
                                         const std::function<double()>& rng,
                                         std::vector<int>& out) {
    out.assign(reads_.size(), +1);

    // Fold the fired Paulis into the diagonal-error normal form: X -> v, Z -> S^2 (a += 2),
    // Y = iXZ -> both (the global phase i is irrelevant to outcomes). Repeated hits cancel.
    touched_.clear();
    bool touches_B = false;
    for (const auto& fq : fired) {
        const int q = fq.first, t = fq.second;
        if (!E_.a[q] && !E_.v[q]) touched_.push_back(q);
        if (t != 0) E_.a[q] = (uint8_t)((E_.a[q] + 2) & 3);   // Z component (Z or Y)
        if (t != 2) E_.v[q] ^= 1;                             // X component (X or Y)
        if (f_.cls[q] != -1) touches_B = true;
    }

    // Errors with support only in A leave the block's joint read law identical to the zero-error
    // one (A and B are tensor factors), so the precomputed plan replays it; only B-touching shots
    // rebuild + re-measure the block.
    if (plan_ok_ && !touches_B) plan_shot_(rng, out);
    else error_shot(E_, rng, out);

    for (int q : touched_) { E_.a[q] = 0; E_.v[q] = 0; }   // sparse reset
}

void FramedPauliShotSampler::base_outcomes(const std::function<double()>& rng,
                                           std::vector<int>& out) {
    if (!plan_ok_) { live_base_shot_(rng, out); return; }   // over-cap / out-of-scope plan: live block
    // E_ is all-zero between shots, so this is the pure zero-error replay: the baked out vector
    // (deterministic A-reads pre-written), the block tree replay, then the coin A-reads as one
    // bulk fair-bit draw — the same pool bits, in the same order, as the E-aware per-read loop.
    out = zero_out_;
    FairPool fp{rng};
    tree_replay_block(tree_, block_pos_, rng, fp, coin_words_, ozw_, out, &last_leaf_,
                      &togg_, &fired_nodes_, &fired_ptr_, &n_fired_, &osign_);
    const int nac = (int)a_coin_pos_.size();
    if (nac) {
        acw_.assign((size_t)(nac + 63) / 64, 0);
        fp.take(nac, acw_.data());
        for (int i = 0; i < nac; ++i)
            out[a_coin_pos_[i]] = ((acw_[i >> 6] >> (i & 63)) & 1) ? -1 : +1;
    }
}

// Live-block base sampling (Stage G3) — the zero-error base outcomes when the tree plan is
// unavailable (node cap / forcing failure): collapse a CLONE of the block directly
// (batch_measure is chi-general).
// For the zero error the A/B partition is FIXED (active set == B, no error ever pulls an A qubit
// here), so the block read list and the A-read kinds precomputed in init_ apply verbatim; the
// per-shot cost is O(block) at any chi — the A/B economics the general-chi design promises.
void FramedPauliShotSampler::live_base_shot_(const std::function<double()>& rng,
                                             std::vector<int>& out) {
    out.assign(reads_.size(), +1);
    if (!block_reads_.empty()) {
        *block_work_ = f_.block;
        batch_measure(*block_work_, block_reads_, rng, bits_);
        for (size_t i = 0; i < block_pos_.size(); ++i)
            out[block_pos_[i]] = bits_[i] ? -1 : +1;
    }
    for (size_t i = 0; i < a_pos_.size(); ++i)
        out[a_pos_[i]] = (a_kind_[i] == 2) ? ((rng() < 0.5) ? +1 : -1)
                                           : (a_kind_[i] == 0 ? +1 : -1);
}

void FramedPauliShotSampler::plan_shot_(const std::function<double()>& rng, std::vector<int>& out) {
    FairPool fp{rng};
    tree_replay_block(tree_, block_pos_, rng, fp, coin_words_, ozw_, out, &last_leaf_,
                      &togg_, &fired_nodes_, &fired_ptr_, &n_fired_, &osign_);
    for (size_t i = 0; i < a_pos_.size(); ++i) {
        const int k = a_pos_[i];
        const int q = reads_[k].second;
        if (E_.a[q] | E_.v[q]) {   // error-touched A-qubit: closed form with the error folded in
            const ReadResult rr = reduced_read(E_, reads_[k].first, q, ax1_, sgn_);
            out[k] = rr.deterministic ? rr.value : (fp.coin() ? -1 : +1);
        } else {
            out[k] = (a_kind_[i] == 2) ? (fp.coin() ? -1 : +1) : (a_kind_[i] == 0 ? +1 : -1);
        }
    }
}

void FramedPauliShotSampler::error_shot(const FactoredDiagError& E,
                                        const std::function<double()>& rng,
                                        std::vector<int>& out) {
    out.assign(reads_.size(), +1);
    // Build the active block (B extended by any error-entangled A-qubits) with the error
    // conjugated in, and partition the reads PER SHOT (entangling CZs can grow the column set).
    framed_active_block(f_, E, *block_work_, &b2g_, &asw_);
    if ((int)g2a_.size() != n_) g2a_.assign(n_, -1);
    for (int c = 0; c < (int)b2g_.size(); ++c) g2a_[b2g_[c]] = c;
    active_reads_.clear();
    active_pos_.clear();
    for (size_t k = 0; k < reads_.size(); ++k) {
        const int col = g2a_[reads_[k].second];
        if (col >= 0) {
            active_reads_.push_back({reads_[k].first, col});
            active_pos_.push_back((int)k);
        } else {   // non-active => product A-qubit; closed form (error folded in)
            const ReadResult rr = reduced_read(E, reads_[k].first, reads_[k].second, ax1_, sgn_);
            out[k] = rr.deterministic ? rr.value : ((rng() < 0.5) ? +1 : -1);
        }
    }
    if (!active_reads_.empty()) {
        batch_measure(*block_work_, active_reads_, rng, bits_);
        for (size_t a = 0; a < active_pos_.size(); ++a)
            out[active_pos_[a]] = bits_[a] ? -1 : +1;
    }
    for (int c = 0; c < (int)b2g_.size(); ++c) g2a_[b2g_[c]] = -1;   // sparse reset
}

// R>0 bad-shot path: the legacy run_ab_obs_shot body (G2) hoisted into the framed engine —
// error_shot's records verbatim, then expectations on the factorized post state
//   |post> = (⊗ non-active single-qubit states) ⊗ |block_post>:
//   <P> = s_pre · Π_q(single-qubit factors on the non-active support) · <block_post|P|_active|block_post>
// where s_pre = i^{P.phase − #Y(P)} = ±1 (P Hermitian), a READ non-active qubit contributes
// (letter == read basis) ? outcome : 0, an UNREAD one its (error-rotated) axis expectation via
// reduced_read (deterministic value, else exact 0), and the block factor is one framed_expectation
// of the Hermitian letter product restricted to active columns on the ALREADY collapsed block —
// so record/expectation correlations are exact by construction.
void FramedPauliShotSampler::error_shot_obs(const FactoredDiagError& E,
                                            const std::function<double()>& rng,
                                            std::vector<int>& out, const std::vector<Pauli>& obsP,
                                            double* exp) {
    // (1) records — error_shot's body, with the g2a_ sparse reset DEFERRED past the expectation
    //     loop (the letter -> block-column mapping is the tensor split's active projection).
    out.assign(reads_.size(), +1);
    framed_active_block(f_, E, *block_work_, &b2g_, &asw_);
    if ((int)g2a_.size() != n_) g2a_.assign(n_, -1);
    for (int c = 0; c < (int)b2g_.size(); ++c) g2a_[b2g_[c]] = c;
    active_reads_.clear();
    active_pos_.clear();
    for (size_t k = 0; k < reads_.size(); ++k) {
        const int col = g2a_[reads_[k].second];
        if (col >= 0) {
            active_reads_.push_back({reads_[k].first, col});
            active_pos_.push_back((int)k);
        } else {
            const ReadResult rr = reduced_read(E, reads_[k].first, reads_[k].second, ax1_, sgn_);
            out[k] = rr.deterministic ? rr.value : ((rng() < 0.5) ? +1 : -1);
        }
    }
    if (!active_reads_.empty()) {
        batch_measure(*block_work_, active_reads_, rng, bits_);
        for (size_t a = 0; a < active_pos_.size(); ++a)
            out[active_pos_[a]] = bits_[a] ? -1 : +1;
        // Cascade convention (E2b): the block factor below reads block_work_ — condition it on
        // the recorded fresh-coin outcomes (no RNG consumed; records above are untouched).
        condition_on_recorded(*block_work_, active_reads_, bits_);
    }
    // (2) expectations on the factorized post state (run_ab_obs_shot step 5, out[] for mw bits).
    const int nb = block_work_->n();
    const int W = (n_ + 63) / 64;
    for (int r = 0; r < (int)obsP.size(); ++r) {
        const Pauli& P = obsP[r];
        const int pre = ((P.phase - P.xz_overlap()) % 4 + 4) % 4;   // i^{ph-#Y} = ±1 (Hermitian)
        double val = (pre == 2) ? -1.0 : 1.0;
        Pauli& Pb = pb_scratch_;
        reset_pauli_inplace(Pb, nb);
        bool block_any = false;
        for (int w = 0; w < W && val != 0.0; ++w) {                 // support bit-scan (x|z words)
            uint64_t sup = P.x[w] | P.z[w];
            while (sup) {
                const int q = (w << 6) + __builtin_ctzll(sup);
                sup &= sup - 1;
                const bool xb = P.xbit(q), zb = P.zbit(q);
                const int sig = xb ? (zb ? 1 : 0) : 2;              // letter 0:X 1:Y 2:Z
                const int col = g2a_[q];
                if (col >= 0) {                                     // block letter (disjoint cols:
                    if (xb) Pb.setx(col);                           //  direct bit set == pmul_into
                    if (zb) Pb.setz(col);                           //  of single_pauli letters)
                    if (sig == 1) Pb.phase = (Pb.phase + 1) & 3;
                    block_any = true;
                } else if (w2r_[q] >= 0) {                          // read non-active qubit
                    const int j = w2r_[q];
                    if (sig != reads_[j].first) { val = 0.0; break; }   // <o·b|σ≠b|o·b> = 0
                    if (out[j] == -1) val = -val;
                } else {                                            // unread product qubit
                    const ReadResult rr = reduced_read(E, sig, q, ax1_, sgn_);
                    if (!rr.deterministic) { val = 0.0; break; }    // off-axis letter: <σ> = 0
                    if (rr.value == -1) val = -val;
                }
            }
        }
        if (val != 0.0 && block_any) {
            const auto pq = framed_expectation(*block_work_, Pb);
            val *= (pq.first - pq.second);
        }
        exp[r] = val;
    }
    for (int c = 0; c < (int)b2g_.size(); ++c) g2a_[b2g_[c]] = -1;   // sparse reset
}

// ════════════════════════════ FramedCircuitShotSampler ════════════════════════════════════════

// The 15 non-identity 2q Paulis in Stim's PAULI_CHANNEL_2 order (IX IY IZ XI XX XY XZ YI YX YY YZ
// ZI ZX ZY ZZ), encoded as (first factor, second factor), 0 = identity else PauliBasis+1. Mirrors
// sample_noise's table so my channels realize the identical law.
static const int kPair2[15][2] = {
    {0,1},{0,2},{0,3},{1,0},{1,1},{1,2},{1,3},{2,0},{2,1},{2,2},{2,3},{3,0},{3,1},{3,2},{3,3}
};

int FramedCircuitShotSampler::pair_id_(int a, int b) {
    if (a > b) std::swap(a, b);
    for (size_t i = 0; i < pair_list_.size(); ++i)
        if (pair_list_[i].first == a && pair_list_[i].second == b) return (int)i;
    pair_list_.push_back({a, b});
    return (int)pair_list_.size() - 1;
}

// Promote a diagonal atom's DiagPauliClifford (γ·X^v·diag(S^a)·∏CZ) to a CliffordTableau by gate
// decomposition — diagonal layer (S/Z/S†) first, then the CZ layer, then X^v (the operator order,
// mirroring the test-only apply_diag_pauli). The global scalar γ is unobservable in conjugation
// (dual_image is γ-insensitive), so the tableau — which carries no global phase — is exact for the
// E†·M·E read conjugation. Used only to compose diagonal atoms that share a general alt.
static CliffordTableau tableau_from_diag(const DiagPauliClifford& D) {
    CliffordTableau T(D.n);
    for (int q = 0; q < D.n; ++q)
        switch (D.a[q] & 3) {
            case 1: T.left_s(q); break;
            case 2: T.left_z(q); break;
            case 3: T.left_sdg(q); break;
            default: break;
        }
    for (int i = 0; i < D.n; ++i)
        for (int j = i + 1; j < D.n; ++j)
            if (D.B.get(i, j)) T.left_cz(i, j);
    for (int q = 0; q < D.n; ++q)
        if (D.v[q]) T.left_x(q);
    return T;
}

// The ordered atoms a fired list references (parallel to compose_fired's lookup): x_atom for an
// X/Y factor, z_atom for a Z/Y factor, in the SAME order compose_fired composes them.
std::vector<const PropResult*>
FramedCircuitShotSampler::gather_atoms_(const std::vector<FiredPauli>& fired) const {
    std::vector<const PropResult*> out;
    for (const FiredPauli& f : fired) {
        const auto [loc, qi] = pt_.find_slot(f.location_index, f.qubit, "gather_atoms");
        if (f.pauli == PauliBasis::X || f.pauli == PauliBasis::Y) out.push_back(&loc->x_atom[qi]);
        if (f.pauli == PauliBasis::Z || f.pauli == PauliBasis::Y) out.push_back(&loc->z_atom[qi]);
    }
    return out;
}

// The composed end-of-circuit error tableau E = atom_last ∘ ... ∘ atom_first of one fired alt that
// references at least one general atom. Single stored-general atom => reuse it (no build). Else
// build E's forward rows (Xrow[a] = E·X_a·E†) by applying the FORWARD action of each atom in
// stream order (CliffordTableau::forward_image) — matches compose_fired's c = c.then(atom)
// chaining. diagonal atoms are promoted to tableaus. Bug note: using conjugate() (= pullback
// T†·P·T) here would store dual rows in Xrow and make the subsequent dual_image(M) call in
// process_shot_general_ return E·M·E† instead of the correct E†·M·E, flipping the measurement
// outcome sign for non-Pauli composed residuals.
std::shared_ptr<const CliffordTableau>
FramedCircuitShotSampler::compose_alt_general_(const std::vector<const PropResult*>& atoms) const {
    if (atoms.size() == 1 && atoms[0]->general) return atoms[0]->general;
    int nd = 0;
    for (const PropResult* a : atoms) if (!a->general) ++nd;
    std::vector<CliffordTableau> temps;
    temps.reserve((size_t)nd);                                 // stable: all pushes precede tab()
    std::vector<int> tidx(atoms.size(), -1);
    for (size_t i = 0; i < atoms.size(); ++i)
        if (!atoms[i]->general) { tidx[i] = (int)temps.size(); temps.push_back(tableau_from_diag(atoms[i]->c_prop)); }
    auto tab = [&](size_t i) -> const CliffordTableau& {
        return atoms[i]->general ? *atoms[i]->general : temps[(size_t)tidx[i]];
    };
    CliffordTableau E(n_);
    for (int a = 0; a < n_; ++a) {
        Pauli x(n_); x.setx(a);
        Pauli z(n_); z.setz(a);
        for (size_t i = 0; i < atoms.size(); ++i) { x = tab(i).forward_image(x); z = tab(i).forward_image(z); }
        E.Xrow[a] = std::move(x);
        E.Zrow[a] = std::move(z);
    }
    E.invalidate_dual();                                       // rows set directly; dual_image reads forward rows
    return std::make_shared<CliffordTableau>(std::move(E));
}

// Conjugate a read Pauli through one DIAGONAL alt: M ← D†·M·D, D = X^v·∏CZ·diag(S^a). The X^v is
// innermost (X_q flips a Z/Y on q), then the canonical diag_conjugate S-power / CZ rules — the
// single source of truth for the leftover-Z / i-phase bookkeeping.
void FramedCircuitShotSampler::conj_by_diag_altside_(Pauli& M, const AltSide& sd) const {
    int dphase = 0;
    for (int q : sd.v) if (M.zbit(q)) dphase += 2;             // X^v M X^v (innermost)
    static thread_local std::vector<uint64_t> zmask;
    zmask.assign(M.z.size(), 0);
    for (const auto& qc : sd.a) diag_spow_conjugate(M, qc.first, qc.second, zmask.data(), dphase);
    for (int pid : sd.cz) {
        const auto pr = pair_list_[(size_t)pid];
        diag_cz_conjugate(M, pr.first, pr.second, zmask.data(), dphase);
    }
    for (size_t w = 0; w < M.z.size(); ++w) M.z[w] ^= zmask[w];
    M.phase = (M.phase + (dphase & 3)) & 3;
}

// General shot: fold nothing into an AltSide — clone the bare state and read each terminal Pauli
// M_k conjugated through the fired residuals (last-fired first, so the composition E_total =
// E_last···E_first is undone outermost-by-E_first). measure_pauli(E†M_kE, coin) on the bare clone
// is EXACTLY the outcome of M_k on the error state E_total·|bare⟩ — one coin per read, record order.
void FramedCircuitShotSampler::process_shot_general_(const int32_t* ev, int nev,
                                                     const std::function<double()>& rng,
                                                     std::vector<int>& out) {
    FramedSuperposition clone = *bare_full_;
    out.assign((size_t)M_, +1);
    for (int k = 0; k < M_; ++k) {
        Pauli M = single_pauli(rd_[k].first, rd_[k].second, n_);
        for (int e = nev - 1; e >= 0; --e) {                  // last-fired residual conjugates first
            const int32_t alt = ev[e];
            if (alt_tab_[(size_t)alt]) M = alt_tab_[(size_t)alt]->dual_image(M);
            else conj_by_diag_altside_(M, sides_[(size_t)alt]);
        }
        out[k] = clone.measure_pauli(M, rng());
    }
    // R>0: after the read collapse, each declared PAULI_EXPECTATION column P evaluates
    // <ψ_post| E†PE |ψ_post> = <bare| E†PE |bare_post> with |bare_post> = clone (the same
    // post-collapse state the reads produced). Conjugate P by the SAME fired chain (last-fired
    // first) and read the exact signed Pauli's expectation off the clone — sign-exact (the ± of
    // the conjugated Pauli folds into FramedSuperposition::expectation's i^{phase} scalar). The
    // declared record-frame (emask/esign) XOR is applied downstream in pack_shot_records, exactly
    // as on the diagonal path — this fills the RAW <E†PE> row (last_expectations()).
    if (exp_on_) {
        for (int r = 0; r < R_; ++r) {
            Pauli P = obsP_[r];
            for (int e = nev - 1; e >= 0; --e) {
                const int32_t alt = ev[e];
                if (alt_tab_[(size_t)alt]) P = alt_tab_[(size_t)alt]->dual_image(P);
                else conj_by_diag_altside_(P, sides_[(size_t)alt]);
            }
            // Hermitian in ⇒ Hermitian out (Clifford conjugation preserves phase parity), so the
            // conjugated P carries a real ±1 only. Assert it: expectation()'s .real() would
            // silently truncate an imaginary residue if this invariant ever broke.
            assert((P.phase & 1) == 0 && "general-shot PE Pauli must stay Hermitian (even phase)");
            exp_[r] = clone.expectation(P);
        }
    }
}

void FramedCircuitShotSampler::add_alt_(ErrorChannel& ch, double p,
                                        const std::vector<FiredPauli>& fired) {
    std::vector<const PropResult*> atoms = gather_atoms_(fired);
    bool gen = false;
    for (const PropResult* a : atoms) if (a->general) { gen = true; break; }
    AltSide sd;
    sd.mflip.assign((size_t)MW_, 0);
    std::shared_ptr<const CliffordTableau> tab;
    if (gen) {
        // Off-diagonal residual: keep the composed error tableau; the AltSide stays empty (the
        // general shot path uses alt_tab_, never this alt's folded diagonal action).
        tab = compose_alt_general_(atoms);
    } else {
        DiagPauliClifford D = compose_fired(pt_, fired);
        D.finalize();
        // (compose_fired handles 1- and 2-factor lists alike; the compile cost lives here and in the
        // table build above — both amortized by the compile cache and the table reuse.)
        for (int q : D.cache->active_a) sd.a.push_back({q, (uint8_t)(D.a[q] & 3)});
        for (const auto& pr : D.cache->b_pairs) sd.cz.push_back(pair_id_(pr.first, pr.second));
        for (int q = 0; q < n_; ++q)
            if (D.v[q]) {
                sd.v.push_back(q);
                // X^v on wire q flips a Z- or Y-read on q (X Z X = -Z, X Y X = -Y); X-reads unaffected.
                const int r = rid_[q];
                if (r >= 0 && rbase_[q] != 0) sd.mflip[r >> 6] ^= 1ull << (r & 63);
            }
    }
    sides_.push_back(std::move(sd));
    alt_tab_.push_back(std::move(tab));
    ErrorChannel::Alt alt;
    alt.p = p;                                    // action data lives in sides_/alt_tab_; the
    ch.alts.push_back(std::move(alt));            // DiagErrorSampler is used for EVENT sampling only
}

FramedCircuitShotSampler::FramedCircuitShotSampler(const FramedSuperposition& bare,
                                                   const Circuit& deferred,
                                                   std::vector<std::pair<int, int>> reads,
                                                   uint64_t noise_seed,
                                                   const PropagationTable* table)
    : sp_(bare, reads), n_(bare.n()), rd_(std::move(reads)) {
    // No chi gate — the engine is chi-general. tier0 shots and pattern-key bad shots replay the
    // TreePlan DAG at any chi (chi<=2 = its depth<=1 instance, byte-identical streams); over the
    // node cap the extraction reports failure and the G3 fallbacks hold: live-block base sampling
    // (tier0) / error_shot (patterns) — both O(block) at any chi. The relabel layers applied on
    // top (relabels/mflips/flips) are chi-INDEPENDENT: the cascade theorem relabels the terminal
    // joint law, whatever block law produced the base sample.
    build_(deferred, noise_seed, table);
    // QEC_FORCE_FALLBACK=1 (read once at construction): every shot takes the retained full-state
    // general path — the engine-neutral A/B oracle lever, R == 0 included (records only there).
    const char* fbe = std::getenv("QEC_FORCE_FALLBACK");
    force_fb_ = (fbe != nullptr && fbe[0] == '1');
    if (force_fb_) {
        static const auto _ffb_quiet = []{ const char* v = std::getenv("XTIM_QUIET"); return v && v[0] && v[0] != '0'; }();
        static bool _ffb_noticed = false;
        if (!_ffb_noticed && !_ffb_quiet) { _ffb_noticed = true; fprintf(stderr, "[xtim] QEC_FORCE_FALLBACK=1: full-state clone per shot (slower; diagnostic)\n"); }
        bare_full_ = std::make_unique<FramedSuperposition>(bare);
        fb_a_.assign(n_, 0);
        fb_v_.assign(n_, 0);
    }
    // A general (off-diagonal residual) circuit reads its terminal Paulis off a fresh bare clone
    // per general shot (process_shot_general_), so keep the full bare state.
    if (has_general_ && !bare_full_) bare_full_ = std::make_unique<FramedSuperposition>(bare);
}

void FramedCircuitShotSampler::begin_run(uint64_t noise_seed) {
    smp_->reseed(noise_seed);
    batch_.S = 0;
    cursor_ = 0;
    memo_.clear();
}

// ── R>0 expectation channels (production since E2b) ────────────────────────────────────────────
bool FramedCircuitShotSampler::enable_expectations(const std::vector<Pauli>& obsP,
                                                   const FramedSuperposition& bare) {
    // R>0 with an off-diagonal (CH-class) residual (has_general_) is handled per general shot by
    // process_shot_general_: it conjugates each declared observable through the SAME fired-residual
    // chain as the terminal reads (last-fired-first) and evaluates the exact signed Pauli on the
    // post-collapse full-state clone (FramedSuperposition::expectation) — sign-exact, no channel.
    // The DIAGONAL closed-form channels built below still serve the diagonal-only shots of a general
    // circuit (any shot that fires no general alt takes the diagonal path), so build them either way.
    obsP_ = obsP;
    R_ = (int)obsP.size();
    exp_.assign((size_t)R_, 0.0);
    if (!bare_full_)                                            // full-state fallback source
        bare_full_ = std::make_unique<FramedSuperposition>(bare);
    Ez_.a.assign(n_, 0);
    Ez_.v.assign(n_, 0);
    Ez_.cz.clear();
    zmask_.assign((size_t)((n_ + 63) / 64), 0);
    no_coins_.clear();
    fb_a_.assign(n_, 0);
    fb_v_.assign(n_, 0);
    // Zero-plan channel: static split at the original read bases + the correction-chain tree
    // with per-leaf posts/corrections (partially evaluated into its path-unique expansion when
    // small — expand_corrected_tree). A refused leaf channel (or an over-cap channel tree) is NOT
    // an error — tier0 shots then take the live-block route, which is exact at any chi.
    std::vector<int8_t> rb((size_t)M_);
    for (int k = 0; k < M_; ++k) rb[k] = (int8_t)rd_[k].first;
    std::vector<int> w2r(n_, -1);
    for (int k = 0; k < M_; ++k) w2r[rd_[k].second] = k;
    build_obs_split(zero_chan_, obsP_, n_, rb, w2r, sp_.factored().cls, sp_.block_qubits(),
                    sp_.a_axis1(), sp_.a_sign(), Ez_);
    std::vector<std::pair<int32_t, FramedSuperposition>> ls;
    zero_chan_.leaves_ok = false;
    if (sp_.rebuild_plan_channel(ls) &&
        build_obs_leaves(zero_chan_, sp_.tree_mut(), ls, sp_.block_read_list()))
        expand_corrected_tree(sp_.tree_mut(), sp_.block_read_list().size());
    // Zero-event fast-path constants (E2b): fold s_pre · a_const · the deterministic A-read
    // outcomes into fast_const, and select the fair-coin A-reads as a mask over the plan's
    // a-coin words (bit i == a_coin_positions()[i]) — eval_zero_fast_ then computes each
    // observable as two masked popcounts against the shot's coin words.
    {
        const std::vector<int>& acp = sp_.a_coin_positions();
        const std::vector<int>& zo = sp_.zero_outs();
        std::vector<int> coin_of(M_ > 0 ? (size_t)M_ : 0, -1);
        for (size_t i = 0; i < acp.size(); ++i) coin_of[acp[i]] = (int)i;
        const size_t aw = (acp.size() + 63) / 64;
        for (auto& sp2 : zero_chan_.split) {
            sp2.fast_const = sp2.s_pre * sp2.a_const;
            sp2.acmask.assign(aw, 0);
            for (int j : sp2.a_reads) {
                if (coin_of[j] >= 0) sp2.acmask[coin_of[j] >> 6] |= 1ull << (coin_of[j] & 63);
                else if (zo[j] == -1) sp2.fast_const = -sp2.fast_const;
            }
        }
    }
    exp_on_ = true;
    return true;
}

// Zero-event tier0 shot on the leaf route (the dominant low-p case): every split factor is a
// precomputed constant except the fair coins — <P_r> = fast_const · leaf.val0[r] with sign =
// parity(leaf.sig[r] & leaf coin words) XOR parity(acmask[r] & A-read coin words). Exactly the
// general untouched path's value (products of ±1 folded at compile time), at two popcounts/obs.
void FramedCircuitShotSampler::eval_zero_fast_(int32_t leaf, uint64_t osign) {
    const std::vector<uint64_t>& coins = sp_.last_coin_words();
    const std::vector<uint64_t>& acw = sp_.a_coin_words();
    const FramedObsChannel::Leaf& lf = zero_chan_.leaves[leaf];
    for (int r = 0; r < R_; ++r) {
        const FramedObsChannel::Split& sp = zero_chan_.split[r];
        if (sp.zero) { exp_[r] = 0.0; continue; }
        double v = sp.fast_const * lf.val0[r];
        int s = r < 64 ? (int)((osign >> r) & 1) : 0;   // corrected tree coins (replay-accumulated)
        const std::vector<uint64_t>& m = lf.sig[r];
        for (size_t w = 0; w < m.size(); ++w)
            s ^= (int)(__builtin_popcountll(m[w] & coins[w]) & 1);
        for (size_t w = 0; w < sp.acmask.size(); ++w)
            s ^= (int)(__builtin_popcountll(sp.acmask[w] & acw[w]) & 1);
        exp_[r] = s ? -v : v;
    }
}

// Per-shot channel evaluation for tier0/pattern shots (called on the BASE outcomes, before the
// relabel layers — the |φ> collapse outcomes the tensor factors read). Reference formulation:
//   <P> = <φ| E†PE |φ>,  E = γ·X^V·Δ  =>  (-1)^{V·z(P)} · <φ| Δ†PΔ |φ>,
// with Δ = the shot's folded S/CZ error EXCLUDING kind-2 pairs (those live in the pattern state,
// the legacy incpair cancellation). |φ> factorizes as (⊗ A posts) ⊗ |block post|; the block
// factor comes from the leaf reference post ± the leaf coins' corrections (legacy esign0
// semantics), or from the LIVE collapsed block when `live` is non-null (no corrections — the
// live state already carries the shot's outcomes).
void FramedCircuitShotSampler::eval_channel_shot_(const FramedObsChannel& ch,
                                                  const FramedPauliShotSampler::TreePlan* tree,
                                                  const int32_t* fired_nodes, int n_fired,
                                                  uint64_t osign, int32_t leaf,
                                                  const std::vector<uint64_t>& coins,
                                                  const FramedSuperposition* live,
                                                  const std::vector<int>& out) {
    const int W = (n_ + 63) / 64;
    for (int r = 0; r < R_; ++r) {
        const Pauli& A = obsP_[r];
        // (1) closed-form diagonal conjugation of A through the folded error (Δ†·A·Δ),
        //     kind-2 pairs excluded — the legacy per-shot zmask fold, verbatim rules.
        int dph = 0;
        bool touched = false;
        for (int32_t q : dirty_q_)
            if (A.xbit(q)) {
                if (!touched) { touched = true; for (int w = 0; w < W; ++w) zmask_[w] = 0; }
                diag_spow_conjugate(A, q, acc_a_[q], zmask_.data(), dph);
            }
        for (int32_t w : dirty_w_) {
            uint64_t bits = czacc_[w];
            while (bits) {
                const int pid = (w << 6) + __builtin_ctzll(bits);
                bits &= bits - 1;
                if (paircls_[pid].kind == 2) continue;   // folded into the pattern STATE
                const int qa = pair_list_[pid].first, qb = pair_list_[pid].second;
                if (A.xbit(qa) || A.xbit(qb)) {
                    if (!touched) { touched = true; for (int ww = 0; ww < W; ++ww) zmask_[ww] = 0; }
                    diag_cz_conjugate(A, qa, qb, zmask_.data(), dph);
                }
            }
        }
        // (2) X^V sign: parity(V ∧ z(A)) — X^V A X^V = -A on Z-overlap.
        int sgn = 0;
        for (int w = 0; w < W; ++w) sgn ^= (int)(__builtin_popcountll(vmask_[w] & A.z[w]) & 1);
        double val;
        if (!touched) {
            const FramedObsChannel::Split& sp = ch.split[r];
            if (sp.zero) { exp_[r] = 0.0; continue; }
            val = sp.s_pre * sp.a_const;
            for (int j : sp.a_reads) if (out[j] == -1) val = -val;
            if (sp.block_any) {
                if (live) {
                    const auto pq = framed_expectation(*live, sp.Pb);
                    val *= (pq.first - pq.second);
                } else {
                    const FramedObsChannel::Leaf& lf = ch.leaves[leaf];
                    val *= lf.val0[r];
                    int s2 = r < 64 ? (int)((osign >> r) & 1)   // corrected tree coins
                                    : [&] {                     // r >= 64: per-node masks
                                          int sx = 0;
                                          for (int j2 = 0; j2 < n_fired; ++j2)
                                              sx ^= (int)((ch.nsig[fired_nodes[j2]][r >> 6] >>
                                                           (r & 63)) & 1);
                                          return sx;
                                      }();
                    const std::vector<uint64_t>& m = lf.sig[r];
                    for (size_t w2 = 0; w2 < m.size(); ++w2)
                        s2 ^= (int)(__builtin_popcountll(m[w2] & coins[w2]) & 1);
                    if (s2) val = -val;
                }
            }
        } else {
            // Error-conjugated observable: build Pc = Δ†AΔ and take the generic tensor split.
            // Scratch Paulis + support bit-scan (no per-shot allocations — the E2b hot-path fix).
            Pauli& Pc = pc_scratch_;
            reset_pauli_inplace(Pc, n_);
            for (int w = 0; w < W; ++w) { Pc.x[w] = A.x[w]; Pc.z[w] = A.z[w] ^ zmask_[w]; }
            Pc.phase = (uint8_t)((A.phase + dph) & 3);
            const int pre = ((Pc.phase - Pc.xz_overlap()) % 4 + 4) % 4;
            val = (pre == 2) ? -1.0 : 1.0;
            Pauli& Pb = pb_scratch_;
            reset_pauli_inplace(Pb, ch.nb);
            bool block_any = false;
            for (int w = 0; w < W && val != 0.0; ++w) {
                uint64_t sup = Pc.x[w] | Pc.z[w];
                while (sup) {
                    const int q = (w << 6) + __builtin_ctzll(sup);
                    sup &= sup - 1;
                    const bool xb = Pc.xbit(q), zb = Pc.zbit(q);
                    const int sig2 = xb ? (zb ? 1 : 0) : 2;
                    const int col = ch.g2a[q];
                    if (col >= 0) {                                 // block letter (disjoint cols:
                        if (xb) Pb.setx(col);                       //  direct bit set == pmul_into
                        if (zb) Pb.setz(col);                       //  of single_pauli letters)
                        if (sig2 == 1) Pb.phase = (Pb.phase + 1) & 3;
                        block_any = true;
                    } else if (rid_[q] >= 0) {                      // read A-qubit: BASE outcome
                        const int j = rid_[q];
                        if (sig2 != (int)ch.rbasis[j]) { val = 0.0; break; }
                        if (out[j] == -1) val = -val;
                    } else {                                        // unread product qubit
                        const ReadResult rr = reduced_read(Ez_, sig2, q, sp_.a_axis1(), sp_.a_sign());
                        if (!rr.deterministic) { val = 0.0; break; }
                        if (rr.value == -1) val = -val;
                    }
                }
            }
            if (val != 0.0 && block_any) {
                const FramedSuperposition* post = live ? live : &ch.leaves[leaf].post;
                const auto pq = framed_expectation(*post, Pb);
                val *= (pq.first - pq.second);
                if (!live) {                                        // leaf coin corrections vs Pc|B
                    const FramedObsChannel::Leaf& lf = ch.leaves[leaf];
                    for (int c = 0; c < (int)lf.coinC.size(); ++c)
                        if (((coins[c >> 6] >> (c & 63)) & 1) &&
                            Pauli::anticommute_bit(lf.coinC[c], Pb))
                            val = -val;
                    for (int j2 = 0; j2 < n_fired; ++j2)   // corrected tree coins vs the SHOT'S Pb
                        if (Pauli::anticommute_bit(tree->nodes[fired_nodes[j2]].corr, Pb))
                            val = -val;
                }
            }
        }
        exp_[r] = sgn ? -val : val;
    }
}

// The retained full-state fallback (the former run_lean_fallback_shot body): clone the full bare
// state, fold the shot's COMPLETE error (a, cz incl. kind-2, v), conjugate, tagged lean collapse,
// LEAN Born reads for the expectations (post-state conditioned on the recorded fresh-coin
// outcomes — the cascade convention, E2b). Counter-tracked; QEC_FORCE_FALLBACK=1 forces every
// shot here (the engine-neutral A/B oracle route, R == 0 included).
void FramedCircuitShotSampler::full_fallback_shot_(const std::function<double()>& rng,
                                                   std::vector<int>& out) {
    for (int32_t q : dirty_q_) fb_a_[q] = acc_a_[q] & 3;
    fb_cz_.clear();
    for (int32_t w : dirty_w_) {
        uint64_t bits = czacc_[w];
        while (bits) {
            const int pid = (w << 6) + __builtin_ctzll(bits);
            bits &= bits - 1;
            fb_cz_.push_back(pair_list_[pid]);
        }
    }
    for (int32_t q : vdirty_) fb_v_[q] = (uint8_t)((vmask_[q >> 6] >> (q & 63)) & 1);
    FramedSuperposition wl = *bare_full_;
    conjugate_by_diag_clifford(wl.U, fb_a_, fb_cz_, fb_v_);
    batch_measure(wl, rd_, rng, fb_bits_);
    out.assign((size_t)M_, +1);
    for (int j = 0; j < M_; ++j)
        if (fb_bits_[j] == 1) out[j] = -1;
    if (R_ > 0) {
        condition_on_recorded(wl, rd_, fb_bits_);   // cascade convention (E2b); no RNG consumed
        for (int r = 0; r < R_; ++r) {
            const auto pq = framed_expectation(wl, obsP_[r]);
            exp_[r] = pq.first - pq.second;
        }
    }
    for (int32_t q : dirty_q_) fb_a_[q] = 0;
    for (int32_t q : vdirty_) fb_v_[q] = 0;
    ++fallback_full_;
}

void FramedCircuitShotSampler::build_(const Circuit& deferred, uint64_t noise_seed,
                                      const PropagationTable* table) {
    M_ = (int)rd_.size();
    MW_ = (M_ + 63) / 64;
    rid_.assign(n_, -1);
    rbase_.assign(n_, 3);
    for (size_t k = 0; k < rd_.size(); ++k) {
        rid_[rd_[k].second] = (int)k;
        rbase_[rd_[k].second] = (uint8_t)rd_[k].first;
    }

    if (table != nullptr) pt_ = *table;              // reuse the caller's (one deep copy)
    else pt_ = build_propagation_table(deferred, /*ppr_retry=*/true);
    if (!pt_.all_in_class)
        throw std::runtime_error("FramedCircuitShotSampler: circuit outside the diagonal+Pauli class");
    has_general_ = pt_.has_general;   // any noise atom with an off-diagonal (CH-class) residual
    std::map<int, int> locof;   // stream index -> location entry
    for (size_t i = 0; i < pt_.locations.size(); ++i) locof[pt_.locations[i].stream_index] = (int)i;

    // One ErrorChannel per independent firing unit (qubit for 1q channels, pair for 2q), with the
    // alternatives at sample_noise's documented probabilities. Flat alternative ids are assigned in
    // construction order — sides_ is built in lockstep, so ids index it directly.
    for (int si = 0; si < (int)deferred.stream.size(); ++si) {
        const Instr& ins = deferred.stream[si];
        if (ins.kind != Instr::Kind::Noise) continue;
        if (locof.find(si) == locof.end())
            throw std::runtime_error("FramedCircuitShotSampler: noise location missing from table");
        switch (ins.channel) {
            case NoiseChannel::X_ERROR:
            case NoiseChannel::Y_ERROR:
            case NoiseChannel::Z_ERROR: {
                const PauliBasis b = ins.channel == NoiseChannel::X_ERROR ? PauliBasis::X
                                   : ins.channel == NoiseChannel::Y_ERROR ? PauliBasis::Y
                                                                          : PauliBasis::Z;
                for (int q : ins.qubits) {
                    ErrorChannel ch;
                    add_alt_(ch, ins.probs[0], {{si, q, b}});
                    channels_.push_back(std::move(ch));
                }
                break;
            }
            case NoiseChannel::DEPOLARIZE1:
            case NoiseChannel::PAULI_CHANNEL_1: {
                const bool dep = ins.channel == NoiseChannel::DEPOLARIZE1;
                static const PauliBasis bs[3] = {PauliBasis::X, PauliBasis::Y, PauliBasis::Z};
                for (int q : ins.qubits) {
                    ErrorChannel ch;
                    for (int i = 0; i < 3; ++i) {
                        const double p = dep ? ins.probs[0] / 3.0 : ins.probs[i];
                        if (p > 0.0) add_alt_(ch, p, {{si, q, bs[i]}});
                    }
                    if (!ch.alts.empty()) channels_.push_back(std::move(ch));
                }
                break;
            }
            case NoiseChannel::DEPOLARIZE2:
            case NoiseChannel::PAULI_CHANNEL_2: {
                const bool dep = ins.channel == NoiseChannel::DEPOLARIZE2;
                for (size_t pi = 0; pi + 1 < ins.qubits.size(); pi += 2) {
                    const int qa = ins.qubits[pi], qb = ins.qubits[pi + 1];
                    ErrorChannel ch;
                    for (int t = 0; t < 15; ++t) {
                        const double p = dep ? ins.probs[0] / 15.0 : ins.probs[t];
                        if (p <= 0.0) continue;
                        std::vector<FiredPauli> fr;
                        if (kPair2[t][0]) fr.push_back({si, qa, (PauliBasis)(kPair2[t][0] - 1)});
                        if (kPair2[t][1]) fr.push_back({si, qb, (PauliBasis)(kPair2[t][1] - 1)});
                        add_alt_(ch, p, fr);
                    }
                    if (!ch.alts.empty()) channels_.push_back(std::move(ch));
                }
                break;
            }
            default:
                throw std::runtime_error("FramedCircuitShotSampler: unsupported noise channel");
        }
    }

    // Pair classification (tier0 relabel algebra on terminal single-qubit reads). STATE-AWARE
    // refinement over the production sampler's read-basis-only rule: a CZ endpoint that is an
    // UNREAD Z-definite product qubit |b⟩ acts as the classical Z^b on its partner — CZ is
    // diagonal on a Z-definite endpoint, so NOTHING in the diagonal class (S fixes the Z axis;
    // X^v only flips b; CZ never entangles a Z-definite endpoint) can break this at shot time.
    //   both endpoints non-Z reads              -> 2 (entangles two read operators: bad)
    //   non-Z read + Z-READ partner             -> 1 (relabel: out[target] ^= base out[source])
    //   non-Z read + unread Z-DEFINITE-A |1⟩    -> 3 (constant flip of out[target]; in the folded
    //                                                 normal form the CZ always sees the BARE sign —
    //                                                 X-crossings were folded into a+=2 on the read)
    //   non-Z read + unread Z-DEFINITE-A |0⟩    -> 0 (Z^0: nothing)
    //   non-Z read + other unread partner       -> 2 (read operator leaves the read set)
    //   anything else (Z/unread combinations)   -> 0 (commutes with every read projector)
    const std::vector<int>& cls = sp_.factored().cls;
    const std::vector<uint8_t>& ax1 = sp_.a_axis1();
    const std::vector<uint8_t>& asg = sp_.a_sign();
    paircls_.assign(pair_list_.size(), {});
    for (size_t id = 0; id < pair_list_.size(); ++id) {
        const int wa = pair_list_[id].first, wb = pair_list_[id].second;
        const bool aXY = rid_[wa] >= 0 && rbase_[wa] != 2;
        const bool bXY = rid_[wb] >= 0 && rbase_[wb] != 2;
        auto zdefA = [&](int w) { return cls[w] == -1 && ax1[w] == 3; };   // A-qubit, Z axis
        PairCls pc;
        if (aXY && bXY) pc.kind = 2;
        else if (aXY || bXY) {
            const int wq = aXY ? wa : wb, wp = aXY ? wb : wa;   // q = the non-Z read, p = partner
            if (rid_[wp] >= 0 && rbase_[wp] == 2) { pc.kind = 1; pc.target = rid_[wq]; pc.source = rid_[wp]; }
            else if (rid_[wp] < 0 && zdefA(wp))   { pc.kind = asg[wp] ? 3 : 0; pc.target = rid_[wq]; }
            else pc.kind = 2;
        }
        paircls_[id] = pc;
    }

    smp_ = std::make_unique<DiagErrorSampler>(n_, channels_, noise_seed);

    // fold scratch
    acc_a_.assign(n_, 0);
    in_dq_.assign(n_, 0);
    czacc_.assign((pair_list_.size() + 63) / 64, 0);
    in_dw_.assign(czacc_.size(), 0);
    vmask_.assign((n_ + 63) / 64, 0);
    mflip_acc_.assign((size_t)MW_, 0);
    E_.a.assign(n_, 0);
    E_.v.assign(n_, 0);
}

void FramedCircuitShotSampler::next_shot(const std::function<double()>& rng, std::vector<int>& out) {
    if (cursor_ >= batch_.S) { smp_->sample_events(4096, batch_); cursor_ = 0; }
    const int s = cursor_++;
    last_nev = batch_.shot_off[s + 1] - batch_.shot_off[s];
    last_ev.assign(&batch_.ev[batch_.shot_off[s]], &batch_.ev[batch_.shot_off[s]] + last_nev);
    process_shot_(&batch_.ev[batch_.shot_off[s]], last_nev, rng, out);
}

void FramedCircuitShotSampler::shot_from_events(const int32_t* ev, int nev,
                                                const std::function<double()>& rng,
                                                std::vector<int>& out) {
    last_nev = nev;
    last_ev.assign(ev, ev + nev);
    process_shot_(ev, nev, rng, out);
}

void FramedCircuitShotSampler::process_shot_(const int32_t* ev, int nev,
                                             const std::function<double()>& rng,
                                             std::vector<int>& out) {
    // ── general (off-diagonal residual) shot: one flag branch on the diagonal path (has_general_
    //    is false for every non-CH circuit, so this is compiled-out cost there).
    //    For CH-class circuits (has_general_=true), the bare state is chi=2 (non-stabilizer),
    //    which means Z/Y measurements can have pp ∉ {0,1/2,1}. The mflip approach (flip sign of
    //    zero-error outcome) is correct only for deterministic measurements (pp∈{0,1}). For a
    //    chi=2 bare state it is WRONG: measuring E·|bare> with a given coin does NOT equal
    //    measuring |bare> with that coin and negating. process_shot_general_ — which conjugates
    //    each read Pauli through the error and measures on the clone — is always correct. So for
    //    any non-zero-error shot in a CH-class circuit, always use that path.
    if (has_general_ && nev > 0) { process_shot_general_(ev, nev, rng, out); return; }

    // ── exact sparse fold of the shot's fired alternatives (event order), the sampler's normal-
    //    form fold: the new event's S-powers conjugate through the ACCUMULATED X (X S^c X = S^-c),
    //    a CZ crossing an accumulated X on one endpoint deposits Z (a += 2) on the other. ──
    dirty_q_.clear(); dirty_w_.clear(); vdirty_.clear();
    bool bad = false;
    long nf = 0;
    for (int32_t e = 0; e < nev; ++e, ++nf) {
        const AltSide& sd = sides_[ev[e]];
        for (const auto& qc : sd.a) {
            const int q = qc.first;
            const uint8_t cc = ((vmask_[q >> 6] >> (q & 63)) & 1) ? (uint8_t)((4 - qc.second) & 3)
                                                                  : qc.second;
            if (cc && !in_dq_[q]) { in_dq_[q] = 1; dirty_q_.push_back(q); }
            acc_a_[q] = (uint8_t)((acc_a_[q] + cc) & 3);
        }
        for (int32_t pid : sd.cz) {
            if (!in_dw_[pid >> 6]) { in_dw_[pid >> 6] = 1; dirty_w_.push_back(pid >> 6); }
            czacc_[pid >> 6] ^= 1ull << (pid & 63);
            const int i = pair_list_[pid].first, j = pair_list_[pid].second;
            if ((vmask_[j >> 6] >> (j & 63)) & 1) {
                if (!in_dq_[i]) { in_dq_[i] = 1; dirty_q_.push_back(i); }
                acc_a_[i] = (uint8_t)((acc_a_[i] + 2) & 3);
            }
            if ((vmask_[i >> 6] >> (i & 63)) & 1) {
                if (!in_dq_[j]) { in_dq_[j] = 1; dirty_q_.push_back(j); }
                acc_a_[j] = (uint8_t)((acc_a_[j] + 2) & 3);
            }
        }
        for (int32_t q : sd.v) { vmask_[q >> 6] ^= 1ull << (q & 63); vdirty_.push_back(q); }
        for (int w = 0; w < MW_; ++w) mflip_acc_[w] ^= sd.mflip[w];
    }
    fired_ += nf;

    // ── classify the FOLDED error ──
    relabels_.clear();
    flips_.clear();
    rots_.clear();
    badp_.clear();
    for (int32_t q : dirty_q_) {
        const int a = acc_a_[q] & 3;
        if (!a) continue;
        const int r = rid_[q];
        if (r >= 0 && rbase_[q] != 2) {
            if (a & 1) rots_.push_back({r, rbase_[q], (uint8_t)a});   // basis-rotating read
            else flips_.push_back(r);         // S^2 = Z flips an X/Y read
        }                                     // (any S-power on a Z-read or unread wire: no effect)
    }
    for (int32_t w : dirty_w_) {
        uint64_t bits = czacc_[w];
        while (bits) {
            const int pid = (w << 6) + __builtin_ctzll(bits);
            bits &= bits - 1;
            const PairCls& pc = paircls_[pid];
            if (pc.kind == 2) badp_.push_back(pid);
            else if (pc.kind == 1) relabels_.push_back({pc.target, pc.source});
            else if (pc.kind == 3) flips_.push_back(pc.target);   // classical Z^1 on the read
        }
    }
    bad = !rots_.empty() || !badp_.empty();

    const PatternPlan* pat = nullptr;
    if (bad) {
        // Pattern memo: key = (rot wires' reads, entangling pair ids). The a∈{1,3} values and
        // everything diagonal-non-rotating stay OUT of the key — they are relabel layers on the
        // pattern's law (the basis swap handles the rotation; a ROT_SIGN correction the sign).
        std::sort(badp_.begin(), badp_.end());
        key_.clear();
        for (const auto& ri : rots_) key_.push_back(ri.read);
        std::sort(key_.begin(), key_.end());
        key_.push_back(-1);
        for (int32_t pid : badp_) key_.push_back(pid);
        static const bool memo_on = [] {
            const char* e = std::getenv("QEC_FRAMED_MEMO");
            bool on = (e == nullptr || e[0] != '0');
            if (!on) {
                const char* q = std::getenv("XTIM_QUIET");
                bool quiet = (q && q[0] && q[0] != '0');
                if (!quiet) fprintf(stderr, "[xtim] QEC_FRAMED_MEMO=0: pattern memo disabled (slower; diagnostic)\n");
            }
            return on;
        }();
        if (memo_on) {
            auto it = memo_.find(key_);
            if (it == memo_.end() && memo_.size() < 4096)
                it = memo_.emplace(key_, build_pattern_(key_)).first;
            // R>0: a pattern is usable only with a valid expectation channel (else the bad-shot
            // path below carries the shot — exact at any active-set size).
            if (it != memo_.end() && it->second.ok &&
                (!exp_on_ || (it->second.chan.split_ok && it->second.chan.leaves_ok)))
                pat = &it->second;
        }
    }

    if (force_fb_) {
        // Forced full-state fallback (QEC_FORCE_FALLBACK=1): records (+ expectations at R>0)
        // from the retained full-state general path — the engine-neutral A/B oracle route.
        full_fallback_shot_(rng, out);
    } else if (!bad || pat) {
        // Cascade path: base outcomes from the (zero-error or pattern) plan, then the
        // deterministic relabel layers (cascade theorem).
        ++tier0_;
        bool live_route = false;
        if (pat) {
            replay_pattern_(*pat, rng, out);
        } else if (exp_on_ && !(sp_.plan_active() && zero_chan_.leaves_ok)) {
            // R>0 without a leaf channel (over-cap channel tree / refused leaf): live-block base
            // outcomes — block_work() then IS the shot's zero-error block post-state and the
            // channel evaluates on it directly (no coin corrections needed).
            sp_.live_base_outcomes(rng, out);
            live_route = true;
        } else {
            sp_.base_outcomes(rng, out);
        }
        // Expectations read the BASE outcomes (the |φ> collapse values) — evaluate BEFORE the
        // relabel layers below rewrite `out` into the reported record.
        if (exp_on_) {
            if (pat) eval_channel_shot_(pat->chan, &pat->tree, pat_fptr_, pat_nfired_,
                                        pat_osign_, pat_leaf_, pat_coins_, nullptr, out);
            else if (live_route) eval_channel_shot_(zero_chan_, nullptr, nullptr, 0, 0, -1,
                                                    no_coins_, &sp_.block_work(), out);
            else if (nev == 0) eval_zero_fast_(sp_.last_leaf(), sp_.last_osign());
            else eval_channel_shot_(zero_chan_, &sp_.tree(), sp_.last_fired_nodes(),
                                    sp_.last_nfired(), sp_.last_osign(), sp_.last_leaf(),
                                    sp_.last_coin_words(), nullptr, out);
        }
        for (const auto& rl : relabels_)                      // relabels read BASE values (CZ acts
            if (out[rl.second] == -1) out[rl.first] = -out[rl.first];   // before X^v in the normal form)
        for (int w = 0; w < MW_; ++w) {
            uint64_t bits = mflip_acc_[w];
            while (bits) { const int r = (w << 6) + __builtin_ctzll(bits); bits &= bits - 1; out[r] = -out[r]; }
        }
        for (int r : flips_) out[r] = -out[r];
        // ROT_SIGN correction for the swapped reads: composed(base, a, v) XOR the mflip layer's
        // v-contribution reduces to (base==X && a==1) || (base==Y && a==3); v drops out.
        for (const auto& ri : rots_)
            if ((ri.base == 0 && ri.a == 1) || (ri.base == 1 && ri.a == 3)) out[ri.read] = -out[ri.read];
    } else {
        // full A/B path with the general folded error.
        ++full_;
        for (int32_t q : dirty_q_) E_.a[q] = acc_a_[q] & 3;
        for (int32_t q : vdirty_) E_.v[q] = (uint8_t)((vmask_[q >> 6] >> (q & 63)) & 1);
        E_.cz.clear();
        for (int32_t w : dirty_w_) {
            uint64_t bits = czacc_[w];
            while (bits) {
                const int pid = (w << 6) + __builtin_ctzll(bits);
                bits &= bits - 1;
                E_.cz.push_back(pair_list_[pid]);
            }
        }
        if (exp_on_) sp_.error_shot_obs(E_, rng, out, obsP_, exp_.data());   // R>0 bad-shot path
        else sp_.error_shot(E_, rng, out);
        for (int32_t q : dirty_q_) E_.a[q] = 0;
        for (int32_t q : vdirty_) E_.v[q] = 0;
    }

    // ── sparse reset of the fold scratch ──
    for (int32_t q : dirty_q_) { acc_a_[q] = 0; in_dq_[q] = 0; }
    for (int32_t w : dirty_w_) { czacc_[w] = 0; in_dw_[w] = 0; }
    for (int32_t q : vdirty_) vmask_[q >> 6] = 0;   // word-clear (idempotent across duplicates)
    for (int w = 0; w < MW_; ++w) mflip_acc_[w] = 0;
}


FramedCircuitShotSampler::PatternPlan
FramedCircuitShotSampler::build_pattern_(const std::vector<int32_t>& key) const {
    PatternPlan P;
    // decode the key: [rot read indices..., -1, entangling pair ids...]
    std::vector<int> rotreads;
    std::vector<int32_t> pids;
    size_t i = 0;
    for (; i < key.size() && key[i] != -1; ++i) rotreads.push_back(key[i]);
    for (++i; i < key.size(); ++i) pids.push_back(key[i]);

    // Dress the block with ONLY the pattern's entangling CZs (the basis rotations are handled by
    // swapping the read list; every diagonal-non-rotating component is a per-shot relabel layer).
    FactoredDiagError Ek;
    Ek.a.assign(n_, 0);
    Ek.v.assign(n_, 0);
    for (int32_t pid : pids) Ek.cz.push_back(pair_list_[pid]);
    FramedSuperposition blk(0);
    std::vector<int> b2g;
    framed_active_block(sp_.factored(), Ek, blk, &b2g, nullptr);

    std::vector<int> g2a(n_, -1);
    for (int c = 0; c < (int)b2g.size(); ++c) g2a[b2g[c]] = c;
    std::vector<uint8_t> swapped(M_, 0);
    for (int r : rotreads) swapped[(size_t)r] = 1;
    FactoredDiagError E0;
    E0.a.assign(n_, 0);
    E0.v.assign(n_, 0);
    for (int k = 0; k < M_; ++k) {
        int basis = rd_[k].first;
        if (swapped[k]) basis = (basis == 0) ? 1 : 0;   // odd S-power: X <-> Y read basis
        const int w = rd_[k].second;
        const int col = g2a[w];
        if (col >= 0) {
            P.block_reads.push_back({basis, col});
            P.block_pos.push_back(k);
        } else {
            const ReadResult rr = reduced_read(E0, basis, w, sp_.a_axis1(), sp_.a_sign());
            P.a_pos.push_back(k);
            P.a_kind.push_back(rr.deterministic ? (rr.value == +1 ? 0 : 1) : 2);
        }
    }
    // Baked out vector + coin A-read list (same replay economics as the zero-error plan).
    P.zero_out.assign((size_t)M_, +1);
    for (size_t a = 0; a < P.a_pos.size(); ++a) {
        if (P.a_kind[a] == 1) P.zero_out[P.a_pos[a]] = -1;
        else if (P.a_kind[a] == 2) P.a_coin_pos.push_back(P.a_pos[a]);
    }
    if (!exp_on_) {
        P.ok = extract_tree_plan(blk, P.block_reads, P.tree);
        if (P.ok) expand_corrected_tree(P.tree, P.block_reads.size());
        return P;
    }
    // R>0: correction-chain tree + leaf capture, then the pattern's expectation channel — the
    // static split at the SWAPPED read bases over the dressed block's column map, the per-leaf
    // posts/corrections on the dressed block, and the bounded path-unique expansion.
    std::vector<std::pair<int32_t, FramedSuperposition>> ls;
    P.ok = extract_tree_plan(blk, P.block_reads, P.tree, /*channel=*/true, &ls);
    if (P.ok) {
        std::vector<int8_t> rb((size_t)M_);
        for (int k = 0; k < M_; ++k) rb[k] = (int8_t)rd_[k].first;
        for (int r : rotreads) rb[(size_t)r] = rb[(size_t)r] == 0 ? 1 : 0;
        std::vector<int> w2r(n_, -1);
        for (int k = 0; k < M_; ++k) w2r[rd_[k].second] = k;
        build_obs_split(P.chan, obsP_, n_, rb, w2r, g2a, blk.n(),
                        sp_.a_axis1(), sp_.a_sign(), Ez_);
        const bool lok = build_obs_leaves(P.chan, P.tree, ls, P.block_reads);
        if (lok) expand_corrected_tree(P.tree, P.block_reads.size());
    }
    return P;
}

void FramedCircuitShotSampler::replay_pattern_(const PatternPlan& P,
                                               const std::function<double()>& rng,
                                               std::vector<int>& out) {
    out = P.zero_out;
    FairPool fp{rng};
    tree_replay_block(P.tree, P.block_pos, rng, fp, pat_coins_, pat_oz_, out, &pat_leaf_,
                      &pat_togg_, &pat_fired_, &pat_fptr_, &pat_nfired_, &pat_osign_);
    const int nac = (int)P.a_coin_pos.size();
    if (nac) {   // bulk fair bits — the same pool bits, same order, as the per-read coin loop
        pat_acw_.assign((size_t)(nac + 63) / 64, 0);
        fp.take(nac, pat_acw_.data());
        for (int i = 0; i < nac; ++i)
            out[P.a_coin_pos[i]] = ((pat_acw_[i >> 6] >> (i & 63)) & 1) ? -1 : +1;
    }
}

}  // namespace qeccore
