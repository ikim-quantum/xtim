// J-a N6 — record-space B (qeccore/record_space_b.hpp) against the σ-dual-basis builder it replaces:
//   * G-B1 (unit half): build_record_table_recB == the sampler's plan_structure table (build_shot_law +
//     build_record_table, the σ dual basis) FIELD for field (pivot_rec, born_rec, coin / born rows, base, Λ as
//     (a, b, EFFECTIVE phase): plan §3.3 convention, §5 Q12) on κ > 0 plans of the [[4,2,2]]-block fixtures of
//     tests/test_twirl_kappa_dual.py (κ = B kernel directions, distinct Born factors per block), with and
//     without a CZ in the residual. The corpus half of G-B1 is tests/test_ja_record_space_b.py.
//   * G-B2 (mechanism, design B-F1 F-9): the B stage's work is independent of inactive generators — the same plans
//     with k = 8, 16, 32 inactive padding checks (R q / M q / DETECTOR: a certified Z_q no residual touches, and an
//     untouched channel) have IDENTICAL B operation counts (touched channels, generator visits, elimination and
//     preimage row operations, pivots), and the table is unchanged. Mutation XTIM_MUTATE_B_PAD (the channel
//     loop visits every generator of a mask, inactive ones too) → the counts grow with the padding: red.
#include "check.hpp"

#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/record_space_b.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/sampler.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/twirl_sampler.hpp"

using namespace qeccore;

namespace {

// tests/test_twirl_kappa_dual.py circuit_text(B), with `pad` inactive Z checks before OUTPUT_QUBITS.
std::string circuit_text(int B, int pad) {
    std::string s;
    auto L = [&](const std::string& x) { s += x + "\n"; };
    for (int b = 0; b < B; ++b) L("RX " + std::to_string(4 * b));
    for (int b = B - 1; b > 0; --b) L("CX " + std::to_string(4 * b) + " " + std::to_string(4 * (b - 1)));
    for (int b = 0; b < B; ++b) {
        for (int d : {4 * b + 1, 4 * b + 2}) {
            L("RX " + std::to_string(d));
            L("T " + std::to_string(d));
            if (b != 2) L("H " + std::to_string(d));
            if (b == 1) L("X " + std::to_string(d));
        }
        L("R " + std::to_string(4 * b + 3));
    }
    for (int b = 0; b < B; ++b) {
        const int a0 = 4 * b, d1 = 4 * b + 1, d2 = 4 * b + 2, a3 = 4 * b + 3;
        L("CX " + std::to_string(a0) + " " + std::to_string(d1));
        L("CX " + std::to_string(a0) + " " + std::to_string(d2));
        L("CX " + std::to_string(d1) + " " + std::to_string(a3));
        L("CX " + std::to_string(d2) + " " + std::to_string(a3));
        L("CX " + std::to_string(a0) + " " + std::to_string(a3));
    }
    L("X_ERROR(0.01) " + std::to_string(4 * B - 1));
    for (int b = 0; b < B; ++b) {
        std::string x = "MPP", z = "MPP";
        for (int i = 0; i < 4; ++i) {
            x += (i ? "*X" : " X") + std::to_string(4 * b + i);
            z += (i ? "*Z" : " Z") + std::to_string(4 * b + i);
        }
        L(x); L("DETECTOR rec[-1]"); L(z); L("DETECTOR rec[-1]");
    }
    for (int i = 0; i < pad; ++i) {
        const std::string q = std::to_string(4 * B + i);
        L("R " + q); L("M " + q); L("DETECTOR rec[-1]");
    }
    std::string o = "OUTPUT_QUBITS out";
    for (int q = 0; q < 4 * B; ++q) o += " " + std::to_string(q);
    L(o);
    return s;
}

// The FamilySampler / TwirlSampler front end (cpp/testapi/family_py.cpp), production options.
struct Bundle {
    std::unique_ptr<FramedSuperposition> bare;
    Circuit deferred;
    std::vector<std::pair<int, int>> reads;
    std::vector<std::vector<int>> detectors;
    std::vector<std::pair<int, std::vector<int>>> obs;
    std::vector<int> final_wire;
    TwirlRecordOptions opt;
    std::unique_ptr<TwirlRecordSampler> s;
};

std::unique_ptr<Bundle> compile(const std::string& text) {
    auto b = std::make_unique<Bundle>();
    ParsedStim ps = parse_stim_circuit(text);
    if (!ps.ok()) return nullptr;
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    pol.hoist_feedback_on_class_exit = true;
    pol.zero_noise_transparent = true;
    NormalizeResult nr = normalize(ps.circuit, pol);
    for (const Instr& ins : nr.coherent.stream)
        if (ins.kind == Instr::Kind::Measure) {
            b->opt.rec_flip.push_back(ins.readout_flip_p);
            b->opt.rec_invert.push_back(ins.invert ? 1 : 0);
        }
    {
        size_t j = 0;
        for (const Instr& ins : nr.normalized.stream)
            if (ins.kind == Instr::Kind::Measure) {
                if (j < b->opt.rec_invert.size() && ins.invert) b->opt.rec_invert[j] ^= 1;
                ++j;
            }
    }
    b->deferred = std::move(nr.normalized);
    b->final_wire = nr.map.final_wire;
    for (const auto& r : nr.map.terminal_reads) b->reads.push_back({(int)r.second, r.first});
    b->detectors = std::move(ps.detectors);
    b->obs.assign(ps.observables.begin(), ps.observables.end());
    b->opt.decisions = std::move(ps.decisions);
    FramedBareState bs = build_bare_state_framed(b->deferred);
    if (bs.rejected) return nullptr;
    b->bare.reset(new FramedSuperposition(std::move(bs.state)));
    b->opt.p_factor = 1.0;
    b->opt.circuit_channels = true;
    b->opt.selfcheck = 0;
    b->s = TwirlRecordSampler::make(*b->bare, b->deferred, b->reads, b->detectors, b->obs, 1, b->opt);
    if (b->s->setup_error() != 0) return nullptr;
    return b;
}

// The sampler's record columns (storage order) from its public accessors.
std::vector<RecColumn> columns_of(const TwirlRecordSampler& s) {
    const std::vector<Pauli> ops = s.record_operators();
    const std::vector<int> ord = s.record_column_order();
    const std::vector<std::vector<uint64_t>> masks = s.record_channel_masks();
    const std::vector<uint8_t> refs = s.record_channel_refs();
    const int nch = s.num_channels();
    std::vector<RecColumn> cols(ops.size());
    for (size_t c = 0; c < ops.size(); ++c) {
        cols[c].M = ops[c];
        cols[c].order = ord[c];
        cols[c].channel = (int)c < nch;
        if ((int)c < nch) {
            cols[c].mask = masks[c];
            cols[c].ref = refs[c];
        }
    }
    return cols;
}

// plan_structure's key (identity prefix): prefix x / z words ‖ a bytes ‖ 0xFF ‖ cz u16 pairs.
std::vector<uint8_t> plan_key(int n, const std::vector<int>& a, const std::vector<std::pair<int, int>>& cz) {
    const int NW = (n + 63) / 64;
    std::vector<uint8_t> k((size_t)16 * NW, 0);
    std::vector<uint8_t> am((size_t)n, 0);
    for (int q : a) am[(size_t)q] = 1;
    k.insert(k.end(), am.begin(), am.end());
    k.push_back(0xFF);
    for (const auto& e : cz)
        for (int v : {e.first, e.second}) { k.push_back((uint8_t)(v & 255)); k.push_back((uint8_t)(v >> 8)); }
    return k;
}

bool row_bit(const std::vector<uint64_t>& r, int c) { return (r[(size_t)(c >> 6)] >> (c & 63)) & 1ULL; }

// The FIELD identity of two canonical tables (plan §3.3 convention); prints the first difference.
bool same_table(const RecTable& d, const RecTable& b, const char* tag) {
    auto fail = [&](const char* what) { std::fprintf(stderr, "  [%s] tables differ: %s\n", tag, what); return false; };
    if (!d.built || !b.built) return fail("unbuilt");
    if (d.m != b.m || d.rho != b.rho || d.kappa != b.kappa) return fail("m / rho / kappa");
    if (d.pivot_rec != b.pivot_rec) return fail("pivot_rec");
    if (d.born_rec != b.born_rec) return fail("born_rec");
    for (int c = 0; c < d.m; ++c) {
        if (row_bit(d.base, c) != row_bit(b.base, c)) return fail("beta");
        for (int k = 0; k < d.rho; ++k)
            if (row_bit(d.coin_row[(size_t)k], c) != row_bit(b.coin_row[(size_t)k], c)) return fail("A");
        for (int j = 0; j < d.kappa; ++j)
            if (row_bit(d.born_row[(size_t)j], c) != row_bit(b.born_row[(size_t)j], c)) return fail("B");
    }
    for (int j = 0; j < d.kappa; ++j) {
        const FreeDir &x = d.lam[(size_t)j], &y = b.lam[(size_t)j];
        if (x.a != y.a || x.b != y.b || fd_eff_phase(x) != fd_eff_phase(y)) return fail("Lambda");
    }
    return true;
}

struct Case { int B; std::vector<std::pair<int, int>> cz; const char* name; };
void test_recb_equals_dual_basis_and_padding_invariance() {
    const Case cases[] = {{2, {}, "B2"}, {2, {{1, 2}}, "B2_cz"}, {3, {}, "B3"}, {3, {{1, 2}, {5, 6}}, "B3_cz"}};
    const int pads[] = {0, 8, 16, 32};
    int compared = 0, kappa_plans = 0;
    for (const Case& cs : cases) {
        RecBOps base_ops;
        RecTable base_tab;
        for (int pad : pads) {
            auto b = compile(circuit_text(cs.B, pad));
            CHECK(b != nullptr);
            if (!b) continue;
            const TwirlRecordSampler& s = *b->s;
            const int n = s.deferred_wires();
            std::vector<int> a;
            for (int blk = 0; blk < cs.B; ++blk) { a.push_back(b->final_wire[(size_t)(4 * blk)]);
                                                   a.push_back(b->final_wire[(size_t)(4 * blk + 1)]); }
            std::vector<std::pair<int, int>> cz;
            for (const auto& e : cs.cz) cz.push_back({b->final_wire[(size_t)e.first], b->final_wire[(size_t)e.second]});
            const std::vector<uint8_t> key = plan_key(n, a, cz);
            const TwirlRecordSampler::PlanStructure ps = s.plan_structure(key.data(), (int)key.size());
            CHECK(!ps.law.fallback);
            CHECK_EQ(ps.law.kappa, cs.B);
            CHECK(ps.rec.built);
            if (ps.law.kappa > 0) ++kappa_plans;
            const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
            RecTable T;
            RecBOps ops;
            const bool built = build_record_table_recB(G, s.bare_state(), ps.nf, columns_of(s), T, &ops);
            CHECK(built);
            const std::string tag = std::string(cs.name) + " pad " + std::to_string(pad);
            CHECK(same_table(ps.rec, T, tag.c_str()));
            ++compared;
            std::fprintf(stderr, "  [%s] kappa %d rho %d kappa_rec %d | touched %ld gen_visits %ld elim_rows %ld "
                         "pre_rows %ld pivots %ld\n", tag.c_str(), ps.law.kappa, T.rho, T.kappa, ops.touched,
                         ops.gen_visits, ops.elim_rows, ops.pre_rows, ops.pivots);
            CHECK(ops.touched > 0);
            if (pad == 0) { base_ops = ops; base_tab = T; continue; }
            // G-B2: identical B work at every padding (the table grows by `pad` constant columns only)
            CHECK_EQ(ops.touched, base_ops.touched);
            CHECK_EQ(ops.gen_visits, base_ops.gen_visits);
            CHECK_EQ(ops.elim_rows, base_ops.elim_rows);
            CHECK_EQ(ops.pre_rows, base_ops.pre_rows);
            CHECK_EQ(ops.pivots, base_ops.pivots);
            CHECK_EQ(T.rho, base_tab.rho);
            CHECK_EQ(T.kappa, base_tab.kappa);
        }
    }
    std::fprintf(stderr, "[record_space_b] %d tables compared (%d kappa > 0 plans)\n", compared, kappa_plans);
    CHECK_EQ(compared, 16);
}

// The READ-direction cases on the same blocks. Block b's X-check record is a single X on a copy wire r_b
// (record_operators); CZ legs from r_b onto block c's data wires d1, d2 dress that record with Z_{d1}Z_{d2}, a
// LOGICAL of block c (a kernel direction): the DETECTOR reads it (B rows over the channels are nonzero). S on r_b
// makes the check a fair coin (a pivot). Legs onto two blocks read the PRODUCT of their directions.
struct RCase {
    int B;
    std::vector<std::pair<int, int>> reads;   // (record block b, data block c): CZ(r_b, d1_c), CZ(r_b, d2_c)
    std::vector<int> coin;                    // record blocks with S on r_b
    const char* name;
};

void test_read_directions() {
    const RCase cases[] = {
        {2, {{0, 0}}, {}, "one_read"},
        {2, {{0, 0}, {1, 1}}, {}, "two_reads"},
        {2, {{0, 0}, {1, 0}, {1, 1}}, {}, "product_read"},
        {2, {{0, 0}, {1, 1}}, {0}, "coin_and_read"},
        {3, {{0, 0}, {1, 1}, {2, 2}, {2, 0}}, {}, "three_reads"},
        {3, {{0, 1}, {1, 1}, {1, 2}, {2, 0}}, {1}, "mixed"},
    };
    const int pads[] = {0, 8, 16, 32};
    int compared = 0, read_plans = 0;
    for (const RCase& cs : cases) {
        RecBOps base_ops;
        for (int pad : pads) {
            auto b = compile(circuit_text(cs.B, pad));
            CHECK(b != nullptr);
            if (!b) continue;
            const TwirlRecordSampler& s = *b->s;
            const int n = s.deferred_wires();
            const std::vector<Pauli> ops_m = s.record_operators();
            auto rec_wire = [&](int blk) {           // the X-check record of block blk: X on one wire
                const Pauli& P = ops_m[(size_t)(2 * blk)];
                int w = -1, cnt = 0;
                for (int q = 0; q < n; ++q) if (P.xbit(q) || P.zbit(q)) { w = q; ++cnt; }
                CHECK(cnt == 1 && P.xbit(w) && !P.zbit(w));
                return w;
            };
            std::vector<int> a;
            for (int blk : cs.coin) a.push_back(rec_wire(blk));
            std::vector<std::pair<int, int>> cz;
            for (const auto& rc : cs.reads)
                for (int d : {4 * rc.second + 1, 4 * rc.second + 2})
                    cz.push_back({rec_wire(rc.first), b->final_wire[(size_t)d]});
            const std::vector<uint8_t> key = plan_key(n, a, cz);
            const TwirlRecordSampler::PlanStructure ps = s.plan_structure(key.data(), (int)key.size());
            CHECK(!ps.law.fallback);
            CHECK(ps.rec.built);
            const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
            RecTable T;
            RecBOps ops;
            CHECK(build_record_table_recB(G, s.bare_state(), ps.nf, columns_of(s), T, &ops));
            const std::string tag = std::string(cs.name) + " pad " + std::to_string(pad);
            CHECK(same_table(ps.rec, T, tag.c_str()));
            ++compared;
            int ch_reads = 0;
            for (int j = 0; j < T.kappa; ++j)
                for (int c = 0; c < s.num_channels(); ++c) ch_reads += row_bit(T.born_row[(size_t)j], c) ? 1 : 0;
            if (ch_reads > 0) ++read_plans;
            std::fprintf(stderr, "  [%s] kappa %d rho %d kappa_rec %d channel reads %d | touched %ld gen_visits %ld "
                         "elim_rows %ld pre_rows %ld pivots %ld\n", tag.c_str(), ps.law.kappa, T.rho, T.kappa,
                         ch_reads, ops.touched, ops.gen_visits, ops.elim_rows, ops.pre_rows, ops.pivots);
            if (pad == 0) { base_ops = ops; continue; }
            CHECK_EQ(ops.touched, base_ops.touched);
            CHECK_EQ(ops.gen_visits, base_ops.gen_visits);
            CHECK_EQ(ops.elim_rows, base_ops.elim_rows);
            CHECK_EQ(ops.pre_rows, base_ops.pre_rows);
            CHECK_EQ(ops.pivots, base_ops.pivots);
        }
    }
    std::fprintf(stderr, "[record_space_b] read directions: %d tables compared, %d with channel reads\n", compared,
                 read_plans);
    CHECK_EQ(compared, 24);
    CHECK_EQ(read_plans, 24);
}

// Random residuals (seeded): S and CZ legs drawn over the X-check record wires and the data wires of every block,
// so channels read single and product directions with every phase pattern the residual can make. Each plan:
// recB == the σ-dual-basis table (field identity). Coverage asserted: channel reads, coins, and directions with
// kb_j = 1 that a channel reads (the β sign path: XTIM_MUTATE_B_KB_DROP must go red here).
void test_random_residuals() {
    std::mt19937_64 rng(20261018);
    int compared = 0, read_plans = 0, coin_plans = 0, fallbacks = 0;
    long kb_read = 0;
    for (int B : {2, 3}) {
        auto b = compile(circuit_text(B, 0));
        CHECK(b != nullptr);
        if (!b) continue;
        const TwirlRecordSampler& s = *b->s;
        const int n = s.deferred_wires();
        const std::vector<Pauli> ops_m = s.record_operators();
        std::vector<int> wires, recw, dataw;
        for (int blk = 0; blk < B; ++blk) {
            const Pauli& P = ops_m[(size_t)(2 * blk)];
            for (int q = 0; q < n; ++q) if (P.xbit(q)) recw.push_back(q);
            for (int d = 0; d < 4; ++d) dataw.push_back(b->final_wire[(size_t)(4 * blk + d)]);
        }
        wires = recw;
        wires.insert(wires.end(), dataw.begin(), dataw.end());
        const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
        const std::vector<RecColumn> cols = columns_of(s);
        for (int trial = 0; trial < 1000; ++trial) {
            std::vector<int> a;
            for (int w : wires) if (rng() % 10 == 0) a.push_back(w);
            std::vector<std::pair<int, int>> cz;
            const int ncz = (int)(rng() % 8);
            for (int e = 0; e < ncz; ++e) {
                const bool leg = rng() % 3 != 0;      // mostly record -> data legs (reads); else any pair
                int x = leg ? recw[(size_t)(rng() % recw.size())] : wires[(size_t)(rng() % wires.size())];
                int y = leg ? dataw[(size_t)(rng() % dataw.size())] : wires[(size_t)(rng() % wires.size())];
                if (x == y) continue;
                if (x > y) std::swap(x, y);
                bool dup = false;
                for (const auto& p : cz) dup |= p.first == x && p.second == y;
                if (!dup) cz.push_back({x, y});
            }
            const std::vector<uint8_t> key = plan_key(n, a, cz);
            const TwirlRecordSampler::PlanStructure ps = s.plan_structure(key.data(), (int)key.size());
            RecTable T;
            RecBOps ops;
            const bool built = build_record_table_recB(G, s.bare_state(), ps.nf, cols, T, &ops);
            if (ps.law.fallback || !ps.rec.built) { CHECK(!built); ++fallbacks; continue; }
            CHECK(built);
            CHECK(same_table(ps.rec, T, ("random B" + std::to_string(B) + " #" + std::to_string(trial)).c_str()));
            ++compared;
            int ch_reads = 0;
            for (int j = 0; j < T.kappa; ++j)
                for (int c = 0; c < s.num_channels(); ++c) ch_reads += row_bit(T.born_row[(size_t)j], c) ? 1 : 0;
            read_plans += ch_reads > 0;
            coin_plans += T.rho > 0;
            kb_read += ops.kb_read;
        }
    }
    std::fprintf(stderr, "[record_space_b] random residuals: %d compared (%d fallback), %d with channel reads, %d with "
                 "coins, %ld read directions with kb = 1\n", compared, fallbacks, read_plans, coin_plans, kb_read);
    CHECK(compared >= 1900);
    CHECK(read_plans >= 300);
    CHECK(coin_plans >= 300);
    CHECK(kb_read >= 30);
}

}  // namespace

int main() {
    RUN(test_recb_equals_dual_basis_and_padding_invariance);
    RUN(test_read_directions);
    RUN(test_random_residuals);
    REPORT();
}
