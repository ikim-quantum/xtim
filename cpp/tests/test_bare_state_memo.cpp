// test_bare_state_memo.cpp — the framed bare-state memo (bare_state_memo.hpp).
//
//   1. READ-SET PIN (mutation-verified): the build is a pure function of (n, Gate instructions in
//      stream order). Every non-gate field of each bundled example (noise channels + probabilities,
//      Measure bases / inverts / readout flips, Resets, Observables, ports) is perturbed: the memo key
//      must not change AND the uncached build must stay byte-identical (bare_states_identical). A
//      gate perturbation (kind, target, insertion, deletion) must change the key.
//   2. HIT IDENTITY: the second build through build_bare_state_framed is served from the memo and is
//      field-for-field identical to the first and to a fresh uncached build; the memo entries are
//      not charged to the state budget (live_state_bytes unchanged by a held entry).
//   3. COMPARATOR MUTATION: bare_states_identical detects a 1-ulp coefficient change, a sigma bit,
//      an eps bit, a tableau phase, and a dropped dual.
//   4. BOUNDS: entry cap evicts LRU; a zero byte cap stores nothing; a rejected build is never stored.
#include "check.hpp"

#include <complex>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "qeccore/bare_state_memo.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/state_budget.hpp"
#include "qeccore/stim_parse.hpp"

using namespace qeccore;

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    CHECK(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Circuit deferred_of(const std::string& text) {
    ParsedStim ps = parse_stim_circuit(text);
    CHECK(ps.ok());
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    return normalize(ps.circuit, pol).normalized;
}

const char* kExamples[] = {
    "cultivation_d3_faithful.stim", "cultivation_d3_rate.stim", "distillation_15_1_3.stim",
    "cube_ccz.stim", "code_switching_faithful.stim", "code_switching_rate.stim",
    "ch_cultivation.stim", "miniature_oracle.stim",
};

std::vector<Circuit> corpus() {
    std::vector<Circuit> out;
    for (const char* name : kExamples) {
        Circuit d = deferred_of(slurp(std::string(XTIM_EXAMPLES_DIR) + "/" + name));
        FramedBareState b = build_bare_state_framed_uncached(d);
        if (b.rejected) continue;   // out of the memo's domain (never stored)
        out.push_back(std::move(d));
    }
    CHECK(out.size() >= 6);
    return out;
}

// ── 1. read-set pin ───────────────────────────────────────────────────────────────────────────
void test_read_set_pin_by_mutation() {
    int mutants = 0, gate_mutants = 0;
    for (const Circuit& d0 : corpus()) {
        const std::string k0 = bare_state_memo_key(d0);
        const FramedBareState b0 = build_bare_state_framed_uncached(d0);
        CHECK(!b0.rejected);
        auto same = [&](const Circuit& d, const char* what) {
            CHECK(bare_state_memo_key(d) == k0);
            FramedBareState b = build_bare_state_framed_uncached(d);
            std::string why;
            if (!bare_states_identical(b0, b, &why))
                std::fprintf(stderr, "  non-gate mutation '%s' changed the build: %s\n", what, why.c_str());
            CHECK(bare_states_identical(b0, b, nullptr));
            ++mutants;
        };
        // (a) every noise channel: another channel, halved probabilities, extra qubit list entry
        {
            Circuit d = d0;
            bool any = false;
            for (Instr& ins : d.stream)
                if (ins.kind == Instr::Kind::Noise) {
                    ins.channel = ins.channel == NoiseChannel::X_ERROR ? NoiseChannel::Z_ERROR
                                                                        : NoiseChannel::X_ERROR;
                    for (double& p : ins.probs) p *= 0.5;
                    ins.origin = Instr::Origin::UserSite;
                    ins.origin_index = 7;
                    any = true;
                }
            if (any) same(d, "noise channel/probs");
        }
        // (b) every Measure: basis cycled, invert flipped, readout flip set
        {
            Circuit d = d0;
            bool any = false;
            for (Instr& ins : d.stream)
                if (ins.kind == Instr::Kind::Measure) {
                    ins.basis = ins.basis == PauliBasis::Z ? PauliBasis::X
                              : ins.basis == PauliBasis::X ? PauliBasis::Y : PauliBasis::Z;
                    ins.invert = !ins.invert;
                    ins.readout_flip_p = 0.3;
                    any = true;
                }
            if (any) same(d, "measure basis/invert/flip");
        }
        // (c) inserted Reset + Observable (with a frame) + ControlledPauli at the front and back
        {
            Circuit d = d0;
            Instr r; r.kind = Instr::Kind::Reset; r.qubits = {0};
            Instr o; o.kind = Instr::Kind::Observable; o.obs = {{0, PauliBasis::X}}; o.obs_frame = {0};
            Instr c; c.kind = Instr::Kind::ControlledPauli; c.basis = PauliBasis::X; c.qubits = {0};
            c.control_rec_offset = 1;
            d.stream.insert(d.stream.begin(), r);
            d.stream.push_back(o);
            d.stream.push_back(c);
            same(d, "inserted Reset/Observable/ControlledPauli");
        }
        // (d) ports and every Noise instruction REMOVED
        {
            Circuit d = d0;
            d.inputs.push_back(InputPort{InputPort::Kind::Bits, "b", 2, {}});
            d.outputs.push_back(OutputPort{"o", {0}});
            std::vector<Instr> kept;
            for (const Instr& ins : d.stream)
                if (ins.kind != Instr::Kind::Noise) kept.push_back(ins);
            d.stream = std::move(kept);
            same(d, "ports + noise removed");
        }
        // GATE mutations must change the key.
        {
            Circuit d = d0;
            for (Instr& ins : d.stream)
                if (ins.kind == Instr::Kind::Gate && ins.targets.size() == 2) {
                    ins.gate = ins.gate == GateKind::CX ? GateKind::CZ : GateKind::CX;
                    break;
                }
            CHECK(bare_state_memo_key(d) != k0);
            ++gate_mutants;
        }
        {
            Circuit d = d0;
            for (Instr& ins : d.stream)
                if (ins.kind == Instr::Kind::Gate) { ins.targets[0] = (ins.targets[0] + 1) % d.n; break; }
            CHECK(bare_state_memo_key(d) != k0);
        }
        {
            Circuit d = d0;
            Instr g; g.kind = Instr::Kind::Gate; g.gate = GateKind::S; g.targets = {0};
            d.stream.push_back(g);
            CHECK(bare_state_memo_key(d) != k0);
        }
        {
            Circuit d = d0;
            for (size_t i = 0; i < d.stream.size(); ++i)
                if (d.stream[i].kind == Instr::Kind::Gate) { d.stream.erase(d.stream.begin() + (long)i); break; }
            CHECK(bare_state_memo_key(d) != k0);
        }
        {
            Circuit d = d0;
            d.n += 1;
            CHECK(bare_state_memo_key(d) != k0);
        }
    }
    std::fprintf(stderr, "  %d non-gate mutants identical, %d gate mutants re-keyed\n", mutants, gate_mutants);
    CHECK(mutants >= 18);
}

// ── 2. hit identity + budget neutrality ───────────────────────────────────────────────────────
void test_hit_identity() {
    bare_state_memo_clear();
    bare_state_memo_reset_stats();
    int hits_seen = 0;
    for (const Circuit& d : corpus()) {
        const int64_t live0 = live_state_bytes();
        FramedBareState first = build_bare_state_framed(d);
        BareStateMemoStats s1 = bare_state_memo_stats();
        FramedBareState second = build_bare_state_framed(d);
        BareStateMemoStats s2 = bare_state_memo_stats();
        CHECK(s2.hits == s1.hits + 1);
        CHECK(s2.stores == s1.stores);
        CHECK(s1.entries >= 1);
        std::string why;
        if (!bare_states_identical(first, second, &why))
            std::fprintf(stderr, "  hit differs from first build: %s\n", why.c_str());
        CHECK(bare_states_identical(first, second, &why));
        FramedBareState fresh = build_bare_state_framed_uncached(d);
        CHECK(bare_states_identical(fresh, second, &why));
        CHECK(bare_states_identical(fresh, first, &why));
        // a hit's state is a live chi-scaled state (charged); the memo entry itself is not: with
        // both results dropped the live total is back where it started although the entry is held.
        { FramedBareState drop1 = std::move(first), drop2 = std::move(second), drop3 = std::move(fresh); }
        CHECK(live_state_bytes() == live0);
        CHECK(bare_state_memo_stats().entries >= 1);
        ++hits_seen;
    }
    CHECK(hits_seen >= 6);
}

// ── 3. comparator mutation ────────────────────────────────────────────────────────────────────
void test_comparator_detects_mutations() {
    Circuit d = deferred_of("RX 0 1 2\nT 0 1 2\nCX 0 1\nCZ 1 2\nMX 0 1 2\n");
    FramedBareState a = build_bare_state_framed_uncached(d);
    CHECK(!a.rejected);
    CHECK(a.state.chi() >= 2);
    std::string why;
    CHECK(bare_states_identical(a, a, &why));
    {
        FramedBareState b = a;
        std::complex<double>& c = b.state.coeff(0);
        double re = c.real();
        re = std::nextafter(re, re + 1.0);              // 1 ulp
        c = std::complex<double>(re, c.imag());
        CHECK(!bare_states_identical(a, b, &why));
        CHECK(why.find("coeff[0]") != std::string::npos);
    }
    {
        FramedBareState b = a;
        b.state.sigma(1)[0] ^= 1;
        CHECK(!bare_states_identical(a, b, &why));
        CHECK(why.find("sigma[1]") != std::string::npos);
    }
    {
        FramedBareState b = a;
        b.state.eps[0] ^= 1;
        CHECK(!bare_states_identical(a, b, &why));
        CHECK(why == "eps");
    }
    {
        FramedBareState b = a;
        b.state.U.Zrow[0].phase ^= 2;
        CHECK(!bare_states_identical(a, b, &why));
        CHECK(why.find("Zrow[0]") != std::string::npos);
    }
    {
        FramedBareState b = a;
        b.state.U.ensure_dual();
        FramedBareState c = a;
        c.state.U.ensure_dual();
        CHECK(bare_states_identical(b, c, &why));
        if (!a.state.U.dual_valid()) {
            CHECK(!bare_states_identical(a, b, &why));
            CHECK(why == "dual validity");
        }
    }
    {
        FramedBareState b = a;
        b.chi += 1;
        CHECK(!bare_states_identical(a, b, &why));
    }
}

// ── 4. bounds + rejected builds ───────────────────────────────────────────────────────────────
void test_bounds() {
    std::vector<Circuit> cs = corpus();
    const BareStateMemoStats dflt = bare_state_memo_stats();
    bare_state_memo_clear();
    // entry cap 1: after building two distinct circuits one entry is held and one was evicted
    bare_state_memo_set_caps(1, size_t(1) << 40);
    bare_state_memo_reset_stats();
    (void)build_bare_state_framed(cs[0]);
    (void)build_bare_state_framed(cs[1]);
    BareStateMemoStats s = bare_state_memo_stats();
    CHECK(s.entries == 1);
    CHECK(s.stores == 2);
    CHECK(s.evictions == 1);
    (void)build_bare_state_framed(cs[1]);        // the held one hits
    CHECK(bare_state_memo_stats().hits == 1);
    (void)build_bare_state_framed(cs[0]);        // the evicted one misses (and evicts cs[1])
    s = bare_state_memo_stats();
    CHECK(s.misses == 3);
    CHECK(s.entries == 1);
    // byte cap 0: nothing stored, every build counted as skipped
    bare_state_memo_set_caps(8, 0);
    CHECK(bare_state_memo_stats().entries == 0);
    bare_state_memo_reset_stats();
    (void)build_bare_state_framed(cs[0]);
    (void)build_bare_state_framed(cs[0]);
    s = bare_state_memo_stats();
    CHECK(s.entries == 0);
    CHECK(s.bytes == 0);
    CHECK(s.stores == 0);
    CHECK(s.skipped == 2);
    CHECK(s.hits == 0);
    // byte cap just above one entry: a second, larger entry evicts the first (LRU by bytes)
    bare_state_memo_set_caps(8, size_t(1) << 40);
    bare_state_memo_reset_stats();
    (void)build_bare_state_framed(cs[0]);
    const size_t one = bare_state_memo_stats().bytes;
    CHECK(one > 0);
    bare_state_memo_set_caps(8, one);
    CHECK(bare_state_memo_stats().entries == 1);
    (void)build_bare_state_framed(cs[1]);
    s = bare_state_memo_stats();
    CHECK(s.entries <= 1);
    CHECK(s.bytes <= one);
    // per-entry cap just under one entry: the build is skipped BEFORE any capture (no copy), the
    // call still returns the state; raising the cap stores it again
    bare_state_memo_set_caps(8, size_t(1) << 40);
    bare_state_memo_clear();
    bare_state_memo_set_entry_cap(one - 1);
    bare_state_memo_reset_stats();
    {
        FramedBareState r = build_bare_state_framed(cs[0]);
        CHECK(!r.rejected);
        s = bare_state_memo_stats();
        CHECK(s.entries == 0);
        CHECK(s.skipped == 1);
        CHECK(s.stores == 0);
        CHECK(s.cap_entry_bytes == one - 1);
    }
    bare_state_memo_set_entry_cap(one);
    (void)build_bare_state_framed(cs[0]);
    CHECK(bare_state_memo_stats().entries == 1);
    bare_state_memo_set_entry_cap(dflt.cap_entry_bytes);
    // a rejected build is never stored (non-commuting magic: T, H, T on one wire)
    bare_state_memo_set_caps(8, size_t(1) << 40);
    bare_state_memo_clear();
    bare_state_memo_reset_stats();
    {
        Circuit d = deferred_of("RX 0 1\nT 0\nCX 0 1\nT 1\nH 0\nT 0\nCX 1 0\nT 0\nH 0\nT 0\nMX 0 1\n");
        FramedBareState r = build_bare_state_framed(d);
        if (r.rejected) {
            s = bare_state_memo_stats();
            CHECK(s.entries == 0);
            CHECK(s.skipped == 1);
            (void)build_bare_state_framed(d);
            CHECK(bare_state_memo_stats().hits == 0);
        } else {
            std::fprintf(stderr, "  (rejection probe built; rejected-not-stored check skipped)\n");
        }
    }
    bare_state_memo_set_caps(dflt.cap_entries, dflt.cap_bytes);
    bare_state_memo_clear();
}

}  // namespace

int main() {
    RUN(test_read_set_pin_by_mutation);
    RUN(test_hit_identity);
    RUN(test_comparator_detects_mutations);
    RUN(test_bounds);
    REPORT();
}
