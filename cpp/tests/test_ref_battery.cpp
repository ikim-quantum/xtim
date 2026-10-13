// test_ref_battery — perf/ref-compile-battery: the work-sized noiseless-invariants battery
// (verification never costs more than compilation) and the failed-TreePlan-extraction memo.
//
//   (1) invariants_shot_count: clamp(W_compile / w1, 1, ceiling); w1 = 0 => ceiling.
//   (2) a cheap circuit (TreePlan replay, w1 = 0) runs the FULL battery: 500 / 200.
//   (3) FACTORY CLASS in CI time: kMagicPE at QEC_TREEPLAN_CAP=2 has no TreePlan, so its noiseless
//       shots collapse the block live (w1 > 0). The work numbers are PINNED (w1 = 42, W_compile =
//       214 => 5 shots) — deleting the kernel's work counter, ignoring the probe or changing the
//       budget definition all change them.
//   (4) one shot sequence: the battery's shots are the first `shots` rows of the ceiling-shot run
//       (sample_program_streamed stops a prefix, never restarts); the DETECTOR verdict equals the
//       full ceiling-shot battery's on every seed; a gauge detector that read 1 on all sized shots
//       continues the SAME sequence (found by scanning seeds).
//   (5) InvariantsTally: the |exp| wobble check and the constant-1 detector check, on synthetic
//       records.
//   (6) memo: init_'s failure is stored; the channel rebuild is SKIPPED explicitly (no memo
//       access, named status); a second compile is answered by the memo, credits the recorded
//       work (same W_compile) and samples the same bytes; a different cap is a different key.
//   (7) memo KEY completeness: changing eps, an amplitude, a frame row (bits or phase), free, the
//       reads or the cap changes the key.
//
// Must end main() with REPORT().
#define REF_COMPILE_NO_MAIN
#include "../apps/ref_compile.cpp"

#include "check.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "qeccore/framed_sampler.hpp"
#include "qeccore/sampler.hpp"

using namespace qeccore;
using refcompile::invariants_shot_count;
using refcompile::InvariantsTally;
using refcompile::noiseless_invariants;

// 3-qubit repetition code, deterministic detectors.
static const char kRep[] =
    "R 0 1 2 3 4\n"
    "X_ERROR(0.01) 0 1 2\n"
    "CX 0 3 1 3\n"
    "CX 1 4 2 4\n"
    "M 3 4\n"
    "DETECTOR rec[-2]\n"
    "DETECTOR rec[-1]\n"
    "M 0 1 2\n"
    "OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]\n";

// chi > 1 block with coupling reads (its TreePlan needs > 2 expansions), a gauge detector and an
// expectation column (R > 0).
static const char kMagicPE[] =
    "R 0 1 2 3\n"
    "H 0 1 2 3\n"
    "T 0 1 2\n"
    "CX 0 1 1 2 2 3\n"
    "H 0 1 2\n"
    "M 0 1 2\n"
    "DETECTOR rec[-1]\n"
    "PAULI_EXPECTATION(0) X3\n";

// kMagicPE plus a product-qubit GAUGE detector (a fair coin noiselessly).
static const char kMagicPEGauge[] =
    "R 0 1 2 3 4\n"
    "H 0 1 2 3 4\n"
    "T 0 1 2\n"
    "CX 0 1 1 2 2 3\n"
    "H 0 1 2\n"
    "M 0 1 2 4\n"
    "DETECTOR rec[-2]\n"
    "DETECTOR rec[-1]\n"
    "PAULI_EXPECTATION(0) X3\n";

using ProgPtr = std::unique_ptr<SamplerProgram, void (*)(SamplerProgram*)>;
static ProgPtr compile_clean(const protoref::LoadedBench& lb) {
    return ProgPtr(compile_sampler_program(protoref::strip_noise(lb.circuit), lb.detectors,
                                           lb.observables, nullptr),
                   &delete_sampler_program);
}

static std::string digest(const PackedRecords& r, int shots) {
    const int MB = PackedRecords::bytes_per_shot(r.num_measurements);
    const int DB = PackedRecords::bytes_per_shot(r.num_detectors);
    std::string s(r.measurements.begin(), r.measurements.begin() + (size_t)shots * MB);
    s.append(r.detectors.begin(), r.detectors.begin() + (size_t)shots * DB);
    s.append(reinterpret_cast<const char*>(r.expectations.data()),
             (size_t)shots * r.num_expectations * sizeof(double));
    return s;
}

static std::string sample_bytes(const protoref::LoadedBench& lb, int shots, uint64_t seed) {
    ProgPtr prog(compile_sampler_program(lb.circuit, lb.detectors, lb.observables, nullptr),
                 &delete_sampler_program);
    PackedRecords r = sample_program(*prog, shots, seed);
    CHECK(!r.rejected);
    return digest(r, shots);
}

static PackedRecords synthetic(const std::vector<double>& exps, const std::vector<int>& det) {
    PackedRecords r;
    r.shots = (int)exps.size();
    r.num_detectors = 1;
    r.num_expectations = 1;
    r.detectors.assign((size_t)r.shots * PackedRecords::bytes_per_shot(1), 0);
    for (int s = 0; s < r.shots; ++s) r.detectors[(size_t)s] = (uint8_t)det[s];
    r.expectations = exps;
    return r;
}

int main() {
    setenv("XTIM_QUIET", "1", 1);

    // (1) the rule
    CHECK(invariants_shot_count(0, 0, 500) == 500);
    CHECK(invariants_shot_count(123456, 0, 500) == 500);
    CHECK(invariants_shot_count(214, 42, 500) == 5);
    CHECK(invariants_shot_count(10, 42, 500) == 1);          // floor 1
    CHECK(invariants_shot_count(0, 42, 500) == 1);
    CHECK(invariants_shot_count(42 * 500, 42, 500) == 500);
    CHECK(invariants_shot_count(42 * 9999, 42, 200) == 200);  // ceiling
    CHECK(invariants_shot_count(42 * 7, 42, 200) == 7);
    CHECK(invariants_shot_count(5, 1, 0) == 0);

    // (2) cheap circuit: the full battery, both ceilings
    {
        protoref::LoadedBench lb = refcompile::bench_from_text(kRep);
        CHECK(lb.ok);
        auto ir = noiseless_invariants(lb, nullptr, 12345, 500);
        CHECK(ir.ok);
        CHECK(ir.shot_work == 0);
        CHECK(ir.shots == 500);
        auto iv = noiseless_invariants(lb, nullptr, 12345, 200);
        CHECK(iv.ok);
        CHECK(iv.shots == 200);
    }

    protoref::LoadedBench lbm = refcompile::bench_from_text(kMagicPE);
    CHECK(lbm.ok);
    setenv("QEC_TREEPLAN_CAP", "2", 1);

    // (3) factory class, pinned work numbers
    {
        tree_fail_memo_clear();
        auto ir = noiseless_invariants(lbm, nullptr, 12345, 500);
        CHECK(ir.ok);
        CHECK_EQ(ir.shot_work, 42u);
        CHECK_EQ(ir.compile_work, 214u);
        CHECK_EQ(ir.sized_shots, 5);
        CHECK(ir.shots >= ir.sized_shots);
        // and the same circuit with its TreePlan (default cap): replay, full battery
        unsetenv("QEC_TREEPLAN_CAP");
        auto full = noiseless_invariants(lbm, nullptr, 12345, 500);
        CHECK(full.ok);
        CHECK(full.shot_work == 0);
        CHECK(full.shots == 500);
        setenv("QEC_TREEPLAN_CAP", "2", 1);
    }

    // (4) one sequence; detector verdict == the full battery's; continuation on a gauge detector
    {
        protoref::LoadedBench lbm = refcompile::bench_from_text(kMagicPEGauge);
        CHECK(lbm.ok);
        int continued = 0, at_sized = 0;
        for (uint64_t seed = 0; seed < 300; ++seed) {
            auto ib = noiseless_invariants(lbm, nullptr, seed, 500);
            ProgPtr prog = compile_clean(lbm);
            PackedRecords full = sample_program(*prog, 500, seed);
            InvariantsTally t;
            for (int s = 0; s < 500; ++s) t.add(full, s);
            CHECK((ib.det_const_nonzero == 0) == (t.const_one == 0));   // the full verdict
            CHECK(ib.ok);
            // the battery's shots are a prefix of this run: re-stream to ib.shots and compare
            ProgPtr prog2 = compile_clean(lbm);
            const int want = ib.shots;
            const ShotHook stop = [want](int s, const PackedRecords&) { return s + 1 < want; };
            PackedRecords pre = sample_program_streamed(*prog2, 500, seed, &stop);
            CHECK(pre.shots == want);
            CHECK(digest(pre, want) == digest(full, want));
            if (ib.shots > ib.sized_shots) ++continued;
            else ++at_sized;
        }
        CHECK(continued >= 1);
        CHECK(at_sized >= 1);
        std::printf("sequence: %d continued past the sized count, %d stopped at it\n", continued,
                    at_sized);
    }

    // (5) the tally on synthetic records
    {
        InvariantsTally a;
        PackedRecords r = synthetic({1.0, -1.0, 1.0, 0.5}, {0, 1, 0, 0});
        for (int s = 0; s < 3; ++s) a.add(r, s);
        CHECK(a.ok());                      // |exp| constant, detector flips
        a.add(r, 3);
        CHECK(!a.ok());                     // |exp| wobble 0.5
        CHECK(a.max_dev() == 0.5);
        InvariantsTally b;
        PackedRecords q = synthetic({1.0, 1.0, 1.0}, {1, 1, 1});
        for (int s = 0; s < 3; ++s) b.add(q, s);
        CHECK(b.const_one == 1);
        CHECK(!b.ok());                     // constant-1 detector
        InvariantsTally c;
        PackedRecords w = synthetic({1.0, 1.0 + 2e-9}, {0, 0});
        c.add(w, 0);
        c.add(w, 1);
        CHECK(!c.ok());                     // the 1e-9 threshold
    }

    // (6) memo
    {
        tree_fail_memo_clear();
        const std::string fresh = sample_bytes(lbm, 300, 7);
        TreeFailMemoStats s1 = tree_fail_memo_stats();
        CHECK(s1.stores == 1);
        CHECK(s1.misses == 1);
        CHECK(s1.hits == 0);                // the channel rebuild never asks the memo
        {
            ProgPtr p(compile_sampler_program(lbm.circuit, lbm.detectors, lbm.observables, nullptr),
                      &delete_sampler_program);
            const auto& sp = sampler_program_framed(*p)->shot_sampler();
            CHECK(sp.zero_plan_status() == FramedPauliShotSampler::ZeroPlan::FailedMemo);
            CHECK(sp.channel_status() ==
                  FramedPauliShotSampler::ChannelPlan::SkippedZeroPlanFailed);
        }
        const std::string again = sample_bytes(lbm, 300, 7);
        TreeFailMemoStats s2 = tree_fail_memo_stats();
        CHECK(s2.hits == 2);
        CHECK(s2.misses == 1);
        CHECK(again == fresh);
        // credit: W_compile is the same with the memo warm and cold
        auto warm = noiseless_invariants(lbm, nullptr, 3, 500);
        tree_fail_memo_clear();
        auto cold = noiseless_invariants(lbm, nullptr, 3, 500);
        CHECK_EQ(warm.compile_work, cold.compile_work);
        CHECK_EQ(warm.shots, cold.shots);
        tree_fail_memo_clear();
        CHECK(sample_bytes(lbm, 300, 7) == fresh);
        {
            ProgPtr p(compile_sampler_program(lbm.circuit, lbm.detectors, lbm.observables, nullptr),
                      &delete_sampler_program);
            CHECK(sampler_program_framed(*p)->shot_sampler().zero_plan_status() ==
                  FramedPauliShotSampler::ZeroPlan::FailedMemo);
        }
        // different cap => different key: the default-cap extraction runs (and succeeds)
        unsetenv("QEC_TREEPLAN_CAP");
        const TreeFailMemoStats s3 = tree_fail_memo_stats();
        (void)sample_bytes(lbm, 300, 7);
        const TreeFailMemoStats s4 = tree_fail_memo_stats();
        CHECK(s4.hits == s3.hits);
        CHECK(s4.misses == s3.misses + 2);
        CHECK(s4.stores == s3.stores);
    }

    // (7) key completeness
    {
        FramedBareState bs = build_bare_state_framed(lbm.deferred);
        CHECK(!bs.rejected);
        const FramedSuperposition& B = bs.state;
        const std::vector<std::pair<int, int>> reads{{2, 0}, {2, 1}, {2, 2}};
        const std::string k0 = tree_fail_memo_key(B, reads, 2);
        CHECK(k0 == tree_fail_memo_key(B, reads, 2));
        CHECK(k0 != tree_fail_memo_key(B, reads, 3));
        CHECK(k0 != tree_fail_memo_key(B, {{2, 0}, {2, 1}}, 2));
        CHECK(k0 != tree_fail_memo_key(B, {{0, 0}, {2, 1}, {2, 2}}, 2));
        { FramedSuperposition M = B; M.eps[0] ^= 1; CHECK(k0 != tree_fail_memo_key(M, reads, 2)); }
        {
            FramedSuperposition M = B;
            CHECK(!M.entries().empty());
            M.entries()[0].second *= -1.0;
            CHECK(k0 != tree_fail_memo_key(M, reads, 2));
        }
        {
            FramedSuperposition M = B;
            CHECK(!M.entries().empty() && !M.entries()[0].first.empty());
            M.entries()[0].first[0] ^= 1;
            CHECK(k0 != tree_fail_memo_key(M, reads, 2));
        }
        { FramedSuperposition M = B; M.U.Zrow[1].flipx(0); CHECK(k0 != tree_fail_memo_key(M, reads, 2)); }
        { FramedSuperposition M = B; M.U.Xrow[2].flipz(1); CHECK(k0 != tree_fail_memo_key(M, reads, 2)); }
        { FramedSuperposition M = B; M.U.Zrow[0].phase ^= 2; CHECK(k0 != tree_fail_memo_key(M, reads, 2)); }
        { FramedSuperposition M = B; M.free.push_back(3); CHECK(k0 != tree_fail_memo_key(M, reads, 2)); }
    }

    REPORT();
}
