// protocol_reference — shared benchmark-loading machinery for the reference pipeline.
//
// HISTORY: this translation unit once hosted the sequential-FRONTIER reference constructor
// (design log §4.10, Task C) — a per-gate split/merge walk of the deferred circuit that
// ballooned χ and ran super-linearly (~50 min at n=1023). It was fully superseded by the
// parity-native (C-4) build_bare_state in normal_form.cpp and removed (C4-Int 7); the parity
// build accepts the same in-class circuits directly (incl. H-on-magic) and rejects out-of-class
// circuits up front in build_pauli_rotation_form, so no frontier fallback is needed.
//
// WHAT REMAINS: the shared `protoref::` benchmark loader — load_bench (parse + reference
// normalization preset: coherentize magic-crossing feedback, strip remaining feedback, defer,
// eliminate Hadamards) and strip_noise. ref_compile.cpp #includes this TU (with
// REF_COMPILE_NO_MAIN's sibling guard) for LoadedBench / load_bench / strip_noise and, transitively,
// the library headers below; test_ref_io.cpp uses the same loader. The include block is kept intact
// because those consumers depend on it transitively.
#include <algorithm>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "qeccore/canonical_stab_sum.hpp"
#include "qeccore/circuit_ir.hpp"
#include "qeccore/ref_io.hpp"
#include "qeccore/deferral.hpp"
#include "qeccore/feedback.hpp"   // strip_feedback: the reference is the feedback-free circuit
#include "qeccore/coherentize_feedback.hpp"   // coherentize magic-crossing feedback before stripping
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"   // normalize() consolidated entry point
#include "qeccore/sampler.hpp"
#include "qeccore/stab_affine.hpp"
#include "qeccore/stab_generators.hpp"
#include "qeccore/stim_parse.hpp"

namespace protoref {

using namespace qeccore;
using cd = std::complex<double>;

// ── Benchmark loading ─────────────────────────────────────────────────────────────────────────

struct LoadedBench {
    Circuit circuit;                       // user circuit (with noise)
    Circuit deferred;                      // deferred circuit (supplied-state space)
    DeferralMap map;
    std::vector<std::vector<int>> detectors;
    std::vector<std::vector<int>> observables;   // index-dense
    bool ok = false;
    std::string error;
};

static LoadedBench load_bench(const std::string& path) {
    LoadedBench lb;
    std::ifstream in(path);
    if (!in) { lb.error = "cannot open " + path; return lb; }
    std::stringstream ss;
    ss << in.rdbuf();
    ParsedStim p = parse_stim_circuit(ss.str());
    if (!p.ok()) {
        lb.error = "parse failed: " + (p.errors.empty() ? "?" : p.errors[0].message);
        return lb;
    }
    lb.circuit = std::move(p.circuit);
    // Reference normalization preset (Strip + defer + map + eliminate-Hadamards). Magic-crossing
    // feedback is coherentized into Clifford entanglers FIRST so the reference sees the same circuit
    // the sampler does (mirrors refcompile::bench_from_text). Remaining (non-crossing) feedback does
    // NOT enter the bare/reference state — the sampler handles it as a post-sampling relabel — so
    // strip those controlled-Paulis before building the reference (a no-op when there is no
    // feedback). lb.circuit (consumed downstream by the sampler / strip_noise) keeps the
    // coherentized+stripped form exactly as before; lb.deferred is the eliminate_hadamards'd
    // consumer circuit, produced together with lb.map by ONE normalize call.
    lb.circuit = coherentize_classically_controlled_feedback(lb.circuit);
    if (circuit_has_feedback(lb.circuit)) lb.circuit = strip_feedback(lb.circuit);
    NormalizePolicy ref_policy;
    ref_policy.feedback = NormalizePolicy::Feedback::Strip;
    ref_policy.defer = true;
    ref_policy.want_map = true;
    NormalizeResult nr = normalize(lb.circuit, ref_policy);
    lb.detectors = std::move(p.detectors);
    int max_obs = -1;
    for (auto& kv : p.observables) if (kv.first > max_obs) max_obs = kv.first;
    lb.observables.assign((size_t)(max_obs + 1), {});
    for (auto& kv : p.observables) lb.observables[kv.first] = kv.second;
    lb.deferred = std::move(nr.normalized);
    lb.map = std::move(nr.map);
    lb.ok = true;
    return lb;
}

[[maybe_unused]] static Circuit strip_noise(const Circuit& c) {
    Circuit out;
    out.n = c.n;
    for (const auto& ins : c.stream)
        if (ins.kind != Instr::Kind::Noise) out.stream.push_back(ins);
    return out;
}

}  // namespace protoref

