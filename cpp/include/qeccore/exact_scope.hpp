// exact_scope.hpp — J-b1 P13 (plan r3.2, r3.1-fix R3-5): the runtime exact-construction counter.
// Every exact-engine entry class bumps a process-global counter when it is constructed, unless an
// EXPLICIT-EXACT SCOPE is open on the constructing thread. Explicit entry points open the scope (RAII here; a
// Python context manager, xtim._exact_scope.explicit_exact, around the public explicit calls). The counter has
// no effect on any output. After F a session-level pytest fixture asserts it stays 0 (no silent exact route).
// Counted classes: SamplerProgram (compile_sampler_program: the CompiledDetectorSampler / CompiledSampler
// backing, and the forced-fault program inside ExactBranchProgram), FramedCircuitShotSampler,
// ExactBranchProgram, the DEM export program (export_dem) and detector_determinism.
#pragma once

namespace qeccore {

// Bump the counter unless an explicit-exact scope is open on this thread.
void note_exact_construction();
long exact_constructions();
void reset_exact_constructions();
void explicit_exact_enter();
void explicit_exact_exit();

struct ExplicitExactScope {
    ExplicitExactScope() { explicit_exact_enter(); }
    ~ExplicitExactScope() { explicit_exact_exit(); }
    ExplicitExactScope(const ExplicitExactScope&) = delete;
    ExplicitExactScope& operator=(const ExplicitExactScope&) = delete;
};

}  // namespace qeccore
