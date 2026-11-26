#include "qeccore/exact_scope.hpp"

#include <atomic>
#include <stdexcept>

namespace qeccore {
namespace {
std::atomic<long> g_exact_constructions{0};
thread_local int t_explicit_depth = 0;
}  // namespace

void note_exact_construction() {
    if (t_explicit_depth == 0) g_exact_constructions.fetch_add(1, std::memory_order_relaxed);
}
long exact_constructions() { return g_exact_constructions.load(std::memory_order_relaxed); }
void reset_exact_constructions() { g_exact_constructions.store(0, std::memory_order_relaxed); }
void explicit_exact_enter() { ++t_explicit_depth; }
void explicit_exact_exit() {
    if (t_explicit_depth <= 0) throw std::logic_error("explicit_exact_exit without a matching enter");
    --t_explicit_depth;
}

}  // namespace qeccore
