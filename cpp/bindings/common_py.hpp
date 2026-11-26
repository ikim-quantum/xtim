// common_py.hpp — J-b1 F: the binding helpers shared by xtim._xtim and xtim._xtim_testapi (the state-budget
// guards, the module-def scope, the packed-array marshallers). Moved verbatim from xtim_py.cpp; each extension
// includes it in ONE translation unit (anonymous namespace).
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "qeccore/bare_state_errors.hpp"
#include "qeccore/state_budget.hpp"
#include "qeccore/stim_parse.hpp"

namespace py = pybind11;
using namespace qeccore;

namespace {

// ── state-budget refusal mid-operation (3.1.10) ──────────────────────────────────────────
// A BareStateCapacityError raised in the middle of an operation that MUTATES a Python-held object
// (a sampler's sample, a state's measurement) can leave that object half-updated. Such an object
// marks itself invalid (budget_invalid_) and every later call on it raises a clear
// XtimCapacityError instead of computing from undefined state. Read-only calls that are refused
// (e.g. BarrierBuffer.materialize, branch_frames) leave their object valid.
template <class Self>
void budget_check_valid(const Self& self) {
    if (!self.budget_invalid_.empty())
        throw qeccore::BareStateCapacityError(
            std::string("this xtim ") + Self::kBudgetWhat + " is no longer usable: " +
            self.budget_invalid_ + " Build a new one (raise XTIM_MAX_STATE_BYTES first if needed).");
}
template <class Self, class F>
auto budget_poison_on_refusal(Self& self, F&& f) -> decltype(f()) {
    budget_check_valid(self);
    try {
        return f();
    } catch (const qeccore::BareStateCapacityError& e) {
        self.budget_invalid_ =
            std::string("an earlier call was refused by the state-memory budget in the middle of "
                        "an operation that mutates it, so its internal state is undefined (") +
            e.what() + ").";
        throw qeccore::BareStateCapacityError(
            std::string(e.what()) + " This xtim " + Self::kBudgetWhat +
            " was refused mid-operation and is now INVALID: further calls on it raise "
            "XtimCapacityError; build a new one.");
    }
}
// Member-function adaptor for pybind: const methods check validity, non-const methods poison the
// object on a mid-operation refusal.
template <auto M> struct BudgetGuard;
template <class R, class C, class... A, R (C::*M)(A...) const>
struct BudgetGuard<M> {
    static R call(const C& self, A... a) {
        budget_check_valid(self);
        return (self.*M)(std::forward<A>(a)...);
    }
};
template <class R, class C, class... A, R (C::*M)(A...)>
struct BudgetGuard<M> {
    static R call(C& self, A... a) {
        return budget_poison_on_refusal(self, [&]() -> R { return (self.*M)(std::forward<A>(a)...); });
    }
};

// A numpy array whose buffer is charged to the state budget for as long as the array (or any
// view of it) is alive: a capsule owns the ticket and the data. Used for the χ-scaled exports.
template <class T>
py::array_t<T> budget_array(const std::vector<py::ssize_t>& shape, const char* what) {
    size_t n = 1;
    for (py::ssize_t d : shape) n *= (size_t)d;
    struct Holder { qeccore::StateBudgetTicket t; std::unique_ptr<T[]> p; };
    auto* h = new Holder();
    try {
        h->t.charge((int64_t)(n * sizeof(T)), what);      // refuses before allocating
        h->p.reset(new T[n ? n : 1]());
    } catch (...) {
        delete h;
        throw;
    }
    py::capsule cap(h, [](void* v) { delete static_cast<Holder*>(v); });
    return py::array_t<T>(shape, h->p.get(), cap);
}

// Module-level def with the scratch call scope (the automatic per-thread scratch release when no
// Python-held owner is left; see state_budget.hpp).
template <typename Func, typename... Extra>
void mdef(py::module_& m, const char* name, Func&& f, const Extra&... extra) {
    m.def(name, std::forward<Func>(f), extra..., py::call_guard<qeccore::ScratchCallScope>());
}


inline py::array_t<uint8_t> packed_to_numpy(const std::vector<uint8_t>& v, int shots, int bytes) {
    py::array_t<uint8_t> a({shots, bytes});
    if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size());
    return a;
}

inline py::array_t<double> exps_to_numpy(const std::vector<double>& v, int shots, int r) {
    py::array_t<double> a({shots, r});
    if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
    return a;
}

}  // namespace
