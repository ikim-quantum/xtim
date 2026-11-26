// exact-out T6 — TEST API (not a public interface): the compiled Born sector table of an S-layer plan, the `cpp`
// backend of the consumer-side frame algebra tests/_frames.py (formerly xtim/frames.py, never exported from `xtim`).
// Moved verbatim from cpp/bindings/xtim_py.cpp; the oracle bodies stay in tests/_frames.py (pure numpy).
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstring>
#include <string>
#include <vector>

#include "qeccore/frame_cosets.hpp"
#include "qeccore/state_budget.hpp"

#include "../bindings/common_py.hpp"

namespace py = pybind11;

namespace {

// The compiled sector table's ABI (3.1.3): exported as _xtim_testapi.FRAME_COSETS_ABI and compared by
// tests/_frames.py against its own FRAME_COSETS_ABI at import — a stale .so (missing symbol or
// another value) is REFUSED by plan_frame_cosets_fast, never silently replaced by the
// pure-Python path.  Bump both together when the binding's signature or output changes.
constexpr int kFrameCosetsAbi = 2;   // 2 = EXP-fix #5: weight within 1e-12 of the oracle (was bit-for-bit); |supp a| <= 64

// 3.1.3 (docs/frame_cosets_cpp.md): the compiled Born SECTOR table of an S-layer plan — the
// `cpp` backend of tests/_frames.plan_frame_cosets_fast.  Inputs are the branch_frames() export's
// arrays (unpacked uint8 bit rows, the numpy layout) and the plan's a-mask; the output is the
// numpy path's list of dicts, bit-for-bit (same sectors, same first-encounter order, same
// fields / dtypes, same `weight` bits).  Refusals map to the numpy path's: ValueError for a
// null-space basis read that is not ±1, AssertionError for an export that is not a full frame.
// Read-only, no engine state; the sampling hot path does not touch this code.
using _u8arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
static py::list py_plan_frame_cosets(
    _u8arr stab_x, _u8arr stab_z, py::array_t<int8_t, py::array::c_style | py::array::forcecast> stab_phase,
    _u8arr destab_x, _u8arr destab_z, py::array_t<int32_t, py::array::c_style | py::array::forcecast> free_rows,
    _u8arr sigma, py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coeff,
    _u8arr branch_x, _u8arr branch_z, _u8arr a_mask) {
    using namespace qeccore::frame_cosets;
    if (stab_x.ndim() != 2 || stab_x.shape(0) != stab_x.shape(1))
        throw py::value_error("plan_frame_cosets: stab_x must be (n, n)");
    const int n = (int)stab_x.shape(0);
    auto need = [&](const py::array& a, int r, int c, const char* name) {
        if (a.ndim() != 2 || a.shape(0) != r || a.shape(1) != c)
            throw py::value_error(std::string("plan_frame_cosets: ") + name + " has the wrong shape");
    };
    need(stab_z, n, n, "stab_z"); need(destab_x, n, n, "destab_x"); need(destab_z, n, n, "destab_z");
    if (stab_phase.ndim() != 1 || stab_phase.shape(0) != n) throw py::value_error("plan_frame_cosets: stab_phase must be (n,)");
    if (free_rows.ndim() != 1) throw py::value_error("plan_frame_cosets: free must be (k,)");
    const int k = (int)free_rows.shape(0);
    if (coeff.ndim() != 1) throw py::value_error("plan_frame_cosets: coeff must be (chi,)");
    const int chi = (int)coeff.shape(0);
    if (sigma.ndim() != 2 || sigma.shape(0) != chi || sigma.shape(1) != k)
        throw py::value_error("plan_frame_cosets: sigma must be (chi, k)");
    need(branch_x, chi, n, "branch_x"); need(branch_z, chi, n, "branch_z");
    if (a_mask.ndim() != 1 || a_mask.shape(0) != n) throw py::value_error("plan_frame_cosets: a_mask must be (n,)");
    std::vector<int> fr(k);
    for (int d = 0; d < k; ++d) {
        fr[d] = (int)free_rows.data()[d];
        if (fr[d] < 0 || fr[d] >= n) throw py::value_error("plan_frame_cosets: free row out of range");
    }
    // χ-scaled transient (branch Pauli rows + sigma + coeff): charged to the state budget first.
    qeccore::StateBudgetTicket frame_ticket(
        (int64_t)chi * (int64_t)(2 * 8 * FrameRef::words(n) + k + 16), "plan_frame_cosets' packed branch table");
    const FrameRef f = FrameRef::from_bits(n, stab_x.data(), stab_z.data(), stab_phase.data(),
                                           destab_x.data(), destab_z.data(), fr.data(), k, sigma.data(),
                                           coeff.data(), chi, branch_x.data(), branch_z.data());
    std::vector<Sector> tab;
    try {
        tab = plan_frame_cosets(f, a_mask.data());
    } catch (const FrameNotFull& e) {
        PyErr_SetString(PyExc_AssertionError, e.what());
        throw py::error_already_set();
    }
    py::list out;
    auto bits = [n](const std::vector<uint8_t>& v) {
        py::array_t<uint8_t> a(n);
        std::memcpy(a.mutable_data(), v.data(), (size_t)n);
        return a;
    };
    for (const Sector& s : tab) {
        py::dict d;
        d["vz"] = bits(s.vz);
        d["fx"] = bits(s.fx);
        d["fz"] = bits(s.fz);
        if (s.has_flip) d["logical_flip"] = py::make_tuple(py::cast(s.xb), py::cast(s.zb));
        else d["logical_flip"] = py::none();
        d["weight"] = py::float_(s.weight);
        d["frame_copy"] = py::bool_(s.frame_copy);
        out.append(std::move(d));
    }
    return out;
}

}  // namespace

void register_frame_cosets(py::module_& m) {
    m.attr("FRAME_COSETS_ABI") = kFrameCosetsAbi;
    mdef(m, "plan_frame_cosets", &py_plan_frame_cosets,
          py::arg("stab_x"), py::arg("stab_z"), py::arg("stab_phase"), py::arg("destab_x"),
          py::arg("destab_z"), py::arg("free"), py::arg("sigma"), py::arg("coeff"),
          py::arg("branch_x"), py::arg("branch_z"), py::arg("a_mask"),
          "3.1.3: the Born SECTOR table of an S-layer plan on a branch_frames() reference "
          "(tests/_frames.plan_frame_cosets_fast's compiled backend; bit-for-bit the numpy "
          "path's list of dicts). docs/frame_cosets_cpp.md.");
}
