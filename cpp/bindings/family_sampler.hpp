// family_sampler.hpp — J-b1 F: THE twirl sampler binding (the record-family law; plan §2.2 item 1). Moved from
// cpp/testapi/family_py.cpp (the J-a / J-b1 test API, where it was built and gated pre-flip) and shared by both
// extensions: xtim._xtim binds PyFamilySampler as TwirlSampler (every record class; nothing switches it), and
// xtim._xtim_testapi as FamilySampler (compile_family_sampler; J-b2 J6 deleted the per-class bisection knob),
// plus the test-only accessors under XTIM_FAMILY_TESTAPI). One binding source, so F-B1a (NEW == FAMP) compares one
// code path's two entry points (plan r3.3). Each extension includes this header in ONE translation unit.
//
// The private FAM surface is reached through TwirlSamplerBindingAccess (public members) and, in the test API only,
// TwirlSamplerTestAccess.
#pragma once

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/amplitudes.hpp"
#include "qeccore/branch_table.hpp"
#include "qeccore/carried_compose.hpp"
#include "qeccore/exact_scope.hpp"   // MUT-INJ-FALLBACK (mutation build only)
#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/record_fidelity.hpp"
#include "qeccore/ref_io.hpp"
#include "qeccore/sampler.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/term_residual.hpp"
#include "qeccore/twirl_sampler.hpp"
#include "common_py.hpp"
#include "injection.hpp"
#include "record_api.hpp"
#include "record_buffer.hpp"
#include "refusal_text.hpp"
#include "twirl_binding_access.hpp"
#include "twirl_front.hpp"
#ifdef XTIM_FAMILY_TESTAPI
#include "../testapi/twirl_test_access.hpp"
#endif

namespace py = pybind11;
using namespace qeccore;

namespace {

// termroute-a m3: ONE meaning for every cache-budget argument (sampler plan_payload_budget, set_plan_payload_budget,
// RecordFidelity cache_budget): None = the default (kDefaultCacheBudget), math.inf = unbounded (engine 0), a positive
// int = that many bytes; anything else (0, negatives, other floats) is refused by name.
inline size_t cache_budget_arg(const py::object& v, const char* what) {
    if (v.is_none()) return kDefaultCacheBudget;
    if (py::isinstance<py::float_>(v)) {
        const double d = v.cast<double>();
        if (std::isinf(d) && d > 0) return 0;
        throw std::invalid_argument(std::string(what) + " must be None (the 512 MiB default), a positive byte count "
                                    "or math.inf (unbounded); " + py::repr(v).cast<std::string>() + " is refused");
    }
    const long long b = v.cast<long long>();
    if (b <= 0)
        throw std::invalid_argument(std::string(what) + " must be None (the 512 MiB default), a positive byte count "
                                    "or math.inf (unbounded); a budget of " + std::to_string(b) +
                                    " bytes is refused");
    return (size_t)b;
}
// A budget as reported (record_report / RecordFidelity.stats): engine 0 = unbounded = math.inf.
inline py::object cache_budget_out(size_t b) {
    return b ? py::object(py::int_(b)) : py::object(py::float_(std::numeric_limits<double>::infinity()));
}

using BA_ = qeccore::TwirlSamplerBindingAccess;
#ifdef XTIM_FAMILY_TESTAPI
using TA_ = qeccore::TwirlSamplerTestAccess;
#endif


py::array_t<uint8_t> packed(const std::vector<uint8_t>& v, long rows, int bpr) {
    py::array_t<uint8_t> a({(py::ssize_t)rows, (py::ssize_t)bpr});
    if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size());
    return a;
}


// ── J-a (N2): Pauli / term-set conversions for the term-route test API ──
py::tuple pauli_xz(const Pauli& P) {
    py::array_t<uint8_t> x((py::ssize_t)P.n), z((py::ssize_t)P.n);
    for (int q = 0; q < P.n; ++q) {
        x.mutable_data()[q] = P.xbit(q) ? 1 : 0;
        z.mutable_data()[q] = P.zbit(q) ? 1 : 0;
    }
    return py::make_tuple(x, z);
}
inline Pauli pauli_of(int n, const std::vector<uint8_t>& x, const std::vector<uint8_t>& z) {
    if ((int)x.size() != n || (int)z.size() != n) throw std::invalid_argument("x / z must have length n");
    Pauli P(n);
    for (int q = 0; q < n; ++q) { if (x[(size_t)q]) P.setx(q); if (z[(size_t)q]) P.setz(q); }
    return P;
}
std::vector<int> set_list(const std::vector<uint64_t>& S, int T) {
    std::vector<int> out;
    for (int j = 0; j < T; ++j) if ((S[(size_t)(j >> 6)] >> (j & 63)) & 1) out.push_back(j);
    return out;
}
std::vector<uint64_t> set_words(const std::vector<int>& S, int T) {
    std::vector<uint64_t> w((size_t)((T + 63) / 64), 0);
    for (int j : S) {
        if (j < 0 || j >= T) throw std::invalid_argument("term index out of range");
        w[(size_t)(j >> 6)] ^= 1ULL << (j & 63);
    }
    return w;
}
inline py::bytes words_bytes(const std::vector<uint64_t>& w) {
    return py::bytes(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(uint64_t));
}
inline py::dict nf_dict(const DiagNormalForm& nf) {
    py::dict d;
    d["prefix"] = pauli_xz(nf.prefix);
    std::vector<int> a;
    for (size_t q = 0; q < nf.a.size(); ++q) if (nf.a[q] & 1) a.push_back((int)q);
    d["a"] = a;
    d["cz"] = nf.cz;
    return d;
}
inline py::tuple term_axes(const TermResidual& tr) {
    const int T = tr.num_terms(), n = tr.n();
    py::array_t<uint8_t> xz({(py::ssize_t)T, (py::ssize_t)(2 * n)});
    py::array_t<int32_t> ph((py::ssize_t)T);
    for (int j = 0; j < T; ++j) {
        const Pauli& B = tr.axis(j);
        for (int q = 0; q < n; ++q) {
            xz.mutable_data()[(size_t)j * 2 * n + q] = B.xbit(q) ? 1 : 0;
            xz.mutable_data()[(size_t)j * 2 * n + n + q] = B.zbit(q) ? 1 : 0;
        }
        ph.mutable_data()[j] = B.phase;
    }
    return py::make_tuple(xz, ph);
}
inline py::list kernel_list(const TermResidual& tr) {
    py::list out;
    for (const auto& kv : tr.kernel())
        out.append(py::make_tuple(kv.j, set_list(kv.k, tr.num_terms()), pauli_xz(kv.N)));
    return out;
}

// A hand-built term set (signed Hermitian axes): G-TK2 (the kernel fold) and G-TK6 (the assert).
struct PyTermSet {
    std::unique_ptr<TermResidual> tr;
    PyTermSet(int n, py::array_t<uint8_t> xz, std::vector<int> phase) {
        const auto buf = xz.unchecked<2>();
        if (buf.shape(1) != 2 * n || (size_t)buf.shape(0) != phase.size())
            throw std::invalid_argument("TermSet: xz must be (T, 2n) with one phase per axis");
        std::vector<Pauli> axes;
        for (py::ssize_t j = 0; j < buf.shape(0); ++j) {
            Pauli B(n);
            for (int q = 0; q < n; ++q) { if (buf(j, q)) B.setx(q); if (buf(j, n + q)) B.setz(q); }
            B.phase = phase[(size_t)j] & 3;
            axes.push_back(B);
        }
        tr.reset(new TermResidual(n, axes));
    }
    py::tuple run(const std::vector<int>& S, const std::vector<int>* seq) const {
        TermShot sh;
        tr->init_shot(sh);
        tr->clear_shot(sh);
        sh.S = set_words(S, tr->num_terms());
        if (seq) tr->fold_sequence_(*seq, sh);
        else tr->reduce_key_(sh);
        return py::make_tuple(pauli_xz(sh.P), set_list(sh.S, tr->num_terms()));
    }
};

// J-a N5: a state in the branch_frames() export format (xtim.frames.dense_state), the bare_frames layout.
py::dict state_frames(const FramedSuperposition& st) {
    const int N = st.n(), K = st.k();
    const auto& brs = st.entries();
    const int chi = (int)brs.size();
    auto bits2d = [](int rows, int cols_) {
        py::array_t<uint8_t> a({rows, cols_});
        std::memset(a.mutable_data(), 0, (size_t)rows * (size_t)cols_);
        return a;
    };
    auto fill = [N](py::array_t<uint8_t>& ax, py::array_t<uint8_t>& az, int r, const Pauli& p) {
        for (int q = 0; q < N; ++q) {
            ax.mutable_data()[(size_t)r * N + q] = p.xbit(q);
            az.mutable_data()[(size_t)r * N + q] = p.zbit(q);
        }
    };
    py::array_t<uint8_t> sx = bits2d(N, N), sz = bits2d(N, N), dx = bits2d(N, N), dz = bits2d(N, N);
    py::array_t<int8_t> sp(N), dp(N);
    py::array_t<uint8_t> eps(N);
    for (int a = 0; a < N; ++a) {
        Pauli g = st.U.Zrow[a];
        if (st.eps[a]) g.phase = (g.phase + 2) & 3;
        fill(sx, sz, a, g);
        sp.mutable_data()[a] = (int8_t)(g.phase & 3);
        fill(dx, dz, a, st.U.Xrow[a]);
        dp.mutable_data()[a] = (int8_t)(st.U.Xrow[a].phase & 3);
        eps.mutable_data()[a] = st.eps[a] ? 1 : 0;
    }
    py::array_t<int32_t> fr(K);
    for (int d = 0; d < K; ++d) fr.mutable_data()[d] = st.free[(size_t)d];
    py::array_t<uint8_t> sig = bits2d(chi, K), bx = bits2d(chi, N), bz = bits2d(chi, N);
    py::array_t<std::complex<double>> co(chi);
    py::array_t<int8_t> bp(chi);
    for (int i = 0; i < chi; ++i) {
        Pauli D(N);
        for (int d = 0; d < K; ++d) {
            sig.mutable_data()[(size_t)i * K + d] = key_bit(brs[(size_t)i].first, d);
            if (key_bit(brs[(size_t)i].first, d)) D = Pauli::multiply(D, st.U.Xrow[(size_t)st.free[(size_t)d]]);
        }
        fill(bx, bz, i, D);
        bp.mutable_data()[i] = (int8_t)(D.phase & 3);
        co.mutable_data()[i] = brs[(size_t)i].second;
    }
    py::dict out;
    out["n"] = N; out["free"] = fr; out["eps"] = eps;
    out["stab_x"] = sx; out["stab_z"] = sz; out["stab_phase"] = sp;
    out["destab_x"] = dx; out["destab_z"] = dz; out["destab_phase"] = dp;
    out["sigma"] = sig; out["coeff"] = co;
    out["branch_x"] = bx; out["branch_z"] = bz; out["branch_phase"] = bp;
    return out;
}

Pauli pauli_from(int n, const py::handle& t) {
    const py::tuple tp = py::reinterpret_borrow<py::object>(t).cast<py::tuple>();
    if (tp.size() != 3) throw std::invalid_argument("a Pauli is (x bits, z bits, phase)");
    const std::vector<uint8_t> x = tp[0].cast<std::vector<uint8_t>>(), z = tp[1].cast<std::vector<uint8_t>>();
    if ((int)x.size() != n || (int)z.size() != n) throw std::invalid_argument("a Pauli on the wrong number of qubits");
    Pauli P(n);
    for (int q = 0; q < n; ++q) { if (x[(size_t)q]) P.setx(q); if (z[(size_t)q]) P.setz(q); }
    P.phase = tp[2].cast<int>() & 3;
    return P;
}

// J-a N5 / J-b1 P12: the layout-3 records of one FAM run — xtim_api::LayoutRecords (cpp/bindings/record_buffer.hpp).
using PyFamilyRecords = xtim_api::LayoutRecords;

// J-b1 P9a: an ideal state — the frame-explicit carried state of a circuit (its bare state with the OUTPUT_QUBITS
// wires) and its frame token. Exported by a producer FAM sampler (ideal_state(): producer token, the producer kept
// alive for branch rebuilds) or built from text / rows (rows-only token: branch 0 only).
struct PyIdealState {
    FramedSuperposition fs{0};
    std::vector<int> wires;
    FrameToken tok;
    py::object producer;                            // the producing FamilySampler (None: no provenance)
};

// J-b1 P9a: state-level injection of a carried state into a consumer's deferred circuit (production's Task 3.0 route,
// cpp/bindings/injection.hpp): bare = carried ⊗ the consumer's fresh block, the port-touching Cliffords baked, the
// split certified against the all-|0> oracle. Remaps deferred / reads / ports into the combined space.
// J-b1 P9b (XQ9 (A), ruling on fixplan-p9b-xq9): a consumer whose non-Clifford gates touch carried wires is no longer
// refused by check_no_magic_on_carried: its bare is composed on the carried free register exactly
// (compose_carried_bare: merged odd axes, the span criterion + gauge, F rows as FastTODD variables). The span
// criterion's failure is the named law limitation LAW_LIMITATION_CARRIED_AXIS / carried_free_x. Consumers without
// magic on carried wires keep the tensor + port-Clifford route unchanged.
#ifdef XTIM_MUTATE_INJ_FALLBACK
// mutation build only (MUT-INJ-FALLBACK): a carried-axis / injection refusal first takes a silent exact route (an
// exact construction outside any explicit scope); the counter tests must turn red.
static void mut_inj_fallback_() { qeccore::note_exact_construction(); }
#else
static inline void mut_inj_fallback_() {}
#endif
struct InjectInfo {
    std::string route = "inject";                   // "inject" (tensor + port Cliffords) or "compose" (XQ9 (A))
    std::string cls, sub;                           // a typed refusal (XtimTwirlRefusal)
    int axes = 0, gauge_moves = 0, free_rows = 0;
};
static std::string inject_carried_(const FramedSuperposition& carried, const std::vector<int>& src_wires,
                                   Circuit& deferred, std::vector<std::pair<int, int>>& reads, std::vector<int>& ports,
                                   std::unique_ptr<FramedSuperposition>& bare_out, InjectInfo* info = nullptr,
                                   std::vector<int>* map_out = nullptr) {
    // J-b1 P9b: the consumer's own bare with the FAM ladder settlement (as build_bare_state_framed_family), so a
    // consumer the family builder accepts in-stage is not refused here by the even-Clifford ladder gap.
    BareState cbs = build_bare_state_parity(deferred, AssemblePath::coset_family, nullptr, /*settle_ladder=*/true);
    if (cbs.rejected) {                             // J-b1 final review MAJOR-3: typed (T-depth > 1 consumer)
        if (info) { info->cls = "OUTSIDE_T_DEPTH_1"; info->sub = "bare"; }
        return "bare state rejected: " + cbs.reject_reason;
    }
    if (ports.size() != src_wires.size())
        return "state-level injection: consumer INPUT_QUBITS declares " + std::to_string(ports.size()) +
               " port qubit(s) but the ideal state carries " + std::to_string(src_wires.size());
    const int N_in = carried.n();
    for (int w : src_wires) if (w < 0 || w >= N_in) return "state-level injection: a carried wire is out of range";
    const int Nc = deferred.n;
    std::vector<uint8_t> is_port((size_t)(Nc > 0 ? Nc : 1), 0);
    std::vector<int> map((size_t)Nc, -1);
    for (size_t j = 0; j < ports.size(); ++j) {
        const int pw = ports[j];
        if (pw < 0 || pw >= Nc) return "port qubit out of deferred range";
        map[(size_t)pw] = src_wires[j];
        is_port[(size_t)pw] = 1;
    }
    for (int w = 0; w < Nc; ++w) if (map[(size_t)w] < 0) map[(size_t)w] = N_in + w;
    std::string err = check_no_magic_on_carried(deferred, is_port, Nc);
    const int combined_n = N_in + Nc;
#ifndef XTIM_MUTATE_XQ9_KEEP_CHECK
    if (!err.empty()) {                             // XQ9 (A): the composed build on the carried free register
        FramedSuperposition S_in = tensor_framed(carried, zero_state(Nc));
        remap_circuit_wires(deferred, map, combined_n);
        CarriedComposeResult cr = compose_carried_bare(S_in, deferred);
        if (info) { info->route = "compose"; info->axes = cr.axes; info->gauge_moves = cr.gauge_moves;
                    info->free_rows = (int)carried.free.size(); }
        if (cr.rejected) {
            if (cr.span_refusal) {
                std::string term;
                for (int q = 0; q < combined_n; ++q) {
                    const bool x = cr.span_term.xbit(q), z = cr.span_term.zbit(q);
                    if (x || z) term += std::string(x && z ? "Y" : x ? "X" : "Z") + "[w" + std::to_string(q) + "]";
                }
                if (info) { info->cls = "LAW_LIMITATION_CARRIED_AXIS"; info->sub = "carried_free_x"; }
                return "state-level injection: law limitation (carried axis, carried_free_x): a consumer "
                       "non-Clifford axis (" + term + ") flips a carried amplitude coordinate (the ideal state's "
                       "free row " + std::to_string(cr.span_row) + "); the carried-branch law reduces phase "
                       "polynomials over carried amplitudes only. Served after J-c (register Clifford); refused";
            }
            if (info) { info->cls = "OUTSIDE_T_DEPTH_1"; info->sub = "composed"; }
            return "state-level injection: the composed consumer was rejected: " + cr.reason;
        }
        bare_out.reset(new FramedSuperposition(std::move(cr.state)));
        for (auto& r : reads) r.second = map[(size_t)r.second];
        for (int& p : ports) p = map[(size_t)p];
        if (map_out) *map_out = map;
        return "";
    }
#else
    if (!err.empty()) return err;                   // mutation build only (MUT-XQ9-KEEP-CHECK): the old refusal
#endif
    Circuit fresh_sub = fresh_only_subcircuit(deferred, is_port, Nc);
    // J-b1 final review MAJOR-3: the fresh block takes the SAME P5 ladder-gap settle as the standalone family build
    // (build_bare_state_framed_family): the plain build first, and only on the ladder-gap rejection the settled
    // rebuild — so every consumer served before builds identically, and a T-depth-1 ladder-gap block is served.
    BareState fbs = build_bare_state(fresh_sub);
#ifndef XTIM_MUTATE_INJ_NOSETTLE
    if (fbs.rejected && fbs.reject_reason == "rotation-form terms anticommute (even-Clifford ladder gap)")
        fbs = build_bare_state_parity(fresh_sub, AssemblePath::coset_family, nullptr, /*settle_ladder=*/true);
#endif
    if (fbs.rejected) {
        if (info) { info->cls = "OUTSIDE_T_DEPTH_1"; info->sub = "bare"; }
        return "state-level injection: fresh magic block rejected: " + fbs.reject_reason;
    }
    FramedSuperposition freshblock = FramedSuperposition::from_css(fbs.state);
    bare_out.reset(new FramedSuperposition(tensor_framed(carried, freshblock)));
    remap_circuit_wires(deferred, map, combined_n);
    std::vector<uint8_t> is_port_comb((size_t)combined_n, 0);
    for (int sw : src_wires) if (sw >= 0 && sw < combined_n) is_port_comb[(size_t)sw] = 1;
    err = apply_stream_cliffords_ports(*bare_out, deferred, is_port_comb);
    if (!err.empty()) return err;
    FramedSuperposition chk = tensor_framed(zero_state(N_in), freshblock);
    err = apply_stream_cliffords_ports(chk, deferred, is_port_comb);
    if (!err.empty()) return err;
    FramedSuperposition oracle = FramedSuperposition::from_css(cbs.state);
    FramedSuperposition oracle_mapped = remap_state_wires(oracle, map, combined_n);
    if (std::abs(std::abs(exact_sum_overlap(chk, oracle_mapped)) - 1.0) > 1e-9)
        return "state-level injection: the fresh/port composition failed its exactness guard; refused";
    for (auto& r : reads) r.second = map[(size_t)r.second];
    for (int& p : ports) p = map[(size_t)p];
    if (map_out) *map_out = map;
    return "";
}

// S8-2a (plan r4.1 M3): the one-thread-at-a-time contract, as a mechanism. A sampler and its plan cache each carry an
// atomic busy flag, try-acquired by compare_exchange and released by this RAII hold; a failed acquire refuses by
// name (XT_SAMPLER_BUSY / XT_CACHE_BUSY) and nothing waits. Scopes:
//   - the public runs (sample, sample_records_core_ — sample_records / sample_barrier) hold the sampler flag and the
//     cache flag for the whole call, taken before any effect on the record sink, eng_ib_ / cls_ib_ or the engine's
//     input branch (final review M1: a refused concurrent call must not touch the running call's state); run_core_
//     (the GIL release inside) takes none. The E-class loop holds each inner sampler's own two flags (its own
//     cache: built by its own ctor) around the inner run_core_: no self-refusal;
//   - the run's Python warnings (drain_) are emitted inside the flag scope: a warnings / logging hook that calls back
//     into the same sampler is refused by its own XT_SAMPLER_BUSY (by name; final review m4);
//   - every GIL-held entry that reads or writes sampler / cache state holds the sampler flag (the record readers and
//     RecordFidelity.evaluate also the cache flag) for its own scope; entries reading only construction constants
//     (num_detectors, output_wires, ...) take none;
//   - the test API's PlanCacheHandle holds the cache flag;
//   - unguarded by design (they read construction constants or are bench-only): the _bench_* test entries,
//     branch_table_digest, born_dec, record_reads. A later entry that reads MUTABLE sampler / cache state must take
//     the flag (final review m5);
//   - the constructor: no API hands an existing cache to a new sampler (every ctor builds its own cache, published
//     only when the ctor returns), so its GIL-released build cannot meet a shared cache and takes no flag.
struct BusyHold {
    std::atomic<bool>* f = nullptr;
    BusyHold(std::atomic<bool>& a, const char* code, const char* what, const char* who) {
        bool expect = false;
#ifndef XTIM_MUTATE_BUSY_NOFLAG
        if (!a.compare_exchange_strong(expect, true, std::memory_order_acquire))
            throw std::runtime_error(std::string(code) + ": " + who + " refused — another thread is using this " + what +
                                     " (one thread at a time per " + what + "; nothing waits)");
#else
        (void)expect; (void)code; (void)what; (void)who;    // mutation build only (BUSY-NOFLAG): never refuses
#endif
        f = &a;
    }
    BusyHold(const BusyHold&) = delete;
    BusyHold& operator=(const BusyHold&) = delete;
    ~BusyHold() {
#ifndef XTIM_MUTATE_BUSY_LEAK
        f->store(false, std::memory_order_release);
#endif
    }
};

// J-b1 P11a TEST API: a plan cache kept alive past its sampler (shared ownership).
struct PyCacheHandle {
    std::shared_ptr<TwirlPlanCache> c;
};

struct PyFamilySampler {
    std::unique_ptr<FramedSuperposition> bare_;
    // J-b1 F (fixplan-F-materialize-c0hat r2 item 3): per-sampler invariants, computed once. The state model key is
    // fixed once the constructor returns (the bare, the generators, the record operators with their folds and the
    // port Born reference are construction outputs); a process-unique serial lets a reader skip the bit-exact key
    // comparison for records this very sampler produced (any other records are compared bit for bit, as before).
    const uint64_t serial_ = next_serial_();
    mutable std::string model_key_;
    mutable bool model_ready_ = false;
    mutable bool armed_ = false;                    // arm_ ran (the term view and its TWPL token are set once)
    mutable std::atomic<bool> busy_{false};         // S8-2a: the sampler flag (BusyHold above)
    BusyHold hold_(const char* who) const { return BusyHold(busy_, "XT_SAMPLER_BUSY", "sampler", who); }
    BusyHold hold_cache_(const char* who) const {
        return BusyHold(BA_::family_plan_cache(*s_)->binding_busy, "XT_CACHE_BUSY", "plan cache", who);
    }
    static uint64_t next_serial_() {               // J-b1 final review m1: salted per extension (prod even, test odd)
        static std::atomic<uint64_t> c{0};
#ifdef XTIM_FAMILY_TESTAPI
        return 2 * (c.fetch_add(1, std::memory_order_relaxed) + 1) + 1;
#else
        return 2 * (c.fetch_add(1, std::memory_order_relaxed) + 1);
#endif
    }
    const std::string& model_key() const {
        if (!model_ready_) { model_key_ = BA_::family_model_key(*s_); model_ready_ = true; }
        return model_key_;
    }
    std::string text_;                              // J-b1 P9a: the circuit text (the producer frame token)
    std::vector<int> out_wires_;                    // J-b1 P9a: OUTPUT_QUBITS → deferred wires (declaration order)
    std::shared_ptr<BranchTable> btab_;             // J-b1 P9a: the consumer's branch table (ideal-state samplers)
    std::string inject_route_;                      // J-b1 P9b: "inject" / "compose" (ideal-state samplers)
    int inject_axes_ = 0, inject_gauge_ = 0;
    py::object producer_;                           // J-b1 P9a: the ideal state's producer (None: rows-only)
    std::vector<FramedSuperposition> bvec_;         // J-b1 P9a: branch id − 1 → the branch vector (u = 0 sector)
    // J-b1 P9b: the consumer branch law. The consumer's construction inputs before injection (a branch's consumer
    // bare is the same pure construction on the branch vector) and, per branch id − 1, its engine branch id (0: an
    // E_min != {} branch, served by the per-E law).
    Circuit cons_def0_;
    std::vector<std::pair<int, int>> cons_reads0_;
    std::vector<int> cons_ports0_, cons_wires_;
    std::vector<int32_t> beng_;
    std::vector<int32_t> eng_ib_;                   // the run's per-shot engine branch ids (empty: branch 0 only)
    std::vector<int32_t> cls_ib_;                   // the run's per-shot per-E engine branch ids (E-class shots)
    // J-b1 P9b: the per-E law. One per-E consumer per distinct E_min: a PyFamilySampler of the same text and options
    // whose ideal state is the first branch of the class on the per-E frame (free = F₀ ∪ E, the E rows VARIABLES);
    // its plans are keyed by this E (its own plan table); every branch of the class is an engine branch of it.
    struct EClass {
        std::vector<int> rows;
        std::unique_ptr<PyFamilySampler> s;
        // The per-E consumer reports its DETECTOR / OBSERVABLE columns relative to ITS bare's reference; this
        // sampler's columns are relative to the ideal's (A1 §2): per column, ref_inner ⊕ ref_ideal (DECISIONs raw).
        std::vector<uint8_t> dmask, omask;
    };
    std::vector<EClass> eclass_;
    std::vector<int32_t> bclass_;                   // branch id − 1 → its E-class + 1 (0: E_min = {})
    // the construction options (a per-E consumer repeats them)
    double ctor_pf_ = 1.0;
    long ctor_sc_ = 0;
    bool ctor_oas_ = false, ctor_dec_ = true;
    uint64_t ctor_seed_ = 1;
    bool recording_ = false;                        // sample_records is running (a per-E row cannot be recorded)
    long branches_with_delta_ = 0;                  // registered branches whose eps delta D != I (non-vacuity)
    long branches_with_port_delta_ = 0;             // ... whose delta reaches a port wire (observable by the consumer)
    Circuit deferred_;
    std::vector<std::pair<int, int>> reads_;
    // RF-ref: per deferred wire, the measurement whose read wire it is (−1 for none) — the strict refusal's diagnostic
    std::vector<int> read_meas_() const {
        std::vector<int> v((size_t)(bare_ ? bare_->n() : 0), -1);
        for (size_t r = 0; r < reads_.size(); ++r)
            if (reads_[r].second >= 0 && reads_[r].second < (int)v.size()) v[(size_t)reads_[r].second] = (int)r;
        return v;
    }
    // Q4 (README stage): per deferred wire, the user qubit it carries (−1: internal) — the strict refusal names qubits
    // by it. Output wires come from final_wire_ (exact under injection too); other wires from the deferral's origin
    // map (cleared on an injected compile, whose wires it does not index).
    std::vector<int> origin_user_;
    std::vector<int> wire_user_() const {
        std::vector<int> v((size_t)(bare_ ? bare_->n() : 0), -1);
        for (size_t d = 0; d < origin_user_.size() && d < v.size(); ++d) v[d] = origin_user_[d];
        for (size_t q = 0; q < final_wire_.size(); ++q)
            if (final_wire_[q] >= 0 && final_wire_[q] < (int)v.size()) v[(size_t)final_wire_[q]] = (int)q;
        return v;
    }
    std::vector<std::vector<int>> detectors_;
    std::vector<std::pair<int, std::vector<int>>> obs_list_;
    TwirlRecordOptions opt_;
    std::unique_ptr<TwirlRecordSampler> s_;
    int num_det_ = 0, num_obs_ = 0, num_dec_ = 0;
    py::list disk_events_;                          // J-b1 P11a: the DISK_LOAD / DISK_SAVE events, kept for the tests
    // J-b1 F: production's binding state (budget guard, OUTPUT_QUBITS map, the feedback-hoist count, the module that
    // built this object — a BarrierBuffer's evaluator comes from it, plan r3.3).
    qeccore::ScratchOwner scratch_owner_;           // Python-held owner (automatic scratch release)
    std::string budget_invalid_;                    // set by a mid-operation budget refusal
    static constexpr const char* kBudgetWhat = "TwirlSampler";
    std::vector<int> final_wire_;
    std::vector<OutputPort> outputs_;
    int feedback_hoisted_ = 0;
    static const char* module_name_() {
#ifdef XTIM_FAMILY_TESTAPI
        return "xtim._xtim_testapi";
#else
        return "xtim._xtim";
#endif
    }
    // S0d (diagnostics policy, design §5.0), as production: ONE drain-emit-outcome step per call, the GIL held,
    // never from the shot loop. The call's events go to xtim._diagnostics.emit_twirl (warnings / logging); the
    // DISK_LOAD / DISK_SAVE ones are also kept for the tests (take_disk_events). `abort` (non-empty): the call's
    // outcome is RuntimeError(abort) raised from the first warning; `collect`: a C++ exception is propagating (it
    // stays the outcome), every event is emitted and nothing is raised.
    std::vector<TwirlRecordSampler::DiagEvent> pending_;   // drain_run_ -> drain_ (events already taken)
    void drain_(const std::string& abort = std::string(), bool collect = false) {
        if (!s_) {
            if (!abort.empty() && !collect) throw std::runtime_error(abort);
            return;
        }
        std::vector<TwirlRecordSampler::DiagEvent> ev = std::move(pending_);
        pending_.clear();
        for (auto& e : s_->take_diagnostics()) ev.push_back(std::move(e));
        if (ev.empty()) {
            if (!abort.empty() && !collect) throw std::runtime_error(abort);
            return;
        }
        static const char* const kKind[] = {"exact_fallback", "selfcheck_window", "disk_load",
                                            "oracle_failure", "born_node", "disk_save", "channels"};
        py::list events;
        for (const auto& e : ev) {
            if (e.kind == TwirlRecordSampler::DiagEvent::DISK_LOAD || e.kind == TwirlRecordSampler::DiagEvent::DISK_SAVE)
                disk_events_.append(py::make_tuple(e.kind == TwirlRecordSampler::DiagEvent::DISK_LOAD ? "load" : "save",
                                                   e.name, e.count));
            events.append(py::make_tuple(kKind[e.kind], e.name, e.count, e.shots, e.detail, e.value));
        }
        py::object ab = abort.empty() ? py::object(py::none()) : py::object(py::str(abort));
        py::module_::import("xtim._diagnostics").attr("emit_twirl")(events, collect ? py::object(py::none()) : ab,
                                                                     collect);
    }

    PyFamilySampler(const std::string& text, double p_factor, long selfcheck,
                    bool oracle_all_shots, bool decisions, uint64_t seed,
                    bool record_route_term = false, const PyIdealState* ideal = nullptr,
                    const std::string& disk_path = std::string(), const std::string& disk_auto_dir = std::string())
        : text_(text) {
        ctor_pf_ = p_factor; ctor_sc_ = selfcheck; ctor_oas_ = oracle_all_shots;
        ctor_dec_ = decisions; ctor_seed_ = seed;      // J-b1 P9b: a per-E consumer repeats them
        opt_.disk_path = disk_path;                 // J-b1 P11a: TWPL (a FAM cache reads / writes v6)
        opt_.disk_auto_dir = disk_auto_dir;         // J-b1 F: production's automatic cache dir (xtim.twirl resolves it)
        std::string err;
        std::string twirl_cls, twirl_sub;           // J-b1 P13: a typed refusal (xtim.errors.XtimTwirlRefusal)
        {
            py::gil_scoped_release rel;
            ParsedStim ps = parse_stim_circuit(text);
            if (!ps.ok()) {
                err = "parse failed";
                for (const auto& e : ps.errors) err += "\n  line " + std::to_string(e.line) + ": " + e.message;
            } else {
                for (const InputPort& p : ps.circuit.inputs)
                    if (p.kind == InputPort::Kind::Qubits) { opt_.input_pauli_qubits = p.qubits; break; }
#ifdef XTIM_MUTATE_MUTRF_FEEDBACK_TWICE
                // mutation build only (MUT-RF): the readout flips read off the USER circuit, so a
                // feedback-driving noisy record (folded into the residual by coherentize) is flipped twice
                for (const Instr& ins : ps.circuit.stream)
                    if (ins.kind == Instr::Kind::Measure) opt_.rec_flip.push_back(ins.readout_flip_p);
#endif
                NormalizeResult nr = normalize_for_twirl_sampler(ps.circuit);
                for (const Instr& ins : nr.coherent.stream)
                    if (ins.kind == Instr::Kind::Measure) {
#ifndef XTIM_MUTATE_MUTRF_FEEDBACK_TWICE
                        opt_.rec_flip.push_back(ins.readout_flip_p);
#endif
                        opt_.rec_invert.push_back(ins.invert ? 1 : 0);
                    }
                {
                    size_t j = 0;
                    for (const Instr& ins : nr.normalized.stream)
                        if (ins.kind == Instr::Kind::Measure) {
                            if (j < opt_.rec_invert.size() && ins.invert) opt_.rec_invert[j] ^= 1;
                            ++j;
                        }
                }
                deferred_ = std::move(nr.normalized);
                for (const auto& r : nr.map.terminal_reads) reads_.push_back({(int)r.second, r.first});
                // J-b1 F: the OUTPUT_QUBITS wire map (production's final_wire_ / outputs_; output_wires(),
                // output_ports(), the ideal-state export and branches() read it). An ideal-state compile remaps it
                // through the injection map below (production's Task 3.0 rule; the P9a factory had omitted it).
                final_wire_ = nr.map.final_wire;
                origin_user_.clear();
                for (const WireOrigin& o : nr.map.wire_origin)
                    origin_user_.push_back(o.wire >= 0 && o.wire < ps.num_user_qubits ? o.wire : -1);
                outputs_ = ps.circuit.outputs;
                feedback_hoisted_ = nr.feedback_hoisted;
                detectors_ = std::move(ps.detectors);
                obs_list_.assign(ps.observables.begin(), ps.observables.end());
                num_det_ = (int)detectors_.size();
                if (decisions) opt_.decisions = std::move(ps.decisions);
                for (const auto& kv : opt_.decisions) if (kv.first + 1 > num_dec_) num_dec_ = kv.first + 1;
                int mx = -1;
                for (const auto& kv : obs_list_) if (kv.first > mx) mx = kv.first;
                num_obs_ = mx + 1;
                FramedBareState bs;
                if (ideal) {                        // J-b1 P9a: the ideal state injected as branch 0
                    cons_def0_ = deferred_;         // J-b1 P9b: the branch law rebuilds per branch from these
                    cons_reads0_ = reads_;
                    cons_ports0_ = opt_.input_pauli_qubits;
                    cons_wires_ = ideal->wires;
                    origin_user_.clear();           // the injected wires are re-indexed: outputs only
                    std::unique_ptr<FramedSuperposition> composed;
                    InjectInfo ii;
                    std::vector<int> imap;
                    const std::string ierr = inject_carried_(ideal->fs, ideal->wires, deferred_, reads_,
                                                             opt_.input_pauli_qubits, composed, &ii, &imap);
                    if (ierr.empty())
                        for (int& fw : final_wire_)
                            if (fw >= 0 && fw < (int)imap.size()) fw = imap[(size_t)fw];
                    inject_route_ = ii.route; inject_axes_ = ii.axes; inject_gauge_ = ii.gauge_moves;
                    if (!ii.cls.empty()) { twirl_cls = ii.cls; twirl_sub = ii.sub; }
                    if (!ierr.empty()) { bs.rejected = true; bs.reject_reason = ierr; }
                    else { bs.rejected = false; bs.state = std::move(*composed); }
                } else {
                    bs = build_bare_state_framed_family(deferred_);   // J-b1 P5: ladder gap settled
                }
                if (bs.rejected && ideal) {
                    mut_inj_fallback_();
                    err = bs.reject_reason;          // an injection refusal (by name)
                } else if (bs.rejected) {
                    // J-b1 P13: OUTSIDE_T_DEPTH_1 (bare), the axes in user qubits (the production text's renderer)
                    err = "bare state rejected: " + bs.reject_reason + user_terms_text(bs.reject_terms, nr, ps) +
                          "; the circuit is outside xtim's model (its non-Clifford gates do not fold into one "
                          "commuting layer of pi/8 rotations)";
                    twirl_cls = "OUTSIDE_T_DEPTH_1";
                    twirl_sub = "bare";
                } else {
                    bare_.reset(new FramedSuperposition(std::move(bs.state)));
                    opt_.p_factor = p_factor;
                    opt_.circuit_channels = true;
                    opt_.selfcheck = selfcheck;
                    opt_.oracle_all_shots = oracle_all_shots;
                    opt_.fam_record_route_term = record_route_term;   // J-b1 P4: the all-term oracle route
                    // compile stage F-C: the normalization's in-class table of deferred_ (plain path only:
                    // an injected ideal state rewrites deferred_, so the table is never passed there)
                    s_ = TwirlRecordSampler::make(*bare_, deferred_, reads_, detectors_, obs_list_, seed, opt_,
                                                  ideal ? std::optional<PropagationTable>() : std::move(nr.class_table));
                    if (s_->setup_error() != 0)
                        err = !BA_::family_refusal(*s_).empty()
                                  ? BA_::family_refusal(*s_)
                                  : "family sampler setup failed (code " + std::to_string(s_->setup_error()) + ")";
                    if (s_->setup_error() == 3) {   // TS:119 (the propagation table, rebuilt off the hot path)
                        // J-b1 rulings on fixplan-p5-c14 (A): typed by the owner's static rule. The bare state is
                        // accepted here, so the circuit is T-depth <= 1 noiselessly; the rule either accepts it
                        // (LAW_LIMITATION_PROPAGATION: the gate-by-gate table meets a same-axis non-Clifford pair
                        // before it merges; served after J-c by the merged-op-list table) or refuses it by its own
                        // law limitation (NONCOMMUTING_MAGIC, the propagation table refusing first).
                        const PropagationTable t = build_propagation_table(deferred_, /*ppr_retry=*/true);
                        const StaticRuleVerdict v = family_static_rule(deferred_, reads_, p_factor);
                        std::string where;
                        if (!t.rejects.empty()) {
                            const RejectInfo& r = t.rejects.front();
                            const std::string b = r.basis == PauliBasis::X ? "X" : r.basis == PauliBasis::Y ? "Y" : "Z";
                            where = "a " + b + " fault at deferred noise site " + std::to_string(r.location_index) +
                                    " on deferred wire " + std::to_string(r.qubit) + user_wire_text(r.qubit, nr, ps) +
                                    " leaves the gate-by-gate propagation table's diagonal + Pauli class at deferred "
                                    "gate " + std::to_string(r.reject_gate_index);
                        }
                        if (v.accept) {
                            err = "twirl sampler: law limitation (propagation): the static rule accepts this circuit "
                                  "(T-depth <= 1), but " + where + ": a fault meets a same-axis non-Clifford pair "
                                  "before it merges; a law limitation of the twirl engine (not physics): the circuit "
                                  "is outside xtim's model today";
                            twirl_cls = "LAW_LIMITATION_PROPAGATION";
                        } else {
                            std::vector<int> site_wires;
                            if (v.site >= 0 && v.site < (int)deferred_.stream.size())
                                site_wires = deferred_.stream[(size_t)v.site].qubits;
                            err = v.message + user_terms_text(v.axes, nr, ps, site_wires) +
                                  " (the propagation table refused it first: " + where + "). It is outside xtim's "
                                  "model";
                            twirl_cls = "NONCOMMUTING_MAGIC";
                        }
                    } else if (s_->setup_error() == 7 && BA_::family_refusal_info(*s_).cls == 1) {
                        const auto info = BA_::family_refusal_info(*s_);   // J-b1 P13: NONCOMMUTING_MAGIC
                        std::vector<int> site_wires;
                        if (info.site >= 0 && info.site < (int)deferred_.stream.size())
                            site_wires = deferred_.stream[(size_t)info.site].qubits;
                        err += user_terms_text(info.axes, nr, ps, site_wires);
                        err += ". It is outside xtim's model";
                        twirl_cls = "NONCOMMUTING_MAGIC";
                    }
                }
            }
        }
        if (err.empty() && s_ && s_->born_obs_index() >= 0)   // J-b1 P10: a port circuit's Born-observable reference
            for (const auto& kv : obs_list_)
                if (kv.first == s_->born_obs_index()) { BA_::family_port_born_ref(*s_, kv.second); break; }
        if (!err.empty()) drain_(std::string(), /*collect=*/true);   // S0d: the construction's events, then its refusal
        if (!err.empty() && !twirl_cls.empty()) {   // J-b1 P13: xtim.errors.XtimTwirlRefusal(message, class, sub)
            py::object exc = py::module_::import("xtim.errors").attr("XtimTwirlRefusal")(err, twirl_cls, twirl_sub);
            PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(exc.ptr())), exc.ptr());
            throw py::error_already_set();
        }
        if (!err.empty()) throw std::invalid_argument(err);
        for (const OutputPort& op : outputs_)          // J-b1 P9a / F: OUTPUT_QUBITS -> deferred wires (declaration order)
            for (int q : op.qubits)
                if (q >= 0 && q < (int)final_wire_.size()) out_wires_.push_back(final_wire_[(size_t)q]);
        drain_();
        if (inject_route_ == "compose")             // J-b1 P9b (XQ9 (A)): ONE INFO event per composed consumer
            py::module_::import("xtim._diagnostics").attr("twirl_log").attr("info")(
                "[twirl_records] carried composition: the consumer's non-Clifford gates on carried wires are "
                "composed on the carried free register (%d merged axes, %d gauge row moves)", inject_axes_,
                inject_gauge_);
    }

    // ── J-b1 P9a: the branch identity layer ──
    static py::bytes dbytes_(const Digest128& d) { return py::bytes(reinterpret_cast<const char*>(&d), sizeof d); }
    static Digest128 dfrom_(const py::object& b) {
        const std::string s = b.cast<std::string>();
        if (s.size() != sizeof(Digest128)) throw std::invalid_argument("a 128-bit digest must be 16 bytes");
        Digest128 d;
        std::memcpy(&d, s.data(), sizeof d);
        return d;
    }
    // J-b1 P9b (P9a-1, review C5): the ideal state is the POST-barrier state — the identity plan's u = 0, z = 0
    // retained state at the noiseless reference, materialize(identity content, P = I, y = base) — on the
    // post-barrier frame every producer branch vector lives on (design Lemma R2′). The frame token hashes it.
    mutable std::unique_ptr<FramedSuperposition> ideal_fs_;
    const FramedSuperposition& ideal_fs() const {
        if (!ideal_fs_) {
            arm_();
            const int n = bare_->n();
            std::string key((size_t)n, '\0');
            key.push_back((char)0xFF);
            const int h = BA_::family_intern_content(*s_, 0, key);
            const auto T = BA_::family_content_table(*s_, h);
            std::vector<uint64_t> y(T->base.begin(), T->base.end());
            if (y.empty()) y.push_back(0);
            std::vector<uint64_t> prec((size_t)2 * ((n + 63) / 64), 0);
            ideal_fs_.reset(new FramedSuperposition(BA_::family_materialize(*s_, h, prec.data(), y.data())));
        }
        return *ideal_fs_;
    }
    FrameToken own_token_() const { return frame_token_producer(text_, ideal_fs()); }
    // The producer's handles of its layout-3 records: (frame_token, the content's 128-bit digest, the Born word z).
    py::list branch_handles(const PyFamilyRecords& R) const {
        same_model(R);
        for (long i = 0; i < R.rows; ++i)            // J-b1 P9b (ruling on fixplan-p9b-frame): 3-stage re-export
            if (R.branch[(size_t)i] != 0) {
                py::object exc = py::module_::import("xtim.errors").attr("XtimTwirlRefusal")(
                    "branch_handles: law limitation (carried axis, branch_frame): record " + std::to_string(i) +
                    " is a carried-branch row of this consumer (input_branch != 0); re-exporting it to a third "
                    "stage needs its frame on this consumer's ideal frame. Served after J-c; refused",
                    "LAW_LIMITATION_CARRIED_AXIS", "branch_frame");
                PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(exc.ptr())), exc.ptr());
                throw py::error_already_set();
            }
        const FrameToken tok = own_token_();
        arm_();
        py::list out;
        for (long i = 0; i < R.rows; ++i) {
            const auto& C = R.content_of(i);
            const int h = BA_::family_intern_content(*s_, C.kind, C.content);
            const auto T = BA_::family_content_table(*s_, h);
            const int nuz = R.nuz[(size_t)i];
            if (nuz != T->rho + T->kappa) throw std::logic_error("branch_handles: outcomes != rho + kappa (engine bug)");
            std::string z;
            for (int j = 0; j < T->kappa; ++j) z.push_back((char)(R.uz[(size_t)i * R.sink.UZW + T->rho + j] & 1));
            py::dict d;
            d["token"] = dbytes_(tok.d);
            Digest128 dg; dg.hi = C.digest_hi; dg.lo = C.digest_lo;
            d["digest"] = dbytes_(dg);
            d["z"] = py::bytes(z);
            out.append(d);
        }
        return out;
    }
    // J-b1 P9b (P9a-2, design §2.10): per record, its content handle and the shot's Pauli frame P_rec · W(u)
    // (W(u) = Π_p W_p^{u_p}, u = the first ρ outcomes) restricted to the OUTPUT_QUBITS wires in declaration order:
    // (handles, frames uint8[rows, 2, n_out], X row then Z row). The consumer passes the frames as input_pauli.
    py::tuple branches(const PyFamilyRecords& R) const {
        py::list H = branch_handles(R);
        const int no = (int)out_wires_.size();
        py::array_t<uint8_t> fr({(py::ssize_t)R.rows, (py::ssize_t)2, (py::ssize_t)no});
        if (R.rows && no) std::memset(fr.mutable_data(), 0, (size_t)R.rows * 2 * (size_t)no);
        const std::vector<int32_t> h = local_handles(R);
        const int n = bare_->n();
        for (long i = 0; i < R.rows; ++i) {
            Pauli F = prec_pauli_row_(R, i, n);
#ifndef XTIM_MUTATE_WU_DROP
            const auto& W = BA_::family_pivot_frames(*s_, h[(size_t)i]);
            for (size_t p = 0; p < W.size(); ++p)
                if (R.uz[(size_t)i * R.sink.UZW + p] & 1) F = Pauli::multiply(F, W[p]);
#endif
            for (int j = 0; j < no; ++j) {
#ifdef XTIM_MUTATE_WU_COLS
                const int w = out_wires_[(size_t)(no - 1 - j)];   // mutation build only: the columns reversed
#else
                const int w = out_wires_[(size_t)j];
#endif
                fr.mutable_data()[((size_t)i * 2 + 0) * no + j] = F.xbit(w);
                fr.mutable_data()[((size_t)i * 2 + 1) * no + j] = F.zbit(w);
            }
        }
        return py::make_tuple(H, fr);
    }
    static Pauli prec_pauli_row_(const PyFamilyRecords& R, long i, int n) {
        Pauli P(n);
        const uint64_t* p = R.prec_row(i);
        for (int q = 0; q < n; ++q) {
            if ((p[q >> 6] >> (q & 63)) & 1ULL) P.setx(q);
            if ((p[R.sink.PNW + (q >> 6)] >> (q & 63)) & 1ULL) P.setz(q);
        }
        return P;
    }
    // The producer's branch vector of a handle (the u = 0 sector: P = I, pivots at y = 0, introducers at z) on the
    // post-barrier frame, its full tagged content key and its canonical E_min (the rows of the post-barrier register
    // outside the bare's free set on which the vector has support in both sectors).
    // J-b1 P9b (ruling on fixplan-p9b-frame, path (R)): the branch's materialize frame against the ideal post-barrier
    // frame, EXACTLY: the stabilizer rows' X/Z bits on every row, and the destabilizer rows (X/Z bits and phase) on
    // every row free in either state (the rows the amplitudes are keyed on). The rows that differ (ascending); empty
    // ⇔ the branch vector lives on the ideal frame. Zrow phases are signs and belong with eps (not the frame).
    static std::vector<int> frame_divergence_(const FramedSuperposition& id, const FramedSuperposition& br) {
        std::vector<int> rows;
        const int n = id.n();
        if (br.n() != n) throw std::logic_error("frame_divergence_: register width differs (engine bug)");
        std::vector<uint8_t> fr((size_t)(n > 0 ? n : 1), 0);
        for (int f : id.free) fr[(size_t)f] = 1;
        for (int f : br.free) fr[(size_t)f] = 1;
        for (int a = 0; a < n; ++a) {
            const Pauli& zi = id.U.Zrow[(size_t)a];
            const Pauli& zb = br.U.Zrow[(size_t)a];
#ifdef XTIM_MUTATE_FRAME_CHECK_OFF
            (void)zi; (void)zb;                          // mutation build only (MUT-FRAME-CHECK-OFF): no frame check
            continue;
#endif
            bool diff = zi.x != zb.x || zi.z != zb.z;
            if (!diff && fr[(size_t)a]) {
                const Pauli& xi = id.U.Xrow[(size_t)a];
                const Pauli& xb = br.U.Xrow[(size_t)a];
                diff = xi.x != xb.x || xi.z != xb.z || ((xi.phase - xb.phase) & 3) != 0;
            }
            if (diff) rows.push_back(a);
        }
        return rows;
    }
    FramedSuperposition branch_vector_(const BranchHandle& h, std::string& full_key, std::vector<int>& emin,
                                       std::vector<int>* frame_rows = nullptr) const {
        arm_();
        const auto& contents = BA_::family_record_contents(*s_);
        int found = -1;
        for (size_t c = 0; c < contents.size(); ++c)
            if (contents[c].digest_hi == h.digest.hi && contents[c].digest_lo == h.digest.lo) {
                if (found >= 0) throw DigestCollision("branch_vector: two producer contents share one digest (refused)");
                found = (int)c;
            }
        if (found < 0)
            throw BranchRefusal("add_branch: the producer has no content with the handle's digest (digest mismatch) — "
                                "refused");
        const auto& C = contents[(size_t)found];
        full_key = tagged_plan_key(C.kind, C.content);
        const auto T = BA_::family_content_table(*s_, found);
        if ((int)h.z.size() != T->kappa) throw BranchRefusal("add_branch: the handle's z has the wrong length — refused");
        const int YW = (int)T->base.size();
        std::vector<uint64_t> y(T->base.begin(), T->base.end());
        for (int j = 0; j < T->kappa; ++j)
            if (h.z[(size_t)j] & 1) for (int w = 0; w < YW; ++w) y[(size_t)w] ^= T->born_row[(size_t)j][(size_t)w];
        const int PNW = (bare_->n() + 63) / 64;
        std::vector<uint64_t> prec((size_t)2 * PNW, 0);
        FramedSuperposition st = BA_::family_materialize(*s_, found, prec.data(), y.data());
        // The frame check (fixplan-p9b-frame ruling (R)): a branch whose frame differs from the ideal's is REFUSED by
        // the caller before E_min is read, so E_min, distinct_emin and the table digest only ever cover frame-equal
        // branches (on which the row index means the same generator in both frames).
        if (frame_rows) *frame_rows = frame_divergence_(ideal_fs(), st);
        if (frame_rows && !frame_rows->empty()) { emin.clear(); return st; }
        // E_min against the POST-barrier F₀ (the ideal state's free rows; review C5 (b)), on the ideal frame.
        std::vector<uint8_t> in_f0((size_t)st.n(), 0);
        for (int f : ideal_fs().free) in_f0[(size_t)f] = 1;
        emin.clear();
        for (int d = 0; d < (int)st.free.size(); ++d) {
            const int row = st.free[(size_t)d];
            if (in_f0[(size_t)row]) continue;
            bool s0 = false, s1 = false;
            for (const auto& e : st.entries()) ((e.first >> d) & 1ULL ? s1 : s0) = true;
            if (s0 && s1) emin.push_back(row);
        }
        std::sort(emin.begin(), emin.end());
        return st;
    }
    int add_branch(const py::dict& hd, const py::object& vector) {
        if (!btab_) throw std::invalid_argument("add_branch: not an ideal-state sampler (compile_family_sampler_from_"
                                                "ideal_state)");
        if (!vector.is_none())
            throw std::invalid_argument("add_branch: an explicit branch vector is supplied at P9b (consumer branch law); "
                                        "pass vector=None (the producer rebuilds it) — refused");
        BranchHandle h;
        h.token = dfrom_(hd["token"]);
        h.digest = dfrom_(hd["digest"]);
        const std::string z = hd["z"].cast<std::string>();
        h.z.assign(z.begin(), z.end());
        if (producer_.is_none()) return btab_->add(h, std::string(), {});   // rows-only: refused by name
        const PyFamilySampler& P = producer_.cast<const PyFamilySampler&>();
        std::string key;
        std::vector<int> emin, frows;
        FramedSuperposition v = P.branch_vector_(h, key, emin, &frows);
        if (!frows.empty()) {   // J-b1 P9b (fixplan-p9b-frame ruling): refused BY NAME, before any table entry
            std::string rl;
            for (size_t j = 0; j < frows.size() && j < 8; ++j) rl += (j ? ", " : "") + std::to_string(frows[j]);
            if (frows.size() > 8) rl += ", ...";
            const std::string msg =
                "add_branch: law limitation (carried axis, branch_frame): a producer coin read hits a dressed (E_S) "
                "row, so this branch's frame differs from the ideal post-barrier frame (frame rows " + rl + "); its "
                "E_min and law are not defined on the ideal frame. Served after J-c (per-plan frame alignment); "
                "refused";
            mut_inj_fallback_();
            py::object exc = py::module_::import("xtim.errors").attr("XtimTwirlRefusal")(
                msg, "LAW_LIMITATION_CARRIED_AXIS", "branch_frame");
            PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(exc.ptr())), exc.ptr());
            throw py::error_already_set();
        }
        // J-b1 P9b: the branch's consumer law input, built EAGERLY here (never inside run(): every per-E step and
        // guard, the span criterion included). The branch is put on its frame — the ideal's rows with the free
        // register F₀ (E_min = {}) or F₀ ∪ E_min (the E rows VARIABLES, never constants) — its constant promoted rows
        // folded, its certified sign differences split off as the Pauli D (on the ports a frames row, elsewhere the
        // shot's P); its consumer bare by the construction of branch 0 (E_min = {}: this sampler, plans shared) or of
        // its E-class's per-E consumer (E_min != {}: that consumer's own plan table).
#ifdef XTIM_MUTATE_XQ9_LAZY
        if (!emin.empty()) {                           // mutation build only (MUT-XQ9-LAZY): the per-E steps deferred
            const int lid = btab_->add(h, key, emin);
            if (lid > (int)bvec_.size()) { bvec_.push_back(std::move(v)); beng_.push_back(0); bclass_.push_back(0); }
            return lid;
        }
#endif
        BranchOnFrame bf = branch_on_frame_(P, v, emin);
        const FramedSuperposition& idf = P.ideal_fs();
        FramedSuperposition& vp = bf.vp;
        Pauli& D = bf.D;
#ifdef XTIM_MUTATE_BR_NO_DELTA
        D = Pauli(idf.n());                            // mutation build only (MUT-BR-NO-DELTA): the eps delta dropped
#endif
        const int pnq = (int)cons_wires_.size();
        std::vector<uint8_t> dport((size_t)(2 * pnq), 0);
        Pauli dspec(bare_->n());
        {
            std::vector<uint8_t> isport((size_t)(idf.n() > 0 ? idf.n() : 1), 0);
            for (int j = 0; j < pnq; ++j) {
                const int w = cons_wires_[(size_t)j];
                isport[(size_t)w] = 1;
                dport[(size_t)j] = D.xbit(w);
                dport[(size_t)(pnq + j)] = D.zbit(w);
            }
            for (int w = 0; w < idf.n(); ++w) {
                if (isport[(size_t)w]) continue;
                if (D.xbit(w)) dspec.setx(w);
                if (D.zbit(w)) dspec.setz(w);
            }
        }
        PyFamilySampler* host = this;
        int cls_ix = 0;
        std::unique_ptr<PyFamilySampler> fresh_host;
        if (!emin.empty()) {
            for (size_t k = 0; k < eclass_.size(); ++k)
                if (eclass_[k].rows == emin) { host = eclass_[k].s.get(); cls_ix = (int)k + 1; break; }
            if (!cls_ix) {                             // a new E-class: its per-E consumer (built here, eagerly)
                PyIdealState ie;
                ie.fs = vp;
                ie.wires = cons_wires_;
                ie.tok = frame_token_rows(vp);
                ie.producer = py::none();
                try {
                    fresh_host.reset(new PyFamilySampler(text_, ctor_pf_, ctor_sc_, ctor_oas_,
                                                         ctor_dec_, ctor_seed_, false, &ie));
                } catch (py::error_already_set& e) {
                    if (e.matches(py::module_::import("xtim.errors").attr("XtimTwirlRefusal")) &&
                        py::str(e.value().attr("sub_reason")).cast<std::string>() == "carried_free_x")
                        refuse_carried_branch_x_(emin, py::str(e.value()).cast<std::string>());
                    throw;
                }
                host = fresh_host.get();
            }
        }
        std::unique_ptr<FramedSuperposition> cb;
        {
            InjectInfo ii;
            const std::string err = consumer_bare_(*host, vp, cb, &ii);
            if (!err.empty()) {
                if (ii.sub == "carried_free_x" && !emin.empty()) refuse_carried_branch_x_(emin, err);
                throw std::logic_error("add_branch: the branch's consumer construction failed where its frame's "
                                       "first construction passed (engine bug): " + err);
            }
        }
        int id;
#ifdef XTIM_MUTATE_BR_DIGEST
        try { id = btab_->add(h, key, emin); }   // mutation build only (MUT-BR-DIGEST): a collision merges silently
        catch (const DigestCollision&) { id = 1; }
#else
        id = btab_->add(h, key, emin);
#endif
        if (id > (int)bvec_.size()) {
            if (fresh_host) {
                EClass ec;
                ec.rows = emin;
                const auto ri = fresh_host->column_reference_(), ro = column_reference_();
                ec.dmask.assign((size_t)num_det_, 0);
                ec.omask.assign((size_t)num_obs_, 0);
                for (int j = 0; j < num_det_; ++j) ec.dmask[(size_t)j] = ri.first[(size_t)j] ^ ro.first[(size_t)j];
                for (int j = 0; j < num_obs_; ++j) ec.omask[(size_t)j] = ri.second[(size_t)j] ^ ro.second[(size_t)j];
                ec.s = std::move(fresh_host);
                eclass_.push_back(std::move(ec));
                cls_ix = (int)eclass_.size();
                host = eclass_.back().s.get();
            }
            // D6: the payload's re-supply is the SAME pure function (no Python, no refusal: validated above)
            const PyFamilySampler* prod = &P;
            const std::vector<int> em = emin;
            std::function<FramedSuperposition()> resupply = [prod, h, em, host]() {
                std::string k2;
                std::vector<int> e2, fr2;
                FramedSuperposition v2 = prod->branch_vector_(h, k2, e2, &fr2);
                if (!fr2.empty() || e2 != em) throw std::logic_error("branch re-supply: a different branch (engine bug)");
                BranchOnFrame b2 = branch_on_frame_(*prod, v2, e2);
                std::unique_ptr<FramedSuperposition> c2;
                const std::string err = consumer_bare_(*host, b2.vp, c2, nullptr);
                if (!err.empty()) throw std::logic_error("branch re-supply failed (engine bug): " + err);
                return std::move(*c2);
            };
            bvec_.push_back(std::move(v));
            beng_.push_back(BA_::family_add_branch(*host->s_, *cb, dspec, dport, std::move(resupply)));
            bclass_.push_back(cls_ix);
            bool dz = false, dp = false;
            for (size_t w = 0; w < D.x.size(); ++w) dz |= (D.x[w] | D.z[w]) != 0;
            for (uint8_t b : dport) dp |= b != 0;
            branches_with_delta_ += dz;
            branches_with_port_delta_ += dp;
        }
        return id;
    }
    // J-b1 P9b: the branch on its frame (free F₀ (E_min = {}) or F₀ ∪ E_min) and its certified sign delta D — the
    // pure first half of a branch payload (add_branch and the D6 re-supply).
    struct BranchOnFrame { FramedSuperposition vp{0}; Pauli D; };
    static BranchOnFrame branch_on_frame_(const PyFamilySampler& P, FramedSuperposition v, std::vector<int>& emin) {
        const FramedSuperposition& idf = P.ideal_fs();
#ifdef XTIM_MUTATE_E_CONST
        if (!emin.empty()) { v = rows_const_(v, emin); emin.clear(); }   // mutation build only (MUT-E-CONST)
#endif
        std::vector<int> target = idf.free;
        for (int r : emin) target.push_back(r);
        BranchOnFrame o;
        o.D = Pauli(idf.n());
        o.vp = branch_on_ideal_(v, idf, o.D, &target);
        return o;
    }
    // ... and its second half: the consumer bare by `host`'s construction (this sampler, or an E-class's).
    static std::string consumer_bare_(const PyFamilySampler& host, const FramedSuperposition& vp,
                                      std::unique_ptr<FramedSuperposition>& cb, InjectInfo* ii) {
        Circuit d = host.cons_def0_;
        auto r = host.cons_reads0_;
        auto pq = host.cons_ports0_;
        return inject_carried_(vp, host.cons_wires_, d, r, pq, cb, ii);
    }
    [[noreturn]] static void refuse_carried_branch_x_(const std::vector<int>& emin, const std::string& why) {
        mut_inj_fallback_();
        std::string rl;
        for (size_t j = 0; j < emin.size(); ++j) rl += (j ? ", " : "") + std::to_string(emin[j]);
        py::object exc = py::module_::import("xtim.errors").attr("XtimTwirlRefusal")(
            "add_branch: law limitation (carried axis, carried_branch_x): a consumer non-Clifford axis flips a "
            "coordinate of this branch's extended rows (E_min = {" + rl + "}); the carried-branch law reduces phase "
            "polynomials over carried amplitudes only. Served after J-c (register Clifford on the E rows); refused. "
            "(" + why + ")",
            "LAW_LIMITATION_CARRIED_AXIS", "carried_branch_x");
        PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(exc.ptr())), exc.ptr());
        throw py::error_already_set();
    }
#ifdef XTIM_MUTATE_E_CONST
    // mutation build only (MUT-E-CONST): the E rows reduced as constants — the branch projected on one sector of each
    static FramedSuperposition rows_const_(const FramedSuperposition& v, const std::vector<int>& rows) {
        FramedSuperposition o = v;
        std::vector<std::pair<AmpKey, std::complex<double>>> keep;
        AmpKey m = 0, val = 0;
        for (int d = 0; d < v.k(); ++d)
            for (int r : rows) if (v.free[(size_t)d] == r) m |= AmpKey(1) << d;
        bool first = true;
        double nrm = 0;
        for (const auto& e : v.entries()) {
            if (first) { val = e.first & m; first = false; }
            if ((e.first & m) == val) { keep.push_back(e); nrm += std::norm(e.second); }
        }
        for (auto& e : keep) e.second /= std::sqrt(nrm);
        o.entries().assign(keep.begin(), keep.end());
        o.sync_alpha_k();
        return o;
    }
#endif
    // J-b1 P9b: an E_min = {} branch vector on the ideal frame: its promoted rows outside F₀ are constant on its
    // support (E_min = {}) and fold into eps (measure_pauli's χ-drop rule), its free register is re-keyed in the
    // ideal state's free order, and its sign differences from the ideal on certified rows are returned as the Pauli
    // D = Π Xrow_a (the destabilizers flip exactly those signs; v = D·v′ up to a global phase), while sign
    // differences on free rows relabel the amplitudes (α′_x = α_{x ⊕ e}). The result has the ideal's (U, F₀, eps).
    static FramedSuperposition branch_on_ideal_(const FramedSuperposition& v, const FramedSuperposition& id, Pauli& D,
                                                const std::vector<int>* target = nullptr) {
        // `target`: the free rows of the result (default the ideal's F₀; the per-E law passes F₀ then E_min)
        const std::vector<int>& tf = target ? *target : id.free;
        const int n = id.n(), kv = v.k(), ki = (int)tf.size();
        std::vector<int> pos((size_t)(n > 0 ? n : 1), -1);
        for (int t = 0; t < ki; ++t) pos[(size_t)tf[(size_t)t]] = t;
        std::vector<int> tgt((size_t)kv, -1);
        std::vector<uint8_t> eps = v.eps;
        int seen = 0;
        for (int d = 0; d < kv; ++d) {
            tgt[(size_t)d] = pos[(size_t)v.free[(size_t)d]];
            if (tgt[(size_t)d] >= 0) { ++seen; continue; }
            // a promoted row outside F₀: constant on the support (E_min = {}) → a certified sign
            int b = -1;
            for (const auto& e : v.entries()) {
                const int bit = (int)((e.first >> d) & 1ULL);
                if (b < 0) b = bit;
                else if (bit != b) throw std::logic_error("add_branch: a promoted row is not constant (E_min bug)");
            }
            if (b == 1) eps[(size_t)v.free[(size_t)d]] ^= 1;
        }
        if (seen != ki) throw std::logic_error("add_branch: the branch lost an ideal free row (frame check bug)");
        FramedSuperposition out = v;
        out.free = tf;
        std::vector<std::pair<AmpKey, std::complex<double>>> ent;
        for (const auto& e : v.entries()) {
            AmpKey k = 0;
            for (int d = 0; d < kv; ++d)
                if (tgt[(size_t)d] >= 0 && ((e.first >> d) & 1ULL)) k |= AmpKey(1) << tgt[(size_t)d];
            ent.push_back({k, e.second});
        }
        D = Pauli(n);
        std::vector<uint8_t> isfree((size_t)(n > 0 ? n : 1), 0);
        for (int f : tf) isfree[(size_t)f] = 1;
        for (int a = 0; a < n; ++a)
            if (!isfree[(size_t)a] && eps[(size_t)a] != id.eps[(size_t)a]) {
                D = Pauli::multiply(D, id.U.Xrow[(size_t)a]);
                eps[(size_t)a] = id.eps[(size_t)a];
            }
        AmpKey flip = 0;
        for (int t = 0; t < ki; ++t) {
            const int f = tf[(size_t)t];
            if (eps[(size_t)f] != id.eps[(size_t)f]) { flip |= AmpKey(1) << t; eps[(size_t)f] = id.eps[(size_t)f]; }
        }
        for (auto& e : ent) e.first ^= flip;
        out.eps = eps;
        out.entries().assign(ent.begin(), ent.end());
        out.sync_alpha_k();
        return out;
    }
    py::dict branch_counters() const {
        py::dict d;
        d["branches"] = btab_ ? btab_->num_branches() : 1;
        d["distinct_emin"] = btab_ ? btab_->distinct_emin() : 0;
        d["map_probes"] = btab_ ? btab_->map_probes() : 0L;
        d["provenance"] = btab_ ? btab_->token().producer : false;
        const auto c = BA_::family_branch_counters(*s_);   // J-b1 P9b: the engine's branch law
        d["engine_branches"] = c.branches;
        d["branch_shots"] = c.shots;
        d["branch_map_probes"] = c.map_probes;
        d["branch_tries"] = c.tries;
        d["branch_trie_nodes"] = c.trie_nodes;
        d["branch_trie_node_evals"] = c.trie_node_evals;
        d["branches_with_delta"] = branches_with_delta_;
        d["branch_resident"] = c.resident;
        d["branch_evictions"] = c.evictions;
        d["branch_resupplies"] = c.resupplies;
        long ev = 0, rs = 0;
        for (const auto& ec : eclass_) {                // the per-E consumers' D6 counters
            const auto ci = BA_::family_branch_counters(*ec.s->s_);
            ev += ci.evictions; rs += ci.resupplies;
        }
        d["perE_evictions"] = ev;
        d["perE_resupplies"] = rs;
        d["branches_with_port_delta"] = branches_with_port_delta_;
        return d;
    }

    // S8 fix round (final review M1): the run's flags are taken HERE, at the public scope, before any effect on the
    // sink / eng_ib_ / cls_ib_ / the engine's input branch — a refused concurrent call returns before touching the
    // running call's state (IbReset only exists past the hold). run_core_ takes none (the caller holds them).
    py::tuple sample(long shots, py::object seed_obj, bool retain, py::object input_pauli,
                     py::object input_branch = py::none()) {
#ifndef XTIM_MUTATE_BUSY_RUNCORE
        const BusyHold hs = hold_("run"), hc = hold_cache_("run");
#endif
        return sample_held_(shots, seed_obj, retain, input_pauli, input_branch);
    }
    // sample() with the sampler and cache flags already held by the caller (sample / sample_records_core_).
    py::tuple sample_held_(long shots, py::object seed_obj, bool retain, py::object input_pauli,
                           py::object input_branch) {
        if (shots < 0) throw std::invalid_argument("shots must be >= 0");
        // J-b1 P9a / P9b: per-shot input branch (branch 0 = the ideal state). E_min = {} branches run on this
        // sampler (engine branch ids); an E_min != {} branch runs on its E-class's per-E consumer (same text and
        // options, its own plan table), on that class's shots only, the rows scattered back (exact per shot).
        std::vector<int32_t> cls;                     // per shot: its E-class + 1 (0: this sampler)
        bool any_cls = false;
        if (!input_branch.is_none()) {
            auto ib = input_branch.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            if (ib.ndim() != 1 || ib.shape(0) != shots)
                throw std::invalid_argument("input_branch: expected shape (" + std::to_string(shots) + ",)");
            const int nb = btab_ ? btab_->num_branches() : 1;
            bool any = false;
            cls.assign((size_t)shots, 0);
            for (long i = 0; i < shots; ++i) {
                const int b = ib.data()[i];
                if (b < 0 || b >= nb) throw std::invalid_argument("input_branch: branch id out of range");
                if (b != 0) {
                    if (bclass_[(size_t)(b - 1)]) { cls[(size_t)i] = bclass_[(size_t)(b - 1)]; any_cls = true; }
                    else if (beng_[(size_t)(b - 1)] == 0)   // C3: every branch is validated (built) at add_branch
                        throw std::logic_error("input_branch: the run met branch " + std::to_string(b) + " whose "
                                               "per-E law was not validated at add_branch (engine bug)");
                    else any = true;
                }
#ifdef XTIM_MUTATE_BR_MAPPROBE
                if (btab_) (void)btab_->map_find(0, b);   // mutation build only (MUT-BR-MAPPROBE): branch 0 probes
#endif
            }
            if (any) {                                // branch-0-only runs never hand the engine a branch array
                eng_ib_.assign((size_t)shots, 0);
                for (long i = 0; i < shots; ++i) {
                    const int b = ib.data()[i];
                    eng_ib_[(size_t)i] = (b && !cls[(size_t)i]) ? beng_[(size_t)(b - 1)] : 0;
                }
            }
            if (any_cls && recording_)
                xtim_api::raise_twirl_refusal("input_branch: law limitation (carried axis, branch_row): an E_min != {} "
                                              "branch's rows are sampled (sample()) but not recorded in J-b1 (its "
                                              "records live on the per-E consumer; served after J-c); refused",
                                              "LAW_LIMITATION_CARRIED_AXIS", "branch_row");   // final review m2
            if (any_cls) {
                cls_ib_.assign((size_t)shots, 0);
                for (long i = 0; i < shots; ++i) {
                    const int b = ib.data()[i];
                    if (cls[(size_t)i]) cls_ib_[(size_t)i] = beng_[(size_t)(b - 1)];
                }
            }
        }
        struct IbReset {
            PyFamilySampler& s;
            ~IbReset() { s.eng_ib_.clear(); s.cls_ib_.clear(); BA_::family_set_input_branch(*s.s_, nullptr, 0); }
        } ib_reset{*this};
        // J-a (N4b): input_pauli frames (PartitionedTwirlSampler forwards None per group)
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame_arr;
        const uint8_t* frame_ptr = nullptr;
        int frame_stride = 0;
        if (!input_pauli.is_none()) {
            const int cols = 2 * (int)opt_.input_pauli_qubits.size();
            if (cols == 0) throw std::invalid_argument("input_pauli supplied but circuit declares no INPUT_QUBITS port");
            frame_arr = input_pauli.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
            if (frame_arr.ndim() != 2 || frame_arr.shape(0) != shots || frame_arr.shape(1) != cols)
                throw std::invalid_argument("input_pauli: expected shape (" + std::to_string(shots) + ", " +
                                            std::to_string(cols) + ")");
            frame_ptr = frame_arr.data();
            frame_stride = cols;
        }
        const int DBB = (num_det_ + 7) / 8, OBB = (num_obs_ + 7) / 8, DECB = (num_dec_ + 7) / 8;
        std::vector<uint8_t> d((size_t)shots * DBB, 0), o((size_t)shots * OBB, 0), dc((size_t)shots * DECB, 0);
        run_core_(shots, seed_obj, retain, frame_ptr, frame_stride, eng_ib_.empty() ? nullptr : eng_ib_.data(), d, o, dc);
        if (any_cls) {
            for (size_t k = 0; k < eclass_.size(); ++k) {
                std::vector<long> rows;
                for (long i = 0; i < shots; ++i) if (cls[(size_t)i] == (int32_t)(k + 1)) rows.push_back(i);
                if (rows.empty()) continue;
                const long m = (long)rows.size();
                std::vector<int32_t> ids((size_t)m);
                std::vector<uint8_t> fr;
                for (long j = 0; j < m; ++j) ids[(size_t)j] = cls_ib_[(size_t)rows[(size_t)j]];
                if (frame_ptr && frame_stride) {
                    fr.resize((size_t)m * (size_t)frame_stride);
                    for (long j = 0; j < m; ++j)
                        std::memcpy(&fr[(size_t)j * frame_stride], frame_ptr + (size_t)rows[(size_t)j] * frame_stride,
                                    (size_t)frame_stride);
                }
                py::object sk = seed_obj.is_none() ? py::object(py::none())
                                                    : py::int_(py::cast<uint64_t>(seed_obj) ^ (0x9E3779B97F4A7C15ULL * (k + 1)));
                PyFamilySampler& in = *eclass_[k].s;
                std::vector<uint8_t> d2((size_t)m * DBB, 0), o2((size_t)m * OBB, 0), dc2((size_t)m * DECB, 0);
#ifndef XTIM_MUTATE_BUSY_RUNCORE
                const BusyHold ihs = in.hold_("run"), ihc = in.hold_cache_("run");   // the inner's own flags
#endif
                in.run_core_(m, sk, false, frame_ptr ? fr.data() : nullptr, frame_stride, ids.data(), d2, o2, dc2);
                const EClass& ec = eclass_[k];
                for (long j = 0; j < m; ++j) {
                    for (int q = 0; q < num_det_; ++q)
                        if (ec.dmask[(size_t)q]) d2[(size_t)j * DBB + (size_t)(q >> 3)] ^= (uint8_t)(1u << (q & 7));
                    for (int q = 0; q < num_obs_; ++q)
                        if (ec.omask[(size_t)q]) o2[(size_t)j * OBB + (size_t)(q >> 3)] ^= (uint8_t)(1u << (q & 7));
                }
                for (long j = 0; j < m; ++j) {
                    const size_t i = (size_t)rows[(size_t)j];
                    if (DBB) std::memcpy(&d[i * DBB], &d2[(size_t)j * DBB], (size_t)DBB);   // (no memcpy of 0 bytes
                    if (OBB) std::memcpy(&o[i * OBB], &o2[(size_t)j * OBB], (size_t)OBB);   //  from an empty vector)
                    if (DECB) std::memcpy(&dc[i * DECB], &dc2[(size_t)j * DECB], (size_t)DECB);
                }
            }
        }
        return py::make_tuple(packed(d, shots, DBB), packed(o, shots, OBB), packed(dc, shots, DECB));
    }
    // J-b1 P9b: the reference values this sampler's DETECTOR / OBSERVABLE columns are relative to (A1 §2: the
    // noiseless run from its bare, frame-free — noiseless_reference_records, as fam_setup_ / family_port_born_ref
    // read it): per detector, per observable, the parity of the reference records.
    std::pair<std::vector<uint8_t>, std::vector<uint8_t>> column_reference_() const {
        const std::vector<uint8_t> R = noiseless_reference_records(*bare_, deferred_, reads_, opt_.rec_invert, nullptr);
        std::vector<uint8_t> dr((size_t)num_det_, 0), orf((size_t)num_obs_, 0);
        for (int j = 0; j < num_det_ && j < (int)detectors_.size(); ++j)
            for (int r : detectors_[(size_t)j]) if (r >= 0 && r < (int)R.size()) dr[(size_t)j] ^= R[(size_t)r] & 1;
        for (const auto& kv : obs_list_)
            if (kv.first >= 0 && kv.first < num_obs_)
                for (int r : kv.second) if (r >= 0 && r < (int)R.size()) orf[(size_t)kv.first] ^= R[(size_t)r] & 1;
        return {dr, orf};
    }
    // One engine run of `shots` shots into d / o / dc (the per-shot engine branch ids `eng`, nullptr: branch 0).
    // The caller holds this sampler's flag and its cache flag (S8 fix round M1).
    void run_core_(long shots, const py::object& seed_obj, bool retain, const uint8_t* frame_ptr, int frame_stride,
                   const int32_t* eng, std::vector<uint8_t>& d, std::vector<uint8_t>& o, std::vector<uint8_t>& dc) {
        const int DBB = (num_det_ + 7) / 8, OBB = (num_obs_ + 7) / 8, DECB = (num_dec_ + 7) / 8;
#ifdef XTIM_MUTATE_BUSY_RUNCORE
        const BusyHold hs = hold_("run"), hc = hold_cache_("run");   // mutation build only: the old (S8-2a) scope
#endif
        // OD-2026-10-23-s8-seed (fixplan-negative-seed): the seed is converted while the GIL is held — a seed that fails
        // the uint64 cast raises here (Python's error, the sampler untouched) instead of crashing the process.
        const bool has_seed = !seed_obj.is_none();
#ifndef XTIM_MUTATE_SEED_CAST_NOGIL
        const uint64_t seed_v = has_seed ? py::cast<uint64_t>(seed_obj) : 0;
#endif
        bool ok = true;
        try {   // S0d: a C++ exception out of run() still emits this call's queued events (collect mode)
            py::gil_scoped_release rel;
#ifdef XTIM_MUTATE_SEED_CAST_NOGIL
            const uint64_t seed_v = has_seed ? py::cast<uint64_t>(seed_obj) : 0;   // mutation build only: the old site
#endif
            if (has_seed && !s_->set_seed(seed_v))
                throw std::runtime_error("family sampler: set_seed refused");
            FamilySink fs;
            fs.dets = d.data(); fs.DBB = DBB;
            fs.obs = o.data(); fs.OBB = OBB;
            fs.dec = dc.data(); fs.DECB = DECB;
            fs.ndec = num_dec_;                       // channel metadata: set_family_sink (the sampler's own)
            fs.born_oi = s_->born_obs_index();
            fs.born_ref = s_->born_obs_ref();
            struct Reset {                            // the sink / frame / branch pointers never outlive the call
                TwirlRecordSampler* s;
                ~Reset() {
                    BA_::family_set_input_branch(*s, nullptr, 0);
                    s->set_input_frame(nullptr, 0);
                    s->set_retain_state(false);
                    s->set_family_sink(nullptr);
                }
            } reset{s_.get()};
            s_->set_family_sink(&fs);
            s_->set_retain_state(retain);
            s_->set_input_frame(frame_ptr, frame_stride);
            if (eng) BA_::family_set_input_branch(*s_, eng, shots);   // J-b1 P9b
            ok = s_->run(shots);
        } catch (...) { drain_(std::string(), /*collect=*/true); throw; }
        if (!ok) {
            const std::string why = BA_::family_refusal(*s_);
            if (!why.empty()) { drain_(std::string(), /*collect=*/true); throw std::runtime_error(why); }
        }
        drain_run_(ok);
    }
    // The run's events, then its one outcome: an oracle failure (an ORACLE_FAILURE event or run() == false)
    // raises RuntimeError("family sampler: <the oracle's message>") from its XtimCorrectnessWarning.
    void drain_run_(bool ok) {
        std::vector<TwirlRecordSampler::DiagEvent> ev = s_->take_diagnostics();
        std::string abort;
        for (const auto& e : ev)
            if (e.kind == TwirlRecordSampler::DiagEvent::ORACLE_FAILURE) { abort = "family sampler: " + e.detail; break; }
        if (!ok && abort.empty()) abort = "family sampler: run() aborted (oracle)";
        if (ev.empty()) {
            if (!abort.empty()) throw std::runtime_error(abort);
            return;
        }
        pending_ = std::move(ev);
        drain_(abort);
    }

    // J-a N5: one run with layout-3 records (every shot term-routed); (dets, obs, decisions) and the records.
    py::tuple sample_records_core_(long shots, py::object seed_obj, py::object input_pauli,
                                   py::object input_branch, std::shared_ptr<PyFamilyRecords>& out_R) {
#ifndef XTIM_MUTATE_BUSY_RUNCORE
        const BusyHold hs = hold_("run"), hc = hold_cache_("run");   // M1: before the record sink is pointed anywhere
#endif
        auto R = std::make_shared<PyFamilyRecords>(shots, bare_->n(), (int)s_->record_operators().size(),
                                                   s_->record_value_words());
        BA_::set_family_record_sink(*s_, &R->sink);
        py::tuple out;
        recording_ = true;
        try {
            out = sample_held_(shots, seed_obj, false, input_pauli, input_branch);
        } catch (...) {
            recording_ = false;
            BA_::set_family_record_sink(*s_, nullptr);
            throw;
        }
        recording_ = false;
        BA_::set_family_record_sink(*s_, nullptr);
        if (R->sink.row != shots)
            throw std::logic_error("family records: " + std::to_string(R->sink.row) + " records for " +
                                   std::to_string(shots) + " shots (engine bug)");
        R->contents = BA_::family_record_contents(*s_);
        R->model = model_key();
        R->model_serial = serial_;
        out_R = std::move(R);
        return out;
    }
    // (dets, obs, decisions, FamilyRecords) — the test API's records object (production binds no records type).
    py::tuple sample_records(long shots, py::object seed_obj, py::object input_pauli,
                             py::object input_branch = py::none()) {
        std::shared_ptr<PyFamilyRecords> R;
        py::tuple out = sample_records_core_(shots, seed_obj, input_pauli, input_branch, R);
        return py::make_tuple(out[0], out[1], out[2], py::cast(R));
    }
    // G-8d: records are read only by a sampler with the producer's state model (identity tuple).
    void same_model(const PyFamilyRecords& R) const {
        if (R.model_serial == serial_) return;       // this sampler's own records: the same model by construction
        if (R.model != model_key())
            throw std::invalid_argument("family records come from a sampler with a different state model (bare "
                                        "state, certified generators, every record operator with its order and "
                                        "inv/ref fold); refused");
    }
    // The producer's content handles resolved into THIS sampler's (by content key; cold per distinct content).
    std::vector<int32_t> local_handles(const PyFamilyRecords& R) const {
        std::vector<int32_t> map(R.contents.size(), -1), h((size_t)R.rows, -1);
        for (long i = 0; i < R.rows; ++i) {
            const int p = R.handle[(size_t)i];
            if (p < 0 || p >= (int)R.contents.size()) throw std::logic_error("record without content (engine bug)");
            if (map[(size_t)p] < 0) map[(size_t)p] = BA_::family_intern_content(*s_, R.contents[(size_t)p].kind, R.contents[(size_t)p].content);
            h[(size_t)i] = map[(size_t)p];
        }
        return h;
    }
    FramedSuperposition materialize_state(const PyFamilyRecords& R, long i, bool tableau = false) const {
        same_model(R);
        const auto& C = R.content_of(i);
        const int h = BA_::family_intern_content(*s_, C.kind, C.content);
        if (C.kind != 0 || R.branch[(size_t)i] != 0) arm_();   // r2 item 4: kind-0 branch-0 reads never arm the term view
        return BA_::family_materialize(*s_, h, R.prec_row(i), R.y_row(i), tableau, R.branch[(size_t)i]);
    }
    // The value route of a record (r2 item 2): tableau on branch-0 rows, C0̂ on carried-branch rows.
    FramedSuperposition value_state(const PyFamilyRecords& R, long i) const {
        R.check(i);
#ifdef XTIM_MUTATE_MAT_C0HAT
        return materialize_state(R, i, false);       // mutation build only (MUT-MAT-C0HAT): the value readers on C0̂
#else
        return materialize_state(R, i, R.branch[(size_t)i] == 0);
#endif
    }
    py::array_t<double> state_expectations(const PyFamilyRecords& R, long i, const py::list& paulis,
                                           bool tableau = false) const {
        FramedSuperposition st = materialize_state(R, i, tableau);
        py::array_t<double> a((py::ssize_t)paulis.size());
        for (size_t k = 0; k < paulis.size(); ++k) {
            const std::complex<double> v = st.pauli_expectation(pauli_from(st.n(), paulis[k]));
            a.mutable_data()[k] = v.real();
        }
        return a;
    }
    py::array_t<int8_t> signs(const PyFamilyRecords& R, const py::list& paulis) const {
        same_model(R);
        R.no_branch_rows("signs");
        std::vector<Pauli> g;
        for (const auto& t : paulis) g.push_back(pauli_from(bare_->n(), t));
        py::array_t<int8_t> a = xtim_api::signs_array(R.rows, g.size());   // J-b1 P11b: the shared builder
        std::vector<int32_t> h = local_handles(R);
        BA_::FamilyRecordSink view = R.sink;
        view.handle = h.data();
        arm_();
        if (R.rows && !g.empty()) BA_::family_signs(*s_, view, R.rows, g, a.mutable_data());
        return a;
    }
    py::dict record_classes() const {
        static const char* names[] = {"deterministic", "coin", "born"};
        const std::vector<int> cls = BA_::family_record_classes(*s_);
        const std::vector<int> order = s_->record_column_order();
        py::dict dets, obs, dec;
        std::vector<int> nd_det, nd_obs;
        for (size_t c = 0; c < cls.size(); ++c) {
            const int kind = order[c] >> 24, idx = order[c] & ((1 << 24) - 1);
            py::dict& d = kind == 0 ? dets : kind == 1 ? obs : dec;
            d[py::int_(idx)] = names[cls[c]];
            if (cls[c] != 0 && kind == 0) nd_det.push_back(idx);
            if (cls[c] != 0 && kind == 1) nd_obs.push_back(idx);
        }
        std::sort(nd_det.begin(), nd_det.end());
        std::sort(nd_obs.begin(), nd_obs.end());
        py::dict r;
        py::dict classes;
        classes["detectors"] = dets; classes["observables"] = obs; classes["decisions"] = dec;
        r["classes"] = classes;
        r["nondeterministic_detectors"] = nd_det;
        r["nondeterministic_observables"] = nd_obs;
        return r;
    }

    // ── J-b1 P11b: the record API (design §4) through the shared builders (cpp/bindings/record_api.hpp) ──
    // J-b1 P12 (fixplan-p12-frames (A)): a content's plan with its C0 as indices into the export's shared axis
    // table T (kmap: kind-2 term j → table index, filled lazily per export).
    xtim_api::PlanExport plan_export_(int h, xtim_api::AxisTable& T, std::vector<int32_t>& kmap) const {
        const auto v = BA_::family_plan_view(*s_, h);
        xtim_api::PlanExport p;
        p.kind = v.kind; p.content = v.content; p.digest_hi = v.digest_hi; p.digest_lo = v.digest_lo;
        p.rho = v.rho; p.kappa = v.kappa; p.born = v.born; p.a = v.a; p.cz = v.cz;
        if (v.kind == 0) {
            xtim_api::kind0_term_index(T, v.a, v.cz, p.term_index);
        } else {
            TermResidual& tr = BA_::family_terms(*s_);
            if ((int)kmap.size() != tr.num_terms()) kmap.assign((size_t)tr.num_terms(), -1);
            for (int j = 0; j < tr.num_terms(); ++j)
                if ((j >> 6) < (int)v.S.size() && ((v.S[(size_t)(j >> 6)] >> (j & 63)) & 1ULL)) {
                    if (kmap[(size_t)j] < 0) kmap[(size_t)j] = T.pauli(tr.axis(j));
                    p.term_index.push_back(kmap[(size_t)j]);
                }
        }
        return p;
    }
    py::dict record_report() const {
        const auto c = s_->record_counters();
        std::map<std::string, long> rc, ct;
        for (const auto& kv : c.routes) rc[kv.first] = kv.second;
        ct["plan_misses"] = (long)(c.plan_misses_kappa0 + c.plan_misses_kappa_pos + c.plan_misses_fallback);
        ct["plan_misses_kappa0"] = (long)c.plan_misses_kappa0;
        ct["plan_misses_kappa_pos"] = (long)c.plan_misses_kappa_pos;
        ct["payload_builds"] = (long)c.payload_builds;
        ct["payload_evictions"] = (long)c.payload_evictions;
        ct["header_bytes"] = (long)c.header_bytes;
        ct["payload_resident_bytes"] = (long)c.payload_resident_bytes;
        ct["reader_tables"] = c.reader_tables;
        ct["reader_lookups"] = (long)c.reader_lookups;
        ct["trie_walks"] = c.trie_walks;
        ct["trie_nodes"] = c.trie_nodes;
        py::dict r = xtim_api::record_report_dict(BA_::family_record_classes(*s_), s_->record_column_order(), rc, ct);
        // J-b1 F: production's run counters (the former record_report keys whose source is the family law; the
        // legacy-only ones — exact_*, ppr_*, sigma_aux, frame_dressed_shots, rec.ops_* of the dual basis,
        // channel_one_counts — are gone, jb2 ruling C2) and the former channel_report's disk_path / feedback_hoisted.
        r["used"] = s_->used();
        r["plans"] = s_->plans();
        py::dict pmiss;
        pmiss["kappa0"] = c.plan_misses_kappa0; pmiss["kappa_pos"] = c.plan_misses_kappa_pos;
        pmiss["fallback"] = c.plan_misses_fallback;
        r["plan_misses"] = pmiss;
        py::dict bt;
        bt["walks"] = c.trie_walks; bt["steps"] = c.trie_steps; bt["nodes"] = c.trie_nodes;
        bt["node_evals"] = c.trie_node_evals; bt["chi_j"] = c.trie_chi_j; bt["max_depth"] = c.trie_max_depth;
        bt["passes"] = c.trie_passes; bt["max_support"] = c.trie_max_support; bt["root_memo_hits"] = c.trie_root_memo_hits;
        r["born_trie"] = bt;
        r["structural_shots"] = c.structural_shots;
        py::dict rec;
        rec["tables"] = c.rec_tables; rec["extras"] = c.rec_extras;
        rec["pivots_from_extras"] = c.rec_pivots_from_extras; rec["dirs_from_extras"] = c.rec_dirs_from_extras;
        rec["kappa_rec_hist"] = c.rec_kappa_hist;
        py::dict od;
        od["record"] = c.obs_draws_record;          // (the PPR family's "ppr" key: legacy-only, removed)
        rec["obs_draws"] = od;
        r["rec"] = rec;
        py::dict pm;
        pm["budget"] = cache_budget_out(c.payload_budget);   // math.inf = unbounded
        pm["resident_bytes"] = c.payload_resident_bytes;
        pm["peak_bytes"] = c.payload_peak_bytes;
        pm["overshoot_bytes"] = c.payload_overshoot_bytes;
        pm["header_bytes"] = c.header_bytes;
        pm["payload_builds"] = c.payload_builds;
        pm["evictions"] = c.payload_evictions;
        pm["reader_lookups"] = c.reader_lookups;
        pm["reader_tables"] = c.reader_tables;
        pm["reader_bytes"] = c.reader_bytes;
        pm["reader_peak_bytes"] = c.reader_peak_bytes;
        pm["reader_overshoot_bytes"] = c.reader_overshoot_bytes;
        pm["reader_evictions"] = c.reader_evictions;
        r["plan_memory"] = pm;
        r["disk_path"] = s_->disk_path();
        r["feedback_hoisted"] = feedback_hoisted_;
        r["semantic_oracle_shots"] = s_->semantic_oracle_shots();   // the selfcheck window's oracle (cumulative)
        {   // the family law's term route: kind-2 (CH / PPR) and route-1 (kind 0 outside the pair universe) plans
            const auto st = BA_::family_route_stats(*s_);
            py::dict tr;
            tr["shots_kind2"] = st.shots_kind2; tr["plans_kind2"] = st.plans_kind2;
            tr["shots_kind0_general"] = st.shots_kind0_general; tr["plans_kind0_general"] = st.plans_kind0_general;
            tr["shots_framed"] = st.shots_framed; tr["shots_record_d"] = st.shots_record_d;
            tr["key_bytes"] = st.key_bytes; tr["key_evictions"] = st.key_evictions;   // termroute-a cache bound
            r["term_route"] = tr;
        }
        r["branches"] = branch_counters();
        r["branch_table_digest"] = btab_ ? py::object(dbytes_(btab_->digest())) : py::object(py::none());
        return r;
    }
    // ── J-b1 F: production's static accessors (the former PyTwirlSampler's; same formats) ──
    // Deferred-space wires of the declared OUTPUT_QUBITS, flattened in declaration order (an ideal-state compile:
    // in the combined register).
    std::vector<int> output_wires() const { return out_wires_; }
    // [(name, circuit qubits, deferred wires)] in declaration order.
    py::list output_ports() const {
        py::list out;
        for (const OutputPort& p : outputs_) {
            std::vector<int> w;
            for (int q : p.qubits) w.push_back(final_wire_.at((size_t)q));
            out.append(py::make_tuple(p.name, p.qubits, w));
        }
        return out;
    }
    // Per measurement record (record order): deferred wire, basis (0 X / 1 Y / 2 Z), deterministic invert, the
    // readout-flip probability (M(p), unscaled by p_factor).
    py::dict record_reads() const {
        const size_t R = reads_.size();
        py::array_t<int32_t> wire((py::ssize_t)R), basis((py::ssize_t)R);
        py::array_t<uint8_t> inv((py::ssize_t)R);
        py::array_t<double> flip((py::ssize_t)R);
        for (size_t r = 0; r < R; ++r) {
            basis.mutable_data()[r] = (int32_t)reads_[r].first;
            wire.mutable_data()[r] = (int32_t)reads_[r].second;
            inv.mutable_data()[r] = r < opt_.rec_invert.size() ? opt_.rec_invert[r] : 0;
            flip.mutable_data()[r] = r < opt_.rec_flip.size() ? opt_.rec_flip[r] : 0.0;
        }
        py::dict d;
        d["wire"] = wire; d["basis"] = basis; d["invert"] = inv; d["flip"] = flip;
        d["n"] = s_->deferred_wires();
        return d;
    }
    // The record operators M_c (storage column order: channels then extras) as (x sites, z sites, phase).
    py::list record_operators() const {
        py::list out;
        for (const Pauli& P : s_->record_operators()) {
            std::vector<int> xs, zs;
            for (int q = 0; q < P.n; ++q) { if (P.xbit(q)) xs.push_back(q); if (P.zbit(q)) zs.push_back(q); }
            out.append(py::make_tuple(xs, zs, (int)(P.phase & 3)));
        }
        return out;
    }
    // plan_structure() (design §4 fields) of a layout-3 plan content ([kind u8][len u32 LE] content, as
    // BarrierBuffer.plan_content(i)[0]): a pure function of the compiled circuit and the content, no buffer.
    py::dict plan_structure_of_content(const py::bytes& pc) const {
        const std::string b = pc.cast<std::string>();
        if (b.size() < 5) throw std::invalid_argument("plan_structure: a layout-3 plan content is [kind u8][len u32] content");
        const int kind = (uint8_t)b[0];
        const uint32_t len = (uint32_t)(uint8_t)b[1] | ((uint32_t)(uint8_t)b[2] << 8) | ((uint32_t)(uint8_t)b[3] << 16) |
                             ((uint32_t)(uint8_t)b[4] << 24);
        if (b.size() != 5 + (size_t)len) throw std::invalid_argument("plan_structure: plan content length mismatch");
        if (kind != 0 && kind != 2) throw std::invalid_argument("plan_structure: plan content kind must be 0 or 2");
        arm_();
        const int n = bare_->n();
        xtim_api::AxisTable T(n);
        std::vector<int32_t> kmap;
        const auto p = plan_export_(BA_::family_intern_content(*s_, kind, b.substr(5)), T, kmap);
        return xtim_api::plan_structure_dict(p, T, n);
    }
    // m1 (plan §1.6): the trie checks walk the diagonal cache AND the FAM route's own plans (kind 2, route 1).
    std::pair<long, long> born_trie_check_all() {
        const auto d = s_->born_trie_check();
        const auto f = BA_::family_born_trie_walk(*s_, false);
        return std::make_pair(d.first + f[0] + f[1], d.second + f[2]);
    }
    std::pair<long, long> born_trie_oracle_all() {
        const auto d = s_->born_trie_oracle();
        const auto f = BA_::family_born_trie_walk(*s_, true);
        return std::make_pair(d.first + f[0] + f[1], d.second + f[2]);
    }
    py::dict frames(const PyFamilyRecords& R) const {
        same_model(R);
        R.no_branch_rows("frames");
        arm_();
        const std::vector<int32_t> h = local_handles(R);
        std::vector<int32_t> pidx((size_t)R.rows, -1);
        std::unordered_map<int32_t, int32_t> first;
        std::vector<xtim_api::PlanExport> plans;
        xtim_api::AxisTable T(R.n);
        std::vector<int32_t> kmap;
        for (long i = 0; i < R.rows; ++i) {
            auto it = first.find(h[(size_t)i]);
            if (it == first.end()) {
                it = first.emplace(h[(size_t)i], (int32_t)plans.size()).first;
                plans.push_back(plan_export_(h[(size_t)i], T, kmap));
            }
            pidx[(size_t)i] = it->second;
        }
        xtim_api::RecordRows rows;
        rows.rows = R.rows; rows.n = R.n; rows.m = R.m; rows.PNW = R.sink.PNW; rows.YW = R.sink.YW; rows.UZW = R.sink.UZW;
        rows.plan = pidx.data(); rows.prec = R.prec.data(); rows.y = R.y.data(); rows.uz = R.uz.data();
        rows.nuz = R.nuz.data(); rows.ff = R.ff.data();
        return xtim_api::frames_dict(rows, plans, T, s_->record_operators());
    }
    py::dict record_plan_structure(const PyFamilyRecords& R, long i) const {
        same_model(R);
        arm_();
        const auto& C = R.content_of(i);
        xtim_api::AxisTable T(R.n);
        std::vector<int32_t> kmap;
        const auto p = plan_export_(BA_::family_intern_content(*s_, C.kind, C.content), T, kmap);
        return xtim_api::plan_structure_dict(p, T, R.n);
    }

#ifdef XTIM_FAMILY_TESTAPI   // test-API-only accessors (cpp/testapi/family_py.cpp defines it)
    py::dict channel_report() const {
        py::dict r;
        r["deterministic_detectors"] = s_->channel_detectors();
        r["deterministic_observables"] = s_->channel_observables();
        r["born_observable"] = s_->born_obs_index();
        py::list fx;
        static const char* kinds[] = {"gauge_detector", "anti_detector", "extra_observable"};
        for (const auto& e : TA_::family_extras(*s_))
            fx.append(py::make_tuple(kinds[e.kind], e.index, e.column, (int)e.off));
        r["family_extras"] = fx;
        r["used"] = s_->used();
        return r;
    }

    // The plan's record table (the TwirlSampler.plan_structure record fields), for the per-plan field
    // identity against (X) (tests/_ja_fixtures.compare_tables). force_general=True (J-a N3, G-9a): the
    // table of the SAME plan built by (G) (build_record_table_general on the plan's C0 written as terms).
    py::dict plan_structure(py::bytes pk_bytes, bool force_general) const {
        const std::string pk = pk_bytes.cast<std::string>();
        const uint8_t* k = pk.empty() ? nullptr : reinterpret_cast<const uint8_t*>(pk.data());
        const TwirlRecordSampler::PlanStructure ps = s_->plan_structure(k, (int)pk.size());
        if (!force_general || ps.law.fallback) {
            py::dict res = table_dict(ps.rec);
            res["fallback"] = ps.law.fallback;
            return res;
        }
        RecTable T;
        std::vector<int> ib;
        TA_::family_general_table_diag(*s_, k, (int)pk.size(), T, &ib);
        py::dict res = table_dict(T);
        res["fallback"] = false;
        res["intro_b"] = ib;
        return res;
    }
    // J-a N3 (G-9b / G-9c): (G)'s table of the plan C0 = rot(S) of the sampler's term view (S: term ids, e.g.
    // term_shot()['S_red'] — the N2 term key's plan); born_law=True adds p_z (the directions' joint Born law
    // by the unchanged BornTrie node definition, index sum_j z_j 2^j).
    py::dict term_plan_structure(const std::vector<int>& S, bool born_law) {
        RecTable T;
        std::vector<int> ib;
        TA_::family_general_table(*s_, set_words(S, terms().num_terms()), T, &ib);
        py::dict res = table_dict(T);
        res["fallback"] = false;
        res["intro_b"] = ib;
        if (born_law) res["p_z"] = TA_::family_born_law(*s_, T);
        return res;
    }
    // <bare|P|bare> for P = i^phase X^x Z^z (the bare state's exact pauli_expectation; N3-D2).
    std::complex<double> bare_expectation(const std::vector<uint8_t>& x, const std::vector<uint8_t>& z,
                                          int phase) const {
        Pauli P = pauli_of(bare_->n(), x, z);
        P.phase = phase & 3;
        return bare_->pauli_expectation(P);
    }
    py::dict table_dict(const RecTable& T) const {
        py::dict res;
        res["rec_built"] = T.built;
        res["m"] = T.m;
        res["rho"] = T.rho;
        res["kappa_rec"] = T.kappa;
        res["pivot_rec"] = T.pivot_rec;
        res["born_rec"] = T.born_rec;
        res["rec_order"] = s_->record_column_order();
        py::array_t<uint8_t> A({(py::ssize_t)T.m, (py::ssize_t)T.rho});
        py::array_t<uint8_t> B({(py::ssize_t)T.m, (py::ssize_t)T.kappa});
        py::array_t<uint8_t> be((py::ssize_t)T.m);
        for (int c = 0; c < T.m; ++c) {
            for (int k = 0; k < T.rho; ++k)
                A.mutable_data()[(size_t)c * T.rho + k] = (uint8_t)((T.coin_row[(size_t)k][(size_t)(c >> 6)] >> (c & 63)) & 1);
            for (int j = 0; j < T.kappa; ++j)
                B.mutable_data()[(size_t)c * T.kappa + j] = (uint8_t)((T.born_row[(size_t)j][(size_t)(c >> 6)] >> (c & 63)) & 1);
            be.mutable_data()[c] = (uint8_t)((T.base.empty() ? 0 : (T.base[(size_t)(c >> 6)] >> (c & 63))) & 1);
        }
        res["rec_A"] = A;
        res["rec_B"] = B;
        res["rec_beta"] = be;
        const int kk = T.lam.empty() ? 0 : (int)T.lam[0].a.size();
        py::array_t<uint8_t> lam({(py::ssize_t)T.kappa, (py::ssize_t)(2 * kk)});
        py::array_t<uint8_t> lph((py::ssize_t)T.kappa);
        for (int j = 0; j < T.kappa; ++j) {
            for (int i = 0; i < kk; ++i) {
                lam.mutable_data()[(size_t)j * 2 * kk + i] = T.lam[(size_t)j].a[(size_t)i];
                lam.mutable_data()[(size_t)j * 2 * kk + kk + i] = T.lam[(size_t)j].b[(size_t)i];
            }
            lph.mutable_data()[j] = (uint8_t)fd_eff_phase(T.lam[(size_t)j]);
        }
        res["rec_lam"] = lam;
        res["rec_lam_phase"] = lph;
        res["folds"] = T.folds;                       // introducers built with the folded beta' (N3-D3)
        py::dict ops;                                 // add_extra operation counts of this build (M-1)
        ops["plane"] = T.ops_plane;
        ops["pivot_reduce"] = T.ops_pivot_reduce;
        ops["greduce"] = T.ops_greduce;
        ops["echelon"] = T.ops_echelon;
        ops["extras"] = T.extras;
        res["ops"] = ops;
        return res;
    }


#endif
    // ── J-a (N2): the term route ──
    // J-b1 P4: the sampler keeps no construction inputs when its term route is unarmed (R9a); the test API's
    // accessors arm the term view on demand from THIS object's copies (deferred_, reads_). Never a run path.
    void arm_() const {                             // idempotent (r2 item 3): armed once, then a flag test
        if (armed_) return;
        BA_::family_arm_terms(*s_, deferred_, reads_);
        armed_ = true;
    }
#ifdef XTIM_FAMILY_TESTAPI
    TermResidual& terms() { arm_(); return TA_::family_terms(*s_); }
    py::list term_alt_faults() const {
        py::list out;
        for (const auto& fr : TA_::family_alt_faults(*s_)) {
            py::list l;
            for (const FiredPauli& f : fr)
                l.append(py::make_tuple(f.location_index, f.qubit,
                                        f.pauli == PauliBasis::X ? 1 : f.pauli == PauliBasis::Y ? 2 : 3));
            out.append(l);
        }
        return out;
    }
    py::dict term_shot(const std::vector<int32_t>& alts) {
        TermRouteShot r;
        arm_();
        TA_::family_term_shot(*s_, alts, r);
        const int T = terms().num_terms();
        py::dict d;
        d["P"] = pauli_xz(r.P0);
        d["S"] = set_list(r.S0, T);
        d["P_red"] = pauli_xz(r.sh.P);
        d["S_red"] = set_list(r.sh.S, T);
        d["key"] = words_bytes(terms().key_of(r.sh.S));
        d["kind"] = r.kind;
        d["plan_id"] = r.plan_id;
        if (r.kind == 0) { d["nf"] = nf_dict(r.nf); d["nf_raw"] = nf_dict(r.nf_raw); }
        else { d["nf"] = py::none(); d["nf_raw"] = py::none(); }
        return d;
    }
    py::object diag_shot(const std::vector<int32_t>& alts) {
        DiagNormalForm nf;
        long id = -1;
        arm_();
        if (!TA_::family_diag_shot(*s_, alts, nf, id)) return py::none();
        py::dict d = nf_dict(nf);
        d["plan_id"] = id;
        return d;
    }
    py::array_t<uint8_t> relabel(const std::vector<uint8_t>& x, const std::vector<uint8_t>& z) const {
        const std::vector<uint8_t> t = TA_::family_relabel(*s_, pauli_of((int)x.size(), x, z));
        py::array_t<uint8_t> a((py::ssize_t)t.size());
        if (!t.empty()) std::memcpy(a.mutable_data(), t.data(), t.size());
        return a;
    }

    py::dict route_stats() const {
        const TA_::FamilyRouteStats st = TA_::family_route_stats(*s_);
        py::dict r;
        r["shots_kind0"] = st.shots_kind0;
        r["shots_kind2"] = st.shots_kind2;
        r["keys"] = st.keys;
        r["plans_kind2"] = st.plans_kind2;
        r["builds_kind2"] = st.builds_kind2;
        r["payload_builds_kind2"] = st.payload_builds_kind2;
        r["oracle_term_shots"] = st.oracle_term_shots;
        r["shots_framed"] = st.shots_framed;
        r["shots_record_d"] = st.shots_record_d;   // J-b1 P4
        r["tables_recb"] = st.tables_recb;         // J-b1 P6
        r["laws_payload"] = st.laws_payload;       // J-b1 (fixplan-coldstart)
        r["tables_recb_parked"] = st.tables_recb_parked;
        r["recb_core_rebuilds"] = st.recb_core_rebuilds;
        r["reader_table_rebuilds"] = st.reader_table_rebuilds;
        r["plans_kind0_general"] = st.plans_kind0_general;
        r["builds_kind0_general"] = st.builds_kind0_general;
        r["payload_builds_kind0_general"] = st.payload_builds_kind0_general;
        r["shots_kind0_general"] = st.shots_kind0_general;
        py::dict ps;
        for (size_t i = 0; i < st.plan_shots.size(); ++i)
            if (st.plan_shots[i]) ps[py::int_(i)] = py::make_tuple(st.plan_kind[i], st.plan_shots[i]);
        r["plan_shots"] = ps;
        return r;
    }

    py::dict oracle_counts() const {
        const std::vector<long> c = TA_::semantic_oracle_class_checks(*s_);
        py::dict r;
        static const char* names[] = {"channel", "born_observable", "born_decision", "gauge_detector",
                                      "anti_detector", "extra_observable"};
        for (int i = 0; i < 6; ++i) r[names[i]] = c[(size_t)i];
        r["shots"] = s_->semantic_oracle_shots();
        return r;
    }
#endif
};


// J-b1 P7: the layout-3 record fidelity (record_fidelity.hpp, record mode) on a FAM sampler's records. Built on the
// sampler's bare state and EVERY record operator; a records object from a sampler with a different state model
// (family_model_key: bare, generators, every record operator with its folds) is refused by name (G-3).
// J-b1 P12: the test API's RecordReader — the FAM sampler wrapper behind a layout-3 BarrierBuffer (kept alive).
// S8-2a: a bound member entry under the sampler flag (as BudgetGuard: &Busy<&PyFamilySampler::f>::call).
template <auto M> struct Busy;
template <class R, class... A, R (PyFamilySampler::*M)(A...) const>
struct Busy<M> {
    static R call(const PyFamilySampler& s, A... a) {
        const BusyHold h = s.hold_("sampler entry");
        return (s.*M)(std::forward<A>(a)...);
    }
};
template <class R, class... A, R (PyFamilySampler::*M)(A...)>
struct Busy<M> {
    static R call(PyFamilySampler& s, A... a) {
        const BusyHold h = s.hold_("sampler entry");
        return (s.*M)(std::forward<A>(a)...);
    }
};

struct FamReader : xtim_api::RecordReader {
    py::object owner;
    PyFamilySampler* s;
    explicit FamReader(py::object o) : owner(std::move(o)), s(&owner.cast<PyFamilySampler&>()) {}
    // S8-2a: every reader holds the sampler flag (the lazy model key, arm_, the content table memo), the readers of
    // plan payloads and RecordFidelity.evaluate also the cache flag.
    const std::string& model() const override { const BusyHold h = s->hold_("model key"); return s->model_key(); }
    // J-b1 F (fixplan-F-materialize-c0hat r2 item 2): the value readers take the gate-level (tableau) route on
    // branch-0 rows — P8's gate proves it equal to C0̂ as a state; C0̂ stays where the frame is the contract (the
    // branch export) and on carried-branch rows (no tableau route there; C0̂'s capacity refusal bounds it).
    FramedSuperposition state(const PyFamilyRecords& R, long i) const override {
        const BusyHold h = s->hold_("materialize"), hc = s->hold_cache_("materialize");
        return s->value_state(R, i);
    }
    py::array_t<int8_t> signs(const PyFamilyRecords& R, const py::list& p) const override {
        const BusyHold h = s->hold_("signs"), hc = s->hold_cache_("signs");
        return s->signs(R, p);
    }
    py::dict frames(const PyFamilyRecords& R) const override {
        const BusyHold h = s->hold_("frames"), hc = s->hold_cache_("frames");
        return s->frames(R);
    }
    py::dict record_layout() const override {
        const int n = s->bare_->n();
        return xtim_api::record_layout_dict(n, (int)s->s_->record_operators().size(), (n + 63) / 64,
                                            s->s_->record_value_words());
    }
    void fidelity(RecordFidelity& rf, const PyFamilyRecords& R, double* out, const uint8_t* corr) const override {
        const BusyHold hs_ = s->hold_("RecordFidelity.evaluate");
#ifndef XTIM_MUTATE_BUSY_RFEVAL
        const BusyHold hc_ = s->hold_cache_("RecordFidelity.evaluate");
#endif
#ifndef XTIM_MUTATE_F1_RF_BRANCH
        // J-b1 F (fixplan-F-branch-readers F-1): rf is built on branch 0's bare; a carried-branch row's state is the
        // branch's own consumer bare, so scoring it here would be silently wrong. Refused by name (J-c serves it).
        R.no_branch_rows("RecordFidelity");
#endif
        s->same_model(R);
        bool k2 = false;
        for (long i = 0; i < R.rows && !k2; ++i)
            if (R.contents[(size_t)R.handle[(size_t)i]].kind != 0) k2 = true;
        if (k2) s->arm_();
        // J-b1 F: the records' content handles are the PRODUCER's; resolve them into this sampler's (as signs() and
        // frames() do) — a buffer of another sampler of the same state model is served, not mis-indexed.
        std::vector<int32_t> h = s->local_handles(R);
        BA_::FamilyRecordSink view = R.sink;
        view.handle = h.data();
        BA_::family_record_fidelity(*s->s_, rf, view, R.rows, out, corr, corr ? &s->out_wires_ : nullptr);
    }
    const FramedSuperposition& bare() const override { return *s->bare_; }
    std::vector<Pauli> record_ops() const override { return BA_::family_record_ops(*s->s_); }
    int n_out() const override { return (int)s->out_wires_.size(); }
    std::vector<int> out_wires() const override { return s->out_wires_; }
    std::vector<int> read_meas() const override { return s->read_meas_(); }
    std::vector<int> wire_user() const override { return s->wire_user_(); }
    py::dict plan_structure(const PyFamilyRecords& R, long i) const override {
        const BusyHold h = s->hold_("plan_structure"), hc = s->hold_cache_("plan_structure");
        return s->record_plan_structure(R, i);
    }
    py::dict born_dec() const override { return xtim_api::born_dec_dict(s->s_->born_dec_ops()); }
    std::vector<std::complex<double>> expectations(const PyFamilyRecords& R, long i,
                                                   const std::vector<Pauli>& ps) const override {
        const BusyHold h = s->hold_("pauli_expectations"), hc = s->hold_cache_("pauli_expectations");
        const FramedSuperposition st = s->value_state(R, i);
        std::vector<std::complex<double>> e;
        for (const Pauli& P : ps) e.push_back(st.pauli_expectation(P));
        return e;
    }
};

// ── J-b1 F: the record surfaces both extensions bind (one body; plan r3.3) ──
// A layout-3 records argument: a BarrierBuffer (its records) or, in the test API, a FamilyRecords object.
inline std::shared_ptr<PyFamilyRecords> records_arg(const py::object& o) {
    if (py::isinstance<xtim_api::BarrierBuffer>(o)) return o.cast<const xtim_api::BarrierBuffer&>().R;
#if defined(XTIM_FAMILY_TESTAPI) || defined(XTIM_MUTATE_RECORDS_FOREIGN)
    return o.cast<std::shared_ptr<PyFamilyRecords>>();
#else
    // J-b1 final review m1: production reads only ITS OWN BarrierBuffer. pybind11 loads a foreign extension's
    // module_local type with the same C++ typeid, so a test-API records object (whose sampler serial lives in another
    // namespace) is refused here by type.
    throw py::type_error("records: expected this module's BarrierBuffer (sample_barrier); a records object of another "
                         "extension is refused");
#endif
}
// sample_barrier: the layout-3 BarrierBuffer; every shot keeps its record. A layout-3 buffer is always compact
// (records only, the state reconstructed on demand): compact=True is accepted, compact=False refused by name.
inline std::unique_ptr<xtim_api::BarrierBuffer> family_sample_barrier(py::object self, long shots, py::object seed,
                                                                      py::object input_pauli, bool compact,
                                                                      py::object input_branch) {
    if (!compact)
        throw std::invalid_argument("sample_barrier: compact=False — a layout-3 BarrierBuffer keeps every shot's "
                                    "record and reconstructs its state on demand (materialize); there is no "
                                    "full-state buffer; refused");
    PyFamilySampler& s = self.cast<PyFamilySampler&>();
    auto B = std::make_unique<xtim_api::BarrierBuffer>();
    py::tuple t = s.sample_records_core_(shots, seed, input_pauli, input_branch, B->R);
    B->rd = std::make_shared<FamReader>(self);
    B->dets_ = t[0]; B->obs_ = t[1]; B->dec_ = t[2];
    B->frames_ = py::none();
    B->module_ = py::module_::import(PyFamilySampler::module_name_());
    return B;
}
// The layout-3 RecordFidelity evaluator of a sampler (evaluate(buffer, correction=None, shots=None)).
inline std::unique_ptr<xtim_api::RecordFidelityL3> family_record_fidelity_l3(py::object self, size_t max_bytes,
                                                                             py::object cache_budget) {
    return std::make_unique<xtim_api::RecordFidelityL3>(std::make_shared<FamReader>(self), max_bytes,
                                                        cache_budget_arg(cache_budget, "RecordFidelity: cache_budget"));
}

}  // namespace
