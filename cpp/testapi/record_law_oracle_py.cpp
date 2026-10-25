// The Stage S1 exact record-law oracle (X), exposed as a TEST API: the extension module
// `xtim._xtim_testapi`, class `RecordLawOracle`. Its own extension (setup.py), linked against the
// engine objects compiled for _xtim: nothing here is reachable from a sampling path, and the
// shipped _xtim is built exactly as without it.
#include "qeccore/exact_scope.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "record_law_oracle.hpp"

namespace py = pybind11;
using namespace qeccore;
namespace ro = qeccore::record_oracle;

namespace {

py::array_t<uint8_t> u8(const std::vector<uint8_t>& v) {
    py::array_t<uint8_t> a((py::ssize_t)v.size());
    auto m = a.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i) m((py::ssize_t)i) = v[i];
    return a;
}

std::vector<uint8_t> vec_u8(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& a) {
    auto r = a.unchecked<1>();
    std::vector<uint8_t> v((size_t)r.shape(0));
    for (size_t i = 0; i < v.size(); ++i) v[i] = r((py::ssize_t)i);
    return v;
}

// Pauli from (qubits, codes) with codes 0 I, 1 X, 2 Y, 3 Z (Y carries phase 1: Hermitian).
Pauli pauli_of(int n, const std::vector<int>& qs, const std::vector<int>& codes) {
    if (qs.size() != codes.size()) throw std::invalid_argument("qubits / codes length mismatch");
    Pauli P(n);
    for (size_t i = 0; i < qs.size(); ++i) {
        const int q = qs[i], c = codes[i];
        if (q < 0 || q >= n) throw std::invalid_argument("qubit out of range");
        if (c == 1 || c == 2) P.flipx(q);
        if (c == 3 || c == 2) P.flipz(q);
    }
    return P;
}

Pauli pauli_xz(int n, const std::vector<uint8_t>& x, const std::vector<uint8_t>& z) {
    if ((int)x.size() != n || (int)z.size() != n) throw std::invalid_argument("x / z must have length n");
    Pauli P(n);
    for (int q = 0; q < n; ++q) { if (x[(size_t)q]) P.setx(q); if (z[(size_t)q]) P.setz(q); }
    return P;
}

py::tuple xz_of(const Pauli& P) {
    std::vector<uint8_t> x((size_t)P.n), z((size_t)P.n);
    for (int q = 0; q < P.n; ++q) { x[(size_t)q] = P.xbit(q); z[(size_t)q] = P.zbit(q); }
    return py::make_tuple(u8(x), u8(z));
}

py::array_t<uint8_t> bits_matrix(const std::vector<std::vector<uint64_t>>& rows, int width) {
    py::array_t<uint8_t> a({(py::ssize_t)rows.size(), (py::ssize_t)width});
    auto m = a.mutable_unchecked<2>();
    for (size_t r = 0; r < rows.size(); ++r)
        for (int j = 0; j < width; ++j)
            m((py::ssize_t)r, j) = ((size_t)j / 64 < rows[r].size()) ? ((rows[r][(size_t)j / 64] >> (j & 63)) & 1) : 0;
    return a;
}

struct PyOracle {
    std::shared_ptr<ro::Program> prog;

    explicit PyOracle(const std::string& text) {
        py::gil_scoped_release rel;
        prog = std::make_shared<ro::Program>(ro::compile(text));
    }

    std::vector<ro::Fault> faults_of(const std::vector<std::tuple<int, std::vector<int>, std::vector<int>>>& fs) const {
        std::vector<ro::Fault> out;
        for (const auto& f : fs) out.push_back({std::get<0>(f), pauli_of(prog->n, std::get<1>(f), std::get<2>(f))});
        return out;
    }

    py::tuple residual(const std::vector<std::tuple<int, std::vector<int>, std::vector<int>>>& fs,
                       py::object frame) const {
        std::unique_ptr<Pauli> F;
        if (!frame.is_none()) {
            auto fr = frame.cast<std::pair<std::vector<int>, std::vector<int>>>();
            F.reset(new Pauli(pauli_of(prog->n, fr.first, fr.second)));
        }
        ro::TermResidual R = ro::residual(*prog, faults_of(fs), F.get());
        py::tuple P = xz_of(R.P);
        return py::make_tuple(u8(R.S), P[0], P[1]);
    }

    py::dict law(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& S) const {
        ro::PlanLaw L;
        {
            std::vector<uint8_t> s = vec_u8(S);
            py::gil_scoped_release rel;
            L = ro::plan_law(*prog, s);
        }
        py::dict d;
        d["m"] = L.m; d["rho"] = L.rho; d["kappa"] = L.kappa;
        d["pivots"] = L.pivots; d["born"] = L.born;
        d["A"] = bits_matrix(L.A, L.rho);
        d["B"] = bits_matrix(L.B, L.kappa);
        d["beta"] = u8(L.beta);
        d["p0"] = L.p0;
        d["cls"] = u8(L.cls);
        d["s_sign"] = u8(L.s_sign);
        {
            const int k = prog->k;
            py::array_t<uint8_t> lam({(py::ssize_t)L.kappa, (py::ssize_t)(2 * k)});
            auto lm = lam.mutable_unchecked<2>();
            for (int j = 0; j < L.kappa; ++j)
                for (int i = 0; i < k; ++i) {
                    lm(j, i) = (uint8_t)((L.lam_a[(size_t)j] >> i) & 1);
                    lm(j, k + i) = (uint8_t)((L.lam_b[(size_t)j] >> i) & 1);
                }
            d["lam"] = lam;
            d["lam_phase"] = L.lam_ph;
        }
        return d;
    }

    py::tuple enumerate(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& S,
                        const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& t) const {
        std::vector<std::pair<uint64_t, double>> v;
        {
            std::vector<uint8_t> s = vec_u8(S), tt = vec_u8(t);
            py::gil_scoped_release rel;
            v = ro::enumerate_law(ro::plan_law(*prog, s), tt);
        }
        py::array_t<uint64_t> keys((py::ssize_t)v.size());
        py::array_t<double> probs((py::ssize_t)v.size());
        auto km = keys.mutable_unchecked<1>();
        auto pm = probs.mutable_unchecked<1>();
        for (size_t i = 0; i < v.size(); ++i) { km((py::ssize_t)i) = v[i].first; pm((py::ssize_t)i) = v[i].second; }
        return py::make_tuple(keys, probs);
    }

    py::array_t<uint8_t> relabel(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& x,
                                 const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& z) const {
        return u8(ro::relabel(*prog, pauli_xz(prog->n, vec_u8(x), vec_u8(z))));
    }

    py::array_t<uint8_t> frame_relabel(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& x,
                                       const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& z,
                                       const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& S) const {
        return u8(ro::frame_relabel(*prog, pauli_xz(prog->n, vec_u8(x), vec_u8(z)), vec_u8(S)));
    }

    py::object term_set_pauli(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& S) const {
        Pauli N;
        if (!ro::term_set_pauli(*prog, vec_u8(S), N)) return py::none();
        return xz_of(N);
    }

    py::list columns() const {
        py::list l;
        static const char* kinds[3] = {"detector", "observable", "decision"};
        for (const auto& c : prog->columns) l.append(py::make_tuple(kinds[c.kind], c.index, c.recs));
        return l;
    }

    py::list noise_sites() const {
        py::list l;
        for (const auto& s : prog->sites) {
            py::dict d;
            d["stream_index"] = s.stream_index;
            d["op_pos"] = s.op_pos;
            d["qubits"] = s.qubits;
            py::list outs;
            for (const auto& o : s.outcomes) outs.append(std::vector<int>(o.begin(), o.end()));
            d["outcomes"] = outs;
            d["probs"] = s.probs;
            l.append(d);
        }
        return l;
    }

    py::tuple terms() const {
        const int T = (int)prog->terms.size(), n = prog->n;
        py::array_t<uint8_t> xz({(py::ssize_t)T, (py::ssize_t)(2 * n)});
        py::array_t<int32_t> ph((py::ssize_t)T);
        auto m = xz.mutable_unchecked<2>();
        auto pm = ph.mutable_unchecked<1>();
        for (int j = 0; j < T; ++j) {
            for (int q = 0; q < n; ++q) {
                m(j, q) = prog->terms[(size_t)j].xbit(q);
                m(j, n + q) = prog->terms[(size_t)j].zbit(q);
            }
            pm(j) = prog->terms[(size_t)j].phase;
        }
        return py::make_tuple(xz, ph, prog->term_stream_index);
    }
};

}  // namespace

void register_family(py::module_& m);   // family_py.cpp (J-a N1)
void register_recb(py::module_& m);     // recb_py.cpp (J-a-b N6/N7)
void register_dem_census(py::module_& m);   // dem_census_py.cpp (EXP-fix: the DEM law census)
void register_compile(py::module_& m);      // compile_py.cpp (compile stage: construction-speed oracles)

PYBIND11_MODULE(_xtim_testapi, m) {
    m.doc() = "xtim test API (not a public interface): the S1 exact record-law oracle; the J-a family sampler.";
    register_family(m);
    register_recb(m);
    register_dem_census(m);
    register_compile(m);
    // J-b1 final review MAJOR-4: this extension links its OWN copy of the engine (exact_scope.cpp included), so its
    // exact constructions are counted here; xtim._exact_scope sums both modules and enters both scopes.
    m.def("_exact_constructions", []() { return qeccore::exact_constructions(); });
    m.def("_reset_exact_constructions", []() { qeccore::reset_exact_constructions(); });
    m.def("_explicit_exact_enter", []() { qeccore::explicit_exact_enter(); });
    m.def("_explicit_exact_exit", []() { qeccore::explicit_exact_exit(); });
    py::class_<PyOracle>(m, "RecordLawOracle",
        "S1 TEST API: the exact joint record law of a shot (design Theorem 1, term-indexed residual). "
        "Compiles the circuit like the TwirlSampler family; never used by a sampling path.")
        .def(py::init<const std::string&>(), py::arg("circuit_text"))
        .def_property_readonly("n", [](const PyOracle& o) { return o.prog->n; })
        .def_property_readonly("k", [](const PyOracle& o) { return o.prog->k; })
        .def_property_readonly("chi", [](const PyOracle& o) { return (int)o.prog->alpha.size(); })
        .def_property_readonly("num_terms", [](const PyOracle& o) { return (int)o.prog->terms.size(); })
        .def_property_readonly("num_ops", [](const PyOracle& o) { return (int)o.prog->ops.size(); })
        .def_property_readonly("input_qubits", [](const PyOracle& o) { return o.prog->input_qubits; })
        .def_property_readonly("rec_flip", [](const PyOracle& o) { return o.prog->rec_flip; })
        .def_property_readonly("rec_invert", [](const PyOracle& o) { return u8(o.prog->rec_invert); })
        .def_property_readonly("reference", [](PyOracle& o) { ro::ensure_reference(*o.prog); return u8(o.prog->reference); })
        .def_property_readonly("emission_offset", [](PyOracle& o) {
            ro::ensure_reference(*o.prog); return u8(o.prog->emission_offset); })
        .def_property_readonly("reads", [](const PyOracle& o) { return o.prog->reads; })
        .def("columns", &PyOracle::columns, "[(kind, index, records)] in column order: detectors, observables, decisions.")
        .def("noise_sites", &PyOracle::noise_sites,
             "Deferred noise sites: [{stream_index, op_pos, qubits, outcomes (codes 0I 1X 2Y 3Z), probs}] "
             "(identity outcome first; zero-probability outcomes dropped).")
        .def("terms", &PyOracle::terms,
             "(xz uint8[T, 2n], phase int32[T], stream_index[T]): the end-of-circuit signed term axes B_j "
             "(term j = the T-type factor exp(-i pi/8 A_j) of the deferred gate at stream_index[j]).")
        .def("residual", &PyOracle::residual, py::arg("faults"), py::arg("frame") = py::none(),
             "faults: [(op_pos, qubits, codes)], frame: (qubits, codes) at t=0 or None -> (S, Px, Pz): "
             "R = P * prod_{j in S} rot(B_j).")
        .def("law", &PyOracle::law, py::arg("S"),
             "Theorem 1 for C0 = rot(S): {m, rho, kappa, pivots, born, A, B, beta, p0, cls, s_sign} (P-free).")
        .def("enumerate", &PyOracle::enumerate, py::arg("S"), py::arg("t"),
             "Exact law of the RAW column vector: (keys uint64 (bit c = y_c), probs).")
        .def("relabel", &PyOracle::relabel, py::arg("x"), py::arg("z"), "Lemma 2: t_c = <P, M_c>.")
        .def("frame_relabel", &PyOracle::frame_relabel, py::arg("x"), py::arg("z"), py::arg("S"),
             "Section 2.7 relabel of a Pauli-image frame component on term set S.")
        .def("term_set_pauli", &PyOracle::term_set_pauli, py::arg("S"),
             "rot(S) as a Pauli (Nx, Nz) up to phase, or None when it is not a Pauli (S not in ker Phi).");
}
