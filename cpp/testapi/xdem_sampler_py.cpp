// xdem Task 6 test API (not a public interface): hooks into qeccore::xdem::XdemSampler's production code paths for
// tests/test_xdem_sampler.py — the noise walk's per-line fire counts, the status / law of a forced fired set, sampling
// with a forced fired set, the postselection span test's premise, and sampling with the test lever force_general;
// Task 7: score_lines with a mutation lever (xdem_lines_fixed); Task 8: k_optimal without the memo
// (xdem_k_optimal).
// Ported from the dc-DEM v1 sampler's test hooks (same arguments, without its plain-DEM flag). Registered in
// record_law_oracle_py.cpp (PYBIND11_MODULE _xtim_testapi). Reader refusals (qeccore::xdem::XdemFileError) surface as
// xtim.XtimXdemError (the exception translator _xtim registers is process-wide).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "qeccore/xdem_file.hpp"
#include "qeccore/xdem_sampler.hpp"

namespace py = pybind11;
namespace xd = qeccore::xdem;

namespace {

// A bit vector as a Python int (bit j of word w = bit 64 w + j).
py::int_ bits_int(const std::vector<uint64_t>& w) {
    const std::string s = xd::to_hex(xd::Bits{w});
    return py::int_(py::reinterpret_steal<py::object>(PyLong_FromString(s.c_str(), nullptr, 16)));
}

py::list cspace_rows(const xd::XdemSampler& S, int id) {
    py::list rows;
    const size_t WO = S.obs_words();
    const auto& c = S.cspaces()[(size_t)id];
    const size_t dim = WO ? c.size() / WO : 0;
    for (size_t i = 0; i < dim; ++i)
        rows.append(bits_int(std::vector<uint64_t>(c.begin() + (long)(i * WO), c.begin() + (long)((i + 1) * WO))));
    return rows;
}

py::list bits_list(const std::vector<xd::Bits>& v) {
    py::list l;
    for (const auto& b : v) l.append(bits_int(b.w));
    return l;
}

}  // namespace

void register_xdem_sampler_hooks(py::module_& m) {
    using u8arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
    using i32arr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
    using i64arr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
    m.def("xdem_fire_counts", [](const std::string& text, long long shots, uint64_t seed,
                                 const std::vector<std::pair<int, int>>& pairs, uint64_t first_block) {
        xd::XdemSampler S(xd::parse(text));
        std::vector<uint64_t> counts, pc;
        {
            py::gil_scoped_release rel;
            S.fire_counts((size_t)shots, seed, first_block, counts, pairs, pc);
        }
        py::array_t<uint64_t> c((py::ssize_t)counts.size());
        for (size_t i = 0; i < counts.size(); ++i) c.mutable_data()[i] = counts[i];
        return py::make_tuple(c, pc);
    }, py::arg("text"), py::arg("shots"), py::arg("seed"), py::arg("pairs") = std::vector<std::pair<int, int>>{},
       py::arg("first_block") = 0,
       "TEST API: (per-line fire counts uint64[num_lines], [#shots where both lines of pair q fired]) over `shots` "
       "shots of the production noise walk (XdemSampler stream, seed, first_block).");

    m.def("xdem_status_of", [](const std::string& text, const std::vector<int>& line_ids, bool force_general) {
        xd::XdemSampler S(xd::parse(text), xd::kDefaultCacheCap, force_general);
        xd::Bits off;
        std::vector<xd::Bits> coins;
        const int id = S.status_of(line_ids, off, coins);
        py::dict d;
        d["cspace"] = cspace_rows(S, id);   // C basis rows as O-bit ints (observable o = bit o)
        d["dim"] = S.cspace_dim(id);
        d["offset"] = bits_int(off.w);      // the sampler's law (File layout: detector d = bit d, obs o = bit D+o)
        d["coins"] = bits_list(coins);
        d["fast_path"] = S.fast_path();
        return d;
    }, py::arg("text"), py::arg("line_ids"), py::arg("force_general") = false,
       "TEST API: the production status of a forced fired multiset (any order, repeats cancel): {cspace: [C basis row "
       "ints over the O observables], dim, offset, coins, fast_path} (offset / coins: the law the sampler uses, D+O-bit "
       "ints, not reduced).");

    m.def("xdem_law_of", [](const std::string& text, const std::vector<int>& line_ids, bool force_general) {
        xd::XdemSampler S(xd::parse(text), xd::kDefaultCacheCap, force_general);
        xd::Bits off;
        std::vector<xd::Bits> coins;
        const int dim = S.law_of(line_ids, off, coins);
        py::dict d;
        d["dim"] = dim;
        d["offset"] = bits_int(off.w);
        d["coins"] = bits_list(coins);
        d["fast_path"] = S.fast_path();
        return d;
    }, py::arg("text"), py::arg("line_ids"), py::arg("force_general") = false,
       "TEST API: the law of a forced multiset with no coin draw (fast path: from the kernel basis of N_O): {dim (C), "
       "offset, coins (D+O-bit ints, not reduced), fast_path}; must equal xdem_status_of's law.");

    m.def("xdem_sample_fixed", [](const std::string& text, const std::vector<int>& line_ids, long long shots,
                                  uint64_t seed, bool force_general) {
        xd::XdemSampler S(xd::parse(text), xd::kDefaultCacheCap, force_general);
        const size_t n = (size_t)shots;
        py::array_t<uint8_t> dets({(py::ssize_t)n, (py::ssize_t)S.det_bytes()});
        py::array_t<uint8_t> obs({(py::ssize_t)n, (py::ssize_t)S.obs_bytes()});
        py::array_t<int32_t> cs({(py::ssize_t)n});
        uint8_t* dp = dets.mutable_data();
        uint8_t* op = obs.mutable_data();
        int32_t* cp = cs.mutable_data();
        {
            py::gil_scoped_release rel;
            S.sample_fixed(line_ids, n, seed, dp, op, cp);
        }
        py::list cl;
        for (size_t i = 0; i < S.cspaces().size(); ++i) cl.append(cspace_rows(S, (int)i));
        return py::make_tuple(dets, obs, cs, cl);
    }, py::arg("text"), py::arg("line_ids"), py::arg("shots"), py::arg("seed"), py::arg("force_general") = false,
       "TEST API: sample with every shot firing exactly line_ids (production composition + law + coins + status): "
       "(dets, obs, cspace, [C basis row ints per table entry]).");

    m.def("xdem_sample_ps", [](const std::string& text, long long shots, uint64_t seed, int threads,
                               const std::vector<uint8_t>& flags, bool force_general) {
        // two fresh samplers: the postselected call and the plain call (the tests compare their laws)
        xd::XdemSampler S1(xd::parse(text), xd::kDefaultCacheCap, force_general);
        xd::XdemSampler S2(xd::parse(text), xd::kDefaultCacheCap, force_general);
        const size_t n = (size_t)shots;
        auto run = [&](bool ps) {
            xd::XdemSampler& S = ps ? S1 : S2;
            py::array_t<uint8_t> dets({(py::ssize_t)n, (py::ssize_t)S.det_bytes()});
            py::array_t<uint8_t> obs({(py::ssize_t)n, (py::ssize_t)S.obs_bytes()});
            py::array_t<int32_t> cs({(py::ssize_t)n});
            uint8_t* dp = dets.mutable_data();
            uint8_t* op = obs.mutable_data();
            int32_t* cp = cs.mutable_data();
            xd::XdemPsCounts k;
            {
                py::gil_scoped_release rel;
                if (ps) S.sample_ps(n, seed, threads, flags, dp, op, cp, k, 0);
                else S.sample(n, seed, threads, dp, op, cp, 0);
            }
            const py::ssize_t m = ps ? (py::ssize_t)k.passed : (py::ssize_t)n;
            dets.resize({m, (py::ssize_t)S.det_bytes()}, false);
            obs.resize({m, (py::ssize_t)S.obs_bytes()}, false);
            cs.resize({m}, false);
            return py::make_tuple(dets, obs, cs, k.early);
        };
        py::tuple a = run(true);
        py::tuple b = run(false);
        auto dims = [](const xd::XdemSampler& S) {
            py::list l;
            for (size_t i = 0; i < S.num_cspaces(); ++i) l.append(S.cspace_dim((int)i));
            return l;
        };
        return py::make_tuple(a, b, S1.fast_path(), dims(S1), dims(S2));
    }, py::arg("text"), py::arg("shots"), py::arg("seed"), py::arg("threads"), py::arg("flags"),
       py::arg("force_general") = false,
       "TEST API: fresh samplers (force_general: the cached exact-law path): sample_ps(shots, seed, threads, flags) "
       "-> (dets, obs, cspace, early) of the survivors (C++ sample_ps for every mask, the empty one included), and on "
       "a second fresh sampler sample(shots, seed, threads) -> (dets, obs, cspace, 0); returns (ps, plain, fast_path, "
       "dims_ps, dims_plain), dims_* = the dim of each C id of that sampler's table.");

    m.def("xdem_span_check", [](const std::string& text, long long shots, uint64_t seed, bool force_general) {
        xd::XdemSampler S(xd::parse(text), xd::kDefaultCacheCap, force_general);
        uint64_t xs = 0;
        uint64_t bad;
        {
            py::gil_scoped_release rel;
            bad = S.span_check((size_t)shots, seed, &xs);
        }
        return py::make_tuple(bad, xs);
    }, py::arg("text"), py::arg("shots"), py::arg("seed"), py::arg("force_general") = false,
       "TEST API: (shots whose correction changes the detector row outside span(img)_D, shots with an extended line) "
       "over the stream's first shots (the postselection span test's premise).");

    m.def("xdem_lines_fixed", [](const std::string& text, const i64arr& fo, const i32arr& fi, const u8arr& dets,
                                 const i64arr& ho, const i32arr& hi, const std::string& correction, int mut,
                                 bool force_general) {
        // explicit (fired CSR, detector rows, hypothesis CSR) triples through score_lines, with a mutation lever;
        // force_general: the sampler's general path (and "optimal"'s generic law_into for [T, K])
        xd::XdemSampler S(xd::parse(text), xd::kDefaultCacheCap, force_general);
        const size_t n = (size_t)dets.shape(0);
        if (dets.ndim() != 2 || (size_t)dets.shape(1) != S.det_bytes() || (size_t)fo.size() != n + 1 ||
            (size_t)ho.size() != n + 1)
            throw std::invalid_argument("xdem_lines_fixed: shapes");
        xd::XdemLinesOptions opt;
        if (correction == "none") opt.corr = xd::Correction::None;
        else if (correction == "r2b") opt.corr = xd::Correction::R2b;
        else if (correction == "optimal") opt.corr = xd::Correction::Optimal;
        else throw std::invalid_argument("xdem_lines_fixed: correction");
        opt.mut = mut;
        const int O = S.num_observables();
        py::array_t<double> po((py::ssize_t)n);
        py::array_t<int8_t> ao((py::ssize_t)n);
        py::array_t<uint8_t> co({(py::ssize_t)n, (py::ssize_t)O});
        double* pp = po.mutable_data();
        int8_t* ap = ao.mutable_data();
        uint8_t* cp = co.mutable_data();
        {
            py::gil_scoped_release rel;
            S.score_lines(n, dets.data(), fo.data(), fi.data(), (size_t)fi.size(), ho.data(), hi.data(),
                          (size_t)hi.size(), pp, opt, ap, cp);
        }
        const auto k = S.counters();
        py::dict cn;
        cn["lines_lossy"] = k.lines_lossy;
        cn["lines_outside"] = k.lines_outside;
        cn["lines_coin_after"] = k.lines_coin_after;
        cn["fast_path"] = S.fast_path();
        cn["kfast_laws"] = k.kfast_laws;
        return py::make_tuple(po, ao, co, cn);
    }, py::arg("text"), py::arg("fired_offsets"), py::arg("fired_ids"), py::arg("dets"), py::arg("offsets"),
       py::arg("ids"), py::arg("correction"), py::arg("mut") = 0, py::arg("force_general") = false,
       "TEST API: score_lines on explicit (fired CSR, detector rows uint8[n, ceil(D/8)], hypothesis CSR) with the "
       "mutation lever mut (0 = none; kXdemMut*) and the test lever force_general: (pass float64[n], action int8[n] "
       "(0 none, 1 applied, 2 withheld), c uint8[n, O], counters dict).");

    m.def("xdem_correction", [](const std::string& text, const u8arr& dets, const i64arr& ho, const i32arr& hi,
                                const std::string& correction, bool force_general) {
        // XdemSampler::correction on a fresh sampler (force_general: the general path), every export member as a list
        xd::XdemSampler S(xd::parse(text), xd::kDefaultCacheCap, force_general);
        const size_t n = (size_t)dets.shape(0);
        if (dets.ndim() != 2 || (size_t)dets.shape(1) != S.det_bytes() || (size_t)ho.size() != n + 1)
            throw std::invalid_argument("xdem_correction: shapes");
        const xd::Correction c = correction == "r2b" ? xd::Correction::R2b : xd::Correction::Optimal;
        xd::XdemCorrectionOut o;
        {
            py::gil_scoped_release rel;
            S.correction(n, dets.data(), ho.data(), hi.data(), (size_t)hi.size(), c, o);
        }
        return py::make_tuple(o.axes_off, o.axes, o.piv_off, o.piv, o.y, o.frame, o.lossy, o.late_off, o.late_axis,
                              o.late_eta, o.tau_off, o.tau_axis, o.applied, o.c);
    }, py::arg("text"), py::arg("dets"), py::arg("offsets"), py::arg("ids"), py::arg("correction"),
       py::arg("force_general") = false,
       "TEST API: XdemSampler::correction(\"r2b\" or \"optimal\") on a fresh sampler, with the test lever force_general: "
       "the XdemCorrectionOut members as lists (axes_off, axes, piv_off, piv, y, frame, lossy, late_off, late_axis, "
       "late_eta, tau_off, tau_axis, applied, c).");

    m.def("xdem_k_optimal", [](const std::string& text, const std::vector<int32_t>& ids, const py::int_& s) {
        // XdemSampler::k_optimal (no memo) for one hypothesis and one detector row s (a Python int, detector d = bit d)
        xd::XdemSampler S(xd::parse(text));
        const int D = S.num_detectors();
        std::vector<uint64_t> d((size_t)(D + 63) / 64 + 1, 0);
        py::int_ v = s;
        const py::int_ mask(0xffffffffffffffffull);
        for (size_t w = 0; w < d.size(); ++w) {
            d[w] = py::cast<uint64_t>(v & mask);
            v = py::int_(v.attr("__rshift__")(64));
        }
        std::vector<xd::KObs> out;
        S.k_optimal(ids.data(), ids.size(), d.data(), out);
        py::list rows;
        for (const auto& o : out) {
            py::dict r;
            r["lossy"] = o.lossy;
            r["P"] = o.P;
            r["S"] = o.S;
            r["y"] = (int)o.y;
            r["frame"] = (int)o.frame;
            rows.append(r);
        }
        return rows;
    }, py::arg("text"), py::arg("ids"), py::arg("s"),
       "TEST API: XdemSampler::k_optimal (SPEC §4.3 steps 1-5, from scratch, no memo): per observable {lossy, P "
       "(detector ids), S (axis ids), y, frame}.");

    m.def("xdem_sample", [](const std::string& text, std::vector<std::pair<long long, uint64_t>> calls, int threads,
                            bool force_general, size_t cache_cap) {
        xd::XdemSampler S(xd::parse(text), cache_cap, force_general);
        py::list out;
        for (const auto& [shots, seed] : calls) {
            const size_t n = (size_t)shots;
            py::array_t<uint8_t> dets({(py::ssize_t)n, (py::ssize_t)S.det_bytes()});
            py::array_t<uint8_t> obs({(py::ssize_t)n, (py::ssize_t)S.obs_bytes()});
            py::array_t<int32_t> cs({(py::ssize_t)n});
            uint8_t* dp = dets.mutable_data();
            uint8_t* op = obs.mutable_data();
            int32_t* cp = cs.mutable_data();
            {
                py::gil_scoped_release rel;
                S.sample(n, seed, threads, dp, op, cp, 0);
            }
            out.append(py::make_tuple(dets, obs, cs));
        }
        py::list cl;
        for (size_t i = 0; i < S.cspaces().size(); ++i) cl.append(cspace_rows(S, (int)i));
        const auto c = S.counters();
        py::dict k;
        k["cache_hits"] = c.cache_hits;
        k["cache_misses"] = c.cache_misses;
        k["cache_clears"] = c.cache_clears;
        k["cache_entries"] = c.cache_entries;
        k["thread_slots"] = c.thread_slots;
        k["ext_shots"] = c.ext_shots;
        k["fast_path"] = S.fast_path();
        return py::make_tuple(out, cl, k);
    }, py::arg("text"), py::arg("calls"), py::arg("threads") = 1, py::arg("force_general") = false,
       py::arg("cache_cap") = xd::kDefaultCacheCap,
       "TEST API: one sampler (force_general: the cached exact-law path), sample(shots, seed) per call in order: "
       "([(dets, obs, cspace)], [C basis row ints per table entry], counters).");
}
