// record_buffer.hpp — J-b1 P12: the layout-3 BarrierBuffer and the RecordFidelity evaluator (design §4; plan r2.3 /
// r3.3), as ONE source the production binding (at F) and the test API both compile. Every class is registered
// py::module_local() (bind_record_buffer), so the two extensions never share a type; the Python readers are
// duck-typed and pick the evaluator from the buffer's own module (`_engine_module()`).
//
// Storage is the FAM sampler's layout-3 FamilyRecordSink (dense per-content handles, P_rec words, y words, the
// outcomes u ‖ z in a RESERVED [rows, UZW] block, frame_folded): every per-shot write is into preallocated storage
// (the profile's per-shot buffer fixes: reserved outcome storage, dense per-plan ids; no σ anywhere). The readers
// that need the producing sampler (materialize, signs, frames, plan views, the fidelity) go through RecordReader,
// which the binding implements on its sampler wrapper and which keeps that sampler alive.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/record_fidelity.hpp"
#include "qeccore/twirl_sampler.hpp"
#include "twirl_binding_access.hpp"
#include "record_api.hpp"

namespace xtim_api {
namespace py = pybind11;
using qeccore::FramedSuperposition;
using qeccore::Pauli;
using BA_ = qeccore::TwirlSamplerBindingAccess;   // J-b1 F: the public bindings' door (was the test access)

// J-b1 (final review): raise xtim.errors.XtimTwirlRefusal(message, class, sub) — a typed refusal by name.
[[noreturn]] inline void raise_twirl_refusal(const std::string& msg, const char* cls, const char* sub) {
    py::object exc = py::module_::import("xtim.errors").attr("XtimTwirlRefusal")(msg, cls, sub);
    PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(exc.ptr())), exc.ptr());
    throw py::error_already_set();
}

// J-a N5: the layout-3 records of one FAM run (rows = shots), owned; the sink views them.
struct LayoutRecords {
    long rows = 0;
    int n = 0, m = 0;
    BA_::FamilyRecordSink sink;
    std::vector<int32_t> handle, nuz, branch;          // J-b1 P9b: branch = the row's engine branch id (0: ideal)
    std::vector<uint8_t> route, uz, ff;
    std::vector<uint64_t> prec, y;
    std::vector<BA_::FamilyRecordContent> contents;   // snapshot after the run
    std::string model;                                               // the producer's family_model_key()
    uint64_t model_serial = 0;                       // J-b1 F (fixplan r2 item 3): the producing sampler's serial

    LayoutRecords(long shots, int n_, int m_, int YW) : rows(shots), n(n_), m(m_) {
        const int PNW = (n + 63) / 64;
        handle.assign((size_t)shots, -1); nuz.assign((size_t)shots, 0); branch.assign((size_t)shots, 0);
        route.assign((size_t)shots, 0); ff.assign((size_t)shots, 0);
        uz.assign((size_t)shots * (size_t)std::max(m, 1), 0);
        prec.assign((size_t)shots * 2 * (size_t)PNW, 0);
        y.assign((size_t)shots * (size_t)std::max(YW, 1), 0);
        sink.cap = shots; sink.row = 0;
        sink.PNW = PNW; sink.YW = YW; sink.UZW = std::max(m, 1);
        sink.handle = handle.data(); sink.route = route.data(); sink.prec = prec.data(); sink.y = y.data();
        sink.uz = uz.data(); sink.nuz = nuz.data(); sink.ff = ff.data(); sink.branch = branch.data();
    }
    void check(long i) const { if (i < 0 || i >= rows) throw std::out_of_range("record index out of range"); }
    void rebind_() {                                  // re-point the sink at this object's storage (after a move)
        sink.handle = handle.data(); sink.route = route.data(); sink.prec = prec.data(); sink.y = y.data();
        sink.uz = uz.data(); sink.nuz = nuz.data(); sink.ff = ff.data(); sink.branch = branch.data();
    }
    // J-b1 P9b: the readers that key or export the retained state on branch 0's bare (frames, signs, the record
    // groups and keys, fidelity, the branch export) do not serve carried-branch rows: refused by name (materialize
    // serves them: the branch's own consumer bare).
    void no_branch_rows(const char* what) const {
        for (long i = 0; i < sink.row && i < rows; ++i)
            if (branch[(size_t)i] != 0)
                raise_twirl_refusal(std::string(what) + ": record " + std::to_string(i) + " is a carried-branch row "
                                    "(input_branch != 0); in J-b1 only materialize reads branch rows; law limitation "
                                    "(carried axis, branch_row), served after J-c; refused",
                                    "LAW_LIMITATION_CARRIED_AXIS", "branch_row");   // J-b1 final review m2: typed
    }
    // J-b1 P12: a copy holding the listed rows only (same contents table and model), for subset readers.
    LayoutRecords gather(const int64_t* ix, long k) const {
        LayoutRecords o(k, n, m, sink.YW);
        const size_t W = 2 * (size_t)sink.PNW;
        for (long j = 0; j < k; ++j) {
            const long i = (long)ix[j];
            check(i);
            o.handle[(size_t)j] = handle[(size_t)i]; o.nuz[(size_t)j] = nuz[(size_t)i];
            o.route[(size_t)j] = route[(size_t)i]; o.ff[(size_t)j] = ff[(size_t)i]; o.branch[(size_t)j] = branch[(size_t)i];
            std::memcpy(&o.uz[(size_t)j * sink.UZW], &uz[(size_t)i * sink.UZW], (size_t)sink.UZW);
            std::memcpy(&o.prec[(size_t)j * W], &prec[(size_t)i * W], W * sizeof(uint64_t));
            std::memcpy(&o.y[(size_t)j * sink.YW], &y[(size_t)i * sink.YW], (size_t)sink.YW * sizeof(uint64_t));
        }
        o.sink.row = k;
        o.contents = contents;
        o.model = model;
        o.model_serial = model_serial;
        return o;
    }
    const uint64_t* prec_row(long i) const { return prec.data() + (size_t)i * 2 * (size_t)sink.PNW; }
    const uint64_t* y_row(long i) const { return y.data() + (size_t)i * (size_t)sink.YW; }
    py::array_t<uint8_t> y_bits(long i) const {
        check(i);
        py::array_t<uint8_t> a((py::ssize_t)m);
        for (int c = 0; c < m; ++c) a.mutable_data()[c] = (uint8_t)((y_row(i)[c >> 6] >> (c & 63)) & 1ULL);
        return a;
    }
    py::array_t<uint8_t> outcomes(long i) const {
        check(i);
        py::array_t<uint8_t> a((py::ssize_t)nuz[(size_t)i]);
        if (nuz[(size_t)i]) std::memcpy(a.mutable_data(), uz.data() + (size_t)i * (size_t)sink.UZW, (size_t)nuz[(size_t)i]);
        return a;
    }
    py::tuple prec_bits(long i) const {
        check(i);
        py::array_t<uint8_t> x((py::ssize_t)n), z((py::ssize_t)n);
        const uint64_t* p = prec_row(i);
        for (int q = 0; q < n; ++q) {
            x.mutable_data()[q] = (uint8_t)((p[q >> 6] >> (q & 63)) & 1ULL);
            z.mutable_data()[q] = (uint8_t)((p[sink.PNW + (q >> 6)] >> (q & 63)) & 1ULL);
        }
        return py::make_tuple(x, z);
    }
    const BA_::FamilyRecordContent& content_of(long i) const {
        check(i);
        const int h = handle[(size_t)i];
        if (h < 0 || h >= (int)contents.size()) throw std::logic_error("record: no content (the run was refused?)");
        return contents[(size_t)h];
    }
    // The layout-3 key: [kind][len u32 LE] content ‖ P_rec words (x then z, LE) ‖ y words (LE).
    py::bytes key(long i) const {
        const auto& C = content_of(i);
        std::string k;
        k.push_back((char)C.kind);
        const uint32_t L = (uint32_t)C.content.size();
        for (int b = 0; b < 4; ++b) k.push_back((char)((L >> (8 * b)) & 0xFF));
        k += C.content;
        k.append(reinterpret_cast<const char*>(prec_row(i)), 2 * (size_t)sink.PNW * sizeof(uint64_t));
        k.append(reinterpret_cast<const char*>(y_row(i)), (size_t)sink.YW * sizeof(uint64_t));
        return py::bytes(k);
    }
    // The plan of a record for a dense oracle: kind 0 {a, cz}; kind 2 {S}.
    py::dict plan(long i) const {
        const auto& C = content_of(i);
        py::dict d;
        d["kind"] = C.kind;
        d["digest"] = py::make_tuple(C.digest_hi, C.digest_lo);
        if (C.kind == 0) {
            std::vector<int> a;
            for (int q = 0; q < n; ++q) if (C.content[(size_t)q]) a.push_back(q);
            py::list cz;
            for (size_t j = (size_t)n + 1; j + 3 < C.content.size(); j += 4) {
                const int f = (unsigned char)C.content[j] | ((unsigned char)C.content[j + 1] << 8);
                const int g = (unsigned char)C.content[j + 2] | ((unsigned char)C.content[j + 3] << 8);
                cz.append(py::make_tuple(f, g));
            }
            d["a"] = a;
            d["cz"] = cz;
        } else {
            std::vector<int> S;
            const uint64_t* w = reinterpret_cast<const uint64_t*>(C.content.data());
            for (int j = 0; j < (int)(C.content.size() * 8); ++j) if ((w[j >> 6] >> (j & 63)) & 1ULL) S.push_back(j);
            d["S"] = S;
        }
        return d;
    }
};

// The sampler-side readers a buffer needs (implemented by the binding on its sampler wrapper).
struct RecordReader {
    virtual ~RecordReader() = default;
    virtual const std::string& model() const = 0;                             // the state model key (bit-exact)
    // J-b1 F (fixplan-F-materialize-c0hat r2 items 1-2): shot i's retained state, the VALUE route (the gate-level
    // route on the frame tableau for branch-0 rows; C0̂ for carried-branch rows, under its capacity refusal).
    virtual FramedSuperposition state(const LayoutRecords& R, long i) const = 0;
    virtual py::array_t<int8_t> signs(const LayoutRecords& R, const py::list& paulis) const = 0;
    virtual py::dict frames(const LayoutRecords& R) const = 0;
    virtual py::dict record_layout() const = 0;
    virtual void fidelity(qeccore::RecordFidelity& rf, const LayoutRecords& R, double* out,
                          const uint8_t* corr) const = 0;   // corr: uint8[rows, 2, n_out] or nullptr
    virtual const FramedSuperposition& bare() const = 0;
    virtual std::vector<Pauli> record_ops() const = 0;
    virtual int n_out() const = 0;                                            // OUTPUT_QUBITS
    virtual std::vector<int> out_wires() const = 0;
    virtual std::vector<int> read_meas() const = 0;                          // per deferred wire: its measurement, or −1
    virtual std::vector<int> wire_user() const = 0;                          // per deferred wire: its user qubit, or −1
    // J-b1 P12 (ruling on fixplan-p12-review: the b9c2e92 BarrierBuffer methods implemented on layout 3)
    virtual py::dict plan_structure(const LayoutRecords& R, long i) const = 0;  // the record's plan (no σ fields)
    virtual py::dict born_dec() const = 0;                                    // the Born-measured decision operators
    virtual std::vector<std::complex<double>> expectations(const LayoutRecords& R, long i,
                                                           const std::vector<Pauli>& ps) const = 0;
};

// J-b1 F (fixplan r2 item 1): materialize / state return the extension's FramedSuperposition HANDLE, as b9c2e92
// does (no eager array export; branch_frames() is the handle's own method). The handle type is py::module_local and
// lives in the TU that binds it (framed_py.hpp), so that TU installs its wrapper here once at module init.
using WrapStateFn = py::object (*)(FramedSuperposition&&);
inline WrapStateFn& wrap_state_hook() {
    static WrapStateFn f = nullptr;
    return f;
}
inline py::object wrap_state(FramedSuperposition&& st) {
    if (!wrap_state_hook()) throw std::logic_error("BarrierBuffer.materialize: no FramedSuperposition handle type "
                                                   "is bound in this extension (binding bug)");
    return wrap_state_hook()(std::move(st));
}

// The layout-3 barrier buffer of one sample_barrier run.
struct BarrierBuffer {
    std::shared_ptr<LayoutRecords> R;
    std::shared_ptr<RecordReader> rd;
    py::object dets_, obs_, dec_;                     // the run's packed columns (as sample())
    mutable py::object frames_;                       // frames() memo (the buffer is immutable)
    mutable std::vector<int32_t> gid_;                // record_group_ids memo
    mutable std::vector<std::string> gkey_;
    py::object module_;                               // the extension that built this buffer

    long num_shots() const { return R->rows; }
    void build_groups_() const {
        if (!gid_.empty() || R->rows == 0) return;
#ifndef XTIM_MUTATE_F2_GROUPS_BRANCH
        // J-b1 F (fixplan-F-branch-readers F-2): the layout-3 key has no branch id, so a branch row and a branch-0
        // row with equal keys would share a group although their states differ. Refused by name (J-c serves it).
        R->no_branch_rows("record groups");
#endif
        std::unordered_map<std::string, int32_t> ix;
        gid_.resize((size_t)R->rows);
        for (long i = 0; i < R->rows; ++i) {
            std::string k = R->key(i);                // py::bytes → std::string
            auto it = ix.find(k);
            if (it == ix.end()) { it = ix.emplace(k, (int32_t)gkey_.size()).first; gkey_.push_back(k); }
            gid_[(size_t)i] = it->second;
        }
    }
    py::array_t<int32_t> record_group_ids() const {
        build_groups_();
        py::array_t<int32_t> a((py::ssize_t)R->rows);
        if (R->rows) std::memcpy(a.mutable_data(), gid_.data(), (size_t)R->rows * sizeof(int32_t));
        return a;
    }
    py::tuple record_groups() const {               // (layout-3 keys of the groups, counts), first-appearance order
        build_groups_();
        py::list keys;
        for (const auto& k : gkey_) keys.append(py::bytes(k));
        py::array_t<int64_t> cnt((py::ssize_t)gkey_.size());
        std::fill(cnt.mutable_data(), cnt.mutable_data() + gkey_.size(), 0);
        for (int32_t g : gid_) cnt.mutable_data()[g] += 1;
        return py::make_tuple(keys, cnt);
    }
    py::dict frames() const {
        if (frames_.is_none() || !frames_) frames_ = rd->frames(*R);
        return frames_.cast<py::dict>();
    }
};

// RecordFidelity on layout-3 records (the REGISTER fidelity on OUTPUT_QUBITS; strict contract checked at construction):
// the evaluator reads each shot's own record (plan content, P_rec, y) from the buffer; it never takes caller-supplied
// bits. correction: a per-shot Pauli on the OUTPUT qubits,
// uint8[rows, 2, n_out] (X mask row, Z mask row; OUTPUT_QUBITS declaration order), applied before the fidelity.
struct RecordFidelityL3 {
    std::unique_ptr<qeccore::RecordFidelity> rf;
    std::shared_ptr<RecordReader> rd;
    std::string model;
    size_t cache_budget = 0;            // termroute-a: held bytes above it are dropped at an evaluate() call (0: never)
    RecordFidelityL3(std::shared_ptr<RecordReader> reader, size_t max_bytes, size_t cache_budget_ = 0)
        : rd(std::move(reader)), cache_budget(cache_budget_) {
        rf = std::make_unique<qeccore::RecordFidelity>(rd->bare(), rd->record_ops(), rd->out_wires(), rd->read_meas(),
                                                       max_bytes, rd->wire_user());
        model = rd->model();
    }
    // shots (optional): the row indices to evaluate (e.g. the post-selected ones); the output has one entry per
    // listed row. The listed rows are gathered into a temporary layout-3 view (O(listed) copies).
    py::array_t<double> evaluate(const BarrierBuffer& B, const py::object& correction, const py::object& shots) const {
        if (!shots.is_none()) {
            auto ix = py::cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>(shots);
            auto sub = std::make_shared<LayoutRecords>(B.R->gather(ix.data(), (long)ix.size()));
            sub->rebind_();
            BarrierBuffer V;
            V.R = sub;
            V.rd = B.rd;
            py::object corr2 = correction;
            if (!correction.is_none()) {
                auto c = py::cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(correction);
                if (c.ndim() != 3 || c.shape(0) != B.R->rows)
                    throw std::invalid_argument("RecordFidelity: correction must be uint8[shots, 2, n_out] over ALL "
                                                "shots of the buffer — refused (shape mismatch)");
                py::array_t<uint8_t> g({(py::ssize_t)ix.size(), c.shape(1), c.shape(2)});
                const size_t row = (size_t)c.shape(1) * (size_t)c.shape(2);
                for (py::ssize_t k = 0; k < ix.size(); ++k)
                    std::memcpy(g.mutable_data() + (size_t)k * row, c.data() + (size_t)ix.data()[k] * row, row);
                corr2 = g;
            }
            return evaluate(V, corr2, py::none());
        }
        if (B.R->model != model || B.rd->model() != model)
            throw std::invalid_argument("RecordFidelity: the buffer comes from a different state model (bare state, "
                                        "certified generators, record operators with their folds) — refused");
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> cref;
        if (!correction.is_none()) {
            cref = py::cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(correction);
            const int no = (int)rd->out_wires().size();
            if (cref.ndim() != 3 || cref.shape(0) != B.R->rows || cref.shape(1) != 2 || cref.shape(2) != no)
                throw std::invalid_argument("RecordFidelity: correction must be uint8[shots, 2, n_out] (X and Z masks "
                                            "on OUTPUT_QUBITS) — refused (shape mismatch)");
            for (py::ssize_t i = 0; i < cref.size(); ++i)
                if (cref.data()[i] > 1)
                    throw std::invalid_argument("RecordFidelity: correction entries must be 0 / 1 (entry " +
                                                std::to_string(i) + " is " + std::to_string((int)cref.data()[i]) +
                                                ") — refused");
        }
        py::array_t<double> out((py::ssize_t)B.R->rows);
        rf->trim_cache(cache_budget);       // termroute-a: between calls only (no content id is live)
        rd->fidelity(*rf, *B.R, out.mutable_data(), correction.is_none() ? nullptr : cref.data());
        return out;
    }
};

// Registers the shared classes on `m` (module_local: each extension owns its copy of the type).
inline void bind_record_buffer(py::module_& m) {
    py::class_<BarrierBuffer>(m, "BarrierBuffer", py::module_local(),
        "J-b1 P12: the layout-3 barrier buffer (design §4): per shot the record key [kind][len] content ‖ P_rec ‖ y, "
        "the outcomes u ‖ z, frame_folded; no σ (sigmas / sig_bits / plan_key / exact_flags do not exist).")
        .def_property_readonly("num_shots", &BarrierBuffer::num_shots)
        .def("dets", [](const BarrierBuffer& b) { return b.dets_; })
        .def("obs", [](const BarrierBuffer& b) { return b.obs_; })
        .def("decisions", [](const BarrierBuffer& b) { return b.dec_; })
        .def("record_keys", [](const BarrierBuffer& b) {
#ifndef XTIM_MUTATE_F2_GROUPS_BRANCH
            b.R->no_branch_rows("record_keys");       // J-b1 F (F-2): the key carries no branch id
#endif
            py::list l;
            for (long i = 0; i < b.R->rows; ++i) l.append(b.R->key(i));
            return l; })
        .def("record_group_ids", &BarrierBuffer::record_group_ids)
        .def("record_groups", &BarrierBuffer::record_groups)
        .def("outcomes", [](const BarrierBuffer& b, long i) { return b.R->outcomes(i); }, py::arg("i"))
        .def("record_outcomes", [](const BarrierBuffer& b, long i) { return b.R->y_bits(i); }, py::arg("i"))
        .def("frame_folded", [](const BarrierBuffer& b, long i) { b.R->check(i); return (int)b.R->ff[(size_t)i]; },
             py::arg("i"))
        .def("plan_content", [](const BarrierBuffer& b, long i) {
            const auto& C = b.R->content_of(i);
            return py::make_tuple(plan_content_bytes(C.kind, C.content), digest_bytes(C.digest_hi, C.digest_lo));
        }, py::arg("i"))
        .def("record_layout", [](const BarrierBuffer& b) { return b.rd->record_layout(); })
        .def("frames", &BarrierBuffer::frames)
        .def("signs", [](const BarrierBuffer& b, const py::list& p) { return b.rd->signs(*b.R, p); }, py::arg("paulis"))
        .def("materialize", [](const BarrierBuffer& b, long i) { b.R->check(i); return wrap_state(b.rd->state(*b.R, i)); },
             py::arg("i"),
             "Shot i's retained post-barrier state, reconstructed on demand from its own record, as a "
             "FramedSuperposition (as b9c2e92). branch_frames(i) is materialize(i).branch_frames().")
        // J-b1 P12: the b9c2e92 methods on layout 3 (state: alias of materialize; branch_frames(i) =
        // materialize(i).branch_frames(), as b9c2e92 — the retained state is reconstructed on demand from the shot's
        // own record; plan_structure without σ fields; born_dec pure data; pauli_expectations_all: every shot's state
        // read on the given literal Paulis, with no export). REMOVED (plan §4.1): sigma, frame_dressing,
        // input_frame_advance, input_frame_dressing, record_tables_held.
        .def("state", [](const BarrierBuffer& b, long i) { b.R->check(i); return wrap_state(b.rd->state(*b.R, i)); },
             py::arg("i"))
        .def("branch_frames", [](const BarrierBuffer& b, long i) {
            b.R->check(i);
            return wrap_state(b.rd->state(*b.R, i)).attr("branch_frames")();
        }, py::arg("i"))
        .def("plan_structure", [](const BarrierBuffer& b, long i) { return b.rd->plan_structure(*b.R, i); },
             py::arg("i"))
        .def("born_dec", [](const BarrierBuffer& b) { return b.rd->born_dec(); })
        .def("pauli_expectations_all", [](const BarrierBuffer& b,
                                          const std::vector<std::tuple<std::vector<int>, std::vector<int>, int>>& reads) {
            const int n = b.rd->bare().n(), nr = (int)reads.size();
            std::vector<Pauli> ps;
            for (const auto& rd : reads) {
                Pauli P(n);
                for (int q : std::get<0>(rd)) {
                    if (q < 0 || q >= n) throw std::out_of_range("pauli_expectations_all: x site out of range");
                    P.setx(q);
                }
                for (int q : std::get<1>(rd)) {
                    if (q < 0 || q >= n) throw std::out_of_range("pauli_expectations_all: z site out of range");
                    P.setz(q);
                }
                P.phase = std::get<2>(rd) & 3;
                ps.push_back(P);
            }
            py::array_t<std::complex<double>> out({(py::ssize_t)b.R->rows, (py::ssize_t)nr});
            auto o = out.mutable_unchecked<2>();
            for (long i = 0; i < b.R->rows; ++i) {
                const std::vector<std::complex<double>> e = b.rd->expectations(*b.R, i, ps);
                for (int r = 0; r < nr; ++r) o(i, r) = e[(size_t)r];
            }
            return out;
        }, py::arg("reads"))
        .def("_engine_module", [](const BarrierBuffer& b) { return b.module_; });
    py::class_<RecordFidelityL3>(m, "RecordFidelity", py::module_local(),
        "The layout-3 register RecordFidelity (the only route); evaluate(buffer, correction=None) reads every "
        "shot's own record.")
        .def("evaluate", [](const RecordFidelityL3& r, const py::object& buf, const py::object& correction,
                            const py::object& shots) {
            if (!py::isinstance<BarrierBuffer>(buf))
                throw py::type_error("RecordFidelity.evaluate: the buffer is not this module's BarrierBuffer (built by "
                                     "another extension); use that module's evaluator");
            return r.evaluate(buf.cast<const BarrierBuffer&>(), correction, shots);
        }, py::arg("buffer"), py::arg("correction") = py::none(), py::arg("shots") = py::none())
        .def("stats", [](const RecordFidelityL3& r) {
            const auto& st = r.rf->stats();
            py::dict d;
            d["shots"] = st.shots; d["memo_hits"] = st.memo_hits; d["misses"] = st.misses; d["plans"] = st.plans;
            d["max_dim_bits"] = st.max_dim_bits;
            d["memo_bytes"] = r.rf->memo_bytes();       // the memo alone
            d["held_bytes"] = r.rf->bytes_held();       // everything held (memo + contents + structures + λ)
            d["kernel_compiles"] = st.kernel_compiles; d["rn_misses"] = st.rn_misses;
            d["rotation_misses"] = st.rotation_misses; d["rn_words"] = st.rn_words;
            d["route"] = st.route; d["out_a"] = st.out_a;
            d["sectors"] = st.sectors; d["max_sectors"] = st.max_sectors;
            d["sectors_nonzero"] = st.sectors_nonzero;
            d["max_r"] = st.max_r; d["max_kappa_prime"] = st.max_kappa_prime;   // S8: the expansion route's sizes
            d["r_hist"] = st.r_hist; d["kappa_prime_hist"] = st.kappa_prime_hist; d["dker_hist"] = st.dker_hist;
            d["cache_budget"] = r.cache_budget ? py::object(py::int_(r.cache_budget))
                                             : py::object(py::float_(HUGE_VAL));   // math.inf = unbounded
            d["cache_flushes"] = st.cache_flushes;      // termroute-a: evaluate() calls that dropped the held bytes
            d["orthogonal_rows"] = st.orthogonal_rows;  // fid-orth: rows scored 0 by the per-content consistency test
            d["projection_skips"] = st.projection_skips;   //   rows that reused the content's previous projection
            d["value_builds"] = st.value_builds;        //   contents whose value structure was built (stage 2)
            return d;
        })
        .def("clear", [](RecordFidelityL3& r) { r.rf->clear(); },
             "Drop everything the evaluator holds: the memo, the interned contents and their expansion structures "
             "(pure speed devices; values are functions of their keys).")
        .def("clear_memo", [](RecordFidelityL3& r) { r.rf->clear_memo(); },
             "Drop the memo alone (contents and kernels persist across evaluate() calls).");
}

}  // namespace xtim_api
