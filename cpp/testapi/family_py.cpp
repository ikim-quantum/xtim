// J-a (N1) test API, PRODUCTION since the J-b1 flip (F, xtim-engine-13): the record-family sampler —
// `xtim._xtim_testapi.compile_family_sampler(text, **sampler_opts)` builds the SAME class the public
// `xtim._xtim.TwirlSampler` binds (cpp/bindings/family_sampler.hpp), plus the test-only accessors. (J-b2 J2: the
// legacy sampler's test-API surface, compile_legacy_sampler, is deleted; J-b2 J6: the per-class subset knob too.)
//
// The front end is the TwirlSampler(text, ...) recipe of cpp/bindings/xtim_py.cpp (parse →
// normalize_for_twirl_sampler → readout flip / invert per record → deferred reads → framed bare state),
// replicated here like record_law_oracle.cpp replicates the normalization preset. sample() emits through
// the FamilySink: the same packed (dets, obs, decisions) layout as TwirlSampler.sample(), with the Born
// observable reference-relative (born_obs_ref applied) and the family extras reference-relative.
//
// J-b1 F: the sampler, its records' reader and the ideal-state / injection layer moved to
// cpp/bindings/family_sampler.hpp (shared with xtim._xtim, which binds the same class as TwirlSampler). This file
// keeps the test-API surface: compile_family_sampler, the test-only accessors (XTIM_FAMILY_TESTAPI)
// and the J-a / J-b1 test cores.
#define XTIM_FAMILY_TESTAPI 1
#include "../bindings/family_sampler.hpp"
#include "../bindings/framed_py.hpp"     // J-b2 J2: this extension's FramedSuperposition (was bound by legacy_py.cpp)
#include "twirl_test_access.hpp"

struct PyFamilyRecordFidelity {
    std::unique_ptr<RecordFidelity> rf;
    std::string model;
    PyFamilyRecordFidelity(PyFamilySampler& s, size_t max_bytes) {
        rf = std::make_unique<RecordFidelity>(*s.bare_, TA_::family_record_ops(*s.s_), s.out_wires_, s.read_meas_(),
                                              max_bytes, s.wire_user_());
        model = TA_::family_model_key(*s.s_);
    }
    // correction: uint8[rows, 2, n_out] output-Pauli masks (X row, Z row; OUTPUT_QUBITS order), as production.
    py::array_t<double> evaluate(PyFamilySampler& s, const PyFamilyRecords& R, const py::object& correction) {
        const BusyHold h = s.hold_("FamilyRecordFidelity.evaluate"), hc = s.hold_cache_("FamilyRecordFidelity.evaluate");
        R.no_branch_rows("FamilyRecordFidelity");
        if (R.model != model || TA_::family_model_key(*s.s_) != model)
            throw std::invalid_argument("FamilyRecordFidelity: the records (or the sampler) come from a different "
                                        "state model (bare state, certified generators, record operators with their "
                                        "folds; checked bit for bit) — refused");
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> c;
        if (!correction.is_none()) {
            c = py::cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(correction);
            if (c.ndim() != 3 || c.shape(0) != R.rows || c.shape(1) != 2 || c.shape(2) != (py::ssize_t)s.out_wires_.size())
                throw std::invalid_argument("FamilyRecordFidelity: correction must be uint8[rows, 2, n_out] — refused "
                                            "(shape mismatch)");
            for (py::ssize_t i = 0; i < c.size(); ++i)
                if (c.data()[i] > 1)
                    throw std::invalid_argument("FamilyRecordFidelity: correction entries must be 0 / 1 — refused");
        }
        bool k2 = false;
        for (long i = 0; i < R.rows && !k2; ++i)
            if (R.handle[(size_t)i] >= 0 && R.handle[(size_t)i] < (int)R.contents.size() &&
                R.contents[(size_t)R.handle[(size_t)i]].kind != 0) k2 = true;
        if (k2) s.arm_();                                // kind-2 contents need the term view
        py::array_t<double> out((py::ssize_t)R.rows);
        TA_::family_record_fidelity(*s.s_, *rf, R.sink, R.rows, out.mutable_data(),
                                    correction.is_none() ? nullptr : c.data(),
                                    correction.is_none() ? nullptr : &s.out_wires_);
        return out;
    }
};

// S8-2a: the cache flag a PlanCacheHandle entry takes.
static std::atomic<bool>& handle_flag(const PyCacheHandle& h) {
#ifndef XTIM_MUTATE_BUSY_HANDLE
    return h.c->binding_busy;
#else
    static std::atomic<bool> unrelated{false};       // mutation build only (BUSY-HANDLE): the handle skips the flag
    (void)h;
    return unrelated;
#endif
}

void register_family(py::module_& m) {
    // J-b2 J2: the test API's FramedSuperposition handle (module_local) and the layout-3 BarrierBuffer's materialize
    // hook, as the production extension binds them (bound by the deleted legacy_py.cpp until J2).
    bind_framed_superposition(m);
    xtim_api::wrap_state_hook() = [](FramedSuperposition&& st) -> py::object {
        return py::cast(PyFramedSuperposition(std::move(st)));
    };
    xtim_api::bind_record_buffer(m);                 // J-b1 P12: BarrierBuffer, RecordFidelity (module_local)
    // S8 zero classes: the per-row code log (record_fidelity.hpp set_row_log_for_test; memo bypassed while on)
    m.def("_rf_row_log", [](xtim_api::RecordFidelityL3& r, bool on) { r.rf->set_row_log_for_test(on); },
          py::arg("rf"), py::arg("on"));
    m.def("_rf_take_row_log", [](xtim_api::RecordFidelityL3& r) {
        const std::vector<uint8_t> v = r.rf->take_row_log_for_test();
        py::array_t<uint8_t> a((py::ssize_t)v.size());
        if (!v.empty()) std::copy(v.begin(), v.end(), a.mutable_data());
        return a;
    }, py::arg("rf"));
    // S8 fix round (final review m1): flip one stored record value of row i, column c (a tampered buffer, test-only)
    m.def("_flip_record_value", [](const py::object& r, long i, int c) {
        const std::shared_ptr<PyFamilyRecords> R = records_arg(r);
        R->check(i);
        if (c < 0 || c >= R->m) throw std::out_of_range("record column out of range");
        R->y[(size_t)i * (size_t)R->sink.YW + (size_t)(c >> 6)] ^= 1ULL << (c & 63);
    }, py::arg("records"), py::arg("i"), py::arg("c"));
    // S8 RF1: λ(u, v) through the memo or the formula; the dense table's byte cap; the memo kind
    m.def("_rf_lambda", [](xtim_api::RecordFidelityL3& r, uint64_t u, uint64_t v, bool direct) {
        return r.rf->lambda_for_test(u, v, direct); }, py::arg("rf"), py::arg("u"), py::arg("v"), py::arg("direct") = false);
    m.def("_rf_set_lambda_budget", [](xtim_api::RecordFidelityL3& r, size_t b) { r.rf->set_lambda_budget_for_test(b); },
          py::arg("rf"), py::arg("bytes"));
    m.def("_rf_lambda_mode", [](xtim_api::RecordFidelityL3& r) { return r.rf->lambda_mode_for_test(); }, py::arg("rf"));
    // S8 RF4: the fibers by the exact enumeration oracle (test-only)
    m.def("_rf_hform_oracle", [](xtim_api::RecordFidelityL3& r, bool on) { r.rf->set_hform_oracle_for_test(on); },
          py::arg("rf"), py::arg("on"));
    py::class_<PyFamilyRecordFidelity>(m, "FamilyRecordFidelity",
        "J-b1 P7 TEST API: the layout-3 register record fidelity (the only route).")
        .def(py::init<PyFamilySampler&, size_t>(), py::arg("sampler"), py::arg("max_bytes") = (size_t)0,
             py::keep_alive<1, 2>())
        .def("evaluate", &PyFamilyRecordFidelity::evaluate, py::arg("sampler"), py::arg("records"),
             py::arg("correction") = py::none())
        .def("stats", [](const PyFamilyRecordFidelity& r) {
            const RecordFidelityStats& st = r.rf->stats();
            py::dict d;
            d["shots"] = st.shots; d["memo_hits"] = st.memo_hits; d["misses"] = st.misses; d["plans"] = st.plans;
            d["max_dim_bits"] = st.max_dim_bits; d["max_r"] = st.max_r; d["max_kappa_prime"] = st.max_kappa_prime;
            return d; });
    // J-b1 final review m1: module_local, so the production extension never accepts a test-API records object
    // (LayoutRecords is in a named namespace: a global registration let pybind11's cross-module cast through).
    py::class_<PyFamilyRecords, std::shared_ptr<PyFamilyRecords>>(m, "FamilyRecords", py::module_local(),
        "J-a N5 TEST API: the layout-3 records of one FAM run.")
        .def_property_readonly("num_records", [](const PyFamilyRecords& R) { return R.rows; })
        .def("y", &PyFamilyRecords::y_bits, py::arg("i"), "raw record values (storage column order)")
        .def("outcomes", &PyFamilyRecords::outcomes, py::arg("i"), "the emission's outcomes (u then z)")
        .def("prec", &PyFamilyRecords::prec_bits, py::arg("i"), "P_rec as (x bits, z bits)")
        .def("handle", [](const PyFamilyRecords& R, long i) { R.check(i); return R.handle[(size_t)i]; }, py::arg("i"))
        .def("route", [](const PyFamilyRecords& R, long i) { R.check(i); return (int)R.route[(size_t)i]; }, py::arg("i"))
        .def("frame_folded", [](const PyFamilyRecords& R, long i) { R.check(i); return (int)R.ff[(size_t)i]; }, py::arg("i"))
        .def("branch", [](const PyFamilyRecords& R, long i) { R.check(i); return (int)R.branch[(size_t)i]; }, py::arg("i"))
        .def("kind", [](const PyFamilyRecords& R, long i) { return R.content_of(i).kind; }, py::arg("i"))
        .def("content", [](const PyFamilyRecords& R, long i) { return py::bytes(R.content_of(i).content); }, py::arg("i"))
        .def("digest", [](const PyFamilyRecords& R, long i) {
                 const auto& C = R.content_of(i);
                 return py::make_tuple(C.digest_hi, C.digest_lo);
             }, py::arg("i"))
        .def("key", &PyFamilyRecords::key, py::arg("i"), "layout 3: [kind][len] content ‖ P_rec words ‖ y words")
        .def("plan_content", [](const PyFamilyRecords& R, long i) {   // J-b1 P11b: the layout-3 accessor
            const auto& C = R.content_of(i);
            return py::make_tuple(xtim_api::plan_content_bytes(C.kind, C.content),
                                  xtim_api::digest_bytes(C.digest_hi, C.digest_lo));
        }, py::arg("i"), "(the layout-3 plan content [kind][len] content, its 128-bit digest)")

        .def("plan", &PyFamilyRecords::plan, py::arg("i"));
    py::class_<PyFamilySampler>(m, "FamilySampler",
        "J-a TEST API (switched OFF in production): a FAM TwirlRecordSampler. Construct with "
        "compile_family_sampler.")
        .def("sample", &PyFamilySampler::sample, py::arg("shots"), py::arg("seed") = py::none(),
             py::arg("retain") = false, py::arg("input_pauli") = py::none(), py::arg("input_branch") = py::none(),
             "(dets, obs, decisions) packed b8 (TwirlSampler.sample's layout). retain=True asks for state "
             "retention (refused by name on a family sampler: sigma export, Q3 / N5).")
        .def_property_readonly("num_detectors", [](const PyFamilySampler& s) { return s.num_det_; })
        .def_property_readonly("num_observables", [](const PyFamilySampler& s) { return s.num_obs_; })
        .def_property_readonly("num_decisions", [](const PyFamilySampler& s) { return s.num_dec_; })
        .def("channel_report", &Busy<&PyFamilySampler::channel_report>::call)
#ifdef XTIM_BENCH_REPLAY
        // BENCH-ONLY (-DXTIM_BENCH_REPLAY): the ledger's fam_fast/<w> replay (J-a final review M1; REPORT-ONLY):
        // the FAM sampler's run() over a recorded event batch, as PyTwirlSampler's hooks
        .def("_bench_replay", [](PyFamilySampler& s, py::object off, py::object ev) {
            if (off.is_none()) { s.s_->bench_replay({}, {}); return; }
            auto o = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(off);
            auto e = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(ev);
            s.s_->bench_replay(std::vector<int32_t>(o.data(), o.data() + o.size()),
                               std::vector<int32_t>(e.data(), e.data() + e.size()));
        }, py::arg("off"), py::arg("ev"))
        .def("_bench_fingerprint", [](const PyFamilySampler& s) { return s.s_->bench_fingerprint(); })
        // J-b1 final review m6: moved here from the production TwirlSampler (test oracles)
        .def("_born_trie_check", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("_born_trie_check");
                 const auto r = s.born_trie_check_all();
                 return py::make_tuple(r.first, r.second);
             }, "Test oracle: every evaluated Born-trie node (diagonal cache, kind-2 and route-1 plans) re-evaluated "
                "and compared bit for bit; (checked, mismatches).")
        .def("_born_trie_oracle", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("_born_trie_oracle");
                 const auto r = s.born_trie_oracle_all();
                 return py::make_tuple(r.first, r.second);
             }, "Independent oracle: every evaluated Born-trie node against the n-qubit state; (checked, mismatches).")
#endif
        .def("sample_records", &PyFamilySampler::sample_records, py::arg("shots"), py::arg("seed") = py::none(),
             py::arg("input_pauli") = py::none(), py::arg("input_branch") = py::none(),
             "J-a N5: (dets, obs, decisions, FamilyRecords) — every shot term-routed, its layout-3 record kept.")
        .def("materialize", [](const PyFamilySampler& s, const PyFamilyRecords& R, long i, bool tableau) { const BusyHold h_ = s.hold_("materialize");
                 return state_frames(s.materialize_state(R, i, tableau));
             }, py::arg("records"), py::arg("i"), py::arg("tableau") = false,
             "J-a N5: the retained state of record i (forced projection) in the bare_frames export format.")
        .def("state_expectations", &Busy<&PyFamilySampler::state_expectations>::call, py::arg("records"), py::arg("i"),
             py::arg("paulis"), py::arg("tableau") = false, "J-a N5: Re<psi_i|P|psi_i> on the retained state of record i, per Pauli (x, z, phase).")
        .def("signs", &Busy<&PyFamilySampler::signs>::call, py::arg("records"), py::arg("paulis"),
             "J-a N5 signs() core: int8 [rows, paulis] = +1 / -1 definite by the H-membership rule, else 0.")
        .def("record_classes", &Busy<&PyFamilySampler::record_classes>::call,
             "J-a N5 record_report() classes (R = I): {classes {detectors, observables, decisions}, "
             "nondeterministic_detectors, nondeterministic_observables}.")
        .def("oracle_counts", &Busy<&PyFamilySampler::oracle_counts>::call)
        .def("ideal_state", [](py::object self, bool provenance) {   // J-b1 P9a (producer)
            const PyFamilySampler& s = self.cast<const PyFamilySampler&>();
            const BusyHold h_ = s.hold_("ideal_state");
            PyIdealState i;
            i.fs = s.ideal_fs();                    // J-b1 P9b (P9a-1): the post-barrier ideal state
            i.wires = s.out_wires_;
            if (provenance) { i.tok = s.own_token_(); i.producer = self; }
            else { i.tok = frame_token_rows(i.fs); i.producer = py::none(); }
            return i;
        }, py::arg("provenance") = true)
        .def("sample_barrier", &family_sample_barrier, py::arg("shots"), py::arg("seed") = py::none(),
             py::arg("input_pauli") = py::none(), py::arg("compact") = true, py::arg("input_branch") = py::none(),
             py::keep_alive<0, 1>(),
             "J-b1 P12 / F: the layout-3 BarrierBuffer (cpp/bindings/family_sampler.hpp, the production body).")
        .def("record_fidelity", &family_record_fidelity_l3, py::arg("max_bytes") = (size_t)0,
             py::arg("cache_budget") = py::none(),
             "The layout-3 reference RecordFidelity evaluator of this sampler (evaluate(buffer, correction=None)).")
        .def("branch_handles", [](const PyFamilySampler& s, const py::object& r) { const BusyHold h_ = s.hold_("branch_handles"); return s.branch_handles(*records_arg(r)); },
             py::arg("records"))
        .def("branches", [](const PyFamilySampler& s, const py::object& r) { const BusyHold h_ = s.hold_("branches"); return s.branches(*records_arg(r)); },
             py::arg("records"),
             "J-b1 P9b: (handles, frames uint8[rows, 2, n_out]) — the shot's Pauli frame P_rec·W(u) on OUTPUT_QUBITS.")

        .def("add_branch", &Busy<&PyFamilySampler::add_branch>::call, py::arg("handle"), py::arg("vector") = py::none())
        .def("branch_counters", &Busy<&PyFamilySampler::branch_counters>::call)
        .def("set_branch_budget", [](PyFamilySampler& s, py::object bytes) {   // J-b1 P9b D6 (test knob)
            const size_t b = bytes.is_none() ? SIZE_MAX : bytes.cast<size_t>();
            TA_::family_set_branch_budget(*s.s_, b);
            for (auto& ec : s.eclass_) TA_::family_set_branch_budget(*ec.s->s_, b);
        }, py::arg("bytes"))
        .def("evict_branches", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("evict_branches");   // J-b1 P9b D6 (test knob)
            TA_::family_evict_branches(*s.s_);
            for (auto& ec : s.eclass_) TA_::family_evict_branches(*ec.s->s_);
        })
        .def("_column_reference", [](const PyFamilySampler& s) { return s.column_reference_(); })   // J-b1 P9b
        .def("_busy_flag", [](const PyFamilySampler& s, const std::string& which, bool on) {   // S8-2a test hook
            std::atomic<bool>& f = which == "sampler" ? s.busy_ : TA_::family_plan_cache(*s.s_)->binding_busy;
            f.store(on);
        }, py::arg("which"), py::arg("on"),
           "S8-2a TEST HOOK: set / clear the sampler ('sampler') or plan-cache ('cache') busy flag, as another thread "
           "holding it would.")
        .def("_alt_atoms_oracle", [](const PyFamilySampler& s) {   // S8-1' 5-a: (checked, mismatches) over every E-class
            long c = 0, b = 0;
            auto add = [&](const std::pair<long, long>& r) { c += r.first; b += r.second; };
            add(TA_::family_alt_atoms_oracle(*s.s_));
            for (const auto& ec : s.eclass_) add(TA_::family_alt_atoms_oracle(*ec.s->s_));
            return py::make_tuple(c, b);
        })
        .def("_ctor_digest", [](const PyFamilySampler& s) {   // S8-0b G-PREP: sha256 of every construction product
            std::string b = TA_::family_ctor_bytes(*s.s_);
            for (const auto& ec : s.eclass_) b += TA_::family_ctor_bytes(*ec.s->s_);
            return py::module_::import("hashlib").attr("sha256")(py::bytes(b)).attr("hexdigest")();
        })
        .def("_eclass_masks", [](const PyFamilySampler& s) {
            py::list l;
            for (const auto& ec : s.eclass_) l.append(py::make_tuple(ec.rows, ec.dmask, ec.omask));
            return l; })
        .def("carried_compose_report", [](const PyFamilySampler& s) { const BusyHold h_ = s.hold_("carried_compose_report");   // J-b1 P9b (XQ9 (A))
            py::dict d;
            d["route"] = s.inject_route_;
            d["axes"] = s.inject_axes_;
            d["gauge_moves"] = s.inject_gauge_;
            return d;
        })
        .def("branch_table_digest", [](const PyFamilySampler& s) {
            if (!s.btab_) throw std::invalid_argument("branch_table_digest: not an ideal-state sampler");
            return PyFamilySampler::dbytes_(s.btab_->digest()); })
        .def("branch_emin", [](const PyFamilySampler& s, int b) { const BusyHold h_ = s.hold_("branch_emin");
            if (!s.btab_) throw std::invalid_argument("branch_emin: not an ideal-state sampler");
            return s.btab_->emin(b); }, py::arg("branch"))
        .def("branch_vector", [](const PyFamilySampler& s, int b) { const BusyHold h_ = s.hold_("branch_vector");   // the registered branch's vector (frames)
            if (b < 1 || b > (int)s.bvec_.size()) throw std::invalid_argument("branch_vector: branch id out of range");
            return state_frames(s.bvec_[(size_t)b - 1]); }, py::arg("branch"))
        .def("_force_branch_digest", [](PyFamilySampler& s, const py::bytes& d) { const BusyHold h_ = s.hold_("_force_branch_digest");   // G-DIG test hook
            if (!s.btab_) throw std::invalid_argument("not an ideal-state sampler");
            s.btab_->force_digest_for_test(PyFamilySampler::dfrom_(d)); })
        .def("record_ops", [](const PyFamilySampler& s) {   // J-b1 P7: (x, z, phase) per record column
            py::list out;
            for (const Pauli& P : TA_::family_record_ops(*s.s_)) {
                std::vector<uint8_t> x((size_t)P.n), z((size_t)P.n);
                for (int q = 0; q < P.n; ++q) { x[(size_t)q] = P.xbit(q); z[(size_t)q] = P.zbit(q); }
                out.append(py::make_tuple(x, z, P.phase & 3));
            }
            return out; })
        .def("route_stats", &Busy<&PyFamilySampler::route_stats>::call,
             "J-a N4a: the kind-2 route's counters {shots_kind0, shots_kind2, keys, plans_kind2, builds_kind2, "
             "payload_builds_kind2, oracle_term_shots, shots_framed, plans/builds/payload_builds/shots_kind0_general "
             "(N4-D1: kind-0 composites outside the pair universe, served by the (G) path), "
             "plan_shots {plan id: (kind, shots)}}.")
        .def("record_report", &Busy<&PyFamilySampler::record_report>::call,
             "J-b1 P11b: record_report() (design §4) — classes, nondeterministic lists, route_counts, counters.")
        .def("frames", &Busy<&PyFamilySampler::frames>::call, py::arg("records"),
             "J-b1 P11b: frames() (design §4) of layout-3 records — the state of row i is P_rec·C0|B> projected onto "
             "the records at y; per-plan arrays indexed by the export-local plan_index.")
        .def("record_plan_structure", &Busy<&PyFamilySampler::record_plan_structure>::call, py::arg("records"), py::arg("i"),
             "J-b1 P11b: plan_structure() (design §4 fields) of record i's plan.")
        .def("record_layout", [](const PyFamilySampler& s) {
            const int n = s.bare_->n();
            return xtim_api::record_layout_dict(n, (int)s.s_->record_operators().size(), (n + 63) / 64,
                                                s.s_->record_value_words());
        })
        .def("record_counters", [](const PyFamilySampler& s) { const BusyHold h_ = s.hold_("record_counters");   // J-b1 P11a: the plan-memory counters (M-K2OWN)
            const auto c = s.s_->record_counters();
            py::dict d;
            d["header_bytes"] = c.header_bytes;
            d["payload_resident_bytes"] = c.payload_resident_bytes;
            d["payload_builds"] = c.payload_builds;
            d["payload_evictions"] = c.payload_evictions;
            d["plan_misses"] = c.plan_misses_kappa0 + c.plan_misses_kappa_pos + c.plan_misses_fallback;
            d["arena_headers"] = TA_::family_plan_cache(*s.s_)->arena_size();
            d["diagonal_headers"] = TA_::family_plan_cache(*s.s_)->size();
            return d;
        })
        .def("_cache_handle", [](const PyFamilySampler& s) { return PyCacheHandle{TA_::family_plan_cache(*s.s_)}; },
             "J-b1 P11a TEST API: shared ownership of the sampler's plan cache (the destroy-then-evict test).")
        .def("take_disk_events", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("take_disk_events");   // J-b1 P11a: the DISK_LOAD / DISK_SAVE events so far
            s.drain_();
            py::list out = s.disk_events_;
            s.disk_events_ = py::list();
            return out;
        })
        .def("set_plan_payload_budget", [](PyFamilySampler& s, size_t b) { const BusyHold h_ = s.hold_("set_plan_payload_budget"); s.s_->set_plan_payload_budget(b); },
             py::arg("bytes"))
        .def("plan_payload_stats", [](const PyFamilySampler& s) { const BusyHold h_ = s.hold_("plan_payload_stats");
                 const TwirlRecordSampler::RecordCounters c = s.s_->record_counters();
                 py::dict r;
                 r["resident_bytes"] = c.payload_resident_bytes;
                 r["peak_bytes"] = c.payload_peak_bytes;
                 r["evictions"] = c.payload_evictions;
                 r["reader_bytes"] = c.reader_bytes;            // J-b1 P7 (m10)
                 r["reader_peak_bytes"] = c.reader_peak_bytes;
                 r["reader_overshoot_bytes"] = c.reader_overshoot_bytes;
                 r["reader_evictions"] = c.reader_evictions;
                 return r; })
        .def("plan_structure", &Busy<&PyFamilySampler::plan_structure>::call, py::arg("plan_key"), py::arg("force_general") = false,
             "the plan's record table; force_general=True: the same plan's table built by (G) (J-a N3, G-9a).")
        // J-a (N3): the (G) builder (the dual-run harness)
        .def("term_plan_structure", &Busy<&PyFamilySampler::term_plan_structure>::call, py::arg("S"), py::arg("born_law") = false,
             "(G)'s table of C0 = rot(S) (S: term ids of term_axes()); born_law=True adds p_z.")
        .def("bare_expectation", &Busy<&PyFamilySampler::bare_expectation>::call, py::arg("x"), py::arg("z"), py::arg("phase") = 0,
             "<bare|i^phase X^x Z^z|bare> (exact; N3-D2's per-plan fold-product check).")
        // m1: the diagonal cache's walk (production) PLUS the FAM route's sampler-owned plans (kind 2, route 1)
        .def("born_trie_check", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("born_trie_check");
            const auto d = s.s_->born_trie_check();
            const auto f = TA_::family_born_trie_walk(*s.s_, false);
            return std::make_pair(d.first + f[0] + f[1], d.second + f[2]);
        })
        .def("born_trie_oracle", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("born_trie_oracle");
            const auto d = s.s_->born_trie_oracle();
            const auto f = TA_::family_born_trie_walk(*s.s_, true);
            return std::make_pair(d.first + f[0] + f[1], d.second + f[2]);
        })
        .def("born_trie_walk_family", [](PyFamilySampler& s, bool oracle) { const BusyHold h_ = s.hold_("born_trie_walk_family"); return TA_::family_born_trie_walk(*s.s_, oracle); },
             py::arg("oracle") = false, "{checked kind 2, checked route 1, bad} over the FAM route's own plans")
        .def("born_trie_nodes", [](const PyFamilySampler& s) { const BusyHold h_ = s.hold_("born_trie_nodes"); return TA_::family_trie_nodes(*s.s_); },
             "non-NaN record-trie nodes by linear scan: {diagonal cache, kind 2, route 1}")
        .def("_intern_content", [](PyFamilySampler& s, int kind, py::bytes c) { const BusyHold h_ = s.hold_("_intern_content");
            return TA_::family_intern_content(*s.s_, kind, std::string(c));
        }, py::arg("kind"), py::arg("content"), "m3 test seam: the layout-3 content handle of (kind, content)")
        // J-a (N2): the term route (TermView, per-alt (P_f, S_f), KerPhi, key, kind, plan id)
        .def("term_axes", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("term_axes"); return term_axes(s.terms()); },
             "(xz uint8[T, 2n], phase int32[T]): the end-of-circuit signed term axes B_j.")
        .def("term_rows", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("term_rows");
                 std::vector<std::vector<int>> r;
                 for (int j = 0; j < s.terms().num_terms(); ++j) r.push_back(s.terms().term_rows(j));
                 return r; })
        .def("term_alt_faults", &Busy<&PyFamilySampler::term_alt_faults>::call,
             "per flat alternative: [(deferred stream index, qubit, code 1X 2Y 3Z)].")
        .def("term_alt_is_ppr", [](const PyFamilySampler& s) { return TA_::family_alt_is_ppr(*s.s_); })
        .def("term_alt_atom", [](PyFamilySampler& s, int a) { const BusyHold h_ = s.hold_("term_alt_atom");
                 TermResidual& tr = s.terms();
                 if (a < 0 || a >= tr.num_alts()) throw std::invalid_argument("alternative id out of range");
                 Pauli P(tr.n());
                 for (int w = 0; w < (int)P.x.size(); ++w) { P.x[(size_t)w] = tr.alt_px(a)[w]; P.z[(size_t)w] = tr.alt_pz(a)[w]; }
                 std::vector<uint64_t> S(tr.alt_s(a), tr.alt_s(a) + tr.term_words());
                 return py::make_tuple(pauli_xz(P), set_list(S, tr.num_terms())); }, py::arg("alt"))
        .def("term_events", [](PyFamilySampler& s, long shots, uint64_t seed) { const BusyHold h_ = s.hold_("term_events");
                 return TA_::family_draw_events(*s.s_, shots, seed); }, py::arg("shots"), py::arg("seed"),
             "the fired alternative ids per shot, as run() draws them after set_seed(seed).")
        .def("term_shot", &Busy<&PyFamilySampler::term_shot>::call, py::arg("alts"),
             "the term route of an alternative list: {P, S (composed), P_red, S_red, key, kind, plan_id, nf}.")
        .def("diag_shot", &Busy<&PyFamilySampler::diag_shot>::call, py::arg("alts"),
             "the diagonal route's canonical normal form {prefix, a, cz, plan_id}, or None (a PPR alternative).")
        .def("term_relabel", &Busy<&PyFamilySampler::relabel>::call, py::arg("x"), py::arg("z"),
             "t(P) per storage column (channels: prefix rows; extras: <P, M_e>).")
        .def("term_kernel", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("term_kernel"); return kernel_list(s.terms()); })
        .def("term_rank", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("term_rank"); return s.terms().rank(); })
        .def("term_num_plans", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("term_num_plans"); return s.terms().num_plan_ids(); })
        .def("term_merge_stats", [](PyFamilySampler& s) { const BusyHold h_ = s.hold_("term_merge_stats");   // J-b1 P5: the static merge's counters (R3-6)
            const TermResidual::MergeStats& m = s.terms().merge_stats();
            py::dict d;
            d["merges"] = m.merges; d["folds"] = m.folds; d["cancels"] = m.cancels; d["passes"] = m.passes;
            d["terms_before"] = m.terms_before; d["terms"] = s.terms().num_terms();
            return d;
        })
        .def("_engine", [](PyFamilySampler& s) {
                 s.arm_();   // J-b1 P4: the C++ test cores read the term view (branch_py.cpp)
                 return py::capsule(static_cast<void*>(s.s_.get()), "xtim.TwirlRecordSampler"); });
    py::class_<PyCacheHandle>(m, "PlanCacheHandle", "J-b1 P11a TEST API: a FAM sampler's plan cache, kept alive.")
        // S8-2a: every handle entry holds the cache flag (a handle outlives its sampler and may be used from any thread)
        .def("evict_all", [](PyCacheHandle& h) {
            const BusyHold b(handle_flag(h), "XT_CACHE_BUSY", "plan cache", "evict_all");
            long k = 0; while (h.c->evict_oldest()) ++k; return k; })
        .def("resident_bytes", [](const PyCacheHandle& h) {
            const BusyHold b(handle_flag(h), "XT_CACHE_BUSY", "plan cache", "resident_bytes"); return h.c->resident_bytes(); })
        .def("arena_size", [](const PyCacheHandle& h) {
            const BusyHold b(handle_flag(h), "XT_CACHE_BUSY", "plan cache", "arena_size"); return h.c->arena_size(); })
        .def("header_bytes", [](const PyCacheHandle& h) {
            const BusyHold b(handle_flag(h), "XT_CACHE_BUSY", "plan cache", "header_bytes"); return h.c->header_bytes(); })
        ;
    py::class_<PyTermSet>(m, "TermSet",
        "J-a N2 TEST API: a hand-built commuting term set (construction throws the term-commutation assert "
        "on an anticommuting pair).")
        .def(py::init<int, py::array_t<uint8_t>, std::vector<int>>(), py::arg("n"), py::arg("xz"), py::arg("phase"))
        .def("kernel", [](const PyTermSet& t) { return kernel_list(*t.tr); })
        .def("rank", [](const PyTermSet& t) { return t.tr->rank(); })
        .def("reduce", [](const PyTermSet& t, const std::vector<int>& S) { return t.run(S, nullptr); }, py::arg("S"),
             "(I, S) -> ((Px, Pz), S_red): reduce_key_.")
        .def("fold_sequence", [](const PyTermSet& t, const std::vector<int>& S, const std::vector<int>& seq) {
                 return t.run(S, &seq); }, py::arg("S"), py::arg("seq"),
             "(I, S) -> ((Px, Pz), S'): the unconditional fold by kernel vectors seq[0], seq[1], ...")
        .def("kind", [](const PyTermSet& t, const std::vector<int>& S) {
                 return t.tr->kind_of_(set_words(S, t.tr->num_terms())); }, py::arg("S_red"));
    // ── J-b1 P9a: ideal states and the branch identity layer ──
    py::class_<PyIdealState>(m, "IdealState",
        "J-b1 P9a TEST API: an ideal (carried) state: its frame-explicit state, OUTPUT_QUBITS wires and frame token.")
        .def_property_readonly("token", [](const PyIdealState& i) { return PyFamilySampler::dbytes_(i.tok.d); })
        .def_property_readonly("provenance", [](const PyIdealState& i) { return i.tok.producer; })
        .def_property_readonly("wires", [](const PyIdealState& i) { return i.wires; });
    // J-b1 P10: an ideal state WITHOUT provenance (rows-only token: branch 0 only) from a state in the
    // branch_frames() export format (production FramedSuperposition.branch_frames(), the test API's state_frames;
    // stab_phase carries the eps sign, eps given separately) and its carried wires: the test-API twin of
    // compile_twirl_sampler_from_state's carried state (_InputState: state, carried_wires) — every carried state,
    // single-qubit stabilizer states included, goes through injection as branch 0 (the prep-splice reroute).
    m.def("ideal_state_from_frames", [](const py::dict& d, const std::vector<int>& wires) {
              const int n = d["n"].cast<int>();
              auto rows = [n](const py::object& a) {
                  auto arr = a.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
                  if (arr.ndim() != 2 || arr.shape(0) != n || arr.shape(1) != n)
                      throw std::invalid_argument("ideal_state_from_frames: a frame row block is not n x n");
                  std::vector<std::vector<uint8_t>> out((size_t)n, std::vector<uint8_t>((size_t)n));
                  for (int a = 0; a < n; ++a)
                      for (int q = 0; q < n; ++q) out[(size_t)a][(size_t)q] = arr.data()[(size_t)a * n + q] & 1;
                  return out;
              };
              const auto eps = d["eps"].cast<std::vector<uint8_t>>();
              auto zph = d["stab_phase"].cast<std::vector<int>>();
              const auto xph = d["destab_phase"].cast<std::vector<int>>();
              if ((int)eps.size() != n || (int)zph.size() != n)
                  throw std::invalid_argument("ideal_state_from_frames: eps / stab_phase length != n");
              for (int a = 0; a < n; ++a) zph[(size_t)a] = (zph[(size_t)a] + (eps[(size_t)a] & 1 ? 2 : 0)) & 3;
              const auto free = d["free"].cast<std::vector<int>>();
              auto sig = d["sigma"].cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
              auto co = d["coeff"].cast<py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>>();
              const int K = (int)free.size(), chi = (int)co.shape(0);
              std::vector<std::pair<std::vector<uint8_t>, std::complex<double>>> ent;
              for (int i = 0; i < chi; ++i) {
                  std::vector<uint8_t> b((size_t)K);
                  for (int k = 0; k < K; ++k) b[(size_t)k] = sig.data()[(size_t)i * K + k] & 1;
                  ent.emplace_back(std::move(b), co.data()[i]);
              }
              PyIdealState i;
              i.fs = framed_from_rows(n, rows(d["stab_x"]), rows(d["stab_z"]), zph, rows(d["destab_x"]),
                                      rows(d["destab_z"]), xph, eps, free, ent);
              for (int w : wires)
                  if (w < 0 || w >= n) throw std::invalid_argument("ideal_state_from_frames: a carried wire is out of range");
              i.wires = wires;
              i.tok = frame_token_rows(i.fs);
              i.producer = py::none();
              return i;
          }, py::arg("frames"), py::arg("wires"));
    m.def("pivot_nonpattern_count", []() { return framed_pivot_nonpattern_count(); },
          "J-b1 P9a: anticommuting reads whose pivot row differs from the pattern-only rule (Lemma R1 gate).");
    m.def("bare_overlap_mapped", [](const PyFamilySampler& a, const PyFamilySampler& b, const std::vector<int>& map) {
              // J-b1 P9b (XQ9 C6): |<bare_a | remap(bare_b)>|, b's wire w moved to a's wire map[w] (a's other wires
              // |0>; remap_state_wires) — the composed consumer bare vs the in-stage build of the spliced circuit.
              if (!a.bare_ || !b.bare_) throw std::invalid_argument("bare_overlap_mapped: no bare state");
              if ((int)map.size() != b.bare_->n()) throw std::invalid_argument("bare_overlap_mapped: map width");
              const FramedSuperposition bm = remap_state_wires(*b.bare_, map, a.bare_->n());
              return std::abs(exact_sum_overlap(*a.bare_, bm));
          }, py::arg("a"), py::arg("b"), py::arg("map"));
    m.def("forced_pivot_nonpattern_count", []() { return family_forced_pivot_nonpattern_count(); },
          "J-b1 P9b: forced records (materialize) whose pivot differs from the pattern-only rule because C0-hat "
          "promoted E_S rows (Lemma R1 gate on the force_record_ -> measure_pauli path; review C-R3).");
    m.def("compile_family_sampler_from_ideal_state",
          [](const PyIdealState& ideal, const std::string& text, double p_factor,
             long selfcheck, bool oracle_all_shots, bool decisions, uint64_t seed) {
              auto s = std::make_unique<PyFamilySampler>(text, p_factor, selfcheck, oracle_all_shots,
                                                         decisions, seed, false, &ideal);
              s->btab_ = std::make_shared<BranchTable>(ideal.tok);
              s->producer_ = ideal.producer;
              return s;
          },
          py::arg("ideal_state"), py::arg("text"), py::arg("p_factor") = 1.0,
          py::arg("selfcheck") = 0, py::arg("oracle_all_shots") = false,
          py::arg("decisions") = true, py::arg("seed") = 1, py::keep_alive<0, 1>(),
          "J-b1 P9a TEST API: a FAM consumer sampler whose INPUT_QUBITS carry the ideal state (branch 0, state-level "
          "injection); add_branch registers producer branches by content handle.");
    m.def("compile_family_sampler",
          [](const std::string& text, const std::string& classes, double p_factor, long selfcheck,
             bool oracle_all_shots, bool decisions, uint64_t seed,
             const std::string& record_route, const std::string& disk_path) {
              // J-b2 J6: every record class is served (the class bits are deleted). `classes` stays accepted as "all"
              // only, so the ledger harness (_drv.py, byte-identical across the trees it measures) keeps working.
              if (classes != "all")
                  throw std::invalid_argument("compile_family_sampler: classes must be 'all' (the record-class "
                                              "bisection knob is deleted at J-b2 J6)");
              if (record_route != "d" && record_route != "term")
                  throw std::invalid_argument("compile_family_sampler: record_route must be 'd' or 'term'");
              return std::make_unique<PyFamilySampler>(text, p_factor, selfcheck, oracle_all_shots,
                                                       decisions, seed,
                                                       record_route == "term", nullptr, disk_path);
          },
          py::arg("text"), py::arg("classes") = "all", py::arg("p_factor") = 1.0,
          py::arg("selfcheck") = 2000, py::arg("oracle_all_shots") = false,
          py::arg("decisions") = true, py::arg("seed") = 1,
          py::arg("record_route") = "d", py::arg("disk_path") = "",
          "J-a TEST API: compile `text` into a FAM twirl sampler (every record class; classes='all' only, for the "
          "ledger harness). J-b1 P4: record_route=\"term\" (oracle) term-routes every shot of a recording run.");
    // J-b1 P5: the deferred stream a FAM sampler's TermResidual reads (the family normalization), with every noise
    // probability times p_factor — the input of the independent static-rule oracle tests/_nc_oracle.py (which
    // shares no code with TermResidual or the bare builder).
    m.def("_deferred_stream", [](const std::string& text, double p_factor) {
        ParsedStim ps = parse_stim_circuit(text);
        if (!ps.ok()) throw std::invalid_argument("_deferred_stream: parse failed");
        NormalizeResult nr = normalize_for_twirl_sampler(ps.circuit);
        py::list out;
        for (const Instr& ins : nr.normalized.stream) {
            py::dict d;
            d["kind"] = (int)ins.kind;
            d["gate"] = (int)ins.gate;
            d["targets"] = ins.targets;
            d["channel"] = (int)ins.channel;
            d["qubits"] = ins.qubits;
            std::vector<double> pr(ins.probs.begin(), ins.probs.end());
            for (double& x : pr) x *= p_factor;
            d["probs"] = pr;
            d["basis"] = (int)ins.basis;
            d["corr"] = std::vector<int>(ins.corr.begin(), ins.corr.end());
            out.append(d);
        }
        return py::make_tuple(nr.normalized.n, out);
    }, py::arg("text"), py::arg("p_factor") = 1.0,
       "J-b1 P5 TEST API: (n, [instruction dicts]) of the family-normalized deferred stream (the NC oracle's input).");
}
