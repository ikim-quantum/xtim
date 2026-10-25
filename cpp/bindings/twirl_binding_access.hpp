// twirl_binding_access.hpp — J-b1 F: the PUBLIC bindings' door to a TwirlRecordSampler's family (FAM) surface. The
// family_* members are private (m6); the test API reaches all of them through TwirlSamplerTestAccess
// (cpp/testapi/twirl_test_access.hpp), the production binding (cpp/bindings/family_sampler.hpp, shared with the test
// API) only the members below: the layout-3 records and their readers (materialize, signs, record fidelity, plan
// views), the carried branches, and the port reference. Every member forwards; no logic lives here.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "qeccore/twirl_sampler.hpp"

namespace qeccore {

struct TwirlSamplerBindingAccess {
    using S = TwirlRecordSampler;
    using FamilyRecordSink = S::FamilyRecordSink;
    using FamilyRecordContent = S::FamilyRecordContent;
    using FamilyPlanView = typename S::FamilyPlanView;
    using FamilyRefusalInfo = S::FamilyRefusalInfo;
    using FamilyBranchCounters = S::FamilyBranchCounters;
    using FamilyRouteStats = S::FamilyRouteStats;

    static const std::string& family_refusal(const S& s) { return s.family_refusal(); }
    static FamilyRefusalInfo family_refusal_info(const S& s) { return s.family_refusal_info(); }
    static void family_port_born_ref(S& s, const std::vector<int>& r) { s.family_port_born_ref(r); }
    static std::shared_ptr<TwirlPlanCache> family_plan_cache(const S& s) { return s.family_plan_cache(); }
    static FamilyPlanView family_plan_view(const S& s, int h) { return s.family_plan_view(h); }
    static const std::vector<Pauli>& family_pivot_frames(const S& s, int h) { return s.family_pivot_frames(h); }
    static std::vector<Pauli> family_record_ops(const S& s) { return s.family_record_ops(); }
    static std::shared_ptr<const RecTable> family_content_table(const S& s, int h) { return s.family_content_table(h); }
    static void family_record_fidelity(S& s, RecordFidelity& rf, const FamilyRecordSink& rs, long rows,
                                       double* out, const uint8_t* corr = nullptr,
                                       const std::vector<int>* out_wires = nullptr) {
        s.family_record_fidelity(rf, rs, rows, out, corr, out_wires);
    }
    static TermResidual& family_terms(S& s) { return s.family_terms(); }
    static void set_family_record_sink(S& s, FamilyRecordSink* rs) { s.set_family_record_sink(rs); }
    static const std::vector<FamilyRecordContent>& family_record_contents(const S& s) {
        return s.family_record_contents();
    }
    static int family_intern_content(S& s, int kind, const std::string& content) {
        return s.family_intern_content(kind, content);
    }
    static FramedSuperposition family_materialize(const S& s, int handle, const uint64_t* prec, const uint64_t* y,
                                                  bool tableau = false, int branch = 0) {
        return s.family_materialize(handle, prec, y, tableau, branch);
    }
    static int family_add_branch(S& s, const FramedSuperposition& st, const Pauli& dspec,
                                 const std::vector<uint8_t>& dport,
                                 std::function<FramedSuperposition()> resupply = nullptr) {
        return s.family_add_branch(st, dspec, dport, std::move(resupply));
    }
    static void family_set_branch_budget(S& s, size_t bytes) { s.family_set_branch_budget(bytes); }
    static void family_evict_branches(S& s) { s.family_evict_branches(); }
    static void family_set_input_branch(S& s, const int32_t* ids, long n) { s.family_set_input_branch(ids, n); }
    static FamilyBranchCounters family_branch_counters(const S& s) { return s.family_branch_counters(); }
    static void family_signs(const S& s, const FamilyRecordSink& rs, long rows, const std::vector<Pauli>& g,
                             int8_t* out) {
        s.family_signs(rs, rows, g, out);
    }
    static std::vector<int> family_record_classes(const S& s) { return s.family_record_classes(); }
    static std::string family_model_key(const S& s) { return s.family_model_key(); }
    static void family_arm_terms(S& s, const Circuit& deferred, const std::vector<std::pair<int, int>>& reads) {
        s.family_arm_terms(deferred, reads);
    }
    static std::vector<long> family_born_trie_walk(S& s, bool oracle) { return s.family_born_trie_walk(oracle); }
    static FamilyRouteStats family_route_stats(const S& s) { return s.family_route_stats(); }
};

}  // namespace qeccore
