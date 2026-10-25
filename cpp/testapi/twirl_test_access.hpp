// twirl_test_access.hpp — J-b1 P3 (m6): the ONE door to a TwirlRecordSampler's FAM test surface (the family_*
// members and their nested types are private; this struct is their declared friend). Test API only: no public
// binding includes this header. Every member forwards; no logic lives here.
#pragma once

#include "qeccore/twirl_sampler.hpp"

namespace qeccore {

struct TwirlSamplerTestAccess {
    using S = TwirlRecordSampler;
    using FamilyExtra = S::FamilyExtra;
    using FamilyRouteStats = S::FamilyRouteStats;
    using FamilyRecordSink = S::FamilyRecordSink;
    using FamilyRecordContent = S::FamilyRecordContent;

    static std::vector<FamilyExtra> family_extras(const S& s) { return s.family_extras(); }
    static const std::string& family_refusal(const S& s) { return s.family_refusal(); }
    static void family_port_born_ref(S& s, const std::vector<int>& r) { s.family_port_born_ref(r); }   // J-b1 P10
    static std::shared_ptr<TwirlPlanCache> family_plan_cache(const S& s) { return s.family_plan_cache(); }   // P11a
    using FamilyPlanView = typename S::FamilyPlanView;                                                      // P11b
    static FamilyPlanView family_plan_view(const S& s, int h) { return s.family_plan_view(h); }
    static const std::vector<Pauli>& family_pivot_frames(const S& s, int h) { return s.family_pivot_frames(h); }   // P9b
    using FamilyRefusalInfo = S::FamilyRefusalInfo;
    static FamilyRefusalInfo family_refusal_info(const S& s) { return s.family_refusal_info(); }
    static std::vector<Pauli> family_record_ops(const S& s) { return s.family_record_ops(); }
    static std::shared_ptr<const RecTable> family_content_table(const S& s, int h) { return s.family_content_table(h); }
    static void family_record_fidelity(S& s, RecordFidelity& rf, const FamilyRecordSink& rs, long rows,
                                       double* out, const uint8_t* corr = nullptr,
                                       const std::vector<int>* out_wires = nullptr) {
        s.family_record_fidelity(rf, rs, rows, out, corr, out_wires);
    }
    static std::vector<long> semantic_oracle_class_checks(const S& s) { return s.semantic_oracle_class_checks(); }
    static TermResidual& family_terms(S& s) { return s.family_terms(); }
    static const std::vector<std::vector<FiredPauli>>& family_alt_faults(const S& s) { return s.family_alt_faults(); }
    static std::vector<uint8_t> family_alt_is_ppr(const S& s) { return s.family_alt_is_ppr(); }
    static std::vector<std::vector<int32_t>> family_draw_events(S& s, long shots, uint64_t seed) {
        return s.family_draw_events(shots, seed);
    }
    static void family_term_shot(S& s, const std::vector<int32_t>& alts, TermRouteShot& out) {
        s.family_term_shot(alts, out);
    }
    static bool family_diag_shot(S& s, const std::vector<int32_t>& alts, DiagNormalForm& nf, long& plan_id) {
        return s.family_diag_shot(alts, nf, plan_id);
    }
    static std::vector<uint8_t> family_relabel(const S& s, const Pauli& P) { return s.family_relabel(P); }
    static void family_general_table(S& s, const std::vector<uint64_t>& Sw, RecTable& out,
                                     std::vector<int>* intro_b = nullptr) {
        s.family_general_table(Sw, out, intro_b);
    }
    static void family_general_table_diag(const S& s, const uint8_t* plan_key, int n_plan, RecTable& out,
                                          std::vector<int>* intro_b = nullptr) {
        s.family_general_table_diag(plan_key, n_plan, out, intro_b);
    }
    static std::vector<double> family_born_law(const S& s, const RecTable& T) { return s.family_born_law(T); }
    static FamilyRouteStats family_route_stats(const S& s) { return s.family_route_stats(); }
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
                                 std::function<FramedSuperposition()> resupply = nullptr) {   // J-b1 P9b
        return s.family_add_branch(st, dspec, dport, std::move(resupply));
    }
    static void family_set_branch_budget(S& s, size_t bytes) { s.family_set_branch_budget(bytes); }
    static void family_evict_branches(S& s) { s.family_evict_branches(); }
    static void family_set_input_branch(S& s, const int32_t* ids, long n) { s.family_set_input_branch(ids, n); }
    using FamilyBranchCounters = S::FamilyBranchCounters;
    static FamilyBranchCounters family_branch_counters(const S& s) { return s.family_branch_counters(); }
    static void family_signs(const S& s, const FamilyRecordSink& rs, long rows, const std::vector<Pauli>& g,
                             int8_t* out) {
        s.family_signs(rs, rows, g, out);
    }
    static std::vector<int> family_record_classes(const S& s) { return s.family_record_classes(); }
    static std::string family_model_key(const S& s) { return s.family_model_key(); }
    static std::string family_ctor_bytes(const S& s) { return s.family_ctor_bytes(); }   // S8-0b G-PREP
    static std::pair<long, long> family_alt_atoms_oracle(const S& s) { return s.family_alt_atoms_oracle(); }   // S8-1'
    static void family_arm_terms(S& s, const Circuit& deferred, const std::vector<std::pair<int, int>>& reads) {
        s.family_arm_terms(deferred, reads);
    }
    static std::vector<long> family_born_trie_walk(S& s, bool oracle) { return s.family_born_trie_walk(oracle); }
    static std::vector<long> family_trie_nodes(const S& s) { return s.family_trie_nodes(); }
};

}  // namespace qeccore
