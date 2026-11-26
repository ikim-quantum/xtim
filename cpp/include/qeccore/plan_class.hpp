#pragma once
// plan_class — the class key map of the diagonal plan key (plancache-design/plan.md §0-§1, owner-signed rev 4).
// An a-bit on a wire with X-support in NO record column changes no pullback C0†M_cC0 (Theorem T3's a-part), so the class
// key of a diagonal-cache key is a ∧ U ∧ ¬prod, U = the wires with X-support in any record column, prod = the product
// wires. lean D1-S3 (owner-signed deletions): the rotation loop serves every circuit with rotations, so a cache key holds no
// CZ pair — the K2' kernel over the pair universe (T3's B-part), its elimination, cap and O3 audit are deleted.
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "qeccore/record_table.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

struct PlanClassifier {
    int n = 0, NW = 0;
    std::vector<uint64_t> umask;        // NW words: U ∧ ¬prod — the a-part of the class key is key & umask
    // the compile's record (the PLAN_CLASS diag event, record_report)
    size_t columns = 0;                 // Σ_c (free Z coordinates + 1) + m(m − 1)/2: the (i) / (ii) condition columns
    std::string cap;                    // "" (no kernel is ever capped: there is no pair universe)

    size_t dim() const { return 0; }    // dim K2' (lean D1-S3: no pair universe, so no kernel)
    // Π in place on a cache key (NW a-words): the a ∧ U ∧ ¬prod strip.
    void project(uint64_t* key) const { for (int w = 0; w < NW; ++w) key[w] &= umask[(size_t)w]; }
    // True iff Π(key) == key (the key is its own class representative).
    bool is_representative(const uint64_t* key) const {
        for (int w = 0; w < NW; ++w) if (key[w] & ~umask[(size_t)w]) return false;
        return true;
    }
};

// Compile the classifier of a sampler's key universe: the certified group G (its signed pure-Z subgroup: the report's
// column count), the record columns (storage order; every column counts) and the product-wire mask.
PlanClassifier compile_plan_classifier(const CertifiedGroupPlanes& G, const std::vector<RecColumn>& cols,
                                       const std::vector<uint64_t>& prodmask, int n);

}  // namespace qeccore
