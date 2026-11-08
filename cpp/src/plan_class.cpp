// plan_class — the class key map (qeccore/plan_class.hpp): the a ∧ U ∧ ¬prod strip and the compile's report.
#include "qeccore/plan_class.hpp"

namespace qeccore {

PlanClassifier compile_plan_classifier(const CertifiedGroupPlanes& G, const std::vector<RecColumn>& cols,
                                       const std::vector<uint64_t>& prodmask, int n) {
    PlanClassifier pc;
    pc.n = n;
    pc.NW = (n + 63) / 64;
    const int m = (int)cols.size();
    // U: wires with X-support in any record column
    pc.umask.assign((size_t)pc.NW, 0);
    for (const RecColumn& c : cols)
        for (int w = 0; w < pc.NW && w < (int)c.M.x.size(); ++w) pc.umask[(size_t)w] |= c.M.x[(size_t)w];
    for (int w = 0; w < pc.NW && w < (int)prodmask.size(); ++w) pc.umask[(size_t)w] &= ~prodmask[(size_t)w];
    // the report's column count: per record its reduced z-part over the free coordinates of the signed pure-Z subgroup
    // (n minus its rref rows, each on its own pivot wire) plus the sign bit, then the (ii) pairs
    size_t zrows = 0;
    for (int r = 0; r < G.rank; ++r) zrows += G.pivcol[(size_t)r] >= n ? 1 : 0;
    pc.columns = (size_t)m * ((size_t)n - zrows + 1) + (size_t)m * (size_t)(m - 1) / 2;
    return pc;
}

}  // namespace qeccore
