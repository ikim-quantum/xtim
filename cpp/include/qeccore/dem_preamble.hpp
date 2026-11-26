// dem_preamble.hpp — the DEM export's shared preamble (factored out of export_dem so the xdem writer reuses it;
// export_dem's bytes are unchanged — the code moved, nothing in it changed).
//
// parse -> DEM normalization preset (KeepCoherent feedback + defer + eliminate-Hadamards, with the diagonal-class
// fallback and the CH-class table) -> propagation-class check -> bare state (FastTODD; capacity refusals) -> terminal
// reads (each on its own wire) -> residual-feedback fold (the reported records are a unitriangular image L of the raw
// records; every record-defined column of `p` is relabelled to its raw set L^T S in place).
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "qeccore/dem_export.hpp"
#include "qeccore/feedback.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/propagation_table.hpp"
#include "qeccore/stim_parse.hpp"

namespace qeccore {

// The DEM path's normalization + propagation class, with the diagonal-class fallback. On a
// fallback that still rejects, the DEFAULT normalization's result (and reject index) is kept.
struct DemNormalized {
    NormalizeResult nr;
    Circuit deferred;
    PropagationTable table;
    bool diagonal_copy = false;   // the fallback deferral was used (a feedback probe must defer the same way)
    bool general = false;         // dem-ch: the table was built with the PPR retry (a CH-class circuit: some atom's
                                  // end error is a PprResidual, PropResult::ppr; no tableau is stored per atom)
};

DemNormalized dem_normalize(const Circuit& user, const NormalizePolicy& policy_in);
// The default normalization alone, no propagation table (`table` default-constructed: all_in_class, no locations;
// diagonal_copy = general = false): the xdem lean route (LEAN-EXPORT-PLAN Task 3). Exact only when the caller then
// requires WriterModel::h_meets_location == false, which implies the default table's all_in_class.
DemNormalized dem_normalize_default_only(const Circuit& user, const NormalizePolicy& policy_in);

struct DemPreamble {
    ParsedStim p;                              // detectors / observables / decisions / PE frames: RAW record sets
                                               // (folded through the feedback relabel when the circuit has feedback)
    DemNormalized dn;
    FramedBareState bare;
    std::vector<std::pair<int, int>> reads;    // terminal reads, record order: (0 X / 1 Y / 2 Z, deferred wire)
    int M = 0;                                 // number of records
    FeedbackPlan fb;                           // residual feedback (empty when the circuit has none)
    FeedbackFold fbf;
};

// Runs the preamble. On a refusal returns false with `res` filled exactly as export_dem fills it (parse_errors /
// rejected + reject_gate_index / error). `include_expectations` and `census` only steer the census bookkeeping.
// build_bare = false skips the bare-state build; `pre.bare` is then empty (rejected = true) and must not be read
// until dem_build_bare(pre, res) builds it. The xdem export (format = Xdem, ruling R9) runs the preamble this way,
// then the writer model (its diagonality check names both locations — a non-diagonal residue would otherwise be
// refused first by the bare-state build's generic "non-commuting magic"), then dem_build_bare for the determinism
// gate. The default export (build_bare = true) is unchanged. (The test API's xdem_header_debug also uses it.)
// class_table = false runs dem_normalize_default_only instead of dem_normalize (no propagation table, so no class
// refusal and no CH-class feedback refusal); the xdem export's lean route passes false.
bool dem_preamble(const std::string& stim_text, bool include_expectations, DemLawCensus* census,
                  DemPreamble& pre, DemExportResult& res, bool build_bare = true, bool class_table = true);

// The preamble's bare-state build (FastTODD; capacity refusals): on a refusal returns false with res.rejected /
// res.reject_gate_index filled exactly as dem_preamble fills them.
bool dem_build_bare(DemPreamble& pre, DemExportResult& res);

}  // namespace qeccore
