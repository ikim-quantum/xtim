// S2-T3: the bare-state build emits a COMPACT Frame. For every circuit of the corpus (examples/, examples/xdem/,
// tests/data/ and the private corpus, given as directories on the command line) the production build
// (AssemblePath::coset_family: compact frame, the Clifford gates appended to its C) is held to the Stage-1 tableau,
// kept as the oracle — the 3.1.8 construction (AssemblePath::coset_rays: from_generators on all of ray_0's generators,
// dense edits, an expanded frame), which _ray_syndrome_oracle_diff pins to the Stage-1 production build:
//   (a) S ⊇ free: every distinguishing index is a frame index of F_S (row_pos ≥ 0);
//   (b) off F_S the base rows are identity pairs at π(a) (the frame's own invariant: compact() validates π);
//   (c) a copy's dense() == the oracle's dense() byte for byte (forward rows, dual validity), and after ensure_dual
//       on both the dual rows too;
//   (d) stabilizer_generators_sparse == stabilizer_generators (bits, phases, order) on the anchor;
//   (e) QW-F: both == the pre-QW-F emitter kept in stab_generators_oracle.hpp;
// plus the shape hazards: a prepared compact frame that is then expanded is no longer prepared (QW-B stale dual) until
// prepare_queries() runs again; resync_dense_size after a size-changing dense edit. Prints the shape (n', |F_S|, |C|, π ≠ id) per
// circuit; private circuits by index only.
#include "check.hpp"
#include <algorithm>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "qeccore/frame.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/parity_assemble.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/stab_generators.hpp"
#include "qeccore/stim_parse.hpp"
#include "stab_generators_oracle.hpp"

using namespace qeccore;

namespace {

bool same_pauli(const Pauli& a, const Pauli& b) { return a.n == b.n && a.phase == b.phase && a.x == b.x && a.z == b.z; }

bool same_tableau(const CliffordTableau& A, const CliffordTableau& B, std::string* why) {
    if (A.n != B.n) { *why = "n"; return false; }
    for (int a = 0; a < A.n; ++a) {
        if (!same_pauli(A.Zrow[(size_t)a], B.Zrow[(size_t)a])) { *why = "Zrow[" + std::to_string(a) + "]"; return false; }
        if (!same_pauli(A.Xrow[(size_t)a], B.Xrow[(size_t)a])) { *why = "Xrow[" + std::to_string(a) + "]"; return false; }
    }
    if (A.dual_valid() != B.dual_valid()) { *why = "dual validity"; return false; }
    if (A.dual_valid())
        for (int a = 0; a < A.n; ++a)
            if (!same_pauli(A.xinv_row(a), B.xinv_row(a)) || !same_pauli(A.zinv_row(a), B.zinv_row(a))) {
                *why = "dual row " + std::to_string(a);
                return false;
            }
    return true;
}

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void stim_files(const std::string& dir, bool recurse, std::vector<std::string>& out) {
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    std::vector<std::string> names;
    while (dirent* e = readdir(d)) {
        const std::string nm = e->d_name;
        if (nm == "." || nm == "..") continue;
        names.push_back(nm);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    for (const std::string& nm : names) {
        const std::string p = dir + "/" + nm;
        struct stat st;
        if (stat(p.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) { if (recurse) stim_files(p, recurse, out); continue; }
        if (nm.size() > 5 && nm.compare(nm.size() - 5, 5, ".stim") == 0) out.push_back(p);
    }
}

// The production twirl front's normalization (bindings/twirl_front.hpp normalize_for_twirl).
bool deferred_of(const std::string& text, Circuit& out) {
    ParsedStim ps = parse_stim_circuit(text);
    if (!ps.ok()) return false;
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    pol.hoist_feedback_on_class_exit = true;
    out = normalize(ps.circuit, pol).normalized;
    return true;
}

struct Tally { int built = 0, compact = 0, pi_moved = 0, skipped = 0; long gens = 0; };

void check_circuit(const std::string& label, const std::string& text, Tally& t) {
    Circuit d;
    if (!deferred_of(text, d)) { std::fprintf(stderr, "  %-44s skipped: does not parse\n", label.c_str()); ++t.skipped; return; }
    // A refusal is a skip only when BOTH paths refuse with the same message (the circuit is out of the bare state's
    // class); a throw of the compact build alone (or a different message) is a failure.
    BareState prod, orc;
    std::string ep, eo;
    try { prod = build_bare_state_parity(d, AssemblePath::coset_family); } catch (const std::exception& e) { ep = e.what(); if (ep.empty()) ep = "?"; }
    try { orc = build_bare_state_parity(d, AssemblePath::coset_rays); } catch (const std::exception& e) { eo = e.what(); if (eo.empty()) eo = "?"; }
    if (!ep.empty() || !eo.empty()) {
        if (ep != eo) std::fprintf(stderr, "  %s: compact build throws '%s', oracle '%s'\n", label.c_str(), ep.c_str(), eo.c_str());
        CHECK(ep == eo);
        std::fprintf(stderr, "  %-44s skipped: both paths refuse (%.80s)\n", label.c_str(), ep.c_str());
        ++t.skipped;
        return;
    }
    CHECK(prod.rejected == orc.rejected);
    if (prod.rejected || orc.rejected) {
        CHECK(prod.reject_gate_index == orc.reject_gate_index);
        std::fprintf(stderr, "  %-44s skipped: refused by both paths (gate %d)\n", label.c_str(), prod.reject_gate_index);
        ++t.skipped;
        return;
    }
    ++t.built;
    const Frame& F = prod.state.U;
    const int n = F.n;
    // (a) S ⊇ free
    for (int a : prod.state.free) CHECK(F.row_pos(a) >= 0);
    // (b) the shape: off F_S every index has a wire (compact() validated the bijection)
    int moved = 0;
    for (int a = 0; a < n; ++a) {
        if (F.row_pos(a) >= 0) continue;
        CHECK(F.pi(a) >= 0);
        if (F.pi(a) != a) ++moved;
    }
    if (!F.expanded()) ++t.compact;
    if (moved) ++t.pi_moved;
    // (c) the expansion is the Stage-1 tableau, byte for byte (then the duals)
    Frame A = F;
    Frame B = orc.state.U;
    std::string why;
    const bool same = same_tableau(A.dense(), B.dense(), &why);
    if (!same) std::fprintf(stderr, "  %s: compact build != Stage-1 tableau: %s\n", label.c_str(), why.c_str());
    CHECK(same);
    A.dense().ensure_dual();
    B.dense().ensure_dual();
    CHECK(same_tableau(A.dense(), B.dense(), &why));
    CHECK(prod.state.eps == orc.state.eps && prod.state.free == orc.state.free);
    // (d) sparse generators == dense generators on the anchor (an AffineState)
    {
        const std::vector<Pauli> dg = stabilizer_generators(prod.state.anchor_state());
        const std::vector<SparsePauli> sg = stabilizer_generators_sparse(prod.state.anchor_state());
        CHECK(dg.size() == sg.size());
        for (size_t i = 0; i < dg.size() && i < sg.size(); ++i) CHECK(same_pauli(to_dense(sg[i]), dg[i]));
        // (e) QW-F: the support-recording emitter == the pre-QW-F full-width emitter (both outputs)
        const std::vector<Pauli> og = qwf_oracle::stabilizer_generators(prod.state.anchor_state());
        const std::vector<SparsePauli> os = qwf_oracle::stabilizer_generators_sparse(prod.state.anchor_state());
        CHECK(og.size() == dg.size() && os.size() == sg.size());
        for (size_t i = 0; i < og.size() && i < dg.size(); ++i) CHECK(same_pauli(og[i], dg[i]));
        for (size_t i = 0; i < os.size() && i < sg.size(); ++i)
            CHECK(os[i].n == sg[i].n && os[i].phase == sg[i].phase && os[i].x == sg[i].x && os[i].z == sg[i].z);
        t.gens += (long)sg.size();
    }
    std::fprintf(stderr, "  %-44s n'=%6d |F_S|=%5zu |C|=%7zu r=%2zu pi!=id=%5d %s\n", label.c_str(), n,
                 F.expanded() ? (size_t)n : F.support_rows().size(), F.gates().size(), prod.state.free.size(), moved,
                 F.expanded() ? "expanded" : "compact");
}

void test_shape_hazards() {
    // a compact frame: 3 identity pairs around a 2-qubit block, a few gates in C
    CliffordTableau U0(2);
    U0.left_h(0);
    U0.left_cx(0, 1);
    Frame F = Frame::compact(5, {1, 3}, {0, 4}, {2, -1, 1, -1, 3}, U0, {{0, 2, -1}, {6, 2, 3}});
    {
        // S2-CLEAN: prepare_queries() is the one threading contract (warm() is gone). A prepared compact frame that is
        // then expanded is no longer prepared (the expansion leaves the dense dual stale, QW-B); preparing it again is
        // the supported order (the twirl sampler's dense_bare_ does it).
        Frame G = F;
        G.prepare_queries();
        CHECK(G.queries_prepared() && !G.expanded());
        G.ensure_expanded();
        CHECK(G.expanded() && !G.queries_prepared());
        G.prepare_queries();
        CHECK(G.queries_prepared() && G.dense().dual_valid());
    }
    {
        Frame G = F;
        G.ensure_expanded();
        G.prepare_queries();                           // expand, then prepare: the supported order
        CHECK(G.expanded() && G.dense().dual_valid());
        CHECK(G.support_rows().size() == 5 && G.row_pos(4) == 4 && G.pi(4) == -1);
        G.dense().grow_identity_qubits(2);
        G.resync_dense_size();
        CHECK(G.n == 7 && G.expanded() && G.dense().n == 7);
    }
    {
        bool threw = false;
        Frame G = F;
        try { G.resync_dense_size(); } catch (const std::logic_error&) { threw = true; }
        CHECK(threw);                                  // only an expanded frame resyncs
    }
}

}  // namespace

int main(int argc, char** argv) {
    test_shape_hazards();
    Tally t;
    int npriv = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string dir = argv[i];
        const bool priv = dir.size() >= 8 && dir.compare(dir.size() - 8, 8, "/private") == 0;
        std::vector<std::string> files;
        stim_files(dir, !priv, files);
        for (const std::string& p : files) {
            const std::string label = priv ? "private[" + std::to_string(npriv++) + "]" : p.substr(p.rfind('/', p.rfind('/') - 1) + 1);
            check_circuit(label, slurp(p), t);
        }
    }
    std::fprintf(stderr, "[frame_compact_build] built %d (compact %d, pi != id on %d), skipped %d; %ld anchor generators"
                 " == the pre-QW-F emitter\n", t.built, t.compact, t.pi_moved, t.skipped, t.gens);
    CHECK(t.built > 0);
    REPORT();
}
