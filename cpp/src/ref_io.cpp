#include "qeccore/ref_io.hpp"
#include <cstddef>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "qeccore/pauli.hpp"
#include "qeccore/stab_generators.hpp"
#include "qeccore/stab_disentangle.hpp"

namespace qeccore {

namespace {

using cd = std::complex<double>;

constexpr int OP_H = 0, OP_S = 1, OP_SDG = 2, OP_X = 3, OP_Y = 4, OP_Z = 5,
              OP_CX = 6, OP_CZ = 7;
const char* const OP_NAMES[8] = {"H", "S", "SDG", "X", "Y", "Z", "CX", "CZ"};

}  // namespace

// Materialize a sum's branch rays |phi_i> = D_{sigma_i}|phi0> (protocol_reference's recipe;
// verify_invariants() is the public flush of any pending gates). PUBLIC (declared in ref_io.hpp)
// so protocol_reference shares this one definition instead of carrying its own static copy.
std::vector<std::unique_ptr<AffineState>> materialize_rays(const CanonicalStabSum& s) {
    s.ensure_frame_current();   // flush pending gates (NOT the O(χ²) verify_invariants audit)
    std::vector<std::unique_ptr<AffineState>> out;
    out.reserve(s.branches.size());
    for (const auto& br : s.branches) {
        auto st = s.anchor->clone();
        for (int a = 0; a < (int)s.free.size(); ++a)
            if (br.sigma[a]) s.U.Xrow[s.free[a]].apply_to(*st);
        out.push_back(std::unique_ptr<AffineState>(static_cast<AffineState*>(st.release())));
    }
    return out;
}

void apply_prep_gate(AffineState& s, const RefFile::PrepGate& g) {
    switch (g.op) {
        case OP_H:   s.apply_h(g.a);   break;
        case OP_S:   s.apply_s(g.a);   break;
        case OP_SDG: s.apply_sdg(g.a); break;
        case OP_X:   s.apply_x(g.a);   break;
        case OP_Y:   s.apply_y(g.a);   break;
        case OP_Z:   s.apply_z(g.a);   break;
        case OP_CX:  s.apply_cx(g.a, g.b); break;
        case OP_CZ:  s.apply_cz(g.a, g.b); break;
        default: throw std::logic_error("apply_prep_gate: bad op");
    }
}

// ── prep synthesis (tableau reduction) ──────────────────────────────────────────────────
//
// Reduce the ray's +1 stabilizer generators to {+Z_0, ..., +Z_{n-1}} by conjugating with
// H/S/X/CX/CZ. The REVERSED, inverted gate list (S <-> SDG; the rest self-inverse) prepares
// the ray from |0...0> up to a unit phase. Now delegates the core reduction to the shared
// disentangle_from_generators helper (also used by canonical_stab_sum.cpp's compute_fsynd).
std::vector<RefFile::PrepGate> synthesize_prep(const AffineState& ray) {
    const int n = ray.n();
    std::vector<Pauli> rows = stabilizer_generators(ray);
    if ((int)rows.size() != n)
        throw std::runtime_error("synthesize_prep: generator count != n");
    // Get the DISENTANGLING gate list (sends |ray> -> e^{i theta}|0...0>).
    // DisentangleGate and RefFile::PrepGate use the SAME op numbering (0:H..7:CZ), so
    // the conversion is a field-by-field copy.
    std::vector<DisentangleGate> dis = disentangle_from_generators(std::move(rows));
    // Invert: reverse order, S <-> SDG (H/X/Y/Z/CX/CZ are involutions). The inverted list
    // maps |0...0> -> e^{i theta}|ray> (the PREP direction).
    std::vector<RefFile::PrepGate> prep;
    prep.reserve(dis.size());
    for (auto it = dis.rbegin(); it != dis.rend(); ++it) {
        int op = it->op;
        if (op == OP_S) op = OP_SDG;
        else if (op == OP_SDG) op = OP_S;
        prep.push_back({op, it->a, it->b});
    }
    return prep;
}

// ── framed anchor reconstruction + ray materialization ─────────────────────────────────
//
// The framed rep drops the affine anchor; for exact-overlap verification we rebuild it from
// the frame alone. |phi0> is determined (up to a global phase every consumer discards) by
// g_a|phi0> = (-1)^{eps[a]}|phi0>, g_a = U.Zrow[a]: fold eps into each stored Pauli's phase
// ((-1) = i^2, so phase += 2*eps[a] mod 4), disentangle, invert, replay onto |0...0> — the
// exact battle-tested synthesize_prep route, no new phase math.
AffineState anchor_from_frame(const FramedSuperposition& L) {
    const int n = L.n();
    if ((int)L.U.Zrow.size() != n || (int)L.eps.size() != n)
        throw std::runtime_error("anchor_from_frame: frame Zrow/eps size != n");
    std::vector<Pauli> rows;
    rows.reserve(n);
    for (int a = 0; a < n; ++a) {
        Pauli P = L.U.Zrow[a];                       // signed generator, sign folded in:
        P.phase = (P.phase + 2 * (int)(L.eps[a] & 1)) & 3;   // (-1)^eps = i^{2 eps}
        rows.push_back(std::move(P));
    }
    // Disentangling gate list G: G|phi0> = e^{i theta}|0...0>. Invert (reverse order,
    // S <-> SDG; the rest are involutions) and replay on |0...0> to prepare |phi0>.
    std::vector<DisentangleGate> dis = disentangle_from_generators(std::move(rows));
    AffineState st(n);
    for (auto it = dis.rbegin(); it != dis.rend(); ++it) {
        DisentangleGate g = *it;
        if (g.op == OP_S) g.op = OP_SDG;
        else if (g.op == OP_SDG) g.op = OP_S;
        apply_disentangle_gate(st, g);
    }
    // ALWAYS-ON correctness check: <phi0| g_a |phi0> must be EXACTLY (-1)^{eps[a]}
    // (ExactPhase integer compare: unit magnitude scale==0, z8 == 4*eps[a] since -1 = zeta8^4).
    // A violation is a REAL phase bug in the frame/eps/reconstruction — fail loud, never loosen.
    for (int a = 0; a < n; ++a) {
        auto probe = st.clone();                     // g_a |phi0>
        L.U.Zrow[a].apply_to(*probe);
        ExactPhase ov = st.inner_product(*probe);
        const int want_z8 = L.eps[a] ? 4 : 0;
        if (ov.is_zero || ov.scale != 0 || ov.z8 != want_z8)
            throw std::runtime_error(
                "anchor_from_frame: reconstructed anchor violates generator " +
                std::to_string(a) + " sign (eps=" + std::to_string((int)L.eps[a]) +
                ", overlap " + (ov.is_zero ? std::string("0")
                                           : "scale=" + std::to_string(ov.scale) +
                                             " z8=" + std::to_string(ov.z8)) + ")");
    }
    return st;
}

// Framed twin of materialize_rays(CanonicalStabSum): anchor reconstructed from the frame,
// then |phi_i> = D_{sigma_i}|phi0| via U.Xrow[free[a]] per set sigma bit of entry i. The
// whole list shares the anchor's arbitrary reconstruction phase (relative structure exact).
std::vector<std::unique_ptr<AffineState>> materialize_rays(const FramedSuperposition& L) {
    AffineState anchor = anchor_from_frame(L);
    const auto& eb = L.entries();
    std::vector<std::unique_ptr<AffineState>> out;
    out.reserve(eb.size());
    for (const auto& e : eb) {
        auto st = std::make_unique<AffineState>(anchor);
        for (int a = 0; a < (int)L.free.size(); ++a)
            if (e.first[a]) L.U.Xrow[L.free[a]].apply_to(*st);
        out.push_back(std::move(st));
    }
    return out;
}

RefFile ref_from_state(const CanonicalStabSum& state, const std::string& source,
                       const std::string& provenance) {
    // CSS-input serialization writes v3 (structure-preserving, direct-load). PRODUCTION writes
    // v4 from the framed state (ref_from_state_v4, Phase B3); this entry stays for CSS-holding
    // callers/tests. The old v2 from_rays writer was removed; the v2 READER (parser/loader/lean
    // fallback) is kept so legacy ref-format-2 files still load.
    return ref_from_state_v3(state, source, provenance);
}

// RETAINED AS TEST INFRASTRUCTURE (2026-07-02): production writes v4 (ref_from_state_v4, framed,
// no anchor). The v3 writer's only callers are test_ref_io's reader-coverage tests — the v3
// READER must live as long as published v3 files exist, and a writer is how the reader stays
// testable on synthetic states (chi 1/2/4) beyond the three committed fixtures. Not dead code.
// ── v3 writer: structure-preserving (direct), no from_rays on load ──────────────────────
// Reads eps/free/sigma + CliffordTableau rows directly from the CanonicalStabSum; anchor
// is synthesized once (same as v2). The loader sets U.Zrow/Xrow directly from stored rows.
RefFile ref_from_state_v3(const CanonicalStabSum& state, const std::string& source,
                          const std::string& provenance) {
    RefFile rf;
    rf.version = 3;
    rf.source = source;
    rf.n = state.n();
    rf.chi = state.chi();
    rf.provenance = provenance;

    state.ensure_frame_current();                    // flush pending gates before reading U/anchor
    const AffineState& anchor = static_cast<const AffineState&>(*state.anchor);
    rf.anchor_prep = synthesize_prep(anchor);        // ONE O(n³) synthesis (same as v2)

    // Probe the rebuilt anchor to determine the global phase correction.
    // synthesize_prep produces a circuit mapping |0...0> → e^{iθ}|anchor>; the phase θ may
    // differ from the original anchor's global phase. We fold the phase correction into each
    // branch coefficient, exactly as ref_from_state (v2) does for per-branch preps.
    AffineState probe_anchor(rf.n);
    for (const auto& g : rf.anchor_prep) apply_prep_gate(probe_anchor, g);
    // inner_product(probe, original) = e^{i(θ_probe - θ_original)} (unit phase since same
    // stabilizer support). The coefficients must be multiplied by this phase.
    ExactPhase anchor_phase_ov = probe_anchor.inner_product(anchor);
    if (anchor_phase_ov.is_zero) throw std::runtime_error("ref_from_state_v3: anchor synthesize_prep overlap is zero");
    cd anchor_phase = anchor_phase_ov.to_complex();

    // Store frame: Zrow[0..n-1] then Xrow[0..n-1]
    const int n = rf.n;
    rf.frame_rows.reserve(2 * n);
    auto store_pauli = [&](const Pauli& P) {
        RefFile::StoredPauli sp;
        sp.phase = P.phase;
        sp.x = P.x;
        sp.z = P.z;
        rf.frame_rows.push_back(std::move(sp));
    };
    for (int a = 0; a < n; ++a) store_pauli(state.U.Zrow[a]);
    for (int a = 0; a < n; ++a) store_pauli(state.U.Xrow[a]);

    // r free generators, eps (n bits), per-branch sigma bitmask
    const int r = (int)state.free.size();
    if (r >= 64) throw std::runtime_error("ref_from_state_v3: r >= 64 unsupported (sigma bitmask is 64-bit)");
    rf.r = r;
    rf.free_gens = state.free;                       // copy indices directly
    rf.eps.assign(state.eps.begin(), state.eps.end());  // copy n sign bits

    for (size_t i = 0; i < state.branches.size(); ++i) {
        RefFile::Branch br;
        // Apply anchor phase correction: c_corrected = c_i * <probe_anchor|anchor>
        // so that when the loader uses probe_anchor (which = e^{iθ}|anchor>) the state matches.
        br.coeff = state.branches[i].c * anchor_phase;
        // Encode sigma as a bitmask: bit d = sigma[d]
        uint64_t mask = 0;
        for (int d = 0; d < r; ++d)
            if (state.branches[i].sigma[d]) mask |= (1ull << d);
        br.sigma_mask = mask;
        rf.branches.push_back(std::move(br));
    }
    return rf;
}

// ── v4 writer: written FROM the framed state — no anchor block, no phase folding ────────
// The frame/free/eps/sigma encodings are byte-identical to v3 (v3's eps is stored explicitly
// and its Zrow/Xrow phases are stored as-is, so nothing about them ever depended on the
// anchor). The ONLY v3 machinery dropped is the anchor_prep synthesis + the anchor-synthesis
// phase v3 folded into every coefficient equally (needed only so the reloaded AFFINE state
// bit-matched; every consumer takes |overlap|). Coefficients are therefore written AS-IS,
// making framed_from_ref(parse(ref_to_text(rf))) reproduce the input field-for-field.
RefFile ref_from_state_v4(const FramedSuperposition& state, const std::string& source,
                          const std::string& provenance) {
    RefFile rf;
    rf.version = 4;
    rf.source = source;
    rf.n = state.n();
    rf.chi = state.chi();
    rf.provenance = provenance;

    const int n = rf.n;
    if ((int)state.U.Zrow.size() != n || (int)state.U.Xrow.size() != n ||
        (int)state.eps.size() != n)
        throw std::runtime_error("ref_from_state_v4: frame Zrow/Xrow/eps size != n");

    // Store frame: Zrow[0..n-1] then Xrow[0..n-1] (identical layout to v3)
    rf.frame_rows.reserve(2 * n);
    auto store_pauli = [&](const Pauli& P) {
        RefFile::StoredPauli sp;
        sp.phase = P.phase;
        sp.x = P.x;
        sp.z = P.z;
        rf.frame_rows.push_back(std::move(sp));
    };
    for (int a = 0; a < n; ++a) store_pauli(state.U.Zrow[a]);
    for (int a = 0; a < n; ++a) store_pauli(state.U.Xrow[a]);

    // r free generators, eps (n bits), per-branch sigma bitmask (identical to v3)
    const int r = (int)state.free.size();
    if (r >= 64) throw std::runtime_error("ref_from_state_v4: r >= 64 unsupported (sigma bitmask is 64-bit)");
    rf.r = r;
    rf.free_gens = state.free;
    rf.eps.assign(state.eps.begin(), state.eps.end());

    const auto& eb = state.entries();
    for (const auto& e : eb) {
        if ((int)e.first.size() != r)
            throw std::runtime_error("ref_from_state_v4: entry sigma size != r");
        RefFile::Branch br;
        br.coeff = e.second;                 // AS-IS: no anchor phase exists to fold
        uint64_t mask = 0;
        for (int d = 0; d < r; ++d)
            if (e.first[d]) mask |= (1ull << d);
        br.sigma_mask = mask;
        rf.branches.push_back(std::move(br));
    }
    return rf;
}

// ── writer ────────────────────────────────────────────────────────────────────────────

std::string ref_to_text(const RefFile& rf) {
    std::string out;
    char buf[320];
    std::snprintf(buf, sizeof buf, "ref-format %d\n", rf.version);
    out += buf;
    out += "source " + rf.source + "\n";
    std::snprintf(buf, sizeof buf, "n %d\n", rf.n);
    out += buf;
    std::snprintf(buf, sizeof buf, "chi %d\n", rf.chi);
    out += buf;
    if (rf.version >= 3) {
        std::snprintf(buf, sizeof buf, "r %d\n", rf.r);
        out += buf;
    }
    out += "# " + rf.provenance + "\n";
    auto emit_gates = [&](const std::vector<RefFile::PrepGate>& gs) {
        char gbuf[160];
        for (const auto& g : gs) {
            if (g.b >= 0) std::snprintf(gbuf, sizeof gbuf, "%s %d %d\n", OP_NAMES[g.op], g.a, g.b);
            else          std::snprintf(gbuf, sizeof gbuf, "%s %d\n", OP_NAMES[g.op], g.a);
            out += gbuf;
        }
    };
    if (rf.version <= 3) {                 // v4 has NO anchor block (retired)
        out += "anchor\n";
        emit_gates(rf.anchor_prep);
        out += "endanchor\n";
    }
    if (rf.version >= 3) {
        // frame block: Zrow[0..n-1] then Xrow[0..n-1]
        // Each row: "<type> <phase> <x_hex_w0> [<x_hex_w1>...] <z_hex_w0> [...]"
        out += "frame\n";
        const int W = (rf.n + 63) / 64;
        if ((int)rf.frame_rows.size() != 2 * rf.n)
            throw std::runtime_error("ref_to_text v3/v4: frame_rows size != 2*n");
        auto emit_row = [&](const char* type, const RefFile::StoredPauli& sp) {
            std::snprintf(buf, sizeof buf, "%s %d", type, sp.phase);
            out += buf;
            // x words
            for (int w = 0; w < W; ++w) {
                std::snprintf(buf, sizeof buf, " %llx",
                              (unsigned long long)(w < (int)sp.x.size() ? sp.x[w] : 0ull));
                out += buf;
            }
            // z words
            for (int w = 0; w < W; ++w) {
                std::snprintf(buf, sizeof buf, " %llx",
                              (unsigned long long)(w < (int)sp.z.size() ? sp.z[w] : 0ull));
                out += buf;
            }
            out += "\n";
        };
        for (int a = 0; a < rf.n; ++a) emit_row("Z", rf.frame_rows[a]);
        for (int a = 0; a < rf.n; ++a) emit_row("X", rf.frame_rows[rf.n + a]);
        out += "endframe\n";
        // free: space-separated indices
        out += "free";
        for (int idx : rf.free_gens) {
            std::snprintf(buf, sizeof buf, " %d", idx);
            out += buf;
        }
        out += "\n";
        // eps: space-separated bits (0/1)
        out += "eps";
        for (uint8_t b : rf.eps) {
            out += (b ? " 1" : " 0");
        }
        out += "\n";
        // branch lines: re im sigma_hex (no endbranch)
        for (const auto& br : rf.branches) {
            std::snprintf(buf, sizeof buf, "branch %.17g %.17g %llx\n",
                          br.coeff.real(), br.coeff.imag(),
                          (unsigned long long)br.sigma_mask);
            out += buf;
        }
    } else {
        // v2: branch with prep suffix + endbranch
        for (const auto& br : rf.branches) {
            std::snprintf(buf, sizeof buf, "branch %.17g %.17g\n", br.coeff.real(),
                          br.coeff.imag());
            out += buf;
            emit_gates(br.prep);
            out += "endbranch\n";
        }
    }
    return out;
}

bool write_ref_file(const RefFile& rf, const std::string& path, std::string* err) {
    std::ofstream o(path);
    if (!o) {
        if (err) *err = "cannot open " + path + " for writing";
        return false;
    }
    o << ref_to_text(rf);
    o.flush();
    if (!o.good()) {
        if (err) *err = "write to " + path + " failed";
        return false;
    }
    return true;
}

// ── parser ────────────────────────────────────────────────────────────────────────────

namespace {

struct LineReader {
    std::istringstream in;
    int lineno = 0;
    explicit LineReader(const std::string& text) : in(text) {}
    // Next significant line (blank/'#' skipped), tokenized. Returns false at EOF.
    bool next(std::vector<std::string>* toks) {
        std::string line;
        while (std::getline(in, line)) {
            ++lineno;
            std::istringstream ls(line);
            std::vector<std::string> t;
            std::string w;
            while (ls >> w) t.push_back(w);
            if (t.empty() || t[0][0] == '#') continue;
            *toks = std::move(t);
            return true;
        }
        return false;
    }
};

bool parse_int(const std::string& s, int* out) {
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') return false;
    // Range check: ERANGE (beyond long) plus explicit [0, INT_MAX] — a silent long->int
    // truncation would alias huge qubit indices onto valid ones and OOM the loader on n.
    if (errno == ERANGE || v < 0 || v > INT_MAX) return false;
    *out = (int)v;
    return true;
}

bool parse_double(const std::string& s, double* out) {
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') return false;
    *out = v;
    return true;
}

std::string at_line(int lineno, const std::string& msg) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "line %d: ", lineno);
    return buf + msg;
}

int op_from_token(const std::string& t) {
    for (int i = 0; i < 8; ++i)
        if (t == OP_NAMES[i]) return i;
    return -1;
}

}  // namespace

RefParseResult parse_ref_text(const std::string& text) {
    RefParseResult res;
    LineReader rd(text);
    std::vector<std::string> t;
    auto fail = [&](const std::string& msg) {
        res.ok = false;
        res.error = at_line(rd.lineno, msg);
        return res;
    };
    // magic + version
    if (!rd.next(&t)) return fail("empty file (expected 'ref-format 2', 3 or 4)");
    if (t[0] != "ref-format" || t.size() != 2)
        return fail("expected 'ref-format <version>' as the first line");
    if (!parse_int(t[1], &res.file.version) ||
        (res.file.version != 2 && res.file.version != 3 && res.file.version != 4))
        return fail("unsupported ref-format version '" + t[1] +
                    "' (expected 2, 3 or 4; regenerate with current ref_compile)");
    const int file_version = res.file.version;
    // header keys: source, n, chi (+ r for v3/v4) — in order, duplicates rejected
    bool have_source = false, have_n = false, have_chi = false, have_r = false;
    while (!(have_source && have_n && have_chi && (file_version < 3 || have_r))) {
        if (!rd.next(&t)) return fail("truncated header (need source, n, chi" +
                                      std::string(file_version >= 3 ? ", r" : "") + ")");
        if (t[0] == "source") {
            if (have_source) return fail("duplicate 'source'");
            if (t.size() != 2) return fail("expected 'source <filename>' (single token)");
            res.file.source = t[1];
            have_source = true;
        } else if (t[0] == "n") {
            if (!have_source) return fail("'n' before 'source'");
            if (have_n) return fail("duplicate 'n'");
            if (t.size() != 2 || !parse_int(t[1], &res.file.n) || res.file.n <= 0)
                return fail("expected 'n <positive int>'");
            have_n = true;
        } else if (t[0] == "chi") {
            if (!have_n) return fail("'chi' before 'n'");
            if (have_chi) return fail("duplicate 'chi'");
            if (t.size() != 2 || !parse_int(t[1], &res.file.chi) || res.file.chi <= 0)
                return fail("expected 'chi <positive int>'");
            have_chi = true;
        } else if (t[0] == "r" && file_version >= 3) {
            if (!have_chi) return fail("'r' before 'chi'");
            if (have_r) return fail("duplicate 'r'");
            if (t.size() != 2 || !parse_int(t[1], &res.file.r) || res.file.r < 0)
                return fail("expected 'r <non-negative int>'");
            have_r = true;
        } else {
            return fail("unexpected header token '" + t[0] + "'");
        }
    }
    // gate-line parser: validates op + qubit ranges, returns error string or "" on success.
    const int n = res.file.n;
    auto parse_gate = [&](const std::vector<std::string>& tk, RefFile::PrepGate* g) -> std::string {
        int op = op_from_token(tk[0]);
        if (op < 0) return "unknown gate token '" + tk[0] + "'";
        const bool two_q = (op == OP_CX || op == OP_CZ);
        const size_t want = two_q ? 3 : 2;
        if (tk.size() != want)
            return std::string("gate ") + OP_NAMES[op] + " expects " +
                   (two_q ? "2 qubit indices" : "1 qubit index");
        g->op = op; g->a = -1; g->b = -1;
        if (!parse_int(tk[1], &g->a) || g->a < 0 || g->a >= n)
            return "qubit index '" + tk[1] + "' out of range [0," + std::to_string(n) + ")";
        if (two_q) {
            if (!parse_int(tk[2], &g->b) || g->b < 0 || g->b >= n)
                return "qubit index '" + tk[2] + "' out of range [0," + std::to_string(n) + ")";
            if (g->a == g->b)
                return std::string(OP_NAMES[op]) + " with identical qubits";
        }
        return "";
    };
    // anchor block (required in v2 and v3; ABSENT in v4 — the anchor block is retired,
    // so a v4 file goes straight to its frame block and an 'anchor' token there fails)
    if (file_version <= 3) {
        if (!rd.next(&t) || t[0] != "anchor" || t.size() != 1)
            return fail("expected 'anchor' block after header");
        bool anchor_closed = false;
        while (rd.next(&t)) {
            if (t[0] == "endanchor") {
                if (t.size() != 1) return fail("'endanchor' takes no arguments");
                anchor_closed = true;
                break;
            }
            RefFile::PrepGate g{0, -1, -1};
            std::string e = parse_gate(t, &g);
            if (!e.empty()) return fail(e);
            res.file.anchor_prep.push_back(g);
        }
        if (!anchor_closed) return fail("truncated anchor block (missing 'endanchor')");
    }

    if (file_version >= 3) {
        // ── v3/v4 body: frame block, free, eps, branch lines ────────────────────────────
        const int r = res.file.r;
        const int W = (n + 63) / 64;  // Pauli word count

        // frame block: "frame\nZ ... (n lines)\nX ... (n lines)\nendframe"
        if (!rd.next(&t) || t[0] != "frame" || t.size() != 1)
            return fail(file_version == 3
                            ? "expected 'frame' block after anchor in v3 format"
                            : "expected 'frame' block after header in v4 format (no anchor block)");
        {
            res.file.frame_rows.resize(2 * n);
            // Parse n Zrow lines (prefix "Z") then n Xrow lines (prefix "X")
            int zcount = 0, xcount = 0;
            bool frame_closed = false;
            while (rd.next(&t)) {
                if (t[0] == "endframe") {
                    if (t.size() != 1) return fail("'endframe' takes no arguments");
                    frame_closed = true;
                    break;
                }
                if (t[0] != "Z" && t[0] != "X")
                    return fail("expected 'Z' or 'X' Pauli row in frame block, got '" + t[0] + "'");
                // Expected: <type> <phase> <W x_words> <W z_words> = 2+2*W tokens
                const size_t want = 2 + 2 * (size_t)W;
                if (t.size() != want)
                    return fail(std::string(t[0] == "Z" ? "Z" : "X") + " row expects " +
                                std::to_string(want) + " tokens (phase + " + std::to_string(W) +
                                " x-words + " + std::to_string(W) + " z-words)");
                int ph = 0;
                if (!parse_int(t[1], &ph) || ph < 0 || ph > 3)
                    return fail("Pauli phase must be 0/1/2/3, got '" + t[1] + "'");
                RefFile::StoredPauli sp;
                sp.phase = ph;
                sp.x.resize(W, 0);
                sp.z.resize(W, 0);
                for (int w = 0; w < W; ++w) {
                    char* hex_end = nullptr;
                    errno = 0;
                    unsigned long long v = std::strtoull(t[2 + w].c_str(), &hex_end, 16);
                    if (hex_end == t[2 + w].c_str() || *hex_end != '\0' || errno == ERANGE)
                        return fail("invalid x-word hex '" + t[2 + w] + "'");
                    sp.x[w] = (uint64_t)v;
                }
                for (int w = 0; w < W; ++w) {
                    char* hex_end = nullptr;
                    errno = 0;
                    unsigned long long v = std::strtoull(t[2 + W + w].c_str(), &hex_end, 16);
                    if (hex_end == t[2 + W + w].c_str() || *hex_end != '\0' || errno == ERANGE)
                        return fail("invalid z-word hex '" + t[2 + W + w] + "'");
                    sp.z[w] = (uint64_t)v;
                }
                if (t[0] == "Z") {
                    if (zcount >= n) return fail("too many Z rows in frame block (expected " + std::to_string(n) + ")");
                    res.file.frame_rows[zcount++] = std::move(sp);
                } else {
                    if (xcount >= n) return fail("too many X rows in frame block (expected " + std::to_string(n) + ")");
                    res.file.frame_rows[n + xcount++] = std::move(sp);
                }
            }
            if (!frame_closed) return fail("truncated frame block (missing 'endframe')");
            if (zcount != n) return fail("frame block has " + std::to_string(zcount) +
                                         " Z rows, expected " + std::to_string(n));
            if (xcount != n) return fail("frame block has " + std::to_string(xcount) +
                                         " X rows, expected " + std::to_string(n));
        }

        // free line: "free <a0> <a1> ... <a_{r-1}>"
        if (!rd.next(&t) || t[0] != "free")
            return fail("expected 'free' line after 'endframe' in v3 format");
        if ((int)t.size() != r + 1)
            return fail("'free' line must have exactly r=" + std::to_string(r) + " indices");
        res.file.free_gens.resize(r);
        for (int d = 0; d < r; ++d) {
            if (!parse_int(t[d + 1], &res.file.free_gens[d]) ||
                res.file.free_gens[d] < 0 || res.file.free_gens[d] >= n)
                return fail("free index '" + t[d + 1] + "' out of range [0," + std::to_string(n) + ")");
        }
        // eps line: "eps <bit0> <bit1> ... <bit_{n-1}>"
        if (!rd.next(&t) || t[0] != "eps")
            return fail("expected 'eps' line after 'free' in v3 format");
        if ((int)t.size() != n + 1)
            return fail("'eps' line must have exactly n=" + std::to_string(n) + " bits");
        res.file.eps.resize(n);
        for (int q = 0; q < n; ++q) {
            if (t[q + 1] == "0") res.file.eps[q] = 0;
            else if (t[q + 1] == "1") res.file.eps[q] = 1;
            else return fail("eps bit must be 0 or 1, got '" + t[q + 1] + "'");
        }
        // branch lines: "branch <re> <im> <sigma_hex>" (one per branch, no endbranch)
        while (rd.next(&t)) {
            if (t[0] != "branch")
                return fail("expected 'branch <re> <im> <sigma_hex>' or end of file, got '" + t[0] + "'");
            if ((int)res.file.branches.size() >= res.file.chi)
                return fail("more branches than chi=" + std::to_string(res.file.chi));
            if (t.size() != 4)
                return fail("expected 'branch <re> <im> <sigma_hex>' (three values)");
            double re, im;
            if (!parse_double(t[1], &re) || !parse_double(t[2], &im))
                return fail("non-numeric branch coefficient");
            if (!std::isfinite(re) || !std::isfinite(im))
                return fail("non-finite branch coefficient (nan/inf)");
            if (re == 0.0 && im == 0.0) return fail("zero branch coefficient (degenerate branch)");
            // parse sigma_hex
            char* hex_end = nullptr;
            errno = 0;
            unsigned long long mask_ull = std::strtoull(t[3].c_str(), &hex_end, 16);
            if (hex_end == t[3].c_str() || *hex_end != '\0' || errno == ERANGE)
                return fail("invalid sigma_hex '" + t[3] + "'");
            RefFile::Branch br;
            br.coeff = cd(re, im);
            br.sigma_mask = (uint64_t)mask_ull;
            res.file.branches.push_back(std::move(br));
        }
        if ((int)res.file.branches.size() != res.file.chi)
            return fail("found " + std::to_string(res.file.branches.size()) +
                        " branch(es), header says chi=" + std::to_string(res.file.chi));
    } else {
        // ── v2 body: branch blocks with per-branch prep + endbranch ─────────────────────
        while (rd.next(&t)) {
            if (t[0] != "branch")
                return fail("expected 'branch <re> <im>' or end of file, got '" + t[0] + "'");
            if ((int)res.file.branches.size() >= res.file.chi)
                return fail("more branches than chi=" + std::to_string(res.file.chi));
            if (t.size() != 3) return fail("expected 'branch <re> <im>' (two numbers)");
            double re, im;
            if (!parse_double(t[1], &re) || !parse_double(t[2], &im))
                return fail("non-numeric branch coefficient");
            if (!std::isfinite(re) || !std::isfinite(im))
                return fail("non-finite branch coefficient (nan/inf)");
            if (re == 0.0 && im == 0.0) return fail("zero branch coefficient (degenerate branch)");
            RefFile::Branch br;
            br.coeff = cd(re, im);
            bool closed = false;
            while (rd.next(&t)) {
                if (t[0] == "endbranch") {
                    if (t.size() != 1) return fail("'endbranch' takes no arguments");
                    closed = true;
                    break;
                }
                RefFile::PrepGate g{0, -1, -1};
                std::string e = parse_gate(t, &g);
                if (!e.empty()) return fail(e);
                br.prep.push_back(g);
            }
            if (!closed) return fail("truncated branch (missing 'endbranch')");
            res.file.branches.push_back(std::move(br));
        }
        if ((int)res.file.branches.size() != res.file.chi)
            return fail("found " + std::to_string(res.file.branches.size()) +
                        " branch(es), header says chi=" + std::to_string(res.file.chi));
    }
    res.ok = true;
    return res;
}

RefParseResult parse_ref_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        RefParseResult res;
        res.error = "cannot open " + path;
        return res;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    RefParseResult res = parse_ref_text(ss.str());
    if (!res.ok) res.error = path + ": " + res.error;
    return res;
}

// ── loader ────────────────────────────────────────────────────────────────────────────

LoadedRef ref_to_state(const RefFile& rf) {
    LoadedRef out;
    if (rf.n <= 0 || rf.chi <= 0 || (int)rf.branches.size() != rf.chi) {
        out.error = "malformed RefFile (n/chi/branch-count inconsistent)";
        return out;
    }
    // Sanity cap BEFORE any state allocation: an int-range but absurd n would OOM below.
    if (rf.n > 1'000'000) {
        out.error = "implausible qubit count n=" + std::to_string(rf.n) + " (cap 1000000)";
        return out;
    }

    if (rf.version == 4) {
        // v4 carries NO anchor block, so there is no affine (CSS) reconstruction. Fail loud
        // rather than fall through to the v2 replay leg (which would silently collapse the
        // empty-prep branches into a wrong chi=1 state).
        out.error = "ref-format 4 is framed-only (anchor block retired); "
                    "load via framed_from_ref / framed_from_ref_validated";
        return out;
    }

    if (rf.version == 3) {
        // ── v3 direct loader: reconstruct CanonicalStabSum without from_rays ───────────────
        // Validate v3-specific fields.
        if (rf.r < 0 || (int)rf.free_gens.size() != rf.r ||
            (int)rf.eps.size() != rf.n ||
            (int)rf.frame_rows.size() != 2 * rf.n) {
            out.error = "malformed v3 RefFile (r/free_gens/eps/frame_rows inconsistent)";
            return out;
        }
        // norm check
        double norm2 = 0.0;
        for (const auto& br : rf.branches) norm2 += std::norm(br.coeff);
        if (!std::isfinite(norm2) || std::abs(norm2 - 1.0) > 1e-9) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "coefficient norm^2 = %.17g != 1 (not a state)", norm2);
            out.error = buf;
            return out;
        }

        // Step 1: replay anchor_prep → anchor AffineState
        AffineState anchor_state(rf.n);
        for (const auto& g : rf.anchor_prep) apply_prep_gate(anchor_state, g);

        // Step 2: assemble CanonicalStabSum, setting U directly from stored frame rows
        // (NO stabilizer_generators, NO from_generators, NO from_rays)
        CanonicalStabSum S(rf.n);
        const int n = rf.n;
        const int W = (n + 63) / 64;

        // Reconstruct Pauli rows from stored StoredPauli
        auto load_pauli = [&](const RefFile::StoredPauli& sp) {
            Pauli P(n);
            P.phase = sp.phase;
            // Copy x/z words (StoredPauli words may be shorter if trailing zeros omitted)
            for (int w = 0; w < W && w < (int)sp.x.size(); ++w) P.x[w] = sp.x[w];
            for (int w = 0; w < W && w < (int)sp.z.size(); ++w) P.z[w] = sp.z[w];
            return P;
        };
        for (int a = 0; a < n; ++a) S.U.Zrow[a] = load_pauli(rf.frame_rows[a]);
        for (int a = 0; a < n; ++a) S.U.Xrow[a] = load_pauli(rf.frame_rows[n + a]);
        S.U.invalidate_dual();  // forward rows set directly; inverse tableau not stored

        // Step 3: set eps from stored bits
        S.eps.assign(rf.eps.begin(), rf.eps.end());

        // Step 4: set anchor
        S.anchor = std::make_unique<AffineState>(std::move(anchor_state));

        // Step 5: set free
        S.free = rf.free_gens;

        // Step 6: set branches (sigma from bitmask, coeff direct)
        const int r = rf.r;
        S.branches.resize(rf.chi);
        for (int i = 0; i < rf.chi; ++i) {
            S.branches[i].sigma.resize(r);
            for (int d = 0; d < r; ++d)
                S.branches[i].sigma[d] = (uint8_t)((rf.branches[i].sigma_mask >> d) & 1);
            S.branches[i].c = rf.branches[i].coeff;
        }

        if (!S.verify_invariants()) {
            out.error = "v3 loaded state violates CanonicalStabSum invariants "
                        "(duplicate/parallel branch rays, or sigma/eps mismatch?)";
            return out;
        }
        if (S.chi() != rf.chi) {
            out.error = "v3 loaded chi " + std::to_string(S.chi()) +
                        " != header chi " + std::to_string(rf.chi);
            return out;
        }
        out.state = std::move(S);
        out.ok = true;
        return out;
    }

    // ── v2 loader (unchanged): replay per-branch prep, assemble via from_rays ─────────────
    // Build anchor base state (empty in v2-compat mode; behavior-preserving with empty anchor_prep).
    AffineState anchor_state(rf.n);
    for (const auto& g : rf.anchor_prep) apply_prep_gate(anchor_state, g);
    std::vector<std::unique_ptr<AffineState>> rays;
    std::vector<cd> coeffs;
    double norm2 = 0.0;
    for (const auto& br : rf.branches) {
        auto st = std::make_unique<AffineState>(anchor_state);   // clone the base
        for (const auto& g : br.prep) apply_prep_gate(*st, g);
        rays.push_back(std::move(st));
        coeffs.push_back(br.coeff);
        norm2 += std::norm(br.coeff);
    }
    // Branch rays of a valid sum are orthonormal, so the norm is sum |c|^2 exactly.
    // !(isfinite) guards NaN: std::abs(NaN - 1.0) > 1e-9 is FALSE and would slip through.
    if (!std::isfinite(norm2) || std::abs(norm2 - 1.0) > 1e-9) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "coefficient norm^2 = %.17g != 1 (not a state)", norm2);
        out.error = buf;
        return out;
    }
    out.state = CanonicalStabSum::from_rays(rf.n, std::move(rays), std::move(coeffs));
    if (!out.state.verify_invariants()) {
        out.error = "loaded state violates CanonicalStabSum invariants "
                    "(duplicate/parallel branch rays?)";
        return out;
    }
    if (out.state.chi() != rf.chi) {
        out.error = "loaded chi " + std::to_string(out.state.chi()) +
                    " != header chi " + std::to_string(rf.chi) +
                    " (parallel branch rays collapsed?)";
        return out;
    }
    out.ok = true;
    return out;
}

// ── lean bare-state sourcing (byte-identical to FramedSuperposition::from_css(ref_to_state(rf).state)) ───────
FramedSuperposition framed_from_ref(const RefFile& rf) {
    // v2 file: the v3/v4 frame fields are empty (frame_rows/free_gens, r==0), so there
    // is nothing to copy directly — defer to the full v2 loader + the existing conversion point.
    if (rf.version < 3) {
        return FramedSuperposition::from_css(ref_to_state(rf).state);
    }

    // v3/v4 (identical frame/free/eps/branch encodings; v4 simply has no anchor block, which
    // this loader never consulted anyway): mirror ref_to_state's v3 direct loader (the
    // load_pauli lambda + Steps 2–6 in `ref_to_state`) EXACTLY, but into a FramedSuperposition
    // — the affine anchor (Steps 1/4) is skipped (the lean rep ignores it).
    const int n = rf.n;
    const int W = (n + 63) / 64;
    FramedSuperposition L(n);

    // Step 2: frame rows -> U.Zrow[0..n-1] / U.Xrow[n..2n-1] (StoredPauli words may be shorter when
    // trailing zeros are omitted; copy only what's present, leave the rest zero — identical to the
    // loader's load_pauli lambda).
    auto load_pauli = [&](const RefFile::StoredPauli& sp) {
        Pauli P(n);
        P.phase = sp.phase;
        for (int w = 0; w < W && w < (int)sp.x.size(); ++w) P.x[w] = sp.x[w];
        for (int w = 0; w < W && w < (int)sp.z.size(); ++w) P.z[w] = sp.z[w];
        return P;
    };
    for (int a = 0; a < n; ++a) L.U.Zrow[a] = load_pauli(rf.frame_rows[a]);
    for (int a = 0; a < n; ++a) L.U.Xrow[a] = load_pauli(rf.frame_rows[n + a]);
    L.U.invalidate_dual();   // forward rows set directly; inverse tableau not stored

    // Step 3: eps from stored bits
    L.eps.assign(rf.eps.begin(), rf.eps.end());

    // Step 5: free
    L.free = rf.free_gens;

    // Step 6: amplitude entries (sigma from bitmask over rf.r bits, coeff direct)
    const int r = rf.r;
    auto& eb = L.entries();
    eb.resize(rf.chi);
    for (int i = 0; i < rf.chi; ++i) {
        eb[i].first.resize(r);
        for (int d = 0; d < r; ++d)
            eb[i].first[d] = (uint8_t)((rf.branches[i].sigma_mask >> d) & 1);
        eb[i].second = rf.branches[i].coeff;
    }
    L.sync_alpha_k();
    return L;
}

// ── validated framed load: the shared consumer entry point ─────────────────────────────
LoadedFramedRef framed_from_ref_validated(const RefFile& rf) {
    LoadedFramedRef out;
    if (rf.version >= 3) {
        // Structural gates (the parser enforces these for files; re-checked here so hand-built
        // RefFiles fail loud too) + the sanity cap + the unit-norm gate the CSS loader carried.
        if (rf.n <= 0 || rf.chi <= 0 || (int)rf.branches.size() != rf.chi) {
            out.error = "malformed RefFile (n/chi/branch-count inconsistent)";
            return out;
        }
        if (rf.n > 1'000'000) {
            out.error = "implausible qubit count n=" + std::to_string(rf.n) + " (cap 1000000)";
            return out;
        }
        if (rf.r < 0 || (int)rf.free_gens.size() != rf.r ||
            (int)rf.eps.size() != rf.n ||
            (int)rf.frame_rows.size() != 2 * rf.n) {
            out.error = "malformed v3/v4 RefFile (r/free_gens/eps/frame_rows inconsistent)";
            return out;
        }
        double norm2 = 0.0;
        for (const auto& br : rf.branches) norm2 += std::norm(br.coeff);
        if (!std::isfinite(norm2) || std::abs(norm2 - 1.0) > 1e-9) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "coefficient norm^2 = %.17g != 1 (not a state)", norm2);
            out.error = buf;
            return out;
        }
        out.state = framed_from_ref(rf);
        out.ok = true;
        return out;
    }
    // v2 legacy: the fully-validating CSS loader (invariants + norm) + the conversion point.
    LoadedRef lr = ref_to_state(rf);
    if (!lr.ok) {
        out.error = lr.error;
        return out;
    }
    out.state = FramedSuperposition::from_css(lr.state);
    out.ok = true;
    return out;
}

LoadedRef load_ref_state(const std::string& path) {
    RefParseResult pr = parse_ref_file(path);
    if (!pr.ok) {
        LoadedRef out;
        out.error = pr.error;
        return out;
    }
    LoadedRef out = ref_to_state(pr.file);
    if (!out.ok) out.error = path + ": " + out.error;
    return out;
}

// The exact sum-vs-sum overlap linear algebra (dense_core, ray_support_fingerprint,
// exact_sum_overlap and variants) lives in ref_overlap.cpp — it is overlap math unrelated to the
// .ref file format. Declarations stay in ref_io.hpp so callers are unaffected.

}  // namespace qeccore
