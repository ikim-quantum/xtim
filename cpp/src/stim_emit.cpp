// Phase 2 of the strict-Stim-superset parser: the Emitter. Turns the phase-1 line tree
// (built in stim_parse.cpp) into a flat, unrolled, desugared ParsedStim. Split out of
// stim_parse.cpp as a pure move (byte-identical) along the phase-1/phase-2 seam; the shared
// line-tree types + caps + the run_emitter entry point live in stim_emit_internal.hpp.
#include "stim_emit_internal.hpp"
#include <cstddef>
#include "qeccore/clifford_frames.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace qeccore {
namespace stim_detail {

namespace {

// A single base-gate step of a parse-time desugaring. `t0`/`t1` index INTO the broadcast group
// (0-based; t1 = -1 for a 1-qubit gate). e.g. SWAP a b desugars to CX(0,1) CX(1,0) CX(0,1).
// All `kind`s are over the existing engine gate set {X,Y,Z,S,SDG,H,CX,CZ} — NO new GateKind.
struct DesugarStep { GateKind kind; int t0; int t1; };
// A desugaring: the broadcast group ARITY plus the word. All 2q desugars validate distinct
// qubits per group like CX/CZ (inherited in emit_word).
struct Desugar { int arity; std::vector<DesugarStep> word; };

// Desugar table for every Stim unitary 1q/2q Clifford gate NOT natively handled above. Each word
// was derived (BFS over {X,Y,Z,S,SDG,H,CX,CZ} against Stim's own tableau) to equal Stim's gate UP
// TO GLOBAL PHASE (global phase is unobservable in sampling) and is asserted equal both densely
// (up-to-phase unitary) and by record-distribution vs real Stim in the tests. Step convention:
// {GateKind, t0, t1}, t0/t1 index the broadcast group; t1=-1 for 1q steps. For a 2q step,
// (t0,t1)=(0,1) is "control=first target, target=second"; (1,0) is the reverse (CX b a).
// Names listed are Stim canonical names + their aliases (ZCY, SWAPCZ) pointing at the same word.
const Desugar* lookup_desugar(const std::string& nm) {
    using G = GateKind;
    auto Sa = [](GateKind k){ return DesugarStep{k, 0, -1}; };          // 1q step on group[0] (a)
    auto Sb = [](GateKind k){ return DesugarStep{k, 1, -1}; };          // 1q step on group[1] (b)
    auto Cab = [](GateKind k){ return DesugarStep{k, 0, 1}; };          // 2q step a->b
    auto Cba = [](GateKind k){ return DesugarStep{k, 1, 0}; };          // 2q step b->a
    static const std::map<std::string, Desugar> table = {
        // ---- 1-qubit (the 24-element Clifford group entries we don't natively name) ----
        {"SQRT_X",     {1, {Sa(G::H), Sa(G::S),   Sa(G::H)}}},          // √X  = H S H
        {"SQRT_X_DAG", {1, {Sa(G::H), Sa(G::SDG), Sa(G::H)}}},          // √X† = H S† H
        {"SQRT_Y",     {1, {Sa(G::H), Sa(G::X)}}},                      // √Y  = H X
        {"SQRT_Y_DAG", {1, {Sa(G::H), Sa(G::Z)}}},                      // √Y† = H Z
        {"H_XY",       {1, {Sa(G::S), Sa(G::Y)}}},                      // X<->Y, Z->-Z
        {"H_YZ",       {1, {Sa(G::SDG), Sa(G::H), Sa(G::S)}}},          // Y<->Z, X->-X
        {"H_NXY",      {1, {Sa(G::S), Sa(G::X)}}},
        {"H_NXZ",      {1, {Sa(G::H), Sa(G::Y)}}},
        {"H_NYZ",      {1, {Sa(G::S), Sa(G::H), Sa(G::SDG)}}},
        {"C_XYZ",      {1, {Sa(G::SDG), Sa(G::H)}}},                    // X->Y->Z->X
        {"C_ZYX",      {1, {Sa(G::H), Sa(G::S)}}},                      // X->Z->Y->X
        {"C_NXYZ",     {1, {Sa(G::S), Sa(G::H), Sa(G::Y)}}},
        {"C_XNYZ",     {1, {Sa(G::S), Sa(G::H)}}},
        {"C_XYNZ",     {1, {Sa(G::S), Sa(G::H), Sa(G::Z)}}},
        {"C_NZYX",     {1, {Sa(G::H), Sa(G::S), Sa(G::X)}}},
        {"C_ZNYX",     {1, {Sa(G::H), Sa(G::SDG)}}},
        {"C_ZYNX",     {1, {Sa(G::H), Sa(G::S), Sa(G::Y)}}},
        // ---- 2-qubit (words derived against Stim's TABLEAU; Cab=CX(first->second),
        //      Cba=CX(second->first); step (0,1)=control first target second) ----
        {"SWAP",       {2, {Cab(G::CX), Cba(G::CX), Cab(G::CX)}}},
        {"ISWAP",      {2, {Cab(G::CX), Sb(G::S),   Cba(G::CX), Cab(G::CX)}}},
        {"ISWAP_DAG",  {2, {Cab(G::CX), Sb(G::SDG), Cba(G::CX), Cab(G::CX)}}},
        {"CXSWAP",     {2, {Cba(G::CX), Cab(G::CX)}}},
        {"SWAPCX",     {2, {Cab(G::CX), Cba(G::CX)}}},
        {"CZSWAP",     {2, {Sa(G::H), Cab(G::CX), Cba(G::CX), Sb(G::H)}}},  // alias SWAPCZ
        {"CY",         {2, {Sa(G::S),  DesugarStep{G::CZ,0,1}, Cab(G::CX)}}},   // alias ZCY
        {"XCZ",        {2, {Cba(G::CX)}}},                              // X-control on 2nd qubit
        {"YCZ",        {2, {Sb(G::S),  DesugarStep{G::CZ,0,1}, Cba(G::CX)}}},
        {"XCX",        {2, {Sa(G::H),  Cab(G::CX), Sa(G::H)}}},
        {"XCY",        {2, {Sa(G::H), Sa(G::S), DesugarStep{G::CZ,0,1}, Cab(G::CX), Sa(G::H)}}},
        {"YCX",        {2, {Sb(G::H), Sb(G::S), DesugarStep{G::CZ,0,1}, Cba(G::CX), Sb(G::H)}}},
        {"YCY",        {2, {Sa(G::SDG), Sb(G::SDG), Sa(G::H), Cab(G::CX), Sa(G::H), Sa(G::S), Sb(G::S)}}},
        {"SQRT_ZZ",     {2, {Sa(G::S),   Sb(G::S),   DesugarStep{G::CZ,0,1}}}},
        {"SQRT_ZZ_DAG", {2, {Sa(G::SDG), Sb(G::SDG), DesugarStep{G::CZ,0,1}}}},
        {"SQRT_XX",     {2, {Sa(G::SDG), Cab(G::CX), Sa(G::H), Sa(G::SDG), Cab(G::CX)}}},
        {"SQRT_XX_DAG", {2, {Sa(G::S),   Cab(G::CX), Sa(G::H), Sa(G::S),   Cab(G::CX)}}},
        {"SQRT_YY",     {2, {Sa(G::S), Cba(G::CX), Sb(G::H), Sa(G::Z), Cba(G::CX), Sa(G::S)}}},
        {"SQRT_YY_DAG", {2, {Sa(G::S), Sb(G::Z), Cba(G::CX), Sb(G::H), Cba(G::CX), Sa(G::SDG)}}},
        {"II",          {2, {}}},                                       // 2q identity: no-op word
    };
    static const std::map<std::string, std::string> alias = {
        {"ZCY", "CY"}, {"SWAPCZ", "CZSWAP"},
    };
    auto ai = alias.find(nm);
    const std::string& key = ai == alias.end() ? nm : ai->second;
    auto it = table.find(key);
    return it == table.end() ? nullptr : &it->second;
}

// ---------- phase 2: the emitter ----------
struct Emitter {
    ParsedStim& out;
    int meas_count = 0;
    // Task 6: IF guard stack accumulated from outermost to innermost IF.
    // Each entry is pushed when we enter an IF body and popped when we exit.
    // Copied into out.detector_if_guards at each emit_detector call.
    std::vector<IfGuard> current_if_guards_;
    int max_qubit = -1;
    int anc_next = 0;               // next fresh MPP/SPP ancilla wire (= 1 + max user qubit, set
                                    // from scan_max_qubit before walk(); never collides with a
                                    // user qubit). Each MPP product consumes one ancilla.
    long long emitted = 0;          // C2: instructions emitted so far (REPEAT-unrolled)
    long long processed = 0;        // C2b: lines DISPATCHED (annotations/errors too) — an
                                    // annotation-only or error-only REPEAT body emits nothing,
                                    // so capping `emitted` alone lets the unroll loop spin
                                    // (found by the parser fuzzer: REPEAT 1e12 { TICK } hung).
    bool overflowed = false;        // C2: set once; short-circuits walk/dispatch
    bool errors_capped = false;     // bound error-collection memory under huge unrolls

    // M-B: cumulative SHIFT_COORDS offset (component-wise; grows as larger shifts arrive). Applied
    // to subsequent QUBIT_COORDS/DETECTOR coordinate args before they are stored, replicating
    // Stim's semantics. Because walk() processes the UNROLLED tree, a SHIFT_COORDS inside a REPEAT
    // accumulates once per iteration exactly as Stim does.
    std::vector<double> coord_shift{};
    std::vector<double> shifted(const std::vector<double>& raw) const {
        std::vector<double> r = raw;
        for (size_t i = 0; i < r.size() && i < coord_shift.size(); ++i) r[i] += coord_shift[i];
        return r;
    }

    static constexpr size_t kMaxErrors = 1000;
    void err(int line, const std::string& m) {
        if (errors_capped) return;
        if (out.errors.size() >= kMaxErrors) {
            errors_capped = true;
            out.errors.push_back({line, "too many errors; further diagnostics suppressed"});
            return;
        }
        out.errors.push_back({line, m});
    }
    void touch(int q) { if (q > max_qubit) max_qubit = q; }

    bool qubit_targets(const RawLine& rl, std::vector<int>& qs) {     // all-plain-qubit lines
        for (const std::string& t : rl.targets) {
            if (!t.empty() && t[0] == '!') {
                err(rl.lineno, "inverted targets (!q) are not supported in v1"); return false; }
            if (t.rfind("sweep", 0) == 0) {
                err(rl.lineno, "sweep bits are not supported in v1"); return false; }
            if (t.rfind("rec", 0) == 0) {
                err(rl.lineno, "rec[-k] as a gate target: only CX/CY/CZ accept a "
                               "measurement-record control (feedback); this gate does not");
                return false; }
            try {
                size_t pos = 0;
                int v = std::stoi(t, &pos);
                if (pos != t.size() || v < 0) { err(rl.lineno, "bad qubit target '" + t + "'"); return false; }
                if (v > kMaxQubitIndex) { err(rl.lineno, "qubit index too large (cap 1000000)"); return false; }
                qs.push_back(v); touch(v);
            } catch (...) { err(rl.lineno, "bad qubit target '" + t + "'"); return false; }
        }
        return true;
    }

    // Emit a broadcast gate. `g2`, when not -1, DESUGARS each instance into the pair (g, g2)
    // applied to the same group — used for T_DAG -> T;S_DAG and CS_DAG -> CS;CZ. The desugared
    // gates are exact diagonal-phase inverses (T·S_DAG = diag(1,ζ₈⁷) = T†; CS·CZ = diag(1,1,1,ζ₈⁶)
    // = CS†), so this is a pure parse-time convenience: no new GateKind, no engine change. The
    // validation (arity, distinct-qubits-per-group, canonical sort) is the analogue's, unchanged.
    void emit_gate(const RawLine& rl, GateKind g, int arity, bool sort_group, int g2 = -1) {
        if (!rl.args.empty()) { err(rl.lineno, rl.name + " takes no () arguments"); return; }
        std::vector<int> qs;
        if (!qubit_targets(rl, qs)) return;
        if (qs.empty() || (int)qs.size() % arity != 0) {
            err(rl.lineno, rl.name + ": target count must be a positive multiple of " + std::to_string(arity));
            return;
        }
        for (size_t g0 = 0; g0 < qs.size(); g0 += arity) {
            std::vector<int> grp(qs.begin() + g0, qs.begin() + g0 + arity);
            for (int a = 0; a < arity; ++a)
                for (int b = a + 1; b < arity; ++b)
                    if (grp[a] == grp[b]) { err(rl.lineno, rl.name + ": duplicate qubit in a target group"); return; }
            if (sort_group) std::sort(grp.begin(), grp.end());
            Instr ins; ins.kind = Instr::Kind::Gate; ins.gate = g; ins.targets = grp;
            out.circuit.stream.push_back(std::move(ins));
            if (g2 >= 0) {
                Instr ins2; ins2.kind = Instr::Kind::Gate; ins2.gate = (GateKind)g2; ins2.targets = std::move(grp);
                out.circuit.stream.push_back(std::move(ins2));
            }
        }
    }

    // Identity gate `I q...`: a no-op. We emit NOTHING (the IR keys on measurement/noise order,
    // never on gate stream indices, so a dropped identity is semantically exact), but we DO touch
    // each target so `circuit.n` counts a qubit Stim would (Stim: `I 5` => num_qubits 6).
    void emit_identity(const RawLine& rl) {
        if (!rl.args.empty()) { err(rl.lineno, rl.name + " takes no () arguments"); return; }
        std::vector<int> qs;
        if (!qubit_targets(rl, qs)) return;   // touches each target
        if (qs.empty()) { err(rl.lineno, rl.name + ": needs at least one target"); return; }
    }

    // Emit a broadcast gate that DESUGARS each instance into a fixed WORD over base gates.
    // The broadcast/arity/distinct-qubit-per-group validation is identical to emit_gate's; the
    // only difference is each instance pushes the word's steps (with targets remapped through the
    // group) instead of a single gate. Used for every Stim Clifford gate not natively in our table
    // (SWAP, ISWAP, SQRT_X, H_XY, CY, ... — see the dispatch table). NO new GateKind, no engine
    // change: a pure parse-time rewrite to {X,Y,Z,S,SDG,H,CX,CZ}.
    void emit_word(const RawLine& rl, const Desugar& d) {
        if (!rl.args.empty()) { err(rl.lineno, rl.name + " takes no () arguments"); return; }
        std::vector<int> qs;
        if (!qubit_targets(rl, qs)) return;
        const int arity = d.arity;
        if (qs.empty() || (int)qs.size() % arity != 0) {
            err(rl.lineno, rl.name + ": target count must be a positive multiple of " + std::to_string(arity));
            return;
        }
        for (size_t g0 = 0; g0 < qs.size(); g0 += arity) {
            std::vector<int> grp(qs.begin() + g0, qs.begin() + g0 + arity);
            for (int a = 0; a < arity; ++a)
                for (int b = a + 1; b < arity; ++b)
                    if (grp[a] == grp[b]) { err(rl.lineno, rl.name + ": duplicate qubit in a target group"); return; }
            for (const DesugarStep& st : d.word) {
                Instr ins; ins.kind = Instr::Kind::Gate; ins.gate = st.kind;
                if (st.t1 < 0) ins.targets = {grp[st.t0]};
                else           ins.targets = {grp[st.t0], grp[st.t1]};
                out.circuit.stream.push_back(std::move(ins));
            }
        }
    }

    // Parse a `rec[-k]` token into k (>=1, in range). Returns true on success, else emits an
    // error and returns false. Shares the exact shape rules used by DETECTOR.
    bool parse_rec_offset(int line, const std::string& nm, const std::string& t, int& k) {
        if (t.size() < 7 || t.rfind("rec[-", 0) != 0 || t.back() != ']') {
            err(line, nm + ": expected rec[-k] control, got '" + t + "'"); return false; }
        const std::string digits = t.substr(5, t.size() - 6);
        try {                                            // pure digits only, fully consumed
            size_t pos = 0;
            k = std::stoi(digits, &pos);
            if (pos != digits.size() || digits.empty() ||
                !std::isdigit((unsigned char)digits[0])) throw 0;
        }
        catch (...) { err(line, nm + ": bad rec[] index '" + t + "'"); return false; }
        if (k < 1 || k > meas_count) {
            err(line, nm + ": rec[-" + std::to_string(k) + "] out of range ("
                  + std::to_string(meas_count) + " measurements so far)"); return false; }
        return true;
    }
    // Parse a plain non-negative qubit token. Returns true on success (and touches it), else errs.
    bool parse_qubit_tok(int line, const std::string& nm, const std::string& t, int& q) {
        if (!t.empty() && t[0] == '!') {
            err(line, nm + ": inverted targets (!q) are not supported in v1"); return false; }
        if (t.rfind("rec", 0) == 0 || t.rfind("sweep", 0) == 0) {
            err(line, nm + ": expected a qubit, got '" + t + "'"); return false; }
        try {
            size_t pos = 0; q = std::stoi(t, &pos);
            if (pos != t.size() || q < 0) { err(line, nm + ": bad qubit target '" + t + "'"); return false; }
        } catch (...) { err(line, nm + ": bad qubit target '" + t + "'"); return false; }
        if (q > kMaxQubitIndex) { err(line, nm + ": qubit index too large (cap 1000000)"); return false; }
        touch(q); return true;
    }

    // Classically-controlled Pauli feedback + normal 2-qubit gates, BROADCAST in PAIRS — Stim's
    // CX/CY/CZ convention. Stim consumes targets as (control, target) PAIRS, each independently:
    //   (rec[-k], qubit)  -> feedback: Pauli on `qubit` gated by record offset k  (Task-1 IR)
    //   (qubit,   qubit)  -> a normal 2-qubit gate on that pair (CX/CZ emit_gate, CY desugar word)
    // A single line may MIX pair kinds (verified against Stim: `CX 0 1 rec[-1] 2` = CX(0,1) + a
    // feedback on q2; multi-feedback `CX rec[-3] 2 rec[-1] 2` = two feedbacks on q2). Pairs are
    // emitted in source order, so N feedbacks become N ControlledPauli IR instrs (the relabel
    // composes them) and normal pairs stay byte-identical to the old emit_gate/emit_word path.
    //   control == sweep[k]   -> rejected (out of scope).
    //   target  == rec/sweep  -> rejected (rec can't be a runnable gate target; Stim rejects the
    //                            samplable cases — `CX q rec` is "record editing").
    //   odd target count      -> rejected (Stim: "requires an even number of targets").
    // `pauli` is the feedback basis (CX->X, CY->Y, CZ->Z). `normal_gate`>=0 means CX/CZ-style
    // single-gate emit for (qubit,qubit) pairs; otherwise the (CY) desugar word is used.
    void emit_c2(const RawLine& rl, PauliBasis pauli, int normal_gate) {
        // Fast path: NO rec/sweep anywhere -> delegate to the byte-identical normal-gate path.
        bool any_rec = false;
        for (const std::string& t : rl.targets)
            if (t.rfind("rec", 0) == 0 || t.rfind("sweep", 0) == 0) { any_rec = true; break; }
        if (!any_rec) {
            if (normal_gate >= 0) emit_gate(rl, (GateKind)normal_gate, 2, false);
            else                  emit_word(rl, *lookup_desugar("CY"));
            return;
        }
        if (!rl.args.empty()) { err(rl.lineno, rl.name + " takes no () arguments"); return; }
        if (rl.targets.empty() || rl.targets.size() % 2 != 0) {
            err(rl.lineno, rl.name + ": requires an even number of targets (control/target pairs)");
            return; }
        for (size_t i = 0; i < rl.targets.size(); i += 2) {
            const std::string& tc = rl.targets[i];
            const std::string& tt = rl.targets[i + 1];
            if (tc.rfind("sweep", 0) == 0 || tt.rfind("sweep", 0) == 0) {
                err(rl.lineno, rl.name + ": sweep-bit-controlled gates are not supported"); return; }
            const bool ctrl_is_rec = tc.rfind("rec", 0) == 0;
            if (tt.rfind("rec", 0) == 0) {   // rec as a TARGET is never a runnable gate
                err(rl.lineno, rl.name + ": rec[] cannot be a gate target (got '" + tt + "')"); return; }
            if (ctrl_is_rec) {               // (rec[-k], qubit) -> feedback
                int k = 0, q = 0;
                if (!parse_rec_offset(rl.lineno, rl.name, tc, k)) return;
                if (!parse_qubit_tok(rl.lineno, rl.name, tt, q)) return;
                Instr ins; ins.kind = Instr::Kind::ControlledPauli; ins.basis = pauli;
                ins.qubits = {q}; ins.control_rec_offset = k;
                out.circuit.stream.push_back(std::move(ins));
            } else {                         // (qubit, qubit) -> a normal 2-qubit gate on this pair
                int a = 0, b = 0;
                if (!parse_qubit_tok(rl.lineno, rl.name, tc, a)) return;
                if (!parse_qubit_tok(rl.lineno, rl.name, tt, b)) return;
                if (a == b) { err(rl.lineno, rl.name + ": duplicate qubit in a target group"); return; }
                if (normal_gate >= 0) {
                    Instr ins; ins.kind = Instr::Kind::Gate; ins.gate = (GateKind)normal_gate;
                    ins.targets = {a, b}; out.circuit.stream.push_back(std::move(ins));
                } else {                     // CY desugar word on (a,b)
                    const int grp[2] = {a, b};
                    for (const DesugarStep& st : lookup_desugar("CY")->word) {
                        Instr ins; ins.kind = Instr::Kind::Gate; ins.gate = st.kind;
                        if (st.t1 < 0) ins.targets = {grp[st.t0]};
                        else           ins.targets = {grp[st.t0], grp[st.t1]};
                        out.circuit.stream.push_back(std::move(ins));
                    }
                }
            }
        }
    }

    // ---- extension point: Task 2 adds emit_measure/emit_reset/emit_noise here ----
    // M-family optional readout-flip probability: at most one arg in [0,1].
    bool measure_arg(const RawLine& rl, double& flip_p) {
        flip_p = 0.0;
        if (rl.args.empty()) return true;
        if (rl.args.size() != 1) { err(rl.lineno, rl.name + ": expected at most one probability argument"); return false; }
        double p = rl.args[0];
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) { err(rl.lineno, rl.name + ": probability out of [0,1]"); return false; }
        flip_p = p; return true;
    }
    void emit_measure(const RawLine& rl, PauliBasis basis, bool then_reset, bool reset_extra_h, bool reset_extra_s) {
        double flip_p = 0.0;
        if (!measure_arg(rl, flip_p)) return;          // Task 4 fills flip_p; here it stays 0
        if (rl.targets.empty()) { err(rl.lineno, rl.name + ": needs at least one target"); return; }
        for (const std::string& tok : rl.targets) {
            std::string t = tok; bool invert = false;
            if (!t.empty() && t[0] == '!') { invert = true; t = t.substr(1); }
            int q; if (!parse_qubit_tok(rl.lineno, rl.name, t, q)) return;
            Instr m; m.kind = Instr::Kind::Measure; m.basis = basis; m.qubits = {q};
            m.invert = invert; m.readout_flip_p = flip_p;
            out.circuit.stream.push_back(std::move(m));
            ++meas_count;
            if (then_reset) emit_reset_one(q, reset_extra_h, reset_extra_s);
        }
    }
    void emit_reset_one(int q, bool extra_h, bool extra_s) {
        Instr r; r.kind = Instr::Kind::Reset; r.qubits = {q};
        out.circuit.stream.push_back(std::move(r));
        // Reset emits |0>, then the basis rotation OUT of Z (rotate_from_z, clifford_frames):
        //   RZ (false,false) -> Z -> {} ; RX (true,false) -> X -> {H} ; RY (true,true) -> Y -> {H,S}.
        const PauliBasis b = extra_s ? PauliBasis::Y : (extra_h ? PauliBasis::X : PauliBasis::Z);
        for (Instr& g : rotate_from_z(b, q)) out.circuit.stream.push_back(std::move(g));
    }
    void emit_reset(const RawLine& rl, bool extra_h, bool extra_s) {
        if (!rl.args.empty()) { err(rl.lineno, rl.name + " takes no () arguments"); return; }
        std::vector<int> qs;
        if (!qubit_targets(rl, qs)) return;
        if (qs.empty()) { err(rl.lineno, rl.name + ": needs at least one target"); return; }
        for (int q : qs) emit_reset_one(q, extra_h, extra_s);
    }
    void emit_noise(const RawLine& rl, NoiseChannel ch, size_t nargs, int group) {
        if (rl.args.size() != nargs) {
            err(rl.lineno, rl.name + ": expected " + std::to_string(nargs) + " probability argument(s)");
            return;
        }
        double sum = 0.0;
        for (double a : rl.args) {
            if (!std::isfinite(a) || a < 0.0 || a > 1.0) {   // I1: also rejects NaN/inf
                err(rl.lineno, rl.name + ": probability out of [0,1]"); return; }
            sum += a;
        }
        if (nargs > 1 && sum > 1.0 + 1e-12) { err(rl.lineno, rl.name + ": probabilities sum to > 1"); return; }
        std::vector<int> qs;
        if (!qubit_targets(rl, qs)) return;
        if (qs.empty() || (int)qs.size() % group != 0) {
            err(rl.lineno, rl.name + ": target count must be a positive multiple of " + std::to_string(group));
            return;
        }
        if (group == 2)
            for (size_t i = 0; i + 1 < qs.size(); i += 2)
                if (qs[i] == qs[i + 1]) { err(rl.lineno, rl.name + ": pair qubits must be distinct"); return; }
        Instr nz; nz.kind = Instr::Kind::Noise; nz.channel = ch; nz.qubits = std::move(qs); nz.probs = rl.args;
        out.circuit.stream.push_back(std::move(nz));
    }
    // ---- extension point: Task 4 adds emit_detector/emit_observable/emit_expectation here ----
    bool rec_targets(const RawLine& rl, std::vector<int>& abs) {
        if (rl.targets.empty()) { err(rl.lineno, rl.name + ": needs at least one rec[-k] target"); return false; }
        for (const std::string& t : rl.targets) {
            if (t.size() < 7 || t.rfind("rec[-", 0) != 0 || t.back() != ']') {
                err(rl.lineno, rl.name + ": expected rec[-k], got '" + t + "'"); return false; }
            int k = 0;
            const std::string digits = t.substr(5, t.size() - 6);
            try {                                            // I2: pure digits only, fully consumed
                size_t pos = 0;
                k = std::stoi(digits, &pos);
                if (pos != digits.size() || digits.empty() ||
                    !std::isdigit((unsigned char)digits[0])) throw 0;
            }
            catch (...) { err(rl.lineno, rl.name + ": bad rec[] index '" + t + "'"); return false; }
            if (k < 1 || k > meas_count) {
                err(rl.lineno, rl.name + ": rec[-" + std::to_string(k) + "] out of range ("
                                + std::to_string(meas_count) + " measurements so far)"); return false; }
            abs.push_back(meas_count - k);
        }
        return true;
    }
    void emit_detector(const RawLine& rl) {                 // M-B: () coords STORED (shift-resolved)
        std::vector<int> abs;
        if (rec_targets(rl, abs)) {
            out.detectors.push_back(std::move(abs));
            // Parallel to out.detectors (declaration order): the shift-resolved DETECTOR() args,
            // or an empty vector for a coordinate-free DETECTOR. == Stim's get_detector_coordinates.
            out.detector_coords.push_back(shifted(rl.args));
            // Task 6: record the IF guard stack at the point of this DETECTOR declaration.
            // The partition layer uses these conditions to determine which superset detector
            // columns are active (live_mask=1) vs absent (live_mask=0) for each decision pattern.
            out.detector_if_guards.push_back(current_if_guards_);
        }
    }
    // M-B: QUBIT_COORDS(x,...) q...  -> store the shift-resolved coords for each target qubit
    // (last write wins, like Stim's get_final_qubit_coordinates). Annotation only: no stream
    // instruction, but we DO touch each target so circuit.n matches Stim (QUBIT_COORDS(..) 5
    // makes num_qubits >= 6).
    void emit_qubit_coords(const RawLine& rl) {
        std::vector<int> qs;
        if (!qubit_targets(rl, qs)) return;       // touches each target; rejects bad targets
        std::vector<double> c = shifted(rl.args);
        for (int q : qs) out.qubit_coords[q] = c;
    }
    // M-B: SHIFT_COORDS(dx,...) accumulates a component-wise offset onto all SUBSEQUENT
    // QUBIT_COORDS/DETECTOR coordinates (cumulative; grows the offset vector as needed).
    void emit_shift_coords(const RawLine& rl) {
        if (rl.args.size() > coord_shift.size()) coord_shift.resize(rl.args.size(), 0.0);
        for (size_t i = 0; i < rl.args.size(); ++i) coord_shift[i] += rl.args[i];
    }
    void emit_observable(const RawLine& rl) {
        if (rl.args.size() != 1 || rl.args[0] < 0 || rl.args[0] != (double)(long long)rl.args[0]) {
            err(rl.lineno, "OBSERVABLE_INCLUDE: needs one non-negative integer argument"); return; }
        if (rl.args[0] > (double)kMaxObservableIndex) {      // C1: bounds dense observable_bits
            err(rl.lineno, "OBSERVABLE_INCLUDE: index too large (cap 1000000)"); return; }
        std::vector<int> abs;
        if (rec_targets(rl, abs)) {
            auto& v = out.observables[(int)rl.args[0]];
            v.insert(v.end(), abs.begin(), abs.end());
        }
    }
    // DECISION(k) rec[-j] ... — mirrors OBSERVABLE_INCLUDE exactly: same argument validation,
    // same rec[-k] resolution to absolute measurement indices. One push per DECISION line.
    void emit_decision(const RawLine& rl) {
        if (rl.args.size() != 1 || rl.args[0] < 0 || rl.args[0] != (double)(long long)rl.args[0]) {
            err(rl.lineno, "DECISION: needs one non-negative integer argument"); return; }
        if (rl.args[0] > (double)kMaxObservableIndex) {
            err(rl.lineno, "DECISION: index too large (cap 1000000)"); return; }
        std::vector<int> abs;
        if (rec_targets(rl, abs)) {
            out.decisions.push_back({(int)rl.args[0], std::move(abs)});
        }
    }
    // Post-walk verification: every qubit declared in an OUTPUT_QUBITS port must NOT appear
    // in any Measure instruction in the stream. Emits a parse error for each violation.
    void verify_output_qubits() {
        if (out.circuit.outputs.empty()) return;
        // Collect all qubits that appear in Measure instructions.
        std::set<int> measured;
        for (const Instr& ins : out.circuit.stream)
            if (ins.kind == Instr::Kind::Measure)
                for (int q : ins.qubits) measured.insert(q);
        // For each output port, check all declared qubits.
        for (const OutputPort& p : out.circuit.outputs)
            for (int q : p.qubits)
                if (measured.count(q))
                    out.errors.push_back({0,
                        "OUTPUT_QUBITS '" + p.name + "': qubit " + std::to_string(q) +
                        " is measured in the circuit (output qubits must be unmeasured)"});
    }
    void emit_expectation(const RawLine& rl) {
        if (rl.args.size() != 1 || rl.args[0] != (double)(long long)rl.args[0]) {
            err(rl.lineno, "PAULI_EXPECTATION: needs one integer label argument"); return; }
        int label = (int)rl.args[0];
        for (int l : out.expectation_labels)
            if (l == label) { err(rl.lineno, "PAULI_EXPECTATION: duplicate label " + std::to_string(label)); return; }
        if (rl.targets.empty()) {
            err(rl.lineno, "PAULI_EXPECTATION: missing *-joined Pauli product"); return; }
        Instr ob; ob.kind = Instr::Kind::Observable;
        std::istringstream ps(rl.targets[0]);
        std::string piece;
        while (std::getline(ps, piece, '*')) {
            if (piece.size() < 2) { err(rl.lineno, "PAULI_EXPECTATION: bad term '" + piece + "'"); return; }
            char c = (char)std::toupper((unsigned char)piece[0]);
            PauliBasis b = c == 'X' ? PauliBasis::X : c == 'Y' ? PauliBasis::Y : PauliBasis::Z;
            if (c != 'X' && c != 'Y' && c != 'Z') { err(rl.lineno, "PAULI_EXPECTATION: bad term '" + piece + "'"); return; }
            int q = 0;
            try { size_t pos = 0; q = std::stoi(piece.substr(1), &pos);
                  if (pos != piece.size() - 1 || q < 0) throw 0; }
            catch (...) { err(rl.lineno, "PAULI_EXPECTATION: bad term '" + piece + "'"); return; }
            if (q > kMaxQubitIndex) { err(rl.lineno, "qubit index too large (cap 1000000)"); return; }
            for (const PauliTerm& t : ob.obs)
                if (t.qubit == q) { err(rl.lineno, "PAULI_EXPECTATION: duplicate qubit in product"); return; }
            ob.obs.push_back({q, b});
            touch(q);
        }
        if (ob.obs.empty()) { err(rl.lineno, "PAULI_EXPECTATION: empty product"); return; }
        // rl.targets[0] = the Pauli product (parsed above).
        // rl.targets[1..] = the declared byproduct frame; each must be rec[-k].
        if (rl.targets.size() >= 2) {
            RawLine frame_rl = rl;                 // reuse rec_targets on the tail
            frame_rl.targets.assign(rl.targets.begin() + 1, rl.targets.end());
            std::vector<int> frame_abs;
            if (!rec_targets(frame_rl, frame_abs)) return;   // errors on a non-rec[-k] tail token
            ob.obs_frame = std::move(frame_abs);
        }
        out.expectation_labels.push_back(label);
        out.circuit.stream.push_back(std::move(ob));
    }

    // MPP P1 P2 ... : measure each space-separated Pauli product Pj independently, in order, each
    // producing ONE measurement record. Desugar each product into the standard ancilla gadget
    // (derived + pinned against real Stim, see docs/xtim_dialect.md):
    //   RX a                       fresh |+> ancilla a (= a wire above every user qubit)
    //   C{p_i} a q_i               controlled-Pauli per factor: X->CX, Y->CY, Z->CZ (control=a)
    //   [Z a]                      iff the product carries a leading `!` (inversion): Z flips the
    //                              X-basis readout, recording the COMPLEMENT bit (matches Stim's
    //                              record sign; an X here would NOT — X commutes with the MX basis)
    //   MX a                       terminal X-measure of a = the MPP record (abandoned ancilla,
    //                              handled by the existing terminal-Pauli deferral)
    // The i^{#Y} normalisation of a Y-containing product is absorbed by Stim's own CY definition
    // (no extra correction needed — verified densely + vs Stim). Each product's MX increments
    // meas_count exactly like a single-qubit M, so rec[]/detectors resolve in Stim's record order.
    void emit_mpp(const RawLine& rl) {
        double flip_p = 0.0;
        if (!measure_arg(rl, flip_p)) return;
        if (rl.targets.empty()) { err(rl.lineno, "MPP: needs at least one Pauli product"); return; }
        for (const std::string& prod : rl.targets) {
            // Parse one product P = (p_0 on q_0) * (p_1 on q_1) * ..., with optional `!` inversion.
            std::string body = prod;
            bool inverted = false;
            if (!body.empty() && body[0] == '!') { inverted = true; body = body.substr(1); }
            if (body.empty()) { err(rl.lineno, "MPP: empty Pauli product '" + prod + "'"); return; }
            std::vector<PauliTerm> terms;
            std::istringstream ps(body);
            std::string piece;
            bool bad = false;
            while (std::getline(ps, piece, '*')) {
                // A leading `!` on a factor also toggles product inversion (Stim semantics).
                if (!piece.empty() && piece[0] == '!') { inverted = !inverted; piece = piece.substr(1); }
                if (piece.size() < 2) { err(rl.lineno, "MPP: bad Pauli term '" + piece + "'"); bad = true; break; }
                char c = (char)std::toupper((unsigned char)piece[0]);
                if (c != 'X' && c != 'Y' && c != 'Z') {
                    err(rl.lineno, "MPP: bad Pauli term '" + piece + "' (expected X/Y/Z<qubit>)"); bad = true; break; }
                PauliBasis b = c == 'X' ? PauliBasis::X : c == 'Y' ? PauliBasis::Y : PauliBasis::Z;
                int q = 0;
                try { size_t pos = 0; q = std::stoi(piece.substr(1), &pos);
                      if (pos != piece.size() - 1 || q < 0) throw 0; }
                catch (...) { err(rl.lineno, "MPP: bad Pauli term '" + piece + "'"); bad = true; break; }
                if (q > kMaxQubitIndex) { err(rl.lineno, "qubit index too large (cap 1000000)"); bad = true; break; }
                for (const PauliTerm& t : terms)
                    if (t.qubit == q) { err(rl.lineno, "MPP: duplicate qubit in product '" + prod + "'"); bad = true; break; }
                if (bad) break;
                terms.push_back({q, b});
                touch(q);
            }
            if (bad) return;
            if (terms.empty()) { err(rl.lineno, "MPP: empty Pauli product '" + prod + "'"); return; }
            // Emit the gadget. anc is fresh (above every user qubit, by scan_max_qubit).
            const int a = anc_next++;
            touch(a);
            auto push_gate = [&](GateKind g, std::vector<int> tg) {
                Instr ins; ins.kind = Instr::Kind::Gate; ins.gate = g; ins.targets = std::move(tg);
                out.circuit.stream.push_back(std::move(ins));
            };
            // HADAMARD-MINIMAL desugar (basis-matched ancilla). The fresh ancilla starts |0>.
            // For an ALL-Z product we read it in the Z basis: copy each data qubit's Z-value onto
            // the ancilla (CX data->anc) and Z-measure it — NO Hadamard anywhere, so a magic-crossing
            // feedback controlled on this record coherentizes to a plain CX and the deferral teleport
            // is a plain CX too (the whole control path stays in the Z basis). Records the SAME bit as
            // the X-basis cat-check (0 = even Z-parity, 1 = odd): both project to the same syndrome
            // sector with the same sign convention. For non-all-Z products we keep the X-basis cat-check
            // (RX; controlled-Pauli; MX) — X/Y reads carry intrinsic basis changes, left for eliminate_hadamards.
            bool all_z = true;
            for (const PauliTerm& t : terms) if (t.p != PauliBasis::Z) { all_z = false; break; }
            PauliBasis read_basis;
            if (all_z) {
                read_basis = PauliBasis::Z;                    // fresh |0> ancilla, read in Z (no H)
                for (const PauliTerm& t : terms)
                    push_gate(GateKind::CX, {t.qubit, a});     // data controls -> copy Z-parity onto anc
            } else {
                read_basis = PauliBasis::X;
                push_gate(GateKind::H, {a});                   // RX a = |0> then H -> |+> (the cat-check)
                for (const PauliTerm& t : terms) {
                    if (t.p == PauliBasis::X)      push_gate(GateKind::CX, {a, t.qubit});
                    else if (t.p == PauliBasis::Z) push_gate(GateKind::CZ, {a, t.qubit});
                    else {                                     // Y -> CY, desugared via the table word
                        const Desugar* d = lookup_desugar("CY");
                        std::vector<int> grp{a, t.qubit};
                        for (const DesugarStep& st : d->word) {
                            if (st.t1 < 0) push_gate(st.kind, {grp[st.t0]});
                            else           push_gate(st.kind, {grp[st.t0], grp[st.t1]});
                        }
                    }
                }
            }
            Instr m; m.kind = Instr::Kind::Measure; m.basis = read_basis; m.qubits = {a};
            m.invert = inverted;                                // record-only flip (same convention both ways)
            m.readout_flip_p = flip_p;
            out.circuit.stream.push_back(std::move(m));
            ++meas_count;
        }
    }

    // MPAD v0 v1 ... : each target is a PADDING VALUE (0 or 1, NOT a qubit). Appends ONE
    // measurement record per target carrying that fixed bit, in order. Desugar each value v to a
    // fresh |0> ancilla (above every user qubit, like MPP) measured in Z (records 0) with the
    // record-only `invert` flag set iff v==1 (fresh|0> -> Z-measures 0 -> invert => record = v).
    // Reuses the existing fresh-ancilla + invert mechanism; no new IR. Each record increments
    // meas_count so rec[]/detectors resolve in Stim's order. Stim accepts an empty MPAD (0 records,
    // 0 qubits) — we match the oracle (no-op), so the empty case simply appends nothing.
    void emit_mpad(const RawLine& rl) {
        if (!rl.args.empty()) { err(rl.lineno, "MPAD takes no () arguments"); return; }
        for (const std::string& tok : rl.targets) {
            int v = -1;
            if (tok == "0") v = 0;
            else if (tok == "1") v = 1;
            else { err(rl.lineno, "MPAD: target must be 0 or 1, got '" + tok + "'"); return; }
            const int a = anc_next++;
            touch(a);
            Instr m; m.kind = Instr::Kind::Measure; m.basis = PauliBasis::Z; m.qubits = {a};
            m.invert = (v == 1); m.readout_flip_p = 0.0;
            out.circuit.stream.push_back(std::move(m));
            ++meas_count;
        }
    }

    // I_ERROR(p...) q... / II_ERROR(p...) q... : identity-ERROR placeholders (noise class). Applies
    // NO error — a NO-OP for sampling regardless of the probability args (verified vs Stim 1.16.0).
    // We emit NOTHING into the circuit stream (0 measurements, 0 state effect) but still touch each
    // qubit target so circuit.n matches Stim (I_ERROR 5 => num_qubits 6). Any () prob args are
    // validated finite in [0,1] (mirrors emit_noise). `arity` is the required target-count divisor:
    // 1 for I_ERROR, 2 for II_ERROR (II_ERROR rejects an odd count, like Stim). An empty target list
    // is accepted (Stim accepts `I_ERROR` / `II_ERROR` with 0 qubits = pure no-op).
    void emit_identity_error(const RawLine& rl, int arity) {
        for (double a : rl.args)
            if (!std::isfinite(a) || a < 0.0 || a > 1.0) {
                err(rl.lineno, rl.name + ": probability out of [0,1]"); return; }
        std::vector<int> qs;
        if (!qubit_targets(rl, qs)) return;     // touches each target; rejects bad targets
        if (arity == 2 && qs.size() % 2 != 0) {
            err(rl.lineno, "II_ERROR: target count must be a multiple of 2"); return; }
        // no-op: emit nothing into the stream
    }

    // MXX/MYY/MZZ q0 q1 [q2 q3 ...] : two-qubit Pauli-parity measurement, broadcast over
    // qubit PAIRS. Desugar by synthesizing an MPP line (P<q0>*P<q1> ...) and reusing the MPP
    // gadget, so Stim's product-sign convention is inherited. `!` on EITHER qubit of a pair
    // inverts that pair's record (Stim semantics); the (p) readout-flip arg is passed through
    // to emit_mpp, which forwards it to emit_measure as a per-record Bernoulli flip probability.
    // P is the per-axis Pauli letter ('X','Y','Z').
    void emit_parity(const RawLine& rl, char P) {
        if (rl.targets.size() % 2 != 0 || rl.targets.empty()) {
            err(rl.lineno, rl.name + ": needs an even, positive number of targets"); return; }
        RawLine mpp; mpp.lineno = rl.lineno; mpp.name = "MPP"; mpp.args = rl.args;
        for (size_t i = 0; i < rl.targets.size(); i += 2) {
            std::string t0 = rl.targets[i], t1 = rl.targets[i + 1];
            bool inv = false;
            if (!t0.empty() && t0[0] == '!') { inv = !inv; t0 = t0.substr(1); }
            if (!t1.empty() && t1[0] == '!') { inv = !inv; t1 = t1.substr(1); }
            std::string prod = (inv ? "!" : "");
            prod += std::string(1, P) + t0 + "*" + std::string(1, P) + t1;
            mpp.targets.push_back(prod);
        }
        emit_mpp(mpp);
    }

    // SPP P / SPP_DAG P : the Clifford Pauli-product rotation exp(∓iπ/4·P) (SPP = minus, SPP_DAG =
    // plus). A unitary GATE — no record. Desugar to a basis-change ladder onto a single PIVOT qubit
    // (the qubit of the first factor) so P → Z_pivot, an S/S_DAG on the pivot, then UNCOMPUTE the
    // ladder (derived against real Stim's tableau + dense unitary up-to-(even-)phase; see
    // docs/xtim_dialect.md):
    //   pre   : per factor, rotate its local Pauli to +Z   X->H ; Y->S_DAG,H ; Z->(nothing)
    //   fan   : per NON-pivot factor q_i   CX q_i pivot     (product collapses onto Z_pivot)
    //   rot   : S pivot  (SPP)  or  S_DAG pivot  (SPP_DAG)  — exp(∓iπ/4 Z) on the pivot
    //   uncompute: reverse fan (CX q_i pivot) then inverse pre  Y->H,S ; X->H ; Z->(nothing)
    // `!`-inversion (Stim allows it on ANY factor; each negates the product): the XOR-PARITY of the
    // `!` prefixes flips P->-P, i.e. flips the rotation sense — implemented by TOGGLING S<->S_DAG on
    // the pivot. Single-factor SPP X0 = exp(-iπ/4 X) = √X (H S H), the sanity anchor. No record, no
    // new GateKind: a pure parse-time rewrite to {S,SDG,H,CX}.
    void emit_spp(const RawLine& rl, bool dag) {
        if (!rl.args.empty()) { err(rl.lineno, rl.name + " takes no () arguments"); return; }
        if (rl.targets.empty()) { err(rl.lineno, rl.name + ": needs at least one Pauli product"); return; }
        for (const std::string& prod : rl.targets) {
            std::vector<PauliTerm> terms;
            bool invert = false;                     // running XOR-parity of `!` factor prefixes
            std::istringstream ps(prod);
            std::string piece;
            bool bad = false;
            while (std::getline(ps, piece, '*')) {
                if (!piece.empty() && piece[0] == '!') { invert = !invert; piece = piece.substr(1); }
                if (piece.size() < 2) { err(rl.lineno, rl.name + ": bad Pauli term '" + piece + "'"); bad = true; break; }
                char c = (char)std::toupper((unsigned char)piece[0]);
                if (c != 'X' && c != 'Y' && c != 'Z') {
                    err(rl.lineno, rl.name + ": bad Pauli term '" + piece + "' (expected X/Y/Z<qubit>)"); bad = true; break; }
                PauliBasis b = c == 'X' ? PauliBasis::X : c == 'Y' ? PauliBasis::Y : PauliBasis::Z;
                int q = 0;
                try { size_t pos = 0; q = std::stoi(piece.substr(1), &pos);
                      if (pos != piece.size() - 1 || q < 0) throw 0; }
                catch (...) { err(rl.lineno, rl.name + ": bad Pauli term '" + piece + "'"); bad = true; break; }
                if (q > kMaxQubitIndex) { err(rl.lineno, "qubit index too large (cap 1000000)"); bad = true; break; }
                for (const PauliTerm& t : terms)
                    if (t.qubit == q) { err(rl.lineno, rl.name + ": duplicate qubit in product '" + prod + "'"); bad = true; break; }
                if (bad) break;
                terms.push_back({q, b});
                touch(q);
            }
            if (bad) return;
            if (terms.empty()) { err(rl.lineno, rl.name + ": empty Pauli product '" + prod + "'"); return; }
            auto push_gate = [&](GateKind g, std::vector<int> tg) {
                Instr ins; ins.kind = Instr::Kind::Gate; ins.gate = g; ins.targets = std::move(tg);
                out.circuit.stream.push_back(std::move(ins));
            };
            const int pivot = terms[0].qubit;
            // pre: rotate each factor's local Pauli to +Z   X->H ; Y->S_DAG,H ; Z->()
            // (shared frame primitive rotate_to_z, clifford_frames)
            for (const PauliTerm& t : terms)
                for (const Instr& gi : rotate_to_z(t.p, t.qubit)) push_gate(gi.gate, gi.targets);
            // fan non-pivot factors onto the pivot
            for (size_t i = 1; i < terms.size(); ++i) push_gate(GateKind::CX, {terms[i].qubit, pivot});
            // pivot rotation; `!`-parity (invert) flips the sense => toggle S<->S_DAG
            push_gate((dag ^ invert) ? GateKind::SDG : GateKind::S, {pivot});
            // uncompute fan (reverse order)
            for (size_t i = terms.size(); i-- > 1;) push_gate(GateKind::CX, {terms[i].qubit, pivot});
            // uncompute pre (inverse rotate_from_z: Y->H,S ; X->H ; Z->())
            for (const PauliTerm& t : terms)
                for (const Instr& gi : rotate_from_z(t.p, t.qubit)) push_gate(gi.gate, gi.targets);
        }
    }

    void dispatch(const RawLine& rl) {
        if (overflowed) return;                              // C2
        const std::string& nm = rl.name;
        // gates
        if (nm == "I")            { emit_identity(rl); return; }   // no-op (touches targets)
        if (nm == "H" || nm == "H_XZ")  { emit_gate(rl, GateKind::H, 1, false); return; }
        if (nm == "S" || nm == "SQRT_Z")     { emit_gate(rl, GateKind::S,   1, false); return; }
        if (nm == "S_DAG" || nm == "SQRT_Z_DAG") { emit_gate(rl, GateKind::SDG, 1, false); return; }
        if (nm == "X")            { emit_gate(rl, GateKind::X,   1, false); return; }
        if (nm == "Y")            { emit_gate(rl, GateKind::Y,   1, false); return; }
        if (nm == "Z")            { emit_gate(rl, GateKind::Z,   1, false); return; }
        // CX/CY/CZ: pair-broadcast dispatch. A line with no rec/sweep delegates to the byte-
        // identical normal path (emit_gate for CX/CZ; the CY desugar word). A line containing
        // rec[-k] is walked in (control,target) pairs, each pair emitting a feedback (rec ctrl)
        // or a normal 2-qubit gate (qubit ctrl) — multi-pair and mixed forms supported. CX->X
        // feedback, CY->Y, CZ->Z; `normal_gate` is the (qubit,qubit) gate for CX/CZ, -1 = CY word.
        if (nm == "CX" || nm == "CNOT" || nm == "ZCX") { emit_c2(rl, PauliBasis::X, (int)GateKind::CX); return; }
        if (nm == "CY" || nm == "ZCY")                 { emit_c2(rl, PauliBasis::Y, -1);               return; }
        if (nm == "CZ" || nm == "ZCZ")                 { emit_c2(rl, PauliBasis::Z, (int)GateKind::CZ); return; }
        if (nm == "T")            { emit_gate(rl, GateKind::T,   1, false); return; }
        if (nm == "T_DAG")        { emit_gate(rl, GateKind::T,   1, false, (int)GateKind::SDG); return; }  // T·S† = T†
        if (nm == "CS")           { emit_gate(rl, GateKind::CS,  2, true);  return; }
        if (nm == "CS_DAG")       { emit_gate(rl, GateKind::CS,  2, true,  (int)GateKind::CZ);  return; }  // CS·CZ = CS†
        if (nm == "CCZ")          { emit_gate(rl, GateKind::CCZ, 3, true);  return; }
        if (nm == "CH")           { emit_gate(rl, GateKind::CH,  2, false); return; }
        // Desugared Stim Cliffords (table below) -> words over {X,Y,Z,S,SDG,H,CX,CZ}, each verified
        // == Stim's gate up to global phase (dense + Stim record-equality tests). Aliases ZCY/SWAPCZ
        // resolve to their canonical entry in the same table.
        if (const Desugar* d = lookup_desugar(nm)) { emit_word(rl, *d); return; }
        // ---- extension point: Task 2 cases (M*/R*/MR*, noise channels) ----
        if (nm == "M" || nm == "MZ")   { emit_measure(rl, PauliBasis::Z, false, false, false); return; }
        if (nm == "MX")                { emit_measure(rl, PauliBasis::X, false, false, false); return; }
        if (nm == "MY")                { emit_measure(rl, PauliBasis::Y, false, false, false); return; }
        if (nm == "R" || nm == "RZ")   { emit_reset(rl, false, false); return; }
        if (nm == "RX")                { emit_reset(rl, true,  false); return; }
        if (nm == "RY")                { emit_reset(rl, true,  true);  return; }
        if (nm == "MR" || nm == "MRZ") { emit_measure(rl, PauliBasis::Z, true, false, false); return; }
        if (nm == "MRX")               { emit_measure(rl, PauliBasis::X, true, true,  false); return; }
        if (nm == "MRY")               { emit_measure(rl, PauliBasis::Y, true, true,  true);  return; }
        if (nm == "X_ERROR")        { emit_noise(rl, NoiseChannel::X_ERROR, 1, 1); return; }
        if (nm == "Y_ERROR")        { emit_noise(rl, NoiseChannel::Y_ERROR, 1, 1); return; }
        if (nm == "Z_ERROR")        { emit_noise(rl, NoiseChannel::Z_ERROR, 1, 1); return; }
        if (nm == "DEPOLARIZE1")    { emit_noise(rl, NoiseChannel::DEPOLARIZE1, 1, 1); return; }
        if (nm == "DEPOLARIZE2")    { emit_noise(rl, NoiseChannel::DEPOLARIZE2, 1, 2); return; }
        if (nm == "PAULI_CHANNEL_1"){ emit_noise(rl, NoiseChannel::PAULI_CHANNEL_1, 3, 1); return; }
        if (nm == "PAULI_CHANNEL_2"){ emit_noise(rl, NoiseChannel::PAULI_CHANNEL_2, 15, 2); return; }
        // ---- extension point: Task 3 cases (annotations + named rejects) ----
        if (nm == "TICK") return;                                  // annotation (no-op)
        if (nm == "QUBIT_COORDS") { emit_qubit_coords(rl); return; }  // M-B: store coords
        if (nm == "SHIFT_COORDS") { emit_shift_coords(rl); return; }  // M-B: accumulate offset
        if (nm == "CORRELATED_ERROR" || nm == "ELSE_CORRELATED_ERROR" || nm == "E") {
            err(rl.lineno, nm + ": not in the v1 channel set; use PAULI_CHANNEL_*"); return; }
        if (nm == "MPP") { emit_mpp(rl); return; }             // desugar to ancilla gadget (Task 1)
        if (nm == "MXX") { emit_parity(rl, 'X'); return; }
        if (nm == "MYY") { emit_parity(rl, 'Y'); return; }
        if (nm == "MZZ") { emit_parity(rl, 'Z'); return; }
        if (nm == "SPP")     { emit_spp(rl, false); return; }   // exp(-iπ/4·P) desugar (Task 2)
        if (nm == "SPP_DAG") { emit_spp(rl, true);  return; }   // exp(+iπ/4·P) desugar (Task 2)
        if (nm == "MPAD") { emit_mpad(rl); return; }            // padding records (0/1), no qubits
        if (nm == "I_ERROR")  { emit_identity_error(rl, 1); return; }   // no-op noise placeholder
        if (nm == "II_ERROR") { emit_identity_error(rl, 2); return; }   // no-op noise placeholder (pairs)
        // ---- extension point: Task 4 cases (DETECTOR/OBSERVABLE_INCLUDE/PAULI_EXPECTATION) ----
        if (nm == "DETECTOR")           { emit_detector(rl); return; }
        if (nm == "OBSERVABLE_INCLUDE") { emit_observable(rl); return; }
        if (nm == "PAULI_EXPECTATION")  { emit_expectation(rl); return; }
        // ---- DECISION(k): declared Born output bit — mirrors OBSERVABLE_INCLUDE ----
        if (nm == "DECISION") { emit_decision(rl); return; }
        // ---- INPUT_QUBITS / INPUT_BITS: decoder-feedback input port declarations ----
        // Tokenizer puts all tokens after the keyword into rl.targets: for
        // `INPUT_QUBITS corr 0 1`, targets=["corr","0","1"]; for `INPUT_BITS dec 1`,
        // targets=["dec","1"]. Name must not be duplicated; INPUT_BITS width must be >0.
        if (nm == "INPUT_QUBITS" || nm == "INPUT_BITS") {
            if (!rl.args.empty()) { err(rl.lineno, nm + " takes no () arguments"); return; }
            const std::string port_nm = rl.targets.empty() ? std::string() : rl.targets[0];
            if (port_nm.empty()) { err(rl.lineno, nm + ": missing input name"); return; }
            for (const InputPort& p : out.circuit.inputs)
                if (p.name == port_nm) { err(rl.lineno, "duplicate input name '" + port_nm + "'"); return; }
            if (nm == "INPUT_QUBITS") {
                std::vector<int> qs;
                for (size_t i = 1; i < rl.targets.size(); ++i) {
                    int q = 0;
                    if (!parse_qubit_tok(rl.lineno, nm, rl.targets[i], q)) return;
                    qs.push_back(q);
                }
                out.circuit.inputs.push_back(InputPort{InputPort::Kind::Qubits, port_nm, 0, qs});
            } else {
                if (rl.targets.size() < 2) { err(rl.lineno, "INPUT_BITS: missing width"); return; }
                int w = 0;
                try {
                    size_t pos = 0;
                    long long wl = std::stoll(rl.targets[1], &pos);
                    if (pos != rl.targets[1].size()) throw std::invalid_argument("bad");
                    w = (int)wl;
                } catch (...) { err(rl.lineno, "INPUT_BITS: invalid width '" + rl.targets[1] + "'"); return; }
                if (w <= 0) { err(rl.lineno, "INPUT_BITS: width must be positive"); return; }
                out.circuit.inputs.push_back(InputPort{InputPort::Kind::Bits, port_nm, w, {}});
            }
            return;
        }
        // ---- OUTPUT_QUBITS: surviving qubit output port declaration ----
        // `OUTPUT_QUBITS name q...`: targets=["name","q0","q1",...].
        // Name must not be duplicated. Unmeasured invariant verified after walk() by
        // verify_output_qubits().
        if (nm == "OUTPUT_QUBITS") {
            if (!rl.args.empty()) { err(rl.lineno, "OUTPUT_QUBITS takes no () arguments"); return; }
            const std::string port_nm = rl.targets.empty() ? std::string() : rl.targets[0];
            if (port_nm.empty()) { err(rl.lineno, "OUTPUT_QUBITS: missing output name"); return; }
            for (const OutputPort& p : out.circuit.outputs)
                if (p.name == port_nm) { err(rl.lineno, "duplicate output name '" + port_nm + "'"); return; }
            std::vector<int> qs;
            for (size_t i = 1; i < rl.targets.size(); ++i) {
                int q = 0;
                if (!parse_qubit_tok(rl.lineno, nm, rl.targets[i], q)) return;
                qs.push_back(q);
            }
            out.circuit.outputs.push_back(OutputPort{port_nm, qs});
            return;
        }
        // Known Stim instructions deliberately unsupported in v1 — give a clear
        // "recognized but unsupported" message instead of "unknown instruction" (which
        // reads like a typo to a Stim user).
        if (nm == "HERALDED_ERASE" || nm == "HERALDED_PAULI_CHANNEL_1") {
            err(rl.lineno, "'" + nm + "' is a Stim heralded/erasure-noise instruction, "
                           "not supported in v1; model erasures with PAULI_CHANNEL_* or "
                           "post-selection");
            return;
        }
        err(rl.lineno, "unknown instruction '" + nm + "'");
    }

    void walk(const std::vector<Node>& nodes) {
        for (const Node& nd : nodes) {
            if (overflowed) return;                          // C2
            if (nd.is_repeat) {
                for (long long r = 0; r < nd.repeat_count && !overflowed; ++r) {
                    // tick per ITERATION too: an empty (or error-pruned) body dispatches
                    // nothing, so the per-line tick alone lets REPEAT 1e12 { } spin forever
                    // (second fuzzer-found hang shape).
                    ++processed;
                    if (processed > kMaxEmittedInstrs) {
                        overflowed = true;
                        out.errors.push_back({nd.repeat_lineno,
                            "circuit too large after REPEAT unrolling (cap 10000000 instructions)"});
                        break;
                    }
                    walk(nd.children);
                }
            } else if (nd.is_if) {
                // Resolve condition port and range-check the bit index.
                int port = -1;
                for (size_t i = 0; i < out.circuit.inputs.size(); ++i)
                    if (out.circuit.inputs[i].name == nd.if_name &&
                        out.circuit.inputs[i].kind == InputPort::Kind::Bits) { port = (int)i; break; }
                if (port < 0) {
                    err(nd.repeat_lineno, "IF: unknown decision port '" + nd.if_name + "'");
                } else if (nd.if_bit < 0 || nd.if_bit >= out.circuit.inputs[port].width) {
                    err(nd.repeat_lineno, "IF: bit index out of range");
                } else {
                    Instr ib{}; ib.kind = Instr::Kind::IfBegin;
                    ib.cond_port = port; ib.cond_bit = nd.if_bit; ib.cond_value = nd.if_value;
                    out.circuit.stream.push_back(ib);
                    // Task 6: push this IF's condition onto the guard stack before walking the
                    // body, so any DETECTOR declared inside inherits the full condition chain.
                    current_if_guards_.push_back(IfGuard{port, nd.if_bit, nd.if_value});
                    walk(nd.children);        // body NOT unrolled — markers kept as a span
                    current_if_guards_.pop_back();
                    out.circuit.stream.push_back(Instr{Instr::Kind::IfEnd});
                }
                ++processed;
                emitted = (long long)out.circuit.stream.size();
            } else {
                dispatch(nd.line);
                ++processed;                                        // counts EVERY dispatched line
                emitted = (long long)out.circuit.stream.size();
                if (!overflowed && (emitted > kMaxEmittedInstrs ||
                                    processed > kMaxEmittedInstrs)) {   // C2(+C2b): one error, stop
                    overflowed = true;
                    // push directly: must be reported even when the error list is already capped
                    out.errors.push_back({nd.line.lineno,
                        "circuit too large after REPEAT unrolling (cap 10000000 instructions)"});
                }
            }
        }
    }
};

}  // namespace

// Phase-2 entry point: construct the Emitter over `out`, place fresh ancillas at `anc_base`,
// walk the tree, and record the qubit count. Phase 1 (stim_parse.cpp) owns the post-walk error
// sort + circuit invalidation.
void run_emitter(ParsedStim& out, const std::vector<Node>& tree, int anc_base) {
    Emitter em{out};
    // MPP/SPP ancillas go ABOVE every user qubit so a desugar wire never collides with one the
    // circuit references later (pre-unroll scan is exact: qubit indices are unroll-invariant).
    em.anc_next = anc_base;
    em.walk(tree);
    // Post-walk: verify OUTPUT_QUBITS invariant (every declared output qubit is unmeasured).
    em.verify_output_qubits();
    out.circuit.n = em.max_qubit + 1;
}

}  // namespace stim_detail
}  // namespace qeccore
