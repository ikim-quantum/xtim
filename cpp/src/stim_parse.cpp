#include "qeccore/stim_parse.hpp"
#include <cstddef>
#include "stim_emit_internal.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <string>

namespace qeccore {

namespace {

using stim_detail::RawLine;
using stim_detail::Node;
using stim_detail::kMaxQubitIndex;
using stim_detail::kMaxRepeatDepth;

// True iff `s` opens with block keyword `kw` (compared case-insensitively),
// followed by whitespace.
//
// The BLOCK keywords (REPEAT, IF) are matched on the raw line BEFORE
// `split_instruction` upper-cases instruction names, so they do not inherit
// the case-insensitivity every ordinary instruction gets for free.  That left
// the grammar inconsistent: `detector`, `decision(0)` and `input_qubits` all
// parsed, while `if` did not, and REPEAT accepted exactly three hand-written
// spellings ("REPEAT"/"repeat"/"Repeat") so `rEpEaT` failed.
//
// The failure was loud (never a wrong answer), but the diagnostic was
// actively misleading: lowercase `if` reported `unknown instruction 'IF'` --
// quoting the name back UPPER-CASED, naming the very keyword the user
// believed they had written -- plus a cascading `unmatched '}'` on a
// blameless line.
static bool opens_block(const std::string& s, const char* kw) {
    size_t n = std::strlen(kw);
    if (s.size() < n + 1) return false;
    for (size_t i = 0; i < n; ++i) {
        if (std::toupper((unsigned char)s[i]) != (unsigned char)kw[i]) return false;
    }
    return s[n] == ' ' || s[n] == '\t';
}

// ---------- phase 1: the line tree ----------
struct LineSplitter {
    std::vector<std::pair<int, std::string>> lines;   // (lineno, content) — comments stripped, trimmed
    explicit LineSplitter(const std::string& text) {
        std::istringstream is(text);
        std::string raw; int no = 0;
        while (std::getline(is, raw)) {
            ++no;
            size_t h = raw.find('#');
            if (h != std::string::npos) raw = raw.substr(0, h);
            size_t b = raw.find_first_not_of(" \t\r");
            if (b == std::string::npos) continue;
            size_t e = raw.find_last_not_of(" \t\r");
            lines.push_back({no, raw.substr(b, e - b + 1)});
        }
    }
};

// Parse "NAME(arg, arg) tok tok ..." into a RawLine. Returns false + msg on malformed shape.
bool split_instruction(const std::string& s, RawLine& out, std::string& msg) {
    size_t i = 0;
    while (i < s.size() && (std::isalnum((unsigned char)s[i]) || s[i] == '_')) ++i;
    if (i == 0) { msg = "expected an instruction name"; return false; }
    out.name = s.substr(0, i);
    std::transform(out.name.begin(), out.name.end(), out.name.begin(),
                   [](unsigned char c){ return (char)std::toupper(c); });
    // Stim >=1.14 gate `[tag]` suffix (e.g. `H[foo] 0`): a bracketed annotation directly after
    // the instruction name and before any `(args)`. Accept-and-ignore it so tagged circuits load
    // (the tag carries no sampling semantics). This is name-adjacent only; target tokens like
    // rec[-1] are handled later and are never adjacent to the name.
    if (i < s.size() && s[i] == '[') {
        size_t close = s.find(']', i);
        if (close == std::string::npos) { msg = "unterminated '[' tag"; return false; }
        i = close + 1;
    }
    if (i < s.size() && s[i] == '(') {
        size_t close = s.find(')', i);
        if (close == std::string::npos) { msg = "unterminated '(' argument list"; return false; }
        std::string inside = s.substr(i + 1, close - i - 1);
        std::istringstream as(inside);
        std::string piece;
        while (std::getline(as, piece, ',')) {
            try { out.args.push_back(std::stod(piece)); }
            catch (...) { msg = "bad numeric argument '" + piece + "'"; return false; }
        }
        i = close + 1;
    }
    std::istringstream ts(s.substr(i));
    std::string tok;
    while (ts >> tok) out.targets.push_back(tok);
    return true;
}

// Build the node tree (handles nested REPEAT). Appends errors; returns the tree.
std::vector<Node> build_tree(const std::vector<std::pair<int, std::string>>& lines,
                             size_t& idx, bool in_block, std::vector<ParseError>& errors,
                             int depth) {
    std::vector<Node> out;
    while (idx < lines.size()) {
        int no = lines[idx].first;
        const std::string& s = lines[idx].second;
        if (s == "}") {
            if (!in_block) errors.push_back({no, "unmatched '}'"});
            ++idx;
            if (in_block) return out;
            continue;
        }
        if (opens_block(s, "REPEAT")) {
            std::istringstream rs(s);
            std::string kw; long long cnt = -1; std::string brace;
            rs >> kw >> cnt >> brace;
            if (cnt < 1 || brace != "{") {
                errors.push_back({no, "malformed REPEAT (expected 'REPEAT <n> {')"});
                ++idx;
                if (!s.empty() && s.back() == '{') {        // header still opened a block: skip its body
                    int depth = 1;
                    while (idx < lines.size() && depth > 0) {
                        const std::string& t = lines[idx].second;
                        if (t == "}") --depth;
                        else if (!t.empty() && t.back() == '{') ++depth;
                        ++idx;
                    }
                }
                continue;
            }
            ++idx;
            if (depth >= kMaxRepeatDepth) {     // C4: bound recursion; skip the block's body
                errors.push_back({no, "REPEAT nesting too deep (cap 256)"});
                int d = 1;
                while (idx < lines.size() && d > 0) {
                    const std::string& t = lines[idx].second;
                    if (t == "}") --d;
                    else if (!t.empty() && t.back() == '{') ++d;
                    ++idx;
                }
                continue;
            }
            Node nd; nd.is_repeat = true; nd.repeat_count = cnt; nd.repeat_lineno = no;
            nd.children = build_tree(lines, idx, true, errors, depth + 1);
            out.push_back(std::move(nd));
            continue;
        }
        // IF [!]name[bit] { — conditional block; reuse REPEAT's brace-matching machinery.
        if (opens_block(s, "IF")) {
            std::istringstream ifs(s);
            std::string kw, cond, brace;
            ifs >> kw >> cond >> brace;
            // Parse cond = [!]name[bit]
            bool if_value = true;
            std::string rest = cond;
            if (!rest.empty() && rest[0] == '!') { if_value = false; rest = rest.substr(1); }
            size_t lb = rest.find('[');
            size_t rb = rest.rfind(']');
            bool malformed = (brace != "{" || cond.empty() ||
                              lb == std::string::npos || rb == std::string::npos || rb <= lb);
            std::string if_name;
            int if_bit = 0;
            if (!malformed) {
                if_name = rest.substr(0, lb);
                std::string bit_str = rest.substr(lb + 1, rb - lb - 1);
                try {
                    size_t pos = 0;
                    long long v = std::stoll(bit_str, &pos);
                    if (pos != bit_str.size() || v < 0) malformed = true;
                    else if_bit = (int)v;
                } catch (...) { malformed = true; }
            }
            if (malformed || if_name.empty()) {
                errors.push_back({no, "malformed IF (expected 'IF [!]name[bit] {')"});
                ++idx;
                if (!s.empty() && s.back() == '{') {        // header opened a block: skip its body
                    int d = 1;
                    while (idx < lines.size() && d > 0) {
                        const std::string& t = lines[idx].second;
                        if (t == "}") --d;
                        else if (!t.empty() && t.back() == '{') ++d;
                        ++idx;
                    }
                }
                continue;
            }
            // Finding 2: inline IF body — non-whitespace content after the opening
            // '{' on the same line (e.g. 'IF dec[0] { X 0 }').  Multi-line syntax is
            // required; the inline form silently swallows the body and following lines.
            {
                std::string inline_tail;
                if (ifs >> inline_tail) {
                    errors.push_back({no,
                        "inline IF body unsupported; put the body on separate lines"});
                    ++idx;
                    continue;
                }
            }
            ++idx;
            if (depth >= kMaxRepeatDepth) {     // C4: bound recursion; skip the block's body
                errors.push_back({no, "IF nesting too deep (cap 256)"});
                int d = 1;
                while (idx < lines.size() && d > 0) {
                    const std::string& t = lines[idx].second;
                    if (t == "}") --d;
                    else if (!t.empty() && t.back() == '{') ++d;
                    ++idx;
                }
                continue;
            }
            Node nd; nd.is_if = true; nd.if_name = if_name; nd.if_bit = if_bit;
            nd.if_value = if_value; nd.repeat_lineno = no;
            nd.children = build_tree(lines, idx, true, errors, depth + 1);
            out.push_back(std::move(nd));
            continue;
        }
        RawLine rl; rl.lineno = no;
        std::string msg;
        if (!split_instruction(s, rl, msg)) errors.push_back({no, msg});
        else { Node nd; nd.line = std::move(rl); out.push_back(std::move(nd)); }
        ++idx;
    }
    if (in_block) errors.push_back({lines.empty() ? 0 : lines.back().first, "REPEAT block not closed"});
    return out;
}

// Recursively scan the (pre-unroll) node tree for the maximum plain integer qubit token across
// every instruction's targets. Used ONLY to place MPP/SPP desugar ancillas ABOVE every user
// qubit so a fresh-ancilla wire can never collide with a qubit the circuit references later
// (qubit indices are unroll-invariant, so the pre-unroll scan is exact). Non-integer / annotation
// tokens (rec[-k], sweep, !-prefixed Pauli products, etc.) are skipped — they carry no plain
// qubit index here; malformed integers are left for the per-instruction emitters to diagnose.
long long scan_max_qubit(const std::vector<Node>& nodes) {
    long long mx = -1;
    auto consider = [&](long long v) { if (v >= 0 && v <= kMaxQubitIndex) mx = std::max(mx, v); };
    for (const Node& nd : nodes) {
        if (nd.is_repeat || nd.is_if) { mx = std::max(mx, scan_max_qubit(nd.children)); continue; }
        for (const std::string& t : nd.line.targets) {
            if (t.empty()) continue;
            // Pauli-product token (MPP/SPP/PAULI_EXPECTATION): a `*`-joined run of P<qubit>
            // factors, optional leading `!`. Pull every embedded qubit index so the ancilla
            // base sits above qubits referenced ONLY inside a product (e.g. q2 in `MPP X0*X2`).
            const char c0 = t[0];
            if (c0 == '!' || c0 == 'X' || c0 == 'Y' || c0 == 'Z' ||
                c0 == 'x' || c0 == 'y' || c0 == 'z') {
                std::string body = (c0 == '!') ? t.substr(1) : t;
                std::istringstream ps(body);
                std::string piece;
                while (std::getline(ps, piece, '*')) {
                    if (piece.size() < 2) continue;
                    char pc = (char)std::toupper((unsigned char)piece[0]);
                    if (pc != 'X' && pc != 'Y' && pc != 'Z') continue;
                    try { size_t pos = 0; long long v = std::stoll(piece.substr(1), &pos);
                          if (pos == piece.size() - 1) consider(v); }
                    catch (...) { /* malformed factor; emitter diagnoses */ }
                }
                continue;
            }
            if (!std::isdigit((unsigned char)c0)) continue;     // rec[-k]/sweep/etc.
            try {
                size_t pos = 0;
                long long v = std::stoll(t, &pos);
                if (pos == t.size()) consider(v);
            } catch (...) { /* not a plain qubit token; emitter will diagnose if needed */ }
        }
    }
    return mx;
}

}  // namespace

ParsedStim parse_stim_circuit(const std::string& text) {
    ParsedStim out;
    LineSplitter ls(text);
    size_t idx = 0;
    std::vector<Node> tree = build_tree(ls.lines, idx, false, out.errors, 0);
    // Phase 2 (stim_emit.cpp): MPP/SPP ancillas go ABOVE every user qubit so a desugar wire never
    // collides with one the circuit references later (pre-unroll scan is exact: qubit indices are
    // unroll-invariant). run_emitter walks the tree into `out` and sets out.circuit.n.
    stim_detail::run_emitter(out, tree, (int)(scan_max_qubit(tree) + 1));
    std::stable_sort(out.errors.begin(), out.errors.end(),  // phase-1 + phase-2 errors in source order
                     [](const ParseError& a, const ParseError& b) { return a.line < b.line; });
    if (!out.errors.empty()) out.circuit = Circuit{};       // any error => unusable circuit
    return out;
}

}  // namespace qeccore
