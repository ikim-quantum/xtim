#pragma once
// Internal seam between stim_parse.cpp (phase 1: the line-tree builder) and
// stim_emit.cpp (phase 2: the Emitter that unrolls the tree into a ParsedStim).
// NOT a public header — only those two translation units include it. The public API
// (parse_stim_circuit -> ParsedStim) lives in qeccore/stim_parse.hpp and is unchanged.
#include "qeccore/stim_parse.hpp"
#include "qeccore/circuit_ir.hpp"
#include <string>
#include <vector>

namespace qeccore {
namespace stim_detail {

// ---------- phase 1: the line tree (shared shape between the two TUs) ----------
struct RawLine { int lineno; std::string name; std::vector<double> args; std::vector<std::string> targets; };
struct Node {                                   // either an instruction, a REPEAT block, or an IF block
    bool is_repeat = false;
    bool is_if = false;                         // valid when set: an IF block
    std::string if_name;                        // condition port name (valid when is_if)
    int if_bit = 0;                             // condition bit index (valid when is_if)
    bool if_value = true;                       // false for `IF !name[i]` (valid when is_if)
    RawLine line;                               // valid when !is_repeat && !is_if
    long long repeat_count = 0;                 // valid when is_repeat
    int repeat_lineno = 0;                      // valid when is_repeat or is_if (header line number)
    std::vector<Node> children;
};

// Hostile-input caps (review C1-C4): keep parse output sizes sane so downstream
// dense allocations (observable_bits, qubit arrays, the unrolled stream) stay bounded.
constexpr long long kMaxQubitIndex = 1'000'000;
constexpr long long kMaxObservableIndex = 1'000'000;
constexpr long long kMaxEmittedInstrs = 10'000'000;
constexpr int kMaxRepeatDepth = 256;

// Phase-2 entry point (defined in stim_emit.cpp). Walks the (pre-scanned) tree into `out`,
// placing fresh MPP/SPP desugar ancillas at and above `anc_base` (= 1 + max user qubit, from
// scan_max_qubit). Sets out.circuit.n. Phase-1 errors already in out.errors are appended to.
void run_emitter(ParsedStim& out, const std::vector<Node>& tree, int anc_base);

}  // namespace stim_detail
}  // namespace qeccore
