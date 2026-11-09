#pragma once
// Process-level memo of the FRAMED bare state (perf/dem-share-bare).
//
// Every bare-state consumer — the exact sampler compile (compile_sampler_program), the twirl
// engine compile (PyTwirlSampler), DEM export, detector_determinism, expectation_frame_check, the
// reference reconstruction — calls build_bare_state_framed on ITS deferred circuit. A DEM export
// after a compile of the same circuit rebuilt the whole pipeline (FastTODD + coset-family assembly
// + per-H canonicalisation: ~70 % of a bt81 DEM, 60 % of its compile). This memo makes the build
// a memoised pure function of its input, so the second caller reuses the first's result.
//
// THE KEY = the build's exact read set. build_bare_state_parity reads `deferred.n` and the Gate
// instructions in stream order (gate kind + targets) and NOTHING else: build_pauli_rotation_form
// skips every non-Gate instruction (pauli_rotation_form.cpp, `if (ins.kind != Kind::Gate)
// continue;`), and every later stage (diagonalize / restrict_to_support / FastTODD / assemble)
// consumes only the form + n. Noise channels and probabilities, Measure bases / inverts / flip
// probabilities, Resets, Observables, feedback, ports: all invisible to the build. The key is the
// exact serialisation of (n, gate list) — never a hash alone, never the user text — so a hit means
// the fresh build would read byte-for-byte the same input. test_bare_state_memo pins this read set
// by MUTATION: every non-gate field of a corpus circuit is perturbed and the uncached build must
// stay byte-identical; a gate perturbation must change the key.
//
// THE VALUE = the fresh build's output, captured verbatim: the frame tableau (dual rows included
// iff the build left them valid), eps, free, k, and the (sigma, coefficient) entries, held in
// PLAIN (non-budgeted) storage. A hit re-materialises a FramedSuperposition exactly as
// FramedSuperposition::from_css does from a CanonicalStabSum (the same field assignments), so the
// returned object has the same contents as a fresh build; bare_states_identical is the
// field-by-field (bit-level for coefficients) comparator, run on every hit under
// XTIM_BARE_MEMO=verify and in the tests.
//
// The memo is NOT chi-scaled state storage: it holds copies outside the state budget
// (state_budget.hpp), so live_state_bytes() and every refusal are exactly what they were without
// it; a hit's FramedSuperposition is charged like a fresh build's. Bounded by count AND bytes
// (XTIM_BARE_MEMO_ENTRIES, XTIM_BARE_MEMO_MB; an entry above the byte cap is never stored), LRU,
// guarded by one mutex (the bindings release the GIL around every compile / export).
//
// Levers (read ONCE per process, function-local statics; tests toggle them in subprocesses). A malformed
// value is refused by name (std::invalid_argument -> ValueError naming the variable and the value):
//   XTIM_BARE_MEMO=0        disable (every call builds fresh; diagnostic / oracle)
//   XTIM_BARE_MEMO=verify   every hit ALSO builds fresh and compares; a difference throws
//                           BareStateInvariantError naming the first differing field
//   XTIM_BARE_MEMO_ENTRIES=<n>   max entries (default 8)
//   XTIM_BARE_MEMO_MB=<n>        max total bytes in MiB (default 256; 0 = store nothing)
//   XTIM_BARE_MEMO_ENTRY_MB=<n>  max bytes of ONE entry in MiB (default 16): a state above it is never
//                                captured (the size is known before any copy), so a large-chi build
//                                (bh20: chi = 2^20, ~97 MB, allocation-bound to build AND to copy)
//                                costs nothing on a miss — its cold DEM and RSS stay what they were.
#include <cstddef>
#include <cstdint>
#include <string>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/normal_form.hpp"

namespace qeccore {

// The fresh (un-memoised) framed build: build_bare_state + FramedSuperposition::from_css at the
// shared conversion point. build_bare_state_framed (normal_form.hpp) = this, memoised.
FramedBareState build_bare_state_framed_uncached(const Circuit& circ);

// The memo key: exact serialisation of the build's read set (n, then every Gate instruction's
// kind + targets in stream order). Two circuits with equal keys are read identically by the build.
std::string bare_state_memo_key(const Circuit& circ);

// Field-by-field equality of two framed bare states (rejected flag / chi / reject index, tableau
// rows incl. the dual when valid, eps, free, k, every (sigma, coefficient) entry with the
// coefficient compared BIT-wise). On a difference returns false and names the field in `why`.
bool bare_states_identical(const FramedBareState& a, const FramedBareState& b, std::string* why);

struct BareStateMemoStats {
    uint64_t hits = 0;        // lookups served from the memo
    uint64_t misses = 0;      // lookups that built fresh
    uint64_t stores = 0;      // fresh builds stored
    uint64_t skipped = 0;     // fresh builds NOT stored (over the byte cap, or rejected)
    uint64_t evictions = 0;   // entries evicted (LRU) to make room
    uint64_t bypassed = 0;    // calls that did not consult the memo (disabled / test hook set)
    size_t entries = 0;       // current entry count
    size_t bytes = 0;         // current held bytes (keys + values, estimated per entry)
    size_t cap_entries = 0;
    size_t cap_bytes = 0;
    size_t cap_entry_bytes = 0;   // one entry above this is never captured (skipped, no copy)
    bool enabled = false;
    bool verify = false;
};
BareStateMemoStats bare_state_memo_stats();
void bare_state_memo_clear();                       // drop every entry (counters kept)
void bare_state_memo_reset_stats();                 // zero the counters (entries kept)
// Re-bound the memo (tests): entries above either cap are evicted immediately. The env levers set
// the initial caps; this overrides them for the rest of the process.
void bare_state_memo_set_caps(size_t max_entries, size_t max_bytes);
void bare_state_memo_set_entry_cap(size_t max_entry_bytes);   // held entries are not re-checked
// Estimated held bytes of one captured entry (what the byte cap counts).
size_t bare_state_entry_bytes(const FramedBareState& s);

}  // namespace qeccore
