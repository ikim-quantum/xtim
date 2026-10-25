#pragma once
// Caller-owned kernel scratch (TLC; the S2k AmpIndex pattern): a kernel's working buffers live in a
// scratch object its CALLER owns and passes down, never in thread_local storage.
//
//   ScratchCharge — the high-water state-budget charge of a χ-scaled scratch: need(bytes, what)
//                   charges only the growth, BEFORE the buffers grow (a refusal leaves the previous
//                   charge unchanged and nothing allocated); released with its owner (or reset()).
//   DualScanScratch — the dual-image scan's scratch (below).
//   ScratchUse    — Debug-build re-entrancy guard on one scratch part (the flag is always present:
//                   layouts do not depend on NDEBUG).
#include <cassert>
#include <cstdint>
#include <vector>

#include "qeccore/state_budget.hpp"

namespace qeccore {

struct ScratchCharge {
    StateBudgetTicket ticket;
    void need(int64_t bytes, const char* what) {
        if (bytes > ticket.bytes()) ticket.charge(bytes - ticket.bytes(), what);
    }
    int64_t bytes() const { return ticket.bytes(); }
};

struct ScratchUse {
#ifndef NDEBUG
    bool& busy;
    explicit ScratchUse(bool& b) : busy(b) { assert(!busy && "kernel scratch re-entered"); busy = true; }
    ~ScratchUse() { busy = false; }
#else
    explicit ScratchUse(bool&) {}
#endif
    ScratchUse(const ScratchUse&) = delete;
    ScratchUse& operator=(const ScratchUse&) = delete;
};

// dual_image_rows_scan's caller-owned scratch (pauli_kernels.hpp): the running products' Z-words
// (flat cnt×W) and phases, and their χ-scaled charge.
struct DualScanScratch {
    std::vector<uint64_t> az;
    std::vector<int> acc_phase;
    ScratchCharge charge;
    bool busy = false;
};

}  // namespace qeccore
