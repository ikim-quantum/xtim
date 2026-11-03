// diag_error_sampler_gen.inc.hpp — INTERNAL (diag_error_sampler.cpp, diag_error_sampler_ranges.cpp): the ONE per-channel
// event generator body (geometric skips over channels [K0, K1), appended k-major to EV), expanded textually into the
// serial draw (every channel into ev_ks_ — the expanded source is the pre-T8 generator's, so is its code) and the
// scaling T8 range tasks (scaling R3: one body, no copy). Used inside a DiagErrorSampler member with `S` and a
// compile-time `Keyed` and the noise counter step `kGammaNoise` in scope.
#pragma once

// clang-format off
#define XTIM_DIAG_GEN_BODY(EV, K0, K1) \
    const int64_t S64 = S;                                                                                       \
    for (int k = (K0); k < (K1); ++k) {                                                                          \
        int64_t s = next_[k];                                                                                    \
        if (s < S64) {                                                                                           \
            const double sc = skipc_[k];                                                                         \
            const int32_t base = altbase_[k];                                                                    \
            const int nalt = altbase_[k + 1] - base;                                                             \
            if constexpr (Keyed) {                                                                               \
                /* The channel's stream position lives in a register for the channel's events (a */              \
                /* local: the event-vector stores cannot alias it) and is written back once. Each */             \
                /* event reads [categorical word if nalt > 1][skip word if not forced]. The words are */         \
                /* addressed, so the NEXT event's skip word is mixed one event ahead — off the */                \
                /* log -> skip -> s dependency chain (same draws, same values). */                               \
                uint64_t kp = kpos_[(size_t)k];                                                                  \
                const uint64_t dcat = (nalt > 1) ? kGammaNoise : 0;                                              \
                const uint64_t dev = dcat + kGammaNoise; /* counter advance per (unforced) event */              \
                uint64_t mnext = krng::mix(kp + dev); /* skip word of the first event */                         \
                do {                                                                                             \
                    const uint64_t pc = kp + kGammaNoise; /* categorical word (nalt > 1) */                      \
                    const uint64_t ps = pc + dcat; /* skip word */                                               \
                    const double uskip = ((mnext >> 11) + 1) * 0x1.0p-53;                                        \
                    mnext = krng::mix(ps + dev); /* one event ahead (off the s chain) */                         \
                    uint64_t aid = (uint64_t)base;                                                               \
                    if (nalt > 1) {                                                                              \
                        const double v = ((krng::mix(pc) >> 11) + 1) * 0x1.0p-53 * ptot_[k];                     \
                        const double* c = cum_.data() + cumoff_[k];                                              \
                        int t = 0;                                                                               \
                        while (t + 1 < nalt && v > c[t]) ++t;                                                    \
                        aid += (uint64_t)t;                                                                      \
                    }                                                                                            \
                    (EV).push_back((aid << 32) | (uint32_t)s);                                                   \
                    if (sc > 0.0) { kp = pc + dcat - kGammaNoise; s += 1; continue; } /* forced: no skip word */ \
                    kp = ps;                                                                                     \
                    const double gp = std::log(uskip) * sc;                                                      \
                    s += 1 + ((gp < 4.0e18) ? (int64_t)gp : ((int64_t)1 << 62));                                 \
                } while (s < S64);                                                                               \
                kpos_[(size_t)k] = kp;                                                                           \
            } else {                                                                                             \
                do {                                                                                             \
                    uint64_t aid = (uint64_t)base;                                                               \
                    if (nalt > 1) {                                                                              \
                        /* Conditional categorical draw, paid per FIRED event only: v uniform on */              \
                        /* (0, p_total]; alternative t iff cum[t-1] < v <= cum[t] ⇒ P = p_t/p_total. */          \
                        const double v = runi_open() * ptot_[k];                                                 \
                        const double* c = cum_.data() + cumoff_[k];                                              \
                        int t = 0;                                                                               \
                        while (t + 1 < nalt && v > c[t]) ++t;                                                    \
                        aid += (uint64_t)t;                                                                      \
                    }                                                                                            \
                    (EV).push_back((aid << 32) | (uint32_t)s);                                                   \
                    if (sc > 0.0) { s += 1; continue; } /* forced channel: fires every shot */                   \
                    /* log(U) <= 0 and sc < 0 ⇒ the product is >= 0: truncation == floor. Clamped */             \
                    /* before the cast (p_total < ~4e-18 could otherwise overflow int64 — UB). */                \
                    const double gp = std::log(runi_open()) * sc;                                                \
                    s += 1 + ((gp < 4.0e18) ? (int64_t)gp : ((int64_t)1 << 62));                                 \
                } while (s < S64);                                                                               \
            }                                                                                                    \
        }                                                                                                        \
        next_[k] = s - S64;                                                                                      \
    }                                                                                                            \
    do {} while (0)
// clang-format on
