#pragma once
// keyed_rng — the ONE random generator of the twirl sampler (S0(b), design §3.4 [E4], D10/D11).
//
// Every draw is the SplitMix64 output finalizer `mix` (Stafford's "Mix13" constants) of a keyed
// counter, γ = 2^64/φ (odd):
//
//   per-shot streams:  draw(seed, shot, tag, ctr) = mix( key(seed, tag) + (shot·2^20 + ctr + 1)·γ )
//   noise stream:      group g, block b, draw c   = mix( key(seed, tag) + (b·R + r0_g + c + 1)·γ )
//   key(seed, tag)  =  mix(seed ^ mix(tag))       — the tag goes THROUGH the hash, never XOR-linearly
//
// `shot` is the CUMULATIVE shot index; `ctr` the draw index inside one (shot, tag) stream. The noise
// stream (geometric skips, which span shots by construction) has one chain per (equal-P GROUP g of mechanisms,
// aligned 4096-shot BLOCK b) — lean N1, qeccore/noise_chain.hpp: tag NOISE (the pruned mechanism table) or
// NOISE_FULL (the full channel table of the runs that read P_f). Lean N2: chain (g, b) OWNS the counter range
// [b·R + r0_g, b·R + r0_g + R_g) of its tag's sequence (R_g = 2·M_g·4096 + 1 bounds its draws, r0_g = Σ_{g'<g} R_g',
// R = Σ R_g), so no two chains of one (seed, tag) share a draw; a block whose range would pass 2^64 is refused.
// Disjointness holds WITHIN one (seed, tag) only: all tags sit on the same γ-mix cycle (the key is a shift on it), so the
// NOISE span can overlap a COIN / BORN / RF per-shot window. That is a rare generator-quality event (≈ 7e-4 expected on
// the s25 circuit at 1e10 shots), not a law change; a per-tag γ would be the fix and would move every stream.
//
// Why this generator (stateless, counter-based):
//  * SplitMix64 (Steele, Lea & Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014)
//    IS counter-based: output i = mix(seed + i·γ); it passes BigCrush. Here the counter of a
//    per-shot draw is shot·2^20 + ctr: for ctr < 2^20 and shot < 2^44 distinct (shot, ctr) give
//    distinct counters, γ is odd (the counter -> key + (i+1)·γ map is a bijection mod 2^64) and mix
//    is a bijection — so no two draws of one (seed, tag) ever coincide (the sampler refuses, by
//    name, a shot index or a per-shot draw count outside that range).
//  * A draw is ADDRESSED: route, call partition and plan-cache warmth cannot change which draw a
//    shot reads, only whether it reads it — the determinism contract D10.
//  * Cost: one mix (3 shift-xor, 2 multiplies) per draw; positioning a stream at a shot is one
//    multiply-add. Pure uint64 arithmetic, identical on every platform (D13: only std::log in the
//    noise skips and the double Born probabilities are platform-sensitive).
#include <cstdint>

namespace qeccore {
namespace krng {

constexpr uint64_t kGamma = 0x9E3779B97F4A7C15ull;
constexpr int kCtrBits = 20;                               // draws per (shot, tag) < 2^20
constexpr uint64_t kMaxCtr = 1ull << kCtrBits;
constexpr uint64_t kMaxShots = 1ull << (64 - kCtrBits);    // cumulative shots per seed < 2^44
// CAPS-2 A4 (re-checked on the lean N1 layout): both limits are refused BY NAME (run_w / run(shots, threads)), never
// wrapped, and neither is reachable by a circuit the sampler can compile, so they stay (owner rule: a limit far beyond
// any practical circuit is acceptable).
//  (a) draws per (shot, stream) < 2^20. The live per-shot streams are COIN (ceil(rho / 64) words, rho <= ng certified
//      generators), BORN (one word per Born step, kappa <= 2k <= 2 * kMaxBareRank = 60) and RF (<= nrf words, nrf =
//      noisy record slots); the noise chains own counter ranges per (group, block), not per shot (no per-shot counter). The
//      check refuses max(nrf, ng, #Born decisions) >= 2^20 (an over-count for COIN). Every record is a deferred wire,
//      so ng, nrf >= 2^20 needs n >= 2^20 wires, and the compile is quadratic in n (measured 2026-11-07, one qubit,
//      N noisy M records: N = 16384 -> 16 s / 1.5 GB, N = 65536 -> 471 s / 22.8 GB; at 2^20: ~5.8 TB, ~5 days).
//  (b) cumulative shots per seed < 2^44 = 1.76e13: at the fastest measured shot (surface d7 r7, 88 ns) that is ~18
//      CPU-days on ONE seed; a new seed resets it (the refusal says so).

inline uint64_t mix(uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Stream tags. Values are part of the byte contract (ENGINE_VERSION).
enum Tag : uint64_t {
    NOISE = 1,   // noise chains of the PRUNED mechanism table: per (group, block) (lean N1; keyed per channel before engine-17)
    COIN  = 2,   // fair coins: coin k = bit (k mod 64) of word floor(k/64)
    BORN  = 3,   // kernel Born steps: step j = uniform from word j
    OBS   = 4,   // RETIRED at J-b2 (the PPR family's Born observable): no engine draw; the value is never reused
    // 5 was DEC (Born decisions) — retired at S2b; the value is never reused (byte contract)
    RF    = 6,   // readout flips (Stim M(p)): per group of equal-p slots one geometric-gap chain on the shot's sequential words (lean L5)
    GAUGE = 7,   // RETIRED at J-b2 (gauge-detector fair coins; gauge detectors are records): the value is never reused
    EXACT = 8,   // RETIRED at J-b2 (the per-shot exact-fallback engine): no engine draw; the value is never reused
    NOISE_FULL = 9,   // noise chains of the FULL channel table (recording / framed / term-recorded runs; lean N1)
    // (OBS / GAUGE / EXACT stay as NAMES only: the ledger's pinned rng_ir.cpp harness keys its uniform draw by OBS)
};

inline uint64_t stream_key(uint64_t seed, uint64_t tag) noexcept { return mix(seed ^ mix(tag)); }
// Position of a per-shot stream: draw ctr = mix(shot_base + (ctr + 1)·γ).
inline uint64_t shot_base(uint64_t key, uint64_t shot) noexcept { return key + (shot << kCtrBits) * kGamma; }
// The reference definition of a per-shot draw (tests pin every stream against it).
inline uint64_t draw(uint64_t seed, uint64_t shot, uint64_t tag, uint64_t ctr) noexcept {
    return mix(shot_base(stream_key(seed, tag), shot) + (ctr + 1) * kGamma);
}
// The pre-N1 per-channel noise stream base (engine-11 .. engine-16). No engine draw uses it since lean N1; it stays as a
// NAME for the ledger's pinned rng_ir.cpp harness (ns_draw/noise, built against every measured tree's headers).
inline uint64_t channel_base(uint64_t seed, uint64_t k) noexcept {
    return mix(stream_key(seed, NOISE) + k * kGamma);
}
// Position of a noise chain whose counter range starts at ctr0 (lean N2; NoiseModel::chain_pos): draw c =
// mix(noise_pos + (c + 1)·γ) — the shot_base of the noise stream.
inline uint64_t noise_pos(uint64_t key, uint64_t ctr0) noexcept { return key + ctr0 * kGamma; }
// [0, 1) with 53 random bits.
inline double u01(uint64_t x) noexcept { return (double)(x >> 11) * 0x1.0p-53; }

// One (seed, tag) per-shot stream positioned at a shot (a multiply-add); next()/word() one mix each.
struct Stream {
    uint64_t key = 0;    // stream_key(seed, tag)
    uint64_t base = 0;   // shot_base(key, shot)
    uint64_t pos = 0;    // base + (ctr + 1)·γ of the last sequential draw
    void set_key(uint64_t seed, uint64_t tag) noexcept { key = stream_key(seed, tag); }
    void at(uint64_t shot) noexcept { base = pos = shot_base(key, shot); }
    uint64_t next() noexcept { pos += kGamma; return mix(pos); }      // sequential: ctr 0, 1, ...
    uint64_t word(uint64_t ctr) const noexcept { return mix(base + (ctr + 1) * kGamma); }
    double u(uint64_t ctr) const noexcept { return u01(word(ctr)); }
};

// Fair coins of one shot: coin k = bit (k mod 64) of COIN word floor(k/64), consumed in order.
// A copy is a stream-neutral fork (the self-check window replays the shot's coins on a copy).
struct CoinBits {
    Stream st;
    uint64_t res = 0;
    int left = 0;
    void at(uint64_t shot) noexcept { st.at(shot); left = 0; }
    bool next() noexcept {
        if (left == 0) { res = st.next(); left = 64; }
        const bool b = res & 1; res >>= 1; --left;
        return b;
    }
};

}  // namespace krng
}  // namespace qeccore
