#pragma once
// Portability shim for GCC/Clang bit-counting builtins on MSVC.
//
// The engine uses __builtin_ctz/clz/popcount (and their *ll 64-bit forms) in ~163
// places across 16 translation units. GCC and Clang (incl. clang-cl) provide these
// natively on every target ISA, so this header is intentionally EMPTY there — which
// keeps Linux/macOS builds byte-identical (the byte-parity engine gate is unaffected).
//
// MSVC (cl.exe) has no __builtin_*. When building the extension with MSVC we
// force-include this header (setup.py adds /FIqeccore/compiler_builtins.hpp) so the
// pervasive call sites compile unchanged. The mappings use _BitScanForward/Reverse and
// a software popcount: the results are identical integers, and we deliberately avoid the
// POPCNT instruction (__popcnt*) so portable wheels run on any x64 CPU.
//
// Note: defining macros over the reserved __builtin_* names is a pragmatic, widely-used
// Windows-port idiom; it is scoped strictly to MSVC, where those names are otherwise
// undefined.

#if defined(_MSC_VER) && !defined(__clang__)

#include <intrin.h>

namespace qeccore_compat {

inline int ctz32(unsigned int x) {  // undefined for x == 0, matching __builtin_ctz
    unsigned long i;
    _BitScanForward(&i, x);
    return static_cast<int>(i);
}
inline int ctz64(unsigned long long x) {  // matching __builtin_ctzll
    unsigned long i;
    _BitScanForward64(&i, x);
    return static_cast<int>(i);
}
inline int clz32(unsigned int x) {  // matching __builtin_clz
    unsigned long i;
    _BitScanReverse(&i, x);
    return 31 - static_cast<int>(i);
}
inline int clz64(unsigned long long x) {  // matching __builtin_clzll
    unsigned long i;
    _BitScanReverse64(&i, x);
    return 63 - static_cast<int>(i);
}
inline int popcount32(unsigned int x) {  // software popcount: no POPCNT instruction
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    return static_cast<int>((x * 0x01010101u) >> 24);
}
inline int popcount64(unsigned long long x) {  // matching __builtin_popcountll
    const unsigned long long m1 = 0x5555555555555555ull;
    const unsigned long long m2 = 0x3333333333333333ull;
    const unsigned long long m4 = 0x0f0f0f0f0f0f0f0full;
    x = x - ((x >> 1) & m1);
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;
    return static_cast<int>((x * 0x0101010101010101ull) >> 56);
}
inline int parity32(unsigned int x) {        // matching __builtin_parity
    return popcount32(x) & 1;
}
inline int parity64(unsigned long long x) {  // matching __builtin_parityll
    return popcount64(x) & 1;
}

}  // namespace qeccore_compat

#define __builtin_ctz(x)        ::qeccore_compat::ctz32(static_cast<unsigned int>(x))
#define __builtin_ctzll(x)      ::qeccore_compat::ctz64(static_cast<unsigned long long>(x))
#define __builtin_clz(x)        ::qeccore_compat::clz32(static_cast<unsigned int>(x))
#define __builtin_clzll(x)      ::qeccore_compat::clz64(static_cast<unsigned long long>(x))
#define __builtin_popcount(x)   ::qeccore_compat::popcount32(static_cast<unsigned int>(x))
#define __builtin_popcountll(x) ::qeccore_compat::popcount64(static_cast<unsigned long long>(x))
#define __builtin_parity(x)     ::qeccore_compat::parity32(static_cast<unsigned int>(x))
#define __builtin_parityll(x)   ::qeccore_compat::parity64(static_cast<unsigned long long>(x))
// __builtin_prefetch is a pure performance HINT (no observable effect) — a no-op on MSVC
// is correct. Variadic macro swallows the optional (rw, locality) args. (twirl_sampler.cpp
// uses it in the warm-path DRAM-latency prefetch; 2026-07-16.)
#define __builtin_prefetch(...) ((void)0)

#endif  // _MSC_VER && !__clang__
