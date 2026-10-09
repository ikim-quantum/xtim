#pragma once
// Minimal CHECK / RUN / REPORT harness for standalone C++ test executables.
// Must end main() with REPORT() or CHECK failures pass silently.
// Mirrors the pattern used in qec_library/cpp/tests/check.hpp.
#include <cstdio>
#include <string>

namespace check { inline int& failures() { static int f = 0; return f; } }

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            ++check::failures();                                               \
        }                                                                      \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        auto _a = (a);                                                         \
        auto _b = (b);                                                         \
        if (!(_a == _b)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s == %s (%lld vs %lld)\n",     \
                         __FILE__, __LINE__, #a, #b,                           \
                         (long long)_a, (long long)_b);                        \
            ++check::failures();                                               \
        }                                                                      \
    } while (0)

#define RUN(fn)                                                                \
    do {                                                                       \
        std::fprintf(stderr, "[run] %s\n", #fn);                               \
        fn();                                                                  \
    } while (0)

#define REPORT()                                                               \
    do {                                                                       \
        if (check::failures()) {                                               \
            std::fprintf(stderr, "%d FAILURE(S)\n", check::failures());        \
            return 1;                                                          \
        }                                                                      \
        std::fprintf(stderr, "OK\n");                                          \
        return 0;                                                              \
    } while (0)
