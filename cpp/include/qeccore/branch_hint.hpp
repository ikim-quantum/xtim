#pragma once
// Branch-prediction hints. GCC/Clang: __builtin_expect (codegen unchanged); MSVC: the plain expression.
#if defined(__GNUC__) || defined(__clang__)
#define XTIM_LIKELY(x)   __builtin_expect(!!(x), 1)
#define XTIM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define XTIM_LIKELY(x)   (x)
#define XTIM_UNLIKELY(x) (x)
#endif
