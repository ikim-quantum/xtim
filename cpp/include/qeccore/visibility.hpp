// QECCORE_SHARED_TYPE marks a class whose C++ type identity must be the same in every extension module of the
// process. The bindings are built with -fvisibility=hidden (pybind11's default); a hidden exception class gets
// separate type info in _xtim and in _xtim_testapi, and on macOS an exception thrown in one module is then not
// caught by a handler compiled into the other: the process-wide translator _xtim registers falls through to
// pybind11's generic mapping (RuntimeError / OverflowError). Default visibility makes the type info one shared
// symbol. Used for the exception classes that translator catches.
#ifndef QECCORE_VISIBILITY_HPP
#define QECCORE_VISIBILITY_HPP

#if defined(__GNUC__) || defined(__clang__)
#define QECCORE_SHARED_TYPE __attribute__((visibility("default")))
#else
#define QECCORE_SHARED_TYPE
#endif

#endif
