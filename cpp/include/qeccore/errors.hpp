#pragma once
#include <stdexcept>
#include <string>

namespace qeccore {
struct QecError : std::runtime_error { using std::runtime_error::runtime_error; };
// Mirrors qec_library.core.errors.AdapterError. Cap breaches and "loud-fail"
// conditions throw this with a message whose key phrase matches the Python
// (e.g. "exceeded", "enumeration depth exceeded", "k=0").
struct AdapterError : QecError { using QecError::QecError; };
}  // namespace qeccore
