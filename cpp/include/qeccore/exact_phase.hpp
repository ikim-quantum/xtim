#pragma once
#include <complex>
#include <cmath>

namespace qeccore {

// Exact stabilizer amplitude: 0, or 2^(scale/2) · ζ₈^z8, ζ₈ = e^{iπ/4}.
struct ExactPhase {
    bool is_zero = false;
    int  scale   = 0;          // power of √2
    int  z8      = 0;          // exponent of ζ₈, kept in [0,8)

    static ExactPhase zero()           { return {true, 0, 0}; }
    static ExactPhase one()            { return {false, 0, 0}; }
    static ExactPhase zeta8(int k)     { return {false, 0, ((k % 8) + 8) % 8}; }

    ExactPhase mul(const ExactPhase& o) const {
        if (is_zero || o.is_zero) return zero();
        return {false, scale + o.scale, ((z8 + o.z8) % 8 + 8) % 8};
    }
    ExactPhase conj() const {
        if (is_zero) return zero();
        return {false, scale, ((-z8) % 8 + 8) % 8};
    }
    ExactPhase mul_inv_sqrt2() const {
        if (is_zero) return zero();
        return {false, scale - 1, z8};
    }
    // Intentional in-place mutator (unlike the value-returning ops above): gate
    // updates fold a phase directly into a member `omega`, e.g. omega.add_z8(2).
    void add_z8(int k) { if (!is_zero) z8 = ((z8 + k) % 8 + 8) % 8; }

    std::complex<double> to_complex() const {
        if (is_zero) return {0.0, 0.0};
        double mag = std::pow(2.0, scale / 2.0);
        return std::polar(mag, M_PI * z8 / 4.0);
    }
    bool exact_eq(const ExactPhase& o) const {
        if (is_zero || o.is_zero) return is_zero && o.is_zero;
        return scale == o.scale && z8 == o.z8;
    }
};

}  // namespace qeccore
