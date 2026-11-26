// test_amp_tiny.cpp — amp_is_tiny(z) == (std::abs(z) < 1e-15) for every z: random magnitudes across the whole
// double range, the decision band around 1e-15 (both components, every ulp neighbour), signed zeros, subnormals,
// infinities and NaNs.
#include "check.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <vector>

#include "qeccore/amplitudes.hpp"

using namespace qeccore;

namespace {
long n = 0;
void one(double re, double im) {
    const cd z(re, im);
    const bool a = amp_is_tiny(z), b = std::abs(z) < 1e-15;
    CHECK(a == b);
    if (a != b) std::fprintf(stderr, "  mismatch at (%.17g, %.17g)\n", re, im);
    ++n;
}
void test_all() {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const double specials[] = {0.0, -0.0, 1e-15, -1e-15, 2e-15, 5e-16, 7.07e-16, 7.08e-16, 1e-15 / std::sqrt(2.0),
                               std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(),
                               std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
    for (double a : specials) for (double b : specials) one(a, b);
    for (int t = 0; t < 200000; ++t) {
        const double e = -330.0 + 660.0 * (double)(rng() % 100000) / 100000.0;   // magnitude 10^e over the double range
        one(u(rng) * std::pow(10.0, e), u(rng) * std::pow(10.0, e));
    }
    for (int t = 0; t < 400000; ++t) {                                    // the band: up to 1e-14, every mix
        const double a = u(rng) * 1e-14, b = u(rng) * 1e-14;
        one(a, b); one(a * 0.1, b); one(a, b * 0.1);
    }
    for (int k = -64; k <= 64; ++k) {                                      // ulp neighbours of the threshold
        double x = 1e-15;
        for (int i = 0; i < std::abs(k); ++i) x = std::nextafter(x, k > 0 ? 2.0 : 0.0);
        one(x, 0.0); one(0.0, x); one(x, x / 2); one(x * 0.70710678118654752, x * 0.70710678118654752);
        one(x * 0.5, x * 0.8660254037844386);
    }
    std::fprintf(stderr, "[amp_tiny] %ld values\n", n);
}
}  // namespace

int main() {
    RUN(test_all);
    REPORT();
}
