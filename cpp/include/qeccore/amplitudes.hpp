#pragma once
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace qeccore {

using cd = std::complex<double>;

// ---------------------------------------------------------------------------
// Abstract interface
// ---------------------------------------------------------------------------
struct Amplitudes {
    virtual ~Amplitudes() = default;

    virtual int k() const = 0;                                   // number of logical indices
    virtual std::unique_ptr<Amplitudes> clone() const = 0;

    // Deep-copy `other` into this instance IN PLACE (per-shot buffer reuse: no realloc when
    // the existing buffers already fit). Backends must match; throws on mismatch.
    virtual void assign_from(const Amplitudes& other) = 0;

    // Read α_x (returns 0 if x is absent from support).
    virtual cd at(const std::vector<uint8_t>& x) const = 0;

    // Number of stored nonzero entries (the "χ").
    virtual int support() const = 0;

    // shift: α_x  <-  α_{x XOR y}   (relabel keys)
    virtual void shift(const std::vector<uint8_t>& y) = 0;

    // diagonal: α_x  <-  g0 * Π_i g[i]^{x_i} * α_x
    virtual void apply_diagonal(cd g0, const std::vector<cd>& g) = 0;

    // axpy: this  <-  this + c * other   (other must have the same k)
    virtual void axpy(cd c, const Amplitudes& other) = 0;

    // inner product: Σ_x  conj(this_x) * other_x
    virtual cd   inner(const Amplitudes& other) const = 0;

    // ||this||^2 = inner(this).real()
    virtual double norm2() const = 0;

    // scale: α_x  <-  c * α_x
    virtual void scale(cd c) = 0;

    // normalize: divide by sqrt(norm2)
    virtual void normalize() = 0;

    // drop_index(i): remove index i (must be constant across all entries first)
    virtual void drop_index(int i) = 0;

    // add_index: append a fresh index; α_{x,0} = α_x, α_{x,1} = 0
    virtual void add_index() = 0;
};

// ---------------------------------------------------------------------------
// Dense (sparse key-list) backend
// ---------------------------------------------------------------------------
struct DenseAmplitudes final : Amplitudes {
    int k_;
    // Sparse list of (bitstring, coefficient) pairs. Keys are unique; only
    // nonzero entries are stored.
    std::vector<std::pair<std::vector<uint8_t>, cd>> b_;

    explicit DenseAmplitudes(int k) : k_(k) {}

    int k() const override { return k_; }

    std::unique_ptr<Amplitudes> clone() const override {
        auto copy = std::make_unique<DenseAmplitudes>(k_);
        copy->b_ = b_;
        return copy;
    }

    void assign_from(const Amplitudes& other) override {
        const auto* od = dynamic_cast<const DenseAmplitudes*>(&other);
        if (!od) throw std::runtime_error("assign_from: backend mismatch");
        k_ = od->k_;
        b_ = od->b_;   // vector copy-assign: reuses this instance's buffers where they fit
    }

    cd at(const std::vector<uint8_t>& x) const override {
        for (auto& [key, val] : b_) {
            if (key == x) return val;
        }
        return cd(0, 0);
    }

    int support() const override {
        return static_cast<int>(b_.size());
    }

    void shift(const std::vector<uint8_t>& y) override {
        // α_x <- α_{x XOR y}: relabel every key x -> x XOR y
        for (auto& [key, val] : b_) {
            for (int i = 0; i < k_; ++i) {
                key[i] ^= y[i];
            }
        }
    }

    void apply_diagonal(cd g0, const std::vector<cd>& g) override {
        for (auto& [key, val] : b_) {
            cd factor = g0;
            for (int i = 0; i < k_; ++i) {
                if (key[i]) {
                    // x_i = 1: multiply by g[i]^1 = g[i]
                    factor *= g[i];
                }
                // x_i = 0: g[i]^0 = 1, no multiply
            }
            val *= factor;
        }
        prune();
    }

    void axpy(cd c, const Amplitudes& other) override {
        // Iterate over other's entries and merge into b_
        const auto* od = dynamic_cast<const DenseAmplitudes*>(&other);
        if (!od) throw std::runtime_error("axpy: backend mismatch");
        for (auto& [okey, oval] : od->b_) {
            cd contribution = c * oval;
            bool found = false;
            for (auto& [key, val] : b_) {
                if (key == okey) {
                    val += contribution;
                    found = true;
                    break;
                }
            }
            if (!found) {
                b_.emplace_back(okey, contribution);
            }
        }
        prune();
    }

    cd inner(const Amplitudes& other) const override {
        const auto* od = dynamic_cast<const DenseAmplitudes*>(&other);
        if (!od) throw std::runtime_error("inner: backend mismatch");
        cd result(0, 0);
        for (auto& [okey, oval] : od->b_) {
            for (auto& [key, val] : b_) {
                if (key == okey) {
                    result += std::conj(val) * oval;
                    break;
                }
            }
        }
        return result;
    }

    double norm2() const override {
        double s = 0.0;
        for (auto& [key, val] : b_) {
            s += std::norm(val);   // std::norm(z) = |z|^2
        }
        return s;
    }

    void scale(cd c) override {
        if (std::abs(c) < 1e-15) {
            b_.clear();
            return;
        }
        for (auto& [key, val] : b_) {
            val *= c;
        }
        prune();
    }

    void normalize() override {
        double n2 = norm2();
        if (n2 < 1e-30) return;  // avoid dividing by ~zero
        double inv = 1.0 / std::sqrt(n2);
        for (auto& [key, val] : b_) {
            val *= inv;
        }
    }

    void drop_index(int i) override {
        // Check that bit i is the same across all entries (survives Release builds)
        if (!b_.empty()) {
            uint8_t ref_bit = b_[0].first[i];
            for (auto& entry : b_) {
                if (entry.first[i] != ref_bit)
                    throw std::runtime_error("drop_index: index not constant across support");
            }
        }
        // Remove byte at position i from every key
        for (auto& [key, val] : b_) {
            key.erase(key.begin() + i);
        }
        k_--;
    }

    void add_index() override {
        // Append a 0 byte to every key
        for (auto& [key, val] : b_) {
            key.push_back(0);
        }
        k_++;
    }

private:
    // Remove entries whose coefficient is effectively zero.
    void prune() {
        auto it = b_.begin();
        while (it != b_.end()) {
            if (std::abs(it->second) < 1e-15) {
                it = b_.erase(it);
            } else {
                ++it;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

inline std::unique_ptr<Amplitudes> make_dense(
    int k,
    const std::vector<std::pair<std::vector<uint8_t>, cd>>& branches)
{
    auto d = std::make_unique<DenseAmplitudes>(k);
    for (auto& [key, val] : branches) {
        if (std::abs(val) < 1e-15) continue;
        bool found = false;
        for (auto& [ekey, eval] : d->b_) {
            if (ekey == key) {
                eval += val;
                found = true;
                break;
            }
        }
        if (!found) {
            d->b_.emplace_back(key, val);
        }
    }
    // prune any keys whose merged coefficient is now ~zero
    auto it = d->b_.begin();
    while (it != d->b_.end()) {
        if (std::abs(it->second) < 1e-15) it = d->b_.erase(it);
        else ++it;
    }
    return d;
}

inline std::unique_ptr<Amplitudes> make_dense_basis0(int k) {
    auto d = std::make_unique<DenseAmplitudes>(k);
    std::vector<uint8_t> zero(k, 0);
    d->b_.emplace_back(zero, cd(1.0, 0.0));
    return d;
}

}  // namespace qeccore
