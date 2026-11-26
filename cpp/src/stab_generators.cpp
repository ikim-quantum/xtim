#include "qeccore/stab_generators.hpp"

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>

#include "qeccore/stab_affine.hpp"
#include "qeccore/stab_overlap.hpp"

namespace qeccore {

namespace {

// ---- Prepared phase pin -----------------------------------------------------
// The amplitude-ratio ORACLE (<psi|g|psi> read as (g|psi>)(b) / psi(b) via
// AffineState::amplitude_at_bits; formerly pin_phase here, removed as dead code) is exact
// and self-contained, but each call re-derives a fresh solve of R y = rhs (a dense
// byte-matrix Gaussian) — O(n·k) materialization + solve PER GENERATOR, the dominant reference-compile cost on wide
// deferred circuits (d=11 memory: n = k = 2316, 2n pins ≈ 47 s). stabilizer_generators
// pins every generator against the SAME state psi and has already paid for its
// PreparedAffine: G_θ with G_θ·R = I_k and H_θ with im(R) = ker(H_θ). That factors
// each pin into one packed solve + one ℤ₄ form evaluation:
//   a0 = ⟨b|ψ⟩    = ω_ψ · 2^{−k/2}                       (rhs = 0 ⇒ y = 0, Q(0) = 0)
//   a1 = ⟨b|g·ψ⟩  = ω_gψ · i^{Q'(y)} · 2^{−k/2},   y = G_θ·x the unique sol of R y = x
// where x = b ⊕ b' is the Pauli's X-translation. prepare() succeeding proves R has
// full column rank, so amplitude_at_bits' kernel is empty and its gauss_sum reduces
// to ζ8^{2·Q'(y)} — the formulas above are EXACTLY that path specialized to r = 0
// (cpp/tests/test_prepared_affine.cpp keeps the dense prepare and the amplitude_at_bits a0 as the
// oracle and requires the generators bit for bit). a0 is g-independent and computed ONCE.
struct PinPrepared {
    const AffineState& psi;
    const PreparedAffine& pa;
    AffineState scratch;           // = psi; per-pin restore of the Pauli-touched entries
    ExactPhase a0;
    std::vector<uint64_t> xs, yw;  // rhs x in slot space (k bits) / solution (k bits); zero outside the words
                                   // the previous pin touched (xswords / ysupp), cleared there at the next pin
    std::vector<int> ysupp;
    std::vector<int> xswords;      // the nonzero words of xs

    // a0 = amplitude_at_bits(b) without the dense solve: rhs = 0 has the unique solution y = 0
    // (prepare() proved full column rank), so the oracle's Gauss sum is the r = 0 one, ζ8^0.
    static ExactPhase amplitude_at_b(const AffineState& s) {
        ExactPhase r = s.omega.mul(ExactPhase::zeta8(0));
        if (r.is_zero) return ExactPhase::zero();
        r.scale -= s.k_;
        return r;
    }

    PinPrepared(const AffineState& s, const PreparedAffine& p)
        : psi(s), pa(p), scratch(s), a0(amplitude_at_b(s)),
          xs((size_t)p.kw_, 0), yw(((size_t)s.k_ + 63) / 64, 0) {}

    // g's supports are xq (its X-qubits) and zq (its Z-qubits), ascending, as the emitter recorded them while it
    // built g (QW-F): every step below walks them instead of g's n-wide words, so a pin costs O(weight) plus the
    // slot-space work, never O(n/64).
    void pin(Pauli& g, const std::vector<int>& xq, const std::vector<int>& zq) {
        // gψ = g|ψ⟩: a Pauli apply touches ONLY {b at its X-qubits, D at the R-row supports of its
        // Z-qubits, omega} — restored below from psi; scratch's R and J are byte-owned copies of
        // psi's that no pin ever mutates. Pauli::apply_to's order: ascending q, every Z before every X.
        for (int q : zq) scratch.apply_z(q);
        for (int q : xq) scratch.apply_x(q);
        if (g.phase % 4 != 0) scratch.omega = scratch.omega.mul(ExactPhase::zeta8((2 * g.phase) % 8));
        // rhs x = b ⊕ b' — exactly g's X-part (apply_x flips b[q] at each X-qubit; apply_z and the
        // phase never touch b). On-support ⟺ H_θ·x = 0 (im(R) = ker(H_θ)) — the exact condition
        // under which the oracle's gf2_solve finds a solution. H_θ row i · x = x[hq[i]] ⊕ (slot
        // part · x restricted to the pivot qubits), so x is gathered into slot space once.
        for (int t : xswords) xs[(size_t)t] = 0;
        xswords.clear();
        for (int q : xq) {
            const int s = pa.slot_of[q];
            if (s < 0) continue;
            if (!xs[(size_t)(s >> 6)]) xswords.push_back(s >> 6);
            xs[(size_t)(s >> 6)] |= 1ull << (s & 63);
        }
        // A Z-type candidate (x = 0) has every parity 0: nothing to check or solve.
        if (!xq.empty())
            for (int i = 0; i < pa.m_; ++i) {
                const uint64_t* hr = pa.Hrow(i);
                uint64_t acc = 0;
                for (int t : xswords) acc ^= hr[t] & xs[(size_t)t];
                if (__builtin_parityll(acc) ^ (int)g.xbit(pa.hq[i]))
                    throw std::logic_error(
                        "stabilizer_generators: candidate has wrong X/Z support "
                        "(rhs off the state's affine support): n=" + std::to_string(psi.n_) +
                        " k=" + std::to_string(psi.k_) + " check_row=" + std::to_string(i));
            }
        // y = G_θ·x (the unique solution — full column rank); G_θ lives on the pivot qubits.
        for (int j : ysupp) yw[(size_t)(j >> 6)] = 0;
        ysupp.clear();
        if (!xswords.empty())
            for (int j = 0; j < psi.k_; ++j) {
                const uint64_t* gr = pa.Grow(j);
                uint64_t acc = 0;
                for (int t : xswords) acc ^= gr[t] & xs[(size_t)t];
                if (__builtin_parityll(acc)) { yw[j >> 6] |= 1ull << (j & 63); ysupp.push_back(j); }
            }
        // c0 = Q'(y) = Σ_{j∈supp(y)} D'[j] + 2·#{J-edges inside supp(y)}   (mod 4).
        // Σ_a popcount(J_a & y) over the support counts every edge twice = the 2·#edges
        // term directly. J bits beyond dim are zero, so truncating to yw's words is exact.
        long long q4 = 0;
        for (int j : ysupp) q4 += scratch.D[j];
        const int tw = std::min(psi.J.words(), (int)yw.size());
        for (int a : ysupp) {
            const uint64_t* jr = psi.J.row(a);
            for (int t = 0; t < tw; ++t) q4 += __builtin_popcountll(jr[t] & yw[t]);
        }
        const int c0 = (int)(((q4 % 4) + 4) % 4);
        // a1 = ω_gψ · ζ8^{2c0}, scale −k — amplitude_at_bits' r = 0 result verbatim.
        ExactPhase a1 = scratch.omega;
        a1.add_z8((2 * c0) % 8);
        a1.scale -= psi.k_;
        // Restore the entries the apply touched (a throw above abandons scratch with its owner).
        scratch.omega = psi.omega;
        for (int q : xq) scratch.b[q] = psi.b[q];
        for (int q : zq) {
            const uint64_t* r = psi.R.row(q);
            for (int u = 0; u < psi.R.words_; ++u)
                for (uint64_t v = r[u]; v; v &= v - 1) {
                    const int j = (u << 6) + __builtin_ctzll(v);
                    scratch.D[j] = psi.D[j];
                }
        }
        // Identical accept/reject arithmetic to the oracle.
        if (a0.is_zero || a1.is_zero || a0.scale != a1.scale)
            throw std::logic_error(
                "stabilizer_generators: candidate has wrong X/Z support "
                "(|<psi|g|psi>| != 1): n=" + std::to_string(psi.n_) +
                " k=" + std::to_string(psi.k_) +
                " a0=(zero=" + std::to_string(a0.is_zero) +
                ",scale=" + std::to_string(a0.scale) +
                ") a1=(zero=" + std::to_string(a1.is_zero) +
                ",scale=" + std::to_string(a1.scale) + ")");
        const int dz = ((a1.z8 - a0.z8) % 8 + 8) % 8;
        if (dz & 1)
            throw std::logic_error(
                "stabilizer_generators: candidate eigenphase is not a 4th root "
                "(zeta8 exponent " + std::to_string(dz) + ")");
        g.phase = (4 - dz / 2) % 4;
    }
};

// The generator synthesis shared by the dense and the sparse output (S2-T3): `sink.begin()` hands out the Pauli
// to fill (all-zero, phase 0), `sink.done(g, xq, zq)` takes it after its pin, with its ascending X / Z supports — the
// same bits and phases in the same order. QW-F: the supports are recorded as the bits are set (a bit already set is
// not recorded twice, the OR of setx/setz), so no step scans g's n-wide words.
template <class Sink>
void emit_stabilizer_generators(const StabState& s, Sink& sink) {
    const AffineState* aff = dynamic_cast<const AffineState*>(&s);
    if (!aff)
        throw std::logic_error(
            "stabilizer_generators: only the affine representation is "
            "supported");

    const int n = aff->n_;
    const int k = aff->k_;
    PreparedAffine pa = PreparedAffine::prepare(*aff);
    PinPrepared pin(*aff, pa);   // shares pa + the state across all n pins
    std::vector<int> xq, zq;     // the current generator's X / Z supports
    auto addx = [&](Pauli& g, int q) { if (!g.xbit(q)) { g.setx(q); xq.push_back(q); } };
    auto addz = [&](Pauli& g, int q) { if (!g.zbit(q)) { g.setz(q); zq.push_back(q); } };

    // ---- Z-type generators: the n−k parity-check (left null-space) rows of R.
    // Each H_θ row is a Z-support: qubit hq[i] plus the pivot qubits of its slot bits; X-part is 0.
    auto for_bits = [](const uint64_t* r, int words, auto&& f) {
        for (int t = 0; t < words; ++t)
            for (uint64_t w = r[t]; w; w &= w - 1) f((t << 6) + __builtin_ctzll(w));
    };
    for (int i = 0; i < pa.m_; ++i) {
        Pauli& g = sink.begin();
        xq.clear();
        zq.clear();
        addz(g, pa.hq[i]);
        for_bits(pa.Hrow(i), pa.kw_, [&](int s) { addz(g, pa.piv[s]); });
        std::sort(zq.begin(), zq.end());
        pin.pin(g, xq, zq);
        sink.done(g, xq, zq);
    }

    // ---- X-type generators: one per free column l of R.
    // Applying X^{a_l} (a_l = column l of R) shifts y → y⊕e_l, changing the
    // amplitude phase by  Q(y⊕e_l) − Q(y) ≡ D_l + 2·(Σ_j J(l,j) y_j − D_l y_l)
    // (mod 4).  The constant D_l is a global phase (pinned empirically). The
    // y-dependent part is 2·c·y, c = J_l ⊕ (D_l mod 2)·e_l, which a Z-operator
    // Z^{z_l} cancels iff (R^T z_l)·y ≡ c·y, i.e. R^T z_l = c.  Since
    // G_θ R = I_k ⇒ R^T G_θ^T = I_k, a solution is z_l = G_θ^T·c, i.e.
    //   z_l = (XOR over j with J(l,j)=1 of Grow(j)) ⊕ (D_l odd ? Grow(l) : 0),
    // accumulated in slot space and scattered to the pivot qubits.
    // The columns of R as qubit lists (one pass over R's set bits, ascending q per column).
    std::vector<int> cstart((size_t)k + 1, 0), crow;
    for (int q = 0; q < n; ++q) for_bits(aff->R.row(q), aff->R.words_, [&](int j) { ++cstart[(size_t)j + 1]; });
    for (int j = 0; j < k; ++j) cstart[(size_t)j + 1] += cstart[(size_t)j];
    crow.resize((size_t)cstart[(size_t)k]);
    {
        std::vector<int> pos(cstart.begin(), cstart.end() - 1);
        for (int q = 0; q < n; ++q) for_bits(aff->R.row(q), aff->R.words_, [&](int j) { crow[(size_t)pos[(size_t)j]++] = q; });
    }
    std::vector<uint64_t> zacc((size_t)pa.kw_, 0);
    auto xor_grow = [&](int j) {
        const uint64_t* gr = pa.Grow(j);
        for (int w = 0; w < pa.kw_; ++w) zacc[(size_t)w] ^= gr[w];
    };
    for (int l = 0; l < k; ++l) {
        Pauli& g = sink.begin();
        xq.clear();
        zq.clear();
        // X-part = column l of R (crow is ascending per column).
        for (int c = cstart[(size_t)l]; c < cstart[(size_t)l + 1]; ++c) addx(g, crow[(size_t)c]);
        // Z-part = G_θ^T · (J row l ⊕ (D_l mod 2)·e_l).
        std::fill(zacc.begin(), zacc.end(), 0);
        if (aff->J.words() > 0) for_bits(aff->J.row(l), aff->J.words(), [&](int j) { xor_grow(j); });
        if (aff->D[l] & 1) xor_grow(l);
        for_bits(zacc.data(), pa.kw_, [&](int s) { addz(g, pa.piv[s]); });
        std::sort(zq.begin(), zq.end());
        pin.pin(g, xq, zq);
        sink.done(g, xq, zq);
    }
}

}  // namespace

std::vector<Pauli> stabilizer_generators(const StabState& s) {
    struct DenseSink {
        int n;
        std::vector<Pauli> gens;
        Pauli& begin() { gens.emplace_back(n); return gens.back(); }
        void done(Pauli&, const std::vector<int>&, const std::vector<int>&) {}
    } sink{s.n(), {}};
    sink.gens.reserve((size_t)sink.n);
    emit_stabilizer_generators(s, sink);
    return std::move(sink.gens);
}

std::vector<SparsePauli> stabilizer_generators_sparse(const StabState& s) {
    // One dense scratch Pauli, cleared over its set bits after each generator; the SparsePauli is the emitter's
    // recorded supports (= from_dense of the scratch): O(weight) per generator beyond the pin, and no n-word row per
    // generator is kept or scanned.
    struct SparseSink {
        Pauli g;
        std::vector<SparsePauli> gens;
        Pauli& begin() { g.phase = 0; return g; }
        void done(Pauli& p, const std::vector<int>& xq, const std::vector<int>& zq) {
            SparsePauli sp;
            sp.n = p.n;
            sp.phase = p.phase;
            sp.x.assign(xq.begin(), xq.end());
            sp.z.assign(zq.begin(), zq.end());
            for (int q : xq) p.x[(size_t)(q >> 6)] = 0;
            for (int q : zq) p.z[(size_t)(q >> 6)] = 0;
            p.phase = 0;
            gens.push_back(std::move(sp));
        }
    } sink{Pauli(s.n()), {}};
    sink.gens.reserve((size_t)s.n());
    emit_stabilizer_generators(s, sink);
    return std::move(sink.gens);
}

}  // namespace qeccore
