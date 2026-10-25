// framed_py.hpp — J-b1 F (plan r3.3): PyFramedSuperposition (the FramedSuperposition handle) and its binding,
// shared by xtim._xtim and xtim._xtim_testapi (the legacy sampler's materialize returns it). Registered
// py::module_local() in each extension, so the two modules never share the type. Moved verbatim from xtim_py.cpp;
// (J-b2 J8: the sigma port law, port_sigma_law, is deleted with the sigma law.)
#pragma once

#include "common_py.hpp"
#include "injection.hpp"
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <complex>
#include "qeccore/framed_superposition.hpp"
#include "qeccore/framed_sampler.hpp"
#include "qeccore/ref_io.hpp"
#include "qeccore/port_contract.hpp"

namespace {

struct PyFramedSuperposition {
    FramedSuperposition state;
    qeccore::ScratchOwner scratch_owner_;   // Python-held owner (automatic scratch release)
    std::string budget_invalid_;            // set by a mid-operation budget refusal
    static constexpr const char* kBudgetWhat = "FramedSuperposition";
    explicit PyFramedSuperposition(const FramedSuperposition& s) : state(s) {}
    explicit PyFramedSuperposition(FramedSuperposition&& s) : state(std::move(s)) {}

    int n() const { return state.n(); }
    int k() const { return state.k(); }

    // |<self|other>| ≈ 1  ⟺  states equal up to global phase.
    bool approx_equal(const PyFramedSuperposition& other, double tol = 1e-12) const {
        auto ov = exact_sum_overlap(state, other.state);
        return std::abs(std::abs(ov) - 1.0) < tol;
    }

    // 3.1.0: the overlap VALUE |<self|other>|^2 (global-phase insensitive), the same exact
    // exact_sum_overlap that approx_equal thresholds. Both states must live on the same
    // qubit set (same n); a mismatch is a caller error, refused loudly. Consumers: per-shot
    // Choi-state fidelity estimation (a stabilizer-superposition inner product per shot).
    double overlap(const PyFramedSuperposition& other) const {
        if (state.n() != other.state.n())
            throw std::invalid_argument(
                "FramedSuperposition.overlap: states have different qubit counts (" +
                std::to_string(state.n()) + " vs " + std::to_string(other.state.n()) +
                "); overlaps are defined on one qubit set");
        auto ov = exact_sum_overlap(state, other.state);
        return std::norm(ov);
    }

    // In-place Clifford gate for mutation testing. kind: 0=H 1=S 2=Sdg 3=X 4=Y 5=Z
    // 6=CX 7=CZ. For single-qubit gates b is ignored (pass 0).
    void apply_clifford_py(uint8_t kind, int a, int b) {
        state.apply_clifford(kind, a, b);
    }

    // Task 2b.1: single-qubit Pauli expectations <X_q> / <Z_q> (real). Wraps the exact
    // FramedSuperposition::pauli_expectation (framed_expectation engine). Used by the Born-
    // decision correlation gate: the retained collapsed state must satisfy <X_0> = ±1.
    double pauli_expectation_x(int q) const {
        Pauli P(state.n()); P.setx(q);
        return state.pauli_expectation(P).real();
    }
    double pauli_expectation_z(int q) const {
        Pauli P(state.n()); P.setz(q);
        return state.pauli_expectation(P).real();
    }
    // Task 2b.2: <Y_q> (real). Y = i·XZ; matches measure_pauli's Y encoding. Used by the
    // ⊗-composition to classify a carried qubit's single-qubit stabilizer state (X/Y/Z axis).
    double pauli_expectation_y(int q) const {
        Pauli P(state.n()); P.setx(q); P.setz(q); P.phase = 1;   // Y = i·XZ
        return state.pauli_expectation(P).real();
    }
    // Diagnostic: <Π_i X_{xs[i]} · Π_j Z_{zs[j]}> (real part). For frame-checking.
    double pauli_expectation_xz(const std::vector<int>& xs, const std::vector<int>& zs) const {
        Pauli P(state.n());
        for (int q : xs) P.setx(q);
        for (int q : zs) P.setz(q);
        return state.pauli_expectation(P).real();
    }
    // 3.1.2 (branch frames, docs/branch_frames.md): export the state as
    //   |psi> = sum_i coeff[i] · D_i |ref>,
    // |ref> = the joint +1 eigenstate of the n SIGNED generators G_a = i^{stab_phase[a]}
    // X^{stab_x[a]} Z^{stab_z[a]} (frame row Zrow[a] with (-1)^{eps[a]} folded in — the
    // certified_stabilizers() convention extended to the free rows), and
    //   D_i = Π_{d ascending : sigma_i[d]} Xrow[free[d]]
    // formed LEFT-TO-RIGHT with the engine's Pauli product (pmul_into ≡ Pauli::multiply(acc,b)),
    // i.e. exactly the oW[i] of framed_measure_anticommuting_general. Read-only; O(chi·n) bit
    // copies on the already-materialized state. Every Pauli is over the STATE's n wires (the
    // indices pauli_expectation_x(q) addresses / TwirlSampler.output_wires() returns).
    py::dict branch_frames() const {
        const int N = state.n();
        const int K = state.k();
        const auto& brs = state.entries();
        const int chi = (int)brs.size();
        auto bits2d = [N](int rows) {
            py::array_t<uint8_t> a({rows, N});
            std::memset(a.mutable_data(), 0, (size_t)rows * (size_t)N);
            return a;
        };
        auto fill_row = [N](py::array_t<uint8_t>& ax, py::array_t<uint8_t>& az, int r,
                            const Pauli& p) {
            uint8_t* x = ax.mutable_data() + (size_t)r * (size_t)N;
            uint8_t* z = az.mutable_data() + (size_t)r * (size_t)N;
            for (int q = 0; q < N; ++q) { x[q] = p.xbit(q) ? 1 : 0; z[q] = p.zbit(q) ? 1 : 0; }
        };
        py::array_t<uint8_t> sx = bits2d(N), sz = bits2d(N), dx = bits2d(N), dz = bits2d(N);
        py::array_t<int8_t> sp(N), dp(N);
        py::array_t<uint8_t> eps_a(N);
        for (int a = 0; a < N; ++a) {
            Pauli g = state.U.Zrow[a];
            if (state.eps[a]) g.phase = (g.phase + 2) & 3;      // ×(-1): the signed generator
            fill_row(sx, sz, a, g);
            sp.mutable_data()[a] = (int8_t)(g.phase & 3);
            const Pauli& d = state.U.Xrow[a];
            fill_row(dx, dz, a, d);
            dp.mutable_data()[a] = (int8_t)(d.phase & 3);
            eps_a.mutable_data()[a] = state.eps[a] ? 1 : 0;
        }
        py::array_t<int32_t> fr(K);
        for (int d = 0; d < K; ++d) fr.mutable_data()[d] = (int32_t)state.free[d];
        // χ-scaled arrays are charged to the state budget for as long as they are alive (3.1.10):
        // the whole export is pre-checked, then each array carries its own ticket.
        const char* what = "branch_frames export (numpy arrays)";
        qeccore::state_budget_precheck(
            (int64_t)chi * ((int64_t)K + 16 + 2 * (int64_t)N + 1), what);
        py::array_t<uint8_t> sig = budget_array<uint8_t>({(py::ssize_t)chi, (py::ssize_t)K}, what);
        py::array_t<std::complex<double>> co = budget_array<std::complex<double>>({(py::ssize_t)chi}, what);
        py::array_t<uint8_t> bx = budget_array<uint8_t>({(py::ssize_t)chi, (py::ssize_t)N}, what);
        py::array_t<uint8_t> bz = budget_array<uint8_t>({(py::ssize_t)chi, (py::ssize_t)N}, what);
        py::array_t<int8_t> bp = budget_array<int8_t>({(py::ssize_t)chi}, what);
        for (int i = 0; i < chi; ++i) {
            Pauli D(N);                                          // identity, phase 0
            for (int d = 0; d < K; ++d) {
                sig.mutable_data()[(size_t)i * (size_t)K + (size_t)d] = qeccore::key_bit(brs[i].first, d);
                if (qeccore::key_bit(brs[i].first, d)) D = Pauli::multiply(D, state.U.Xrow[state.free[d]]);
            }
            fill_row(bx, bz, i, D);
            bp.mutable_data()[i] = (int8_t)(D.phase & 3);
            co.mutable_data()[i] = brs[i].second;
        }
        py::dict out;
        out["n"] = N;
        out["free"] = fr;
        out["eps"] = eps_a;
        out["stab_x"] = sx; out["stab_z"] = sz; out["stab_phase"] = sp;
        out["destab_x"] = dx; out["destab_z"] = dz; out["destab_phase"] = dp;
        out["sigma"] = sig; out["coeff"] = co;
        out["branch_x"] = bx; out["branch_z"] = bz; out["branch_phase"] = bp;
        return out;
    }
    // decoder-feedback perf (BUCKET FOLD guard, audit §1/§4 "ρ_ports check", COMPLETE / magic-aware):
    // a CANONICAL fingerprint of this state's reduced density matrix on the carried port `wires`.
    // Two states return EQUAL bytes IFF ρ_port is identical (so injecting either representative gives
    // identical stage-k+1 physics — stage k+1 measures ONLY the port). run_protocol folds all record
    // buckets that share one port_signature into ONE compiled law; the fingerprint being COMPLETE is
    // what makes that sound (audit's stabilizer-only sketch was INSUFFICIENT — the Steane T-teleport
    // carries a MAGIC state whose logical frame ⟨Ȳ⟩=±1/√2 varies within a decision pattern; a
    // stabilizer-sign-only guard silently mis-folds it and fails the bp_osd ML-oracle gate at 7σ).
    //
    // The fingerprint has TWO parts:
    //  (1) STABILIZER part — the port-restricted +1 Hermitian stabilizer subgroup with signs, in a
    //      canonical RREF. certified_stabilizers() gives the n−k exact stabilizer generators; a
    //      symplectic Gauss-Jordan over the 2·|non-port| coordinates isolates the port subgroup, and
    //      a second Gauss-Jordan over the 2·|port| coordinates canonicalizes it (all phase algebra via
    //      Pauli::multiply, so each surviving generator's Hermitian sign s∈{±1} is exact).
    //  (2) MAGIC/LOGICAL part — the reduced logical subsystem (k_port = |port|−#stab generators). For
    //      k_port==0 the port is a pure stabilizer state → part (1) is already complete. For k_port==1
    //      the logical qubit's Bloch vector (⟨L_X⟩,⟨L_Z⟩,⟨L_X·L_Z⟩ for a CANONICAL logical pair
    //      derived from the shared stabilizer subgroup — the values are stabilizer-multiple invariant)
    //      pins the pure/mixed logical state exactly. For k_port≥2 the guard cannot cheaply certify a
    //      complete fingerprint, so it emits a FALLBACK flag (leading byte) that run_protocol treats
    //      as "un-foldable" → per-record path (never a silent fold of an un-foldable case).
    py::bytes port_signature(const std::vector<int>& wires) const {
        const int N = state.n();
        std::vector<uint8_t> in_port((size_t)(N > 0 ? N : 1), 0);
        for (int w : wires)
            if (w >= 0 && w < N) in_port[(size_t)w] = 1;

        std::vector<Pauli> rows = state.certified_stabilizers();

        // Gauss-Jordan eliminate a coordinate (qubit q, which: 0=x-bit, 1=z-bit) using rows
        // [pivot_start..): pick a pivot row with that bit set, clear it from every OTHER row via
        // Pauli::multiply (XOR of x/z bits + exact phase), swap the pivot to `pivot_start`, and
        // advance. Returns the new pivot_start (unchanged if no pivot found).
        auto eliminate = [&](int q, int which, int pivot_start) -> int {
            int piv = -1;
            for (int i = pivot_start; i < (int)rows.size(); ++i) {
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) { piv = i; break; }
            }
            if (piv < 0) return pivot_start;
            std::swap(rows[pivot_start], rows[piv]);
            for (int i = 0; i < (int)rows.size(); ++i) {
                if (i == pivot_start) continue;
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) rows[i] = Pauli::multiply(rows[i], rows[pivot_start]);
            }
            return pivot_start + 1;
        };

        // Phase 1: eliminate all non-port coordinates → rows[r..] are supported only on `wires`.
        int r = 0;
        for (int q = 0; q < N; ++q) {
            if (in_port[(size_t)q]) continue;
            r = eliminate(q, 0, r);
            r = eliminate(q, 1, r);
        }
        std::vector<Pauli> port(rows.begin() + r, rows.end());

        // Phase 2: canonical RREF of the port subgroup over the 2·|port| port coordinates.
        rows = std::move(port);
        int pr = 0;
        for (int w : wires) {
            if (w < 0 || w >= N) continue;
            pr = eliminate(w, 0, pr);
            pr = eliminate(w, 1, pr);
        }
        // Keep only the independent (non-identity) port generators.
        std::vector<Pauli> gens(rows.begin(), rows.begin() + pr);
        // Deterministic order: by (leading port coordinate). RREF from the fixed wire scan already
        // yields a unique echelon; sort by the packed port support to remove residual row order.
        auto packed = [&](const Pauli& g) {
            std::vector<uint8_t> key;
            key.reserve(2 * wires.size() + 1);
            for (int w : wires) key.push_back(g.xbit(w) ? 1 : 0);
            for (int w : wires) key.push_back(g.zbit(w) ? 1 : 0);
            // Hermitian sign s = i^{(phase − x·z) mod 4} ∈ {+1,−1} → 0/1.
            int d = ((g.phase - g.xz_overlap()) % 4 + 4) % 4;   // 0 or 2 for a +1 stabilizer
            key.push_back(d == 0 ? 0 : 1);
            return key;
        };
        std::vector<std::vector<uint8_t>> keys;
        keys.reserve(gens.size());
        for (const auto& g : gens) keys.push_back(packed(g));
        std::sort(keys.begin(), keys.end());

        const int W = (int)wires.size();
        const int ns = (int)gens.size();          // # independent port stabilizer generators
        const int k_port = W - ns;                // logical DOF carried on the port

        std::string out;
        out.reserve(keys.size() * (2 * W + 1) + 64);
        // Leading FLAG byte: 0/1 = complete fingerprint (foldable); ≥2 = k_port≥2 fallback marker
        // (run_protocol refuses to fold on it). Then port width + stabilizer-generator count so
        // different-width / different-rank states never collide.
        out.push_back((char)(k_port >= 2 ? 2 : k_port));
        // W header is a single byte.  W>64 already short-circuits to flag=2 above
        // so W>255 is unreachable today; if the fallback threshold is ever raised
        // past 255 this header must widen or port_signature_struct will misalign.
        out.push_back((char)(W & 0xFF));
        out.push_back((char)(ns & 0xFF));
        for (const auto& k : keys)
            out.append(reinterpret_cast<const char*>(k.data()), k.size());
        if (k_port == 0)
            return py::bytes(out);                 // port fully stabilized — stabilizer part is complete
        if (k_port >= 2 || W > 64)
            return py::bytes(out);                 // fallback: stabilizer part only + flag=2 (unfoldable)

        // ── MAGIC/LOGICAL part (k_port == 1): append the carried logical qubit's Bloch vector ──
        // A valid logical is a port Pauli in the NORMALIZER (commuting with every stabilizer) but not
        // in the stabilizer group.  We compute the normalizer as the GF(2) null space of the
        // symplectic-constraint matrix and pick the two independent (anticommuting) logicals.  ⟨L⟩ is
        // invariant under multiplying L by any +1 stabilizer, so the Bloch coordinates are canonical
        // given the (shared) stabilizer subgroup — comparable across buckets without a canonical L.
        // Symplectic coordinate: 2W bits, low W = x over wires, high W = z over wires.
        using u128 = unsigned __int128;
        auto to_sv = [&](const Pauli& g) -> u128 {         // Pauli → port symplectic vector
            u128 v = 0;
            for (int i = 0; i < W; ++i) {
                if (g.xbit(wires[i])) v |= (u128)1 << i;
                if (g.zbit(wires[i])) v |= (u128)1 << (W + i);
            }
            return v;
        };
        auto sp = [&](u128 a, u128 b) -> int {             // symplectic product a_x·b_z + a_z·b_x
            u128 axbz = (a & (((u128)1 << W) - 1)) & (b >> W);
            u128 azbx = (a >> W) & (b & (((u128)1 << W) - 1));
            u128 t = axbz ^ azbx;
            return (__builtin_popcountll((uint64_t)t) + __builtin_popcountll((uint64_t)(t >> 64))) & 1;
        };
        std::vector<u128> S;                               // stabilizer symplectic vectors
        S.reserve(gens.size());
        for (const auto& g : gens) S.push_back(to_sv(g));
        // Constraint rows: A_i · v = sp(v, S_i).  A_i = (S_i.z ‖ S_i.x) (swap halves) so that dotting
        // with v = (v_x ‖ v_z) gives v_x·S_i.z + v_z·S_i.x = sp(v,S_i).
        std::vector<u128> A;
        A.reserve(S.size());
        for (u128 s : S) {
            u128 lo = s & (((u128)1 << W) - 1), hi = s >> W;
            A.push_back(hi | (lo << W));
        }
        // Row-reduce A to echelon; record pivot columns.
        std::vector<int> pivcol;
        std::vector<u128> pivrow;
        for (u128 row : A) {
            for (size_t p = 0; p < pivrow.size(); ++p)
                if ((row >> pivcol[p]) & 1) row ^= pivrow[p];
            if (row == 0) continue;
            int lead = 0;
            for (int c = 0; c < 2 * W; ++c) if ((row >> c) & 1) { lead = c; break; }
            // clear this column from previous pivot rows (reduced echelon)
            for (size_t p = 0; p < pivrow.size(); ++p)
                if ((pivrow[p] >> lead) & 1) pivrow[p] ^= row;
            pivrow.push_back(row);
            pivcol.push_back(lead);
        }
        std::vector<char> is_piv(2 * W, 0);
        for (int c : pivcol) is_piv[c] = 1;
        // Null-space basis: one vector per free column (bit set there + back-substituted pivots).
        std::vector<u128> nullb;
        for (int fc = 0; fc < 2 * W; ++fc) {
            if (is_piv[fc]) continue;
            u128 v = (u128)1 << fc;
            for (size_t p = 0; p < pivrow.size(); ++p)
                if ((pivrow[p] >> fc) & 1) v |= (u128)1 << pivcol[p];
            nullb.push_back(v);
        }
        // Logical operators = null-space vectors NOT in the stabilizer span.  Reduce each by S; keep
        // the nonzero residues; take the first two independent (anticommuting) as L_a, L_b.
        // Build a reduced basis of S (for span membership tests).
        std::vector<u128> Sred, Spiv;
        for (u128 s : S) {
            int lead = -1; u128 r = s;
            for (size_t p = 0; p < Sred.size(); ++p) {
                int lp = 0; for (int c = 0; c < 2 * W; ++c) if ((Spiv[p] >> c) & 1) { lp = c; break; }
                if ((r >> lp) & 1) r ^= Sred[p];
            }
            if (r) { Sred.push_back(r); Spiv.push_back(r); }
        }
        auto in_span_S = [&](u128 v) -> bool {
            for (size_t p = 0; p < Sred.size(); ++p) {
                int lp = 0; for (int c = 0; c < 2 * W; ++c) if ((Sred[p] >> c) & 1) { lp = c; break; }
                if ((v >> lp) & 1) v ^= Sred[p];
            }
            return v == 0;
        };
        u128 La = 0, Lb = 0; bool haveA = false, haveB = false;
        for (u128 v : nullb) {
            if (in_span_S(v)) continue;
            if (!haveA) { La = v; haveA = true; }
            else if (sp(v, La)) { Lb = v; haveB = true; break; }
        }
        auto sv_to_pauli = [&](u128 v) -> Pauli {
            Pauli P(N);
            for (int i = 0; i < W; ++i) {
                if ((v >> i) & 1) P.setx(wires[i]);
                if ((v >> (W + i)) & 1) P.setz(wires[i]);
            }
            return P;
        };
        auto herm_exp = [&](u128 v) -> long long {
            // Hermitian expectation ⟨i^{x·z} X^x Z^z⟩ ∈ [−1,1]; quantize to 1e-9 (exact-value stable,
            // distinguishes ±1/√2 cleanly, never merges genuinely-different reduced states).
            Pauli H = sv_to_pauli(v);
            H.phase = H.xz_overlap() & 3;
            double e = state.pauli_expectation(H).real();
            return llround(e * 1e9);
        };
        auto append_i64 = [&](long long v) {
            for (int b = 0; b < 8; ++b) out.push_back((char)((v >> (8 * b)) & 0xFF));
        };
        if (haveA && haveB) {
            append_i64(herm_exp(La));
            append_i64(herm_exp(Lb));
            // The Bloch third coordinate: ⟨L_a·L_b⟩ (Hermitian) — its GF(2) support is La^Lb.
            append_i64(herm_exp(La ^ Lb));
        } else {
            out[0] = (char)2;                      // degenerate — be conservative (unfoldable)
        }
        return py::bytes(out);
    }

    // Task 6: structured accessor — one owner of the wire format.
    // Calls port_signature(wires) to get the canonical byte string and decodes it using
    // EXACTLY the same logic as fold.py's _parse_port_signature.  No second implementation
    // of Gauss-Jordan or Bloch derivation — the byte string is the single source of truth.
    //
    // Layout mirror of _parse_port_signature(b):
    //   b[0] = flag, b[1] = W, b[2] = ns
    //   then ns entries each (2*W + 1) bytes: first 2*W = x/z support; last 1 = Hermitian sign
    //   then remainder = little-endian signed int64s (Bloch coords)
    //     each v mapped to (abs(v), 1 if v < 0 else 0)
    //
    // Returns dict { flag:int, W:int, ns:int,
    //                keys:dict{bytes->int},
    //                blochs:list[list[int,int]] }
    py::dict port_signature_struct(const std::vector<int>& wires) const {
        // Delegate to the single authoritative emitter.
        py::bytes raw = port_signature(wires);
        std::string s = raw.cast<std::string>();
        const uint8_t* b = reinterpret_cast<const uint8_t*>(s.data());
        const size_t blen = s.size();

        int flag = (blen > 0) ? (int)b[0] : 0;
        int W    = (blen > 1) ? (int)b[1] : 0;
        int ns   = (blen > 2) ? (int)b[2] : 0;

        const int ksz = 2 * W + 1;
        size_t off = 3;

        // keys: dict mapping 2W-byte support bytes -> Hermitian sign int
        py::dict keys;
        for (int i = 0; i < ns; ++i) {
            if (off + (size_t)ksz > blen) break;
            py::bytes support(reinterpret_cast<const char*>(b + off), 2 * W);
            int sign = (int)b[off + 2 * W];
            keys[support] = sign;
            off += (size_t)ksz;
        }

        // blochs: list of [magnitude, sign] pairs (signed int64, little-endian)
        py::list blochs;
        const size_t tail_len = (blen > off) ? (blen - off) : 0;
        const size_t n_i64 = tail_len / 8;
        for (size_t j = 0; j < n_i64; ++j) {
            int64_t v = 0;
            for (int byte = 0; byte < 8; ++byte)
                v |= (int64_t)b[off + 8 * j + (size_t)byte] << (8 * byte);
            int64_t mag  = (v < 0) ? -v : v;
            int     sgn  = (v < 0) ? 1 : 0;
            py::list pair;
            pair.append((long long)mag);
            pair.append(sgn);
            blochs.append(pair);
        }

        py::dict res;
        res["flag"]   = flag;
        res["W"]      = W;
        res["ns"]     = ns;
        res["keys"]   = keys;
        res["blochs"] = blochs;
        return res;
    }

    // decoder-feedback perf (PAULI-ORBIT FOLD companion to port_signature): the CANONICAL port
    // symplectic OPERATORS whose signs the fingerprint records — the port stabilizer generators (in
    // the SAME sorted order as port_signature's stabilizer supports) plus the carried logical pair
    // (La, Lb) in the SAME order as the Bloch coordinates (⟨La⟩, ⟨Lb⟩, ⟨La·Lb⟩).  These are a PURE
    // FUNCTION of the stabilizer SUPPORT structure (signs never enter the canonicalization or the
    // null-space logical selection), so they are IDENTICAL for every state in one sign-stripped ORBIT
    // class — the constant GF(2) solve matrix run_protocol uses to recover, per shot, the residual
    // port-Pauli P_i whose anticommutation reproduces that shot's sign vector vs the orbit rep.  A
    // Pauli P anticommutes with (and hence flips the sign of) operator O iff sp(P,O)=1, so
    //     M · P = Δsigns   with M's rows = [stab_1 … stab_ns, La, Lb] symplectic vectors
    // and columns the 2W port-Pauli coordinates (low W = X over wires, high W = Z over wires) — the
    // SAME coordinate convention as input_pauli's frame slice.  Returns ``ok=False`` for the
    // un-foldable / degenerate cases (flag≥2, W>64) so the caller falls back to the per-record path.
    py::dict port_orbit_operators(const std::vector<int>& wires) const {
        py::dict res;
        const int N = state.n();
        const int W = (int)wires.size();
        res["W"] = W;
        if (W > 64) { res["ok"] = false; return res; }
        std::vector<uint8_t> in_port((size_t)(N > 0 ? N : 1), 0);
        for (int w : wires)
            if (w >= 0 && w < N) in_port[(size_t)w] = 1;

        std::vector<Pauli> rows = state.certified_stabilizers();
        auto eliminate = [&](int q, int which, int pivot_start) -> int {
            int piv = -1;
            for (int i = pivot_start; i < (int)rows.size(); ++i) {
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) { piv = i; break; }
            }
            if (piv < 0) return pivot_start;
            std::swap(rows[pivot_start], rows[piv]);
            for (int i = 0; i < (int)rows.size(); ++i) {
                if (i == pivot_start) continue;
                bool b = which == 0 ? rows[i].xbit(q) : rows[i].zbit(q);
                if (b) rows[i] = Pauli::multiply(rows[i], rows[pivot_start]);
            }
            return pivot_start + 1;
        };
        int r = 0;
        for (int q = 0; q < N; ++q) {
            if (in_port[(size_t)q]) continue;
            r = eliminate(q, 0, r);
            r = eliminate(q, 1, r);
        }
        std::vector<Pauli> port(rows.begin() + r, rows.end());
        rows = std::move(port);
        int pr = 0;
        for (int w : wires) {
            if (w < 0 || w >= N) continue;
            pr = eliminate(w, 0, pr);
            pr = eliminate(w, 1, pr);
        }
        std::vector<Pauli> gens(rows.begin(), rows.begin() + pr);
        // Emit the stabilizer symplectic vectors in the SAME sorted key order port_signature uses
        // (key = [x over wires ‖ z over wires ‖ sign]; sort strips residual row order).  We sort by
        // the (support-only) 2W-bit key — distinct stabilizers have distinct support so the sign byte
        // is never a tiebreaker (matches port_signature's std::sort on the full key).
        auto sv_bits = [&](const Pauli& g) {
            std::vector<uint8_t> v; v.reserve(2 * W);
            for (int w : wires) v.push_back(g.xbit(w) ? 1 : 0);
            for (int w : wires) v.push_back(g.zbit(w) ? 1 : 0);
            return v;
        };
        std::vector<std::vector<uint8_t>> stab_sv;
        stab_sv.reserve(gens.size());
        for (const auto& g : gens) stab_sv.push_back(sv_bits(g));
        std::sort(stab_sv.begin(), stab_sv.end());

        const int ns = (int)gens.size();
        const int k_port = W - ns;
        res["ns"] = ns;
        res["k_port"] = (k_port >= 2 ? 2 : k_port);
        if (k_port >= 2) { res["ok"] = false; return res; }

        py::list stab_list;
        for (auto& v : stab_sv) {
            py::array_t<uint8_t> a((py::ssize_t)v.size());
            std::memcpy(a.mutable_data(), v.data(), v.size());
            stab_list.append(a);
        }
        res["stab"] = stab_list;

        py::list log_list;
        if (k_port == 1) {
            // Recompute (La, Lb) EXACTLY as port_signature does — a pure function of the supports.
            using u128 = unsigned __int128;
            auto to_sv = [&](const Pauli& g) -> u128 {
                u128 v = 0;
                for (int i = 0; i < W; ++i) {
                    if (g.xbit(wires[i])) v |= (u128)1 << i;
                    if (g.zbit(wires[i])) v |= (u128)1 << (W + i);
                }
                return v;
            };
            auto sp = [&](u128 a, u128 b) -> int {
                u128 axbz = (a & (((u128)1 << W) - 1)) & (b >> W);
                u128 azbx = (a >> W) & (b & (((u128)1 << W) - 1));
                u128 t = axbz ^ azbx;
                return (__builtin_popcountll((uint64_t)t) + __builtin_popcountll((uint64_t)(t >> 64))) & 1;
            };
            std::vector<u128> S;
            for (const auto& g : gens) S.push_back(to_sv(g));
            std::vector<u128> A;
            for (u128 s : S) {
                u128 lo = s & (((u128)1 << W) - 1), hi = s >> W;
                A.push_back(hi | (lo << W));
            }
            std::vector<int> pivcol; std::vector<u128> pivrow;
            for (u128 row : A) {
                for (size_t p = 0; p < pivrow.size(); ++p)
                    if ((row >> pivcol[p]) & 1) row ^= pivrow[p];
                if (row == 0) continue;
                int lead = 0;
                for (int c = 0; c < 2 * W; ++c) if ((row >> c) & 1) { lead = c; break; }
                for (size_t p = 0; p < pivrow.size(); ++p)
                    if ((pivrow[p] >> lead) & 1) pivrow[p] ^= row;
                pivrow.push_back(row); pivcol.push_back(lead);
            }
            std::vector<char> is_piv(2 * W, 0);
            for (int c : pivcol) is_piv[c] = 1;
            std::vector<u128> nullb;
            for (int fc = 0; fc < 2 * W; ++fc) {
                if (is_piv[fc]) continue;
                u128 v = (u128)1 << fc;
                for (size_t p = 0; p < pivrow.size(); ++p)
                    if ((pivrow[p] >> fc) & 1) v |= (u128)1 << pivcol[p];
                nullb.push_back(v);
            }
            std::vector<u128> Sred, Spiv;
            for (u128 s : S) {
                u128 rr = s;
                for (size_t p = 0; p < Sred.size(); ++p) {
                    int lp = 0; for (int c = 0; c < 2 * W; ++c) if ((Spiv[p] >> c) & 1) { lp = c; break; }
                    if ((rr >> lp) & 1) rr ^= Sred[p];
                }
                if (rr) { Sred.push_back(rr); Spiv.push_back(rr); }
            }
            auto in_span_S = [&](u128 v) -> bool {
                for (size_t p = 0; p < Sred.size(); ++p) {
                    int lp = 0; for (int c = 0; c < 2 * W; ++c) if ((Sred[p] >> c) & 1) { lp = c; break; }
                    if ((v >> lp) & 1) v ^= Sred[p];
                }
                return v == 0;
            };
            u128 La = 0, Lb = 0; bool haveA = false, haveB = false;
            for (u128 v : nullb) {
                if (in_span_S(v)) continue;
                if (!haveA) { La = v; haveA = true; }
                else if (sp(v, La)) { Lb = v; haveB = true; break; }
            }
            if (!(haveA && haveB)) { res["ok"] = false; return res; }
            auto emit = [&](u128 v) {
                py::array_t<uint8_t> a((py::ssize_t)(2 * W));
                uint8_t* d = a.mutable_data();
                for (int i = 0; i < 2 * W; ++i) d[i] = (uint8_t)((v >> i) & 1);
                log_list.append(a);
            };
            emit(La); emit(Lb);
        }
        res["logicals"] = log_list;
        res["ok"] = true;
        return res;
    }

    // E3 (exact-residual arc, Task 3): PURE-DATA exposure of the state's certified stabilizer
    // generators as full n-qubit symplectic rows + exact phases.  Row b is
    // certified_stabilizers()[b] — the IDENTICAL order as the σ bit order of
    // BarrierBuffer.sigmas() (twirl_sampler.hpp "parallel to bare.certified_stabilizers()") and
    // as port_sigma_law's comb bit order.  Coordinate convention matches port_orbit_operators /
    // port_sigma_law rows, widened to all n qubits: low n = x bits, high n = z bits.  The
    // generator OPERATOR is  i^phase · X^x · Z^z  (Pauli rep, phase ∈ {0..3}); on the bare state
    // every row has +1 eigenvalue.  Consumers: the adaptq per-(check, plan) indefiniteness
    // classifier (GF(2) span basis + exact Pauli-product phase bookkeeping).  Returns dict:
    //   n:int, ngens:int, xz: uint8[ngens, 2n], phase: uint8[ngens]
    py::dict certified_symplectic() const {
        const int N = state.n();
        const std::vector<Pauli> gens = state.certified_stabilizers();
        const int ngens = (int)gens.size();
        py::array_t<uint8_t> xz({(py::ssize_t)ngens, (py::ssize_t)(2 * N)});
        py::array_t<uint8_t> ph((py::ssize_t)ngens);
        uint8_t* xd = xz.mutable_data();
        uint8_t* pd = ph.mutable_data();
        for (int b = 0; b < ngens; ++b) {
            const Pauli& g = gens[(size_t)b];
            uint8_t* row = xd + (size_t)b * (size_t)(2 * N);
            for (int q = 0; q < N; ++q) {
                row[q] = g.xbit(q) ? 1 : 0;
                row[N + q] = g.zbit(q) ? 1 : 0;
            }
            pd[b] = (uint8_t)(((g.phase % 4) + 4) % 4);
        }
        py::dict res;
        res["n"] = N;
        res["ngens"] = ngens;
        res["xz"] = xz;
        res["phase"] = ph;
        return res;
    }

    // Task 2b.2-pre: Born-measure a single-qubit Pauli (basis 0:X 1:Y 2:Z) on qubit q with
    // random u∈[0,1); returns ±1 and COLLAPSES the state (mutates in place). This is the exact
    // per-shot decision-operator read the engine performs internally (emit_decisions) exposed
    // for the frame-explicit acceptance oracle: on the physical bare, measuring the decision
    // operator collapses the survivors physically.
    int measure_pauli(int basis, int q, double u) {
        Pauli P(state.n());
        if (basis == 0) P.setx(q);
        else if (basis == 2) P.setz(q);
        else { P.setx(q); P.setz(q); P.phase = 1; }   // Y = i·XZ
        return state.measure_pauli(P, u);
    }
};

// ── Task 3.0: multi-qubit carried-port state-level injection helpers ──────────────────────
// Grow a FramedSuperposition by `k` fresh |0> qubits at columns n..n+k-1 (tensor with I on the
// tableau, +Z_j stabilisers ⇒ |0>, eps 0). free/amplitudes untouched (the new qubits are not
// logical; sigma keys are over `free`), so a magic (chi>=2) state grows verbatim.
// grow_framed: cpp/bindings/injection.hpp (J-b1 P9a)

// Harness continuation (2026-10-07): relabel the qubits of a FramedSuperposition by a
// permutation `perm` (perm[q] = the NEW index of old qubit q).  |0…0> is permutation-invariant,
// so U' = Π·U is a valid frame for Π|ψ>: every Xrow/Zrow Pauli gets its column bits moved
// (rows keep their generator index, eps/free/amplitudes untouched); the dual is invalidated.
// Used with grow_framed by adaptq.fidelity to bring two samplers' states (different deferred
// layouts) onto one layout for an exact port overlap.
static void permute_framed(FramedSuperposition& fs, const std::vector<int>& perm) {
    const int n = fs.n();
    if ((int)perm.size() != n)
        throw std::invalid_argument("permute: perm has " + std::to_string(perm.size()) +
                                    " entries for an n=" + std::to_string(n) + " state");
    std::vector<uint8_t> seen((size_t)(n > 0 ? n : 1), 0);
    for (int q = 0; q < n; ++q) {
        const int t = perm[(size_t)q];
        if (t < 0 || t >= n || seen[(size_t)t])
            throw std::invalid_argument("permute: perm is not a permutation of 0..n-1");
        seen[(size_t)t] = 1;
    }
    auto remap = [&](const Pauli& src) {
        Pauli dst(n);
        dst.phase = src.phase;
        for (int q = 0; q < n; ++q) {
            if (src.xbit(q)) dst.setx(perm[(size_t)q]);
            if (src.zbit(q)) dst.setz(perm[(size_t)q]);
        }
        return dst;
    };
    for (int a = 0; a < n; ++a) {
        fs.U.Xrow[(size_t)a] = remap(fs.U.Xrow[(size_t)a]);
        fs.U.Zrow[(size_t)a] = remap(fs.U.Zrow[(size_t)a]);
    }
    fs.U.invalidate_dual();
}


inline void bind_framed_superposition(py::module_& m) {
    // S2.1: FramedSuperposition handle (created by _bare_state_of).
    // S2.2: adds approx_equal (exact state comparison) + apply_clifford (mutation testing).
    auto fs_cls = py::class_<PyFramedSuperposition>(m, "FramedSuperposition", py::module_local(),
          "Handle for a qeccore::FramedSuperposition (a collapsed bare state). "
          "Create via _xtim.bare_state_of(text) / input_state_of(text); a carried state reaches a consumer "
          "sampler as an ideal state (xtim.compile_twirl_sampler_from_ideal_state). "
          "S2.2: approx_equal compares states up to global phase; apply_clifford mutates in-place.")
        .def("approx_equal", &BudgetGuard<&PyFramedSuperposition::approx_equal>::call,
             py::arg("other"), py::arg("tol") = 1e-12,
             "Return True iff |<self|other>| >= 1 - tol (equal up to global phase). "
             "Uses exact_sum_overlap (framed ray materialisation + inner product). "
             "S2.2 correctness oracle: compare sample_barrier states against the "
             "independent _project_bare_onto_syndrome path.")
        .def("overlap", &BudgetGuard<&PyFramedSuperposition::overlap>::call, py::arg("other"),
             "3.1.0: return |<self|other>|^2 exactly (global-phase insensitive) — the value "
             "approx_equal thresholds. Both states must have the same qubit count "
             "(ValueError otherwise). Per-shot Choi-state fidelity estimation uses it.")
        .def("apply_clifford", &BudgetGuard<&PyFramedSuperposition::apply_clifford_py>::call,
             py::arg("kind"), py::arg("a"), py::arg("b") = 0,
             "Apply a Clifford gate to the state in-place. "
             "kind: 0=H 1=S 2=Sdg 3=X 4=Y 5=Z 6=CX 7=CZ. "
             "For single-qubit gates (0-5) b is unused (pass 0). "
             "S2.2 mutation check: apply Z (kind=5) to qubit 0, verify approx_equal fails.")
        .def("pauli_expectation_x", &BudgetGuard<&PyFramedSuperposition::pauli_expectation_x>::call, py::arg("q"),
             "Task 2b.1: exact real single-qubit <X_q> on the collapsed state. Used by the "
             "Born-decision correlation gate (retained state must satisfy <X_0> = ±1).")
        .def("pauli_expectation_z", &BudgetGuard<&PyFramedSuperposition::pauli_expectation_z>::call, py::arg("q"),
             "Task 2b.1: exact real single-qubit <Z_q> on the collapsed state.")
        .def("pauli_expectation_y", &BudgetGuard<&PyFramedSuperposition::pauli_expectation_y>::call, py::arg("q"),
             "Task 2b.2: exact real single-qubit <Y_q> on the collapsed state (Y = i·XZ). "
             "Used by the ⊗-composition to classify a carried qubit's stabilizer axis.")
        .def("branch_frames", &BudgetGuard<&PyFramedSuperposition::branch_frames>::call,
             "3.1.2: export the state as |psi> = sum_i coeff[i] D_i |ref> — dict of numpy arrays: "
             "n, free, eps, stab_x/stab_z/stab_phase (signed generators G_a of |ref>), "
             "destab_x/destab_z/destab_phase (d_a), sigma, coeff, branch_x/branch_z/branch_phase "
             "(D_i = prod_{d asc: sigma_i[d]} d_{free[d]}, engine Pauli phases). Read-only. "
             "Contract + consumer algebra: docs/branch_frames.md and xtim.frames.")
        .def("pauli_expectation_xz", &BudgetGuard<&PyFramedSuperposition::pauli_expectation_xz>::call,
             py::arg("xs"), py::arg("zs"), "Diagnostic: <ΠX_xs·ΠZ_zs> (real).")
        .def("port_signature", &BudgetGuard<&PyFramedSuperposition::port_signature>::call,
             py::arg("wires"),
             "Canonical bytes fingerprint of the reduced stabilizer state on the carried port "
             "`wires` (port-restricted reduced tableau + signs). Equal bytes ⟺ identical reduced "
             "density matrix on `wires`. The BUCKET-FOLD guard (run_protocol): fold record buckets "
             "of a decision pattern iff they share one port_signature.")
        .def("port_signature_struct", &BudgetGuard<&PyFramedSuperposition::port_signature_struct>::call,
             py::arg("wires"),
             "Task 6: structured accessor for the port_signature byte format. Decodes the raw "
             "bytes (via port_signature) into a dict with keys flag, W, ns, keys, blochs — "
             "same fields and semantics as fold.py's _parse_port_signature. One owner of the "
             "wire format: changing port_signature automatically updates this accessor.")
        .def("port_orbit_operators", &BudgetGuard<&PyFramedSuperposition::port_orbit_operators>::call,
             py::arg("wires"),
             "PAULI-ORBIT FOLD companion to port_signature: the canonical port symplectic OPERATORS "
             "whose signs the fingerprint records — dict{ok, W, ns, k_port, stab:[2W-uint8 …], "
             "logicals:[La,Lb]} — in the SAME order as port_signature (stab supports sorted; La,Lb "
             "match the Bloch coords). A pure function of the stabilizer SUPPORTS ⇒ identical across "
             "one sign-stripped orbit class. Coordinate layout matches input_pauli (low W = X, high "
             "W = Z over wires). ok=False for k_port≥2 / W>64 (un-foldable).")
        .def("certified_symplectic", &BudgetGuard<&PyFramedSuperposition::certified_symplectic>::call,
             "E3 (exact-residual arc): the state's certified stabilizer generators as full "
             "n-qubit symplectic rows + exact phases (PURE DATA). Returns dict{n, ngens, "
             "xz:(ngens,2n) uint8 (low n = x bits, high n = z bits), phase:(ngens,) uint8}. "
             "Row b == certified_stabilizers()[b] == σ bit b of BarrierBuffer.sigmas() == "
             "port_sigma_law comb bit b. Generator operator = i^phase·X^x·Z^z; every row has "
             "+1 eigenvalue on the bare state.")
        .def("measure_pauli", &BudgetGuard<&PyFramedSuperposition::measure_pauli>::call,
             py::arg("basis"), py::arg("q"), py::arg("u"),
             "Task 2b.2-pre: Born-measure single-qubit Pauli (basis 0:X 1:Y 2:Z) on qubit q "
             "with random u∈[0,1); returns ±1 and COLLAPSES the state in place. The exact "
             "per-shot decision read the engine performs internally, exposed for the oracle.")
        .def("grow", [](PyFramedSuperposition& self, int k) {
                 budget_poison_on_refusal(self, [&] { grow_framed(self.state, k); });
             },
             py::arg("k"),
             "Harness continuation (2026-10-07): append k fresh |0> qubits at indices n..n+k-1 "
             "(the same grow the state-level injection uses). In place.")
        .def("permute", [](PyFramedSuperposition& self, const std::vector<int>& perm) {
                 budget_poison_on_refusal(self, [&] { permute_framed(self.state, perm); }); },
             py::arg("perm"),
             "Harness continuation (2026-10-07): relabel qubits, perm[q] = new index of old "
             "qubit q (a permutation of 0..n-1; ValueError otherwise). In place; exact.")
        .def("pauli_expectation", [](const PyFramedSuperposition& self, const std::vector<int>& xs,
                                     const std::vector<int>& zs, int phase) {
                 budget_check_valid(self);
                 Pauli P(self.state.n());
                 for (int q : xs) {
                     if (q < 0 || q >= self.state.n()) throw std::out_of_range("pauli_expectation: x site out of range");
                     P.setx(q);
                 }
                 for (int q : zs) {
                     if (q < 0 || q >= self.state.n()) throw std::out_of_range("pauli_expectation: z site out of range");
                     P.setz(q);
                 }
                 P.phase = phase & 3;
                 return self.state.pauli_expectation(P);
             },
             py::arg("xs"), py::arg("zs"), py::arg("phase") = 0,
             "Harness continuation (2026-10-07): exact <psi| i^phase X^xs Z^zs |psi> as a complex "
             "(the literal operator product, the certified_symplectic convention; a Y site is "
             "X·Z = -iY, so pass phase = number of Y sites mod 4 for a Hermitian read). No copy.")
        .def("pauli_expectations", [](const PyFramedSuperposition& self,
                                      const std::vector<std::tuple<std::vector<int>, std::vector<int>, int>>& reads) {
                 budget_check_valid(self);
                 std::vector<std::complex<double>> out;
                 out.reserve(reads.size());
                 const int n = self.state.n();
                 for (const auto& rd : reads) {
                     Pauli P(n);
                     for (int q : std::get<0>(rd)) {
                         if (q < 0 || q >= n) throw std::out_of_range("pauli_expectations: x site out of range");
                         P.setx(q);
                     }
                     for (int q : std::get<1>(rd)) {
                         if (q < 0 || q >= n) throw std::out_of_range("pauli_expectations: z site out of range");
                         P.setz(q);
                     }
                     P.phase = std::get<2>(rd) & 3;
                     out.push_back(self.state.pauli_expectation(P));
                 }
                 return out;
             },
             py::arg("reads"),
             "Harness continuation (2026-10-07): pauli_expectation over a list of (xs, zs, phase) "
             "in one call; returns the list of complex values.")
        .def("copy", [](const PyFramedSuperposition& self) {
                 budget_check_valid(self);
                 return PyFramedSuperposition(self.state); },
             "Harness continuation (2026-10-07): an independent deep copy of the state.")
        .def_property_readonly("n", &PyFramedSuperposition::n, "Number of qubits.")
        .def_property_readonly("k", &PyFramedSuperposition::k,
             "Number of logical (free) DOF; k==0 ⇒ syndrome fully determines the state.");
}

}  // namespace
