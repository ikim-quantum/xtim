#include "qeccore/pauli_rotation_form.hpp"
#include <cstddef>

#include <cassert>
#include <stdexcept>

namespace qeccore {

// ---------------------------------------------------------------------------
// Pauli key (phase-independent): x words || z words.
// ---------------------------------------------------------------------------
std::string pauli_key(const Pauli& p) {
    std::string s;
    s.reserve((p.x.size() + p.z.size()) * sizeof(uint64_t));
    auto push = [&](const std::vector<uint64_t>& w) {
        for (uint64_t v : w)
            for (int b = 0; b < 8; ++b) s.push_back(char((v >> (8 * b)) & 0xff));
    };
    push(p.x);
    push(p.z);
    return s;
}

static inline int mod4(int p) { return ((p % 4) + 4) % 4; }
static inline int mod16(int c) { return ((c % 16) + 16) % 16; }

static Pauli pauliXq(int n, int q) { Pauli p(n); p.setx(q); return p; }
static Pauli pauliZq(int n, int q) { Pauli p(n); p.setz(q); return p; }

// ---------------------------------------------------------------------------
// Clifford conjugation g Q g^{-1}, returning a *signed* Pauli (phase ∈ {0,2}
// for sign on a Hermitian Pauli, or {0..3} in general). Computed exactly by
// mapping each generator X_q / Z_q of Q to its Heisenberg image under g.
// ---------------------------------------------------------------------------
struct XZImage { Pauli imgX, imgZ; };

static XZImage conj_generators(int n, GateKind g, const std::vector<int>& t) {
    XZImage r{Pauli(n), Pauli(n)};
    int a = t[0];
    switch (g) {
        case GateKind::X:  // X X X = X ; X Z X = -Z
            r.imgX = pauliXq(n, a);
            r.imgZ = pauliZq(n, a); r.imgZ.phase = 2;
            break;
        case GateKind::Z:  // Z X Z = -X ; Z Z Z = Z
            r.imgX = pauliXq(n, a); r.imgX.phase = 2;
            r.imgZ = pauliZq(n, a);
            break;
        case GateKind::Y:  // Y X Y = -X ; Y Z Y = -Z
            r.imgX = pauliXq(n, a); r.imgX.phase = 2;
            r.imgZ = pauliZq(n, a); r.imgZ.phase = 2;
            break;
        case GateKind::H:  // H X H = Z ; H Z H = X
            r.imgX = pauliZq(n, a);
            r.imgZ = pauliXq(n, a);
            break;
        case GateKind::S:  // S X S^d = Y = i XZ ; S Z S^d = Z
            r.imgX = pauliXq(n, a); r.imgX.setz(a); r.imgX.phase = 1;  // i XZ = Y
            r.imgZ = pauliZq(n, a);
            break;
        case GateKind::SDG:  // S^d X S = -Y = -i XZ ; S^d Z S = Z
            r.imgX = pauliXq(n, a); r.imgX.setz(a); r.imgX.phase = 3;  // -i XZ = -Y
            r.imgZ = pauliZq(n, a);
            break;
        default:
            throw std::logic_error("conj_generators: 2-qubit / non-Clifford handled elsewhere");
    }
    return r;
}

// Two-qubit gate generator images. control a, target b.
static Pauli img_X(int n, GateKind g, int a, int b, int q) {
    Pauli p(n);
    if (g == GateKind::CX) {
        if (q == a) { p.setx(a); p.setx(b); }   // X_a -> X_a X_b
        else        { p.setx(b); }               // X_b -> X_b
    } else { // CZ
        if (q == a) { p.setx(a); p.setz(b); }    // X_a -> X_a Z_b
        else        { p.setx(b); p.setz(a); }    // X_b -> X_b Z_a
    }
    return p;
}
static Pauli img_Z(int n, GateKind g, int a, int b, int q) {
    Pauli p(n);
    if (g == GateKind::CX) {
        if (q == a) { p.setz(a); }               // Z_a -> Z_a
        else        { p.setz(a); p.setz(b); }    // Z_b -> Z_a Z_b
    } else { // CZ : Z_a -> Z_a, Z_b -> Z_b
        p.setz(q);
    }
    return p;
}

// Conjugate a signed Pauli Q by Clifford gate g (single or two qubit): g Q g^{-1}.
static Pauli conjugate(const Pauli& Q, GateKind g, const std::vector<int>& t) {
    int n = Q.n;
    int a = t[0];
    int b = (t.size() > 1) ? t[1] : -1;
    bool twoq = (g == GateKind::CX || g == GateKind::CZ);

    Pauli result(n);
    result.phase = mod4(Q.phase);
    for (int q = 0; q < n; ++q) {
        bool xb = Q.xbit(q), zb = Q.zbit(q);
        if (!xb && !zb) continue;
        bool touched = twoq ? (q == a || q == b) : (q == a);
        if (!touched) {
            if (xb) result = Pauli::multiply(result, pauliXq(n, q));
            if (zb) result = Pauli::multiply(result, pauliZq(n, q));
            continue;
        }
        if (twoq) {
            if (xb) result = Pauli::multiply(result, img_X(n, g, a, b, q));
            if (zb) result = Pauli::multiply(result, img_Z(n, g, a, b, q));
        } else {
            XZImage im = conj_generators(n, g, t);
            if (xb) result = Pauli::multiply(result, im.imgX);
            if (zb) result = Pauli::multiply(result, im.imgZ);
        }
    }
    return result;
}

// Inverse gate kind (only S/SDG are non-self-inverse in our Clifford set).
static GateKind inverse_gate(GateKind g) {
    if (g == GateKind::S) return GateKind::SDG;
    if (g == GateKind::SDG) return GateKind::S;
    return g;  // X,Y,Z,H,CX,CZ self-inverse
}

// Conjugate a Z-string by the inverse-frame of the whole C list:
//   P = C^{-1} · Zstring · C, with C = [g_1 .. g_m] (g_1 applied first).
//   = g_1^{-1} ( g_2^{-1} ( ... ( g_m^{-1} Z g_m ) ... ) g_1 ).
// Iterate i = m..1, each step P := g_i^{-1} P g_i = conjugate(P, inv(g_i)).
static Pauli conjugate_into_zero_frame(const Pauli& Zstring,
                                       const std::vector<CliffGate>& C) {
    Pauli P = Zstring;
    for (auto it = C.rbegin(); it != C.rend(); ++it)
        P = conjugate(P, inverse_gate(it->kind), it->targets);
    return P;
}

// ---------------------------------------------------------------------------
// Hermitian representative of a bit pattern: i^{(#Y) mod 2} X^x Z^z.
// (Y = i XZ is Hermitian.) Stored T-layer Pauli is ALWAYS this rep (phase 0/1).
// ---------------------------------------------------------------------------
static Pauli hermitian_rep(const Pauli& bits) {
    Pauli Q = bits;
    Q.phase = Q.xz_overlap() & 1;
    return Q;
}

// Split a signed Pauli into (Hermitian rep, sign ∈ {+1,-1}): Qsigned = sign·Q_h.
static std::pair<Pauli, int> split_sign(const Pauli& Qsigned) {
    Pauli Q = hermitian_rep(Qsigned);
    int delta = mod4(Qsigned.phase - Q.phase);  // must be 0 or 2
    assert(delta == 0 || delta == 2);
    return {Q, (delta == 0) ? 1 : -1};
}

static std::string pauli_str(const Pauli& p) {
    std::string s;
    for (int q = 0; q < p.n; ++q) {
        bool xb = p.xbit(q), zb = p.zbit(q);
        if (xb && zb) s += "Y" + std::to_string(q);
        else if (xb)  s += "X" + std::to_string(q);
        else if (zb)  s += "Z" + std::to_string(q);
    }
    if (s.empty()) s = "I";
    return s;
}

// ---------------------------------------------------------------------------
// Fold a signed magic generator (coeff c along Pauli Qsigned) into the ORDERED
// T-list, WITHOUT rejecting.  The generator is the most-recently-applied magic,
// so it enters at the OUTPUT (back) end and slides toward the INPUT (front).
//
//   Scan from back -> front:
//     * same Pauli (key match): combine coeffs mod 16; erase if 0; STOP.
//     * commutes: keep scanning past it.
//     * anticommutes: STOP, INSERT here (it cannot slide past).
//   If the scan reaches the front without folding: prepend (insert at front).
// ---------------------------------------------------------------------------
static void fold_term(PauliRotationForm& f, const Pauli& Qsigned, int c) {
    auto [Q, sgn] = split_sign(Qsigned);
    int cc = mod16(c * sgn);
    if (cc == 0) return;  // trivial rotation
    std::string key = pauli_key(Q);

    // Walk from the back (output) toward the front (input). `pos` is the index
    // at which we would INSERT (i.e. one past the entry we just inspected,
    // moving leftward).  Start pos = size (append at back).
    int i = (int)f.terms.size() - 1;
    for (; i >= 0; --i) {
        PauliRotationForm::Entry& e = f.terms[i];
        if (pauli_key(e.pauli) == key) {
            // same Pauli -> combine coeffs.
            int nc = mod16(e.coeff + cc);
            if (nc == 0) f.terms.erase(f.terms.begin() + i);
            else e.coeff = nc;
            return;  // folded (STOP)
        }
        if (Pauli::commute(Q, e.pauli)) {
            continue;  // slide past
        }
        // anticommutes -> STOP, insert AFTER this entry (to its output side).
        f.terms.insert(f.terms.begin() + (i + 1),
                       PauliRotationForm::Entry{Q, cc});
        return;
    }
    // reached the input end without folding -> prepend at front.
    f.terms.insert(f.terms.begin(), PauliRotationForm::Entry{Q, cc});
}

// ---------------------------------------------------------------------------
// Main builder (deferred fold-then-check).
// ---------------------------------------------------------------------------
PauliRotationForm build_pauli_rotation_form(const Circuit& circuit) {
    PauliRotationForm f;
    f.n = circuit.n;
    int n = circuit.n;
    using cd = std::complex<double>;
    const double PI = 3.14159265358979323846;
    const cd e_i_pi8 = cd(std::cos(PI / 8.0), std::sin(PI / 8.0));

    for (int k = 0; k < (int)circuit.stream.size(); ++k) {
        const Instr& ins = circuit.stream[k];
        if (ins.kind != Instr::Kind::Gate) continue;
        GateKind g = ins.gate;
        const std::vector<int>& t = ins.targets;

        switch (g) {
            // ---- Clifford gates: append to C (act last). T unchanged. ----
            case GateKind::X: case GateKind::Y: case GateKind::Z:
            case GateKind::S: case GateKind::SDG: case GateKind::H:
            case GateKind::CX: case GateKind::CZ:
                f.cliffords.push_back(CliffGate{g, t});
                break;

            // ---- non-Clifford diagonal: expand to Z-string π/8 generators ----
            case GateKind::T: {
                // T = e^{iπ/8} · exp(-i(π/8) Z) : coeff -1 on Z_q, global *= e^{iπ/8}.
                f.global_phase *= e_i_pi8;
                Pauli P = conjugate_into_zero_frame(pauliZq(n, t[0]), f.cliffords);
                fold_term(f, P, -1);
                break;
            }
            case GateKind::CS: {
                // CS = exp(i(π/8)(I-Z_a)(I-Z_b))
                //    = e^{iπ/8} · exp(-iπ/8 Z_a) exp(-iπ/8 Z_b) exp(+iπ/8 Z_aZ_b).
                int a = t[0], b = t[1];
                f.global_phase *= e_i_pi8;
                Pauli Za(n); Za.setz(a);
                Pauli Zb(n); Zb.setz(b);
                Pauli Zab(n); Zab.setz(a); Zab.setz(b);
                // Fold in order (each acts as the next-most-recent magic).
                fold_term(f, conjugate_into_zero_frame(Za, f.cliffords), -1);
                fold_term(f, conjugate_into_zero_frame(Zb, f.cliffords), -1);
                fold_term(f, conjugate_into_zero_frame(Zab, f.cliffords), +1);
                break;
            }
            case GateKind::CCZ: {
                // CCZ = exp(i(π/8)(I-Z_a)(I-Z_b)(I-Z_c))
                //  = e^{iπ/8} over: Z_a:-1,Z_b:-1,Z_c:-1, ZaZb:+1,ZaZc:+1,ZbZc:+1, ZaZbZc:-1.
                int a = t[0], b = t[1], c = t[2];
                f.global_phase *= e_i_pi8;
                auto Zstr = [&](std::vector<int> qs) {
                    Pauli p(n); for (int q : qs) p.setz(q); return p;
                };
                struct GC { Pauli P; int c; };
                std::vector<GC> gens = {
                    {Zstr({a}), -1}, {Zstr({b}), -1}, {Zstr({c}), -1},
                    {Zstr({a,b}), +1}, {Zstr({a,c}), +1}, {Zstr({b,c}), +1},
                    {Zstr({a,b,c}), -1},
                };
                for (auto& gen : gens)
                    fold_term(f, conjugate_into_zero_frame(gen.P, f.cliffords), gen.c);
                break;
            }
            case GateKind::CH: {
                // CH = (I⊗R_y(π/4)_t) · CZ(c,t) · (I⊗R_y(−π/4)_t)   [rightmost applied first].
                // Time order (first→last): rot(+1, Y_t) ; CZ(c,t) ; rot(−1, Y_t).
                // R_y(θ)=exp(−iθY/2) has determinant 1 ⇒ NO global phase (unlike T/CS/CCZ).
                // KEEP IN LOCKSTEP with the propagation-side decomposition (ppr_residual.cpp,
                // case GateKind::CH): same three sub-ops, same sign convention; each side is
                // independently pinned by its dense-unitary oracle.
                int c = t[0], tgt = t[1];
                Pauli Yt(n); Yt.setx(tgt); Yt.setz(tgt); Yt.phase = 1;  // Hermitian Y = i·XZ
                fold_term(f, conjugate_into_zero_frame(Yt, f.cliffords), +1);
                f.cliffords.push_back(CliffGate{GateKind::CZ, {c, tgt}});
                fold_term(f, conjugate_into_zero_frame(Yt, f.cliffords), -1);
                break;
            }
            default:
                // Out-of-class gate (e.g. H/CH on a live magic wire, or any gate the
                // diagonal-Clifford+T form cannot represent). Reject CLEANLY carrying the
                // deferred-stream index `k`, matching the OLD sweep's reject_gate_index
                // convention (>= 0 ⇒ propagation/class reject ⇒ exit 3), instead of throwing.
                f.rejected = true;
                f.reject_gate_index = k;
                f.reject_reason = "out-of-class gate at deferred index " + std::to_string(k) +
                                  ": not representable as a diagonal-Clifford + commuting-T form";
                return f;
        }
    }

    // ---- THE SINGLE decision -------------------------------------------------
    // Even-coeff entries are Clifford and may sit BETWEEN odd entries; settling
    // them into C conjugates the odd Paulis they pass, which can change those
    // odd axes (e.g. a CLIFFORD X between two Z's rotates the right Z to Y, so the
    // two magic rotations no longer commute). So the decisive commute test must
    // be run on the odd axes AFTER conceptually settling the evens to the output
    // (right) end.  We do this on a COPY of the axes (the materialized T is left
    // intact — applyRotation handles arbitrary even entries exactly).
    //
    // Settle rule: pull the leftmost even entry E (exponent t = coeff/2 of
    // exp(iπ/4 E)) rightward.  For each entry P to its right with t ODD and
    // {E,P}=0, the axis becomes E·P (the product); otherwise unchanged. Only the
    // AXIS matters for the commute test (signs are irrelevant to commutation).
    {
        struct W { Pauli p; int coeff; };
        std::vector<W> work;
        for (const auto& e : f.terms) work.push_back({e.pauli, e.coeff});

        // repeatedly remove the leftmost even entry, conjugating entries to its
        // right.
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t p = 0; p < work.size(); ++p) {
                if (work[p].coeff % 2 == 0) {
                    Pauli E = work[p].p;
                    int t = (work[p].coeff / 2);  // exponent parity is what matters
                    bool t_odd = (t & 1);
                    for (size_t r = p + 1; r < work.size(); ++r) {
                        if (t_odd && !Pauli::commute(E, work[r].p)) {
                            // axis -> E·P (product); phase irrelevant to commute.
                            Pauli prod = Pauli::multiply(E, work[r].p);
                            prod.phase = 0;
                            work[r].p = prod;
                        }
                    }
                    work.erase(work.begin() + p);
                    changed = true;
                    break;
                }
            }
        }

        // remaining `work` entries are all odd magic; test mutual commutation.
        for (size_t i = 0; i < work.size() && !f.rejected; ++i)
            for (size_t j = i + 1; j < work.size(); ++j)
                if (!Pauli::commute(work[i].p, work[j].p)) {
                    f.rejected = true;
                    f.reject_reason = "non-commuting magic: " + pauli_str(work[i].p) +
                                      " and " + pauli_str(work[j].p) + " anticommute";
                    break;
                }
    }
    return f;
}

}  // namespace qeccore
