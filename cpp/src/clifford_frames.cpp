#include "qeccore/clifford_frames.hpp"
#include <cstddef>

namespace qeccore {
namespace {
Instr g1(GateKind g, int q) {
    Instr i; i.kind = Instr::Kind::Gate; i.gate = g; i.targets = {q}; return i;
}
}  // namespace

// Single-qubit Clifford R mapping the +1/-1 eigenstates of Pauli `b` to |0>/|1> (Z basis):
//   Z: {}            (already in Z basis)
//   X: {H}           (H|+>=|0>, H|->=|1>)
//   Y: {SDG, H}      (|+i> -> |0>, |-i> -> |1>)
std::vector<Instr> rotate_to_z(PauliBasis b, int q) {
    std::vector<Instr> out;
    if (b == PauliBasis::X) {
        out.push_back(g1(GateKind::H, q));
    } else if (b == PauliBasis::Y) {
        out.push_back(g1(GateKind::SDG, q));
        out.push_back(g1(GateKind::H, q));
    }
    return out;
}

// Inverse of rotate_to_z: maps |0>/|1> back to the +1/-1 eigenstates of `b`.
//   Z: {} ; X: {H} ; Y: {H, S}  (reverse of {SDG,H} with S=SDG^-1)
std::vector<Instr> rotate_from_z(PauliBasis b, int q) {
    std::vector<Instr> out;
    if (b == PauliBasis::X) {
        out.push_back(g1(GateKind::H, q));
    } else if (b == PauliBasis::Y) {
        out.push_back(g1(GateKind::H, q));
        out.push_back(g1(GateKind::S, q));
    }
    return out;
}

namespace {

bool is_gate(const Instr& i, GateKind g) {
    return i.kind == Instr::Kind::Gate && i.gate == g;
}
bool gate_on(const Instr& i, GateKind g, int q) {
    return is_gate(i, g) && i.targets.size() == 1 && i.targets[0] == q;
}
// Does instruction `i` touch qubit `q` (as a gate target, measure/reset/feedback qubit,
// or noise wire)? Used to verify a fold target is idle between H and its boundary.
bool touches(const Instr& i, int q) {
    switch (i.kind) {
        case Instr::Kind::Gate:
            for (int t : i.targets) if (t == q) return true;
            return false;
        case Instr::Kind::Measure:
        case Instr::Kind::Reset:
        case Instr::Kind::Noise:
        case Instr::Kind::ControlledPauli:
            for (int t : i.qubits) if (t == q) return true;
            return false;
        case Instr::Kind::Observable:
            for (const PauliTerm& pt : i.obs) if (pt.qubit == q) return true;
            return false;
    }
    return false;
}

}  // namespace

// ===========================================================================
// Single-qubit Clifford frame (symplectic + sign) tracker for frame-pushing.
// ===========================================================================
//
// A 1-qubit Clifford F is represented by its conjugation action on the Pauli
// group: the signed images  F X F† = sx · Px  and  F Z F† = sz · Pz, where each
// image Pauli is encoded as (xbit,zbit): I=(0,0) X=(1,0) Z=(0,1) Y=(1,1), and the
// sign is in {+1,-1}. The image of Y is derived: Y = i·X·Z so F Y F† = i·(FXF†)(FZF†).
//
// Signed Pauli with i-phase: phase in Z4 (powers of i), pauli=(x,z).
// Multiplication: X^{x1}Z^{z1} · X^{x2}Z^{z2} = (-1)^{z1·x2} X^{x1⊕x2} Z^{z1⊕z2}.
struct SP {            // signed (i-phased) Pauli
    int ph;            // phase = i^ph, ph in {0,1,2,3}
    int x, z;          // X^x Z^z
};
static SP sp_mul(const SP& a, const SP& b) {
    int ph = (a.ph + b.ph + 2 * (a.z & b.x)) & 3;   // (-1)^{z1 x2} = i^{2 z1 x2}
    return SP{ph, a.x ^ b.x, a.z ^ b.z};
}

// A 1-qubit Clifford as signed images of X and Z (images are HERMITIAN Paulis,
// so phase is 0 or 2, i.e. sign ±1). We store full Z4 phase for safe arithmetic.
struct Frame {
    SP xim{0, 1, 0};   // F X F† = +X  (identity)
    SP zim{0, 0, 1};   // F Z F† = +Z
    bool is_identity() const {
        return xim.ph == 0 && xim.x == 1 && xim.z == 0 &&
               zim.ph == 0 && zim.x == 0 && zim.z == 1;
    }
    // F Y F† = i · (F X F†)(F Z F†).
    SP yim() const { SP r = sp_mul(xim, zim); r.ph = (r.ph + 1) & 3; return r; }
    // F P F† for a *unsigned* basis Pauli p in {X=0,Y=1,Z=2}.
    SP image(int p) const { return p == 0 ? xim : (p == 2 ? zim : yim()); }
};

// Compose a single-qubit GATE g (applied AFTER F, i.e. F' = g∘F) into F.
// New images: F' P F'† = g (F P F†) g†. So conjugate F's images by g.
// We give g's conjugation action on X and Z directly, then map a signed Pauli.
static SP conj_basis(GateKind g, int p /*0=X,1=Y,2=Z*/) {
    // returns g·p·g† in the convention  i^ph · X^x Z^z. Note: +Y = i^1·XZ → SP{1,1,1};
    // −Y = i^3·XZ → SP{3,1,1}; +X={0,1,0}; −X={2,1,0}; +Z={0,0,1}; −Z={2,0,1}.
    switch (g) {
        case GateKind::H:    // X<->Z, Y->-Y
            if (p == 0) return SP{0,0,1};            // HXH=Z
            if (p == 2) return SP{0,1,0};            // HZH=X
            return SP{3,1,1};                         // HYH=-Y
        case GateKind::S:    // S X S†=Y, S Z S†=Z, S Y S†=-X
            if (p == 0) return SP{1,1,1};            // X->+Y
            if (p == 2) return SP{0,0,1};            // Z->Z
            return SP{2,1,0};                         // Y->-X
        case GateKind::SDG:  // S† X S=-Y, Z->Z, Y->X
            if (p == 0) return SP{3,1,1};            // X->-Y
            if (p == 2) return SP{0,0,1};            // Z->Z
            return SP{0,1,0};                         // Y->+X
        case GateKind::X:    // X X X=X, X Z X=-Z, X Y X=-Y
            if (p == 0) return SP{0,1,0};
            if (p == 2) return SP{2,0,1};
            return SP{3,1,1};                         // -Y
        case GateKind::Y:    // Y X Y=-X, Y Z Y=-Z, Y Y Y=Y
            if (p == 0) return SP{2,1,0};
            if (p == 2) return SP{2,0,1};
            return SP{1,1,1};                         // +Y
        case GateKind::Z:    // Z X Z=-X, Z Z Z=Z, Z Y Z=-Y
            if (p == 0) return SP{2,1,0};
            if (p == 2) return SP{0,0,1};
            return SP{3,1,1};                         // -Y
        default: return SP{0, p==0?1:(p==1?1:0), p==2?1:(p==1?1:0)};
    }
}
// Map a signed Pauli through gate g's conjugation: g·(SP)·g†.
static SP conj_sp(GateKind g, const SP& s) {
    // decode (x,z) -> basis index; carry the *residual* phase beyond the bare Pauli's
    // intrinsic phase (Y = i^1·XZ carries 1; X,Z carry 0). conj_basis returns the image
    // of the bare +Pauli, so we add (s.ph - intrinsic(p)).
    int p, intrinsic;
    if (s.x == 0 && s.z == 0) { return s; }          // identity Pauli
    if (s.x == 1 && s.z == 0) { p = 0; intrinsic = 0; }   // X
    else if (s.x == 0 && s.z == 1) { p = 2; intrinsic = 0; }  // Z
    else { p = 1; intrinsic = 1; }                   // Y (x=1,z=1) = i^1 XZ
    SP base = conj_basis(g, p);
    base.ph = (base.ph + s.ph - intrinsic + 4) & 3;
    return base;
}
static Frame compose_gate(const Frame& F, GateKind g) {
    Frame r;
    r.xim = conj_sp(g, F.xim);
    r.zim = conj_sp(g, F.zim);
    return r;
}

// Decode a Hermitian signed Pauli (an image of a Clifford) into (basis index 0=X/1=Y/2=Z,
// sign true if -1). The SP encodes  i^ph · X^x Z^z. For X/Z the Hermitian phase is ph∈{0,2};
// for Y = i^1 XZ (= i^{ph-1}Y) it is ph∈{1,3}: ph≡(1) → +Y, ph≡(3) → −Y.
static void sp_signed(const SP& s, int& basis_idx, bool& minus) {
    if (s.x && s.z) {              // Y-type:  i^ph XZ = i^{ph-1} Y
        basis_idx = 1;
        minus = (((s.ph - 1) & 3) == 2);     // i^2 = -1
    } else {                        // X (1,0), Z (0,1), I (0,0)
        basis_idx = (s.x ? 0 : 2);
        minus = ((s.ph & 3) == 2);
    }
}

// Emit a gate sequence realizing the single-qubit Clifford F on wire q (for FLUSH).
// Brute-force over the 24-element group: BFS shortest word in {H,S,X} that matches F's
// (xim,zim). Precomputed once. Returns the literal IR gates (applied in stream order).
static const std::vector<GateKind>& frame_word(const Frame& F);

// ===========================================================================
namespace {

// Try to push the pending frames (Fa,Fb) on a CZ/CX gate G(a,b) to the RIGHT:
//   G ∘ (Fa ⊗ Fb)  =  (Fa' ⊗ Fb') ∘ G'    with G' a *bare* 2-qubit Clifford in
//   {CZ, CX(a,b), CX(b,a)} and Fa',Fb' single-qubit residues.
// We search: for each candidate G' and the resulting required residues, check exact
// equality of the 2-qubit symplectic tableau. Returns true + fills G'/Fa'/Fb' on success.
// SOUND: only returns true when the factorization is EXACT (verified by tableau equality).

// 2-qubit Pauli image tracking: we track images of X0,Z0,X1,Z1 (wire-local indices 0/1)
// as signed 2-qubit Paulis. A 2-qubit Clifford = these 4 signed images. Build the tableau
// of  G ∘ (Fa⊗Fb)  and of each candidate (Fa'⊗Fb') ∘ G', compare.

// 2-qubit signed Pauli: phase i^ph, (x0,z0,x1,z1).
struct SP2 { int ph; int x0,z0,x1,z1; };
SP2 sp2_mul(const SP2& a, const SP2& b) {
    int ph = (a.ph + b.ph + 2*((a.z0 & b.x0) ^ (a.z1 & b.x1))) & 3;
    return SP2{ph, a.x0^b.x0, a.z0^b.z0, a.x1^b.x1, a.z1^b.z1};
}
// embed a 1-qubit SP on wire 0 or 1.
SP2 embed(const SP& s, int wire) {
    return wire == 0 ? SP2{s.ph, s.x, s.z, 0, 0} : SP2{s.ph, 0, 0, s.x, s.z};
}
struct Tab2 { SP2 x0, z0, x1, z1; };  // images of X0,Z0,X1,Z1

// conjugation of a 2-qubit basis generator by a *bare* 2-qubit gate (control c, target t
// in wire-local 0/1 indexing); returns image SP2 for generator gen in {0:X0,1:Z0,2:X1,3:Z1}.
SP2 conj2_bare(GateKind g, int c, int t, int gen) {
    // start from the identity image of the generator
    SP2 X0{0,1,0,0,0}, Z0{0,0,1,0,0}, X1{0,0,0,1,0}, Z1{0,0,0,0,1};
    SP2 base = (gen==0?X0:gen==1?Z0:gen==2?X1:Z1);
    if (g == GateKind::CZ) {
        // CZ: X0->X0 Z1, Z0->Z0, X1->Z0 X1, Z1->Z1   (symmetric)
        if (gen==0) return SP2{0,1,0,0,1};
        if (gen==1) return Z0;
        if (gen==2) return SP2{0,0,1,1,0};
        return Z1;
    }
    // CX with control c, target t (c,t in {0,1}, c!=t).
    // CX: X_c -> X_c X_t ; Z_c -> Z_c ; X_t -> X_t ; Z_t -> Z_c Z_t.
    auto Xq = [](int w){ return w==0?SP2{0,1,0,0,0}:SP2{0,0,0,1,0}; };
    auto Zq = [](int w){ return w==0?SP2{0,0,1,0,0}:SP2{0,0,0,0,1}; };
    // decode generator -> (which wire, X or Z)
    int gw = (gen==0||gen==1)?0:1;
    bool isX = (gen==0||gen==2);
    if (isX && gw==c) return sp2_mul(Xq(c), Xq(t));   // X_c -> X_c X_t
    if (isX && gw==t) return Xq(t);                    // X_t -> X_t
    if (!isX && gw==c) return Zq(c);                   // Z_c -> Z_c
    if (!isX && gw==t) return sp2_mul(Zq(c), Zq(t));   // Z_t -> Z_c Z_t
    return base;
}
// Tableau of a bare 2-qubit gate.
Tab2 bare_tab(GateKind g, int c, int t) {
    return Tab2{ conj2_bare(g,c,t,0), conj2_bare(g,c,t,1),
                conj2_bare(g,c,t,2), conj2_bare(g,c,t,3) };
}
// Tableau of (Fa ⊗ Fb): image of X0 is embed(Fa.image(X)), etc.
Tab2 tensor_tab(const Frame& Fa, const Frame& Fb) {
    return Tab2{ embed(Fa.xim,0), embed(Fa.zim,0),
                embed(Fb.xim,1), embed(Fb.zim,1) };
}
// Compose tableaus: (T_after ∘ T_before) — image under T_before then T_after.
// If M = U∘V (V first), then M P M† = U (V P V†) U†. So map V's image generators
// through U's tableau (substitute generators by U's images).
SP2 apply_tab(const Tab2& U, const SP2& s) {
    // s is a product of generators X0^x0 Z0^z0 X1^x1 Z1^z1 with phase. Substitute each.
    SP2 acc{ s.ph, 0,0,0,0 };
    if (s.x0) acc = sp2_mul(acc, U.x0);
    if (s.z0) acc = sp2_mul(acc, U.z0);
    if (s.x1) acc = sp2_mul(acc, U.x1);
    if (s.z1) acc = sp2_mul(acc, U.z1);
    return acc;
}
Tab2 compose_tab(const Tab2& U, const Tab2& V) {
    return Tab2{ apply_tab(U, V.x0), apply_tab(U, V.z0),
                apply_tab(U, V.x1), apply_tab(U, V.z1) };
}
bool sp2_eq(const SP2& a, const SP2& b) {
    return a.ph==b.ph && a.x0==b.x0 && a.z0==b.z0 && a.x1==b.x1 && a.z1==b.z1;
}
bool tab_eq(const Tab2& a, const Tab2& b) {
    return sp2_eq(a.x0,b.x0) && sp2_eq(a.z0,b.z0) && sp2_eq(a.x1,b.x1) && sp2_eq(a.z1,b.z1);
}

}  // namespace (push helpers)

// Enumerate the 24 single-qubit Cliffords as Frames (canonical, with a shortest word).
static const std::vector<std::pair<Frame, std::vector<GateKind>>>& all_frames() {
    static std::vector<std::pair<Frame, std::vector<GateKind>>> tbl;
    if (!tbl.empty()) return tbl;
    // BFS from identity over generators {H,S,X} (X needed to reach all signs).
    auto key = [](const Frame& F) {
        // canonical key from images (phase mod 4, x, z for X and Z images).
        return ((((F.xim.ph<<2 | F.xim.x<<1 | F.xim.z) << 6)
                | (F.zim.ph<<2 | F.zim.x<<1 | F.zim.z)));
    };
    std::vector<int> seen;
    auto contains = [&](int k){ for (int s : seen) if (s==k) return true; return false; };
    Frame I;
    tbl.push_back({I, {}});
    seen.push_back(key(I));
    size_t head = 0;
    const GateKind gens[3] = {GateKind::H, GateKind::S, GateKind::X};
    while (head < tbl.size()) {
        Frame cur = tbl[head].first;
        std::vector<GateKind> word = tbl[head].second;
        ++head;
        for (GateKind g : gens) {
            Frame nf = compose_gate(cur, g);
            int k = key(nf);
            if (!contains(k)) {
                seen.push_back(k);
                std::vector<GateKind> nw = word; nw.push_back(g);
                tbl.push_back({nf, nw});
            }
        }
    }
    return tbl;
}
static bool frame_eq(const Frame& a, const Frame& b) {
    return a.xim.ph==b.xim.ph && a.xim.x==b.xim.x && a.xim.z==b.xim.z &&
           a.zim.ph==b.zim.ph && a.zim.x==b.zim.x && a.zim.z==b.zim.z;
}
static const std::vector<GateKind>& frame_word(const Frame& F) {
    static const std::vector<GateKind> empty;
    for (const auto& pr : all_frames()) if (frame_eq(pr.first, F)) return pr.second;
    return empty;  // unreachable for valid Frames
}

// Try to factor  G(a,b) ∘ (Fa ⊗ Fb)  =  (Fa' ⊗ Fb') ∘ G'  with G' bare in
// {CZ, CX(0,1), CX(1,0)} (wire-local). On success returns true and sets out_*.
static bool try_push(GateKind G, const Frame& Fa, const Frame& Fb,
                     GateKind& outG, int& outC, int& outT, Frame& outFa, Frame& outFb) {
    // target tableau  L = G ∘ (Fa⊗Fb)
    Tab2 L = compose_tab(bare_tab(G, 0, 1), tensor_tab(Fa, Fb));
    struct Cand { GateKind g; int c, t; };
    const Cand cands[3] = { {GateKind::CZ,0,1}, {GateKind::CX,0,1}, {GateKind::CX,1,0} };
    const auto& F24 = all_frames();
    for (const Cand& cd : cands) {
        Tab2 GT = bare_tab(cd.g, cd.c, cd.t);
        for (const auto& pa : F24) for (const auto& pb : F24) {
            // R = (Fa'⊗Fb') ∘ G'
            Tab2 R = compose_tab(tensor_tab(pa.first, pb.first), GT);
            if (tab_eq(L, R)) {
                outG = cd.g; outC = cd.c; outT = cd.t;
                outFa = pa.first; outFb = pb.first;
                return true;
            }
        }
    }
    return false;
}

namespace {

bool is_single_qubit_clifford(GateKind g) {
    return g==GateKind::H || g==GateKind::S || g==GateKind::SDG ||
           g==GateKind::X || g==GateKind::Y || g==GateKind::Z;
}

// basis index encoding for the Frame::image API: X=0, Y=1, Z=2.
static int basis_to_idx(PauliBasis b) {
    return b == PauliBasis::X ? 0 : (b == PauliBasis::Y ? 1 : 2);
}
static PauliBasis idx_to_basis(int p) {
    return p == 0 ? PauliBasis::X : (p == 1 ? PauliBasis::Y : PauliBasis::Z);
}

// Frame-pushing pass: pushes single-qubit Clifford frames rightward through the two-qubit
// Clifford bulk (CZ/CX), rewriting gates so leading H's reach the boundaries. Frames are
// ABSORBED into terminal single-qubit measurement bases (with a record-bit flip carrying
// any sign), and FLUSHED as explicit gates at any other boundary (Reset / non-Clifford /
// unknown gate / feedback / end). A frame that cannot factor through a two-qubit gate as a
// single-qubit residue is flushed and the gate emitted verbatim — always sound.
Circuit push_frames(const Circuit& in) {
    Circuit out; out.n = in.n;
    std::vector<Frame> frame(in.n);              // per-wire pending Clifford (identity init)

    auto flush = [&](int q) {
        if (frame[q].is_identity()) return;
        for (GateKind g : frame_word(frame[q])) {
            Instr i; i.kind = Instr::Kind::Gate; i.gate = g; i.targets = {q};
            out.stream.push_back(std::move(i));
        }
        frame[q] = Frame{};
    };
    auto flush_all = [&]() { for (int q = 0; q < in.n; ++q) flush(q); };

    for (const Instr& cur : in.stream) {
        if (cur.kind == Instr::Kind::Gate && cur.targets.size() == 1 &&
            is_single_qubit_clifford(cur.gate)) {
            int q = cur.targets[0];
            frame[q] = compose_gate(frame[q], cur.gate);   // F' = g ∘ F (g applied after)
            continue;
        }
        if (cur.kind == Instr::Kind::Gate && cur.targets.size() == 2 &&
            (cur.gate == GateKind::CZ || cur.gate == GateKind::CX)) {
            int a = cur.targets[0], b = cur.targets[1];
            // emitted CX uses (control=first target, target=second). Build local tableau
            // with wire 0 = a, wire 1 = b; for CX the control is a (wire 0).
            GateKind G = cur.gate;
            GateKind outG; int outC, outT; Frame nfa, nfb;
            // tableau of the incoming gate in local coords:
            // for CZ symmetric; for CX control=wire0(a), target=wire1(b).
            // try_push expects G with local control 0 / target 1 for CX.
            bool ok = try_push(G, frame[a], frame[b], outG, outC, outT, nfa, nfb);
            if (ok) {
                // emit G' with local wires mapped back: wire0->a, wire1->b.
                int wa = a, wb = b;
                Instr g; g.kind = Instr::Kind::Gate; g.gate = outG;
                if (outG == GateKind::CZ) g.targets = {wa, wb};
                else g.targets = { outC==0?wa:wb, outT==0?wa:wb };
                out.stream.push_back(std::move(g));
                frame[a] = nfa; frame[b] = nfb;
                continue;
            }
            // can't factor -> flush both, emit gate verbatim.
            flush(a); flush(b);
            out.stream.push_back(cur);
            continue;
        }
        // ---- Terminal measurement: ABSORB the pending frame into the read basis. ----
        // Measuring Pauli P_b after applying frame F == measuring (F† P_b F) before F.
        // We find the basis Pauli Q with  F Q F† = ± P_b  and read Q instead; a minus
        // sign flips the recorded outcome bit (XOR into `invert`).
        if (cur.kind == Instr::Kind::Measure && cur.qubits.size() == 1) {
            int q = cur.qubits[0];
            if (frame[q].is_identity()) { out.stream.push_back(cur); continue; }
            int pb = basis_to_idx(cur.basis);
            // find Q in {X,Y,Z} whose F-image is ± P_b
            Instr m = cur;
            bool found = false;
            for (int Q = 0; Q < 3; ++Q) {
                SP img = frame[q].image(Q);                // F (basis Q) F†
                int ip; bool minus; sp_signed(img, ip, minus);
                if (ip == pb) {
                    m.basis = idx_to_basis(Q);
                    if (minus) m.invert = !m.invert;       // measuring -P flips the bit
                    found = true; break;
                }
            }
            if (found) { frame[q] = Frame{}; out.stream.push_back(std::move(m)); continue; }
            // Should not happen for a valid Clifford frame; fall back to flush.
            flush(q); out.stream.push_back(cur); continue;
        }
        // Any other boundary / unknown gate: flush touched wires, emit verbatim.
        switch (cur.kind) {
            case Instr::Kind::Reset:
            case Instr::Kind::Measure:           // multi-qubit measure (rare): flush its wires
                for (int q : cur.qubits) flush(q);
                break;
            case Instr::Kind::Gate:
                for (int q : cur.targets) flush(q);
                break;
            case Instr::Kind::ControlledPauli:
            case Instr::Kind::Noise:
                for (int q : cur.qubits) flush(q);
                break;
            case Instr::Kind::Observable:
                flush_all();
                break;
        }
        out.stream.push_back(cur);
    }
    flush_all();
    return out;
}

}  // namespace

// Boundary folds (Task 2): H;M(Z)->MX/MY and R;H[;S]->RX/RY. Operates on an already
// frame-pushed deferred circuit; copies everything else verbatim.
static Circuit apply_boundary_folds(const Circuit& deferred) {
    // No-op early-out: nothing to do if there is no single-qubit H anywhere.
    bool has_h = false;
    for (const Instr& i : deferred.stream) {
        if (is_gate(i, GateKind::H)) { has_h = true; break; }
    }
    if (!has_h) return deferred;

    const std::vector<Instr>& in = deferred.stream;
    const size_t N = in.size();
    std::vector<bool> consumed(N, false);  // indices folded away (skip on emit)

    Circuit out;
    out.n = deferred.n;

    for (size_t idx = 0; idx < N; ++idx) {
        if (consumed[idx]) continue;
        const Instr& cur = in[idx];

        // ---- Fold 2: R q [; H q [; S q]]  ->  RX q / RY q ----
        // The deferred circuit represents RX/RY as a bare Reset (|0>) followed by the
        // basis rotation gates of rotate_from_z. Re-absorb a directly-following H (and
        // optional S) on the SAME qubit back onto the reset's basis tag. We require the
        // H to be the very next instruction touching q.
        if (cur.kind == Instr::Kind::Reset && cur.qubits.size() == 1) {
            int q = cur.qubits[0];
            // Find the next instruction (after idx) that touches q.
            size_t j = idx + 1;
            while (j < N && consumed[j]) ++j;
            // It must be `H q` immediately (no other instruction touching q first).
            // Because the stream is ordered and Reset just wrote |0> on q, the only
            // operations between can be ones NOT touching q; but to stay maximally
            // conservative we require the immediate next stream slot to be H q.
            if (j < N && gate_on(in[j], GateKind::H, q)) {
                // Optional trailing `S q` to make RY.
                size_t k = j + 1;
                while (k < N && consumed[k]) ++k;
                bool ry = (k < N && gate_on(in[k], GateKind::S, q));
                // Emit a plain Reset, then the basis rotation (RX={H}, RY={H,S}).
                Instr r = cur;
                out.stream.push_back(r);
                PauliBasis b = ry ? PauliBasis::Y : PauliBasis::X;
                for (Instr& g : rotate_from_z(b, q)) out.stream.push_back(std::move(g));
                consumed[j] = true;
                if (ry) consumed[k] = true;
                continue;
            }
            // No fold: emit the reset verbatim.
            out.stream.push_back(cur);
            continue;
        }

        // ---- Fold 1: [SDG q;] H q ; ... idle ... ; M q (Z basis)  ->  MX/MY q ----
        // An H on q whose only later use of q is the terminal Z-basis measurement folds
        // into a basis-changed measurement. H toggles Z<->X; a preceding SDG gives Y.
        if (is_gate(cur, GateKind::H) && cur.targets.size() == 1) {
            int q = cur.targets[0];
            // Does a preceding (un-consumed) SDG q immediately precede this H, making MY?
            // Look backward over the EMITTED stream for an SDG q we just wrote.
            bool have_sdg_prefix = !out.stream.empty()
                && gate_on(out.stream.back(), GateKind::SDG, q);

            // Scan forward: the next instruction touching q must be a terminal Z-basis
            // measurement, with nothing else touching q in between.
            size_t j = idx + 1;
            bool ok = false;
            size_t meas_idx = N;
            while (j < N) {
                if (consumed[j]) { ++j; continue; }
                if (touches(in[j], q)) {
                    if (in[j].kind == Instr::Kind::Measure
                        && in[j].qubits.size() == 1
                        && in[j].qubits[0] == q
                        && in[j].basis == PauliBasis::Z) {
                        // Must be terminal: no further instruction touches q after it.
                        meas_idx = j;
                        ok = true;
                    }
                    break;  // first touch of q decides
                }
                ++j;
            }
            if (ok) {
                // Confirm terminality: no instruction after meas_idx touches q.
                for (size_t m = meas_idx + 1; m < N && ok; ++m)
                    if (!consumed[m] && touches(in[m], q)) ok = false;
            }
            if (ok) {
                // Build the new measurement: same record-flip flags, basis X (or Y if SDG).
                Instr mz = in[meas_idx];
                if (have_sdg_prefix) {
                    // Drop the already-emitted SDG and emit MY instead.
                    out.stream.pop_back();
                    mz.basis = PauliBasis::Y;
                } else {
                    mz.basis = PauliBasis::X;
                }
                out.stream.push_back(std::move(mz));
                consumed[meas_idx] = true;
                // The H itself is dropped (not emitted).
                continue;
            }
            // No fold: emit the H verbatim.
            out.stream.push_back(cur);
            continue;
        }

        // Everything else: verbatim.
        out.stream.push_back(cur);
    }

    return out;
}

// See header. Push single-qubit Clifford frames through the Clifford bulk so leading
// H's reach the boundaries (rewriting CZ<->CX etc.), then absorb boundary-adjacent H's
// into prep/measure basis tags. SOUND: any frame that cannot be pushed is flushed
// verbatim; every transform is an exact symplectic factorization.
Circuit eliminate_hadamards(const Circuit& deferred) {
    // No-op early-out: nothing to do if there is no single-qubit H anywhere.
    bool has_h = false;
    for (const Instr& i : deferred.stream)
        if (is_gate(i, GateKind::H)) { has_h = true; break; }
    if (!has_h) return deferred;
    return apply_boundary_folds(push_frames(deferred));
}

}  // namespace qeccore
