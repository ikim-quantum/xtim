// injection.hpp — the state-level injection helpers of the twirl sampler bindings (D2 / Task 3.0), shared by the
// production binding (cpp/bindings/xtim_py.cpp) and the J-b1 P9a test API (the family sampler built from an ideal
// state). Moved here verbatim from xtim_py.cpp (static → inline); no behaviour change.
#pragma once

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/stim_parse.hpp"

namespace qeccore {

// Grow a FramedSuperposition by `k` fresh |0> qubits at columns n..n+k-1 (tensor with I on the tableau, +Z_j
// stabilisers => |0>, eps 0); free / amplitudes untouched.
inline void grow_framed(FramedSuperposition& fs, int k) {
    if (k <= 0) return;
    const int old_n = fs.n();
    fs.U.grow_identity_qubits(k);
    fs.eps.resize((size_t)old_n + k, 0);   // new stabilisers g_{n+j}=Z_{n+j}: g|0>=+|0>
}

// D2 — the n-qubit computational-basis |0…0> as a FramedSuperposition (χ=1, one amplitude entry
// with empty sigma and coeff 1).  A default-constructed FramedSuperposition has an EMPTY container
// (χ=0); this seeds the single branch, matching FramedSuperposition::from_css(|0>).
inline FramedSuperposition zero_state(int n) {
    FramedSuperposition s(n);                 // frame = identity (Z-stabilised |0>), free empty
    s.entries().clear();
    s.entries().emplace_back(qeccore::AmpKey(0), std::complex<double>(1.0, 0.0));
    s.sync_alpha_k();
    return s;
}

// D2 — TENSOR two FramedSuperpositions:  out = A ⊗ B  (A on wires [0,A.n), B on wires
// [A.n, A.n+B.n)).  The frame is the DIRECT SUM of the two symplectic frames (block-diagonal
// tableau — B's rows shifted by A.n in column space); eps concatenates; the distinguished
// (magic/logical) `free` rows concatenate (B's shifted by A.n); the amplitude container is the
// OUTER PRODUCT (χ multiplies) — sigma = (A's sigma over A.free) ++ (B's sigma over B.free),
// coeff = c_A · c_B.  This is a genuine general engine primitive: it composes a carried patch of
// ANY χ with a freshly-built magic block of ANY χ.  (Verified byte-identical to build_bare_state
// on the all-|0> instance at the injection site — see the composition guard.)
inline FramedSuperposition tensor_framed(const FramedSuperposition& A,
                                         const FramedSuperposition& B) {
    const int nA = A.n(), nB = B.n();
    FramedSuperposition out(A);              // A verbatim (frame + eps + free + amplitudes)
    grow_framed(out, nB);                    // append nB fresh |0> columns (identity rows N_A+q)
    // Splice B's frame rows into columns [nA, nA+nB): row (nA+q) becomes B.Xrow[q]/Zrow[q] with
    // every column bit shifted up by nA (B is on its own qubits; no overlap with A's columns).
    auto shift_into = [&](const Pauli& src, Pauli& dst) {
        dst.phase = src.phase;
        for (int q = 0; q < nB; ++q) {
            if (src.xbit(q)) dst.setx(nA + q);
            if (src.zbit(q)) dst.setz(nA + q);
        }
    };
    for (int q = 0; q < nB; ++q) {
        Pauli xr(nA + nB), zr(nA + nB);
        shift_into(B.U.Xrow[(size_t)q], xr);
        shift_into(B.U.Zrow[(size_t)q], zr);
        out.U.Xrow[(size_t)(nA + q)] = xr;
        out.U.Zrow[(size_t)(nA + q)] = zr;
        out.eps[(size_t)(nA + q)] = B.eps[(size_t)q];
    }
    out.U.invalidate_dual();
    // free: A.free (indices < nA, unchanged) ++ (B.free shifted by nA).
    const int kA = (int)A.free.size();
    qeccore::check_amp_key_width((long long)kA + (long long)B.free.size(), "tensor");
    for (int f : B.free) out.free.push_back(nA + f);
    out.sync_alpha_k();
    // amplitudes: outer product.  out entry m*chiB + i has sigma = A[m].sigma ++ B[i].sigma
    // (ordered by out.free = A.free ++ shifted B.free) and coeff = A[m].c · B[i].c.
    const auto& ea = A.entries();
    const auto& eb = B.entries();
    const int kB = (int)B.free.size();
    FramedSuperposition::EntryVec prod;
    prod.reserve(ea.size() * eb.size());
    for (const auto& a : ea)
        for (const auto& b : eb) {
            // sigma = A's key ++ B's key (B's bits shifted up by kA; kA + kB <= 64 checked above)
            const qeccore::AmpKey sig = a.first | (kB == 0 ? 0 : (b.first << kA));
            prod.emplace_back(sig, a.second * b.second);
        }
    out.entries() = std::move(prod);
    out.sync_alpha_k();
    return out;
}

// Apply the Clifford Gate instructions of a (measurement/noise-inclusive) deferred circuit to a
// FramedSuperposition, in stream order — turning an INITIAL state into the PRE-MEASUREMENT state
// the twirl sampler's bare contract requires (build_bare_state does this from |0>; injection does
// it from the carried input). Returns "" on success, or a loud message on an unsupported op:
//   * a non-Clifford gate (T/CS/CCZ/CH) — fresh-magic guard already refuses these upstream;
//   * a mid-stream Reset (build_bare_state's provenance would differ) — out of scope.
inline std::string apply_stream_cliffords(FramedSuperposition& fs, const Circuit& deferred) {
    for (const Instr& ins : deferred.stream) {
        if (ins.kind == Instr::Kind::Reset) {
            // A reset re-initialises to |0>. Our fresh wires arrive |0>; a reset on them is a
            // no-op. A reset on any other wire (a carried/derived wire) is a genuine mid-stream
            // reinitialisation we do not model here — refuse loudly.
            for (int q : ins.qubits) {
                Pauli zq(fs.n());
                if (q >= 0 && q < fs.n()) zq.z[(size_t)(q >> 6)] |= (1ull << (q & 63));
                if (std::abs(fs.expectation(zq) - 1.0) > 1e-9)
                    return "state-level injection: mid-stream RESET on a non-|0> wire is out of "
                           "scope (Clifford consumer stages only)";
            }
            continue;
        }
        if (ins.kind != Instr::Kind::Gate) continue;   // Noise/Measure/Observable/feedback: not state ops
        uint8_t kind;
        switch (ins.gate) {
            case GateKind::H:   kind = 0; break;
            case GateKind::S:   kind = 1; break;
            case GateKind::SDG: kind = 2; break;
            case GateKind::X:   kind = 3; break;
            case GateKind::Y:   kind = 4; break;
            case GateKind::Z:   kind = 5; break;
            case GateKind::CX:  kind = 6; break;
            case GateKind::CZ:  kind = 7; break;
            default:
                return "state-level injection: non-Clifford gate on a consumer stage is out of "
                       "scope (only Clifford consumer stages carry patches)";
        }
        const int a = ins.targets.empty() ? -1 : ins.targets[0];
        const int b = ins.targets.size() > 1 ? ins.targets[1] : -1;
        fs.apply_clifford(kind, a, b);
    }
    return "";
}

// D2 — like apply_stream_cliffords, but ONLY bakes gates that TOUCH a port wire (`is_port[w]`).
// Gates supported entirely on FRESH (non-port) wires are SKIPPED — they are already baked into the
// fresh block (built by build_bare_state on the fresh-only sub-circuit).  A non-Clifford gate that
// touches a port wire is the CROSSING case (front-pushed magic on a carried wire) — returned as a
// precise loud message.  A non-Clifford gate on fresh wires only is skipped (already in the block).
// Wire indices here are in the COMBINED space (post-remap): port wires = src_wires, fresh = the
// appended |0> columns.  `is_port` is indexed over the combined width.
// Harness continuation (2026-10-07): which instructions of the consumer stream are PRE-BAKED
// into the fresh block (built by build_bare_state on the fresh-only sub-circuit) and which are
// applied IN STREAM ORDER by the port baker.  A fresh-only gate commutes to the front only while
// every wire it touches is still UNCONTAMINATED — no earlier instruction joined that wire to a
// port wire or to a contaminated wire.  Before this rule every fresh-only gate was pre-baked,
// which is unsound once a fresh ancilla has interacted with the port (a second syndrome round's
// deferred H after its CX): the exactness guard refused such stages.  Now those gates are baked in
// order; the guard still certifies the split.  Stages whose fresh-only gates all precede their
// contamination (every previously accepted stage) classify identically — byte-identical.
inline std::vector<uint8_t> prebake_mask(const Circuit& deferred, const std::vector<uint8_t>& is_port) {
    const size_t nw = is_port.size();
    std::vector<uint8_t> contaminated(nw, 0);
    std::vector<uint8_t> mask(deferred.stream.size(), 0);
    auto hot = [&](int q) {
        return q >= 0 && q < (int)nw && (is_port[(size_t)q] || contaminated[(size_t)q]);
    };
    for (size_t i = 0; i < deferred.stream.size(); ++i) {
        const Instr& ins = deferred.stream[i];
        if (ins.kind != Instr::Kind::Gate) continue;
        bool any_hot = false;
        for (int q : ins.targets) if (hot(q)) any_hot = true;
        if (!any_hot) { mask[i] = 1; continue; }         // fresh-only, uncontaminated: pre-baked
        for (int q : ins.targets)
            if (q >= 0 && q < (int)nw && !is_port[(size_t)q]) contaminated[(size_t)q] = 1;
    }
    return mask;
}

inline std::string apply_stream_cliffords_ports(FramedSuperposition& fs, const Circuit& deferred,
                                                const std::vector<uint8_t>& is_port) {
    auto touches_port = [&](const Instr& ins) {
        for (int q : ins.targets)
            if (q >= 0 && q < (int)is_port.size() && is_port[(size_t)q]) return true;
        return false;
    };
    const std::vector<uint8_t> prebaked = prebake_mask(deferred, is_port);
    for (size_t i = 0; i < deferred.stream.size(); ++i) {
        const Instr& ins = deferred.stream[i];
        if (ins.kind == Instr::Kind::Reset) {
            if (!touches_port(ins)) continue;   // fresh-wire reset: fresh block arrives |0>
            for (int q : ins.qubits) {
                Pauli zq(fs.n());
                if (q >= 0 && q < fs.n()) zq.z[(size_t)(q >> 6)] |= (1ull << (q & 63));
                if (std::abs(fs.expectation(zq) - 1.0) > 1e-9)
                    return "state-level injection: mid-stream RESET on a non-|0> carried wire is "
                           "out of scope (Clifford consumer stages only)";
            }
            continue;
        }
        if (ins.kind != Instr::Kind::Gate) continue;
        if (prebaked[i]) continue;              // fresh-only, uncontaminated: in the block already
        uint8_t kind;
        switch (ins.gate) {
            case GateKind::H:   kind = 0; break;
            case GateKind::S:   kind = 1; break;
            case GateKind::SDG: kind = 2; break;
            case GateKind::X:   kind = 3; break;
            case GateKind::Y:   kind = 4; break;
            case GateKind::Z:   kind = 5; break;
            case GateKind::CX:  kind = 6; break;
            case GateKind::CZ:  kind = 7; break;
            default:
                return "state-level injection: front-pushed magic supported on carried wires — "
                       "a non-Clifford gate entangles a fresh magic qubit with a carried port "
                       "qubit; prepare the magic state upstream of the carried-fresh entanglement "
                       "(prep-then-teleport), then this composes";
        }
        const int a = ins.targets.empty() ? -1 : ins.targets[0];
        const int b = ins.targets.size() > 1 ? ins.targets[1] : -1;
        fs.apply_clifford(kind, a, b);
    }
    return "";
}

// D2 — build the FRESH-ONLY sub-circuit: the consumer deferred stream restricted to gates whose
// support lies entirely on FRESH (non-port) wires, with a Reset on each fresh wire prepended so
// build_bare_state sees them as |0>.  Wire indices are the ORIGINAL consumer indices (0..Nc-1);
// build_bare_state consumes width `Nc` and reports χ = 2^k for the k fresh magic states.  Port
// wires appear in NO instruction (their gates are the crossing/port gates, excluded here and baked
// later), so the fresh block is a genuine tensor factor on the fresh wires.
inline Circuit fresh_only_subcircuit(const Circuit& deferred, const std::vector<uint8_t>& is_port,
                                     int Nc) {
    Circuit out;
    out.n = Nc;
    for (int w = 0; w < Nc; ++w)
        if (w < (int)is_port.size() && !is_port[(size_t)w]) {
            Instr r; r.kind = Instr::Kind::Reset; r.qubits = {w};
            out.stream.push_back(r);
        }
    const std::vector<uint8_t> prebaked = prebake_mask(deferred, is_port);
    for (size_t i = 0; i < deferred.stream.size(); ++i) {
        const Instr& ins = deferred.stream[i];
        if (ins.kind != Instr::Kind::Gate) continue;   // magic + Cliffords are Gate instrs
        if (!prebaked[i]) continue;                    // port / post-contamination gates baked later
        out.stream.push_back(ins);
    }
    return out;
}

// D2 — STATE-BLIND crossing check (the §6.1 narrow refusal, made general).  A fresh magic state must
// be prepared BEFORE it entangles with any carried port qubit (prep-then-teleport).  We propagate a
// "port-tainted" mark along the stream: port wires start tainted; any multi-qubit gate spreads taint
// to all its wires if ANY is tainted (they become correlated with the carried state).  A non-Clifford
// gate acting on a tainted wire means its front-pushed magic support reaches the carried state — the
// injection cannot factor it out.  Returns the precise loud message on that case, else "".  This is
// conservative (state-blind), so it never depends on the carried state collapsing the magic away
// (the all-|0> exactness oracle alone would MISS a control-in-superposition crossing).
inline std::string check_no_magic_on_carried(const Circuit& deferred,
                                             const std::vector<uint8_t>& is_port_orig, int Nc) {
    std::vector<uint8_t> tainted((size_t)(Nc > 0 ? Nc : 1), 0);
    for (int w = 0; w < Nc; ++w)
        if (w < (int)is_port_orig.size() && is_port_orig[(size_t)w]) tainted[(size_t)w] = 1;
    auto is_nonclifford = [](GateKind g) {
        return !(g == GateKind::H || g == GateKind::S || g == GateKind::SDG ||
                 g == GateKind::X || g == GateKind::Y || g == GateKind::Z ||
                 g == GateKind::CX || g == GateKind::CZ);
    };
    for (const Instr& ins : deferred.stream) {
        if (ins.kind != Instr::Kind::Gate) continue;
        bool any_tainted = false;
        for (int q : ins.targets)
            if (q >= 0 && q < Nc && tainted[(size_t)q]) { any_tainted = true; break; }
        if (is_nonclifford(ins.gate) && any_tainted)
            return "state-level injection: front-pushed magic supported on carried wires — a "
                   "non-Clifford gate acts on a qubit already entangled with a carried port; "
                   "prepare the magic state upstream of the carried-fresh entanglement "
                   "(prep-then-teleport), then this composes";
        if (any_tainted)                               // a multi-qubit gate spreads the taint
            for (int q : ins.targets) if (q >= 0 && q < Nc) tainted[(size_t)q] = 1;
    }
    return "";
}

// D2 — relabel a FramedSuperposition's qubit wires by the INJECTIVE map `map` (source wire w ->
// map[w] in a width-`new_n` space).  Frame row w moves to row map[w]; each row's column bits move
// w -> map[w]; eps and `free` follow.  Wires of the new space not hit by `map` are |0> (identity
// Z-stabiliser).  Used ONLY by the compile-time composition exactness guard (compare the mapped
// all-|0> oracle against the composed-with-|0> instance).
inline FramedSuperposition remap_state_wires(const FramedSuperposition& s,
                                             const std::vector<int>& map, int new_n) {
    const int old_n = s.n();
    FramedSuperposition out(new_n);   // all-|0>: Xrow[a]=X_a, Zrow[a]=Z_a, eps 0, free empty
    auto move_pauli = [&](const Pauli& src) {
        Pauli dst(new_n);
        dst.phase = src.phase;
        for (int q = 0; q < old_n; ++q) {
            const int nq = (q < (int)map.size()) ? map[(size_t)q] : q;
            if (src.xbit(q)) dst.setx(nq);
            if (src.zbit(q)) dst.setz(nq);
        }
        return dst;
    };
    for (int a = 0; a < old_n; ++a) {
        const int na = (a < (int)map.size()) ? map[(size_t)a] : a;
        out.U.Xrow[(size_t)na] = move_pauli(s.U.Xrow[(size_t)a]);
        out.U.Zrow[(size_t)na] = move_pauli(s.U.Zrow[(size_t)a]);
        out.eps[(size_t)na] = s.eps[(size_t)a];
    }
    out.U.invalidate_dual();
    for (int f : s.free) out.free.push_back((f < (int)map.size()) ? map[(size_t)f] : f);
    out.sync_alpha_k();
    // amplitudes: sigma order follows s.free order, which maps 1:1 to out.free order (same
    // sequence, relabelled), so the (sigma, coeff) entries copy verbatim.
    out.entries() = s.entries();
    out.sync_alpha_k();
    return out;
}

// Remap every qubit index in a Circuit's instruction stream through `map` (map[w] = new index),
// and set the new width. Record indices (Measure order) are UNCHANGED — only wire indices move.
inline void remap_circuit_wires(Circuit& c, const std::vector<int>& map, int new_n) {
    auto mp = [&](int q) -> int { return (q >= 0 && q < (int)map.size()) ? map[(size_t)q] : q; };
    for (Instr& ins : c.stream) {
        for (int& q : ins.targets) q = mp(q);
        for (int& q : ins.qubits) q = mp(q);
        for (PauliTerm& t : ins.obs) t.qubit = mp(t.qubit);
    }
    c.n = new_n;
}

// Harness continuation (2026-10-07): build a FramedSuperposition from EXPLICIT rows — the
// port RESTRICTION of a carried state, re-synthesised by adaptq.choi_segments from exact reads
// (port stabilizer group + destabilizers + the free sector's density matrix).  Pure data in:
//   n; zrows/xrows: n rows of (xbits[n], zbits[n], phase) — Zrow[a] = i^phase X^x Z^z (the
//   signed generator g_a with eps folded separately), Xrow[a] its conjugate partner;
//   eps[a]; free: the distinguished (logical / magic) row indices; entries: (sigma over free,
//   coeff).  Validated: a symplectic frame (<Xrow_i, Zrow_j> = delta_ij, Xrows commute, Zrows
//   commute), Hermitian rows (phase parity == #Y sites parity), sigma widths, unit norm.
inline FramedSuperposition framed_from_rows(
        int n, const std::vector<std::vector<uint8_t>>& zx, const std::vector<std::vector<uint8_t>>& zz,
        const std::vector<int>& zph, const std::vector<std::vector<uint8_t>>& xx,
        const std::vector<std::vector<uint8_t>>& xz, const std::vector<int>& xph,
        const std::vector<uint8_t>& eps, const std::vector<int>& free,
        const std::vector<std::pair<std::vector<uint8_t>, std::complex<double>>>& entries) {
    if (n <= 0) throw std::invalid_argument("from_rows: n must be positive");
    auto row = [&](const std::vector<uint8_t>& x, const std::vector<uint8_t>& z, int ph, const char* what) {
        if ((int)x.size() != n || (int)z.size() != n)
            throw std::invalid_argument(std::string("from_rows: ") + what + " row width != n");
        Pauli P(n);
        int ny = 0;
        for (int q = 0; q < n; ++q) {
            if (x[(size_t)q]) P.setx(q);
            if (z[(size_t)q]) P.setz(q);
            if (x[(size_t)q] && z[(size_t)q]) ++ny;
        }
        P.phase = ph & 3;
        if (((P.phase ^ ny) & 1) != 0)
            throw std::invalid_argument(std::string("from_rows: ") + what + " row is not Hermitian");
        return P;
    };
    if ((int)zx.size() != n || (int)zz.size() != n || (int)zph.size() != n ||
        (int)xx.size() != n || (int)xz.size() != n || (int)xph.size() != n || (int)eps.size() != n)
        throw std::invalid_argument("from_rows: expected n rows of each kind and n eps");
    FramedSuperposition fs(n);
    for (int a = 0; a < n; ++a) {
        fs.U.Zrow[(size_t)a] = row(zx[(size_t)a], zz[(size_t)a], zph[(size_t)a], "Z");
        fs.U.Xrow[(size_t)a] = row(xx[(size_t)a], xz[(size_t)a], xph[(size_t)a], "X");
        fs.eps[(size_t)a] = eps[(size_t)a] & 1;
    }
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            const bool xz_anti = Pauli::anticommute_bit(fs.U.Xrow[(size_t)a], fs.U.Zrow[(size_t)b]) != 0;
            if (xz_anti != (a == b))
                throw std::invalid_argument("from_rows: <Xrow_" + std::to_string(a) + ", Zrow_" +
                                            std::to_string(b) + "> != delta");
            if (a < b) {
                if (Pauli::anticommute_bit(fs.U.Xrow[(size_t)a], fs.U.Xrow[(size_t)b]))
                    throw std::invalid_argument("from_rows: Xrows anticommute");
                if (Pauli::anticommute_bit(fs.U.Zrow[(size_t)a], fs.U.Zrow[(size_t)b]))
                    throw std::invalid_argument("from_rows: Zrows anticommute");
            }
        }
    fs.U.invalidate_dual();
    std::vector<uint8_t> seen((size_t)n, 0);
    for (int f : free) {
        if (f < 0 || f >= n || seen[(size_t)f]) throw std::invalid_argument("from_rows: bad free index");
        seen[(size_t)f] = 1;
        fs.free.push_back(f);
    }
    const size_t k = free.size();
    qeccore::check_amp_key_width((long long)k, "from_rows");
    double norm = 0.0;
    fs.entries().clear();
    for (const auto& e : entries) {
        if (e.first.size() != k) throw std::invalid_argument("from_rows: sigma width != len(free)");
        fs.entries().emplace_back(qeccore::pack_key(e.first, "from_rows"), e.second);
        norm += std::norm(e.second);
    }
    if (fs.entries().empty() || std::abs(norm - 1.0) > 1e-9)
        throw std::invalid_argument("from_rows: amplitudes must be non-empty and unit-norm");
    fs.sync_alpha_k();
    return fs;
}

}  // namespace qeccore
