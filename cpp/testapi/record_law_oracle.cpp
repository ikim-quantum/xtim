// record_law_oracle — Stage S1 exact oracle (X). See record_law_oracle.hpp for the mathematics.
// TEST / ORACLE infrastructure: nothing on a production path includes or calls this TU.
#include "record_law_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "qeccore/clifford_tableau.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/sampler.hpp"      // noiseless_reference_records
#include "qeccore/stim_parse.hpp"

namespace qeccore {
namespace record_oracle {

namespace {

[[noreturn]] void refuse(const std::string& why) {
    throw std::invalid_argument("record_law_oracle: " + why);
}

inline bool bit(const std::vector<uint64_t>& w, int q) { return (w[(size_t)q >> 6] >> (q & 63)) & 1ULL; }

// Hermitian single-wire read B_r = X / Y / Z on wire w (Y = i X Z: phase 1).
Pauli read_pauli(int n, int basis, int w) {
    Pauli P(n);
    if (basis == 0) P.setx(w);
    else if (basis == 2) P.setz(w);
    else { P.setx(w); P.setz(w); P.phase = 1; }
    return P;
}

// A T-type factor exp(-i(pi/8) s * Z_{qs}) with sign s = +1 / -1: axis s * prod Z.
Pauli zaxis(int n, std::initializer_list<int> qs, int sign) {
    Pauli A(n);
    for (int q : qs) A.setz(q);
    A.phase = sign > 0 ? 0 : 2;
    return A;
}

void conj_clifford(Pauli& M, GateKind g, int a, int b) {
    switch (g) {
        case GateKind::X: CliffordTableau::conj_x(M, a); break;
        case GateKind::Z: CliffordTableau::conj_z(M, a); break;
        case GateKind::Y: CliffordTableau::conj_z(M, a); CliffordTableau::conj_x(M, a); break;
        case GateKind::S: CliffordTableau::conj_s(M, a); break;
        case GateKind::SDG: CliffordTableau::conj_sdg(M, a); break;
        case GateKind::H: CliffordTableau::conj_h(M, a); break;
        case GateKind::CX: CliffordTableau::conj_cx(M, a, b); break;
        case GateKind::CZ: CliffordTableau::conj_cz(M, a, b); break;
        default: throw std::logic_error("record_law_oracle: conj_clifford on a non-Clifford gate");
    }
}

// The twirl-sampler normalization preset (xtim_py.cpp normalize_for_twirl_sampler), replicated so
// the oracle reads the same deferred circuit the engine samples.
NormalizeResult normalize_like_twirl_sampler(const Circuit& user) {
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    pol.hoist_feedback_on_class_exit = true;
    pol.zero_noise_transparent = true;
    return normalize(user, pol);
}

// Categorical outcomes of one deferred Noise instruction, split into independent sites.
void add_sites(Program& prog, const Instr& ins, int stream_index, int op_pos) {
    auto push = [&](std::vector<int> qs, std::vector<std::vector<uint8_t>> outs, std::vector<double> ps) {
        NoiseSite s;
        s.stream_index = stream_index;
        s.op_pos = op_pos;
        s.qubits = std::move(qs);
        double tot = 0.0;
        for (double p : ps) tot += p;
        s.outcomes.push_back(std::vector<uint8_t>(s.qubits.size(), 0));
        s.probs.push_back(1.0 - tot);
        for (size_t i = 0; i < outs.size(); ++i)
            if (ps[i] > 0.0) { s.outcomes.push_back(outs[i]); s.probs.push_back(ps[i]); }
        prog.sites.push_back(std::move(s));
    };
    const auto& p = ins.probs;
    switch (ins.channel) {
        case NoiseChannel::X_ERROR: case NoiseChannel::Y_ERROR: case NoiseChannel::Z_ERROR: {
            const uint8_t c = ins.channel == NoiseChannel::X_ERROR ? 1 : ins.channel == NoiseChannel::Y_ERROR ? 2 : 3;
            for (int q : ins.qubits) push({q}, {{c}}, {p.at(0)});
            break;
        }
        case NoiseChannel::DEPOLARIZE1:
            for (int q : ins.qubits) push({q}, {{1}, {2}, {3}}, {p.at(0) / 3, p.at(0) / 3, p.at(0) / 3});
            break;
        case NoiseChannel::PAULI_CHANNEL_1:
            for (int q : ins.qubits) push({q}, {{1}, {2}, {3}}, {p.at(0), p.at(1), p.at(2)});
            break;
        case NoiseChannel::DEPOLARIZE2: case NoiseChannel::PAULI_CHANNEL_2: {
            if (ins.qubits.size() % 2) refuse("odd two-qubit noise target list");
            for (size_t i = 0; i < ins.qubits.size(); i += 2) {
                std::vector<std::vector<uint8_t>> outs;
                std::vector<double> ps;
                for (int t = 0; t < 15; ++t) {
                    outs.push_back({(uint8_t)pc2_first_code(t), (uint8_t)pc2_second_code(t)});
                    ps.push_back(ins.channel == NoiseChannel::DEPOLARIZE2 ? p.at(0) / 15 : p.at((size_t)t));
                }
                push({ins.qubits[i], ins.qubits[i + 1]}, outs, ps);
            }
            break;
        }
        case NoiseChannel::CORRELATED: {
            const std::vector<double> q = corr_outcome_probs(ins);
            std::vector<std::vector<uint8_t>> outs;
            for (int t = 0; t < corr_terms(ins); ++t) {
                std::vector<uint8_t> o(ins.qubits.size(), 0);
                for (size_t i = 0; i < ins.qubits.size(); ++i) o[i] = (uint8_t)corr_code(ins, t, (int)i);
                outs.push_back(o);
            }
            push(ins.qubits, outs, q);
            break;
        }
    }
}

// k-qubit exact Pauli on the free register: i^ph X^a Z^b (a, b bit d <-> free row d).
struct FreePauli {
    uint64_t a = 0, b = 0;
    int ph = 0;
};
// Sign of i^ph X^a Z^b relative to its Hermitian canonical form i^{a.b} X^a Z^b (Hermitian input).
int herm_sign(const FreePauli& p) {
    return (((p.ph - __builtin_popcountll(p.a & p.b)) & 3) == 2) ? 1 : 0;
}
FreePauli fmul(const FreePauli& p, const FreePauli& q) {
    FreePauli r;
    r.a = p.a ^ q.a;
    r.b = p.b ^ q.b;
    r.ph = (p.ph + q.ph + 2 * (__builtin_popcountll(p.b & q.a) & 1)) & 3;
    return r;
}

// <phi| i^ph X^a Z^b |phi> over the free-register support (exact formula; real for Hermitian).
std::complex<double> free_expectation(const Program& prog,
                                      const std::unordered_map<uint64_t, std::complex<double>>& amp,
                                      const FreePauli& L) {
    std::complex<double> s = 0.0;
    for (const auto& e : prog.alpha) {
        const uint64_t x = e.first;
        auto it = amp.find(x ^ L.a);
        if (it == amp.end()) continue;
        const double sg = (__builtin_popcountll(L.b & x) & 1) ? -1.0 : 1.0;
        s += std::conj(it->second) * e.second * sg;
    }
    static const std::complex<double> ip[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    return ip[L.ph & 3] * s;
}

}  // namespace

Program compile(const std::string& text) {
    Program prog;
    ParsedStim ps = parse_stim_circuit(text);
    if (!ps.ok()) {
        std::string err = "parse failed";
        for (const auto& e : ps.errors) err += "\n  line " + std::to_string(e.line) + ": " + e.message;
        refuse(err);
    }
    if (circuit_has_branches(ps.circuit)) refuse("IF blocks are not supported");
    for (const InputPort& p : ps.circuit.inputs)
        if (p.kind == InputPort::Kind::Qubits) { prog.input_qubits = p.qubits; break; }
    NormalizeResult nr = normalize_like_twirl_sampler(ps.circuit);
    if (nr.feedback_rejected) refuse("feedback survived coherentization");
    for (const Instr& ins : nr.coherent.stream)
        if (ins.kind == Instr::Kind::Measure) {
            prog.rec_flip.push_back(ins.readout_flip_p);
            prog.rec_invert.push_back(ins.invert ? 1 : 0);
        }
    {
        size_t j = 0;
        for (const Instr& ins : nr.normalized.stream)
            if (ins.kind == Instr::Kind::Measure) {
                if (j < prog.rec_invert.size() && ins.invert) prog.rec_invert[j] ^= 1;
                ++j;
            }
    }
    prog.deferred = std::move(nr.normalized);
    const int n = prog.deferred.n;
    prog.n = n;
    for (const auto& r : nr.map.terminal_reads) prog.reads.push_back({(int)r.second, r.first});
    if (prog.reads.size() != prog.rec_invert.size()) refuse("record count mismatch");

    FramedBareState bs = build_bare_state_framed(prog.deferred);
    if (bs.rejected) refuse("bare state rejected: " + bs.reject_reason);
    prog.bare = std::move(bs.state);
    prog.k = prog.bare.k();
    if (prog.k > 64) refuse("more than 64 free rows");
    // The bare builder puts every reference sign in the frame rows (eps = 0), so the frame state is
    // sum_x alpha_x |emb x> exactly. Checked, not assumed.
    for (uint8_t e : prog.bare.eps)
        if (e) throw std::logic_error("record_law_oracle: bare state with a nonzero eps (builder invariant)");
    prog.fixed_rows.assign((size_t)(n + 63) / 64, 0);
    {
        std::vector<uint8_t> isfree((size_t)n, 0);
        for (int f : prog.bare.free) isfree[(size_t)f] = 1;
        for (int a = 0; a < n; ++a)
            if (!isfree[(size_t)a]) prog.fixed_rows[(size_t)a >> 6] |= 1ULL << (a & 63);
    }
    for (const auto& e : prog.bare.entries()) {
        uint64_t x = 0;
        for (int d = 0; d < prog.k; ++d) if (qeccore::key_bit(e.first, d)) x |= 1ULL << d;
        prog.alpha.push_back({x, e.second});
    }
    prog.bare.U.ensure_dual();

    // ── ops + terms + noise sites (one walk of the deferred stream) ──
    auto term = [&](const Pauli& A, int si) {
        prog.ops.push_back({1, GateKind::X, -1, -1, (int)prog.local_axes.size()});
        prog.local_axes.push_back(A);
        prog.term_stream_index.push_back(si);
    };
    auto cliff = [&](GateKind g, int a, int b = -1) { prog.ops.push_back({0, g, a, b, -1}); };
    // A read is terminal iff nothing after it touches its wire (reads on other wires commute).
    std::vector<uint8_t> read_done((size_t)n, 0);
    auto untouched = [&](const std::vector<int>& qs) {
        for (int q : qs) if (read_done[(size_t)q]) refuse("an operation on a wire after its read");
    };
    for (size_t si = 0; si < prog.deferred.stream.size(); ++si) {
        const Instr& ins = prog.deferred.stream[si];
        const int s = (int)si;
        switch (ins.kind) {
            case Instr::Kind::Gate: {
                untouched(ins.targets);
                const auto& t = ins.targets;
                switch (ins.gate) {
                    case GateKind::T: term(zaxis(n, {t[0]}, +1), s); break;
                    case GateKind::CS:   // diag(1,1,1,i) = exp(i pi/8 (1 - Za - Zb + ZaZb))
                        term(zaxis(n, {t[0]}, +1), s); term(zaxis(n, {t[1]}, +1), s);
                        term(zaxis(n, {t[0], t[1]}, -1), s);
                        break;
                    case GateKind::CCZ:  // exp(i pi |111><111|) = exp(i pi/8 (1 - sum Z + sum ZZ - ZZZ))
                        term(zaxis(n, {t[0]}, +1), s); term(zaxis(n, {t[1]}, +1), s);
                        term(zaxis(n, {t[2]}, +1), s);
                        term(zaxis(n, {t[0], t[1]}, -1), s); term(zaxis(n, {t[0], t[2]}, -1), s);
                        term(zaxis(n, {t[1], t[2]}, -1), s);
                        term(zaxis(n, {t[0], t[1], t[2]}, +1), s);
                        break;
                    case GateKind::CH: { // CH = (I x V) CZ (I x V^dag), V = S H T H S^dag (V Z V^dag = H)
                        const int c = t[0], q = t[1];
                        // V^dag = S H T^dag H S^dag: time order S^dag, H, T^dag, H, S
                        cliff(GateKind::SDG, q); cliff(GateKind::H, q);
                        term(zaxis(n, {q}, -1), s);
                        cliff(GateKind::H, q); cliff(GateKind::S, q);
                        cliff(GateKind::CZ, c, q);
                        // V = S H T H S^dag: time order S^dag, H, T, H, S
                        cliff(GateKind::SDG, q); cliff(GateKind::H, q);
                        term(zaxis(n, {q}, +1), s);
                        cliff(GateKind::H, q); cliff(GateKind::S, q);
                        break;
                    }
                    default: cliff(ins.gate, t[0], t.size() > 1 ? t[1] : -1); break;
                }
                break;
            }
            case Instr::Kind::Noise:
                untouched(ins.qubits);
                add_sites(prog, ins, s, (int)prog.ops.size());
                break;
            case Instr::Kind::Measure:
                untouched(ins.qubits);
                for (int q : ins.qubits) read_done[(size_t)q] = 1;
                break;
            case Instr::Kind::Observable: break;   // PAULI_EXPECTATION: non-collapsing, no record
            default: refuse("unsupported deferred instruction kind");
        }
    }
    // End-of-circuit term axes: push each local axis through every later op. In class the terms
    // commute, so a later term factor leaves it unchanged — asserted (non-commuting magic refuses).
    const int T = (int)prog.local_axes.size();
    prog.terms.resize((size_t)T);
    {
        std::vector<int> pos((size_t)T, -1);
        for (size_t i = 0; i < prog.ops.size(); ++i) if (prog.ops[i].kind == 1) pos[(size_t)prog.ops[i].term] = (int)i;
        for (int j = 0; j < T; ++j) {
            Pauli B = prog.local_axes[(size_t)j];
            for (size_t i = (size_t)pos[(size_t)j] + 1; i < prog.ops.size(); ++i) {
                const Op& o = prog.ops[i];
                if (o.kind == 0) conj_clifford(B, o.g, o.a, o.b);
                else {
                    // the other factor's axis at this time is its local axis
                    if (Pauli::anticommute_bit(B, prog.local_axes[(size_t)o.term]))
                        refuse("non-commuting magic terms (outside the one-layer class)");
                }
            }
            prog.terms[(size_t)j] = std::move(B);
        }
    }
    prog.wire_terms.assign((size_t)n, {});
    for (int j = 0; j < T; ++j)
        for (int w = 0; w < n; ++w)
            if (prog.terms[(size_t)j].xbit(w) || prog.terms[(size_t)j].zbit(w)) prog.wire_terms[(size_t)w].push_back(j);

    // ── record columns ──
    auto add_col = [&](int kind, int index, const std::vector<int>& recs) {
        Column c;
        c.kind = kind;
        c.index = index;
        std::vector<int> r = recs;
        std::sort(r.begin(), r.end());
        for (size_t i = 0; i < r.size();) {   // duplicates cancel in a parity
            size_t j = i;
            while (j < r.size() && r[j] == r[i]) ++j;
            if ((j - i) & 1) c.recs.push_back(r[i]);
            i = j;
        }
        for (int x : c.recs) if (x < 0 || x >= (int)prog.reads.size()) refuse("record index out of range");
        prog.columns.push_back(std::move(c));
    };
    for (size_t d = 0; d < ps.detectors.size(); ++d) add_col(0, (int)d, ps.detectors[d]);
    {
        int mx = -1;
        for (const auto& kv : ps.observables) mx = std::max(mx, kv.first);
        for (int o = 0; o <= mx; ++o) {
            auto it = ps.observables.find(o);
            add_col(1, o, it == ps.observables.end() ? std::vector<int>{} : it->second);
        }
    }
    {
        std::map<int, std::vector<int>> dec;
        for (const auto& kv : ps.decisions) {
            if (dec.count(kv.first)) refuse("two DECISION lines with one index");
            dec[kv.first] = kv.second;
        }
        int mx = -1;
        for (const auto& kv : dec) mx = std::max(mx, kv.first);
        for (int d = 0; d <= mx; ++d) {
            auto it = dec.find(d);
            add_col(2, d, it == dec.end() ? std::vector<int>{} : it->second);
        }
    }
    for (const Column& c : prog.columns) {          // reads of one column sit on distinct wires
        std::vector<int> ws;
        for (int r : c.recs) ws.push_back(prog.reads[(size_t)r].second);
        std::sort(ws.begin(), ws.end());
        if (std::adjacent_find(ws.begin(), ws.end()) != ws.end())
            throw std::logic_error("record_law_oracle: two reads of one column on one wire");
    }
    return prog;
}

void ensure_reference(Program& prog) {
    if (prog.reference_ready) return;
    prog.reference = noiseless_reference_records(prog.bare, prog.deferred, prog.reads, prog.rec_invert, nullptr);
    prog.emission_offset.clear();
    for (const Column& c : prog.columns) {
        int off = 0;
        for (int r : c.recs) {
            off ^= prog.rec_invert[(size_t)r];
            if (c.kind < 2) off ^= prog.reference[(size_t)r];
        }
        prog.emission_offset.push_back((uint8_t)(off & 1));
    }
    prog.reference_ready = true;
}

TermResidual residual(const Program& prog, const std::vector<Fault>& faults, const Pauli* frame) {
    std::vector<size_t> order(faults.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return faults[a].op_pos < faults[b].op_pos; });
    TermResidual R;
    R.P = Pauli(prog.n);
    R.S.assign(prog.terms.size(), 0);
    if (frame) R.P = Pauli::multiply(*frame, R.P);
    size_t fi = 0;
    for (size_t i = 0; i <= prog.ops.size(); ++i) {
        // a fault inserted here acts AFTER everything so far: R <- F * R (F leftmost; the Pauli
        // part moves no rotation, every term factor of R so far sits to the right of P).
        while (fi < order.size() && faults[order[fi]].op_pos == (int)i) {
            const Pauli& F = faults[order[fi]].P;
            if (F.n != prog.n) refuse("fault Pauli has the wrong width");
            R.P = Pauli::multiply(F, R.P);
            ++fi;
        }
        if (i == prog.ops.size()) break;
        const Op& o = prog.ops[i];
        if (o.kind == 0) conj_clifford(R.P, o.g, o.a, o.b);
        else if (Pauli::anticommute_bit(R.P, prog.local_axes[(size_t)o.term]))
            R.S[(size_t)o.term] ^= 1;   // exp(-i pi/8 A) P exp(i pi/8 A) = P * rot(A)
    }
    if (fi != order.size()) refuse("fault op_pos out of range");
    return R;
}

namespace {

// Q_r = rot(S)^dag B_r rot(S) for every read, support-sparse (only the terms on the read's wire).
Pauli pulled_read(const Program& prog, int r, const std::vector<uint8_t>& S) {
    const int basis = prog.reads[(size_t)r].first, w = prog.reads[(size_t)r].second;
    Pauli Q = read_pauli(prog.n, basis, w);
    const Pauli Br = Q;
    for (int j : prog.wire_terms[(size_t)w]) {
        if (!S[(size_t)j]) continue;
        const Pauli& B = prog.terms[(size_t)j];
        if (!Pauli::anticommute_bit(Br, B)) continue;
        Pauli iB = B;
        iB.phase = (iB.phase + 1) & 3;
        Q = Pauli::multiply(Q, iB);   // rot(B)^dag Q rot(B) = Q * (iB) for {Q, B} = 0
    }
    return Q;
}

int column_anticommutes(const Program& prog, const Column& c, const Pauli& P) {
    int t = 0;
    for (int r : c.recs) {
        const int basis = prog.reads[(size_t)r].first, w = prog.reads[(size_t)r].second;
        const int px = P.xbit(w), pz = P.zbit(w);
        t ^= basis == 0 ? pz : basis == 2 ? px : (px ^ pz);
    }
    return t;
}

}  // namespace

PlanLaw plan_law(const Program& prog, const std::vector<uint8_t>& S) {
    if (S.size() != prog.terms.size()) refuse("term set has the wrong size");
    const int n = prog.n, m = (int)prog.columns.size(), NW = (n + 63) / 64;
    PlanLaw L;
    L.m = m;
    // Q_c' = U^dag Q_c U (frame coordinates), Q_c = prod_r Q_r.
    std::vector<Pauli> Qf;
    Qf.reserve((size_t)m);
    for (const Column& c : prog.columns) {
        Pauli Q(n);
        for (int r : c.recs) Q = Pauli::multiply(Q, pulled_read(prog, r, S));
        Qf.push_back(prog.bare.U.conjugate(Q));
    }
    // Pivot elimination on phi(Q_c) = X-part on the fixed rows. Each stored row carries its
    // combination over pivot slots.
    struct Row { std::vector<uint64_t> v; int lead; std::vector<uint64_t> comb; };
    std::vector<Row> rows;
    L.A.assign((size_t)m, {});
    std::vector<int> slot_of((size_t)m, -1);
    auto lead_of = [&](const std::vector<uint64_t>& v) {
        for (int w = 0; w < NW; ++w) if (v[(size_t)w]) return w * 64 + __builtin_ctzll(v[(size_t)w]);
        return -1;
    };
    std::vector<std::vector<uint64_t>> combs((size_t)m);
    for (int c = 0; c < m; ++c) {
        std::vector<uint64_t> v((size_t)NW);
        for (int w = 0; w < NW; ++w) v[(size_t)w] = Qf[(size_t)c].x[(size_t)w] & prog.fixed_rows[(size_t)w];
        std::vector<uint64_t> comb;   // over pivot slots
        for (const Row& R : rows)
            if (bit(v, R.lead)) {
                for (int w = 0; w < NW; ++w) v[(size_t)w] ^= R.v[(size_t)w];
                if (comb.size() < R.comb.size()) comb.resize(R.comb.size(), 0);
                for (size_t w = 0; w < R.comb.size(); ++w) comb[w] ^= R.comb[w];
            }
        const int ld = lead_of(v);
        if (ld >= 0) {
            const int slot = (int)L.pivots.size();
            L.pivots.push_back(c);
            slot_of[(size_t)c] = slot;
            std::vector<uint64_t> cb = comb;
            if ((int)cb.size() * 64 <= slot) cb.resize((size_t)slot / 64 + 1, 0);
            cb[(size_t)slot / 64] ^= 1ULL << (slot & 63);
            // keep rows reduced at their lead bits (full reduction not needed: we test lead bits in order)
            for (Row& R : rows)
                if (bit(R.v, ld)) {
                    for (int w = 0; w < NW; ++w) R.v[(size_t)w] ^= v[(size_t)w];
                    if (R.comb.size() < cb.size()) R.comb.resize(cb.size(), 0);
                    for (size_t w = 0; w < cb.size(); ++w) R.comb[w] ^= cb[w];
                }
            rows.push_back({v, ld, cb});
            L.A[(size_t)c].assign((size_t)slot / 64 + 1, 0);
            L.A[(size_t)c][(size_t)slot / 64] |= 1ULL << (slot & 63);   // y_p = u_p
        } else {
            L.A[(size_t)c] = comb;   // phi_c = sum over the comb's pivots
            combs[(size_t)c] = comb;
        }
    }
    L.rho = (int)L.pivots.size();
    // K_c = Q_c * prod_{p in comb} Q_p (frame coords); its fixed-row X-part must vanish.
    std::vector<FreePauli> Kh((size_t)m);
    std::vector<uint8_t> nonpiv((size_t)m, 0);
    for (int c = 0; c < m; ++c) {
        if (slot_of[(size_t)c] >= 0) continue;
        nonpiv[(size_t)c] = 1;
        Pauli K = Qf[(size_t)c];
        const auto& cb = combs[(size_t)c];
        for (int s = 0; s < L.rho; ++s)
            if ((size_t)s / 64 < cb.size() && ((cb[(size_t)s / 64] >> (s & 63)) & 1))
                K = Pauli::multiply(K, Qf[(size_t)L.pivots[(size_t)s]]);
        for (int w = 0; w < NW; ++w)
            if (K.x[(size_t)w] & prog.fixed_rows[(size_t)w])
                throw std::logic_error("record_law_oracle: K_c has a fixed-row X part (elimination bug)");
        FreePauli f;
        for (int d = 0; d < prog.k; ++d) {
            const int row = prog.bare.free[(size_t)d];
            if (K.xbit(row)) f.a |= 1ULL << d;
            if (K.zbit(row)) f.b |= 1ULL << d;
        }
        f.ph = K.phase & 3;   // the frame state is |emb x> (eps = 0, checked at compile)
        Kh[(size_t)c] = f;
    }
    // Born basis over the free parts (record order), with exact signs beta.
    struct BRow { uint64_t a, b; int lead; uint64_t comb; };
    std::vector<BRow> brows;
    std::vector<FreePauli> Lam;
    L.B.assign((size_t)m, {});
    L.beta.assign((size_t)m, 0);
    L.s_sign.assign((size_t)m, 0);
    for (int c = 0; c < m; ++c) {
        if (!nonpiv[(size_t)c]) continue;
        uint64_t a = Kh[(size_t)c].a, b = Kh[(size_t)c].b, comb = 0;
        auto lead2 = [](uint64_t a_, uint64_t b_) {
            return a_ ? __builtin_ctzll(a_) : (b_ ? 64 + __builtin_ctzll(b_) : -1);
        };
        for (const BRow& R : brows) {
            const bool hit = R.lead < 64 ? ((a >> R.lead) & 1) : ((b >> (R.lead - 64)) & 1);
            if (hit) { a ^= R.a; b ^= R.b; comb ^= R.comb; }
        }
        const int ld = lead2(a, b);
        if (ld >= 0) {
            const int j = (int)Lam.size();
            if (j >= 64) throw std::logic_error("record_law_oracle: kappa > 64");
            const uint64_t cb = comb ^ (1ULL << j);
            for (BRow& R : brows) {
                const bool hit = ld < 64 ? ((R.a >> ld) & 1) : ((R.b >> (ld - 64)) & 1);
                if (hit) { R.a ^= a; R.b ^= b; R.comb ^= cb; }
            }
            brows.push_back({a, b, ld, cb});
            Lam.push_back(Kh[(size_t)c]);
            L.born.push_back(c);
            L.B[(size_t)c] = {1ULL << j};
            L.beta[(size_t)c] = 0;
        } else {
            // K_c = beta * prod_{j in comb} Lambda_j (exact).
            FreePauli prod;
            int sfac = 0;
            for (int j = 0; j < (int)Lam.size(); ++j)
                if ((comb >> j) & 1) { prod = fmul(prod, Lam[(size_t)j]); sfac ^= herm_sign(Lam[(size_t)j]); }
            if (prod.a | prod.b) L.s_sign[(size_t)c] = (uint8_t)(herm_sign(prod) ^ sfac);
            if (prod.a != Kh[(size_t)c].a || prod.b != Kh[(size_t)c].b)
                throw std::logic_error("record_law_oracle: Born decomposition mismatch");
            const int d = (Kh[(size_t)c].ph - prod.ph) & 3;
            if (d == 1 || d == 3)
                throw std::logic_error("record_law_oracle: non-Hermitian ratio (commuting-family bug)");
            L.B[(size_t)c] = {comb};
            L.beta[(size_t)c] = d == 2 ? 1 : 0;
        }
    }
    L.kappa = (int)Lam.size();
    for (const FreePauli& f : Lam) { L.lam_a.push_back(f.a); L.lam_b.push_back(f.b); L.lam_ph.push_back(f.ph & 3); }
    if (L.kappa > 30) refuse("kappa > 30 (P0 table too large)");
    // P0(z) = 2^-kappa sum_J (-1)^{z.J} <Lambda_J>.
    std::unordered_map<uint64_t, std::complex<double>> amp;
    amp.reserve(prog.alpha.size() * 2);
    for (const auto& e : prog.alpha) amp[e.first] = e.second;
    const size_t NZ = (size_t)1 << L.kappa;
    std::vector<double> E(NZ, 0.0);
    for (size_t J = 0; J < NZ; ++J) {
        FreePauli LJ;
        for (int j = 0; j < L.kappa; ++j) if ((J >> j) & 1) LJ = fmul(LJ, Lam[(size_t)j]);
        E[J] = free_expectation(prog, amp, LJ).real();
    }
    L.p0.assign(NZ, 0.0);
    for (size_t z = 0; z < NZ; ++z) {
        double s = 0.0;
        for (size_t J = 0; J < NZ; ++J) s += (__builtin_popcountll(z & J) & 1) ? -E[J] : E[J];
        L.p0[z] = s / (double)NZ;
    }
    L.cls.assign((size_t)m, 0);
    for (int c = 0; c < m; ++c) {
        bool coin = false, born = false;
        for (uint64_t w : L.A[(size_t)c]) coin |= w != 0;
        for (uint64_t w : L.B[(size_t)c]) born |= w != 0;
        L.cls[(size_t)c] = coin ? 1 : born ? 2 : 0;
    }
    return L;
}

std::vector<uint8_t> relabel(const Program& prog, const Pauli& P) {
    std::vector<uint8_t> t;
    for (const Column& c : prog.columns) t.push_back((uint8_t)column_anticommutes(prog, c, P));
    return t;
}

std::vector<uint8_t> frame_relabel(const Program& prog, const Pauli& Pp, const std::vector<uint8_t>& S) {
    std::vector<uint8_t> t = relabel(prog, Pp);
    for (size_t j = 0; j < prog.terms.size(); ++j) {
        if (!S[j] || !Pauli::anticommute_bit(Pp, prog.terms[j])) continue;
        for (size_t c = 0; c < prog.columns.size(); ++c)
            t[c] ^= (uint8_t)column_anticommutes(prog, prog.columns[c], prog.terms[j]);
    }
    return t;
}

bool term_set_pauli(const Program& prog, const std::vector<uint8_t>& S, Pauli& N) {
    // rot(S)^dag v rot(S) = v * prod_{j in S, {v,B_j}=0} (i B_j) (the terms commute). rot(S) is a
    // Pauli N iff every X_q / Z_q maps to +-itself; N anticommutes with exactly the ones that flip.
    N = Pauli(prog.n);
    for (int q = 0; q < prog.n; ++q)
        for (int zx = 0; zx < 2; ++zx) {
            Pauli v(prog.n);
            if (zx) v.setz(q); else v.setx(q);
            Pauli acc = v;
            for (size_t j = 0; j < prog.terms.size(); ++j) {
                if (!S[j] || !Pauli::anticommute_bit(v, prog.terms[j])) continue;
                Pauli iB = prog.terms[j];
                iB.phase = (iB.phase + 1) & 3;
                acc = Pauli::multiply(acc, iB);
            }
            if (acc.x != v.x || acc.z != v.z) return false;
            if (((acc.phase - v.phase) & 3) == 2) { if (zx) N.setx(q); else N.setz(q); }
            else if (((acc.phase - v.phase) & 3) != 0)
                throw std::logic_error("record_law_oracle: non-Hermitian term-set image");
        }
    return true;
}

std::vector<std::pair<uint64_t, double>> enumerate_law(const PlanLaw& L, const std::vector<uint8_t>& t) {
    if (L.m > 63) refuse("enumerate_law: more than 63 columns");
    if (L.rho + L.kappa > 30) refuse("enumerate_law: rho + kappa > 30");
    if ((int)t.size() != L.m) refuse("enumerate_law: relabel size");
    auto dotw = [](const std::vector<uint64_t>& row, uint64_t v) {
        return row.empty() ? 0 : (__builtin_popcountll(row[0] & v) & 1);
    };
    for (int c = 0; c < L.m; ++c)
        if (L.A[(size_t)c].size() > 1 || L.B[(size_t)c].size() > 1) refuse("enumerate_law: rho > 64");
    std::unordered_map<uint64_t, double> out;
    const double pu = std::ldexp(1.0, -L.rho);
    for (uint64_t u = 0; u < (1ULL << L.rho); ++u)
        for (uint64_t z = 0; z < (1ULL << L.kappa); ++z) {
            const double p = pu * L.p0[(size_t)z];
            uint64_t y = 0;
            for (int c = 0; c < L.m; ++c) {
                const int b = dotw(L.A[(size_t)c], u) ^ dotw(L.B[(size_t)c], z) ^ L.beta[(size_t)c] ^ t[(size_t)c];
                y |= (uint64_t)b << c;
            }
            out[y] += p;
        }
    std::vector<std::pair<uint64_t, double>> v(out.begin(), out.end());
    std::sort(v.begin(), v.end());
    return v;
}

}  // namespace record_oracle
}  // namespace qeccore
