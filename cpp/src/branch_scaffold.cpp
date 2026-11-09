// branch_scaffold — J-a N7: the per-plan carried-branch scaffolding. See qeccore/branch_scaffold.hpp.
#include "qeccore/branch_scaffold.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

#include "qeccore/amplitudes.hpp"
#include "qeccore/record_table.hpp"   // phi_of

namespace qeccore {

namespace {

inline uint64_t mix64(uint64_t x) {           // splitmix64 finalizer
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline bool bget(const std::vector<uint64_t>& v, int i) {
    return (size_t)(i >> 6) < v.size() && ((v[(size_t)(i >> 6)] >> (i & 63)) & 1ULL);
}

}  // namespace

Digest128 plan_key_digest128_(const std::string& key) {
    // Two independent word-wise multiply-xorshift chains over the key bytes (8-byte little-endian words, the
    // tail zero-padded), each seeded differently and closed with the length and a splitmix64 finalizer.
    uint64_t a = 0x243F6A8885A308D3ull, b = 0x13198A2E03707344ull;
    const size_t n = key.size();
    for (size_t o = 0; o < n; o += 8) {
        uint64_t w = 0;
        std::memcpy(&w, key.data() + o, n - o < 8 ? n - o : 8);
        a = mix64(a ^ w) * 0x9E3779B97F4A7C15ull;
        b = mix64(b + w * 0xC2B2AE3D27D4EB4Full) ^ (b >> 29);
    }
    Digest128 d;
    d.lo = mix64(a ^ (uint64_t)n);
    d.hi = mix64(b ^ ((uint64_t)n * 0x165667B19E3779F9ull));
    return d;
}

std::string tagged_plan_key(int kind, const std::string& content_key) {
    std::string k(1, (char)kind);
    k += content_key;
    return k;
}

PlanScaffold::PlanScaffold(PlanTermView v, const CertifiedGroupPlanes& G, long* es_counter)
    : v_(std::move(v)), G_(&G), es_count_(es_counter), id_(tagged_plan_key(v_.kind, v_.key)),
      dig_(plan_key_digest128_(id_)) {
#ifdef XTIM_MUTATE_ES_EAGER
    (void)ensure_es_();                       // mutation build only (MUT-ES-EAGER): E_S in the payload build
#endif
}

void PlanScaffold::assert_commuting_() const {
    std::vector<int> J;
    for (int j = 0; j < (int)v_.axes.size(); ++j) if (bget(v_.S, j)) J.push_back(j);
    for (size_t a = 0; a < J.size(); ++a)
        for (size_t b = a + 1; b < J.size(); ++b)
            if (Pauli::anticommute_bit(v_.axes[(size_t)J[a]], v_.axes[(size_t)J[b]]))
                throw std::logic_error("branch_scaffold: two axes of the plan's term set anticommute (engine bug)");
}

const std::vector<uint64_t>& PlanScaffold::ensure_es_() {
    if (es_built_) return es_;
    assert_commuting_();                      // once per plan, on the lazy path
    const CertifiedGroupPlanes& G = *G_;
    es_.assign((size_t)G.words, 0);
    std::vector<uint64_t> ph;
#ifdef XTIM_MUTATE_ES_SKIP
    bool skipped = false;                     // mutation build only (MUT-ES): the first j ∈ S skipped
#endif
    for (int j = 0; j < (int)v_.axes.size(); ++j) {
        if (!bget(v_.S, j)) continue;
#ifdef XTIM_MUTATE_ES_SKIP
        if (!skipped) { skipped = true; continue; }
#endif
        phi_of(G, v_.axes[(size_t)j], ph);    // bit a = ⟨B_j, g_a⟩
        for (size_t w = 0; w < es_.size() && w < ph.size(); ++w) es_[w] |= ph[w];
    }
    es_built_ = true;
    if (es_count_) ++*es_count_;
    return es_;
}

std::vector<uint64_t> PlanScaffold::moved_rows_() const {
    // C0† g_a C0 = g_a · Π_{j∈S, {g_a,B_j}=0} (iB_j): moved (≠ ±g_a) iff the anticommuting axes do not multiply to
    // a scalar, i.e. their symplectic vectors do not sum to 0.
    const CertifiedGroupPlanes& G = *G_;
    std::vector<uint64_t> out((size_t)G.words, 0);
    for (int a = 0; a < (int)G.gens.size(); ++a) {
        const Pauli& g = G.gens[(size_t)a];
        Pauli acc(g.n);
        bool any = false;
        for (int j = 0; j < (int)v_.axes.size(); ++j)
            if (bget(v_.S, j) && Pauli::anticommute_bit(g, v_.axes[(size_t)j])) {
                acc = Pauli::multiply(acc, v_.axes[(size_t)j]);
                any = true;
            }
        if (!any) continue;
        bool scalar = true;
        for (size_t w = 0; w < acc.x.size(); ++w) scalar &= !acc.x[w] && !acc.z[w];
        if (!scalar) out[(size_t)(a >> 6)] |= 1ULL << (a & 63);
    }
    return out;
}

const std::vector<TermImage>& PlanScaffold::term_images_(const FramedSuperposition& bare) {
    if (img_built_) {
        if (&bare != img_bare_)
            throw std::logic_error("branch_scaffold: term_images_ called with a different bare state (the images are "
                                   "cached for the first one)");
        return img_;
    }
    img_bare_ = &bare;
    const std::vector<uint64_t>& es = ensure_es_();
    const CertifiedGroupPlanes& G = *G_;
    const int N = bare.n(), kk = bare.k();
    std::vector<uint8_t> is_free((size_t)(N > 0 ? N : 1), 0);
    for (int d = 0; d < kk; ++d) is_free[(size_t)bare.free[(size_t)d]] = 1;
    std::vector<int> nonfree;                 // σ index a -> its frame row (certified rows, ascending)
    for (int q = 0; q < N; ++q) if (!is_free[(size_t)q]) nonfree.push_back(q);
    if ((int)nonfree.size() < (int)G.gens.size())
        throw std::logic_error("branch_scaffold: more certified generators than certified frame rows (engine bug)");
    std::vector<int> erows;                   // the E_S rows (σ order) -> frame rows
    for (int a = 0; a < (int)G.gens.size(); ++a) if (bget(es, a)) erows.push_back(nonfree[(size_t)a]);
    std::vector<uint64_t> ph;
    img_.clear();
    for (int j = 0; j < (int)v_.axes.size(); ++j) {
        if (!bget(v_.S, j)) continue;
        const Pauli& B = v_.axes[(size_t)j];
        const Pauli Q = bare.U.dual_valid() ? bare.U.conjugate(B) : bare.U.dual_image(B);
        // the frame image's X on certified rows is exactly the generators B anticommutes with (⊆ E_S)
        phi_of(G, B, ph);
        for (int a = 0; a < (int)G.gens.size(); ++a)
            if ((Q.xbit(nonfree[(size_t)a]) ? 1 : 0) != (bget(ph, a) ? 1 : 0))
                throw std::logic_error("branch_scaffold: term image X on certified rows != its anticommutation "
                                       "pattern (frame / σ-order mismatch) (engine bug)");
        TermImage t;
        t.term = j;
        t.x.assign((size_t)(kk + (int)erows.size()), 0);
        t.z.assign(t.x.size(), 0);
        for (int d = 0; d < kk; ++d) {
            t.x[(size_t)d] = Q.xbit(bare.free[(size_t)d]) ? 1 : 0;
            t.z[(size_t)d] = Q.zbit(bare.free[(size_t)d]) ? 1 : 0;
        }
        for (size_t e = 0; e < erows.size(); ++e) {
            t.x[(size_t)kk + e] = Q.xbit(erows[e]) ? 1 : 0;
            t.z[(size_t)kk + e] = Q.zbit(erows[e]) ? 1 : 0;
        }
        int par = 0;                          // free_dir_of's convention: Z·eps over every wire
        for (int q = 0; q < N; ++q) if (Q.zbit(q) && bare.eps[(size_t)q]) par ^= 1;
        t.phase = (uint8_t)(Q.phase & 3);
        t.par = (uint8_t)par;
#ifdef XTIM_MUTATE_ALPHA_READ
        {   // mutation build only (MUT-ALPHA-READ): the payload reads an amplitude (its lowest mantissa bit)
            const auto* d = dynamic_cast<const DenseAmplitudes*>(bare.alpha.get());
            if (d && !d->b_.empty()) {
                const double re = d->b_[0].second.real();
                uint64_t u;
                std::memcpy(&u, &re, sizeof u);
                t.par ^= (uint8_t)(u & 1);
            }
        }
#endif
        img_.push_back(std::move(t));
    }
    img_built_ = true;
    return img_;
}

std::vector<Pauli> PlanScaffold::pivot_frames_(const std::vector<Pauli>& Qpiv) const {
    const CertifiedGroupPlanes& G = *G_;
    const int rho = (int)Qpiv.size(), ng = (int)G.gens.size();
    // The pivot planes φ(Q_p) as rows with combination tracking, reduced to RREF (pivot column pc_r per row).
    struct Row { std::vector<uint64_t> v; std::vector<uint8_t> comb; int pc; };
    std::vector<Row> R;
    for (int p = 0; p < rho; ++p) {
        Row r;
        phi_of(G, Qpiv[(size_t)p], r.v);
        r.comb.assign((size_t)rho, 0);
        r.comb[(size_t)p] = 1;
        for (const Row& o : R)
            if (bget(r.v, o.pc)) {
                for (size_t w = 0; w < r.v.size(); ++w) r.v[w] ^= o.v[w];
                for (int q = 0; q < rho; ++q) r.comb[(size_t)q] ^= o.comb[(size_t)q];
            }
        r.pc = -1;
        for (int a = 0; a < ng; ++a) if (bget(r.v, a)) { r.pc = a; break; }
        if (r.pc < 0)
            throw std::logic_error("branch_scaffold: dependent pivot pullback planes (engine bug)");
        for (Row& o : R)
            if (bget(o.v, r.pc)) {
                for (size_t w = 0; w < o.v.size(); ++w) o.v[w] ^= r.v[w];
                for (int q = 0; q < rho; ++q) o.comb[(size_t)q] ^= r.comb[(size_t)q];
            }
        R.push_back(std::move(r));
    }
    // Row r = Σ_p E[r][p] φ(Q_p) has 1 at pc_r and 0 at every other pivot column, so with c'_r = e_{pc_r},
    // φ(Q_p')·(Σ_r E[r][p] c'_r) = δ_pp': g_p := Π_{a ∈ c_p} g_a, c_p = {pc_r : E[r][p] = 1}.
    std::vector<Pauli> W;
    W.reserve((size_t)rho);
    const int n = G.n_qubits;
    for (int p = 0; p < rho; ++p) {
        std::vector<uint8_t> c((size_t)ng, 0);
        for (const Row& r : R) if (r.comb[(size_t)p]) c[(size_t)r.pc] ^= 1;
        Pauli g(n);
        for (int a = 0; a < ng; ++a) if (c[(size_t)a]) g = Pauli::multiply(g, G.gens[(size_t)a]);
        Pauli w = g;                          // g_p on the LEFT
        for (int j = 0; j < (int)v_.axes.size(); ++j) {
            if (!bget(v_.S, j) || !Pauli::anticommute_bit(g, v_.axes[(size_t)j])) continue;
            Pauli f = v_.axes[(size_t)j];
#ifdef XTIM_MUTATE_W_SIGN
            f.phase = (uint8_t)((f.phase + 1) & 3);   // mutation build only (MUT-W-SIGN): +iB_j
#else
            f.phase = (uint8_t)((f.phase + 3) & 3);   // −iB_j
#endif
            w = Pauli::multiply(w, f);
        }
        W.push_back(std::move(w));
    }
    return W;
}

long PlanTableSet::insert(const std::vector<int>& emin, const std::string& key) {
    auto& T = tables_[emin];
    const Digest128 d = digest_of_(key);
    auto it = T.find(d);
    if (it != T.end()) {
#ifndef XTIM_MUTATE_DIG_NOCMP
        if (it->second.key != key)
            throw DigestCollision("branch_scaffold: plan digest collision: two different content keys share one "
                                  "128-bit digest (refused by name; the full-key compare never merges plans)");
#endif   // mutation build only (MUT-DIG-NOCMP): digests compared only, a collision merges silently
        return it->second.id;
    }
    const long id = next_id_++;
    T.emplace(d, Entry{key, id});
    return id;
}

long PlanTableSet::resolve(const std::vector<int>& emin, const Digest128& d, const std::string& key) const {
    auto t = tables_.find(emin);
    if (t == tables_.end()) return -1;
    auto it = t->second.find(d);
    if (it == t->second.end()) return -1;
#ifndef XTIM_MUTATE_DIG_NOCMP
    if (it->second.key != key)
        throw DigestCollision("branch_scaffold: plan digest collision on resolve: the handle's digest belongs to a "
                              "different content key (refused by name)");
#endif
    return it->second.id;
}

long PlanTableSet::size() const {
    long s = 0;
    for (const auto& kv : tables_) s += (long)kv.second.size();
    return s;
}

}  // namespace qeccore
