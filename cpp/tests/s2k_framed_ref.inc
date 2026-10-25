// GENERATED (S2k fix plan r2, m5) — DO NOT EDIT. The framed kernels of stage/s2k b3f2016
// (cpp/src/framed_superposition.cpp lines 171-393 and 428-1022: framed_canonicalise ... batch_measure),
// copied VERBATIM except that every function (and t_measure_entries) is renamed `*_ref`, so the
// test TU holds an independent oracle of the per-coordinate key loops the set-bit / popcount
// rewrite replaced. b3f2016 is byte-identical to ab2ff72 on the full corpus (0 diffs, every leg).
static thread_local FramedSuperposition::EntryVec t_measure_entries_ref;
// forward declarations (the original TU had them from framed_superposition.hpp)
int framed_measure_anticommuting_general_ref(FramedSuperposition& L, const Pauli& P, const Pauli& Q,
                                             const std::vector<int>& A, double u);
FramedRefRead framed_reference_read_toward_ref(FramedSuperposition& L, int pauli, int q, int want);
// ───────────────────────────── lean canonicalise (frame coords; tableau-only) ────────────────
void framed_canonicalise_ref(FramedSuperposition& L) {
    int r = (int)L.free.size();
    int x = L.chi();
    if (r == 0 || x == 0) return;
    auto& brs = L.entries();
    const int xw = (x + 63) / 64;
    // χ-scaled scratch: r difference columns + ≤ r reduced columns + one residual, xw words each.
    struct SlotScratch {
        std::vector<std::vector<uint64_t>> col;
        std::vector<std::vector<uint64_t>> red;
        std::vector<uint64_t> resid;
        ScratchMark mark{0, 0};
    };
    static thread_local SlotScratch sc;   // one TLS object: buffers + scratch charge
    static constexpr ScratchReleaser slot_rel = +[] {
        std::vector<std::vector<uint64_t>>().swap(sc.col);
        std::vector<std::vector<uint64_t>>().swap(sc.red);
        std::vector<uint64_t>().swap(sc.resid);
    };
    std::vector<std::vector<uint64_t>>& col = sc.col;
    std::vector<std::vector<uint64_t>>& red = sc.red;
    std::vector<uint64_t>& resid = sc.resid;
    scratch_need(sc.mark, slot_rel, (int64_t)(2 * r + 1) * (heap_chunk_bytes((int64_t)xw * 8) + 24),
              "framed canonicalise scratch");
    col.assign(r, std::vector<uint64_t>(xw, 0));
    for (int i = 1; i < x; ++i)
        for (int d = 0; d < r; ++d)
            if (key_bit(brs[i].first ^ brs[0].first, d)) col[d][i >> 6] |= 1ull << (i & 63);

    static thread_local std::vector<int> pivot_cols; pivot_cols.clear();
    red.clear();
    static thread_local std::vector<std::vector<uint8_t>> comp; comp.clear();
    static thread_local std::vector<uint8_t> expand;
    bool rows_edited = false;
    auto lead_bit = [&](const std::vector<uint64_t>& v) {
        for (int w = 0; w < xw; ++w) if (v[w]) return (w << 6) + __builtin_ctzll(v[w]);
        return -1;
    };
    for (int c = 0; c < r; ++c) {
        bool varying = false;
        for (int w = 0; w < xw; ++w) if (col[c][w]) { varying = true; break; }
        if (!varying) continue;
        resid = col[c];
        expand.assign(pivot_cols.size(), 0);
        for (size_t k = 0; k < pivot_cols.size(); ++k) {
            const int lb = lead_bit(red[k]);
            if (lb >= 0 && ((resid[lb >> 6] >> (lb & 63)) & 1)) {
                for (int w = 0; w < xw; ++w) resid[w] ^= red[k][w];
                for (size_t t = 0; t < comp[k].size(); ++t) expand[t] = (uint8_t)(expand[t] ^ comp[k][t]);
            }
        }
        bool zero = true;
        for (int w = 0; w < xw; ++w) if (resid[w]) { zero = false; break; }
        if (!zero) {
            std::vector<uint8_t> cc(pivot_cols.size() + 1, 0);
            for (size_t t = 0; t < expand.size(); ++t) cc[t] = expand[t];
            cc[pivot_cols.size()] = 1;
            for (auto& v : comp) v.push_back(0);
            pivot_cols.push_back(c);
            red.push_back(resid);
            comp.push_back(std::move(cc));
        } else {
            rows_edited = true;
            const Pauli Xc = L.U.Xrow[L.free[c]];
            for (size_t k = 0; k < expand.size(); ++k) {
                if (!expand[k]) continue;
                const int p = pivot_cols[k];
                L.U.Zrow[L.free[c]] = Pauli::multiply(L.U.Zrow[L.free[c]], L.U.Zrow[L.free[p]]);
                L.U.Xrow[L.free[p]] = Pauli::multiply(L.U.Xrow[L.free[p]], Xc);
                L.eps[L.free[c]] = (uint8_t)(L.eps[L.free[c]] ^ L.eps[L.free[p]]);
                for (auto& br : brs) br.first ^= (AmpKey)key_bit(br.first, p) << c;
            }
        }
    }
    if (rows_edited) L.U.invalidate_dual();

    int rp = (int)pivot_cols.size();
    static thread_local std::vector<uint8_t> is_pivot; is_pivot.assign(r, 0);
    for (int c2 : pivot_cols) is_pivot[c2] = 1;
    for (int col2 = 0; col2 < r; ++col2) {
        if (is_pivot[col2]) continue;
        if (key_bit(brs[0].first, col2)) L.eps[L.free[col2]] ^= 1;
    }
    for (int d = 0; d < rp; ++d) L.free[d] = L.free[pivot_cols[d]];
    L.free.resize(rp);
    for (auto& br : brs) {   // gather the pivot columns (ascending, pivot_cols[d] >= d) to 0..rp-1
        AmpKey nk = 0;
        for (int d = 0; d < rp; ++d) nk |= (AmpKey)key_bit(br.first, pivot_cols[d]) << d;
        br.first = nk;
    }
    L.sync_alpha_k();
}

// ───────────────────────────── LEAN PAULI EXPECTATION ────────────────────────────────────────
// Partner-lookup crossover chi for the Born pair-sums comes from the shared partner_hash_chi_min()
// in qeccore/pauli_kernels.hpp (same default 64 + QEC_PARTNER_HASH_MIN override) so the chi-gated
// scan/hash branch matches the oracle bit-for-bit.
namespace {

// MIRRORS CanonicalStabSum::born_from_conjugated EXACTLY on the lean fields (eps/free/branches/chi).
// Same Case A / Case B split, same packed synd build, same chi-gated scan-vs-hash partner pairing,
// same i-ascending `expt` accumulation, same ipow table — so the float result is bit-identical.
std::pair<double, double> framed_born_from_conjugated_ref(const FramedSuperposition& L, const Pauli& Q) {
    const int N = L.n();
    const int W = (N + 63) / 64;
    bool anyx = false;
    for (int w = 0; w < W; ++w) if (Q.x[w]) { anyx = true; break; }

    if (!anyx) {
        // Case A. Q = i^{Q.phase} Z^{q_z}; s0 = i^{Q.phase}·∏_{a:q_z[a]}(-1)^{eps[a]} ∈ {±1}.
        int s0 = ((Q.phase & 3) == 2) ? -1 : 1;
        for (int a = 0; a < N; ++a) if (Q.zbit(a) && L.eps[a]) s0 = -s0;
        // Stabiliser-observable sub-case: Q.z has NO support on any FREE generator. Then lam_br = s0
        // for EVERY branch, so ⟨P⟩ = s0·Σnorm(c_br) = s0 EXACTLY (±1). Return the exact result with
        // no float accumulation (the branch coeffs carry ~1e-15 collapse drift) so the downstream
        // forced-conditional gate reads an exact hard bit. (Support on NON-free generators only
        // flips s0's sign via eps, already folded above.)
        bool free_support = false;
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (Q.zbit(L.free[d])) { free_support = true; break; }
        if (!free_support)
            return (s0 == +1) ? std::pair<double, double>{1.0, 0.0}
                              : std::pair<double, double>{0.0, 1.0};
        double pp = 0.0;
        for (const auto& br : L.entries()) {
            int dot = 0;
            for (int d = 0; d < (int)L.free.size(); ++d)
                if (Q.zbit(L.free[d]) && key_bit(br.first, d)) dot ^= 1;
            int lam = dot ? -s0 : s0;
            if (lam == +1) pp += std::norm(br.second);
        }
        if (pp < 0) pp = 0; if (pp > 1) pp = 1;
        return {pp, 1.0 - pp};
    }

    // Case B: build the packed per-branch sign vectors (bit a of row i = synd[i][a]).
    const int x = L.chi();
    const auto& brs = L.entries();
    // χ-scaled scratch: x packed sign rows (+ the partner hash at large χ).
    struct SlotScratch {
        std::vector<uint64_t> synd_pk;
        std::unordered_map<std::string, int> sig_idx;
        ScratchMark mark{0, 0};
    };
    static thread_local SlotScratch sc;   // one TLS object: buffers + scratch charge
    static constexpr ScratchReleaser slot_rel = +[] {
        std::vector<uint64_t>().swap(sc.synd_pk);
        std::unordered_map<std::string, int>().swap(sc.sig_idx);
    };
    std::vector<uint64_t>& synd_pk = sc.synd_pk;
    std::unordered_map<std::string, int>& sig_idx = sc.sig_idx;
    scratch_need(sc.mark, slot_rel, (int64_t)x * (8 * (int64_t)W +
                            (x < partner_hash_chi_min() ? 0 : partner_hash_bytes_per_entry(8 * W))),
              "framed Born-probability scratch");
    static thread_local std::vector<uint64_t> eps_pk; eps_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (L.eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);
    synd_pk.resize((size_t)x * W);
    for (int i = 0; i < x; ++i) {
        uint64_t* row = synd_pk.data() + (size_t)i * W;
        for (int w = 0; w < W; ++w) row[w] = eps_pk[w];
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (key_bit(brs[i].first, d)) row[L.free[d] >> 6] ^= 1ull << (L.free[d] & 63);
    }
    static const std::complex<double> Iunit(0, 1);
    static const std::complex<double> ipow[4] = {
        std::pow(Iunit, 0.0), std::pow(Iunit, 1.0), std::pow(Iunit, 2.0), std::pow(Iunit, 3.0)};
    static thread_local std::vector<uint64_t> tgt; tgt.resize(W);
    std::complex<double> expt(0, 0);
    if (x < partner_hash_chi_min()) {
        for (int i = 0; i < x; ++i) {
            const uint64_t* si = synd_pk.data() + (size_t)i * W;
            for (int w = 0; w < W; ++w) tgt[w] = si[w] ^ Q.x[w];      // partner signature of i
            for (int j = 0; j < x; ++j) {
                const uint64_t* sj = synd_pk.data() + (size_t)j * W;
                bool match = true;
                for (int w = 0; w < W; ++w) if (sj[w] != tgt[w]) { match = false; break; }
                if (!match) continue;
                int par = 0;
                for (int w = 0; w < W; ++w) par += __builtin_popcountll(Q.z[w] & sj[w]);
                expt += std::conj(brs[i].second) * brs[j].second
                        * ipow[(Q.phase + 2 * (par & 1)) & 3];
            }
        }
    } else {
        sig_idx.clear();
        sig_idx.reserve((size_t)x * 2);
        for (int j = 0; j < x; ++j) {
            const uint64_t* sj = synd_pk.data() + (size_t)j * W;
            sig_idx.emplace(
                std::string(reinterpret_cast<const char*>(sj), (size_t)W * 8), j);
        }
        for (int i = 0; i < x; ++i) {
            const uint64_t* si = synd_pk.data() + (size_t)i * W;
            for (int w = 0; w < W; ++w) tgt[w] = si[w] ^ Q.x[w];      // partner signature of i
            auto it = sig_idx.find(
                std::string(reinterpret_cast<const char*>(tgt.data()), (size_t)W * 8));
            if (it == sig_idx.end()) continue;                        // no partner: term vanishes
            const int j = it->second;
            const uint64_t* sj = synd_pk.data() + (size_t)j * W;
            int par = 0;
            for (int w = 0; w < W; ++w) par += __builtin_popcountll(Q.z[w] & sj[w]);
            expt += std::conj(brs[i].second) * brs[j].second
                    * ipow[(Q.phase + 2 * (par & 1)) & 3];
        }
    }
    double pp = 0.5 * (1.0 + std::real(expt));
    if (pp < 0) pp = 0; if (pp > 1) pp = 1;
    return {pp, 1.0 - pp};
}
}  // namespace

std::pair<double, double> framed_expectation_ref(const FramedSuperposition& L, const Pauli& P) {
    // Q = U†PU: cached inverse rows when the dual is valid (conjugate, O(weight)); else the forward
    // rows (dual_image, O(n²)) — never force a dual rebuild. Both give the same unique Q, so the
    // result matches born_probabilities bit-for-bit (same Q feeds the identical Born math below).
    Pauli Q = L.U.dual_valid() ? L.U.conjugate(P) : L.U.dual_image(P);
    return framed_born_from_conjugated_ref(L, Q);
}
std::pair<double, double> framed_expectation_frame_ref(const FramedSuperposition& L, const Pauli& Q) {
    return framed_born_from_conjugated_ref(L, Q);
}

int framed_measure_anticommuting_ref(FramedSuperposition& L, int pauli, int q, const Pauli& P,
                               const std::vector<int>& A, double u) {
    // Single-qubit lab read: conjugate via the cached inverse (O(n)) and delegate. labP = P (built
    // here identically to the old in-body labP from pauli/q), Q = U† P U.
    Pauli Q = L.U.conjugate_single(pauli, q);
    return framed_measure_anticommuting_general_ref(L, P, Q, A, u);
}

int framed_measure_anticommuting_general_ref(FramedSuperposition& L, const Pauli& P, const Pauli& Q,
                                       const std::vector<int>& A, double u) {
    const int N = L.n();
    const int x = L.chi();
    const auto& brs = L.entries();
    const int W = (N + 63) / 64;

    // χ-scaled scratch of this measurement, charged as one high-water estimate BEFORE anything
    // (state or scratch) is touched, so a budget refusal leaves L unchanged.
    struct Surv { std::complex<double> coeff; int rep; };
    struct SlotScratch {
        std::vector<uint64_t> synd_pk, post_pk, fsynd, fsyndT;
        std::vector<std::complex<double>> c0;
        std::vector<Pauli> oW, Wgs, Qfs;
        std::vector<const Pauli*> Wtg;
        std::vector<int> order;
        std::vector<Surv> surv;
        std::vector<std::vector<uint64_t>> redcol;
        std::unordered_map<std::string, int> post_idx;
        ScratchMark mark{0, 0};
    };
    static thread_local SlotScratch sc;   // one TLS object: buffers + scratch charge
    static constexpr ScratchReleaser slot_rel = +[] {
        std::vector<uint64_t>().swap(sc.synd_pk); std::vector<uint64_t>().swap(sc.post_pk);
        std::vector<uint64_t>().swap(sc.fsynd); std::vector<uint64_t>().swap(sc.fsyndT);
        std::vector<std::complex<double>>().swap(sc.c0);
        std::vector<Pauli>().swap(sc.oW); std::vector<Pauli>().swap(sc.Wgs); std::vector<Pauli>().swap(sc.Qfs);
        std::vector<const Pauli*>().swap(sc.Wtg);
        std::vector<int>().swap(sc.order);
        std::vector<Surv>().swap(sc.surv);
        std::vector<std::vector<uint64_t>>().swap(sc.redcol);
        std::unordered_map<std::string, int>().swap(sc.post_idx);
        FramedSuperposition::EntryVec().swap(t_measure_entries_ref);
    };
    std::vector<uint64_t>& synd_pk = sc.synd_pk;
    std::vector<uint64_t>& post_pk = sc.post_pk;
    std::vector<uint64_t>& fsynd = sc.fsynd;
    std::vector<uint64_t>& fsyndT = sc.fsyndT;
    std::vector<std::complex<double>>& c0 = sc.c0;
    std::vector<Pauli>& oW = sc.oW;
    std::vector<Pauli>& Wgs = sc.Wgs;
    std::vector<Pauli>& Qfs = sc.Qfs;
    std::vector<const Pauli*>& Wtg = sc.Wtg;
    std::vector<int>& order = sc.order;
    std::vector<Surv>& surv = sc.surv;
    std::vector<std::vector<uint64_t>>& redcol = sc.redcol;
    std::unordered_map<std::string, int>& post_idx = sc.post_idx;
    {
        const int64_t pauli_b = (int64_t)sizeof(Pauli) + 2 * heap_chunk_bytes(8 * (int64_t)W);
        const int64_t per = 3 * 8 * (int64_t)W            // synd_pk, post_pk, fsynd rows
                          + 16 + 4 + (int64_t)sizeof(Surv) // c0, order, surv
                          + 3 * pauli_b + 8                // oW, Wgs, Qfs, Wtg
                          + 2 * (((int64_t)N + 7) / 8)     // fsyndT and redcol: N bits per branch
                          + (x < partner_hash_chi_min() ? 0 : partner_hash_bytes_per_entry(8 * W + 1));
        scratch_need(sc.mark, slot_rel, (int64_t)x * per + (int64_t)N * (16 + 48), "framed measurement scratch");
    }

    static thread_local std::vector<int> free_pos; free_pos.assign(N, -1);
    for (int d = 0; d < (int)L.free.size(); ++d) free_pos[L.free[d]] = d;
    static thread_local std::vector<uint64_t> eps_pk; eps_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (L.eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);
    synd_pk.resize((size_t)x * W);
    for (int i = 0; i < x; ++i) {
        uint64_t* row = synd_pk.data() + (size_t)i * W;
        for (int w = 0; w < W; ++w) row[w] = eps_pk[w];
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (key_bit(brs[i].first, d)) row[L.free[d] >> 6] ^= 1ull << (L.free[d] & 63);
    }
    auto synd_bit = [&](int i, int a) -> uint64_t {
        return (synd_pk[(size_t)i * W + (a >> 6)] >> (a & 63)) & 1ull;
    };

    c0.resize(x);
    for (int i = 0; i < x; ++i) c0[i] = brs[i].second;

    oW.resize(x);
    for (int i = 0; i < x; ++i) {
        reset_pauli_inplace(oW[i], N);
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (key_bit(brs[i].first, d)) pmul_into(oW[i], L.U.Xrow[L.free[d]]);
    }

    int p = -1;
    for (int a : A) if (free_pos[a] >= 0) {
        bool vary = false;
        for (int i = 1; i < x; ++i) if (synd_bit(i, a) != synd_bit(0, a)) { vary = true; break; }
        if (vary) { p = a; break; }
    }
    if (p < 0) for (int a : A) if (free_pos[a] >= 0) { p = a; break; }
    if (p < 0) p = A[0];

    Pauli s_lab = L.U.Zrow[p];
    if (L.eps[p]) s_lab.phase = (s_lab.phase + 2) & 3;

    Pauli Qr = Q;
    {
        static thread_local std::vector<uint64_t> amask; amask.assign(W, 0);
        for (int a : A) if (a != p) amask[a >> 6] |= 1ull << (a & 63);
        for (int i = 0; i < x; ++i) {
            uint64_t* row = synd_pk.data() + (size_t)i * W;
            if ((row[p >> 6] >> (p & 63)) & 1ull)
                for (int w = 0; w < W; ++w) row[w] ^= amask[w];
        }
    }
    for (int a : A) {
        if (a == p) continue;
        pmul_into(L.U.Zrow[a], L.U.Zrow[p]);
        pmul_into(L.U.Xrow[p], L.U.Xrow[a]);
        conj_cx_inplace(Qr, p, a);
    }

    static const std::complex<double> Iunit(0, 1);
    std::complex<double> Iph = std::pow(Iunit, (double)(Qr.phase & 3));
    auto qz_par = [&](int j) {
        const uint64_t* row = synd_pk.data() + (size_t)j * W;
        int s = 0; for (int w = 0; w < W; ++w) s += __builtin_popcountll(Qr.z[w] & row[w]);
        return s & 1;
    };
    auto mu = [&](int j) -> std::complex<double> { return Iph * (double)(qz_par(j) ? -1 : 1); };

    post_pk.resize((size_t)x * W);
    for (int i = 0; i < x; ++i) {
        const uint64_t* src = synd_pk.data() + (size_t)i * W;
        uint64_t* dst = post_pk.data() + (size_t)i * W;
        for (int w = 0; w < W; ++w) dst[w] = src[w];
        dst[p >> 6] &= ~(1ull << (p & 63));
    }
    auto post_eq = [&](int i, int j) {
        const uint64_t* a = post_pk.data() + (size_t)i * W;
        const uint64_t* b = post_pk.data() + (size_t)j * W;
        for (int w = 0; w < W; ++w) if (a[w] != b[w]) return false;
        return true;
    };

    // Born pairing: branch i's partner is the UNIQUE j with the same post-key (pivot column
    // zeroed) and the OPPOSITE pivot bit — the framed rep keeps the full packed syndrome rows
    // distinct (distinct branch sigma keys map injectively onto synd rows; the amask rotation
    // above is a bit-p-conditioned XOR by a p-free mask, so it preserves distinctness), hence at
    // most one j per i. CHI-GATED (see partner_hash_chi_min, same crossover as the Born hash in
    // framed_born_from_conjugated_ref and the oracle's measure Step-2 hash): small chi keeps the
    // O(chi^2 W) scan; large chi uses the O(chi W) (post-key, pivot-bit)->index hash. BOTH keep
    // the outer i ascending with exactly one partner term each, so `expt` accumulates in the
    // identical FP order — bit-for-bit identical either side of the crossover.
    std::complex<double> expt(0, 0);
    if (x < partner_hash_chi_min()) {
        for (int i = 0; i < x; ++i)
            for (int j = 0; j < x; ++j) {
                if (!post_eq(i, j)) continue;
                if (synd_bit(i, p) == synd_bit(j, p)) continue;
                expt += std::conj(brs[i].second) * brs[j].second * mu(j);
            }
    } else {
        post_idx.clear();
        post_idx.reserve((size_t)x * 2);
        for (int j = 0; j < x; ++j) {
            std::string key(reinterpret_cast<const char*>(post_pk.data() + (size_t)j * W),
                            (size_t)W * 8);
            key.push_back((char)synd_bit(j, p));           // distinguish the two pivot-bit members
            // INVARIANT (defended): each (post-key, pivot-bit) keys at most one branch, so the
            // hash sums exactly the pairs the scan keeps (a duplicate would drop one).
            auto ins = post_idx.emplace(std::move(key), j);
            assert(ins.second && "collapse pairing hash: duplicate (post-key,pivot) (non-canonical state)");
            (void)ins;
        }
        for (int i = 0; i < x; ++i) {
            std::string key(reinterpret_cast<const char*>(post_pk.data() + (size_t)i * W),
                            (size_t)W * 8);
            key.push_back((char)(1 - synd_bit(i, p)));     // the partner has the opposite pivot bit
            auto it = post_idx.find(key);
            if (it == post_idx.end()) continue;            // no partner: branch i is unpaired here
            const int j = it->second;
            expt += std::conj(brs[i].second) * brs[j].second * mu(j);
        }
    }
    double pp = 0.5 * (1.0 + std::real(expt));
    if (pp < 0) pp = 0;
    if (pp > 1) pp = 1;
    int m = (u < pp) ? +1 : -1;
    double pm = (m == +1) ? pp : (1.0 - pp);
    double inv = (pm > 0) ? 1.0 / std::sqrt(pm) : 0.0;

    order.resize(x);
    for (int i = 0; i < x; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        const uint64_t* a = post_pk.data() + (size_t)i * W;
        const uint64_t* b = post_pk.data() + (size_t)j * W;
        for (int w = 0; w < W; ++w) if (a[w] != b[w]) {
            const int t = __builtin_ctzll(a[w] ^ b[w]);
            return ((a[w] >> t) & 1ull) == 0;
        }
        return false;
    });
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    surv.clear();
    for (int beg = 0; beg < x; ) {
        int end = beg + 1;
        while (end < x && post_eq(order[beg], order[end])) ++end;
        int w0 = -1, w1 = -1;
        for (int t = beg; t < end; ++t) { const int i = order[t]; if (synd_bit(i, p)) w1 = i; else w0 = i; }
        beg = end;
        if (w0 >= 0 && w1 >= 0) {
            std::complex<double> nu = inv_sqrt2 * (c0[w0] + (double)m * std::conj(mu(w0)) * c0[w1]) * inv;
            if (std::abs(nu) < 1e-12) continue;
            surv.push_back({nu, w0});
        } else {
            int w = (w0 >= 0) ? w0 : w1;
            std::complex<double> nu = inv_sqrt2 * c0[w] * inv;
            if (std::abs(nu) < 1e-12) continue;
            surv.push_back({nu, w});
        }
    }

    Pauli newpiv = P;
    if (m == -1) newpiv.phase = (newpiv.phase + 2) & 3;
    Pauli new_dp = L.U.Zrow[p];
    L.U.Zrow[p] = newpiv;
    {
        const uint64_t* npx = newpiv.x.data();
        const uint64_t* npz = newpiv.z.data();
        for (int a = 0; a < N; ++a) {
            if (a == p) continue;
            const Pauli& Xa = L.U.Xrow[a];
            int ac = 0;
            for (int w = 0; w < W; ++w) ac += __builtin_popcountll(Xa.x[w] & npz[w]) + __builtin_popcountll(Xa.z[w] & npx[w]);
            if (ac & 1) pmul_into(L.U.Xrow[a], new_dp);
        }
    }
    L.U.Xrow[p] = new_dp;
    L.U.invalidate_dual();

    const int xs = (int)surv.size();
    if (xs == 0) {
        L.entries().clear(); L.free.clear(); L.sync_alpha_k(); L.eps.assign(N, 0);
        return m;
    }
    fsynd.assign((size_t)xs * W, 0);
    for (int g = 0; g < xs; ++g) {
        const uint64_t* src = synd_pk.data() + (size_t)surv[g].rep * W;
        uint64_t* dst = fsynd.data() + (size_t)g * W;
        for (int w = 0; w < W; ++w) dst[w] = src[w];
        dst[p >> 6] &= ~(1ull << (p & 63));
    }
    auto fbit = [&](int g, int a) -> uint64_t {
        return (fsynd[(size_t)g * W + (a >> 6)] >> (a & 63)) & 1ull;
    };

    {
        const int gw = (xs + 63) >> 6;
        fsyndT.assign((size_t)N * gw, 0);
        for (int g = 0; g < xs; ++g)
            for (int a = 0; a < N; ++a)
                if (fbit(g, a)) fsyndT[(size_t)a * gw + (g >> 6)] |= 1ull << (g & 63);
        static thread_local std::vector<uint64_t> onesg; onesg.assign(gw, ~0ull);
        if (xs & 63) onesg[gw - 1] = (1ull << (xs & 63)) - 1;
        static thread_local std::vector<int> basis; basis.clear();
        redcol.clear();
        static thread_local std::vector<std::vector<uint8_t>> comp; comp.clear();
        size_t nb = 0;
        static thread_local std::vector<uint64_t> ecol, resid;
        ecol.resize(gw); resid.resize(gw);
        static thread_local std::vector<uint8_t> expand;
        for (int a = 0; a < N; ++a) {
            const uint64_t* ta = fsyndT.data() + (size_t)a * gw;
            const uint64_t b0 = ta[0] & 1ull;
            for (int w = 0; w < gw; ++w) ecol[w] = ta[w] ^ (b0 ? onesg[w] : 0ull);
            bool allzero = true; for (int w = 0; w < gw; ++w) if (ecol[w]) { allzero = false; break; }
            if (allzero) continue;
            resid = ecol;
            expand.assign(nb, 0);
            for (size_t bi = 0; bi < nb; ++bi) {
                const std::vector<uint64_t>& rc = redcol[bi];
                int pr = -1;
                for (int w = 0; w < gw; ++w) if (rc[w]) { pr = (w << 6) + __builtin_ctzll(rc[w]); break; }
                if (pr >= 0 && ((resid[pr >> 6] >> (pr & 63)) & 1ull)) {
                    for (int w = 0; w < gw; ++w) resid[w] ^= rc[w];
                    for (size_t k = 0; k < comp[bi].size(); ++k) expand[k] ^= comp[bi][k];
                }
            }
            bool rz = true; for (int w = 0; w < gw; ++w) if (resid[w]) { rz = false; break; }
            if (rz) {
                Pauli dofa = L.U.Xrow[a];
                for (size_t k = 0; k < expand.size(); ++k) if (expand[k]) {
                    int abase = basis[k];
                    pmul_into(L.U.Zrow[a], L.U.Zrow[abase]);
                    pmul_into(L.U.Xrow[abase], dofa);
                    for (int g = 0; g < xs; ++g) {
                        uint64_t* row = fsynd.data() + (size_t)g * W;
                        row[a >> 6] ^= ((row[abase >> 6] >> (abase & 63)) & 1ull) << (a & 63);
                    }
                }
            } else {
                for (size_t bi = 0; bi < nb; ++bi) comp[bi].push_back(0);
                if (nb >= redcol.size()) { redcol.emplace_back(); comp.emplace_back(); }
                redcol[nb] = resid;
                std::vector<uint8_t>& cc = comp[nb]; cc.assign(nb + 1, 0); cc[nb] = 1;
                for (size_t k = 0; k < expand.size(); ++k) cc[k] = expand[k];
                basis.push_back(a);
                ++nb;
            }
        }
    }

    L.eps.assign(N, 0);
    for (int a = 0; a < N; ++a) L.eps[a] = (uint8_t)fbit(0, a);
    static thread_local std::vector<uint64_t> varies; varies.assign(W, 0);
    for (int g = 1; g < xs; ++g) {
        const uint64_t* rg = fsynd.data() + (size_t)g * W;
        for (int w = 0; w < W; ++w) varies[w] |= rg[w] ^ fsynd[w];
    }
    static thread_local std::vector<int> nfree; nfree.clear();
    for (int w = 0; w < W; ++w) {
        uint64_t bits = varies[w]; const int base = w << 6;
        while (bits) { nfree.push_back(base + __builtin_ctzll(bits)); bits &= bits - 1; }
    }
    L.free = nfree;
    L.sync_alpha_k();

    static thread_local std::vector<uint64_t> epsn_pk; epsn_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (L.eps[a]) epsn_pk[a >> 6] |= 1ull << (a & 63);
    const Pauli& labP = P;     // the standard lab Pauli of the read (single-qubit or general)
    static const std::complex<double> ipw[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    Wgs.resize(xs);
    Wtg.resize(xs);
    for (int g = 0; g < xs; ++g) {
        Wgs[g] = oW[surv[g].rep];
        int ac = 0;
        for (int w = 0; w < W; ++w)
            ac += __builtin_popcountll(Wgs[g].x[w] & labP.z[w]) + __builtin_popcountll(Wgs[g].z[w] & labP.x[w]);
        if (ac & 1) pmul_into(Wgs[g], s_lab);
        Wtg[g] = &Wgs[g];
    }
    Qfs.resize(xs);
    { DualScanScratch ds; dual_image_rows_scan(L.U.Xrow, L.U.Zrow, N, Wtg.data(), xs, Qfs.data(), ds); }   // TLC: owned scan scratch
    // The new reference is survivor 0's branch state (eps ← fsynd row 0 above), NOT the projected
    // σ=0 state R0 = √2·Π_m|ref_old> the W_g act on: R0 = D_{x0}|ref_new> with x0 = Qfs[0].x
    // (= row-0 syndrome ⊕ R0's syndrome, the frame-X part of survivor 0's W). So branch g is
    //   W_g·R0 = W_g·D_{x0}|ref_new> = U·(Qf_g X^{x0})|eps> = i^{ph_g}(-1)^{z_g·(eps⊕x0)} D_{x_g⊕x0}|ref_new>:
    // label x_g ⊕ x0 (supported on nfree — the survivors' syndromes differ only there) and the sign
    // parity against eps ⊕ x0. x0 = 0 whenever survivor 0 is the σ=0 branch (the previous code's
    // implicit assumption): then this is exactly the old label/sign. Otherwise the old code dropped
    // x0 — a relative Pauli frame error D_{x0} on the post-measurement state (wrong signs on every
    // later read anticommuting with it; xtim audit finding G).
    // x0 = 0 (the common case) keeps the old arithmetic verbatim; otherwise fold x0 into the
    // sign mask once and XOR it into every label.
    const Pauli& Q0 = Qfs[0];
    bool x0_any = false;
    for (int w = 0; w < W; ++w) x0_any |= (Q0.x[w] != 0);
    if (x0_any)
        for (int w = 0; w < W; ++w) epsn_pk[w] ^= Q0.x[w];          // parity against eps ⊕ x0
    FramedSuperposition::EntryVec& nb = t_measure_entries_ref; nb.clear();
    for (int g = 0; g < xs; ++g) {
        const Pauli& Qf = Qfs[g];
        FramedSuperposition::Entry br;
        br.first = 0;
        if (!x0_any) {
            for (int d = 0; d < (int)nfree.size(); ++d) br.first |= (AmpKey)Qf.xbit(nfree[d]) << d;
        } else {
            for (int d = 0; d < (int)nfree.size(); ++d)
                br.first |= (AmpKey)(Qf.xbit(nfree[d]) ^ Q0.xbit(nfree[d])) << d;
        }
#ifndef NDEBUG
        {   // x_g ⊕ x0 must live on the new free set (else the branch is not representable)
            static thread_local std::vector<uint8_t> isnf; isnf.assign(N, 0);
            for (int f : nfree) isnf[f] = 1;
            for (int a = 0; a < N; ++a) assert(isnf[a] || !(Qf.xbit(a) ^ (x0_any && Q0.xbit(a))));
        }
#endif
        int zs = 0; for (int w = 0; w < W; ++w) zs += __builtin_popcountll(Qf.z[w] & epsn_pk[w]);
        std::complex<double> gph = ipw[Qf.phase & 3];
        if (zs & 1) gph = -gph;
        br.second = surv[g].coeff * gph;
        nb.push_back(std::move(br));
    }
    L.entries().swap(nb);
    framed_canonicalise_ref(L);   // restore Minimality (independent free set) for the post-measurement rep
    return m;
}

// ───────────────────────────── Case-A commuting split (shared by batch_measure_ref + ref read) ────
// For a COMMUTING read with conjugate Qf (= U†PU, all X-bits zero), classify each branch by the
// read eigenvalue bit (0:+1, 1:-1) and return the +1/-1 branch weights in p0/p1. The chosen bit is
// returned as a convenience (b argument for callers that want the deterministic outcome). No state
// mutation. This is the exact eigenvalue column the affine measure_single Case A computes (s0 ·
// (-1)^{q_z·σ̃_i}), re-expressed on the lean fields.
static int framed_caseA_eig_ref(const FramedSuperposition& L, const Pauli& Qf, double& p0, double& p1) {
    const int N = L.n();
    const auto& brs = L.entries();
    int zpar = 0; for (int a = 0; a < N; ++a) if (Qf.zbit(a) && L.eps[a]) zpar ^= 1;
    int base = (((Qf.phase & 3) == 2) ? 1 : 0) ^ zpar;
    p0 = 0; p1 = 0;
    int firstbit = 0;
    for (int i = 0; i < L.chi(); ++i) {
        int sp = 0;
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (key_bit(brs[i].first, d) && Qf.zbit(L.free[d])) sp ^= 1;
        int b = base ^ sp;
        if (i == 0) firstbit = b;
        (b ? p1 : p0) += std::norm(brs[i].second);
    }
    return firstbit;
}

// Collapse a commuting Case-A read to the eigenvalue bit `bsel` (0:+1, 1:-1): keep the branches with
// that eigenvalue, renormalise, re-canonicalise. Shares the exact keep/renorm/canonicalise block
// the affine measure_single Case A (and batch_measure_ref's inline split) use. Mutates L in place.
static void framed_split_to_outcome_ref(FramedSuperposition& L, const Pauli& Qf, int bsel) {
    const int N = L.n();
    const auto& brs = L.entries();
    int zpar = 0; for (int a = 0; a < N; ++a) if (Qf.zbit(a) && L.eps[a]) zpar ^= 1;
    int base = (((Qf.phase & 3) == 2) ? 1 : 0) ^ zpar;
    FramedSuperposition::EntryVec nb;
    for (int i = 0; i < L.chi(); ++i) {
        int sp = 0;
        for (int d = 0; d < (int)L.free.size(); ++d)
            if (key_bit(brs[i].first, d) && Qf.zbit(L.free[d])) sp ^= 1;
        if ((base ^ sp) == bsel) nb.push_back(brs[i]);
    }
    double nrm = 0; for (auto& br : nb) nrm += std::norm(br.second);
    double s = nrm > 0 ? 1.0 / std::sqrt(nrm) : 1.0;
    for (auto& br : nb) br.second *= s;
    L.entries().swap(nb);
    framed_canonicalise_ref(L);
}

// ───────────────────────────── LEAN REFERENCE-COLLAPSE PRIMITIVE ──────────────────────────────
// Classify + collapse a single-qubit read at the u=0 reference, mirroring the affine loop body's
// born_probabilities + is_commuting_single + measure_single(...,0.0). MUTATES L to the reference
// post-state. Reference convention (matches CanonicalStabSum::measure_single with u=0):
//   out = (pp>1e-12) ? +1 : -1, i.e. m = (0 < pp) ? +1 : -1.
FramedRefRead framed_reference_read_ref(FramedSuperposition& L, int pauli, int q) {
    return framed_reference_read_toward_ref(L, pauli, q, 0);
}

FramedRefRead framed_reference_read_toward_ref(FramedSuperposition& L, int pauli, int q, int want) {
    const int N = L.n();
    FramedRefRead r;
    r.chi_pre = L.chi();
    Pauli P = single_pauli(pauli, q, N);
    // (pp,pm): Born probs P(+1),P(-1) BEFORE collapse — bit-identical to born_probabilities.
    auto pr = framed_expectation_ref(L, P);
    r.pp = pr.first; r.pm = pr.second;
    r.is_coin = (r.pp > 1e-12 && r.pm > 1e-12);
    if (want == 0) r.out = (r.pp > 1e-12) ? +1 : -1;
    else           r.out = (r.pm > 1e-12) ? -1 : +1;
    // Qf = U† P U: Case B iff it has any X-bit (the is_commuting_single test).
    Pauli Qf = L.U.dual_valid() ? L.U.conjugate(P) : L.U.dual_image(P);
    bool caseB = false;
    const int W = (N + 63) / 64;
    for (int w = 0; w < W; ++w) if (Qf.x[w]) { caseB = true; break; }
    r.commuting = !caseB;
    if (caseB) {
        // Anticommuting collapse: m = (u < pp) ? +1 : -1, so u=0 selects +1 (the u=0 reference the
        // affine measure_single uses) and u=1 selects -1 (only reached when pm > 1e-12).
        std::vector<int> A;
        for (int a = 0; a < N; ++a) if (Qf.xbit(a)) A.push_back(a);
        int m = framed_measure_anticommuting_general_ref(L, P, Qf, A, r.out == +1 ? 0.0 : 1.0);
        (void)m;   // m == r.out by construction
    } else {
        // Commuting read. Deterministic (pp or pm ≈ 0): no state change. Coin (χ-discriminating):
        // keep the branches consistent with the reference outcome (bit 0:+1 / 1:-1).
        if (r.is_coin) {
            const int bsel = (r.out == +1) ? 0 : 1;
            framed_split_to_outcome_ref(L, Qf, bsel);
        }
    }
    return r;
}

// ───────────────────────────── BATCH measurement (commuting reads, no post-state) ────────────
// ── Batch-conjugate single-qubit reads: out[k] = U† P_{read k} U, all in ONE forward-row pass.
//    The reads are single-qubit, so this is dual_image_rows_scan over the SAME 2N forward rows for
//    every target — amortizing the row iteration that a per-read conjugate_single (O(n²) each on a
//    stale dual) re-pays |reads| times. Bit-identical to conjugate_single(reads[k]...) per read (same
//    forward-row decomposition + phase convention as dual_image). `Qs` is resized to reads.size().
static void batch_conjugate_reads_ref(const FramedSuperposition& L,
                                  const std::vector<std::pair<int, int>>& reads,
                                  std::vector<Pauli>& Qs) {
    const int N = L.n();
    const size_t M = reads.size();
    Qs.resize(M);
    if (M == 0) return;
    static thread_local std::vector<Pauli> Tg;   // single-qubit lab Paulis (targets)
    static thread_local std::vector<const Pauli*> Tp;
    Tg.resize(M); Tp.resize(M);
    for (size_t k = 0; k < M; ++k) {
        reset_pauli_inplace(Tg[k], N);
        const int pauli = reads[k].first, q = reads[k].second;
        if (pauli == 0) Tg[k].setx(q);
        else if (pauli == 1) { Tg[k].setx(q); Tg[k].setz(q); Tg[k].phase = 1; }
        else Tg[k].setz(q);
        Tp[k] = &Tg[k];
    }
    { DualScanScratch ds; dual_image_rows_scan(L.U.Xrow, L.U.Zrow, N, Tp.data(), (int)M, Qs.data(), ds); }   // TLC: owned scan scratch
}

// chi==1 batch, conjugates already computed (Qpre[k] = U† P_{read k} U). Shared by batch_chi1_ref and
// batch_measure_ref's pass-2 (which precomputes conjugates in one forward pass). Behaviour-identical to
// the per-read conjugate_single form; only the conjugation source differs.
static void batch_chi1_pre_ref(const FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                           const std::vector<Pauli>& Qpre, std::function<double()> rng,
                           std::vector<int>& out) {
    const int N = L.n(), W = (N + 63) / 64;
    static thread_local std::vector<uint64_t> eps_pk; eps_pk.assign(W, 0);
    for (int a = 0; a < N; ++a) if (L.eps[a]) eps_pk[a >> 6] |= 1ull << (a & 63);
    static thread_local std::vector<MeasCoin> coins; coins.clear();
    out.assign(reads.size(), 0);
    for (size_t k = 0; k < reads.size(); ++k) {
        Pauli acc = Qpre[k];
        int sign = 0;
        for (auto& c : coins)
            if ((acc.x[c.pivot >> 6] >> (c.pivot & 63)) & 1ull) { pmul_into(acc, c.Qc); sign ^= c.o; }
        int xnz = -1;
        for (int w = 0; w < W; ++w) if (acc.x[w]) { xnz = (w << 6) + __builtin_ctzll(acc.x[w]); break; }
        if (xnz < 0) {
            int zpar = 0; for (int w = 0; w < W; ++w) zpar += __builtin_popcountll(acc.z[w] & eps_pk[w]);
            int ob = (((acc.phase & 3) == 2) ? 1 : 0) ^ (zpar & 1) ^ sign;
            out[k] = ob;
        } else {
            int o = (rng() < 0.5) ? 0 : 1;
            out[k] = o ^ sign;
            coins.push_back({std::move(acc), xnz, o});
        }
    }
}

void batch_chi1_ref(const FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                std::function<double()> rng, std::vector<int>& out) {
    static thread_local std::vector<Pauli> Qpre;
    batch_conjugate_reads_ref(L, reads, Qpre);
    batch_chi1_pre_ref(L, reads, Qpre, rng, out);
}

// COLUMN-form couple test — no conjugation needed. For a single-qubit read P_q, the conjugate
// Qf = U† P_q U has, on forward index b, an X-bit iff ⟨P_q, Zrow[b]⟩ and a Z-bit iff ⟨P_q, Xrow[b]⟩
//   (dual_image's α_b/β_b). For a single-qubit P_q the symplectic product ⟨P_q,R⟩ reads ONLY
//   column q of R: Z-read → R.xbit(q); X-read → R.zbit(q); Y-read → R.xbit(q)^R.zbit(q). So Qf
//   couples a free generator fa iff that column bit is set in Zrow[fa] OR Xrow[fa] — an O(|free|)
//   column lookup, NOT an O(n²) dual_image. Only the ONE picked read is then fully conjugated.
// Shared with the TreePlan extractor (framed_sampler.cpp) so its forced-collapse order matches
// batch_measure_ref's Pass-1 by construction.
bool framed_read_couples_free_ref(const FramedSuperposition& L, int pauli, int q) {
    auto sp_col = [](const Pauli& R, int pauli, int q) -> int {
        if (pauli == 2) return (int)R.xbit(q);                       // ⟨Z_q,R⟩ = R.x[q]
        if (pauli == 0) return (int)R.zbit(q);                       // ⟨X_q,R⟩ = R.z[q]
        return (int)R.xbit(q) ^ (int)R.zbit(q);                      // ⟨Y_q,R⟩ = R.x[q]^R.z[q]
    };
    for (int fa : L.free)
        if (sp_col(L.U.Zrow[fa], pauli, q) || sp_col(L.U.Xrow[fa], pauli, q)) return true;
    return false;
}

void batch_measure_ref(FramedSuperposition& L, const std::vector<std::pair<int, int>>& reads,
                   std::function<double()> rng, std::vector<int>& out) {
    const int N = L.n(), W = (N + 63) / 64;
    out.assign(reads.size(), -1);
    while (L.chi() > 1) {
        int pick = -1; Pauli Qf;
        for (size_t k = 0; k < reads.size(); ++k) if (out[k] == -1) {
            if (framed_read_couples_free_ref(L, reads[k].first, reads[k].second)) { pick = (int)k; break; }
        }
        if (pick >= 0) Qf = L.U.conjugate_single(reads[pick].first, reads[pick].second);
        if (pick < 0) break;
        bool caseB = false; for (int w = 0; w < W; ++w) if (Qf.x[w]) { caseB = true; break; }
        if (!caseB) {
            double p0, p1;
            int b = framed_caseA_eig_ref(L, Qf, p0, p1);
            double tot = p0 + p1, pr0 = tot > 0 ? p0 / tot : 0.5;
            int bsel = (rng() < pr0) ? 0 : 1;
            (void)b;
            framed_split_to_outcome_ref(L, Qf, bsel);
            out[pick] = bsel;
        } else {
            static thread_local std::vector<int> A; A.clear();
            for (int a = 0; a < N; ++a) if (Qf.xbit(a)) A.push_back(a);
            Pauli P = single_pauli(reads[pick].first, reads[pick].second, N);
            int m = framed_measure_anticommuting_ref(L, reads[pick].first, reads[pick].second, P, A, rng());
            out[pick] = (m == -1) ? 1 : 0;
        }
    }
    // Pass-2 (chi==1): the branch-independent remainder. batch_chi1_ref conjugates the remaining reads in
    // ONE batched forward pass (dual_image_rows_scan), not |rem| × O(n²) per-read conjugate_single.
    static thread_local std::vector<std::pair<int, int>> rem; rem.clear();
    static thread_local std::vector<int> remidx; remidx.clear();
    for (size_t k = 0; k < reads.size(); ++k) if (out[k] == -1) { rem.push_back(reads[k]); remidx.push_back((int)k); }
    static thread_local std::vector<int> ro; batch_chi1_ref(L, rem, rng, ro);
    for (size_t j = 0; j < remidx.size(); ++j) out[remidx[j]] = ro[j];
}
