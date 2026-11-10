// noise_chain — the mechanism tables and the block generator of the twirl sampler's NOISE stream (lean N1; the contract
// is in qeccore/noise_chain.hpp).
#include "qeccore/noise_chain.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace qeccore {

namespace {
// Vose's alias table over w[0, n) (all > 0), in index order: acceptance prob[i], alias[i] (an index in [0, n)).
void alias_table(const double* w, int n, double* prob, int32_t* alias) {
    double W = 0.0;
    for (int i = 0; i < n; ++i) W += w[i];
    std::vector<double> pr((size_t)n);
    std::vector<int> sm, lg;
    for (int i = 0; i < n; ++i) {
        pr[(size_t)i] = w[i] * n / W;
        (pr[(size_t)i] < 1.0 ? sm : lg).push_back(i);
    }
    while (!sm.empty() && !lg.empty()) {
        const int l = sm.back(); sm.pop_back();
        const int g = lg.back(); lg.pop_back();
        prob[l] = pr[(size_t)l];
        alias[l] = g;
        pr[(size_t)g] = (pr[(size_t)g] + pr[(size_t)l]) - 1.0;
        (pr[(size_t)g] < 1.0 ? sm : lg).push_back(g);
    }
    for (int i : lg) { prob[i] = 2.0; alias[i] = i; }   // (rounding leftovers: always accepted)
    for (int i : sm) { prob[i] = 2.0; alias[i] = i; }
}

// One mechanism's outcomes (alternative ids and probabilities) appended to T; groups are formed by finish().
void add_mech(NoiseTable& T, int k, const std::vector<int32_t>& alts, const std::vector<double>& w) {
    if (T.mo0.empty()) T.mo0.push_back(0);
    const int o0 = (int)T.oalt.size(), n = (int)alts.size();
    double P = 0.0;
    for (double x : w) P += x;
    T.oalt.insert(T.oalt.end(), alts.begin(), alts.end());
    T.ow.insert(T.ow.end(), w.begin(), w.end());
    T.oprob.resize((size_t)o0 + n);
    T.oalias.resize((size_t)o0 + n);
    alias_table(w.data(), n, T.oprob.data() + o0, T.oalias.data() + o0);
    T.mo0.push_back(o0 + n);
    T.mch.push_back(k);
    T.mP.push_back(std::min(1.0, P));
}

// The groups: mechanisms of bitwise-equal P, ascending P bits; channel (= mechanism) order inside a group.
void finish(NoiseTable& T) {
    if (T.mo0.empty()) T.mo0.push_back(0);
    std::map<uint64_t, std::vector<int32_t>> byp;
    for (int m = 0; m < T.M(); ++m) {
        uint64_t bits;
        std::memcpy(&bits, &T.mP[(size_t)m], 8);
        byp[bits].push_back(m);
    }
    T.events = 0.0;
    for (auto& [bits, ms] : byp) {
        double P;
        std::memcpy(&P, &bits, 8);
        NoiseTable::Group g;
        g.P = P;
        g.sc = P >= 1.0 ? 1.0 : 1.0 / std::log1p(-P);
        g.m0 = (int32_t)T.gm.size();
        T.gm.insert(T.gm.end(), ms.begin(), ms.end());
        g.m1 = (int32_t)T.gm.size();
        g.r0 = T.R;
        g.r = 2 * ((uint64_t)ms.size() << NoiseModel::kBShift) + 1;   // ≤ one gap + one outcome word per cell, + 1
        T.R += g.r;
        T.groups.push_back(g);
        T.events += P * (double)ms.size();
    }
    T.blast = T.R ? (0 - T.R) / T.R : ~(uint64_t)0;                  // (b + 1)·R ≤ 2^64  <=>  b ≤ (2^64 − R) / R
}
}  // namespace

size_t NoiseTable::bytes() const {
    return mo0.capacity() * 4 + oalt.capacity() * 4 + ow.capacity() * 8 + oprob.capacity() * 8 + oalias.capacity() * 4 +
           mch.capacity() * 4 + mP.capacity() * 8 + groups.capacity() * sizeof(Group) + gm.capacity() * 4;
}
size_t NoiseModel::bytes() const {
    return pruned_.bytes() + full_.bytes() + alt_k_.capacity() * 4 + alt_p_.capacity() * 8;
}

NoiseModel::NoiseModel(const std::vector<ErrorChannel>& channels, const std::vector<uint32_t>& rec,
                       const std::vector<size_t>& off)
    : K_((int)channels.size()) {
    pruned_.tag = krng::NOISE;
    full_.tag = krng::NOISE_FULL;
    size_t A = 0;
    for (const ErrorChannel& ch : channels) A += ch.alts.size();
    if (off.size() != A + 1) throw std::logic_error("noise table: records / alternatives mismatch (engine bug)");
    alt_k_.reserve(A);
    alt_p_.reserve(A);
    auto span = [&](size_t a) { return std::make_pair(rec.data() + off[a], off[a + 1] - off[a]); };
    auto is_null = [&](size_t a) {
        const auto [p, len] = span(a);
        for (size_t i = 0; i < len; ++i) if (p[i]) return false;
        return true;
    };
    auto same = [&](size_t a, size_t c) {
        const auto [p, lp] = span(a);
        const auto [q, lq] = span(c);
        return lp == lq && std::memcmp(p, q, lp * sizeof(uint32_t)) == 0;
    };
    std::vector<int32_t> fa, pa;      // the full / pruned mechanism's outcome alternatives
    std::vector<double> fw, pw;
    int32_t a = 0;
    for (int k = 0; k < K_; ++k) {
        const ErrorChannel& ch = channels[(size_t)k];
        if (ch.alts.empty()) throw std::invalid_argument("noise table: need at least one alternative per channel");
        fa.clear(); fw.clear(); pa.clear(); pw.clear();
        double pt = 0.0;
        for (const ErrorChannel::Alt& alt : ch.alts) {
            if (!(alt.p >= 0.0)) throw std::invalid_argument("noise table: need p >= 0");
            pt += alt.p;
            alt_k_.push_back(k);
            alt_p_.push_back(alt.p);
            if (alt.p > 0.0) {
                fa.push_back(a);
                fw.push_back(alt.p);
                if (!is_null((size_t)a)) {
                    size_t j = 0;
                    while (j < pa.size() && !same((size_t)pa[j], (size_t)a)) ++j;
                    if (j == pa.size()) { pa.push_back(a); pw.push_back(alt.p); }
                    else pw[j] += alt.p;
                }
            }
            ++a;
        }
        if (!(pt < 1.0 + 1e-9)) throw std::invalid_argument("noise table: need sum of alternative p <= 1");
        if (!fa.empty()) add_mech(full_, k, fa, fw);
        if (!pa.empty()) add_mech(pruned_, k, pa, pw);
    }
    finish(full_);
    finish(pruned_);
}

void NoiseModel::refuse_block_(const NoiseTable& T, uint64_t b) {
    throw std::invalid_argument("twirl sampler: noise block " + std::to_string(b) + " (cumulative shots from " +
        std::to_string(b) + " * 4096) would pass the noise stream's 2^64 counters (each 4096-shot block owns R = " +
        std::to_string(T.R) + " counters, 2 * mechanisms * 4096 + groups; about 2 * mechanisms * shots >= 2^64); "
        "pass a new seed to continue");
}

void NoiseModel::block_events(const NoiseTable& T, uint64_t seed, uint64_t b, std::vector<uint64_t>& ev) const {
    ev.clear();
    walk(T, seed, b, [&](int32_t alt, int s) { ev.push_back(((uint64_t)(uint32_t)alt << 32) | (uint32_t)s); });
}

void NoiseModel::shot_major(const std::vector<uint64_t>& ev, int lo, int S, DiagErrorSampler::SparseBatch& out,
                            std::vector<int32_t>& cnt) {
    const uint32_t ulo = (uint32_t)lo, uS = (uint32_t)S;
    cnt.assign((size_t)S + 1, 0);
    for (uint64_t e : ev) {
        const uint32_t s = (uint32_t)e - ulo;
        if (s < uS) ++cnt[(size_t)s + 1];
    }
    for (int s = 0; s < S; ++s) cnt[(size_t)s + 1] += cnt[(size_t)s];
    out.S = S;
    out.shot_off.assign(cnt.begin(), cnt.end());
    out.ev.resize((size_t)cnt[(size_t)S]);
    for (uint64_t e : ev) {
        const uint32_t s = (uint32_t)e - ulo;
        if (s < uS) out.ev[(size_t)cnt[s]++] = (int32_t)(e >> 32);
    }
    for (int s = 0; s < S; ++s) {                     // per shot ascending alternative id (= ascending channel)
        int32_t* p = out.ev.data() + out.shot_off[(size_t)s];
        const int c = out.shot_off[(size_t)s + 1] - out.shot_off[(size_t)s];
        if (c < 2) continue;
        if (c <= 16) {
            for (int i = 1; i < c; ++i) {
                const int32_t x = p[i];
                int j = i;
                while (j > 0 && p[j - 1] > x) { p[j] = p[j - 1]; --j; }
                p[j] = x;
            }
        } else {
            std::sort(p, p + c);
        }
    }
}

}  // namespace qeccore
