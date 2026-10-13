// bare_state_memo.cpp — the process-level framed bare-state memo (see bare_state_memo.hpp).
#include "qeccore/bare_state_memo.hpp"

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "qeccore/bare_state_errors.hpp"
#include "qeccore/clifford_tableau.hpp"
#include "qeccore/framed_superposition.hpp"

namespace qeccore {

namespace {

// ── Levers (read once) ────────────────────────────────────────────────────────────────────────
struct MemoConfig {
    bool enabled = true;
    bool verify = false;
    bool trace = false;
    size_t cap_entries = 8;
    size_t cap_bytes = size_t(256) << 20;
    size_t cap_entry_bytes = size_t(16) << 20;
};

size_t parse_size_env(const char* name, size_t dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    unsigned long long x = std::strtoull(v, &end, 10);
    if (end == v || *end != '\0') {
        std::fprintf(stderr, "[xtim] %s='%s' is not an integer; using %zu\n", name, v, dflt);
        return dflt;
    }
    return (size_t)x;
}

const MemoConfig& memo_config() {
    static const MemoConfig cfg = [] {
        MemoConfig c;
        const char* m = std::getenv("XTIM_BARE_MEMO");
        if (m && *m) {
            std::string s(m);
            if (s == "0" || s == "off") c.enabled = false;
            else if (s == "verify") c.verify = true;
            else if (s == "trace") c.trace = true;
            else if (s == "verify,trace" || s == "trace,verify") { c.verify = true; c.trace = true; }
            else if (s != "1" && s != "on")
                std::fprintf(stderr, "[xtim] XTIM_BARE_MEMO='%s' not understood (0|1|verify|trace); "
                                     "memo enabled\n", m);
        }
        c.cap_entries = parse_size_env("XTIM_BARE_MEMO_ENTRIES", c.cap_entries);
        c.cap_bytes = parse_size_env("XTIM_BARE_MEMO_MB", c.cap_bytes >> 20) << 20;
        c.cap_entry_bytes = parse_size_env("XTIM_BARE_MEMO_ENTRY_MB", c.cap_entry_bytes >> 20) << 20;
        return c;
    }();
    return cfg;
}

// ── Captured value: the fresh build's output in PLAIN (non-budgeted) storage ──────────────────
struct Captured {
    CliffordTableau U{0};                     // dual rows kept iff the build left them valid
    std::vector<uint8_t> eps;
    std::vector<int> free;
    int chi = 0;
    std::vector<std::vector<uint8_t>> sigma;  // entry i's sign pattern over `free`
    std::vector<std::complex<double>> coeff;  // entry i's coefficient
    size_t bytes = 0;                         // key + value estimate (what the byte cap counts)

    static Captured capture(const FramedBareState& s, size_t key_bytes) {
        Captured c;
        c.U = s.state.U;
        c.eps = s.state.eps;
        c.free = s.state.free;
        c.chi = s.chi;
        const int x = s.state.chi();
        c.sigma.reserve((size_t)x);
        c.coeff.reserve((size_t)x);
        for (int i = 0; i < x; ++i) {
            c.sigma.push_back(s.state.sigma(i));
            c.coeff.push_back(s.state.coeff(i));
        }
        c.bytes = key_bytes + bare_state_entry_bytes(s);
        return c;
    }

    // Exactly FramedSuperposition::from_css's field assignments, from the captured copies.
    FramedBareState restore() const {
        FramedBareState out;
        FramedSuperposition L(U.n);
        L.U = U;
        L.eps = eps;
        L.free = free;
        auto& eb = L.entries();
        eb.resize(sigma.size());
        for (size_t i = 0; i < sigma.size(); ++i) {
            eb[i].first = sigma[i];
            eb[i].second = coeff[i];
        }
        L.sync_alpha_k();
        out.state = std::move(L);
        out.rejected = false;
        out.reject_gate_index = -1;
        out.chi = chi;
        return out;
    }
};

struct Memo {
    std::mutex m;
    using Entry = std::pair<std::string, std::shared_ptr<const Captured>>;
    std::list<Entry> lru;                                   // front = most recently used
    std::unordered_map<std::string, std::list<Entry>::iterator> index;
    size_t bytes = 0;
    size_t cap_entries = 0, cap_bytes = 0, cap_entry_bytes = 0;
    bool caps_init = false;
    BareStateMemoStats st;

    void init_caps_locked() {
        if (caps_init) return;
        cap_entries = memo_config().cap_entries;
        cap_bytes = memo_config().cap_bytes;
        cap_entry_bytes = memo_config().cap_entry_bytes;
        caps_init = true;
    }
    void evict_to_caps_locked(size_t extra_bytes) {
        while (!lru.empty() && (lru.size() + 1 > cap_entries || bytes + extra_bytes > cap_bytes)) {
            Entry& e = lru.back();
            bytes -= e.second->bytes;
            index.erase(e.first);
            lru.pop_back();
            ++st.evictions;
        }
    }
    std::shared_ptr<const Captured> lookup(const std::string& key) {
        std::lock_guard<std::mutex> g(m);
        init_caps_locked();
        auto it = index.find(key);
        if (it == index.end()) { ++st.misses; return nullptr; }
        lru.splice(lru.begin(), lru, it->second);   // touch
        ++st.hits;
        return it->second->second;
    }
    // Would an entry of `bytes` be stored? Asked BEFORE the capture copy, so a state over the
    // per-entry cap (or the total cap, or with no entry slots) costs nothing on a miss.
    bool admits(size_t bytes) {
        std::lock_guard<std::mutex> g(m);
        init_caps_locked();
        if (bytes > cap_entry_bytes || bytes > cap_bytes || cap_entries == 0) { ++st.skipped; return false; }
        return true;
    }
    void store(const std::string& key, std::shared_ptr<const Captured> c) {
        std::lock_guard<std::mutex> g(m);
        init_caps_locked();
        if (index.count(key)) return;                 // raced: another thread stored it first
        if (c->bytes > cap_entry_bytes || c->bytes > cap_bytes || cap_entries == 0) { ++st.skipped; return; }
        evict_to_caps_locked(c->bytes);
        bytes += c->bytes;
        lru.emplace_front(key, std::move(c));
        index[key] = lru.begin();
        ++st.stores;
    }
    // Most-recent key (trace diagnostics: why did a lookup miss?).
    std::string front_key() {
        std::lock_guard<std::mutex> g(m);
        return lru.empty() ? std::string() : lru.front().first;
    }
};

Memo& memo() {
    static Memo* m = new Memo();   // never destroyed: outlives every static-destruction-order user
    return *m;
}

uint64_t fnv64(const std::string& s) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : s) { h ^= c; h *= 0x100000001B3ull; }
    return h;
}

void put_i32(std::string& s, int v) {
    char b[4];
    std::memcpy(b, &v, 4);
    s.append(b, 4);
}
int get_i32(const std::string& s, size_t off) {
    int v;
    std::memcpy(&v, s.data() + off, 4);
    return v;
}

// Trace helper: first differing gate between two keys (same encoding), as text.
std::string key_diff(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) return "memo empty";
    const int na = get_i32(a, 0), nb = get_i32(b, 0);
    if (na != nb) return "n differs (" + std::to_string(na) + " vs " + std::to_string(nb) + ")";
    size_t pa = 4, pb = 4;
    int gi = 0;
    while (pa < a.size() && pb < b.size()) {
        const int ka = get_i32(a, pa), kb = get_i32(b, pb);
        const int ta = get_i32(a, pa + 4), tb = get_i32(b, pb + 4);
        const size_t la = 8 + 4 * (size_t)ta, lb = 8 + 4 * (size_t)tb;
        if (ka != kb || ta != tb || a.compare(pa, la, b, pb, lb) != 0) {
            std::string r = "first differing gate at index " + std::to_string(gi) + ": kind " +
                            std::to_string(ka) + " [";
            for (int t = 0; t < ta; ++t) r += (t ? "," : "") + std::to_string(get_i32(a, pa + 8 + 4 * t));
            r += "] vs kind " + std::to_string(kb) + " [";
            for (int t = 0; t < tb; ++t) r += (t ? "," : "") + std::to_string(get_i32(b, pb + 8 + 4 * t));
            return r + "]";
        }
        pa += la; pb += lb; ++gi;
    }
    if (pa < a.size()) return "this circuit has more gates (" + std::to_string(gi) + " shared)";
    if (pb < b.size()) return "memo entry has more gates (" + std::to_string(gi) + " shared)";
    return "keys equal";
}

size_t gate_count(const std::string& key) {
    size_t p = 4, g = 0;
    while (p < key.size()) { p += 8 + 4 * (size_t)get_i32(key, p + 4); ++g; }
    return g;
}

bool paulis_equal(const Pauli& a, const Pauli& b) {
    return a.n == b.n && a.phase == b.phase && a.x == b.x && a.z == b.z;
}

}  // namespace

// ── Public API ────────────────────────────────────────────────────────────────────────────────

std::string bare_state_memo_key(const Circuit& circ) {
    std::string key;
    key.reserve(4 + 16 * circ.stream.size());
    put_i32(key, circ.n);
    for (const Instr& ins : circ.stream) {
        if (ins.kind != Instr::Kind::Gate) continue;   // the build's read set: Gate instructions only
        put_i32(key, (int)ins.gate);
        put_i32(key, (int)ins.targets.size());
        for (int t : ins.targets) put_i32(key, t);
    }
    return key;
}

size_t bare_state_entry_bytes(const FramedBareState& s) {
    const CliffordTableau& U = s.state.U;
    auto rows_bytes = [](const std::vector<Pauli>& rows) {
        size_t b = rows.size() * sizeof(Pauli);
        for (const Pauli& p : rows) b += (p.x.size() + p.z.size()) * sizeof(uint64_t);
        return b;
    };
    size_t b = sizeof(Captured) + rows_bytes(U.Xrow) + rows_bytes(U.Zrow);
    if (U.dual_valid())
        for (int a = 0; a < U.n; ++a)
            b += 2 * sizeof(Pauli) + (U.xinv_row(a).x.size() + U.xinv_row(a).z.size() +
                                      U.zinv_row(a).x.size() + U.zinv_row(a).z.size()) * sizeof(uint64_t);
    b += s.state.eps.size() + s.state.free.size() * sizeof(int);
    const size_t x = (size_t)s.state.chi(), r = (size_t)s.state.k();
    b += x * (sizeof(std::vector<uint8_t>) + sizeof(std::complex<double>) + r + 32 /* heap chunk */);
    return b;
}

bool bare_states_identical(const FramedBareState& a, const FramedBareState& b, std::string* why) {
    auto fail = [&](const std::string& w) { if (why) *why = w; return false; };
    if (a.rejected != b.rejected) return fail("rejected flag");
    if (a.reject_gate_index != b.reject_gate_index) return fail("reject_gate_index");
    if (a.chi != b.chi) return fail("chi field");
    if (a.rejected) return true;
    const FramedSuperposition& A = a.state;
    const FramedSuperposition& B = b.state;
    if (A.n() != B.n()) return fail("n");
    if (A.U.Xrow.size() != B.U.Xrow.size() || A.U.Zrow.size() != B.U.Zrow.size()) return fail("tableau row count");
    for (size_t i = 0; i < A.U.Xrow.size(); ++i)
        if (!paulis_equal(A.U.Xrow[i], B.U.Xrow[i])) return fail("Xrow[" + std::to_string(i) + "]");
    for (size_t i = 0; i < A.U.Zrow.size(); ++i)
        if (!paulis_equal(A.U.Zrow[i], B.U.Zrow[i])) return fail("Zrow[" + std::to_string(i) + "]");
    if (A.U.dual_valid() != B.U.dual_valid()) return fail("dual validity");
    if (A.U.dual_valid())
        for (int i = 0; i < A.n(); ++i) {
            if (!paulis_equal(A.U.xinv_row(i), B.U.xinv_row(i))) return fail("Xinv[" + std::to_string(i) + "]");
            if (!paulis_equal(A.U.zinv_row(i), B.U.zinv_row(i))) return fail("Zinv[" + std::to_string(i) + "]");
        }
    if (A.eps != B.eps) return fail("eps");
    if (A.free != B.free) return fail("free");
    if (A.k() != B.k()) return fail("k");
    if (A.chi() != B.chi()) return fail("chi (entry count)");
    if (A.alpha->k() != B.alpha->k()) return fail("alpha k");
    for (int i = 0; i < A.chi(); ++i) {
        if (A.sigma(i) != B.sigma(i)) return fail("sigma[" + std::to_string(i) + "]");
        const std::complex<double>& ca = A.coeff(i);
        const std::complex<double>& cb = B.coeff(i);
        if (std::memcmp(&ca, &cb, sizeof ca) != 0) return fail("coeff[" + std::to_string(i) + "] (bits)");
    }
    return true;
}

BareStateMemoStats bare_state_memo_stats() {
    Memo& M = memo();
    std::lock_guard<std::mutex> g(M.m);
    M.init_caps_locked();
    BareStateMemoStats s = M.st;
    s.entries = M.lru.size();
    s.bytes = M.bytes;
    s.cap_entries = M.cap_entries;
    s.cap_bytes = M.cap_bytes;
    s.cap_entry_bytes = M.cap_entry_bytes;
    s.enabled = memo_config().enabled;
    s.verify = memo_config().verify;
    return s;
}

void bare_state_memo_clear() {
    Memo& M = memo();
    std::lock_guard<std::mutex> g(M.m);
    M.lru.clear();
    M.index.clear();
    M.bytes = 0;
}

void bare_state_memo_reset_stats() {
    Memo& M = memo();
    std::lock_guard<std::mutex> g(M.m);
    M.st = BareStateMemoStats{};
}

void bare_state_memo_set_caps(size_t max_entries, size_t max_bytes) {
    Memo& M = memo();
    std::lock_guard<std::mutex> g(M.m);
    M.init_caps_locked();
    M.cap_entries = max_entries;
    M.cap_bytes = max_bytes;
    // evict_to_caps_locked bounds (size + 1, bytes + extra) for an incoming entry; bound the
    // current contents exactly here.
    while (!M.lru.empty() && (M.lru.size() > M.cap_entries || M.bytes > M.cap_bytes)) {
        Memo::Entry& e = M.lru.back();
        M.bytes -= e.second->bytes;
        M.index.erase(e.first);
        M.lru.pop_back();
        ++M.st.evictions;
    }
}

void bare_state_memo_set_entry_cap(size_t max_entry_bytes) {
    Memo& M = memo();
    std::lock_guard<std::mutex> g(M.m);
    M.init_caps_locked();
    M.cap_entry_bytes = max_entry_bytes;
}

FramedBareState build_bare_state_framed(const Circuit& circ) {
    const MemoConfig& cfg = memo_config();
    Memo& M = memo();
    // Bypass: disabled, or a test seam is installed (g_bare_state_test_hook corrupts the assembled
    // state on purpose — a memoised result would hide it).
    if (!cfg.enabled || g_bare_state_test_hook) {
        { std::lock_guard<std::mutex> g(M.m); ++M.st.bypassed; }
        return build_bare_state_framed_uncached(circ);
    }
    const std::string key = bare_state_memo_key(circ);
    if (std::shared_ptr<const Captured> hit = M.lookup(key)) {
        FramedBareState out = hit->restore();
        if (cfg.trace)
            std::fprintf(stderr, "[xtim] bare-state memo HIT  n=%d gates=%zu chi=%d key=%016llx\n",
                         circ.n, gate_count(key), out.chi, (unsigned long long)fnv64(key));
        if (cfg.verify) {
            FramedBareState fresh = build_bare_state_framed_uncached(circ);
            std::string why;
            if (!bare_states_identical(out, fresh, &why))
                throw BareStateInvariantError(
                    "bare-state memo verify: the memoised state differs from a fresh build (" + why +
                    ") — the memo key does not capture the build's full input; refusing");
            if (cfg.trace) std::fprintf(stderr, "[xtim] bare-state memo VERIFIED (fresh rebuild identical)\n");
        }
        return out;
    }
    if (cfg.trace)
        std::fprintf(stderr, "[xtim] bare-state memo MISS n=%d gates=%zu key=%016llx (%s)\n",
                     circ.n, gate_count(key), (unsigned long long)fnv64(key),
                     key_diff(key, M.front_key()).c_str());
    FramedBareState out = build_bare_state_framed_uncached(circ);
    if (out.rejected) {
        { std::lock_guard<std::mutex> g(M.m); ++M.st.skipped; }
        return out;
    }
    const size_t nbytes = key.size() + bare_state_entry_bytes(out);
    if (M.admits(nbytes)) {                       // size known before any copy: over a cap = no copy
        auto cap = std::make_shared<const Captured>(Captured::capture(out, key.size()));
        M.store(key, std::move(cap));
    }
    if (cfg.trace) {
        BareStateMemoStats s = bare_state_memo_stats();
        std::fprintf(stderr, "[xtim] bare-state memo %s n=%d chi=%d bytes=%zu (held %zu entries, %zu bytes; caps %zu entries, %zu bytes, %zu per entry)\n",
                     s.entries && M.front_key() == key ? "STORE" : "SKIP ", circ.n, out.chi, nbytes,
                     s.entries, s.bytes, s.cap_entries, s.cap_bytes, s.cap_entry_bytes);
    }
    return out;
}

}  // namespace qeccore
