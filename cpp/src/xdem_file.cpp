// xdem_file.cpp — the xdem reader, the canonical emitter, the composite and the law (see qeccore/xdem_file.hpp;
// SPEC §1.2, §1.3, §3). The law is the dc-DEM v1 reader's law_into (19f2f78) on an AxisView, reading the composite's eta.
#include "qeccore/xdem_file.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <string_view>

namespace qeccore::xdem {

namespace {

using SV = std::string_view;

[[noreturn]] void refuse(int line, const std::string& msg) {
    throw XdemFileError("xdem: line " + std::to_string(line) + ": " + msg);
}

inline std::string S(SV s) { return std::string(s); }
inline bool is_ws(char c) { return c == ' ' || c == '\t'; }

inline bool eq_ci(SV a, const char* b) {
    const size_t n = std::strlen(b);
    if (a.size() != n) return false;
    for (size_t i = 0; i < n; ++i) {
        char c = a[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (c != b[i]) return false;
    }
    return true;
}

// One non-empty text line: leading whitespace removed; a comment ('#' outside a [tag]) and the whitespace before it
// removed. Trailing whitespace of a line without a comment is kept (the canonical-spacing check reads it).
struct Item {
    int lineno;
    SV body;
};

// An instruction's head: name, optional [tag], optional (args); `rest` = the text after the head.
struct Head {
    SV name, tag, args, rest;
    bool has_tag = false, has_args = false;
};

bool parse_head(SV b, Head& h) {
    size_t i = 0;
    while (i < b.size() && (std::isalnum((unsigned char)b[i]) || b[i] == '_')) ++i;
    if (i == 0) return false;
    h.name = b.substr(0, i);
    if (i < b.size() && b[i] == '[') {
        const size_t e = b.find(']', i);
        if (e == SV::npos) return false;
        h.has_tag = true;
        h.tag = b.substr(i + 1, e - i - 1);
        i = e + 1;
    }
    if (i < b.size() && b[i] == '(') {
        const size_t e = b.find(')', i);
        if (e == SV::npos) return false;
        h.has_args = true;
        h.args = b.substr(i + 1, e - i - 1);
        i = e + 1;
    }
    if (i < b.size() && !is_ws(b[i])) return false;
    h.rest = b.substr(i);
    return true;
}

void split_ws(SV s, std::vector<SV>& out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && is_ws(s[i])) ++i;
        const size_t b = i;
        while (i < s.size() && !is_ws(s[i])) ++i;
        if (i > b) out.push_back(s.substr(b, i - b));
    }
}

inline bool all_digits(SV s) {
    if (s.empty()) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    return true;
}

// stim's parens arguments (read_parens_arguments / read_normal_double, stim 1.16): comma-separated items, whitespace
// around each skipped; an item is at most 63 characters of [0-9.eE+-] that strtod consumes whole and that is finite
// (so no hex float, inf or nan); an empty item reads as 0 (stim: `error() D0` is `error(0) D0`). Returns false where
// stim refuses; out receives the values.
bool stim_args(SV a, std::vector<double>& out) {
    out.clear();
    for (;;) {
        size_t e = 0;
        while (e < a.size() && a[e] != ',') ++e;
        SV x = a.substr(0, e);
        while (!x.empty() && is_ws(x.front())) x.remove_prefix(1);
        while (!x.empty() && is_ws(x.back())) x.remove_suffix(1);
        if (x.size() > 63) return false;
        for (char c : x)
            if (!((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')) return false;
        char buf[64];   // as stim's read_normal_double: at most 63 characters, NUL-terminated for strtod
        std::memcpy(buf, x.data(), x.size());
        buf[x.size()] = '\0';
        char* end = nullptr;
        const double v = std::strtod(buf, &end);
        if (end != buf + x.size() || std::isinf(v) || std::isnan(v)) return false;
        out.push_back(v);
        if (e == a.size()) return true;
        a.remove_prefix(e + 1);
    }
}

// stim's target prefixes are case-blind (d/D, l/L): the upper-case prefix of a target token, or 0.
inline char dl_prefix(SV k) {
    if (k.empty()) return 0;
    const char c = k[0] == 'd' ? 'D' : k[0] == 'l' ? 'L' : k[0];
    return c == 'D' || c == 'L' ? c : 0;
}

constexpr long long kIntMax = (1LL << 31) - 1;
constexpr long long kShiftSat = 1LL << 61;
constexpr int64_t kObsTag = 1LL << 40;   // a pending flip target: detector d as d, observable k as kObsTag + k

const char* const kLineFields[3] = {"x", "rot", "late"};
const char* const kAxisFields[4] = {"sign", "img", "gauge", "hidden"};

inline void set_bit(std::vector<uint64_t>& w, size_t j) { w[j >> 6] |= 1ull << (j & 63); }
inline bool any(const uint64_t* w, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (w[i]) return true;
    return false;
}

std::string axis_list(const uint64_t* comb, size_t aw) {
    std::string s;
    for (size_t w = 0; w < aw; ++w)
        for (uint64_t x = comb[w]; x; x &= x - 1) {
            if (!s.empty()) s += ' ';
            s += "Q" + std::to_string(w * 64 + (size_t)__builtin_ctzll(x));
        }
    return s;
}

// Row elimination over GF(2) on the first `kw` words of each row (pivot = lowest set bit), carrying the remaining
// words along, with the combination of input rows tracked in `aw` words. reduce(row) returns true when the key part
// reduces to 0 (the row is then left reduced, its combination in its last aw words); otherwise it is inserted.
struct Elim {
    size_t kw, tw, aw, nrow = 0;   // key words, total words per row (key + carried + combination)
    std::vector<uint64_t> rows;
    std::vector<int32_t> prow;     // pivot bit -> row
    Elim(size_t key_bits, size_t kw_, size_t carried, size_t aw_)
        : kw(kw_), tw(kw_ + carried + aw_), aw(aw_), prow(key_bits, -1) {}
    bool reduce_or_insert(uint64_t* v) {
        for (size_t w = 0; w < kw; ++w) {
            while (v[w]) {
                const size_t p = w * 64 + (size_t)__builtin_ctzll(v[w]);
                const int32_t r = prow[p];
                if (r < 0) {
                    prow[p] = (int32_t)nrow++;
                    rows.insert(rows.end(), v, v + tw);
                    return false;
                }
                const uint64_t* b = &rows[(size_t)r * tw];
                for (size_t t = w; t < tw; ++t) v[t] ^= b[t];
            }
        }
        return true;
    }
};

class Reader {
public:
    explicit Reader(const std::string& text) : text_(text) {}
    File run();
    // run()'s allocation failed (parse reports it as XdemCapacityError): the sizes read up to that point
    std::string capacity_message() const {
        const long long D = ext_ ? F.D : max_det_ + 1, O = ext_ ? F.O : max_obs_ + 1;
        return "xdem: out of memory reading the file at line " + std::to_string(cur_line_) + " (" +
               std::to_string(D) + " detectors, " + std::to_string(O) + " observables, " +
               std::to_string(F.axes.size()) + " axes, " + std::to_string(F.p.size()) +
               " error lines so far, repeat blocks unrolled)";
    }

private:
    const std::string& text_;
    std::vector<Item> items_;
    File F;
    bool ext_ = false;
    int header_line_ = 0;
    long long shift_ = 0;
    bool seen_error_ = false;
    long long max_det_ = -1, max_obs_ = -1;
    int max_line_ = 0;
    int cur_line_ = 0;                   // the line being read (capacity_message)
    std::vector<int> axis_line_;
    std::vector<int64_t> pend_;          // pending flip targets of every line (kObsTag encoding)
    std::vector<uint64_t> pend_off_{0};
    std::vector<SV> tok_;
    std::vector<int64_t> tgt_;
    std::vector<int32_t> fld_;
    std::vector<double> args_;           // parens arguments (stim_args scratch)

    void split_lines();
    void recognize_v1();
    void canonical_spacing(const Item& it);
    long long int_token(SV digits, int line, const char* what, SV whole);
    void handle(const Item& it);
    void header(const Item& it);
    void axis(const Item& it, const Head& h);
    void error(const Item& it, const Head& h);
    void finish();
    void check_header_rank();
    void check_operators();
};

void Reader::split_lines() {
    const char* s = text_.data();
    const size_t n = text_.size();
    size_t pos = 0;
    int lineno = 0;
    while (pos < n) {
        size_t nl = text_.find('\n', pos);
        if (nl == std::string::npos) nl = n;
        size_t e = nl;
        if (e > pos && s[e - 1] == '\r') --e;   // CRLF is read as LF
        ++lineno;
        size_t b = pos;
        while (b < e && is_ws(s[b])) ++b;
        bool in_tag = false;
        size_t c = e;
        for (size_t i = b; i < e; ++i) {
            if (s[i] == '[') in_tag = true;
            else if (s[i] == ']') in_tag = false;
            else if (s[i] == '#' && !in_tag) { c = i; break; }
        }
        if (c != e)
            while (c > b && is_ws(s[c - 1])) --c;   // a comment and the whitespace before it
        size_t t = c;
        while (t > b && is_ws(s[t - 1])) --t;
        if (t > b) items_.push_back({lineno, SV(s + b, c - b)});
        pos = nl + 1;
    }
}

// dc v1 recognition (before anything else): the format marker shift_detectors(6463, ...) or a dc:-prefixed tag.
void Reader::recognize_v1() {
    if (text_.find("dc:") == std::string::npos && text_.find("6463") == std::string::npos) return;
    for (const Item& it : items_) {
        Head h;
        if (!parse_head(it.body, h)) continue;
        bool v1 = h.has_tag && h.tag.substr(0, 3) == "dc:";
        if (!v1 && h.has_args && eq_ci(h.name, "shift_detectors")) {
            const std::string a = S(h.args);
            char* end = nullptr;
            const double v = std::strtod(a.c_str(), &end);
            v1 = end != a.c_str() && v == 6463.0;
        }
        if (v1)
            refuse(it.lineno, "dc-DEM v1 file: re-export the circuit with format=\"xdem\" (an xdem cannot be built from "
                              "a v1 file: it lacks the strict tau and the hidden-syndrome register)");
    }
}

void Reader::canonical_spacing(const Item& it) {
    const SV b = it.body;
    for (size_t i = 0; i < b.size(); ++i) {
        if (b[i] == '\t') refuse(it.lineno, "tab (tokens are separated by single spaces)");
        if (b[i] == ' ' && i + 1 < b.size() && b[i + 1] == ' ') refuse(it.lineno, "double space");
    }
    if (!b.empty() && b.back() == ' ') refuse(it.lineno, "trailing space");
}

// An unsigned integer token. Extended files: canonical (no leading zero except "0", <= 2^31 - 1); a conventional DEM
// keeps stim's spelling (leading zeros accepted), the bound still applies. Returns -1 when `digits` is not decimal.
long long Reader::int_token(SV digits, int line, const char* what, SV whole) {
    if (!all_digits(digits)) return -1;
    long long v = 0;
    bool over = false;
    for (char c : digits) {
        v = v * 10 + (c - '0');
        if (v > kIntMax) { over = true; break; }
    }
    if (ext_ && (over || (digits.size() > 1 && digits[0] == '0')))
        refuse(line, "non-canonical integer '" + S(digits) + "'");
    if (over) refuse(line, std::string(what) + " '" + S(whole) + "' exceeds 2^31 - 1");
    return v;
}

void Reader::header(const Item& it) {
    header_line_ = it.lineno;
    canonical_spacing(it);
    split_ws(it.body, tok_);
    const int ln = it.lineno;
    if (tok_.size() < 2) refuse(ln, "header field missing (expected the version after 'xdem')");
    const long long v = int_token(tok_[1], ln, "version", tok_[1]);
    if (v < 0) refuse(ln, "unknown xdem version " + S(tok_[1]));
    if (v != 1) refuse(ln, "unknown xdem version " + std::to_string(v));
    const char* names[5] = {"detectors", "observables", "gauge", "hidden", "tc"};
    long long val[5];
    for (int f = 0; f < 5; ++f) {
        const size_t i = 2 + 2 * (size_t)f;
        if (i >= tok_.size()) refuse(ln, std::string("header field missing (expected '") + names[f] + "')");
        if (tok_[i] != names[f])
            refuse(ln, "header field '" + S(tok_[i]) + "' (expected '" + names[f] + "')");
        if (i + 1 >= tok_.size()) refuse(ln, std::string("header field '") + names[f] + "' has no value");
        val[f] = int_token(tok_[i + 1], ln, names[f], tok_[i + 1]);
        if (val[f] < 0) refuse(ln, std::string("header field '") + names[f] + "' has a bad value '" +
                                   S(tok_[i + 1]) + "'");
    }
    if (tok_.size() > 12) refuse(ln, "header field '" + S(tok_[12]) + "' (expected the end of the line)");
    if (val[0] + val[1] > kMaxColumns)
        refuse(ln, std::to_string(val[0] + val[1]) + " readout columns exceed 2^31 - 1");
    F.D = (int)val[0];
    F.O = (int)val[1];
    F.r = (int)val[2];
    F.m = (int)val[3];
    F.tc = val[4];
}

void Reader::axis(const Item& it, const Head& h) {
    const int ln = it.lineno;
    if (!ext_) refuse(ln, "axis line in a file with no xdem header");
    if (seen_error_) refuse(ln, "axis line after the first error line");
    canonical_spacing(it);
    if (h.has_tag) refuse(ln, "tags are not part of xdem 1");
    if (h.has_args) refuse(ln, "an axis line takes no arguments");
    split_ws(h.rest, tok_);
    if (tok_.empty()) refuse(ln, "axis line without an axis id");
    const SV id = tok_[0];
    const long long a = id.size() > 1 && id[0] == 'Q' ? int_token(id.substr(1), ln, "axis id", id) : -1;
    if (a < 0) refuse(ln, "bad axis id '" + S(id) + "'");
    if (a != (long long)F.axes.size())
        refuse(ln, "axis Q" + std::to_string(a) + " out of order (expected Q" + std::to_string(F.axes.size()) + ")");
    Axis ax;
    ax.img.w.assign(words_for((long long)F.D + F.O), 0);
    ax.k.assign(words_for(F.r), 0);
    int cur = -1, next = 0;
    size_t cnt = 0;
    long long prev = -1;
    auto close = [&]() {
        if (cur == 0 && cnt == 0) refuse(ln, "sign takes only '-'");
        if (cur > 0 && cnt == 0) refuse(ln, std::string("empty ") + kAxisFields[cur] + " field");
    };
    for (size_t t = 1; t < tok_.size(); ++t) {
        const SV k = tok_[t];
        int kw = -1;
        for (int f = 0; f < 4; ++f)
            if (k == kAxisFields[f]) kw = f;
        if (kw >= 0) {
            close();
            if (kw < next)
                refuse(ln, std::string(kAxisFields[kw]) +
                               " field out of order (the fields are sign, img, gauge, hidden, each at most once)");
            cur = kw;
            next = kw + 1;
            cnt = 0;
            prev = -1;
            continue;
        }
        if (cur < 0) refuse(ln, "unknown axis field '" + S(k) + "'");
        if (cur == 0) {
            if (k != "-" || cnt) refuse(ln, "sign takes only '-'");
            ax.b = 1;
            ++cnt;
            continue;
        }
        long long col = -1;
        if (cur == 1) {
            const long long j = k.size() > 1 && (k[0] == 'D' || k[0] == 'L')
                                    ? int_token(k.substr(1), ln, "img index", k) : -1;
            if (j < 0) refuse(ln, "img token '" + S(k) + "'");
            if (k[0] == 'D') {
                if (j + shift_ >= F.D)
                    refuse(ln, "undeclared detector D" + std::to_string(j + shift_) + " (header: " +
                                   std::to_string(F.D) + " detectors)");
                col = j + shift_;
            } else {
                if (j >= F.O)
                    refuse(ln, "undeclared observable L" + std::to_string(j) + " (header: " + std::to_string(F.O) +
                                   " observables)");
                col = F.D + j;
            }
        } else {
            const char want = cur == 2 ? 'g' : 'h';
            const long long q = k.size() > 1 && k[0] == want ? int_token(k.substr(1), ln, kAxisFields[cur], k) : -1;
            if (q < 0) refuse(ln, std::string(kAxisFields[cur]) + " token '" + S(k) + "'");
            if (cur == 2 && q >= F.r)
                refuse(ln, "gauge qubit " + S(k) + " out of range (header: gauge " + std::to_string(F.r) + ")");
            if (cur == 3 && q >= F.m)
                refuse(ln, "hidden-syndrome qubit " + S(k) + " out of range (header: hidden " + std::to_string(F.m) +
                               ")");
            col = q;
        }
        if (col == prev) refuse(ln, "duplicate " + S(k) + " in " + kAxisFields[cur] + " field");
        if (col < prev) refuse(ln, std::string("unsorted ") + kAxisFields[cur] + " field");
        prev = col;
        ++cnt;
        if (cur == 1) set_bit(ax.img.w, (size_t)col);
        else if (cur == 2) set_bit(ax.k, (size_t)col);
        else ax.hidden.push_back((int32_t)col);
    }
    close();
    if (!any(ax.img.w.data(), ax.img.w.size()) && !any(ax.k.data(), ax.k.size()) && ax.hidden.empty())
        refuse(ln, "axis Q" + std::to_string(a) + " has no qubit");
    F.axes.push_back(std::move(ax));
    axis_line_.push_back(ln);
}

void Reader::error(const Item& it, const Head& h) {
    const int ln = it.lineno;
    if (h.has_tag && ext_) refuse(ln, "tags are not part of xdem 1");
    if (!h.has_args) refuse(ln, "error instruction needs exactly one probability");
    double p;
    {
        std::vector<double>& v = args_;   // stim's number rule (stim_args): no hex float, inf or nan
        if (!stim_args(h.args, v) || v.size() != 1)
            refuse(ln, "bad error probability '" + S(h.args) + "' (error instruction needs exactly one probability)");
        p = v[0];
        if (!(p >= 0.0 && p <= 1.0)) {
            SV a = h.args;
            while (!a.empty() && is_ws(a.front())) a.remove_prefix(1);
            while (!a.empty() && is_ws(a.back())) a.remove_suffix(1);
            refuse(ln, "error probability " + S(a) + " outside [0, 1]");
        }
    }
    split_ws(h.rest, tok_);
    size_t nt = 0;   // the targets are tok_[0, nt)
    while (nt < tok_.size() && tok_[nt] != "x" && tok_[nt] != "rot" && tok_[nt] != "late") ++nt;
    if (nt < tok_.size()) {
        if (!ext_) refuse(ln, "x/rot/late field in a file with no xdem header");
        if (h.name != "error") refuse(ln, "instruction '" + S(h.name) + "' (xdem spells it 'error')");
        canonical_spacing(it);
    }
    tgt_.clear();
    for (size_t t = 0; t < nt; ++t) {
        const SV k = tok_[t];
        if (k == "^") {   // a decomposition hint: the line flips the XOR of all its targets
            if (t == 0 || t + 1 == nt)   // stim refuses a separator first or last, and two in a row
                refuse(ln, "a separator (^) is the first or last target of the error instruction");
            if (tok_[t - 1] == "^") refuse(ln, "adjacent separators (^ ^) in the error instruction");
            continue;
        }
        const char pre = dl_prefix(k);   // d/D, l/L: stim's target prefixes are case-blind
        const long long j = k.size() > 1 && pre ? int_token(k.substr(1), ln, "target", k) : -1;
        if (j < 0) refuse(ln, "bad error target '" + S(k) + "'");
        if (pre == 'D') {
            const long long d = j + shift_;
            if (ext_ && d >= F.D)
                refuse(ln, "undeclared detector D" + std::to_string(d) + " (header: " + std::to_string(F.D) +
                               " detectors)");
            if (d > max_det_) { max_det_ = d; max_line_ = ln; }
            tgt_.push_back(d);
        } else {
            if (ext_ && j >= F.O)
                refuse(ln, "undeclared observable L" + std::to_string(j) + " (header: " + std::to_string(F.O) +
                               " observables)");
            if (j > max_obs_) { max_obs_ = j; max_line_ = ln; }
            tgt_.push_back(kObsTag + j);
        }
    }
    bool asc = true;
    for (size_t i = 1; i < tgt_.size() && asc; ++i) asc = tgt_[i] > tgt_[i - 1];
    if (asc) {
        pend_.insert(pend_.end(), tgt_.begin(), tgt_.end());
    } else {   // a repeated target cancels mod 2 (also across `^`)
        std::sort(tgt_.begin(), tgt_.end());
        for (size_t i = 0; i < tgt_.size();) {
            size_t j = i;
            while (j < tgt_.size() && tgt_[j] == tgt_[i]) ++j;
            if ((j - i) & 1) pend_.push_back(tgt_[i]);
            i = j;
        }
    }
    pend_off_.push_back(pend_.size());
    // fields x, rot, late: each optional, at most once, in this order; tokens ascending by axis, no axis repeated
    int cur = -1, next = 0;
    size_t cnt = 0;
    long long prev = -1;
    const long long A = (long long)F.axes.size();
    auto close = [&]() {
        if (cur >= 0 && cnt == 0) refuse(ln, std::string("empty ") + kLineFields[cur] + " field");
    };
    for (size_t t = nt; t < tok_.size(); ++t) {
        const SV k = tok_[t];
        int kw = -1;
        for (int f = 0; f < 3; ++f)
            if (k == kLineFields[f]) kw = f;
        if (kw >= 0) {
            close();
            if (kw < next)
                refuse(ln, std::string(kLineFields[kw]) +
                               " field out of order (the fields are x, rot, late, each at most once)");
            cur = kw;
            next = kw + 1;
            cnt = 0;
            prev = -1;
            continue;
        }
        long long a = -1;
        uint8_t e = 0;
        if (cur == 0) {
            a = k.size() > 1 && k[0] == 'Q' ? int_token(k.substr(1), ln, "axis id", k) : -1;
            if (a < 0) refuse(ln, "x token '" + S(k) + "'");
        } else {
            a = k.size() > 2 && (k[0] == '+' || k[0] == '-') && k[1] == 'Q' ? int_token(k.substr(2), ln, "axis id", k)
                                                                            : -1;
            if (a < 0) refuse(ln, std::string(kLineFields[cur]) + " token '" + S(k) + "'");
            e = k[0] == '+' ? 1 : 3;
        }
        if (a >= A) refuse(ln, "undeclared axis Q" + std::to_string(a));
        if (a == prev) refuse(ln, "axis Q" + std::to_string(a) + " repeated in " + kLineFields[cur]);
        if (a < prev) refuse(ln, std::string("unsorted ") + kLineFields[cur] + " field");
        prev = a;
        ++cnt;
        if (cur == 0) F.tau_ax.push_back((int32_t)a);
        else if (cur == 1) F.rot.push_back({(int32_t)a, e});
        else F.late.push_back({(int32_t)a, e});
    }
    close();
    F.p.push_back(p);
    F.src_line.push_back(ln);
    F.tau_off.push_back(F.tau_ax.size());
    F.rot_off.push_back(F.rot.size());
    F.late_off.push_back(F.late.size());
    seen_error_ = true;
}

void Reader::handle(const Item& it) {
    const int ln = it.lineno;
    cur_line_ = ln;
    Head h;
    if (!parse_head(it.body, h)) refuse(ln, "unparseable line '" + S(it.body) + "'");
    if (h.name == "xdem") refuse(ln, "an xdem header is only valid as the first line");
    if (eq_ci(h.name, "error")) return error(it, h);
    if (eq_ci(h.name, "axis")) {   // an xdem extension: one spelling (SPEC §3.2), unlike stim's case-blind names
        if (h.name != "axis") refuse(ln, "instruction '" + S(h.name) + "' (xdem spells it 'axis')");
        return axis(it, h);
    }
    if (h.has_tag && ext_) refuse(ln, "tags are not part of xdem 1");
    split_ws(h.rest, tok_);
    // stim's number rule on every instruction's parens arguments
    if (h.has_args && !stim_args(h.args, args_)) refuse(ln, "bad argument '" + S(h.args) + "' (not a real number)");
    if (eq_ci(h.name, "detector")) {
        if (tok_.size() != 1)   // stim: exactly one target
            refuse(ln, "detector takes exactly 1 target, got " + std::to_string(tok_.size()));
        for (SV k : tok_) {
            const long long j = k.size() > 1 && dl_prefix(k) == 'D' ? int_token(k.substr(1), ln, "detector", k) : -1;
            if (j < 0) refuse(ln, "bad detector target '" + S(k) + "'");
            const long long d = j + shift_;
            if (ext_ && d >= F.D)
                refuse(ln, "declares detector D" + std::to_string(d) + " (header: " + std::to_string(F.D) +
                               " detectors)");
            if (d > max_det_) { max_det_ = d; max_line_ = ln; }
        }
    } else if (eq_ci(h.name, "logical_observable")) {
        if (h.has_args) refuse(ln, "logical_observable takes no arguments");   // stim: 0 arguments, 1 target
        if (tok_.size() != 1)
            refuse(ln, "logical_observable takes exactly 1 target, got " + std::to_string(tok_.size()));
        for (SV k : tok_) {
            const long long j = k.size() > 1 && dl_prefix(k) == 'L' ? int_token(k.substr(1), ln, "observable", k) : -1;
            if (j < 0) refuse(ln, "bad logical_observable target '" + S(k) + "'");
            if (ext_ && j >= F.O)
                refuse(ln, "declares observable L" + std::to_string(j) + " (header: " + std::to_string(F.O) +
                               " observables)");
            if (j > max_obs_) { max_obs_ = j; max_line_ = ln; }
        }
    } else if (eq_ci(h.name, "shift_detectors")) {
        if (tok_.size() != 1) refuse(ln, "shift_detectors needs exactly one target");
        const long long k = int_token(tok_[0], ln, "shift_detectors target", tok_[0]);
        if (k < 0) refuse(ln, "bad shift_detectors target '" + S(tok_[0]) + "'");
        shift_ = std::min(shift_ + k, kShiftSat);
    } else {
        refuse(ln, "unknown instruction '" + S(h.name) + "'");
    }
}

File Reader::run() {
    split_lines();
    recognize_v1();
    size_t first = 0;
    if (!items_.empty()) {
        const SV b = items_[0].body;
        size_t e = 0;
        while (e < b.size() && !is_ws(b[e])) ++e;
        if (b.substr(0, e) == "xdem") {
            ext_ = true;
            F.extended = true;
            header(items_[0]);
            first = 1;
        }
    }
    F.flip_off.assign(1, 0);
    F.tau_off.assign(1, 0);
    F.rot_off.assign(1, 0);
    F.late_off.assign(1, 0);
    // blocks: `repeat N {` opens (a line of its own, as stim writes it), a line `}` closes; unrolled in place
    const size_t n = items_.size();
    std::vector<long long> count(n, -1);
    std::vector<size_t> close(n, 0), open;
    for (size_t i = first; i < n; ++i) {
        const Item& it = items_[i];
        const SV b = it.body;
        SV t = b;
        while (!t.empty() && is_ws(t.back())) t.remove_suffix(1);
        if (t == "}") {
            if (open.empty()) refuse(it.lineno, "'}' closes no repeat block");
            close[open.back()] = i;
            open.pop_back();
            count[i] = -2;
            continue;
        }
        if (t.empty() || t.back() != '{') {
            if (t.find('{') != SV::npos || t.find('}') != SV::npos)
                refuse(it.lineno, "bad repeat block line (stim writes `repeat N {` and `}` on lines of their own)");
            continue;
        }
        Head h;
        if (!parse_head(t.substr(0, t.size() - 1), h) || !eq_ci(h.name, "repeat") || h.has_args)
            refuse(it.lineno, "bad repeat block header '" + S(t) + "'");
        if (h.has_tag && ext_) refuse(it.lineno, "tags are not part of xdem 1");
        split_ws(h.rest, tok_);
        const long long N = tok_.size() == 1 ? int_token(tok_[0], it.lineno, "repeat count", tok_[0]) : -1;
        if (N < 0) refuse(it.lineno, "bad repeat block header '" + S(t) + "'");
        count[i] = N;
        open.push_back(i);
    }
    if (!open.empty()) refuse(items_[open.back()].lineno, "repeat block not closed");
    std::function<void(size_t, size_t)> walk = [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            if (count[i] == -1) {
                handle(items_[i]);
            } else if (count[i] >= 0) {
                for (long long c = 0; c < count[i]; ++c) walk(i + 1, close[i]);
                i = close[i];
            }
        }
    };
    walk(first, n);
    finish();
    return std::move(F);
}

void Reader::finish() {
    const size_t L = F.p.size();
    if (ext_) {
        if (F.tc > (int64_t)L)
            refuse(header_line_, "tc " + std::to_string(F.tc) + " out of range (the file has " + std::to_string(L) +
                                     " error lines)");
    } else {
        const long long D = max_det_ + 1, O = max_obs_ + 1;
        if (D + O > kMaxColumns) refuse(max_line_, std::to_string(D + O) + " readout columns exceed 2^31 - 1");
        F.D = (int)D;
        F.O = (int)O;
        F.r = F.m = 0;
        F.tc = (int64_t)L;
    }
    F.flip_off = std::move(pend_off_);
    F.flip_cols.resize(pend_.size());
    for (size_t t = 0; t < pend_.size(); ++t)
        F.flip_cols[t] = (int32_t)(pend_[t] >= kObsTag ? F.D + (pend_[t] - kObsTag) : pend_[t]);
    std::vector<int64_t>().swap(pend_);
    if (ext_ && !F.axes.empty()) {
        check_header_rank();
        check_operators();
    }
}

// §3.5(b): rank[img | kappa] = rank kappa. Rows (kappa_a | img_a | {a}) eliminated on the kappa columns: a row whose
// kappa part reduces to 0 must have a zero img part.
void Reader::check_header_rank() {
    const size_t A = F.axes.size(), rw = words_for(F.r), nw = words_for((long long)F.D + F.O), aw = words_for((long long)A);
    Elim E(rw * 64, rw, nw, aw);
    std::vector<uint64_t> v(E.tw);
    for (size_t a = 0; a < A; ++a) {
        std::fill(v.begin(), v.end(), 0);
        std::copy(F.axes[a].k.begin(), F.axes[a].k.end(), v.begin());
        std::copy(F.axes[a].img.w.begin(), F.axes[a].img.w.end(), v.begin() + (long)rw);
        set_bit(v, (rw + nw) * 64 + a);
        if (E.reduce_or_insert(v.data()) && any(v.data() + rw, nw))
            refuse(axis_line_[a], "header: rank[img | kappa] != rank kappa (axes " +
                                      axis_list(v.data() + rw + nw, aw) + ": kappa sums to 0, img does not)");
    }
}

// §3.5(c): every line's x field is orthogonal to ker(kappa | sigma): a basis of the kernel (the combinations of the
// rows (kappa_a | sigma_a) that reduce to 0), then one parity per kernel vector per line (bit-sliced over the kernel).
void Reader::check_operators() {
    const size_t A = F.axes.size(), rw = words_for(F.r), mw = words_for(F.m), aw = words_for((long long)A);
    Elim E((rw + mw) * 64, rw + mw, 0, aw);   // key: kappa in words [0, rw), sigma in words [rw, rw + mw)
    std::vector<uint64_t> v(E.tw), ker;   // kernel vectors, aw words each
    size_t nk = 0;
    for (size_t a = 0; a < A; ++a) {
        std::fill(v.begin(), v.end(), 0);
        std::copy(F.axes[a].k.begin(), F.axes[a].k.end(), v.begin());
        for (int32_t hq : F.axes[a].hidden) set_bit(v, rw * 64 + (size_t)hq);
        set_bit(v, (rw + mw) * 64 + a);
        if (E.reduce_or_insert(v.data())) {
            ker.insert(ker.end(), v.begin() + (long)(rw + mw), v.end());
            ++nk;
        }
    }
    if (!nk) return;
    const size_t kw = words_for((long long)nk);
    std::vector<uint64_t> member(A * kw, 0), acc(kw);   // member[a] = the kernel vectors containing axis a
    for (size_t j = 0; j < nk; ++j)
        for (size_t w = 0; w < aw; ++w)
            for (uint64_t x = ker[j * aw + w]; x; x &= x - 1)
                set_bit(member, (w * 64 + (size_t)__builtin_ctzll(x)) * kw * 64 + j);
    for (size_t i = 0; i < F.num_lines(); ++i) {
        if (F.tau_off[i] == F.tau_off[i + 1]) continue;
        std::fill(acc.begin(), acc.end(), 0);
        for (uint64_t t = F.tau_off[i]; t < F.tau_off[i + 1]; ++t) {
            const uint64_t* mrow = &member[(size_t)F.tau_ax[t] * kw];
            for (size_t w = 0; w < kw; ++w) acc[w] ^= mrow[w];
        }
        for (size_t w = 0; w < kw; ++w)
            if (acc[w]) {
                const size_t j = w * 64 + (size_t)__builtin_ctzll(acc[w]);
                refuse(F.src_line[i], "x field is not an operator (it anticommutes with an odd number of the axes " +
                                          axis_list(&ker[j * aw], aw) +
                                          ", whose gauge and hidden-syndrome parts sum to 0)");
            }
    }
}

// GF(2) rank of rows of nw words each (Elim with the whole row as key, nothing carried or tracked).
size_t rank_of(std::vector<std::vector<uint64_t>> rows, size_t nw) {
    Elim E(nw * 64, nw, 0, 0);
    for (auto& v : rows) E.reduce_or_insert(v.data());
    return E.nrow;
}

}  // namespace

// ---------------------------------------------------------------------------------------------- public

File parse(const std::string& text) {
    Reader R(text);
    try {
        return R.run();
    } catch (const std::bad_alloc&) {
        throw XdemCapacityError(R.capacity_message());
    }
}

std::string to_hex(const Bits& b) {
    static const char* d = "0123456789abcdef";
    int top = (int)b.w.size() - 1;
    while (top >= 0 && b.w[(size_t)top] == 0) --top;
    if (top < 0) return "0";
    std::string s;
    for (int i = top; i >= 0; --i)
        for (int nib = 15; nib >= 0; --nib) {
            const int v = (int)((b.w[(size_t)i] >> (4 * nib)) & 15u);
            if (s.empty() && v == 0) continue;
            s.push_back(d[v]);
        }
    return s;
}

std::string emit(const File& F) {
    auto u = [](long long v) { return std::to_string(v); };
    std::string out = "xdem 1 detectors " + u(F.D) + " observables " + u(F.O) + " gauge " + u(F.r) + " hidden " +
                      u(F.m) + " tc " + u(F.tc) + "\n";
    auto target = [&](int j) { return j < F.D ? "D" + u(j) : "L" + u(j - F.D); };
    for (size_t a = 0; a < F.axes.size(); ++a) {
        const Axis& ax = F.axes[a];
        out += "axis Q" + u((long long)a);
        if (ax.b) out += " sign -";
        const char* kw = " img";
        for (size_t w = 0; w < ax.img.w.size(); ++w)
            for (uint64_t x = ax.img.w[w]; x; x &= x - 1) {
                out += kw; kw = ""; out += " " + target((int)(w * 64 + (size_t)__builtin_ctzll(x)));
            }
        kw = " gauge";
        for (size_t w = 0; w < ax.k.size(); ++w)
            for (uint64_t x = ax.k[w]; x; x &= x - 1) {
                out += kw; kw = ""; out += " g" + u((long long)(w * 64 + (size_t)__builtin_ctzll(x)));
            }
        kw = " hidden";
        for (int32_t h : ax.hidden) { out += kw; kw = ""; out += " h" + u(h); }
        out += "\n";
    }
    char pb[40];
    for (size_t i = 0; i < F.num_lines(); ++i) {
        std::snprintf(pb, sizeof pb, "%.17g", F.p[i]);     // the default export's probability spelling
        out += "error("; out += pb; out += ")";
        for (uint64_t t = F.flip_off[i]; t < F.flip_off[i + 1]; ++t) out += " " + target(F.flip_cols[t]);
        if (F.tau_off[i] < F.tau_off[i + 1]) {
            out += " x";
            for (uint64_t t = F.tau_off[i]; t < F.tau_off[i + 1]; ++t) out += " Q" + u(F.tau_ax[t]);
        }
        auto rots = [&](const char* kw2, const std::vector<uint64_t>& off, const std::vector<Rot>& v) {
            if (off[i] == off[i + 1]) return;
            out += kw2;
            for (uint64_t t = off[i]; t < off[i + 1]; ++t) out += (v[t].e == 1 ? " +Q" : " -Q") + u(v[t].axis);
        };
        rots(" rot", F.rot_off, F.rot);
        rots(" late", F.late_off, F.late);
        out += "\n";
    }
    return out;
}

Composed compose(const File& F, const std::vector<int>& ids_in) {
    for (int id : ids_in)
        if (id < 0 || (size_t)id >= F.num_lines())
            throw XdemFileError("xdem: line id " + std::to_string(id) + " out of range (the file has " +
                                std::to_string(F.num_lines()) + " error lines)");
    std::vector<int> ids(ids_in), order;
    std::sort(ids.begin(), ids.end());
    for (int id : ids) {                                  // file order; a line listed twice cancels (M_i^2 = I)
        if (!order.empty() && order.back() == id) order.pop_back();
        else order.push_back(id);
    }
    const size_t A = F.axes.size();
    Composed c;
    c.flip.w.assign(words_for((long long)F.D + F.O), 0);
    c.tau.assign((A + 63) / 64, 0);
    std::vector<uint8_t> eta(A, 0);
    auto rotate = [&](int id, const std::vector<uint64_t>& off, const std::vector<Rot>& v) {
        for (uint64_t t = off[(size_t)id]; t < off[(size_t)id + 1]; ++t) {
            const int a = v[t].axis;
            const bool neg = (c.tau[(size_t)a >> 6] >> (a & 63)) & 1ull;   // tau of the EARLIER lines only
            eta[(size_t)a] = (uint8_t)((eta[(size_t)a] + (neg ? 4 - v[t].e : v[t].e)) & 3);
        }
    };
    for (int id : order) {
        rotate(id, F.rot_off, F.rot);                     // the rotation acts first (SPEC §1.1) ...
        rotate(id, F.late_off, F.late);
        for (uint64_t t = F.flip_off[(size_t)id]; t < F.flip_off[(size_t)id + 1]; ++t) {
            const int j = F.flip_cols[t];
            c.flip.w[(size_t)j >> 6] ^= 1ull << (j & 63);
        }
        for (uint64_t t = F.tau_off[(size_t)id]; t < F.tau_off[(size_t)id + 1]; ++t) {   // ... then the X part
            const int a = F.tau_ax[t];
            c.tau[(size_t)a >> 6] ^= 1ull << (a & 63);
        }
    }
    for (size_t a = 0; a < A; ++a)
        if (eta[a]) c.eta.emplace_back((int)a, (int)eta[a]);
    return c;
}

// Ported from the dc-DEM v1 reader's law_into (19f2f78, lines 833-993 of its source): (D, O, r, AxisView) for the
// File, c.eta for c.E, and the two W-phase refusals turned into internal errors (unreachable after the header check
// §3.5(b)).
void law_into(int D, int O, int r, const AxisView& ax, const Composed& c, LawScratch& S) {
    const int n = D + O;
    const size_t nw = words_for(n);
    S.off.assign(c.flip.w.begin(), c.flip.w.end());
    S.rank = 0;
    S.odd.clear();
    S.oddE.clear();
    // E_a of an odd axis = the exponent of its FIRST entry in c.eta; c.eta strictly ascending by axis (compose(),
    // the sampler) makes that the entry itself, otherwise the first-entry search below applies.
    bool ascending = true;
    for (size_t i = 0; i < c.eta.size(); ++i) {
        const int a = c.eta[i].first, e = c.eta[i].second;
        if (i && a <= c.eta[i - 1].first) ascending = false;
        if (e == 2) {
            const uint64_t* img = ax[(size_t)a].img.w.data();
            for (size_t w = 0; w < S.off.size(); ++w) S.off[w] ^= img[w];
        } else {
            S.odd.push_back(a);
            S.oddE.push_back(e);
        }
    }
    if (S.odd.empty()) return;
    if (!ascending)
        for (size_t t = 0; t < S.odd.size(); ++t)
            for (const auto& [aa, ee] : c.eta)
                if (aa == S.odd[t]) { S.oddE[t] = ee; break; }
    // coin rows: row_j = XOR over odd a with k_a bit j of img_a (built by iterating each k_a's bits once); coins =
    // their fully reduced RREF, rows processed in order j = 0..r-1, kept in place (coin i = row crow[i]).
    S.rows.assign((size_t)r * nw, 0);
    for (int a : S.odd) {
        const uint64_t* img = ax[(size_t)a].img.w.data();
        const std::vector<uint64_t>& k = ax.kappa((size_t)a);
        for (size_t kw = 0; kw < k.size(); ++kw)
            for (uint64_t y = k[kw]; y; y &= y - 1) {
                uint64_t* row = &S.rows[(kw * 64 + (size_t)__builtin_ctzll(y)) * nw];
                for (size_t w = 0; w < nw; ++w) row[w] ^= img[w];
            }
    }
    // The coins are fully reduced at every step (coin i has bit piv[j] only for j = i), so reducing a vector x
    // modulo the coins is the XOR of the coins whose pivot bit is set in x on entry: one pass over x & pivmask
    // (XOR-ing a coin changes no other pivot bit of x), coin of pivot p = row prow[p].
    S.piv.clear();
    S.crow.clear();
    S.pm.assign(nw, 0);
    S.prow.resize((size_t)n);
    auto reduce = [&](uint64_t* x) {
        for (size_t w = 0; w < nw; ++w)
            for (uint64_t y = x[w] & S.pm[w]; y; y &= y - 1) {
                const uint64_t* ci = &S.rows[(size_t)S.prow[w * 64 + (size_t)__builtin_ctzll(y)] * nw];
                for (size_t v = 0; v < nw; ++v) x[v] ^= ci[v];
            }
    };
    for (int j = 0; j < r; ++j) {
        uint64_t* row = &S.rows[(size_t)j * nw];
        reduce(row);
        int p = -1;
        for (size_t w = nw; w-- > 0;)
            if (row[w]) { p = (int)(w * 64 + 63 - (size_t)__builtin_clzll(row[w])); break; }
        if (p < 0) continue;
        for (size_t i = 0; i < S.piv.size(); ++i) {
            uint64_t* ci = &S.rows[(size_t)S.crow[i] * nw];
            if ((ci[(size_t)p >> 6] >> (p & 63)) & 1ull)
                for (size_t w = 0; w < nw; ++w) ci[w] ^= row[w];
        }
        S.crow.push_back(j);
        S.piv.push_back(p);
        S.pm[(size_t)p >> 6] ^= 1ull << (p & 63);
        S.prow[(size_t)p] = j;
    }
    const size_t nc = S.piv.size();
    // W phases, bit-sliced over the columns j: u_a[j] = parity(s_j . img_a) for the private-pivot W basis s_j
    // (j not a pivot); ph(j) = sum E_a u_a[j] (mod 4) in (lo, hi); bb(j) = XOR b_a u_a[j].
    S.lo.assign(nw, 0);
    S.hi.assign(nw, 0);
    S.bb.assign(nw, 0);
    S.u.resize(nw);
    // linearity: the columns of U (rows u_a, t = row index) span a self-orthogonal space of GF(2)^m
    const size_t m = S.odd.size(), mw = (m + 63) / 64;
    S.cols.assign((size_t)n * mw, 0);
    uint64_t* u = S.u.data();
    for (size_t t = 0; t < m; ++t) {
        const Axis& x = ax[(size_t)S.odd[t]];
        const uint64_t* img = x.img.w.data();
        for (size_t w = 0; w < nw; ++w) u[w] = img[w];
        reduce(u);
        const uint64_t tb = 1ull << (t % 64);
        for (size_t w = 0; w < nw; ++w)
            for (uint64_t y = u[w]; y; y &= y - 1)
                S.cols[(w * 64 + (size_t)__builtin_ctzll(y)) * mw + t / 64] |= tb;
        const uint64_t e3 = S.oddE[t] == 3 ? ~0ull : 0ull, bm = x.b ? ~0ull : 0ull;
        for (size_t w = 0; w < nw; ++w) {
            const uint64_t uw = u[w];
            const uint64_t carry = S.lo[w] & uw;
            S.lo[w] ^= uw;
            S.hi[w] ^= carry ^ (e3 & uw);
            S.bb[w] ^= bm & uw;
        }
    }
    {
        // basis: one flat arena (nb rows of mw words), kept fully reduced, pivot p = lowest set bit of the vector
        // when inserted (bpm = pivot mask, brow[p] = its row); a column is reduced in place in the cols arena in one
        // pass over c & bpm, then checked pairwise and for self-parity, in that order.
        S.basis.clear();
        S.bpm.assign(mw, 0);
        S.brow.resize(m);
        size_t nb = 0;
        for (int j = 0; j < n; ++j) {
            uint64_t* cj = &S.cols[(size_t)j * mw];
            for (size_t w = 0; w < mw; ++w)
                for (uint64_t y = cj[w] & S.bpm[w]; y; y &= y - 1) {
                    const uint64_t* bk = &S.basis[(size_t)S.brow[w * 64 + (size_t)__builtin_ctzll(y)] * mw];
                    for (size_t v = 0; v < mw; ++v) cj[v] ^= bk[v];
                }
            size_t p = SIZE_MAX;
            for (size_t w = 0; w < mw && p == SIZE_MAX; ++w)
                if (cj[w]) p = w * 64 + (size_t)__builtin_ctzll(cj[w]);
            if (p == SIZE_MAX) continue;
            for (size_t k = 0; k < nb; ++k) {
                const uint64_t* bk = &S.basis[k * mw];
                int par = 0;
                for (size_t w = 0; w < mw; ++w) par ^= __builtin_parityll(bk[w] & cj[w]);
                if (par)
                    throw std::logic_error(
                        "xdem law: internal: W phase not linear (the odd axes' images are not pairwise even on W) "
                        "(unreachable after the header check §3.5(b), AUDIT Cor. 1.2(ii))");
            }
            int self = 0;
            for (size_t w = 0; w < mw; ++w) self ^= __builtin_parityll(cj[w]);
            if (self)
                throw std::logic_error(
                    "xdem law: internal: odd phase on W (the file's k vectors admit an odd dependency) "
                    "(unreachable after the header check §3.5(b), AUDIT Cor. 1.2(ii))");
            for (size_t k = 0; k < nb; ++k) {
                uint64_t* bk = &S.basis[k * mw];
                if ((bk[p / 64] >> (p % 64)) & 1ull)
                    for (size_t w = 0; w < mw; ++w) bk[w] ^= cj[w];
            }
            S.basis.insert(S.basis.end(), cj, cj + mw);
            S.bpm[p / 64] |= 1ull << (p % 64);
            S.brow[p] = (int)nb++;
        }
    }
    for (size_t w = 0; w < nw; ++w) {
        if (S.lo[w] & ~S.pm[w])
            throw std::logic_error(
                "xdem law: internal: odd phase on W (the file's k vectors admit an odd dependency) "
                "(unreachable after the header check §3.5(b), AUDIT Cor. 1.2(ii))");
        S.off[w] ^= (S.hi[w] ^ S.bb[w]) & ~S.pm[w];
    }
    reduce(S.off.data());
    // descending pivot order
    S.ord.resize(nc);
    for (size_t i = 0; i < nc; ++i) S.ord[i] = i;
    std::sort(S.ord.begin(), S.ord.end(), [&](size_t x, size_t y) { return S.piv[x] > S.piv[y]; });
    S.coins.resize(nc * nw);
    for (size_t i = 0; i < nc; ++i) {
        const uint64_t* ci = &S.rows[(size_t)S.crow[S.ord[i]] * nw];
        std::copy(ci, ci + nw, &S.coins[i * nw]);
    }
    S.rank = (int)nc;
}

void law_into(const File& F, const Composed& c, LawScratch& S) {
    AxisView v;
    v.base = F.axes.data();
    v.nbase = F.axes.size();
    law_into(F.D, F.O, F.r, v, c, S);
}

Law law(const File& F, const Composed& c) {
    LawScratch S;
    law_into(F, c, S);
    const size_t nw = words_for((long long)F.D + F.O);
    Law L;
    L.offset.w = std::move(S.off);
    L.coins.resize((size_t)S.rank);
    for (size_t i = 0; i < L.coins.size(); ++i)
        L.coins[i].w.assign(S.coins.begin() + (long)(i * nw), S.coins.begin() + (long)((i + 1) * nw));
    return L;
}

std::vector<int> late_axes(const File& F) {
    std::vector<uint8_t> seen(F.axes.size(), 0);
    for (const Rot& x : F.late) seen[(size_t)x.axis] = 1;
    std::vector<int> out;
    for (size_t a = 0; a < seen.size(); ++a)
        if (seen[a]) out.push_back((int)a);
    return out;
}

std::pair<bool, std::string> r2b_available(const File& F) {
    for (size_t i = (size_t)F.tc; i < F.num_lines(); ++i) {
        const std::string where = "line " + std::to_string(F.src_line[i]) + " (index \xe2\x89\xa5 tc) ";
        if (F.flip_off[i] < F.flip_off[i + 1] && F.flip_cols[F.flip_off[i]] < F.D)   // ascending: detectors first
            return {false, where + "flips detector D" + std::to_string(F.flip_cols[F.flip_off[i]])};
        if (F.rot_off[i] < F.rot_off[i + 1] || F.late_off[i] < F.late_off[i + 1])
            return {false, where + "carries a rotation"};
    }
    for (int a : late_axes(F)) {
        const Bits& img = F.axes[(size_t)a].img;
        for (size_t w = 0; w < img.w.size(); ++w)
            if (img.w[w]) {
                const long long j = (long long)(w * 64) + __builtin_ctzll(img.w[w]);
                if (j < F.D)
                    return {false, "late axis Q" + std::to_string(a) + " has detector D" + std::to_string(j) +
                                       " in its img"};
                break;
            }
    }
    return {true, ""};
}

int invisible_residues(const File& F) {
    const size_t nw = words_for((long long)F.D + F.O), dw = words_for(F.D);
    std::vector<uint8_t> in(F.axes.size(), 0);
    for (const Rot& x : F.rot) in[(size_t)x.axis] = 1;
    std::vector<std::vector<uint64_t>> full, det;
    for (size_t a = 0; a < in.size(); ++a) {
        if (!in[a]) continue;
        full.push_back(F.axes[a].img.w);
        std::vector<uint64_t> d(F.axes[a].img.w.begin(), F.axes[a].img.w.begin() + (long)dw);
        if (F.D & 63) d[dw - 1] &= (1ull << (F.D & 63)) - 1;
        det.push_back(std::move(d));
    }
    return (int)rank_of(std::move(full), nw) - (int)rank_of(std::move(det), dw);
}

}  // namespace qeccore::xdem
