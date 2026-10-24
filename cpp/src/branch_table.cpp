#include "qeccore/branch_table.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "qeccore/engine_version.hpp"

namespace qeccore {
namespace {
void put_pauli(std::string& s, const Pauli& P) {
    s.append(reinterpret_cast<const char*>(P.x.data()), P.x.size() * sizeof(uint64_t));
    s.append(reinterpret_cast<const char*>(P.z.data()), P.z.size() * sizeof(uint64_t));
    s.push_back((char)(P.phase & 3));
}
std::string frame_rows(const FramedSuperposition& fs) {
    std::string s;
    const int32_t n = fs.n();
    s.append(reinterpret_cast<const char*>(&n), sizeof n);
    for (int a = 0; a < fs.n(); ++a) { put_pauli(s, fs.U.Zrow[(size_t)a]); put_pauli(s, fs.U.Xrow[(size_t)a]); }
    for (size_t a = 0; a < fs.eps.size(); ++a) s.push_back((char)(fs.eps[a] & 1));
    s.push_back('|');
    for (int f : fs.free) s.append(reinterpret_cast<const char*>(&f), sizeof f);
    return s;
}
}  // namespace

FrameToken frame_token_producer(const std::string& circuit_text, const FramedSuperposition& fs) {
    std::string s = std::string("producer|") + kEngineVersion + "|";
    const Digest128 c = plan_key_digest128_(circuit_text);
    s.append(reinterpret_cast<const char*>(&c), sizeof c);
    s += frame_rows(fs);
    return FrameToken{plan_key_digest128_(s), true};
}

FrameToken frame_token_rows(const FramedSuperposition& fs) {
    return FrameToken{plan_key_digest128_(std::string("rows|") + frame_rows(fs)), false};
}

int BranchTable::add(const BranchHandle& h, const std::string& full_key, const std::vector<int>& emin) {
    if (!tok_.producer)
        throw BranchRefusal("add_branch: this ideal state carries a rows-only frame token (no producer provenance: "
                            "built from text or rows); it accepts branch 0 only — refused");
#ifndef XTIM_MUTATE_BR_TOKEN
    if (h.token != tok_.d)
        throw BranchRefusal("add_branch: the handle's frame token is not this ideal state's (a branch of another "
                            "producer or frame) — refused");
#endif   // mutation build only (MUT-BR-TOKEN): the token check skipped
#ifndef XTIM_MUTATE_BR_DIGEST
    if (plan_key_digest128_(full_key) != h.digest)
        throw BranchRefusal("add_branch: the handle's plan digest is not the producer content's (digest mismatch) — "
                            "refused");
#endif   // mutation build only (MUT-BR-DIGEST): the full-key compare skipped
    if (!std::is_sorted(emin.begin(), emin.end()))
        throw std::invalid_argument("add_branch: E_min must be an ascending row list");
    {   // the content identity is E_min-independent: one digest → one full key across every table
        const Digest128 d = force_ ? forced_ : plan_key_digest128_(full_key);
        auto ins = keys_.emplace(std::make_pair(d.hi, d.lo), full_key);
        if (!ins.second && ins.first->second != full_key)
            throw DigestCollision("add_branch: plan digest collision: two different producer contents share one "
                                  "128-bit digest (refused by name; the full-key compare never merges branches)");
    }
    const long pid = tables_.insert(emin, full_key);       // DigestCollision on a shared digest
    auto key = std::make_pair(pid, h.z);
    auto it = ids_.find(key);
    if (it != ids_.end()) return it->second;
    br_.push_back(Entry{h, full_key, emin, pid});
    const int id = (int)br_.size();
    ids_.emplace(std::move(key), id);
    emins_.emplace(emin, 0);
    return id;
}

const std::vector<int>& BranchTable::emin(int b) const {
    static const std::vector<int> empty;
    if (b == 0) return empty;
    if (b < 0 || b > (int)br_.size()) throw std::out_of_range("branch id out of range");
    return br_[(size_t)b - 1].emin;
}

Digest128 BranchTable::digest() const {
    std::vector<std::string> rows;
    for (const Entry& e : br_) {
        std::string r(reinterpret_cast<const char*>(&e.h.digest), sizeof e.h.digest);
        r.push_back('|');
        r.append(e.h.z.begin(), e.h.z.end());
        r.push_back('|');
        for (int a : e.emin) r.append(reinterpret_cast<const char*>(&a), sizeof a);
        rows.push_back(std::move(r));
    }
    std::sort(rows.begin(), rows.end());
    std::string s(reinterpret_cast<const char*>(&tok_.d), sizeof tok_.d);
    for (const std::string& r : rows) { s += '#'; s += r; }
    return plan_key_digest128_(s);
}

long* BranchTable::map_find(long plan, int branch) {
    ++probes_;
    auto it = map_.find(((uint64_t)plan << 20) ^ (uint64_t)branch);
    return it == map_.end() ? nullptr : &it->second;
}

void BranchTable::map_put(long plan, int branch, long root) {
    if (branch == 0) throw std::logic_error("branch map: branch 0's root is inline in the plan payload (engine bug)");
    map_[((uint64_t)plan << 20) ^ (uint64_t)branch] = root;
}

}  // namespace qeccore
