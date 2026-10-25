#pragma once
// branch_table — J-b1 P9a: the IDENTITY layer of the carried-branch design (design.final.md §2.10, "Identity by
// CONTENT"; plan §2.1 P9a). Reached through the test API until F.
//
//   * frame_token = H(ENGINE_VERSION, circuit digest, frame rows (U, F₀, eps), n) for a producer-exported ideal state;
//     a state without provenance gets a rows-only token H(frame rows, n) and accepts branch 0 only.
//   * A branch is identified by its handle h = (frame_token, plan_key_digest₁₂₈, z): the digest of the producer's
//     tagged content key, and the record's Born word z (the u = 0 sector). Every registration is backed by a FULL-KEY
//     compare against the producer's content (a digest collision is a named refusal, never a merge).
//   * Branches are registered per distinct E_min (one plan table per E_min; the dense internal id is never exported).
//   * branch_table_digest = H(frame_token, sorted (key digest, z, E_min)).
//   * The sparse (plan, branch) → trie-root map holds branches ≠ 0 only (branch 0's root stays inline in the plan
//     payload): a branch-0 shot never probes it (op counter).
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/branch_scaffold.hpp"
#include "qeccore/framed_superposition.hpp"

namespace qeccore {

struct FrameToken {
    Digest128 d;
    bool producer = false;          // false: rows-only (no provenance): branch 0 only
};
// The producer token of an exported ideal state, and the rows-only token.
FrameToken frame_token_producer(const std::string& circuit_text, const FramedSuperposition& fs);
FrameToken frame_token_rows(const FramedSuperposition& fs);

struct BranchHandle {
    Digest128 token;                // the producer's frame token
    Digest128 digest;               // plan_key_digest128_ of the producer's tagged content key
    std::vector<uint8_t> z;         // the Born word (one byte per direction)
};

// Named refusals of add_branch.
struct BranchRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class BranchTable {
public:
    explicit BranchTable(FrameToken t) : tok_(t) {}
    const FrameToken& token() const { return tok_; }
    // Register a producer branch. full_key: the producer's tagged content key the handle's digest stands for (resolved
    // by the caller); emin: the branch's canonical E (sparse row list, ascending). Returns the branch id (>= 1; branch
    // 0 is the ideal state itself); an identical registration returns the same id. Refusals by name: a rows-only
    // ideal state (no provenance); a handle of another frame token; a digest that is not the full key's; a digest
    // that two different keys share (DigestCollision).
    int add(const BranchHandle& h, const std::string& full_key, const std::vector<int>& emin);
    int num_branches() const { return (int)br_.size() + 1; }
    int distinct_emin() const { return (int)emins_.size(); }
    const std::vector<int>& emin(int b) const;
    Digest128 digest() const;                              // branch_table_digest
    // The sparse (plan, branch) map: branch 0 is never stored (inline); every lookup counts one probe.
    long* map_find(long plan, int branch);
    void map_put(long plan, int branch, long root);
    long map_probes() const { return probes_; }
    void force_digest_for_test(const Digest128& d) { tables_.force_digest_for_test(d); forced_ = d; force_ = true; }

private:
    struct Entry { BranchHandle h; std::string key; std::vector<int> emin; long plan_id; };
    FrameToken tok_;
    std::vector<Entry> br_;                                // id − 1 → entry
    std::map<std::pair<long, std::vector<uint8_t>>, int> ids_;   // (plan id, z) → branch id
    PlanTableSet tables_;                                  // one table per distinct E_min
    std::map<std::vector<int>, int> emins_;
    std::unordered_map<uint64_t, long> map_;               // (plan, branch) → trie root (branch ≠ 0 only)
    long probes_ = 0;
    std::map<std::pair<uint64_t, uint64_t>, std::string> keys_;   // every registered digest → its full key
    bool force_ = false;
    Digest128 forced_;
};

}  // namespace qeccore
