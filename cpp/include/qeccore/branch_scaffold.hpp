#pragma once
// branch_scaffold — J-a N7 (switched OFF in production; reached only through the test API until J-b1): the
// per-plan scaffolding of the carried-branch design (design.final.md §2.10; xstage-r3 §1-§3; plan §1.1 rows
// 13-14, §3.8, r2.3, r2.6, R-JA-ES).
//
// A plan's TERM VIEW is its canonical content key and a term form of its residual C0 = Π_{j∈S} rot(B_j):
//   * kind 0 (diagonal): the stripped key (a ‖ 0xFF ‖ cz) and diag_plan_axes(nf) with S = every axis
//     (record_general.hpp: S_q = rot(−Z_q), CZ = rot(−Z_f) rot(−Z_s) rot(Z_f Z_s), up to a global phase);
//   * kind 2: the term key and the sampler's TermResidual axes with S = the reduced term set.
// Per plan (the J-b1 payload part, evicted with the payload):
//   * the plan's IDENTITY key = its kind byte ‖ the content key (tagged_plan_key: a kind-0 and a kind-2 content key
//     can be equal as bytes, so identities never compare bare content keys); its 128-bit digest
//     (plan_key_digest128_) is never an identity on its own: every resolve compares the FULL key
//     (PlanTableSet::resolve);
//   * E_S := {a ∈ G₀ : ∃ j ∈ S, ⟨g_a, B_j⟩ = 1} (Lemma R2′; normative per R-JA-ES), a bitset over the certified
//     generators in σ bit order, computed LAZILY on the first materialize / branches-core request and cached
//     (Y-5): the payload build never computes it (G-LAZY counts the computations);
//   * the term images B̂_j (j ∈ S): U† B_j U on the register F₀ ∪ E_S (its X part on certified rows lies in E_S
//     by the definition of E_S; Z on the other certified rows is a sign through eps), lazy with E_S;
//   * W_p per pivot (design §2.10): W_p = g_p · Π_{j∈S, ⟨g_p,B_j⟩=1} (−iB_j), g_p ∈ G₀ the generator dual to
//     the pivots (⟨Q_p', g_p⟩ = δ_pp' over the pivot pullbacks), g_p on the LEFT, the product in ascending j.
// The plan tables: PlanTableSet holds one table per distinct E_min (a sparse row list; ∅ for every plan
// today), each a map digest → (full key, internal id); resolve compares the full key, and a digest that two
// different keys share is a NAMED refusal (DigestCollision), never a merge.
// Nothing here reads the bare amplitudes α (G-ALPHA): only the frame (U, F₀, eps) and the certified rows.
// Preconditions (asserted once, on the lazy path): the axes in S commute pairwise (C0 = Π rot(B_j) is then
// order-free; moved_rows_ and pivot_frames_ rely on it).
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qeccore/framed_superposition.hpp"
#include "qeccore/pauli.hpp"
#include "qeccore/twirl_planes.hpp"

namespace qeccore {

struct Digest128 {
    uint64_t hi = 0, lo = 0;
    bool operator==(const Digest128& o) const { return hi == o.hi && lo == o.lo; }
    bool operator!=(const Digest128& o) const { return !(*this == o); }
};
// The 128-bit digest of a plan identity key (two independent 64-bit mixes of the key bytes).
Digest128 plan_key_digest128_(const std::string& key);
// The identity key of a plan: its kind byte, then its canonical content key.
std::string tagged_plan_key(int kind, const std::string& content_key);

// One plan's term view (see above).
struct PlanTermView {
    int kind = 0;                       // 0 diagonal, 2 kind 2
    std::string key;                    // the canonical content key
    std::vector<Pauli> axes;            // B_j (signed Hermitian, n wires)
    std::vector<uint64_t> S;            // the plan's term set (words over axes)
};

// B̂_j: U† B_j U on the register F₀ ∪ E_S, coordinates = the k free rows (bare.free order) then the E_S rows
// (ascending σ index): x, z on those coordinates, and free_dir_of's sign convention: phase = the frame image's phase,
// par = Σ_q z_q·eps_q over EVERY frame row q (the free and E_S rows included, as FreeDir's par does).
struct TermImage {
    int term = -1;
    std::vector<uint8_t> x, z;          // k + |E_S| coordinates
    uint8_t phase = 0, par = 0;
};

class PlanScaffold {
public:
    // The payload build: identity key, digest, kind. Never computes E_S (Y-5). G: the sampler's certified group
    // (outlives the scaffold); es_counter (optional): incremented once per E_S computation (the G-LAZY counter).
    PlanScaffold(PlanTermView v, const CertifiedGroupPlanes& G, long* es_counter = nullptr);
    const PlanTermView& view() const { return v_; }
    const std::string& identity_key() const { return id_; }
    const Digest128& digest() const { return dig_; }
    // E_S as a bitset over the certified generators (σ bit order, G.words words): lazy, cached.
    const std::vector<uint64_t>& ensure_es_();
    bool es_built() const { return es_built_; }
    // B̂_j for j ∈ S, ascending j (lazy with E_S). bare: the sampler's bare state (its frame only); every call must
    // pass the same object (std::logic_error otherwise: the images are cached).
    const std::vector<TermImage>& term_images_(const FramedSuperposition& bare);
    // W_p per pivot, given the pivot records' pullbacks Q_p (pivot slot order). Throws std::logic_error if the
    // pullback planes are dependent (an engine bug: pivots have independent planes, Theorem 1).
    std::vector<Pauli> pivot_frames_(const std::vector<Pauli>& Qpiv) const;
    // The certified rows moved by Φ(S) (C0† g_a C0 ≠ ±g_a), a bitset like E_S: a subset of E_S (R-JA-ES).
    std::vector<uint64_t> moved_rows_() const;

private:
    PlanTermView v_;
    const CertifiedGroupPlanes* G_;
    long* es_count_;
    std::string id_;
    Digest128 dig_;
    bool es_built_ = false;
    std::vector<uint64_t> es_;
    bool img_built_ = false;
    const FramedSuperposition* img_bare_ = nullptr;
    std::vector<TermImage> img_;
    void assert_commuting_() const;
};

// A digest shared by two different keys: refused by name on the full-key compare.
struct DigestCollision : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One plan table per distinct E_min (sparse row list, ascending σ indices; ∅ = the only table today).
class PlanTableSet {
public:
    // Insert (or find) the plan with this IDENTITY key (tagged_plan_key) in the E_min table; returns its internal id
    // (dense, first sight; never exported). A digest equal to another key's: DigestCollision.
    long insert(const std::vector<int>& emin, const std::string& key);
    // Resolve a handle (E_min, digest, full key) to the internal id: FULL-key compare; −1 if absent;
    // DigestCollision if the digest is held by a different key.
    long resolve(const std::vector<int>& emin, const Digest128& d, const std::string& key) const;
    int num_tables() const { return (int)tables_.size(); }
    long size() const;
    // TEST hook (G-DIG): every key's digest is replaced by `forced` (a forced collision).
    void force_digest_for_test(const Digest128& forced) { forced_ = forced; force_ = true; }

private:
    struct Entry { std::string key; long id; };
    Digest128 digest_of_(const std::string& key) const { return force_ ? forced_ : plan_key_digest128_(key); }
    struct DigestHash {
        size_t operator()(const Digest128& d) const { return (size_t)(d.lo ^ (d.hi * 0x9E3779B97F4A7C15ull)); }
    };
    std::map<std::vector<int>, std::unordered_map<Digest128, Entry, DigestHash>> tables_;
    long next_id_ = 0;
    bool force_ = false;
    Digest128 forced_;
};

}  // namespace qeccore
