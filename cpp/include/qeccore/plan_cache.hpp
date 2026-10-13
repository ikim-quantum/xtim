#pragma once
// plan_cache — generic, reusable disk-cache serialization primitives, extracted from the (now
// research-only) twirl_plan cache so other compile-artifact caches can share them (P3-T3 per-branch
// boundary-reference cache is the first consumer). NOTHING here is twirl-specific.
//
// The primitives are:
//   * fnv1a64            — a byte-span integrity hash (catches a structurally-valid but corrupt blob),
//   * Writer / Reader    — append-only / bounds-checked binary (de)serializers for raw bytes, trivially
//                          copyable PODs, and length-prefixed strings / vectors,
//   * begin_blob / finish_blob / open_blob — the magic + version + trailing-hash FRAMING, with the
//     corrupt-entry silent-rebuild contract: open_blob returns a Reader with ok==false on ANY anomaly
//     (bad magic, truncation, hash mismatch) so the caller can silently rebuild.
//
// Byte-identity contract: PODs are copied as their raw native bytes (no decimal round-trip), so a
// round-tripped payload is BIT-identical to the freshly written one. Caches built on this are
// machine-local (the caller keys them on the engine-binary hash), so raw native-endian POD copies
// are safe — a blob never crosses to a different binary / architecture.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace qeccore {
namespace plan_cache {

// FNV-1a 64-bit over a byte span — the blob's integrity check.
uint64_t fnv1a64(const char* p, size_t n);

// Append-only binary writer.
struct Writer {
    std::string buf;
    void raw(const void* p, size_t n) { buf.append(reinterpret_cast<const char*>(p), n); }
    template <class T> void pod(const T& v) { raw(&v, sizeof(T)); }
    void str(const std::string& s) { pod((uint64_t)s.size()); raw(s.data(), s.size()); }
    template <class T> void vpod(const std::vector<T>& v) {   // T trivially copyable
        pod((uint64_t)v.size());
        if (!v.empty()) raw(v.data(), v.size() * sizeof(T));
    }
};

// Bounds-checked binary reader over [p, end). Any short read sets ok=false (sticky).
struct Reader {
    const char* p = nullptr;
    const char* end = nullptr;
    bool ok = true;
    void raw(void* d, size_t n) {
        if (!ok || (size_t)(end - p) < n) { ok = false; return; }
        std::memcpy(d, p, n); p += n;
    }
    template <class T> void pod(T& v) { raw(&v, sizeof(T)); }
    void str(std::string& s) {
        uint64_t n = 0; pod(n);
        if (!ok || (size_t)(end - p) < n) { ok = false; return; }
        s.assign(p, (size_t)n); p += n;
    }
    template <class T> void vpod(std::vector<T>& v) {
        uint64_t n = 0; pod(n);
        if (!ok || (size_t)(end - p) < n * sizeof(T)) { ok = false; return; }
        v.resize((size_t)n);
        if (n) { std::memcpy(v.data(), p, (size_t)n * sizeof(T)); p += (size_t)n * sizeof(T); }
    }
    // True iff the whole framed payload was consumed exactly (trailing bytes ⇒ corrupt / version skew).
    bool at_end() const { return p == end; }
};

// Blob layout: [4-byte magic][uint32 version][payload...][uint64 FNV-1a hash of the payload].
// (The hash covers the payload only — the version is guarded by an explicit equality check on read.)

// Start a blob: writes the 4-byte magic + the version. The caller then serializes the payload into
// the returned Writer.
Writer begin_blob(const char magic[4], uint32_t version);

// Finish a blob: append the trailing integrity hash (over the payload — everything past magic+version)
// and return the completed blob. `w` must have come from begin_blob.
std::string finish_blob(Writer& w);

// Open a blob: validate the magic and the trailing integrity hash. On success, return a Reader
// positioned at the VERSION field (so the caller reads + checks the version, then the payload, then
// calls at_end()). On ANY anomaly (too short, bad magic, hash mismatch) the Reader has ok==false.
Reader open_blob(const std::string& blob, const char magic[4]);

}  // namespace plan_cache
}  // namespace qeccore
