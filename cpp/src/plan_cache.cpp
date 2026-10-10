// plan_cache — generic disk-cache serialization primitives (see plan_cache.hpp). Extracted from the
// (research-only) twirl_plan cache so future compile-artifact caches share one audited framing.
#include "qeccore/plan_cache.hpp"

namespace qeccore {
namespace plan_cache {

uint64_t fnv1a64(const char* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= (uint8_t)p[i]; h *= 1099511628211ull; }
    return h;
}

Writer begin_blob(const char magic[4], uint32_t version) {
    Writer w;
    w.raw(magic, 4);
    w.pod(version);
    return w;
}

std::string finish_blob(Writer& w) {
    // Hash the payload only (everything past the 4-byte magic + 4-byte version).
    const uint64_t h = fnv1a64(w.buf.data() + 8, w.buf.size() - 8);
    w.pod(h);
    return std::move(w.buf);
}

Reader open_blob(const std::string& blob, const char magic[4]) {
    Reader bad;
    bad.ok = false;
    if (blob.size() < 16) return bad;                  // 4 magic + 4 version + >=0 payload + 8 hash
    if (std::memcmp(blob.data(), magic, 4) != 0) return bad;
    const size_t payload_len = blob.size() - 8 - 8;    // minus magic+version and the trailing hash
    uint64_t stored = 0;
    std::memcpy(&stored, blob.data() + blob.size() - 8, 8);
    if (fnv1a64(blob.data() + 8, payload_len) != stored) return bad;
    Reader r;
    r.p = blob.data() + 4;                              // positioned at the version field
    r.end = blob.data() + blob.size() - 8;             // excludes the trailing hash
    r.ok = true;
    return r;
}

}  // namespace plan_cache
}  // namespace qeccore
