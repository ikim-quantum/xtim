// test_threads — threads (fixplan threads/fixplan.md, gates G1 / G2): TwirlRecordSampler::run(shots, threads) emits
// the bytes of run(shots) for every thread count. The C++ driver of the TSan gate (G2): built with -fsanitize=thread
// in a scratch build directory it must report nothing.
//
// The circuits: a small repetition code (always), cultivation d5 (tests/data, always) and the private CCZ factory a
// when present (tests/private/ccz_factory_a.stim, or $XTIM_PRIVATE_FIXTURES). Each is compiled the way the bindings
// compile a TwirlSampler (normalize for the twirl sampler, the framed family bare state, circuit channels), then for
// T in {2, 7}: fresh and warm samplers, a seeded call and a continuation, compared byte for byte with T = 1 (dets, obs,
// decisions) and on the shot-determined counters (used, clean, diag, channel_one_counts).
//
// Must end main() with REPORT() (check.hpp).
#include "check.hpp"

#include <cstdio>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "qeccore/circuit_ir.hpp"
#include "qeccore/normal_form.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/bare_state_errors.hpp"
#include "qeccore/record_fidelity.hpp"
#include "qeccore/thread_pool.hpp"
#include "qeccore/twirl_sampler.hpp"
#include "../testapi/twirl_test_access.hpp"   // the layout-3 record sink (test API: header-only forwarders)
#include <new>

// Fix-2 F1: the test's allocation-failure seam (this executable only): with g_new_countdown armed, the allocation that
// many allocations ahead ON THE CALLING THREAD (fix-3: a thread_local flag, so the failure is deterministic and never
// lands in a worker) throws std::bad_alloc, once.
static std::atomic<long>& new_countdown_();
static thread_local bool t_alloc_seam = false;
void* operator new(std::size_t n) {
    std::atomic<long>& c = new_countdown_();
    if (t_alloc_seam && c.load(std::memory_order_relaxed) >= 0 && c.fetch_sub(1) == 0) throw std::bad_alloc();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace qeccore;

static const char kRep[] =
    "R 0 1 2 3 4\n"
    "X_ERROR(0.05) 0 1 2\n"
    "CX 0 3 1 3\n"
    "DEPOLARIZE2(0.02) 0 3 1 3\n"
    "CX 1 4 2 4\n"
    "DEPOLARIZE2(0.02) 1 4 2 4\n"
    "M(0.01) 3 4\n"
    "DETECTOR rec[-2]\n"
    "DETECTOR rec[-1]\n"
    "X_ERROR(0.05) 0 1 2\n"
    "M 0 1 2\n"
    "OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]\n";

struct Bundle {
    bool ok = false;
    std::string error;
    Circuit deferred;
    std::vector<std::pair<int, int>> reads;
    std::vector<std::vector<int>> detectors;
    std::vector<std::pair<int, std::vector<int>>> observables;
    TwirlRecordOptions opt;
    std::unique_ptr<FramedSuperposition> bare;
    int ndet = 0, nobs = 0, ndec = 0;
    std::vector<int> out_wires;                    // OUTPUT_QUBITS -> deferred wires (the bindings' out_wires_)
};

// The bindings' TwirlSampler front end (family_sampler.hpp, PyFamilySampler's constructor), port-free circuits.
static Bundle compile(const std::string& text, long selfcheck) {
    Bundle b;
    ParsedStim ps = parse_stim_circuit(text);
    if (!ps.ok()) { b.error = "parse failed"; return b; }
    NormalizePolicy pol;
    pol.coherentize = true;
    pol.coherentize_all = true;
    pol.defer = true;
    pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject;
    pol.hoist_feedback_on_class_exit = true;
    pol.zero_noise_transparent = true;
    pol.sparse_class_table = true;
    NormalizeResult nr = normalize(ps.circuit, pol);
    for (const Instr& ins : nr.coherent.stream)
        if (ins.kind == Instr::Kind::Measure) {
            b.opt.rec_flip.push_back(ins.readout_flip_p);
            b.opt.rec_invert.push_back(ins.invert ? 1 : 0);
        }
    {
        size_t j = 0;
        for (const Instr& ins : nr.normalized.stream)
            if (ins.kind == Instr::Kind::Measure) {
                if (j < b.opt.rec_invert.size() && ins.invert) b.opt.rec_invert[j] ^= 1;
                ++j;
            }
    }
    b.deferred = std::move(nr.normalized);
    for (const OutputPort& op : ps.circuit.outputs)
        for (int q : op.qubits)
            if (q >= 0 && q < (int)nr.map.final_wire.size()) b.out_wires.push_back(nr.map.final_wire[(size_t)q]);
    for (const auto& r : nr.map.terminal_reads) b.reads.push_back({(int)r.second, r.first});
    b.detectors = ps.detectors;
    b.observables.assign(ps.observables.begin(), ps.observables.end());
    b.opt.decisions = ps.decisions;
    b.ndet = (int)b.detectors.size();
    for (const auto& kv : b.observables) b.nobs = std::max(b.nobs, kv.first + 1);
    for (const auto& kv : b.opt.decisions) b.ndec = std::max(b.ndec, kv.first + 1);
    FramedBareState bs = build_bare_state_framed_family(b.deferred);
    if (bs.rejected) { b.error = "bare state rejected: " + bs.reject_reason; return b; }
    b.bare = std::make_unique<FramedSuperposition>(std::move(bs.state));
    b.opt.circuit_channels = true;
    b.opt.selfcheck = selfcheck;
    b.opt.p_factor = 1.0;
    b.ok = true;
    return b;
}

struct Out {
    std::vector<uint8_t> d, o, c;
    long used = 0, clean = 0, diag = 0, early = 0, late = 0;
    std::vector<long long> chan;
    std::vector<std::string> rec;                // per recorded row: kind ‖ content ‖ route ‖ P_rec ‖ y ‖ u z ‖ ff
};
enum Mode { kSample = 0, kPostselect = 1, kRecord = 2, kRecordPs = 3 };

// One sampler, a list of calls (shots, seed; seed 0 = continue), at `threads`.
static bool run_calls(const Bundle& b, const std::vector<std::pair<long, uint64_t>>& calls, int threads, Out& out,
                      Mode mode = kSample) {
    using TA = TwirlSamplerTestAccess;
    auto s = TwirlRecordSampler::make(*b.bare, b.deferred, b.reads, b.detectors, b.observables, 1, b.opt);
    if (s->setup_error()) { std::fprintf(stderr, "setup_error %d\n", s->setup_error()); return false; }
    const int DBB = (b.ndet + 7) / 8, OBB = (b.nobs + 7) / 8, DECB = (b.ndec + 7) / 8;
    std::vector<uint8_t> flags((size_t)b.ndet, 0);
    for (int q = 0; q < b.ndet; q += 2) flags[(size_t)q] = 1;
    for (const auto& c : calls) {
        if (c.second && !s->set_seed(c.second)) return false;
        std::vector<uint8_t> d((size_t)c.first * DBB, 0), o((size_t)c.first * OBB, 0), dc((size_t)c.first * DECB, 0);
        // the layout-3 record buffers (as PyFamilyRecords holds them)
        const int n = b.bare->n(), PNW = (n + 63) / 64, YW = s->record_value_words(),
                  UZW = s->record_operator_count();
        std::vector<int32_t> handle((size_t)c.first, -1), nuz((size_t)c.first, 0), branch((size_t)c.first, 0);
        std::vector<uint8_t> route((size_t)c.first, 0), uz((size_t)c.first * UZW, 0), ff((size_t)c.first, 0);
        std::vector<uint64_t> prec((size_t)c.first * 2 * PNW, 0), y((size_t)c.first * YW, 0);
        TA::FamilyRecordSink rs;
        rs.cap = c.first; rs.PNW = PNW; rs.YW = YW; rs.UZW = UZW;
        rs.handle = handle.data(); rs.route = route.data(); rs.prec = prec.data(); rs.y = y.data();
        rs.uz = uz.data(); rs.nuz = nuz.data(); rs.ff = ff.data(); rs.branch = branch.data();
        const bool rec = mode == kRecord || mode == kRecordPs, ps = mode == kPostselect || mode == kRecordPs;
        if (rec) TA::set_family_record_sink(*s, &rs);
        if (ps) s->set_postselect(flags);
        FamilySink fs;
        fs.dets = DBB ? d.data() : nullptr; fs.DBB = DBB;
        fs.obs = OBB ? o.data() : nullptr; fs.OBB = OBB;
        fs.dec = DECB ? dc.data() : nullptr; fs.DECB = DECB;
        fs.ndec = b.ndec;
        fs.born_oi = s->born_obs_index();
        fs.born_ref = s->born_obs_ref();
        s->set_family_sink(&fs);
        const bool ok = s->run(c.first, threads);
        s->set_family_sink(nullptr);
        if (rec) TA::set_family_record_sink(*s, nullptr);
        if (ps) { out.early += s->postselect_early(); out.late += s->postselect_late(); s->set_postselect({}); }
        if (!ok) { std::fprintf(stderr, "run failed (threads %d)\n", threads); return false; }
        if (!ps && fs.row != c.first) { std::fprintf(stderr, "rows %ld != %ld\n", fs.row, c.first); return false; }
        if (rec && rs.row != fs.row) { std::fprintf(stderr, "records %ld != rows %ld\n", rs.row, fs.row); return false; }
        d.resize((size_t)fs.row * DBB); o.resize((size_t)fs.row * OBB); dc.resize((size_t)fs.row * DECB);
        if (rec) {
            const auto& C = TA::family_record_contents(*s);
            for (long r = 0; r < rs.row; ++r) {
                const int h = handle[(size_t)r];
                if (h < 0 || (size_t)h >= C.size()) { std::fprintf(stderr, "row %ld without content\n", r); return false; }
                std::string k(1, (char)C[(size_t)h].kind);
                k += C[(size_t)h].content;
                k.push_back((char)route[(size_t)r]);
                k.append(reinterpret_cast<const char*>(&prec[(size_t)r * 2 * PNW]), 2 * (size_t)PNW * 8);
                k.append(reinterpret_cast<const char*>(&y[(size_t)r * YW]), (size_t)YW * 8);
                k.append(reinterpret_cast<const char*>(&uz[(size_t)r * UZW]), (size_t)UZW);
                k.push_back((char)nuz[(size_t)r]);
                k.push_back((char)ff[(size_t)r]);
                out.rec.push_back(std::move(k));
            }
        }
        if (threads > 1 && s->threads_effective() != threads)
            std::fprintf(stderr, "  note: threads %d effective %d (%s)\n", threads, s->threads_effective(),
                         s->threads_limit().c_str());
        out.d.insert(out.d.end(), d.begin(), d.end());
        out.o.insert(out.o.end(), o.begin(), o.end());
        out.c.insert(out.c.end(), dc.begin(), dc.end());
    }
    out.used = s->used();
    out.clean = s->clean_shots();
    out.diag = s->diag_shots();
    out.chan = s->channel_one_counts();
    return true;
}

static void check_circuit(const char* name, const std::string& text, long selfcheck,
                          const std::vector<std::pair<long, uint64_t>>& calls, const std::vector<int>& threads,
                          Mode mode = kSample) {
    Bundle b = compile(text, selfcheck);
    if (!b.ok) { std::fprintf(stderr, "%s: %s\n", name, b.error.c_str()); CHECK(b.ok); return; }
    if ((mode == kPostselect || mode == kRecordPs) && b.ndet == 0) return;
    Out ref;
    const bool ok1 = run_calls(b, calls, 1, ref, mode);
    CHECK(ok1);
    if (!ok1) return;
    for (int T : threads) {
        Out got;
        const bool ok = run_calls(b, calls, T, got, mode);
        CHECK(ok);
        if (!ok) continue;
        const bool same = got.d == ref.d && got.o == ref.o && got.c == ref.c && got.rec == ref.rec;
        const bool cnt = got.used == ref.used && got.clean == ref.clean && got.diag == ref.diag && got.chan == ref.chan &&
                         got.early == ref.early && got.late == ref.late;
        if (!same || !cnt)
            std::fprintf(stderr, "FAIL %s threads %d: bytes %s, counters %s\n", name, T, same ? "ok" : "DIFFER",
                         cnt ? "ok" : "DIFFER");
        CHECK(same);
        CHECK(cnt);
    }
    static const char* mn[] = {"sample", "postselect", "record", "record+postselect"};
    std::printf("  %s [%s]: threads 1 vs", name, mn[mode]);
    for (int T : threads) std::printf(" %d", T);
    std::printf(" ok\n");
}

// born_trie_merge: a lane's grown copy merges by path; a node both evaluated must agree bit for bit (engine-bug throw).
static void test_trie_merge() {
    const double nan = std::nan("");
    std::vector<BornNode> dst = {{0.5, {1, 0}}, {nan, {0, 0}}};          // root evaluated, kid 0 allocated
    std::vector<BornNode> src = dst;                                     // the lane's copy, grown:
    src[1].p0 = 0.25;                                                    //   kid 0 evaluated
    src[0].kid[1] = 2;                                                   //   kid 1 allocated and evaluated
    src.push_back({0.75, {0, 3}});
    src.push_back({nan, {0, 0}});                                        //   its kid 1 allocated
    std::vector<BornNode> other = dst;                                   // a second lane: kid 0 evaluated too
    other[1].p0 = 0.25;
    born_trie_merge(dst, src);
    born_trie_merge(dst, other);
    CHECK(dst.size() == 4);
    CHECK(dst[0].p0 == 0.5 && dst[1].p0 == 0.25);
    CHECK(dst[0].kid[1] != 0 && dst[dst[0].kid[1]].p0 == 0.75 && dst[dst[0].kid[1]].kid[1] != 0);
    std::vector<BornNode> bad = dst;
    bad[1].p0 = std::nextafter(0.25, 1.0);                               // one ulp off
    bool threw = false;
    try { born_trie_merge(dst, bad); } catch (const std::logic_error&) { threw = true; }
    CHECK(threw);
    std::printf("  born_trie_merge: by path, mismatch refused ok\n");
}

// The pool's withdrawal: with every worker held by another call, a threaded call's worker jobs are never claimed — the
// caller withdraws them and does all the work itself (the lanes pull their work), with the same bytes.
template <class F>
static void with_busy_pool(F f) {
    const int W = std::max(1, ThreadPool::instance().workers());   // hold every worker the pool has (no growth below)
    std::atomic<int> started{0};
    std::atomic<bool> release{false};
    std::thread hold([&] {
        ThreadPool::instance().run(W + 1, [&](int i) {
            if (i) { ++started; while (!release) std::this_thread::yield(); }
            else while (started < W) std::this_thread::yield();   // every job claimed before this call can withdraw one
        });
    });
    while (started < W) std::this_thread::yield();
    f();                                             // a 2-thread call: its worker job waits behind the held ones
    release = true;
    hold.join();
}

// threads T6: RecordFidelity over one recording run's rows, threads 2 / 7 == 1 bit for bit.
static void check_fidelity(const char* name, const std::string& text, long shots, bool want_class = false) {
    const int lever = want_class ? 2 : 0;           // explaw T4(b): the class route forced on every evaluator (Shared lever)
    using TA = TwirlSamplerTestAccess;
    Bundle b = compile(text, 0);
    if (!b.ok || b.out_wires.empty()) { std::fprintf(stderr, "%s: %s\n", name, b.ok ? "no output" : b.error.c_str()); CHECK(false); return; }
    auto s = TwirlRecordSampler::make(*b.bare, b.deferred, b.reads, b.detectors, b.observables, 1, b.opt);
    CHECK(s->setup_error() == 0);
    if (s->setup_error()) return;
    const int n = b.bare->n(), PNW = (n + 63) / 64, YW = s->record_value_words(), UZW = s->record_operator_count();
    std::vector<int32_t> handle((size_t)shots, -1), nuz((size_t)shots, 0), branch((size_t)shots, 0);
    std::vector<uint8_t> route((size_t)shots, 0), uz((size_t)shots * UZW, 0), ff((size_t)shots, 0);
    std::vector<uint64_t> prec((size_t)shots * 2 * PNW, 0), y((size_t)shots * YW, 0);
    TA::FamilyRecordSink rs;
    rs.cap = shots; rs.PNW = PNW; rs.YW = YW; rs.UZW = UZW;
    rs.handle = handle.data(); rs.route = route.data(); rs.prec = prec.data(); rs.y = y.data();
    rs.uz = uz.data(); rs.nuz = nuz.data(); rs.ff = ff.data(); rs.branch = branch.data();
    TA::set_family_record_sink(*s, &rs);
    const bool ok = s->run(shots, 4);
    TA::set_family_record_sink(*s, nullptr);
    CHECK(ok && rs.row == shots);
    if (!ok) return;
    std::vector<double> ref((size_t)shots), got((size_t)shots);
    {
        RecordFidelity rf(*b.bare, TA::family_record_ops(*s), b.out_wires, {}, 0);
        rf.set_class_route_for_test(lever);
        TA::family_record_fidelity(*s, rf, rs, shots, ref.data());
        if (want_class) {                            // explaw T4(b): the class route is what the lanes run below
            std::printf("  %s [fidelity]: class_contents %ld class_rows %ld max_ksec %ld\n", name, rf.stats().class_contents,
                        rf.stats().class_rows, rf.stats().max_ksec);
            CHECK(rf.stats().class_contents >= 1 && rf.stats().classes_nonzero == 3);
        }
    }
    for (int T : {2, 7}) {
        if (T == 7) {                                // first a 2-thread call whose worker job is withdrawn
            RecordFidelity rf(*b.bare, TA::family_record_ops(*s), b.out_wires, {}, 0);
        rf.set_class_route_for_test(lever);
            std::vector<double> w((size_t)shots, -1.0);
            with_busy_pool([&] { TA::family_record_fidelity(*s, rf, rs, shots, w.data(), nullptr, nullptr, 2); });
            CHECK(std::memcmp(ref.data(), w.data(), ref.size() * sizeof(double)) == 0);
        }
        RecordFidelity rf(*b.bare, TA::family_record_ops(*s), b.out_wires, {}, 0);
        rf.set_class_route_for_test(lever);
        TA::family_record_fidelity(*s, rf, rs, shots, got.data(), nullptr, nullptr, T);
        TA::family_record_fidelity(*s, rf, rs, shots, got.data(), nullptr, nullptr, T);   // warm: the merged memo
        const bool same = std::memcmp(ref.data(), got.data(), ref.size() * sizeof(double)) == 0;
        if (!same) std::fprintf(stderr, "FAIL %s fidelity threads %d differs\n", name, T);
        CHECK(same);
        {   // fid-routine B2: a whole-cache flush whose frees run on T pool jobs, then the same values from a rebuild
            const long plans = rf.stats().plans;
            rf.trim_cache(rf.bytes_held() / 2, -1, T);   // a partial pass first (its evicted structures freed on T jobs)
            const long part_ev = rf.stats().boundary_evictions;
            std::vector<double> part((size_t)shots, -1.0);
            TA::family_record_fidelity(*s, rf, rs, shots, part.data(), nullptr, nullptr, T);
            CHECK(std::memcmp(ref.data(), part.data(), ref.size() * sizeof(double)) == 0);
            CHECK(rf.trim_cache(1, -1, T) || plans == 0);
            std::vector<double> again((size_t)shots, -1.0);
            TA::family_record_fidelity(*s, rf, rs, shots, again.data(), nullptr, nullptr, T);
            const bool same2 = std::memcmp(ref.data(), again.data(), ref.size() * sizeof(double)) == 0;
            if (!same2) std::fprintf(stderr, "FAIL %s fidelity threads %d after a flush differs\n", name, T);
            CHECK(same2);
            if (T == 7) std::printf("  %s [fidelity]: %ld contents before the flush (%ld evicted by the partial pass)\n",
                                    name, plans, part_ev);
            // fix round 1: a bad_alloc between the waves (the call fails) / at the memo reserve (it does not); then a
            // flush (slot ids recycled) and a call equal to the serial reference — no lane overlay survives a throw
            for (int point : {1, 2}) {
                std::vector<double> f((size_t)shots, -1.0);
                rf.set_fault_for_test(point);
                bool threw = false;
                try { TA::family_record_fidelity(*s, rf, rs, shots, f.data(), nullptr, nullptr, T); }
                catch (const std::bad_alloc&) { threw = true; }
                rf.set_fault_for_test(0);
                if (rf.threads_effective() >= 2) {
                    CHECK(threw == (point == 1));
                    if (point == 2) CHECK(std::memcmp(ref.data(), f.data(), ref.size() * sizeof(double)) == 0);
                }
                rf.trim_cache(1, -1, T);
                std::vector<double> g((size_t)shots, -1.0);
                TA::family_record_fidelity(*s, rf, rs, shots, g.data(), nullptr, nullptr, T);
                const bool ok = std::memcmp(ref.data(), g.data(), ref.size() * sizeof(double)) == 0;
                if (!ok) std::fprintf(stderr, "FAIL %s fidelity threads %d after fault point %d differs\n", name, T, point);
                CHECK(ok);
            }
            // fix round 2 (re-review R1): a throw INSIDE the merge (point 3: lanes 1.. keep their computed overlays;
            // point 4: at a merged entry's note) on a flushed evaluator, then a flush and the rows in reversed order:
            // a lane overlay or untracked memo entry that survived would be read under a recycled id
            std::vector<int32_t> h2(handle.rbegin(), handle.rend()), nz2(nuz.rbegin(), nuz.rend()), br2(branch.rbegin(), branch.rend());
            std::vector<uint8_t> ro2(route.rbegin(), route.rend()), ff2(ff.rbegin(), ff.rend()), uz2(uz.size());
            std::vector<uint64_t> pr2(prec.size()), y2(y.size());
            for (long i = 0; i < shots; ++i) {
                const long j = shots - 1 - i;
                std::memcpy(&pr2[(size_t)i * 2 * PNW], &prec[(size_t)j * 2 * PNW], 2 * (size_t)PNW * 8);
                if (YW) std::memcpy(&y2[(size_t)i * YW], &y[(size_t)j * YW], (size_t)YW * 8);
                if (UZW) std::memcpy(&uz2[(size_t)i * UZW], &uz[(size_t)j * UZW], (size_t)UZW);
            }
            TA::FamilyRecordSink r2 = rs;
            r2.handle = h2.data(); r2.route = ro2.data(); r2.prec = pr2.data(); r2.y = y2.data(); r2.uz = uz2.data();
            r2.nuz = nz2.data(); r2.ff = ff2.data(); r2.branch = br2.data();
            for (int point : {3, 4}) {
                rf.trim_cache(1, -1, T);
                std::vector<double> f((size_t)shots, -1.0);
                rf.set_fault_for_test(point);
                bool threw = false;
                try { TA::family_record_fidelity(*s, rf, rs, shots, f.data(), nullptr, nullptr, T); }
                catch (const std::bad_alloc&) { threw = true; }
                rf.set_fault_for_test(0);
                const auto au = rf.audit_memo_for_test();   // {live contents' memo bytes, held_memo, stray entries}
                CHECK(au[2] == 0 && au[0] == au[1]);
                rf.trim_cache(1, -1, T);
                std::vector<double> g((size_t)shots, -1.0);
                bool ran = true;
                try { TA::family_record_fidelity(*s, rf, r2, shots, g.data(), nullptr, nullptr, T); }
                catch (const std::exception& e) { ran = false; std::fprintf(stderr, "FAIL %s T%d point %d: %s\n", name, T, point, e.what()); }
                long bad = 0;
                for (long i = 0; ran && i < shots; ++i) bad += std::memcmp(&g[(size_t)i], &ref[(size_t)(shots - 1 - i)], 8) != 0;
                if (bad) std::fprintf(stderr, "FAIL %s fidelity T%d after fault point %d: %ld reversed rows differ\n", name, T, point, bad);
                CHECK(ran && bad == 0);
                if (T == 7) std::printf("  %s [fidelity]: fault point %d %s; reversed rows equal\n", name, point,
                                        threw ? "threw" : "did not fire (one lane)");
            }
        }
        if (rf.threads_effective() < 2) std::fprintf(stderr, "  note: %s fidelity threads %d effective %d (%s)\n",
                                                     name, T, rf.threads_effective(), rf.threads_limit().c_str());
    }
    // scaling F1: the new contents are built in parallel and installed in first-row order — a refusal (a malformed
    // content) at row r stands at r: the same message, every row before r evaluated, the evaluator holding the same
    // bytes and contents as the serial row-order intern (r = 0: before the lazy record images; r = mid: after)
    const int bad = TA::family_intern_content(*s, 0, std::string("malformed"));
    for (long r : {0L, shots / 2}) {
        std::vector<int32_t> h2(handle);
        h2[(size_t)r] = bad;
        if (r + 3 < shots) h2[(size_t)r + 3] = bad;
        TA::FamilyRecordSink rs2 = rs;
        rs2.handle = h2.data();
        auto eval = [&](int T, std::vector<double>& o, size_t& held, long& plans) {
            RecordFidelity rf(*b.bare, TA::family_record_ops(*s), b.out_wires, {}, 0);
        rf.set_class_route_for_test(lever);
            o.assign((size_t)shots, -7.0);
            std::string msg;
            try { TA::family_record_fidelity(*s, rf, rs2, shots, o.data(), nullptr, nullptr, T); }
            catch (const std::exception& e) { msg = e.what(); }
            held = rf.bytes_held();
            plans = rf.stats().plans;
            return msg;
        };
        std::vector<double> o1, oT;
        size_t h1 = 0, hT = 0;
        long p1 = 0, pT = 0;
        const std::string m1 = eval(1, o1, h1, p1);
        CHECK(!m1.empty());
        for (int T : {2, 7}) {
            const std::string mT = eval(T, oT, hT, pT);
            const bool same = mT == m1 && std::memcmp(o1.data(), oT.data(), o1.size() * sizeof(double)) == 0 &&
                              hT == h1 && pT == p1;
            if (!same) std::fprintf(stderr, "FAIL %s fidelity refusal at row %ld, threads %d: '%s' vs '%s', held %zu vs %zu, "
                                    "contents %ld vs %ld\n", name, r, T, mT.c_str(), m1.c_str(), hT, h1, pT, p1);
            CHECK(same);
        }
    }
    std::printf("  %s [fidelity]: threads 1 vs 2 7 ok\n", name);
}

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) return std::string();
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// The busy-pool withdrawal on a sampler call (lanes pull blocks: the caller's lane runs them all).
static void check_withdrawn(const char* name, const std::string& text) {
    Bundle b = compile(text, 0);
    if (!b.ok) { CHECK(b.ok); return; }
    Out ref, got;
    CHECK(run_calls(b, {{9000, 2}}, 1, ref, kRecordPs));
    bool ok = false;
    with_busy_pool([&] { ok = run_calls(b, {{9000, 2}}, 2, got, kRecordPs); });
    CHECK(ok && got.d == ref.d && got.o == ref.o && got.rec == ref.rec && got.used == ref.used);
    std::printf("  %s [withdrawn worker]: ok\n", name);
}

// Fix F2: a lane's state-budget refusal (injected by the test seam) is not the call's: the super-block re-runs on
// the home and the call continues serially — the serial bytes, budget_fallbacks reported.
static uint64_t g_fail_lane_at = 0;
static void lane_budget_hook(uint64_t shot, bool lane) {
    if (lane && shot == g_fail_lane_at) throw BareStateCapacityError("test seam: a lane's chi-scaled allocation refused");
}
static void check_lane_budget_refusal(const char* name, const std::string& text) {
    Bundle b = compile(text, 0);
    if (!b.ok) { CHECK(b.ok); return; }
    for (Mode m : {kSample, kRecordPs}) {
        Out ref, got;
        CHECK(run_calls(b, {{9000, 3}, {3000, 0}}, 1, ref, m));
        g_fail_lane_at = 700;                        // inside the second super-block (2 lanes: shots 512..2559)
        g_twirl_shot_end_test_hook = &lane_budget_hook;
        const bool ok = run_calls(b, {{9000, 3}, {3000, 0}}, 2, got, m);
        g_twirl_shot_end_test_hook = nullptr;
        CHECK(ok && got.d == ref.d && got.o == ref.o && got.c == ref.c && got.rec == ref.rec && got.used == ref.used &&
              got.chan == ref.chan && got.early == ref.early && got.late == ref.late);
    }
    std::printf("  %s [lane budget refusal -> serial re-run]: ok\n", name);
}

// Fix F3: a lane fails in block b while a deferred (term-route) shot before b's first shot fails too: the deferred
// shot's failure is the call's (serially it comes first).
static uint64_t g_fail_home_at = 0;
static void order_hook(uint64_t shot, bool lane) {
    if ((!lane && shot == g_fail_home_at) || (lane && shot == g_fail_lane_at))
        throw std::runtime_error("test seam: shot " + std::to_string(shot));
}
static void check_failure_order(const char* name, const std::string& text, bool same_block = false) {
    using TA = TwirlSamplerTestAccess;
    Bundle b = compile(text, 0);
    if (!b.ok) { CHECK(b.ok); return; }
    auto probe = TwirlRecordSampler::make(*b.bare, b.deferred, b.reads, b.detectors, b.observables, 1, b.opt);
    const auto ev = TA::family_draw_events(*probe, 2048, 4);
    const auto ppr = TA::family_alt_is_ppr(*probe);
    long A = -1, Bs = -1;
    for (long i = 0; i < (long)ev.size(); ++i) {
        bool p = false;
        for (int32_t e : ev[(size_t)i]) p = p || ppr[(size_t)e];
        // a cold sampler on 2 lanes: super-block 0 = shots [0, 512) in blocks of 64 (twirl_sampler_threads.cpp)
        if (p && A < 0 && i < 64) A = i;                        // a deferred shot in block 0
        if (!same_block && !p && A >= 0 && i >= 128 && i < 512 && Bs < 0) Bs = i;   // a lane shot in a later block
        // fix R2: a lane shot after the deferred one in the SAME block (the failing lane deferred it first)
        if (same_block && !p && A >= 0 && i > A && i < 64 && Bs < 0) Bs = i;
    }
    CHECK(A >= 0 && Bs >= 0);
    if (A < 0 || Bs < 0) return;
    g_fail_home_at = (uint64_t)A;
    g_fail_lane_at = (uint64_t)Bs;
    g_twirl_shot_end_test_hook = &order_hook;
    std::string msg;
    auto s = TwirlRecordSampler::make(*b.bare, b.deferred, b.reads, b.detectors, b.observables, 4, b.opt);
    try { s->run(9000, 2); } catch (const std::exception& e) { msg = e.what(); }
    g_twirl_shot_end_test_hook = nullptr;
    long born = 0;                                   // fix R3: the call's one BORN_NODE event at most
    for (const auto& e : s->take_diagnostics()) born += e.kind == TwirlRecordSampler::DiagEvent::BORN_NODE;
    CHECK(born <= 1);
    const bool ok = msg == "test seam: shot " + std::to_string(A);
    if (!ok) std::fprintf(stderr, "FAIL %s failure order: got '%s', want shot %ld (lane shot %ld)\n", name, msg.c_str(), A, Bs);
    CHECK(ok);
    std::printf("  %s [failure order: deferred shot %ld before lane shot %ld%s]: ok\n", name, A, Bs,
                same_block ? ", same block" : "");
}

// Fix-2 F1 (coordinator ruling): a FAILED call commits nothing — the noise stream and the cumulative shot index go back
// to their values at the call's entry — so the next seed=None call gives the same bytes at every thread count. Failures
// injected: a lane's shot and a noise-generation task (threaded), and an allocation failure inside the engine (any T;
// at T = 1 mid-chunk: run_w draws 4096-shot chunks ahead of its shots).
static uint64_t g_fail_shot = UINT64_MAX;
static void fail_shot_hook(uint64_t shot, bool) {
    if (shot == g_fail_shot) throw std::runtime_error("test seam: shot " + std::to_string(shot));
}
static std::atomic<int> g_gen_calls{0};          // (the tasks run on several threads)
static int g_gen_fail_at = -1;
static void fail_gen_hook(int) {
    if (g_gen_calls.fetch_add(1) == g_gen_fail_at) throw std::runtime_error("test seam: generation task");
}
static std::atomic<long> g_new_countdown{-1};    // >= 0: the allocation that many allocations ahead fails (once)
static std::atomic<long>& new_countdown_() { return g_new_countdown; }
static long g_rows_written = 0;                  // the sink's rows of the last run_dets call (a failed call's too)
static bool run_dets(TwirlRecordSampler& s, const Bundle& b, long shots, int threads, std::vector<uint8_t>& d) {
    const int DBB = (b.ndet + 7) / 8, OBB = (b.nobs + 7) / 8, DECB = (b.ndec + 7) / 8;
    d.assign((size_t)shots * DBB, 0);
    std::vector<uint8_t> o((size_t)shots * OBB, 0), dc((size_t)shots * DECB, 0);
    FamilySink fs;
    fs.dets = DBB ? d.data() : nullptr; fs.DBB = DBB;
    fs.obs = OBB ? o.data() : nullptr; fs.OBB = OBB;
    fs.dec = DECB ? dc.data() : nullptr; fs.DECB = DECB;
    fs.ndec = b.ndec;
    fs.born_oi = s.born_obs_index();
    fs.born_ref = s.born_obs_ref();
    s.set_family_sink(&fs);
    bool ok = false;
    t_alloc_seam = true;
    try { ok = s.run(shots, threads); } catch (const std::exception&) { ok = false; }
    t_alloc_seam = false;
    g_new_countdown.store(-1);
    g_rows_written = fs.row;
    s.set_family_sink(nullptr);
    d.insert(d.end(), o.begin(), o.end());
    return ok;
}
static void check_failure_commits_nothing(const char* name, const std::string& text) {
    using TA = TwirlSamplerTestAccess;
    Bundle b = compile(text, 0);
    if (!b.ok) { CHECK(b.ok); return; }
    std::vector<uint8_t> warm, ref;                  // the reference: 5000 shots, then 9000 (no failure), serial
    {
        auto r = TwirlRecordSampler::make(*b.bare, b.deferred, b.reads, b.detectors, b.observables, 3, b.opt);
        CHECK(run_dets(*r, b, 5000, 1, warm));
        CHECK(run_dets(*r, b, 9000, 1, ref));
    }
    struct Case { const char* what; int T; int kind; };   // kind 0 lane shot, 1 generation task, 2 allocation
    for (const Case c : {Case{"allocation", 1, 2}, Case{"allocation", 2, 2}, Case{"allocation", 7, 2},
                         Case{"lane shot", 2, 0}, Case{"lane shot", 7, 0}, Case{"generation task", 7, 1}}) {
        // (an allocation failure: the countdown counts the calling thread's allocations only; at T > 1 that thread's
        // share of the work depends on the block schedule, so a countdown the call does not reach is retried smaller —
        // the case then fails deterministically, never in a worker)
        std::unique_ptr<TwirlRecordSampler> s;
        std::vector<uint8_t> w, d, cont;
        long used0 = 0;
        bool ok = true;
        for (long cd : {20000L, 6000L, 2000L, 600L, 200L}) {
            s = TwirlRecordSampler::make(*b.bare, b.deferred, b.reads, b.detectors, b.observables, 3, b.opt);
            CHECK(run_dets(*s, b, 5000, c.T, w));
            used0 = s->used();
            if (c.kind == 0) { g_fail_shot = 5000 + 7777; g_twirl_shot_end_test_hook = &fail_shot_hook; }
            if (c.kind == 1) { g_gen_calls.store(0); g_gen_fail_at = 20; g_twirl_gen_task_test_hook = &fail_gen_hook; }
            if (c.kind == 2) g_new_countdown.store(cd);
            ok = run_dets(*s, b, 20000, c.T, d);
            g_twirl_shot_end_test_hook = nullptr;
            g_twirl_gen_task_test_hook = nullptr;
            if (!ok || c.kind != 2) break;
        }
        const long ran = g_rows_written;             // shots run before the failure (their rows are discarded)
        CHECK(!ok);
        CHECK(TA::family_shot_index(*s) == 5000);    // the call's entry index
        CHECK(s->used() == used0);                   // fix-3: the failed call's shots are not counted
        if (c.kind == 2 && c.T == 1) CHECK(ran > 0 && ran % 4096 != 0);   // a serial failure mid-chunk
        CHECK(run_dets(*s, b, 9000, c.T, cont));     // seed=None continuation at the call's thread count
        const bool same = w == warm && cont == ref;
        if (!same) std::fprintf(stderr, "FAIL %s commit-nothing %s T=%d (ran %ld)\n", name, c.what, c.T, ran);
        CHECK(same);
        if (same) std::printf("  %s [failed call commits nothing: %s, T=%d, %ld shots run]: ok\n", name, c.what, c.T, ran);
    }
}

// fix round 2: a bad_alloc during run()'s task pushes leaves no queued task naming the dead call (later runs: every
// job of theirs runs exactly once, and no stale task runs a dead call's job)
static void check_pool_push_failure() {
    for (int at : {1, 3, 7}) {
        std::atomic<int> ran{0};
        thread_pool_fail_push_for_test(at);
        bool threw = false;
        try { ThreadPool::instance().run(8, [&](int) { ran.fetch_add(1); }); } catch (const std::bad_alloc&) { threw = true; }
        thread_pool_fail_push_for_test(-1);
        CHECK(threw && ran.load() == 0);
        CHECK(ThreadPool::instance().queued_for_test() == 0);   // no task names the dead call
        for (int rep = 0; rep < 20; ++rep) {
            std::unique_ptr<std::atomic<int>[]> hit(new std::atomic<int>[8]());
            ThreadPool::instance().run(8, [&](int i) { hit[i].fetch_add(1); });
            for (int i = 0; i < 8; ++i) CHECK(hit[i].load() <= 1);
            CHECK(hit[0].load() == 1);
        }
    }
}

int main(int argc, char** argv) {
    const long scale = argc > 1 ? std::atol(argv[1]) : 1;   // shot multiplier (TSan runs use the default)
    check_pool_push_failure();
    const std::vector<int> threads = {2, 7};
    test_trie_merge();
    check_withdrawn("rep", kRep);
    check_lane_budget_refusal("rep", kRep);
    {
        const std::string ch = slurp(std::string(XTIM_TESTS_DIR) + "/data/ch_cultivation.stim");
        CHECK(!ch.empty());
        if (!ch.empty()) {
            check_failure_order("ch_cultivation", ch);
            check_failure_order("ch_cultivation", ch, true);
            check_lane_budget_refusal("ch_cultivation", ch);
        }
    }
    check_circuit("rep", kRep, 0, {{4097 * scale, 3}, {5000 * scale, 0}}, threads);
    check_circuit("rep_window", kRep, 300, {{9000 * scale, 3}}, threads);
    const std::string d5 = slurp(std::string(XTIM_TESTS_DIR) + "/data/cultivation_d5.stim");
    CHECK(!d5.empty());
    if (!d5.empty()) {
        check_failure_commits_nothing("d5", d5);
        check_circuit("d5", d5, 0, {{6000 * scale, 5}, {3000 * scale, 0}}, threads);
        check_circuit("d5_window", d5, 1000, {{8000 * scale, 5}}, threads);
    }
    const char* pf = std::getenv("XTIM_PRIVATE_FIXTURES");
    const std::string fa = slurp((pf ? std::string(pf) : std::string(XTIM_TESTS_DIR) + "/private") + "/ccz_factory_a.stim");
    if (!fa.empty()) {
        check_circuit("factory_a", fa, 0, {{3000 * scale, 2}, {2000 * scale, 0}}, threads);
        check_circuit("factory_a", fa, 0, {{3000 * scale, 2}}, threads, kRecordPs);
    } else {
        std::printf("  factory_a: private fixture absent (skipped)\n");
    }
    for (const char* nm : {"ch_cultivation.stim", "ja_seed23.stim", "cultivation_d3_rate.stim"}) {   // CH / PPR: deferral
        const std::string t = slurp(std::string(XTIM_TESTS_DIR) + "/data/" + nm);
        CHECK(!t.empty());
        if (t.empty()) continue;
        for (Mode m : {kSample, kPostselect, kRecord, kRecordPs})
            check_circuit(nm, t, 0, {{5000 * scale, 4}, {3000 * scale, 0}}, threads, m);
    }
    for (const char* nm : {"magna_color5_fid.stim", "adaptq_qrm_magic_producer.stim"}) {
        const std::string t = slurp(std::string(XTIM_TESTS_DIR) + "/data/" + nm);
        CHECK(!t.empty());
        if (!t.empty()) check_fidelity(nm, t, 4000 * scale);
    }
    {   // explaw T4(b): ksec > 0 contents on the class route (tests/test_explaw.py t4_encoded(3), t4_bell_pair(2); the
        // route forced by the lever, since the cost rule keeps these small ones on the sector sum): several contents
        // built on lanes in parallel, the output's classes computed once under the shared lock
        std::string enc = "RX 0\nT 0\nR 1 2\nCX 0 1\nCX 0 2\n", bell = enc;
        for (int i = 0; i < 3; ++i) {
            const int c = 3 + 2 * i, r = 4 + 2 * i;
            enc += "R " + std::to_string(c) + "\nRX " + std::to_string(r) + "\nX_ERROR(0.5) " + std::to_string(c) +
                   "\nCCZ " + std::to_string(i) + " " + std::to_string(c) + " " + std::to_string(r) + "\n";
        }
        for (int i = 0; i < 3; ++i) enc += "M " + std::to_string(3 + 2 * i) + "\nDECISION(" + std::to_string(i) + ") rec[-1]\n";
        enc += "OUTPUT_QUBITS out 0 1 2\n";
        for (int i = 0; i < 2; ++i) {
            const int c = 3 + 3 * i, r = 4 + 3 * i, r2 = 5 + 3 * i;
            bell += "R " + std::to_string(c) + "\nRX " + std::to_string(r) + "\nR " + std::to_string(r2) + "\nCX " +
                    std::to_string(r) + " " + std::to_string(r2) + "\nX_ERROR(0.5) " + std::to_string(c) + "\nCCZ " +
                    std::to_string(i % 3) + " " + std::to_string(c) + " " + std::to_string(r) + "\nCCZ " +
                    std::to_string((i + 1) % 3) + " " + std::to_string(c) + " " + std::to_string(r2) + "\n";
        }
        for (int i = 0; i < 2; ++i) bell += "M " + std::to_string(3 + 3 * i) + "\nDECISION(" + std::to_string(i) + ") rec[-1]\n";
        bell += "OUTPUT_QUBITS out 0 1 2\n";
        check_fidelity("t4_encoded3", enc, 4000 * scale, true);
        check_fidelity("t4_bell2", bell, 4000 * scale, true);
    }
    if (!d5.empty())
        for (Mode m : {kPostselect, kRecord, kRecordPs})
            check_circuit("d5", d5, 0, {{6000 * scale, 5}, {2000 * scale, 0}}, threads, m);
    REPORT();
}
