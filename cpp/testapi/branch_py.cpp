// J-a-b (N7, switched OFF in production): the TEST API of the branch scaffolding (qeccore/branch_scaffold.hpp) on a FAM
// sampler of xtim._xtim_testapi.compile_family_sampler (its TwirlRecordSampler via FamilySampler._engine()). Registered
// from register_recb (recb_py.cpp).
//   branch_plan(fam, key=None, S=None) — one plan's scaffolding: a DIAGONAL plan key (plan_structure's format; kind 0,
//       term form diag_plan_axes) or a reduced term set S of the sampler's term view (kind 2): content key, digest, the
//       axes and S, E_S (lazy: requested here), the certified rows moved by Φ(S), the term images B̂_j, the record
//       table's pivots / coin rows, the pivot pullbacks' dual generators g_p and W_p;
//   branch_workload(fam, keys, branches, force_collision) — payload builds (scaffold + plan table insert) of every
//       diagonal key, the branches core on request (E_S + B̂_j), resolve by FULL key; the E_S computation counter;
//   alpha_payload_check(fam, keys, ulps) — G-ALPHA: every payload field of every key with the bare state and with a
//       copy whose first amplitude moved by `ulps` ulp; the Born-trie node values compared and counted apart;
//   plan_payload_fields() — the payload accessor list (every field alpha_payload_check reports or excludes);
//   emin_categories(fam, S_list, reduce) — F3-2: per term set (one first-order fault), the surviving-component
//       structure after the records (cosets of K = ker(M·A) over subsets of S): Pauli frame / LOGICAL / EXTENDED;
//   bare_frames(fam) — the sampler's bare state in the branch_frames() export format (xtim.frames.dense_state).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "qeccore/record_space_b.hpp"
#include "qeccore/amplitudes.hpp"
#include "qeccore/branch_scaffold.hpp"
#include "qeccore/normalize.hpp"
#include "qeccore/record_general.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/sampler.hpp"
#include "qeccore/stim_parse.hpp"
#include "qeccore/term_residual.hpp"
#include "qeccore/twirl_kernel.hpp"
#include "qeccore/twirl_sampler.hpp"
#include "twirl_test_access.hpp"

namespace py = pybind11;
using namespace qeccore;
using TA_ = qeccore::TwirlSamplerTestAccess;   // J-b1 P3 (m6)

namespace {

TwirlRecordSampler& engine_of(const py::object& fam) {
    py::capsule c = fam.attr("_engine")();
    if (std::strcmp(c.name(), "xtim.TwirlRecordSampler") != 0)
        throw std::invalid_argument("branch: expected a FamilySampler (compile_family_sampler)");
    return *static_cast<TwirlRecordSampler*>(c.get_pointer());
}

std::vector<RecColumn> columns_of(const TwirlRecordSampler& s) {
    const std::vector<Pauli> ops = s.record_operators();
    const std::vector<int> ord = s.record_column_order();
    const std::vector<std::vector<uint64_t>> masks = s.record_channel_masks();
    const std::vector<uint8_t> refs = s.record_channel_refs();
    const int nch = s.num_channels();
    std::vector<RecColumn> cols(ops.size());
    for (size_t c = 0; c < ops.size(); ++c) {
        cols[c].M = ops[c];
        cols[c].order = ord[c];
        cols[c].channel = (int)c < nch;
        if ((int)c < nch) { cols[c].mask = masks[c]; cols[c].ref = refs[c]; }
    }
    return cols;
}

py::tuple pauli_t(const Pauli& P) {
    py::array_t<uint8_t> x((py::ssize_t)P.n), z((py::ssize_t)P.n);
    for (int q = 0; q < P.n; ++q) { x.mutable_data()[q] = P.xbit(q); z.mutable_data()[q] = P.zbit(q); }
    return py::make_tuple(x, z, (int)P.phase);
}
std::vector<int> bits_list(const std::vector<uint64_t>& v, int nbits) {
    std::vector<int> o;
    for (int i = 0; i < nbits; ++i) if ((size_t)(i >> 6) < v.size() && ((v[(size_t)(i >> 6)] >> (i & 63)) & 1)) o.push_back(i);
    return o;
}
std::vector<uint64_t> words_of(const std::vector<int>& S, int T) {
    std::vector<uint64_t> w((size_t)((T + 63) / 64), 0);
    for (int j : S) {
        if (j < 0 || j >= T) throw std::invalid_argument("branch: term id out of range");
        w[(size_t)(j >> 6)] ^= 1ULL << (j & 63);
    }
    return w;
}
std::string words_str(const std::vector<uint64_t>& w) {
    return std::string(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(uint64_t));
}
DiagNormalForm prefix_free(const DiagNormalForm& nf, int n) {
    DiagNormalForm nf0 = nf;
    nf0.prefix = Pauli(n);
    return nf0;
}
// The stripped content key of a diagonal plan: a bytes ‖ 0xFF ‖ cz u16 pairs (plan_structure's key minus the prefix).
std::string diag_content_key(const DiagNormalForm& nf, int n) {
    std::string k;
    for (int q = 0; q < n; ++q) k.push_back((char)(q < (int)nf.a.size() ? (nf.a[(size_t)q] & 1) : 0));
    k.push_back((char)0xFF);
    for (const auto& e : nf.cz)
        for (int v : {e.first, e.second}) { k.push_back((char)(v & 255)); k.push_back((char)(v >> 8)); }
    return k;
}

// One plan: its term view, record table and pivot pullbacks.
struct PlanInputs {
    PlanTermView v;
    RecTable T;
    std::vector<Pauli> Qpiv;
    DiagNormalForm nf;           // kind 0
};
PlanInputs plan_inputs(TwirlRecordSampler& s, const CertifiedGroupPlanes& G, const std::vector<RecColumn>& cols,
                       const py::object& key, const py::object& S) {
    PlanInputs in;
    const int n = s.deferred_wires();
    if (!S.is_none()) {                                   // kind 2: a reduced term set of the sampler's term view
        TermResidual& tv = TA_::family_terms(s);
        const std::vector<uint64_t> Sw = words_of(py::cast<std::vector<int>>(S), tv.num_terms());
        in.v.kind = 2;
        in.v.key = words_str(tv.key_of(Sw));
        for (int j = 0; j < tv.num_terms(); ++j) in.v.axes.push_back(tv.axis(j));
        in.v.S = Sw;
        TA_::family_general_table(s, Sw, in.T);
        for (int c : in.T.pivot_rec) in.Qpiv.push_back(term_pullback(tv, Sw, cols[(size_t)c].M));
        return in;
    }
    const std::string k = py::cast<std::string>(py::bytes(key));
    const TwirlRecordSampler::PlanStructure ps = s.plan_structure(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size());
    if (ps.law.fallback || !ps.rec.built) throw std::invalid_argument("branch_plan: a fallback plan");
    in.nf = prefix_free(ps.nf, n);
    in.v.kind = 0;
    in.v.key = diag_content_key(in.nf, n);
    in.v.axes = diag_plan_axes(in.nf, n);
    in.v.S.assign((in.v.axes.size() + 63) / 64, 0);
    for (size_t j = 0; j < in.v.axes.size(); ++j) in.v.S[j >> 6] |= 1ULL << (j & 63);
    in.T = ps.rec;
    for (int c : in.T.pivot_rec) in.Qpiv.push_back(diag_pullback(in.nf, cols[(size_t)c].M));
    (void)G;
    return in;
}

py::dict scaffold_dict(PlanScaffold& sc, const PlanInputs& in, const FramedSuperposition& bare, int ng) {
    py::dict d;
    d["kind"] = in.v.kind;
    d["content_key"] = py::bytes(in.v.key);
    d["identity_key"] = py::bytes(sc.identity_key());
    d["digest"] = py::make_tuple(sc.digest().hi, sc.digest().lo);
    py::list ax;
    for (const Pauli& B : in.v.axes) ax.append(pauli_t(B));
    d["axes"] = ax;
    d["S"] = bits_list(in.v.S, (int)in.v.axes.size());
    d["es"] = bits_list(sc.ensure_es_(), ng);
    d["moved"] = bits_list(sc.moved_rows_(), ng);
    py::list im;
    for (const TermImage& t : sc.term_images_(bare))
        im.append(py::make_tuple(t.term, t.x, t.z, (int)t.phase, (int)t.par));
    d["images"] = im;
    d["pivot_rec"] = in.T.pivot_rec;
    py::array_t<uint8_t> A({(py::ssize_t)in.T.m, (py::ssize_t)in.T.rho});
    for (int c = 0; c < in.T.m; ++c)
        for (int k = 0; k < in.T.rho; ++k)
            A.mutable_data()[(size_t)c * in.T.rho + k] = (uint8_t)((in.T.coin_row[(size_t)k][(size_t)(c >> 6)] >> (c & 63)) & 1);
    d["coin_rows"] = A;
    py::list W, Q;
    for (const Pauli& w : sc.pivot_frames_(in.Qpiv)) W.append(pauli_t(w));
    {   // g_p (W_p without its dressing): C0† W_p C0 restricted... recovered as W_p · Π_{j: {g_p, B_j} = 0} (iB_j)
        py::list gl;
        for (const Pauli& w0 : sc.pivot_frames_(in.Qpiv)) {
            Pauli g = w0;                       // W_p = g_p · Π(−iB_j) over the B_j anticommuting with g_p ⇔ with W_p
            for (int j = (int)in.v.axes.size() - 1; j >= 0; --j) {
                if (!((in.v.S[(size_t)(j >> 6)] >> (j & 63)) & 1) || !Pauli::anticommute_bit(w0, in.v.axes[(size_t)j]))
                    continue;
                Pauli f = in.v.axes[(size_t)j];
                f.phase = (uint8_t)((f.phase + 1) & 3);   // (−iB)^{-1} = iB
                g = Pauli::multiply(g, f);
            }
            gl.append(pauli_t(g));
        }
        d["g"] = gl;
    }
    for (const Pauli& q : in.Qpiv) Q.append(pauli_t(q));
    d["W"] = W;
    d["Qpiv"] = Q;
    if (in.v.kind == 0) {
        std::vector<int> a;
        for (size_t q = 0; q < in.nf.a.size(); ++q) if (in.nf.a[q] & 1) a.push_back((int)q);
        d["a"] = a;
        d["cz"] = in.nf.cz;
    }
    return d;
}

// ── G-ALPHA: the payload of one plan, field by field, as bytes ──
void put_bits(std::string& s, const std::vector<uint64_t>& w) { s.append(words_str(w)); s.push_back('|'); }
void put_fd(std::string& s, const FreeDir& d) {
    s.append(reinterpret_cast<const char*>(d.a.data()), d.a.size());
    s.append(reinterpret_cast<const char*>(d.b.data()), d.b.size());
    s.push_back((char)d.phase); s.push_back((char)d.par); s.push_back('|');
}
void put_pauli(std::string& s, const Pauli& P) { put_bits(s, P.x); put_bits(s, P.z); s.push_back((char)P.phase); }

// The PAYLOAD fields of a plan (plan r2.3): compared byte for byte by alpha_payload_check. The trie node values are
// the one excluded group (reported separately).
const std::vector<std::string>& compared_fields() {
    static const std::vector<std::string> f = {
        "header.content_key", "header.digest", "header.kind",
        "rec.m", "rec.rho", "rec.kappa", "rec.CWX", "rec.base", "rec.coin_row", "rec.born_row", "rec.pivot_rec",
        "rec.born_rec", "rec.lam", "rec.folds", "rec.base_ch", "rec.base_ex", "rec.coin_ch", "rec.coin_ex",
        "rec.born_ch", "rec.born_ex", "law.det_signs", "law.kernel_logicals",
        "branch.E_S", "branch.W_p", "branch.term_images", "branch.emin_table_index"};
    return f;
}
const std::vector<std::string>& excluded_fields() {
    static const std::vector<std::string> f = {"rec_trie.p0", "rec_trie.children", "reader_memo.values",
                                               "signs.value_cache", "ppr.fold_p1"};
    return f;
}

// kind-2 plans (Sw non-empty): (G)'s table, the term view; no σ law (the PPR family's fields are excluded).
std::map<std::string, std::string> payload_of_kind2(TwirlRecordSampler& s, const CertifiedGroupPlanes& G,
                                                    const FramedSuperposition& bare, const std::vector<RecColumn>& cols,
                                                    const std::vector<uint64_t>& Sw, PlanTableSet& tables);

std::map<std::string, std::string> payload_of(TwirlRecordSampler& s, const CertifiedGroupPlanes& G,
                                              const FramedSuperposition& bare, const std::vector<RecColumn>& cols,
                                              const std::string& key, PlanTableSet& tables) {
    std::map<std::string, std::string> P;
    for (const std::string& f : compared_fields()) P[f] = "";     // every listed field exists (possibly empty)
    const int n = s.deferred_wires();
    const TwirlRecordSampler::PlanStructure ps0 = s.plan_structure(reinterpret_cast<const uint8_t*>(key.data()), (int)key.size());
    const DiagNormalForm nf = prefix_free(ps0.nf, n);
    // the law and the record table, built as the plan payload builds them (record-space B; J-b2 J7b), with THIS
    // bare state
    const ShotLaw law = build_shot_law(G, nf);
    RecTable T;
    if (!law.fallback) build_record_table_recB(G, bare, nf, cols, T);
    auto& f = P;
    PlanTermView tvw;
    tvw.kind = 0;
    tvw.key = diag_content_key(nf, n);
    tvw.axes = diag_plan_axes(nf, n);
    tvw.S.assign((tvw.axes.size() + 63) / 64, 0);
    for (size_t j = 0; j < tvw.axes.size(); ++j) tvw.S[j >> 6] |= 1ULL << (j & 63);
    PlanScaffold sc(tvw, G);
    f["header.content_key"] = tvw.key;
    f["header.digest"] = words_str({sc.digest().hi, sc.digest().lo});
    f["header.kind"] = "0";
    f["rec.m"] = std::to_string(T.m); f["rec.rho"] = std::to_string(T.rho); f["rec.kappa"] = std::to_string(T.kappa);
    f["rec.CWX"] = std::to_string(T.CWX); f["rec.folds"] = std::to_string(T.folds);
    put_bits(f["rec.base"], T.base);
    for (const auto& r : T.coin_row) put_bits(f["rec.coin_row"], r);
    for (const auto& r : T.born_row) put_bits(f["rec.born_row"], r);
    for (int c : T.pivot_rec) f["rec.pivot_rec"] += std::to_string(c) + ",";
    for (int c : T.born_rec) f["rec.born_rec"] += std::to_string(c) + ",";
    for (const FreeDir& d : T.lam) put_fd(f["rec.lam"], d);
    // the emission block's channel / extras split of every row (build_rec_cold_): bits [0, nchan) and [nchan, m)
    const int nch = s.num_channels();
    auto split = [&](const std::vector<uint64_t>& row, std::string& ch, std::string& ex) {
        for (int c = 0; c < T.m; ++c) {
            const char b = (char)('0' + ((row[(size_t)(c >> 6)] >> (c & 63)) & 1));
            (c < nch ? ch : ex).push_back(b);
        }
        ch.push_back('|'); ex.push_back('|');
    };
    if (T.built) {
        split(T.base, f["rec.base_ch"], f["rec.base_ex"]);
        for (const auto& r : T.coin_row) split(r, f["rec.coin_ch"], f["rec.coin_ex"]);
        for (const auto& r : T.born_row) split(r, f["rec.born_ch"], f["rec.born_ex"]);
    } else {
        f["rec.base_ch"] = f["rec.base_ex"] = f["rec.coin_ch"] = f["rec.coin_ex"] = f["rec.born_ch"] = f["rec.born_ex"] = "";
    }
    put_bits(f["law.det_signs"], law.det_signs);
    for (const Pauli& L : law.kernel_logicals) put_pauli(f["law.kernel_logicals"], L);
    put_bits(f["branch.E_S"], sc.ensure_es_());
    std::vector<Pauli> Qpiv;
    for (int c : T.pivot_rec) Qpiv.push_back(diag_pullback(nf, cols[(size_t)c].M));
    for (const Pauli& w : sc.pivot_frames_(Qpiv)) put_pauli(f["branch.W_p"], w);
    for (const TermImage& t : sc.term_images_(bare)) {
        std::string& o = f["branch.term_images"];
        o += std::to_string(t.term) + ":";
        o.append(reinterpret_cast<const char*>(t.x.data()), t.x.size());
        o.append(reinterpret_cast<const char*>(t.z.data()), t.z.size());
        o.push_back((char)t.phase); o.push_back((char)t.par); o.push_back('|');
    }
    f["branch.emin_table_index"] = std::to_string(tables.insert({}, sc.identity_key()));
    return P;
}

std::map<std::string, std::string> payload_of_kind2(TwirlRecordSampler& s, const CertifiedGroupPlanes& G,
                                                    const FramedSuperposition& bare, const std::vector<RecColumn>& cols,
                                                    const std::vector<uint64_t>& Sw, PlanTableSet& tables) {
    std::map<std::string, std::string> f;
    for (const std::string& name : compared_fields()) f[name] = "";
    TermResidual& tv = TA_::family_terms(s);
    RecTable T;
    build_record_table_general(G, bare, tv, Sw, cols, T);
    PlanTermView v;
    v.kind = 2;
    v.key = words_str(tv.key_of(Sw));
    for (int j = 0; j < tv.num_terms(); ++j) v.axes.push_back(tv.axis(j));
    v.S = Sw;
    PlanScaffold sc(v, G);
    f["header.content_key"] = v.key;
    f["header.digest"] = words_str({sc.digest().hi, sc.digest().lo});
    f["header.kind"] = "2";
    f["rec.m"] = std::to_string(T.m); f["rec.rho"] = std::to_string(T.rho); f["rec.kappa"] = std::to_string(T.kappa);
    f["rec.CWX"] = std::to_string(T.CWX); f["rec.folds"] = std::to_string(T.folds);
    put_bits(f["rec.base"], T.base);
    for (const auto& r : T.coin_row) put_bits(f["rec.coin_row"], r);
    for (const auto& r : T.born_row) put_bits(f["rec.born_row"], r);
    for (int c : T.pivot_rec) f["rec.pivot_rec"] += std::to_string(c) + ",";
    for (int c : T.born_rec) f["rec.born_rec"] += std::to_string(c) + ",";
    for (const FreeDir& d : T.lam) put_fd(f["rec.lam"], d);
    put_bits(f["branch.E_S"], sc.ensure_es_());
    std::vector<Pauli> Qpiv;
    for (int c : T.pivot_rec) Qpiv.push_back(term_pullback(tv, Sw, cols[(size_t)c].M));
    for (const Pauli& w : sc.pivot_frames_(Qpiv)) put_pauli(f["branch.W_p"], w);
    for (const TermImage& t : sc.term_images_(bare)) {
        std::string& o = f["branch.term_images"];
        o += std::to_string(t.term) + ":";
        o.append(reinterpret_cast<const char*>(t.x.data()), t.x.size());
        o.append(reinterpret_cast<const char*>(t.z.data()), t.z.size());
        o.push_back((char)t.phase); o.push_back((char)t.par); o.push_back('|');
    }
    f["branch.emin_table_index"] = std::to_string(tables.insert({}, sc.identity_key()));
    return f;
}

// ── G-ALPHA twin: the FamilySampler front end (cpp/testapi/family_py.cpp, replicated for the twin only) with the bare
// state's first amplitude moved by `ulps` ulp. Its plan_structure / family_general_table run the PRODUCTION builders
// (build_shot_law + build_rec_; (G)) on the perturbed bare.
struct Twin {
    std::unique_ptr<FramedSuperposition> bare;
    Circuit deferred;
    std::vector<std::pair<int, int>> reads;
    std::vector<std::vector<int>> detectors;
    std::vector<std::pair<int, std::vector<int>>> obs;
    TwirlRecordOptions opt;
    std::unique_ptr<TwirlRecordSampler> s;
};
std::unique_ptr<Twin> make_twin(const std::string& text, int ulps) {
    auto t = std::make_unique<Twin>();
    ParsedStim ps = parse_stim_circuit(text);
    if (!ps.ok()) throw std::invalid_argument("alpha twin: parse failed");
    for (const InputPort& p : ps.circuit.inputs)
        if (p.kind == InputPort::Kind::Qubits) { t->opt.input_pauli_qubits = p.qubits; break; }
    NormalizePolicy pol;
    pol.coherentize = true; pol.coherentize_all = true; pol.defer = true; pol.want_map = true;
    pol.feedback = NormalizePolicy::Feedback::Reject; pol.hoist_feedback_on_class_exit = true;
    pol.zero_noise_transparent = true;
    NormalizeResult nr = normalize(ps.circuit, pol);
    for (const Instr& ins : nr.coherent.stream)
        if (ins.kind == Instr::Kind::Measure) {
            t->opt.rec_flip.push_back(ins.readout_flip_p);
            t->opt.rec_invert.push_back(ins.invert ? 1 : 0);
        }
    {
        size_t j = 0;
        for (const Instr& ins : nr.normalized.stream)
            if (ins.kind == Instr::Kind::Measure) {
                if (j < t->opt.rec_invert.size() && ins.invert) t->opt.rec_invert[j] ^= 1;
                ++j;
            }
    }
    t->deferred = std::move(nr.normalized);
    for (const auto& r : nr.map.terminal_reads) t->reads.push_back({(int)r.second, r.first});
    t->detectors = std::move(ps.detectors);
    t->obs.assign(ps.observables.begin(), ps.observables.end());
    t->opt.decisions = std::move(ps.decisions);
    FramedBareState bs = build_bare_state_framed(t->deferred);
    if (bs.rejected) throw std::invalid_argument("alpha twin: bare state rejected");
    t->bare.reset(new FramedSuperposition(std::move(bs.state)));
    auto* dense = dynamic_cast<DenseAmplitudes*>(t->bare->alpha.get());
    if (!dense || dense->b_.empty()) throw std::invalid_argument("alpha twin: no dense amplitudes");
    double re = dense->b_[0].second.real();
    for (int u = 0; u < ulps; ++u) re = std::nextafter(re, re > 0 ? 2.0 * re + 1.0 : -2.0);
    dense->b_[0].second = std::complex<double>(re, dense->b_[0].second.imag());
    t->opt.p_factor = 1.0;
    t->opt.circuit_channels = true;
    t->opt.selfcheck = 0;
    t->s = TwirlRecordSampler::make(*t->bare, t->deferred, t->reads, t->detectors, t->obs, 1, t->opt);
    if (t->s->setup_error() != 0) throw std::invalid_argument("alpha twin: setup failed");
    TA_::family_arm_terms(*t->s, t->deferred, t->reads);   // J-b1 P4: the twin's term view, from its own inputs
    return t;
}
// Every field of a plan's production law and record table, as bytes (Λ and kernel logicals with their exact splits).
std::string law_table_bytes(const ShotLaw& law, const RecTable& T) {
    std::string o;
    o += std::to_string(law.fallback) + "," + std::to_string(law.r) + "," + std::to_string(law.kappa) + ";";
    put_bits(o, law.det_signs);
    for (const Pauli& L : law.kernel_logicals) put_pauli(o, L);
    o += "T" + std::to_string(T.built) + "," + std::to_string(T.m) + "," + std::to_string(T.rho) + "," +
         std::to_string(T.kappa) + "," + std::to_string(T.CWX) + "," + std::to_string(T.folds) + ";";
    put_bits(o, T.base);
    for (const auto& r : T.coin_row) put_bits(o, r);
    for (const auto& r : T.born_row) put_bits(o, r);
    for (int c : T.pivot_rec) o += std::to_string(c) + ",";
    for (int c : T.born_rec) o += std::to_string(c) + ",";
    for (const FreeDir& d : T.lam) put_fd(o, d);
    o += std::to_string(T.ops_plane) + "," + std::to_string(T.ops_pivot_reduce) + "," + std::to_string(T.ops_greduce) +
         "," + std::to_string(T.ops_echelon) + "," + std::to_string(T.extras);
    return o;
}

}  // namespace

void register_branch(py::module_& m) {
    m.def("alpha_twin_check",
          [](const py::object& fam, const std::string& text,
             const std::vector<py::bytes>& keys, const std::vector<std::vector<int>>& S_list, int ulps) {
              TwirlRecordSampler& s = engine_of(fam);
              std::unique_ptr<Twin> tw;
              {
                  py::gil_scoped_release rel;
                  tw = make_twin(text, ulps);
              }
              TwirlRecordSampler& t = *tw->s;
              long plans = 0, diff = 0, amp_diff = 0;
              {   // the twin's amplitudes do differ (the perturbation is real)
                  const auto& a = s.bare_state().entries();
                  const auto& b = t.bare_state().entries();
                  for (size_t i = 0; i < a.size() && i < b.size(); ++i) amp_diff += a[i].second != b[i].second;
              }
              std::vector<std::string> bad;
              for (const py::bytes& kb : keys) {
                  const std::string k = kb.cast<std::string>();
                  const auto* kp = reinterpret_cast<const uint8_t*>(k.data());
                  const TwirlRecordSampler::PlanStructure a = s.plan_structure(kp, (int)k.size());
                  const TwirlRecordSampler::PlanStructure b = t.plan_structure(kp, (int)k.size());
                  ++plans;
                  if (law_table_bytes(a.law, a.rec) != law_table_bytes(b.law, b.rec)) { ++diff; bad.push_back(k); }
              }
              for (const std::vector<int>& S : S_list) {
                  const std::vector<uint64_t> Sw = words_of(S, TA_::family_terms(s).num_terms());
                  RecTable a, b;
                  TA_::family_general_table(s, Sw, a);
                  TA_::family_general_table(t, Sw, b);
                  ++plans;
                  if (law_table_bytes(ShotLaw(), a) != law_table_bytes(ShotLaw(), b)) ++diff;
              }
              py::dict d;
              d["plans"] = plans;
              d["diffs"] = diff;
              d["amplitudes_moved"] = amp_diff;
              return d;
          },
          py::arg("fam"), py::arg("text"), py::arg("keys"),
          py::arg("S_list") = std::vector<std::vector<int>>(), py::arg("ulps") = 1,
          "J-a-b TEST API (N7, G-ALPHA): the PRODUCTION plan builders (plan_structure: build_shot_law + build_rec_; "
          "family_general_table: (G)) of `fam` vs a twin sampler of `text` whose bare amplitudes moved by `ulps` ulp: "
          "every law / table byte compared per plan.");
    m.def("branch_plan",
          [](const py::object& fam, const py::object& key, const py::object& S) {
              TwirlRecordSampler& s = engine_of(fam);
              const int n = s.deferred_wires();
              const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
              const std::vector<RecColumn> cols = columns_of(s);
              const PlanInputs in = plan_inputs(s, G, cols, key, S);
              long cnt = 0;
              PlanScaffold sc(in.v, G, &cnt);
              py::dict d = scaffold_dict(sc, in, s.bare_state(), (int)G.gens.size());
              d["es_computations"] = cnt;
              return d;
          },
          py::arg("fam"), py::arg("key") = py::none(), py::arg("S") = py::none(),
          "J-a-b TEST API (N7): one plan's branch scaffolding (diagonal plan key, or a reduced term set S).");
    m.def("branch_workload",
          [](const py::object& fam, const std::vector<py::bytes>& keys, bool branches, bool force_collision) {
              TwirlRecordSampler& s = engine_of(fam);
              const int n = s.deferred_wires();
              const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
              long es_count = 0;
              PlanTableSet tables;
              if (force_collision) tables.force_digest_for_test(Digest128{0x5EED, 0xC011});
              std::vector<PlanScaffold> plans;
              std::vector<long> ids;
              long refused = 0;
              for (const py::bytes& kb : keys) {        // the payload builds
                  const std::string k = kb.cast<std::string>();
                  const TwirlRecordSampler::PlanStructure ps =
                      s.plan_structure(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size());
                  PlanTermView v;
                  const DiagNormalForm nf = prefix_free(ps.nf, n);
                  v.kind = 0;
                  v.key = diag_content_key(nf, n);
                  v.axes = diag_plan_axes(nf, n);
                  v.S.assign((v.axes.size() + 63) / 64, 0);
                  for (size_t j = 0; j < v.axes.size(); ++j) v.S[j >> 6] |= 1ULL << (j & 63);
                  plans.emplace_back(v, G, &es_count);
                  try { ids.push_back(tables.insert({}, plans.back().identity_key())); }
                  catch (const DigestCollision&) { ++refused; ids.push_back(-2); }
              }
              const long es_after_build = es_count;
              long resolved = 0, mismatched = 0;
              for (size_t i = 0; i < plans.size(); ++i) {
                  if (branches) { (void)plans[i].ensure_es_(); (void)plans[i].term_images_(s.bare_state()); }
                  try {
                      const long id = tables.resolve({}, force_collision ? Digest128{0x5EED, 0xC011} : plans[i].digest(),
                                                     plans[i].identity_key());
                      if (id == ids[i]) ++resolved; else ++mismatched;
                  } catch (const DigestCollision&) { ++refused; }
              }
              py::dict d;
              d["plans"] = (long)plans.size();
              d["distinct"] = tables.size();
              d["tables"] = tables.num_tables();
              d["es_after_payload_builds"] = es_after_build;
              d["es_computations"] = es_count;
              d["resolved"] = resolved;
              d["mismatched"] = mismatched;
              d["refused"] = refused;
              return d;
          },
          py::arg("fam"), py::arg("keys"), py::arg("branches") = false, py::arg("force_collision") = false,
          "J-a-b TEST API (N7, G-LAZY / G-DIG): payload builds of every diagonal key, the branches core on request.");
    m.def("plan_payload_fields", []() {
              py::dict d;
              d["compared"] = compared_fields();
              d["excluded"] = excluded_fields();
              return d;
          }, "J-a-b TEST API (N7, G-ALPHA): the plan payload accessor list.");
    m.def("alpha_payload_check",
          [](const py::object& fam, const std::vector<py::bytes>& keys, const std::vector<std::vector<int>>& S_list,
             int ulps) {
              TwirlRecordSampler& s = engine_of(fam);
              const int n = s.deferred_wires();
              const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
              const std::vector<RecColumn> cols = columns_of(s);
              const FramedSuperposition& B0 = s.bare_state();
              FramedSuperposition B1(B0);                // α perturbed by `ulps` ulp in its first entry
              auto* dense = dynamic_cast<DenseAmplitudes*>(B1.alpha.get());
              if (!dense || dense->b_.empty()) throw std::invalid_argument("alpha_payload_check: no dense amplitudes");
              {
                  double re = dense->b_[0].second.real();
                  for (int u = 0; u < ulps; ++u) re = std::nextafter(re, re > 0 ? 2.0 * re + 1.0 : -2.0);
                  dense->b_[0].second = std::complex<double>(re, dense->b_[0].second.imag());
              }
              PlanTableSet t0, t1;
              std::map<std::string, long> diffs;
              for (const std::string& f : compared_fields()) diffs[f] = 0;
              long nodes = 0, moved = 0, plans = 0, kappa_plans = 0;
              for (const py::bytes& kb : keys) {
                  const std::string k = kb.cast<std::string>();
                  const auto p0 = payload_of(s, G, B0, cols, k, t0);
                  const auto p1 = payload_of(s, G, B1, cols, k, t1);
                  for (const std::string& f : compared_fields()) {
                      auto a = p0.find(f), b = p1.find(f);
                      if (a == p0.end() || b == p1.end())
                          throw std::logic_error("alpha_payload_check: field '" + f + "' not produced");
                      if (a->second != b->second) ++diffs[f];
                  }
                  for (const auto& kv : p0)
                      if (std::find(compared_fields().begin(), compared_fields().end(), kv.first) == compared_fields().end())
                          throw std::logic_error("alpha_payload_check: payload field '" + kv.first + "' is not listed");
                  ++plans;
                  // the excluded group: Born-trie node values (BornTrie::p0_from_scratch on the plan's directions)
                  const TwirlRecordSampler::PlanStructure ps =
                      s.plan_structure(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size());
                  if (!ps.rec.built || ps.rec.kappa == 0) continue;
                  ++kappa_plans;
                  BornTrie bt0(B0), bt1(B1);
                  const int K = std::min(ps.rec.kappa, 10);
                  std::vector<uint8_t> z((size_t)K, 0);
                  for (int j = 0; j < K; ++j)
                      for (uint64_t zp = 0; zp < (1ULL << j); ++zp) {
                          for (int i = 0; i < j; ++i) z[(size_t)i] = (uint8_t)((zp >> i) & 1);
                          const double a = bt0.p0_from_scratch(ps.rec.lam, z.data(), j);
                          const double b = bt1.p0_from_scratch(ps.rec.lam, z.data(), j);
                          ++nodes;
                          if (std::memcmp(&a, &b, sizeof(double)) != 0) ++moved;
                      }
              }
              TermResidual& tv = TA_::family_terms(s);
              for (const std::vector<int>& S : S_list) {             // kind-2 plans
                  const std::vector<uint64_t> Sw = words_of(S, tv.num_terms());
                  const auto p0 = payload_of_kind2(s, G, B0, cols, Sw, t0);
                  const auto p1 = payload_of_kind2(s, G, B1, cols, Sw, t1);
                  for (const std::string& f : compared_fields()) if (p0.at(f) != p1.at(f)) ++diffs[f];
                  ++plans;
                  RecTable T;
                  build_record_table_general(G, B0, tv, Sw, cols, T);
                  if (T.kappa == 0) continue;
                  ++kappa_plans;
                  BornTrie bt0(B0), bt1(B1);
                  const int K = std::min(T.kappa, 10);
                  std::vector<uint8_t> z((size_t)K, 0);
                  for (int j = 0; j < K; ++j)
                      for (uint64_t zp = 0; zp < (1ULL << j); ++zp) {
                          for (int i = 0; i < j; ++i) z[(size_t)i] = (uint8_t)((zp >> i) & 1);
                          const double a = bt0.p0_from_scratch(T.lam, z.data(), j);
                          const double b = bt1.p0_from_scratch(T.lam, z.data(), j);
                          ++nodes;
                          if (std::memcmp(&a, &b, sizeof(double)) != 0) ++moved;
                      }
              }
              py::dict d;
              d["plans"] = plans;
              d["kappa_plans"] = kappa_plans;
              d["field_diffs"] = diffs;
              d["trie_nodes"] = nodes;
              d["trie_moved"] = moved;
              return d;
          },
          py::arg("fam"), py::arg("keys"), py::arg("S_list") = std::vector<std::vector<int>>(), py::arg("ulps") = 1,
          "J-a-b TEST API (N7, G-ALPHA): payload fields with the bare amplitudes vs the amplitudes moved by `ulps` ulp "
          "(diagonal plan keys and kind-2 term sets).");
    m.def("emin_categories",
          [](const py::object& fam, const std::vector<std::vector<int>>& S_list, bool reduce) {
              TwirlRecordSampler& s = engine_of(fam);
              const int n = s.deferred_wires();
              const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
              TermResidual& tv = TA_::family_terms(s);
              const int T = tv.num_terms(), ng = (int)G.gens.size();
              const std::vector<std::vector<uint64_t>> masks = s.record_channel_masks();
              const std::vector<Pauli> rops = s.record_operators();
              const int nch = (int)masks.size(), nrec = std::max(nch, (int)rops.size()), CW = (nrec + 63) / 64;
              // per term: its anticommutation pattern over the certified rows, and its record signature M·a_j
              std::vector<std::vector<uint64_t>> acert((size_t)T), sig((size_t)T);
              for (int j = 0; j < T; ++j) {
                  phi_of(G, tv.axis(j), acert[(size_t)j]);
                  sig[(size_t)j].assign((size_t)CW, 0);
                  for (int c = 0; c < nch; ++c) {
                      int p = 0;
                      for (size_t w = 0; w < masks[(size_t)c].size() && w < acert[(size_t)j].size(); ++w)
                          p ^= __builtin_popcountll(masks[(size_t)c][w] & acert[(size_t)j][w]) & 1;
                      if (p) sig[(size_t)j][(size_t)(c >> 6)] |= 1ULL << (c & 63);
                  }
                  for (int c = nch; c < nrec; ++c)          // the extras (records outside the group): their own operators
                      if (Pauli::anticommute_bit(rops[(size_t)c], tv.axis(j)))
                          sig[(size_t)j][(size_t)(c >> 6)] |= 1ULL << (c & 63);
              }
              TermShot sh;
              tv.init_shot(sh);
              py::list out;
              for (const std::vector<int>& S : S_list) {
                  std::vector<int> J = S;
                  if (reduce && !S.empty()) {
                      tv.clear_shot(sh);
                      sh.S = words_of(S, T);
                      tv.reduce_key_(sh);
                      J = bits_list(sh.S, T);
                  }
                  // K = ker(Σ_{j∈T'} sig_j) over subsets T' of J: eliminate the signatures with combination tracking
                  struct Row { std::vector<uint64_t> v, comb; int lead; };
                  std::vector<Row> R;
                  std::vector<std::vector<uint64_t>> kern;
                  const int s_ = (int)J.size(), JW = (s_ + 63) / 64;
                  for (int t = 0; t < s_; ++t) {
                      Row r{sig[(size_t)J[(size_t)t]], std::vector<uint64_t>((size_t)JW, 0), -1};
                      r.comb[(size_t)(t >> 6)] |= 1ULL << (t & 63);
                      for (const Row& o : R)
                          if ((r.v[(size_t)(o.lead >> 6)] >> (o.lead & 63)) & 1) {
                              for (int w = 0; w < CW; ++w) r.v[(size_t)w] ^= o.v[(size_t)w];
                              for (int w = 0; w < JW; ++w) r.comb[(size_t)w] ^= o.comb[(size_t)w];
                          }
                      for (int w = 0; w < CW && r.lead < 0; ++w) if (r.v[(size_t)w]) r.lead = w * 64 + __builtin_ctzll(r.v[(size_t)w]);
                      if (r.lead < 0) kern.push_back(r.comb);
                      else R.push_back(std::move(r));
                  }
                  // the surviving components differ on the certified rows touched by A·K
                  std::vector<uint64_t> emin((size_t)G.words, 0);
                  for (const auto& k : kern) {
                      std::vector<uint64_t> x((size_t)G.words, 0);
                      for (int t = 0; t < s_; ++t)
                          if ((k[(size_t)(t >> 6)] >> (t & 63)) & 1)
                              for (int w = 0; w < G.words; ++w) x[(size_t)w] ^= acert[(size_t)J[(size_t)t]][(size_t)w];
                      for (int w = 0; w < G.words; ++w) emin[(size_t)w] |= x[(size_t)w];
                  }
                  const int dimK = (int)kern.size();
                  const std::vector<int> erows = bits_list(emin, ng);
                  const char* cat = S.empty() ? "pauli" : J.empty() ? "pauli (kernel)" : dimK == 0 ? "pauli_frame"
                                    : erows.empty() ? "logical" : "extended";
                  out.append(py::make_tuple(cat, (int)S.size(), (int)J.size(), dimK, erows, J));
              }
              return out;
          },
          py::arg("fam"), py::arg("S_list"), py::arg("reduce") = true,
          "J-a-b TEST API (N7, F3-2): per term set, (category, |S|, |S_red|, dim K, E_min rows, S_red); the record "
          "signatures cover the channels (masks over the certified rows) and the extras (their own operators).");
    m.def("bare_frames", [](const py::object& fam) {
              const FramedSuperposition& st = engine_of(fam).bare_state();
              const int N = st.n(), K = st.k();
              const auto& brs = st.entries();
              const int chi = (int)brs.size();
              auto bits2d = [](int rows, int cols_) {
                  py::array_t<uint8_t> a({rows, cols_});
                  std::memset(a.mutable_data(), 0, (size_t)rows * (size_t)cols_);
                  return a;
              };
              auto fill = [N](py::array_t<uint8_t>& ax, py::array_t<uint8_t>& az, int r, const Pauli& p) {
                  for (int q = 0; q < N; ++q) {
                      ax.mutable_data()[(size_t)r * N + q] = p.xbit(q);
                      az.mutable_data()[(size_t)r * N + q] = p.zbit(q);
                  }
              };
              py::array_t<uint8_t> sx = bits2d(N, N), sz = bits2d(N, N), dx = bits2d(N, N), dz = bits2d(N, N);
              py::array_t<int8_t> sp(N), dp(N);
              py::array_t<uint8_t> eps(N);
              for (int a = 0; a < N; ++a) {                 // xtim_py.cpp branch_frames(), the same export
                  Pauli g = st.U.Zrow[a];
                  if (st.eps[a]) g.phase = (g.phase + 2) & 3;
                  fill(sx, sz, a, g);
                  sp.mutable_data()[a] = (int8_t)(g.phase & 3);
                  fill(dx, dz, a, st.U.Xrow[a]);
                  dp.mutable_data()[a] = (int8_t)(st.U.Xrow[a].phase & 3);
                  eps.mutable_data()[a] = st.eps[a] ? 1 : 0;
              }
              py::array_t<int32_t> fr(K);
              for (int d = 0; d < K; ++d) fr.mutable_data()[d] = st.free[(size_t)d];
              py::array_t<uint8_t> sig = bits2d(chi, K), bx = bits2d(chi, N), bz = bits2d(chi, N);
              py::array_t<std::complex<double>> co(chi);
              py::array_t<int8_t> bp(chi);
              for (int i = 0; i < chi; ++i) {
                  Pauli D(N);
                  for (int d = 0; d < K; ++d) {
                      sig.mutable_data()[(size_t)i * K + d] = key_bit(brs[(size_t)i].first, d);
                      if (key_bit(brs[(size_t)i].first, d)) D = Pauli::multiply(D, st.U.Xrow[(size_t)st.free[(size_t)d]]);
                  }
                  fill(bx, bz, i, D);
                  bp.mutable_data()[i] = (int8_t)(D.phase & 3);
                  co.mutable_data()[i] = brs[(size_t)i].second;
              }
              py::dict out;
              out["n"] = N; out["free"] = fr; out["eps"] = eps;
              out["stab_x"] = sx; out["stab_z"] = sz; out["stab_phase"] = sp;
              out["destab_x"] = dx; out["destab_z"] = dz; out["destab_phase"] = dp;
              out["sigma"] = sig; out["coeff"] = co;
              out["branch_x"] = bx; out["branch_z"] = bz; out["branch_phase"] = bp;
              return out;
          },
          py::arg("fam"), "J-a-b TEST API: the FAM sampler's bare state in the branch_frames() export format.");
    m.def("record_operators", [](const py::object& fam) {
              py::list l;
              for (const Pauli& P : engine_of(fam).record_operators()) l.append(pauli_t(P));
              return l;
          }, py::arg("fam"), "J-a-b TEST API: the record operators M_c (storage order), exact phase.");
    m.def("record_channel_masks", [](const py::object& fam) {
              py::list l;
              for (const auto& w : engine_of(fam).record_channel_masks()) {
                  py::array_t<uint64_t> a((py::ssize_t)w.size());
                  if (!w.empty()) std::memcpy(a.mutable_data(), w.data(), w.size() * sizeof(uint64_t));
                  l.append(a);
              }
              return l;
          }, py::arg("fam"), "J-a-b TEST API: the record channels' certified-generator masks (sigma words).");
    m.def("certified_generators", [](const py::object& fam) {
              py::list l;
              for (const Pauli& P : engine_of(fam).certified_generators()) l.append(pauli_t(P));
              return l;
          }, py::arg("fam"), "J-a-b TEST API: the certified generators g_a (sigma bit order), exact phase.");
}
