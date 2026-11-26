// J-a-b (N6, switched OFF in production): the TEST API of record-space B (qeccore/record_space_b.hpp), on a FAM
// sampler of xtim._xtim_testapi.compile_family_sampler (its TwirlRecordSampler via FamilySampler._engine()).
//   recb_plan_structure(fam, key) — the plan's record-space-B table as plain fields (the G-9 convention of
//       tests/test_ja_dual_run.py), plus B's op counts (J-b2 J7b: the σ dual basis "d" side is deleted);
//   recb_replay(fam, keys, passes, mode) — the B-banking replay: every key built `passes` times by one builder; mode
//       "recb" (build_record_table_recB) or "plan_structure" (the ledger's t_build call itself). Returns a digest of
//       the tables. The measured function is the noinline recb_build_one_ (callgrind --toggle-collect).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "twirl_test_access.hpp"

#include "qeccore/fault_record_law.hpp"
#include "qeccore/pauli_kernels.hpp"
#include "qeccore/record_space_b.hpp"
#include "qeccore/record_table.hpp"
#include "qeccore/twirl_planes.hpp"
#include "qeccore/twirl_sampler.hpp"

namespace py = pybind11;
using namespace qeccore;

namespace {

TwirlRecordSampler& engine_of(const py::object& fam) {
    py::capsule c = fam.attr("_engine")();
    if (std::strcmp(c.name(), "xtim.TwirlRecordSampler") != 0)
        throw std::invalid_argument("recb: expected a FamilySampler (compile_family_sampler)");
    return *static_cast<TwirlRecordSampler*>(c.get_pointer());
}

// The sampler's record columns (storage order) from its public accessors.
std::vector<RecColumn> columns_of(const TwirlRecordSampler& s) {
    const std::vector<Pauli> ops = s.record_operators();
    const std::vector<int> ord = s.record_column_order();
    const std::vector<std::vector<uint64_t>> masks = s.record_channel_masks();
    const std::vector<uint8_t> refs = s.record_channel_refs();
    const int nch = s.num_channels();
    std::vector<RecColumn> cols(ops.size());
    for (size_t c = 0; c < ops.size(); ++c) {
        cols[c].M = from_dense(ops[c]);
        cols[c].order = ord[c];
        cols[c].channel = (int)c < nch;
        if ((int)c < nch) {
            cols[c].mask = mask_list(masks[c]);
            cols[c].ref = refs[c];
        }
    }
    return cols;
}

py::dict table_dict(const RecTable& T) {
    py::dict res;
    res["rec_built"] = T.built;
    res["m"] = T.m;
    res["rho"] = T.rho;
    res["kappa_rec"] = T.kappa;
    res["pivot_rec"] = T.pivot_rec;
    res["born_rec"] = T.born_rec;
    py::array_t<uint8_t> A({(py::ssize_t)T.m, (py::ssize_t)T.rho});
    py::array_t<uint8_t> B({(py::ssize_t)T.m, (py::ssize_t)T.kappa});
    py::array_t<uint8_t> be((py::ssize_t)T.m);
    for (int c = 0; c < T.m; ++c) {
        for (int k = 0; k < T.rho; ++k)
            A.mutable_data()[(size_t)c * T.rho + k] = (uint8_t)((T.coin_row[(size_t)k][(size_t)(c >> 6)] >> (c & 63)) & 1);
        for (int j = 0; j < T.kappa; ++j)
            B.mutable_data()[(size_t)c * T.kappa + j] = (uint8_t)((T.born_row[(size_t)j][(size_t)(c >> 6)] >> (c & 63)) & 1);
        be.mutable_data()[c] = (uint8_t)((T.base.empty() ? 0 : (T.base[(size_t)(c >> 6)] >> (c & 63))) & 1);
    }
    res["rec_A"] = A;
    res["rec_B"] = B;
    res["rec_beta"] = be;
    const int kk = T.lam.empty() ? 0 : (int)T.lam[0].a.size();
    py::array_t<uint8_t> lam({(py::ssize_t)T.kappa, (py::ssize_t)(2 * kk)});
    py::array_t<uint8_t> lph((py::ssize_t)T.kappa);
    for (int j = 0; j < T.kappa; ++j) {
        for (int i = 0; i < kk; ++i) {
            lam.mutable_data()[(size_t)j * 2 * kk + i] = T.lam[(size_t)j].a[(size_t)i];
            lam.mutable_data()[(size_t)j * 2 * kk + kk + i] = T.lam[(size_t)j].b[(size_t)i];
        }
        lph.mutable_data()[j] = (uint8_t)fd_eff_phase(T.lam[(size_t)j]);
    }
    res["rec_lam"] = lam;
    res["rec_lam_phase"] = lph;
    return res;
}

// A content digest of a table's canonical fields (FNV-1a), Λ by (a, b, effective phase).
uint64_t table_digest(const RecTable& T, uint64_t h) {
    auto mix = [&](uint64_t v) { h = (h ^ v) * 0x100000001B3ull; };
    mix((uint64_t)T.m); mix((uint64_t)T.rho); mix((uint64_t)T.kappa);
    for (int c : T.pivot_rec) mix((uint64_t)c);
    for (int c : T.born_rec) mix((uint64_t)c);
    for (uint64_t w : T.base) mix(w);
    for (const auto& r : T.coin_row) for (uint64_t w : r) mix(w);
    for (const auto& r : T.born_row) for (uint64_t w : r) mix(w);
    for (const FreeDir& d : T.lam) {
        for (uint8_t v : d.a) mix(v);
        for (uint8_t v : d.b) mix(v);
        mix((uint64_t)fd_eff_phase(d));
    }
    return h;
}

DiagNormalForm prefix_free(const DiagNormalForm& nf, int n) {
    DiagNormalForm nf0 = nf;
    nf0.prefix = Pauli(n);
    return nf0;
}

// The two measured builders (B-banking). Same inputs: the certified planes, the bare state, the parsed plan (identity
// prefix) and the record columns.
__attribute__((noinline)) void recb_build_one_(const CertifiedGroupPlanes& G, const FramedSuperposition& bare,
                                               const DiagNormalForm& nf, const std::vector<RecColumn>& cols,
                                               RecTable& T) {
    build_record_table_recB(G, bare, nf, cols, T);
}

// S2-T6: the record columns' oracle. A circuit's sampler (compile_family_sampler, selfcheck 0) against the pre-S2-T6
// provenance (ProvenanceRREF, the dense combination-tracking echelon over the generators): every channel column's mask
// (words) and ref must equal the oracle's decomposition of its operator M and G.reduce's sign; Π_{mask} g must equal M
// up to the sign ref (x, z and phase); every extra must reduce to LOGICAL / ANTI. The DEM's twirl-law columns
// (RecordColumns, built on a fresh CertifiedGroupPlanes of the same generators) are held to the same oracle on every
// column. Returns None when the circuit does not compile to a sampler (a refusal), else the counts and the first
// mismatches (an empty list on agreement).
py::object record_columns_oracle(const std::string& text) {
    using namespace pybind11::literals;
    py::object fam;
    try {
        fam = py::module_::import("xtim._xtim_testapi").attr("compile_family_sampler")(text, "selfcheck"_a = 0);
    } catch (const py::error_already_set&) {
        return py::none();
    }
    TwirlRecordSampler& s = engine_of(fam);
    const std::vector<SparsePauli>& gens = s.certified_generators();
    const int n = s.deferred_wires();
    const ProvenanceRREF P = ProvenanceRREF::build(gens, n);
    const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(std::vector<SparsePauli>(gens), n);
    const RecordColumns RC(G);
    const std::vector<RecColumn> cols = columns_of(s);
    const std::vector<Pauli> ops = s.record_operators();
    const int nch = s.num_channels(), GW = ((int)gens.size() + 63) / 64;
    py::list bad;
    long channels = 0, extras = 0, rc_checked = 0;
    auto fail = [&](const std::string& what, size_t c) {
        if (py::len(bad) < 20) bad.append(what + " (column " + std::to_string(c) + ")");
    };
    for (size_t c = 0; c < cols.size(); ++c) {
        const Pauli& M = ops[c];
        const Membership m = G.reduce(M);
        const bool chan = (int)c < nch;
        std::vector<uint64_t> omask;
        uint8_t oref = 0;
        if (m.verdict == Membership::IN_GROUP) {
            if (!P.decompose(M, omask)) fail("oracle provenance residual nonzero", c);
            oref = (uint8_t)((m.rep.phase >> 1) & 1);
        }
        if (chan) {
            ++channels;
            if (m.verdict != Membership::IN_GROUP) { fail("channel not IN_GROUP", c); continue; }
            const std::vector<uint64_t> mk = mask_words(cols[c].mask, GW);
            if (mk != omask) fail("channel mask != oracle", c);
            if ((cols[c].ref & 1) != oref) fail("channel ref != oracle", c);
            Pauli Pm(n);
            for (int i = 0; i < (int)gens.size(); ++i)
                if ((mk[(size_t)(i >> 6)] >> (i & 63)) & 1) pmul_into(Pm, gens[(size_t)i]);
            if (Pm.x != M.x || Pm.z != M.z) fail("product of the mask's generators != M (support)", c);
            else if (((M.phase - Pm.phase) & 3) != 2 * (cols[c].ref & 1))
                fail("product of the mask's generators != M (sign ref)", c);
        } else {
            ++extras;
            if (m.verdict == Membership::IN_GROUP) fail("extra is IN_GROUP", c);
        }
        const RecColumn rc = RC.column(M, cols[c].order);   // the DEM twirl-law route (fault_record_law)
        ++rc_checked;
        if (rc.channel != (m.verdict == Membership::IN_GROUP)) fail("RecordColumns channel flag", c);
        if (rc.channel) {
            const std::vector<uint64_t> mk = mask_words(rc.mask, GW);
            if (mk != omask) fail("RecordColumns mask != oracle", c);
            if ((rc.ref & 1) != oref) fail("RecordColumns ref != oracle", c);
        }
    }
    py::dict out;
    out["n"] = n;
    out["gens"] = (int)gens.size();
    out["rank"] = G.rank;
    out["channels"] = channels;
    out["extras"] = extras;
    out["record_columns"] = rc_checked;
    out["mismatches"] = bad;
    return out;
}

// S2-T7: the per-wire record rows' oracle. The sampler's stored rows (TwirlRecordSampler::family_wire_rows) against
// the pre-S2-T7 dense builders, written out here from the public accessors (the record operators, the certified
// generators and the channel masks): qcol_x / qcol_z (per wire, every record column whose operator has Z / X there),
// pxrow / pzrow (lean L6: per wire, the XOR over the generators with Z / X there of their channel-membership rows),
// xrow_x / xrow_z (the extras' columns at the extras' bit positions) and the readout-flip columns rf_col (the
// transpose of the channels' slot sets). Checks: the stored lists are well formed (ascending, split at nchan), the
// words regenerated from the lists equal the sampler's regenerated words (the digest's form), and both equal the old
// builders' words exactly (sizes included). Returns None on a refusal, else the shape, the storage and the first
// mismatches.
py::object wire_rows_oracle(const std::string& text) {
    using namespace pybind11::literals;
    py::object fam;
    try {
        fam = py::module_::import("xtim._xtim_testapi").attr("compile_family_sampler")(text, "selfcheck"_a = 0);
    } catch (const py::error_already_set&) {
        return py::none();
    }
    TwirlRecordSampler& s = engine_of(fam);
    const TwirlSamplerTestAccess::FamilyWireRows W = TwirlSamplerTestAccess::family_wire_rows(s);
    const std::vector<Pauli> ops = s.record_operators();
    const std::vector<SparsePauli>& gens = s.certified_generators();
    const std::vector<std::vector<uint64_t>> masks = s.record_channel_masks();
    const int n = s.deferred_wires(), nch = s.num_channels(), m = (int)ops.size();
    const int ng = (int)gens.size();
    py::list bad;
    auto fail = [&](const std::string& what) { if (py::len(bad) < 20) bad.append(what); };
    // shape (the old widths)
    const int CW = (nch + 63) / 64, QCW = std::max(1, (m + 63) / 64), nextra = m - nch;
    const int XR = nextra > 0 ? std::max(1, (nextra + 63) / 64) : 0;
    if (W.n != n || W.nchan != nch || W.nextra != nextra || W.CW != CW || W.QCW != QCW || W.XR != XR)
        fail("shape differs from the old widths");
    // the old builders
    auto each_bit = [](const std::vector<uint64_t>& w, auto&& f) {
        for (size_t k = 0; k < w.size(); ++k)
            for (uint64_t x = w[k]; x; x &= x - 1) f((int)(k * 64) + __builtin_ctzll(x));
    };
    std::vector<uint64_t> qx((size_t)n * QCW, 0), qz((size_t)n * QCW, 0);
    for (int c = 0; c < m; ++c) {
        each_bit(ops[(size_t)c].z, [&](int q) { qx[(size_t)q * QCW + (size_t)(c >> 6)] |= 1ULL << (c & 63); });
        each_bit(ops[(size_t)c].x, [&](int q) { qz[(size_t)q * QCW + (size_t)(c >> 6)] |= 1ULL << (c & 63); });
    }
    std::vector<uint64_t> xx((size_t)n * XR, 0), xz((size_t)n * XR, 0);
    for (int e = 0; e < nextra; ++e) {
        each_bit(ops[(size_t)(nch + e)].z, [&](int q) { xx[(size_t)q * XR + (size_t)(e >> 6)] |= 1ULL << (e & 63); });
        each_bit(ops[(size_t)(nch + e)].x, [&](int q) { xz[(size_t)q * XR + (size_t)(e >> 6)] |= 1ULL << (e & 63); });
    }
    std::vector<uint64_t> px(std::max<size_t>((size_t)n * CW, 1), 0), pz(px);
    {
        std::vector<uint64_t> gen_chan((size_t)std::max(ng, 1) * (size_t)std::max(CW, 1), 0);
        for (int c = 0; c < nch; ++c)
            for (int i = 0; i < ng; ++i)
                if ((masks[(size_t)c][(size_t)(i >> 6)] >> (i & 63)) & 1)
                    gen_chan[(size_t)i * CW + (size_t)(c >> 6)] ^= 1ULL << (c & 63);
        for (int i = 0; i < ng; ++i) {
            const uint64_t* gr = &gen_chan[(size_t)i * CW];
            for (int32_t q : gens[(size_t)i].z) for (int t = 0; t < CW; ++t) px[(size_t)q * CW + t] ^= gr[t];
            for (int32_t q : gens[(size_t)i].x) for (int t = 0; t < CW; ++t) pz[(size_t)q * CW + t] ^= gr[t];
        }
    }
    std::vector<uint64_t> rf;
    if (W.nrf > 0) {
        if ((int)W.crf_off.size() != nch + 1) fail("crf offsets: not nchan + 1");
        rf.assign(std::max<size_t>((size_t)W.nrf * CW, 1), 0);
        for (int c = 0; c + 1 < (int)W.crf_off.size(); ++c)
            for (int32_t i = W.crf_off[(size_t)c]; i < W.crf_off[(size_t)c + 1]; ++i) {
                const int sl = W.crf_slot[(size_t)i];
                if (sl < 0 || sl >= W.nrf || (i > W.crf_off[(size_t)c] && sl <= W.crf_slot[(size_t)i - 1])) {
                    fail("crf slot out of range or not ascending");
                    continue;
                }
                rf[(size_t)sl * CW + (size_t)(c >> 6)] |= 1ULL << (c & 63);
            }
    }
    // the stored lists: well formed, and their words
    auto from_lists = [&](const std::vector<int32_t>& off, const std::vector<int32_t>& mid, const std::vector<int32_t>& col,
                          const char* nm, std::vector<uint64_t>& all, std::vector<uint64_t>& ch,
                          std::vector<uint64_t>& ex) {
        all.assign((size_t)n * QCW, 0);
        ch.assign(std::max<size_t>((size_t)n * CW, 1), 0);
        ex.assign((size_t)n * XR, 0);
        if ((int)off.size() != n + 1 || (int)mid.size() != n || off[0] != 0 || (size_t)off[(size_t)n] != col.size()) {
            fail(std::string(nm) + ": offsets malformed");
            return;
        }
        for (int q = 0; q < n; ++q) {
            const int32_t b = off[(size_t)q], md = mid[(size_t)q], e = off[(size_t)q + 1];
            if (!(b <= md && md <= e)) { fail(std::string(nm) + ": mid outside the row"); return; }
            for (int32_t i = b; i < e; ++i) {
                const int c = col[(size_t)i];
                if (c < 0 || c >= m || (i > b && c <= col[(size_t)i - 1]) || ((i < md) != (c < nch))) {
                    fail(std::string(nm) + ": entry out of range, not ascending or on the wrong side of mid");
                    return;
                }
                all[(size_t)q * QCW + (size_t)(c >> 6)] |= 1ULL << (c & 63);
                if (c < nch) ch[(size_t)q * CW + (size_t)(c >> 6)] |= 1ULL << (c & 63);
                else ex[(size_t)q * XR + (size_t)((c - nch) >> 6)] |= 1ULL << ((c - nch) & 63);
            }
        }
    };
    std::vector<uint64_t> lqx, lpx, lxx, lqz, lpz, lxz;
    from_lists(W.xoff, W.xmid, W.xcol, "x rows", lqx, lpx, lxx);
    from_lists(W.zoff, W.zmid, W.zcol, "z rows", lqz, lpz, lxz);
    auto eq = [&](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b, const std::string& what) {
        if (a != b) fail(what);
    };
    eq(W.qcol_x, qx, "qcol_x (regenerated) != old builder"); eq(W.qcol_z, qz, "qcol_z (regenerated) != old builder");
    eq(W.pxrow, px, "pxrow (regenerated) != old builder"); eq(W.pzrow, pz, "pzrow (regenerated) != old builder");
    eq(W.xrow_x, xx, "xrow_x (regenerated) != old builder"); eq(W.xrow_z, xz, "xrow_z (regenerated) != old builder");
    eq(lqx, qx, "qcol_x (lists) != old builder"); eq(lqz, qz, "qcol_z (lists) != old builder");
    eq(lpx, px, "pxrow (lists) != old builder"); eq(lpz, pz, "pzrow (lists) != old builder");
    eq(lxx, xx, "xrow_x (lists) != old builder"); eq(lxz, xz, "xrow_z (lists) != old builder");
    if (W.dense_x.empty() != W.dense_z.empty()) fail("the density switch kept one of the two word blocks");
    if (!W.dense_x.empty()) {                                // the density switch's stored words
        eq(W.dense_x, qx, "stored dense X words != old builder");
        eq(W.dense_z, qz, "stored dense Z words != old builder");
    }
    int rf_dense_slots = 0;
    if (W.nrf > 0) {
        std::vector<uint64_t> lrf(std::max<size_t>((size_t)W.nrf * CW, 1), 0);
        if ((int)W.rf_off.size() != W.nrf + 1 || W.rf_off[0] != 0 || (size_t)W.rf_off.back() != W.rf_chan.size()) {
            fail("rf offsets malformed");
        } else {
            for (int sl = 0; sl < W.nrf; ++sl)
                for (int32_t i = W.rf_off[(size_t)sl]; i < W.rf_off[(size_t)sl + 1]; ++i) {
                    const int c = W.rf_chan[(size_t)i];
                    if (c < 0 || c >= nch || (i > W.rf_off[(size_t)sl] && c <= W.rf_chan[(size_t)i - 1])) {
                        fail("rf entry out of range or not ascending");
                        break;
                    }
                    lrf[(size_t)sl * CW + (size_t)(c >> 6)] |= 1ULL << (c & 63);
                }
        }
        eq(W.rf_col, rf, "rf_col (regenerated) != the transpose of the slot sets");
        eq(lrf, rf, "rf_col (lists) != the transpose of the slot sets");
        if ((int)W.rf_obs.size() != W.nrf) fail("rf_obs size != nrf");
        // fix round 1: the per-slot density switch's stored words equal the slot's column of the transpose
        if ((int)W.rf_dslot.size() != W.nrf) fail("rf_dslot size != nrf");
        else {
            const int Wd = std::max(CW, 1);
            int nd = 0;
            for (int sl = 0; sl < W.nrf; ++sl) {
                const int d = W.rf_dslot[(size_t)sl];
                if (d < 0) continue;
                if (d != nd++) { fail("rf dense rows not in slot order"); break; }
                if ((size_t)(d + 1) * Wd > W.rf_dwords.size()) { fail("rf dense row out of range"); break; }
                for (int k = 0; k < CW; ++k)
                    if (W.rf_dwords[(size_t)d * Wd + k] != rf[(size_t)sl * CW + k]) { fail("rf dense words != the transpose"); break; }
            }
            if ((size_t)nd * Wd != W.rf_dwords.size()) fail("rf dense words: extra rows");
            rf_dense_slots = nd;
        }
    }
    long weight = 0;
    for (const Pauli& M : ops)
        for (size_t k = 0; k < M.x.size(); ++k) weight += __builtin_popcountll(M.x[k]) + __builtin_popcountll(M.z[k]);
    py::dict out;
    out["n"] = n;
    out["nchan"] = nch;
    out["nextra"] = nextra;
    out["nrf"] = W.nrf;
    out["entries"] = (long)(W.xcol.size() + W.zcol.size());
    out["weight"] = weight;                                  // Σ |M_c.x| + |M_c.z| over the record columns
    out["rf_entries"] = (long)W.rf_chan.size();
    out["bytes"] = W.bytes;
    out["rf_bytes"] = W.rf_bytes;
    out["dense_words"] = !W.dense_x.empty();                 // the density switch kept the full rows' words
    out["rf_dense_slots"] = rf_dense_slots;                  // readout-flip slots that keep their words
    out["dense_bytes"] = (size_t)8 * (2 * (size_t)n * QCW + 2 * std::max<size_t>((size_t)n * CW, 1) + 2 * (size_t)n * XR);
    out["rf_dense_bytes"] = (size_t)8 * ((size_t)W.nrf * CW + (size_t)nch * (size_t)((W.nrf + 63) / 64));
    out["mismatches"] = bad;
    return out;
}

}  // namespace

void register_branch(py::module_& m);   // branch_py.cpp (J-a-b N7)

void register_recb(py::module_& m) {
    register_branch(m);
    m.def("recb_plan_structure",
          [](const py::object& fam, py::bytes key) {
              TwirlRecordSampler& s = engine_of(fam);
              const std::string k = key.cast<std::string>();
              const TwirlRecordSampler::PlanStructure ps =
                  s.plan_structure(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size());
              py::dict out;
              out["fallback"] = ps.law.fallback || !ps.rec.built;
              out["kappa"] = ps.law.kappa;
              out["nchan"] = s.num_channels();
              if (ps.law.fallback || !ps.rec.built) return out;
              const int n = s.deferred_wires();
              const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
              RecTable T;
              RecBOps ops;
              build_record_table_recB(G, s.bare_state(), ps.nf, columns_of(s), T, &ops);
              out["b"] = table_dict(T);
              py::dict o;
              o["touched"] = ops.touched;
              o["gen_visits"] = ops.gen_visits;
              o["elim_rows"] = ops.elim_rows;
              o["pre_rows"] = ops.pre_rows;
              o["pivots"] = ops.pivots;
              o["kb_read"] = ops.kb_read;
              o["mask_words"] = ops.mask_words;     // lean L4
              o["law_reduces"] = ops.law_reduces;
              out["ops"] = o;
              return out;
          },
          py::arg("fam"), py::arg("key"),
          "J-a-b TEST API (N6): record-space B's table of one plan key (plan_structure's format) and its op counts.");
    m.def("_record_columns_oracle", &record_columns_oracle, py::arg("text"),
          "S2-T6 TEST API: the sampler's record columns (mask / ref / M) and the DEM's RecordColumns against the "
          "pre-S2-T6 provenance oracle; None on a refusal, else counts and the first mismatches.");
    m.def("_wire_rows_oracle", &wire_rows_oracle, py::arg("text"),
          "S2-T7 TEST API: the sampler's per-wire record rows and readout-flip columns (as stored) against the "
          "pre-S2-T7 dense builders; None on a refusal, else the shape, the storage and the first mismatches.");
    m.def("recb_replay",
          [](const py::object& fam, const std::vector<py::bytes>& keys, int passes, const std::string& mode) {
              TwirlRecordSampler& s = engine_of(fam);
              if (mode != "recb" && mode != "plan_structure")
                  throw std::invalid_argument("recb_replay: mode must be 'recb' or 'plan_structure'");
              std::vector<std::string> ks;
              for (const py::bytes& b : keys) ks.push_back(b.cast<std::string>());
              uint64_t h = 0xcbf29ce484222325ull;
              long built = 0;
              {
                  py::gil_scoped_release rel;           // pure C++ below; Python objects only after this scope
                  if (mode == "plan_structure") {
                      for (int p = 0; p < passes; ++p)
                          for (const std::string& k : ks) {
                              const TwirlRecordSampler::PlanStructure ps =
                                  s.plan_structure(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size());
                              if (p == 0) { h = table_digest(ps.rec, h); built += ps.rec.built; }
                          }
                  } else {
                      const int n = s.deferred_wires();
                      const CertifiedGroupPlanes G = CertifiedGroupPlanes::build(s.certified_generators(), n);
                      const std::vector<RecColumn> cols = columns_of(s);
                      std::vector<DiagNormalForm> nfs;
                      for (const std::string& k : ks)
                          nfs.push_back(prefix_free(
                              s.plan_structure(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size()).nf, n));
                      RecTable T;
                      for (int p = 0; p < passes; ++p)
                          for (const DiagNormalForm& nf : nfs) {
                              recb_build_one_(G, s.bare_state(), nf, cols, T);
                              if (p == 0) { h = table_digest(T, h); built += T.built; }
                          }
                  }
              }
              return py::make_tuple(h, built);
          },
          py::arg("fam"), py::arg("keys"), py::arg("passes") = 1, py::arg("mode") = "recb",
          "J-a-b TEST API (N6, B-banking): build every plan key `passes` times by one builder; returns "
          "(table digest, tables built). Releases the GIL while building: the sampler must not be used from another "
          "thread meanwhile (benchmark use).");
}
