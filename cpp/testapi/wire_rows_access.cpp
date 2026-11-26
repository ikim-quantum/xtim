// wire_rows_access.cpp — S2-T7 (test API), moved out of the engine by S2-FINALFIX F2: the definition of
// TwirlRecordSampler::family_wire_rows (declared in qeccore/twirl_sampler.hpp, reached through TwirlSamplerTestAccess).
// Compiled into xtim._xtim_testapi only, so the shipped _xtim carries no copy: no run path reads it. It needs the
// sampler's Impl, hence the engine's internal header (the engine's own TUs include it the same way).
#include "../src/twirl_sampler_impl.inc.hpp"

namespace qeccore {

// The per-wire rows and the readout-flip columns as stored (see the header), and the pre-S2-T7 dense words regenerated
// from them (WireRows::words — the digest's form; rf_col from the columns' lists).
TwirlRecordSampler::FamilyWireRows TwirlRecordSampler::family_wire_rows() const {
    const Impl& I = *impl_;
    FamilyWireRows o;
    o.n = I.n; o.nchan = I.nchan; o.nextra = I.nextra_; o.CW = I.CW; o.QCW = I.QCW_; o.XR = I.XR_; o.nrf = I.nrf;
    const WireRows& R = I.wr_;
    o.xoff = R.xoff; o.xmid = R.xmid; o.xcol = R.xcol; o.zoff = R.zoff; o.zmid = R.zmid; o.zcol = R.zcol;
    const Impl::RfCols& F = I.rfc_;
    o.rf_off = F.off; o.rf_chan = F.chan; o.crf_off = F.coff; o.crf_slot = F.cslot; o.rf_obs = F.obs;
    o.qcol_x = R.words(true, WireRows::ALL, I.QCW_, 0);
    o.qcol_z = R.words(false, WireRows::ALL, I.QCW_, 0);
    o.pxrow = R.words(true, WireRows::CHANNELS, I.CW, 1);
    o.pzrow = R.words(false, WireRows::CHANNELS, I.CW, 1);
    o.xrow_x = R.words(true, WireRows::EXTRAS, I.XR_, 0);
    o.xrow_z = R.words(false, WireRows::EXTRAS, I.XR_, 0);
    o.dense_x = R.dx; o.dense_z = R.dz;
    o.rf_dslot = I.rfc_.dslot; o.rf_dwords = I.rfc_.dwords;
    if (I.nrf > 0) {
        o.rf_col.assign(std::max<size_t>((size_t)I.nrf * (size_t)I.CW, 1), 0);
        for (int sl = 0; sl < I.nrf; ++sl)
            for (int32_t i = F.off[(size_t)sl]; i < F.off[(size_t)sl + 1]; ++i)
                toggle_bit(&o.rf_col[(size_t)sl * (size_t)I.CW], F.chan[(size_t)i]);
    }
    o.bytes = R.bytes();
    o.rf_bytes = F.bytes();
    return o;
}

}  // namespace qeccore
