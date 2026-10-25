// twirl_sampler — V2-T3: TwirlRecordSampler, the end-to-end twirl record sampler.
//
// MOVE-only extraction of the framed_bench `twirl_records` mode body (byte-equality gated
// at fixed seeds: identical RNG draw order, identical counters, identical oracles). The
// constructor is the mode's setup phase (channel setup, propagation table, per-alt sparse
// atoms + masks + flat blob, plan caches, disk load); run() is the shot loop. See the
// header and the original mode comments below for the algorithm.


#include "twirl_sampler_impl.inc.hpp"   // includes, the file-local helpers, TwirlRecordSampler::Impl

namespace qeccore {

namespace {
// V3-T3 R2: structural FNV-64 of the deferred circuit + terminal reads for the AUTO plan-cache
// file key. Noise probability VALUES are deliberately EXCLUDED (the plan cache is noise-blind:
// one auto file serves whole p-sweeps); everything shape-defining is mixed (kinds, gates,
// targets, channels, qubit lists, measure bases/inverts, observables, feedback offsets, reads).
// Mirrors the pybind deferred_signature content, hashed instead of concatenated.
uint64_t deferred_signature_fnv(const Circuit& deferred,
                                const std::vector<std::pair<int, int>>& reads) {
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 0x100000001B3ull; };
    mix((uint64_t)(uint32_t)deferred.n);
    for (const Instr& ins : deferred.stream) {
        mix(0x100u + (uint64_t)(int)ins.kind);
        switch (ins.kind) {
            case Instr::Kind::Gate:
                mix((uint64_t)(int)ins.gate);
                for (int t : ins.targets) mix((uint64_t)(uint32_t)t);
                break;
            case Instr::Kind::Noise:
                mix((uint64_t)(int)ins.channel);               // probs EXCLUDED (noise-blind)
                for (int q : ins.qubits) mix((uint64_t)(uint32_t)q);
                if (ins.channel == NoiseChannel::CORRELATED) {   // chain term supports: structure
                    mix(0xC0DEull);
                    for (unsigned char c : ins.corr) mix(0x40u + (uint64_t)c);
                }
                break;
            case Instr::Kind::Measure:
                mix((uint64_t)(int)ins.basis);
                mix(ins.invert ? 0x21u : 0x2Eu);               // flip PROBABILITY excluded
                for (int q : ins.qubits) mix((uint64_t)(uint32_t)q);
                break;
            case Instr::Kind::Reset:
                for (int q : ins.qubits) mix((uint64_t)(uint32_t)q);
                break;
            case Instr::Kind::Observable:
                for (const PauliTerm& t : ins.obs) {
                    mix((uint64_t)(uint32_t)t.qubit);
                    mix(0x70u + (uint64_t)(int)t.p);
                }
                for (int r : ins.obs_frame) mix(0xF00u ^ (uint64_t)(uint32_t)r);
                break;
            case Instr::Kind::ControlledPauli:
                mix((uint64_t)(int)ins.basis);
                mix((uint64_t)(uint32_t)ins.control_rec_offset);
                for (int q : ins.qubits) mix((uint64_t)(uint32_t)q);
                break;
            default: break;   // IfBegin/IfEnd: no shape-defining content
        }
    }
    mix(0x5EAD5ull);                                           // reads-section separator
    for (const auto& pr : reads) {
        mix((uint64_t)(uint32_t)pr.first);
        mix((uint64_t)(uint32_t)pr.second);
    }
    return h;
}
}  // namespace

void (*g_twirl_alt_blob_test_hook)(std::vector<uint64_t>&) = nullptr;

// The run_w instances are compiled in twirl_sampler_fam.cpp, never here (J-b2 J3: the only instances).
extern template bool TwirlRecordSampler::Impl::run_w<8>(long);
extern template bool TwirlRecordSampler::Impl::run_w<0>(long);

TwirlRecordSampler::Impl::RunFn TwirlRecordSampler::Impl::pick_run_() const {
    return CW <= 8 ? &Impl::run_w<8> : &Impl::run_w<0>;
}

TwirlRecordSampler::Impl::Impl(const FramedSuperposition& bare_in, const Circuit& deferred,
                               const std::vector<std::pair<int, int>>& reads,
                               const std::vector<std::vector<int>>& detectors,
                               const std::vector<std::pair<int, std::vector<int>>>& observables,
                               uint64_t seed, const TwirlRecordOptions& opt)
    : Impl(bare_in, deferred, reads, detectors, observables, seed, opt,
           std::make_shared<TwirlPlanCache>()) {}

TwirlRecordSampler::Impl::Impl(const FramedSuperposition& bare_in, const Circuit& deferred,
                               const std::vector<std::pair<int, int>>& reads,
                               const std::vector<std::vector<int>>& detectors,
                               const std::vector<std::pair<int, std::vector<int>>>& observables,
                               uint64_t seed, const TwirlRecordOptions& opt,
                               std::shared_ptr<TwirlPlanCache> plans,
                               std::optional<PropagationTable>&& class_table)
    : bare(bare_in), n(bare_in.n()), disk_path(opt.disk_path), selfcheck(opt.selfcheck),
      cache(std::move(plans)),
      trie_(bare_in) {
    int want_nchan = opt.want_nchan;           // --nchan N: channel-count scaling measurements
    const bool circ_chan = opt.circuit_channels;   // --circuit-channels: compile DETECTORs to σ-masks
    synthetic_channels_ = !circ_chan;              // speed-kill T1: mrng-drawn sch ⇒ set_seed refuses
    if (want_nchan < 1) want_nchan = 1;
    const double factor = opt.p_factor;
    gens = bare.certified_stabilizers();
    ng = (int)gens.size();
    GW = (ng + 63) / 64;
    G = CertifiedGroupPlanes::build(gens, n);
    // The PPR-retry atoms' dense error tableaux (PropResult::general) are not built: this sampler
    // reads their residuals (`ppr`) only. A caller's class_table is this very table (all in class).
    if (class_table) table = std::move(*class_table);
    else table = build_propagation_table(deferred, /*ppr_retry=*/true, AtomSet::ChannelSupport,
                                         /*build_general=*/false);
    if (!table.all_in_class) { setup_error = 3; return; }   // the bindings type and word the refusal
    // CH-class (non-diagonal) atoms: their shots take the term route (kind 2, the ppr_records class);
    // shots whose fired alts are all-diagonal keep the abelian fast path.
    ppr_tables = table.has_general;

    // S channels: 16 singles + random products up to --nchan (default 32, matching the
    // twirl_collapse gate; --nchan 128 covers a full d5-scale detector set (108 channels) for
    // the channel-count scaling measurement the review demanded — per-channel cost is linear in
    // CHANNEL WORDS, and CW = ceil(nchan/64)).
    auto set_bit = [&](std::vector<uint64_t>& w, int i) { w[i >> 6] |= (1ULL << (i & 63)); };
    // Synthetic channels are dead weight under circuit_channels (the compiled channels
    // replace them below) — and the random-product loop must cap the subset size at ng:
    // drawing sz > ng DISTINCT indices never terminates (2026-07-16 fix: a 2-qubit gauge
    // circuit with ng=2 hung the compile). Skipping in circuit mode changes nothing
    // downstream (mrng is local); synthetic-mode draw sequences are unchanged whenever
    // ng > 5, i.e. on every existing benchmark.
    if (!circ_chan) {
        for (int i = 0; i < std::min(ng, 16) && (int)sch.size() < want_nchan; ++i) {
            SChan c; c.GS = gens[i]; c.mask.assign(GW, 0); set_bit(c.mask, i); sch.push_back(std::move(c));
        }
        std::mt19937_64 mrng(seed + 4242);
        while ((int)sch.size() < want_nchan && ng >= 2) {
            const int sz = std::min(ng, 2 + (int)(mrng() % 4));
            std::set<int> pick;
            while ((int)pick.size() < sz) pick.insert((int)(mrng() % ng));
            SChan c; c.mask.assign(GW, 0); Pauli P(n);
            for (int idx : pick) { P = Pauli::multiply(P, gens[idx]); set_bit(c.mask, idx); }
            c.GS = P; sch.push_back(std::move(c));
        }
    }
    // ── V2-T2: CIRCUIT channel compilation (--circuit-channels). Each DETECTOR's combined
    // record operator W = Π_k W_k (terminal reads) is classified against the certified group:
    //   IN_GROUP → σ-mask channel: bit_D = ref ⊕ (mask·σ); the mask is the generator
    //              PROVENANCE (the combination the RREF consumed), ref the exact reduction
    //              sign ((1−ε)/2 — the noiseless value);
    //   LOGICAL  → gauge channel (twirled semantics: DECLARED, excluded from the 5σ gate);
    //   ANTI     → refused (counted loudly; cannot be a deterministic record channel).
    // OBSERVABLEs with non-IN_GROUP operators are refused per the spec (workflow arc).
    if (circ_chan) {
        sch.clear();
        // provenance RREF over the generators' symplectic rows (combo-tracking)
        const int NW2 = (n + 63) / 64;
        struct PRow { std::vector<uint64_t> s; std::vector<uint64_t> c; int pc; };
        std::vector<PRow> prref;
        auto sympl = [&](const Pauli& P) {
            std::vector<uint64_t> s(2 * (size_t)NW2, 0);
            for (int w = 0; w < NW2; ++w) { s[w] = P.x[w]; s[NW2 + w] = P.z[w]; }
            return s;
        };
        auto sfirst = [&](const std::vector<uint64_t>& s) {
            for (size_t w = 0; w < s.size(); ++w)
                if (s[w]) return (int)w * 64 + __builtin_ctzll(s[w]);
            return -1;
        };
        for (int i = 0; i < ng; ++i) {
            PRow r; r.s = sympl(gens[i]); r.c.assign(GW, 0); set_bit(r.c, i);
            for (const PRow& e : prref)
                if ((r.s[e.pc >> 6] >> (e.pc & 63)) & 1) {
                    for (size_t w = 0; w < r.s.size(); ++w) r.s[w] ^= e.s[w];
                    for (int w = 0; w < GW; ++w) r.c[w] ^= e.c[w];
                }
            r.pc = sfirst(r.s);
            if (r.pc >= 0) prref.push_back(std::move(r));
        }
        auto record_pauli = [&](int k) {
            Pauli P(n);
            const int b2 = reads[(size_t)k].first, q = reads[(size_t)k].second;
            if (b2 == 0) P.setx(q);
            else if (b2 == 2) P.setz(q);
            else { P.setx(q); P.setz(q); P.phase = 1; }          // Y = i·XZ
            return P;
        };
        auto classify = [&](const std::vector<int>& recs, std::vector<uint64_t>& mask,
                            uint8_t& ref) {
            Pauli W(n);
            for (int k : recs) W = Pauli::multiply(W, record_pauli(k));
            Membership m = G.reduce(W);
            if (m.verdict != Membership::IN_GROUP) return m.verdict;
            std::vector<uint64_t> s = sympl(W);
            mask.assign(GW, 0);
            for (const PRow& e : prref)
                if ((s[e.pc >> 6] >> (e.pc & 63)) & 1) {
                    for (size_t w = 0; w < s.size(); ++w) s[w] ^= e.s[w];
                    for (int w = 0; w < GW; ++w) mask[w] ^= e.c[w];
                }
            for (uint64_t w : s)
                if (w) throw std::logic_error("provenance residual nonzero on IN_GROUP");
            if (m.rep.phase & 1) throw std::logic_error("non-Hermitian detector reduction");
            ref = (uint8_t)((m.rep.phase >> 1) & 1);
            return m.verdict;
        };
        auto record_product = [&](const std::vector<int>& recs) {
            Pauli Wp(n);
            for (int k : recs) Wp = Pauli::multiply(Wp, record_pauli(k));
            return Wp;
        };
        for (size_t di = 0; di < detectors.size(); ++di) {
            std::vector<uint64_t> mask;
            uint8_t ref = 0;
            const auto v = classify(detectors[di], mask, ref);
            if (v == Membership::IN_GROUP) {
                // S0(f): no channel cap (run-width channel rows; the old 512 cap refused here).
                rec_cols_.push_back(RecColumn{record_product(detectors[di]), rec_order_(0, (int)di), true,
                                              mask, ref});
                SChan c; c.mask = std::move(mask); sch.push_back(std::move(c));
                chan_det.push_back((int)di);
                chan_ref.push_back(ref);
            } else if (v == Membership::LOGICAL) {
                fam_add_(0, (int)di, detectors[di], reads);   // J-a: a record extra (gauge detector)
            } else {
                fam_add_(1, (int)di, detectors[di], reads);   // J-a: a record extra (ANTI detector)
            }
        }
        // Task 2b.1: the empty gate relaxes — a barrier of only Born DECISIONs is valid (zero
        // syndrome = trivial decode). Fatal ONLY when there is no channel of ANY kind; when
        // DECISIONs are declared they are classified below and re-checked there. Decision-free
        // circuits keep the exact pre-2b.1 behavior (sch.empty() ⇒ error here).
        // V2-T4: deterministic (IN_GROUP) observables ARE record channels, compiled exactly
        // like detector channels and appended AFTER them (channel layout in the header). The
        // ch/diag benches' logical observables remain the refusal case (chan_obs empty there,
        // so the bench's byte-equality gate sees an unchanged channel set).
        for (const auto& kv : observables) {
            std::vector<uint64_t> mask;
            uint8_t ref = 0;
            const auto ov = classify(kv.second, mask, ref);
            if (ov == Membership::IN_GROUP) {
                rec_cols_.push_back(RecColumn{record_product(kv.second), rec_order_(1, kv.first), true,
                                              mask, ref});
                SChan c; c.mask = std::move(mask); sch.push_back(std::move(c));
                chan_obs.push_back(kv.first);
                chan_ref.push_back(ref);
            } else if (ov == Membership::LOGICAL && born_idx < 0) {
                // V3: FIRST LOGICAL observable on a diagonal table → Born-weighted channel
                // (spec guards: a second logical observable falls to refusal below — the
                // biased coins would be conditionally independent, ⟨W₁W₂⟩ wrong; ANTI-class
                // observables — e.g. declared-byproduct-frame readouts whose raw record
                // product anticommutes with certified generators, code_switching — stay
                // refused: they need the frame fold, a recorded V3 open item).
                born_idx = kv.first;
                born_W = record_product(kv.second);
            } else {                                        // J-a: a second LOGICAL / an ANTI observable
                fam_add_(2, kv.first, kv.second, reads);
            }
        }
        if (born_idx >= 0) born_obs = true;
        // ── Task 2b.1: DECISION classification ────────────────────────────────────────────
        // Each declared DECISION(k) parity is classified by the SAME G.reduce code path.
        //   IN_GROUP → deterministic decision: a σ-mask channel appended to `sch` (bit =
        //              ref ⊕ mask·σ); dec_chan_* map that channel back to decision index k.
        //   LOGICAL / ANTI → Born-measurement channel: store (k, W_dec); Born-sampled per shot.
        // (A DECISION may be EITHER class — a deterministic decision is a trivial coin; unlike a
        // DETECTOR, a nondeterministic DECISION is ALLOWED. Never inferred, always verified.)
        for (const auto& kv : opt.decisions) {
            const int k = kv.first;
            if (k + 1 > num_dec_) num_dec_ = k + 1;
            std::vector<uint64_t> mask;
            uint8_t ref = 0;
            const auto dv = classify(kv.second, mask, ref);
            if (dv == Membership::IN_GROUP) {
                rec_cols_.push_back(RecColumn{record_product(kv.second), rec_order_(2, k), true, mask, ref});
                SChan c; c.mask = std::move(mask); sch.push_back(std::move(c));
                dec_chan_slot_.push_back((int)sch.size() - 1);
                dec_chan_idx_.push_back(k);
                dec_chan_ref_.push_back(ref);
                // FIX: keep chan_ref.size() == sch.size() (nchan). Without this,
                // chan_ref is empty when there are only det-decision channels (no
                // regular detectors/observables), and the readout-flip layer's loop
                // `chan_ref[c] = ...` over c < nchan causes an OOB null-deref (segfault).
                chan_ref.push_back(ref);
                // Store records for the readout-flip layer (rf-mask + rec_invert accounting).
                dec_chan_recs_.push_back(kv.second);
            } else {
                // LOGICAL or ANTI: Born channel (the record operator, measured on the state).
                born_dec.push_back({k, record_product(kv.second)});
                // Deterministic record invert (Stim `!` + the H-elimination gadget's folded sign):
                // the recorded decision bit = measure_pauli(W_dec) ⊕ ⊕_{r∈recs} rec_invert[r].
                uint8_t inv = 0;
                for (int r : kv.second)
                    if (r >= 0 && r < (int)opt.rec_invert.size()) inv ^= (opt.rec_invert[(size_t)r] & 1);
                born_dec_inv.push_back(inv);
                born_dec_recs.push_back(kv.second);
            }
        }
        has_decisions_ = !opt.decisions.empty();
        // S2b: the extras' columns (after every channel; storage order Born observable, Born decisions).
        if (born_idx >= 0)
            rec_cols_.push_back(RecColumn{born_W, rec_order_(1, born_idx), false, {}, 0});
        for (const auto& bd : born_dec)
            rec_cols_.push_back(RecColumn{bd.second, rec_order_(2, bd.first), false, {}, 0});
        if (!famx_.empty()) fam_cols_();                     // J-a: the family extras (FAM samplers only)
        has_born_dec_ = !born_dec.empty();
        dec_scratch_.assign((size_t)(num_dec_ > 0 ? num_dec_ : 0), 0);
        {   // S7: an event (logging INFO on xtim.twirl, the Python layer), not stderr
            char msg[256];
            snprintf(msg, sizeof msg, "[twirl_records] circuit channels: %zu deterministic (+%zu observable), "
                                      "0 gauge, 0 anti, 0 refused observables, %zu born-decisions "   // J-b2 J6 (Q2): literal zeros
                                      "(+%zu det-decisions)",
                     chan_det.size(), chan_obs.size(), born_dec.size(), dec_chan_idx_.size());
            diag_.push_back({TwirlRecordSampler::DiagEvent::CHANNELS, "", (long)chan_det.size(), 0, msg});
        }
    }
    nchan = (int)sch.size();
    CW = (nchan + 63) / 64;                     // channel-space word count
    run_fn_ = pick_run_();                      // J1: the run_w instance for the final CW
    // S2b: the record family's columns. Synthetic-channel mode (bench only) has σ-mask channels and
    // no record operators: give each the product of its generators (raw = mask·σ, sign 0).
    if (!circ_chan) {
        rec_cols_.clear();
        for (int c = 0; c < nchan; ++c) {
            Pauli Mc(n);
            for (int w = 0; w < GW && w < (int)sch[(size_t)c].mask.size(); ++w)
                for (uint64_t word = sch[(size_t)c].mask[(size_t)w]; word; word &= word - 1)
                    Mc = Pauli::multiply(Mc, gens[(size_t)(w * 64 + __builtin_ctzll(word))]);
            rec_cols_.push_back(RecColumn{Mc, rec_order_(0, c), true, sch[(size_t)c].mask, 0});
        }
    }
    nextra_ = (int)rec_cols_.size() - nchan;
    if (nextra_ < 0)
        throw std::logic_error("twirl_records: record columns fewer than the channels (engine bug)");
    for (int c = 0; c < nchan; ++c)
        if (!rec_cols_[(size_t)c].channel || rec_cols_[(size_t)c].mask != sch[(size_t)c].mask)
            throw std::logic_error("twirl_records: record column / channel slot mismatch (engine bug)");
    XW = std::max(1, (nextra_ + 63) / 64);
    xbits_.assign((size_t)XW, 0);
    refmask_ch_.assign((size_t)std::max(CW, 1), 0);
    for (int c = 0; c < nchan; ++c)
        if (rec_cols_[(size_t)c].ref & 1) refmask_ch_[(size_t)(c >> 6)] |= 1ULL << (c & 63);
    // r7.1: per-wire record-column rows (the transpose of the record operators): qcol_x_[q] = the
    // columns whose operator has Z on wire q (a prefix X on q flips them), qcol_z_[q] = the columns
    // with X on q. t(P) = ⊕_{q∈supp P.x} qcol_x_[q] ⊕ ⊕_{q∈supp P.z} qcol_z_[q] (record_values).
    QCW_ = std::max(1, ((int)rec_cols_.size() + 63) / 64);
    qcol_x_.assign((size_t)n * (size_t)QCW_, 0);
    qcol_z_.assign((size_t)n * (size_t)QCW_, 0);
    for (int c = 0; c < (int)rec_cols_.size(); ++c) {
        const Pauli& M = rec_cols_[(size_t)c].M;
        for (int q = 0; q < n && q < M.n; ++q) {
            if (M.zbit(q)) qcol_x_[(size_t)q * QCW_ + (size_t)(c >> 6)] |= 1ULL << (c & 63);
            if (M.xbit(q)) qcol_z_[(size_t)q * QCW_ + (size_t)(c >> 6)] |= 1ULL << (c & 63);
        }
    }
    obs_col_ = (born_idx >= 0) ? 0 : -1;
    dec_col_.clear();
    for (size_t j = 0; j < born_dec.size(); ++j) dec_col_.push_back((born_idx >= 0 ? 1 : 0) + (int)j);
    // S0(f): run-width channel-row storage (no channel cap); ≥ 1 word so .data() is never null.
    bits_.assign((size_t)std::max(CW, 1), 0);
    oracle_abd_.assign((size_t)std::max(CW, 1), 0);
    oracle_nfd_.assign((size_t)std::max(CW, 1), 0);
    oracle_lin_.assign((size_t)std::max(CW, 1), 0);
    sem_raw_bits_.assign((size_t)std::max(CW, 1), 0);
    // ── Readout-flip layer build (circuit-channel mode). Assign a slot to each noisy record;
    // per channel record the noisy slots it contains; fold deterministic inverts into refs. ──
    if (circ_chan && !opt.rec_flip.empty()) {
        std::vector<int> rec_slot((size_t)opt.rec_flip.size(), -1);
        auto slot_of = [&](int rec) -> int {
            if (rec < 0 || rec >= (int)opt.rec_flip.size()) return -1;
            if (opt.rec_flip[(size_t)rec] <= 0.0) return -1;
            if (rec_slot[(size_t)rec] < 0) {
                rec_slot[(size_t)rec] = nrf++;
                rf_prob.push_back(opt.rec_flip[(size_t)rec] * factor);
            }
            return rec_slot[(size_t)rec];
        };
        auto inv_of = [&](int rec) -> int {
            return (rec >= 0 && rec < (int)opt.rec_invert.size()) ? (opt.rec_invert[(size_t)rec] & 1) : 0;
        };
        // detector channels [0, chan_det.size()), then observable channels
        std::vector<std::vector<int>> chan_records((size_t)nchan);
        for (size_t c = 0; c < chan_det.size(); ++c)
            chan_records[c] = detectors[(size_t)chan_det[c]];
        for (size_t k = 0; k < chan_obs.size(); ++k)
            for (const auto& kv : observables)
                if (kv.first == chan_obs[k]) { chan_records[chan_det.size() + k] = kv.second; break; }
        // Task 2b.1 FIX: fill chan_records for IN_GROUP decision channels so that:
        //   (a) slot_of() discovers their noisy records for rf-mask accounting, and
        //   (b) rec_invert is properly folded into chan_ref[c] (and synced to dec_chan_ref_[j]).
        for (size_t j = 0; j < dec_chan_slot_.size(); ++j)
            chan_records[(size_t)dec_chan_slot_[j]] = dec_chan_recs_[j];
        // first pass: discover all noisy slots (channels + born observable)
        for (int c = 0; c < nchan; ++c) for (int r : chan_records[(size_t)c]) slot_of(r);
        std::vector<int> born_records;
        if (born_idx >= 0)
            for (const auto& kv : observables)
                if (kv.first == born_idx) { born_records = kv.second; break; }
        for (int r : born_records) slot_of(r);
        // Task 2b.1: discover the Born decisions' noisy record slots (M(p) readout flips).
        for (const auto& recs : born_dec_recs) for (int r : recs) slot_of(r);
        for (const FamX& fx : famx_) for (int r : fx.recs) slot_of(r);   // J-a: the family extras' slots
        RFW = (nrf + 63) / 64;
        rf_coins.assign((size_t)RFW, 0);
        chan_rf_mask.assign((size_t)nchan, std::vector<uint64_t>((size_t)RFW, 0));
        for (int c = 0; c < nchan; ++c) {
            int inv = 0;
            for (int r : chan_records[(size_t)c]) {
                const int sl = slot_of(r);
                if (sl >= 0) chan_rf_mask[(size_t)c][sl >> 6] |= 1ull << (sl & 63);
                inv ^= inv_of(r);
            }
            chan_ref[(size_t)c] = (uint8_t)(chan_ref[(size_t)c] ^ (inv & 1));  // deterministic invert
        }
        // Task 2b.1 FIX: sync dec_chan_ref_[j] from chan_ref[c] now that rec_invert has been
        // folded in. emit_decisions() reads dec_chan_ref_[j] for the decision reference bit —
        // without this sync, rec_invert (`!` in Stim) on a DECISION(k) record would be ignored.
        for (size_t j = 0; j < dec_chan_slot_.size(); ++j)
            dec_chan_ref_[j] = chan_ref[(size_t)dec_chan_slot_[j]];
        if (born_idx >= 0) {
            born_rf_mask.assign((size_t)RFW, 0);
            int inv = 0;
            for (int r : born_records) {
                const int sl = slot_of(r);
                if (sl >= 0) born_rf_mask[sl >> 6] |= 1ull << (sl & 63);
                inv ^= inv_of(r);
            }
            born_inv_ = (uint8_t)(inv & 1);      // the record invert, XORed at every emission (audit I/J)
            if (circ_chan && opt.input_pauli_qubits.empty()) {
                // Twirl normalizes coherentize_all: feedback is IN the bare state (no relabel plan).
                const std::vector<uint8_t> R =
                    noiseless_reference_records(bare, deferred, reads, opt.rec_invert, nullptr);
                int par = 0;
                for (int r : born_records) par ^= R[(size_t)r];
                born_ref_ = par & 1;
            }
        }
        // Task 2b.1: per-born-decision readout-flip mask (the deterministic invert is already in
        // born_dec_inv from opt.rec_invert). emit_decisions XORs the parity of the shot's rf coins
        // over this mask into the decision bit — inert when no record carries an M(p) probability.
        if (!famx_.empty()) {                    // J-a: the family extras' readout-flip slot masks
            famx_rf_.assign(famx_.size() * (size_t)RFW, 0);
            for (size_t i = 0; i < famx_.size(); ++i)
                for (int r : famx_[i].recs) {
                    const int sl = slot_of(r);
                    if (sl >= 0) famx_rf_[i * (size_t)RFW + (size_t)(sl >> 6)] ^= 1ull << (sl & 63);
                }
        }
        if (!born_dec.empty()) {
            born_dec_rf_mask.assign(born_dec.size(), std::vector<uint64_t>((size_t)RFW, 0));
            for (size_t j = 0; j < born_dec.size(); ++j)
                for (int r : born_dec_recs[j]) {
                    const int sl = slot_of(r);
                    if (sl >= 0) born_dec_rf_mask[j][(size_t)(sl >> 6)] |= 1ull << (sl & 63);
                }
        }
        // FIX B: precompute fixed-point thresholds rf_thresh[sl] = (uint64_t)(p_sl * 2^64).
        // p_sl is in (0,1] (slot_of only creates slots for p>0).  The cast saturates correctly:
        // p_sl=1.0 gives (uint64_t)(1.0 * 2^64) = 0 due to wrap, but p_sl=1 is only possible
        // in test circuits; real circuits have p<<1.  For safety we clamp: if p_sl >= 1.0 set
        // thresh = UINT64_MAX (always fire).
        rf_thresh.resize((size_t)nrf);
        for (int sl = 0; sl < nrf; ++sl) {
            const double p = rf_prob[(size_t)sl];
            rf_thresh[(size_t)sl] = (p >= 1.0) ? UINT64_MAX : (uint64_t)(p * 18446744073709551616.0); // 2^64
        }
        // FIX A: build transposed column layout rf_col[sl] = CW-word bitvector of channels
        // that have slot sl in their rf mask.  Also build rf_obs_bit[sl] from born_rf_mask.
        // rf_col_data_ is a flat nrf*CW allocation; rf_col[sl] points into it.
        rf_col_data_.assign(std::max<size_t>((size_t)nrf * (size_t)CW, 1), 0);   // J-b1 F: >= 1 (CW = 0: &v[0] defined)
        rf_col.resize((size_t)nrf);
        for (int sl = 0; sl < nrf; ++sl)
            rf_col[(size_t)sl] = rf_col_data_.data() + (size_t)sl * (size_t)CW;
        // Transpose chan_rf_mask -> rf_col: for each channel c and each set bit sl in its mask,
        // set bit c in rf_col[sl].
        for (int c = 0; c < nchan; ++c) {
            for (int w = 0; w < RFW; ++w) {
                uint64_t word = chan_rf_mask[(size_t)c][(size_t)w];
                while (word) {
                    const int bit = __builtin_ctzll(word);
                    word &= word - 1;
                    const int sl = w * 64 + bit;
                    rf_col[(size_t)sl][c >> 6] |= 1ull << (c & 63);
                }
            }
        }
        // Build rf_obs_bit[sl] from born_rf_mask (1 if slot sl is set in the born obs mask).
        rf_obs_bit.assign((size_t)nrf, 0);
        if (!born_rf_mask.empty()) {
            for (int sl = 0; sl < nrf; ++sl)
                rf_obs_bit[(size_t)sl] = (uint8_t)((born_rf_mask[sl >> 6] >> (sl & 63)) & 1);
        }
    }
    // ── NOISELESS-REFERENCE SEMANTICS (P0, adjudicated 2026-07-24, option (b)) ──────
    // Deterministic DETECTOR/OBSERVABLE channels are REFERENCE-RELATIVE — exactly Stim's
    // and the production sampler's semantics (sampler.cpp refdet/refobs: "the emitted bit
    // is parity(shot) XOR parity(noiseless reference)").  The engine's internal channel bit
    // is already the detection EVENT relative to THIS COMPILE'S BARE STATE (mask·σ ⊕ readout
    // coins), and the reporting layer (xtim_py sinks, framed_bench) XORs chan_ref into it.
    //
    // Stim DEFINES the reference as "the run with every noise channel frozen off".  That
    // definition splits into exactly two cases here, and BOTH are exact — no text
    // inspection, no propagation, no heuristic:
    //
    //   • NO INPUT PORT — a self-contained circuit (stage 0, any port-free consumer stage,
    //     and every existing pin/golden/bench).  The engine already HOLDS the noise-free
    //     run: the bare state IS it.  So the reference parity equals the bare parity, the
    //     internal bit is ALREADY the event, and  chan_ref = 0  EXACTLY.
    //
    //     This replaces (2026-07-24) the earlier attempt to reach the same number by
    //     conjugating the text's own deterministic X/Y/Z gates to circuit end with
    //     propagate_atom and subtracting their induced flip.  That was an APPROXIMATION with
    //     four documented failure modes (a Pauli leaving the Pauli class in propagation; a
    //     flip composed from non-Pauli text such as S;S = Z; a flip absorbed by
    //     eliminate_hadamards into a measurement invert; a Pauli in a PRODUCER stage before
    //     OUTPUT_QUBITS) — in each of which it silently fell back to the raw sign.  Zeroing
    //     is exact wherever the subtraction was merely trying to be.
    //
    //   • CARRIED INPUT PORT — a consumer stage compiled from_state.  The ideal (noise-free)
    //     CARRIED state is a PROTOCOL-level fact that no function of this compile's text can
    //     recover.  The SAME composed text
    //         X q (carried-state prep) ; INPUT_QUBITS ; X q (body) ; M q ; DETECTOR
    //     must report 0 when the carried |1⟩ is a Born branch (teleport-style feedback) and
    //     1 when it is an X_ERROR deviation — the two protocols are byte-identical here and
    //     demand OPPOSITE references, so no local rule can serve both.  The engine therefore
    //     does not guess: it reports the ABSOLUTE parity (chan_ref = the raw classify reduce
    //     sign, with rec_invert folded in above), and the layer that KNOWS the protocol —
    //     the noiseless-reference pass in xtim/feedback.py, which replays the stage chain at
    //     p_factor = 0 and XORs the resulting per-(stage, decision-pattern) reference — turns
    //     absolute parities into EVENTS before the decoder ever sees them.
    //
    //     CONTRACT: a from_state compile driven directly through the Python sampler API
    //     reports ABSOLUTE parity (tests/test_observable_reference.py
    //     ::test_carried_deviation_still_reported pins exactly this — a carried |1⟩ read in Z
    //     reports 1, and an X input frame corrects it to 0).
    //
    // DECISION channels are untouched by design: they are RAW recorded parity by contract
    // (they feed feedback/decoding) and occupy channel slots
    // [chan_det.size() + chan_obs.size(), nchan).
    if (circ_chan && opt.input_pauli_qubits.empty()) {
        const size_t ndo = chan_det.size() + chan_obs.size();
        for (size_t c = 0; c < ndo; ++c) chan_ref[c] = 0;
    }
    // DiagErrorSampler channels from the deferred noise stream (probs × factor); sides = the per-alternative
    // FiredPauli lists, flat-alt-id order == construction order. J-b1 F (F item 2): the ONE copy of the loop
    // (twirl_alternatives_, twirl_sampler_fam.cpp), built before fam_setup_, which reuses it. Its switch is
    // exhaustive over NoiseChannel and compiled with -Werror=switch (the deleted TS:689 `default:`, F item 1).
    std::vector<ErrorChannel> chans;
    twirl_alternatives_(deferred, factor, chans, sides);
    fam_setup_(deferred, reads, opt, chans);    // the family sampler (twirl_sampler_fam.cpp)
    if (setup_error) return;
    if (born_obs) {                             // acceptance test data (post-selected stats)
        det_refw.assign((size_t)CW, 0);
        det_maskw.assign((size_t)CW, 0);
        for (size_t c = 0; c < chan_det.size(); ++c) {
            det_maskw[c >> 6] |= 1ull << (c & 63);
            if (chan_ref[c]) det_refw[c >> 6] |= 1ull << (c & 63);
        }
    }
    // Per-circuit prefix qubit rows (CW words per qubit): bit c of pxrow[q] =
    // parity(zcol(q) & mask_c) (X-support contribution to plane1 lands on zcol), pzrow from xcol.
    pxrow.assign(std::max<size_t>((size_t)n * CW, 1), 0); pzrow.assign(std::max<size_t>((size_t)n * CW, 1), 0);   // J-b1 F: >= 1 (CW = 0)
    {
        std::vector<uint64_t> col(GW);
        for (int q = 0; q < n; ++q) {
            const uint64_t* zc = G.col(q, false);
            const uint64_t* xc = G.col(q, true);
            for (int c = 0; c < nchan; ++c) {
                col.assign(zc, zc + GW);
                if (par(col, sch[c].mask)) pxrow[(size_t)q * CW + (c >> 6)] |= (1ull << (c & 63));
                col.assign(xc, xc + GW);
                if (par(col, sch[c].mask)) pzrow[(size_t)q * CW + (c >> 6)] |= (1ull << (c & 63));
            }
        }
    }
    chtok = 0x2545F4914F6CDD1Dull;               // channel-set token
    for (const SChan& c : sch) for (uint64_t w : c.mask) { chtok ^= w; chtok *= 0x100000001B3ull; }

    smp.reset(new DiagErrorSampler(n, chans, seed, /*keyed=*/true));   // S0(b): per-channel keyed

    // ── Sparse per-alt atom extraction (Tier C substrate) ───────────────────────────────────
    // Per flat alternative: compose its (1–2 Pauli) fired list ONCE at setup via the exact
    // compose_fired, then store the sparse content. Per shot the residual is accumulated by the
    // SAME recurrence then() uses — r.a[j] = a1[j] + a2[j]·(1−2·v1[j]) + 2·(B2·v1)[j], v ^= v2,
    // B ^= B2 — but sparsely over the atom's support with membership-flag dirty lists (the
    // through-zero dedup lesson), so a shot costs O(Σ atom support), not fired × O(n²) B copies.
    // γ is never tracked (discarded at the normal form). Self-checked EXACTLY against the
    // compose_fired chain on the first --selfcheck diag shots.
    alt_atoms.assign(sides.size(), AltAtom());
    // ── PPR alts (has_general tables): an alt touching a PPR-retry atom cannot go through
    // the diagonal compose (c_prop is invalid there); its shots take the term route (kind 2).
    // J-b2 J6b: the dressing's per-alt axes are deleted with the dressing.
    alt_is_ppr.assign(sides.size(), 0);
    {
        std::set<std::pair<int, int>> ps;
        // S8-1' (plan r2.3 option 5-a): each alternative's sparse content (av, czp, vq) is composed SPARSELY from
        // its atoms' own sparse contents (each atom's (a, B pairs, v) extracted once), by the recurrence of
        // DiagPauliClifford::then — r.a[j] = a1[j] + a2[j]·(1 − 2·v1[j]) + 2·(B2·v1)[j] (mod 4), v ^= v2, B ^= B2 —
        // over the atoms' supports only, instead of a dense identity(n).then(...) chain per alternative. γ is never
        // read. The result is the same (av ascending q, a ≠ 0; czp the odd-multiplicity pairs, ascending; vq
        // ascending) as compose_fired + finalize, which the test API's family_alt_atoms_oracle() recomputes.
        struct SpAtom { std::vector<std::pair<int, uint8_t>> a; std::vector<std::pair<int, int>> b; std::vector<int> v; };
        std::unordered_map<const PropAtom*, SpAtom> sp;
        auto sparse_of = [&](const PropAtom& A) -> const SpAtom& {
            auto it = sp.find(&A);
            if (it != sp.end()) return it->second;
            SpAtom& S = sp[&A];
            const DiagPauliClifford& D = A.c_prop;
            for (int q = 0; q < n; ++q) if (D.a[(size_t)q] & 3) S.a.push_back({q, (uint8_t)(D.a[(size_t)q] & 3)});
            for (int q = 0; q < n; ++q) if (D.v[(size_t)q]) S.v.push_back(q);
            const int bw = D.B.words();
            for (int j = 0; j < n && bw; ++j) {
                const uint64_t* row = D.B.row(j);
                for (int w = (j + 1) >> 6; w < bw; ++w)
                    for (uint64_t word = row[w]; word; word &= word - 1) {
                        const int l = w * 64 + __builtin_ctzll(word);
                        if (l > j) S.b.push_back({j, l});
                    }
            }
            return S;
        };
        std::vector<int> acc_a((size_t)n, 0);
        std::vector<uint8_t> acc_v((size_t)n, 0), bv((size_t)n, 0), touched((size_t)n, 0);
        std::vector<int> tq, bq;
        std::vector<std::pair<int, int>> pairs;
        std::vector<const PropAtom*> seq;
        for (size_t ai = 0; ai < sides.size(); ++ai) {
            const std::vector<FiredPauli>& fr = sides[ai];
            bool isppr = false;
            if (ppr_tables) {
                for (const FiredPauli& f : fr) {
                    const auto [loc, qi] = table.find_slot(f.location_index, f.qubit, "twirl_records");
                    if ((f.pauli == PauliBasis::X || f.pauli == PauliBasis::Y) && loc->atom(qi, PauliBasis::X, "twirl_records").ppr) isppr = true;
                    if ((f.pauli == PauliBasis::Z || f.pauli == PauliBasis::Y) && loc->atom(qi, PauliBasis::Z, "twirl_records").ppr) isppr = true;
                }
            }
            alt_is_ppr[ai] = isppr ? 1 : 0;
            if (isppr) continue;                       // AltAtom stays empty (never composed diagonally)
            seq.clear();                               // the atoms in compose_fired's order (stream; Y = X then Z)
            for (const FiredPauli& f : fr) {
                const auto [loc, qi] = table.find_slot(f.location_index, f.qubit, "compose_fired");
                if (f.pauli == PauliBasis::X || f.pauli == PauliBasis::Y) seq.push_back(&loc->atom(qi, PauliBasis::X, "compose_fired"));
                if (f.pauli == PauliBasis::Z || f.pauli == PauliBasis::Y) seq.push_back(&loc->atom(qi, PauliBasis::Z, "compose_fired"));
            }
            tq.clear(); pairs.clear();
            auto touch = [&](int q) { if (!touched[(size_t)q]) { touched[(size_t)q] = 1; tq.push_back(q); } };
            for (const PropAtom* pa : seq) {
                const SpAtom& U2 = sparse_of(*pa);
                for (const auto& e : U2.a) {           // a2[j]·(1 − 2·v1[j])
                    touch(e.first);
                    acc_a[(size_t)e.first] += acc_v[(size_t)e.first] ? -(int)e.second : (int)e.second;
                }
                bq.clear();                            // 2·(B2·v1)[j]
                for (const auto& pr : U2.b) {
                    if (acc_v[(size_t)pr.second]) { if (!bv[(size_t)pr.first]) bq.push_back(pr.first); bv[(size_t)pr.first] ^= 1; }
                    if (acc_v[(size_t)pr.first]) { if (!bv[(size_t)pr.second]) bq.push_back(pr.second); bv[(size_t)pr.second] ^= 1; }
                }
                for (int q : bq)                       // (bq may repeat a wire: it is consumed once)
                    if (bv[(size_t)q]) {
                        bv[(size_t)q] = 0;
#ifndef XTIM_MUTATE_ALT_SPARSE_CARRY
                        touch(q);
                        acc_a[(size_t)q] += 2;
#endif
                    }
                for (int q : U2.v) { touch(q); acc_v[(size_t)q] ^= 1; }
                pairs.insert(pairs.end(), U2.b.begin(), U2.b.end());
            }
            std::sort(tq.begin(), tq.end());
            AltAtom& A = alt_atoms[ai];
            for (int q : tq) {
                const int av = ((acc_a[(size_t)q] % 4) + 4) % 4;
                if (av) A.av.push_back({q, (uint8_t)av});
            }
            for (int q : tq) if (acc_v[(size_t)q]) A.vq.push_back(q);
            for (int q : tq) { acc_a[(size_t)q] = 0; acc_v[(size_t)q] = 0; touched[(size_t)q] = 0; }
            std::sort(pairs.begin(), pairs.end());
            for (size_t i = 0; i < pairs.size();) {    // B = ⊕ B_atoms: the pairs of odd multiplicity
                size_t j = i;
                while (j < pairs.size() && pairs[j] == pairs[i]) ++j;
                if ((j - i) & 1) { ps.insert(pairs[i]); A.czp.push_back(pairs[i]); }
                i = j;
            }
        }
        upairs.assign(ps.begin(), ps.end());
        for (size_t i = 0; i < upairs.size(); ++i) pidx[upairs[i]] = (int)i;
        for (size_t ai = 0; ai < alt_atoms.size(); ++ai) {
            AltAtom& A = alt_atoms[ai];
            A.czidx.reserve(A.czp.size());
            for (const auto& pr : A.czp) A.czidx.push_back(pidx[pr]);
        }
    }
    PW = (int)(upairs.size() + 63) / 64;
    NW = (n + 63) / 64;
    a_acc.assign((size_t)n, 0); v_acc.assign((size_t)n, 0);
    in_a.assign((size_t)n, 0); in_v.assign((size_t)n, 0);
    czw.assign((size_t)PW, 0);
    in_w.assign((size_t)PW, 0);
    amask_acc.assign((size_t)NW, 0);   // maintained packed a&1 (the memo key part)
    // Product-wire word mask (canon-touch test in O(NW) instead of an O(n) byte scan).
    prodmask.assign((size_t)NW, 0);
    {
        const CertifiedGroupPlanes::ZAxisRREF& Z = G.z_axis_rref();
        for (int q = 0; q < n; ++q)
            if (Z.prod_sign[q] != 0) prodmask[q >> 6] |= (1ULL << (q & 63));
    }
    ZR = &G.z_axis_rref();

    // ── ABELIAN FAST PATH substrate (2026-07-15, "why isn't warm just sampling?"). ─────────
    // The memo KEY's mod-2 content is an XOR-ADDITIVE function of the fired set (every
    // composition cross-term is even), so per shot: key = XOR of per-alt precomputed masks —
    // NO content materialisation on a warm hit. The composed prefix decomposes as
    //   x-part: v_tot = ⊕_j v_j                                             (abelian)
    //   z-part: z_tot = ⊕_j z_j ⊕ ⊕_{k<j}[ (s_j∧s_k) ⊕ (s_j∧v_k) ⊕ (B_j·v_k) ]
    // (the pairwise S-carry is EXACT for all coincidence counts: C(c,2) mod 2 == bit1(c mod 4)),
    // and channel bits are pxrow/pzrow XORs — LINEAR in these masks, so per-alt channel rows
    // XOR too; only the ordered-PAIR term needs per-shot work (~fired²/2 small word ops).
    // Strip interaction: the canonical key is (smask & ~prodmask, kept pairs) and the −Z-partner
    // zflips ride per-occurrence XOR (a cancelling pair cancels its flips) → per-alt zrow
    // includes the alt's own stripped-pair flips; product-wire z bits are inert (pzrow ≡ 0
    // there). ORACLE: selfcheck window compares abelian key == materialised+canon key and
    // abelian bits == σ-path bits EXACTLY, or aborts.
    alt_masks.assign(alt_atoms.size(), AltMasks());
    for (size_t ai = 0; ai < alt_atoms.size(); ++ai) {
        const AltAtom& A = alt_atoms[ai];
        AltMasks& M = alt_masks[ai];
        M.smask.assign((size_t)NW, 0); M.vmask.assign((size_t)NW, 0);
        M.kmask.assign((size_t)NW, 0); M.czk.assign((size_t)PW, 0);
        M.zxrow.assign((size_t)CW, 0); M.zzrow.assign((size_t)CW, 0);
        for (const auto& qa : A.av) {
            if (qa.second & 1) M.smask[qa.first >> 6] |= (1ULL << (qa.first & 63));
            if (qa.second & 2)
                for (int w2 = 0; w2 < CW; ++w2) M.zzrow[w2] ^= pzrow[(size_t)qa.first * CW + w2];
        }
        for (int w2 = 0; w2 < NW; ++w2) M.kmask[w2] = M.smask[w2] & ~prodmask[w2];
        for (int q : A.vq) {
            M.vmask[q >> 6] |= (1ULL << (q & 63));
            for (int w2 = 0; w2 < CW; ++w2) M.zxrow[w2] ^= pxrow[(size_t)q * CW + w2];
        }
        for (size_t i = 0; i < A.czp.size(); ++i) {
            const int a2 = A.czp[i].first, b2 = A.czp[i].second;
            const int s1 = ZR->prod_sign[a2], s2 = ZR->prod_sign[b2];
            if (s1 == 0 && s2 == 0) {
                const int pi2 = A.czidx[i];
                M.czk[pi2 >> 6] ^= (1ULL << (pi2 & 63));
            } else if (s1 != 0 && s2 == 0) {
                if (s1 < 0) for (int w2 = 0; w2 < CW; ++w2) M.zzrow[w2] ^= pzrow[(size_t)b2 * CW + w2];
            } else if (s1 == 0 && s2 != 0) {
                if (s2 < 0) for (int w2 = 0; w2 < CW; ++w2) M.zzrow[w2] ^= pzrow[(size_t)a2 * CW + w2];
            }                                              // both stripped: global phase
        }
    }
    ab_kmask.assign((size_t)NW, 0); ab_czk.assign((size_t)PW, 0);
    ab_T.assign((size_t)NW, 0);                       // pair-term scratch
    // ── The plan key for the abelian fast path AND the plan cache (S0(c)): (kmask NW ‖ czk PW)
    //    words over upairs indexing — no pair-list expansion, no pair-vector hashing on the hit
    //    path. The cache stores it once per plan (inline in the permanent header); its flat
    //    (hash, header) table is the index every route probes. A TWPL key with a pair outside
    //    upairs is unreachable under this alt set and is skipped at load.
    KW = NW + PW;
    ab_kw.assign((size_t)KW, 0);                      // per-shot key scratch
    ab_S.assign((size_t)NW, 0); ab_V.assign((size_t)NW, 0);     // running pair-term masks
    cache->set_universe(n, upairs, G.identity_token());
    upair_keep_.assign((size_t)PW, 0);
    for (size_t i = 0; i < upairs.size(); ++i)
        if (ZR->prod_sign[upairs[i].first] == 0 && ZR->prod_sign[upairs[i].second] == 0)
            upair_keep_[i >> 6] |= 1ULL << (i & 63);
    plan_budget_ = opt.plan_payload_budget;
    budget_pub_.store(plan_budget_, std::memory_order_relaxed);
    // ── Flat per-alt blob (2026-07-15): the AoS AltMasks walk was DRAM-latency-bound (~6
    //    scattered heap vectors per fired alt). One contiguous block per alt + prefetch makes
    //    the warm path stream. Layout (uint64 words):
    //      [0]            header: nc | np<<32  (czk entry count, czp pair count)
    //      [1]            htok: GF(2)-linear hash token of this alt's key contribution
    //      [2 .. +NW]     kmask (dense)        [.. +NW] smask   [.. +NW] vmask
    //      [.. +CW]       zrow = zxrow ^ zzrow (emission row; only ever used XORed together)
    //      [.. +2*nc]     czk entries (word-idx, value)
    //      [.. +np]       czp pairs (x | y<<32)
    alt_off.assign(alt_masks.size(), 0);
    {
        size_t tot = 0;
        for (size_t ai = 0; ai < alt_masks.size(); ++ai) {
            size_t nc = 0;
            for (int w = 0; w < PW; ++w) if (alt_masks[ai].czk[(size_t)w]) ++nc;
            tot += 2 + 3 * (size_t)NW + (size_t)CW + 2 * nc + alt_atoms[ai].czp.size();
        }
        alt_blob.reserve(tot);
        std::vector<uint64_t> contrib((size_t)KW);
        for (size_t ai = 0; ai < alt_masks.size(); ++ai) {
            const AltMasks& M = alt_masks[ai];
            const AltAtom& A = alt_atoms[ai];
            alt_off[ai] = (uint32_t)alt_blob.size();
            size_t nc = 0;
            for (int w = 0; w < PW; ++w) if (M.czk[(size_t)w]) ++nc;
            alt_blob.push_back((uint64_t)nc | ((uint64_t)A.czp.size() << 32));
            for (int w = 0; w < NW; ++w) contrib[(size_t)w] = M.kmask[(size_t)w];
            for (int w = 0; w < PW; ++w) contrib[(size_t)(NW + w)] = M.czk[(size_t)w];
            alt_blob.push_back(lhash(contrib.data()) ^ 0xcbf29ce484222325ull);  // htok (C removed)
            for (int w = 0; w < NW; ++w) alt_blob.push_back(M.kmask[(size_t)w]);
            for (int w = 0; w < NW; ++w) alt_blob.push_back(M.smask[(size_t)w]);
            for (int w = 0; w < NW; ++w) alt_blob.push_back(M.vmask[(size_t)w]);
            for (int w = 0; w < CW; ++w) alt_blob.push_back(M.zxrow[(size_t)w] ^ M.zzrow[(size_t)w]);
            for (int w = 0; w < PW; ++w)
                if (M.czk[(size_t)w]) { alt_blob.push_back((uint64_t)w); alt_blob.push_back(M.czk[(size_t)w]); }
            for (size_t i = 0; i < A.czp.size(); ++i)
                alt_blob.push_back((uint64_t)(uint32_t)A.czp[i].first |
                                   ((uint64_t)(uint32_t)A.czp[i].second << 32));
        }
        if (g_twirl_alt_blob_test_hook) g_twirl_alt_blob_test_hook(alt_blob);   // TEST seam (ctest only)
        // One-time blob-vs-AoS verification (exact, every alt) — the blob is a pure re-encoding.
        for (size_t ai = 0; ai < alt_masks.size(); ++ai) {
            const AltMasks& M = alt_masks[ai];
            const AltAtom& A = alt_atoms[ai];
            const uint64_t* B = &alt_blob[alt_off[ai]];
            const uint32_t nc = (uint32_t)B[0], np = (uint32_t)(B[0] >> 32);
            for (int w = 0; w < NW; ++w) contrib[(size_t)w] = M.kmask[(size_t)w];
            for (int w = 0; w < PW; ++w) contrib[(size_t)(NW + w)] = M.czk[(size_t)w];
            if (B[1] != (lhash(contrib.data()) ^ 0xcbf29ce484222325ull)) {
                family_refusal_ = "twirl_records: BLOB VERIFY FAIL (htok) at alt " + std::to_string(ai) +
                                  ": the packed alt blob differs from its source masks (engine bug)";
                setup_error = 5; return;   // the bindings raise family_refusal_ (S7: no stderr copy)
            }
            const uint64_t* km = B + 2;
            const uint64_t* sm = km + NW;
            const uint64_t* vm = sm + NW;
            const uint64_t* zr = vm + NW;
            const uint64_t* ce = zr + CW;
            const uint64_t* pp = ce + 2 * nc;
            bool ok = np == A.czp.size();
            for (int w = 0; ok && w < NW; ++w)
                ok = km[w] == M.kmask[(size_t)w] && sm[w] == M.smask[(size_t)w] &&
                     vm[w] == M.vmask[(size_t)w];
            for (int w = 0; ok && w < CW; ++w)
                ok = zr[w] == (M.zxrow[(size_t)w] ^ M.zzrow[(size_t)w]);
            std::vector<uint64_t> cz2((size_t)PW, 0);
            for (uint32_t i = 0; i < nc; ++i) cz2[(size_t)ce[2 * i]] = ce[2 * i + 1];
            if (ok) ok = cz2 == M.czk;
            for (uint32_t i = 0; ok && i < np; ++i)
                ok = (uint32_t)pp[i] == (uint32_t)A.czp[i].first &&
                     (uint32_t)(pp[i] >> 32) == (uint32_t)A.czp[i].second;
            if (!ok) {
                family_refusal_ = "twirl_records: BLOB VERIFY FAIL at alt " + std::to_string(ai) +
                                  ": the packed alt blob differs from its source masks (engine bug)";
                setup_error = 5; return;
            }
        }
    }

    set_keys_(seed);   // S0(b): every per-shot stream keyed on (seed, shot, tag, ctr)
    // V3-T3 R2: automatic plan-cache path — <dir>/<fnv64(deferred-signature)>-<group-token>.twpl.
    // Noise-blind key (signature excludes noise values) + the group identity token; the TWPL
    // framing re-verifies the token on load, so stale/corrupt/skew files rebuild silently.
    if (disk_path.empty() && !opt.disk_auto_dir.empty()) {
        char key[64];
        snprintf(key, sizeof key, "/%016llx-%016llx.twpl",
                 (unsigned long long)deferred_signature_fnv(deferred, reads),
                 (unsigned long long)G.identity_token());
        disk_path = opt.disk_auto_dir + key;
    }
    if (!disk_path.empty()) {
        const bool ld = cache->load_file(disk_path, G.identity_token());
        diag_.push_back(       // S0d: INFO on logger xtim.twirl (Python layer), not stderr
            {TwirlRecordSampler::DiagEvent::DISK_LOAD, ld ? "LOADED" : "absent/rebuild",
                             (long)cache->size(), 0, disk_path});
    }
    chancnt.assign((size_t)nchan, 0);             // per-channel 1-bit counts
    nf.prefix = Pauli(n);
    // Task 4 / J-a N4b: the input Pauli frame's port (the term route composes the supplied frames).
    has_input_frame_ = !opt.input_pauli_qubits.empty();
    if (has_input_frame_) frame_port_qubits_ = opt.input_pauli_qubits;
}

}  // namespace qeccore

namespace qeccore {
// ── fixplan-coldstart (B) (J-b1): the members TwirlRecordSampler::Impl shares with the family TU, defined ONCE
// here (out of line; twirl_sampler_fam.cpp sees declarations only), so the linked bodies of the FAM hot path are
// this TU's — as the linker already chose — and stay put when J-b2 removes the legacy run_w. They stay `inline`
// (their in-class definitions were implicitly inline: the inliner's declared-inline limits are unchanged, so this TU's
// codegen does not move) and `used` (the out-of-line copy the family TU links is always emitted here; J-b2 J3 added
// the attribute, which the comment claimed and the code lacked: without the legacy run_w nothing in this TU calls them).
inline __attribute__((used)) void TwirlRecordSampler::Impl::payload_boundary_() {                    // between shots ONLY (no shot holds a plan)
        size_t tot = cache->resident_bytes();
        sampler_resident_pub_.store(tot, std::memory_order_relaxed);
        if (tot > payload_peak_) payload_peak_ = tot;
        // r6.4 eviction order: the readers' UNHELD tables go first (never a held one), then payloads
        if (plan_budget_ && reader_bytes_.load(std::memory_order_relaxed) &&
            tot + reader_bytes_.load(std::memory_order_relaxed) > plan_budget_) {
            std::lock_guard<std::mutex> lk(rec_memo_mu_);
            reader_trim_locked_(tot, nullptr);
        }
        // S8-4: the Born root memos are charged with the payloads and cleared FIRST under pressure (exact: a cleared
        // root is recomputed bit for bit); the payload eviction below then sees the payload bytes alone, as before.
        if (plan_budget_) {
            const size_t mb = born_memo_bytes_();
            if (mb && tot + mb > plan_budget_) born_memo_clear_();
        }
        // r8 (R6-2): readers never evict a sampler payload — the sampler evicts on ITS OWN bytes only
        // (the readers' bytes were trimmed above; a table a reader still holds stays, reported).
        if (!plan_budget_ || tot <= plan_budget_) return;
        if (tot - plan_budget_ > payload_overshoot_) payload_overshoot_ = tot - plan_budget_;
        while (tot > plan_budget_ && cache->evict_oldest())
            tot = cache->resident_bytes();
        sampler_resident_pub_.store(tot, std::memory_order_relaxed);
    }

inline __attribute__((used)) const uint8_t* TwirlRecordSampler::Impl::emit_decisions(const uint64_t* bits) {
        if (!has_decisions_) return nullptr;
        std::fill(dec_scratch_.begin(), dec_scratch_.end(), (uint8_t)0);
        for (size_t j = 0; j < dec_chan_slot_.size(); ++j) {
            const int c = dec_chan_slot_[j];
            const int b = (int)((bits[c >> 6] >> (c & 63)) & 1) ^ (int)dec_chan_ref_[j];
            dec_scratch_[(size_t)dec_chan_idx_[j]] = (uint8_t)b;
        }
        if (!has_born_dec_) return dec_scratch_.data();
        dec_raw_.assign(born_dec.size(), 0);
        for (size_t j = 0; j < born_dec.size(); ++j) {
            if (!rec_route_)
                throw std::logic_error("emit_decisions: a Born decision without record rows (engine bug)");
            const uint8_t raw = xbit_(dec_col_[j]) ? 1 : 0;
            dec_raw_[j] = raw;
            uint8_t bit = raw;
            bit ^= born_dec_inv[j];                      // deterministic record invert (Stim `!` etc.)
            if (!born_dec_rf_mask.empty() && !born_dec_rf_mask[j].empty()) {
                int par = 0;                             // M(p) readout-flip coins (drawn this shot)
                for (int w = 0; w < RFW; ++w)
                    par += __builtin_popcountll(rf_coins[(size_t)w] & born_dec_rf_mask[j][(size_t)w]);
                bit ^= (uint8_t)(par & 1);
            }
            dec_scratch_[(size_t)born_dec[j].first] = bit;
        }
        return dec_scratch_.data();
    }

inline __attribute__((used)) void TwirlRecordSampler::Impl::record_values_words_(const RecTable& T, const uint64_t* px, const uint64_t* pz, int pnw,
                              const uint8_t* uz, int n_uz, uint64_t* out) const {
        if (n_uz != T.rho + T.kappa)
            throw std::invalid_argument("record_outcomes: the record carries " + std::to_string(n_uz) +
                                        " outcomes, its plan has rho + kappa_rec = " +
                                        std::to_string(T.rho + T.kappa));
        record_rows_values_(T, uz, out);
        // t(P) by the per-wire column rows (r7.1): the same GF(2) sum as ⟨P, M_c⟩ per column, reordered.
        const int nw = std::min(pnw, (n + 63) / 64);
        if (T.CWX != QCW_)
            throw std::logic_error("record_values: table width " + std::to_string(T.CWX) +
                                   " differs from the column-row width " + std::to_string(QCW_) + " (engine bug)");
        const int W = QCW_;
#ifdef XTIM_MUTATE_R71_XROW_SKIP_FIRST
        bool r71_first = true;                        // mutation build only (R71)
#endif
        for (int w = 0; w < nw; ++w) {
            uint64_t xw = px[w];
            while (xw) {
                const int q = w * 64 + __builtin_ctzll(xw); xw &= xw - 1;
#ifdef XTIM_MUTATE_R71_XROW_SKIP_FIRST
                if (r71_first) { r71_first = false; continue; }   // R71: the prefix's first X row dropped
#endif
                const uint64_t* r = &qcol_x_[(size_t)q * QCW_];
                for (int k = 0; k < W; ++k) out[k] ^= r[k];
            }
            uint64_t zw = pz[w];
            while (zw) {
                const int q = w * 64 + __builtin_ctzll(zw); zw &= zw - 1;
                const uint64_t* r = &qcol_z_[(size_t)q * QCW_];
                for (int k = 0; k < W; ++k) out[k] ^= r[k];
            }
        }
    }

inline __attribute__((used)) DiagNormalForm TwirlRecordSampler::Impl::parse_plan_key(const uint8_t* plan_key, int n_plan) const {
        DiagNormalForm nf;
        nf.a.assign((size_t)n, 0);
        nf.prefix = Pauli(n);
        nf.diagonal_class = true;
        const int NW = (n + 63) / 64;
        if (plan_key && n_plan > 0) {
            int p = 0;
            auto rd_words = [&](std::vector<uint64_t>& dst) {
                for (int w = 0; w < NW; ++w) {
                    uint64_t v = 0;
                    for (int b = 0; b < 8; ++b)
                        if (p < n_plan) v |= ((uint64_t)plan_key[p++]) << (b * 8);
                    if (w < (int)dst.size()) dst[(size_t)w] = v;
                }
            };
            rd_words(nf.prefix.x);
            rd_words(nf.prefix.z);
            for (int q = 0; q < n; ++q)
                if (p < n_plan) nf.a[(size_t)q] = (uint8_t)(plan_key[p++] & 1);
            if (p < n_plan && plan_key[p] == 0xFF) ++p;   // separator
            while (p + 3 < n_plan) {
                int f = (int)plan_key[p] | ((int)plan_key[p + 1] << 8);
                int s = (int)plan_key[p + 2] | ((int)plan_key[p + 3] << 8);
                // T5 step 0 hardening (T3 review Minor): plan_structure accepts ARBITRARY caller
                // bytes (unlike materialize_shot, which only ever replays sampler-serialized
                // keys), so refuse an out-of-range cz wire before build_shot_law consumes it.
                if (f < 0 || s < 0 || f >= n || s >= n)
                    throw std::invalid_argument(
                        "plan_structure: cz pair (" + std::to_string(f) + ", " +
                        std::to_string(s) + ") out of range [0, " + std::to_string(n) +
                        ") — malformed plan key");
                nf.cz.emplace_back(f, s);
                p += 4;
            }
        }
        return nf;
    }

inline __attribute__((noinline, cold))  void TwirlRecordSampler::Impl::build_rec_cold_(const CachedPlan& plan) const {
        if (!plan.resident)
            throw std::logic_error("ensure_rec_: payload not resident (evicted inside a shot) (engine bug)");
        DiagNormalForm nf0;
        nf0.prefix = Pauli(n);
        nf0.a.assign((size_t)n, 0);
        nf0.diagonal_class = true;
        for (int q = 0; q < n; ++q) if (cache->key_a(plan, q)) nf0.a[(size_t)q] = 1;
        cache->key_cz(plan, nf0.cz);
        auto T = std::make_shared<RecTable>();
        build_rec_table_fam_(plan, nf0, *T);       // J-b1 P6 / J-b2 J7b: record-space B (the σ dual basis is gone)
        // r9: the emission block — the record rows split into channel words (internal convention: the base
        // row XORed with the classify signs) and extras words, laid out contiguously in emission order.
        {
            const int nr = 1 + T->rho + T->kappa, cw = std::max(CW, 1);
            std::vector<uint64_t>& E = plan.emit;
            E.assign((size_t)nr * (size_t)(cw + XW), 0);
            std::vector<uint64_t> ch, exw;
            auto put = [&](int r, const std::vector<uint64_t>& row, const uint64_t* chxor) {
                split_row_(row, ch, exw, chxor);
                std::copy(ch.begin(), ch.begin() + cw, E.begin() + (size_t)r * cw);
                std::copy(exw.begin(), exw.begin() + XW, E.begin() + (size_t)nr * cw + (size_t)r * XW);
            };
            put(0, T->base, refmask_ch_.data());
            for (int k = 0; k < T->rho; ++k) put(1 + k, T->coin_row[(size_t)k], nullptr);
            for (int j = 0; j < T->kappa; ++j) put(1 + T->rho + j, T->born_row[(size_t)j], nullptr);
            // r9 condition (a): the block and the readers' rows cannot drift — rejoin and compare exactly.
            auto rejoin_ok = [&](int r, const std::vector<uint64_t>& row, const uint64_t* chxor) {
                return rejoin_ok_(E, nr, cw, T->m, r, row, chxor);
            };
            bool ok = rejoin_ok(0, T->base, refmask_ch_.data());
            for (int k = 0; ok && k < T->rho; ++k) ok = rejoin_ok(1 + k, T->coin_row[(size_t)k], nullptr);
            for (int j = 0; ok && j < T->kappa; ++j) ok = rejoin_ok(1 + T->rho + j, T->born_row[(size_t)j], nullptr);
            if (!ok) throw std::logic_error("ensure_rec_: the emission block differs from the record table (engine bug)");
            plan.emit_rho = T->rho;
            plan.emit_kappa = T->kappa;
        }
        ++rec_tables_;
        rec_extras_ += T->extras;
        rec_ops_plane_ += T->ops_plane; rec_ops_pivot_ += T->ops_pivot_reduce;
        rec_ops_greduce_ += T->ops_greduce; rec_ops_echelon_ += T->ops_echelon;
        for (int c : T->pivot_rec) if (c >= nchan) ++rec_pivots_extras_;
        for (int c : T->born_rec) if (c >= nchan) ++rec_dirs_extras_;
        if ((int)rec_kappa_hist_.size() <= T->kappa) rec_kappa_hist_.resize((size_t)T->kappa + 1, 0);
        ++rec_kappa_hist_[(size_t)T->kappa];
        plan.rec = std::move(T);
        plan.rec_trie.clear();
        plan.emit_built = true;
        cache->reaccount(plan);
    }

inline __attribute__((used)) const CachedPlan& TwirlRecordSampler::Impl::identity_plan_() {
        if (!plan0_) {
            std::vector<uint64_t> k0((size_t)KW, 0);
            DiagNormalForm nf0;
            nf0.prefix = Pauli(n);
            nf0.a.assign((size_t)n, 0);
            nf0.diagonal_class = true;
            plan0_ = cache->get_or_build(G, bare, k0.data(), lhash(k0.data()), nf0, true);
            if (!plan0_) throw std::logic_error("identity_plan_: key derivation (engine bug)");
        } else if (!plan0_->resident) {
            std::vector<uint64_t> k0((size_t)KW, 0);
            DiagNormalForm nf0;
            nf0.prefix = Pauli(n);
            nf0.a.assign((size_t)n, 0);
            nf0.diagonal_class = true;
            plan0_ = cache->get_or_build(G, bare, k0.data(), lhash(k0.data()), nf0, false);
        }
        return *plan0_;
    }

template <bool EX, bool KEEPU>
inline __attribute__((noinline)) bool TwirlRecordSampler::Impl::record_rows_(const CachedPlan& plan, uint64_t* bits) {
        ensure_rec_(plan);
        // r9: everything the κ = 0 hot path reads is the plan's contiguous emission block. EX (the circuit
        // has extras) and KEEPU (the shot is retained: σ-4 and the record read u) are fixed per run — one
        // body, instantiated per variant. The clear stays unconditional (the trie's z land here).
        const int cw = std::max(CW, 1);
        const int rho = plan.emit_rho, kap = plan.emit_kappa, nr = 1 + rho + kap;
        const uint64_t* E = plan.emit.data();
        const uint64_t* EX0 = E + (size_t)nr * cw;
        for (int w = 0; w < CW; ++w) bits[w] ^= E[w];
#ifdef XTIM_MUTATE_R73_NO_PUSH_RETAIN
        constexpr bool keep_u = false;            // mutation build only (R73): u never recorded
#else
        constexpr bool keep_u = KEEPU;
#endif
        if (EX) for (int w = 0; w < XW; ++w) xbits_[(size_t)w] ^= EX0[w];
        rec_uz_.clear();
        for (int k = 0; k < rho; ++k) {
            const bool bit = coins_.next();
            if (keep_u) rec_uz_.push_back((uint8_t)bit);
#ifdef XTIM_MUTATE_R63_SKIP_COIN0
            if (k == 0) continue;                 // mutation build only (R63): coin 0's row not applied
#endif
            if (bit) {
                const uint64_t* rc = E + (size_t)(1 + k) * cw;
                for (int w = 0; w < CW; ++w) bits[w] ^= rc[w];
                if (EX) {
                    const uint64_t* rx = EX0 + (size_t)(1 + k) * XW;
                    for (int w = 0; w < XW; ++w) xbits_[(size_t)w] ^= rx[w];
                }
            }
        }
        if (kap > 0) {
            const RecTable& T = *plan.rec;        // κ > 0: the directions and the trie
            const long n0 = trie_.nodes, e0 = trie_.node_evals;
            const size_t z0 = rec_uz_.size();
            const int bad = trie_.walk(T.lam, plan.rec_trie, 0, born_at_shot_(), rec_uz_);
            if (bad >= 0)
                throw std::logic_error("record_rows_: a non-Hermitian record direction (engine bug)");
            if (trie_.nodes != n0) cache->reaccount(plan);
            if (__builtin_expect(trie_.node_evals != e0 && !born_node_event_, 0)) {
                born_node_event_ = true;
                diag_.push_back({TwirlRecordSampler::DiagEvent::BORN_NODE, "", (long)bare.chi(),
                                 (long)T.kappa, "", trie_.timed_eval_seconds});
            }
            for (int j = 0; j < kap; ++j)
                if (rec_uz_[z0 + (size_t)j]) {
                    const uint64_t* bc = E + (size_t)(1 + rho + j) * cw;
                    for (int w = 0; w < CW; ++w) bits[w] ^= bc[w];
                    if (EX) {
                        const uint64_t* bx = EX0 + (size_t)(1 + rho + j) * XW;
                        for (int w = 0; w < XW; ++w) xbits_[(size_t)w] ^= bx[w];
                    }
                }
        }
        rec_uz_width_ = KEEPU ? rho + kap : -1;   // retained shots only read rec_uz_ (C2 asserts there)
        rec_route_ = true;
        return true;
    }

inline __attribute__((used)) bool TwirlRecordSampler::Impl::semantic_oracle_(const DiagErrorSampler::SparseBatch& b, int32_t e0, int32_t e1, long shot) {
        sem_work_ = bare;
        if (e1 > e0) {
            fired.clear();
            for (int32_t e = e0; e < e1; ++e)
                for (const FiredPauli& f : sides[(size_t)b.ev[e]]) fired.push_back(f);
            DiagPauliClifford D = compose_fired(table, fired);
            D.finalize();
            const DiagNormalForm r = residual_normal_form(D);
            if (!r.diagonal_class) return true;            // out of the diagonal class: no oracle
            for (int q = 0; q < n && q < (int)r.a.size(); ++q)
                if (r.a[q] & 1) sem_work_.apply_clifford(/*S*/ 1, q, 0);
            for (const std::pair<int, int>& e : r.cz) sem_work_.apply_clifford(/*CZ*/ 7, e.first, e.second);
            for (int q = 0; q < n; ++q) {
                if (r.prefix.xbit(q)) sem_work_.apply_clifford(/*X*/ 3, q, 0);
                if (r.prefix.zbit(q)) sem_work_.apply_clifford(/*Z*/ 5, q, 0);
            }
        }
        auto product = [&](const std::vector<uint64_t>& mask) {
            Pauli P(n);
            for (int w = 0; w < GW && w < (int)mask.size(); ++w)
                for (uint64_t word = mask[(size_t)w]; word; word &= word - 1)
                    P = Pauli::multiply(P, gens[(size_t)(w * 64 + __builtin_ctzll(word))]);
            return P;
        };
        // (0) the emitted channel bits must be a possible record of R|bare⟩; condition on them.
        for (int c = 0; c < nchan; ++c) {
            const Pauli Gc = product(sch[(size_t)c].mask);
            const int bit = (int)((sem_raw_bits_[c >> 6] >> (c & 63)) & 1);
            const double p1 = sem_work_.born_p1(Gc);
            if (p1 > 1e-9 && p1 < 1.0 - 1e-9) {
                (void)sem_work_.measure_pauli(Gc, bit ? std::nextafter(1.0, 0.0) : 0.0);
            } else if ((p1 > 0.5) == (bit != 0)) {
                fprintf(stderr, "twirl_records SEMANTICS ORACLE FAIL shot=%ld: channel %d emitted %d "
                                "but is definite %+d on the shot's residual state\n",
                        shot, c, bit, p1 > 0.5 ? 1 : -1);
                return false;
            }
        }
        // (a) S2b: every extra (the Born observable, then the Born decisions in record order) is a
        // possible record of the channel-conditioned state: project each at its emitted RAW value,
        // and a definite one must agree.
        auto project_at = [&](const Pauli& W, int raw, const char* what, int idx) -> bool {
            const double p1 = sem_work_.born_p1(W);
            if (p1 > 1e-9 && p1 < 1.0 - 1e-9) {
                (void)sem_work_.measure_pauli(W, raw ? std::nextafter(1.0, 0.0) : 0.0);
            } else if ((p1 > 0.5) == (raw != 0)) {
                fprintf(stderr, "twirl_records SEMANTICS ORACLE FAIL shot=%ld: %s %d emitted raw %d "
                                "but is definite %+d on the record-conditioned state\n",
                        shot, what, idx, raw, p1 > 0.5 ? 1 : -1);
                return false;
            }
            return true;
        };
        if (born_obs && !project_at(born_W, obs_bit_cur_raw_ & 1, "born observable", 0)) return false;
        if (has_born_dec_)
            for (size_t j = 0; j < born_dec.size(); ++j)
                if (!project_at(born_dec[j].second, dec_raw_[j] & 1, "born decision", (int)j)) return false;
        ++sem_checked_;
        return true;
    }

// record_rows_'s four variants are instantiated HERE only (the family TU references them through rows_fn_).
template bool TwirlRecordSampler::Impl::record_rows_<false, false>(const CachedPlan&, uint64_t*);
template bool TwirlRecordSampler::Impl::record_rows_<false, true>(const CachedPlan&, uint64_t*);
template bool TwirlRecordSampler::Impl::record_rows_<true, false>(const CachedPlan&, uint64_t*);
template bool TwirlRecordSampler::Impl::record_rows_<true, true>(const CachedPlan&, uint64_t*);

// ── Public surface ────────────────────────────────────────────────────────────────────────
TwirlRecordSampler::TwirlRecordSampler(const FramedSuperposition& bare, const Circuit& deferred,
                                       const std::vector<std::pair<int, int>>& reads,
                                       const std::vector<std::vector<int>>& detectors,
                                       const std::vector<std::pair<int, std::vector<int>>>& observables,
                                       uint64_t seed, const TwirlRecordOptions& opt)
    : impl_(new Impl(bare, deferred, reads, detectors, observables, seed, opt)) {}

struct TwirlRecordSampler::OwnCaches {
    std::shared_ptr<TwirlPlanCache> plans;
};

TwirlRecordSampler::TwirlRecordSampler(const FramedSuperposition& bare, const Circuit& deferred,
                                       const std::vector<std::pair<int, int>>& reads,
                                       const std::vector<std::vector<int>>& detectors,
                                       const std::vector<std::pair<int, std::vector<int>>>& observables,
                                       uint64_t seed, const TwirlRecordOptions& opt, OwnCaches&& caches,
                                       std::optional<PropagationTable>&& class_table)
    : impl_(new Impl(bare, deferred, reads, detectors, observables, seed, opt,
                     std::move(caches.plans), std::move(class_table))) {}

std::unique_ptr<TwirlRecordSampler> TwirlRecordSampler::make(
        const FramedSuperposition& bare, const Circuit& deferred,
        const std::vector<std::pair<int, int>>& reads,
        const std::vector<std::vector<int>>& detectors,
        const std::vector<std::pair<int, std::vector<int>>>& observables,
        uint64_t seed, const TwirlRecordOptions& opt, std::optional<PropagationTable> class_table) {
    OwnCaches c;                                    // allocation order: the plan cache, then the
    c.plans = std::make_shared<TwirlPlanCache>();   // sampler and its Impl (J-b2 J5: the PPR cache is gone)
    return std::unique_ptr<TwirlRecordSampler>(
        new TwirlRecordSampler(bare, deferred, reads, detectors, observables, seed, opt, std::move(c),
                               std::move(class_table)));
}

TwirlRecordSampler::~TwirlRecordSampler() = default;

int TwirlRecordSampler::setup_error() const { return impl_->setup_error; }
bool TwirlRecordSampler::run(long shots) { return impl_->run(shots); }
bool TwirlRecordSampler::set_seed(uint64_t seed) { return impl_->set_seed(seed); }
void TwirlRecordSampler::set_plan_payload_budget(size_t bytes) {
    impl_->plan_budget_ = bytes;
    impl_->budget_pub_.store(bytes, std::memory_order_relaxed);
}
void TwirlRecordSampler::set_retain_state(bool on) { impl_->retain_state = on; }
void TwirlRecordSampler::set_input_frame(const uint8_t* data, int stride) {
    if (!impl_) return;
    impl_->frame_data_   = data;
    impl_->frame_stride_ = stride;
}
FramedSuperposition TwirlRecordSampler::materialize_shot(const uint8_t* plan_key, int n_plan,
                                                         const uint8_t* coins, int n_coins) const {
    return impl_->materialize_shot(plan_key, n_plan, coins, n_coins);
}

TwirlRecordSampler::PlanStructure TwirlRecordSampler::plan_structure(const uint8_t* plan_key,
                                                                     int n_plan) const {
    PlanStructure out;
    out.nf = impl_->parse_plan_key(plan_key, n_plan);
    out.law = build_shot_law(impl_->G, out.nf);
    // J-b2 J7b: the plan's record table by record-space B (the readers' builder; the σ dual basis is gone)
    if (!out.law.fallback && out.nf.diagonal_class &&
        !build_record_table_recB(impl_->G, impl_->bare, out.nf, impl_->rec_cols_, out.rec))
        throw std::logic_error("plan_structure: record-space B reports a fallback on a non-fallback plan (engine bug)");
    return out;
}
std::vector<uint8_t> TwirlRecordSampler::record_channel_refs() const {
    std::vector<uint8_t> r;
    for (int c = 0; c < impl_->nchan; ++c) r.push_back(impl_->rec_cols_[(size_t)c].ref & 1);
    return r;
}
std::vector<Pauli> TwirlRecordSampler::record_operators() const {
    std::vector<Pauli> o;
    for (const RecColumn& c : impl_->rec_cols_) o.push_back(c.M);
    return o;
}
int TwirlRecordSampler::record_value_words() const {
    return std::max(1, ((int)impl_->rec_cols_.size() + 63) / 64);
}
std::vector<int> TwirlRecordSampler::record_column_order() const {
    std::vector<int> o;
    for (const RecColumn& c : impl_->rec_cols_) o.push_back(c.order);
    return o;
}
std::pair<long, long> TwirlRecordSampler::born_trie_check() {
    Impl& I = *impl_;
    long checked = 0, bad = 0;
    // A test hook: the from-scratch evaluations below run on the sampler's walker (its one reusable
    // working buffer — no allocation inside p0_from_scratch, the t_node ledger window), so its exact
    // counters (record_report()["born_trie"]) are saved here and restored on return: the check
    // never moves a reported counter.
    struct Restore {
        BornTrie& t;
        const long walks, steps, nodes, node_evals, chi_j, passes, max_support;
        const int max_depth;
        ~Restore() {
            t.walks = walks; t.steps = steps; t.nodes = nodes; t.node_evals = node_evals;
            t.chi_j = chi_j; t.passes = passes; t.max_support = max_support; t.max_depth = max_depth;
        }
    } restore{I.trie_, I.trie_.walks, I.trie_.steps, I.trie_.nodes, I.trie_.node_evals,
              I.trie_.chi_j, I.trie_.passes, I.trie_.max_support, I.trie_.max_depth};
    std::vector<uint8_t> z;
    for (size_t i = 0; i < I.cache->size(); ++i) {
        const CachedPlan& p = I.cache->header(i);
        // S2b: the walked trie is the RECORD trie over the record table's canonical directions.
        if (!p.resident || !p.rec || p.rec_trie.empty()) continue;
        const std::vector<FreeDir>& lam = p.rec->lam;
        // DFS over (node, depth, z-prefix).
        std::vector<std::pair<uint32_t, std::vector<uint8_t>>> st{{0u, {}}};
        while (!st.empty()) {
            auto [node, pre] = st.back();
            st.pop_back();
            const BornNode nd = p.rec_trie[node];
            if (std::isnan(nd.p0)) continue;
            const double q = I.trie_.p0_from_scratch(lam, pre.data(), (int)pre.size());
            ++checked;
            if (std::memcmp(&q, &nd.p0, sizeof(double)) != 0) ++bad;
            for (int b = 0; b < 2; ++b)
                if (nd.kid[b]) { z = pre; z.push_back((uint8_t)b); st.emplace_back(nd.kid[b], z); }
        }
    }
    return {checked, bad};
}
std::pair<long, long> TwirlRecordSampler::born_trie_oracle() const {
    const Impl& I = *impl_;
    long checked = 0, bad = 0;
    for (size_t i = 0; i < I.cache->size(); ++i) {
        const CachedPlan& p = I.cache->header(i);
        if (!p.resident || !p.rec || p.rec_trie.empty()) continue;
        std::vector<Pauli> L;
        for (const FreeDir& d : p.rec->lam) L.push_back(derive_kernel_logical(I.bare, I.G, d));
        std::vector<std::pair<uint32_t, std::vector<uint8_t>>> st{{0u, {}}};
        while (!st.empty()) {
            auto [node, pre] = st.back();
            st.pop_back();
            const BornNode nd = p.rec_trie[node];
            if (std::isnan(nd.p0)) continue;
            FramedSuperposition w = I.bare;
            for (size_t j = 0; j < pre.size(); ++j)
                (void)w.measure_pauli(L[j], pre[j] ? std::nextafter(1.0, 0.0) : 0.0);
            const double q = w.born_p1(L[pre.size()]);
            ++checked;
            if (std::memcmp(&q, &nd.p0, sizeof(double)) != 0) ++bad;
            for (int b = 0; b < 2; ++b)
                if (nd.kid[b]) { std::vector<uint8_t> z = pre; z.push_back((uint8_t)b); st.emplace_back(nd.kid[b], z); }
        }
    }
    return {checked, bad};
}
// T5 step 0 (exact-residual arc): pure-data forwarding of the Impl's born_dec vector — the
// Born-measured decision operators + record inverts, declaration order (see the header doc).
std::vector<TwirlRecordSampler::BornDecOp> TwirlRecordSampler::born_dec_ops() const {
    std::vector<BornDecOp> out;
    out.reserve(impl_->born_dec.size());
    for (size_t j = 0; j < impl_->born_dec.size(); ++j)
        out.push_back(BornDecOp{impl_->born_dec[j].first, impl_->born_dec[j].second,
                                impl_->born_dec_inv[j]});
    return out;
}

long TwirlRecordSampler::used() const { return impl_->used; }
int TwirlRecordSampler::born_obs_index() const { return impl_->born_idx; }
int TwirlRecordSampler::born_obs_ref() const { return impl_->born_ref_; }
long TwirlRecordSampler::obs_ones() const { return impl_->obs_ones_; }
long TwirlRecordSampler::ps_accepted() const { return impl_->ps_acc_; }
long TwirlRecordSampler::ps_obs_ones() const { return impl_->ps_obs_ones_; }
long TwirlRecordSampler::semantic_oracle_shots() const { return impl_->sem_checked_; }
const std::string& TwirlRecordSampler::disk_path() const { return impl_->disk_path; }
long TwirlRecordSampler::clean_shots() const { return impl_->clean; }
long TwirlRecordSampler::diag_shots() const { return impl_->diag; }
int TwirlRecordSampler::deferred_wires() const { return impl_->n; }
int TwirlRecordSampler::num_channels() const { return impl_->nchan; }
int TwirlRecordSampler::channel_words() const { return impl_->CW; }
int TwirlRecordSampler::num_decisions() const { return impl_->num_dec_; }
size_t TwirlRecordSampler::plans() const { return impl_->cache->size(); }
TwirlRecordSampler::RecordCounters TwirlRecordSampler::record_counters() const {
    const Impl& I = *impl_;
    RecordCounters c;
    c.routes.emplace_back("fast", I.fast_derived_);    // attributed per run (uncounted hot loop)
    c.routes.emplace_back("clean", I.clean);
    for (int r = 0; r < Impl::R_COUNT; ++r) c.routes.emplace_back(Impl::route_name_(r), I.route_[r]);
    c.trie_walks = I.trie_.walks;
    c.trie_steps = I.trie_.steps;
    c.trie_nodes = I.trie_.nodes;
    c.trie_node_evals = I.trie_.node_evals;
    c.trie_chi_j = I.trie_.chi_j;
    c.trie_max_depth = I.trie_.max_depth;
    c.trie_passes = I.trie_.passes;
    c.trie_max_support = I.trie_.max_support;
    c.trie_root_memo_hits = I.trie_.root_memo_hits;
    c.rec_tables = I.rec_tables_; c.rec_extras = I.rec_extras_;
    c.rec_ops_plane = I.rec_ops_plane_; c.rec_ops_pivot_reduce = I.rec_ops_pivot_;
    c.rec_ops_greduce = I.rec_ops_greduce_; c.rec_ops_echelon = I.rec_ops_echelon_;
    c.rec_pivots_from_extras = I.rec_pivots_extras_; c.rec_dirs_from_extras = I.rec_dirs_extras_;
    c.rec_kappa_hist = I.rec_kappa_hist_;
    c.obs_draws_record = I.obs_draws_record_;
    c.structural_shots = I.structural_shots_;
    c.plan_misses_kappa0 = I.cache->misses_kappa0();
    c.plan_misses_kappa_pos = I.cache->misses_kappa_pos();
    c.plan_misses_fallback = I.cache->misses_fallback();
    c.payload_budget = I.plan_budget_;
    c.payload_resident_bytes = I.cache->resident_bytes();
    c.payload_peak_bytes = std::max(I.payload_peak_, c.payload_resident_bytes);
    c.payload_overshoot_bytes = I.payload_overshoot_;
    c.reader_lookups = I.reader_lookups_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(I.rec_memo_mu_);
        c.reader_tables = (long)I.rec_memo_.size();
        c.reader_bytes = I.reader_bytes_.load(std::memory_order_relaxed);
        c.reader_peak_bytes = I.reader_peak_;
        c.reader_overshoot_bytes = I.reader_overshoot_;
        c.reader_evictions = I.reader_evictions_;
    }
    c.header_bytes = I.cache->header_bytes();
    c.payload_builds = I.cache->payload_builds();
    c.payload_evictions = I.cache->evictions();
    return c;
}
std::vector<TwirlRecordSampler::DiagEvent> TwirlRecordSampler::take_diagnostics() {
    std::vector<DiagEvent> out;
    out.swap(impl_->diag_);
    return out;
}
const std::vector<long long>& TwirlRecordSampler::channel_one_counts() const { return impl_->chancnt; }
double TwirlRecordSampler::draw_seconds() const { return impl_->t_draw; }
double TwirlRecordSampler::wall_seconds() const { return impl_->wall; }
#ifdef XTIM_BENCH_REPLAY
void TwirlRecordSampler::bench_record(bool on) {
    impl_->bench_rec_on_ = on;
    impl_->bench_rec_off_.assign(1, 0);
    impl_->bench_rec_ev_.clear();
}
const std::vector<int32_t>& TwirlRecordSampler::bench_rec_off() const { return impl_->bench_rec_off_; }
const std::vector<int32_t>& TwirlRecordSampler::bench_rec_ev() const { return impl_->bench_rec_ev_; }
void TwirlRecordSampler::bench_replay(std::vector<int32_t> off, std::vector<int32_t> ev) {
    const int alts = impl_->smp->alt_total();
    if (!off.empty()) {
        if (off[0] != 0 || (size_t)off.back() != ev.size())
            throw std::invalid_argument("bench replay: offsets must start at 0 and end at len(ev)");
        for (size_t i = 1; i < off.size(); ++i)
            if (off[i] < off[i - 1]) throw std::invalid_argument("bench replay: offsets decrease");
        for (int32_t e : ev)
            if (e < 0 || e >= alts) throw std::invalid_argument("bench replay: alt id out of range");
    }
    impl_->bench_src_ = !off.empty();
    impl_->bench_src_off_ = std::move(off);
    impl_->bench_src_ev_ = std::move(ev);
}
// FNV-1a over what an event id MEANS to the hot loop's source: the flat alt count, the channel
// count, and per alt its owning channel and record-flip mask (the alt numbering contract).
uint64_t TwirlRecordSampler::bench_fingerprint() const {
    const DiagErrorSampler& d = *impl_->smp;
    uint64_t h = 1469598103934665603ull;
    auto mixin = [&](uint64_t v) { for (int b = 0; b < 8; ++b) { h ^= (v >> (8 * b)) & 0xff; h *= 1099511628211ull; } };
    mixin((uint64_t)d.n()); mixin((uint64_t)d.K()); mixin((uint64_t)d.alt_total());
    for (int a = 0; a < d.alt_total(); ++a) { mixin((uint64_t)d.channel_of(a)); mixin(d.flips(a)); }
    return h;
}
// Noise component alone: the production event source for `shots` shots in 4096-shot batches
// (the run() batch partition), from the CURRENT stream position. Returns seconds; *events = total.
long TwirlRecordSampler::bench_build_plans(const std::vector<std::string>& keys) {
    Impl& I = *impl_;
    std::vector<uint64_t> kw((size_t)I.KW);
    long built = 0;
    for (const std::string& k : keys) {
        DiagNormalForm nf = I.parse_plan_key(reinterpret_cast<const uint8_t*>(k.data()), (int)k.size());
        std::fill(kw.begin(), kw.end(), 0);
        for (int q = 0; q < I.n; ++q)
            if ((nf.a[(size_t)q] & 1) && !((I.prodmask[(size_t)(q >> 6)] >> (q & 63)) & 1)) kw[(size_t)(q >> 6)] |= 1ULL << (q & 63);
        bool ok = true;
        for (const auto& pr : nf.cz) {
            const auto it = I.pidx.find(pr);
            if (it == I.pidx.end()) { ok = false; break; }
            const int i = it->second;
            if ((I.upair_keep_[(size_t)(i >> 6)] >> (i & 63)) & 1) kw[(size_t)(I.NW + (i >> 6))] ^= 1ULL << (i & 63);
        }
        if (!ok) continue;
        const CachedPlan* plan = I.cache->get_or_build(I.G, I.bare, kw.data(), I.lhash(kw.data()), nf, false);
        if (!plan) throw std::logic_error("bench_build_plans: key derivation disagrees (engine bug)");
        if (!plan->fallback) I.ensure_rec_(*plan);
        ++built;
    }
    return built;
}
double TwirlRecordSampler::bench_noise_only(long shots, long* events) {
    DiagErrorSampler::SparseBatch b;
    long done = 0, ev = 0;
    auto t0 = clk::now();
    while (done < shots) {
        const int S = (int)std::min<long>(4096, shots - done);
        impl_->smp->sample_events(S, b);
        ev += (long)b.ev.size();
        done += S;
    }
    const double sec = std::chrono::duration<double>(clk::now() - t0).count();
    if (events) *events = ev;
    return sec;
}
#endif
const std::vector<int>& TwirlRecordSampler::channel_detectors() const { return impl_->chan_det; }
const std::vector<int>& TwirlRecordSampler::channel_observables() const { return impl_->chan_obs; }
uint64_t TwirlRecordSampler::group_token() const { return impl_->G.identity_token(); }
const FramedSuperposition& TwirlRecordSampler::bare_state() const { return impl_->bare; }
const std::vector<Pauli>& TwirlRecordSampler::certified_generators() const { return impl_->gens; }
std::vector<std::vector<uint64_t>> TwirlRecordSampler::record_channel_masks() const {
    std::vector<std::vector<uint64_t>> out;
    out.reserve(impl_->sch.size());
    for (const auto& c : impl_->sch) out.push_back(c.mask);
    return out;
}

}  // namespace qeccore
