# TLC budget manifest (stage/tlc; OUTPUTS BYTE-IDENTICAL; budget ACCOUNTING CHANGED — owner signature needed)

## The change
Before TLC the framed engine's chi-scaled kernel scratch (measurement / Born / canonicalise / dual-image scan) lived in
`thread_local` slots registered with the thread-scratch registry: charged to the state budget while the thread held it,
released when the last Python owner was dropped (or `_release_thread_state_scratch()`). After TLC it is caller-owned
(`FramedScratch` / `DualScanScratch`): charged by the owner's high-water ticket BEFORE the buffers grow (same estimates,
same names in refusals), released with the owner — the exact engine releases at the end of every program compile / run
(sampler.cpp), cold callers per call. `_thread_scratch_bytes()` / `release_thread_state_scratch()` now cover only the
CanonicalStabSum oracle's remaining thread slots.

## Re-pinned assertions (tests/test_state_budget.py: 2246b5a, trimmed in 67e7416 and the next commit) — each strictly required (the old assertion fails)
| test | old | new | mechanism |
|---|---|---|---|
| test_measurement_scratch_is_charged_and_released | after `sample(10)` on BH12: `_thread_scratch_bytes() > 4096*100`; after `_release_thread_state_scratch()`: `live_rel <= live - scr`, `scr_rel == 0` | `live after sample == live after compile`; the size bound moves to C++ (test_tlc_scratch `test_scratch_charge`: fmag charge > 100 B x chi; refusal before growth leaves the part with no buffer and no charge) | scratch is owned per compile / run, not per thread |
| test_measurement_scratch_refused_before_allocating | `res["scr"] > 4096*8` | `res["released"]` (live unchanged across the call) | the call owns its Born scratch |
| test_live_total_returns_to_baseline_automatically | `res["peak"] > res["live0"] and res["scr"] > 0` | `res["peak"] > res["live0"]` | the framed engine holds no thread scratch |
New tests (not re-pins): test_measurement_scratch_not_retained_across_live_samplers (k = 4 live samplers hold exactly k x one
sampler's states); tests/test_tlc_scratch_lifetime.py (runs independent of interleaved programs). Unchanged and passing:
test_refusal_mid_sampling_invalidates_the_exact_sampler, test_refusal_while_compiling_is_a_capacity_error_not_a_rejection.
Note (review m-7): the CanonicalStabSum oracle's dual-scan charge and the framed one are now separate tickets; the measured
peaks below show no rise.

## Peak charged bytes per corpus circuit (instrumented builds; tools/peaks.py: bare, exact+auto 500 shots (chi <= 4096), DEM, twirl
2000 shots + barrier 300 + 4 materialize; each stage from a reset high-water mark; one fresh process per circuit)
158 circuits: peak DOWN on 20, equal on 138, UP on 0; every stage verdict identical; no circuit flips at 2 GiB (or at any budget
except inside the listed [TLC, parent) intervals, where TLC now SUCCEEDS where the parent refused).

## k live exact samplers (BH12, chi = 4096; each compiled + sample(20), held alive)
| k | peak 6c44b52 | peak TLC | held live 6c44b52 | held live TLC |
|---|---|---|---|---|
| 1 | 4456072 | 4422968 | 2817816 | 196608 |
| 2 | 3751560 | 3718456 | 3014424 | 393216 |
| 4 | 4144776 | 4111672 | 3407640 | 786432 |
| 8 | 4931208 | 4898104 | 4194072 | 1572864 |

## Per-circuit table

| circuit | chi | peak 6c44b52 (B) | peak TLC (B) | TLC/parent | flips iff XTIM_MAX_STATE_BYTES in | verdicts equal | stages that moved |
|---|---|---|---|---|---|---|---|
| blowup/circ/15to1.stim | 2 | 1024 | 1024 | 1.000 | - | yes |  |
| blowup/circ/bh10.stim | 1024 | 524288 | 524288 | 1.000 | - | yes |  |
| blowup/circ/bh12.stim | 4096 | 2097152 | 2097152 | 1.000 | - | yes |  |
| blowup/circ/bh14.stim | 16384 | 4276224 | 4276224 | 1.000 | - | yes |  |
| blowup/circ/bh16.stim | 65536 | 18808832 | 18808832 | 1.000 | - | yes |  |
| blowup/circ/bh18.stim | 262144 | 82051072 | 82051072 | 1.000 | - | yes |  |
| blowup/circ/bh2.stim | 4 | 2048 | 2048 | 1.000 | - | yes |  |
| blowup/circ/bh20.stim | 1048576 | 355467264 | 355467264 | 1.000 | - | yes |  |
| blowup/circ/bh4.stim | 16 | 8192 | 8192 | 1.000 | - | yes |  |
| blowup/circ/bh6.stim | 64 | 32768 | 32768 | 1.000 | - | yes |  |
| blowup/circ/bh8.stim | 256 | 131072 | 131072 | 1.000 | - | yes |  |
| blowup/circ/bt27.stim | 512 | 853176 | 845056 | 0.990 | [845056, 853176) | yes | exact |
| blowup/circ/bt27_p0.stim | 512 | 853176 | 845056 | 0.990 | [845056, 853176) | yes | exact |
| blowup/circ/bt81.stim | 512 | 1557176 | 1501952 | 0.965 | [1501952, 1557176) | yes | exact |
| blowup/circ/bt81_p0.stim | 512 | 1510072 | 1501952 | 0.995 | [1501952, 1510072) | yes | exact |
| blowup/circ/cult_d3.stim | 2 | 2372 | 2372 | 1.000 | - | yes |  |
| blowup/circ/cult_d3_p0.stim | 2 | 1024 | 1024 | 1.000 | - | yes |  |
| blowup/circ/cult_d5.stim | 2 | 21976 | 21544 | 0.980 | [21544, 21976) | yes | exact |
| blowup/circ/cult_d5_p0.stim | 2 | 21592 | 21544 | 0.998 | [21544, 21592) | yes | exact |
| private/ccz_variant_00.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_01.stim | 1 | 3497 | 3497 | 1.000 | - | yes |  |
| private/ccz_variant_02.stim | 1 | 3497 | 3497 | 1.000 | - | yes |  |
| private/ccz_variant_03.stim | 1 | 3497 | 3497 | 1.000 | - | yes |  |
| private/ccz_variant_04.stim | 1 | 3605 | 3605 | 1.000 | - | yes |  |
| private/ccz_variant_05.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_06.stim | 1 | 3497 | 3497 | 1.000 | - | yes |  |
| private/ccz_variant_07.stim | 1 | 3497 | 3497 | 1.000 | - | yes |  |
| private/ccz_variant_08.stim | 1 | 3497 | 3497 | 1.000 | - | yes |  |
| private/ccz_variant_09.stim | 1 | 3605 | 3605 | 1.000 | - | yes |  |
| private/ccz_variant_10.stim | 512 | 2384896 | 2384896 | 1.000 | - | yes |  |
| private/ccz_variant_11.stim | 1 | 4649 | 4649 | 1.000 | - | yes |  |
| private/ccz_variant_12.stim | 1 | 4649 | 4649 | 1.000 | - | yes |  |
| private/ccz_variant_13.stim | 1 | 4649 | 4649 | 1.000 | - | yes |  |
| private/ccz_variant_14.stim | 1 | 4757 | 4757 | 1.000 | - | yes |  |
| private/ccz_variant_15.stim | 512 | 2384896 | 2384896 | 1.000 | - | yes |  |
| private/ccz_variant_16.stim | 1 | 4649 | 4649 | 1.000 | - | yes |  |
| private/ccz_variant_17.stim | 1 | 4649 | 4649 | 1.000 | - | yes |  |
| private/ccz_variant_18.stim | 1 | 4649 | 4649 | 1.000 | - | yes |  |
| private/ccz_variant_19.stim | 1 | 4757 | 4757 | 1.000 | - | yes |  |
| private/ccz_variant_20.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_21.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_22.stim | -1 | 0 | 0 | 1.000 | - | yes |  |
| private/ccz_variant_23.stim | -1 | 0 | 0 | 1.000 | - | yes |  |
| private/ccz_variant_24.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_25.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_26.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_27.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_28.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_29.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_30.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_31.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_32.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_33.stim | 1 | 3497 | 3497 | 1.000 | - | yes |  |
| tests/private/ccz_factory_b.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_34.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_35.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_36.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| private/ccz_variant_37.stim | 1 | 3497 | 3497 | 1.000 | - | yes |  |
| private/ccz_variant_38.stim | 512 | 2384896 | 2384896 | 1.000 | - | yes |  |
| private/ccz_variant_39.stim | 512 | 2384896 | 2384896 | 1.000 | - | yes |  |
| private/ccz_variant_40.stim | 1 | 4649 | 4649 | 1.000 | - | yes |  |
| private/ccz_variant_41.stim | 512 | 2384896 | 2384896 | 1.000 | - | yes |  |
| private/ccz_variant_42.stim | 512 | 2384896 | 2384896 | 1.000 | - | yes |  |
| private/ccz_variant_43.stim | 1 | 4649 | 4649 | 1.000 | - | yes |  |
| private/ccz_variant_44.stim | 1 | 3497 | 3497 | 1.000 | - | yes |  |
| private/ccz_variant_45.stim | -1 | 0 | 0 | 1.000 | - | yes |  |
| private/ccz_variant_46.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| examples/ch_cultivation.stim | 2 | 1024 | 1024 | 1.000 | - | yes |  |
| examples/code_switching_faithful.stim | 2 | 11584 | 9952 | 0.859 | [9952, 11584) | yes | auto,exact |
| examples/code_switching_rate.stim | 2 | 10392 | 9976 | 0.960 | [9976, 10392) | yes | exact,twirl |
| examples/cube_ccz.stim | 8 | 8920 | 8536 | 0.957 | [8536, 8920) | yes | auto,exact |
| examples/cultivation_d3_faithful.stim | 2 | 3812 | 3812 | 1.000 | - | yes | dem |
| examples/cultivation_d3_rate.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| examples/cultivation_d5.stim | 2 | 21976 | 21544 | 0.980 | [21544, 21976) | yes | exact |
| examples/distillation_15_1_3.stim | 2 | 5744 | 5552 | 0.967 | [5552, 5744) | yes | exact |
| examples/miniature_oracle.stim | 2 | 3164 | 2972 | 0.939 | [2972, 3164) | yes | dem,exact |
| rule/examples/E1/e1_AA.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E1/e1_SD6.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E1/e1_ro_AA.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E1/e1_ro_SD6.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E10/e10_AA.stim | 2 | 13204 | 12960 | 0.982 | [12960, 13204) | yes | auto,exact |
| rule/examples/E10/e10_SD6.stim | 2 | 13204 | 12960 | 0.982 | [12960, 13204) | yes | auto,exact |
| rule/examples/E10/e10_twin_AA.stim | 1 | 520 | 520 | 1.000 | - | yes |  |
| rule/examples/E10/e10_twin_SD6.stim | 1 | 520 | 520 | 1.000 | - | yes |  |
| rule/examples/E11/e11_AA.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E11/e11_SD6.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E2/t0_AA.stim | 2 | 13540 | 13152 | 0.971 | [13152, 13540) | yes | exact |
| rule/examples/E2/t0_SD6.stim | 2 | 13540 | 13152 | 0.971 | [13152, 13540) | yes | exact |
| rule/examples/E2/t0_twin_AA.stim | 1 | 604 | 604 | 1.000 | - | yes |  |
| rule/examples/E2/t0_twin_SD6.stim | 1 | 604 | 604 | 1.000 | - | yes |  |
| rule/examples/E2/t1_AA.stim | 2 | 14120 | 13732 | 0.973 | [13732, 14120) | yes | exact |
| rule/examples/E2/t1_SD6.stim | 2 | 14120 | 13732 | 0.973 | [13732, 14120) | yes | exact |
| rule/examples/E2/t1_twin_AA.stim | 1 | 604 | 604 | 1.000 | - | yes |  |
| rule/examples/E2/t1_twin_SD6.stim | 1 | 604 | 604 | 1.000 | - | yes |  |
| rule/examples/E3/e3_AA.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E3/e3_SD6.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E4/e4_AA.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E4/e4_SD6.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E5/e5_AA.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E5/e5_SD6.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E6/e6_AA.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E6/e6_SD6.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E7/e7_AA.stim | 2 | 14508 | 14120 | 0.973 | [14120, 14508) | yes | exact |
| rule/examples/E7/e7_SD6.stim | 2 | 14508 | 14120 | 0.973 | [14120, 14508) | yes | exact |
| rule/examples/E7/e7_twin_AA.stim | 1 | 604 | 604 | 1.000 | - | yes |  |
| rule/examples/E7/e7_twin_SD6.stim | 1 | 604 | 604 | 1.000 | - | yes |  |
| rule/examples/E8/e8_AA.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E8/e8_SD6.stim | 1 | 488 | 488 | 1.000 | - | yes |  |
| rule/examples/E9/e9_AA.stim | 1 | 508 | 508 | 1.000 | - | yes |  |
| rule/examples/E9/e9_SD6.stim | 1 | 508 | 508 | 1.000 | - | yes |  |
| stateroute/circ/15to1_ts.stim | 1 | 141 | 141 | 1.000 | - | yes |  |
| stateroute/circ/bh10_ts.stim | 1 | 219 | 219 | 1.000 | - | yes |  |
| stateroute/circ/bh12_ts.stim | 1 | 247 | 247 | 1.000 | - | yes |  |
| stateroute/circ/bh14_ts.stim | 1 | 275 | 275 | 1.000 | - | yes |  |
| stateroute/circ/bh16_ts.stim | 1 | 303 | 303 | 1.000 | - | yes |  |
| stateroute/circ/bh18_ts.stim | 1 | 331 | 331 | 1.000 | - | yes |  |
| stateroute/circ/bh20_ts.stim | 1 | 359 | 359 | 1.000 | - | yes |  |
| stateroute/circ/bh2_ts.stim | 1 | 141 | 141 | 1.000 | - | yes |  |
| stateroute/circ/bh4_ts.stim | 1 | 141 | 141 | 1.000 | - | yes |  |
| stateroute/circ/bh6_ts.stim | 1 | 163 | 163 | 1.000 | - | yes |  |
| stateroute/circ/bh8_ts.stim | 1 | 191 | 191 | 1.000 | - | yes |  |
| stateroute/circ/bt27_ts.stim | 1 | 707 | 707 | 1.000 | - | yes |  |
| stateroute/circ/bt81_ts.stim | 1 | 2003 | 2003 | 1.000 | - | yes |  |
| stateroute/circ/cult_d3_ts.stim | 2 | 364 | 364 | 1.000 | - | yes |  |
| stateroute/circ/cult_d5_ts.stim | 2 | 1276 | 1276 | 1.000 | - | yes |  |
| tests/data/adaptq_born5_dec_enum_producer.stim | 32 | 4512 | 4512 | 1.000 | - | yes |  |
| tests/data/adaptq_qrm_magic_producer.stim | 2 | 396 | 396 | 1.000 | - | yes |  |
| tests/data/adaptq_steane_h_producer.stim | 1 | 141 | 141 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0008_cube+rm.stim | 16 | 2960 | 2960 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0010_cube.stim | 8 | 1128 | 1128 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0013_cube.stim | 8 | 1128 | 1128 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0015_cube+cube.stim | 8 | 1128 | 1128 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0020_rm.stim | 2 | 288 | 288 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0022_rm+rm.stim | 4 | 916 | 916 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0028_cube.stim | 8 | 1128 | 1128 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0030_cube.stim | 8 | 1128 | 1128 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0043_rm.stim | 2 | 282 | 282 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0045_rm.stim | 2 | 288 | 288 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0050_cube+cube.stim | 64 | 9024 | 9024 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0051_rm.stim | 2 | 312 | 312 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0062_cube+rm.stim | 16 | 3024 | 3024 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0064_cube.stim | 8 | 1128 | 1128 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0067_cube+rm.stim | 16 | 3312 | 3312 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0068_rm.stim | 2 | 312 | 312 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0069_cube+cube.stim | 64 | 9664 | 9664 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0076_rm+rm.stim | 4 | 964 | 964 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0089_cube+rm.stim | 16 | 2864 | 2864 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0090_rm+rm.stim | 4 | 988 | 988 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0097_rm+rm.stim | 4 | 916 | 916 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0104_rm.stim | 2 | 300 | 300 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0116_cube+rm.stim | 16 | 2768 | 2768 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0122_cube+rm.stim | 2 | 396 | 396 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0129_cube.stim | 8 | 1128 | 1128 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0137_rm.stim | 2 | 300 | 300 | 1.000 | - | yes |  |
| tests/data/dem_factored_law/v2_777_0149_cube.stim | 8 | 1128 | 1128 | 1.000 | - | yes |  |
| tests/private/ccz_factory_b.stim | 512 | 1795072 | 1795072 | 1.000 | - | yes |  |
| tests/private/ccz_factory_a.stim | 512 | 1878016 | 1878016 | 1.000 | - | yes |  |
| tests/data/merlin_switch3_bt27_p0.stim | 512 | 357376 | 357376 | 1.000 | - | yes |  |
