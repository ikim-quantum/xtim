# tests/data/xdem

Fixtures of the xdem tests. Each file below is a byte-for-byte copy of the listed source, an internal working file
of the format's development (md5 of the copy = md5 of the source, checked at copy time).

| file | source | md5 |
|---|---|---|
| `ex422_obs.xdem` | `ex422.xdem` | `8dcb410dad750f2480fafcb83b66e458` |
| `circuits/ex422_obs.stim` | `ex422_obs.stim` | `187c7e1f66a0f120c1c7c6323cca8076` |
| `circuits/toy2obs.stim` | `toy2obs.stim` | `df827c224893fc4b2d9c5f5c6471b33a` |
| `circuits/rm15rep.stim` | `rm15rep.stim` | `4f598a0478cc6f4682249cfcd6da9a23` |
| `circuits/ccz_toy.stim` | `ccz_toy.stim` | `e8b56d6fa0e6730bab456aa3938cb0fb` |
| `circuits/ccz_toy_obs.stim` | `ccz_toy_obs.stim` | `5da93be5b0ddb7842c30547115345b45` |
| `circuits/ccz_toy_obs_p5.stim` | `ccz_toy_obs_p5.stim` | `7aaf3fbf37de1ba182f1bc6e236b3741` |

`ex422_obs.xdem` is the SPEC §3.7 worked file. `circuits/ex422_obs.stim` is the circuit that
`tests/test_xdem_transition.py` exported with the dc v1 engine to check that file (both were removed in Task 11; see
commit 19f2f78).

## Writer fixtures (xdem Task 4)

Written by this branch's writer: `xtim.Circuit(text).detector_error_model_text(format="xdem")` with the default
options, where `text` is `tests/_xdem_circuits.circuit(<name>)` (an `xtim/_examples` circuit; `:gatenoise` /
`:noobs` are `_xdem_circuits.variant`, a copy of `tests/test_dc_export.py::load` at commit 19f2f78), or, for
`tdag_cycle`, the circuit `TDAG_CYCLE` of `tests/test_xdem_export.py`: T, T_DAG, T, T_DAG on one qubit with an
X_ERROR before each gate and an MX detector, so one fault re-touches the axis after E = 0 mod 4 (the same circuit as
the dc fixtures' `tdag_cycle`, whose README, `tests/data/dc/README.md`, was removed in Task 11; see commit 19f2f78). A circuit
whose output is the conventional DEM (no rotating line, SPEC §3.4) has the `.dem` suffix.
`tests/test_xdem_export.py::test_fixtures_are_this_writer` re-exports each one and compares the bytes.

| file | circuit | md5 |
|---|---|---|
| `cultivation_d3_faithful.xdem` | `cultivation_d3_faithful` | `8aaad390727145d98d896babedee17d0` |
| `cultivation_d5.xdem` | `cultivation_d5` | `19c8f8a9f8a47b7040393944a02e2594` |
| `distillation_15_1_3.dem` | `distillation_15_1_3` | `90491726ca091605e6c16fffcc1b1bb8` |
| `distillation_15_1_3_gatenoise.xdem` | `distillation_15_1_3:gatenoise` | `23ce034e3304745fc7c1dc842daf13d7` |
| `color5_cultivation.xdem` | `color5_cultivation` | `af251761ef9874cc4d292b95a6857d96` |
| `miniature_oracle.xdem` | `miniature_oracle` | `3454a449a28442f34d566e218d4bb6ad` |
| `code_switching_faithful_noobs.xdem` | `code_switching_faithful:noobs` | `64b5d2120efb5ff02b69d020439ef8f5` |
| `code_switching_rate_noobs.xdem` | `code_switching_rate:noobs` | `83bea6824680b723dfce4be86dd45bea` |
| `tdag_cycle.xdem` | `tdag_cycle` | `8016de2a9378dadadbaa52a63506792e` |

## Sampler fixtures (xdem Task 6)

| file | made by | md5 |
|---|---|---|
| `synth/surface_d5r5.dem` | `str(stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3).detector_error_model()) + "\n"` (the pins' `surface_d5r5` text; the golden ctest's conventional DEM) | `321a23e1dd67d8f4c775ffe502b41c11` |
| `synth/surface_d5r5_decomposed.dem` | the same circuit's `str(....detector_error_model(decompose_errors=True)) + "\n"` (stim 1.16.0, Linux x86_64; the pins' `surface_d5r5_decomposed` text) | `c53e9492937f441ad077aa31fbf3978d` |
| `synth/surface_d5r5_flat.dem` | the same circuit's `str(....detector_error_model().flattened()) + "\n"` (stim 1.16.0, Linux x86_64; the pins' `surface_d5r5_flat` text) | `a2ea9d31a79bdebf170626ebd55e2805` |
| `synth/boundary_r64.xdem` | `tests/_xdem_synth.py::boundary_text(64, 60, 4)` (fast path at its bound) | `b0851a48c43fec699ac897087c96c94a` |
| `synth/boundary_r65.xdem` | `tests/_xdem_synth.py::boundary_text(65, 60, 4)` (general path) | `975c0d9958f95ac3180c22119a01c74f` |
| `plain_sampler_sha256.json` | byte copy of an internal baseline file of the same name (Task 0: the dc-DEM v1 sampler's plain-DEM mode at BASE 19f2f78) | `648f5013d16910acdfd3c4193a348f6c` |

`tests/test_xdem_sampler.py::test_synth_boundary_fixtures_are_the_generated_text` regenerates the two boundary texts and compares them byte for byte, on every platform. `test_synth_surface_fixture_is_the_pinned_text` compares `synth/surface_d5r5.dem`, `synth/surface_d5r5_decomposed.dem` and `synth/surface_d5r5_flat.dem` with live stim on Linux x86_64 only (stim 1.16.0 prints these probabilities with other digits elsewhere); the sampler pins read the committed texts.
