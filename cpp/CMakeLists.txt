cmake_minimum_required(VERSION 3.16)
project(qeccore_xtim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(QECCORE_PORTABLE "Drop -march=native for reproducible builds" OFF)
# -ffp-contract=off: no FMA contraction, so a native build's float results (exported p) equal the portable build's
# bit for bit (setup.py _FP_ARGS explains).
add_compile_options(-O3 -funroll-loops -ffp-contract=off -Wall -Wextra)
if(NOT QECCORE_PORTABLE)
  add_compile_options(-march=native)
endif()

find_package(Threads REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# All engine sources (mirrors setup.py's glob of cpp/src/*.cpp).
file(GLOB QECCORE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(qeccore STATIC ${QECCORE_SRCS})
target_include_directories(qeccore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qeccore PUBLIC Threads::Threads)
# Test build: turn on cheap-but-not-free invariant checks (AffineState full-column-rank check
# on apply_h's canonicalize() skip path). setup.py (the shipped _xtim) does NOT define it.
target_compile_definitions(qeccore PUBLIC QECCORE_CHECK_INVARIANTS)

enable_testing()

# Helper: declare a test executable linked against qeccore + tests/ headers.
# This pattern mirrors qec_library/cpp/CMakeLists.txt.
function(qeccore_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE tests)
  # tests/ may textually #include apps/ sources (e.g. ref_compile.cpp).
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE qeccore)
endfunction()

qeccore_test(test_set_seed_parity)
add_test(NAME seed_parity COMMAND test_set_seed_parity)

# threads (fixplan threads/fixplan.md G1 / G2): run(shots, threads) == run(shots) byte for byte; the TSan gate's driver.
qeccore_test(test_threads)
target_compile_definitions(test_threads PRIVATE XTIM_TESTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests")
add_test(NAME threads COMMAND test_threads)

qeccore_test(test_planes_walk)
add_test(NAME planes_walk COMMAND test_planes_walk)

# T6: CertifiedGroupPlanes::build / ProvenanceRREF on set bits == the dense oracles (random sets + the corpus).
qeccore_test(test_planes_build)
target_compile_definitions(test_planes_build PRIVATE XTIM_TESTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests")
add_test(NAME planes_build COMMAND test_planes_build)
qeccore_test(test_planes_sparse)
target_compile_definitions(test_planes_sparse PRIVATE XTIM_TESTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests")
add_test(NAME planes_sparse COMMAND test_planes_sparse)

qeccore_test(test_inner_product_form)
add_test(NAME inner_product_form COMMAND test_inner_product_form)

qeccore_test(test_amp_tiny)
add_test(NAME amp_tiny COMMAND test_amp_tiny)

qeccore_test(test_keyed_rng)
add_test(NAME keyed_rng COMMAND test_keyed_rng)

qeccore_test(test_coherentize_all)
add_test(NAME coherentize_all COMMAND test_coherentize_all)

qeccore_test(test_frame_cosets)
add_test(NAME frame_cosets COMMAND test_frame_cosets)

qeccore_test(test_bare_state_guards)
add_test(NAME bare_state_guards COMMAND test_bare_state_guards)

qeccore_test(test_fast_todd_wide)
add_test(NAME fast_todd_wide COMMAND test_fast_todd_wide)

qeccore_test(test_exact_engine_audit)
add_test(NAME exact_engine_audit COMMAND test_exact_engine_audit)

qeccore_test(test_restrict_to_support)
add_test(NAME restrict_to_support COMMAND test_restrict_to_support)
qeccore_test(test_prepared_affine)
add_test(NAME prepared_affine COMMAND test_prepared_affine)
qeccore_test(test_trivial_completion)
add_test(NAME trivial_completion COMMAND test_trivial_completion)

qeccore_test(test_channel_support_atoms)
add_test(NAME channel_support_atoms COMMAND test_channel_support_atoms)

qeccore_test(test_dem_fault_law)
add_test(NAME dem_fault_law COMMAND test_dem_fault_law)
qeccore_test(test_dem_general_law)
add_test(NAME dem_general_law COMMAND test_dem_general_law)
qeccore_test(test_pe_structure_split)
add_test(NAME pe_structure_split COMMAND test_pe_structure_split)

# M2 (magic lane): the DEM export's sparse per-alternative composition (CZ layers included) == the dense chain,
# field for field and byte for byte on the magic corpus (set_dem_compose_mode_for_test).
qeccore_test(test_dem_compose_cz)
add_test(NAME dem_compose_cz COMMAND test_dem_compose_cz ${CMAKE_CURRENT_SOURCE_DIR}/..)

qeccore_test(test_tableau_lean)
add_test(NAME tableau_lean COMMAND test_tableau_lean)
qeccore_test(test_bare_state_memo)
target_compile_definitions(test_bare_state_memo PRIVATE
  XTIM_EXAMPLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/data")
add_test(NAME bare_state_memo COMMAND test_bare_state_memo)

qeccore_test(test_ref_battery)
add_test(NAME ref_battery COMMAND test_ref_battery)

qeccore_test(test_born_trie)
add_test(NAME born_trie COMMAND test_born_trie)
qeccore_test(test_plan_cache_model)
add_test(NAME plan_cache_model COMMAND test_plan_cache_model)
# plancache-design (K2' / Π): the kernel compiler against the exact-pullback oracle (O3's independent twin)
# plancache-design §4 (O5): the slab arena's ownership invariants, the blob round trip
qeccore_test(test_plan_arena)
add_test(NAME plan_arena COMMAND test_plan_arena)
qeccore_test(test_record_table_canonical)
add_test(NAME record_table_canonical COMMAND test_record_table_canonical)
qeccore_test(test_materialize_anchor)
add_test(NAME materialize_anchor COMMAND test_materialize_anchor)

# lean D1-S1: the rotation records' one-time decode-vs-recompute verification is a NAMED refusal (the test seam corrupts a word)
qeccore_test(test_rotation_verify)
add_test(NAME rotation_verify COMMAND test_rotation_verify)
qeccore_test(test_amp_index)
add_test(NAME amp_index COMMAND test_amp_index)
qeccore_test(test_pauli_kernels)
add_test(NAME pauli_kernels COMMAND test_pauli_kernels)
# S2-T1: SparsePauli conversions + kernels vs the dense Pauli routines (oracle).
qeccore_test(test_sparse_pauli)
add_test(NAME sparse_pauli COMMAND test_sparse_pauli)
# S2-T2: the compact Frame vs today's dense CliffordTableau (random frames; the bundled corpus from the two directories).
qeccore_test(test_frame)
add_test(NAME frame COMMAND test_frame ${CMAKE_CURRENT_SOURCE_DIR}/../examples ${CMAKE_CURRENT_SOURCE_DIR}/../tests/data)
# S2-T3: the compact bare-state frame vs the Stage-1 tableau (coset_rays oracle) on the corpus + the private corpus.
qeccore_test(test_frame_compact_build)
add_test(NAME frame_compact_build COMMAND test_frame_compact_build ${CMAKE_CURRENT_SOURCE_DIR}/../examples
         ${CMAKE_CURRENT_SOURCE_DIR}/../tests/data ${CMAKE_CURRENT_SOURCE_DIR}/../tests/private)
qeccore_test(test_framed_kernels)
add_test(NAME framed_kernels COMMAND test_framed_kernels)
qeccore_test(test_tlc_scratch)
add_test(NAME tlc_scratch COMMAND test_tlc_scratch)
# J-a N2: the term-indexed residual (G-TK2 kernel fold, G-TK6 commutation assert, dense composition).
qeccore_test(test_term_residual)
add_test(NAME term_residual COMMAND test_term_residual)
# J-a N3: the (G) builder's pullback (dense oracle; == diag_pullback on the diagonal plans' term form).
qeccore_test(test_record_general)
add_test(NAME record_general COMMAND test_record_general)
# J-a N6 (J-a-b): record-space B == the sigma dual basis (G-B1 unit half); B work independent of inactive padding (G-B2).
qeccore_test(test_record_space_b)
add_test(NAME record_space_b COMMAND test_record_space_b)
# J-a N7 (J-a-b): branch scaffolding -- E_S (G-ES, es_four_terms), W_p == C0 g_p C0^dag (G-W unit), digest + full-key resolve (G-DIG), lazy E_S (G-LAZY).
qeccore_test(test_branch_scaffold)
add_test(NAME branch_scaffold COMMAND test_branch_scaffold)
# S2b (low-mask guard -> Debug assert): the death tests run only where asserts are compiled in. In a
# Debug configuration (cmake -DCMAKE_BUILD_TYPE=Debug) this test is registered and must run and pass;
# the binary fails `--require-debug` if NDEBUG is defined, so the Debug gate can never pass vacuously.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_test(NAME framed_kernels_debug_asserts COMMAND test_framed_kernels --require-debug)
endif()

qeccore_test(test_record_fidelity_ref)
add_test(NAME record_fidelity_ref COMMAND test_record_fidelity_ref)
# S8 RF2: the closed-form Pauli expansion of a kind-0 content vs brute force (+ padding invariance, dressing, refusal).
qeccore_test(test_rf_expansion)
add_test(NAME rf_expansion COMMAND test_rf_expansion)
# explaw E1: the coset-of-X-span Born law vs dense brute force and BornTrie::enumerate (same leaves), both routes.
qeccore_test(test_born_law_coset)
add_test(NAME born_law_coset COMMAND test_born_law_coset)
# xdem Task 2 (SPEC §3, §1.2, §1.3): the §3.7 file, compose, the emitter fixed point, a refusal.
qeccore_test(test_xdem_file)
add_test(NAME xdem_file COMMAND test_xdem_file)
# xdem Task 3 (SPEC §2): canonical coordinates on ex422, bhat = beta on K, the strict tau invariant, the E sign.
qeccore_test(test_xdem_writer)
add_test(NAME xdem_writer COMMAND test_xdem_writer ${CMAKE_CURRENT_SOURCE_DIR}/../xtim/_examples)
# xdem Task 6 (SPEC §1.2, §4.7): ordered composition (Review Focus 1), stream identity (threads, prefix, first_block).
qeccore_test(test_xdem_sampler)
add_test(NAME xdem_sampler COMMAND test_xdem_sampler ${CMAKE_CURRENT_SOURCE_DIR}/../tests/data/xdem)
# xdem Task 6: FNV references (the conventional surface d5r5 text = the v1 sampler's plain-DEM bytes at BASE; xdem
# files pinned).
qeccore_test(test_xdem_sampler_golden)
add_test(NAME xdem_sampler_golden COMMAND test_xdem_sampler_golden ${CMAKE_CURRENT_SOURCE_DIR}/../tests/data/xdem)
# xdem Task 11 (moved from the dc-DEM route): the canonical-form check (C1'/C2' verdicts, require_canonical on an
# extended export) and the parser's source provenance.
qeccore_test(test_xdem_canonical)
add_test(NAME xdem_canonical COMMAND test_xdem_canonical
         ${CMAKE_CURRENT_SOURCE_DIR}/../tests/data/xdem/circuits/ex422_obs.stim)
# xdem Task 11 (moved from the dc-DEM route, its final review I1): the C table read under the sampler's lock while
# another thread samples (run it under -fsanitize=thread / address for the race check).
qeccore_test(test_xdem_sampler_concurrency)
add_test(NAME xdem_sampler_concurrency COMMAND test_xdem_sampler_concurrency)
# xdem Task 12 (SPEC §4.7): order-of-magnitude speed guard (caps 10x the measured best-of-3 µs/shot; immune to load).
qeccore_test(test_xdem_speed_tripwire)
add_test(NAME xdem_speed_tripwire COMMAND test_xdem_speed_tripwire ${CMAKE_CURRENT_SOURCE_DIR}/../tests/data/xdem)
# LEAN-EXPORT-PLAN: lean xdem route == oracle route; the lean pieces' unit oracles.
qeccore_test(test_xdem_lean)
add_test(NAME xdem_lean COMMAND test_xdem_lean ${CMAKE_CURRENT_SOURCE_DIR}/../xtim/_examples
         ${CMAKE_CURRENT_SOURCE_DIR}/../tests/data/xdem)
# output-state helpers (internal design note)
qeccore_test(test_output_state)
add_test(NAME output_state COMMAND test_output_state)
