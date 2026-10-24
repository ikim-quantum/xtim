cmake_minimum_required(VERSION 3.16)
project(qeccore_xtim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(QECCORE_PORTABLE "Drop -march=native for reproducible builds" OFF)
add_compile_options(-O3 -funroll-loops -Wall -Wextra)
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

qeccore_test(test_channel_support_atoms)
add_test(NAME channel_support_atoms COMMAND test_channel_support_atoms)

qeccore_test(test_dem_factored_law)
add_test(NAME dem_factored_law COMMAND test_dem_factored_law)
qeccore_test(test_dem_fault_law)
add_test(NAME dem_fault_law COMMAND test_dem_fault_law)
qeccore_test(test_dem_general_law)
add_test(NAME dem_general_law COMMAND test_dem_general_law)
qeccore_test(test_pe_budget_message)
add_test(NAME pe_budget_message COMMAND test_pe_budget_message)

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
qeccore_test(test_record_table_canonical)
add_test(NAME record_table_canonical COMMAND test_record_table_canonical)
qeccore_test(test_materialize_anchor)
add_test(NAME materialize_anchor COMMAND test_materialize_anchor)
qeccore_test(test_amp_index)
add_test(NAME amp_index COMMAND test_amp_index)
qeccore_test(test_pauli_kernels)
add_test(NAME pauli_kernels COMMAND test_pauli_kernels)
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
