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
qeccore_test(test_pe_budget_message)
add_test(NAME pe_budget_message COMMAND test_pe_budget_message)

qeccore_test(test_kernel_dual_basis)
add_test(NAME kernel_dual_basis COMMAND test_kernel_dual_basis)

qeccore_test(test_bare_state_memo)
target_compile_definitions(test_bare_state_memo PRIVATE
  XTIM_EXAMPLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../examples")
add_test(NAME bare_state_memo COMMAND test_bare_state_memo)

qeccore_test(test_ref_battery)
add_test(NAME ref_battery COMMAND test_ref_battery)
