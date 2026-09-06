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
