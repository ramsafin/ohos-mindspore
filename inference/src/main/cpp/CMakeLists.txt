cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

project(inference LANGUAGES C CXX)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed! Retry in a separate build directory.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building on system: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Cross-compilation: ${CMAKE_CROSSCOMPILING}")

include(options)

# --- Libraries ---
add_subdirectory(src)

# --- Testing ---
if(ENABLE_TESTING)
  enable_testing()

  include(gtest)
  include(GoogleTest)
  inference_enable_gtest()

  add_subdirectory(tests)
endif()
