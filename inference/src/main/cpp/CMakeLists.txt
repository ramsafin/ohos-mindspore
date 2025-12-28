cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

project(inference LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_TESTING "Build unit tests" ON)
option(ENABLE_HOST_TESTS "Build host unit tests" ON)
option(ENABLE_DEVICE_TESTS "Build device unit tests" ON)

# --- Library (core) ---
add_library(${PROJECT_NAME}_core SHARED
  src/core.cpp
)

target_include_directories(${PROJECT_NAME}_core
  PUBLIC ${CMAKE_SOURCE_DIR}/include
)

message(STATUS "Building on system: ${CMAKE_SYSTEM_NAME}")

if (CMAKE_SYSTEM_NAME STREQUAL "OHOS")
#  target_link_libraries(inference
#    PUBLIC
#      ace_napi.z
#      hilog_ndk.z
#      mindspore_lite_ndk
#  )

#  if(DEFINED PACKAGE_FIND_FILE)
#  target_include_directories(${PROJECT_NAME}
#    PUBLIC ${PACKAGE_FIND_FILE}
#  )
#  endif()
endif()

# --- Testing ---
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
