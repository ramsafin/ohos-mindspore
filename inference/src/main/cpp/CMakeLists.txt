cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

project(App LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Library ---
add_library(inference SHARED
    napi_init.cpp
)

 target_include_directories(inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
 )

 if(DEFINED PACKAGE_FIND_FILE)
 target_include_directories(inference PUBLIC
    ${PACKAGE_FIND_FILE}
 )
 endif()

target_link_libraries(inference PUBLIC
    ace_napi.z
    hilog_ndk.z
    mindspore_lite_ndk
)