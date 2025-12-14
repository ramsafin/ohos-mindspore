cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

project(App LANGUAGES C CXX)

# --- OHOS-specific ---
if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

# --- Library ---
add_library(inference SHARED
    napi_init.cpp
    src/api.cpp
    src/backend.cpp
    src/context.cpp
)

 target_include_directories(inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
 )

target_link_libraries(inference PUBLIC
    ace_napi.z
    hilog_ndk.z
    mindspore_lite_ndk
)