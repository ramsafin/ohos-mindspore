include_guard(GLOBAL)

include(FetchContent)

option(FETCH_GTEST "Fetch GoogleTest" ON)

function(inference_enable_gtest)
#  if (CMAKE_CROSSCOMPILING)
#    message(STATUS "GoogleTest disabled (cross-compiling)")
#    return()
#  endif()

  if (NOT FETCH_GTEST)
    find_package(GTest REQUIRED)
    return()
  endif()

  message(STATUS "Fetching GoogleTest")

  # GoogleTest configuration
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP true
  )

  FetchContent_MakeAvailable(googletest)

  if (TARGET gtest)
    target_include_directories(gtest SYSTEM
      INTERFACE
        $<BUILD_INTERFACE:${googletest_SOURCE_DIR}/googletest/include>
    )
  endif()

  if (MSVC)
    foreach(tgt gtest gtest_main gmock gmock_main)
      if (TARGET ${tgt})
        target_compile_options(${tgt} PRIVATE /WX-)
      endif()
    endforeach()
  endif()
endfunction()
