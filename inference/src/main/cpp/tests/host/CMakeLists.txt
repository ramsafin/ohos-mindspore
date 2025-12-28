include(FetchContent)


set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

FetchContent_MakeAvailable(googletest)

target_compile_options(gtest PRIVATE /WX-)
target_compile_options(gtest_main PRIVATE /WX-)

add_executable(unit_tests_host
  test_main.cpp
  test_shape.cpp
)

target_link_libraries(unit_tests_host
  PRIVATE
    GTest::gtest_main
    ${PROJECT_NAME}_core
)

include(GoogleTest)
gtest_discover_tests(unit_tests_host)
