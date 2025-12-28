if (CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR "Host tests must not be built when cross-compiling")
endif()

add_executable(unit_tests_host
  test_main.cpp
  test_shape.cpp
)

target_link_libraries(unit_tests_host
  PRIVATE
    GTest::gtest
    ${PROJECT_NAME}_core
)

target_compile_features(unit_tests_host
  PRIVATE cxx_std_${CMAKE_CXX_STANDARD}
)

gtest_discover_tests(unit_tests_host
  PROPERTIES
    LABELS "host"
)
