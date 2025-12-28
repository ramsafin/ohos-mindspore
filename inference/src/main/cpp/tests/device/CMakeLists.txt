if (NOT CMAKE_CROSSCOMPILING OR NOT CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  message(FATAL_ERROR "Device tests can only be built when cross-compiling for OHOS")
endif()

add_executable(unit_tests_device
  test_main.cpp
)

target_link_libraries(unit_tests_device
  PRIVATE
    ${PROJECT_NAME}_core
)

target_compile_features(unit_tests_device
  PRIVATE cxx_std_${CMAKE_CXX_STANDARD}
)

add_test(
  NAME unit_tests_device
  COMMAND unit_tests_device
)

set_tests_properties(unit_tests_device
  PROPERTIES
    LABELS "device"
)
