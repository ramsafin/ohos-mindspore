if (NOT ENABLE_TESTING)
  return()
endif()

# Host = not cross-compiling
if (ENABLE_HOST_TESTS AND NOT CMAKE_CROSSCOMPILING)
  message(STATUS "Building tests for host")
  add_subdirectory(host)
elseif (ENABLE_HOST_TESTS)
  message(STATUS "Host tests skipped (cross-compiling)")
endif()

# Device = cross-compiling for OHOS
if (ENABLE_DEVICE_TESTS AND CMAKE_CROSSCOMPILING AND CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  message(STATUS "Building tests for device")
  add_subdirectory(device)
elseif (ENABLE_DEVICE_TESTS)
  message(STATUS "Device tests skipped (not OHOS cross-compile)")
endif()
