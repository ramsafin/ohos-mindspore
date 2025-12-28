# Host = not cross-compiling
if (ENABLE_HOST_TESTS AND NOT CMAKE_CROSSCOMPILING)
  add_subdirectory(host)
endif()

# Device = cross-compiling for OHOS
if (ENABLE_DEVICE_TESTS AND CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  add_subdirectory(device)
endif()
