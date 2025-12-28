add_library(${PROJECT_NAME}_core SHARED
  core/core.cpp
)

add_library(${PROJECT_NAME}::core
  ALIAS ${PROJECT_NAME}_core
)

target_include_directories(${PROJECT_NAME}_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(${PROJECT_NAME}_core
  PUBLIC cxx_std_${CMAKE_CXX_STANDARD}
)

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
