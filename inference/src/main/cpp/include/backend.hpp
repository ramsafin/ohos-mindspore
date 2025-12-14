#ifndef APP_BACKEND_HPP
#define APP_BACKEND_HPP

#include "napi/native_api.h" // napi_env

namespace inference {

enum class Backend {
  MOCK,
  CPU,
  NPU
};

void Execute_MOCK(napi_env env, void* data);

void Execute_CPU(napi_env env, void* data);

} // namespace inference

#endif //APP_BACKEND_HPP
