#ifndef APP_API_HPP
#define APP_API_HPP

#include "napi/native_api.h"  // napi_value, napi_env, napi_callback_info

namespace inference {

napi_value RunInference(napi_env env, napi_callback_info info);

} // inference

#endif //APP_API_HPP
