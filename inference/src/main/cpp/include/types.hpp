#ifndef APP_TYPES_HPP
#define APP_TYPES_HPP

#include "napi/native_api.h"

#include <vector>
#include <string>

#include "rawfile/raw_file_manager.h"
#include "resourcemanager/ohresmgr.h"

namespace inference {

struct WorkData {
    napi_env env;
    napi_async_work work;
    napi_deferred deferred;
    
    NativeResourceManager *nativeResMgr;
    
    // payload
    std::vector<double> input;
    std::vector<double> output;
    
    bool ok{true};
    std::string error;
};

} // namespace inference

#endif //APP_TYPES_HPP
