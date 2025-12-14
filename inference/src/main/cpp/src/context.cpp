#include "context.hpp"
#include "backend.hpp"

#include <mindspore/types.h>
#include <mindspore/context.h>

namespace inference {

MindSporeContext::MindSporeContext() 
{
    ctx = OH_AI_ContextCreate();
    
    if (ctx == nullptr) {
        // Failed to create MindSpore context
        return;
    }
    
    // MindSpore context is created
}

MindSporeContext::~MindSporeContext() 
{
    if (ctx != nullptr) {
        // Destroying MindSpore context
        OH_AI_ContextDestroy(&ctx);
        // MindSpore context is destroyed
    }
}

bool MindSporeContext::IsCreated() const 
{
    return ctx != nullptr;
}

void MindSporeContext::EnableBackend(Backend backend) {
    if (ctx == nullptr) {
        // Called on invalid context
        return;
    }
    
    OH_AI_DeviceInfoHandle info = nullptr;
    
    switch (backend) {
        case Backend::CPU:
            info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
            break;
        case Backend::NPU:
            info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_KIRIN_NPU);
            break;
        case Backend::MOCK:
            // Use mock device (no actual inference)
            break;
        default:
            // Requested not supported backend
            break;
    }
    
    if (info == nullptr) {
        // Failed to create device info for requested backend
        return;
    }
    
    OH_AI_ContextAddDeviceInfo(ctx, info);
    
    // CPU device added to context
}

} // namespace inference