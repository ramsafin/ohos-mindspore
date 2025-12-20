#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

#include "inference/types.hpp"

// MindSpore Lite
#include <mindspore/context.h>
#include <mindspore/model.h>
#include <mindspore/tensor.h>
#include <mindspore/types.h>
#include <mindspore/status.h>

namespace inference {

struct MSContext final {
    OH_AI_ContextHandle handle{nullptr};

    MSContext() {
        handle = OH_AI_ContextCreate();
        if (!handle) {
            throw std::runtime_error("OH_AI_ContextCreate returned null");
        }
    }

    ~MSContext() {
        if (handle) {
            OH_AI_ContextDestroy(&handle);
        }
    }
};

struct MSDeviceInfo final {
    OH_AI_DeviceInfoHandle handle{nullptr};

    explicit MSDeviceInfo(OH_AI_DeviceType type) {
        handle = OH_AI_DeviceInfoCreate(type);

        if (!handle) {
            throw std::runtime_error("OH_AI_DeviceInfoCreate returned null");
        }
    }

    ~MSDeviceInfo() {
        if (handle) {
//            OH_AI_DeviceInfoDestroy(&handle);
        }
    }
};

struct MSModel final {
    OH_AI_ModelHandle handle{nullptr};

    MSModel() {
        handle = OH_AI_ModelCreate();
        if (!handle) {
            throw std::runtime_error("OH_AI_ModelCreate returned null");
        }
    }
    ~MSModel() {
        if (handle) {
            OH_AI_ModelDestroy(&handle);
        }
    }
};

inline void check(OH_AI_Status status, const char *what) {
    if (status != OH_AI_STATUS_SUCCESS) {
        std::ostringstream os;
        os << what << " failed, status=0x" << std::hex << static_cast<uint32_t>(status);
        throw std::runtime_error(os.str());
    }
}

inline std::string shape_to_string(const int64_t *shape, size_t num) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < num; i++) {
        if (i) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "]";
    return os.str();
}

inline void dump_tensor(OH_AI_TensorHandle tensor, const char *kind, size_t idx) {
    const char *name = OH_AI_TensorGetName(tensor);
    OH_AI_DataType data_type = OH_AI_TensorGetDataType(tensor);

    size_t shape_num = 0;
    const int64_t *shape = OH_AI_TensorGetShape(tensor, &shape_num);

    size_t bytes = OH_AI_TensorGetDataSize(tensor);

    std::ostringstream os;

    os << kind << "[" << idx << "] "
       << "name=" << (name ? name : "<null>") << " dtype=" << static_cast<int>(data_type)
       << " shape=" << (shape ? shape_to_string(shape, shape_num) : "<null>") << " bytes=" << bytes;

    auto str = os.str();

    std::cerr << str << std::endl;
}

struct Context final {
public:
    explicit Context(ModelConfig config) : config_{std::move(config)} {}

    Tensor run(const TensorView &in) {
        [[maybe_unused]] std::scoped_lock lock{mutex_};

        MSContext ctx;
        OH_AI_ContextSetThreadNum(ctx.handle, 1);
        OH_AI_ContextSetThreadAffinityMode(ctx.handle, 0);

        MSDeviceInfo cpu_device{OH_AI_DEVICETYPE_CPU};
        OH_AI_ContextAddDeviceInfo(ctx.handle, cpu_device.handle);

        // build model from raw buffer
        MSModel model;
        auto status = OH_AI_ModelBuild(model.handle, config_.model_data.data(), config_.model_data.size(),
                                       OH_AI_MODELTYPE_MINDIR, ctx.handle);

        check(status, "OH_AI_ModelBuild");


        // Fetch model inputs
        auto inputs = OH_AI_ModelGetInputs(model.handle);

        if (inputs.handle_num != 1 || inputs.handle_list == nullptr) {
            throw std::runtime_error("Expected exactly 1 input tensor");
        }

        OH_AI_TensorHandle input0 = inputs.handle_list[0];

        // Validate dtype
        const OH_AI_DataType dt = OH_AI_TensorGetDataType(input0);
        if (dt != OH_AI_DATATYPE_NUMBERTYPE_FLOAT32) { // name may vary; use your header’s exact enum
            throw std::runtime_error("Model input dtype is not float32");
        }

        // Expect [1,3,224,224]
        size_t shape_rank = 0;
        const int64_t *shape_ptr = OH_AI_TensorGetShape(input0, &shape_rank);

        if (shape_rank != 4 || shape_ptr[0] != 1 || shape_ptr[1] != 3 || shape_ptr[2] != 224 || shape_ptr[3] != 224) {
            throw std::runtime_error("Model input shape mismatch");
        }

        // Validate buffer sizes and copy input
        const size_t expected_elems = 1 * 3 * 224 * 224;
        if (in.data.size() != expected_elems) {
            throw std::runtime_error("Input length mismatch (expected 150528 floats)");
        }

        void *dst = OH_AI_TensorGetMutableData(input0);
        const size_t dst_bytes = OH_AI_TensorGetDataSize(input0);
        if (!dst || dst_bytes != expected_elems * sizeof(float)) {
            throw std::runtime_error("Input tensor buffer invalid size");
        }

        std::memcpy(dst, in.data.data(), dst_bytes);

        // Predict
        OH_AI_TensorHandleArray outputs{};
        check(OH_AI_ModelPredict(model.handle, inputs, &outputs, nullptr, nullptr), "OH_AI_ModelPredict");

        if (outputs.handle_num != 1 || outputs.handle_list == nullptr) {
            throw std::runtime_error("Expected exactly 1 output tensor");
        }

        OH_AI_TensorHandle out0 = outputs.handle_list[0];

        // 6) Read output and copy to our Tensor
        const size_t out_bytes = OH_AI_TensorGetDataSize(out0);
        const void *out_ptr = OH_AI_TensorGetData(out0);

        if (!out_ptr || out_bytes != 1000 * sizeof(float)) {
            throw std::runtime_error("Output tensor buffer invalid size");
        }

        Tensor out;
        out.shape = {1, 1000};
        out.data.resize(1000);
        std::memcpy(out.data.data(), out_ptr, out_bytes);

        return out;
    }

private:
    mutable std::mutex mutex_;
    ModelConfig config_;
};

} // namespace inference