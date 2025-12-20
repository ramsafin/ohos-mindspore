#pragma once

#include "napi/native_api.h"

#include "inference/types.hpp"

#include <string>

namespace napi {

inline void throw_with_message(napi_env env, const std::string &msg) {
    // what is the error code (nullptr)? should we use it?
    napi_throw_error(env, nullptr, msg.c_str());
}

inline napi_value make_error(napi_env env, const std::string &msg) {
    napi_value js_error_message{};
    napi_create_string_utf8(env, msg.c_str(), NAPI_AUTO_LENGTH, &js_error_message);

    napi_value js_error{};
    napi_create_error(env, nullptr, js_error_message, &js_error);

    return js_error;
}

inline bool get_property(napi_env env, napi_value js_object, const char *name, napi_value *out) {
    return napi_get_named_property(env, js_object, name, out) == napi_ok;
}

inline bool get_string(napi_env env, napi_value js_string, std::string &out) {
    napi_valuetype js_type = napi_string;
    if (napi_typeof(env, js_string, &js_type) != napi_ok || js_type != napi_string) {
        return false;
    }

    size_t length = 0;
    if (napi_get_value_string_utf8(env, js_string, nullptr, 0, &length) != napi_ok) {
        return false;
    }

    std::string tmp(length + 1, '\0');
    if (napi_get_value_string_utf8(env, js_string, tmp.data(), tmp.size(), &length) != napi_ok) {
        return false;
    }

    out.assign(tmp.data(), length); // exclude '\0'
    return true;
}

inline bool parse_shape(napi_env env, napi_value js_shape, inference::Shape &shape, std::string &err) {
    // shape: Uint32Array

    bool is_typed_arr = false;
    if (napi_is_typedarray(env, js_shape, &is_typed_arr) != napi_ok || !is_typed_arr) {
        err = "shape must be Uint32Array";
        return false;
    }

    napi_typedarray_type js_type{};
    size_t length = 0;
    void *data = nullptr;
    napi_value js_arr_buffer{};
    size_t byte_offset = 0;

    if (napi_get_typedarray_info(env, js_shape, &js_type, &length, &data, &js_arr_buffer, &byte_offset) != napi_ok) {
        err = "napi_get_typedarray_info failed for shape";
        return false;
    }


    if (js_type != napi_uint32_array) {
        err = "shape must be Uint32Array";
        return false;
    }

    shape.resize(length);

    if (length > 0) {
        std::memcpy(shape.data(), data, length * sizeof(std::uint32_t));
    }

    return true;
}

inline napi_value make_shape(napi_env env, const inference::Shape &shape) {
    const size_t length = shape.size();
    const size_t bytes = length * sizeof(std::uint32_t);

    void *data = nullptr;
    napi_value js_arr_buffer{};

    if (napi_create_arraybuffer(env, bytes, &data, &js_arr_buffer) != napi_ok) {
        // todo: handle array buffer creation error
        return nullptr;
    }

    if (bytes > 0) {
        std::memcpy(data, shape.data(), bytes);
    }

    napi_value js_shape = nullptr;
    if (napi_create_typedarray(env, napi_uint32_array, length, js_arr_buffer, 0, &js_shape) != napi_ok) {
        // todo: handle typed array creation error
        return nullptr;
    }

    return js_shape;
}

inline bool parse_model_config(napi_env env, napi_value js_config, inference::ModelConfig &config, std::string &err) {
    // js_config: { device: string, modelData: ArrayBuffer }

    // device: string
    napi_value js_device{};

    if (!get_property(env, js_config, "device", &js_device)) {
        err = "ModelConfig must have { device }";
        return false;
    }

    if (!get_string(env, js_device, config.device)) {
        err = "ModelConfig.device must be a string";
        return false;
    }

    // modelData: ArrayBuffer (binary data)
    napi_value js_model_data{};
    if (!get_property(env, js_config, "modelData", &js_model_data)) {
        err = "ModelConfig must have { modelData }";
        return false;
    }

    bool is_array_buffer = false;
    if (napi_is_arraybuffer(env, js_model_data, &is_array_buffer) != napi_ok) {
        return false;
    }

    if (!is_array_buffer) {
        err = "ModelConfig.modelData must be an ArrayBuffer";
        return false;
    }

    void *data = nullptr;
    size_t byte_length = 0;
    if (napi_get_arraybuffer_info(env, js_model_data, &data, &byte_length) != napi_ok) {
        err = "napi_get_arraybuffer_info failed";
        return false;
    }

    config.model_data.resize(byte_length);

    if (byte_length > 0) {
        std::memcpy(config.model_data.data(), data, byte_length);
    }

    return true;
}

inline bool parse_tensor(napi_env env, napi_value js_tensor, inference::Tensor &tensor, std::string &err) {
    // obj: {shape: number[], data: Float32Array}

    // shape: number[]
    napi_value js_shape{};
    if (!get_property(env, js_tensor, "shape", &js_shape)) {
        err = "InputTensor must have { shape }";
        return false;
    }

    if (!parse_shape(env, js_shape, tensor.shape, err)) {
        return false;
    }

    // data (tensor)
    napi_value js_data{};
    if (!get_property(env, js_tensor, "data", &js_data)) {
        err = "InputTensor must have { data }";
        return false;
    }

    bool is_typed_array = false;
    if (napi_is_typedarray(env, js_data, &is_typed_array) != napi_ok) {
        return false;
    }

    if (!is_typed_array) {
        err = "InputTensor.data must be a Float32Array";
        return false;
    }

    napi_typedarray_type js_arr_type = napi_float32_array;
    size_t length = 0; // bug: length could be the number of elements or bytes (need to work around using array buffer)
    void *data = nullptr;
    napi_value js_arr_buffer{};
    size_t byte_offset = 0; // optional?

    if (napi_get_typedarray_info(env, js_data, &js_arr_type, &length, &data, &js_arr_buffer, &byte_offset) != napi_ok) {
        return false;
    }

    if (js_arr_type != napi_float32_array) {
        err = "InputTensor.data must be Float32Array";
        return false;
    }

    tensor.data.resize(length / sizeof(float)); // resize to the actual length, not number of bytes

    if (length > 0) {
        std::memcpy(tensor.data.data(), data, length);
    }

    return true;
}

inline napi_value make_tensor(napi_env env, const inference::Tensor &tensor) {
    // allocate native buffer and attach to external ArrayBuffer finalizer (destructor)
    const size_t length = tensor.data.size();
    auto *heap = new float[length]{}; // non-owning buffer (JS-owned)

    if (length > 0) {
        // why not use std::copy? it should be the modern way of doing this
        std::memcpy(heap, tensor.data.data(), length * sizeof(float));
    }

    const auto deleter = [](napi_env /*env*/, void *data, void * /*hint*/) { delete[] static_cast<float *>(data); };
    napi_value js_arr_buffer{};
    napi_create_external_arraybuffer(env, heap, length * sizeof(float), deleter, nullptr, &js_arr_buffer);

    napi_value js_data{};
    napi_create_typedarray(env, napi_float32_array, length, js_arr_buffer, 0, &js_data);

    napi_value js_shape = make_shape(env, tensor.shape);

    napi_value js_tensor{};
    napi_create_object(env, &js_tensor);
    napi_set_named_property(env, js_tensor, "data", js_data);
    napi_set_named_property(env, js_tensor, "shape", js_shape);

    return js_tensor;
}

inline inference::TensorView as_view(const inference::Tensor &tensor) {
    return {                       // clear structure creation (.field = data)
            .shape = tensor.shape, // consider using view here
            .data = std::span<const float>{tensor.data.data(), tensor.data.size()}};
}

} // namespace napi
