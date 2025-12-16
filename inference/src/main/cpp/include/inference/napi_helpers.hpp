#pragma once

#include "napi/native_api.h"

#include "inference/types.hpp"

#include <string>

namespace napi {

inline void throw_with_message(napi_env env, const std::string &msg) { napi_throw_error(env, nullptr, msg.c_str()); }

inline napi_value make_error(napi_env env, const std::string &msg) {
    napi_value m = nullptr;
    napi_value err = nullptr;
    napi_create_string_utf8(env, msg.c_str(), NAPI_AUTO_LENGTH, &m);
    napi_create_error(env, nullptr, m, &err);
    return err;
}

inline bool get_named(napi_env env, napi_value obj, const char *name, napi_value *out) {
    return napi_get_named_property(env, obj, name, out) == napi_ok;
}

inline bool get_string(napi_env env, napi_value v, std::string &out) {
    size_t len = 0;
    if (napi_get_value_string_utf8(env, v, nullptr, 0, &len) != napi_ok) {
        return false;
    }
    out.resize(len);
    return napi_get_value_string_utf8(env, v, out.data(), out.size() + 1, &len) == napi_ok;
}

inline bool get_shape_u32(napi_env env, napi_value arr, inference::Shape &shape) {
    bool is_array = false;
    if (napi_is_array(env, arr, &is_array) != napi_ok || !is_array) {
        return false;
    }

    uint32_t n = 0;
    if (napi_get_array_length(env, arr, &n) != napi_ok) {
        return false;
    }

    shape.clear();
    shape.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        napi_value el{};
        if (napi_get_element(env, arr, i, &el) != napi_ok) {
            return false;
        }

        uint32_t v = 0;
        if (napi_get_value_uint32(env, el, &v) != napi_ok) {
            return false;
        }

        shape.push_back(v);
    }

    return true;
}

inline bool parse_model_config(napi_env env, napi_value obj, inference::ModelConfig &cfg, std::string &err) {
    napi_value device_v{};
    napi_value model_v{};

    if (!get_named(env, obj, "device", &device_v)) {
        err = "ModelConfig must have { device }";
        return false;
    }

    if (!get_named(env, obj, "modelData", &model_v)) {
        err = "ModelConfig must have { modelData }";
        return false;
    }

    if (!get_string(env, device_v, cfg.device)) {
        err = "ModelConfig.device must be a string";
        return false;
    }

    bool is_array_buffer = false;
    if (napi_is_arraybuffer(env, model_v, &is_array_buffer) != napi_ok) {
        return false;
    }

    if (!is_array_buffer) {
        err = "ModelConfig.modelData must be an ArrayBuffer";
        return false;
    }

    void *data_ptr = nullptr;
    size_t byte_len = 0;
    if (napi_get_arraybuffer_info(env, model_v, &data_ptr, &byte_len) != napi_ok) {
        return false;
    }

    cfg.model_data.resize(byte_len);
    if (byte_len > 0) {
        std::memcpy(cfg.model_data.data(), data_ptr, byte_len);
    }

    return true;
}

inline bool parse_input_tensor(napi_env env, napi_value obj, inference::Tensor &in, std::string &err) {
    napi_value data_v{};
    napi_value shape_v{};
    if (!get_named(env, obj, "data", &data_v)) {
        err = "InputTensor must have { data }";
        return false;
    }

    if (!get_named(env, obj, "shape", &shape_v)) {
        err = "InputTensor must have { shape }";
        return false;
    }

    if (!get_shape_u32(env, shape_v, in.shape)) {
        err = "InputTensor.shape must be number[] (uint32)";
        return false;
    }

    bool is_typed_array = false;
    if (napi_is_typedarray(env, data_v, &is_typed_array) != napi_ok) {
        return false;
    }

    if (!is_typed_array) {
        err = "InputTensor.data must be a Float32Array";
        return false;
    }

    napi_typedarray_type array_type{};
    size_t length = 0;
    void *data_ptr = nullptr;
    napi_value array_buffer{};
    size_t byte_offset = 0;

    if (napi_get_typedarray_info(env, data_v, &array_type, &length, &data_ptr, &array_buffer, &byte_offset) !=
        napi_ok) {
        return false;
    }

    if (array_type != napi_float32_array) {
        err = "InputTensor.data must be Float32Array";
        return false;
    }

    in.data.resize(length);

    if (length > 0) {
        std::memcpy(in.data.data(), data_ptr, length * sizeof(float));
    }

    return true;
}

inline napi_value make_output_tensor(napi_env env, const inference::Tensor &out) {
    // allocate native buffer and attach to external ArrayBuffer finalizer
    const size_t n = out.data.size();
    auto *heap = new float[n];
    if (n > 0) {
        std::memcpy(heap, out.data.data(), n * sizeof(float));
    }

    napi_value array_buffer = nullptr;
    napi_create_external_arraybuffer(
        env, heap, n * sizeof(float),
        [](napi_env /*env*/, void *data, void * /*hint*/) { delete[] static_cast<float *>(data); }, nullptr,
        &array_buffer);

    napi_value typed_array = nullptr;
    napi_create_typedarray(env, napi_float32_array, n, array_buffer, 0, &typed_array);

    // shape JS array
    napi_value shape_arr = nullptr;
    napi_create_array_with_length(env, out.shape.size(), &shape_arr);

    for (uint32_t i = 0; i < out.shape.size(); i++) {
        napi_value v{};
        napi_create_uint32(env, out.shape[i], &v);
        napi_set_element(env, shape_arr, i, v);
    }

    napi_value obj = nullptr;
    napi_create_object(env, &obj);
    napi_set_named_property(env, obj, "data", typed_array);
    napi_set_named_property(env, obj, "shape", shape_arr);
    return obj;
}

} // namespace napi
