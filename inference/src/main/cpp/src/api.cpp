#include "api.hpp"

#include "backend.hpp"
#include "types.hpp"

#include <vector>

namespace inference {

static void Complete(napi_env env, napi_status, void *data) {
    auto *d = static_cast<WorkData *>(data);

    napi_value result;
    napi_create_object(env, &result);

    napi_value ok;
    napi_get_boolean(env, d->ok, &ok);
    napi_set_named_property(env, result, "ok", ok);

    if (d->ok) {
        napi_value out;
        napi_create_array_with_length(env, d->output.size(), &out);

        for (size_t i = 0; i < d->output.size(); i++) {
            napi_value value;
            napi_create_double(env, d->output[i], &value);
            napi_set_element(env, out, i, value);
        }
        napi_set_named_property(env, result, "output", out);
        napi_resolve_deferred(env, d->deferred, result);
    } else {
        napi_value err;
        napi_create_string_utf8(env, d->error.c_str(), NAPI_AUTO_LENGTH, &err);
        napi_reject_deferred(env, d->deferred, err);
    }

    napi_delete_async_work(env, d->work);
    delete d;
}

napi_value RunInference(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    bool isArray = false;
    napi_is_array(env, argv[0], &isArray);

    if (!isArray) {
        napi_throw_type_error(env, nullptr, "Expected number[]");
        return nullptr;
    }

    uint32_t len = 0;
    napi_get_array_length(env, argv[0], &len);

    // allocate shared work data
    auto *data = new WorkData();

    data->env = env;
    data->input.resize(len);

    for (uint32_t i = 0; i < len; i++) {
        napi_value v;
        napi_get_element(env, argv[0], i, &v);
        napi_get_value_double(env, v, &data->input[i]);
    }

    napi_value promise;
    napi_create_promise(env, &data->deferred, &promise);

    napi_value name;
    napi_create_string_utf8(env, "RunInference", NAPI_AUTO_LENGTH, &name);

    napi_create_async_work(env, nullptr, name,     // context
                           Execute_MOCK, Complete, // execute and complete callbacks
                           data, &data->work       // data ad result
    );

    napi_queue_async_work(env, data->work);
    return promise;
}

} // namespace inference