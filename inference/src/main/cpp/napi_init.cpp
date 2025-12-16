#include "napi/native_api.h"

#include <memory>
#include <cstring>

#include "inference/context.hpp"
#include "inference/napi_helpers.hpp"

namespace {

struct ContextWrap final {
    std::shared_ptr<inference::Context> context;
    bool closed{false};
};

struct CreateCtxWork final {
    napi_env env{};
    napi_deferred deferred{};
    napi_async_work work{};

    inference::ModelConfig config;
    std::shared_ptr<inference::Context> context;
    std::string error;
};

struct RunWork final {
    napi_env env{};
    napi_deferred deferred{};
    napi_async_work work{};

    std::shared_ptr<inference::Context> context;
    inference::Tensor input_owned; // safe: copied from JS
    inference::Tensor output_owned;
    std::string error;
};

napi_value ctx_run(napi_env env, napi_callback_info info) {
    napi_value js_this{};
    size_t argc = 1;
    napi_value args[1]{};

    if (napi_get_cb_info(env, info, &argc, args, &js_this, nullptr) != napi_ok) {
        return nullptr;
    }

    ContextWrap *wrap = nullptr;
    if (napi_unwrap(env, js_this, reinterpret_cast<void **>(&wrap)) != napi_ok) {
        return nullptr;
    }

    if (!wrap || wrap->closed || !wrap->context) {
        napi::throw_with_message(env, "Context is closed");
        return nullptr;
    }

    if (argc < 1) {
        napi::throw_with_message(env, "run(inputTensor) missing input");
        return nullptr;
    }

    auto *work = new RunWork();
    work->env = env;
    work->context = wrap->context;

    if (!napi::parse_input_tensor(env, args[0], work->input_owned, work->error)) {
        napi::throw_with_message(env, work->error);
        delete work;
        return nullptr;
    }

    napi_value promise = nullptr;
    napi_create_promise(env, &work->deferred, &promise);

    napi_value resource = nullptr;
    napi_create_string_utf8(env, "inference.run", NAPI_AUTO_LENGTH, &resource);

    napi_create_async_work(
        env, nullptr, resource,
        [](napi_env /*env*/, void *data) {
            auto *work = static_cast<RunWork *>(data);
            try {
                inference::TensorView view{work->input_owned.shape, work->input_owned.data};
                work->output_owned = work->context->run(view);
            } catch (const std::exception &e) {
                work->error = e.what();
            }
        },
        [](napi_env env, napi_status /*status*/, void *data) {
            std::unique_ptr<RunWork> work(static_cast<RunWork *>(data));
            if (!work->error.empty()) {
                napi_reject_deferred(env, work->deferred, napi::make_error(env, work->error));
            } else {
                napi_value out = napi::make_output_tensor(env, work->output_owned);
                napi_resolve_deferred(env, work->deferred, out);
            }
            napi_delete_async_work(env, work->work);
        },
        work, &work->work);

    napi_queue_async_work(env, work->work);
    return promise;
}

napi_value create_wrapped_context_object(napi_env env, std::shared_ptr<inference::Context> context) {
    napi_value obj = nullptr;
    napi_create_object(env, &obj);

    auto *wrap = new ContextWrap{std::move(context), false};
    napi_wrap(
        env, obj, wrap, [](napi_env /*env*/, void *data, void * /*hint*/) { delete static_cast<ContextWrap *>(data); },
        nullptr, nullptr);

    napi_property_descriptor props[] = {
        {"run", nullptr, ctx_run, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, obj, sizeof(props) / sizeof(props[0]), props);
    return obj;
}

} // namespace

napi_value NAPI_Global_createContext(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1]{};

    if (napi_get_cb_info(env, info, &argc, args, nullptr, nullptr) != napi_ok) {
        return nullptr;
    }

    if (argc < 1) {
        napi::throw_with_message(env, "createContext(config) missing config");
        return nullptr;
    }

    auto *work = new CreateCtxWork();
    work->env = env;

    if (!napi::parse_model_config(env, args[0], work->config, work->error)) {
        napi::throw_with_message(env, work->error);
        delete work;
        return nullptr;
    }

    napi_value promise = nullptr;
    napi_create_promise(env, &work->deferred, &promise);

    napi_value resource = nullptr;
    napi_create_string_utf8(env, "inference.createContext", NAPI_AUTO_LENGTH, &resource);

    napi_create_async_work(
        env, nullptr, resource,
        [](napi_env /*env*/, void *data) {
            auto *w = static_cast<CreateCtxWork *>(data);
            try {
                w->context = std::make_shared<inference::Context>(std::move(w->config));
            } catch (const std::exception &e) {
                w->error = e.what();
            }
        },
        [](napi_env env, napi_status /*status*/, void *data) {
            std::unique_ptr<CreateCtxWork> work(static_cast<CreateCtxWork *>(data));

            if (!work->error.empty()) {
                napi_reject_deferred(env, work->deferred, napi::make_error(env, work->error));
            } else {
                napi_value ctx_obj = create_wrapped_context_object(env, work->context);
                napi_resolve_deferred(env, work->deferred, ctx_obj);
            }
            napi_delete_async_work(env, work->work);
        },
        work, &work->work);

    napi_queue_async_work(env, work->work);
    return promise;
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"createContext", nullptr, NAPI_Global_createContext, nullptr, nullptr, nullptr, napi_default, nullptr}};
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "inference",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterInferenceModule() { napi_module_register(&module); }
