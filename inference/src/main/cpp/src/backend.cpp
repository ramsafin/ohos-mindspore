#include "backend.hpp"
#include "types.hpp"

namespace inference {

void Execute_MOCK(napi_env, void *data) {
    auto *d = static_cast<WorkData *>(data);
    d->output.assign(d->input.size(), 1.0f);
}

void Execute_CPU(napi_env, void *data) {
    // ...
}

} // namespace inference
