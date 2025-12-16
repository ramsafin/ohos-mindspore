#pragma once

#include <mutex>

#include "inference/types.hpp"

namespace inference {

struct Context final {
public:
    explicit Context(ModelConfig config) : config_{std::move(config)} {}

    Tensor run(const TensorView &in) {
        std::scoped_lock lock{mutex_};

        if (config_.device == "MOCK") {
            return Tensor{Shape{1, 4}, std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}};
        }

        // CPU, NPU, Kirin-NPU, etc.
        return Tensor{Shape{1, 2}, std::vector<float>{0.3f, 0.7f}};
    }

private:
    mutable std::mutex mutex_;
    ModelConfig config_;
};

} // namespace inference