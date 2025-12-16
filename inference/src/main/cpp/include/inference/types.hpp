#pragma once

#include <span>
#include <vector>
#include <string>
#include <cstdint>

namespace inference {

// no enums yet for simplicity, just plain strings
using Device = std::string;

// owning vector (safely copying JS shape)
using Shape = std::vector<std::uint32_t>;

// owning tensor (safely copies JS tensor)
struct Tensor final {
    Shape shape;
    std::vector<float> data;
};

// non-owning tensor (dangerous, requires scoped handling via napi_ref?)
struct TensorView final {
    Shape shape;
    std::span<float const> data;
};

struct ModelConfig final {
    Device device;
    // prototype: owning copy
    std::vector<std::uint8_t> model_data;
};

} // namespace inference
