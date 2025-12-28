#pragma once

#include <cstdint> // uint32_t

namespace inference::core::types {

/**
 * Semantic layout of tensor dimensions.
 *
 * Layout describes how tensor dimensions should be interpreted,
 * not how data is physically stored in memory.
 *
 * At the core layer, layout is intentionally minimal and is only
 * used where semantic differences matter (e.g. image tensors).
 *
 * Conventions:
 * - UNDEFINED indicates that no semantic layout is specified.
 * - Layout is required only when dimension meaning is important.
 * - Physical storage order and strides are not represented here.
 */
enum class Layout : uint32_t {
  UNDEFINED = 0,
  NCHW, // Batch, Channel, Height, Width
  NHWC, // Batch, Height, Width, Channel
};

} // namespace inference::core::types
