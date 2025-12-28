#pragma once

#include <cstdint>
#include <vector>

namespace inference::core::types {

// Shape representation.
//
// Currently defined as a vector of dimension sizes.
// Future implementations may replace this alias with
// a dedicated Shape structure without changing semantics.
using Shape = std::vector<uint32_t>;

/**
 * Calculates the total number of elements in a tensor shape
 * as the product of its dimensions.
 *
 * This function performs no overflow checks on the dimension product.
 *
 * Conventions:
 * - An empty shape represents a scalar tensor and yields 1.
 * - Any dimension equal to 0 yields a zero-size tensor.
 *
 * @param shape Tensor shape
 * @return Total number of elements
 */
inline uint64_t numel(const Shape &shape) {
  uint64_t num_elems = 1;

  for (uint32_t dim : shape) {
    num_elems *= dim;
  }

  return num_elems;
}

} // namespace inference::core::types