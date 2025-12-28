#pragma once

#include <cstdint> // uint8_t
#include <vector>

#include "inference/core/dtype.hpp"
#include "inference/core/layout.hpp"
#include "inference/core/shape.hpp"

namespace inference::core::types {

struct Tensor {
  /**
   * Raw tensor storage.
   *
   * Owns a contiguous buffer of bytes representing tensor data.
   * Interpretation of this memory is defined by `dtype`, `shape`,
   * and `layout`, and is performed via tensor views.
   */
  std::vector<uint8_t> buffer;

  /**
   * Tensor shape (dimensions).
   *
   * See Shape documentation for conventions.
   */
  Shape shape;

  /**
   * Semantic layout of tensor dimensions.
   *
   * Layout is required only when dimension meaning matters
   * (e.g. image tensors). May be UNDEFINED otherwise.
   */
  Layout layout = Layout::UNDEFINED;

  /**
   * Logical data type of tensor elements.
   *
   * Describes how raw memory in `buffer` should be interpreted.
   */
  DataType dtype = DataType::UNDEFINED;
};


} // namespace inference::core::types
