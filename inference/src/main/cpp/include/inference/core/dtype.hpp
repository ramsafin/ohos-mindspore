#pragma once

#include <cstddef> // size_t
#include <cstdint> // uint32_t

namespace inference::core::types {

/**
 * Logical data type of a tensor element.
 *
 * This enum describes how raw tensor memory should be interpreted.
 * It is independent of storage, layout, or backend implementation.
 */
enum class DataType : uint32_t {
  UNDEFINED = 0,
  FLOAT32,
  UINT8,
};

/**
 * Returns the size in bytes of a single element of the given data type.
 *
 * Conventions:
 * - Returns 0 for DataType::UNDEFINED or unsupported data types.
 * - Performs no validation or error reporting.
 *
 * @param dtype DataType value
 * @return Element size in bytes, or 0 if undefined
 */
inline size_t element_size(DataType dtype) {
  switch (dtype) {
  case DataType::FLOAT32:
    return 4;
  case DataType::UINT8:
    return 1;
  default:
    return 0;
  }
}

} // namespace inference::core::types
