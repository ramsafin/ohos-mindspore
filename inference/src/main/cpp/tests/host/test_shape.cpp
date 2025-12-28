#include <gtest/gtest.h>

#include "inference/core/shape.hpp"

using namespace inference::core;

TEST(CoreShapeTests, Empty) {
  types::Shape shape;
  EXPECT_TRUE(shape.empty());
}

TEST(CoreShapeTests, NumElements) {
  types::Shape shape;
  EXPECT_EQ(types::numel(shape), 1);
}
