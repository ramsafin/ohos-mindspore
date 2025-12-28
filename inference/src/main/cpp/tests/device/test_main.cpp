#include "inference/core/tensor.hpp"

using namespace inference::core;

int main(int argc, char **argv) {
  types::Tensor tensor; // empty tensor
  const size_t numel = types::numel(tensor.shape);
  return 0;
}
