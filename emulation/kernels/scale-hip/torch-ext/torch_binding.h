#pragma once

#include <torch/torch.h>

// Implemented in csrc/scale.hip.
void scale(torch::Tensor &out, torch::Tensor const &input, double factor);
