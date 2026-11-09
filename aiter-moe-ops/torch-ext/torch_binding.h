#pragma once

// Stable-ABI signatures of the package's ops. Each is defined in csrc/<op>_torch.hip,
// which converts the tensors and calls the AITER kernel built from the submodule.
#include <torch/csrc/stable/tensor.h>

#include <optional>

void topk_softmax(torch::stable::Tensor& topk_weights,
                  torch::stable::Tensor& topk_indices,
                  torch::stable::Tensor& token_expert_indices,
                  const torch::stable::Tensor& gating_output,
                  const torch::stable::Tensor& softmax_workspace,
                  bool need_renorm,
                  const std::optional<torch::stable::Tensor>& num_valid_rows);
