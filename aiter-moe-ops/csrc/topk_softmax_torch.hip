// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch op for AITER's topk_softmax (MoE routing: softmax over the expert logits,
// then the top-k experts per token). The kernel is compiled unchanged from the AITER
// submodule (aiter/csrc/kernels/topk_softmax_kernels.cu); this file only checks the
// arguments and converts tensors. Shared experts (AITER's num_shared_experts) are not
// exposed yet: their output column layout still has to be confirmed against AITER.
#include "aiter_torch_bridge.h"
#include "moe_op.h"

using aiter_bridge::view;
using torch::headeronly::ScalarType;
using torch::stable::Tensor;

namespace {

constexpr const char* kOp = "topk_softmax";

void check_routing_output(const Tensor& t, const Tensor& gating_output, const char* name, ScalarType dtype,
                          int64_t min_cols)
{
    STD_TORCH_CHECK(t.get_device_index() == gating_output.get_device_index(), kOp, ": ", name,
                    " must be on the same device as gating_output");
    STD_TORCH_CHECK(t.scalar_type() == dtype, kOp, ": ", name, " has the wrong dtype");
    STD_TORCH_CHECK(t.dim() == 2, kOp, ": ", name, " must be 2-D [num_tokens, k]");
    STD_TORCH_CHECK(t.stride(1) == 1, kOp, ": ", name, " must be contiguous in its last dim");
    STD_TORCH_CHECK(t.size(1) >= min_cols, kOp, ": ", name, " has too few columns");
}

} // namespace

void topk_softmax(Tensor& topk_weights,
                  Tensor& topk_indices,
                  Tensor& token_expert_indices,
                  const Tensor& gating_output,
                  const Tensor& softmax_workspace,
                  bool need_renorm,
                  const std::optional<Tensor>& num_valid_rows)
{
    STD_TORCH_CHECK(gating_output.is_cuda(), kOp, ": gating_output must be on a GPU");
    STD_TORCH_CHECK(gating_output.dim() == 2, kOp, ": gating_output must be 2-D [num_tokens, num_experts]");
    STD_TORCH_CHECK(gating_output.stride(1) == 1, kOp, ": gating_output must be contiguous in its last dim");
    const ScalarType g = gating_output.scalar_type();
    STD_TORCH_CHECK(g == ScalarType::Float || g == ScalarType::Half || g == ScalarType::BFloat16, kOp,
                    ": gating_output must be float32, float16 or bfloat16");

    const int64_t num_tokens = gating_output.size(0);
    const int64_t topk       = topk_indices.size(1);
    const int64_t routing    = gating_output.size(1);
    STD_TORCH_CHECK(topk >= 1 && topk <= routing, kOp, ": topk must be in [1, num_experts]");
    check_routing_output(topk_weights, gating_output, "topk_weights", ScalarType::Float, topk);
    check_routing_output(topk_indices, gating_output, "topk_indices", ScalarType::Int, 1);
    check_routing_output(token_expert_indices, gating_output, "token_expert_indices", ScalarType::Int, topk);
    STD_TORCH_CHECK(topk_weights.size(0) == num_tokens && topk_indices.size(0) == num_tokens &&
                        token_expert_indices.size(0) == num_tokens,
                    kOp, ": outputs must have num_tokens rows");

    // AITER reads the workspace only for a non-power-of-2 or >256 routing-expert count.
    const bool pow2           = (routing & (routing - 1)) == 0;
    const int64_t ws_required = (!pow2 || routing > 256) ? num_tokens * routing : 0;
    STD_TORCH_CHECK(softmax_workspace.scalar_type() == ScalarType::Float, kOp,
                    ": softmax_workspace must be float32");
    STD_TORCH_CHECK(softmax_workspace.numel() >= ws_required, kOp, ": softmax_workspace needs at least ",
                    ws_required, " elements");
    if(num_valid_rows.has_value())
    {
        STD_TORCH_CHECK(num_valid_rows->scalar_type() == ScalarType::Int && num_valid_rows->numel() >= 1, kOp,
                        ": num_valid_rows must be an int32 tensor with at least one element");
    }

    if(num_tokens == 0)
        return;

    aiter_bridge::CallScope scope(gating_output);
    aiter::topk_softmax(view(topk_weights, kOp),
                        view(topk_indices, kOp),
                        view(token_expert_indices, kOp),
                        view(gating_output, kOp),
                        view(softmax_workspace, kOp),
                        need_renorm,
                        /*num_shared_experts=*/0,
                        /*shared_expert_scoring_func=*/"",
                        view(num_valid_rows, kOp));
}
