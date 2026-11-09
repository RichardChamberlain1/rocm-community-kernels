// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared helpers for calling AITER kernels from Torch ops on the Torch stable ABI.
// AITER's kernels take its own framework-neutral aiter_tensor_t and launch on
// AITER's thread-local stream; these helpers convert stable-ABI tensors and point
// that stream at Torch's current one. Nothing here changes AITER.
#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <optional>

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

// The shim's stream accessor is guarded by USE_CUDA, so declare it here (as the
// activation package does). On ROCm it returns the current HIP stream.
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

namespace aiter_bridge {

using torch::stable::Tensor;

inline AiterDtype to_aiter_dtype(torch::headeronly::ScalarType t, const char* op)
{
    using torch::headeronly::ScalarType;
    switch(t)
    {
    case ScalarType::Float: return AITER_DTYPE_fp32;
    case ScalarType::Half: return AITER_DTYPE_fp16;
    case ScalarType::BFloat16: return AITER_DTYPE_bf16;
    case ScalarType::Int: return AITER_DTYPE_i32;
    case ScalarType::Long: return AITER_DTYPE_i64;
    case ScalarType::Short: return AITER_DTYPE_i16;
    case ScalarType::Char: return AITER_DTYPE_i8;
    case ScalarType::Byte: return AITER_DTYPE_u8;
    default: STD_TORCH_CHECK(false, op, ": unsupported dtype");
    }
}

// A non-owning AITER view of a Torch tensor on the GPU.
inline aiter_tensor_t view(const Tensor& t, const char* op)
{
    STD_TORCH_CHECK(t.is_cuda(), op, ": tensors must be on a GPU");
    STD_TORCH_CHECK(t.dim() <= 8, op, ": tensors may have at most 8 dims");
    aiter_tensor_t v{};
    v.ptr       = t.data_ptr();
    v.numel_    = static_cast<size_t>(t.numel());
    v.ndim      = static_cast<int>(t.dim());
    v.dtype_    = to_aiter_dtype(t.scalar_type(), op);
    v.device_id = t.get_device_index();
    for(int i = 0; i < v.ndim; ++i)
    {
        v.shape[i]   = t.size(i);
        v.strides[i] = t.stride(i);
    }
    return v;
}

inline std::optional<aiter_tensor_t> view(const std::optional<Tensor>& t, const char* op)
{
    if(!t.has_value())
        return std::nullopt;
    return view(*t, op);
}

// Set up an AITER call on `t`'s device: a device guard, AITER's stream pointed at
// Torch's current stream, and AITER_CHECK failures raised as exceptions (which
// Torch turns into Python errors) instead of aborting the process.
class CallScope
{
    public:
    explicit CallScope(const Tensor& t) : device_guard_(t.get_device_index())
    {
        void* stream = nullptr;
        TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream));
        aiter::setCurrentHIPStream(static_cast<hipStream_t>(stream));
    }

    private:
    torch::stable::accelerator::DeviceGuard device_guard_;
    aiter_detail::AiterThrowGuard throw_guard_;
};

} // namespace aiter_bridge
