#include <torch/csrc/stable/library.h>

#include "registration.h"
#include "torch_binding.h"

// Stable-ABI registration: one build per ROCm version covers every Torch >= 2.10,
// and adding an op later needs no version bump.
STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("topk_softmax("
          "Tensor! topk_weights, "
          "Tensor! topk_indices, "
          "Tensor! token_expert_indices, "
          "Tensor gating_output, "
          "Tensor softmax_workspace, "
          "bool need_renorm, "
          "Tensor? num_valid_rows) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("topk_softmax", TORCH_BOX(&topk_softmax));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
