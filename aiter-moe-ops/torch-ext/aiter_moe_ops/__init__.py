"""MoE kernels from ROCm/aiter for AMD ROCm GPUs.

Built from the ``aiter/`` submodule (`ROCm/aiter <https://github.com/ROCm/aiter>`_,
unchanged). One package per AITER operation family: this one starts with
``topk_softmax``, AITER's MoE router (softmax over the expert logits, then the top-k
experts per token), from ``csrc/kernels/topk_softmax_kernels.cu``.

``topk_softmax`` keeps the signature of ``aiter/ops/moe_op.py``; ``route`` allocates the
outputs and returns them.
"""

from typing import NamedTuple, Optional

import torch
from torch import Tensor

from ._ops import add_op_namespace_prefix, ops

__kernel_metadata__ = {
    "license": "apache-2.0",
}


@torch.library.register_fake(add_op_namespace_prefix("topk_softmax"))
def _topk_softmax_fake(*args, **kwargs) -> None:
    return None


def _softmax_workspace(gating_output: Tensor) -> Tensor:
    # AITER reads the workspace only for a non-power-of-2 or >256 expert count;
    # the size logic mirrors aiter/ops/moe_op.py.
    num_experts = gating_output.shape[-1]
    num_tokens = gating_output.numel() // num_experts
    pow2 = num_experts & (num_experts - 1) == 0
    size = num_tokens * num_experts if (not pow2 or num_experts > 256) else 0
    return torch.empty(size, dtype=torch.float32, device=gating_output.device)


def topk_softmax(
    topk_weights: Tensor,
    topk_indices: Tensor,
    token_expert_indices: Tensor,
    gating_output: Tensor,
    need_renorm: bool,
    num_shared_experts: int = 0,
    shared_expert_scoring_func: str = "",
    num_valid_rows: Optional[Tensor] = None,
) -> None:
    """Fill preallocated routing outputs (AITER's signature).

    ``gating_output`` is ``[num_tokens, num_experts]`` (float32, float16 or bfloat16).
    Writes ``topk_weights`` (float32) and ``topk_indices`` (int32), both
    ``[num_tokens, topk]`` in descending order of weight, and ``token_expert_indices``
    (int32), where entry ``[t, k]`` is ``k * num_tokens + t``. With ``need_renorm``
    each row of weights sums to 1. ``num_valid_rows`` (int32 device tensor) marks rows
    at or past its first element as padding: their ids are -1 and weights 0.

    Shared experts are not supported by this package yet.
    """
    if num_shared_experts or shared_expert_scoring_func:
        raise NotImplementedError("aiter-moe-ops: shared experts are not supported yet")
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        _softmax_workspace(gating_output),
        need_renorm,
        num_valid_rows,
    )


class Routing(NamedTuple):
    topk_weights: Tensor  # [num_tokens, topk] float32
    topk_ids: Tensor  # [num_tokens, topk] int32


def route(
    gating_output: Tensor,
    topk: int,
    renormalize: bool = False,
    num_valid_rows: Optional[Tensor] = None,
) -> Routing:
    """Top-k softmax routing for ``gating_output`` of shape ``[..., num_experts]``."""
    num_experts = gating_output.shape[-1]
    logits = gating_output.reshape(-1, num_experts)
    num_tokens = logits.shape[0]
    weights = torch.empty(num_tokens, topk, dtype=torch.float32, device=logits.device)
    ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=logits.device)
    token_expert = torch.empty(
        num_tokens, topk, dtype=torch.int32, device=logits.device
    )
    topk_softmax(
        weights, ids, token_expert, logits, renormalize, num_valid_rows=num_valid_rows
    )
    lead = gating_output.shape[:-1]
    return Routing(weights.reshape(*lead, topk), ids.reshape(*lead, topk))


__all__ = [
    "__kernel_metadata__",
    "Routing",
    "route",
    "topk_softmax",
]
