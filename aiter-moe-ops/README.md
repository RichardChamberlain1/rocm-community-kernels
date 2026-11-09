---
license: apache-2.0
tags:
  - kernels
---

# aiter-moe-ops

Mixture-of-experts kernels from [ROCm/aiter](https://github.com/ROCm/aiter), built
ahead of time for AMD Instinct GPUs (MI300X / gfx942, MI355X / gfx950) and published as
a Hugging Face Hub kernel. One package per AITER operation family: this one starts with
`topk_softmax`, AITER's MoE router.

## Where the code comes from

The kernels are built straight from [ROCm/aiter](https://github.com/ROCm/aiter), a git
submodule at `aiter/` pinned to release
[v0.1.29](https://github.com/ROCm/aiter/releases/tag/v0.1.29),
`a0c4b69c146d94e393984bbe553b704e8c89a5e4`. No upstream file is copied or changed.

Compiled from AITER without changes:

- `aiter/csrc/kernels/topk_softmax_kernels.cu` (Apache-2.0; © Advanced Micro Devices,
  Inc., the vLLM team and NVIDIA, adapted from TensorRT-LLM);
- the 12 AITER headers it and the bridge include, each listed in `build.toml`.
  `ENABLE_CK` is not defined, so AITER's `aiter_hip_common.h` uses its own
  `ck_tile_shim.h` and Composable Kernel is not needed.

This package adds only:

- `csrc/aiter_torch_bridge.h`: converts Torch tensors to AITER's framework-neutral
  `aiter_tensor_t`, points AITER's stream at Torch's current stream and turns failed
  `AITER_CHECK`s into Python errors. Shared by every op in the package.
- `csrc/topk_softmax_torch.hip`: checks the arguments and calls `aiter::topk_softmax`.
- `torch-ext/torch_binding.cpp`: registers the op on the Torch stable ABI, so one build
  per ROCm version serves every Torch from 2.10, and later ops need no version bump.
- `torch-ext/aiter_moe_ops/`: the Python API, with AITER's signature from
  `aiter/ops/moe_op.py`, and a fake kernel so the op works with `torch.compile`.

`build.toml` is generated from `sync/aiter-moe-ops.toml` in the repository root by
`scripts/update_aiter_package.py`, which lists every header the build needs by following
`#include` lines.

## Functions

### `route(gating_output, topk, renormalize=False, num_valid_rows=None) -> Routing`

Top-k softmax routing. `gating_output` is `[..., num_experts]` (float32, float16 or
bfloat16). Returns `Routing(topk_weights, topk_ids)`, both `[..., topk]`: float32
weights and int32 expert ids, in descending order of weight. With `renormalize` each
row of weights sums to 1.

### `topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, need_renorm, num_shared_experts=0, shared_expert_scoring_func="", num_valid_rows=None)`

AITER's signature: fills preallocated `[num_tokens, topk]` buffers. `topk_weights` is
float32; `topk_indices` and `token_expert_indices` are int32, and
`token_expert_indices[t, k] = k * num_tokens + t`. `num_valid_rows` (an int32 device
tensor) marks rows at or past its first element as padding: their ids are -1 and their
weights 0. It is read on the device, so a replayed graph follows each step's count.

Shared experts (`num_shared_experts > 0`) are not supported by this package yet.

## Quickstart

```python
import torch
from kernels import get_kernel

moe = get_kernel("amd/aiter-moe-ops", version=1)

router_logits = torch.randn(1024, 64, device="cuda", dtype=torch.bfloat16)
weights, ids = moe.route(router_logits, topk=8, renormalize=True)
```

## Building and testing

```bash
git clone --recurse-submodules https://github.com/ROCm/rocm-community-kernels.git
cd rocm-community-kernels/aiter-moe-ops
nix build .#bundle        # one build per ROCm version (Torch stable ABI)
nix run .#ci-test         # build, then pytest -m kernels_ci on a ROCm GPU
```

Tests load the kernel through `get_kernel` and compare with a PyTorch reference.

## Updating to a newer AITER

The `aiter-sync` workflow proposes new AITER releases every week. By hand, from the
repository root:

```bash
python3 scripts/update_aiter_package.py sync/aiter-moe-ops.toml --to v0.1.30
git add -A
```

Then check whether `aiter::topk_softmax`'s declaration in `aiter/csrc/include/moe_op.h`
changed (the bridge calls it), build, and run the tests.

## Adding an op

1. Add the AITER kernel source to `sources` in `sync/aiter-moe-ops.toml`.
1. Add `csrc/<op>_torch.hip` (argument checks, `aiter_bridge::view` for each tensor, an
   `aiter_bridge::CallScope`, then the AITER call), its declaration in
   `torch-ext/torch_binding.h` and its schema in `torch-ext/torch_binding.cpp`.
1. Add the Python function and fake kernel, and tests.
1. Run `scripts/update_aiter_package.py sync/aiter-moe-ops.toml`.

On the stable ABI, adding an op doesn't need a new version.

## License

Apache-2.0, see `LICENSE`. The kernel is Apache-2.0 (© Advanced Micro Devices, Inc., the
vLLM team and NVIDIA); AITER's headers are MIT (© Advanced Micro Devices, Inc.).
